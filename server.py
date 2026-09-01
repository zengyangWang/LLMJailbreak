import os
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = REPO_ROOT.parent
NLTK_DATA_ROOT = BUNDLE_ROOT / "nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_ROOT))
VENDORED_FASTCHAT_ROOT = REPO_ROOT / "FastChat"
if str(VENDORED_FASTCHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_FASTCHAT_ROOT))

# 强制设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制设置 HF 缓存目录 (可选，防止去默认路径找)
os.environ["HF_HOME"] = str(BUNDLE_ROOT / "huggingface_cache")

import argparse

import tempfile
import uvicorn
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 引入项目现有代码
from main import main as run_attack_logic
from initialize_args import initialize_args
from data.language import Language
from attack_categories import (
    ENCODING_PERTURBATION_CATEGORIES,
    EXTRA_CATEGORIES,
    PROMPT_STRATEGY_CATEGORIES,
    infer_attack_category,
)
from baseline.Template.Template_single_main import (
    _load_templates as _load_template_defs,
    _apply_template_text as _apply_template_text,
)
from baseline.Translate.Translate_single_main import Translate_single_main
from baseline.SensitivePinyin.SensitivePinyin_single_main import (
    SensitivePinyin_single_main,
)
from baseline.NoiseInjection.NoiseInjection_single_main import (
    NoiseInjection_single_main,
)
from baseline.CodeChameleon.CodeChameleon_single_main import (
    CodeChameleon_single_main,
)
from baseline.Caesar.Caesar_single_main import Caesar_single_main
from baseline.Rot13.Rot13_single_main import Rot13_single_main
from baseline.Morse.Morse_single_main import (
    Morse_single_main,
)

app = FastAPI()

# 支持的“基线攻击方法”（原有 20+ 个），仍然保留，供需要完整攻击流程的场景使用；
# 在此基础上，我们只额外集成 Template 拆分出来的小方法，以及整体的 Translate。
SUPPORTED_ATTACK_METHODS: List[str] = [
    "AutoDAN",
    "PAIR",
    "TAP",
    "GPTFuzz",
    "GCG",
    "AdvPrompter",
    "AmpleGCG",
    "DrAttack",
    "MultiJail",
    "PAP",
    "JailBroken",
    "LLMAdaptive",
    "MJP",
    "Cipher",
    "ReNeLLM",
    "CodeChameleon",
    "FlipAttack",
    "Coldattack",
    "Actorattack",
    "FuzzLLM",
    "ICA",
    "SATA",
    "SensitivePinyin",
    "NoiseInjection",
    "Translate",
]
SUPPORTED_ATTACK_METHODS_SET = {m.lower() for m in SUPPORTED_ATTACK_METHODS}
DEFAULT_ATTACK_METHOD = "GCG"

# 37 类“攻击类别”（与 JailBreak_910B 完全一致）
SUPPORTED_CATEGORIES: List[str] = sorted(
    set(PROMPT_STRATEGY_CATEGORIES)
    | set(ENCODING_PERTURBATION_CATEGORIES)
    | set(EXTRA_CATEGORIES)
)
SUPPORTED_CATEGORIES_SET = set(SUPPORTED_CATEGORIES)


def _discover_template_categories() -> List[str]:
    """
    动态扫描模板文件，找出“确实有模板覆盖”的类别。
    这样可以避免某些类别当前没有对应模板而触发 No template found 异常。
    仅保留提示策略类 + EXTRA_CATEGORIES（排除编码/噪声等算法类）。
    """
    cats: Set[str] = set()
    for lang_code, lang_enum in (("en", Language.ENGLISH), ("zh", Language.CHINESE_SIMPLIFIED)):
        try:
            templates = _load_template_defs(lang_enum)
        except Exception:
            continue
        for tpl in templates:
            cat = infer_attack_category(
                attack_method="template",
                template_name=tpl.get("name"),
                template_id=tpl.get("id"),
                lang_used=lang_code,
            )
            if cat in PROMPT_STRATEGY_CATEGORIES or cat in EXTRA_CATEGORIES:
                cats.add(cat)
    return sorted(cats)


# 仅将“确实有模板”的类别拆成“小方法”暴露给前端
TEMPLATE_SUBMETHODS: List[str] = _discover_template_categories()

# 兼容旧的英文 key（如 noise 等；注意此处不再把 translate 映射为类别）
_LEGACY_METHOD_TO_CATEGORY: Dict[str, str] = {
    "noise": "噪声注入",
    "noise_injection": "噪声注入",
    "code_chameleon": "Code Chameleon 加密转换",
    "caesar": "Caesar 加密",
    "rot13": "rot13 编码",
    "morse": "摩斯电码",
    "sensitive_pinyin": "同音词替换",
}

_BASELINE_METHOD_NAMES = {m: m for m in SUPPORTED_ATTACK_METHODS}

# --- 1. 定义对应的 Pydantic 模型 ---

class AttackTask(BaseModel):
    RiskID: int = Field(..., alias="risk_id")  # 对应risk_id
    Goal: str = Field(..., alias="goal")
    Target: Optional[str] = Field(None, alias="target")  # 可选；可由语言推断默认 target
    Language: str = Field("ENGLISH", alias="language")
    IsPublic: bool = Field(False, alias="is_public")

class AttackRequest(BaseModel):
    # AttackMethod 兼容字符串或列表（如 ["SATA"]）；内部会统一转换为单个方法名
    AttackMethod: Union[str, List[str]] = Field(..., alias="attack_method")
    TargetModel: Optional[str] = Field("", alias="target_model")  # 可选；GCG 默认本地权重
    DefenseType: str = Field("None_defense", alias="defense_type")
    ExpName: str = Field("api_exp", alias="exp_name")
    Tasks: List[AttackTask] = Field(..., alias="tasks")
    Config: Optional[Dict[str, Any]] = Field(default={}, alias="config")


class MethodListResponse(BaseModel):
    total: int
    attack_methods: List[str]

# --- 2. 辅助函数 ---
def normalize_language_enum(lang_str: str) -> str:
    """
    规范化外部传入的 language 字段，输出与 Language 枚举一致的名字：
    - 首选返回 Language.__members__ 中已有的键（ENGLISH / CHINESE_SIMPLIFIED / CHINESE_TRADITIONAL）
    - 兼容常见别名（zh/en、中文、简体中文、繁體中文 等）
    - 对未知/非法值统一回退为 CHINESE_SIMPLIFIED，避免出现 UNKNOWN 等非法枚举名
    """
    if lang_str is None:
        return "CHINESE_SIMPLIFIED"
    key = str(lang_str).strip().upper()
    if not key:
        return "CHINESE_SIMPLIFIED"

    # 若本身就是合法的 Language 枚举键，直接返回
    if key in Language.__members__:
        return key

    # 常见中文/英文别名映射到标准键
    zh_aliases = {
        "ZH",
        "CN",
        "CHINESE",
        "CHINESE_SIMPLIFIED",
        "CHINESE_TRADITIONAL",
        "SIMPLIFIED_CHINESE",
        "TRADITIONAL_CHINESE",
        "中文",
        "简体中文",
        "繁體中文",
    }
    en_aliases = {
        "EN",
        "ENGLISH",
    }
    if key in zh_aliases:
        return "CHINESE_SIMPLIFIED"
    if key in en_aliases:
        return "ENGLISH"

    # 兜底：所有无法识别的 language 一律当作中文简体，防止出现 UNKNOWN
    return "CHINESE_SIMPLIFIED"


def _normalize_language_code(raw: Optional[str]) -> str:
    """
    将各种语言标记统一为 'zh' 或 'en'，用于选择模板与编码前缀。
    逻辑与 JailBreak_910B/template_attack.py 中的 normalize_language 基本一致，
    但这里面向本项目的 Language 枚举和字符串。
    """
    if raw is None:
        return "en"
    key = str(raw).strip().upper()
    zh_aliases = {
        "ZH",
        "CN",
        "CHINESE",
        "CHINESE_SIMPLIFIED",
        "CHINESE_TRADITIONAL",
        "SIMPLIFIED_CHINESE",
        "TRADITIONAL_CHINESE",
        "中文",
        "简体中文",
        "繁體中文",
    }
    en_aliases = {
        "EN",
        "ENGLISH",
    }
    if key in zh_aliases:
        return "zh"
    if key in en_aliases:
        return "en"
    # 默认当作英文模板
    return "en"


def _to_language_enum(lang_str: str) -> Language:
    """
    将请求中的 language 字段转换为内部的 Language 枚举。
    目前仅区分“中文 vs 非中文”即可满足 Template/Translate/噪声等攻击需要。
    """
    upper = normalize_language_enum(lang_str)
    if "CHINESE" in upper or upper in ("ZH", "CN", "中文", "简体中文", "繁體中文"):
        return Language.CHINESE_SIMPLIFIED
    return Language.ENGLISH

def default_target_for_language(lang_upper: str) -> str:
    # 参考 main.py 中的 prime map
    prime_map = {
        "ENGLISH": "Sure, here is",
        "CHINESE_SIMPLIFIED": "好的，这是",
        "CHINESE_TRADITIONAL": "好的，這是",
        "FRENCH": "Bien sûr, voici",
        "GERMAN": "Gern, hier ist",
        "SPANISH": "Claro, aquí está",
        "ITALIAN": "Certo, ecco",
        "JAPANESE": "はい、こちらは",
    }
    return prime_map.get(lang_upper, "Sure, here is")

def select_default_target_model_path(base_dir: str) -> str:
    # 本地常见可用权重优先级
    candidates = [
        os.path.join(base_dir, "llm_weights", "vicuna-13b-v1.5"),
        os.path.join(base_dir, "llm_weights", "llama-2-7b-chat-hf"),
        os.path.join(base_dir, "llm_weights", "vicuna-7b-v1.5"),
        os.path.join(base_dir, "llm_weights", "llama-2-7b"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    # 兜底：保持 initialize_args 的默认
    return ""


def _config_with_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JailBreak_910B server config keys to the baseline argument names
    used by this repository. Existing baseline-style keys keep priority.
    """
    normalized = dict(config or {})

    if "samples" in normalized and "noise_num_samples" not in normalized:
        normalized["noise_num_samples"] = normalized["samples"]
    if "gap_noise_range" in normalized and "noise_gap_noise_range_str" not in normalized:
        gap_range = normalized["gap_noise_range"]
        if isinstance(gap_range, (list, tuple)) and len(gap_range) == 2:
            normalized["noise_gap_noise_range_str"] = f"{gap_range[0]},{gap_range[1]}"
        else:
            normalized["noise_gap_noise_range_str"] = str(gap_range)
    if "gap_noise_prob" in normalized and "noise_gap_noise_prob" not in normalized:
        normalized["noise_gap_noise_prob"] = normalized["gap_noise_prob"]

    if "sensitive_dir" in normalized and "sensitive_vocab_dir" not in normalized:
        normalized["sensitive_vocab_dir"] = normalized["sensitive_dir"]
    if "sensitive_styles" in normalized and "sensitive_pinyin_styles_str" not in normalized:
        sensitive_styles = normalized["sensitive_styles"]
        if isinstance(sensitive_styles, (list, tuple)):
            normalized["sensitive_pinyin_styles_str"] = ",".join(
                str(style).strip() for style in sensitive_styles if str(style).strip()
            )
        else:
            normalized["sensitive_pinyin_styles_str"] = str(sensitive_styles)

    return normalized


def _normalize_key(s: Optional[str]) -> str:
    return (s or "").strip()


def _resolve_requested_category(attack_method: str) -> Tuple[str, Optional[str]]:
    """
    返回 (mode, category)
    - mode="category": attack_method 是 37 类之一，category 为其本身
    - mode="legacy_template": attack_method=template，表示“直接套任意模板”（不按类别过滤）
    - mode="legacy_mapped": attack_method 为旧 key（noise/caesar/...），映射到对应类别
    - mode="unknown": 不认识
    """
    raw = _normalize_key(attack_method)
    if raw in SUPPORTED_CATEGORIES_SET:
        return ("category", raw)
    # Preserve the canonical baseline method names exposed by this project.
    # Lowercase legacy aliases such as "translate" are still mapped below.
    if raw in _BASELINE_METHOD_NAMES:
        return ("baseline", None)
    low = raw.lower()
    if low == "template":
        return ("legacy_template", None)
    if low == "translate":
        return ("legacy_mapped", "翻译模式")
    if low in _LEGACY_METHOD_TO_CATEGORY:
        return ("legacy_mapped", _LEGACY_METHOD_TO_CATEGORY[low])
    return ("unknown", None)


def _pick_template_for_category(
    *,
    preferred_lang_code: str,
    category: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    从 Template JSON 中选择一个模板：
    - category=None 表示不按类别过滤（legacy template 模式）
    - preferred_lang_code: 'zh' 或 'en'
    - 支持 config.template_id / config.template_name 精确指定
    - 语言优先级：先尝试 preferred_lang_code；若该语言下无匹配模板，则自动回退到另一种语言
    """
    # 与 JailBreak_910B 中 _pick_template_for_category 的逻辑保持一致：
    # 先尝试首选语言，再在必要时回退到另一种语言，避免因为某个类别只在 zh/en 之一出现而报错。
    langs_to_try = [preferred_lang_code, "zh" if preferred_lang_code == "en" else "en"]

    template_id = _normalize_key(config.get("template_id"))
    template_name = _normalize_key(config.get("template_name"))

    last_err: Optional[Exception] = None

    for lang_code in langs_to_try:
        try:
            # 将 'zh' / 'en' 映射到 Language 枚举
            lang_enum = (
                Language.CHINESE_SIMPLIFIED
                if lang_code == "zh"
                else Language.ENGLISH
            )
            templates = _load_template_defs(lang_enum)

            # 可按类别过滤
            if category:
                filtered: List[Dict[str, str]] = []
                for tpl in templates:
                    cat = infer_attack_category(
                        attack_method="template",
                        template_name=tpl.get("name"),
                        template_id=tpl.get("id"),
                        lang_used=lang_code,
                    )
                    if cat == category:
                        filtered.append(tpl)
                templates = filtered

            if not templates:
                # 当前语言下没有符合该类别的模板，尝试下一种语言
                continue

            # 精确指定优先
            if template_id:
                for tpl in templates:
                    if str(tpl.get("id", "")).strip() == template_id:
                        return tpl
            if template_name:
                for tpl in templates:
                    if str(tpl.get("name", "")).strip() == template_name:
                        return tpl

            # 稳定默认：按 id 排序取第一个
            templates_sorted = sorted(
                templates,
                key=lambda x: str(x.get("id", "")),
            )
            return templates_sorted[0]
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise ValueError("No template found for category")


def _generate_prompts_for_category(
    *,
    category: str,
    tasks: List["AttackTask"],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    针对一批 tasks 生成 37 类中的某一类攻击 prompt。
    返回结果格式与 JailBreak_910B/server.py 一致：
    - 单条时使用 jailbreak_prompt
    - 多条时使用 jailbreak_prompt_list
    """
    results: List[Dict[str, Any]] = []
    baseline_config = _config_with_aliases(config)

    # --- 1) 明确走“算法型”的类别（生成多条） ---
    if category == "噪声注入":
        for task in tasks:
            lang_enum = _to_language_enum(task.Language)
            lang_upper = normalize_language_enum(task.Language)
            args_dict = dict(baseline_config)
            record = NoiseInjection_single_main(
                args_dict=args_dict,
                goal=task.Goal,
                target=task.Target or default_target_for_language(lang_upper),
                language=lang_enum,
            )
            prompts = [
                str(p).strip()
                for p in (record.get("all_generated_prompts") or [])
                if str(p).strip()
            ]
            if not prompts:
                continue
            results.append(
                {
                    "risk_id": task.RiskID,
                    "goal": task.Goal,
                    "jailbreak_prompt": "",
                    "jailbreak_prompt_list": prompts,
                    "language": task.Language,
                    "is_public": task.IsPublic,
                    "model_output": "",
                    "is_jailbroken": False,
                    "iteration": None,
                    "attack_method": category,
                }
            )
        return results

    if category == "同音词替换":
        # 可选启用敏感词拼音替换；否则 fallback 到模板类别
        if bool(config.get("use_sensitive_pinyin", False)):
            for task in tasks:
                lang_enum = _to_language_enum(task.Language)
                lang_upper = normalize_language_enum(task.Language)
                args_dict = dict(baseline_config)
                record = SensitivePinyin_single_main(
                    args_dict=args_dict,
                    goal=task.Goal,
                    target=task.Target or default_target_for_language(lang_upper),
                    language=lang_enum,
                )
                prompts = [
                    str(p).strip()
                    for p in (record.get("all_generated_prompts") or [])
                    if str(p).strip()
                ]
                if not prompts:
                    continue
                results.append(
                    {
                        "risk_id": task.RiskID,
                        "goal": task.Goal,
                        "jailbreak_prompt": "",
                        "jailbreak_prompt_list": prompts,
                        "language": task.Language,
                        "is_public": task.IsPublic,
                        "model_output": "",
                        "is_jailbroken": False,
                        "iteration": None,
                        "attack_method": category,
                    }
                )
            return results
        # 若未启用敏感词算法，则在后面统一走模板类别

    if category == "翻译模式":
        # 与 JailBreak_910B 对齐：默认尝试翻译服务；只有显式传 False 时回退模板。
        flag = config.get("use_translate_service", None)
        use_translate_service = True if flag is None else bool(flag)
        if use_translate_service:
            for task in tasks:
                lang_enum = _to_language_enum(task.Language)
                lang_upper = normalize_language_enum(task.Language)
                args_dict = dict(baseline_config)
                record = Translate_single_main(
                    args_dict=args_dict,
                    target_model=None,
                    goal=task.Goal,
                    target=task.Target or default_target_for_language(lang_upper),
                    language=lang_enum,
                )
                raw_adv = record.get("adv_prompt")
                prompts: List[str] = []
                if isinstance(raw_adv, list):
                    prompts = [str(p).strip() for p in raw_adv if str(p).strip()]
                elif isinstance(raw_adv, str) and raw_adv not in ("", "NULL"):
                    prompts = [raw_adv]
                if not prompts:
                    continue
                results.append(
                    {
                        "risk_id": task.RiskID,
                        "goal": task.Goal,
                        "jailbreak_prompt": "",
                        "jailbreak_prompt_list": prompts,
                        "language": task.Language,
                        "is_public": task.IsPublic,
                        "model_output": "",
                        "is_jailbroken": False,
                        "iteration": None,
                        "attack_method": category,
                    }
                )
            if results:
                return results
            # 如果翻译服务不可用或没有成功生成结果，则继续走下面的模板回退。

    # --- 2) 编码类：Code Chameleon / Caesar / ROT13 / Morse ---
    if category == "Code Chameleon 加密转换":
        for task in tasks:
            lang_enum = _to_language_enum(task.Language)
            lang_upper = normalize_language_enum(task.Language)
            args_dict = dict(baseline_config)
            record = CodeChameleon_single_main(
                args_dict=args_dict,
                goal=task.Goal,
                target=task.Target or default_target_for_language(lang_upper),
                language=lang_enum,
            )
            jb = str(record.get("adv_prompt", "") or "").strip()
            if not jb:
                continue
            results.append(
                {
                    "risk_id": task.RiskID,
                    "goal": task.Goal,
                    "jailbreak_prompt": jb,
                    "jailbreak_prompt_list": [],
                    "language": task.Language,
                    "is_public": task.IsPublic,
                    "model_output": "",
                    "is_jailbroken": False,
                    "iteration": None,
                    "attack_method": category,
                }
            )
        return results

    if category == "Caesar 加密":
        for task in tasks:
            lang_enum = _to_language_enum(task.Language)
            lang_upper = normalize_language_enum(task.Language)
            record = Caesar_single_main(
                args_dict=dict(baseline_config),
                goal=task.Goal,
                target=task.Target or default_target_for_language(lang_upper),
                language=lang_enum,
            )
            jb = str(record.get("adv_prompt", "") or "").strip()
            results.append(
                {
                    "risk_id": task.RiskID,
                    "goal": task.Goal,
                    "jailbreak_prompt": jb,
                    "jailbreak_prompt_list": [],
                    "language": task.Language,
                    "is_public": task.IsPublic,
                    "model_output": "",
                    "is_jailbroken": False,
                    "iteration": None,
                    "attack_method": category,
                }
            )
        return results

    if category == "rot13 编码":
        for task in tasks:
            lang_enum = _to_language_enum(task.Language)
            lang_upper = normalize_language_enum(task.Language)
            record = Rot13_single_main(
                args_dict=dict(baseline_config),
                goal=task.Goal,
                target=task.Target or default_target_for_language(lang_upper),
                language=lang_enum,
            )
            jb = str(record.get("adv_prompt", "") or "").strip()
            results.append(
                {
                    "risk_id": task.RiskID,
                    "goal": task.Goal,
                    "jailbreak_prompt": jb,
                    "jailbreak_prompt_list": [],
                    "language": task.Language,
                    "is_public": task.IsPublic,
                    "model_output": "",
                    "is_jailbroken": False,
                    "iteration": None,
                    "attack_method": category,
                }
            )
        return results

    if category == "摩斯电码":
        for task in tasks:
            lang_enum = _to_language_enum(task.Language)
            lang_upper = normalize_language_enum(task.Language)
            record = Morse_single_main(
                args_dict=dict(baseline_config),
                goal=task.Goal,
                target=task.Target or default_target_for_language(lang_upper),
                language=lang_enum,
            )
            jb = str(record.get("adv_prompt", "") or "").strip()
            results.append(
                {
                    "risk_id": task.RiskID,
                    "goal": task.Goal,
                    "jailbreak_prompt": jb,
                    "jailbreak_prompt_list": [],
                    "language": task.Language,
                    "is_public": task.IsPublic,
                    "model_output": "",
                    "is_jailbroken": False,
                    "iteration": None,
                    "attack_method": category,
                }
            )
        return results

    # --- 3) 其余类别：全部走模板筛选（通常为单条） ---
    for task in tasks:
        lang_code = _normalize_language_code(task.Language)
        tpl = _pick_template_for_category(
            preferred_lang_code=lang_code,
            category=category,
            config=config,
        )
        jb = _apply_template_text(str(tpl.get("template", "")), str(task.Goal or ""))
        results.append(
            {
                "risk_id": task.RiskID,
                "goal": task.Goal,
                "jailbreak_prompt": jb,
                "jailbreak_prompt_list": [],
                "language": task.Language,
                "is_public": task.IsPublic,
                "model_output": "",
                "is_jailbroken": False,
                "iteration": None,
                "attack_method": category,
            }
        )

    return results


def _cleanup_after_attack(attack_lower: str):
    """
    针对一次攻击方法调用后的统一清理逻辑：
    - 对使用本地 vLLM 的方法显式关闭模型引擎
    - 对使用 PyTorch 模型的方法删除模型引用并清理显存
    - 通用地触发 Python GC 与 CUDA 显存回收
    """
    try:
        # AutoDAN: 关闭本地变异模型引擎，释放显存
        if attack_lower == "autodan":
            try:
                # 1. 关闭 AutoDAN 专用的变异模型缓存
                from baseline.AutoDAN.AutoDAN_single_main import (
                    autodan_shutdown_mutators,
                    _AUTODAN_LOCAL_MUTATOR_VLLM_CACHE,
                )
                autodan_shutdown_mutators()
                
                # 2. 强制清理缓存字典
                _AUTODAN_LOCAL_MUTATOR_VLLM_CACHE.clear()
            except Exception:
                pass
            
            # 3. 额外清理：确保 global_vllm_models 中的所有模型都被关闭
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass
            
            # 4. 强制多次垃圾回收和显存清理
            try:
                import gc
                import torch
                for _ in range(5):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass

        # PAIR: 清理所有 vLLM 模型（attackLM, targetLM, evaluator）
        if attack_lower == "pair":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                # 关闭所有缓存的 vLLM 模型
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # TAP: 清理所有 vLLM 模型（evaluator）
        if attack_lower == "tap":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # PAP: 清理所有 vLLM 模型（攻击模型）
        if attack_lower == "pap":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # GCG: 删除 PyTorch 模型引用并清理显存
        if attack_lower == "gcg":
            try:
                import sys
                # 删除可能缓存的模型引用
                modules_to_check = ['main', 'baseline.GCG.GCG_single_main']
                for module_name in modules_to_check:
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        # 删除模型和分词器引用
                        for attr in ['target_model', 'target_tokenizer', 'model', 'tokenizer']:
                            if hasattr(module, attr):
                                try:
                                    delattr(module, attr)
                                except Exception:
                                    pass
            except Exception:
                pass

        # Actorattack: 关闭内部 vLLMTextGenerator 缓存的所有 vLLM EngineCore
        if attack_lower == "actorattack":
            # 注意：Actorattack 内部为了兼容单独运行，会把 baseline/Actorattack
            # 目录加入 sys.path，并通过顶层模块名 `llm_vllm` 导入。
            # 因此这里需要同时尝试两种导入路径，确保命中实际持有 _CACHED_MODELS 的模块。
            try:
                # 情况 1：通过包路径导入（baseline.Actorattack.llm_vllm）
                from baseline.Actorattack.llm_vllm import shutdown_all_vllm_models as _shutdown_pkg

                _shutdown_pkg()
            except Exception:
                pass
            try:
                # 情况 2：通过顶层模块名导入（llm_vllm）
                import llm_vllm  # type: ignore

                if hasattr(llm_vllm, "shutdown_all_vllm_models"):
                    llm_vllm.shutdown_all_vllm_models()
            except Exception:
                pass

        # GPTFuzz: 关闭本地变异模型缓存
        if attack_lower == "gptfuzz":
            try:
                from baseline.GPTFuzz.GPTFuzz_single_main import (
                    shutdown_all_gptfuzz_vllm,
                )

                shutdown_all_gptfuzz_vllm()
            except Exception:
                pass

        # JailBroken: 关闭缓存的 vLLM 攻击模型
        if attack_lower == "jailbroken":
            try:
                from baseline.JailBroken.llm import shutdown_all_jailbroken_vllm

                shutdown_all_jailbroken_vllm()
            except Exception:
                pass

        # MJP: 清理 vLLM 模型
        if attack_lower == "mjp":
            try:
                from baseline.JailBroken.llm import shutdown_all_jailbroken_vllm
                shutdown_all_jailbroken_vllm()
            except Exception:
                pass

        # AdvPrompter: 清理 vLLM 模型
        if attack_lower == "advprompter":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # AmpleGCG: 清理 vLLM 模型
        if attack_lower == "amplegcg":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # Translate: 清理 vLLM 模型
        if attack_lower == "translate":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # Template: 清理 vLLM 模型
        if attack_lower == "template":
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                for model_path in list(global_vllm_models.keys()):
                    try:
                        model = global_vllm_models[model_path]
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass
                global_vllm_models.clear()
            except Exception:
                pass

        # MultiJail: 清理 vLLM 模型（使用自己的 vLLM 实现）
        if attack_lower == "multijail":
            try:
                # MultiJail 使用自己的 LocalVLLM 类，需要删除实例引用
                import sys
                if 'baseline.MultiJail.MultiJail_single_main' in sys.modules:
                    module = sys.modules['baseline.MultiJail.MultiJail_single_main']
                    for attr in ['target_llm', 'model']:
                        if hasattr(module, attr):
                            try:
                                obj = getattr(module, attr)
                                if hasattr(obj, 'model') and hasattr(obj.model, 'shutdown'):
                                    obj.model.shutdown()
                                delattr(module, attr)
                            except Exception:
                                pass
            except Exception:
                pass

        # 通用清理：主动触发 GC 与 CUDA 显存回收
        # 这对所有方法都适用，放在最后确保所有引用都被删除后再清理
        try:
            import gc
            import torch

            # 保底清理：确保 global_vllm_models 中的所有模型都被关闭
            # 即使方法特定的清理失败，这里也会再尝试一次
            try:
                from GPTEvaluatorAgent.language_models import global_vllm_models
                if global_vllm_models:
                    for model_path in list(global_vllm_models.keys()):
                        try:
                            model = global_vllm_models[model_path]
                            if hasattr(model, 'shutdown'):
                                model.shutdown()
                        except Exception:
                            pass
                    global_vllm_models.clear()
            except Exception:
                pass

            # 多次执行 gc，确保所有循环引用被清理
            for _ in range(5):  # 从3次增加到5次，更彻底
                gc.collect()
            
            if torch.cuda.is_available():
                try:
                    # 同步所有 CUDA 设备
                    torch.cuda.synchronize()
                    # 清空所有 GPU 的缓存
                    torch.cuda.empty_cache()
                    # 再次同步确保清理完成
                    torch.cuda.synchronize()
                    # IPC 收集（用于多进程共享内存）
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    # 最后再清空一次缓存
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # 任何清理异常都不应打断接口返回
        pass


@app.get("/attack_method_list", response_model=MethodListResponse)
async def attack_method_list():
    """
    返回当前服务支持的攻击方法列表，供前端/调用方动态查询。

    成功响应:
    {
        "total": int,
        "attack_methods": [str, ...]
    }

    失败响应:
    {
        "error": str
    }
    """
    try:
        # 前端需要看到的攻击方法列表：
        # 1) 保留主项目原有 list：完整 baseline + Template 展开的中文小方法；
        # 2) 再补齐 JailBreak_910B 额外类别，避免破坏原有前端顺序和展示习惯。
        attack_methods: List[str] = list(
            dict.fromkeys(
                SUPPORTED_ATTACK_METHODS + TEMPLATE_SUBMETHODS + SUPPORTED_CATEGORIES
            )
        )
        return MethodListResponse(
            total=len(attack_methods),
            attack_methods=attack_methods,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# --- 3. 核心接口 ---
@app.post("/attack_batch")  # 对应后端的调用路径
async def handle_attack_batch(request: AttackRequest):
    temp_csv_path = None
    try:
        # 基本参数合法性检查：攻击方法是否受支持（兼容单个或多个）
        raw_method = request.AttackMethod
        if isinstance(raw_method, list):
            methods = [str(m).strip() for m in raw_method if str(m).strip()]
        else:
            m = str(raw_method).strip() if raw_method is not None else ""
            # 兼容字符串形式的 JSON 列表，例如 "[\"ICA\",\"SATA\"]"
            if m.startswith("[") and m.endswith("]"):
                try:
                    parsed = json.loads(m)
                    if isinstance(parsed, list):
                        methods = [str(x).strip() for x in parsed if str(x).strip()]
                    else:
                        methods = [m] if m else []
                except Exception:
                    methods = [m] if m else []
            else:
                methods = [m] if m else []

        if not methods:
            raise HTTPException(status_code=400, detail="AttackMethod is required.")

        # 允许两种类型的 key：
        # 1) 37 类攻击类别（中文类别名，或 legacy key 如 noise/translate）
        # 2) 原有基线攻击方法（AutoDAN/GCG/...）
        invalid_methods: List[str] = []
        for m in methods:
            lower = m.lower()
            mode, category = _resolve_requested_category(m)
            if (
                lower not in SUPPORTED_ATTACK_METHODS_SET
                and mode == "unknown"
            ):
                invalid_methods.append(m)
        if invalid_methods:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported attack method(s) {invalid_methods}. "
                    f"Supported category names are: {', '.join(SUPPORTED_CATEGORIES)}; "
                    f"baseline methods are: {', '.join(SUPPORTED_ATTACK_METHODS)}"
                ),
            )

        # Step 1: 将 Tasks 列表转换为 main.py 能读取的 CSV 文件（所有方法共用一次）
        # 我们只提取算法需要的列：goal, target, language
        csv_data = []
        for task in request.Tasks:
            lang_upper = normalize_language_enum(task.Language)
            # 若未提供 target，则根据语言给默认值（GCG 需要 target）
            target_text = (task.Target or "").strip()
            if not target_text:
                target_text = default_target_for_language(lang_upper)
            csv_data.append({
                "goal": task.Goal,
                "target": target_text,
                "language": lang_upper
            })
        
        df = pd.DataFrame(csv_data)
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w+')
        df.to_csv(temp_file.name, index=False)
        temp_csv_path = temp_file.name
        temp_file.close()

        # Step 2: 针对每个攻击方法依次运行主流程，并汇总结果
        final_results = []
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for method in methods:
            # 先判断是否属于“37 类攻击类别”体系
            mode, category = _resolve_requested_category(method)
            config = request.Config or {}

            if mode == "legacy_template":
                # 旧接口：不按类别过滤，直接随机挑选一个模板
                for task in request.Tasks:
                    lang_code = _normalize_language_code(task.Language)
                    tpl = _pick_template_for_category(
                        preferred_lang_code=lang_code,
                        category=None,
                        config=config,
                    )
                    jb = _apply_template_text(
                        str(tpl.get("template", "")), str(task.Goal or "")
                    )
                    final_results.append(
                        {
                            "risk_id": task.RiskID,
                            "goal": task.Goal,
                            "jailbreak_prompt": jb,
                            "jailbreak_prompt_list": [],
                            "language": task.Language,
                            "is_public": task.IsPublic,
                            "model_output": "",
                            "is_jailbroken": False,
                            "iteration": None,
                            "attack_method": method,
                        }
                    )
                # 一个请求里混用“类别 + 基线方法”时，类别部分不会走主流程
                continue

            if mode in ("category", "legacy_mapped"):
                assert category is not None
                cat_results = _generate_prompts_for_category(
                    category=category,
                    tasks=request.Tasks,
                    config=config,
                )
                if category == "翻译模式" and not cat_results:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "status": "error",
                            "error": "translation_service_unavailable",
                            "message": (
                                "Translate 子服务不可用或未生成翻译提示词。"
                                "请确认 LibreTranslate 已在 http://127.0.0.1:47891 运行。"
                            ),
                        },
                    )
                final_results.extend(cat_results)
                continue

            # ===========================
            # 下面是原有“基线攻击”逻辑
            # ===========================
            # Step 2.1: 准备参数 overrides（每个方法单独一份，避免相互污染）
            overrides = _config_with_aliases(config)

            # GCG：固定（或自动探测）本地模型路径，忽略外部 target_model
            attack_lower = method.lower()
            if attack_lower == "gcg":
                target_model_path = select_default_target_model_path(base_dir)
            else:
                # 其他方法：根据传入的模型名称做本地路径转接（保持兼容）
                model_selector = (request.TargetModel or "")
                model_lower = model_selector.lower()
                if "vicuna" in model_lower:
                    target_model_path = os.path.join(
                        base_dir, "llm_weights", "vicuna-13b-v1.5"
                    )
                elif "llama" in model_lower:
                    target_model_path = os.path.join(
                        base_dir, "llm_weights", "llama-2-7b-chat-hf"
                    )
                else:
                    target_model_path = (
                        request.TargetModel
                        or select_default_target_model_path(base_dir)
                    )

            overrides.update(
                {
                    # 与命令行 --attack 等价；此处统一使用规范化后的单个方法名
                    "attack": method,
                    "target_model_path": target_model_path,
                    "defense_type": request.DefenseType,
                    "exp_name": request.ExpName,
                    # 强制关闭 PyTorch 多进程 spawn 带来的部分问题，使用 API 时通常不需要 split
                    "data_split": False,
                    "instructions_path": temp_csv_path,
                }
            )

            # Step 2.2: 初始化参数并运行算法
            try:
                args = initialize_args(overrides=overrides, parse_cli=False)
                raw_results = run_attack_logic(args)
                if attack_lower == "translate":
                    has_translate_prompt = any(
                        (isinstance(item.get("adv_prompt"), list) and any(str(p).strip() for p in item.get("adv_prompt", [])))
                        or (isinstance(item.get("adv_prompt"), str) and item.get("adv_prompt", "").strip() not in ("", "NULL"))
                        or (isinstance(item.get("all_generated_prompts"), list) and any(str(p).strip() for p in item.get("all_generated_prompts", [])))
                        for item in raw_results
                    )
                    if not has_translate_prompt:
                        return JSONResponse(
                            status_code=503,
                            content={
                                "status": "error",
                                "error": "translation_service_unavailable",
                                "message": (
                                    "Translate 子服务不可用或未生成翻译提示词。"
                                    "请确认 LibreTranslate 已在 http://127.0.0.1:47891 运行。"
                                ),
                            },
                        )
            finally:
                # 无论攻击成功或失败，都执行一次资源清理
                _cleanup_after_attack(attack_lower)

            # Step 2.3: 结果缝合 (Re-mapping)
            for idx, item in enumerate(raw_results):
                # 获取对应的原始 Task 信息
                if idx >= len(request.Tasks):
                    break
                original_task = request.Tasks[idx]

                # === 结果字段整理 ===
                # 约定：
                # - jailbreak_prompt: 始终为字符串
                # - jailbreak_prompt_list: 数组；对于“天然多条攻击数据”的方法，使用该字段承载所有提示，jailbreak_prompt 置空
                # 兼容不同基线输出字段：
                # - adv_prompt 可能是 str / list / JSON-encoded list
                # - 某些方法使用 attack_prompt（如 TAP/PAIR/GPTFuzz）
                # - 部分方法使用 all_generated_prompts 承载所有候选 prompt

                raw_adv = item.get("adv_prompt")
                raw_attack_prompt = item.get("attack_prompt")
                all_generated = item.get("all_generated_prompts")

                # 1) 尝试收集“多条攻击提示”的列表
                jailbreak_prompt_list: List[str] = []

                # 1.1 all_generated_prompts 优先：AdvPrompter/AmpleGCG/ReNeLLM/SensitivePinyin/NoiseInjection 等
                if isinstance(all_generated, list):
                    for p in all_generated:
                        if isinstance(p, str):
                            p_str = p.strip()
                        else:
                            p_str = str(p).strip()
                        if p_str:
                            jailbreak_prompt_list.append(p_str)

                # 1.2 某些基线直接把列表放在 adv_prompt（如 Cipher/JailBroken/Template/Translate）
                if not jailbreak_prompt_list and isinstance(raw_adv, list):
                    for p in raw_adv:
                        if isinstance(p, str):
                            p_str = p.strip()
                        else:
                            p_str = str(p).strip()
                        if p_str:
                            jailbreak_prompt_list.append(p_str)

                # 1.3 某些方法会将列表编码成 JSON 字符串放在 adv_prompt（排除 Actorattack）
                # 对于 Actorattack，我们希望 adv_prompt 作为“单条字符串”返回给上游，
                # 因此不再在此处拆分成多条 jailbreak_prompt_list。
                if (
                    not jailbreak_prompt_list
                    and attack_lower != "actorattack"
                    and isinstance(raw_adv, str)
                    and raw_adv.strip().startswith("[")
                ):
                    try:
                        parsed = json.loads(raw_adv)
                        if isinstance(parsed, list):
                            for p in parsed:
                                p_str = str(p).strip()
                                if p_str:
                                    jailbreak_prompt_list.append(p_str)
                    except Exception:
                        # 解析失败则退回单条逻辑
                        pass

                # 2) 决定单条 jailbreak_prompt 字段
                if jailbreak_prompt_list:
                    # 多条攻击数据：全部放入 jailbreak_prompt_list，单条字段置空字符串
                    jailbreak_prompt = ""
                else:
                    # 单条攻击数据：与原有逻辑兼容，但强制转换为字符串
                    adv_prompt_val = raw_adv
                    # 如果 adv_prompt 是 list，取第一条作为代表
                    if isinstance(adv_prompt_val, list):
                        adv_prompt_val = adv_prompt_val[0] if adv_prompt_val else ""
                    # 若 adv_prompt 缺失或为 "NULL"，尝试使用 attack_prompt
                    if not adv_prompt_val or adv_prompt_val == "NULL":
                        adv_prompt_val = raw_attack_prompt
                        if isinstance(adv_prompt_val, list):
                            adv_prompt_val = (
                                adv_prompt_val[0] if adv_prompt_val else ""
                            )
                    # 若仍然为空，再尝试从 all_generated_prompts 中取第一条
                    if (
                        (not adv_prompt_val or adv_prompt_val == "NULL")
                        and isinstance(all_generated, list)
                        and len(all_generated) > 0
                    ):
                        adv_prompt_val = all_generated[0]

                    # 最终保证为字符串
                    if adv_prompt_val is None or adv_prompt_val == "NULL":
                        jailbreak_prompt = ""
                    else:
                        jailbreak_prompt = str(adv_prompt_val)

                # 尽量保持向后兼容；如果只生成 prompt，这些字段可能为空或缺失
                model_output = item.get("language_model_output", "")
                is_jb_val = item.get("is_JB", None)
                if isinstance(is_jb_val, str):
                    is_jb = is_jb_val.strip().lower() == "true"
                elif isinstance(is_jb_val, bool):
                    is_jb = is_jb_val
                else:
                    is_jb = False

                result_obj = {
                    "risk_id": original_task.RiskID,
                    "goal": item.get("original_prompt", original_task.Goal),
                    "jailbreak_prompt": jailbreak_prompt,
                    "jailbreak_prompt_list": jailbreak_prompt_list,
                    "language": original_task.Language,
                    "is_public": original_task.IsPublic,
                    "model_output": model_output,
                    "is_jailbroken": is_jb,
                    "iteration": item.get("attack_iterations", None),
                    "attack_method": method,
                }
                final_results.append(result_obj)

        # Step 3: 构造最终响应
        return {
            "status": "success",
            "attack_method": methods if len(methods) > 1 else methods[0],
            "result_count": len(final_results),
            "results": final_results,
            "message": "Attack finished successfully",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": str(e),
            "error_details": traceback.format_exc()
        }
    finally:
        # 清理临时 CSV
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=39002)
