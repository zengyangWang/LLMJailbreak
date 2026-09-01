import json
import os
from typing import Dict, List, Optional

from baseline.Template.config_template import TARGET_TEMPERATURE, TARGET_TOP_P
from baseline.Template.llm_template import TargetLLM_template
from data.language import Language
from utils.test_utils import test_prefixes


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def Template_initial(args_dict: dict):
    args = Args(args_dict)

    target_model_path = args.target_model_path
    tensor_parallel_size = args.tensor_parallel_size
    target_max_n_tokens = args.target_max_n_tokens
    target_temperature = TARGET_TEMPERATURE
    target_top_p = TARGET_TOP_P

    target_model = TargetLLM_template(
        model_path=target_model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_n_tokens=target_max_n_tokens,
        temperature=target_temperature,
        top_p=target_top_p,
    )
    return target_model


def _templates_file_for_language(language: Language) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if language == Language.CHINESE_SIMPLIFIED:
        return os.path.join(base_dir, "template_zh.json")
    else:
        return os.path.join(base_dir, "template_en.json")


def _load_templates(language: Language) -> List[Dict[str, str]]:
    path = _templates_file_for_language(language)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Templates file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Templates JSON must be a list, got: {type(data)}")
    cleaned: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("id", "")).strip()
        text = str(item.get("template", "")).strip()
        name = str(item.get("name", tid)).strip() or tid
        if tid and text:
            cleaned.append({"id": tid, "name": name, "template": text})
    return cleaned


def _apply_template_text(template_text: str, seed_prompt: str) -> str:
    """
    Replace placeholders used by the templates.
    Supports {query} and {encoded_query}.
    """
    seed = seed_prompt if seed_prompt is not None else ""
    result = template_text.replace("{query}", seed)
    result = result.replace("{encoded_query}", seed)
    return result


def _generate_template_prompts_for_goal(
    goal: str,
    language: Language,
) -> List[str]:
    """
    根据给定的 goal 和语言加载对应模板，并生成一组攻击提示词。
    逻辑与 JailBreak_910B 中的 generate_template_attack_rows 保持一致，
    但这里针对的是单个 goal，直接返回 prompt 列表，方便上层复用。
    """
    templates = _load_templates(language)
    generated_prompts: List[str] = []
    for tpl in templates:
        attack = _apply_template_text(tpl["template"], goal)
        generated_prompts.append(attack)
    return generated_prompts


def Template_single_main(
    args_dict: dict,
    target_model: Optional[TargetLLM_template],
    goal: str,
    target: str,
    language: Language,
):
    args = Args(args_dict)

    output_record = {
        "original_prompt": goal,
        "target": target,
        "attack_prompt": "NULL",
        "is_JB_Judge": None,
        "is_JB": None,
        "is_JB_Agent": None,
    }

    # 仅产出可用于输入 LLM 的 prompt，不实际调用目标模型、不评估成功与否
    generated_prompts = _generate_template_prompts_for_goal(goal=goal, language=language)

    # 汇总输出记录
    output_record["attack_iterations"] = len(generated_prompts)
    # 不做评估与打分
    output_record["is_JB"] = "None"
    # 让导出表多行展开：返回列表
    output_record["adv_prompt"] = generated_prompts if len(generated_prompts) > 0 else "NULL"
    # 不产出模型回复
    output_record["language_model_output"] = "NULL"

    return output_record


