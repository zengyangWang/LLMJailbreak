import base64
import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Any

from data.language import Language
from baseline.JailBroken.llm import VLLMTextGenerator
from baseline.JailBroken.mutation.mutation_base import Instance
from baseline.JailBroken.mutation.Artificial import Artificial
from baseline.JailBroken.mutation.Auto_obfuscation import Auto_obfuscation
from baseline.JailBroken.mutation.Auto_payload_splitting import Auto_payload_splitting
from baseline.JailBroken.mutation.Base64 import Base64
from baseline.JailBroken.mutation.Base64_input_only import Base64_input_only
from baseline.JailBroken.mutation.Base64_raw import Base64_raw
from baseline.JailBroken.mutation.Combination_1 import Combination_1
from baseline.JailBroken.mutation.Combination_2 import Combination_2
from baseline.JailBroken.mutation.Combination_3 import Combination_3
from baseline.JailBroken.mutation.Disemvowel import Disemvowel
from baseline.JailBroken.mutation.Leetspeak import Leetspeak
from baseline.JailBroken.mutation.Rot13 import Rot13


_TEMPLATE_PATH = Path(__file__).resolve().with_name("jailbroken_template.json")
with _TEMPLATE_PATH.open("r", encoding="utf-8") as fp:
    _DEFAULT_TEMPLATES: Dict[str, List[Dict[str, str]]] = json.load(fp)

_BASE64_TEMPLATE_KEYS = {
    "base64_output_only",
}


def _language_key(language: Language) -> str:
    name = getattr(language, "name", str(language)).upper()
    if "CHINESE" in name:
        return "cn"
    return "en"


def _format_query(template_key: str, raw_query: str) -> str:
    if template_key in _BASE64_TEMPLATE_KEYS:
        return base64.b64encode(raw_query.encode("utf-8")).decode("utf-8")
    return raw_query


def _fill_template(template: str, query_text: str) -> str:
    return template.replace("{query}", query_text)


@dataclass
class Args:
    jailbroken_template_path: str
    jailbroken_temperature: float
    jailbroken_top_p: float
    jailbroken_max_new_tokens: int
    tensor_parallel_size: int
    attack_model_path: str
    target_model_path: str
    jailbroken_gpu_memory_utilization: float
    target_max_n_tokens: int
    target_temperature: float
    target_top_p: float

    def __init__(self, args_dict: Dict[str, Any]) -> None:
        for key, value in args_dict.items():
            setattr(self, key, value)
        # Defaults fall back to general target decoding parameters when unset.
        if not hasattr(self, "jailbroken_template_path") or not self.jailbroken_template_path:
            self.jailbroken_template_path = str(_TEMPLATE_PATH)
        if not hasattr(self, "jailbroken_temperature"):
            self.jailbroken_temperature = getattr(self, "target_temperature", 0.7)
        if not hasattr(self, "jailbroken_top_p"):
            self.jailbroken_top_p = getattr(self, "target_top_p", 0.9)
        if not hasattr(self, "jailbroken_max_new_tokens"):
            self.jailbroken_max_new_tokens = getattr(self, "target_max_n_tokens", 256)
        if not hasattr(self, "jailbroken_gpu_memory_utilization"):
            self.jailbroken_gpu_memory_utilization = 0.5
        # 对 vLLM 显存占用比例设置一个合理的下限，避免 13B Vicuna 仅能装下权重而
        # 没有剩余空间用于 KV cache，从而触发
        # "No available memory for the cache blocks" 这类错误。
        # 在 80GiB GPU 上，权重大约占用 24GiB，0.3*80GiB 约等于 24GiB，
        # 会导致可用 KV cache 显存为负。这里将下限提升到 0.4（≈32GiB），
        # 既能容纳权重，又能留出一定 KV cache 空间。
        try:
            min_util = 0.4
            if self.jailbroken_gpu_memory_utilization < min_util:
                self.jailbroken_gpu_memory_utilization = min_util
        except Exception:
            # 任何异常都不影响后续逻辑，保持兼容性
            pass


@lru_cache(maxsize=None)
def _load_template_store(path: str) -> Dict[str, List[Dict[str, str]]]:
    template_path = Path(path)
    if template_path.is_file():
        try:
            with template_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return dict(_DEFAULT_TEMPLATES)


def _collect_templates(
    language_key: str,
    template_store: Dict[str, List[Dict[str, str]]],
    fallback: str = "en",
) -> List[Tuple[str, str]]:
    if language_key in template_store:
        entries = template_store[language_key]
    else:
        entries = template_store.get(fallback, [])
    formatted = []
    for entry in entries:
        for key, value in entry.items():
            formatted.append((key, value))
    return formatted


def _prepare_prompts(
    query: str,
    language_key: str,
    template_store: Dict[str, List[Dict[str, str]]],
) -> List[Tuple[str, str]]:
    mutations = []
    for key, template in _collect_templates(language_key, template_store):
        mutated_query = _format_query(key, query)
        prompt = _fill_template(template, mutated_query)
        mutations.append((key, prompt))
    return mutations


def _normalize_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return _normalize_text(value[0]) if value else ""
    return str(value)


def _format_instance_prompt(instance: Instance) -> str:
    template = getattr(instance, "jailbreak_prompt", None) or "{query}"
    query_text = _normalize_text(getattr(instance, "query", ""))
    return template.replace("{query}", query_text)


def _build_mutations(attack_model: VLLMTextGenerator) -> List[Any]:
    mutations: List[Any] = [
        Artificial(),
        Base64(),
        Base64_input_only(),
        Base64_raw(),
        Combination_1(),
        Combination_2(),
        Combination_3(),
        Disemvowel(),
        Leetspeak(),
        Rot13(),
    ]
    # 需要模型生成改写的策略仅在可用时启用
    if attack_model is not None:
        mutations.extend(
            [
                Auto_obfuscation(model=attack_model),
                Auto_payload_splitting(model=attack_model),
            ]
        )
    return mutations


def _collect_mutation_prompts(
    goal: str, language_key: str, attack_model: VLLMTextGenerator
) -> List[str]:
    base_instance = Instance(query=goal.strip(), language=language_key)
    prompts: List[str] = []
    seen = set()

    # 保留原始提示词，便于参考
    if base_instance.query:
        prompts.append(base_instance.query)
        seen.add(base_instance.query)

    for mutation in _build_mutations(attack_model):
        try:
            mutated_dataset = mutation([base_instance.copy()])
        except Exception:
            continue
        for inst in mutated_dataset:
            prompt_text = _format_instance_prompt(inst).strip()
            if prompt_text and prompt_text not in seen:
                prompts.append(prompt_text)
                seen.add(prompt_text)
    return prompts


def JailBroken_initial(args_dict: Dict[str, Any]) -> Tuple[Args, VLLMTextGenerator]:
    args = Args(args_dict)
    attack_model_path = getattr(args, "attack_model_path", args.target_model_path)
    attack_llm = VLLMTextGenerator(
        attack_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.jailbroken_gpu_memory_utilization,
    )
    return args, attack_llm


def JailBroken_single_main(
    args_dict: Dict[str, Any],
    attack_model: VLLMTextGenerator,
    goal: str,
    target: str,
    language: Language,
) -> Dict[str, Any]:
    args = Args(args_dict)
    language_key = _language_key(language)

    prompts = _collect_mutation_prompts(goal, language_key, attack_model)
    if not prompts:
        prompts = [goal.strip()]

    return {
        "adv_prompt": prompts,
        "language_model_output": "",
        "attack_iterations": len(prompts),
        "is_JB": False,
    }

