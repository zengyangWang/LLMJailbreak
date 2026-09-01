import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from FastChat.fastchat.model.model_adapter import get_conversation_template

from data.language import Language
from baseline.JailBroken.llm import VLLMTextGenerator
from baseline.JailBroken.mutation.MJPChoices import MJPChoices
from baseline.JailBroken.mutation.mutation_base import (
    Instance as MutationInstance,
    JailbreakDataset,
)
from utils.test_utils import test_prefixes


def _language_key(language: Language) -> str:
    name = getattr(language, "name", str(language)).upper()
    if "CHINESE" in name:
        return "cn"
    return "en"


@dataclass
class Args:
    mjp_template_path: str
    mjp_prompt_type: str
    mjp_batch_num: int
    mjp_max_new_tokens: int
    mjp_temperature: float
    mjp_top_p: float
    mjp_gpu_memory_utilization: float
    tensor_parallel_size: int
    target_model_path: str
    template_name: str

    def __init__(self, args_dict: Dict[str, Any]) -> None:
        for key, value in args_dict.items():
            setattr(self, key, value)
        if not getattr(self, "mjp_template_path", None):
            self.mjp_template_path = str(
                Path(__file__).resolve().parent.parent
                / "JailBroken"
                / "mutation"
                / "mjp_template.json"
            )
        if not getattr(self, "mjp_prompt_type", None):
            self.mjp_prompt_type = "JQ+COT+MC"
        if not getattr(self, "mjp_batch_num", None):
            self.mjp_batch_num = 1
        if not getattr(self, "mjp_max_new_tokens", None):
            self.mjp_max_new_tokens = getattr(self, "target_max_n_tokens", 256)
        if not getattr(self, "mjp_temperature", None):
            self.mjp_temperature = getattr(self, "target_temperature", 0.7)
        if not getattr(self, "mjp_top_p", None):
            self.mjp_top_p = getattr(self, "target_top_p", 0.9)
        if not getattr(self, "mjp_gpu_memory_utilization", None):
            self.mjp_gpu_memory_utilization = 0.3
        # 为 vLLM 显存占用比例设置一个合理下限，避免 13B Vicuna 仅能装下权重而
        # 没有剩余空间用于 KV cache，从而触发
        # "No available memory for the cache blocks" 这类错误。
        # 在 80GiB GPU 上，权重大约占用 24GiB，0.3*80GiB ≈ 24GiB，会导致可用
        # KV cache 显存为负；这里将下限提升到 0.4（≈32GiB），既能容纳权重，又能
        # 留出一定 KV cache 空间。
        try:
            min_util = 0.4
            if self.mjp_gpu_memory_utilization < min_util:
                self.mjp_gpu_memory_utilization = min_util
        except Exception:
            # 任何异常都不影响后续逻辑
            pass


class TargetModelAdapter:
    """
    Thin wrapper to adapt the vLLM generator to the interface expected by
    ``MJPChoices`` (single-string generate calls).
    """

    def __init__(
        self,
        generator: VLLMTextGenerator,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self._generator = generator
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

    def generate(self, prompt, clear_old_history: bool = True) -> str:
        if isinstance(prompt, list):
            prompt_text = "\n".join(str(item) for item in prompt)
        else:
            prompt_text = str(prompt)
        return self._generator.generate_one(
            prompt_text,
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )


def _load_templates(path: str) -> Dict[str, List[str]]:
    template_path = Path(path)
    if template_path.is_file():
        try:
            with template_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    # fallback minimal structure
    return {"en": [], "cn": []}


def _build_conversation(template_name: str):
    conv = get_conversation_template(template_name)
    try:
        conv = conv.copy()
        if hasattr(conv, "reset"):
            conv.reset()
        else:
            conv.messages = []
        return conv
    except AttributeError:
        pass


def MJP_initial(
    args_dict: Dict[str, Any]
) -> Tuple[Args, TargetModelAdapter, VLLMTextGenerator, Dict[str, List[str]]]:
    args = Args(args_dict)
    generator = VLLMTextGenerator(
        args.target_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.mjp_gpu_memory_utilization,
    )
    adapter = TargetModelAdapter(
        generator=generator,
        max_new_tokens=args.mjp_max_new_tokens,
        temperature=args.mjp_temperature,
        top_p=args.mjp_top_p,
    )
    templates = _load_templates(args.mjp_template_path)
    return args, adapter, generator, templates


def MJP_single_main(
    args_dict: Dict[str, Any],
    adapter: TargetModelAdapter,
    generator: VLLMTextGenerator,
    templates: Dict[str, List[str]],
    goal: str,
    target: str,
    language: Language,
) -> Dict[str, Any]:
    args = Args(args_dict)
    language_key = _language_key(language)

    conv = _build_conversation(args.template_name)

    instance = MutationInstance(
        query=goal,
        language=language_key,
        attack_attrs={"conversation": conv},
    )
    dataset = JailbreakDataset([instance])

    mutator = MJPChoices(args.mjp_prompt_type, adapter)
    mutated_dataset = mutator(dataset)
    mutated_instance = mutated_dataset[0]
    conversation = mutated_instance.attack_attrs["conversation"]

    final_prompt = conversation.get_prompt()

    best_result = None
    for iteration in range(1, args.mjp_batch_num + 1):
        response = generator.generate_one(
            final_prompt,
            max_tokens=args.mjp_max_new_tokens,
            temperature=args.mjp_temperature,
            top_p=args.mjp_top_p,
        )
        lowered = response.lower()
        jailbroken = not any(prefix.lower() in lowered for prefix in test_prefixes)
        if target and target.strip():
            jailbroken = jailbroken or (target.lower() in lowered)
        if jailbroken:
            best_result = (response, iteration)
            break
        if best_result is None:
            best_result = (response, iteration)

    model_output, iteration = best_result if best_result else ("", 0)
    is_jb = False
    if model_output:
        lowered_output = model_output.lower()
        is_jb = not any(prefix.lower() in lowered_output for prefix in test_prefixes)
        if target and target.strip():
            is_jb = is_jb or (target.lower() in lowered_output)

    return {
        "adv_prompt": final_prompt,
        "language_model_output": model_output,
        "attack_iterations": iteration,
        "is_JB": bool(is_jb),
    }

