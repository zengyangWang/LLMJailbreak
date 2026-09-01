from __future__ import annotations

from typing import Iterable, List, Sequence, Union, Tuple, Dict

from vllm import LLM as VLLM
from vllm import SamplingParams


_CACHED_MODELS: Dict[Tuple[str, int, float], VLLM] = {}


def _get_or_create_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    *,
    max_num_seqs: int = 32,
    dtype: str = "float16",
    trust_remote_code: bool = True,
) -> VLLM:
    """
    Reuse loaded vLLM models whenever possible to reduce GPU memory footprint.
    """
    cache_key = (model_path, tensor_parallel_size, gpu_memory_utilization)
    if cache_key not in _CACHED_MODELS:
        _CACHED_MODELS[cache_key] = VLLM(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
        )
    return _CACHED_MODELS[cache_key]


def shutdown_all_vllm_models():
    """
    显式关闭 Actorattack 内部使用的所有 vLLM 引擎实例，并清空缓存。
    - 调用底层 VLLM.shutdown（若存在）
    - 删除 _CACHED_MODELS 中的引用，让 CUDA 显存得以回收
    """
    global _CACHED_MODELS
    for key, model in list(_CACHED_MODELS.items()):
        try:
            if hasattr(model, "shutdown"):
                model.shutdown()
        except Exception:
            # 清理失败不应中断主流程
            pass
    _CACHED_MODELS.clear()


class VLLMTextGenerator:
    """
    Minimal text generation wrapper around vLLM for the JailBroken attack.
    """

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.6,
        max_num_seqs: int = 32,
        dtype: str = "float16",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.model = _get_or_create_model(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompt_list: List[str] = [prompts]
        else:
            prompt_list = list(prompts)
        if not prompt_list:
            return []
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(0.0, temperature),
            top_p=max(0.0, min(top_p, 1.0)),
        )
        results = self.model.generate(
            prompt_list,
            sampling_params=sampling_params,
        )
        outputs = []
        for res in results:
            if not res.outputs:
                outputs.append("")
            else:
                outputs.append(res.outputs[0].text)
        return outputs

    def generate_one(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        outputs = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0] if outputs else ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        tokenizer = self.model.get_tokenizer()
        if not tokenizer.chat_template:
            # Fallback template (Simple User/Assistant) which works for many models like Vicuna
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
            
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Error applying chat template: {e}. Using manual fallback.")
            prompt = ""
            for msg in messages:
                if msg['role'] == 'user':
                    prompt += f"USER: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    prompt += f"ASSISTANT: {msg['content']}\n"
            prompt += "ASSISTANT:"

        return self.generate_one(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
