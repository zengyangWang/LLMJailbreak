from __future__ import annotations

from typing import Iterable, List, Sequence, Union, Tuple, Dict

from vllm import LLM as VLLM
from vllm import SamplingParams


_CACHED_MODELS: Dict[Tuple[str, int, float], VLLM] = {}


def _get_or_create_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    *,
    max_num_seqs: int = 32,
    dtype: str = "bfloat16",
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
        )
    return _CACHED_MODELS[cache_key]


def shutdown_all_jailbroken_vllm():
    """
    关闭 JailBroken 攻击中缓存的所有 vLLM 引擎，并清空全局缓存。
    """
    global _CACHED_MODELS
    for _, model in list(_CACHED_MODELS.items()):
        try:
            if hasattr(model, "shutdown"):
                model.shutdown()
        except Exception:
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
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_num_seqs: int = 32,
        dtype: str = "bfloat16",
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
        )

    def shutdown(self) -> None:
        """
        显式关闭底层 vLLM 引擎（若未通过全局函数关闭），并从缓存中移除当前模型实例。
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "shutdown"):
                    try:
                        self.model.shutdown()
                    except Exception:
                        pass
        finally:
            try:
                # 从全局缓存中移除
                key = (
                    self.model_path,
                    int(self.tensor_parallel_size),
                    float(self.gpu_memory_utilization),
                )
                if key in _CACHED_MODELS:
                    del _CACHED_MODELS[key]
            except Exception:
                pass

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_tokens: int = 256,
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
        max_tokens: int = 256,
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

