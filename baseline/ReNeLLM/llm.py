from typing import List, Optional, Tuple

import torch
import gc

from utils.utils import load_model_and_tokenizer
from baseline.ReNeLLM.renellm_core import ReNeLLMGenerator


class AttackHFModel:
    """
    轻量 HF 文本生成封装：提供 generate(prompt, max_new_tokens, temperature) -> str。
    仅用于 ReNeLLM 的“重写”阶段，不做复杂采样控制。
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # 尝试设置 pad/eos，避免 generate 报错
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            try:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            except Exception:
                pass

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        sequences = getattr(output_ids, "sequences", output_ids)
        if not getattr(self.model.config, "is_encoder_decoder", False):
            input_len = inputs["input_ids"].shape[1]
            sequences = sequences[:, input_len:]
        text = self.tokenizer.decode(
            sequences[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text.strip()

    def shutdown(self) -> None:
        """
        释放 HF 模型与 CUDA 显存。
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
        finally:
            try:
                del self.model
            except Exception:
                pass
            try:
                del self.tokenizer
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception:
                    pass


class LocalReNeLLM:
    """
    ReNeLLM 适配器（Generation Only）：
    - 使用 AttackHFModel 作为重写模型
    - 基于 ReNeLLMGenerator 生成越狱提示集合
    - 不执行目标模型推理，返回 ([], adv_prompts_list)
    """

    def __init__(
        self,
        args,
        test_generation_kwargs: Optional[dict] = None,  # 保留以兼容现有加载逻辑（未使用）
    ) -> None:
        # 1) 加载用于“重写/嵌套”的攻击模型
        attack_model_path = getattr(args, "attack_model_path", None) or getattr(
            args, "rewrite_model", None
        )
        if not attack_model_path:
            raise ValueError("ReNeLLM 需要 attack_model_path 或 rewrite_model 参数。")

        device = f"cuda:{getattr(args, 'device_id', 0)}"
        atk_model, atk_tokenizer = load_model_and_tokenizer(
            attack_model_path, device=device
        )
        self.attack_model = AttackHFModel(atk_model, atk_tokenizer)

        # 2) 初始化 ReNeLLM 核心引擎（语言在 generate 时动态覆盖）
        self.generator = ReNeLLMGenerator(
            attack_model=self.attack_model,
            language="en",
            max_new_tokens=int(getattr(args, "rewrite_max_new_tokens", 192)),
            temperature=float(getattr(args, "rewrite_temperature", 0.7)),
        )

        # 3) 进化次数（生成轮数）
        self.evo_max = int(getattr(args, "renellm_evo_max", 20))

    def _normalize_lang(self, language: Optional[str]) -> str:
        s = str(language or "en").lower()
        # 兼容多种中文标记
        if any(
            key in s
            for key in ["zh", "chinese", "cn", "chinese_simplified", "简体", "中文"]
        ):
            return "zh"
        return "en"

    def generate(
        self, args, goal: str, language: Optional[str] = "en"
    ) -> Tuple[List[str], List[str]]:
        """
        仅生成 Prompt（不做推理）：
        - 返回 ([], adv_prompts_list)
        """
        lang_code = self._normalize_lang(language)
        self.generator.language = lang_code  # 动态切换模板语言

        adv_prompts: List[str] = []
        for _ in range(self.evo_max):
            prompt_variant = self.generator.run(goal)
            adv_prompts.append(prompt_variant)

        # 兼容调用方签名：输出列表为空
        return [], adv_prompts

    def shutdown(self) -> None:
        """
        显式释放内部 HF 攻击模型占用的显存。
        """
        try:
            if hasattr(self, "attack_model") and self.attack_model is not None:
                if hasattr(self.attack_model, "shutdown"):
                    try:
                        self.attack_model.shutdown()
                    except Exception:
                        pass
        finally:
            try:
                del self.attack_model
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

