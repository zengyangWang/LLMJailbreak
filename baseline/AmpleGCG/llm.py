import torch
import gc
from pathlib import Path
from vllm import LLM as vllm
from vllm import SamplingParams
import emoji
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm
from utils.string_utils import compose_system_prompt, language_code
from FastChat.fastchat.model.model_adapter import get_conversation_template


def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace="")
    model_response = model_response.replace("```", "")
    return model_response


class AmpleLLM:
    def __init__(self, args):
        self.device = "cuda:0"
        self.num_beams = args.num_beams
        self.max_new_tokens = args.ample_max_new_tokens
        self.min_new_tokens = args.ample_min_new_tokens
        self.diversity_penalty = args.ample_diversity_penalty
        repo_root = Path(__file__).resolve().parents[2]
        model_dirs = {
            "llama2": "AmpleGCG-llama2-sourced-llama2-7b-chat",
            "vicuna": "AmpleGCG-llama2-sourced-vicuna-7b",
        }
        if args.attack_source not in model_dirs:
            raise ValueError(
                f"Unsupported AmpleGCG attack_source: {args.attack_source!r}. "
                f"Expected one of: {sorted(model_dirs)}"
            )
        self.atk_model_path = str(repo_root / "llm_weights" / model_dirs[args.attack_source])
        if not Path(self.atk_model_path).is_dir():
            raise FileNotFoundError(
                f"AmpleGCG local attack model not found: {self.atk_model_path}. "
                "Download the corresponding osunlp AmpleGCG model into LLMJailbreak/llm_weights."
            )

        self.attack_llm = AutoModelForCausalLM.from_pretrained(
            self.atk_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.attack_tokenzier = AutoTokenizer.from_pretrained(self.atk_model_path, local_files_only=True)

        self.attack_tokenzier.padding_side = "left"

        if not self.attack_tokenzier.pad_token:
            self.attack_tokenzier.pad_token = self.attack_tokenzier.eos_token
        gen_kwargs = {
            "pad_token_id": self.attack_tokenzier.pad_token_id,
            "eos_token_id": self.attack_tokenzier.eos_token_id,
            "bos_token_id": self.attack_tokenzier.bos_token_id,
        }

        gen_config = {
            "do_sample": True,            
            "temperature": 1.0,            
            "top_p": 0.9,                 
            
            "num_beams": 1,                
            "diversity_penalty": 0.0,       
            "num_beam_groups": 1,           
            
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "num_return_sequences": self.num_beams, 
        }
        self.gen_config = GenerationConfig(**gen_kwargs, **gen_config)

    def generate_suffix(self, goal):
        prompt = "### Query:{goal} ### Prompt:"
        max_len = getattr(self.attack_tokenzier, "model_max_length", 512)
        if not isinstance(max_len, int) or max_len <= 0:
            max_len = 512
        max_len = min(max_len, 512)
        input_ids = self.attack_tokenzier(
            prompt.format(goal=goal),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.device)
        output = self.attack_llm.generate(
            **input_ids, generation_config=self.gen_config
        )
        output = output[:, input_ids["input_ids"].shape[-1] :]
        adv_suffixes = self.attack_tokenzier.batch_decode(
            output, skip_special_tokens=True
        )
        return adv_suffixes

    def shutdown(self):
        """
        释放 HF 模型占用的 CUDA 显存，供 nohup 长服务复用。
        """
        try:
            if hasattr(self, "attack_llm") and self.attack_llm is not None:
                try:
                    # 将权重迁回 CPU，便于 CUDA 快速回收
                    self.attack_llm.to("cpu")
                except Exception:
                    pass
        finally:
            try:
                del self.attack_llm
            except Exception:
                pass
            try:
                del self.attack_tokenzier
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

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalVLLM(LLM):
    def __init__(
        self,
        args,
        gpu_memory_utilization=0.3,
        test_generation_kwargs=None,
        suffix_dict={},
    ):
        super().__init__()
        self.model_path = args.target_model_path
        # 仅生成 Prompt 模式：不初始化 vLLM 引擎以避免显存占用
        self.model = None

        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.suffix_dict = suffix_dict
        self.batch_size = 50

    def generate(self, args, pert_goal, language=None):
        # 仅生成 Prompt 生成器模式：不再进行目标模型推理，直接返回后缀列表
        suffix = self.suffix_dict[args.test_data_idx]
        # 保留原始推理逻辑（已移除），以便必要时回滚参考
        # outputs = []
        # for bi in tqdm(range(0, len(suffix), self.batch_size), desc="Batch Generation"):
        #     batch_suffix = suffix[bi : bi + self.batch_size]
        #     def _join_seed_and_suffix(seed: str, suf: str) -> str:
        #         if not seed:
        #             return suf
        #         if not suf:
        #             return seed
        #         return seed + ("" if suf[:1].isspace() else " ") + suf
        #     batch_pert_goals = [_join_seed_and_suffix(pert_goal, s) for s in batch_suffix]
        #     try:
        #         lang = language if language is not None else getattr(args, "current_language", None)
        #         full_prompts = []
        #         for p in batch_pert_goals:
        #             conv = get_conversation_template(self.model_path)
        #             if getattr(conv, "name", "") == "llama-2" and getattr(conv, "sep2", None):
        #                 conv.sep2 = conv.sep2.strip()
        #             final_system = compose_system_prompt(getattr(conv, "system_message", ""), lang)
        #             conv.set_system_message(final_system)
        #             conv.append_message(conv.roles[0], p)
        #             conv.append_message(conv.roles[1], None)
        #             full_prompts.append(conv.get_prompt())
        #         batch_pert_goals = full_prompts
        #     except Exception:
        #         pass
        #     response = self.model.generate(
        #         prompts=batch_pert_goals, sampling_params=self.sampling_params
        #     )
        #     batch_response = [r.outputs[0].text for r in response]
        #     outputs.extend(batch_response)
        # return outputs, suffix
        return [], suffix
