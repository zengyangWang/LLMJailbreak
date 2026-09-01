from vllm import LLM as vllm
from vllm import SamplingParams
import emoji
from tqdm import tqdm
import gc
import torch


def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace="")
    model_response = model_response.replace("```", "")
    return model_response


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
        gpu_memory_utilization=0.4,
        test_generation_kwargs=None,
        suffix_dict={},
    ):
        super().__init__()
        self.model_path = args.target_model_path
        # 仅生成 Prompt 模式：不初始化 vLLM 引擎，避免显存与时间开销
        self.model = None

        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.suffix_dict = suffix_dict
        self.batch_size = 50

    def generate(self, args, pert_goal):
        # 仅生成 Prompt：返回后缀列表，跳过目标模型推理
        suffix = self.suffix_dict[args.test_data_idx]
        return [], suffix


class AdvPrompter:
    def __init__(
        self,
        args,
        gpu_memory_utilization=0.32,
        test_generation_kwargs=None,
        suffix_dict={},
    ):
        super().__init__()
        self.model_path = args.adv_prompter_model_path
        self.batch_size = 50
        self.model = vllm(
            self.model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            max_num_seqs=self.batch_size,
        )

        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.max_iters = args.adv_prompter_max_iters

    def generate_suffix(self, goal):
        prompts = [goal] * self.max_iters
        response = self.model.generate(
            prompts=prompts, sampling_params=self.sampling_params
        )
        output = [r.outputs[0].text for r in response]
        return output

    def shutdown(self):
        """
        释放 vLLM 引擎与 CUDA 显存，确保服务长时间运行时无残留占用。
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                # vLLM 提供显式关闭方法
                if hasattr(self.model, "shutdown"):
                    self.model.shutdown()
        except Exception:
            # 即便关闭失败也继续做 Python 与 CUDA 的清理
            pass
        finally:
            try:
                # 删除引用并清理缓存
                del self.model
            except Exception:
                pass
            # 触发 Python 垃圾回收与 CUDA 显存回收
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    # 进一步回收共享内存句柄
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    def __del__(self):
        # 兜底：对象被回收时再次尝试释放资源
        try:
            self.shutdown()
        except Exception:
            pass
