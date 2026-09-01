import openai
import anthropic
import os
import time
import torch
from transformers import AutoConfig
import gc
from typing import Dict, List
import google.generativeai as palm
import requests
import json
import re
import emoji

from GPTEvaluatorAgent.common import _extract_json


def remove_code_blocks(text):
    pattern = r"```.*?```"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace="")
    model_response = remove_code_blocks(model_response)

    model_response = model_response.replace("```", "")

    return model_response


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self, prompts_list: List, max_n_tokens: int, temperature: float
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):

        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([self.tokenizer.encode("}")[1], 29913, 9092, 16675])


class ChatGPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = f""" <reason>Error</reason>  <score>Error</score> """
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 100
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "YOUR_KEY_HERE"}

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        url = self.url
        headers = self.headers
        # filter text
        for i in range(len(conv)):
            if "content" in conv[i]:
                conv[i]["content"] = text_process(conv[i]["content"])
        # for _ in range(self.API_MAX_RETRY):
        retry_num = 0
        while True:

            if retry_num >= self.API_MAX_RETRY:
                raise Exception("Retry too many times")
            retry_num += 1
            try:
                data = {
                    "model": self.model_name,
                    "messages": conv,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_n_tokens,
                    "request_timeout": self.API_TIMEOUT,
                }

                response = requests.post(url, headers=headers, data=json.dumps(data))
                print(f"""\n{'=' * 80}\n Response: {response}\n{'=' * 80}\n""")
                response_json = response.json()
                print(
                    f"""\n{'=' * 80}\n Response json: {response_json}\n{'=' * 80}\n"""
                )
                output = response_json["choices"][0]["message"]["content"]
                print(f"""\n{'=' * 80}\n Output: {output} \n{'=' * 80}\n""")
                if not re.search("<reason>.*?</reason>", output) or not re.search(
                    "<score>.*?</score>", output
                ):
                    out_dic = _extract_json(output)
                    if out_dic == None:
                        output = f""" <reason>Error</reason>  <score>1</score> """
                        print(
                            f"""\n{'=' * 80}\n Error Output: {output} \n{'=' * 80}\n"""
                        )
                    else:
                        reason = out_dic["reason"]
                        score = out_dic["score"]
                        output = (
                            f""" <reason>{reason}</reason>  <score>{score}</score> """
                        )
                        print(
                            f"""\n{'=' * 80}\n Dict Output: {output} \n{'=' * 80}\n"""
                        )
                break
            except Exception as e:
                print(f"\n\nException: {str(e)}")
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class Claude:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = anthropic.Anthropic(
            api_key=self.API_KEY,
        )

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of conversations
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class PaLM:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of dictionaries,
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv, temperature=temperature, top_p=top_p
                )
                output = completion.last

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[: (max_n_tokens * 4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(1)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_timeout=self.API_TIMEOUT,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


# ---- Local vLLM based judge model ----
from vllm import LLM as vllm
from vllm import SamplingParams

# 防止重复加载模型
global_vllm_models: Dict[str, vllm] = {}

# 统一限制本地 vLLM 的最大上下文长度，避免像 Qwen3 这类超长上下文模型在显存有限时
# 申请过大的 KV cache。默认 8192 token，对当前各攻击方法已经足够。
# 如需更长上下文，可通过环境变量 LOCAL_VLLM_MAX_MODEL_LEN 覆盖。
DEFAULT_VLLM_MAX_MODEL_LEN = int(os.getenv("LOCAL_VLLM_MAX_MODEL_LEN", "8192"))


class LocalVLLM(LanguageModel):
    API_TIMEOUT = 20

    def __init__(
        self,
        model_path,
        gpu_memory_utilization: float = 0.3,
        tensor_parallel_size: int = 1,
    ):
        """
        本地 vLLM 封装：
        - 默认将 gpu_memory_utilization 从 0.5/0.4 再降到 0.3，以降低显存占用，减少在多模型并行场景下的启动失败概率。
        - 在实例化前会根据当前 GPU 空闲显存自动下调 gpu_memory_utilization，确保
          free_mem >= gpu_memory_utilization * total_mem，从而避免 vLLM 抛出
          “Free memory on device ... is less than desired GPU memory utilization ...” 报错。
        - 若需要占用更多/更少显存，可在调用时显式传入 gpu_memory_utilization 覆盖默认值。
        """
        self.model_name = model_path
        # 根据当前空闲显存动态调整显存占用比例，避免启动阶段因为阈值略高而失败
        try:
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gib = free_bytes / (1024**3)
                total_gib = total_bytes / (1024**3)
                # 目标：gpu_memory_utilization 不超过 free/total，再预留一点冗余 (0.01)
                safe_ratio = free_gib / total_gib - 0.01
                # 给出一个下限，避免极端情况下为负或过小
                safe_ratio = max(0.05, safe_ratio)
                adjusted_ratio = min(float(gpu_memory_utilization), safe_ratio)
                if adjusted_ratio < gpu_memory_utilization:
                    print(
                        f"[LocalVLLM] Adjust gpu_memory_utilization from "
                        f"{gpu_memory_utilization} to {adjusted_ratio:.3f} "
                        f"(free {free_gib:.2f}/{total_gib:.2f} GiB)",
                        flush=True,
                    )
                gpu_memory_utilization = adjusted_ratio
        except Exception as e:
            # 任何探测/调整失败都不影响后续流程，回退到原始参数
            print(
                f"[LocalVLLM] Failed to probe CUDA memory, keep "
                f"gpu_memory_utilization={gpu_memory_utilization}: {e}",
                flush=True,
            )
        if model_path not in global_vllm_models:
            # 不得超过模型配置声明的上下文上限（例如 Vicuna-13B 为 4096）。
            model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            declared_max_len = getattr(model_config, "max_position_embeddings", None)
            effective_max_len = (
                min(DEFAULT_VLLM_MAX_MODEL_LEN, int(declared_max_len))
                if declared_max_len else DEFAULT_VLLM_MAX_MODEL_LEN
            )
            global_vllm_models[model_path] = vllm(
                model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="bfloat16",
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=effective_max_len,
            )
        self.model = global_vllm_models[model_path]

    def batched_generate(
        self,
        prompts_list: List[str],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else 0.0,
            top_p=top_p if top_p is not None else 1.0,
            max_tokens=max_n_tokens,
        )
        results = self.model.generate(
            prompts_list, sampling_params=sampling_params, use_tqdm=False
        )
        outputs = [r.outputs[0].text for r in results]
        return outputs

    def shutdown(self):
        """
        释放底层 vLLM 引擎占用的显存，并从全局缓存中移除。
        """
        try:
            backend = getattr(self, "model", None)
            if backend is not None and hasattr(backend, "shutdown"):
                try:
                    backend.shutdown()
                except Exception:
                    # 忽略底层关闭过程中的非致命异常
                    pass
        finally:
            try:
                if self.model_name in global_vllm_models:
                    del global_vllm_models[self.model_name]
            except Exception:
                pass
