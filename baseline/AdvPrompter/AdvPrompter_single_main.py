import gc
import os
import numpy as np
import torch.nn as nn
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random
from utils.test_utils import test_prefixes
from baseline.AdvPrompter.llm import AdvPrompter
import torch


class Args:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def AdvPrompter_initial(args_dict):
    args = Args(args_dict)
    return args


def AdvPrompter_generate_suffix(args, pert_goals):
    test_generation_kwargs = {
        "max_tokens": args.adv_prompter_max_new_tokens,
        "temperature": 0.6,
    }
    # 降低 vLLM 预留显存占比，避免因显存不足初始化失败
    gpu_util = getattr(args, "adv_prompter_gpu_memory_utilization", 0.32)
    attack_llm = AdvPrompter(
        args,
        gpu_memory_utilization=gpu_util,
        test_generation_kwargs=test_generation_kwargs,
    )
    try:
        suffix_dict = {}
        for goal_idx in range(len(pert_goals[args.test_data_idx : args.end_index])):
            pert_goal_i = pert_goals[args.test_data_idx + goal_idx]
            suffix = attack_llm.generate_suffix(pert_goal_i)
            suffix_dict[args.test_data_idx + goal_idx] = suffix
        return suffix_dict
    finally:
        # 显式关闭 vLLM 引擎并回收显存，确保 nohup 服务持续运行
        try:
            attack_llm.shutdown()
        except Exception:
            pass
        # 删除对象引用并做兜底清理
        try:
            del attack_llm
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


def AdvPrompter_single_main(args_dict, target_model, goal, target, language):
    # initialize
    args = Args(args_dict)

    # 1) 获取后缀列表（忽略输出）
    _outputs_ignored, suffix_list = target_model.generate(args, goal)

    # 2) 批量拼接 Goal + Suffix (+按需空格)
    all_prompts = []
    for suf_i in suffix_list:
        if goal and suf_i and not suf_i[:1].isspace():
            base_prompt = goal + " " + suf_i
        else:
            base_prompt = goal + suf_i
        final_prompt = base_prompt
        all_prompts.append(final_prompt)

    # 3) 返回全部生成的 Prompt
    return {"all_prompts": all_prompts}
