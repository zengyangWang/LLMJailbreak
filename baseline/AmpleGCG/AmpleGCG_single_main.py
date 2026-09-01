from utils.test_utils import test_prefixes
from baseline.AmpleGCG.llm import AmpleLLM
import gc
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


def AmpleGCG_initial(args_dict):
    args = Args(args_dict)
    return args  # , attackLM, targetLM


def AmpleGCG_generate_suffix(args, pert_goals):
    attack_llm = AmpleLLM(args)
    try:
        suffix_dict = {}
        for goal_idx in range(len(pert_goals[args.test_data_idx : args.end_index])):
            pert_goal_i = pert_goals[args.test_data_idx + goal_idx]
            suffix = attack_llm.generate_suffix(pert_goal_i)
            suffix_dict[args.test_data_idx + goal_idx] = suffix
        return suffix_dict
    finally:
        # 显式释放 HF 模型占用的显存，保证服务长期运行稳定
        try:
            attack_llm.shutdown()
        except Exception:
            pass
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


def AmpleGCG_single_main(args_dict, target_model, goal, target, language):
    # initialize
    args = Args(args_dict)

    # 1) 生成后缀（保持原样）：从 target_model 拿到后缀列表；忽略输出列表
    _outputs_ignored, suffix_list = target_model.generate(args, goal, language)

    # 2) 批量处理：拼接 Goal + Suffix (+空格)
    all_prompts = []
    for suf_i in suffix_list:
        # 按需在 goal 与 suffix 之间插入空格
        if goal and suf_i and not suf_i[:1].isspace():
            base_prompt = goal + " " + suf_i
        else:
            base_prompt = goal + suf_i
        final_prompt = base_prompt
        all_prompts.append(final_prompt)

    # 保序去重：相同的随机采样结果只作为一个候选 prompt 返回。
    all_prompts = list(dict.fromkeys(all_prompts))

    # 兼容保留：旧版越狱检测与输出（废弃，已移除推理）
    # is_JB_flag = False
    # jb_suffix, jb_output, jb_iter = "", "", 0
    # for i in range(len(_outputs_ignored)):
    #     c_output = _outputs_ignored[i]
    #     is_JB_flag = not any([prefix.lower() in c_output.lower() for prefix in test_prefixes])
    #     if is_JB_flag:
    #         jb_suffix = all_prompts[i]
    #         jb_output = c_output
    #         jb_iter = i
    #         break

    # 3) 新返回格式：返回所有生成的最终 Prompt
    return {"all_prompts": all_prompts}
