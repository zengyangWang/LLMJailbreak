from typing import Dict, List

from utils.test_utils import test_prefixes  # 保留引用（与其他 baseline 对齐，尽管本方法不再使用）


def ReNeLLM_single_main(args_dict, target_model, goal, target, language):
    """
    Generation Only 控制器：
    - 调用 target_model.generate(args, goal, language)
    - 直接返回生成的越狱提示（按 language 选择模板，无额外语言指令拼接）
    - 返回: {"all_prompts": List[str]}
    """
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = Args(args_dict)

    # 1) 从目标模型获取 ReNeLLM 生成的提示列表（忽略输出列表）
    _outputs_ignored, prompt_list = target_model.generate(args, goal, language)

    # 2) 直接返回生成的 prompt（语言由内部模板自适应，无需额外后缀）
    final_prompts: List[str] = [p or "" for p in prompt_list]

    # 3) 返回统一字典格式
    return {"all_prompts": final_prompts}


