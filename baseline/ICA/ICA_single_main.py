import os
import json
from typing import Dict, Any


class Args:
    def __init__(self, args_dict: Dict[str, Any]):
        for key, value in args_dict.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        attributes = []
        for key, value in self.__dict__.items():
            attributes.append(f"{key}={getattr(self, key)}")
        return ", ".join(attributes)


def ICA_initial(args_dict: Dict[str, Any]):
    """
    初始化 ICA 参数对象（对标 AmpleGCG 的样式）。
    """
    args = Args(args_dict)
    return args


def _load_ica_dataset() -> Dict[str, Any]:
    """
    加载与本文件同目录下的 ica_dataset.json。
    使用绝对路径拼接，确保在不同工作目录下也能正确读取。
    """
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "ica_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def ICA_single_main(args_dict: Dict[str, Any], target_model, goal: str, target: str, language) -> Dict[str, Any]:
    """
    仅基于少量示例（In-Context/Few-shot）拼接最终的攻击 Prompt。
    不进行模型推理（Generation Only）。
    """
    # 初始化参数对象
    args = Args(args_dict)

    # 读取 few-shot 数据
    dataset = _load_ica_dataset()

    # 语言选择：包含 "CHINESE" 使用中文数据，否则使用英文数据
    language_name = getattr(language, "name", str(language)).upper()
    lang_key = "zh" if "CHINESE" in language_name else "en"
    data_list = dataset.get(lang_key, [])
    if not data_list:
        # 兜底：若指定语言列表为空，则回退到英文
        data_list = dataset.get("en", [])

    # 读取 n_shots，默认 5
    n_shots = getattr(args, "ica_n_shots", 5)
    try:
        n_shots = int(n_shots)
    except Exception:
        n_shots = 5
    if n_shots < 0:
        n_shots = 0

    # 构造 few-shot 提示词
    prompt_blocks = []
    if len(data_list) > 0 and n_shots > 0:
        for i in range(n_shots):
            example = data_list[i % len(data_list)]
            example_q = example.get("q", "")
            example_a = example.get("a", "")
            prompt_blocks.append(f"User: {example_q}\nAssistant: {example_a}")

    # 拼接最终目标
    prompt_blocks.append(f"User: {goal}\nAssistant:")
    final_prompt = "\n".join(prompt_blocks)

    # 返回单次 Prompt 的结果（不做推理）
    return {
        "adv_prompt": final_prompt,
        "language_model_output": "SKIPPED (Generation Only)",
        "is_JB": False,
    }


