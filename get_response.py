from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd  # 用于读取和合并 Excel
from tqdm.auto import tqdm  # 进度条
from glob import glob  # 用于匹配 defense*.xlsx
from pathlib import Path
import argparse

###############################################################################
# 模型相关配置：在这里集中管理 4 个待测试模型
###############################################################################

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "models" / "bench_models"

# 已通过 modelscope 下载好的本地目录结构如下：
# - models/bench_models/Qwen/Qwen3-14B
# - models/bench_models/Qwen/Qwen3-32B
# - models/bench_models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# - models/bench_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# 因此这里直接使用「本地路径」加载模型，不再走远程仓库。
MODEL_CONFIGS = {
    "qwen3-14b": {
        "local_subdir": "Qwen/Qwen3-14B",
    },
    "qwen3-32b": {
        "local_subdir": "Qwen/Qwen3-32B",
    },
    "deepseek-r1-llama-8b": {
        "local_subdir": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    },
    "deepseek-r1-qwen-14b": {
        "local_subdir": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    },
}

# 全局模型与 tokenizer（由 load_model_and_tokenizer 填充）
model = None
tokenizer = None


def load_model_and_tokenizer(model_key: str, device: str = "cuda:6"):
    """
    根据给定 key 加载指定模型和 tokenizer。
    - model_key: MODEL_CONFIGS 的键，例如 "qwen3-14b"
    - device:   例如 "cuda:5" 或 "auto"
    """
    global model, tokenizer

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"未知模型标识 {model_key}，可选值：{list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_key]
    local_dir = MODEL_ROOT / cfg["local_subdir"]

    if not local_dir.exists():
        raise FileNotFoundError(f"本地模型目录不存在：{local_dir}")

    # 先加载 tokenizer（直接从本地目录）
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
    )

    # 再加载模型（直接从本地目录）
    model_local = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,   # 可以是 "cuda:5" 或 "auto"
    ).eval()

    # PyTorch 2.x 加速编译（如果可用）
    if hasattr(torch, "compile"):
        model_local = torch.compile(model_local)

    model = model_local

def get_batch_response(prompts):
    """
    批量生成回答
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():     # 关闭反向传播
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=False,         # 关闭采样可加速
            use_cache=True,
        )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()         # 主动释放显存
    return results
def merge_excel_files(file_paths, output_path):
    """
    读取多个 Excel 并按行合并，去重后保存到 output_path
    """
    dfs = []
    for p in file_paths:
        df = pd.read_excel(p)
        dfs.append(df)

    if len(dfs) == 0:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates()
    merged.to_excel(output_path, index=False)
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="针对指定模型在 Excel 数据集上批量生成回答")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-32b",
        choices=list(MODEL_CONFIGS.keys()),
        help=f"选择要使用的模型，可选：{list(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "test_data_20260108_quchong.xlsx"),
        help="输入数据集 Excel 路径（默认使用 test_data_20260108_quchong.xlsx）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出结果 Excel 路径（默认在 input 同目录生成 *_<model>_response.xlsx）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:6",
        help='模型加载设备，例如 "cuda:5"、"cuda:0" 或 "auto"',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="推理批大小",
    )
    return parser.parse_args()

# # 将 L47-51 中的路径转换为列表并执行合并
# file_paths = [
#     "/mnt/data/shuhuizhen/product/get_response/jailbreak_result_translate.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/jailbreak_result_template.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/jailbreak_result_sensitive_pinyin.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/jailbreak_result_noise.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/jailbreak_result_code_chameleon_caesar_rot13_morse.xlsx",
# ]
# file_paths = [p for p in glob("/mnt/data/shuhuizhen/product/get_response/defense*.xlsx")]
# file_paths = [
#     "/mnt/data/shuhuizhen/product/get_response/defense_None_defense__vicuna-13b-v1.5__attack_MJP__260102_1155_12_attack_results.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/defense_None_defense__vicuna-13b-v1.5__attack_Coldattack__260102_1237_40_attack_results.xlsx",
#     "/mnt/data/shuhuizhen/product/get_response/defense_None_defense__vicuna-13b-v1.5__attack_DrAttack__251229_2337_17_attack_results.xlsx",

# ]
if __name__ == "__main__":
    args = parse_args()

    # 1) 确定输入 / 输出路径
    input_path = args.input
    in_path = Path(input_path)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = in_path.suffix or ".xlsx"
        stem = in_path.stem
        output_path = in_path.with_name(f"{stem}_{args.model}_response{suffix}")

    # 2) 读取数据集：
    #    - 如果已有输出文件，则在其基础上“断点续跑”
    #    - 否则从原始输入文件开始
    if output_path.exists():
        print(f"检测到已存在输出文件，将在其基础上继续生成：{output_path}")
        df = pd.read_excel(output_path)
    else:
        df = pd.read_excel(input_path)

    print(df.head())

    if "prompt" not in df.columns:
        raise ValueError("Excel 中缺少 'prompt' 列：请确认文件内容。")

    # 如果还没有 response 列，先创建一列空值
    if "response" not in df.columns:
        df["response"] = None

    # 找出还没有生成 response 的行（NaN 或 空字符串）
    mask = df["response"].isna() | (df["response"].astype(str).str.strip() == "")
    todo_indices = df.index[mask].tolist()

    if not todo_indices:
        print("所有样本均已存在 response，无需重新生成。")
        print(f"结果文件位置：{output_path}")
        raise SystemExit(0)

    print(f"共有 {len(df)} 条样本，其中 {len(todo_indices)} 条需要生成/补全 response。")

    # 3) 加载指定模型（只在确实有需要生成的样本时才加载）
    print(f"加载模型：{args.model}（device={args.device}）")
    load_model_and_tokenizer(args.model, device=args.device)

    # 4) 按需分批推理并清理结果，每个 batch 都落盘，支持中断后续跑
    batch_size = args.batch_size
    total_batches = (len(todo_indices) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(
        tqdm(
            range(0, len(todo_indices), batch_size),
            total=total_batches,
            desc="Processing batches",
            unit="batch",
        ),
        start=1,
    ):
        batch_ids = todo_indices[start : start + batch_size]
        batch_prompts = []
        for idx in batch_ids:
            p = df.at[idx, "prompt"]
            if pd.isna(p):
                p = ""
            batch_prompts.append(str(p))

        if not batch_prompts:
            continue

        print(f"Processing batch {batch_idx}/{total_batches}, size={len(batch_prompts)}")
        batch_out = get_batch_response(batch_prompts)

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 保证上一批次计算完成（仅在有 GPU 时调用）

        cleaned = []
        for p, r in zip(batch_prompts, batch_out):
            if isinstance(r, str):
                cleaned.append(r[len(p) :].strip() if r.startswith(p) else r.strip())
            else:
                cleaned.append(str(r))

        # 写回到对应行
        for idx, resp in zip(batch_ids, cleaned):
            df.at[idx, "response"] = resp

        # 每个 batch 都保存一次，支持中断后续跑
        df.to_excel(output_path, index=False)
        print(
            f"已完成 {batch_idx}/{total_batches} 个 batch，"
            f"累计处理 {start + len(batch_ids)} / {len(todo_indices)} 条未完成样本，"
            f"中间结果已保存到：{output_path}"
        )

    print(f"全部完成，最终结果已写入：{output_path}")
