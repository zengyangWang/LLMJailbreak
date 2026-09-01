"""
Stage 2: 收集 Stage 1 生成的攻击 prompt，用 4 个目标模型批量获取回复
---------------------------------------------------------------------------
功能：
  1. 扫描 Stage 1 输出的 exp_results/advbench_pipeline/ 目录下所有 .xlsx 文件
  2. 提取其中的 jailbreak_prompt（攻击后的 prompt）
  3. 对 4 个目标模型分别推理，结果合并保存到 advbench_pipeline/stage2_responses/
  4. 支持断点续跑（检测已存在的 response 列）

用法：
  python advbench_pipeline/stage2_get_responses.py --device cuda:0 --batch_size 4
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

# 把项目根目录加入 sys.path，便于复用 get_response.py 的函数
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from get_response import MODEL_CONFIGS, MODEL_ROOT, load_model_and_tokenizer  # noqa: E402

# ============================================================
# 配置
# ============================================================
STAGE1_RESULT_DIR = REPO_ROOT / "exp_results" / "advbench_pipeline"
STAGE2_OUTPUT_DIR = REPO_ROOT / "advbench_pipeline" / "stage2_responses"
STAGE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_METHODS = ["GCG", "AmpleGCG", "PAIR", "AutoDAN", "JailBroken", "Template"]

# Stage 1 每种攻击的 exp_name 与对应 attack 方法映射（与 stage1 脚本保持一致）
EXP_NAME_MAP = {
    "advbench_GCG": "GCG",
    "advbench_AmpleGCG": "AmpleGCG",
    "advbench_PAIR": "PAIR",
    "advbench_AutoDAN": "AutoDAN",
    "advbench_JailBroken": "JailBroken",
    "advbench_Template": "Template",
}

# jailbreak_prompt 可能在不同列名下
PROMPT_COL_CANDIDATES = ["jailbreak_prompt", "adv_prompt", "attack_prompt", "prompt"]
# 原始问题列候选
GOAL_COL_CANDIDATES = ["prompt", "goal", "seed_prompt", "query", "original_prompt"]


def _pick_col(df: pd.DataFrame, candidates: list) -> str | None:
    col_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    return None


def collect_attack_prompts() -> pd.DataFrame:
    """
    扫描 Stage 1 输出目录，合并所有攻击方法的 xlsx 结果，
    返回包含以下列的 DataFrame：
        attack_method, original_prompt, jailbreak_prompt
    """
    all_rows = []

    for xlsx_path in sorted(STAGE1_RESULT_DIR.rglob("*_attack_results.xlsx")):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"  [WARN] 读取失败 {xlsx_path}: {e}")
            continue

        prompt_col = _pick_col(df, PROMPT_COL_CANDIDATES)
        goal_col = _pick_col(df, GOAL_COL_CANDIDATES)
        attack_col = _pick_col(df, ["attack_method"])

        if prompt_col is None:
            print(f"  [WARN] 找不到 prompt 列，跳过 {xlsx_path.name}")
            continue

        seen_originals = set()  # 每个 xlsx 内，每个原始 prompt 只取第 1 个变体
        for _, row in df.iterrows():
            jailbreak_prompt = str(row[prompt_col]).strip()
            if not jailbreak_prompt or jailbreak_prompt in ("NULL", "nan", "None", ""):
                continue
            original_prompt = str(row[goal_col]).strip() if goal_col else ""
            attack_method = str(row[attack_col]).strip() if attack_col else xlsx_path.stem
            # 去重：同一攻击方法下，同一原始 prompt 只保留第 1 条
            dedup_key = original_prompt
            if dedup_key in seen_originals:
                continue
            seen_originals.add(dedup_key)
            all_rows.append(
                {
                    "attack_method": attack_method,
                    "original_prompt": original_prompt,
                    "jailbreak_prompt": jailbreak_prompt,
                    "source_file": xlsx_path.name,
                }
            )

    df_all = pd.DataFrame(all_rows)
    print(f"\n[Stage 2] 共收集到 {len(df_all)} 条攻击 prompt（{df_all['attack_method'].nunique()} 种攻击方法）")
    if len(df_all) > 0:
        print(df_all["attack_method"].value_counts().to_string())
    return df_all


def get_batch_response_local(prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    """
    使用当前全局 model / tokenizer 批量推理，返回仅包含新生成内容的字符串列表。
    """
    import get_response as gr  # 引用全局 model/tokenizer

    tok = gr.tokenizer
    mdl = gr.model

    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # 只取新生成的部分
    new_tokens = outputs[:, input_len:]
    results = tok.batch_decode(new_tokens, skip_special_tokens=True)

    del inputs, outputs, new_tokens
    torch.cuda.empty_cache()
    return [r.strip() for r in results]


def run_model(
    model_key: str,
    df_prompts: pd.DataFrame,
    device: str,
    batch_size: int,
) -> None:
    """
    对指定模型跑完所有攻击 prompt，结果保存到 STAGE2_OUTPUT_DIR/<model_key>.csv
    支持断点续跑。
    """
    out_path = STAGE2_OUTPUT_DIR / f"{model_key}.csv"

    # 断点续跑：读取已有结果
    if out_path.exists():
        print(f"\n[{model_key}] 检测到已有结果，尝试断点续跑: {out_path}")
        df_out = pd.read_csv(out_path)
    else:
        df_out = df_prompts.copy()
        df_out["response"] = None
        df_out["target_model"] = model_key

    # 找出还未生成 response 的行
    mask = df_out["response"].isna() | (df_out["response"].astype(str).str.strip() == "")
    todo_idx = df_out.index[mask].tolist()

    if not todo_idx:
        print(f"[{model_key}] 所有 {len(df_out)} 条 response 已存在，跳过推理。")
        return

    print(f"\n[{model_key}] 加载模型（device={device}）...")
    load_model_and_tokenizer(model_key, device=device)

    total = len(todo_idx)
    print(f"[{model_key}] 共需推理 {total} 条（已有 {len(df_out) - total} 条）")

    for start in tqdm(range(0, total, batch_size), desc=f"{model_key}", unit="batch"):
        batch_idx = todo_idx[start : start + batch_size]
        batch_prompts = [str(df_out.at[i, "jailbreak_prompt"]) for i in batch_idx]

        responses = get_batch_response_local(batch_prompts, max_new_tokens=256)

        for i, resp in zip(batch_idx, responses):
            df_out.at[i, "response"] = resp
            df_out.at[i, "target_model"] = model_key

        # 每 batch 落盘
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[{model_key}] 完成！结果: {out_path}")

    # 释放显存
    import get_response as gr
    try:
        del gr.model
        gr.model = None
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()


def merge_all_results() -> pd.DataFrame:
    """合并所有模型的推理结果为一个宽表，保存到 stage2_all_responses.csv"""
    dfs = []
    for model_key in MODEL_CONFIGS:
        p = STAGE2_OUTPUT_DIR / f"{model_key}.csv"
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
    if not dfs:
        print("[Stage 2] 没有找到任何结果文件。")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged_path = STAGE2_OUTPUT_DIR / "stage2_all_responses.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\n[Stage 2] 合并完成！共 {len(merged)} 条记录，保存至: {merged_path}")
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: 批量获取目标模型回复")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="指定要运行的模型，默认运行全部 4 个模型",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备，如 cuda:0")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批大小")
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="仅收集攻击 prompt 并保存，不运行推理",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: 收集 Stage 1 的攻击 prompt
    df_prompts = collect_attack_prompts()

    if len(df_prompts) == 0:
        print("[ERROR] 未找到任何攻击 prompt，请先运行 stage1_run_attacks.sh")
        sys.exit(1)

    # 保存收集到的攻击 prompts（便于排查）
    prompts_save_path = STAGE2_OUTPUT_DIR / "collected_attack_prompts.csv"
    df_prompts.to_csv(prompts_save_path, index=False, encoding="utf-8-sig")
    print(f"[Stage 2] 收集的攻击 prompt 已保存到: {prompts_save_path}")

    if args.collect_only:
        print("[Stage 2] --collect_only 模式，不运行推理，退出。")
        sys.exit(0)

    # Step 2: 对每个目标模型依次推理
    for model_key in args.models:
        print(f"\n{'='*50}")
        print(f"  目标模型: {model_key}")
        print(f"{'='*50}")
        run_model(
            model_key=model_key,
            df_prompts=df_prompts,
            device=args.device,
            batch_size=args.batch_size,
        )

    # Step 3: 合并所有模型结果
    merge_all_results()

    print("\n[Stage 2] 全部完成！")
