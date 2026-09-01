import requests
import json
import numpy as np
import time
import pandas as pd
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parent


def _safe_json_str(v):
    if v is None:
        return ""
    # pandas 可能给出 float('nan')
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def evaluate_response(prompt, response):
    """
    调用本地安全评估服务，对 (prompt, response) 进行风险评估。
    返回: has_issue, risk_zhs, scores, score
    """
    url = "http://localhost:8080/api/validations/text/sync"

    headers = {
        "Content-Type": "application/json",
        "APIKey": "firewall-api-key-2024",
    }

    payload = [
        {
            "prompt": _safe_json_str(prompt),
            "response": _safe_json_str(response),
        }
    ]

    resp = requests.post(
        url,
        headers=headers,
        json=payload,  # 等价于 --post-data + application/json
        timeout=30,
    )

    # 状态码
    print("status:", resp.status_code)

    # 解析结果
    data = resp.json()
    has_issue = data.get("hasIssue", False)
    scores = [
        risk.get("score")
        for task in data.get("tasks", [])
        for risk in task.get("risks", [])
        if risk.get("score") is not None
    ]
    risk_zhs = [
        risk.get("risk_zh")
        for task in data.get("tasks", [])
        for risk in task.get("risks", [])
        if risk.get("risk_zh") is not None
    ]

    print("risk_zhs:", risk_zhs)
    print("hasIssue:", has_issue)
    print("scores:", scores)

    score = float(np.mean(scores)) if scores else 0.0
    print("score:", score)
    time.sleep(0.05)
    return has_issue, risk_zhs, scores, score


def parse_args():
    parser = argparse.ArgumentParser(
        description="对包含 prompt/response 的 Excel 结果文件进行安全评估（支持断点续跑）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            REPO_ROOT
            / "data"
            / "test_data_20260108_quchong_qwen3-32b_response.xlsx"
        ),
        help="输入结果文件（已包含 prompt/jailbreak_prompt 和 response 列）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="评估结果输出路径（默认在 input 同目录生成 *_evaluated.xlsx，且支持断点续跑）",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="每处理多少条样本保存一次中间结果",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = input_path.suffix or ".xlsx"
        stem = input_path.stem
        output_path = input_path.with_name(f"{stem}_evaluated{suffix}")

    # 1) 读取数据：如果已有评估结果文件，则在其基础上断点续跑
    if output_path.exists():
        print(f"检测到已存在评估结果文件，将在其基础上续跑：{output_path}")
        df = pd.read_excel(output_path)
    else:
        df = pd.read_excel(input_path)

    print(df.head())

    # 2) 确定 prompt 列：兼容 'jailbreak_prompt' 或 'prompt'
    if "jailbreak_prompt" in df.columns:
        prompt_col = "jailbreak_prompt"
    elif "prompt" in df.columns:
        prompt_col = "prompt"
    else:
        raise ValueError("Excel 中既没有 'jailbreak_prompt' 列，也没有 'prompt' 列，无法评估。")

    if "response" not in df.columns:
        raise ValueError("Excel 中缺少 'response' 列：请确认先用 get_response.py 生成回答。")

    # 3) 初始化 / 检查评估结果列，显式设置合适的 dtype，避免后续写入 list/bool 出错
    # has_issue: 使用 object，允许存放 bool/None
    if "has_issue" not in df.columns:
        df["has_issue"] = pd.Series([pd.NA] * len(df), dtype="object")
    else:
        df["has_issue"] = df["has_issue"].astype("object")

    # risk_zhs / scores: 一行一个 list，用 object 存放
    for col in ["risk_zhs", "scores"]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")
        else:
            df[col] = df[col].astype("object")

    # score: 单个浮点数
    if "score" not in df.columns:
        df["score"] = np.nan
    else:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # 4) 找出尚未评估的行（以 has_issue 是否为 NaN 为准）
    mask_todo = df["has_issue"].isna()
    todo_indices = df.index[mask_todo].tolist()

    if not todo_indices:
        print("所有样本均已完成评估，无需重复调用。")
        print(f"评估结果文件位置：{output_path}")
        raise SystemExit(0)

    print(f"共有 {len(df)} 条样本，其中 {len(todo_indices)} 条尚未评估。")

    # 5) 逐条调用评估服务，支持中断续跑
    save_every = max(1, int(args.save_every))
    processed = 0

    try:
        for i, idx in enumerate(todo_indices, start=1):
            row = df.loc[idx]
            prompt = row[prompt_col]
            response = row["response"]

            h, rz, scs, s = evaluate_response(prompt, response)
            df.at[idx, "has_issue"] = h
            df.at[idx, "risk_zhs"] = rz
            df.at[idx, "scores"] = scs
            df.at[idx, "score"] = s

            processed += 1

            # 每隔 save_every 条保存一次
            if processed % save_every == 0 or i == len(todo_indices):
                df.to_excel(output_path, index=False)
                print(
                    f"已评估 {processed}/{len(todo_indices)} 条待评估样本，"
                    f"中间结果已保存到：{output_path}"
                )

    except KeyboardInterrupt:
        # 捕获中断，保存当前进度
        df.to_excel(output_path, index=False)
        print("\n检测到手动中断，已保存当前进度。")
        print(f"当前评估结果已写入：{output_path}")
        raise SystemExit(1)

    print(f"全部评估完成，最终结果已写入：{output_path}")
