"""
Stage 2 单模型推理脚本 -- 每张卡单独运行，支持并行
用法:
  CUDA_VISIBLE_DEVICES=4 python advbench_pipeline/stage2_single_model.py \
      --model deepseek-r1-llama-8b --device cuda:0 --batch_size 8
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from get_response import MODEL_CONFIGS, load_model_and_tokenizer  # noqa: E402

PROMPTS_CSV   = REPO_ROOT / "advbench_pipeline" / "stage2_responses" / "collected_attack_prompts.csv"
RESPONSES_DIR = REPO_ROOT / "advbench_pipeline" / "stage2_responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["attack_method", "original_prompt", "jailbreak_prompt",
          "source_file", "response", "target_model"]


def load_base_prompts():
    with open(PROMPTS_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _init_rows(model_key, base_rows):
    rows = []
    for br in base_rows:
        row = {k: br.get(k, "") for k in FIELDS}
        row["response"] = ""
        row["target_model"] = model_key
        rows.append(row)
    return rows


def load_or_init_output(model_key, base_rows):
    out_path = RESPONSES_DIR / f"{model_key}.csv"
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(base_rows):
            print(f"[WARN] {out_path.name} 行数({len(rows)}) != base({len(base_rows)}), 重新初始化")
            rows = _init_rows(model_key, base_rows)
    else:
        rows = _init_rows(model_key, base_rows)
    return out_path, rows


def save(out_path, rows):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def get_batch_response(prompts, max_new_tokens=256):
    import get_response as gr
    tok = gr.tokenizer
    mdl = gr.model

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    inputs = tok(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048)
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = mdl.generate(**inputs, max_new_tokens=max_new_tokens,
                               do_sample=False, use_cache=True)

    new_tokens = outputs[:, input_len:]
    results = tok.batch_decode(new_tokens, skip_special_tokens=True)
    del inputs, outputs, new_tokens
    torch.cuda.empty_cache()
    return [r.strip() for r in results]


def run(model_key, device, batch_size):
    base_rows = load_base_prompts()
    out_path, rows = load_or_init_output(model_key, base_rows)

    todo_idx = [i for i, r in enumerate(rows)
                if not r.get("response", "").strip()
                or r["response"] in ("nan", "None", "NULL")]

    if not todo_idx:
        print(f"[{model_key}] 所有 {len(rows)} 条已完成，跳过。")
        return

    print(f"[{model_key}] 共 {len(rows)} 条，待推理 {len(todo_idx)} 条")
    print(f"[{model_key}] 加载模型 (device={device})...")
    load_model_and_tokenizer(model_key, device=device)

    for start in tqdm(range(0, len(todo_idx), batch_size),
                      desc=model_key, unit="batch"):
        batch_idx = todo_idx[start: start + batch_size]
        prompts = [rows[i]["jailbreak_prompt"] for i in batch_idx]
        responses = get_batch_response(prompts)
        for i, resp in zip(batch_idx, responses):
            rows[i]["response"] = resp
            rows[i]["target_model"] = model_key
        save(out_path, rows)

    has = sum(1 for r in rows if r.get("response", "").strip())
    print(f"[{model_key}] 完成！response={has}/{len(rows)}, 结果: {out_path}")

    import get_response as gr
    try:
        del gr.model
        gr.model = None
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()
    run(args.model, args.device, args.batch_size)
