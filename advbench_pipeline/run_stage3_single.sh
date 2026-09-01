#!/usr/bin/env bash
# 单独对指定模型的 response CSV 跑 LlamaGuard-3-8B 评估
# 用法: bash advbench_pipeline/run_stage3_single.sh [csv路径] [GPU] [batch_size]
#
# 示例:
#   bash advbench_pipeline/run_stage3_single.sh \
#       advbench_pipeline/stage2_responses/qwen3-14b.csv \
#       cuda:5 \
#       16

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"

INPUT="${1:-advbench_pipeline/stage2_responses/qwen3-14b.csv}"
DEVICE="${2:-cuda:2}"
BATCH="${3:-16}"

echo "================================================"
echo "  Stage 3 评估（单模型）"
echo "  输入:   $INPUT"
echo "  GPU:    $DEVICE"
echo "  Batch:  $BATCH"
echo "================================================"

PYTHON_BIN="${PYTHON_BIN:-$BUNDLE_ROOT/my_restored_env/bin/python3}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "        Set PYTHON_BIN=/path/to/python to override it." >&2
    exit 1
fi

"$PYTHON_BIN" \
    advbench_pipeline/stage3_evaluate.py \
    --input "$INPUT" \
    --device "$DEVICE" \
    --batch_size "$BATCH"
