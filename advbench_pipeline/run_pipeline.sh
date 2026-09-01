#!/usr/bin/env bash
# =============================================================================
# 总控脚本：串联三个阶段的完整 pipeline
#
# 用法：
#   bash advbench_pipeline/run_pipeline.sh [--skip-stage1] [--skip-stage2] [--skip-stage3]
#
# 参数说明：
#   --skip-stage1  跳过攻击 prompt 生成（Stage 1），直接从已有结果进入 Stage 2
#   --skip-stage2  跳过目标模型推理（Stage 2）
#   --skip-stage3  跳过 LlamaGuard 评估（Stage 3）
#
# GPU 配置（按需修改）：
#   STAGE1_GPUS  Stage 1 使用的 GPU（攻击生成，白盒需要较多显存）
#   STAGE2_GPU   Stage 2 使用的单 GPU（目标模型推理）
#   STAGE3_GPU   Stage 3 使用的单 GPU（LlamaGuard 评估）
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================
# GPU 配置 —— 按照实际情况修改
# ============================================================
STAGE2_GPU="cuda:6"
STAGE3_GPU="cuda:6"
STAGE2_BATCH=4
STAGE3_BATCH=8

# ============================================================
# 解析命令行参数
# ============================================================
SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_STAGE3=false

for arg in "$@"; do
    case $arg in
        --skip-stage1) SKIP_STAGE1=true ;;
        --skip-stage2) SKIP_STAGE2=true ;;
        --skip-stage3) SKIP_STAGE3=true ;;
        *) echo "[WARN] 未知参数: $arg" ;;
    esac
done

echo "=============================================="
echo "  LLMJailbreak Advbench Pipeline"
echo "  Stage 1 (攻击生成): $([ "$SKIP_STAGE1" = true ] && echo 跳过 || echo 运行)"
echo "  Stage 2 (模型推理): $([ "$SKIP_STAGE2" = true ] && echo 跳过 || echo 运行)"
echo "  Stage 3 (Guard评估): $([ "$SKIP_STAGE3" = true ] && echo 跳过 || echo 运行)"
echo "=============================================="

# ============================================================
# Stage 1: 生成攻击 prompt（6 种攻击方法）
# ============================================================
if [ "$SKIP_STAGE1" = false ]; then
    echo ""
    echo "=============================="
    echo ">>> Stage 1: 攻击 prompt 生成"
    echo "=============================="
    bash "$SCRIPT_DIR/stage1_run_attacks.sh"
    echo ">>> Stage 1 完成"
else
    echo ">>> Stage 1 已跳过"
fi

# ============================================================
# Stage 2: 4 个目标模型批量推理
# ============================================================
if [ "$SKIP_STAGE2" = false ]; then
    echo ""
    echo "=============================="
    echo ">>> Stage 2: 目标模型批量推理"
    echo "=============================="
    python "$SCRIPT_DIR/stage2_get_responses.py" \
        --device "$STAGE2_GPU" \
        --batch_size "$STAGE2_BATCH"
    echo ">>> Stage 2 完成"
else
    echo ">>> Stage 2 已跳过"
fi

# ============================================================
# Stage 3: LlamaGuard-3-8B 评估
# ============================================================
if [ "$SKIP_STAGE3" = false ]; then
    echo ""
    echo "=============================="
    echo ">>> Stage 3: LlamaGuard 评估"
    echo "=============================="
    python "$SCRIPT_DIR/stage3_evaluate.py" \
        --device "$STAGE3_GPU" \
        --batch_size "$STAGE3_BATCH"
    echo ">>> Stage 3 完成"
else
    echo ">>> Stage 3 已跳过"
fi

echo ""
echo "=============================================="
echo "  Pipeline 全部完成！"
echo "  结果目录: $REPO_ROOT/advbench_pipeline/"
echo "    ├── logs/                      Stage 1 运行日志"
echo "    ├── stage2_responses/          Stage 2 模型回复（每模型一个 CSV）"
echo "    │   └── stage2_all_responses.csv  合并后的全量回复"
echo "    └── stage3_evaluation/"
echo "        ├── stage3_evaluation_results.csv  完整评估结果"
echo "        └── asr_summary.csv               ASR 汇总表"
echo "=============================================="
