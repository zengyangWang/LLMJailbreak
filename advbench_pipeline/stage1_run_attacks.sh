#!/usr/bin/env bash
# =============================================================================
# Stage 1: 针对 advbench 数据集生成攻击 prompt
# 包含 6 种攻击方法：
#   白盒: GCG, AmpleGCG
#   黑盒: PAIR, AutoDAN
#   模板: JailBroken, Template
# 每种方法跑完后结果保存在 exp_results/advbench/ 下
# =============================================================================

set -e
cd "$(dirname "$0")/.."  # 切到 LLMJailbreak 根目录

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-$(cd .. && pwd)/huggingface_cache}"

DATA_PATH="./data/advbench.csv"
SAVE_PATH="./exp_results/advbench_pipeline/"
LLAMA_GUARD_PATH="./llm_weights/llama-guard-3-8b"
TARGET_MODEL_PATH="./llm_weights/vicuna-13b-v1.5"
QWEN_PATH="./llm_weights/qwen3-4b-instruct-2507"

# 日志目录
LOG_DIR="./advbench_pipeline/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Stage 1: 攻击 prompt 生成"
echo "数据集: $DATA_PATH"
echo "结果目录: $SAVE_PATH"
echo "=========================================="

# ------------------------------------------------------------------------------
# 1. GCG (白盒) - 需要加载目标模型权重，GPU显存较大
# ------------------------------------------------------------------------------
echo ""
echo "[1/6] GCG 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack GCG \
    --target_model_path "$TARGET_MODEL_PATH" \
    --defense_type None_defense \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --gcg_attack_budget 100 \
    --gcg_suffix 20 \
    --target_max_n_tokens 150 \
    --exp_name advbench_GCG \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/gcg.log"

# ------------------------------------------------------------------------------
# 2. AmpleGCG (白盒) - 先用 AmpleGCG 模型生成 suffix，再在本地目标模型评测
# ------------------------------------------------------------------------------
echo ""
echo "[2/6] AmpleGCG 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack AmpleGCG \
    --target_model_path "$TARGET_MODEL_PATH" \
    --defense_type None_defense \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --attack_source llama2 \
    --num_beams 100 \
    --ample_max_new_tokens 20 \
    --ample_min_new_tokens 20 \
    --tensor_parallel_size 2 \
    --exp_name advbench_AmpleGCG \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/amplegcg.log"

# ------------------------------------------------------------------------------
# 3. PAIR (黑盒) - 使用 Qwen 作为攻击模型 + Llama-Guard 作为内置评估器
# ------------------------------------------------------------------------------
echo ""
echo "[3/6] PAIR 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack PAIR \
    --target_model_path "$TARGET_MODEL_PATH" \
    --attack_model_path "$QWEN_PATH" \
    --defense_type None_defense \
    --evaluator_is_local \
    --evaluator_model_path "$LLAMA_GUARD_PATH" \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --n_iterations 20 \
    --n_streams_pair 3 \
    --keep_last_n 3 \
    --tensor_parallel_size 2 \
    --exp_name advbench_PAIR \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/pair.log"

# ------------------------------------------------------------------------------
# 4. AutoDAN (黑盒) - 遗传算法 + 变异
# ------------------------------------------------------------------------------
echo ""
echo "[4/6] AutoDAN 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack AutoDAN \
    --target_model_path "$TARGET_MODEL_PATH" \
    --defense_type None_defense \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --gcg_attack_budget 100 \
    --gpt_mutate gpt-3.5-turbo \
    --autodan_mutate_local_model_path "$QWEN_PATH" \
    --exp_name advbench_AutoDAN \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/autodan.log"

# ------------------------------------------------------------------------------
# 5. JailBroken (模板) - 多种编码/变换策略
# ------------------------------------------------------------------------------
echo ""
echo "[5/6] JailBroken 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack JailBroken \
    --target_model_path "$TARGET_MODEL_PATH" \
    --attack_model_path "$QWEN_PATH" \
    --defense_type None_defense \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --jailbroken_max_new_tokens 256 \
    --jailbroken_temperature 0.7 \
    --jailbroken_top_p 0.9 \
    --jailbroken_gpu_memory_utilization 0.5 \
    --tensor_parallel_size 2 \
    --exp_name advbench_JailBroken \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/jailbroken.log"

# ------------------------------------------------------------------------------
# 6. Template (模板) - 固定模板填充
# ------------------------------------------------------------------------------
echo ""
echo "[6/6] Template 攻击..."
export CUDA_VISIBLE_DEVICES=6,7
python -u main.py \
    --attack Template \
    --target_model_path "$TARGET_MODEL_PATH" \
    --defense_type None_defense \
    --instructions_path "$DATA_PATH" \
    --save_result_path "$SAVE_PATH" \
    --target_max_n_tokens 150 \
    --tensor_parallel_size 2 \
    --exp_name advbench_Template \
    --resume_exp \
    2>&1 | tee "$LOG_DIR/template.log"

echo ""
echo "=========================================="
echo "Stage 1 完成！结果保存在: $SAVE_PATH"
echo "=========================================="
