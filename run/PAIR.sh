export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1

python -u main.py \
    --target_model_path llm_weights/qwen3-4b-instruct-2507 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model_path llm_weights/qwen3-4b-instruct-2507 \
    --evaluator_model llm_weights/llama-guard-3-8b \
    --evaluator_is_local \
    --evaluator_model_path llm_weights/llama-guard-3-8b \
    --tensor_parallel_size 2 \
    --instructions_path ./data/seed_prompt.xlsx \
    --save_result_path ./exp_results/ \
    --n_iterations 30 \
    --n_streams_pair 3 \
    --keep_last_n 3 \
    --exp_name PAIR \
    --resume_exp