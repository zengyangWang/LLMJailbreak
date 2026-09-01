export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -u main.py \
    --target_model_path llm_weights/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack PAP \
    --attack_model_path llm_weights/vicuna-13b-v1.5 \
    --tensor_parallel_size 4 \
    --instructions_path ./data/SeedData_short.xlsx \
    --save_result_path ./exp_results/ \
    --exp_name pap