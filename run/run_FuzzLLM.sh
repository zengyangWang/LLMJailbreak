export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -u main.py \
    --defense_type None_defense \
    --attack FuzzLLM \
    --fuzzllm_template_type RP \
    --instructions_path ./data/SeedData_short.xlsx \
    --save_result_path ./exp_results/ \
    --exp_name FuzzLLM