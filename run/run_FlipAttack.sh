export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -u main.py \
    --defense_type None_defense \
    --attack FlipAttack \
    --flip_attack_flip_mode FWO \
    --flip_attack_cot \
    --flip_attack_lang_gpt \
    --flip_attack_few_shot \
    --instructions_path ./data/SeedData_short.xlsx \
    --save_result_path ./exp_results/ \
    --exp_name flipattack