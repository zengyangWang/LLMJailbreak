import sys
import os
import json

# Add current directory to sys.path to allow imports from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import PreAttackConfig
from preattack import PreAttack
# from inattack import InAttack
from actor_utils import reload_clients

def Actorattack_single_main(args_dict, target_model, goal, target, language):
    """
    Single main function for Actorattack.
    
    Args:
        args_dict (dict): Arguments dictionary.
        target_model (str or object): Target model path or object. 
                                      Actorattack expects a path string for local models if using vLLM wrapper.
        goal (str): The malicious goal.
        target (str): The target response (not used much in Actorattack as it has its own judge).
        language (str): Language (not explicitly handled in Actorattack, assumes goal language).
        
    Returns:
        dict: Result dictionary with keys 'adv_prompt', 'language_model_output', 'attack_iterations', 'is_JB'.
    """
    
    # Set API key from args if provided
    if args_dict.get('openai_key'):
        os.environ['GPT_API_KEY'] = args_dict['openai_key']
        # You might want to map other keys if args support them
        reload_clients()
    
    # Extract arguments
    # We assume args_dict has 'attack_model_path' or we default to 'gpt-4o'
    # If 'attack_model_path' is not in args, we might need to add it to initialize_args.py
    # For now, let's check if there's a generic attack model arg. 
    # initialize_args.py has 'attack_model_path' for PAIR/TAP. We can reuse it.
    
    attack_model_name = args_dict.get('attack_model_path', 'gpt-4o')
    target_model_path = args_dict.get('target_model_path', '')
    
    # If target_model is passed as an object (from main.py loading), we might have an issue 
    # because Actorattack is designed to create its own client.
    # However, we modified utils.py to accept a path.
    # So we should ensure we pass the path string.
    
    # PreAttack Config
    # We need to ensure paths to prompts are correct relative to the execution directory (LLMJailbreak root)
    # Actorattack config defaults are relative to its own folder usually, but let's check config.py
    # config.py has defaults like './prompts/1_extract.txt'. 
    # If running from LLMJailbreak root, './prompts' won't work if prompts are in 'baseline/Actorattack/prompts'.
    # We need to fix paths in config or pass absolute paths.
    
    base_path = os.path.join(os.getcwd(), 'baseline', 'Actorattack')
    
    # Determine prompt suffix based on language
    prompt_suffix = ""
    # Check if language is CHINESE_SIMPLIFIED (handle both Enum and string cases)
    if (hasattr(language, 'name') and language.name == 'CHINESE_SIMPLIFIED') or str(language) == 'CHINESE_SIMPLIFIED':
        prompt_suffix = "_zh"
    
    pre_config = PreAttackConfig(
        model_name=attack_model_name,
        actor_num=args_dict.get('actor_num', 3), # We might need to add this arg
        behavior_csv=None,
        extract_prompt=os.path.join(base_path, f'prompts/1_extract{prompt_suffix}.txt'),
        network_prompt=os.path.join(base_path, f'prompts/2_network{prompt_suffix}.txt'),
        actor_prompt=os.path.join(base_path, f'prompts/3_actor{prompt_suffix}.txt'),
        query_prompt=os.path.join(base_path, f'prompts/4_queries{prompt_suffix}.txt'),
        json_format_prompt=os.path.join(base_path, f'prompts/5_json_format{prompt_suffix}.txt'),
        more_actor_prompt=os.path.join(base_path, f'prompts/3_more_actor{prompt_suffix}.txt'),
        tensor_parallel_size=args_dict.get('tensor_parallel_size', 4)
    )
    
    pre_attacker = PreAttack(pre_config)
    
    # Run PreAttack
    print(f"Running PreAttack for goal: {goal}")
    pre_result = pre_attacker.infer_single(goal)
    
    # Extract all generated queries from the result
    generated_queries = []
    if pre_result and 'actors' in pre_result:
        for actor_entry in pre_result['actors']:
            if 'queries' in actor_entry:
                generated_queries.extend(actor_entry['queries'])

    # Return the queries as the adv_prompt
    # We return the list of queries as a JSON string so all of them are preserved
    return {
        "adv_prompt": json.dumps(generated_queries, ensure_ascii=False),
        "language_model_output": "",
        "attack_iterations": 0,
        "is_JB": False
    }

    # InAttack Config
    # in_config = InAttackConfig(
    #     attack_model_name=attack_model_name,
    #     target_model_name=target_model_path,
    #     pre_attack_data_path='',
    #     step_judge_prompt=os.path.join(base_path, 'prompts/attack_step_judge.txt'),
    #     modify_prompt=os.path.join(base_path, 'prompts/attack_modify.txt'),
    #     early_stop=True,
    #     dynamic_modify=True
    # )
    
    # in_attacker = InAttack(in_config)
    
    # # Run InAttack
    # print(f"Running InAttack...")
    # attack_result = in_attacker.attack_single(pre_result)
    
    # # Parse result
    # attempts = attack_result.get('attempts', [])
    
    # best_attempt = None
    # for attempt in attempts:
    #     if attempt.get('final_score') == 5:
    #         best_attempt = attempt
    #         break
    
    # if not best_attempt and attempts:
    #     best_attempt = attempts[-1]
        
    # if best_attempt:
    #     dialog_hist = best_attempt.get('dialog_hist', [])
    #     # Extract conversation as prompt and response
    #     # The last message is assistant response (if successful or finished)
    #     # The messages before are the attack context.
        
    #     if dialog_hist:
    #         last_msg = dialog_hist[-1]
    #         if last_msg['role'] == 'assistant':
    #             response = last_msg['content']
    #             # Construct prompt string from history
    #             prompt_msgs = dialog_hist[:-1]
    #             prompt_str = json.dumps(prompt_msgs, ensure_ascii=False) # Or format nicely
    #         else:
    #             response = ""
    #             prompt_str = json.dumps(dialog_hist, ensure_ascii=False)
            
    #         is_jb = (best_attempt.get('final_score') == 5)
            
    #         return {
    #             "adv_prompt": prompt_str,
    #             "language_model_output": response,
    #             "attack_iterations": len(attempts),
    #             "is_JB": is_jb
    #         }

    # return {
    #     "adv_prompt": "",
    #     "language_model_output": "",
    #     "attack_iterations": 0,
    #     "is_JB": False
    # }
