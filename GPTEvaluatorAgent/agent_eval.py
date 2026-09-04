import json
import argparse
from GPTEvaluatorAgent.main import gpt_agent_evaluation
from tqdm import tqdm
from GPTEvaluatorAgent.judges import load_agent_judge

# from GPTEvaluatorAgent.common import detect_repetitions
from utils.test_utils import detect_repetitions


def process_response(model_response):
    length = detect_repetitions(model_response)
    if length > 300:
        model_response = model_response[:10]
    return model_response


def _cold_candidate_outputs(item):
    outputs = item.get("all_model_outputs")
    if isinstance(outputs, list) and outputs:
        return outputs
    return None


def _evaluate_cold_candidates(
    evaluator, item, only_heuristic_true=False
):
    """Evaluate each COLD batch response without mixing candidates."""
    outputs = _cold_candidate_outputs(item) or []
    heuristic_results = item.get("all_is_JB", [])
    if not isinstance(heuristic_results, list):
        heuristic_results = []
    goal = item["original_prompt"]
    target = item["target"]

    jailbreak_results = []
    reasons = []
    scores = []
    for index, response in enumerate(outputs):
        heuristic_is_jb = (
            bool(heuristic_results[index])
            if index < len(heuristic_results)
            else True
        )
        if only_heuristic_true and not heuristic_is_jb:
            jailbroken, reason, score = False, "Not JB", -1
        else:
            jailbroken, reason, score = gpt_agent_evaluation(
                evaluator, goal, target, process_response(response)
            )
        jailbreak_results.append(jailbroken)
        reasons.append(reason)
        scores.append(score)

    item["all_is_JB_Agent"] = jailbreak_results
    item["all_reasons"] = reasons
    item["all_scores"] = scores
    item["is_JB_Agent"] = jailbreak_results[0] if jailbreak_results else False
    item["reason"] = reasons[0] if reasons else ""
    item["score"] = scores[0] if scores else -1
    item["is_JB_Agent_any"] = any(jailbreak_results)


def agent_evaluation(args, data):

    GPT_Evaluator = load_agent_judge(args=args)
    new_data = []
    if args.judge_only_true:
        print("Only using agent to evaluate the jailbreak results")
        for item in tqdm(data, desc="Evaluating"):
            if _cold_candidate_outputs(item):
                _evaluate_cold_candidates(
                    GPT_Evaluator, item, only_heuristic_true=True
                )
                new_data.append(item)
                continue
            model_response = item["language_model_output"]
            model_response = process_response(model_response)
            goal = item["original_prompt"]
            target = item["target"]
            if item["is_JB"]:
                jailbroken, reason, score = gpt_agent_evaluation(
                    GPT_Evaluator, goal, target, model_response
                )
            else:
                jailbroken = False
                reason = "Not JB"
                score = -1
            item["is_JB_Agent"] = jailbroken
            item["reason"] = reason
            item["score"] = score
            new_data.append(item)
    elif args.agent_recheck:
        recheck_info = []
        recheck_str = [
            "The heuristic method determines that the result is jailbreak.",
            "The result of the heuristic judgment is that there is no jailbreak",
        ]
        for item in tqdm(data, desc="Evaluating"):
            if _cold_candidate_outputs(item):
                _evaluate_cold_candidates(GPT_Evaluator, item)
                new_data.append(item)
                continue
            print("Find resume_exp is True, check the data_id: ", item["data_id"])
            if "reason" not in item:
                print("First time to check the data_id: ", item["data_id"])
            elif recheck_str[0] in item["reason"] or recheck_str[1] in item["reason"]:
                print("Need to recheck the data_id: ", item["data_id"])
                # append the experiment info into the txt file
                if len(recheck_info) == 0:
                    recheck_info = [
                        args.instructions_path,
                        args.defense_type,
                        args.attack,
                        args.target_model_path,
                    ]
                    recheck_info = [
                        str(args.instructions_path),
                        str(args.defense_type),
                        str(args.attack),
                        str(args.target_model_path),
                    ]
                    recheck_info_str = ", ".join(recheck_info)
                    with open("recheck_info.txt", "a") as f:
                        f.write(recheck_info_str)
                        f.write("\n\n")

            else:
                print("Skip the data_id: ", item["data_id"])
                new_data.append(item)
                continue
            model_response = item["language_model_output"]
            model_response = process_response(model_response)
            goal = item["original_prompt"]
            target = item["target"]

            jailbroken, reason, score = gpt_agent_evaluation(
                GPT_Evaluator, goal, target, model_response
            )
            item["is_JB_Agent"] = jailbroken
            item["reason"] = reason
            item["score"] = score
            new_data.append(item)
    else:
        for item in tqdm(data, desc="Evaluating"):
            candidate_outputs = _cold_candidate_outputs(item)
            if candidate_outputs:
                completed = item.get("all_is_JB_Agent")
                if (
                    args.resume_exp
                    and isinstance(completed, list)
                    and len(completed) == len(candidate_outputs)
                ):
                    print("Skip the data_id: ", item["data_id"])
                else:
                    _evaluate_cold_candidates(GPT_Evaluator, item)
                new_data.append(item)
                continue
            if args.resume_exp:
                print("Find resume_exp is True, check the data_id: ", item["data_id"])
                if item["is_JB_Agent"] != "None":
                    print("Skip the data_id: ", item["data_id"])
                    new_data.append(item)
                    continue
                else:
                    print("Start the data_id: ", item["data_id"])
            model_response = item["language_model_output"]
            model_response = process_response(model_response)
            goal = item["original_prompt"]
            target = item["target"]

            jailbroken, reason, score = gpt_agent_evaluation(
                GPT_Evaluator, goal, target, model_response
            )
            item["is_JB_Agent"] = jailbroken
            item["reason"] = reason
            item["score"] = score
            new_data.append(item)
    return new_data
