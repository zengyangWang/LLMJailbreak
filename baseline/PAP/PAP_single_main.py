import random
from baseline.PAP.config import (
    ATTACK_TEMPERATURE,
    ATTACK_TOP_P,
)
from baseline.PAP.llm_pap import AttackerLLM_pap
from data.language import Language
from baseline.PAP.persuasion_template import get_persuasion_template


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def PAP_initial(args_dict: dict):
    args = Args(args_dict)

    attack_model_path = args.attack_model_path
    tensor_parallel_size = args.tensor_parallel_size
    attack_max_n_tokens = args.attack_max_n_tokens
    attack_temperature = ATTACK_TEMPERATURE
    attack_top_p = ATTACK_TOP_P

    attack_model = AttackerLLM_pap(
        model_path=attack_model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_n_tokens=attack_max_n_tokens,
        temperature=attack_temperature,
        top_p=attack_top_p,
    )
    return attack_model


def PAP_single_main(
    args_dict: dict,
    attack_model: AttackerLLM_pap,
    goal: str,
    target: str,
    language: Language,
):
    output_record = {
        "original_prompt": goal,
        "target": target,
    }

    persuasion_template, tag = get_persuasion_template(language)
    persuasion_methods = list[str](persuasion_template.keys())
    chosen_method = random.choice(persuasion_methods)
    chosen_template = persuasion_template[chosen_method]

    attack = attack_model.get_attack(template=chosen_template, tag=tag, goal=goal)

    output_record["adv_prompt"] = attack

    return output_record
