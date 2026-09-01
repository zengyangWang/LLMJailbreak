from typing import List
from baseline.FuzzLLM.constraint import get_constraints
from baseline.FuzzLLM.template import get_templates
from data.language import Language


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def FuzzLLM_single_main(
    args_dict: dict,
    goal: str,
    target: str,
    language: Language,
):
    args = Args(args_dict)

    template_type = args.fuzzllm_template_type

    output_record = {
        "original_prompt": goal,
        "target": target,
    }

    attack = get_attack(goal, language, template_type)

    output_record["adv_prompt"] = attack

    return output_record


def construct_RP_attacks(goal: str, language: Language) -> List[str]:
    templates = get_templates("RP", language)
    constraints = get_constraints("RP", language)
    attacks = []

    for template in templates:
        for constraint in constraints:
            attack = template.format(question=goal, RP_constraint=constraint)
            attacks.append(attack)

    return attacks


def construct_OC_attacks(goal: str, language: Language) -> List[str]:
    templates = get_templates("OC", language)
    constraints = get_constraints("OC", language)
    attacks = []

    for template in templates:
        for constraint in constraints:
            attack = template.format(question=goal, OC_constraint=constraint)
            attacks.append(attack)

    return attacks


def construct_PE_attacks(goal: str, language: Language) -> List[str]:

    templates = get_templates("PE", language)
    constraints = get_constraints("PE", language)
    attacks = []

    for template in templates:
        for constraint in constraints:
            attack = template.format(question=goal, PE_constraint=constraint)
            attacks.append(attack)

    return attacks


def construct_RPOC_attacks(goal: str, language: Language) -> List[str]:

    templates = get_templates("RPOC", language)
    RP_constraints = get_constraints("RP", language)
    OC_constraints = get_constraints("OC", language)
    attacks = []

    for template in templates:
        for RP_constraint in RP_constraints:
            for OC_constraint in OC_constraints:
                attack = template.format(
                    question=goal,
                    RP_constraint=RP_constraint,
                    OC_constraint=OC_constraint,
                )
            attacks.append(attack)

    return attacks


def construct_RPPE_attacks(goal: str, language: Language) -> List[str]:

    templates = get_templates("RPPE", language)
    RP_constraints = get_constraints("RP", language)
    PE_constraints = get_constraints("PE", language)
    attacks = []

    for template in templates:
        for RP_constraint in RP_constraints:
            for PE_constraint in PE_constraints:
                attack = template.format(
                    question=goal,
                    RP_constraint=RP_constraint,
                    PE_constraint=PE_constraint,
                )
            attacks.append(attack)

    return attacks


def construct_PEOC_attacks(goal: str, language: Language) -> List[str]:
    templates = get_templates("PEOC", language)
    PE_constraints = get_constraints("PE", language)
    OC_constraints = get_constraints("OC", language)
    attacks = []

    for template in templates:
        for PE_constraint in PE_constraints:
            for OC_constraint in OC_constraints:
                attack = template.format(
                    question=goal,
                    PE_constraint=PE_constraint,
                    OC_constraint=OC_constraint,
                )
                attacks.append(attack)

    return attacks


def construct_RPPEOC_attacks(goal: str, language: Language) -> List[str]:
    templates = get_templates("RPPEOC", language)
    RP_constraints = get_constraints("RP", language)
    PE_constraints = get_constraints("PE", language)
    OC_constraints = get_constraints("OC", language)
    attacks = []

    for template in templates:
        for RP_constraint in RP_constraints:
            for PE_constraint in PE_constraints:
                for OC_constraint in OC_constraints:
                    attack = template.format(
                        question=goal,
                        RP_constraint=RP_constraint,
                        PE_constraint=PE_constraint,
                        OC_constraint=OC_constraint,
                    )
                    attacks.append(attack)

    return attacks


def get_attack(goal: str, language: Language, template_type: str) -> List[str]:
    if template_type == "RP":
        return construct_RP_attacks(goal, language)
    elif template_type == "OC":
        return construct_OC_attacks(goal, language)
    elif template_type == "PE":
        return construct_PE_attacks(goal, language)
    elif template_type == "RPOC":
        return construct_RPOC_attacks(goal, language)
    elif template_type == "RPPE":
        return construct_RPPE_attacks(goal, language)
    elif template_type == "PEOC":
        return construct_PEOC_attacks(goal, language)
    elif template_type == "RPPEOC":
        return construct_RPPEOC_attacks(goal, language)
    else:
        raise ValueError(f"Invalid template type: {template_type}")
