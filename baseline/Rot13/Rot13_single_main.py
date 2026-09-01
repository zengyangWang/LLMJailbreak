from data.language import Language
from encoding_attacks_single import generate_rot13_prompt


def _lang_short(language: Language) -> str:
    if language in (Language.CHINESE_SIMPLIFIED, Language.CHINESE_TRADITIONAL):
        return "zh"
    return "en"


def Rot13_single_main(
    args_dict: dict,
    goal: str,
    target: str,
    language: Language,
):
    attack = generate_rot13_prompt(str(goal or ""), lang_code=_lang_short(language))
    return {
        "original_prompt": goal,
        "target": target,
        "attack_prompt": "NULL",
        "adv_prompt": attack,
        "attack_iterations": 1,
        "language_model_output": "SKIPPED (Generation Only)",
        "is_JB_Judge": None,
        "is_JB": False,
        "is_JB_Agent": None,
        "attack_method": "Rot13",
    }
