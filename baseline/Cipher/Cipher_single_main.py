from baseline.Cipher.experts.AsciiExpert import AsciiExpert
from baseline.Cipher.experts.CaserExpert import CaesarExpert
from baseline.Cipher.experts.ExpertBase import ExpertBase
from baseline.Cipher.experts.GBKExpert import GBKExpert
from baseline.Cipher.experts.MorseExpert import MorseExpert
from baseline.Cipher.experts.SelfDefineCipher import SelfDefineCipher
from baseline.Cipher.experts.UTF8Expert import UTF8Expert
from baseline.Cipher.experts.UnicodeExpert import UnicodeExpert
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


def get_experts(language: Language) -> list[ExpertBase]:
    if language == Language.ENGLISH:
        return [MorseExpert(), CaesarExpert(), AsciiExpert(), SelfDefineCipher()]
    elif language == Language.CHINESE_SIMPLIFIED:
        return [
            UnicodeExpert(),
            UTF8Expert(),
            GBKExpert(),
        ]


def Cipher_single_main(
    args_dict: dict,
    goal: str,
    target: str,
    language: Language,
):
    args = Args(args_dict)

    output_record = {
        "original_prompt": goal,
        "target": target,
    }

    experts = get_experts(language)

    attacks = []

    for expert in experts:
        attack = expert.get_attack(goal)
        attacks.append(attack)

    output_record["adv_prompt"] = attacks

    return output_record
