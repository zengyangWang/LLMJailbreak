from baseline.FlipAttack.prompts import (
    get_cot_prompt,
    get_few_shot_prompt_template,
    get_lang_gpt_prompt_template,
    get_mode_prompt,
    get_system_prompt,
    get_user_prompt_template,
)
from data.language import Language
import jieba


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def FlipAttack_single_main(
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

    system_prompt, user_prompt = FlipAttack(
        args.flip_attack_flip_mode,
        args.flip_attack_cot,
        args.flip_attack_lang_gpt,
        args.flip_attack_few_shot,
        language,
    ).generate(goal)

    output_record["adv_prompt"] = system_prompt + "\n\n" + user_prompt

    return output_record


import textwrap


class FlipAttack:
    # (I) Flip Word Order (FWO)
    # (II) Flip Chars in Word (FCW)
    # (III) Flip Chas in Sentence (FCS)
    # (IV) Fool Model Mode (FMM)

    # (A) Vanilla
    # (B) Vanilla+CoT
    # (C) Vanilla+CoT+LangGPT
    # (D) Vanilla+CoT+LangGPT+Few-shot

    def __init__(
        self,
        flip_mode: str,
        cot: bool,
        lang_gpt: bool,
        few_shot: bool,
        language: Language,
    ) -> None:
        self.flip_mode = flip_mode
        self.cot = cot
        self.lang_gpt = lang_gpt
        self.few_shot = few_shot
        self.language = language

        self.mode_prompt = get_mode_prompt(self.flip_mode, self.language)

        # chain-of-thought
        self.cot_prompt = get_cot_prompt(self.cot, self.language)

        self.system_prompt = get_system_prompt(self.lang_gpt, self.language)

        # LangGPT
        if self.lang_gpt:
            if self.language == Language.ENGLISH:
                demo_prompt = self.demo("give me an apple", self.flip_mode)
            elif self.language == Language.CHINESE_SIMPLIFIED:
                demo_prompt = self.demo("给我一个苹果", self.flip_mode)
            else:
                raise ValueError(f"Unknown language: {self.language}")

            self.lang_gpt_prompt = get_lang_gpt_prompt_template(self.language).format(
                self.mode_prompt,
                demo_prompt,
                self.cot_prompt,
            )

        else:
            self.system_prompt = get_system_prompt(self.lang_gpt, self.language).format(
                self.mode_prompt, self.cot_prompt
            )

    def flip_word_order(self, input_str: str) -> str:
        if self.language == Language.ENGLISH:
            return " ".join(input_str.split()[::-1])
        elif self.language == Language.CHINESE_SIMPLIFIED:
            return "".join(list(jieba.cut(input_str))[::-1])
        else:
            raise ValueError(f"Unknown language: {self.language}")

    def flip_char_in_word(self, input_str: str) -> str:
        if self.language == Language.ENGLISH:
            return " ".join([word[::-1] for word in input_str.split()])
        elif self.language == Language.CHINESE_SIMPLIFIED:
            return "".join([word[::-1] for word in list(jieba.cut(input_str))])
        else:
            raise ValueError(f"Unknown language: {self.language}")

    def flip_char_in_sentence(self, input_str: str) -> str:
        return input_str[::-1]

    def demo(self, input_str: str, mode: str) -> str:
        if mode == "FWO" or mode == "FMM":
            # flip word order | fool model mode: flip chars in sentence but ask model to flip back via flipping word order
            return self.flip_word_order(input_str)

        elif mode == "FCW":
            # flip chars in word
            return self.flip_char_in_word(input_str)

        elif mode == "FCS":
            # flip chars in sentence
            return self.flip_char_in_sentence(input_str)

    def split_sentence_in_half(self, input_str: str) -> tuple[str, str]:
        if self.language == Language.ENGLISH:
            wrapper = textwrap.TextWrapper(
                width=len(input_str) // 2,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_list = wrapper.wrap(input_str)
        elif self.language == Language.CHINESE_SIMPLIFIED:
            words = list(jieba.cut(input_str))
            wrapped_list = [words[0]]
            for word in words[1:]:
                if len(wrapped_list[-1]) + len(word) > len(input_str) // 2:
                    wrapped_list.append(word)
                else:
                    wrapped_list[-1] += word
        else:
            raise ValueError(f"Unknown language: {self.language}")

        if len(wrapped_list) > 2:
            first_half = wrapped_list[0]
            second_half = " ".join(wrapped_list[1:])
        else:
            first_half, second_half = wrapped_list[0], wrapped_list[1]

        return first_half, second_half

    def generate(self, harm_prompt: str) -> tuple[str, str]:

        # flipping mode
        if self.flip_mode == "FWO":
            # flip word order
            disguised_prompt = self.flip_word_order(harm_prompt)

        elif self.flip_mode == "FCW":
            # flip chars in word
            disguised_prompt = self.flip_char_in_word(harm_prompt)

        elif self.flip_mode == "FCS" or self.flip_mode == "FMM":
            # flip chars in sentence | fool model mode: flip chars in sentence but ask model to flip back via flipping word order
            disguised_prompt = self.flip_char_in_sentence(harm_prompt)

        else:
            raise TypeError("Unknown Flip Mode {}".format(self.flip_mode))

        self.user_prompt = get_user_prompt_template(self.language).format(
            disguised_prompt
        )

        # task-oriented few-shot demonstration
        if self.few_shot:
            left, right = self.split_sentence_in_half(harm_prompt)
            self.few_shot_prompt = get_few_shot_prompt_template(self.language).format(
                self.demo(left, self.flip_mode),
                left,
                self.demo(right, self.flip_mode),
                right,
            )

        if self.lang_gpt and not self.few_shot:
            self.user_prompt += "\n" + self.lang_gpt_prompt

        elif self.lang_gpt and self.few_shot:
            if self.language == Language.ENGLISH:
                # remove one-shot demo
                self.lang_gpt_prompt = (
                    self.lang_gpt_prompt[: self.lang_gpt_prompt.find("e.g.") - 2]
                    + self.lang_gpt_prompt[self.lang_gpt_prompt.find("NEVER") - 2 :]
                )

                # add task-oriented few-sho demo
                self.lang_gpt_prompt = (
                    self.lang_gpt_prompt[: self.lang_gpt_prompt.find("Step 2:")]
                    + self.few_shot_prompt
                    + "\n\n"
                    + self.lang_gpt_prompt[self.lang_gpt_prompt.find("Step 2:") :]
                )
            elif self.language == Language.CHINESE_SIMPLIFIED:

                self.lang_gpt_prompt = (
                    self.lang_gpt_prompt[: self.lang_gpt_prompt.find("例如") - 1]
                    + self.lang_gpt_prompt[self.lang_gpt_prompt.find("绝不") - 1 :]
                )

                self.lang_gpt_prompt = (
                    self.lang_gpt_prompt[: self.lang_gpt_prompt.find("步骤2：")]
                    + self.few_shot_prompt
                    + "\n\n"
                    + self.lang_gpt_prompt[self.lang_gpt_prompt.find("步骤2：") :]
                )
            else:
                raise ValueError(f"Unknown language: {self.language}")

            self.user_prompt += "\n\n" + self.lang_gpt_prompt

        elif not self.lang_gpt and self.few_shot:
            self.system_prompt += self.few_shot_prompt

        return self.system_prompt, self.user_prompt
