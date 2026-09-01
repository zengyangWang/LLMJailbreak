import re
from typing import List
from FastChat.fastchat.model import get_conversation_template
from baseline.TAP.system_prompts import (
    get_evaluator_system_prompt_for_judge,
    get_evaluator_system_prompt_for_on_topic,
)
from GPTEvaluatorAgent.language_models import ChatGPT, LocalVLLM
from data.language import Language


def load_evaluator(args):
    if args.evaluator_is_local:
        return LocalEvaluator(args)
    else:
        return GPTEvaluator(args)


class EvaluatorBase:
    def __init__(self, args):
        self.max_n_tokens = args.evaluator_max_n_tokens
        self.temperature = args.evaluator_temperature
        self.model_name = args.evaluator_model_path.split("/")[-1]

    def get_evaluator_prompt(
        self, attack_prompt: str, target_response: str, language: Language
    ):
        en_prompt = (
            f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        )
        zh_hans_prompt = f"[提示词]：{attack_prompt}\n[AI助手的回复]：{target_response}"
        prompt = {
            Language.ENGLISH: en_prompt,
            Language.CHINESE_SIMPLIFIED: zh_hans_prompt,
        }
        if language not in prompt:
            raise ValueError(f"Invalid language: {language}")
        return prompt[language]

    def get_evaluator_prompt_on_topic(self, attack_prompt: str, language: Language):
        en_prompt = f"[PROMPT]:{attack_prompt}"
        zh_hans_prompt = f"[提示词]：{attack_prompt}"
        prompt = {
            Language.ENGLISH: en_prompt,
            Language.CHINESE_SIMPLIFIED: zh_hans_prompt,
        }
        if language not in prompt:
            raise ValueError(f"Invalid language: {language}")
        return prompt[language]

    def judge_score(
        self,
        attack_prompt_list: List[str],
        target_response_list: List[str],
        goal: str,
        language: Language,
    ):
        raise NotImplementedError

    def on_topic_score(
        self, attack_prompt_list: List[str], goal: str, language: Language
    ):
        raise NotImplementedError

    def process_output_judge_score(self, raw_output):
        # Captures numbers enclosed in double square brackets, i.e., strings of the form "[[<number>]]"
        pattern = r"\[\[(\d+)\]\]"
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None

        if output is None:
            output = 1

        return output

    def process_output_on_topic_score(self, raw_output):
        # Captures yes/no in double square brackets, i.e., "[[YES]]" or "[[NO]]"
        pattern = r"\[\[(yes|no)\]\]"
        match = re.search(pattern, raw_output.lower())
        output = int(match.group(1) == "yes") if match else None

        if output is None:
            output = 1

        return output

    def shutdown(self):
        """
        尝试关闭底层评估模型（例如本地 vLLM 模型），释放显存。
        """
        try:
            backend = getattr(self, "evaluator_model", None)
            if backend is not None and hasattr(backend, "shutdown"):
                backend.shutdown()
        except Exception:
            # 评估器关闭失败不应中断主流程
            pass


class GPTEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(GPTEvaluator, self).__init__(args)
        self.evaluator_model = ChatGPT(model_name=self.model_name)

    def create_conv(self, full_prompt: str, system_prompt: str):
        conv = get_conversation_template(self.model_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)

        return conv.to_openai_api_messages()

    def judge_score(
        self,
        attack_prompt_list: List[str],
        target_response_list: List[str],
        goal: str,
        language: Language,
    ):
        convs_list = [
            self.create_conv(
                self.get_evaluator_prompt(prompt, response, language),
                system_prompt=get_evaluator_system_prompt_for_judge(goal, language),
            )
            for prompt, response in zip(attack_prompt_list, target_response_list)
        ]

        print(
            f"Querying evaluator with {len(attack_prompt_list)} prompts (to evaluate judge scores)",
            flush=True,
        )

        raw_outputs = self.evaluator_model.batched_generate(
            convs_list, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )

        outputs = [
            self.process_output_judge_score(raw_output) for raw_output in raw_outputs
        ]
        return outputs

    def on_topic_score(
        self, attack_prompt_list: List[str], goal: str, language: Language
    ):
        convs_list = [
            self.create_conv(
                self.get_evaluator_prompt_on_topic(prompt, language),
                system_prompt=get_evaluator_system_prompt_for_on_topic(goal, language),
            )
            for prompt in attack_prompt_list
        ]

        print(
            f"Querying evaluator with {len(attack_prompt_list)} prompts (to evaluate on-topic scores)",
            flush=True,
        )

        raw_outputs = self.evaluator_model.batched_generate(
            convs_list, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )
        outputs = [
            self.process_output_on_topic_score(raw_output) for raw_output in raw_outputs
        ]
        return outputs


class LocalEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(LocalEvaluator, self).__init__(args)
        model_path = args.evaluator_model_path
        self.evaluator_model = LocalVLLM(
            model_path=model_path, tensor_parallel_size=args.tensor_parallel_size
        )

    def create_conv(self, user_content: str, system_prompt: str):
        conv = get_conversation_template(self.model_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], user_content)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def judge_score(
        self,
        attack_prompt_list: List[str],
        target_response_list: List[str],
        goal: str,
        language: Language,
    ):
        convs_list = [
            self.create_conv(
                self.get_evaluator_prompt(prompt, response, language),
                system_prompt=get_evaluator_system_prompt_for_judge(goal, language),
            )
            for prompt, response in zip(attack_prompt_list, target_response_list)
        ]

        print(
            f"Querying evaluator with {len(attack_prompt_list)} prompts (to evaluate judge scores)",
            flush=True,
        )

        raw_outputs = self.evaluator_model.batched_generate(
            convs_list, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )

        outputs = [
            self.process_output_judge_score(raw_output) for raw_output in raw_outputs
        ]
        return outputs

    def on_topic_score(
        self, attack_prompt_list: List[str], goal: str, language: Language
    ):
        convs_list = [
            self.create_conv(
                self.get_evaluator_prompt_on_topic(prompt, language),
                system_prompt=get_evaluator_system_prompt_for_on_topic(goal, language),
            )
            for prompt in attack_prompt_list
        ]

        print(
            f"Querying evaluator with {len(attack_prompt_list)} prompts (to evaluate on-topic scores)",
            flush=True,
        )

        raw_outputs = self.evaluator_model.batched_generate(
            convs_list, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )
        outputs = [
            self.process_output_on_topic_score(raw_output) for raw_output in raw_outputs
        ]
        return outputs
