r"""
'MJPChoices', generate jailbreak prompts based on locally stored templates and
construct conversation variants according to the chosen strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .mutation_base import MutationBase, Instance

_TEMPLATE_PATH = Path(__file__).with_name("mjp_template.json")
if _TEMPLATE_PATH.is_file():
    with _TEMPLATE_PATH.open("r", encoding="utf-8") as fp:
        _MJP_TEMPLATES = json.load(fp)
else:
    _MJP_TEMPLATES = {"en": [], "cn": []}


def _language_key(language: str) -> str:
    if not language:
        return "en"
    lang = language.lower()
    if any(tag in lang for tag in ("cn", "zh", "zho")):
        return "cn"
    return "en"


__all__ = ["MJPChoices"]


class MJPChoices(MutationBase):
    """Generate jailbreak prompts based on the chosen multi-step strategy."""

    def __init__(self, prompt_type: str, target_model, prompt_templates=None) -> None:
        super().__init__()
        self.prompt_type = prompt_type
        self.target_model = target_model
        self._templates = prompt_templates or _MJP_TEMPLATES

    def _get_mutated_instance(self, instance, *args, **kwargs) -> List[Instance]:
        conv = instance.attack_attrs['conversation']
        query = instance.query.strip()

        language = self._detect_language(instance)
        lang_key = _language_key(language)
        prompts = self._templates.get(lang_key, self._templates.get("en", []))
        jailbreak_prompt = prompts[0] if prompts else ""
        ack_prompt = prompts[1] if len(prompts) > 1 else ""

        cot_suffix = self._cot_suffix(language)
        if 'If you are not sure' in query or '如果不确定' in query:
            input_msg_cot = query
        else:
            input_msg_cot = f"{query} {cot_suffix}".strip()

        user_prompt = jailbreak_prompt if jailbreak_prompt else None
        assistant_prompt = ack_prompt if ack_prompt else None

        conv = self.get_conv_from_type(conv, self.prompt_type, user_prompt, assistant_prompt, query, input_msg_cot)
        instance.attack_attrs['conversation'] = conv
        return [instance]

    def _detect_language(self, instance) -> str:
        lang = getattr(instance, 'language', None)
        if isinstance(lang, str) and lang:
            return lang
        if hasattr(instance, 'attack_attrs'):
            lang = instance.attack_attrs.get('language')
            if isinstance(lang, str) and lang:
                return lang
        return 'en'

    def _cot_suffix(self, language: str) -> str:
        lang = (language or 'en').lower()
        if any(tag in lang for tag in ("cn", "zh", "zho")):
            return '如果不确定，请基于常识给出你最合理的回答。'
        return 'If you are not sure, please provide your best effort answer based on general knowledge.'

    def get_conv_from_type(self, conv, prompt_type, user_prompt, assistant_prompt, input_msg, input_msg_cot):
        if prompt_type == 'DQ':
            conv.append_message(conv.roles[0], input_msg)
        elif prompt_type in {'JQ', 'JQ+MC'}:
            if user_prompt:
                conv.append_message(conv.roles[0], user_prompt)
                target_response = self.target_model.generate(user_prompt)
                conv.append_message(conv.roles[1], target_response)
            conv.append_message(conv.roles[0], input_msg_cot)
        elif 'JQ+COT' in prompt_type:
            if user_prompt:
                conv.append_message(conv.roles[0], user_prompt)
            if assistant_prompt:
                conv.append_message(conv.roles[1], assistant_prompt)
            conv.append_message(conv.roles[0], input_msg_cot)
        else:
            raise ValueError('Invalid prompt type')
        return conv
