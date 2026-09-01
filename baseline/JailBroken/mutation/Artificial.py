import json
from pathlib import Path
from typing import List

from .mutation_base import MutationBase, Instance


def _language_key(language) -> str:
    if hasattr(language, "name"):
        language = language.name
    if isinstance(language, str):
        lower = language.lower()
        if "chinese" in lower or lower.startswith("zh") or lower.startswith("cn"):
            return "cn"
        return "en"
    return "en"


class Artificial(MutationBase):
    """
    Artificial mutation rule for Jailbroken. Loads prompts from the local
    ``jailbroken_template.json`` file instead of the EasyJailbreak seed helper.
    """

    def __init__(self, attr_name: str = "query"):
        self.attr_name = attr_name
        template_path = Path(__file__).resolve().parents[1] / "jailbroken_template.json"
        try:
            with template_path.open("r", encoding="utf-8") as fp:
                self.prompts = json.load(fp)
        except FileNotFoundError:
            self.prompts = {"en": [], "cn": []}

    def _get_mutated_instance(self, instance) -> List[Instance]:
        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Attribute '{self.attr_name}' not found in instance")

        mutated_results: List[Instance] = []
        lang_key = _language_key(getattr(instance, "language", "en"))
        prompt_seeds = self.prompts.get(lang_key) or self.prompts.get("en", [])

        for prompt_seed in prompt_seeds:
            new_instance = instance.copy()
            for value in prompt_seed.values():
                new_instance.jailbreak_prompt = value
            new_instance.parents.append(instance)
            instance.children.append(new_instance)
            mutated_results.append(new_instance)

        return mutated_results
