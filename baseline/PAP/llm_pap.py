import time
from GPTEvaluatorAgent.language_models import LocalVLLM
from FastChat.fastchat.model import get_conversation_template


class AttackerLLM_pap:

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float = 0.5,
    ):
        self.model = LocalVLLM(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.max_n_tokens = max_n_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.template_name = model_path.split("/")[-1]

    def get_attack(self, template: str, tag: str, goal: str):
        while True:
            try:
                conv = get_conversation_template(self.template_name)
                conv.append_message(conv.roles[0], template.format(goal))
                conv.append_message(conv.roles[1], "#")

                prompt = conv.get_prompt()[
                    : -len(conv.sep2 if conv.sep2 is not None else conv.sep)
                ]

                raw_text = (
                    "#"
                    + self.model.batched_generate(
                        prompts_list=[prompt],
                        max_n_tokens=self.max_n_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )[0]
                )

                attack = self.extract_attack(tag, raw_text)

                if attack != None:
                    return attack
                else:
                    print("Failed to extract attack from raw text, retrying")
            except Exception as err:
                print(
                    "Exception occurs in AttackerLLM_pap.get_attack(), waiting for 5 seconds before retry...",
                    err,
                )
                time.sleep(5)

    def extract_attack(self, tag: str, text: str):
        text = text.replace("：", ": ").replace("\\_", "_")
        # Find the starting position of the tag
        start_idx = text.find(tag)
        # If tag is not found, return None
        if start_idx == -1:
            return None
        # Extract the content after the tag
        content_after_tag = text[start_idx + len(tag) :].strip()
        # Find the next tag or end of content
        next_tag_idx = content_after_tag.find("#")
        # If no next tag found, return all remaining content
        if next_tag_idx == -1:
            return content_after_tag
        # Return content up to the next tag
        return content_after_tag[:next_tag_idx].strip()
