from FastChat.fastchat.model.model_adapter import get_conversation_template
from GPTEvaluatorAgent.language_models import LocalVLLM


class TargetLLM_translate:

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

    def get_response(self, attack: str):
        conv = get_conversation_template(self.template_name)
        conv.append_message(conv.roles[0], attack)
        conv.append_message(conv.roles[1], None)

        return self.model.batched_generate(
            prompts_list=[conv.get_prompt()],
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )[0]


