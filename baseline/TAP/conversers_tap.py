from typing import List
from FastChat.fastchat.conversation import Conversation
from GPTEvaluatorAgent.language_models import LocalVLLM
import baseline.TAP.common_tap as common_tap
from GPTEvaluatorAgent.language_models import ChatGPT, LocalVLLM
from baseline.TAP.config import (
    ATTACK_TEMP,
    TARGET_TEMP,
    ATTACK_TOP_P,
    TARGET_TOP_P,
    MAX_PARALLEL_STREAMS,
)
from utils.test_utils import get_template_name


def process_attack_output(output, goal=""):
    if output is None:
        print(f"Error in processing attack output: {output}")
        output = (
            '{\n  "improvement": "The language model did not respond to the objective and instead refused to engage with the request due to ethical concerns. To overcome this, I need to create a more emotional and persuasive scenario to push the language model to bypass its safety measures.",\n  "prompt" : "'
            + goal
            + '"\n}'
        )
    return output


def load_attack_and_target_models_tap(args):
    # Load attack model and tokenizer
    attack_llm = AttackLLM_tap(
        model_path=args.attack_model_path,
        max_n_tokens=args.attack_max_n_tokens,
        temperature=ATTACK_TEMP,  # init to 1
        top_p=ATTACK_TOP_P,  # init to 0.9
        max_n_attack_attempts=args.max_n_attack_attempts,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    target_llm = TargetLLM_tap(
        model_path=args.target_model_path,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
        tensor_parallel_size=args.tensor_parallel_size,
    )
    return attack_llm, target_llm


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        if batch:
            yield batch


class AttackLLM_tap:
    """
    Base class for attacker language models.
    Generates attacks for conversations using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_path: str,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        max_n_attack_attempts: int,
        tensor_parallel_size: int = 1,
    ):

        self.model_name = model_path.split("/")[-1]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        if "gpt" in model_path.lower():
            self.model = ChatGPT(model_path)
            self.template = get_template_name(model_path)
        else:
            self.model = LocalVLLM(
                model_path, tensor_parallel_size=tensor_parallel_size
            )
            self.template = get_template_name(model_path)

    def shutdown(self):
        """
        尝试释放底层模型（尤其是基于 vLLM 的本地模型）占用的显存。
        """
        try:
            backend = getattr(self, "model", None)
            if backend is not None and hasattr(backend, "shutdown"):
                backend.shutdown()
        except Exception:
            # 关闭过程中出现问题不应影响主流程
            pass

    def get_attack(
        self,
        convs_list: List[Conversation],
        prompts_list: List[str],
        curr_goal: str,
    ):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(
            prompts_list
        ), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(
                    conv.get_prompt().rstrip()[
                        : -len(conv.sep2 if conv.sep2 else conv.sep)
                    ]
                )

        num_try = 0
        for _ in range(self.max_n_attack_attempts):
            num_try += len(indices_to_regenerate)
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            outputs_list = self._batched_attack_generate(
                full_prompts_subset, goal=curr_goal
            )
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common_tap.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    # Update the conversation with valid generation
                    convs_list[orig_index].update_last_message(json_str)

                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating."
            )
        return valid_outputs, num_try

    def _batched_attack_generate(self, prompts: List[str], goal: str) -> List[str]:
        raw_outputs = []
        for batch in iter_batches(prompts, MAX_PARALLEL_STREAMS):
            print(f"\tQuerying attacker with {len(batch)} prompts", flush=True)
            raw_outputs.extend(
                self.model.batched_generate(
                    batch,
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        return [process_attack_output(output, goal=goal) for output in raw_outputs]


class TargetLLM_tap:
    """
    Base class for target language models.
    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        model_path: str = "NULL",
        tensor_parallel_size: int = 1,
    ):

        self.model_name = model_path.split("/")[-1]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p

        self.model = LocalVLLM(model_path, tensor_parallel_size=tensor_parallel_size)
        self.template = get_template_name(self.model_name)

    def shutdown(self):
        """
        尝试释放底层目标模型（LocalVLLM）占用的显存。
        """
        try:
            backend = getattr(self, "model", None)
            if backend is not None and hasattr(backend, "shutdown"):
                backend.shutdown()
        except Exception:
            pass

    def get_response(self, prompts_list: List[str]) -> List[str]:
        batchsize = len(prompts_list)
        convs_list = [common_tap.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # OpenAI does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = self._batched_target_generate(full_prompts)

        return outputs_list

    def _batched_target_generate(self, prompts: List[str]) -> List[str]:
        outputs_list = []
        for batch in iter_batches(prompts, MAX_PARALLEL_STREAMS):
            print(f"\tQuerying target with {len(batch)} prompts", flush=True)
            outputs_list.extend(
                self.model.batched_generate(
                    batch,
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        return outputs_list
