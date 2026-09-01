import torch
import fastchat 
from utils.string_utils import compose_system_prompt

def load_conversation_template(template_name, language=None):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    # 注入语言化的 system prompt
    try:
        orig_system = getattr(conv_template, "system", "") or getattr(conv_template, "system_message", "")
        final_system = compose_system_prompt(orig_system, language)
        if hasattr(conv_template, "set_system_message"):
            conv_template.set_system_message(final_system)
        elif hasattr(conv_template, "system"):
            conv_template.system = final_system
        elif hasattr(conv_template, "system_message"):
            conv_template.system_message = final_system
    except Exception:
        pass
    
    return conv_template


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)

        # `prompt` is returned at the end of this method.  Keep the tokenization
        # that corresponds to this exact string so get_input_ids() does not
        # tokenize the same prompt a second time.
        full_prompt_toks = encoding.input_ids
        toks = full_prompt_toks

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.instruction else ''
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
            else:
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        # Keep the tokens for the exact prompt returned above.  The local
        # variable `toks` is overwritten while calculating the different slice
        # boundaries, so use the separately preserved full-prompt tokens.
        self._last_prompt_toks = full_prompt_toks

        self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        self.get_prompt(adv_string=adv_string)
        input_ids = torch.tensor(
            self._last_prompt_toks[:self._target_slice.stop],
            dtype=torch.long,
        )

        return input_ids
