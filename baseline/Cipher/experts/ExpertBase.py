class ExpertBase:
    def __init__(
        self,
        system_prompt: str,
        demonstrations: list[str],
        constraints: str,
        demonstration_template: str,
    ):
        self.system_prompt = system_prompt
        self.demonstrations = demonstrations
        self.constraints = constraints
        self.demonstration_template = demonstration_template

    def get_attack(self, goal: str):
        encode_demonstrations = ""

        for i, demonstration in enumerate(self.demonstrations):
            encode_demonstrations += (
                self.demonstration_template.format(i + 1)
                + self.encode(demonstration)
                + "\n"
            )

        prompt = self.system_prompt + "\n" + encode_demonstrations + self.constraints

        encoded_query = self.encode(goal)

        return prompt.format(encoded_query=encoded_query)

    def encode(self, text: str) -> str:
        raise NotImplementedError

    def decode(self, text: str) -> str:
        raise NotImplementedError
