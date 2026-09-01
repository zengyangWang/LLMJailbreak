from baseline.CodeChameleon.prompts import get_decryption_function, get_prompt_template
from data.language import Language


def get_attack(goal: str, language: Language) -> str:
    encrypted_goal = encrypt_binary_tree(goal)

    decryption_function = get_decryption_function(language)

    prompt_template = get_prompt_template(language)

    prompt = prompt_template.format(
        goal=encrypted_goal, decryption_function=decryption_function
    )

    return prompt


def encrypt_binary_tree(sentence: str) -> str:
    class TreeNode:
        """A node in the binary tree."""

        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def build_tree(words, start, end):
        """Builds the binary tree from the list of words."""
        if start > end:
            return None

        # Middle element to make it a balanced tree
        mid = (start + end) // 2
        node = TreeNode(words[mid])

        # Recursively build the left and right subtrees
        node.left = build_tree(words, start, mid - 1)
        node.right = build_tree(words, mid + 1, end)

        return node

    def tree_to_json(node):
        """Converts a tree to a JSON representation."""
        if node is None:
            return None
        return {
            "value": node.value,
            "left": tree_to_json(node.left),
            "right": tree_to_json(node.right),
        }

    words = sentence.split()
    root = build_tree(words, 0, len(words) - 1)
    tree_representation = tree_to_json(root)
    return tree_representation


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def CodeChameleon_single_main(
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

    attack = get_attack(goal, language)

    output_record["adv_prompt"] = attack

    return output_record
