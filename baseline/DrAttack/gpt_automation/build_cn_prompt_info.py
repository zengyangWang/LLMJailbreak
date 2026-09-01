#!/usr/bin/env python3
"""Generate Chinese prompt parsing entries with HanLP dependency parser."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List, Any

import hanlp

# 默认的中文 prompts 信息文件
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "attack_prompt_data",
    "prompts_information_cn.json",
)


def resolve_model(registry, name: str):
    """Resolve a registered HanLP model shortcut to its component or path."""
    if hasattr(registry, name):
        return getattr(registry, name)
    return name


def build_pipeline(
    tokenizer_name: str = "FINE_ELECTRA_SMALL_ZH",
    pos_name: str = "CTB9_POS_ELECTRA_SMALL",
    dep_name: str = "CTB9_DEP_ELECTRA_SMALL",
):
    pipeline = hanlp.pipeline()

    tok_component = resolve_model(hanlp.pretrained.tok, tokenizer_name)
    if isinstance(tok_component, str):
        tok_component = hanlp.load(tok_component)
    pipeline.append(tok_component, output_key="tok")

    if pos_name:
        pos_component = resolve_model(hanlp.pretrained.pos, pos_name)
        if isinstance(pos_component, str):
            pos_component = hanlp.load(pos_component)
        pipeline.append(pos_component, input_key="tok", output_key="pos")

    dep_component = resolve_model(hanlp.pretrained.dep, dep_name)
    if isinstance(dep_component, str):
        dep_component = hanlp.load(dep_component)
    pipeline.append(dep_component, input_key="tok", output_key="dep")

    return pipeline


def read_existing_data(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def write_data(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pos_to_type(pos: str) -> str:
    if not pos:
        return "structure"
    if pos.startswith("V"):
        return "verb"
    if pos.startswith("N"):
        return "noun"
    if pos.startswith("AD") or pos.startswith("D"):
        return "instruction"
    if pos in {"P", "CC", "CS", "DEC", "DEG", "DER", "SP", "LC", "PU"}:
        return "structure"
    return "instruction"


def label_for_pos(pos: str, relation: str, is_root: bool) -> (str, str):
    if pos.startswith("V"):
        return "Verb", "Verb Phrase" if is_root else "Verb"
    if pos.startswith("N") or pos.startswith("NR") or pos.startswith("NT"):
        return "Noun", "Noun Phrase" if relation in {"nsubj", "dobj", "obj", "obl", "root"} or is_root else "Noun"
    if pos.startswith("AD") or pos.startswith("D"):
        return "Adverb", "Adverbial Phrase" if is_root else "Adverb"
    if pos == "P":
        return "Preposition", "Prepositional Phrase" if is_root else "Preposition"
    if pos.startswith("A"):
        return "Adjective", "Adjective Phrase" if is_root else "Adjective"
    if pos in {"CC", "CS"}:
        return "Conjunction", "Conjunction Phrase" if is_root else "Conjunction"
    return "Token", "Clause" if is_root else "Token"


def merge_entry(container: Dict, key: str, value):
    if key not in container:
        container[key] = value
    else:
        existing = container[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            container[key] = [existing, value]


def build_tree(tokens: List[str], pos_tags: List[str], heads: List[int], relations: List[str]) -> Dict:
    children: Dict[int, List[int]] = defaultdict(list)
    root = None
    for idx, head in enumerate(heads):
        parent = head - 1  # HanLP 依存是 1-based，root 为 0
        if parent < 0:
            root = idx
        else:
            children[parent].append(idx)
    if root is None:
        raise ValueError("Failed to locate root in dependency parse result")

    def dfs(node: int, depth: int, is_root=False):
        token = tokens[node]
        pos = pos_tags[node] if pos_tags else ""
        relation = relations[node] if relations else ""
        word_label, phrase_label = label_for_pos(pos, relation, is_root)

        value_dict: Dict = {}
        merge_entry(value_dict, word_label, token)
        for child in children.get(node, []):
            child_dict = dfs(child, depth + 1)
            for k, v in child_dict.items():
                merge_entry(value_dict, k, v)

        if phrase_label == word_label:
            return {word_label: value_dict[word_label] if len(value_dict) == 1 else value_dict}
        return {phrase_label: value_dict}

    return dfs(root, 1, is_root=True)


def compute_levels(heads: List[int]) -> List[int]:
    children: Dict[int, List[int]] = defaultdict(list)
    root = None
    for idx, head in enumerate(heads):
        parent = head - 1
        if parent < 0:
            root = idx
        else:
            children[parent].append(idx)
    levels = [0] * len(heads)

    def assign(node: int, depth: int):
        levels[node] = depth
        for child in children.get(node, []):
            assign(child, depth + 1)

    if root is not None:
        assign(root, 1)
    return levels


def normalize_tree(node: Any, parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: normalize_tree(v, k) for k, v in node.items()}
    if isinstance(node, list):
        if len(node) == 1:
            return normalize_tree(node[0], parent_key)
        normalized = {}
        for idx, item in enumerate(node, start=1):
            key_base = parent_key if parent_key else "item"
            new_key = f"{key_base}_{idx}"
            normalized[new_key] = normalize_tree(item, parent_key)
        return normalized
    return node


def extract_fields(parsed: Any) -> (List[str], List[str], List[int], List[str]):
    if isinstance(parsed, dict):
        tokens = parsed.get("tok") or parsed.get("tok/fine") or parsed.get("tok/coarse")
        pos_tags = parsed.get("pos") or parsed.get("upos") or parsed.get("xpos") or []
        heads = parsed.get("head")
        relations = parsed.get("deprel", [])
        return tokens, pos_tags, heads, relations
    if hasattr(parsed, "tokens") or hasattr(parsed, "__iter__"):
        tokens = []
        pos_tags = []
        heads = []
        relations = []
        for token in parsed:
            form = None
            head = None
            rel = ""
            xpos = ""

            if isinstance(token, Mapping):
                form = token.get("form") or token.get("tok") or token.get("text")
                head = token.get("head")
                rel = token.get("deprel") or token.get("relation") or ""
                xpos = token.get("xpos") or token.get("upos") or token.get("pos") or ""
            else:
                form = getattr(token, "form", None) or getattr(token, "tok", None) or getattr(token, "text", None)
                head = getattr(token, "head", None)
                rel = getattr(token, "deprel", None) or getattr(token, "relation", None) or ""
                xpos = getattr(token, "xpos", None) or getattr(token, "upos", None) or getattr(token, "pos", None) or ""
            if form is None:
                continue
            tokens.append(form)
            pos_tags.append(xpos or "")
            heads.append(int(head) if head is not None else 0)
            relations.append(rel or "")
        return tokens, pos_tags, heads, relations
    return None, None, None, None


def create_entry(sentence: str, pipeline, prompt_id: int) -> Dict:
    parsed = pipeline(sentence)

    if isinstance(parsed, dict):
        tokens = parsed.get("tok")
        pos_tags = parsed.get("pos", [])
        dep = parsed.get("dep")
    else:
        tokens = None
        pos_tags = []
        dep = parsed

    if tokens is None:
        raise ValueError("HanLP pipeline output missing tokens (tok).")

    tokens, dep_pos, heads, relations = extract_fields(dep)
    if tokens is None or heads is None:
        raise ValueError("HanLP dependency output missing required fields 'tok' or 'head'.")

    if not pos_tags:
        pos_tags = dep_pos or [""] * len(tokens)
    elif len(pos_tags) != len(tokens):
        pos_tags = dep_pos or pos_tags

    tree = build_tree(tokens, pos_tags, heads, relations)
    tree = normalize_tree(tree)
    levels = compute_levels(heads)
    types = [pos_to_type(pos_tags[i] if pos_tags else "") for i in range(len(tokens))]

    return {
        "parsing_tree_dictionary": tree,
        "prompt_id": prompt_id,
        "prompt": sentence,
        "substitutable": [],
        "harmless": [],
        "synonym": {},
        "opposite": {},
        "words": tokens,
        "words_level": levels,
        "words_type": types,
    }


def generate_prompt_info(
    sentence: str,
    output_path: str = DEFAULT_OUTPUT,
    tokenizer_name: str = "FINE_ELECTRA_SMALL_ZH",
    pos_model_name: str = "CTB9_POS_ELECTRA_SMALL",
    dep_model_name: str = "CTB9_DEP_ELECTRA_SMALL",
    prompt_id: int | None = None,
) -> Dict:
    pipeline = build_pipeline(tokenizer_name, pos_model_name, dep_model_name)

    data = read_existing_data(output_path)
    if prompt_id is None:
        max_id = max((entry.get("prompt_id", 0) for entry in data.values()), default=0)
        prompt_id = max_id + 1

    entry = create_entry(sentence, pipeline, prompt_id)
    data[sentence] = entry
    write_data(output_path, data)
    return entry


def main():
    arg_parser = argparse.ArgumentParser(description="Generate Chinese prompt info using HanLP dependency parser")
    arg_parser.add_argument("sentence", help="Chinese sentence to parse")
    arg_parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to prompts_information_cn.json")
    arg_parser.add_argument("--tokenizer", default="FINE_ELECTRA_SMALL_ZH", help="HanLP tokenizer model name")
    arg_parser.add_argument("--pos-model", default="CTB9_POS_ELECTRA_SMALL", help="HanLP POS model name")
    arg_parser.add_argument("--dep-model", default="CTB9_DEP_ELECTRA_SMALL", help="HanLP dependency model name")
    arg_parser.add_argument("--prompt-id", type=int, default=None, help="Optional prompt id to assign")
    args = arg_parser.parse_args()

    entry = generate_prompt_info(
        sentence=args.sentence,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        pos_model_name=args.pos_model,
        dep_model_name=args.dep_model,
        prompt_id=args.prompt_id,
    )
    print(json.dumps(entry, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
