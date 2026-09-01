import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from data.language import Language


# -----------------------------
# Lexicon loading
# -----------------------------


def _resolve_vocab_dir(vocab_dir: str) -> str:
    """
    Resolve vocab_dir to an absolute path.
    - If vocab_dir is absolute: keep it.
    - If relative: resolve against repo root (two levels up from baseline/*).
    """
    if os.path.isabs(vocab_dir):
        return vocab_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    return os.path.join(repo_root, vocab_dir)


def load_sensitive_words(vocab_dir: str) -> List[str]:
    """
    从目录下所有 .txt 文件加载敏感词，每行一个词条，空行跳过。
    """
    terms: List[str] = []
    vocab_dir = _resolve_vocab_dir(vocab_dir)
    if not os.path.isdir(vocab_dir):
        return terms
    for name in os.listdir(vocab_dir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(vocab_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    term = line.strip()
                    if term:
                        terms.append(term)
        except Exception:
            continue
    # 去重并按长度降序排序（有利于一些简单策略）
    uniq = sorted(set(terms), key=lambda x: (-len(x), x))
    return uniq


# -----------------------------
# Trie for maximum forward matching
# -----------------------------


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"]
    is_end: bool

    def __init__(self) -> None:
        self.children = {}
        self.is_end = False


class SensitiveTrie:
    def __init__(self, terms: Iterable[str]) -> None:
        self.root = TrieNode()
        self.max_term_len = 0
        for term in terms:
            self._insert(term)

    def _insert(self, term: str) -> None:
        node = self.root
        for ch in term:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        self.max_term_len = max(self.max_term_len, len(term))

    def find_longest_matches(self, text: str) -> List[Tuple[int, int, str]]:
        """
        贪心最大正向匹配：返回不重叠的“最长词条”命中。
        返回 (start, end_exclusive, matched_term) 列表。
        """
        matches: List[Tuple[int, int, str]] = []
        i = 0
        n = len(text)
        while i < n:
            node = self.root
            j = i
            last_end = -1
            last_term = ""
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_end:
                    last_end = j
                    last_term = text[i:last_end]
            if last_end != -1:
                matches.append((i, last_end, last_term))
                i = last_end
            else:
                i += 1
        return matches


# -----------------------------
# Pinyin rendering
# -----------------------------


def _lazy_import_pypinyin():
    try:
        from pypinyin import Style, pinyin  # type: ignore

        return Style, pinyin
    except Exception as e:
        raise ImportError(
            "缺少依赖 pypinyin：请先安装 `pypinyin`（已在 requirements.txt 中补充）。"
        ) from e


def to_pinyin_concat(text: str) -> str:
    """转拼音（无声调）并拼接。"""
    Style, pinyin = _lazy_import_pypinyin()
    syls = pinyin(text, style=Style.NORMAL, strict=False)
    return "".join(s[0] for s in syls)


def to_pinyin_spaced(text: str) -> str:
    Style, pinyin = _lazy_import_pypinyin()
    syls = pinyin(text, style=Style.NORMAL, strict=False)
    return " ".join(s[0] for s in syls)


def to_pinyin_dotted(text: str) -> str:
    Style, pinyin = _lazy_import_pypinyin()
    syls = pinyin(text, style=Style.NORMAL, strict=False)
    return ".".join(s[0] for s in syls)


def to_initials(text: str) -> str:
    Style, pinyin = _lazy_import_pypinyin()
    syls = pinyin(text, style=Style.NORMAL, strict=False)
    return "".join(s[0][0] for s in syls if s and s[0])


STYLE_FUNCS = {
    "concat": to_pinyin_concat,
    "space": to_pinyin_spaced,
    "dot": to_pinyin_dotted,
    "initials": to_initials,
}


def replace_with_spans(text: str, spans: List[Tuple[int, int, str]], convert) -> str:
    """
    按 spans 替换并重建文本：matched_term -> convert(term)。
    spans 需不重叠且按 start 升序（find_longest_matches 已满足）。
    """
    if not spans:
        return text
    parts: List[str] = []
    prev = 0
    for start, end, term in spans:
        if start > prev:
            parts.append(text[prev:start])
        parts.append(convert(term))
        prev = end
    if prev < len(text):
        parts.append(text[prev:])
    return "".join(parts)


# -----------------------------
# Baseline interface
# -----------------------------


class Args:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def _is_zh_language(language: Language) -> bool:
    return language in (Language.CHINESE_SIMPLIFIED, Language.CHINESE_TRADITIONAL)


def _parse_styles(styles: Optional[Sequence[str]], styles_str: Optional[str]) -> List[str]:
    if styles_str:
        items = [s.strip() for s in str(styles_str).split(",")]
        styles = [s for s in items if s]
    if not styles:
        styles = ["concat", "space", "dot", "initials"]
    return [s for s in styles if s in STYLE_FUNCS]


def SensitivePinyin_single_main(
    args_dict: dict,
    goal: str,
    target: str,
    language: Language,
):
    """
    敏感词拼音替换攻击（仅生成 prompt，不调用目标模型、不做评估）。

    产出字段：
    - adv_prompt: 取第一条生成的 prompt（无命中则为 "NULL"）
    - all_generated_prompts: 所有风格生成的 prompt 列表
    """
    args = Args(args_dict)

    output_record = {
        "original_prompt": goal,
        "target": target,
        "attack_prompt": "NULL",
        "adv_prompt": "NULL",
        "all_generated_prompts": [],
        "attack_iterations": 0,
        "language_model_output": "SKIPPED (Generation Only)",
        "is_JB_Judge": None,
        "is_JB": False,
        "is_JB_Agent": None,
        # 额外元信息（方便导出分析）
        "attack_method": "SensitivePinyin",
        "lang_used": "zh" if _is_zh_language(language) else "non-zh",
        "template_id": "sensitive_pinyin",
        "template_name": "Sensitive Pinyin",
    }
    seed_text = str(goal or "").strip()
    if not _is_zh_language(language):
        prompts = list(set([seed_text]))
        output_record["all_generated_prompts"] = prompts
        return output_record

    if not seed_text:
        prompts = list(set([seed_text]))
        output_record["all_generated_prompts"] = prompts
        return output_record

    vocab_dir = getattr(args, "sensitive_vocab_dir", "data/SensitiveVocabulary")
    styles = _parse_styles(
        getattr(args, "sensitive_pinyin_styles", None),
        getattr(args, "sensitive_pinyin_styles_str", None),
    )

    terms = load_sensitive_words(vocab_dir)
    trie = SensitiveTrie(terms)
    spans = trie.find_longest_matches(seed_text)
    if not spans:
        # 如果没有检测到敏感词，则 prompts 只包含原始文本
        prompts = [seed_text]
    else:
        prompts: List[str] = []
        for style in styles:
            func = STYLE_FUNCS[style]
            jailbreak = replace_with_spans(seed_text, spans, func)
            prompts.append(jailbreak)

    output_record["all_generated_prompts"] = prompts
    output_record["attack_iterations"] = len(prompts)
    output_record["adv_prompt"] = prompts[0] if prompts else "NULL"
    output_record["matched_terms"] = [t for (_, _, t) in spans] if spans else []
    output_record["matched_span_count"] = len(spans) if spans else 0

    return output_record


