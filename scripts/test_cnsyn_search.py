#!/usr/bin/env python3
"""
一个简单的测试脚本，用于批量验证 cnsyn.search 的查询结果。
"""

import argparse
from typing import Iterable, List

try:
    from cnsyn import search  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - 运行环境缺依赖时提示
    raise ImportError(
        "无法导入 cnsyn 库，请先在当前环境安装或激活包含该库的环境。"
    ) from exc


def run_search_tests(words: Iterable[str], origins: Iterable[str], topk: int) -> None:
    """
    针对一组词和词源依次调用 cnsyn.search，并打印结果。
    """
    for word in words:
        print(f"\n==== 查询词：{word} ====")
        for origin in origins:
            try:
                result: List[str] = search(word, topK=topk, origin=origin)
            except Exception as exc:  # pragma: no cover - 仅用于排错输出
                print(f"[ERROR] origin={origin} 查询失败：{exc}")
                continue

            if not isinstance(result, list):
                print(f"[WARN] origin={origin} 返回类型异常：{type(result)}")
                continue

            print(f"origin={origin:>7} | topK={topk:>2} | 命中数量={len(result):>2}")
            if result:
                preview = "、".join(result[: min(10, len(result))])
                print(f"  → 预览：{preview}")
            else:
                print("  → 暂无结果")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 cnsyn.search 功能")
    parser.add_argument(
        "--words",
        nargs="+",
        default=["中山广场", "政府"],
        help="需要测试的查询词（默认包含两个示例）。",
    )
    parser.add_argument(
        "--origins",
        nargs="+",
        default=["all", "wiki", "cndict"],
        help="词源过滤，可选值包括 all/wiki/cndict（默认全测）。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="每个查询返回的最大同义词数量。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk 必须为正整数")
    run_search_tests(args.words, args.origins, args.topk)


if __name__ == "__main__":
    main()

