"""把上游已裁决的 token 边界转换为确定性半开区间。"""
from __future__ import annotations

from collections.abc import Iterable


def sentence_bounds(token_count: int, *, cut_after: Iterable[int] = ()) -> list[tuple[int, int]]:
    """按注入切点返回保序的 token 半开区间。

    ``cut_after`` 中的值直接表示半开区间的 ``end``，范围为 ``1..token_count``。
    本函数不读取 token 文本，也不判断码点、语言或结构作用；边界候选的产生、竞争和
    supersede 属于 occurrence/span 解析层。没有上游证据时只保留完整输入段。
    """
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("token_count 必须为整数")
    if token_count < 0:
        raise ValueError("token_count 不得为负数")
    if token_count == 0:
        return []

    cuts: set[int] = set()
    for end in cut_after:
        if isinstance(end, bool) or not isinstance(end, int):
            raise TypeError("切点必须为整数")
        if end <= 0 or end > token_count:
            raise ValueError("切点必须位于 1..token_count")
        cuts.add(end)
    cuts.add(token_count)

    spans: list[tuple[int, int]] = []
    start = 0
    for end in sorted(cuts):
        if end > start:
            spans.append((start, end))
            start = end
    return spans


__all__ = ["sentence_bounds"]
