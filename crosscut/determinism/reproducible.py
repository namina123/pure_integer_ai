"""crosscut.determinism.reproducible — assert_reproducible（确定性核验）。

同 seed 跑两遍，结果不同即抛（检出非确定性），一致则写 golden 并返回 True。

先做值比较（能直接检出 random 等非确定性，包括返回 float 的情况），
再哈希记 golden（若结果含不可哈希/float 类型会抛 TypeError——本身是 core 违规信号）。
"""
from __future__ import annotations

from typing import Any, Callable

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.determinism.golden import golden


def assert_reproducible(run_fn: Callable[[int], Any], seed: int) -> bool:
    """同 seed 跑两遍；结果不同即抛（检出非确定性），一致则写 golden 并返回 True。"""
    r1 = run_fn(seed)
    r2 = run_fn(seed)
    if r1 != r2:
        raise AssertionError(
            "assert_reproducible: 两遍结果不一致——存在非确定性"
        )
    h1 = Hasher(seed).h63(r1)
    golden.record(seed, r1, h1)
    return True
