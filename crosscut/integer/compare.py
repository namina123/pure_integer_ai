"""crosscut.integer.compare — 交叉积比较（全系统比序唯一零误差路径）。

比较两个有理数 a/b 与 c/d（分母 b,d > 0）的序，**不计算商**，全程整数乘法 +
比较，零误差、零定点。覆盖选优 / 阈值 / 排序。任何"比序"语义强制走本模块，
禁止走定点近似比大小（定点区间重叠时唯一可靠回退）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float


def cross_compare(a: int, b: int, c: int, d: int) -> int:
    """返回 sign(a·d − b·c) ∈ {-1,0,1}，即 a/b 与 c/d 的序（b,d>0）。"""
    assert_no_float(a, b, c, d, _where="cross_compare")
    if b <= 0 or d <= 0:
        raise ValueError(f"分母须为正: b={b}, d={d}")
    diff = a * d - b * c
    return (diff > 0) - (diff < 0)


def cross_lt(a: int, b: int, c: int, d: int) -> bool:
    """a/b < c/d。"""
    return cross_compare(a, b, c, d) < 0


def cross_le(a: int, b: int, c: int, d: int) -> bool:
    """a/b ≤ c/d。"""
    return cross_compare(a, b, c, d) <= 0


def cross_gt(a: int, b: int, c: int, d: int) -> bool:
    """a/b > c/d。"""
    return cross_compare(a, b, c, d) > 0


def cross_ge(a: int, b: int, c: int, d: int) -> bool:
    """a/b ≥ c/d。阈值判定常用：cross_ge(分子, 分母, p, q) 即"成功率 ≥ p/q"。"""
    return cross_compare(a, b, c, d) >= 0


def cross_eq(a: int, b: int, c: int, d: int) -> bool:
    """a/b == c/d。"""
    return cross_compare(a, b, c, d) == 0


# ---- 比较算子 opcode（刀 D 比较 cue·canonical ordering home） ----
# 语言 cue 层的比较方向标记（非 VM OPCODE_LT/GT·后者是图即程序符号域 symbol_domain）。
# comparison_op_of（cue_words.py）把比较 OP 词（大于/小于/不小于/不大于）映射到此 opcode·
# comparison_proof_fn（comparison_proof.py）dispatch 到 cross_compare 验序。纯整数·确定性。
CMP_GT = 1   # 大于 / greater_than  ·a > b
CMP_LT = 2   # 小于 / less_than     ·a < b
CMP_GE = 3   # 不小于 / at_least    ·a ≥ b
CMP_LE = 4   # 不大于 / at_most     ·a ≤ b
CMP_EQ = 5   # 等于 / equal_to      ·a == b（等式标记·非序方向·piece 2.1 比较族补全·cross_eq 验 sign==0）
