"""crosscut.determinism.cross_radix — longdiv 真值桥（直接版 vs limb 版逐位比对）。

cross_radix_check：比对 longdiv 与 longdiv_limb 在各 (a,b,k) 上的 (M,r)；
空 DiffReport = 两实现逐位一致（移植锚点可信）。

这是确定性验证族的一环：两个独立实现（直接大整数 divmod vs 显式 base-2^30 schoolbook）
交叉对拍，任一实现有 bug 即报差异。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.integer.algebraic_fraction import longdiv, longdiv_limb


@dataclass
class DiffReport:
    """longdiv vs longdiv_limb 的逐位差异报告（空 = 一致）。"""

    diffs: list  # list[tuple]

    def __bool__(self) -> bool:
        return bool(self.diffs)


def cross_radix_check(cases: list[tuple[int, int, int]] | None = None) -> DiffReport:
    """比对 longdiv 与 longdiv_limb 在各 (a,b,k) 上的 (M,r)；空 DiffReport=一致。"""
    if cases is None:
        cases = [
            (1, 3, 4), (7, 3, 2), (0, 5, 8), (-1, 3, 4), (-7, 3, 2),
            (2, 7, 3), (123456789, 1000, 3), (-(2 ** 50), 99991, 5),
        ]
    diffs: list = []
    for a, b, k in cases:
        d = longdiv(a, b, k)
        L = longdiv_limb(a, b, k)
        if d.M != L.M or d.r != L.r:
            diffs.append((a, b, k, ("direct", d.M, d.r), ("limb", L.M, L.r)))
    return DiffReport(diffs)
