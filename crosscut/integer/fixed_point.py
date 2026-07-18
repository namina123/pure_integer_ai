"""crosscut.integer.fixed_point — 定点 M/B^k 算术 + 区间回退 + Rational.div。

FixedQuotient(M, r, k, b) 的值 = M/B^k + r/(b·B^k)。算术在"定点输入"上精确：
  add/sub ：(M1±M2)/B^k，对定点输入精确，无除法。
  mul     ：(M1·M2)/B^{2k} 截断回 k 位（一次 ÷B^k）。
  weighted_sum ：Σ w_i·M_i 的整数运算，全程无除法。
误差界 error_bound = r/(b·B^k) < 1/B^k（仅对 longdiv 直出精确；算术后为粗界）。
真值粗区间 to_rational_interval = [M/B^k, (M+1)/B^k)。

Rational.div（延后到此）= rational_div：a/b → longdiv 定点商。
"""
from __future__ import annotations

from typing import Iterable

from pure_integer_ai.crosscut.integer.constants import BASE, DEFAULT_K
from pure_integer_ai.crosscut.integer.valtypes import FixedQuotient, Rational
from pure_integer_ai.crosscut.integer import compare
from pure_integer_ai.crosscut.integer.algebraic_fraction import longdiv


def _require_same_k(a: FixedQuotient, b: FixedQuotient) -> int:
    if a.k != b.k:
        raise ValueError(f"定点运算要求同 k: {a.k} vs {b.k}")
    return a.k


def add(a: FixedQuotient, b: FixedQuotient) -> FixedQuotient:
    """定点加（同 k）：(M1+M2)/B^k。"""
    k = _require_same_k(a, b)
    return FixedQuotient(a.M + b.M, 0, k, 1)


def sub(a: FixedQuotient, b: FixedQuotient) -> FixedQuotient:
    """定点减（同 k）：(M1−M2)/B^k。"""
    k = _require_same_k(a, b)
    return FixedQuotient(a.M - b.M, 0, k, 1)


def mul(a: FixedQuotient, b: FixedQuotient) -> FixedQuotient:
    """定点乘（同 k）：(M1·M2)/B^{2k} 截断回 k 位（一次 ÷B^k）。"""
    k = _require_same_k(a, b)
    p = a.M * b.M
    bk = BASE ** k
    return FixedQuotient(p // bk, p % bk, k, bk)


def weighted_sum(weights: Iterable[int], fqs: Iterable[FixedQuotient]) -> FixedQuotient:
    """加权求和（同 k）：(Σ w_i·M_i)/B^k。全程整数，无除法。"""
    fqs = list(fqs)
    if not fqs:
        raise ValueError("weighted_sum: 空列表")
    k = fqs[0].k
    for f in fqs:
        if f.k != k:
            raise ValueError(f"weighted_sum 要求同 k: {f.k} vs {k}")
    m_total = 0
    for w, f in zip(list(weights), fqs):
        m_total += w * f.M
    return FixedQuotient(m_total, 0, k, 1)


def to_rational_interval(fq: FixedQuotient) -> tuple[Rational, Rational]:
    """真值粗区间 [M/B^k, (M+1)/B^k)——真值 ∈ [lo, hi)。"""
    bk = BASE ** fq.k
    return (Rational(fq.M, bk), Rational(fq.M + 1, bk))


def error_bound(fq: FixedQuotient) -> Rational:
    """精确误差界 r/(b·B^k) < 1/B^k。"""
    bk = BASE ** fq.k
    return Rational(fq.r, fq.b * bk)


def rational_div(a: Rational, b: Rational, k: int = DEFAULT_K) -> FixedQuotient:
    """Rational 除法（Rational.div 的实现）：a/b → 定点商。

    a/b = (a.num·b.den)/(a.den·b.num)；归一到除数>0 后 longdiv。
    """
    num = a.num * b.den
    den = a.den * b.num
    if den == 0:
        raise ZeroDivisionError("rational_div: 除数为 0")
    if den < 0:
        num, den = -num, -den
    return longdiv(num, den, k)


# ---- 定点比序：区间不重叠直接比 M，重叠回退交叉积（constants.ON_INTERVAL_OVERLAP） ----

def compare_fq(a: FixedQuotient, b: FixedQuotient) -> int:
    """两个定点商的比序。同 k 同 b 时直接比 M；否则回退交叉积（零误差）。

    定点近似比大小在区间重叠时不可靠，强制回退交叉积（全系统比序唯一零误差路径）。
    """
    if a.k == b.k and a.b == b.b:
        # 同精度同除数：M 大者值大（M/B^k 严格单调）
        return (a.M > b.M) - (a.M < b.M)
    # 回退交叉积：a ≈ a.M/a.bk，b ≈ b.M/b.bk（用 M/B^k 主项）
    bk_a = BASE ** a.k
    bk_b = BASE ** b.k
    # a/bk_a vs b.M/bk_b ⟹ a.M·bk_b vs b.M·bk_a
    diff = a.M * bk_b - b.M * bk_a
    return (diff > 0) - (diff < 0)
