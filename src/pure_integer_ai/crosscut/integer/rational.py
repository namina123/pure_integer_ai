"""crosscut.integer.rational — Rational 算术（闭运算，永不 float）。

依赖 valtypes（值类型骨架）、compare（交叉积）。

API（自由函数式；canonical 构造器 make()）：
  make(num, den=1) —— 规范化 + gcd 归约
  ZERO / ONE / from_pair(=make 别名)
  add / sub / mul / neg / eq / sign / is_zero
  eq 走交叉积（compare.cross_eq）；并把 Rational 的 == / hash 接到交叉积，
  使裸 Rational(2,4) == Rational(1,2) 成立（"eq 走交叉积"落点）。

div 不在此实现（返回 FixedQuotient，依赖 fixed_point）——见 fixed_point.rational_div。
"""
from __future__ import annotations

import math

from pure_integer_ai.crosscut.integer.valtypes import Rational
from pure_integer_ai.crosscut.integer import compare
from pure_integer_ai.crosscut.guards import float_guard

ZERO = Rational(0, 1)
ONE = Rational(1, 1)


def make(num: int, den: int = 1) -> Rational:
    """canonical 构造：den>0（符号移到 num）+ gcd 归约。

    内联 math.gcd（C builtin）：PR 有理高斯消元热路径经 add/sub/mul 高频调 make，
    math.gcd 语义与 Euclid 逐位等价（同返非负 gcd），bit-identical。入口留单次
    assert_no_float 守纯整数（DEBUG 可关）。

    **perf round3 fast path**（cProfile n=4：make 76.9s/39%·5.69M 调）：
    - inline float 检查替 assert_no_float 函数调用（float_guard.DEBUG live-read·守 FloatViolation 语义）
    - num==0 -> ZERO 单例（gcd(0,den)=den->Rational(0,1)=ZERO·frozen 不可变复用安全·skip gcd+构造+__post_init__）
    - den==1 -> 跳 gcd（gcd(num,1)=1·Rational(num,1) 走 __post_init__ fast path 廉价）
    bit-identical：同 canonical 结果（数学等价·fast path 只跳冗余 gcd/构造）。
    """
    if den == 0:
        raise ValueError("Rational.den 须非零")
    if float_guard.DEBUG:
        if isinstance(num, float) or isinstance(den, float):
            raise float_guard.FloatViolation(
                f"float detected (rational.make): {num!r}, {den!r}")
    if den < 0:
        num, den = -num, -den
    if num == 0:
        return ZERO
    if den == 1:
        return Rational(num, 1)
    g = math.gcd(num, den)
    return Rational(num // g, den // g)


# API 名别名（与设计 API 对齐）
from_pair = make


def eq(a: Rational, b: Rational) -> bool:
    """交叉积判等：a/b == c/d ⟺ a·d == b·c。"""
    return compare.cross_eq(a.num, a.den, b.num, b.den)


def sign(a: Rational) -> int:
    """-1 / 0 / 1（符号承载于 num）。"""
    return (a.num > 0) - (a.num < 0)


def is_zero(a: Rational) -> bool:
    return a.num == 0


def neg(a: Rational) -> Rational:
    return Rational(-a.num, a.den)


def add(a: Rational, b: Rational) -> Rational:
    """a/b + c/d = (a·d + c·b)/(b·d)。"""
    return make(a.num * b.den + b.num * a.den, a.den * b.den)


def sub(a: Rational, b: Rational) -> Rational:
    """a/b − c/d = (a·d − c·b)/(b·d)。"""
    return make(a.num * b.den - b.num * a.den, a.den * b.den)


def mul(a: Rational, b: Rational) -> Rational:
    """a/b · c/d = (a·c)/(b·d)。"""
    return make(a.num * b.num, a.den * b.den)


def div(a: Rational, b: Rational):
    """Rational 除法不闭于 Rational（结果可能需定点）——见 fixed_point.rational_div。"""
    raise NotImplementedError(
        "Rational.div 在 fixed_point.rational_div 实现（返回 FixedQuotient）"
    )


# ---- 把 Rational 的 == / hash 接到交叉积（"eq 走交叉积"落点） ----

def _rational_eq(self: Rational, other: object) -> bool:
    if not isinstance(other, Rational):
        return NotImplemented
    return compare.cross_eq(self.num, self.den, other.num, other.den)


def _rational_hash(self: Rational) -> int:
    g = math.gcd(self.num, self.den)
    sn, sd = self.num // g, self.den // g
    if sd < 0:
        sn, sd = -sn, -sd
    return hash((sn, sd))


Rational.__eq__ = _rational_eq  # type: ignore[assignment]
Rational.__hash__ = _rational_hash  # type: ignore[assignment]
