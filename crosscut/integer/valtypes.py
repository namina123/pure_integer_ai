"""crosscut.integer.valtypes — 抽象值类型声明层（不含算术实现）。

定义 Rational 与 FixedQuotient 的不可变值类型骨架 + 不变量声明。
算术作为自由函数放在 rational.py / fixed_point.py（避免模块级真环：
rational.div 返回 FixedQuotient，FixedQuotient.to_rational_interval 返回 Rational）。

【longdiv 不变量】
  正确不变量： a·B^k == M·b + r ,  0 ≤ r < b   （M=(a·B^k)//b, r=(a·B^k)%b）
  值重建：     a/b == M/B^k + r/(b·B^k) ，误差 = r/(b·B^k) < 1/B^k
  故 FixedQuotient 必须携带 b（error_bound / to_rational_interval 要用）。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float


@dataclass(frozen=True)
class Rational:
    """有理数 num/den。den > 0（符号承载于 num）。纯值，无算术方法。"""

    num: int
    den: int = 1

    def __post_init__(self) -> None:
        # perf round3 fast path（cProfile n=4：__post_init__ 5.69M 调·16.2s）：num/den 均 int（>99% 常见）
        # -> 跳 assert_no_float 函数调用（type(v) is int C 级快于 isinstance·bit-identical·int 必非 float）。
        # 非 int（float 渗入·AST 禁·defense）落全检 assert_no_float。
        if type(self.num) is int and type(self.den) is int:
            if self.den <= 0:
                raise ValueError(f"Rational.den 须为正，got den={self.den}")
            return
        assert_no_float(self.num, self.den, _where="Rational")
        if self.den <= 0:
            raise ValueError(f"Rational.den 须为正，got den={self.den}")


@dataclass(frozen=True)
class FixedQuotient:
    """代分数定点商：值 = M/B^k + r/(b·B^k) = a/b（其中 a·B^k = M·b + r）。

    字段：
      M —— 定点整数（= (a·B^k)//b）
      r —— 余数，0 ≤ r < b（= (a·B^k)%b）
      k —— 精度位数（≥ 0）
      b —— 除数（> 0；error_bound / to_rational_interval 需要它）
    """

    M: int
    r: int
    k: int
    b: int

    def __post_init__(self) -> None:
        assert_no_float(self.M, self.r, self.k, self.b, _where="FixedQuotient")
        if self.b <= 0:
            raise ValueError(f"FixedQuotient.b 须为正，got b={self.b}")
        if self.k < 0:
            raise ValueError(f"FixedQuotient.k 须 ≥ 0，got k={self.k}")
        if not (0 <= self.r < self.b):
            raise ValueError(
                f"FixedQuotient.r 须 ∈ [0, b)，got r={self.r}, b={self.b}"
            )
