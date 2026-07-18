"""crosscut.integer.isqrt — 纯整数开方（图代数脱离浮点的支点）。

图代数的谱半径 / 向量范数都含开方（√λmax、‖v‖=√Σvᵢ²）。纯整数开方让图代数
全程落在整数域——"后继推理脱离浮点"的支点。

数学事实：代数数 √n 绝对可表示（最小多项式 x²−n 的正根），不是"逼近一个未知值"。
本模块给出 √n 的按需任意精度整数展开——floor(√n · S)（S = 整数缩放因子），
与 longdiv 代分数同构（定点商 + 误差界）。

吸收来源（诚实标注，不贪功）：
  - 整数牛顿迭代 x ← (x + n//x)//2（经典，自上界初值单调收敛到 floor(√n)）。
  - digit-by-digit 逐位开方等价但更繁；牛顿法同精度更简，采用之。

API（自由函数式；原子 isqrt_floor + 精度包装，与 rational/algebraic_fraction 同风格）：
  isqrt_floor(n)            —— floor(√n)（整数牛顿，零浮点，移植锚点）
  isqrt(n, precision_bits=0) —— floor(√n · 2^precision_bits)（位精度主接口）
  sqrt_scaled(n, scale=1)   —— floor(√n · scale)（通用定点；scale=10^d→十进制 d 位）
  SqrtRef(n)                —— 代数数 √n 的符号持有（精确身份），按需展开任意精度

不变量（floor 语义，定义即正确，无需外部 oracle）：
  原子  M = isqrt_floor(N)   ⟹  M² ≤ N < (M+1)²  且 M ≥ 0
  定点  M = sqrt_scaled(n,S) ⟹  M² ≤ n·S² < (M+1)²（真值 √n·S ∈ [M, M+1)，误差 < 1/S）

确定性：整数牛顿单调收敛，同入同出（可复现）。零浮点（assert_no_float 守入口）。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float


def isqrt_floor(n: int) -> int:
    """floor(√n)（n ≥ 0）。整数牛顿迭代，零浮点，移植锚点。

    初值取 2^ceil(bitlen/2)（保证 ≥ √n 的上界），迭代 x ← (x + n//x)//2 单调不增，
    收敛到 floor(√n)。终止判据：后继 y ≥ 当前 x 时返回 x（已到不动点 floor）。
    守不变量 M² ≤ n < (M+1)²（断言式门，违例即 bug）。
    """
    assert_no_float(n, _where="isqrt_floor")
    if n < 0:
        raise ValueError(f"isqrt_floor: n 须非负，got n={n}")
    if n < 2:
        return n  # 0 → 0, 1 → 1
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            break
        x = y
    assert x * x <= n < (x + 1) * (x + 1), f"isqrt_floor 不变量破坏: n={n}, x={x}"
    return x


def sqrt_scaled(n: int, scale: int = 1) -> int:
    """floor(√n · scale)（通用定点整数开方）。scale ≥ 0，n ≥ 0。

    归约到原子：floor(√n · scale) = isqrt_floor(n · scale²)。
    误差 < 1/scale（真值 √n·scale ∈ [M, M+1)）。scale=1 → 整数开方；scale=10^d → d 位十进制。
    """
    assert_no_float(n, scale, _where="sqrt_scaled")
    if n < 0:
        raise ValueError(f"sqrt_scaled: n 须非负，got n={n}")
    if scale < 0:
        raise ValueError(f"sqrt_scaled: scale 须非负，got scale={scale}")
    if scale == 0:
        return 0
    return isqrt_floor(n * scale * scale)


def isqrt(n: int, precision_bits: int = 0) -> int:
    """floor(√n · 2^precision_bits)（位精度主接口）。n ≥ 0，precision_bits ≥ 0。

    图代数范数/谱半径自然落在二进制定点（向量范数 ‖v‖² = Σvᵢ² 是整数，开方按位展开）。
    归约：floor(√n · 2^p) = isqrt_floor(n << (2·p)) = sqrt_scaled(n, 1<<p)。
    误差 < 2^(-precision_bits)（真值 ∈ [M/2^p, (M+1)/2^p)）。
    """
    assert_no_float(n, precision_bits, _where="isqrt")
    if n < 0:
        raise ValueError(f"isqrt: n 须非负，got n={n}")
    if precision_bits < 0:
        raise ValueError(f"isqrt: precision_bits 须 ≥ 0，got p={precision_bits}")
    return isqrt_floor(n << (2 * precision_bits))


@dataclass(frozen=True)
class SqrtRef:
    """代数数 √n 的符号持有（精确身份，不存近似值）。

    设计核心：代数数以符号持有（最小多项式 x²−n 的正根 = 精确身份），
    用时按需展开任意精度整数定点。SqrtRef 只存被开方数 n（精确），展开是派生（可复算）。
    谱半径 √λmax / 向量范数开方经此持有——代数数精确收敛非固定精度逼近。

    expand(scale) / expand_bits(p) / expand_digits(d) 按需给整数定点（floor(√n·S)），
    均守 floor 不变量（M² ≤ n·S² < (M+1)²），误差 < 1/scale。
    """

    n: int

    def __post_init__(self) -> None:
        assert_no_float(self.n, _where="SqrtRef")
        if self.n < 0:
            raise ValueError(f"SqrtRef.n 须非负，got n={self.n}")

    def expand(self, scale: int = 1) -> int:
        """按需展开：floor(√n · scale)（通用定点）。"""
        return sqrt_scaled(self.n, scale)

    def expand_bits(self, precision_bits: int = 0) -> int:
        """按需展开：floor(√n · 2^precision_bits)（位精度）。"""
        return isqrt(self.n, precision_bits)

    def expand_digits(self, num_digits: int = 0) -> int:
        """按需展开：floor(√n · 10^num_digits)（十进制 num_digits 位）。"""
        assert_no_float(num_digits, _where="SqrtRef.expand_digits")
        if num_digits < 0:
            raise ValueError(f"expand_digits: num_digits 须 ≥ 0，got {num_digits}")
        return sqrt_scaled(self.n, 10 ** num_digits)
