"""crosscut.integer.algebraic_fraction — 代分数引擎 RA(B,k)（禁浮点的解药 + mod 原子源）。

longdiv_{B,k}(a, b) = FixedQuotient(M, r, k, b)，守正确不变量：

      a · B^k == M · b + r ,   0 ≤ r < b        （M = (a·B^k)//b，r = (a·B^k)%b）

值重建： a/b == M/B^k + r/(b·B^k) ，误差 = r/(b·B^k) < 1/B^k （由不变量直接推出）。
B = 2^30（大数器天然 base），B^k = 2^{30k}，定点商 M/B^k = M >> 30k（右移，跨宿主 bit 一致）。

两个实现，交叉验证一致（determinism.cross_radix_check 的真值桥素材）：
  longdiv      —— 直接大整数 divmod（Python 任意精度，正确性 oracle）。
  longdiv_limb —— 显式 base-2^30 schoolbook（移植锚点，C/Rust 1:1 对照）。
两者均按 floor 语义处理负被除数（余数 r ∈ [0, b)）。

mod(a,b) == a % b == longdiv(a,b,k=0).r  —— mod 单一真相源委托此处。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.integer.constants import BASE, DEFAULT_K
from pure_integer_ai.crosscut.integer.valtypes import FixedQuotient
from pure_integer_ai.crosscut.integer import bigint
from pure_integer_ai.crosscut.guards.float_guard import assert_no_float


def longdiv(a: int, b: int, k: int = DEFAULT_K) -> FixedQuotient:
    """代分数归纳除法（直接大整数 divmod；正确性 oracle）。

    返回 FixedQuotient(M, r, k, b)：M=(a·B^k)//b，r=(a·B^k)%b，守不变量。
    b 须 > 0；r ∈ [0, b)（Python floor divmod，负被除数亦然）。
    """
    assert_no_float(a, b, k, _where="longdiv")
    if b <= 0:
        raise ValueError(f"longdiv: 除数 b 须为正，got b={b}")
    if k < 0:
        raise ValueError(f"longdiv: 精度 k 须 ≥ 0，got k={k}")
    bk = BASE ** k
    dividend = a * bk
    M = dividend // b
    r = dividend % b
    # 断言式门：守正确不变量
    assert dividend == M * b + r, f"longdiv 不变量破坏: {a}·B^{k} != M·b+r"
    assert 0 <= r < b, f"longdiv 余数越界: r={r}, b={b}"
    return FixedQuotient(M, r, k, b)


def _schoolbook_divmod(digits_msb_first: list[int], divisor: int) -> tuple[list[int], int]:
    """base-B schoolbook 除法（MSB→LSB）。返回 (商位 MSB-first, 余数∈[0,divisor))。

    商位 qd ∈ [0, B)：因每步 prev_r < divisor ⟹ prev_r·B+d ≤ (divisor-1)·B+(B-1)
    = divisor·B-1 < divisor·B ⟹ qd ≤ B-1。
    """
    q: list[int] = []
    r = 0
    for d in digits_msb_first:
        r = r * BASE + d
        qd = r // divisor
        q.append(qd)
        r -= qd * divisor
    return q, r


def longdiv_limb(a: int, b: int, k: int = DEFAULT_K) -> FixedQuotient:
    """代分数归纳除法（显式 base-2^30 schoolbook；移植锚点）。与 longdiv 逐位一致。"""
    assert_no_float(a, b, k, _where="longdiv_limb")
    if b <= 0:
        raise ValueError(f"longdiv_limb: 除数 b 须为正，got b={b}")
    if k < 0:
        raise ValueError(f"longdiv_limb: 精度 k 须 ≥ 0，got k={k}")

    sign = -1 if a < 0 else 1
    mag = -a if a < 0 else a
    # 被除数 = mag · B^k = mag 的 base-B 位（MSB-first）+ k 个零位
    _, mag_limbs = bigint.to_limbs(mag)  # LSB-first
    msb_first = list(reversed(mag_limbs)) if mag_limbs else [0]
    div_digits = msb_first + [0] * k
    q_digits, mag_r = _schoolbook_divmod(div_digits, b)  # q MSB-first, mag_r∈[0,b)

    q_lsb = list(reversed(q_digits))
    mag_M = bigint.from_limbs((1, q_lsb)) if q_lsb else 0

    # floor 语义 + 非负余数：mag·B^k = mag_M·b + mag_r
    if sign > 0 or mag_r == 0:
        M = sign * mag_M
        r = mag_r
    else:
        # a<0 且 mag_r≠0：调整使 r∈[0,b)、M 为 floor
        M = -mag_M - 1
        r = b - mag_r

    assert a * (BASE ** k) == M * b + r, f"longdiv_limb 不变量破坏: a={a},b={b},k={k}"
    assert 0 <= r < b, f"longdiv_limb 余数越界: r={r}, b={b}"
    return FixedQuotient(M, r, k, b)


def mod(a: int, b: int) -> int:
    """mod 原子（= a % b = longdiv(a,b,k=0).r）。单一真相源委托此处。"""
    return longdiv(a, b, 0).r
