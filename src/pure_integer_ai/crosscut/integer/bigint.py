"""crosscut.integer.bigint — base 2^30 limb 大数器（移植锚点）。

形式价值高于性能价值：Python 原生 int 任意精度即真值源，本模块把它显式表示为
base 2^30 的 limb 序列，使移植到 C/Rust 时有 1:1 limb 对照（同算法保 bit 一致）。

表示：sign-magnitude。sign ∈ {-1,0,1}；limbs 为 |n| 的小端 base-2^30 序列，
每 limb ∈ [0, BASE)，无前导零 limb（0 → (0, [])）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.integer.constants import BASE, BASE_BITS
from pure_integer_ai.crosscut.guards.float_guard import assert_no_float

_MASK = BASE - 1  # 2^30 - 1


# ---- int <-> limbs ----

def to_limbs(n: int) -> tuple[int, list[int]]:
    """int → (sign, limbs)。sign ∈ {-1,0,1}；limbs 小端，每 limb ∈ [0, BASE)，无前导零。"""
    assert_no_float(n, _where="to_limbs")
    if n == 0:
        return (0, [])
    sign = 1 if n > 0 else -1
    m = n if sign > 0 else -n
    limbs: list[int] = []
    while m:
        limbs.append(m & _MASK)
        m >>= BASE_BITS
    return (sign, limbs)


def from_limbs(rep: tuple[int, list[int]]) -> int:
    """(sign, limbs) → int。接受 to_limbs 的返回，故 from_limbs(to_limbs(n)) == n。"""
    sign, limbs = rep
    n = 0
    for i, limb in enumerate(limbs):
        assert 0 <= limb < BASE, f"limb 越界 [0,{BASE}): {limb}"
        n += limb << (BASE_BITS * i)
    return sign * n


# ---- limb 级（magnitude）运算 ----

def _strip(limbs: list[int]) -> list[int]:
    """去前导零 limb（原地）。"""
    while limbs and limbs[-1] == 0:
        limbs.pop()
    return limbs


def limb_cmp(a: list[int], b: list[int]) -> int:
    """magnitude 比较：-1 / 0 / 1。"""
    a, b = _strip(list(a)), _strip(list(b))
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            return -1 if x < y else 1
    return 0


def limb_add(a: list[int], b: list[int]) -> list[int]:
    """magnitude 相加（base 2^30，带进位）。"""
    out: list[int] = []
    carry = 0
    for i in range(max(len(a), len(b))):
        s = carry
        if i < len(a):
            s += a[i]
        if i < len(b):
            s += b[i]
        out.append(s & _MASK)
        carry = s >> BASE_BITS
    if carry:
        out.append(carry)
    return _strip(out)


def limb_sub(a: list[int], b: list[int]) -> list[int]:
    """magnitude 相减（要求 a >= b，否则抛）。"""
    assert limb_cmp(a, b) >= 0, "limb_sub 要求 a >= b"
    out: list[int] = []
    borrow = 0
    for i in range(len(a)):
        cur = a[i] - borrow - (b[i] if i < len(b) else 0)
        if cur < 0:
            cur += BASE
            borrow = 1
        else:
            borrow = 0
        out.append(cur)
    return _strip(out)


def limb_mul(a: list[int], b: list[int]) -> list[int]:
    """magnitude 相乘（schoolbook，base 2^30）。"""
    a, b = _strip(list(a)), _strip(list(b))
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        carry = 0
        for j, bj in enumerate(b):
            s = out[i + j] + ai * bj + carry
            out[i + j] = s & _MASK
            carry = s >> BASE_BITS
        if carry:
            out[i + len(b)] += carry
    return _strip(out)


# ---- 便利（int 级，等价于 limb 操作） ----

def shift_right(n: int) -> int:
    """右移一个 limb（÷2^30 的退化右移）。等价于丢掉最低 limb。"""
    assert_no_float(n, _where="shift_right")
    return n >> BASE_BITS


def bit_eq(a: int, b: int) -> bool:
    """bit 级相等：按 canonical (sign, limbs) 表示比较（移植对拍锚点）。"""
    return to_limbs(a) == to_limbs(b)
