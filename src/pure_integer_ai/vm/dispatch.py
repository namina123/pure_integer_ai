"""vm.dispatch — opcode dispatch（CALC_ROLE 解析优先/迭代兜底·纯整数 Rational 算术）。

§九 2a 所指层：opcode → 墙内纯整数实现。算子在 Rational 上精确实现（零浮点）。

CALC_ROLE（解析优先哲学·§analytical-preference）：
  CALC_ROLE_ANALYTICAL —— 闭式精确（ADD/SUB/MUL/DIV/EQ/LT/GT 全闭式·Rational 闭运算）
  CALC_ROLE_ITERATIVE  —— 迭代兜底（首版无·sqrt 等迭代算子随 VM 扩展·isqrt 已在 crosscut 精确）
首版 10 opcode 全 ANALYTICAL。DIV 走**有理倒数乘**（a/b = a·(d/c)·精确·非 fixed_point 近似）。

dispatch 函数本身可被 vm_core 直接调（测试/training 层 execute_composes_value 验机制）·gate DISPATCH_MODE 装饰位
零读取（机制不读 gate·无条件跑·cognition 永不调 VM 单向依赖守·故 gate 永无接线点读·永 inert·OFF/ON 等价 bit-identical·见 gates.py 装饰位范式）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.crosscut.integer.rational import (
    Rational, ZERO, ONE, make, add, sub, mul, eq, sign, is_zero,
)
from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_NOP, OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV,
    OPCODE_EQ, OPCODE_LT, OPCODE_GT, OPCODE_LOAD, OPCODE_STORE,
    OPCODE_PUSH_IMM, OPCODE_JZ, OPCODE_JMP, OPCODE_HALT,
)

# ---- CALC_ROLE（解析优先哲学） ----
CALC_ROLE_ANALYTICAL = 1   # 闭式精确
CALC_ROLE_ITERATIVE = 2    # 迭代兜底（首版无）

# opcode → CALC_ROLE（首版全 ANALYTICAL·迭代算子随 VM 扩展登记）
# A1 控制流 opcode（PUSH_IMM/JZ/JMP/HALT）登记 ANALYTICAL 保 calc_role 完整（无 KeyError）·
# 它们不经 dispatch_binary（is_binary_opcode 不含）·控制流在 vm_core.execute 内联处理（触 pc）。
_CALC_ROLE: dict[int, int] = {
    OPCODE_NOP: CALC_ROLE_ANALYTICAL,
    OPCODE_ADD: CALC_ROLE_ANALYTICAL,
    OPCODE_SUB: CALC_ROLE_ANALYTICAL,
    OPCODE_MUL: CALC_ROLE_ANALYTICAL,
    OPCODE_DIV: CALC_ROLE_ANALYTICAL,
    OPCODE_EQ: CALC_ROLE_ANALYTICAL,
    OPCODE_LT: CALC_ROLE_ANALYTICAL,
    OPCODE_GT: CALC_ROLE_ANALYTICAL,
    OPCODE_LOAD: CALC_ROLE_ANALYTICAL,
    OPCODE_STORE: CALC_ROLE_ANALYTICAL,
    OPCODE_PUSH_IMM: CALC_ROLE_ANALYTICAL,
    OPCODE_JZ: CALC_ROLE_ANALYTICAL,
    OPCODE_JMP: CALC_ROLE_ANALYTICAL,
    OPCODE_HALT: CALC_ROLE_ANALYTICAL,
}


def calc_role(opcode: int) -> int:
    """返回 opcode 的 CALC_ROLE（ANALYTICAL 闭式 / ITERATIVE 迭代）。"""
    assert_int(opcode, _where="calc_role.opcode")
    if opcode not in _CALC_ROLE:
        raise KeyError(f"calc_role: 未知 opcode {opcode!r}")
    return _CALC_ROLE[opcode]


def is_binary_opcode(opcode: int) -> bool:
    """是否二元算子（ADD/SUB/MUL/DIV/EQ/LT/GT·操作数从栈取）。"""
    return opcode in (OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV,
                      OPCODE_EQ, OPCODE_LT, OPCODE_GT)


def reciprocal(b: Rational) -> Rational:
    """有理倒数 1/b = den/num（b≠0·精确·纯整数）。"""
    if is_zero(b):
        raise ZeroDivisionError("reciprocal: b=0")
    return make(b.den, b.num)


def rdiv(a: Rational, b: Rational) -> Rational:
    """有理除 a/b = a·(1/b)（精确·Rational 闭运算·非 fixed_point 近似）。

    Rational 除闭于 Rational（a/b ÷ c/d = (a·d)/(b·c)）·与 rational.div（→FixedQuotient）
    不同：本函数留在精确有理域·VM DIV opcode 用此（零损失）。
    """
    return mul(a, reciprocal(b))


def dispatch_binary(opcode: int, a: Rational, b: Rational) -> Rational:
    """二元算子 dispatch（栈机：a 先入栈在下·b 后入在上·运算 a op b）。

    返回 Rational（EQ/LT/GT 返回 ONE/ZERO 作为布尔）。
    """
    assert_no_float(opcode, _where="dispatch_binary.opcode")
    assert_int(opcode, _where="dispatch_binary.opcode")
    if opcode == OPCODE_ADD:
        return add(a, b)
    if opcode == OPCODE_SUB:
        return sub(a, b)
    if opcode == OPCODE_MUL:
        return mul(a, b)
    if opcode == OPCODE_DIV:
        return rdiv(a, b)
    if opcode == OPCODE_EQ:
        return ONE if eq(a, b) else ZERO
    if opcode == OPCODE_LT:
        return ONE if sign(sub(a, b)) < 0 else ZERO
    if opcode == OPCODE_GT:
        return ONE if sign(sub(a, b)) > 0 else ZERO
    raise ValueError(f"dispatch_binary: 非二元 opcode {opcode!r}")
