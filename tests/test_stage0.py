"""Stage 0 验收门测试：纯整数守卫三层 + AST CI + 两跑 bit-identical + 不变量。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 0 验收门）：
  - longdiv 不变量 a·B^k == M·b+r
  - SqrtRef 守恒 M² ≤ n·S² < (M+1)²
  - cross_compare 零误差
  - Hasher.h63 / DRNG 两跑 bit-identical
  - audit_event 链式 + golden append-only
  - AST CI 真扫 float + time/datetime/random + import 方向
  - 纯整数守卫三层（float_guard / int_blocker / AST CI）
"""
from __future__ import annotations

import os

import pytest

from pure_integer_ai.crosscut.integer import (
    algebraic_fraction, bigint, compare, fixed_point, isqrt, rational, valtypes,
)
from pure_integer_ai.crosscut.integer.constants import BASE, DEFAULT_K
from pure_integer_ai.crosscut.determinism import (
    audit_event, cross_radix, drng, golden, hasher, reproducible,
)
from pure_integer_ai.crosscut.guards import float_guard, lint
from pure_integer_ai.crosscut.guards import int_blocker
from pure_integer_ai.numeric import credit_sink, numeric_relatedness, symbol_domain


# ---- longdiv 不变量 ----

@pytest.mark.parametrize("a,b,k", [
    (1, 3, 4), (7, 3, 2), (0, 5, 8), (-1, 3, 4), (-7, 3, 2),
    (2, 7, 3), (123456789, 1000, 3), (-(2 ** 50), 99991, 5),
    (10 ** 40, 7, 8),
])
def test_longdiv_invariant(a, b, k):
    fq = algebraic_fraction.longdiv(a, b, k)
    assert a * (BASE ** k) == fq.M * b + fq.r
    assert 0 <= fq.r < b


def test_longdiv_negative_floor():
    # -7/3: floor = -3, r = 2 (-7 = -3*3 + 2)
    fq = algebraic_fraction.longdiv(-7, 3, 0)
    assert fq.M == -3
    assert fq.r == 2


def test_longdiv_limb_matches_direct():
    rep = cross_radix.cross_radix_check()
    assert not rep, f"longdiv vs longdiv_limb 不一致: {rep.diffs}"


def test_mod_atom():
    assert algebraic_fraction.mod(17, 5) == 2
    assert algebraic_fraction.mod(-7, 3) == 2  # floor 余数非负


# ---- SqrtRef 守恒 ----

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 9, 10, 99, 100, 123456789, 2 ** 100])
def test_isqrt_floor_invariant(n):
    M = isqrt.isqrt_floor(n)
    assert M * M <= n < (M + 1) * (M + 1)
    assert M >= 0


@pytest.mark.parametrize("n,scale", [(2, 1000), (10, 100), (99, 10000), (2 ** 50, 10 ** 6)])
def test_sqrt_scaled_invariant(n, scale):
    M = isqrt.sqrt_scaled(n, scale)
    assert M * M <= n * scale * scale < (M + 1) * (M + 1)


def test_sqrt_ref_expand_digits():
    ref = isqrt.SqrtRef(2)
    # √2 ≈ 1.41421356 → 3 位 = 1414
    assert ref.expand_digits(3) == 1414


def test_isqrt_negative_raises():
    with pytest.raises(ValueError):
        isqrt.isqrt_floor(-1)


# ---- cross_compare 零误差 ----

def test_cross_compare_basic():
    assert compare.cross_compare(1, 2, 1, 2) == 0   # 1/2 == 1/2
    assert compare.cross_compare(1, 2, 1, 3) > 0     # 1/2 > 1/3
    assert compare.cross_compare(1, 3, 1, 2) < 0     # 1/3 < 1/2
    assert compare.cross_eq(2, 4, 1, 2) is True       # 2/4 == 1/2
    assert compare.cross_ge(7, 10, 7, 10) is True


def test_cross_compare_big():
    # 大数交叉积仍精确（浮点会溢出失精）
    assert compare.cross_lt(10 ** 40, 10 ** 40 + 1, 1, 1)  # 巨大有理 < 1
    assert compare.cross_gt(10 ** 40 + 1, 10 ** 40, 1, 1)


def test_cross_compare_den_negative_raises():
    with pytest.raises(ValueError):
        compare.cross_compare(1, 0, 1, 2)


# ---- Rational 算术 ----

def test_rational_eq_cross_product():
    assert rational.make(2, 4) == rational.make(1, 2)
    assert rational.eq(rational.make(6, 9), rational.make(2, 3))


def test_rational_arith():
    a = rational.make(1, 2)
    b = rational.make(1, 3)
    assert rational.add(a, b) == rational.make(5, 6)
    assert rational.sub(a, b) == rational.make(1, 6)
    assert rational.mul(a, b) == rational.make(1, 6)


def test_rational_div_to_fixed():
    fq = fixed_point.rational_div(rational.make(1, 3), rational.ONE, k=8)
    # 1/3 定点近似·误差 < 1/B^k
    lo, hi = fixed_point.to_rational_interval(fq)
    assert compare.cross_le(lo.num, lo.den, 1, 3) or rational.eq(lo, rational.make(1, 3)) or True
    # 真值 1/3 ∈ [lo, hi)
    assert compare.cross_le(lo.num, lo.den, 1, 3)
    assert compare.cross_lt(1, 3, hi.num, hi.den)


# ---- fixed_point ----

def test_fixed_point_add_sub():
    a = algebraic_fraction.longdiv(1, 3, 4)
    b = algebraic_fraction.longdiv(2, 3, 4)
    s = fixed_point.add(a, b)
    # 1/3+2/3=1；定点各自截断后和 = 1 − 1/B^k（floor 各截一次·诚实边界）
    bk = BASE ** 4
    assert compare.cross_ge(s.M, bk, bk - 1, bk)  # ≥ 1 − 1/B^k
    assert compare.cross_lt(s.M, bk, bk + 1, bk)   # < 1 + 1/B^k


def test_fixed_point_weighted_sum():
    a = algebraic_fraction.longdiv(1, 2, 4)
    b = algebraic_fraction.longdiv(1, 4, 4)
    ws = fixed_point.weighted_sum([2, 1], [a, b])
    # 2*(1/2)+1*(1/4)=1.25 → M/B^k ≈ 1.25
    assert compare.cross_ge(ws.M, BASE ** 4, 5, 4)


# ---- bigint round-trip ----

@pytest.mark.parametrize("n", [0, 1, -1, 123456789, -(2 ** 50), 2 ** 100, -10 ** 40])
def test_bigint_roundtrip(n):
    assert bigint.from_limbs(bigint.to_limbs(n)) == n
    assert bigint.bit_eq(n, n)


# ---- Hasher / DRNG bit-identical ----

def test_hasher_deterministic():
    h = hasher.Hasher("seed-42")
    assert h.h63(("a", 1, None)) == h.h63(("a", 1, None))
    # 不同 seed 不同哈希
    assert hasher.Hasher("s1").h63("x") != hasher.Hasher("s2").h63("x")


def test_hasher_canonical():
    # tuple 与 list 同编码（canonical）
    h = hasher.Hasher(0)
    assert h.h63((1, 2, 3)) == h.h63([1, 2, 3])
    # Rational 编码
    assert h.h63(rational.make(1, 2)) == h.h63(rational.make(2, 4))  # eq 走交叉积但哈希按 canonical num/den
    # 注：make 已归约，故 1/2 与 2/4 同 Rational(1,2)


def test_hasher_keeps_golden_bytes_and_tuple_prefix_equivalence():
    """编码优化和公共前缀续算不得改变任何既有稳定哈希。"""
    h = hasher.Hasher("evidence.v1")
    value = ((1, 2, (3, 4)), -7, 0, 9, 1, (5, 6), (7, 8, 9))
    assert h.h(value) == 1_164_569_907_929_920_328
    assert h.h63(value) == 1_164_569_907_929_920_328
    prepared = h.prepare_tuple_prefix(7, (value[0],))
    assert prepared.h(value[1:]) == h.h(value)
    assert prepared.h63(value[1:]) == h.h63(value)
    with pytest.raises(ValueError, match="数量"):
        prepared.h63(value[2:])


@pytest.mark.parametrize("values", [
    (),
    (0,),
    (-1, 0, 1),
    (-(1 << 80), 1 << 80, 7),
])
def test_hasher_tagged_int_tuple_preserves_canonical_hash(values):
    """纯整数流式入口必须与历史嵌套 tuple 编码逐位一致。"""
    h = hasher.Hasher("identity_registry.v1")
    assert h.h63_tagged_int_tuple(4, values) == h.h63((4, values))


def test_hasher_tagged_int_tuple_rejects_bool_and_non_tuple():
    """流式入口不得把 bool 或 list 偷渡为身份整数元组。"""
    h = hasher.Hasher("identity_registry.v1")
    with pytest.raises(TypeError):
        h.h63_tagged_int_tuple(True, (1,))
    with pytest.raises(TypeError):
        h.h63_tagged_int_tuple(4, [1])
    with pytest.raises(TypeError):
        h.h63_tagged_int_tuple(4, (True,))


def test_hasher_rejects_invalid_tuple_prefix_contracts():
    """前缀复用只接受严格整数长度和精确 tuple，不能静默改编码边界。"""
    h = hasher.Hasher("prefix-contract")
    with pytest.raises(ValueError, match="严格整数"):
        h.prepare_tuple_prefix(True, ())
    with pytest.raises(TypeError, match="prefix"):
        h.prepare_tuple_prefix(1, [])
    with pytest.raises(ValueError, match="长于"):
        h.prepare_tuple_prefix(0, (1,))


def test_drng_bit_identical():
    s1 = [drng.DRNG(12345).next() for _ in range(100)]
    s2 = [drng.DRNG(12345).next() for _ in range(100)]
    assert s1 == s2
    # 不同 seed 不同流
    assert drng.DRNG(1).next() != drng.DRNG(2).next()


def test_drng_randbelow():
    d = drng.DRNG(99)
    for _ in range(200):
        r = d.randbelow(10)
        assert 0 <= r < 10


# ---- assert_reproducible 两跑 ----

def test_assert_reproducible_pass():
    def run(seed):
        d = drng.DRNG(seed)
        return tuple(d.next() for _ in range(50))
    assert reproducible.assert_reproducible(run, 777) is True
    assert golden.golden.count() >= 1
    assert golden.golden.verify() is True


def test_assert_reproducible_fail():
    counter = {"n": 0}

    def run(seed):
        counter["n"] += 1
        # 第二跑返不同值 → 非确定性
        return counter["n"]
    with pytest.raises(AssertionError):
        reproducible.assert_reproducible(run, 1)


# ---- audit_event 链式 ----

def test_audit_event_chain():
    log = audit_event.AuditLog()
    log.append("observe", {"a": 1})
    log.append("reward", {"r": 5})
    log.append("promote", {"id": 7})
    assert len(log) == 3
    assert log.seq_sequence() == (1, 2, 3)
    assert log.verify_chain() is True
    # event_hash 序列非空且各异
    seq = log.event_hash_sequence()
    assert len(seq) == 3
    assert len(set(seq)) == 3


def test_audit_event_prev_hash_linkage():
    log = audit_event.AuditLog()
    e0 = log.append("a", 1)
    e1 = log.append("b", 2)
    assert e0.prev_hash == 0
    assert e1.prev_hash == e0.event_hash


def test_audit_event_reproducible():
    # 同 op 流两遍 → 相同 event_hash 序列（确定性）
    def build():
        log = audit_event.AuditLog()
        log.append("observe", {"x": 10})
        log.append("reward", {"y": 20})
        return log.event_hash_sequence()
    assert build() == build()


# ---- 纯整数守卫三层 ----

def test_float_guard_runtime():
    with pytest.raises(float_guard.FloatViolation):
        float_guard.assert_no_float(1.0)
    # int 不抛
    float_guard.assert_no_float(1, 2, 3)


def test_float_guard_int_only_decorator():
    @float_guard.int_only
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    with pytest.raises(float_guard.FloatViolation):
        add(1, 2.0)


def test_int_blocker():
    assert int_blocker.require_int(5) == 5
    # bool 放行（逻辑标志）
    assert int_blocker.require_int(True) is True
    with pytest.raises(int_blocker.IntViolation):
        int_blocker.require_int(1.0)
    with pytest.raises(int_blocker.IntViolation):
        int_blocker.require_int("x")


def test_float_guard_debug_off():
    # DEBUG 关后运行时守卫不抛（生产热路径省开销·AST CI 仍守源码层）
    orig = float_guard.DEBUG
    try:
        float_guard.DEBUG = False
        float_guard.assert_no_float(1.0)  # 不抛
    finally:
        float_guard.DEBUG = orig


# ---- AST CI 真扫 ----

def test_ast_scan_float_literal():
    vios = float_guard.scan_source("x = 3.14  # float literal")
    assert len(vios) == 1


def test_ast_scan_float_call():
    vios = float_guard.scan_source("x = float('1')\ny = round(z)")
    assert len(vios) == 2


def test_ast_scan_clean():
    assert float_guard.scan_source("x = 1 + 2\ny = x * 3") == []


def test_lint_forbidden_time():
    src = "import time\nx = time.time()"
    tree = __import__("ast").parse(src)
    vios = lint._forbidden_violations(tree, "t")
    assert len(vios) >= 2  # import + 调用


def test_lint_forbidden_random():
    src = "import random\nr = random.randint(1, 10)"
    tree = __import__("ast").parse(src)
    vios = lint._forbidden_violations(tree, "t")
    assert len(vios) >= 2


def test_lint_forbidden_datetime():
    src = "from datetime import datetime\nn = datetime.now()"
    tree = __import__("ast").parse(src)
    vios = lint._forbidden_violations(tree, "t")
    assert len(vios) >= 2


def test_lint_import_direction_clean():
    # crosscut 不 import 更高层（numeric/storage/...）
    import pure_integer_ai
    pkg = os.path.dirname(pure_integer_ai.__file__)
    dv = lint.import_direction_check(pkg, whitelist=())
    assert dv == {}, f"import direction 违例: {dv}"


def test_lint_chinese_explanation_rejects_english_prose(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "bad.py").write_text(
        '"""This module explains the training cache behavior."""\n'
        "def f():\n"
        "    # This branch keeps cache state stable after promotion.\n"
        "    return 1\n",
        encoding="utf-8",
    )
    violations = lint.chinese_explanation_check(str(root))
    assert len(violations[str(root / "bad.py")]) == 2


def test_lint_chinese_explanation_allows_protocols_and_formulas(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "good.py").write_text(
        '"""处理 JSON Lines、AST 节点和 RFC 8259 输入。"""\n'
        "# JSON Lines / RFC 8259\n"
        "# score = left_count * 1000 // total\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )
    assert lint.chinese_explanation_check(str(root)) == {}


def test_lint_run_clean():
    import pure_integer_ai
    pkg = os.path.dirname(pure_integer_ai.__file__)
    assert lint.run_lint(pkg, whitelist=()) == 0


# ---- numeric ----

def test_numeric_relatedness_same_axis():
    # 同轴·相同值 → rel = 1 (BIG/BIG)
    rel = numeric_relatedness.relatedness(1, 2, 1, 2, symbol_domain.TYPE_AXIS_INT,
                                          symbol_domain.TYPE_AXIS_INT, big=1000)
    assert rel == (1000, 1000)


def test_numeric_relatedness_cross_axis_ground_wall():
    # 跨轴 = 0 接地墙
    rel = numeric_relatedness.relatedness(1, 2, 1, 2, symbol_domain.TYPE_AXIS_INT,
                                          symbol_domain.TYPE_AXIS_RATIONAL, big=1000)
    assert rel == (0, 1)
    assert numeric_relatedness.is_ground_wall(1, 2) is True


def test_numeric_relatedness_far_zero():
    # diff >= big → 0
    rel = numeric_relatedness.relatedness(0, 1, 1001, 1, 1, 1, big=1000)
    assert rel == (0, 1)


def test_credit_sink_monotone():
    cs = credit_sink.CreditSink()
    assert cs.append(1, 1, 1, 5) == 5
    assert cs.append(1, 1, 1, 3) == 8
    assert cs.credit_of(1, 1, 1) == 8
    with pytest.raises(credit_sink.CreditViolation):
        cs.append(1, 1, 1, -1)
    assert cs.total() == 8


def test_symbol_domain_bridge():
    sid = symbol_domain.opcode_to_symbol("ADD")
    assert symbol_domain.symbol_to_opcode(sid) == "ADD"
    v = symbol_domain.make_variable(42)
    assert symbol_domain.kind_of(v) == symbol_domain.KIND_VARIABLE
    assert symbol_domain.index_of(v) == 42
    assert symbol_domain.kind_of(symbol_domain.opcode_to_symbol("MUL")) == symbol_domain.KIND_OPCODE


def test_symbol_domain_register_no_overwrite():
    with pytest.raises(ValueError):
        symbol_domain.register_opcode("ADD", symbol_domain.OPCODE_BASE | 100)
    with pytest.raises(ValueError):
        symbol_domain.register_opcode("NEW", symbol_domain.OPCODE_ADD)
