"""S4 迭代等价 corpus 测试（doc/重来_S3S4迭代机制设计_2026-07-16 §四·Mode B cross-verify）。

S4 = 两路同函数异 shape 等价对·Mode B 两路独立 execute_composes_value + rational.eq → agreement。
6 form 覆盖**三个迭代 builder**：Sigma（Σk/k²/k³/奇/偶）+ Prod（n! 累乘）+ Recur（n! 递推 a*i）。
**★ Mode B 绕过 discover**（直 build 两树 + execute·S3 discover:217 排除 ctrl 在此路径不适用）→
Sigma+Prod+Recur 迭代【等价验证】能力经 S4 Mode B 可达。

反 theater：片3 每 form 直 cross_verify_pair AGREE（非"建了=活了"）+ 负控 disagree + 片5 formal_train POST
端到端（factorial Prod/Recur·审1 LOW-1·锁真 pipeline 非 unit-only）+ 期望值 host 独立重算 + 确定性 + 异参名。

诚实边界：Mode B = 统计学内一致非真理（agreement 非 identity·stable≠correct·#479 墙内弱）·仅验证非泛化
（S3 discover-泛化迭代 defer）·factorial 两路皆迭代（两迭代 builder 互验·R6 真守因 Prod/Recur 异 builder）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair
from pure_integer_ai.experiments.collection import load_arith_s4_corpus
from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import WEANING_POST


# ---- fixture helper：建两棵独立 COMPOSES 树（同 space·异 root·镜像 test_mode_b_cross_verify:68-90）----

def _build_pair(dsl_a: str, dsl_b: str):
    """建 backend + space + 两棵独立 COMPOSES 树（root_a=source_a / root_b=source_b）·返 (graph, root_a, root_b)。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    root_a = ci.ensure("__s4_a", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(dsl_a, concept_index=ci, edge_store=es,
                              backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_a)
    root_b = ci.ensure("__s4_b", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(dsl_b, concept_index=ci, edge_store=es,
                              backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_b)
    return g, root_a, root_b


# ============ 片1 corpus 结构 ============

def test_s4_corpus_structure():
    """6 form·每 item 有 arith_source + arith_source_b + arith_specs（≥2 探针）。"""
    items = load_arith_s4_corpus()
    assert len(items) == 6
    for it in items:
        assert it.arith_source is not None and it.arith_source_b is not None
        assert it.arith_source.startswith("lambda "), f"source_a 须 lambda DSL: {it.arith_source!r}"
        assert it.arith_source_b.startswith("lambda "), f"source_b 须 lambda DSL: {it.arith_source_b!r}"
        assert len(it.arith_specs) >= 2, "每 form ≥2 探针（多探针 cross-verify）"
        for spec in it.arith_specs:
            assert len(spec.input_args) == 1, "arity-1（单 param）"


def test_s4_corpus_distinct_params_no_reserved():
    """异参名铁律：6 form 各异 param（n/m/s/o/e/f）·无 reserved a/i·同 item source_a/b 同 param。"""
    items = load_arith_s4_corpus()
    params = []
    for it in items:
        head = it.arith_source.split("lambda", 1)[1].split(":", 1)[0].strip()
        params.append(head)
        assert head not in ("a", "i"), f"param 禁 reserved a/i: {head!r}"
        head_b = it.arith_source_b.split("lambda", 1)[1].split(":", 1)[0].strip()
        assert head == head_b, f"同 item source_a/b 须同 param: {head!r} vs {head_b!r}"
    assert len(set(params)) == 6, f"6 form 须各异参名: {params}"


# ============ 片2 期望值正确性（host arithmetic 独立重算·反同源 theater） ============

def test_s4_expected_correct():
    """每 form 探针 expected = 真值（host 独立重算·非依赖 _s4_* 以防同源 theater）。"""
    items = load_arith_s4_corpus()
    def real_sum(n):    # Σ k
        return sum(range(1, n + 1))
    def real_sq(n):     # Σ k²
        return sum(k * k for k in range(1, n + 1))
    def real_cube(n):   # Σ k³
        return sum(k ** 3 for k in range(1, n + 1))
    def real_odd(n):    # Σ (2k-1)
        return sum(2 * k - 1 for k in range(1, n + 1))
    def real_even(n):   # Σ (2k)
        return sum(2 * k for k in range(1, n + 1))
    def real_fact(n):   # n!
        f = 1
        for k in range(2, n + 1):
            f *= k
        return f
    real = {"n": real_sum, "m": real_sq, "s": real_cube, "o": real_odd, "e": real_even, "f": real_fact}
    for it in items:
        p = it.arith_source.split("lambda", 1)[1].split(":", 1)[0].strip()
        fn = real[p]
        for spec in it.arith_specs:
            n = spec.input_args[0]
            assert spec.expected == (fn(n), 1), \
                f"param={p} n={n}: expected={spec.expected} 真值={(fn(n), 1)}"


# ============ 片3 ★反 theater 核心：每 form cross_verify_pair AGREE ============

def test_s4_cross_verify_each_form_agree():
    """★ 反 theater：每 form source_a vs source_b·全探针两路 execute + rational.eq AGREE。

    Mode B 绕过 discover·直 build 两树 + execute_composes_value + rational.eq·镜像 test_mode_b_cross_verify
    片1。all_agree=True 证两路同函数异 shape（R6 真守）。含 factorial（Prod vs Recur·两迭代 builder）。
    """
    items = load_arith_s4_corpus()
    for it in items:
        probes = tuple(spec.input_args for spec in it.arith_specs)
        g, root_a, root_b = _build_pair(it.arith_source, it.arith_source_b)
        cv = cross_verify_pair(g, root_a, root_b, probes)
        assert cv.n_probes == len(probes), f"{it.arith_source}: 探针计数"
        assert cv.n_valid == len(probes), \
            f"{it.arith_source}: 全探针两路执行成功（n_valid={cv.n_valid}/{len(probes)}·有 vacate=DSL 执行失败）"
        assert cv.n_agree == cv.n_valid, \
            f"{it.arith_source}: 全一致（n_agree={cv.n_agree}/{cv.n_valid}·有 disagree=两路≠）"
        assert cv.all_agree is True, \
            f"{it.arith_source} 两路等价须 AGREE（all_agree=False=机制 gap 非 theater）"


def test_s4_cross_verify_disagree_negative_control():
    """反 theater 负控：Σ k 迭代 vs n*n 平方·异值→all_agree=False（证机制真比对非 theater·镜像 test:120）。"""
    items = load_arith_s4_corpus()
    sigma_sum = next(it for it in items if it.arith_source_b.endswith("Sigma(1, n, i)"))
    g, root_a, root_b = _build_pair("lambda n: n * n", sigma_sum.arith_source_b)
    cv = cross_verify_pair(g, root_a, root_b, ((5,),))
    assert cv.n_valid == 1 and cv.n_agree == 0 and cv.all_agree is False, \
        "n*n(25) vs Σk(15) 异值→disagree（机制真比对）"


# ============ 片4 确定性 + bit-identical ============

def test_s4_corpus_deterministic():
    """load 两次完全相同（确定性 in-memory·纯数据·无随机）。"""
    a = load_arith_s4_corpus()
    b = load_arith_s4_corpus()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.arith_source == y.arith_source
        assert x.arith_source_b == y.arith_source_b
        assert x.arith_specs == y.arith_specs


def test_s4_bit_identical_defaults():
    """bit-identical 守：S4 item 显式带 arith_source_b·不染既有 CollectedItem 默认（默认 None）。"""
    from pure_integer_ai.experiments.collection import CollectedItem
    plain = CollectedItem()
    assert plain.arith_source_b is None, "既有 CollectedItem 默认 arith_source_b=None（bit-identical）"
    for it in load_arith_s4_corpus():
        assert it.arith_source_b is not None
    assert gates.MODE_B_CROSS_VERIFY_MODE is False, "gate 默认 OFF（S4 跑须显式翻）"


# ============ 片5 formal_train POST 端到端（审1 LOW-1·锁真 pipeline 非 unit-only） ============

def test_s4_post_e2e_factorial_prod_vs_recur(monkeypatch):
    """formal_train POST 端到端：factorial（Prod 累乘 vs Recur 递推 a*i）→ cross_verify AGREE→reward=1。

    镜像 test_mode_b_cross_verify:190（test_post_cross_verify_on_agree）·锁 factorial（新 Prod/Recur form）
    经 _run_verify_round POST 分支（formal_train:712-735）真产 reward=1（非 unit-only·审1 LOW-1）。
    """
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    items = load_arith_s4_corpus()
    fact = next(it for it in items if it.arith_source.startswith("lambda f: Prod"))
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    res = r.run_round_full(ctx, fact, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1, "factorial Prod vs Recur POST cross-verify 须 reward=1（两迭代 builder AGREE）"
    assert res.episode.judge_G5_active is True


def test_s4_post_e2e_off_bit_identical(monkeypatch):
    """OFF=0 bit-identical：gate OFF + arith_source_b 在→POST 短路 reward=0（镜像 test_mode_b:175）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', False)
    items = load_arith_s4_corpus()
    fact = next(it for it in items if it.arith_source.startswith("lambda f: Prod"))
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    res = r.run_round_full(ctx, fact, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "gate OFF → POST 短路 reward=0（bit-identical·双 False 短路第一项）"
    assert res.episode.judge_G5_active is False
