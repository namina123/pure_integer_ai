"""Phase B 种子床扩测试（§十四-bis·mereology part-of 预序闭包 + PURE_ALIAS 等价闭包）。

B.1 build_mereology_ancestor_map_external（self-gate·external 双滤·isa_ancestor_map engine 复用）+ whole_of（None 冷启动）。
B.2 build_pure_alias_closure_external（self-gate·positive epistemic==EPI_STRUCTURED 滤·symmetrize·transitive_closure 首个 live caller）。
★ 反 theater regression：cue-derived(EPI_CUE) PURE_ALIAS 边**不进**等价闭包（anti-self-proving·Phase B §十四-bis 承重）。

铁律：纯整数 / bit-identical（两 gate default OFF→返 {}·逐字现状）/ 反 theater（external-only 滤·cue 自证边排除）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import (
    SOURCE_CONCEPTNET, SOURCE_DERIVED, EPI_STRUCTURED, EPI_CUE,
    SUBTYPE_PURE_ALIAS, SUBTYPE_METAPHOR,
)
from pure_integer_ai.storage.edge_types import EDGE_MEREOLOGY, EDGE_REFERS_TO
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.process.abstraction import (
    build_mereology_ancestor_map_external, whole_of, build_pure_alias_closure_external,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 Phase B 闭包 gate（守测试隔离）。"""
    saved = (gates.MEREOLOGY_CLOSURE_MODE, gates.PURE_ALIAS_CLOSURE_MODE)
    gates.MEREOLOGY_CLOSURE_MODE = False
    gates.PURE_ALIAS_CLOSURE_MODE = False
    yield
    (gates.MEREOLOGY_CLOSURE_MODE, gates.PURE_ALIAS_CLOSURE_MODE) = saved


def _add_edge(ctx, frm, to, *, edge_type, source, epistemic, subtype=None):
    """直接 add 边（精确控 source/epistemic/subtype·镜像 edge_store.add）。"""
    ctx.edge_store.add(
        space_id_from=frm[0], local_id_from=frm[1],
        space_id_to=to[0], local_id_to=to[1],
        edge_type=edge_type, strength=1, source=source,
        epistemic_origin=epistemic, subtype=subtype, tier=TIER_PRIMARY,
    )


# ============ B.1 mereology part-of 预序闭包 ============

def test_b1_mereology_closure_gate_off_empty():
    """gate OFF→返 {}（self-gate·bit-identical）·即使有 mereology 边。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    wheel = ctx.concept_index.ensure("wheel", space_id=sid)
    car = ctx.concept_index.ensure("car", space_id=sid)
    _add_edge(ctx, wheel, car, edge_type=EDGE_MEREOLOGY,
              source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    assert build_mereology_ancestor_map_external(ctx.backend, space_id=sid) == {}, \
        "gate OFF→self-gate 返 {}（bit-identical）"


def test_b1_mereology_closure_partof_transitive():
    """gate ON·part-of 预序闭包：轮子→车→车队·轮子的 whole 集含车+车队（part-of 传递·A part-of B ∧ B part-of C）。"""
    ctx = make_train_context(DictBackend())
    gates.MEREOLOGY_CLOSURE_MODE = True
    sid = ctx.space_id
    wheel = ctx.concept_index.ensure("wheel", space_id=sid)
    car = ctx.concept_index.ensure("car", space_id=sid)
    fleet = ctx.concept_index.ensure("fleet", space_id=sid)
    _add_edge(ctx, wheel, car, edge_type=EDGE_MEREOLOGY, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    _add_edge(ctx, car, fleet, edge_type=EDGE_MEREOLOGY, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    am = build_mereology_ancestor_map_external(ctx.backend, space_id=sid)
    assert am.get(wheel) == {car, fleet}, "part-of 传递闭包：轮子的 whole 集 = {车, 车队}"
    assert am.get(car) == {fleet}


def test_b1_whole_of_none_coldstart_and_deepest():
    """whole_of 冷启动返 None（非 ref·mereology-specific·审1 MED-6）·有 whole 返最具体（最深）。"""
    ctx = make_train_context(DictBackend())
    gates.MEREOLOGY_CLOSURE_MODE = True
    sid = ctx.space_id
    wheel = ctx.concept_index.ensure("wheel", space_id=sid)
    car = ctx.concept_index.ensure("car", space_id=sid)
    fleet = ctx.concept_index.ensure("fleet", space_id=sid)
    lone = ctx.concept_index.ensure("lone", space_id=sid)
    _add_edge(ctx, wheel, car, edge_type=EDGE_MEREOLOGY, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    _add_edge(ctx, car, fleet, edge_type=EDGE_MEREOLOGY, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    am = build_mereology_ancestor_map_external(ctx.backend, space_id=sid)
    assert whole_of(am, lone) is None, "无 whole→None（冷启动·part 非自身 whole·异 nearest_isa_ancestor return-ref）"
    assert whole_of(am, wheel) == car, "最近 whole = 车（最具体·非车队 fleet）"


def test_b1_mereology_external_filter_excludes_non_external():
    """外源双滤：source≠CONCEPTNET 的 mereology 边不进闭包（anti-self-proving·forward-sound）。"""
    ctx = make_train_context(DictBackend())
    gates.MEREOLOGY_CLOSURE_MODE = True
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    c = ctx.concept_index.ensure("c", space_id=sid)
    _add_edge(ctx, a, b, edge_type=EDGE_MEREOLOGY, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    _add_edge(ctx, b, c, edge_type=EDGE_MEREOLOGY, source=SOURCE_DERIVED, epistemic=EPI_STRUCTURED)
    am = build_mereology_ancestor_map_external(ctx.backend, space_id=sid)
    assert am.get(a) == {b}, "external 边入闭包（a→b·ConceptNet+EPI_STRUCTURED）"
    assert c not in am.get(b, set()), "non-external 边不入闭包（b→c·SOURCE_DERIVED 排除）"


# ============ B.2 PURE_ALIAS 等价闭包 ============

def test_b2_pure_alias_closure_gate_off_empty():
    """gate OFF→返 {}（self-gate·bit-identical）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    _add_edge(ctx, a, b, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    assert build_pure_alias_closure_external(ctx.backend, space_id=sid) == {}, \
        "gate OFF→self-gate 返 {}（bit-identical）"


def test_b2_pure_alias_equivalence_symmetric_over_mixed_directions():
    """gate ON·等价类对称：单向 a→b + 单向 b→c（observe/lemmatizer style）→ a/b/c 同等价类（symmetrize 解方向混合）。"""
    ctx = make_train_context(DictBackend())
    gates.PURE_ALIAS_CLOSURE_MODE = True
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    c = ctx.concept_index.ensure("c", space_id=sid)
    _add_edge(ctx, a, b, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    _add_edge(ctx, b, c, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    equiv = build_pure_alias_closure_external(ctx.backend, space_id=sid)
    assert equiv.get(a, {a}) == {a, b, c}, "symmetrize 后 a/b/c 同等价类（含自身）"
    assert equiv.get(c, {c}) == {a, b, c}, "等价类对称（c 的类同 a·symmetrize 解方向混合→真等价 非 preorder）"


def test_b2_pure_alias_anti_self_proving_excludes_cue():
    """★ regression guard（承重·审2命门）：cue-derived(EPI_CUE) PURE_ALIAS 边**不进**等价闭包。

    Phase B §十四-bis 纠 observe.py:331 alias_cue_pairs 误标后·cue alias 标 EPI_CUE→
    positive 滤 epistemic==EPI_STRUCTURED 排除之→anti-self-proving（cue 自证 alias 不混入等价类·防 floor 自证闭环）。
    """
    ctx = make_train_context(DictBackend())
    gates.PURE_ALIAS_CLOSURE_MODE = True
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    cue = ctx.concept_index.ensure("cue", space_id=sid)
    # boot external a↔b（EPI_STRUCTURED）— 入等价类
    _add_edge(ctx, a, b, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    _add_edge(ctx, b, a, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    # cue-derived b→cue（EPI_CUE·observe alias_cue_pairs Phase B 纠标后）— 不进等价类
    _add_edge(ctx, b, cue, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_CUE, subtype=SUBTYPE_PURE_ALIAS)
    equiv = build_pure_alias_closure_external(ctx.backend, space_id=sid)
    assert equiv.get(a, {a}) == {a, b}, "a/b 同类（boot external EPI_STRUCTURED）"
    assert cue not in equiv.get(a, {a}), "★ cue alias（EPI_CUE）排除·anti-self-proving 命门"
    assert cue not in equiv, "cue 节点不进任何等价类（其唯一 PURE_ALIAS 边 EPI_CUE 全滤）"


def test_b2_pure_alias_metaphor_excluded():
    """select 预滤 subtype=PURE_ALIAS + purity_filter：METAPHOR(subtype=2) 边不进等价闭包（即使 EPI_STRUCTURED）。"""
    ctx = make_train_context(DictBackend())
    gates.PURE_ALIAS_CLOSURE_MODE = True
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    m = ctx.concept_index.ensure("m", space_id=sid)
    _add_edge(ctx, a, b, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_PURE_ALIAS)
    _add_edge(ctx, b, m, edge_type=EDGE_REFERS_TO, source=SOURCE_DERIVED,
              epistemic=EPI_STRUCTURED, subtype=SUBTYPE_METAPHOR)
    equiv = build_pure_alias_closure_external(ctx.backend, space_id=sid)
    assert equiv.get(a, {a}) == {a, b}, "PURE_ALIAS 边入等价类"
    assert m not in equiv.get(a, {a}), "METAPHOR 边排除（select 预滤 subtype=PURE_ALIAS·purity_filter 冗余守）"
