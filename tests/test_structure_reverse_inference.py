"""对应泛化 v2 结构反推测试（cue↔rel·学全·doc/重来_对应泛化_结构反推_学全_2026-07-17）。

片1 机制（两审 APPROVE·条件全收口）：
  - structure_match_count 表（APPEND_ONLY·distinct forming-sample tally·relation-specific·修审2 BLOCKER 2）
  - _collect_cue_slot_candidates + tally_cue_slot_matches（cue-blind 对齐 REALIZES-R-skeleton cue slot·三路分离·审1C3/审2条件1
    + SHADOW 创建·审2条件2）
  - _structure_match_ok（distinct≥K + specificity·审1 CONDITION 1 + 守 CUE_CLUSTER_MODE·审1 C3）+ D:11 删∨两 gate 共存（审2条件3）
    + rel_kind 桥（ATTR_RELATION_PRIMITIVE·审2条件4a）

反 theater（心脏·§四非循环三层·两审代码核证 PASS）：R 来自 REALIZES oracle（source==CONCEPTNET·非 cue）·
W 观察·提升反馈在 source filter 断。学全（W 可新词·非 oracle/frozenset）。

铁律：纯整数 / bit-identical（gate OFF·零 tally·零 D:11 翻）/ 反 theater（specificity 过滤和/的误晋·distinct 防刷数）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_REALIZES, EDGE_COMPOSES, EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_CONCEPTNET, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR, ATTR_OPERAND, ATTR_CUE_SIG
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives, REL_CAUSES, REL_SUBSET,
)
from pure_integer_ai.cognition.understanding.emergent_relation_signal import record_emergent_relation_signal_shadow
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator, _collect_cue_slot_candidates, tally_cue_slot_matches,
    recognize_operators,
)
from pure_integer_ai.training.promote import promote_edge, _structure_match_ok, PROMOTE_STRUCTURE_MATCH_MIN
from pure_integer_ai.storage.structure_match_count import (
    register_structure_match_count, record_structure_match,
    read_structure_match_count, read_structure_match_per_rel,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测后复位动过的 gate（ORACLE_PROMOTE_MODE / CUE_CLUSTER_MODE / REALIZES_MODE）。"""
    saved = (gates.ORACLE_PROMOTE_MODE, gates.CUE_CLUSTER_MODE, gates.REALIZES_MODE)
    gates.ORACLE_PROMOTE_MODE = False
    gates.CUE_CLUSTER_MODE = False
    gates.REALIZES_MODE = False
    yield
    (gates.ORACLE_PROMOTE_MODE, gates.CUE_CLUSTER_MODE, gates.REALIZES_MODE) = saved


def _ctx():
    """建 train context + 注册 structure_match_count 表（tally/promote 读写得）。"""
    ctx = make_train_context(DictBackend())
    register_structure_match_count(ctx.backend)
    return ctx


def _ensure(ctx, surface):
    return ctx.concept_index.ensure(surface, space_id=ctx.space_id)


def _add_composes(ctx, frm, to, order_index):
    ctx.edge_store.add(
        space_id_from=frm[0], local_id_from=frm[1],
        space_id_to=to[0], local_id_to=to[1],
        edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED, order_index=order_index)


def _build_alignable_skel(ctx, root, slot_refs, cue_at=None, cue_ref=None):
    """对齐就绪 skeleton：root ATTR_OPERATOR(NOP) → slot 叶（ATTR_OPERAND make_variable(i)·PARAM slot）·
    cue_at 位叶加挂 ATTR_CUE_SIG=cue_ref（闭类 cue token·6a-3）。镜像 _build_skel_tree 但补 ATTR_OPERAND（_align_walk 须）。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(slot_refs):
        record_composes_attr(ctx.backend, ref=tok, kind=ATTR_OPERAND,
                             int_a=make_variable(ti), int_b=0)
        _add_composes(ctx, root, tok, ti)
        if ti == cue_at and cue_ref is not None:
            record_composes_attr(ctx.backend, ref=tok, kind=ATTR_CUE_SIG,
                                 int_a=cue_ref[0], int_b=cue_ref[1])


def _build_input_tree(ctx, root, leaf_refs):
    """input 程序：root ATTR_OPERATOR(NOP) → plain concept 叶（无 attr·CONCEPT_LEAF）。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(leaf_refs):
        _add_composes(ctx, root, tok, ti)


# ============ A. structure_match_count 表（片1-1·修审2 BLOCKER 2·落盘 + distinct 去重） ============

def test_record_structure_match_dedup():
    """A1：同 (W,R,input_root) 重记 → 首次 new=True·重记 new=False（append-only 幂等·distinct sample 不重计·防刷数）。"""
    ctx = _ctx()
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    s1 = _ensure(ctx, "__s1")
    n1 = record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                                rel_kind=REL_CAUSES, sample_root=s1)
    n2 = record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                                rel_kind=REL_CAUSES, sample_root=s1)
    assert (n1, n2) == (True, False), "首次 new=True·重记 new=False（幂等去重）"


def test_read_structure_match_count_distinct():
    """A2：3 distinct input_root → count 3·重放同 3 → 仍 3（幂等·抗刷数·非次数）。"""
    ctx = _ctx()
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(3):
        s = _ensure(ctx, f"__s{i}")
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=s)
    assert read_structure_match_count(ctx.backend, space_id=sid,
                                      word_ref=w, rel_kind=REL_CAUSES) == 3
    # 重放同 3 sample → 仍 3（append-only 不堆叠）
    for i in range(3):
        s = _ensure(ctx, f"__s{i}")
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=s)
    assert read_structure_match_count(ctx.backend, space_id=sid,
                                      word_ref=w, rel_kind=REL_CAUSES) == 3


def test_read_structure_match_per_rel():
    """A3：W 落 CAUSES×2 + SUBSET×1 → per_rel={CAUSES:2, SUBSET:1}（specificity 用）。"""
    ctx = _ctx()
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(2):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__c{i}"))
    record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                           rel_kind=REL_SUBSET, sample_root=_ensure(ctx, "__su0"))
    per = read_structure_match_per_rel(ctx.backend, space_id=sid, word_ref=w)
    assert per == {REL_CAUSES: 2, REL_SUBSET: 1}


def test_read_structure_match_table_unregistered_graceful():
    """A4：表未注册 → read 返 0/{}（向后兼容·镜像 experience_count 范式）。"""
    ctx = make_train_context(DictBackend())   # 不 register_structure_match_count
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    assert read_structure_match_count(ctx.backend, space_id=sid,
                                      word_ref=w, rel_kind=REL_CAUSES) == 0
    assert read_structure_match_per_rel(ctx.backend, space_id=sid, word_ref=w) == {}


    record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                           rel_kind=REL_CAUSES, sample_root=w)   # 不抛（KeyError skip）


# ============ B. _structure_match_ok（片1-3·distinct≥K + specificity + 守 CUE_CLUSTER） ============

def test_structure_match_ok_below_k():
    """B1：count < K(=3) → False（即使 specific）。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(2):   # 2 < 3
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    assert not _structure_match_ok(ctx.backend, w[0], w[1], REL_CAUSES)


def test_structure_match_ok_at_k_specific():
    """B2：count=K + specific（仅 CAUSES·other=0）→ True。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    assert _structure_match_ok(ctx.backend, w[0], w[1], REL_CAUSES)


def test_structure_match_ok_cue_cluster_guard():
    """B3（审1 C3）：CUE_CLUSTER_MODE OFF → 恒 False（ATTR_CUE_SIG 不写→cue slot 无→soft-fail 非静默）。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = False   # 显式 OFF
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN + 2):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    assert not _structure_match_ok(ctx.backend, w[0], w[1], REL_CAUSES)


def test_structure_match_ok_specificity_filter():
    """B4（审1 CONDITION 1）：count=3 CAUSES + count=3 SUBSET → 不特异（count ≯ other）→ False（过滤通用连接词误晋）。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    w = _ensure(ctx, "和")   # 通用连接词·落两 R 各 3
    for i in range(3):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__c{i}"))
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_SUBSET, sample_root=_ensure(ctx, f"__u{i}"))
    assert not _structure_match_ok(ctx.backend, w[0], w[1], REL_CAUSES), "和 不特异 CAUSES → 不晋（specificity 滤）"


def test_structure_match_ok_specificity_pass_dominant():
    """B5：count=4 CAUSES + count=1 SUBSET → 特异（4>1）→ True（引发 主 CAUSES·偶匹 SUBSET 仍晋）。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    w = _ensure(ctx, "引发")
    for i in range(4):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__c{i}"))
    record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                           rel_kind=REL_SUBSET, sample_root=_ensure(ctx, "__u0"))
    assert _structure_match_ok(ctx.backend, w[0], w[1], REL_CAUSES)


# ============ C. promote D:11 两 gate 共存 + rel_kind 桥（片1-3·审2条件3+4a）============

def _make_d11_shadow(ctx, w, rel_ref):
    """建 D:11 SHADOW 边 word→rel_ref（record_emergent_relation_signal_shadow·generator 关后 tally 唯一创建者）。"""
    record_emergent_relation_signal_shadow(ctx.edge_store, w, rel_ref, space_id=ctx.space_id)


def test_promote_d11_oracle_mode_promotes():
    """C1：ORACLE_PROMOTE_MODE ON + structure_match≥K + specific → promote PRIMARY（rel_kind 桥读 ATTR_RELATION_PRIMITIVE）。"""
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = True
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    w = _ensure(ctx, "引发")
    _make_d11_shadow(ctx, w, rel_causes)
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    ref = (w[0], w[1], rel_causes[0], rel_causes[1], EDGE_RELATION_SIGNAL)
    assert promote_edge(ctx.edge_store, ctx.node_store, ref, backend=ctx.backend)
    row = ctx.edge_store.get(space_id_from=w[0], local_id_from=w[1],
                             space_id_to=rel_causes[0], local_id_to=rel_causes[1],
                             edge_type=EDGE_RELATION_SIGNAL)
    assert row["tier"] == TIER_PRIMARY, "oracle 模式结构匹配达阈 → flip PRIMARY"


def test_promote_d11_oracle_mode_below_k_no_promote():
    """C2：ORACLE_PROMOTE_MODE ON + count<K → 不 promote（结构匹配未达阈）。"""
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = True
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    w = _ensure(ctx, "引发")
    _make_d11_shadow(ctx, w, rel_causes)
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN - 1):   # 2 < 3
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    ref = (w[0], w[1], rel_causes[0], rel_causes[1], EDGE_RELATION_SIGNAL)
    assert not promote_edge(ctx.edge_store, ctx.node_store, ref, backend=ctx.backend)


def test_promote_d11_oracle_mode_off_bit_identical_no_call():
    """C3（审2条件3）：ORACLE_PROMOTE_MODE OFF → _structure_match_ok 不被调·走 experience∨teacher 旧路径（bit-identical）。
    无 experience feed + 无 teacher → 不 promote（旧路径行为不变·knife4 回归守）。"""
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = False   # 既有路径
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    w = _ensure(ctx, "引发")
    _make_d11_shadow(ctx, w, rel_causes)
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN + 2):
        record_structure_match(ctx.backend, space_id=sid, word_ref=w,
                               rel_kind=REL_CAUSES, sample_root=_ensure(ctx, f"__s{i}"))
    ref = (w[0], w[1], rel_causes[0], rel_causes[1], EDGE_RELATION_SIGNAL)
    # OFF → experience∨teacher 路径·无 experience_count feed（表未注册）+ 无 teacher → 不 promote
    assert not promote_edge(ctx.edge_store, ctx.node_store, ref, backend=ctx.backend)


# ============ D. tally_cue_slot_matches（片1-2·三路分离 + SHADOW 创建·审1C3/审2条件1+2）============

def _build_causes_realizes_skel(ctx):
    """建 CAUSES-REALIZES cue-skeleton（cue 在 slot1=使）+ REALIZES→__REL_CAUSES__ 边 + DiscoveredOperator。"""
    sid = ctx.space_id
    skel_root = _ensure(ctx, "__skel_causes")
    s0 = _ensure(ctx, "__cs0"); cue = _ensure(ctx, "使"); s2 = _ensure(ctx, "__cs2")
    _build_alignable_skel(ctx, skel_root, [s0, cue, s2], cue_at=1, cue_ref=cue)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    build_realizes_edge(ctx.edge_store, skel_root, rel_causes, space_id=sid)
    op = DiscoveredOperator(name="__op_causes", skeleton_ref=skel_root, arity=3, sample_count=4)
    return skel_root, cue, rel_prims, rel_causes, op


def test_collect_cue_slot_candidates_cue_blind_capture():
    """D1（审2条件1 三路分离核心）：新词引发 落 使-skeleton cue slot → cue-blind 捕引发（非精确匹配·突破闭类限制）。"""
    ctx = _ctx()
    skel_root, cue, _, _, _ = _build_causes_realizes_skel(ctx)
    # input：[tok1, 引发(new word), tok2]·引发 在 cue slot 位（slot1）
    in_root = _ensure(ctx, "__in1")
    t1 = _ensure(ctx, "__t1"); w = _ensure(ctx, "引发"); t3 = _ensure(ctx, "__t3")
    _build_input_tree(ctx, in_root, [t1, w, t3])
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map
    amap = build_isa_ancestor_map(ctx.backend, space_id=ctx.space_id)
    cands = _collect_cue_slot_candidates(skel_root, in_root, ctx.backend, arity=3, ancestor_map=amap)
    assert cands == [w], "★ 新词引发 落 cue slot → cue-blind 捕引发（精确匹配轨会拒·tally 轨捕）"


def test_tally_creates_shadow_on_cue_slot_match():
    """D2（审2条件2）：REALIZES-R-skeleton + input W 落 cue slot → tally 建 D:11 SHADOW W→rel（generator 关后唯一创建者）。"""
    ctx = _ctx()
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    in_root = _ensure(ctx, "__in2")
    t1 = _ensure(ctx, "__t1"); w = _ensure(ctx, "引发"); t3 = _ensure(ctx, "__t3")
    _build_input_tree(ctx, in_root, [t1, w, t3])
    n = tally_cue_slot_matches([in_root], discovered_operators=[op],
                               graph=ctx.concept_graph, edge_store=ctx.edge_store,
                               backend=ctx.backend, space_id=ctx.space_id, rel_primitives=rel_prims)
    assert n == 1, "首次 (W,R,input) → 建 1 SHADOW"
    # SHADOW 边 W→rel_causes 存在
    rows = ctx.edge_store.query_from(w[0], w[1], edge_type=EDGE_RELATION_SIGNAL)
    assert any((r["space_id_to"], r["local_id_to"]) == rel_causes for r in rows), "SHADOW W→__REL_CAUSES__ 建边"
    # structure_match_count 记 1 distinct sample
    assert read_structure_match_count(ctx.backend, space_id=ctx.space_id,
                                      word_ref=w, rel_kind=REL_CAUSES) == 1


def test_tally_no_realizes_no_shadow():
    """D3：skeleton 无 REALIZES 边 → 不参与 tally（非'已学结构'basis）→ 零 SHADOW。"""
    ctx = _ctx()
    sid = ctx.space_id
    skel_root = _ensure(ctx, "__skel_nore")
    s0 = _ensure(ctx, "__ns0"); cue = _ensure(ctx, "使"); s2 = _ensure(ctx, "__ns2")
    _build_alignable_skel(ctx, skel_root, [s0, cue, s2], cue_at=1, cue_ref=cue)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="__op_nore", skeleton_ref=skel_root, arity=3, sample_count=4)
    in_root = _ensure(ctx, "__in3")
    _build_input_tree(ctx, in_root, [_ensure(ctx, "__nt1"), _ensure(ctx, "引发"), _ensure(ctx, "__nt3")])
    n = tally_cue_slot_matches([in_root], discovered_operators=[op],
                               graph=ctx.concept_graph, edge_store=ctx.edge_store,
                               backend=ctx.backend, space_id=sid, rel_primitives=rel_prims)
    assert n == 0, "无 REALIZES → 不 tally"


def test_tally_distinct_samples_accumulate_idempotent():
    """D4：3 distinct input_root（各引发 在 cue slot）→ tally count 3·重放同 3 → 仍 3（append-only 幂等）+ SHADOW 仅 1 边。"""
    ctx = _ctx()
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    in_roots = []
    for i in range(3):
        ir = _ensure(ctx, f"__inD{i}")
        _build_input_tree(ctx, ir, [_ensure(ctx, f"__d1_{i}"), _ensure(ctx, "引发"), _ensure(ctx, f"__d3_{i}")])
        in_roots.append(ir)
    n1 = tally_cue_slot_matches(in_roots, discovered_operators=[op],
                                graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                backend=ctx.backend, space_id=ctx.space_id, rel_primitives=rel_prims)
    assert read_structure_match_count(ctx.backend, space_id=ctx.space_id,
                                      word_ref=_ensure(ctx, "引发"), rel_kind=REL_CAUSES) == 3
    # 重放同 3 input → count 仍 3（append-only）·SHADOW 创建返 0（已建·record_emergent_relation_signal_shadow 幂等）
    n2 = tally_cue_slot_matches(in_roots, discovered_operators=[op],
                                graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                backend=ctx.backend, space_id=ctx.space_id, rel_primitives=rel_prims)
    assert read_structure_match_count(ctx.backend, space_id=ctx.space_id,
                                      word_ref=_ensure(ctx, "引发"), rel_kind=REL_CAUSES) == 3
    assert n2 == 0, "重放已存 input → SHADOW 幂等不重建"


def test_three_path_separation_tally_vs_recognize():
    """D5（审1C3/审2条件1·三路分离）：新词引发 落 使-skeleton cue slot → tally 捕（D1）·
    但 recognize 精确匹配轨拒（引发≠使·无 Recognition·held-out verify sound·不污染下游）。"""
    ctx = _ctx()
    gates.CUE_CLUSTER_MODE = True   # recognize 精确匹配轨 cue-aware（6a-3 闭命门2）
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    in_root = _ensure(ctx, "__in_sep")
    t1 = _ensure(ctx, "__sep1"); w = _ensure(ctx, "引发"); t3 = _ensure(ctx, "__sep3")
    _build_input_tree(ctx, in_root, [t1, w, t3])
    # recognize 精确匹配轨：引发 ≠ 使（ATTR_CUE_SIG）→ 不命中 → 0 Recognition
    recs = recognize_operators([in_root], discovered_operators=[op],
                               backend=ctx.backend, space_id=ctx.space_id)
    assert recs == [], "★ recognize 精确匹配拒新词引发（闭命门2 sound）·不产 Recognition"
    # tally 轨：cue-blind 捕引发（独立轨·不污染 recognize）
    cands = _collect_cue_slot_candidates(skel_root, in_root, ctx.backend, arity=3,
                                         ancestor_map={})
    assert cands == [w], "tally 轨独立捕引发（三路分离）"


# ============ E. 非循环不变量（审1 C1/C5 + §四第三层·反馈断·tally 不染 CONCEPTNET oracle）============

def test_tally_shadow_source_not_conceptnet():
    """E1（审1 C1/C5·非循环心脏）：tally 建 SHADOW 边 source=SOURCE_BARE_TEXT（**非** SOURCE_CONCEPTNET）→
    永不过 REALIZES oracle 单 source 滤（_has_external_causes source==CONCEPTNET）→ 反馈环断 → 非自证。
    tally 只读 oracle（经 REALIZES exemplar）·永不写 CONCEPTNET 边（ConceptNet 不可变锚）。"""
    ctx = _ctx()
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    in_root = _ensure(ctx, "__in_e1")
    _build_input_tree(ctx, in_root, [_ensure(ctx, "__e1a"), _ensure(ctx, "引发"), _ensure(ctx, "__e1c")])
    # tally 前 CONCEPTNET-source 边数（boot REALIZES exemplar 的 oracle 边·概念上由 boot 直注）
    before = sum(1 for r in ctx.backend.select("edge") if r.get("source") == SOURCE_CONCEPTNET)
    tally_cue_slot_matches([in_root], discovered_operators=[op],
                           graph=ctx.concept_graph, edge_store=ctx.edge_store,
                           backend=ctx.backend, space_id=ctx.space_id, rel_primitives=rel_prims)
    # tally 后 CONCEPTNET-source 边数不变（tally 不染 oracle）
    after = sum(1 for r in ctx.backend.select("edge") if r.get("source") == SOURCE_CONCEPTNET)
    assert after == before, "★ tally 不写 CONCEPTNET-source 边（oracle 不可变·反馈断·非自证）"
    # tally 建的 SHADOW 边 source=BARE_TEXT
    w = _ensure(ctx, "引发")
    rows = ctx.edge_store.query_from(w[0], w[1], edge_type=EDGE_RELATION_SIGNAL)
    assert rows and all(r["source"] == SOURCE_BARE_TEXT for r in rows), "SHADOW source=BARE_TEXT（非 CONCEPTNET）"


# ============ F. held-out 机制层预验（片2·审1 CONDITION 1·审2 F-1·fixture true/false-positive·非 empirical）============

def test_mechanism_heldout_true_positive_end_to_end():
    """F1（机制层 held-out true-positive·学全预验·审2 F-1 诚实框法）：3 distinct 引发 inputs（新词·不在
    oracle/frozenset）落 CAUSES-REALIZES cue slot → tally count=3 + specific（仅 CAUSES）→ promote 引发→CAUSES
    PRIMARY（端到端·新词泛化）。**机制层预验·非 empirical**：fixture 用 build_realizes_edge 直建 skeleton→rel
    exemplar（绕 oracle filter）+ 占位 token·非 boot 真 ConceptNet-grounded·非真语料泛化率测量。empirical 真
    false-positive 率（真 ConceptNet 留出词·n=20+ 全量跑）defer 断奶后 W4 探针·超阈 reject。"""
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = True
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    # 3 distinct held-out inputs·引发（新词）在 cue slot
    in_roots = []
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        ir = _ensure(ctx, f"__in_tp{i}")
        _build_input_tree(ctx, ir, [_ensure(ctx, f"__tp1_{i}"), _ensure(ctx, "引发"), _ensure(ctx, f"__tp3_{i}")])
        in_roots.append(ir)
    tally_cue_slot_matches(in_roots, discovered_operators=[op],
                           graph=ctx.concept_graph, edge_store=ctx.edge_store,
                           backend=ctx.backend, space_id=sid, rel_primitives=rel_prims)
    w = _ensure(ctx, "引发")
    assert read_structure_match_count(ctx.backend, space_id=sid, word_ref=w,
                                      rel_kind=REL_CAUSES) == PROMOTE_STRUCTURE_MATCH_MIN
    ref = (w[0], w[1], rel_causes[0], rel_causes[1], EDGE_RELATION_SIGNAL)
    assert promote_edge(ctx.edge_store, ctx.node_store, ref, backend=ctx.backend), "★ 引发 端到端 promote"
    row = ctx.edge_store.get(space_id_from=w[0], local_id_from=w[1],
                             space_id_to=rel_causes[0], local_id_to=rel_causes[1],
                             edge_type=EDGE_RELATION_SIGNAL)
    assert row["tier"] == TIER_PRIMARY, "新词引发 → CAUSES PRIMARY（真泛化·学全）"


def _build_subset_realizes_skel(ctx):
    """建 SUBSET-REALIZES cue-skeleton（cue 在 slot1=是）+ REALIZES→__REL_SUBSET__（同形·与 CAUSES-skeleton 共 cue-blind 候选池）。"""
    sid = ctx.space_id
    skel_root = _ensure(ctx, "__skel_subset")
    s0 = _ensure(ctx, "__ss0"); cue = _ensure(ctx, "是"); s2 = _ensure(ctx, "__ss2")
    _build_alignable_skel(ctx, skel_root, [s0, cue, s2], cue_at=1, cue_ref=cue)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_subset = rel_prims[REL_SUBSET]
    build_realizes_edge(ctx.edge_store, skel_root, rel_subset, space_id=sid)
    op = DiscoveredOperator(name="__op_subset", skeleton_ref=skel_root, arity=3, sample_count=4)
    return skel_root, cue, rel_subset, op


def test_mechanism_heldout_false_positive_specificity_filter():
    """F2（机制层 false-positive 反例测·审1 CONDITION 1·审2 F-1 诚实框法）：和（通用连接词）落 CAUSES×3 + SUBSET×3
    （同形两 REALIZES-skeleton 各 cue-blind 匹）→ tally count(和,CAUSES)=3 但不特异（other=3）→ promote **不晋**
    （specificity 滤·和 非因果 cue·防误晋 theater）。**机制层反例测·非 empirical 硬闸**：单一人造同形场景验
    specificity 机制有效·非真语料 false-positive 率测量（empirical defer 断奶后 W4 探针·超阈 reject）。"""
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = True
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    _, _, rel_prims, rel_causes, op_c = _build_causes_realizes_skel(ctx)
    _, _, rel_subset, op_s = _build_subset_realizes_skel(ctx)
    # 3 distinct 和 inputs·各 cue-blind 匹两同形 skeleton（cue slot1）→ tally(和,CAUSES)++ 与 tally(和,SUBSET)++
    in_roots = []
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        ir = _ensure(ctx, f"__in_fp{i}")
        _build_input_tree(ctx, ir, [_ensure(ctx, f"__fp1_{i}"), _ensure(ctx, "和"), _ensure(ctx, f"__fp3_{i}")])
        in_roots.append(ir)
    tally_cue_slot_matches(in_roots, discovered_operators=[op_c, op_s],
                           graph=ctx.concept_graph, edge_store=ctx.edge_store,
                           backend=ctx.backend, space_id=sid, rel_primitives=rel_prims)
    w = _ensure(ctx, "和")
    assert read_structure_match_count(ctx.backend, space_id=sid, word_ref=w,
                                      rel_kind=REL_CAUSES) == PROMOTE_STRUCTURE_MATCH_MIN
    assert read_structure_match_count(ctx.backend, space_id=sid, word_ref=w,
                                      rel_kind=REL_SUBSET) == PROMOTE_STRUCTURE_MATCH_MIN, "和 匹两 skeleton 各 3"
    ref = (w[0], w[1], rel_causes[0], rel_causes[1], EDGE_RELATION_SIGNAL)
    assert not promote_edge(ctx.edge_store, ctx.node_store, ref, backend=ctx.backend), \
        "★ 和 不特异 CAUSES（count=3 ≯ other=3）→ specificity 滤·不晋（false-positive 硬闸）"
    row = ctx.edge_store.get(space_id_from=w[0], local_id_from=w[1],
                             space_id_to=rel_causes[0], local_id_to=rel_causes[1],
                             edge_type=EDGE_RELATION_SIGNAL)
    assert row["tier"] == TIER_SHADOW, "和 留 SHADOW 不晋 PRIMARY（specificity 闸守）"


# ============ G. 生产 promote 链（审1 post-impl B1·theater 闭命门·驱动生产 gate _promote_eligible）============

def test_promote_eligible_production_gate_promotes_tally_shadow():
    """G1（审1 post-impl B1 修复验证·生产 promote 链 theater 闭命门）：
    F1/F2 直接调 promote_edge **绕过生产入口** _promote_eligible（formal_train 阶段4 唯一 promote gate）→
    测不出 promote_report 缺 ORACLE_PROMOTE 分支致 tally 建 SHADOW 被预筛淘汰的 theater（审1 B1·致命）。
    本测驱动**生产 gate** _promote_eligible：fixture 同 F1（引发 落 CAUSES-REALIZES cue slot·tally count=3 +
    specific·建 SHADOW）→ _promote_eligible 扫 TIER_SHADOW → promote_report[ORACLE_PROMOTE 分支] eligible=True →
    promote_edge flip PRIMARY。**无 B1 修**（promote_report 缺分支）→ eligible=False → SHADOW 被淘汰 → 不晋 = theater。"""
    from pure_integer_ai.experiments.formal_train import _promote_eligible
    ctx = _ctx()
    gates.ORACLE_PROMOTE_MODE = True
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    skel_root, cue, rel_prims, rel_causes, op = _build_causes_realizes_skel(ctx)
    in_roots = []
    for i in range(PROMOTE_STRUCTURE_MATCH_MIN):
        ir = _ensure(ctx, f"__in_g{i}")
        _build_input_tree(ctx, ir, [_ensure(ctx, f"__g1_{i}"), _ensure(ctx, "引发"), _ensure(ctx, f"__g3_{i}")])
        in_roots.append(ir)
    tally_cue_slot_matches(in_roots, discovered_operators=[op],
                           graph=ctx.concept_graph, edge_store=ctx.edge_store,
                           backend=ctx.backend, space_id=sid, rel_primitives=rel_prims)
    w = _ensure(ctx, "引发")
    assert ctx.concept_graph.relation_cue_candidates(
        REL_CAUSES, space_id=sid) == [], "SHADOW 不得进入生成 cue 候选"
    # ★ 生产 promote gate（formal_train 阶段4 唯一入口·扫 SHADOW → promote_report 预筛 → promote_edge flip）
    promote_count, _oov = _promote_eligible(ctx, teacher=None)
    row = ctx.edge_store.get(space_id_from=w[0], local_id_from=w[1],
                             space_id_to=rel_causes[0], local_id_to=rel_causes[1],
                             edge_type=EDGE_RELATION_SIGNAL)
    assert row is not None and row["tier"] == TIER_PRIMARY, \
        "★ 生产 gate _promote_eligible 经 promote_report→promote_edge 翻 tally SHADOW→PRIMARY（B1 theater 闭）"
    assert promote_count >= 1, "生产 gate 至少翻 1 条（引发→CAUSES）"
    assert ctx.concept_graph.relation_cue_candidates(
        REL_CAUSES, space_id=sid) == [w], "晋升须失效旧空缓存并立即进入生成候选"
