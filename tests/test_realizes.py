"""Phase D §十六-bis D.1 测试（REALIZES labeled bed·option-b oracle-pair-match）。

build_realizes_edge（幂等+self-loop）+ label_realizes_is_a（option-b oracle-pair-match·外源 IS_A 标 REALIZES→__REL_SUBSET__）。
反 theater：oracle = 外源 EDGE_IS_A（SOURCE_CONCEPTNET/SOURCE_CHINESE_KB·**排 cue EPI_CUE**·anti-self-proving）·非读 `_CUE_WORDS`。

铁律：纯整数 / bit-identical（gate REALIZES_MODE default OFF→零 REALIZES 边）/ 反 theater（cue IS_A 排除）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import (
    EDGE_REALIZES, EDGE_IS_A, EDGE_CAUSES, EDGE_COMPOSES, is_registered_edge_type,
)
from pure_integer_ai.storage.edge_store import (
    SOURCE_CONCEPTNET, SOURCE_BARE_TEXT, EPI_STRUCTURED, EPI_CUE,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives, REL_SUBSET, REL_CAUSES,
)
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.cognition.process.structure_discover import (
    DiscoveredOperator, label_realizes_is_a, label_realizes_causes,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    saved = gates.REALIZES_MODE
    gates.REALIZES_MODE = False
    yield
    gates.REALIZES_MODE = saved


def _add_edge(ctx, frm, to, *, edge_type, source, epistemic, strength=1, order_index=None):
    ctx.edge_store.add(
        space_id_from=frm[0], local_id_from=frm[1],
        space_id_to=to[0], local_id_to=to[1],
        edge_type=edge_type, strength=strength, source=source,
        epistemic_origin=epistemic, order_index=order_index, tier=TIER_PRIMARY,
    )


def _build_disc_tree(ctx, root, tokens):
    """建发现式 COMPOSES 树：root（ATTR_OPERATOR·internal）→ token 叶（无 attr=CONCEPT_LEAF）·reading 序。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(tokens):
        _add_edge(ctx, root, tok, edge_type=EDGE_COMPOSES, source=SOURCE_BARE_TEXT,
                  epistemic=EPI_STRUCTURED, order_index=ti)


# ============ EDGE_REALIZES 注册 + build_realizes_edge ============

def test_realizes_registered():
    """EDGE_REALIZES=27 是 C9-bis 登记合法 edge_type（完备性 #1）。"""
    assert EDGE_REALIZES == 27
    assert is_registered_edge_type(EDGE_REALIZES), "27 在 REGISTERED_EDGE_TYPES"


def test_build_realizes_edge_idempotent():
    """幂等：同 (skeleton→rel-type) 重复 build → 仍 1 边（query_from skip·跨 round re-discover 不 corrupt）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel_test", space_id=sid)
    rel = (sid, 888)
    n1 = build_realizes_edge(ctx.edge_store, skel, rel, space_id=sid)
    n2 = build_realizes_edge(ctx.edge_store, skel, rel, space_id=sid)
    assert (n1, n2) == (1, 0), "首建 1·重 build 0（幂等 skip）"
    rows = ctx.backend.select("edge", where={
        "space_id_from": skel[0], "local_id_from": skel[1], "edge_type": EDGE_REALIZES})
    assert len(rows) == 1, "2× build → 1 行（幂等无堆叠）"


def test_build_realizes_edge_self_loop():
    """自环不建（skeleton≠rel-type）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel_self", space_id=sid)
    assert build_realizes_edge(ctx.edge_store, skel, skel, space_id=sid) == 0, "自环→0"


# ============ label_realizes_is_a（option-b oracle-pair-match） ============

def test_label_realizes_gate_off_zero():
    """gate OFF→返 0（self-gate·bit-identical）·即使有外源 IS_A + forming_roots。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("苹果", space_id=sid)
    b = ctx.concept_index.ensure("水果", space_id=sid)
    root = ctx.concept_index.ensure("__disc_test", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    _add_edge(ctx, a, b, edge_type=EDGE_IS_A, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    # gate OFF（fixture 默认）
    assert label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                               rel_primitives=rel_prims, space_id=sid) == 0, "gate OFF→0"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert rows == [], "gate OFF→零 REALIZES 边"


def test_label_realizes_oracle_match_external_isa():
    """★ gate ON + forming-sample token-pair (苹果,水果) 命中外源 EDGE_IS_A → REALIZES skeleton→__REL_SUBSET__。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("苹果", space_id=sid)   # child
    b = ctx.concept_index.ensure("水果", space_id=sid)   # parent
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])   # ordered: 苹果 在 水果 前（child→parent 方向）
    _add_edge(ctx, a, b, edge_type=EDGE_IS_A, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_subset = rel_prims[REL_SUBSET]
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=rel_prims, space_id=sid)
    assert n == 1, "oracle 命中→建 1 REALIZES 边"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert len(rows) == 1
    r = rows[0]
    assert (r["space_id_from"], r["local_id_from"]) == skel, "from = skeleton"
    assert (r["space_id_to"], r["local_id_to"]) == rel_subset, "to = __REL_SUBSET__"
    assert r["epistemic_origin"] == EPI_STRUCTURED, "honest EPI_STRUCTURED labeled bed"


def test_label_realizes_excludes_cue_isa():
    """★ anti-self-proving：cue-derived(EPI_CUE) IS_A 边**不**作 oracle（排·防 Cue 泄漏·Phase A审2命门）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("苹果", space_id=sid)
    b = ctx.concept_index.ensure("水果", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    # cue IS_A（EPI_CUE·observe "X 是 Y" 系词提取）— 不作 oracle
    _add_edge(ctx, a, b, edge_type=EDGE_IS_A, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=rel_prims, space_id=sid)
    assert n == 0, "★ cue IS_A（EPI_CUE）排除·anti-self-proving（仅外源 CONCEPTNET/CHINESE_KB 作 oracle）"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert rows == [], "cue ISA 不触发 REALIZES"


def test_label_realizes_no_match_no_edge():
    """forming-sample token-pair 不在 oracle IS_A → 无 REALIZES（无证据不标·诚实）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("猫", space_id=sid)
    b = ctx.concept_index.ensure("鱼", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    # 无 IS_A 边（猫 鱼无 is_a 关系）
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                            rel_primitives=rel_prims, space_id=sid)
    assert n == 0, "无 oracle 命中→不标（无证据·诚实·非 theater）"


def test_label_realizes_empty_forming_roots_skipped():
    """empty forming_roots（load 重建）→ skip（REALIZES 已在图·resume 不重标）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=())   # load 重建·forming_roots 空
    assert label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                               rel_primitives=rel_prims, space_id=sid) == 0, "empty forming_roots→skip"


# ============ label_realizes_causes（option-b oracle-pair-match·镜像 IS_A） ============

def test_label_realizes_causes_gate_off_zero():
    """gate OFF→返 0（self-gate·bit-identical）·即使有外源 CAUSES + forming_roots。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("雨", space_id=sid)
    b = ctx.concept_index.ensure("洪", space_id=sid)
    root = ctx.concept_index.ensure("__disc_test", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    _add_edge(ctx, a, b, edge_type=EDGE_CAUSES, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    # gate OFF（fixture 默认）
    assert label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                 rel_primitives=rel_prims, space_id=sid) == 0, "gate OFF→0"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert rows == [], "gate OFF→零 REALIZES 边"


def test_label_realizes_causes_oracle_match_external():
    """★ gate ON + forming-sample token-pair (雨,洪) 命中外源 EDGE_CAUSES → REALIZES skeleton→__REL_CAUSES__。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("雨", space_id=sid)   # cause
    b = ctx.concept_index.ensure("洪", space_id=sid)   # effect
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])   # ordered: 雨 在 洪 前（cause→effect 方向）
    _add_edge(ctx, a, b, edge_type=EDGE_CAUSES, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=rel_prims, space_id=sid)
    assert n == 1, "oracle 命中→建 1 REALIZES 边"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert len(rows) == 1
    r = rows[0]
    assert (r["space_id_from"], r["local_id_from"]) == skel, "from = skeleton"
    assert (r["space_id_to"], r["local_id_to"]) == rel_causes, "to = __REL_CAUSES__"
    assert r["epistemic_origin"] == EPI_STRUCTURED, "honest EPI_STRUCTURED labeled bed"


def test_label_realizes_causes_excludes_cue():
    """★ anti-self-proving：cue-derived(EPI_CUE) CAUSES 边**不**作 oracle（排·防 使 cue 泄漏·condition-6 命门）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("雨", space_id=sid)
    b = ctx.concept_index.ensure("洪", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    # cue CAUSES（EPI_CUE·observe "X 使 Y" 指向词提取·source=BARE_TEXT）— 不作 oracle
    _add_edge(ctx, a, b, edge_type=EDGE_CAUSES, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=rel_prims, space_id=sid)
    assert n == 0, "★ cue CAUSES（EPI_CUE）排除·anti-self-proving（仅外源 CONCEPTNET 作 oracle）"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert rows == [], "cue CAUSES 不触发 REALIZES"


def test_label_realizes_causes_excludes_observe_structured():
    """★ condition-6 循环命门：observe-derived CAUSES（source≠CONCEPTNET·即便 EPI_STRUCTURED）**不**作 oracle。

    防使-skeleton 循环：使 cue → observe 建结构化 CAUSES（source=raw.source≠CONCEPTNET）→ 若作 oracle →
    labeler 自证"实现 CAUSES"= theater。oracle 须=外源 ConceptNet KB（source==SOURCE_CONCEPTNET）。
    本测与 excludes_cue 互补：证判别键是 **source**（非 epistemic）·两路 observe 派生全排除。
    """
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("雨", space_id=sid)
    b = ctx.concept_index.ensure("洪", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    # observe structured CAUSES（source=raw.source=BARE_TEXT 非 CONCEPTNET·epistemic=STRUCTURED）— 不作 oracle
    _add_edge(ctx, a, b, edge_type=EDGE_CAUSES, source=SOURCE_BARE_TEXT, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=rel_prims, space_id=sid)
    assert n == 0, "★ observe-derived CAUSES（source≠CONCEPTNET）排除·condition-6 循环命门"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert rows == [], "非外源 KB CAUSES 不触发 REALIZES"


def test_label_realizes_causes_directional():
    """方向性：oracle CAUSES(雨→洪) 命中 ordered (雨,洪)；反向 tokens [洪,雨] 无 oracle CAUSES(洪→雨)→不命中。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    yu = ctx.concept_index.ensure("雨", space_id=sid)
    hong = ctx.concept_index.ensure("洪", space_id=sid)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    _add_edge(ctx, yu, hong, edge_type=EDGE_CAUSES, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    # 正向 skeleton：tokens=[雨, 洪] → ordered (雨,洪) 命中
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [yu, hong])
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    assert label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                 rel_primitives=rel_prims, space_id=sid) == 1, "正向 (雨→洪) 命中"
    # 反向 skeleton：tokens=[洪, 雨] → ordered (洪,雨)·oracle 无 CAUSES(洪→雨)→不命中
    skel2 = ctx.concept_index.ensure("__skel2", space_id=sid)
    root2 = ctx.concept_index.ensure("__disc2", space_id=sid)
    _build_disc_tree(ctx, root2, [hong, yu])
    op2 = DiscoveredOperator(name="n2", skeleton_ref=skel2, arity=2, sample_count=1,
                             name_ref=(0, 0), forming_roots=(root2,))
    assert label_realizes_causes([op2], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                 rel_primitives=rel_prims, space_id=sid) == 0, "反向 (洪→雨) 无 oracle→不命中"


def test_label_realizes_causes_no_match_no_edge():
    """forming-sample token-pair 不在 oracle CAUSES → 无 REALIZES（无证据不标·诚实）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    a = ctx.concept_index.ensure("猫", space_id=sid)
    b = ctx.concept_index.ensure("鱼", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [a, b])
    # 无 CAUSES 边（猫 鱼 无因果关系）
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n = label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                              rel_primitives=rel_prims, space_id=sid)
    assert n == 0, "无 oracle 命中→不标（无证据·诚实·非 theater）"


def test_label_realizes_causes_empty_forming_roots_skipped():
    """empty forming_roots（load 重建）→ skip（REALIZES 已在图·resume 不重标）。"""
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=2, sample_count=1,
                            name_ref=(0, 0), forming_roots=())   # load 重建·forming_roots 空
    assert label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                 rel_primitives=rel_prims, space_id=sid) == 0, "empty forming_roots→skip"


# ============ 跨关系隔离（LOW-1+LOW-3 审补·锁 correspondence 不变量） ============

def test_label_realizes_cross_relation_isolation():
    """★ LOW-1+LOW-3（审补）：单 skeleton forming-sample 同时含外源 IS_A 对 + 外源 CAUSES 对 →
    两 labeler 各自写 REALIZES 到**不同** target（__REL_SUBSET__ / __REL_CAUSES__）·互不干扰。

    锁 correspondence 层不变量：一骨架可携多条关系 label（IS_A + CAUSES 共存）·两 labeler 独立·
    target 区分靠 rel_primitives[REL_SUBSET] vs [REL_CAUSES]（非合并）。防未来重构把两 labeler
    合并 target 后无测可抓。同时证 CAUSES 命中发生在**非首对**（(雨,洪) 是 4 叶树第 6 对·LOW-3）。
    """
    ctx = make_train_context(DictBackend())
    gates.REALIZES_MODE = True
    sid = ctx.space_id
    skel = ctx.concept_index.ensure("__skel", space_id=sid)
    apple = ctx.concept_index.ensure("苹果", space_id=sid)
    fruit = ctx.concept_index.ensure("水果", space_id=sid)
    yu = ctx.concept_index.ensure("雨", space_id=sid)
    hong = ctx.concept_index.ensure("洪", space_id=sid)
    root = ctx.concept_index.ensure("__disc", space_id=sid)
    _build_disc_tree(ctx, root, [apple, fruit, yu, hong])   # reading 序 4 叶
    # 外源 IS_A（苹果→水果）+ 外源 CAUSES（雨→洪）·两不同 token 对·两不同关系
    _add_edge(ctx, apple, fruit, edge_type=EDGE_IS_A, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    _add_edge(ctx, yu, hong, edge_type=EDGE_CAUSES, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_subset = rel_prims[REL_SUBSET]
    rel_causes = rel_prims[REL_CAUSES]
    op = DiscoveredOperator(name="n", skeleton_ref=skel, arity=4, sample_count=1,
                            name_ref=(0, 0), forming_roots=(root,))
    n_isa = label_realizes_is_a([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                rel_primitives=rel_prims, space_id=sid)
    n_causes = label_realizes_causes([op], graph=ctx.concept_graph, edge_store=ctx.edge_store,
                                     rel_primitives=rel_prims, space_id=sid)
    assert (n_isa, n_causes) == (1, 1), "两 labeler 各命中各自 oracle 对→各 1 边·独立"
    rows = ctx.backend.select("edge", where={"edge_type": EDGE_REALIZES})
    assert len(rows) == 2, "单骨架携 2 REALIZES 边（IS_A + CAUSES·target 区分）"
    targets = {(r["space_id_to"], r["local_id_to"]) for r in rows}
    assert targets == {rel_subset, rel_causes}, "target = {__REL_SUBSET__, __REL_CAUSES__}·互异·非合并"
    for r in rows:
        assert (r["space_id_from"], r["local_id_from"]) == skel, "from = 同一 skeleton"
