"""命门③ 候选 C：slot_lca 抽象约束测试（doc/重来_命门③_句子组装_结构抽象活化_设计_2026-07-18）。

候选 C=抽象活化：dispatch_slot 内容词位（非 cue 位）按 slot 的 IS_A LCA 类过滤候选
（c IS_A slot_lca 留·reflexive-transitive·c==slot_lca or slot_lca in ancestors(c)）·
空集 fallback 走 collide 不阻断生成。slot_lca 由 _cluster_by_lca（set_lca 簇内 slot 全 token LCA）
写 ATTR_SLOT_ROLE=9·已 live 零消费者·候选 C = 加读侧+生成消费。

**反 theater**：slot_lca 由 set_lca 涌现（IS_A 图结构计算非自证）·非硬编码语法角色
（abstraction.py:22-24"抽象层绝不用 role_seq/位置桶"）·主谓宾作下层涌现。
**bit-identical**：gate SLOT_LCA_CONSTRAINT_MODE OFF -> 不进过滤 -> candidates 不变 -> 走 collide 返
LINEAGE_CONCEPT_FILL=1·逐字现状。
**防御**：current_slot_lca=() / 候选空集 -> fallback 走 collide（不阻断生成）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO, EDGE_IS_A
from pure_integer_ai.storage.selection_pref_count import register_selection_pref_count
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    RoleSlot, PathResult, LANG_ZH, LINEAGE_CONCEPT_FILL,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def lca_env():
    """候选 C 单测环境（dict backend·core space·struct_ref + 2 candidate·REFERS_TO 接通 activate）。"""
    b = DictBackend()
    bootstrap(b)
    register_selection_pref_count(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ns, ci
    b.close()


def _make_graph(b, surface_map):
    """建 ConceptGraph·注入 surface_of resolver（surface_map: ref->str·测试隔离）。"""
    def _resolver(ref):
        return surface_map.get(ref)
    return ConceptGraph(b, surface_of=_resolver)


def _setup_struct_with_candidates(ci, es, ns, sid):
    """struct_ref + 2 candidate（cand_b 先 ensure ref 小·cand_a 后 ref 大·都 REFERS_TO struct）。

    activate_candidates(struct) 返 [cand_b, cand_a]（REFERS_TO from=candidate to=struct）。
    gate OFF tiebreak 选 ref 小=cand_b·gate ON filter 留 IS_A slot_lca 的 candidate->选 cand_a（对照证 filter）。
    """
    struct_ref = ci.ensure("struct", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    cand_b = ci.ensure("cand_b", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)   # 先 ensure·ref 小
    cand_a = ci.ensure("cand_a", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)   # 后 ensure·ref 大
    for cand in (cand_a, cand_b):
        es.add(space_id_from=cand[0], local_id_from=cand[1],
               space_id_to=struct_ref[0], local_id_to=struct_ref[1],
               edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_TEACHER,
               epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY)
    return struct_ref, cand_a, cand_b


def _add_isa(es, child, parent):
    """EDGE_IS_A child->parent（child IS_A parent·from=child to=parent·build_isa_ancestor_map 读此建 ancestor_map）。"""
    es.add(space_id_from=child[0], local_id_from=child[1],
           space_id_to=parent[0], local_id_to=parent[1],
           edge_type=EDGE_IS_A, strength=1, source=SOURCE_TEACHER,
           epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY)


def _workmem(*, slot_lca=(), is_cue=False, produced=None):
    """SimpleNamespace workmem·显式带候选 C 字段（current_slot_lca）+ 候选 B 字段（防 dispatch_slot 读 getattr 默认）。"""
    return SimpleNamespace(
        current_slot_lca=slot_lca,
        current_slot_is_cue=is_cue,
        current_cue_sig=(),
        current_slot_idx=0,
        current_rel_kind=0,
        current_cue_slots=frozenset(),
        prior_topic_refs=[],
        produced_refs=produced or [],
    )


# ============ gate OFF bit-identical ============

def test_slot_lca_gate_off_falls_through_collide(lca_env):
    """gate OFF -> 无 LCA 过滤 -> [cand_b, cand_a] -> collide 都 0 -> tiebreak 选 ref 小=cand_b。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    slot_lca = ci.ensure("lca", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=slot_lca)   # gate OFF current_slot_lca 不被读
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = False
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, f"gate OFF 走 collide 返 src=1·got {src}"
        assert word == "b词", f"gate OFF tiebreak 选 ref 小 cand_b·got {word!r}"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_gate_off_minimal_workmem_no_crash(lca_env):
    """gate OFF + workmem 无 current_slot_lca（SimpleNamespace 最小）-> getattr 默认守·不 crash·走 collide。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = SimpleNamespace(prior_topic_refs=[], produced_refs=[])   # 无 current_slot_lca
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = False
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "gate OFF 无 current_slot_lca getattr 默认 ()->走 collide"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


# ============ gate ON 抽象约束过滤 ============

def test_slot_lca_gate_on_filters_to_isa_descendant(lca_env):
    """★gate ON + cand_a IS_A slot_lca + cand_b 不 -> filter 留 [cand_a] -> 选 cand_a（非 ref 小 cand_b）。

    反 theater 心脏：内容词按学来的抽象类约束过滤·非硬编码语法角色。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    slot_lca = ci.ensure("lca", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _add_isa(es, cand_a, slot_lca)   # cand_a IS_A slot_lca（留）·cand_b 不 IS_A（滤）
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=slot_lca)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "a词", f"filter 留 IS_A slot_lca 的 cand_a·got {word!r}"
        assert src == LINEAGE_CONCEPT_FILL, "filter 后仍走 collide 选词返 src=1"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_gate_on_empty_fallback(lca_env):
    """gate ON + 全候选不 IS_A slot_lca -> cand_f 空 -> fallback candidates 不变 -> [cand_b, cand_a] -> 选 ref 小 cand_b。

    mirror hub filter `if cand_f:` 空集退化走 collide·不阻断生成。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    slot_lca = ci.ensure("lca", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # 不建任何 IS_A 边 -> cand_a/cand_b 都不 IS_A slot_lca -> cand_f 空 -> fallback
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=slot_lca)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "b词", f"空集 fallback 走 collide 选 ref 小 cand_b·got {word!r}"
        assert src == LINEAGE_CONCEPT_FILL
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_gate_on_empty_lca_skips(lca_env):
    """gate ON + current_slot_lca=()（无约束位）-> 跳过过滤 -> [cand_b, cand_a] -> 选 ref 小 cand_b。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    slot_lca = ci.ensure("lca", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _add_isa(es, cand_a, slot_lca)
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=())   # ()=无约束位->跳过
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "b词", f"无约束位跳过 filter 选 ref 小 cand_b·got {word!r}"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_gate_on_root_class_all_match(lca_env):
    """gate ON + slot_lca 是根类（cand_a/cand_b 都 IS_A 根）-> 全留 -> [cand_b, cand_a] -> 选 ref 小 cand_b。

    非 theater：根类全匹配=filter 无效果退化 fallback·sound。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    root = ci.ensure("root", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _add_isa(es, cand_a, root)   # 都 IS_A root
    _add_isa(es, cand_b, root)
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=root)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "b词", f"根类全留选 ref 小 cand_b·got {word!r}"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_gate_on_reflexive_self_match(lca_env):
    """gate ON + candidate==slot_lca -> is_a_descendant_of(c, c) True（reflexive）-> 留。

    守 reflexive-transitive 判定（c==slot_lca or ancestor in ancestors(c)）·非仅 nearest。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    # slot_lca=cand_a 自身（reflexive 命中）·不建 IS_A 边（验 c==ancestor 短路）
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    workmem = _workmem(slot_lca=cand_a)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "a词", f"reflexive cand_a==slot_lca 留选 cand_a·got {word!r}"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


def test_slot_lca_is_a_descendant_of_reflexive_transitive(lca_env):
    """is_a_descendant_of 单元：c==ancestor True / 直接父 True / 传递祖先 True / 无关 False。

    守 design 决策 reflexive-transitive（非仅 nearest==slot_lca·c 可深 slot_lca 下多层）。"""
    b, sid, es, ns, ci = lca_env
    _ = NodeStore(b)
    c = ci.ensure("c", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    parent = ci.ensure("parent", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    grand = ci.ensure("grand", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    unrelated = ci.ensure("unrelated", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    es = EdgeStore(b)
    _add_isa(es, c, parent)    # c IS_A parent
    _add_isa(es, parent, grand)   # parent IS_A grand -> c 传递 IS_A grand
    graph = _make_graph(b, {})
    assert graph.is_a_descendant_of(c, c) is True, "reflexive: c==c"
    assert graph.is_a_descendant_of(c, parent) is True, "直接父"
    assert graph.is_a_descendant_of(c, grand) is True, "传递祖先（nearest==parent != grand 但 c IS_A grand True）"
    assert graph.is_a_descendant_of(c, unrelated) is False, "无关"


def test_slot_lca_gate_on_multi_slot_per_slot_stash(lca_env):
    """★multi-slot per-slot stash 选对（post-impl 审建议·mirror B multi-slot test 范式）：

    generate slot loop 每 slot 设 current_slot_lca·dispatch_slot 读对当前 slot_idx 的 LCA·非串台。
    模拟 2-slot：slot0 LCA=lca0（留 cand_a）·slot1 LCA=lca1（留 cand_b·异 slot0）。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    lca0 = ci.ensure("lca0", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    lca1 = ci.ensure("lca1", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _add_isa(es, cand_a, lca0)   # cand_a IS_A lca0（slot0 留 cand_a）
    _add_isa(es, cand_b, lca1)   # cand_b IS_A lca1（slot1 留 cand_b·异 slot0）
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    saved = gates.SLOT_LCA_CONSTRAINT_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    try:
        # slot0：LCA=lca0 -> filter 留 [cand_a] -> 选 cand_a
        workmem0 = _workmem(slot_lca=lca0)
        slot0 = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
        word0, _ = dispatch_slot(slot0, PathResult(), graph, workmem0, LANG_ZH)
        assert word0 == "a词", f"slot0 LCA=lca0 留 cand_a·got {word0!r}"
        # slot1：LCA=lca1 -> filter 留 [cand_b] -> 选 cand_b（非 slot0 的 cand_a·不串台）
        workmem1 = _workmem(slot_lca=lca1)
        slot1 = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
        word1, _ = dispatch_slot(slot1, PathResult(), graph, workmem1, LANG_ZH)
        assert word1 == "b词", f"slot1 LCA=lca1 留 cand_b·非串台·got {word1!r}"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved


# ============ MED-1 混合 case（CUE_SLOT_FILL_MODE OFF + SLOT_LCA_CONSTRAINT_MODE ON）============

def test_mixed_case_cue_slot_off_lca_on_sound(lca_env):
    """★MED-1 诚实边界（审2）：CUE_SLOT_FILL_MODE OFF + SLOT_LCA_CONSTRAINT_MODE ON + cue 位（current_slot_is_cue=True）
    -> cue 位不早 return（B gate OFF）-> 走 LCA filter。sound 守：不崩 + 空集 fallback 走 collide·bit-identical。

    语义错位（cue slot ATTR_SLOT_ROLE 是 cue token LCA 非内容词抽象）但生产 B+C 共翻不触发·仅实验场景。"""
    b, sid, es, ns, ci = lca_env
    struct_ref, cand_a, cand_b = _setup_struct_with_candidates(ci, es, ns, sid)
    slot_lca = ci.ensure("lca", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _add_isa(es, cand_a, slot_lca)
    graph = _make_graph(b, {cand_a: "a词", cand_b: "b词"})
    # cue 位（current_slot_is_cue=True）但 CUE_SLOT_FILL_MODE OFF -> 不早 return -> 走 filter
    workmem = _workmem(slot_lca=slot_lca, is_cue=True)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved_lca = gates.SLOT_LCA_CONSTRAINT_MODE
    saved_cue = gates.CUE_SLOT_FILL_MODE
    gates.SLOT_LCA_CONSTRAINT_MODE = True
    gates.CUE_SLOT_FILL_MODE = False   # 混合 case：B OFF cue 不早 return
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        # sound 守：不崩 + filter 留 cand_a（IS_A slot_lca）-> 选 cand_a（非 ref 小 cand_b）
        assert word == "a词", f"混合 case cue slot 走 LCA filter 留 cand_a·got {word!r}"
        assert src == LINEAGE_CONCEPT_FILL, "filter 后走 collide 返 src=1"
    finally:
        gates.SLOT_LCA_CONSTRAINT_MODE = saved_lca
        gates.CUE_SLOT_FILL_MODE = saved_cue


# ============ gate 注册 ============

def test_slot_lca_gate_registered_default_off():
    """gate SLOT_LCA_CONSTRAINT_MODE 已注册·default OFF（bit-identical 守 CI）。"""
    from pure_integer_ai.config import gates
    assert hasattr(gates, "SLOT_LCA_CONSTRAINT_MODE"), "SLOT_LCA_CONSTRAINT_MODE 已注册"
    assert gates.SLOT_LCA_CONSTRAINT_MODE is False, "default OFF（gate OFF 逐字不变·bit-identical）"


# ============ formal_train 生产入口 flip 契约守卫（防再漏翻·mirror B post-impl §七）============

def test_formal_train_flips_slot_lca_constraint_mode_in_production_try_finally():
    """契约守卫：生产 context-local override 必须覆盖完整 slot LCA gate 链。

    漏翻后果：生产训练 SLOT_LCA_CONSTRAINT_MODE 保持 default False->dispatch_slot 内容词位 LCA filter
    永不 fire->内容词无抽象类约束=抽象活化不活 = theater。
    """
    import inspect
    import pure_integer_ai.experiments.formal_train as ft
    import pure_integer_ai.experiments.train_gate_profile as gate_profile
    train_src = inspect.getsource(ft)
    profile_src = inspect.getsource(gate_profile)
    assert "training_gate_token = push_production_training_gates()" in train_src
    assert "reset_production_training_gates(training_gate_token)" in train_src
    for gate_name in [
        "SLOT_LCA_CONSTRAINT_MODE",
        "COMPOSES_COMBINE_MODE",
        "CORRESPONDENCE_SLOT_MODE",
        "REALIZES_MODE",
        "CUE_CLUSTER_MODE",
        "ORACLE_PROMOTE_MODE",
        "CUE_SLOT_FILL_MODE",
    ]:
        assert f'"{gate_name}": True' in profile_src
