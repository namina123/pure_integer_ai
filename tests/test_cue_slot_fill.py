"""命门③ 候选 B：无 learned relation cue 时的骨架 cue fallback 测试。

候选 B=结构活化：dispatch_slot cue 位读
workmem.current_cue_sig[slot_idx]（cue token ConceptRef）-> surface_of(cue_token) 直出功能词·
返 CUE_SLOT_FILL=3。若存在 learned D:11 relation cue，则转入候选竞争；该路径由 correspondence 测试覆盖。

**反 theater**：cue token 由 cue 聚簇（_cluster_by_cue sustainable-split）涌现（闭类 是/使 重复≥K
vs 开类内容词<K·天然区分·无须词表·守 §十五 C5）·非硬编码语法角色。
**bit-identical**：gate CUE_SLOT_FILL_MODE OFF -> cue 位走 collide 返 LINEAGE_CONCEPT_FILL=1·逐字现状。
**防御**：cue_sig 缺/越界/None/surface None -> fall-through 走 collide（不阻断生成）。
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
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.storage.selection_pref_count import register_selection_pref_count
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    RoleSlot, PathResult, LANG_ZH, CUE_SLOT_FILL, LINEAGE_CONCEPT_FILL,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def cue_env():
    """候选 B 单测环境（dict backend·core space·struct_ref + fallback candidate·REFERS_TO 接通 activate）。"""
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


def _make_graph(b, cue_surface_map):
    """建 ConceptGraph·注入 surface_of resolver（cue_surface_map: ref->str·测试隔离·不依赖 ORDINAL_SURFACE_MODE）。"""
    def _resolver(ref):
        return cue_surface_map.get(ref)
    return ConceptGraph(b, surface_of=_resolver)


def _setup_struct_with_candidate(ci, es, ns, sid):
    """struct_ref + fallback candidate（REFERS_TO candidate->struct·activate_candidates(struct) 返 [candidate]）。"""
    struct_ref = ci.ensure("struct", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    candidate = ci.ensure("cand", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    es.add(space_id_from=candidate[0], local_id_from=candidate[1],
           space_id_to=struct_ref[0], local_id_to=struct_ref[1],
           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_TEACHER,
           epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY)
    return struct_ref, candidate


def _workmem(*, is_cue, cue_sig, slot_idx, produced=None):
    """SimpleNamespace workmem·显式带候选 B 三字段（current_slot_is_cue/cue_sig/slot_idx）。"""
    return SimpleNamespace(
        current_slot_is_cue=is_cue,
        current_cue_sig=cue_sig,
        current_slot_idx=slot_idx,
        current_rel_kind=0,
        current_cue_slots=frozenset(),
        prior_topic_refs=[],
        produced_refs=produced or [],
    )


# ============ gate OFF bit-identical ============

def test_cue_slot_fill_gate_off_falls_through_collide(cue_env):
    """gate OFF -> cue 位走 collide·返 LINEAGE_CONCEPT_FILL=1（bit-identical·逐字现状）。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    graph = _make_graph(b, {})   # 无 surface 注入（gate OFF surface_of 不被 cue 路径读）
    workmem = _workmem(is_cue=True, cue_sig=(None,), slot_idx=0)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = False
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, f"gate OFF cue 位走 collide 返 src=1·got {src}"
        assert word is not None, "gate OFF fall-through collide 仍返词"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_off_minimal_workmem_no_crash(cue_env):
    """gate OFF + workmem 无 cue 三字段（SimpleNamespace 最小）-> getattr 默认守·不 crash·走 collide。

    镜像既有 _correspondence_bonus caller getattr 范式（审1 LOW-1 契约级显式 default）。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    graph = _make_graph(b, {})
    workmem = SimpleNamespace(prior_topic_refs=[], produced_refs=[])   # 无 cue 字段
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = False
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "gate OFF 无 cue 字段 getattr 默认 False->走 collide"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


# ============ gate ON cue 位直出 ============

def test_cue_slot_fill_gate_on_cue_slot_returns_cue_token(cue_env):
    """★gate ON + cue 位 + cue_sig valid + surface 非 None -> 直出 cue token 功能词·返 CUE_SLOT_FILL=3。

    无 learned relation cue 时，骨架 cue 是稳定 fallback。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    cue_token = ci.ensure("是", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)   # 闭类功能词
    graph = _make_graph(b, {cue_token: "是"})
    workmem = _workmem(is_cue=True, cue_sig=(cue_token,), slot_idx=0)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "是", f"cue token 直出功能词'是'·got {word!r}"
        assert src == CUE_SLOT_FILL, f"cue 位返 CUE_SLOT_FILL=3·got {src}"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_on_multi_slot_picks_correct_idx(cue_env):
    """多 slot cue_sig：slot_idx=1 取 cue_sig[1] 非 [0]（workmem.current_slot_idx 走 workmem·HIGH-1）。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    cue0 = ci.ensure("是", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    cue1 = ci.ensure("使", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    graph = _make_graph(b, {cue0: "是", cue1: "使"})
    # cue_sig=(None, cue1)·slot_idx=1 -> 取 cue1="使"（非 cue0）
    workmem = _workmem(is_cue=True, cue_sig=(None, cue1), slot_idx=1)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert word == "使", f"slot_idx=1 取 cue_sig[1]='使'·got {word!r}"
        assert src == CUE_SLOT_FILL
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


# ============ gate ON fall-through 防御（5 case）============

def test_cue_slot_fill_gate_on_non_cue_slot_falls_through(cue_env):
    """gate ON + 非 cue 位（current_slot_is_cue=False）-> 走 collide·返 LINEAGE_CONCEPT_FILL=1。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    cue_token = ci.ensure("是", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    graph = _make_graph(b, {cue_token: "是"})
    workmem = _workmem(is_cue=False, cue_sig=(cue_token,), slot_idx=0)   # 非 cue 位
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "非 cue 位走 collide"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_on_cue_sig_none_at_idx_falls_through(cue_env):
    """gate ON + cue 位 + cue_sig[slot_idx] is None（非 cue 位的占位）-> fall-through collide。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    graph = _make_graph(b, {})
    workmem = _workmem(is_cue=True, cue_sig=(None,), slot_idx=0)   # cue_sig[0]=None
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "cue_sig[slot_idx]=None->fall-through collide"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_on_empty_cue_sig_falls_through(cue_env):
    """gate ON + cue 位 + cue_sig=()（冷启动/未 stash）-> fall-through collide。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    graph = _make_graph(b, {})
    workmem = _workmem(is_cue=True, cue_sig=(), slot_idx=0)   # 空 cue_sig
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "空 cue_sig->fall-through collide"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_on_slot_idx_out_of_range_falls_through(cue_env):
    """gate ON + cue 位 + slot_idx >= len(cue_sig)（length-guard 不等退化）-> fall-through collide。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    cue_token = ci.ensure("是", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    graph = _make_graph(b, {cue_token: "是"})
    # cue_sig 长 1·slot_idx=5 越界
    workmem = _workmem(is_cue=True, cue_sig=(cue_token,), slot_idx=5)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "slot_idx 越界->fall-through collide"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


def test_cue_slot_fill_gate_on_surface_none_falls_through(cue_env):
    """gate ON + cue 位 + cue_sig valid + surface_of 返 None（无 surface 对应）-> fall-through collide。

    守承重不变量：surface_of 无对应返 None 非 ""（judge.J2s truthiness）-> cue 路径 fall-through 不产空词。"""
    b, sid, es, ns, ci = cue_env
    struct_ref, _ = _setup_struct_with_candidate(ci, es, ns, sid)
    cue_token = ci.ensure("是", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # surface_of 返 None（cue_token 无 surface 映射·resolver get 返 None）
    graph = _make_graph(b, {})
    workmem = _workmem(is_cue=True, cue_sig=(cue_token,), slot_idx=0)
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    saved = gates.CUE_SLOT_FILL_MODE
    gates.CUE_SLOT_FILL_MODE = True
    try:
        word, src = dispatch_slot(slot, PathResult(), graph, workmem, LANG_ZH)
        assert src == LINEAGE_CONCEPT_FILL, "surface None->fall-through collide（不产空词）"
    finally:
        gates.CUE_SLOT_FILL_MODE = saved


# ============ gate 注册 + CUE_SLOT_FILL 常量 ============

def test_cue_slot_fill_gate_registered_default_off():
    """gate CUE_SLOT_FILL_MODE 已注册·default OFF（bit-identical 守 CI）。"""
    from pure_integer_ai.config import gates
    assert hasattr(gates, "CUE_SLOT_FILL_MODE"), "CUE_SLOT_FILL_MODE 已注册"
    assert gates.CUE_SLOT_FILL_MODE is False, "default OFF（gate OFF 逐字不变·bit-identical）"


def test_cue_slot_fill_constant_value():
    """CUE_SLOT_FILL=3·lineage 值集 {1,2,3}·与 LINEAGE_CONCEPT_FILL=1/LINEAGE_DEF_REPLAY=2 不撞。"""
    from pure_integer_ai.cognition.shared.types import (
        CUE_SLOT_FILL, LINEAGE_CONCEPT_FILL, LINEAGE_DEF_REPLAY,
    )
    assert CUE_SLOT_FILL == 3
    assert LINEAGE_CONCEPT_FILL == 1
    assert LINEAGE_DEF_REPLAY == 2
    assert len({LINEAGE_CONCEPT_FILL, LINEAGE_DEF_REPLAY, CUE_SLOT_FILL}) == 3, "三值互异"


# ============ formal_train 生产入口 flip 契约守卫（防再漏翻·post-impl 审 §8）============

def test_formal_train_flips_cue_slot_fill_mode_in_production_try_finally():
    """契约守卫：生产 context-local override 必须覆盖完整 cue slot gate 链。

    漏翻后果：生产训练 CUE_SLOT_FILL_MODE 保持 default False->dispatch_slot cue 位早 return 永不 fire->
    CUE_SLOT_FILL=0->命门③ 候选 B 生产场景下不活 = theater。
    """
    import inspect
    import pure_integer_ai.experiments.formal_train as ft
    import pure_integer_ai.experiments.train_gate_profile as gate_profile
    train_src = inspect.getsource(ft)
    profile_src = inspect.getsource(gate_profile)
    assert "training_gate_token = push_production_training_gates()" in train_src
    assert "reset_production_training_gates(training_gate_token)" in train_src
    for gate_name in [
        "CUE_SLOT_FILL_MODE",
        "COMPOSES_COMBINE_MODE",
        "CORRESPONDENCE_SLOT_MODE",
        "REALIZES_MODE",
        "CUE_CLUSTER_MODE",
        "ORACLE_PROMOTE_MODE",
    ]:
        assert f'"{gate_name}": True' in profile_src
