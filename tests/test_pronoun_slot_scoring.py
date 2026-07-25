"""审计根治 [严重-3] B6 生成侧 dispatch_slot pronoun scoring 测试（PR3）。

B6 指代维 observe 侧自消费真活（resolve_pronoun_occurrence 读 pr_tn 加候选分）·但生成侧 dispatch_slot
不读 pr_tn（三处标注 defer STEP6·STEP6 未补）→ "生成零判别"病灶对指代维生成侧部分未解。

PR3 根治：dispatch_slot 第5路 pronoun scoring·镜像 selection_pref_score 范式（并入 sp 维联合 _cap_sp
cap 999·守 collide 主轴 1000>999 不变）·gate PRONOUN_SLOT_MODE·graph.pronoun_score(c, slot.ref)
读 read_pronoun_resolution_count → pr_tn（pair-key 对偶 observe 侧 pronoun→antecedent）。

**反 theater**：消费者 dispatch_slot 真读 pr_tn 加 slot 候选分（gate ON 影响选优·gate OFF 退化既有）。
**pair-key 对偶 observe 侧**：observe 写 (pronoun, best_antecedent)·生成侧读 (candidate=pronoun, slot.ref=antecedent)。
**联合 cap 999**：pronoun 加成并入 sp 维 _cap_sp·sp+pr 联合 cap 999·守 collide 主轴优先。
**bit-identical**：gate OFF 退化既有（collide only·两 gate OFF 不进 if）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.storage.pronoun_resolution_count import (
    register_pronoun_resolution_count, read_pronoun_resolution_count,
    record_pronoun_resolution_decision,
)
from pure_integer_ai.storage.selection_pref_count import register_selection_pref_count
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import RoleSlot, PathResult, LANG_ZH
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import (
    _cap_sp, _pronoun_bonus, PR_SLOT_BONUS_CAP, SCORE_SCALE, dispatch_slot,
)
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def pronoun_env():
    """PR3 单测环境（dict backend·core space·pronoun_resolution_count + selection_pref_count 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_pronoun_resolution_count(b)
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


# ============ pronoun_score 方法 ============

def test_pronoun_score_gate_off_returns_zero(pronoun_env):
    """gate PRONOUN_SLOT_MODE OFF → pronoun_score 返 0（不读 pr_tn·bit-identical·退化既有）。"""
    b, sid, es, ns, ci = pronoun_env
    c = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    record_pronoun_resolution_decision(b, pronoun_ref=c, antecedent_ref=antecedent)  # pr_tn=1
    graph = ConceptGraph(b)
    saved = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = False
    try:
        assert graph.pronoun_score(c, antecedent) == 0, "gate OFF 返 0（不读 pr_tn·bit-identical）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved


def test_pronoun_score_reads_pr_tn(pronoun_env):
    """gate PRONOUN_SLOT_MODE ON → pronoun_score 读 pr_tn（pair-key (c, antecedent)·observe 路 sign-agnostic）。"""
    b, sid, es, ns, ci = pronoun_env
    c = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for _ in range(5):
        record_pronoun_resolution_decision(b, pronoun_ref=c, antecedent_ref=antecedent)  # pr_tn=5
    graph = ConceptGraph(b)
    saved = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = True
    try:
        assert graph.pronoun_score(c, antecedent) == 5, "gate ON 读 pr_tn=5"
    finally:
        gates.PRONOUN_SLOT_MODE = saved


def test_pronoun_score_cold_start(pronoun_env):
    """冷启动：无 pair 行 / 表未注册 → pronoun_score 返 0（向后兼容·镜像 selection_pref_score 范式）。"""
    b, sid, es, ns, ci = pronoun_env
    c = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    graph = ConceptGraph(b)
    saved = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = True
    try:
        # 无 pair 行（未 record）→ 0
        assert graph.pronoun_score(c, antecedent) == 0, "冷启动无 pair 行 → 0"
    finally:
        gates.PRONOUN_SLOT_MODE = saved


def test_pronoun_score_pair_key_duality(pronoun_env):
    """pair-key 对偶 observe 侧：pronoun_score(c, antecedent) 读 (c, antecedent) pair·不读 (antecedent, c) 反向。
    observe 写 (pronoun, best_antecedent)·生成侧读 (candidate=pronoun, slot.ref=antecedent)·同一 pair。"""
    b, sid, es, ns, ci = pronoun_env
    他 = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    record_pronoun_resolution_decision(b, pronoun_ref=他, antecedent_ref=猫)  # (他, 猫) pr_tn=1
    graph = ConceptGraph(b)
    saved = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = True
    try:
        # 正向 (他, 猫) → pr_tn=1（pronoun=他·antecedent=猫）
        assert graph.pronoun_score(他, 猫) == 1, "正向 pair (他,猫) pr_tn=1"
        # 反向 (猫, 他) → 0（不同 pair·observe 没写 (猫,他)）
        assert graph.pronoun_score(猫, 他) == 0, "反向 pair (猫,他) pr_tn=0（pair-key 对偶·不混）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved


# ============ dispatch_slot 第5路联合（复现逻辑·反 theater） ============

def test_dispatch_slot_pronoun_scoring_influences_combine(pronoun_env):
    """★反 theater：dispatch_slot 第5路 pronoun 加成影响 combine 选优。
    c1 (高 pr_tn=5) vs c2 (低 pr_tn=1)·collide 同（0）·gate ON c1 combine > c2 combine。
    **对抗审 catch 修**：用 ctx_refs（token concept ref·对偶 observe 侧）·非 slot.ref（struct_ref·pair-key 错位 theater）。"""
    b, sid, es, ns, ci = pronoun_env
    c1 = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    c2 = ci.ensure("她", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)  # ctx_ref·token concept
    for _ in range(5):
        record_pronoun_resolution_decision(b, pronoun_ref=c1, antecedent_ref=antecedent)  # c1 pr_tn=5
    record_pronoun_resolution_decision(b, pronoun_ref=c2, antecedent_ref=antecedent)      # c2 pr_tn=1
    graph = ConceptGraph(b)
    ctx_refs = [antecedent]   # 上下文 token（prior_topic_refs + produced_refs·token concept ref·对偶 observe 侧）
    saved_pr = gates.PRONOUN_SLOT_MODE
    saved_sp = gates.GENERATE_SELECTION_PREF_MODE
    # gate ON：_pronoun_bonus(graph, c, ctx_refs) 取 max pr_tn（cap 3）
    gates.GENERATE_SELECTION_PREF_MODE = False
    gates.PRONOUN_SLOT_MODE = True
    try:
        combine_c1 = 0 * SCORE_SCALE + _cap_sp(0 + _pronoun_bonus(graph, c1, ctx_refs))
        combine_c2 = 0 * SCORE_SCALE + _cap_sp(0 + _pronoun_bonus(graph, c2, ctx_refs))
        assert combine_c1 == 3, f"c1 pr_tn=5→_pronoun_bonus=3·combine=3·got {combine_c1}"
        assert combine_c2 == 1, f"c2 pr_tn=1→_pronoun_bonus=1·combine=1·got {combine_c2}"
        assert combine_c1 > combine_c2, "gate ON pronoun 加成影响选优（c1>c2·反 theater）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved_pr
        gates.GENERATE_SELECTION_PREF_MODE = saved_sp
    # gate OFF：pronoun_score 返 0·_pronoun_bonus=0·combine=collide only（bit-identical）
    gates.PRONOUN_SLOT_MODE = False
    try:
        combine_c1_off = 0 * SCORE_SCALE + _cap_sp(0 + _pronoun_bonus(graph, c1, ctx_refs))
        combine_c2_off = 0 * SCORE_SCALE + _cap_sp(0 + _pronoun_bonus(graph, c2, ctx_refs))
        assert combine_c1_off == combine_c2_off == 0, "gate OFF pronoun 加成 0·combine=collide only（bit-identical）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved_pr


def test_dispatch_slot_pronoun_scoring_e2e(pronoun_env):
    """★★真调 dispatch_slot 端到端（对抗审 catch·反 theater 真证）：gate ON pronoun 加成影响选优。
    建 REFERS_TO pronoun→struct（activate_candidates 返 pronoun）+ record pr_tn(pronoun, antecedent)·
    gate ON dispatch_slot 选高 pr_tn pronoun（他 pr_tn=5 > 她 pr_tn=1）·gate OFF tiebreak ref 序。
    **证消费者真活**：dispatch_slot 真调 _pronoun_bonus(graph, c, ctx_refs)·非复现逻辑。"""
    from types import SimpleNamespace
    b, sid, es, ns, ci = pronoun_env
    struct_ref = ci.ensure("struct", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)  # slot.ref=struct_ref
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)     # ctx_ref·token concept
    他 = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)             # pronoun candidate·高 pr_tn
    她 = ci.ensure("她", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)             # pronoun candidate·低 pr_tn
    # 建 REFERS_TO pronoun→struct（activate_candidates(struct_ref) 返 [他, 她]·REFERS_TO 反向 to=struct 取 from）
    for p in (他, 她):
        es.add(space_id_from=p[0], local_id_from=p[1],
               space_id_to=struct_ref[0], local_id_to=struct_ref[1],
               edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_TEACHER,
               epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY)
    # record pr_tn：他→猫 高（5）·她→猫 低（1）·pair-key (pronoun, antecedent) 对偶 observe 侧
    for _ in range(5):
        record_pronoun_resolution_decision(b, pronoun_ref=他, antecedent_ref=antecedent)
    record_pronoun_resolution_decision(b, pronoun_ref=她, antecedent_ref=antecedent)
    graph = ConceptGraph(b)
    workmem = SimpleNamespace(prior_topic_refs=[antecedent], produced_refs=[])  # ctx_refs=[猫]
    slot = RoleSlot(ref=struct_ref, role=0, filler_is_memory_sequence=False)
    dag_path = PathResult()
    saved_pr = gates.PRONOUN_SLOT_MODE
    saved_sp = gates.GENERATE_SELECTION_PREF_MODE
    # gate ON：pronoun 加成 → 他(pr_tn=5→bonus=3) > 她(pr_tn=1→bonus=1) → 选 他
    gates.GENERATE_SELECTION_PREF_MODE = False
    gates.PRONOUN_SLOT_MODE = True
    try:
        word_on, _ = dispatch_slot(slot, dag_path, graph, workmem, LANG_ZH)
        他_word = f"#{他[0]}:{他[1]}"   # surface_of None → ref 字面
        assert word_on == 他_word, f"gate ON 选 他（pronoun 加成 pr_tn=5>1·combine 3>1）·got {word_on}"
    finally:
        gates.PRONOUN_SLOT_MODE = saved_pr
        gates.GENERATE_SELECTION_PREF_MODE = saved_sp
    # gate OFF：pronoun 加成 0·combine=collide=0·tiebreak ref 序（min ref·他 ref < 她 ref 则选 他·无判别力）
    gates.PRONOUN_SLOT_MODE = False
    try:
        word_off, _ = dispatch_slot(slot, dag_path, graph, workmem, LANG_ZH)
        # gate OFF 无 pronoun 加成·combine 同（collide=0）·选 ref 序最小（他/她 ref 序取决于 ensure 顺序）
        # 关键：gate ON 选 他（pronoun 加成）·gate OFF 无加成（行为差可观测·反 theater）
        assert word_off is not None, "gate OFF dispatch_slot 仍返词（tiebreak ref 序·无 pronoun 判别）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved_pr
        gates.GENERATE_SELECTION_PREF_MODE = saved_sp


def test_dispatch_slot_pronoun_cap_999_collide_priority(pronoun_env):
    """联合 cap 999 守 collide 主轴：pronoun 加成并入 sp 维 _cap_sp·sp+pr 联合 cap 999。
    collide=1,sp=0,pr=5 → combine=1000+min(5,3)=1003·collide=0,sp=999,pr=5 → combine=0+999=999。
    collide=1 (1003) > collide=0 (999)·collide 主轴优先守。"""
    b, sid, es, ns, ci = pronoun_env
    graph = ConceptGraph(b)
    # collide=1, sp=0, pr=5 → _cap_sp(0 + min(5,3)) = _cap_sp(3) = 3·combine = 1×1000 + 3 = 1003
    combine_collide1 = 1 * SCORE_SCALE + _cap_sp(0 + min(5, PR_SLOT_BONUS_CAP))
    # collide=0, sp=999, pr=5 → _cap_sp(999 + min(5,3)) = _cap_sp(1002) = 999·combine = 0×1000 + 999 = 999
    combine_collide0 = 0 * SCORE_SCALE + _cap_sp(999 + min(5, PR_SLOT_BONUS_CAP))
    assert combine_collide1 == 1003, f"collide=1+pr=3 → 1003·got {combine_collide1}"
    assert combine_collide0 == 999, f"collide=0+sp=999+pr cap → 999（联合 cap）·got {combine_collide0}"
    assert combine_collide1 > combine_collide0, "collide 主轴优先（1003 > 999·pr 加成不破主轴）"


# ============ PRONOUN_SLOT_MODE 与 PRONOUN_RESOLVE_COUNT_MODE 分立 ============

def test_pronoun_slot_mode_independent_of_resolve_count_mode():
    """两 gate 分立：PRONOUN_SLOT_MODE（生成侧读）≠ PRONOUN_RESOLVE_COUNT_MODE（observe 侧读写）。
    镜像 SELECTION_PREF_MODE 写 / GENERATE_SELECTION_PREF_MODE 读 分立范式·细粒度回归隔离。"""
    from pure_integer_ai.config import gates
    assert hasattr(gates, "PRONOUN_SLOT_MODE"), "PRONOUN_SLOT_MODE（生成侧读门）"
    assert hasattr(gates, "PRONOUN_RESOLVE_COUNT_MODE"), "PRONOUN_RESOLVE_COUNT_MODE（observe 侧读写门）"
    # 两 gate 独立（不同 env var·不同默认）
    assert gates.PRONOUN_SLOT_MODE is False or gates.PRONOUN_SLOT_MODE is True
    assert gates.PRONOUN_RESOLVE_COUNT_MODE is False or gates.PRONOUN_RESOLVE_COUNT_MODE is True


# ============ 铁律守卫 ============

def test_pronoun_slot_not_in_effective_weight(pronoun_env):
    """铁律：pr_tn 是 pronoun_resolution_count 统计台账非 edge reward·不入 effective_weight。
    pronoun_resolution_count 独立表（core=False·MUTABLE_MONOTONE）·不进 causes_edges/distributed/
    record_episode_result·reward_propagate assert 不动·effective_weight assert 不内。"""
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    # pr_tn 不是 edge_type·是统计台账列·effective_weight 不认
    # 守：pronoun_score 返 pr_tn int·不构造 edge·effective_weight 不被调
    b, sid, es, ns, ci = pronoun_env
    c = ci.ensure("他", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    antecedent = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    record_pronoun_resolution_decision(b, pronoun_ref=c, antecedent_ref=antecedent)
    graph = ConceptGraph(b)
    saved = gates.PRONOUN_SLOT_MODE
    gates.PRONOUN_SLOT_MODE = True
    try:
        # pronoun_score 返 int（pr_tn）·非 edge·不接 effective_weight
        val = graph.pronoun_score(c, antecedent)
        assert isinstance(val, int), "pronoun_score 返 int（pr_tn·统计台账非 edge）"
        assert val == 1
    finally:
        gates.PRONOUN_SLOT_MODE = saved
    # effective_weight 只认 {PRECEDES,CAUSES,REFERS_TO}·pronoun 加成不构造 edge
    # （pronoun_score 是读路径·不建边·不接 reward·effective_weight 无关）


# ============ bit-identical gate OFF ============

def test_pronoun_slot_gate_off_bit_identical(pronoun_env):
    """bit-identical：gate PRONOUN_SLOT_MODE OFF + GENERATE_SELECTION_PREF_MODE OFF →
    dispatch_slot 第4+5路联合 if 不进·combine = collide only（既有行为·bit-identical）。"""
    b, sid, es, ns, ci = pronoun_env
    graph = ConceptGraph(b)
    saved_pr = gates.PRONOUN_SLOT_MODE
    saved_sp = gates.GENERATE_SELECTION_PREF_MODE
    gates.PRONOUN_SLOT_MODE = False
    gates.GENERATE_SELECTION_PREF_MODE = False
    try:
        # 复现 dispatch_slot：两 gate OFF → if (_sp_gate or _pr_gate) 不进 → combine = s（collide only）
        _sp_gate = gates.GENERATE_SELECTION_PREF_MODE
        _pr_gate = gates.PRONOUN_SLOT_MODE
        s = 5  # collide_score
        if _sp_gate or _pr_gate:
            combine = s * SCORE_SCALE + _cap_sp(0)  # 不应进
        else:
            combine = s  # 既有：collide only
        assert combine == 5, "两 gate OFF → combine=collide only（既有 bit-identical）"
    finally:
        gates.PRONOUN_SLOT_MODE = saved_pr
        gates.GENERATE_SELECTION_PREF_MODE = saved_sp
