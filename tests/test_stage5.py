"""Stage 5 验收门测试：卷三结果建模（reward=ΠG·ΣwJ·防塌三柱·假收敛识别）。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 5 验收门）：
  - reward=ΠG·ΣwJ 合成（模块3）
  - G_meta 5字段 {G4,G2p,G3a,G3b,G5} veto 写回（D1 跨卷·early return 前写）
  - 防塌三柱缺一即塌（模块4·pillar1/2/3）
  - 假收敛识别（sn/tn→1+PR方差→0+负通路=0=塌非收敛·模块5）
  - 三空间三时间尺度（模块6·快/中/慢环）
  - 路径填槽+逐槽分派（模块1/2·target_lang 偏好·DEF_REPLAY 血统）
  - 确定性 bit-identical
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO, EDGE_PROPERTY, EDGE_COOCCURS,
)
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, PathData, PathResult, Step, Episode, GMeta,
    OutputResult, OutputPart, RoleSlot, JudgeWeights, CollapseReport,
    ConceptRef, EdgeRef,
    LINEAGE_CONCEPT_FILL, LINEAGE_DEF_REPLAY,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, REWARD_DEAD_END, G_META_DEAD_END,
    INTENT_QUESTION, J3_CAUSES_WEIGHT, J3_PRECEDES_WEIGHT,
    LANG_ZH, LANG_EN, LANG_NONE, DOMAIN_TEXT, DOMAIN_CODE, DOMAIN_MATH,
    WEANING_PRE, WEANING_POST,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot, modality_serialize
from pure_integer_ai.cognition.result.generate import (
    generate_output, structure_units, _path_acyclic, LAYER_UNIT_CAP,
)
from pure_integer_ai.cognition.result.judge import (
    judge, check_closure, slot_fill_rate, path_strength_weighted,
    counterfactual_value_check, self_proof_check, _ARITH_DOMAINS,
)
from pure_integer_ai.cognition.result.anti_collapse import (
    anti_collapse_verify, integer_variance, inject_seeded_exploration,
    linkage_four_conditions_hold, THETA_VARIANCE,
)
from pure_integer_ai.cognition.result.convergence import (
    convergence_check, EpisodeHistory, StatRecord, CONVERGENCE_WINDOW,
)
from pure_integer_ai.cognition.result.tri_space import (
    tri_space_coordination, query_memory_ranked, query_negative_memories,
)
from pure_integer_ai.cognition.process.episode import episode_loop
from pure_integer_ai.crosscut.integer.rational import make, ZERO, ONE
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ci
    b.close()


def _ref(sid, lid):
    return (sid, lid)


def _edge(es, sid, frm, to, et, *, strength=1, sn=0, tn=0, subtype=None,
          source=SOURCE_BARE_TEXT):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=source, tier=TIER_PRIMARY,
           subtype=subtype, sn=sn, tn=tn)


def _graph(b, *, surface_map=None, lang_map=None):
    return ConceptGraph(b,
                       surface_of=(lambda r: surface_map.get(r)) if surface_map else None,
                       lang_of=(lambda r: lang_map.get(r)) if lang_map else None)


def _dag(sid, *, sink=None, edges=None, struct_unit_refs=None,
         topo_layers=None, terminal=TERMINAL_REACHED_SINK):
    return PathResult(
        path=PathData(edges=edges or [], struct_unit_refs=struct_unit_refs or []),
        terminal=terminal, sink=sink,
        topo_layers=topo_layers or [],
        convergence={}, source=None,
    )


# ============ 模块1 generate（路径填槽+回放） ============

def test_generate_single_pass_fills_slots(core):
    b, sid, es, ci = core
    U = ci.ensure("unit", space_id=sid, tier=TIER_PRIMARY)
    W = ci.ensure("word", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [101, 102])           # role_seq 两槽
    _edge(es, sid, W[1], U[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={U: "unit", W: "word"})
    wm = WorkMemory()
    dag = _dag(sid, sink=U, struct_unit_refs=[U], topo_layers=[[U]])
    out = generate_output(dag, g, wm, LANG_ZH)
    assert len(out.parts) == 1
    assert len(out.parts[0].words) == 2          # 两槽都填
    assert out.reached_sink is True              # sink=U 在产出单元
    assert all(src == LINEAGE_CONCEPT_FILL
               for src in out.lineage.values())
    assert U in wm.produced_refs                  # carry


def test_generate_dag_acyclic_assert(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    g = _graph(b, surface_map={U: "u"})
    # 构造含环的 path.edges（自环）
    cyclic = _dag(sid, sink=U, edges=[(sid, 1, sid, 1, EDGE_PRECEDES)],
                  topo_layers=[[U]])
    with pytest.raises(AssertionError):
        generate_output(cyclic, g, WorkMemory(), LANG_ZH)


def test_path_acyclic_detection():
    sid = 1
    # 无环：1→2→3
    dag = _dag(sid, edges=[(sid, 1, sid, 2, EDGE_PRECEDES),
                           (sid, 2, sid, 3, EDGE_PRECEDES)])
    assert _path_acyclic(dag)
    # 有环：1→2→1
    dag_cycle = _dag(sid, edges=[(sid, 1, sid, 2, EDGE_PRECEDES),
                                 (sid, 2, sid, 1, EDGE_PRECEDES)])
    assert not _path_acyclic(dag_cycle)


# ============ 模块2 dispatch_slot（逐槽分派） ============

def test_dispatch_slot_concept_fill(core):
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    W = ci.ensure("word", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={C: "c", W: "w"})
    word, src = dispatch_slot(RoleSlot(ref=C, role=1), _dag(sid), g,
                              WorkMemory(), LANG_ZH)
    assert src == LINEAGE_CONCEPT_FILL
    assert word in ("c", "w")


def test_dispatch_slot_memory_replay(core):
    b, sid, es, ci = core
    M = ci.ensure("mem", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("a", space_id=sid, tier=TIER_PRIMARY)
    Bb = ci.ensure("b", space_id=sid, tier=TIER_PRIMARY)
    # 记忆序列：def_array ref_space_id!=0
    b.insert("def_array", {"space_id": sid, "local_id": M[1],
                           "order_index": 0, "ref_space_id": sid,
                           "ref_local_id": A[1]})
    b.insert("def_array", {"space_id": sid, "local_id": M[1],
                           "order_index": 1, "ref_space_id": sid,
                           "ref_local_id": Bb[1]})
    g = _graph(b, surface_map={A: "a", Bb: "b"})
    word, src = dispatch_slot(RoleSlot(ref=M, filler_is_memory_sequence=True),
                              _dag(sid), g, WorkMemory(), LANG_ZH)
    assert src == LINEAGE_DEF_REPLAY
    assert word == "ab"                          # 序列回放直出


def test_dispatch_slot_target_lang_preference(core):
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    WZH = ci.ensure("zhword", space_id=sid, tier=TIER_PRIMARY)
    WEN = ci.ensure("enword", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, WZH[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, WEN[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={C: "c", WZH: "中", WEN: "en"},
               lang_map={C: LANG_ZH, WZH: LANG_ZH, WEN: LANG_EN})
    # target_lang=ZH → 同 lang 优先（选 WZH "中"·非 EN "en"）
    word, src = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, WorkMemory(), LANG_ZH)
    assert word == "中"
    assert src == LINEAGE_CONCEPT_FILL


def test_dispatch_slot_target_lang_fallback_cross_lang(core):
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    WEN = ci.ensure("enword", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, WEN[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={C: "c", WEN: "en"},
               lang_map={C: LANG_ZH, WEN: LANG_EN})
    # 无同 lang 候选→回退跨 lang（防空槽）
    word, src = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, WorkMemory(), LANG_ZH)
    assert src == LINEAGE_CONCEPT_FILL
    assert word in ("c", "en")


# ============ S4 决断 2 生成侧 selection_pref pair-rate（dispatch_slot sel_pref 维·gate 守） ============

def test_dispatch_slot_sel_pref_off_zero_call_bit_identical(core):
    """片2 bit-identical 守：gate OFF → graph.selection_pref_score 零调（if 外短路·守 984 测）。

    OFF 时 dispatch_slot scored 不重算·sel_pref 维整块跳过·零 IO·ancestor_map_cache 不 build。
    反 theater：spy 计调用·OFF 须 0（若 if 内默认值而非外短路·会误调）。
    """
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    W1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    W2 = ci.ensure("w2", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W1[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, W2[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={C: "c", W1: "w1", W2: "w2"})
    call_count = {"n": 0}
    real_sp = g.selection_pref_score
    def spy_sp(c, ctx_refs):
        call_count["n"] += 1
        return real_sp(c, ctx_refs)
    g.selection_pref_score = spy_sp   # instance-level spy

    saved = gates.GENERATE_SELECTION_PREF_MODE
    gates.GENERATE_SELECTION_PREF_MODE = False
    try:
        word, src = dispatch_slot(RoleSlot(ref=C, role=1), _dag(sid), g,
                                  WorkMemory(), LANG_ZH)
    finally:
        gates.GENERATE_SELECTION_PREF_MODE = saved
    assert call_count["n"] == 0, \
        f"gate OFF → selection_pref_score 零调（if 外短路 bit-identical）·got {call_count['n']}"
    assert src == LINEAGE_CONCEPT_FILL
    assert word in ("c", "w1", "w2"), "OFF 行为不变（bit-identical·同既有测）"


def test_dispatch_slot_sel_pref_on_calls_score(core):
    """片2 gate ON：selection_pref_score 真调（ON 路径活·非 theater）。

    ON 时 dispatch_slot scored 重算（复用 s + sel_pref 亚主轴）·graph.selection_pref_score 被调。
    （OFF/ON 选词差异的反 theater e2e 见 test_stage_s4_selection_pref_dock.py 片3）
    """
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    W1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    W2 = ci.ensure("w2", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W1[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, W2[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={C: "c", W1: "w1", W2: "w2"})
    call_count = {"n": 0}
    real_sp = g.selection_pref_score
    def spy_sp(c, ctx_refs):
        call_count["n"] += 1
        return real_sp(c, ctx_refs)
    g.selection_pref_score = spy_sp

    saved = gates.GENERATE_SELECTION_PREF_MODE
    gates.GENERATE_SELECTION_PREF_MODE = True
    try:
        word, src = dispatch_slot(RoleSlot(ref=C, role=1), _dag(sid), g,
                                  WorkMemory(), LANG_ZH)
    finally:
        gates.GENERATE_SELECTION_PREF_MODE = saved
    assert call_count["n"] >= 1, \
        f"gate ON → selection_pref_score 须被调（ON 路径活·非 theater）·got {call_count['n']}"


def test_collide_score_picks_cooccurring_candidate(core):
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    W1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    W2 = ci.ensure("w2", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W1[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, W2[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    # W2 与 ctx 概念共现·W1 不共现 → collide_score 选 W2
    CTX = ci.ensure("ctx", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W2[1], CTX[1], EDGE_COOCCURS)
    g = _graph(b, surface_map={C: "c", W1: "w1", W2: "w2"})
    wm = WorkMemory()
    wm.prior_topic_refs = [CTX]
    word, _ = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, wm, LANG_NONE)
    assert word == "w2"                          # 共现选词


# ============ 模块3 judge（ΠG·ΣwJ·G_meta 5字段） ============

def _out(parts, reached_sink=True):
    return OutputResult(parts=parts, reached_sink=reached_sink)


def test_judge_synthesis_all_g_pass(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1, 2])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)   # CAUSES 锚
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)],
               struct_unit_refs=[U])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT,
                       intent=IntentType(type=INTENT_QUESTION, sink=U,
                                          is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a", "b"])], reached_sink=True)
    g = _graph(b)
    reward, gm = judge(out, dag, inp, g, JudgeWeights(1, 1, 1), WorkMemory())
    assert reward > 0                             # ΠG=1·ΣwJ>0
    assert gm.vetoed is False                     # 无 veto


def test_judge_g4_veto_empty_slot(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    dag = _dag(sid, sink=U)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(sink=U))
    out = _out([OutputPart(U, [])], reached_sink=True)   # 空槽=未绑定
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(), WorkMemory())
    assert reward == 0
    assert gm.G4 is True                          # 早退前写 vetoed
    assert gm.G2p is False                        # 后续未判


def test_judge_g2p_veto_sink_not_reached(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    dag = _dag(sid, sink=U)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(sink=U))
    out = _out([OutputPart(U, ["a"])], reached_sink=False)   # sink 未达
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(), WorkMemory())
    assert reward == 0
    assert gm.G2p is True
    assert gm.G4 is False                         # J4 通过


def test_judge_g3a_veto_no_causes_anchor(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    # 因果推理意图但 path 无 CAUSES 锚
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_PRECEDES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(sink=U, is_causal_reasoning=True))
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(), WorkMemory())
    assert reward == 0
    assert gm.G3a is True


def test_judge_h3_structural_reasoning_skips_j3(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    # 结构序推理意图·纯 PRECEDES 路径·J3 归零 G3a=1 跳过不罚
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_PRECEDES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(sink=U,
                                          is_structural_sequence_reasoning=True))
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1),
                       WorkMemory())
    assert reward > 0                             # 不罚·J3=0 但 J1/J2>0
    assert gm.G3a is False
    assert gm.vetoed is False


def test_judge_g3b_veto_value_conflict(core):
    b, sid, es, ci = core
    # G1+#774 选 b：G3b 改全局扫命题节点（ATTR_PROPOSITION）·非旧 part.unit PROPERTY 扫（theater·fork §一）。
    # U 须是命题节点（ATTR_PROPOSITION 标记）·其两 PROPERTY 出边指向不同值=结构值冲突。
    from pure_integer_ai.storage.composes_attr import (
        register_composes_attr, record_composes_attr, ATTR_PROPOSITION,
    )
    register_composes_attr(b)
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    record_composes_attr(b, ref=U, kind=ATTR_PROPOSITION, int_a=0, int_b=0)   # U 标记为命题节点
    _edge(es, sid, U[1], 10, EDGE_PROPERTY)
    _edge(es, sid, U[1], 11, EDGE_PROPERTY)
    dag = _dag(sid, sink=U)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(sink=U, has_value_claim=True))
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(), WorkMemory())
    assert reward == 0
    assert gm.G3b is True                         # R4 写回·early return 写 vetoed


def test_judge_g5_veto_self_proof_fail(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_MATH,
                       intent=IntentType(sink=U, is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    # 自证机 fail（Mode A 教师 ground-truth 判错）
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1),
                       WorkMemory(), self_proof_fn=lambda o, p, gg: 0)
    assert reward == 0
    assert gm.G5 is True


def test_judge_g5_self_proof_none_miss_vetoes(core):
    """stub #3：self_proof_fn 返 None（miss/退场）→ G5 veto（防脏 reward·区别于无机制 pass=1）。"""
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_MATH,
                       intent=IntentType(sink=U, is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    # miss（self_proof_fn 返 None）→ veto（防占位 pass 产脏 reward·E4 红线）
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1),
                       WorkMemory(), self_proof_fn=lambda o, p, gg: None)
    assert reward == 0
    assert gm.G5 is True


def test_judge_g5_no_mechanism_passes(core):
    """stub #3：self_proof_fn=None（无机制/TEACHER_MODE OFF/Mode B defer）→ G5 pass=1 占位（bit-identical）。"""
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_MATH,
                       intent=IntentType(sink=U, is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    # 无机制 → G5 不 veto（pass=1 占位）·reward 由结构门 J 驱动 > 0
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1), WorkMemory())
    assert gm.G5 is False
    assert reward > 0


def test_self_proof_check_3state_pre_miss_vetoes():
    """R1：fn 返 None + WEANING_PRE（教师 miss·stub#3）→ veto=0（防脏 reward·E4 红线）。"""
    fn = lambda o, p, gg: None
    assert self_proof_check(None, None, None,
                            weaning_phase=WEANING_PRE, self_proof_fn=fn) == 0


def test_self_proof_check_3state_post_vacates():
    """R1 核心：fn 返 None + WEANING_POST（无子图/StepLimit/单路径）→ vacate=1（G5 非承重·不奖死循环不误杀）。"""
    fn = lambda o, p, gg: None    # 模拟 VM-proof fn 捕获 StepLimitExceeded→返 None
    assert self_proof_check(None, None, None,
                            weaning_phase=WEANING_POST, self_proof_fn=fn) == 1


def test_self_proof_check_3state_int_passthrough():
    """R1：fn 返 int 直通（1=verified / 0=mismatch 硬否决）·两 phase 同。"""
    assert self_proof_check(None, None, None, weaning_phase=WEANING_PRE,
                            self_proof_fn=lambda o, p, gg: 1) == 1
    assert self_proof_check(None, None, None, weaning_phase=WEANING_POST,
                            self_proof_fn=lambda o, p, gg: 0) == 0


def test_judge_g5_none_post_vacates(core):
    """R1 集成：fn 返 None + WEANING_POST → G5 vacate（不 veto）·结构 reward 流入（非脏 reward）。"""
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_MATH, weaning_phase=WEANING_POST,
                       intent=IntentType(sink=U, is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    # POST + fn 返 None（模拟 StepLimit/单路径）→ vacate：G5 不 veto·reward>0
    reward, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1), WorkMemory(),
                       self_proof_fn=lambda o, p, gg: None)
    assert gm.G5 is False
    assert reward > 0


def test_judge_g_meta_five_fields_all_vetoable(core):
    """D1 跨卷最严重：5字段各自 early-return 写 vetoed·Episode 不漏记。"""
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    # G5 域 + value_claim + causal + CAUSES 锚·全设齐后单测 G5 veto
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_CODE,
                       intent=IntentType(sink=U, is_causal_reasoning=True,
                                          has_value_claim=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    _, gm = judge(out, dag, inp, _graph(b), JudgeWeights(1, 1, 1), WorkMemory(),
                  self_proof_fn=lambda o, p, gg: 0)
    assert gm.G5 is True and gm.vetoed is True


def test_judge_j3path_weight_ratio(core):
    b, sid, es, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    _edge(es, sid, 3, 4, EDGE_PRECEDES)
    g = _graph(b)
    dag_c = _dag(sid, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    dag_p = _dag(sid, edges=[(sid, 3, sid, 4, EDGE_PRECEDES)])
    sc = path_strength_weighted(dag_c, g)
    sp = path_strength_weighted(dag_p, g)
    assert sc == J3_CAUSES_WEIGHT * 1            # 10
    assert sp == J3_PRECEDES_WEIGHT * 1           # 1
    assert sc == 10 * sp                          # 1:10


def test_slot_fill_rate():
    out = _out([OutputPart((1, 1), ["a", "b"]), OutputPart((1, 2), ["c"])])
    assert slot_fill_rate(out, None) == 1000      # 全槽填满
    out2 = _out([OutputPart((1, 1), ["a", ""])])
    assert slot_fill_rate(out2, None) == 500


# ============ 模块4 anti_collapse（防塌三柱） ============

def _episode(*, reward=1, terminal=TERMINAL_REACHED_SINK, pr_vector=None,
             g4=False, g2p=False, g3a=False, g3b=False, g5=False,
             veto_count=0, dead_end=0):
    return Episode(
        episode_id=0, run_id=1, reward=reward, terminal=terminal,
        pr_vector=pr_vector if pr_vector is not None else {1: make(1, 2)},
        judge_G4_active=g4, judge_G2p_active=g2p, judge_G3a_active=g3a,
        judge_G3b_active=g3b, judge_G5_active=g5,
        judge_veto_count=veto_count, dead_end_count=dead_end,
        vetoed=g4 or g2p or g3a or g3b or g5,
    )


def test_anti_collapse_pillar1_judge_active():
    ep = _episode(g4=True, veto_count=1)
    rep = anti_collapse_verify(ep)
    assert rep.pillar1_ok is True               # judge 在工作非自媚


def test_anti_collapse_pillar2_neg_pathway():
    ep = _episode(reward=0, veto_count=1)        # judge veto
    rep = anti_collapse_verify(ep)
    assert rep.pillar2_ok is True
    assert rep.failure_count == 1
    ep2 = _episode(reward=REWARD_DEAD_END, terminal=TERMINAL_DEAD_END,
                   dead_end=1)
    rep2 = anti_collapse_verify(ep2)
    assert rep2.pillar2_ok is True
    assert rep2.neg_reward_count == 1            # 负值只来自步进死路


def test_anti_collapse_pillar3_low_variance_injects(core):
    b, sid, es, ci = core
    # 低方差 PR 向量（趋平）→ 注入 seeded 探索
    _edge(es, sid, 1, 2, EDGE_PRECEDES)
    _edge(es, sid, 2, 3, EDGE_PRECEDES)
    from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper
    edges = b.select("edge")
    prw = A3PRWrapper.build(edges)
    prw.solve([(sid, 1)])
    e_set = {(sid, 1)}
    old = gates.EXPLORATION_MODE
    gates.EXPLORATION_MODE = True
    try:
        ep = _episode(pr_vector={n: prw.seed_rank(n) for n in prw.matrix.nodes},
                      veto_count=1)
        rep = anti_collapse_verify(ep, pr_wrapper=prw, e_set=e_set)
        assert rep.pillar3_ok is True
        # seeded 探索注入了新种子（e_set 扩张·确定性·非墙钟随机）
        assert len(e_set) >= 2
    finally:
        gates.EXPLORATION_MODE = old


def test_integer_variance_zero_and_nonzero():
    # 全相同→方差 0（趋平）
    assert integer_variance({1: make(1, 2), 2: make(1, 2)}) == 0
    # 不同→方差 > 0
    assert integer_variance({1: make(1, 4), 2: make(3, 4)}) > 0
    assert integer_variance({}) == 0


def test_inject_seeded_exploration_deterministic(core):
    b, sid, es, ci = core
    _edge(es, sid, 1, 2, EDGE_PRECEDES)
    _edge(es, sid, 2, 3, EDGE_PRECEDES)
    from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper
    # 两个**独立** prw（同初始态）验同 seed → 同注入节点（determinism·bit-identical·非墙钟随机）。
    # 须独立实例：inject_seeded_exploration 的 add_seed **mutate** prw（PR 矩阵变）·共享实例则第二次
    # 调用读到第一次注入后的态 → 非确定性测试污染（#995 修前 buggy 元组序侥幸对此 mutate 稳定·掩盖缺陷）。
    prw1 = A3PRWrapper.build(b.select("edge"))
    prw1.solve([(sid, 1)])
    prw2 = A3PRWrapper.build(b.select("edge"))
    prw2.solve([(sid, 1)])
    e1 = {(sid, 1)}
    e2 = {(sid, 1)}
    n1 = inject_seeded_exploration(prw1, e1, seed=12345)
    n2 = inject_seeded_exploration(prw2, e2, seed=12345)
    assert n1 == n2
    assert n1 in e1 and n1 in e2


def test_inject_seeded_exploration_picks_lowest_value_not_tuple_order():
    """#995：候选选 min 须**值序**（cross_compare）非 (num,den) 元组序。

    构造两候选 seed_rank：A=make(1,2)=0.5（元组(1,2)）/ B=make(2,5)=0.4（元组(2,5)）。
    值序 0.4<0.5 → 最低 x_c = B（正确）。但 (num,den) 元组序 (1,2)<(2,5)（因 1<2）→ min 误选 A（高值 0.5）。
    旧 rank_key=(x.num,x.den,tie) 选 A（错·违 docstring"x_c 最低"）·修后 cross_compare 选 B（对）。
    用最小 stand-in 实现 inject_seeded_exploration 消费的契约（matrix.nodes/seed_rank/add_seed）·
    真 A3PRWrapper 无法稳定产出此精确分歧分数对（PR solve 依赖拓扑权重）·controlled fixture 验选序正确性。
    """
    class _Matrix:
        def __init__(self, nodes):
            self.nodes = list(nodes)
    class _PRW:
        def __init__(self, nodes, ranks):
            self.matrix = _Matrix(nodes)
            self._ranks = ranks
        def seed_rank(self, n):
            return self._ranks[n]
        def add_seed(self, n):
            pass
    node_a, node_b = (1, 2), (1, 3)
    # A=0.5 元组(1,2)·B=0.4 元组(2,5)·值序 B<A 但元组序 A<B（bug 暴露点）
    prw = _PRW([node_a, node_b], {node_a: make(1, 2), node_b: make(2, 5)})
    e = {(1, 1)}   # 两候选均不在 e → 都进 candidates
    target = inject_seeded_exploration(prw, e, seed=12345)
    assert target == node_b   # 最低值 x_c=0.4（B）非 0.5（A）·buggy 版此处返 node_a → 断言失败


def test_linkage_four_conditions_hold():
    # pr_vector 非空 + reward 符号契约 → 闭合
    ep = _episode(reward=1, terminal=TERMINAL_REACHED_SINK,
                  pr_vector={1: make(1, 2)})
    assert linkage_four_conditions_hold(ep) is True
    # 符号契约破：DEAD_END 但 reward>=0 → 未闭合
    ep_bad = _episode(reward=1, terminal=TERMINAL_DEAD_END,
                      pr_vector={1: make(1, 2)})
    assert linkage_four_conditions_hold(ep_bad) is False
    # pr_vector 空 → 未闭合
    ep_empty = _episode(reward=1, pr_vector={})
    assert linkage_four_conditions_hold(ep_empty) is False


# ============ 模块5 convergence（假收敛识别） ============

def test_convergence_collapse_signal_false_convergence():
    h = EpisodeHistory()
    # sn/tn→1均匀 + PR方差→0 + 负通路failure=0 = 塌信号
    h.append(StatRecord(failure_count=0, pr_variance=0,
                        sn_tn_ratio_variance=0,
                        conduction_rate=500, promote_rate=100))
    rep = convergence_check(h)
    assert rep.collapse_signal is True
    assert rep.real_convergence is False         # 塌信号=假收敛


def test_convergence_neg_pathway_inactive_false_convergence():
    h = EpisodeHistory()
    # 负通路不活跃（无 failure）但非塌信号（比率方差高）→ 假收敛
    h.append(StatRecord(failure_count=0, pr_variance=500,
                        sn_tn_ratio_variance=500,
                        conduction_rate=500, promote_rate=100))
    rep = convergence_check(h)
    assert rep.collapse_signal is False
    assert rep.real_convergence is False         # 负通路不活跃=假收敛


def test_convergence_real_with_neg_pathway_active():
    h = EpisodeHistory()
    # 负通路活跃 + 平台 + 比率方差低（非→1均匀）→ steady + real
    h.append(StatRecord(failure_count=2, pr_variance=500,
                        sn_tn_ratio_variance=50,
                        conduction_rate=500, promote_rate=100))
    h.append(StatRecord(failure_count=1, pr_variance=500,
                        sn_tn_ratio_variance=50,
                        conduction_rate=500, promote_rate=100))
    rep = convergence_check(h)
    assert rep.collapse_signal is False
    assert h.neg_reward_count_recent == 0        # 无死路负 reward（诊断）
    assert rep.steady_state is True
    assert rep.real_convergence is True


def test_convergence_window_fifo():
    h = EpisodeHistory()
    h.window = CONVERGENCE_WINDOW
    for i in range(CONVERGENCE_WINDOW + 5):
        h.append(StatRecord(failure_count=1, pr_variance=100,
                            sn_tn_ratio_variance=100))
    assert len(h) == CONVERGENCE_WINDOW           # FIFO 淘汰最旧


def test_neg_pathway_active_from_single_source_unit():
    """B2：neg_pathway_active_from 单点源正确性（D2 断奶用·formal_train 调此非内联）。

    failure_count=judge_veto+dead_end>0 → True·全 0 → False·空 → False。
    与 convergence_check.neg_pathway_active（failure_count_recent>0）同源 M7 同口径。
    """
    from pure_integer_ai.cognition.result.convergence import neg_pathway_active_from
    from pure_integer_ai.cognition.shared.types import Episode

    def _ep(jv=0, de=0):
        e = Episode()
        e.judge_veto_count = jv
        e.dead_end_count = de
        return e

    assert neg_pathway_active_from([]) is False                  # 空→无负通路证据
    assert neg_pathway_active_from([_ep()]) is False             # 全 0→False
    assert neg_pathway_active_from([_ep(jv=1)]) is True          # judge_veto→True
    assert neg_pathway_active_from([_ep(de=2)]) is True          # dead_end→True
    assert neg_pathway_active_from([_ep(), _ep(jv=1)]) is True   # 任一>0→True


# ============ 模块6 tri_space（三空间协同） ============

@pytest.fixture
def mem_setup():
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    yield b, ms
    b.close()


def test_tri_space_positive_reward_does_not_seed_next_episode(mem_setup):
    """M-00：正 reward 不得把旧 memory_item 注入下一 episode。"""
    b, ms = mem_setup
    # #728 纠偏 A：info_ref concept ref（非 memory_ref 行 id）·ms.put 传 info_ref_space/info_ref_id
    ms.put(10, content_hash=100, session_id=None,
           info_ref_space=ms.space_id, info_ref_id=100)
    ms.put(11, content_hash=101, session_id=None,
           info_ref_space=ms.space_id, info_ref_id=101)
    ms.record_use(10, success=True)               # success_count=2
    ms.record_use(11, success=False)              # 负经验
    ep = _episode(reward=1)
    wm = WorkMemory()
    wm.replay_candidates.append((ms.space_id, 999))
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.replay_candidates == []


def test_tri_space_negative_reward_does_not_control_next_recall(mem_setup):
    """M-00：负 reward 也不得产生跨 episode 排除指令。"""
    b, ms = mem_setup
    ms.put(10, content_hash=100, info_ref_space=ms.space_id, info_ref_id=100)
    ms.put(11, content_hash=101, info_ref_space=ms.space_id, info_ref_id=101)
    ms.record_use(10, success=False)              # success_rate=0<1/2=负经验
    ms.record_use(11, success=True)               # 正经验
    ep = _episode(reward=REWARD_DEAD_END, terminal=TERMINAL_DEAD_END)
    wm = WorkMemory()
    wm.exclude_refs.add((ms.space_id, 999))
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.exclude_refs == set()


def test_tri_space_no_memory_space_defer():
    ep = _episode(reward=1)
    wm = WorkMemory()
    tri_space_coordination(ep, workmem=wm, memory_space=None)
    assert wm.replay_candidates == []


def test_query_memory_ranked_orders_by_success(mem_setup):
    b, ms = mem_setup
    ms.put(1, content_hash=1, info_ref_space=ms.space_id, info_ref_id=1)
    ms.put(2, content_hash=2, info_ref_space=ms.space_id, info_ref_id=2)
    ms.record_use(1, success=True)                # rate=1
    ms.record_use(2, success=False)               # rate=0
    ranked = query_memory_ranked(ms, WorkMemory())
    assert ranked[0][0] == (ms.space_id, 1)       # 高 rate 在前·info_ref concept ref


# ============ 验收门 + 端到端 + 确定性 ============

def test_end_to_end_episode_loop_reached_sink(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag_edges = b.select("edge")
    g = _graph(b, surface_map={U: "u", (sid, 1): "x", (sid, 2): "y"})
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT,
                       intent=IntentType(type=INTENT_QUESTION, sink=U,
                                          is_causal_reasoning=True),
                       key_skeleton=[U])
    wm = WorkMemory()
    gen_fn = lambda pr, w, i: generate_output(pr, g, w, LANG_ZH)
    jdg_fn = lambda o, pr, i, w: judge(o, pr, i, g, JudgeWeights(1, 1, 1), w)
    out, ep = episode_loop(inp, dag_edges, [U], wm,
                           IntentType(type=INTENT_QUESTION, sink=U,
                                      is_causal_reasoning=True),
                           generate_fn=gen_fn, judge_fn=jdg_fn,
                           edge_store=es, backend=b)
    assert ep.terminal == TERMINAL_REACHED_SINK
    assert ep.reward >= 0
    assert out is not None and len(out.parts) >= 1


def test_end_to_end_episode_loop_dead_end_negative_reward(core):
    b, sid, es, ci = core
    # 无后继孤立节点→步进死路→reward<0
    U = ci.ensure("iso", space_id=sid, tier=TIER_PRIMARY)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       intent=IntentType(type=INTENT_QUESTION, sink=(sid, 999)))
    wm = WorkMemory()
    out, ep = episode_loop(inp, [], [U], wm,
                           IntentType(type=INTENT_QUESTION, sink=(sid, 999)),
                           generate_fn=None, judge_fn=None,
                           edge_store=es, backend=b)
    assert ep.terminal == TERMINAL_DEAD_END
    assert ep.reward == REWARD_DEAD_END
    assert ep.reward < 0


def test_anti_collapse_three_pillars_end_to_end():
    """防塌三柱缺一即塌：三柱全 active = 不塌（柱③需方差够·趋平须注入·stub#1 falsifiable）。"""
    # 方差够（不同值）→ 柱③ dormant OK·柱① judge active·柱② veto 负通路
    ep = _episode(reward=0, g4=True, veto_count=1,
                  pr_vector={1: make(1, 2), 2: make(3, 4)})  # 不同值→方差够
    rep = anti_collapse_verify(ep)
    assert rep.pillar1_ok and rep.pillar2_ok and rep.pillar3_ok


def test_anti_collapse_pillar3_low_variance_no_injection_fails():
    """柱③ falsifiable：方差趋平 + 无注入（EXPLORATION_MODE OFF/无 pr_wrapper）→ 柱③ 失守 False。"""
    ep = _episode(reward=0, g4=True, veto_count=1,
                  pr_vector={1: make(1, 2), 2: make(1, 2)})  # 方差 0 趋平
    old = gates.EXPLORATION_MODE
    gates.EXPLORATION_MODE = False   # 默认 OFF·不注入
    try:
        rep = anti_collapse_verify(ep)   # 无 pr_wrapper·无注入
        assert rep.pillar1_ok and rep.pillar2_ok   # 柱①② 仍 active
        assert rep.pillar3_ok is False   # 趋平未注入=柱③ 失守（stub#1 theater 修·可证伪）
    finally:
        gates.EXPLORATION_MODE = old


def test_determinism_judge_bit_identical(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1, 2])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)],
               struct_unit_refs=[U])
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT,
                       intent=IntentType(sink=U, is_causal_reasoning=True),
                       key_skeleton=[U])
    out = _out([OutputPart(U, ["a", "b"])], reached_sink=True)
    g = _graph(b)
    r1, gm1 = judge(out, dag, inp, g, JudgeWeights(1, 1, 1), WorkMemory())
    r2, gm2 = judge(out, dag, inp, g, JudgeWeights(1, 1, 1), WorkMemory())
    assert r1 == r2
    assert (gm1.G4, gm1.G2p, gm1.G3a, gm1.G3b, gm1.G5) == \
           (gm2.G4, gm2.G2p, gm2.G3a, gm2.G3b, gm2.G5)


def test_determinism_generate_bit_identical(core):
    b, sid, es, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    W = ci.ensure("w", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, W[1], U[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    g = _graph(b, surface_map={U: "u", W: "w"})
    dag = _dag(sid, sink=U, struct_unit_refs=[U], topo_layers=[[U]])
    o1 = generate_output(dag, g, WorkMemory(), LANG_ZH)
    o2 = generate_output(dag, g, WorkMemory(), LANG_ZH)
    assert o1.parts == o2.parts
    assert o1.lineage == o2.lineage
    assert o1.reached_sink == o2.reached_sink
