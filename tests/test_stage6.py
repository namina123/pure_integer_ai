"""Stage 6 验收门测试：录放层 + 训练编排（断奶前教师合法·断奶后退场·oracle·dump 续训）。

覆盖（doc/重来_落地规划与实施顺序.md §二 Stage 6 验收门）：
  - verify_teacher_boundary 白黑词汇表机械核查（拒越界·§9 A2）
  - RecordableLLMTeacher MODE_RECORD/REPLAY/OFF·miss→None 无 fallback（E4）·幂等·verify_boundary 拒写
  - weaning 双曲线趋势 D1·window_rounds=4 runs·非布尔阈值·下限防假断奶
  - oracle B1-B4 占位校验 + H2 calibrate_weights 最大化 agreement + conduction_rate
  - promote 三重（频次/reward/定义）SHADOW→PRIMARY tier flip·MUTABLE_MONOTONE
  - cursor per-space dump 续训跨 run·新 run_id·E1 终 dump / E8 stage-skip / E4 replay 覆盖率前置
  - stages 五阶段配比 + G5/C6 harness 真接线（teacher.judge_ground_truth → self_proof_fn·weaning pre）
  - 确定性 bit-identical（dump/load + REPLAY 重跑）
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO, EDGE_PROPERTY, EDGE_IS_A,
)
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, PathResult, PathData, Episode, GMeta,
    OutputResult, OutputPart, JudgeWeights, ConceptRef,
    TERMINAL_REACHED_SINK, INTENT_QUESTION,
    DOMAIN_TEXT, DOMAIN_MATH, DOMAIN_CODE, LANG_ZH, WEANING_PRE, WEANING_POST,
    J3_CAUSES_WEIGHT, J3_PRECEDES_WEIGHT,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.result.judge import judge
from pure_integer_ai.cognition.result.convergence import CONVERGENCE_WINDOW
from pure_integer_ai.cognition.process.episode import episode_loop
from pure_integer_ai.teacher.teacher_boundary import (
    verify_teacher_boundary, is_acceptable, Violation,
    KIND_NAME, KIND_DEFINE, KIND_REWARD, KIND_ERROR_LABEL,
    KIND_PLACEMENT, KIND_ORDER, KIND_PUNCTUATION, KIND_COOCCURS,
)
from pure_integer_ai.teacher.recordable_teacher import (
    RecordableLLMTeacher, register_recording_table,
    MODE_OFF, MODE_RECORD, MODE_REPLAY,
    CONTENT_META_DEFINITION, CONTENT_KNOWLEDGE,
    GT_PASS, GT_FAIL,
)
from pure_integer_ai.teacher.weaning import (
    weaning_check, WeaningMetrics, WEANING_WINDOW_ROUNDS,
    METRIC_CONDUCTION, METRIC_REALIZES, METRIC_JUDGE_SELF, METRIC_OOV_PROMOTE,
)
from pure_integer_ai.training.oracle import (
    validate_b1_b4, calibrate_weights, conduction_rate, agreement_rate,
    WEIGHT_GRID,
)
from pure_integer_ai.training.promote import (
    promote_edge, promote_concept, promote_report,
    PROMOTE_FREQ_MIN,
)
from pure_integer_ai.training.cursor import (
    dump_run, load_run, CursorState, cursor_resume, check_replay_coverage,
    DUMP_TABLES,
)
from pure_integer_ai.training.stages import (
    build_judge_fn, stage_gate_config, stage_metric_gate, is_skippable,
    stage_active_gates, StageMetrics,
    STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD, STAGE4_PROMOTE_WEAN,
    STAGE5_MULTIMODAL, SKIPPABLE_STAGES,
    FLOOR_GRAPH_SIZE_S1, FLOOR_CONDUCTION_S3,
)
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    register_recording_table(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ns, ci
    b.close()


def _edge(es, sid, frm, to, et, *, strength=1, sn=0, tn=0, tier=TIER_PRIMARY,
          source=SOURCE_BARE_TEXT, subtype=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=source, tier=tier,
           subtype=subtype, sn=sn, tn=tn)


def _graph(b, *, surface_map=None):
    return ConceptGraph(b,
                       surface_of=(lambda r: surface_map.get(r)) if surface_map else None)


def _dag(sid, *, sink=None, edges=None, terminal=TERMINAL_REACHED_SINK):
    return PathResult(
        path=PathData(edges=edges or [], struct_unit_refs=[]),
        terminal=terminal, sink=sink, topo_layers=[], convergence={}, source=None,
    )


def _out(parts, reached_sink=True):
    return OutputResult(parts=parts, reached_sink=reached_sink)


# ============ teacher_boundary 白黑词汇表（§9 A2） ============

def test_boundary_whitelist_passes():
    for k in (KIND_NAME, KIND_DEFINE, KIND_REWARD, KIND_ERROR_LABEL):
        assert verify_teacher_boundary({"kind": k, "text": "ok"}) == []
    assert is_acceptable({"kind": KIND_DEFINE, "text": "apple is a fruit"})


def test_boundary_blacklist_rejected():
    for k in (KIND_PLACEMENT, KIND_ORDER, KIND_PUNCTUATION, KIND_COOCCURS):
        vs = verify_teacher_boundary({"kind": k})
        assert len(vs) == 1
        assert vs[0].code == "blacklisted"


def test_boundary_not_whitelisted_rejected():
    vs = verify_teacher_boundary({"kind": 999})
    assert vs[0].code == "not_whitelisted"


def test_boundary_no_kind_rejected():
    vs = verify_teacher_boundary({"text": "x"})
    assert vs[0].code == "no_kind"


def test_boundary_struct_directive_in_text_rejected():
    # 白表 kind 但 text 含黑关键词（结构指令越界）→ 拒写
    vs = verify_teacher_boundary({"kind": KIND_DEFINE, "text": "place at placement X"})
    assert vs[0].code == "struct_directive"


# ============ RecordableLLMTeacher 录放层（E4 miss→None 无 fallback） ============

def _llm_factory():
    """假 LLM（MODE_RECORD 离线用·确定性·不调真 LLM）。"""
    def llm_call(kind, args):
        if kind == KIND_DEFINE:
            return {"kind": KIND_DEFINE, "content_type": args[-1],
                    "text": f"define_{args[2]}", "response_int": 0}
        if kind == KIND_REWARD:
            return {"kind": KIND_REWARD, "content_type": CONTENT_META_DEFINITION,
                    "text": None, "response_int": 1}
        return {"kind": kind, "content_type": CONTENT_META_DEFINITION,
                "text": "x", "response_int": 1}
    return llm_call


def test_teacher_record_then_replay(core):
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    r = t.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    assert r is not None and r["text"] == "define_5"   # llm 用 lid 作 text
    # REPLAY 同 key 命中（零 LLM·bit-identical）
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    r2 = t2.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    assert r2 == r


def test_teacher_replay_miss_returns_none_no_fallback(core):
    """E4·miss→None 无 fallback（不静默调真 LLM·破 bit 可复现）。"""
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    assert t.define((sid, 999), "unknown") is None     # miss→None
    assert t.confirm_causes((sid, 1), (sid, 2)) is None  # miss→None
    assert t.confirm_is_a((sid, 1), (sid, 2)) is None    # miss→None（对称 confirm_causes·补 IS_A 来源③ 缺口）


def test_teacher_confirm_is_a_record_replay(core):
    """IS_A 来源③ confirm_is_a（对称 confirm_causes）·录放层 round-trip + miss→None。"""
    b, sid, es, ns, ci = core
    def is_a_llm(kind, args):
        # confirm_is_a args=("confirm_is_a", c_sid, c_lid, p_sid, p_lid) → True（proper subset 确认）
        return {"kind": KIND_DEFINE, "content_type": CONTENT_META_DEFINITION,
                "text": None, "response_int": 1}
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=is_a_llm)
    assert t.confirm_is_a((sid, 1), (sid, 2)) is True   # 录制 + 返 True
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    assert t2.confirm_is_a((sid, 1), (sid, 2)) is True  # 重放
    assert t2.confirm_is_a((sid, 9), (sid, 8)) is None  # miss→None（E4 无 fallback）


def test_teacher_off_returns_none(core):
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_OFF)
    assert t.define((sid, 5), "apple") is None
    assert t.judge_ground_truth(_out([]), _dag(sid), None) is None  # 退场→None 无 fallback（stub #3）


def test_teacher_record_idempotent(core):
    """幂等：同 key 重录跳过（录一次即足·守可复现）。"""
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    t.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    t.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    keys = t.recorded_keys(KIND_DEFINE)
    assert len(keys) == 1   # 同 key 只录一次


def test_teacher_record_rejects_boundary_violation(core):
    """越界响应拒写（verify_teacher_boundary·§9 A2·不进核心）。"""
    b, sid, es, ns, ci = core
    def bad_llm(kind, args):
        return {"kind": KIND_PLACEMENT, "content_type": CONTENT_META_DEFINITION,
                "text": "x", "response_int": 0}   # 黑表 kind
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=bad_llm)
    t.define((sid, 5), "apple")
    assert t.recorded_keys() == []   # 越界拒写·不录


def test_teacher_replay_coverage(core):
    """E4 replay 覆盖率（续训前置校验·未达标禁续训）。"""
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    t.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    needed = [
        (KIND_DEFINE, ("define", sid, 5, "apple", CONTENT_META_DEFINITION)),  # hit
        (KIND_DEFINE, ("define", sid, 999, "x", CONTENT_META_DEFINITION)),    # miss
    ]
    recorded, total = t2.replay_coverage(needed)
    assert (recorded, total) == (1, 2)


def test_teacher_ground_truth_record_replay(core):
    """G5/C6 Mode A 教师 ground-truth 录放（self_proof_fn 注入契约）。"""
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    out = _out([OutputPart((sid, 1), ["a"])], reached_sink=True)
    dag = _dag(sid, sink=(sid, 1))
    assert t.judge_ground_truth(out, dag, None) == GT_PASS   # 录制 + 返
    # REPLAY 读回
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    assert t2.judge_ground_truth(out, dag, None) == GT_PASS


# ============ weaning D1-D5/E2 六闸门（#358 完整实现·非布尔阈值） ============

def _metrics(rounds, *, cond=0, real=0, judge=0, oov=0,
             interv=0, reten=0, dep=0):
    return WeaningMetrics(rounds=rounds, conduction_rate=cond,
                          realizes_rate=real, judge_self_rate=judge,
                          oov_promote_rate=oov, intervention_rate=interv,
                          holdout_retention=reten, dependency=dep)


def _all_gates_true():
    """六闸门硬前置全 True（D1 方向性由 history 提供·D2-D5/E2 由 kwargs 提供）。"""
    return dict(neg_pathway_active=True, judge_source_independent=True,
                probe_set_disjoint=True, mode_b_prevalidated=True, e2_passed=True)


def test_weaning_plateau_ready():
    """六闸门全过 → ready（D1 双规定曲线方向性 + 4 能力指标平台 + D2-D5/E2 全 True）。

    双曲线趋近渐近线：4 能力指标早期上升·最近 4 runs 窗口平台化 ∧ 终值达下限。
    D1 曲线① intervention 单调降∧降至阈值以下 / 曲线② retention 后窗不回升∧达下限 / 依赖度低。
    """
    hist = [
        _metrics(1, cond=100, real=100, judge=100, oov=20,
                 interv=800, reten=600, dep=200),
        _metrics(2, cond=594, real=395, judge=595, oov=95,
                 interv=500, reten=800, dep=180),
        _metrics(3, cond=597, real=398, judge=598, oov=98,
                 interv=400, reten=750, dep=160),
        _metrics(4, cond=599, real=399, judge=599, oov=99,
                 interv=300, reten=720, dep=140),
        _metrics(5, cond=600, real=400, judge=600, oov=100,
                 interv=200, reten=710, dep=120),
    ]
    rep = weaning_check(hist, **_all_gates_true())
    assert rep.ready is True
    assert all(rep.plateaued.values())
    assert rep.intervention_decreasing is True
    assert rep.retention_stable is True
    assert rep.dependency_low is True


def test_weaning_insufficient_window_not_ready():
    hist = [_metrics(1, cond=800, real=800, judge=800, oov=200)]
    rep = weaning_check(hist)
    assert rep.ready is False   # 窗口不足 4 runs


def test_weaning_floor_not_met_not_ready():
    """全 0 平台假断奶：增量 0 但下限未达 → not ready（D1 诚实）。"""
    hist = [_metrics(i) for i in range(1, 6)]   # 全 0
    rep = weaning_check(hist)
    assert rep.ready is False
    assert rep.floors_met is False


def test_weaning_still_rising_not_ready():
    """度量仍在快速上升（未平台化）→ not ready。"""
    hist = [
        _metrics(1, cond=500, real=400, judge=600, oov=100),
        _metrics(2, cond=600, real=500, judge=700, oov=200),
        _metrics(3, cond=700, real=600, judge=800, oov=300),
        _metrics(4, cond=800, real=700, judge=900, oov=400),
    ]
    rep = weaning_check(hist)
    assert rep.ready is False
    assert not all(rep.plateaued.values())


def test_weaning_window_rounds_is_four():
    assert WEANING_WINDOW_ROUNDS == 4


# ============ oracle B1-B4 + H2 calibrate_weights ============

def test_oracle_validate_b1_b4_passes():
    validate_b1_b4()   # 不抛 = 占位值与系统常量一致
    assert J3_CAUSES_WEIGHT == 10 and J3_PRECEDES_WEIGHT == 1
    assert CONVERGENCE_WINDOW == 1000


def test_oracle_calibrate_weights_maximizes_agreement():
    """H2：网格搜索选 agreement 最大的权重（reward 跨 0 依 w3 幅度·非纯符号）。"""
    samples = [object() for _ in range(5)]

    def judge_fn(s, *, weights=None):
        weights = weights or JudgeWeights(1, 1, 1)
        # reward = w3 - 4：w3>4 → 正·w3≤4 → 非正（幅度跨 0·非纯符号）
        return (weights.w3 - 4, GMeta())

    def teacher_gt(s):
        return GT_PASS   # 教师全 pass·需 w3>4 才 reward>0 命中

    w = calibrate_weights(samples, judge_fn, teacher_gt)
    # w3>4 才达满 agreement（5 hits）·tiebreak 权重和最小 → w3=5（非 8）
    assert w.w3 == 5
    assert w.w1 == 1 and w.w2 == 1


def test_oracle_conduction_rate():
    samples = [object() for _ in range(4)]

    def judge_fn(s):
        return (1 if id(s) % 2 == 0 else 0, GMeta())

    rate = conduction_rate(samples, judge_fn)
    # 约一半导通·×1000
    assert 0 <= rate <= 1000


def test_oracle_agreement_rate():
    samples = [object() for _ in range(4)]

    def judge_fn(s, *, weights=None):
        return (1, GMeta())

    def teacher_gt(s):
        return GT_PASS

    assert agreement_rate(samples, judge_fn, teacher_gt, JudgeWeights()) == 1000


# ============ promote 三重 SHADOW→PRIMARY ============

def test_promote_edge_three_evidence_tier_flip(core):
    """三重全达 → SHADOW→PRIMARY tier flip。"""
    b, sid, es, ns, ci = core
    # SHADOW CAUSES 边·频次达 + reward 达 + 结构锚（IS_A 出边）
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=4, tn=1)
    _edge(es, sid, 1, 3, EDGE_IS_A, tier=TIER_PRIMARY)   # 结构锚
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    flipped = promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES))
    assert flipped is True
    row = es.get(space_id_from=sid, local_id_from=1,
                 space_id_to=sid, local_id_to=2, edge_type=EDGE_CAUSES)
    assert row["tier"] == TIER_PRIMARY


def test_promote_edge_missing_freq_not_flipped(core):
    """频次未达 → 不晋。"""
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=1, tn=0)  # freq=1 < 3
    _edge(es, sid, 1, 3, EDGE_IS_A, tier=TIER_PRIMARY)
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    assert promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES)) is False


def test_promote_edge_missing_reward_not_flipped(core):
    """reward 未达（tn 远大于 sn）→ 不晋。"""
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=1, tn=10)
    _edge(es, sid, 1, 3, EDGE_IS_A, tier=TIER_PRIMARY)
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    assert promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES)) is False


def test_promote_edge_missing_definition_not_flipped(core):
    """无结构锚无教师确认 → 不晋。"""
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=4, tn=1)
    # 无 IS_A/PROPERTY/REFERS_TO 出边
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    assert promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES)) is False


def test_promote_edge_teacher_confirm_satisfies_definition(core):
    """教师确认（录放层）满足定义证据③。"""
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=4, tn=1)

    class FakeTeacher:
        def confirm_causes(self, a, b):
            return True
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    flipped = promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES),
                           teacher=FakeTeacher())
    assert flipped is True


def test_promote_edge_already_primary_idempotent(core):
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_PRIMARY, sn=4, tn=1)
    ns.put(sid, 1, node_type=1, tier=TIER_PRIMARY)
    ns.put(sid, 2, node_type=1, tier=TIER_PRIMARY)
    assert promote_edge(es, ns, (sid, 1, sid, 2, EDGE_CAUSES)) is True   # 幂等


def test_promote_edge_monotone_violation_on_demotion(core):
    """MUTABLE_MONOTONE：tier 降抛违例。"""
    from pure_integer_ai.storage.discipline import MonotoneViolation
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_PRIMARY)
    with pytest.raises(MonotoneViolation):
        es.set_tier(space_id_from=sid, local_id_from=1,
                    space_id_to=sid, local_id_to=2,
                    edge_type=EDGE_CAUSES, new_tier=TIER_SHADOW)


def test_promote_concept_tier_max_edges(core):
    """节点 tier = max 其出/入边 tier（§十二⑤·stub #8：旧版无条件晋 PRIMARY·今按 max 边 tier）。"""
    b, sid, es, ns, ci = core
    # 有 PRIMARY 出边 → max=PRIMARY → 晋
    ns.put(sid, 1, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 2, node_type=1, tier=TIER_SHADOW)
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_PRIMARY)
    assert promote_concept(es, ns, (sid, 1)) is True
    assert ns.get(sid, 1)["tier"] == TIER_PRIMARY
    # 入边 PRIMARY 也算（节点 2 经入边晋）
    assert promote_concept(es, ns, (sid, 2)) is True
    assert ns.get(sid, 2)["tier"] == TIER_PRIMARY
    # 只有 SHADOW 边 → max=SHADOW·不升
    ns.put(sid, 3, node_type=1, tier=TIER_SHADOW)
    ns.put(sid, 4, node_type=1, tier=TIER_SHADOW)
    _edge(es, sid, 3, 4, EDGE_PRECEDES, tier=TIER_SHADOW)
    assert promote_concept(es, ns, (sid, 3)) is False
    assert ns.get(sid, 3)["tier"] == TIER_SHADOW
    # 无边节点 → 不冒晋（stub #8 修：旧版会错晋 PRIMARY）
    ns.put(sid, 5, node_type=1, tier=TIER_SHADOW)
    assert promote_concept(es, ns, (sid, 5)) is False
    assert ns.get(sid, 5)["tier"] == TIER_SHADOW


def test_promote_report_diagnostics(core):
    b, sid, es, ns, ci = core
    _edge(es, sid, 1, 2, EDGE_CAUSES, tier=TIER_SHADOW, sn=4, tn=1)
    _edge(es, sid, 1, 3, EDGE_IS_A, tier=TIER_PRIMARY)
    rep = promote_report(es, (sid, 1, sid, 2, EDGE_CAUSES))
    assert rep["freq"] is True and rep["reward"] is True and rep["definition"] is True
    assert rep["eligible"] is True


# ============ cursor dump 续训（per-space·新 run_id·E1/E8/E4） ============

def test_cursor_dump_load_roundtrip(core):
    """per-space dump → load 还原（跨 run 续训·bit-identical）。"""
    b, sid, es, ns, ci = core
    # 建图：两节点 + 一边
    ns.put(sid, 1, node_type=1, tier=TIER_PRIMARY)
    ns.put(sid, 2, node_type=1, tier=TIER_PRIMARY)
    _edge(es, sid, 1, 2, EDGE_CAUSES, sn=3, tn=1)
    with tempfile.TemporaryDirectory() as d:
        dumped = dump_run(b, d, "run_001", spaces=[sid])
        assert dumped == [sid]
        # 文件存在 + per-space 物理分开
        assert os.path.isfile(os.path.join(d, "run_001", f"space_{sid}.dump"))
        # load 到新 backend（新 run_id 从终 dump 起·E8）
        b2 = DictBackend()
        bootstrap(b2)
        loaded = load_run(b2, d, "run_001")
        assert loaded == [sid]
        # 行还原
        rows_edge = b2.select("edge", where={"space_id_from": sid,
                                              "local_id_from": 1})
        assert len(rows_edge) == 1
        assert rows_edge[0]["sn"] == 3
        rows_node = b2.select("concept_node", where={"space_id": sid})
        assert len(rows_node) == 2


def test_cursor_dump_bit_identical(core):
    """两跑 dump 内容一致（确定性 bit-identical）。"""
    b, sid, es, ns, ci = core
    ns.put(sid, 1, node_type=1, tier=TIER_PRIMARY)
    _edge(es, sid, 1, 2, EDGE_CAUSES)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dump_run(b, d1, "r", spaces=[sid])
        dump_run(b, d2, "r", spaces=[sid])
        with open(os.path.join(d1, "r", f"space_{sid}.dump"), encoding="utf-8") as f1, \
             open(os.path.join(d2, "r", f"space_{sid}.dump"), encoding="utf-8") as f2:
            assert f1.read() == f2.read()


def test_cursor_per_space_physical_separation(core):
    """两 space 各自独立 dump 文件（C5·三空间物理分开·用户铁律）。"""
    b, sid, es, ns, ci = core
    reg = SpaceRegistry(b)
    sp2 = AbstractSpace.create(reg, "core2")
    sid2 = sp2.space_id
    ns.put(sid, 1, node_type=1, tier=TIER_PRIMARY)
    ns.put(sid2, 1, node_type=1, tier=TIER_PRIMARY)
    with tempfile.TemporaryDirectory() as d:
        dumped = dump_run(b, d, "r", spaces=[sid, sid2])
        assert dumped == sorted([sid, sid2])
        assert os.path.isfile(os.path.join(d, "r", f"space_{sid}.dump"))
        assert os.path.isfile(os.path.join(d, "r", f"space_{sid2}.dump"))


def test_cursor_resume_stage_skip():
    """E8：已完成 skippable 跳过·非 skippable 保留。"""
    state = CursorState(base_run_id="run_001", run_id="run_002",
                        completed={STAGE1_SKELETON, STAGE2_CAUSES_ABS})
    todo = cursor_resume(state, list((STAGE1_SKELETON, STAGE2_CAUSES_ABS,
                                      STAGE3_REWARD, STAGE4_PROMOTE_WEAN)),
                         skippable=SKIPPABLE_STAGES)
    # 阶段1/2 已完成 skippable→跳·阶段3/4 保留
    assert STAGE1_SKELETON not in todo
    assert STAGE2_CAUSES_ABS not in todo
    assert STAGE3_REWARD in todo       # non-skippable 保留
    assert STAGE4_PROMOTE_WEAN in todo


def test_cursor_is_skippable_classification():
    assert is_skippable(STAGE1_SKELETON) is True
    assert is_skippable(STAGE2_CAUSES_ABS) is True
    assert is_skippable(STAGE3_REWARD) is False   # reward 须重标定 H2
    assert is_skippable(STAGE4_PROMOTE_WEAN) is False


def test_cursor_replay_coverage_preflight(core):
    """E4：续训前 replay 覆盖率 ≥ 阈值才允许续训。"""
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    t.define((sid, 5), "apple", content_type=CONTENT_META_DEFINITION)
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    needed_hit = [(KIND_DEFINE, ("define", sid, 5, "apple", CONTENT_META_DEFINITION))]
    assert check_replay_coverage(t2, needed_hit) is True
    needed_miss = [(KIND_DEFINE, ("define", sid, 999, "x", CONTENT_META_DEFINITION))]
    assert check_replay_coverage(t2, needed_miss) is False


def test_cursor_replay_coverage_b7_partial_miss_threshold(core):
    """B7 放宽阈值至 9/10：部分 miss（≥90%）放行续训·<90% 系统性 miss 仍拦。

    首版 1/1（100%）致 --resume 实际不可用（真实语料任一 miss 即 raise）·改 9/10 允许小量 miss。
    """
    b, sid, es, ns, ci = core
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    for i in range(10):   # 录制 10 个 define（local_id 0..9）
        t.define((sid, i), f"w{i}", content_type=CONTENT_META_DEFINITION)
    t2 = RecordableLLMTeacher(b, mode=MODE_REPLAY)

    def _needed(hit_ids, miss_ids):
        return ([(KIND_DEFINE, ("define", sid, i, f"w{i}", CONTENT_META_DEFINITION))
                 for i in hit_ids]
                + [(KIND_DEFINE, ("define", sid, m, f"miss{m}", CONTENT_META_DEFINITION))
                   for m in miss_ids])

    # 9 命中 / 10 总 = 90% ≥ 9/10 → 放行（B7 新行为·首版 1/1 会拦）
    needed_90 = _needed(range(9), [999])
    assert check_replay_coverage(t2, needed_90) is True
    # 8 命中 / 10 总 = 80% < 9/10 → 拦（系统性 miss）
    needed_80 = _needed(range(8), [998, 999])
    assert check_replay_coverage(t2, needed_80) is False


def test_cursor_dump_tables_cover_core():
    """终 dump 涉及核心表（concept_node/edge/def_array/memory_item）。"""
    assert "concept_node" in DUMP_TABLES
    assert "edge" in DUMP_TABLES
    assert "memory_item" in DUMP_TABLES


# ============ stages 五阶段 + G5/C6 harness 真接线 ============

def test_stage_gate_config_observe_only_before_reward():
    """阶段1-2 observe only（reward/promote off·破死锁）。"""
    s1 = stage_gate_config(STAGE1_SKELETON)
    assert s1.observe_active is True
    assert s1.reward_active is False
    assert s1.promote_active is False
    s2 = stage_gate_config(STAGE2_CAUSES_ABS)
    assert s2.reward_active is False


def test_stage_gate_config_reward_opens_at_stage3():
    s3 = stage_gate_config(STAGE3_REWARD)
    assert s3.reward_active is True
    assert s3.promote_active is False   # promote 阶段4 才开


def test_stage_gate_config_promote_at_stage4():
    s4 = stage_gate_config(STAGE4_PROMOTE_WEAN)
    assert s4.promote_active is True
    assert s4.teacher_active is True    # 断奶前教师在位
    assert s4.weaning_phase == WEANING_PRE


def test_stage_gate_config_multimodal_defer():
    s5 = stage_gate_config(STAGE5_MULTIMODAL)
    assert s5.observe_active is False   # defer·非训练


def test_stage_metric_gate_floor():
    assert stage_metric_gate(STAGE1_SKELETON,
                             StageMetrics(graph_size=FLOOR_GRAPH_SIZE_S1)) is True
    assert stage_metric_gate(STAGE1_SKELETON,
                             StageMetrics(graph_size=1)) is False
    assert stage_metric_gate(STAGE3_REWARD,
                             StageMetrics(conduction_rate=FLOOR_CONDUCTION_S3)) is True


def test_build_judge_fn_teacher_off_returns_none_self_proof(core):
    """TEACHER_MODE OFF → self_proof_fn=None（bit-identical 占位·不产 vacuous reward）。"""
    b, sid, es, ns, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    g = _graph(b, surface_map={U: "u"})
    try:
        gates.TEACHER_MODE = False
        jdg_fn = build_judge_fn(g, JudgeWeights(1, 1, 1),
                                teacher=object(), weaning_phase=WEANING_PRE)
        out = _out([OutputPart(U, ["a"])], reached_sink=True)
        dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
        inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                           domain=DOMAIN_MATH,
                           intent=IntentType(sink=U, is_causal_reasoning=True),
                           key_skeleton=[U])
        reward, gm = jdg_fn(out, dag, inp, WorkMemory())
        # self_proof_fn=None → G5 pass=1 占位·reward>0（非 vacuous）
        assert gm.G5 is False
        assert reward > 0
    finally:
        gates.TEACHER_MODE = False


def test_build_judge_fn_wires_teacher_ground_truth_to_g5(core):
    """G5/C6 harness 真接线：teacher.judge_ground_truth → self_proof_fn·weaning pre·G5 veto 流回。"""
    b, sid, es, ns, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    g = _graph(b, surface_map={U: "u"})
    # 录制教师 ground-truth=fail（判错）
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    # 用一个 fail llm 覆盖 GT
    def fail_llm(kind, args):
        if kind == KIND_REWARD:
            return {"kind": KIND_REWARD, "content_type": CONTENT_META_DEFINITION,
                    "text": None, "response_int": GT_FAIL}
        return {"kind": kind, "content_type": CONTENT_META_DEFINITION,
                "text": "x", "response_int": 1}
    t_fail = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=fail_llm)
    out = _out([OutputPart(U, ["a"])], reached_sink=True)
    dag = _dag(sid, sink=U, edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
    # 录制 fail
    t_fail.judge_ground_truth(out, dag, g)
    # REPLAY 模式教师·真接线
    t_replay = RecordableLLMTeacher(b, mode=MODE_REPLAY)
    try:
        gates.TEACHER_MODE = True
        jdg_fn = build_judge_fn(g, JudgeWeights(1, 1, 1),
                                teacher=t_replay, weaning_phase=WEANING_PRE)
        inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                           domain=DOMAIN_CODE,
                           intent=IntentType(sink=U, is_causal_reasoning=True,
                                              has_value_claim=True),
                           key_skeleton=[U])
        reward, gm = jdg_fn(out, dag, inp, WorkMemory())
        assert gm.G5 is True        # 教师 ground-truth=fail → G5 veto 流回
        assert reward == 0
    finally:
        gates.TEACHER_MODE = False


def test_build_judge_fn_weaning_post_no_teacher(core):
    """断奶后（WEANING_POST）教师退场·self_proof_fn=None（D 墙·Mode B defer）。"""
    b, sid, es, ns, ci = core
    g = _graph(b)
    try:
        gates.TEACHER_MODE = True
        jdg_fn = build_judge_fn(g, JudgeWeights(1, 1, 1),
                                teacher=object(), weaning_phase=WEANING_POST)
        # weaning_post → self_proof_fn=None·即使 TEACHER_MODE ON 也不接教师
        out = _out([OutputPart((sid, 1), ["a"])], reached_sink=True)
        dag = _dag(sid, sink=(sid, 1),
                   edges=[(sid, 1, sid, 2, EDGE_CAUSES)])
        inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                           domain=DOMAIN_MATH,
                           intent=IntentType(sink=(sid, 1), is_causal_reasoning=True),
                           key_skeleton=[(sid, 1)])
        reward, gm = jdg_fn(out, dag, inp, WorkMemory())
        assert gm.G5 is False   # self_proof_fn=None → pass=1 占位
    finally:
        gates.TEACHER_MODE = False


def test_stage_active_gates_respect_global_off():
    """全局 TRAINING_MODE/TEACHER_MODE OFF 时阶段配比不生效（bit-identical 默认）。"""
    try:
        gates.TRAINING_MODE = False
        gates.TEACHER_MODE = False
        cfg = stage_gate_config(STAGE3_REWARD)
        active = stage_active_gates(cfg)
        assert active["reward"] is False     # TRAINING_MODE OFF
        assert active["teacher"] is False    # TEACHER_MODE OFF
    finally:
        gates.TRAINING_MODE = False
        gates.TEACHER_MODE = False


def test_end_to_end_episode_with_teacher_harness(core):
    """端到端：episode_loop + build_judge_fn 真接线·教师 ground-truth 经录放层进 G5。"""
    b, sid, es, ns, ci = core
    U = ci.ensure("u", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U, [1])
    _edge(es, sid, 1, 2, EDGE_CAUSES, strength=1)
    dag_edges = b.select("edge")
    g = _graph(b, surface_map={U: "u", (sid, 1): "x", (sid, 2): "y"})
    # 录制教师 GT=pass
    t = RecordableLLMTeacher(b, mode=MODE_RECORD, llm_call=_llm_factory())
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT,
                       intent=IntentType(type=INTENT_QUESTION, sink=U,
                                          is_causal_reasoning=True),
                       key_skeleton=[U])
    wm = WorkMemory()
    gen_fn = lambda pr, w, i: generate_output(pr, g, w, LANG_ZH)
    # 先用 RECORD 教师跑一遍录 GT
    jdg_rec = build_judge_fn(g, JudgeWeights(1, 1, 1), teacher=t,
                             weaning_phase=WEANING_PRE)
    try:
        gates.TEACHER_MODE = True
        episode_loop(inp, dag_edges, [U], wm,
                     IntentType(type=INTENT_QUESTION, sink=U,
                                is_causal_reasoning=True),
                     generate_fn=gen_fn, judge_fn=jdg_rec,
                     edge_store=es, backend=b)
        # REPLAY 教师重跑·零 LLM·bit-identical·G5 经录放层
        t_rp = RecordableLLMTeacher(b, mode=MODE_REPLAY)
        jdg_rp = build_judge_fn(g, JudgeWeights(1, 1, 1), teacher=t_rp,
                                weaning_phase=WEANING_PRE)
        wm2 = WorkMemory()
        out, ep = episode_loop(inp, dag_edges, [U], wm2,
                               IntentType(type=INTENT_QUESTION, sink=U,
                                          is_causal_reasoning=True),
                               generate_fn=gen_fn, judge_fn=jdg_rp,
                               edge_store=es, backend=b)
        assert ep.terminal == TERMINAL_REACHED_SINK
        assert ep.reward >= 0
        assert ep.judge_G5_active is False   # GT=pass → G5 未 veto
    finally:
        gates.TEACHER_MODE = False
