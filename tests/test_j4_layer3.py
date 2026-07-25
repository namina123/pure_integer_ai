"""#733 J4 指代层3 反 theater e2e 测试（2026-07-07）。

验证：
  - layer 3 mechanism（observe resolve_pronoun_occurrence 扩候选·OCCURRENCE 边 + effective_weight 衰减）
  - ② fix（check_closure 查 workmem.dangling_units·绕旧 ② theater 4 重 kill switch）
  - 诚实边界（stable≠correct / honest forgetting / 扩候选覆盖范围非语义消解）

设计文档：doc/重来_任务0733_J4指代层3_设计.md
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SUBTYPE_OCCURRENCE, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.types import (
    SpaceContext, OutputResult, OutputPart, InputPayload, Segment, IntentType,
    JudgeWeights, PathData, PathResult,
    STAGE_TRAINING, WEANING_PRE,
    INTENT_QUESTION, TERMINAL_REACHED_SINK,
    MODALITY_LANGUAGE, LANG_ZH, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.refers_occurrence import (
    resolve_pronoun_occurrence, OCCURRENCE_STRENGTH,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.judge import check_closure, judge


def _seg(tokens, **kw):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """建 backend + 三空间 + SpaceContext（同 test_stage3 范式·跨 backend 验 bit-identical）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    mem_read = MemorySpace.create(reg, "mem_read")
    mem_interact = MemorySpace.create(reg, "mem_interact")
    comp = CompanionSpace.create(reg, "comp1")
    c = SpaceContext(
        core=core, memory_read=mem_read, memory_interact=mem_interact,
        companion=comp, stage=STAGE_TRAINING, memory_active=False,
        weaning_phase=WEANING_PRE,
    )
    yield c
    b.close()


def _setup(ctx):
    """建 EdgeStore + ConceptIndex + WorkMemory + 小明 antecedent。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    wm = WorkMemory()
    xiaoming = ci.ensure("小明", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    return es, ci, wm, xiaoming


# ============ 层3 mechanism（observe resolve_pronoun_occurrence 扩候选） ============

def test_layer3_resolves_beyond_window(ctx):
    """层3 正（#733·§十四:1291）：超 N=3 FIFO 窗口·读 OCCURRENCE 边历史先行词·解析代词非悬空。

    场景：seg0 小明·seg1 "他"→小明（FIFO·写 OCCURRENCE 边）·seg2-4 empty 逐出 seg0/seg1·
    seg5 "他"（FIFO 仅 empty segs 无候选）→ 层3 读 OCCURRENCE 边→解析小明。
    无层3则 FIFO 空候选→悬空（None）·层3 扩候选覆盖范围。
    """
    es, ci, wm, xiaoming = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    # seg0: 小明（antecedent 入 FIFO）
    wm.push_segment(0, [xiaoming])
    # seg1: "他"→小明（FIFO 候选·写 OCCURRENCE 边 pronoun_ref→小明·memory_time_attach=1）
    ant1 = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant1 == xiaoming
    # seg2-4: empty（逐出 seg0/seg1 出 N=3 FIFO·FIFO 现仅 empty segs 无候选）
    wm.push_segment(2, [])
    wm.push_segment(3, [])
    wm.push_segment(4, [])
    # 层3：读 OCCURRENCE 边（他→小明）·score=effective_weight=max(0,1000-(10-1))=991 ≥1·解析小明
    ant2 = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=10)
    assert ant2 == xiaoming   # 层3 解析·非 None·扩候选覆盖范围


def test_layer3_no_occurrence_edge_dangling(ctx):
    """层3 负控：FIFO 无候选且无 prior OCCURRENCE 边·层3 零候选→悬空返 None（② fire 真碎句）。

    "她"从未解析过·无 OCCURRENCE 边·层3 零候选·FIFO 亦空→悬空。② fix 记 _segment_dangling=1。
    """
    es, ci, wm, _ = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    # FIFO empty·"她"无 prior OCCURRENCE 边（从未解析过"她"）→ 层3 零候选→悬空
    ant = resolve_pronoun_occurrence(
        es, ci, "她", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant is None
    # ② fix：悬空记 _segment_dangling=1（observe 段末标 struct_ref 进 dangling_units）
    assert wm._segment_dangling == 1


def test_layer3_decay_below_theta_dangling(ctx):
    """层3 honest forgetting：OCCURRENCE 边衰减超 OCCURRENCE_STRENGTH 窗→score=0→候选不入→悬空。

    OCCURRENCE 边 memory_time_attach=1·timestamp_seq=OCCURRENCE_STRENGTH+100·
    logical_age=OCCURRENCE_STRENGTH+99 > OCCURRENCE_STRENGTH → effective_weight=max(0,1000-(1099))=0·
    候选不入·FIFO 亦空→悬空。诚实边界：层3 不保证解析·仅提供窗内候选·超窗 honest forgetting。
    """
    es, ci, wm, xiaoming = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    wm.push_segment(0, [xiaoming])
    # seg1: "他"→小明（写 OCCURRENCE 边·memory_time_attach=1）
    resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    # 逐出 seg0/seg1 出 FIFO
    wm.push_segment(2, [])
    wm.push_segment(3, [])
    wm.push_segment(4, [])
    # 层3 读 OCCURRENCE 边·logical_age=OCCURRENCE_STRENGTH+99 > 窗→score=0→候选不入→悬空
    ant = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid,
        timestamp_seq=OCCURRENCE_STRENGTH + 100)
    assert ant is None   # honest forgetting·超窗悬空
    assert wm._segment_dangling == 1   # ② fix·悬空记（同 test_layer3_no_occurrence_edge_dangling）


def test_layer3_wrong_prior_propagates(ctx):
    """层3 stable≠correct（#733 诚实边界）：prior OCCURRENCE 边指向错误先行词·层3 传播错误非语义纠正。

    seg0 苹果（错误 FIFO 启发式候选）·seg1 "他"→苹果（FIFO 错解析·写 OCCURRENCE 边 他→苹果）·
    seg5 "他"→层3 读边→解析苹果（错误传播）·非语义纠正到小明。验 stable≠correct·#479 墙。
    """
    es, ci, wm, xiaoming = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    pingguo = ci.ensure("苹果", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    # seg0: 苹果（错误先行词·模拟 FIFO 启发式错解析）
    wm.push_segment(0, [pingguo])
    # seg1: "他"→苹果（FIFO 候选·写 OCCURRENCE 边 他→苹果·错误）
    ant1 = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant1 == pingguo   # FIFO 启发式错解析到苹果
    # 逐出 seg0/seg1
    wm.push_segment(2, [])
    wm.push_segment(3, [])
    wm.push_segment(4, [])
    # 层3 读 OCCURRENCE 边（他→苹果）·解析苹果（错误传播·非语义纠正到小明）
    ant2 = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=10)
    assert ant2 == pingguo   # 错误先行词传播·stable≠correct
    assert ant2 != xiaoming  # 非正确先行词·层3 不语义纠正


def test_layer3_bit_identical(ctx):
    """层3 bit-identical（#733 确定性）：同 backend 类两独立实例跑·解析结果 + OCCURRENCE 边 bit-identical。

    层3 候选排序 key=(-score, ref)·确定性 tiebreak·两跑一致。@pytest.fixture params=dict/sqlite
    各跑两遍同 backend 类·验类内 bit-identical（跨 backend 类由 sort 兜底·非本测职责）。
    """
    def run(backend_cls):
        b = backend_cls()
        bootstrap(b)
        reg = SpaceRegistry(b)
        core = AbstractSpace.create(reg, "core")
        mem_read = MemorySpace.create(reg, "mem_read")
        comp = CompanionSpace.create(reg, "comp1")
        c = SpaceContext(
            core=core, memory_read=mem_read, memory_interact=mem_read,
            companion=comp, stage=STAGE_TRAINING, memory_active=False,
            weaning_phase=WEANING_PRE,
        )
        es = EdgeStore(b)
        ci = ConceptIndex(b, comp)
        wm = WorkMemory()
        xiaoming = ci.ensure("小明", space_id=core.space_id, tier=TIER_PRIMARY)
        wm.push_segment(0, [xiaoming])
        resolve_pronoun_occurrence(
            es, ci, "他", work_memory=wm, memory_space_id=mem_read.space_id, timestamp_seq=1)
        wm.push_segment(2, [])
        wm.push_segment(3, [])
        wm.push_segment(4, [])
        ant = resolve_pronoun_occurrence(
            es, ci, "他", work_memory=wm, memory_space_id=mem_read.space_id, timestamp_seq=10)
        occ = b.select("edge", where={"edge_type": EDGE_REFERS_TO,
                                      "subtype": SUBTYPE_OCCURRENCE})
        b.close()
        return ant, occ
    # 两独立 backend（同 ctx 当前 param·dict 或 sqlite）跑同输入
    backend_cls = DictBackend if isinstance(ctx.core.backend, DictBackend) else SQLiteBackend
    ant_a, occ_a = run(backend_cls)
    ant_b, occ_b = run(backend_cls)
    assert ant_a == ant_b                              # 解析结果一致
    assert occ_a == occ_b                              # OCCURRENCE 边 bit-identical
    assert len(occ_a) == 2                             # 两次解析·两 OCCURRENCE 边


# ============ ② fix（check_closure 查 dangling_units·绕旧 ② theater） ============

def test_judge_2_theater_fix_check_closure():
    """② fix（#733）：check_closure ② 查 workmem.dangling_units·绕旧 ② theater 4 重 kill switch。

    旧 ② 查 produced_refs 代词悬空·4 重 kill switch（produced_refs 是 struct_refs / surface_of 生产 None /
    out_edges 全类型 / EDGE_REFERS_TO 未 import 破函数）·从未 fire。改读 dangling_units·非 theater 真闭合判据。
    验：① 槽绑定仍守（空槽 fire）·② 输出 unit ∈ dangling_units → fire（J4 veto）·② 不在 → 不 fire。
    """
    U = (1, 100)
    # ① 空槽仍 fire（既有行为不变·非 ②）
    wm0 = WorkMemory()
    out0 = OutputResult(parts=[OutputPart(unit=U, words=[])], reached_sink=True)
    assert check_closure(out0, None, None, wm0) is False   # ① 空槽 fire
    # ② 输出 unit ∈ dangling_units → fire（J4 veto·② fix 真闭合判据）
    wm1 = WorkMemory()
    wm1.dangling_units.add(U)
    out1 = OutputResult(parts=[OutputPart(unit=U, words=["a"])], reached_sink=True)
    assert check_closure(out1, None, None, wm1) is False   # ② fire·输出含悬空段
    # ② 输出 unit 不在 dangling_units → ② 不 fire·① 通过 → True
    wm2 = WorkMemory()
    out2 = OutputResult(parts=[OutputPart(unit=U, words=["a"])], reached_sink=True)
    assert check_closure(out2, None, None, wm2) is True    # ② 不 fire·无悬空
    # ② 多 part·任一 unit ∈ dangling_units → fire
    wm3 = WorkMemory()
    wm3.dangling_units.add((1, 200))
    out3 = OutputResult(parts=[OutputPart(unit=U, words=["a"]),
                                OutputPart(unit=(1, 200), words=["b"])],
                        reached_sink=True)
    assert check_closure(out3, None, None, wm3) is False   # 第二 part unit 悬空→fire


def test_judge_2_theater_fix_e2e_observe_to_judge(ctx):
    """② fix e2e（#733·端到端非 theater 闭环验·设计文档第五节 item 6 承诺）。

    全链路：observe（悬空代词 → resolve_pronoun_occurrence 返 None → _segment_dangling++ →
    段末标 struct_ref 进 dangling_units）→ judge（check_closure ② 查 output.unit ∈ dangling_units
    → G4 veto → reward=0）。对照：代词 FIFO 解析成功 → dangling_units 空 → ② 不 fire（G4_vetoed False）。
    非 check_closure 单元·验 observe→dangling_units→judge→reward 真闭环。
    """
    def _judge_with(wm, struct_ref):
        """构造 output（unit=struct_ref）+ 调 judge·返 (reward, GMeta)。"""
        output = OutputResult(parts=[OutputPart(unit=struct_ref, words=["a"])],
                              reached_sink=True)
        dag = PathResult(path=PathData(edges=[], struct_unit_refs=[struct_ref]),
                         terminal=TERMINAL_REACHED_SINK, sink=struct_ref,
                         topo_layers=[], convergence={}, source=None)
        inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
                           domain=DOMAIN_TEXT,
                           intent=IntentType(type=INTENT_QUESTION, sink=struct_ref,
                                             is_causal_reasoning=False))
        g = ConceptGraph(ctx.core.backend)
        return judge(output, dag, inp, g, JudgeWeights(1, 1, 1), wm)

    # 场景1：单段"他"无先行词 → 悬空 → dangling_units 含 struct_ref → judge ② fire G4 veto
    wm1 = WorkMemory()
    obs1 = ObservePipeline(ctx, work_memory=wm1)
    raw1 = InputPayload(segments=[_seg(["他"], role_seq=[1])],
                        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    result1 = obs1.observe(raw1)
    assert len(result1.struct_refs) == 1
    dangling_struct = result1.struct_refs[0]
    # observe→dangling_units 真活（非 theater）·悬空代词段 struct_ref 进集
    assert dangling_struct in wm1.dangling_units
    # judge ② 查 dangling_units → G4 veto → reward=0（端到端闭环验）
    reward1, gm1 = _judge_with(wm1, dangling_struct)
    assert reward1 == 0
    assert gm1.G4_vetoed is True   # ② fix → G4 veto 端到端真活

    # 场景2："小明"+"他"→ FIFO 解析成功 → dangling_units 空 → ② 不 fire（G4_vetoed False）
    wm2 = WorkMemory()
    obs2 = ObservePipeline(ctx, work_memory=wm2)
    raw2 = InputPayload(segments=[_seg(["小明"], role_seq=[1]),
                                   _seg(["他"], role_seq=[1])],
                        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    result2 = obs2.observe(raw2)
    assert len(wm2.dangling_units) == 0   # FIFO 解析"他"→小明·无悬空
    struct2 = result2.struct_refs[-1]     # "他"段 struct_ref
    reward2, gm2 = _judge_with(wm2, struct2)
    assert gm2.G4_vetoed is False   # ② 不 fire·对照场景1（reward 可能因他 G 为 0·但 G4 不 veto）
