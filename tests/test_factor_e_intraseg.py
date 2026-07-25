"""factor E 层1 同段指代候选 反 theater e2e 测试（2026-07-09）。

验证（doc/重来_factorE_层1指代_intra_seg_设计_2026-07-09 §五 6 测）：
  - 层1 mechanism（resolve_pronoun_occurrence 同段前序 token 候选·gate PRONOUN_INTRASEG_MODE·score=k+1 近因）
  - bit-identical（gate OFF = current·同段前指仍悬空·既有测零回归）
  - e2e reward>0（observe 同段"动物...它们"→层1 解析→非 dangling→judge G4 不 veto→reward>0）

设计文档：doc/重来_factorE_层1指代_intra_seg_设计_2026-07-09.md
根因：judge.py:58 注释声称"层1 单句指代已解析"是 theater（#733 只实施层3+② fix·层1 候选生成从未实现）·
      语料"动物...它们"同段前指→无候选→dangling→J4 ②→reward=0。层1 补全解 factor E。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.edge_types import EDGE_COOCCURS, EDGE_REFERS_TO
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
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.judge import judge
from pure_integer_ai.config import gates


def _seg(tokens, **kw):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """建 backend + 三空间 + SpaceContext（同 test_j4_layer3 范式·跨 backend 验 bit-identical）。"""
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
    """建 EdgeStore + ConceptIndex + WorkMemory。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    wm = WorkMemory()
    return es, ci, wm


@pytest.fixture
def intraseg_on():
    """翻 PRONOUN_INTRASEG_MODE ON·测后复位（镜像生产 formal_train try/finally）。"""
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True
    yield
    gates.PRONOUN_INTRASEG_MODE = saved


def _ensure_concepts(ci, ctx, surfaces):
    """建 core 空间概念点 list（保序）。"""
    return [ci.ensure(s, space_id=ctx.core.space_id, tier=TIER_PRIMARY) for s in surfaces]


# ============ 层1 mechanism（同段前序 token 候选） ============

def test_intraseg_resolves_same_segment(ctx, intraseg_on):
    """层1 正（factor E）：单段"动物 是 类群 之一 它们"·gate ON → "它们"解析到同段前序 token（非 None）。

    cur_refs=[动物,是,类群,之一]（pronoun 它们 未入）·层1 候选全 4·score=k+1·最近=之一(k=3,score=4)胜。
    无层1（gate OFF）则 FIFO 空+层3 空→None（见 test_intraseg_gate_off_dangling）。
    """
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    refs = _ensure_concepts(ci, ctx, ["动物", "是", "类群", "之一"])
    wm._current_segment_refs = list(refs)   # 同段前序 token（pronoun 它们 未入）
    ant = resolve_pronoun_occurrence(
        es, ci, "它们", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant is not None                 # 层1 解析·非悬空
    assert ant == refs[-1]                 # 最近先行词（之一·k=3 score=4）胜
    assert wm._segment_dangling == 0       # 解析成功·不记悬空


def test_intraseg_gate_off_dangling(ctx):
    """层1 负控（bit-identical 守）：gate OFF → 层1 块跳过·FIFO 空+层3 空 → None·悬空。

    gate OFF = current（#733 后现状）·同段前指仍悬空·既有测零回归·bit-identical。
    """
    assert gates.PRONOUN_INTRASEG_MODE is False   # 默认 OFF
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    refs = _ensure_concepts(ci, ctx, ["动物", "是", "类群", "之一"])
    wm._current_segment_refs = list(refs)   # 同段前序（gate OFF 不读此字段）
    ant = resolve_pronoun_occurrence(
        es, ci, "它们", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant is None                      # gate OFF·层1 跳过·FIFO+层3 空→悬空
    assert wm._segment_dangling == 1        # ② fix·悬空记


def test_intraseg_recency_picks_closest(ctx, intraseg_on):
    """层1 近因序：cur_refs=[A,B,C]·"它"→C（k=2 score=3 最高）·非 A/B。

    score=k+1·越近 token（k 大）分越高·sort key=(-score, ref)→最近先行词胜。
    """
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    refs = _ensure_concepts(ci, ctx, ["A", "B", "C"])
    wm._current_segment_refs = list(refs)
    ant = resolve_pronoun_occurrence(
        es, ci, "它", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
    assert ant == refs[2]                   # C 最近（k=2 score=3）胜
    assert ant != refs[0]                   # 非 A（最远 k=0 score=1）


def test_intraseg_hub_filtered(ctx, intraseg_on):
    """层1 hub 过滤（gate EXCLUDE_FUNCTION_MODE·hub 不代）：hubA 出 hub_set → "它"→plainB（非 hubA）。

    加 COOCCURS 边使 A 成 hub（degree≥8）·cur_refs=[A,B]·层1 候选 A 被 hub_set 过滤·B 入→"它"→B。
    """
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    hubA = ci.ensure("hubA", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    plainB = ci.ensure("plainB", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    # COOCCURS 边使 hubA degree=8（hub·≥THETA_HUB_DEGREE=8）·plainB degree=4（非 hub）
    for tgt in ["x1", "x2", "x3", "x4"]:
        t = ci.ensure(tgt, space_id=ctx.core.space_id, tier=TIER_PRIMARY)
        es.add(space_id_from=hubA[0], local_id_from=hubA[1],
               space_id_to=t[0], local_id_to=t[1],
               edge_type=EDGE_COOCCURS, strength=2, source=SOURCE_BARE_TEXT)
    saved = gates.EXCLUDE_FUNCTION_MODE
    gates.EXCLUDE_FUNCTION_MODE = True      # 开 hub 过滤
    try:
        wm._current_segment_refs = [hubA, plainB]
        ant = resolve_pronoun_occurrence(
            es, ci, "它", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
        assert ant == plainB                # hubA 过滤·plainB 胜
        assert ant != hubA
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


def test_intraseg_all_hub_fallback(ctx, intraseg_on):
    """层1 factor F 软 hub fallback：cur_refs 全 hub（集中语料·内容词全 hub）→ fallback 收全→"它"→最近（非悬空）。

    集中语料实测（动物/生物/类群 全 hub）·硬 hub 过滤→零候选→悬空→reward=0。软 fallback→收全 hub→最近胜→
    解析非悬空→reward>0（trainability > dangling·stable≠correct·它→功能词可能·特征过滤 defer）。
    """
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    hubA = ci.ensure("hubA", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    hubB = ci.ensure("hubB", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    # COOCCURS 边使 hubA + hubB 均 degree≥8（全 hub·模拟集中语料内容词）
    for src, tgt_list in [(hubA, ["a1", "a2", "a3", "a4"]), (hubB, ["b1", "b2", "b3", "b4"])]:
        for tn in tgt_list:
            t = ci.ensure(tn, space_id=ctx.core.space_id, tier=TIER_PRIMARY)
            es.add(space_id_from=src[0], local_id_from=src[1],
                   space_id_to=t[0], local_id_to=t[1],
                   edge_type=EDGE_COOCCURS, strength=2, source=SOURCE_BARE_TEXT)
    saved = gates.EXCLUDE_FUNCTION_MODE
    gates.EXCLUDE_FUNCTION_MODE = True      # 开 hub 过滤
    try:
        wm._current_segment_refs = [hubA, hubB]   # 全 hub
        ant = resolve_pronoun_occurrence(
            es, ci, "它", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1)
        assert ant is not None             # ★ fallback 收全→非悬空（硬过滤会 None·reward=0）
        assert ant == hubB                 # 最近（k=1 score=2）胜·非悬空
        assert wm._segment_dangling == 0
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


def test_intraseg_pronoun_never_antecedent(ctx, intraseg_on):
    """对抗审 Bug#1：代词永不作先行词·未解析代词 SHADOW ref 不入层1候选·后代词不解析到代词。

    observe(["他","它"]·他 无内容词先行词→dangling·SHADOW 他 ref。无 fix：SHADOW 他 ref 入 _current_segment_refs
    →它 层1 候选含它→它 解析到 他（pronoun→pronoun·写 OCCURRENCE 污染边·_segment_dangling=1）。
    fix（observe append skip is_pronoun）：他 SHADOW ref 不入候选→它 零候选→dangling（_segment_dangling=2）。
    """
    wm = WorkMemory()
    obs = ObservePipeline(ctx, work_memory=wm)
    raw = InputPayload(segments=[_seg(["他", "它"], role_seq=[1, 1])],
                       source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    obs.observe(raw)
    # 两代词均无内容词先行词·fix 后均 dangling（它 不解析到 他）·无 fix 则 _segment_dangling=1（它→他）
    assert wm._segment_dangling == 2
    # 无 pronoun→pronoun OCCURRENCE 边（它 不解析到 他·Bug#1 fix 守·无污染）
    occ = ctx.core.backend.select("edge", where={"edge_type": EDGE_REFERS_TO,
                                                   "subtype": SUBTYPE_OCCURRENCE})
    assert len(occ) == 0


def test_intraseg_bit_identical(ctx, intraseg_on):
    """层1 bit-identical（确定性）：同 backend 类两独立实例跑·层1 解析结果 bit-identical。

    层1 候选 sort key=(-score, ref)·确定性 tiebreak·两跑一致。
    """
    def run(backend_cls):
        b = backend_cls()
        bootstrap(b)
        reg = SpaceRegistry(b)
        core = AbstractSpace.create(reg, "core")
        mem_read = MemorySpace.create(reg, "mem_read")
        comp = CompanionSpace.create(reg, "comp1")
        es = EdgeStore(b)
        ci = ConceptIndex(b, comp)
        wm = WorkMemory()
        refs = [ci.ensure(s, space_id=core.space_id, tier=TIER_PRIMARY)
                for s in ["动物", "类群", "之一"]]
        wm._current_segment_refs = list(refs)
        ant = resolve_pronoun_occurrence(
            es, ci, "它们", work_memory=wm, memory_space_id=mem_read.space_id,
            timestamp_seq=1)
        b.close()
        return ant
    backend_cls = DictBackend if isinstance(ctx.core.backend, DictBackend) else SQLiteBackend
    ant_a = run(backend_cls)
    ant_b = run(backend_cls)
    assert ant_a == ant_b                    # 两跑解析一致·bit-identical


# ============ e2e（observe 同段前指 → judge reward>0·核心验） ============

def test_intraseg_e2e_observe_to_reward(ctx):
    """层1 e2e（factor E·端到端 reward>0 核心验）：observe"动物...它们"同段前指 →

    gate OFF：它们 dangling → struct_ref 入 dangling_units → judge G4 veto → reward=0。
    gate ON ：层1 解析它们 → 非 dangling → judge G4 不 veto → reward>0。
    非 check_closure 单元·验 observe→dangling_units→judge→reward 真闭环（factor E → reward>0）。
    """
    def _judge_with(wm, struct_ref):
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

    tokens = ["动物", "是", "类群", "之一", "它们", "拥有", "形态"]

    # 场景1：gate OFF → 同段前指悬空 → dangling_units 含 struct_ref → G4 veto → reward=0
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = False
    try:
        wm1 = WorkMemory()
        obs1 = ObservePipeline(ctx, work_memory=wm1)
        raw1 = InputPayload(segments=[_seg(tokens, role_seq=[1] * len(tokens))],
                            source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
        result1 = obs1.observe(raw1)
        struct1 = result1.struct_refs[0]
        assert struct1 in wm1.dangling_units          # gate OFF·它们悬空→struct_ref 入集
        reward1, gm1 = _judge_with(wm1, struct1)
        assert reward1 == 0
        assert gm1.G4_vetoed is True                  # ② fire·G4 veto
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved

    # 场景2：gate ON → 层1 解析它们 → 非 dangling → G4 不 veto → reward>0
    gates.PRONOUN_INTRASEG_MODE = True
    try:
        wm2 = WorkMemory()
        obs2 = ObservePipeline(ctx, work_memory=wm2)
        raw2 = InputPayload(segments=[_seg(tokens, role_seq=[1] * len(tokens))],
                            source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
        result2 = obs2.observe(raw2)
        struct2 = result2.struct_refs[0]
        assert len(wm2.dangling_units) == 0           # 层1 解析它们→无悬空
        reward2, gm2 = _judge_with(wm2, struct2)
        assert reward2 > 0                             # ★ factor E → reward>0 真流
        assert gm2.G4_vetoed is False                 # ② 不 fire·对照场景1
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved
