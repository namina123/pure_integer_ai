"""B6 指代维 方案3 tn+fn 路 pronoun_resolution_count 测试（2026-07-10·#894）。

验证（doc/重来_纠偏轮_round2_任务文档_2026-07-10 §五 B6 + §十二.2 方案3 三路写）：
  - storage：record_decision 写 pr_tn / record_dangling 写 pr_fn self-loop / read_count / read_agg / 表未注册 skip
  - observe 写：resolve 决策→pr_tn++ (pronoun,antecedent) / 悬空→pr_fn++ self-loop (pronoun,pronoun)·gate 守
  - consumer 自消费（反 theater·生产非纸面闭合）：gate ON 读历史 pr_tn 加候选分·改变 best_ref 选择
  - bit-identical：gate OFF 不写不读·候选排序不变·两跑一致

设计：方案3 三路写（observe tn + 失败侧 fn 零教师 + 教师 sn P2 defer）·per-occurrence 决策时写·
      独立 episode 符号·避 β_arith（§十二.1 病根=判据来自 episode 末标量）。
诚实边界：指代维 reward=J4 bool veto（非 graded·与 B4/B5 count 进 _seed_weight 不同）·consumer 在 observe 侧自消费·
          reward>0 鲁棒（J4 只查 dangling 不查 antecedent 质量）·pr_sn 教师 P2 defer·代词消解结构非墙 vs sense 消歧 #479 真墙。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.storage.pronoun_resolution_count import (
    register_pronoun_resolution_count,
    record_pronoun_resolution_decision,
    record_pronoun_resolution_dangling,
    read_pronoun_resolution_count,
    read_pronoun_resolution_agg,
    PRONOUN_RESOLUTION_COUNT_TABLE,
)
from pure_integer_ai.cognition.shared.types import SpaceContext, STAGE_TRAINING, WEANING_PRE
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """建 backend + 三空间 + SpaceContext（同 test_factor_e_intraseg 范式·跨 backend 验 bit-identical）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_pronoun_resolution_count(b)   # B6 表（ObservePipeline 亦注册·幂等·直测场景手动注册）
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
def resolve_count_on():
    """翻 PRONOUN_RESOLVE_COUNT_MODE ON·测后复位（镜像生产 formal_train try/finally）。"""
    saved = gates.PRONOUN_RESOLVE_COUNT_MODE
    gates.PRONOUN_RESOLVE_COUNT_MODE = True
    yield
    gates.PRONOUN_RESOLVE_COUNT_MODE = saved


def _ensure(ci, ctx, surfaces, *, tier=TIER_PRIMARY, space="core"):
    """建概念点 list（保序）。space='core'/'mem'。"""
    sid = ctx.core.space_id if space == "core" else ctx.memory_read.space_id
    return [ci.ensure(s, space_id=sid, tier=tier) for s in surfaces]


# ============ storage：record_decision / record_dangling / read ============

def test_record_decision_creates_row(ctx):
    """record_decision 首次：insert(pr_tn=1, pr_fn=0, pr_sn=0)。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    record_pronoun_resolution_decision(b, pronoun_ref=p_ref, antecedent_ref=a_ref)
    rows = b.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
        "space_id_from": p_ref[0], "local_id_from": p_ref[1],
        "space_id_to": a_ref[0], "local_id_to": a_ref[1]})
    assert len(rows) == 1
    r = rows[0]
    assert r["pr_tn"] == 1
    assert r["pr_fn"] == 0
    assert r["pr_sn"] == 0


def test_record_decision_increments(ctx):
    """record_decision 已存在：pr_tn += 1·pr_fn/pr_sn 不动。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    for _ in range(3):
        record_pronoun_resolution_decision(b, pronoun_ref=p_ref, antecedent_ref=a_ref)
    sn, tn, fn = read_pronoun_resolution_count(b, p_ref, a_ref)
    assert tn == 3
    assert fn == 0
    assert sn == 0


def test_record_dangling_self_loop(ctx):
    """record_dangling 首次：insert(pr_fn=1, pr_tn=0)·self-loop from==to=pronoun。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    record_pronoun_resolution_dangling(b, pronoun_ref=p_ref)
    rows = b.select(PRONOUN_RESOLUTION_COUNT_TABLE, where={
        "space_id_from": p_ref[0], "local_id_from": p_ref[1],
        "space_id_to": p_ref[0], "local_id_to": p_ref[1]})   # self-loop from==to
    assert len(rows) == 1
    r = rows[0]
    assert r["pr_fn"] == 1
    assert r["pr_tn"] == 0
    assert r["pr_sn"] == 0
    # self-loop 标记：from==to=pronoun
    assert r["space_id_from"] == r["space_id_to"]
    assert r["local_id_from"] == r["local_id_to"]


def test_record_dangling_increments(ctx):
    """record_dangling 已存在：pr_fn += 1·pr_tn/pr_sn 不动。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    for _ in range(2):
        record_pronoun_resolution_dangling(b, pronoun_ref=p_ref)
    sn, tn, fn = read_pronoun_resolution_count(b, p_ref, p_ref)   # self-loop key
    assert fn == 2
    assert tn == 0
    assert sn == 0


def test_read_count_cold_start(ctx):
    """read_count 冷启动（无行）→ None。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    assert read_pronoun_resolution_count(b, p_ref, a_ref) is None


def test_read_count_table_unregistered():
    """read_count 表未注册 → None（向后兼容·bare fixture）。"""
    b = DictBackend()
    bootstrap(b)   # 不注册 pronoun_resolution_count
    assert read_pronoun_resolution_count(b, (1, 1), (1, 2)) is None
    b.close()


def test_read_agg(ctx):
    """read_agg：聚合 pronoun 的所有 antecedent 行 sum_pr_sn/tn/fn。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    b_ref = ci.ensure("B", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    # (P,A) pr_tn=2·(P,B) pr_tn=1·(P,P) pr_fn=1 self-loop
    record_pronoun_resolution_decision(b, pronoun_ref=p_ref, antecedent_ref=a_ref)
    record_pronoun_resolution_decision(b, pronoun_ref=p_ref, antecedent_ref=a_ref)
    record_pronoun_resolution_decision(b, pronoun_ref=p_ref, antecedent_ref=b_ref)
    record_pronoun_resolution_dangling(b, pronoun_ref=p_ref)
    sn, tn, fn = read_pronoun_resolution_agg(b, p_ref)
    assert sn == 0
    assert tn == 3        # (P,A)=2 + (P,B)=1
    assert fn == 1        # (P,P) self-loop


def test_read_agg_cold_start(ctx):
    """read_agg 冷启动（无行）→ (0, 0, 0)。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    p_ref = ci.ensure("P", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    assert read_pronoun_resolution_agg(b, p_ref) == (0, 0, 0)


def test_record_table_unregistered_skip():
    """record 表未注册 → KeyError 静默 skip（向后兼容·bare fixture·镜像 record_selection_pref_cooccur）。"""
    b = DictBackend()
    bootstrap(b)   # 不注册 pronoun_resolution_count
    record_pronoun_resolution_decision(b, pronoun_ref=(1, 1), antecedent_ref=(1, 2))   # 不 raise
    record_pronoun_resolution_dangling(b, pronoun_ref=(1, 1))                          # 不 raise
    b.close()


# ============ observe 写：resolve 决策 pr_tn / 悬空 pr_fn·gate 守 ============

def test_resolve_writes_pr_tn_on_decision(ctx, resolve_count_on):
    """gate ON·resolve 决策（选 best antecedent）→ (pronoun, antecedent) pr_tn=1·per-occurrence。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    refs = _ensure(ci, ctx, ["动物", "是", "类群", "之一"])
    wm._current_segment_refs = list(refs)
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True   # 层1 候选源（factor E·决策有候选）
    try:
        ant = resolve_pronoun_occurrence(
            es, ci, "它们", work_memory=wm, memory_space_id=mem_sid,
            timestamp_seq=1, backend=b)
        assert ant is not None                       # 层1 解析·决策成功
        # pronoun_ref = ci.ensure("它们", mem_sid, SHADOW)
        pronoun_ref = ci.ensure("它们", space_id=mem_sid, tier=TIER_SHADOW)
        sn, tn, fn = read_pronoun_resolution_count(b, pronoun_ref, ant)
        assert tn == 1                                # ★ 决策写 pr_tn
        assert fn == 0
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved


def test_resolve_writes_pr_fn_on_dangling(ctx, resolve_count_on):
    """gate ON·resolve 悬空（无候选）→ (pronoun, pronoun) self-loop pr_fn=1·per-occurrence·§九.2 病灶落 pronoun。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    # gate OFF 层1 + FIFO 空 + 层3 空 → 悬空（无候选）
    assert gates.PRONOUN_INTRASEG_MODE is False
    ant = resolve_pronoun_occurrence(
        es, ci, "它们", work_memory=wm, memory_space_id=mem_sid,
        timestamp_seq=1, backend=b)
    assert ant is None                                # 悬空
    assert wm._segment_dangling == 1                  # 既有 ② fix 不变
    pronoun_ref = ci.ensure("它们", space_id=mem_sid, tier=TIER_SHADOW)
    sn, tn, fn = read_pronoun_resolution_count(b, pronoun_ref, pronoun_ref)   # self-loop
    assert fn == 1                                    # ★ 悬空写 pr_fn self-loop
    assert tn == 0


def test_resolve_gate_off_no_write(ctx):
    """gate OFF·resolve 决策 → 不写 count 表（bit-identical·既有 _segment_dangling 不变）。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    refs = _ensure(ci, ctx, ["动物", "是", "类群", "之一"])
    wm._current_segment_refs = list(refs)
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True
    try:
        assert gates.PRONOUN_RESOLVE_COUNT_MODE is False   # gate OFF
        ant = resolve_pronoun_occurrence(
            es, ci, "它们", work_memory=wm, memory_space_id=mem_sid,
            timestamp_seq=1, backend=b)
        assert ant is not None                        # 层1 解析（factor E·gate OFF count 不影响解析）
        rows = b.select(PRONOUN_RESOLUTION_COUNT_TABLE, where=None)
        assert len(rows) == 0                         # ★ gate OFF 不写 count
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved


def test_resolve_gate_off_dangling_no_write(ctx):
    """gate OFF·resolve 悬空 → 不写 pr_fn（既有 _segment_dangling++ 不变·bit-identical）。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    assert gates.PRONOUN_INTRASEG_MODE is False
    assert gates.PRONOUN_RESOLVE_COUNT_MODE is False
    ant = resolve_pronoun_occurrence(
        es, ci, "它们", work_memory=wm, memory_space_id=mem_sid,
        timestamp_seq=1, backend=b)
    assert ant is None
    assert wm._segment_dangling == 1                  # 既有不变
    rows = b.select(PRONOUN_RESOLUTION_COUNT_TABLE, where=None)
    assert len(rows) == 0                             # ★ gate OFF 悬空不写 pr_fn


# ============ consumer 自消费（反 theater·生产非纸面闭合）============

def test_consumer_reads_pr_tn_changes_best_ref(ctx, resolve_count_on):
    """★ consumer 真活（反 theater）：gate ON 读历史 pr_tn 加候选分·改变 best_ref 选择。

    cur_refs=[B, A]·A 最近（k=1 score=2）·B 远（k=0 score=1）。
    gate OFF：A 胜（score 2 > 1·recency 主）。
    gate ON + 预写 (pronoun, B) pr_tn=3：B score=1+min(3,3)=4 > A score=2+0=2 → B 胜（bonus 翻盘）。
    reward>0 鲁棒（A/B 均非悬空 antecedent·J4 只查 dangling）。
    """
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    b_ref = ci.ensure("B", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    pronoun_ref = ci.ensure("它", space_id=mem_sid, tier=TIER_SHADOW)   # 预建 pronoun ref（resolve 复用）
    # 预写 (pronoun, B) pr_tn=3（历史 B 常被选）
    for _ in range(3):
        record_pronoun_resolution_decision(b, pronoun_ref=pronoun_ref, antecedent_ref=b_ref)
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True   # 层1 候选源
    try:
        wm._current_segment_refs = [b_ref, a_ref]   # A 最近（k=1 score=2）·B 远（k=0 score=1）
        ant = resolve_pronoun_occurrence(
            es, ci, "它", work_memory=wm, memory_space_id=mem_sid,
            timestamp_seq=1, backend=b)
        assert ant == b_ref              # ★ consumer bonus 翻盘·B 胜（非 recency 的 A）
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved


def test_consumer_gate_off_recency_wins(ctx):
    """对照（bit-identical）：gate OFF 不读 pr_tn·recency 主·A 胜（最近）。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    b_ref = ci.ensure("B", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    pronoun_ref = ci.ensure("它", space_id=mem_sid, tier=TIER_SHADOW)
    for _ in range(3):
        record_pronoun_resolution_decision(b, pronoun_ref=pronoun_ref, antecedent_ref=b_ref)
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True
    try:
        assert gates.PRONOUN_RESOLVE_COUNT_MODE is False   # gate OFF
        wm._current_segment_refs = [b_ref, a_ref]   # A 最近（k=1 score=2）
        ant = resolve_pronoun_occurrence(
            es, ci, "它", work_memory=wm, memory_space_id=mem_sid,
            timestamp_seq=1, backend=b)
        assert ant == a_ref              # ★ gate OFF·recency 主·A 胜（pr_tn 不读）
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved


def test_consumer_cold_start_bit_identical(ctx, resolve_count_on):
    """consumer 冷启动（无历史 pr_tn=0）→ bonus 0 → 候选排序 bit-identical（recency 主）。"""
    b = ctx.core.backend
    es, ci, wm = _setup(ctx)
    mem_sid = ctx.memory_read.space_id
    a_ref = ci.ensure("A", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    b_ref = ci.ensure("B", space_id=ctx.core.space_id, tier=TIER_PRIMARY)
    saved = gates.PRONOUN_INTRASEG_MODE
    gates.PRONOUN_INTRASEG_MODE = True
    try:
        wm._current_segment_refs = [b_ref, a_ref]   # A 最近（k=1 score=2）·无预写历史
        ant = resolve_pronoun_occurrence(
            es, ci, "它", work_memory=wm, memory_space_id=mem_sid,
            timestamp_seq=1, backend=b)
        assert ant == a_ref              # 冷启动·bonus 0·recency 主·A 胜（= gate OFF 行为）
    finally:
        gates.PRONOUN_INTRASEG_MODE = saved


# ============ bit-identical（gate OFF 两跑一致）============

def test_bit_identical_gate_off(ctx):
    """bit-identical：gate OFF 两独立跑·resolve 结果 + count 表状态一致。"""
    def run(backend_cls):
        b = backend_cls()
        bootstrap(b)
        register_pronoun_resolution_count(b)
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
        saved = gates.PRONOUN_INTRASEG_MODE
        gates.PRONOUN_INTRASEG_MODE = True
        try:
            ant = resolve_pronoun_occurrence(
                es, ci, "它们", work_memory=wm, memory_space_id=mem_read.space_id,
                timestamp_seq=1, backend=b)
        finally:
            gates.PRONOUN_INTRASEG_MODE = saved
        rows = b.select(PRONOUN_RESOLUTION_COUNT_TABLE, where=None)
        b.close()
        return ant, len(rows)
    backend_cls = DictBackend if isinstance(ctx.core.backend, DictBackend) else SQLiteBackend
    ant_a, n_a = run(backend_cls)
    ant_b, n_b = run(backend_cls)
    assert ant_a == ant_b                # 两跑解析一致
    assert n_a == n_b == 0               # gate OFF 不写 count·两跑均 0 行
