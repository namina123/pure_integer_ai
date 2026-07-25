"""阶段1 地基层测试：experience_count 表（§八-bis）+ chapter_seq_table（缺口①·修正分析九v2）。

覆盖（doc/重来_experience_count落地设计指引.md + doc/重来_篇章结构层级设计_缺口①补.md）：
  - experience_count：schema/两源列/R1 符号/base_freq append-only/effective_freq/冷启动/ctx 默认 0/dump
  - chapter_seq_table：schema/attach/read/幂等/冷启动/表未注册向后兼容/APPEND_ONLY 纪律
  - e2e：observe 读 segment.chapter_seq → attach（写路径）+ generate 读 chapter_seq 变化点作 M5 分页候选（反 theater）
  - dump_tables 含两表（1e·跨 run 续训还原）
  - 铁律：纯整数 / MUTABLE_MONOTONE（experience_count）/ APPEND_ONLY（chapter_seq）/ core=False / 不污染 def_array
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap, discipline as disc
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.experience_count import (
    EXPERIENCE_COUNT_TABLE, register_experience_count,
    record_base_freq, record_experience_outcome,
    read_experience_count, read_effective_freq,
)
from pure_integer_ai.storage.chapter_seq import (
    CHAPTER_SEQ_TABLE, register_chapter_seq, attach_chapter_seq,
)
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, SpaceContext, PathData, PathResult,
    STAGE_TRAINING, MODALITY_LANGUAGE, LANG_ZH, DOMAIN_TEXT,
    TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.experiments.formal_train import FormalTrainConfig


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    yield b
    b.close()


@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """建 backend + 三空间 + SpaceContext（observe e2e·镜像 test_stage3 ctx）。"""
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
    )
    yield c
    b.close()


@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + ConceptIndex（generate e2e·镜像 test_stage5 core）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ci
    b.close()


def _seg(tokens, **kw):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


def _dag(sid, *, sink=None, struct_unit_refs=None, topo_layers=None):
    return PathResult(
        path=PathData(edges=[], struct_unit_refs=struct_unit_refs or []),
        terminal=TERMINAL_REACHED_SINK, sink=sink,
        topo_layers=topo_layers or [], convergence={}, source=None,
    )


# ============ experience_count 表 ============

def test_experience_count_schema_discipline(backend):
    """schema 8 列 + MUTABLE_MONOTONE + core=False（observe_tn 方案3 tn路加·镜像 op_confidence 范式）。"""
    register_experience_count(backend)
    meta = backend._tables[EXPERIENCE_COUNT_TABLE]
    assert meta["core"] is False
    assert meta["discipline"] == disc.DISC_MUTABLE_MONOTONE
    assert meta["columns"] == ["space_id", "local_id", "ctx_code", "speaker_code",
                                "base_freq", "e_sn", "e_tn", "observe_tn"]


def test_record_base_freq_creates_row(backend):
    """record_base_freq 建行 (base_freq, 0, 0)·镜像 edge_store.add 写 base_strength 初值。"""
    register_experience_count(backend)
    record_base_freq(backend, ref=(1, 10), base_freq=100)
    assert read_experience_count(backend, (1, 10)) == (100, 0, 0)


def test_record_base_freq_idempotent_skip(backend):
    """base_freq append-only first-write-wins·重写 skip（reward 不调·公约守）。"""
    register_experience_count(backend)
    record_base_freq(backend, ref=(1, 10), base_freq=100)
    record_base_freq(backend, ref=(1, 10), base_freq=999)   # 重写→skip
    assert read_experience_count(backend, (1, 10)) == (100, 0, 0)


def test_record_base_freq_table_unregistered_skip():
    """表未注册（bare fixture）→静默 skip 向后兼容（镜像 record_concept_identity）。"""
    b = DictBackend()
    bootstrap(b)
    record_base_freq(b, ref=(1, 10), base_freq=100)   # 不崩
    assert read_experience_count(b, (1, 10)) is None
    b.close()


def test_record_experience_outcome_success(backend):
    """R1：reward>0→e_sn++&e_tn++（镜像 op_confidence.record_op_outcome）。"""
    register_experience_count(backend)
    record_experience_outcome(backend, ref=(1, 10), reward=5)
    assert read_experience_count(backend, (1, 10)) == (0, 1, 1)


def test_record_experience_outcome_failure(backend):
    """R1：reward≤0→e_tn++ only（e_sn 不降·率自然降）。"""
    register_experience_count(backend)
    record_experience_outcome(backend, ref=(1, 10), reward=0)
    assert read_experience_count(backend, (1, 10)) == (0, 0, 1)
    record_experience_outcome(backend, ref=(1, 10), reward=-1)
    assert read_experience_count(backend, (1, 10)) == (0, 0, 2)


def test_record_experience_outcome_accumulate(backend):
    """R1 累积：success/fail/zero/success 序列→e_sn/e_tn 正确·e_sn 单调。"""
    register_experience_count(backend)
    ref = (1, 10)
    for r in (5, -1, 0, 3):   # success/fail/zero/success
        record_experience_outcome(backend, ref=ref, reward=r)
    # e_sn：reward>0 两次 → 2 · e_tn：四次参与 → 4
    assert read_experience_count(backend, ref) == (0, 2, 4)


def test_record_experience_outcome_does_not_touch_base_freq(backend):
    """reward 路径不调 base_freq（防塌柱① reward CAUSES-only 铁律·base_freq append-only）。"""
    register_experience_count(backend)
    ref = (1, 10)
    record_base_freq(backend, ref=ref, base_freq=100)
    record_experience_outcome(backend, ref=ref, reward=5)
    record_experience_outcome(backend, ref=ref, reward=-1)
    base_freq, e_sn, e_tn = read_experience_count(backend, ref)
    assert base_freq == 100   # reward 未调 base_freq
    assert e_sn == 1 and e_tn == 2


def test_record_base_freq_after_outcome_skipped(backend):
    """异常顺序（reward feed 先建行 base_freq=0）→ record_base_freq skip→base_freq 留 0
    （first-write-wins·诚实降级·断奶后新概念无通识先验只靠 exp 自积累·正常流 base_freq 先注入不撞此）。"""
    register_experience_count(backend)
    ref = (1, 10)
    record_experience_outcome(backend, ref=ref, reward=5)   # 先建行 (base_freq=0, e_sn=1, e_tn=1)
    record_base_freq(backend, ref=ref, base_freq=100)        # 行已存在→幂等 skip
    base_freq, e_sn, e_tn = read_experience_count(backend, ref)
    assert base_freq == 0          # 异常顺序 base_freq 留 0（诚实降级·非 bug）
    assert e_sn == 1 and e_tn == 1
    assert read_effective_freq(backend, ref) == 1            # 0(base) + 1(e_tn)


def test_read_effective_freq(backend):
    """effective_freq = base_freq + e_tn（消费者读·非列·通识基线+经验积累）。"""
    register_experience_count(backend)
    ref = (1, 10)
    record_base_freq(backend, ref=ref, base_freq=100)
    record_experience_outcome(backend, ref=ref, reward=5)   # e_tn=1
    record_experience_outcome(backend, ref=ref, reward=-1)  # e_tn=2
    assert read_effective_freq(backend, ref) == 102   # 100 + 2


def test_read_cold_start(backend):
    """冷启动：无行→read None·effective_freq 0（消费者按 0 处理）。"""
    register_experience_count(backend)
    assert read_experience_count(backend, (1, 99)) is None
    assert read_effective_freq(backend, (1, 99)) == 0


def test_read_table_unregistered_returns_none():
    """表未注册→read None·effective_freq 0（向后兼容·bare fixture 消费者不崩）。"""
    b = DictBackend()
    bootstrap(b)
    assert read_experience_count(b, (1, 10)) is None
    assert read_effective_freq(b, (1, 10)) == 0
    b.close()


def test_experience_count_ctx_speaker_default_zero(backend):
    """第一刀单 key：ctx_code/speaker_code 恒 0（退化·第二刀阶段6 启用复合 key）。"""
    register_experience_count(backend)
    record_experience_outcome(backend, ref=(1, 10), reward=5)
    rows = backend.select(EXPERIENCE_COUNT_TABLE, where={"space_id": 1, "local_id": 10})
    assert rows[0]["ctx_code"] == 0
    assert rows[0]["speaker_code"] == 0


# ============ chapter_seq_table 表 ============

def test_chapter_seq_schema_discipline(backend):
    """schema 5 列 + APPEND_ONLY + core=False（镜像 concept_identity 范式·不复活 CONTAINS）。"""
    register_chapter_seq(backend)
    meta = backend._tables[CHAPTER_SEQ_TABLE]
    assert meta["core"] is False
    assert meta["discipline"] == disc.DISC_APPEND_ONLY
    assert meta["columns"] == ["space_id", "local_id", "chapter_seq",
                                "section_seq", "doc_seq"]


def test_attach_and_read_chapter_seq(backend):
    """attach 落表 → graph read 对齐（镜像 attach_role_seq 语义但独立表）。"""
    register_chapter_seq(backend)
    attach_chapter_seq(backend, ref=(1, 10), chapter_seq=3, section_seq=2)
    g = ConceptGraph(backend)
    assert g.read_chapter_seq((1, 10)) == (3, 2, 0)   # doc_seq 默认 0


def test_attach_chapter_seq_doc_seq(backend):
    """doc_seq 首版单文档=0·可显式传（多文档远期）。"""
    register_chapter_seq(backend)
    attach_chapter_seq(backend, ref=(1, 10), chapter_seq=1, section_seq=1, doc_seq=5)
    g = ConceptGraph(backend)
    assert g.read_chapter_seq((1, 10)) == (1, 1, 5)


def test_attach_chapter_seq_idempotent(backend):
    """APPEND_ONLY：同 struct_ref 重 attach→幂等 skip（章节标记稳定写一次）。"""
    register_chapter_seq(backend)
    attach_chapter_seq(backend, ref=(1, 10), chapter_seq=1, section_seq=1)
    attach_chapter_seq(backend, ref=(1, 10), chapter_seq=2, section_seq=2)   # 重写→skip
    rows = backend.select(CHAPTER_SEQ_TABLE, where={"space_id": 1, "local_id": 10})
    assert len(rows) == 1                    # 无重复行
    assert (rows[0]["chapter_seq"], rows[0]["section_seq"]) == (1, 1)   # first-write-wins


def test_attach_chapter_seq_table_unregistered_skip():
    """表未注册（bare fixture / observe 热路径）→静默 skip 向后兼容（不崩）。"""
    b = DictBackend()
    bootstrap(b)
    attach_chapter_seq(b, ref=(1, 10), chapter_seq=1, section_seq=1)   # 不崩
    b.close()


def test_read_chapter_seq_cold_start(backend):
    """无行→None（无标记文本·退化同流水账·章节承载 defer 钥匙①）。"""
    register_chapter_seq(backend)
    g = ConceptGraph(backend)
    assert g.read_chapter_seq((1, 99)) is None


def test_read_chapter_seq_table_unregistered_returns_none():
    """表未注册→read None（bare ConceptGraph 向后兼容·generate 热路径不崩）。"""
    b = DictBackend()
    bootstrap(b)
    g = ConceptGraph(b)
    assert g.read_chapter_seq((1, 10)) is None
    b.close()


def test_chapter_seq_append_only_rejects_update(backend):
    """APPEND_ONLY 纪律：update 被拒（章节标记写一次·改走 INSERT/版本化非 update）。"""
    register_chapter_seq(backend)
    attach_chapter_seq(backend, ref=(1, 10), chapter_seq=1, section_seq=1)
    with pytest.raises(disc.AppendOnlyViolation):
        backend.update(CHAPTER_SEQ_TABLE,
                       where={"space_id": 1, "local_id": 10},
                       set_={"chapter_seq": 2})


# ============ e2e：observe 写路径 ============

def test_observe_attaches_chapter_seq(ctx):
    """observe 读 segment.chapter_seq → attach_chapter_seq 落表（写路径·反 theater·使 attach 非死码）。"""
    register_chapter_seq(ctx.core.backend)
    seg = _seg(["小明", "吃", "苹果"], chapter_seq=3, section_seq=2)
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    res = ObservePipeline(ctx).observe(raw)
    rows = ctx.core.backend.select(CHAPTER_SEQ_TABLE, where={})
    assert len(rows) == 1
    assert (rows[0]["chapter_seq"], rows[0]["section_seq"], rows[0]["doc_seq"]) == (3, 2, 0)
    # graph read 对齐 struct_ref
    g = ConceptGraph(ctx.core.backend)
    assert g.read_chapter_seq(res.struct_refs[0]) == (3, 2, 0)


def test_observe_no_chapter_seq_no_row(ctx):
    """segment.chapter_seq=0（默认·无标记主流文本）→不调 attach→无行（向后兼容 bit-identical）。"""
    register_chapter_seq(ctx.core.backend)
    seg = _seg(["小明", "吃", "苹果"])   # chapter_seq 默认 0
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)
    rows = ctx.core.backend.select(CHAPTER_SEQ_TABLE, where={})
    assert len(rows) == 0


def test_observe_chapter_seq_table_unregistered_no_crash(ctx):
    """chapter_seq_table 未注册（bare ctx）+ segment 有 chapter_seq→attach try/except skip 不崩。"""
    seg = _seg(["小明", "吃", "苹果"], chapter_seq=1, section_seq=1)
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)   # 不崩（attach 内 try/except KeyError skip）


# ============ e2e：generate 9d 章节边界分页候选（反 theater） ============

def test_generate_chapter_boundary_triggers_carry(core, monkeypatch):
    """9d 反 theater：chapter_seq 变化点作 M5 分页边界候选→carry 触发（证明属性可读有人读）。"""
    b, sid, es, ci = core
    register_chapter_seq(b)
    U1 = ci.ensure("u1", space_id=sid, tier=TIER_PRIMARY)
    U2 = ci.ensure("u2", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U1, [1])
    attach_role_seq(b, U2, [2])
    attach_chapter_seq(b, ref=U1, chapter_seq=1, section_seq=1)   # 章 1
    attach_chapter_seq(b, ref=U2, chapter_seq=2, section_seq=1)   # 章 2（boundary）
    g = ConceptGraph(b, surface_of=lambda r: {U1: "u1", U2: "u2"}.get(r))
    dag = _dag(sid, sink=U2, struct_unit_refs=[U1, U2], topo_layers=[[U1, U2]])

    import pure_integer_ai.cognition.result.generate as gen_mod
    calls = []
    monkeypatch.setattr(gen_mod, "carry_to_workmem",
                        lambda wm, parts: calls.append(len(parts)))
    out = generate_output(dag, g, WorkMemory(), LANG_ZH)

    assert len(out.parts) == 2                  # 两单元都产出
    assert len(calls) >= 1                       # 章边界（U1 ch1→U2 ch2）触发 carry


def test_generate_no_chapter_seq_no_boundary_carry(core, monkeypatch):
    """回归 bit-identical：无 chapter_seq（表未注册）→read 返 None→无 boundary→无 carry。"""
    b, sid, es, ci = core
    # 不注册 chapter_seq_table（bare·向后兼容）
    U1 = ci.ensure("u1", space_id=sid, tier=TIER_PRIMARY)
    U2 = ci.ensure("u2", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, U1, [1])
    attach_role_seq(b, U2, [2])
    g = ConceptGraph(b, surface_of=lambda r: {U1: "u1", U2: "u2"}.get(r))
    dag = _dag(sid, sink=U2, struct_unit_refs=[U1, U2], topo_layers=[[U1, U2]])

    import pure_integer_ai.cognition.result.generate as gen_mod
    calls = []
    monkeypatch.setattr(gen_mod, "carry_to_workmem",
                        lambda wm, parts: calls.append(1))
    out = generate_output(dag, g, WorkMemory(), LANG_ZH)

    assert len(out.parts) == 2
    # 无 chapter_seq → chap_no 恒 0 → 无 boundary · LAYER_UNIT_CAP(256) 未达 → carry 零触发
    assert len(calls) == 0


# ============ dump_tables 含两表（1e·跨 run 续训还原） ============

def test_dump_tables_includes_both():
    """FormalTrainConfig.dump_tables 含两表（跨 run 续训还原·load_run table-agnostic 自动载）。"""
    tables = FormalTrainConfig.dump_tables
    assert EXPERIENCE_COUNT_TABLE in tables
    assert CHAPTER_SEQ_TABLE in tables
