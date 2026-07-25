"""旧 memory_item schema 与非语言 reward sink 迁移核证。

覆盖（plan velvet-juggling-garden.md + doc/重来·主线重审与重画.md §十三 + §十五决策1 + doc/重来_纠偏设计全实施落地计划_修正分析十二.md §3 阶段11）：
  - 11c schema：memory_item 加 5 列（seg_type/info_ref_space/info_ref_id/context_tag/round_id）+ SEG_ 常量 + put 退化
  - 11a 实例化：make_train_context 挂 TrainContext 两层 MemorySpace / SpaceContext 训练期守 None（bit-identical）
  - 非语言兼容写：reward>0 SEG_EPISODIC / reward<0 SEG_NEGATIVE / reward==0 不写
  - M-00 边界：语言 scalar reward 不写任何长期对象
  - bit-identical：memory_read=None → 既有行为不变（落点① experience_count feed 仍活）
  - 反 theater：写活（no-op→真 insert）vs 不写对比 + info_ref 单 sink 两列（与 experience_count 聚合正交）·消费者 defer 诚实标

铁律：纯整数（5 新列 TYPE_INT·content_hash=0 占位）/ 单向依赖（reward_propagate L4→memory_space L0 向下）/
  MUTABLE_MONOTONE（memory_item 加列不改 status flip 单调）/ append-only（每 episode new_local_id 写一次）/
  不写死（SEG_/STATUS_ 常量）/ 最少边（不建记忆边·§十五决策1）/ bit-identical（默认 0/None 退化）。

★反 theater 诚实边界（plan §反 theater）：11d 写活 + 读 defer（G5 晋升闸未落 code + tri_space 中环五连断 4/5）·
  非伪闭环。锚 = no-op→真 insert 行为变 + 兑现"训练期建阅读记忆种子"（line998-1001）·非"中环激活"。
★关键决断（用户 2026-07-05）：(1) 真写+消费者 defer 诚实标（doc 第一刀本意·B1 layer-defer 范式）·
  (2) info_ref 单 sink 两列（与 experience_count 概念聚合正交不重复·最少冗余）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import (
    MemorySpace, SEG_EPISODIC, SEG_NEGATIVE, STATUS_EXPERIENCE,
)
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.experience_count import register_experience_count, pack_ctx_code
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_MATH, DOMAIN_TEXT, INTENT_QUESTION, MODALITY_ARITH,
    MODALITY_LANGUAGE, PathData, PathResult, TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.experiments.formal_train import make_train_context, _build_space_ctx, TrainContext


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """bootstrap（memory_item 表 core=True 已注册）+ register experience_count（落点① 对照用）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def core_mem():
    """DictBackend + core 空间 + memory_read 空间 + EdgeStore + ConceptIndex。

    返 (backend, core_sid, mem, es, ci)——建 concept + CAUSES 边 + propagate（memory_read=mem）。
    """
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    mem = MemorySpace.create(reg, "memory_read")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, mem, es, ci
    b.close()


# ---- helpers ----

def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           sn=sn, tn=tn)


def _path_result(edges, *, sink=None):
    pd = PathData()
    pd.edges = list(edges)
    return PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=sink)


_CTX_TAG = (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)
_LANGUAGE_CTX_TAG = (
    DOMAIN_TEXT, MODALITY_LANGUAGE, 0, INTENT_QUESTION)


def _prop(pr, reward, es, b, *, memory_read=None, workmem=None,
          ctx_tag=_CTX_TAG):
    """以指定模态调用旧 scalar reward 传播入口。"""
    propagate_reward(pr, [], reward, ctx_tag, INTENT_QUESTION,
                     workmem or WorkMemory(), edge_store=es, backend=b,
                     memory_read=memory_read)


# ============ 11c schema 扩列 ============

def test_seg_constants():
    """SEG_ 段类型常量值（落点② reward 符号契约·reward_propagate.py 落地② 用）。"""
    assert SEG_EPISODIC == 1   # reward>0 正经验
    assert SEG_NEGATIVE == 2   # reward<0 负经验


def test_memory_item_schema_has_new_columns(backend):
    """11c：memory_item 表含 5 新列（put 后 select 行 keys 含）。"""
    reg = SpaceRegistry(backend)
    mem = MemorySpace.create(reg, "mem_read")
    mem.put(1, 123456)
    row = backend.select("memory_item", where={"space_id": mem.space_id, "local_id": 1})[0]
    for col in ("seg_type", "info_ref_space", "info_ref_id", "context_tag", "round_id"):
        assert col in row, f"memory_item 缺 11c 扩列 {col}"


def test_put_default_zero_bit_identical(backend):
    """11c bit-identical：ms.put(local_id, content_hash) 退化 → 5 新列全 0（回归 test_stage1:203 范式）。"""
    reg = SpaceRegistry(backend)
    mem = MemorySpace.create(reg, "mem_read")
    mem.put(1, 123456)   # 既有调用签名·不传新参
    row = backend.select("memory_item", where={"space_id": mem.space_id, "local_id": 1})[0]
    assert row["status"] == STATUS_EXPERIENCE
    assert row["seg_type"] == 0
    assert row["info_ref_space"] == 0
    assert row["info_ref_id"] == 0
    assert row["context_tag"] == 0
    assert row["round_id"] == 0


def test_put_with_seg_info(backend):
    """11c：put 带新参 → 5 列正确写入（落点② reward 写用）。"""
    reg = SpaceRegistry(backend)
    mem = MemorySpace.create(reg, "mem_read")
    mem.put(2, 789, seg_type=SEG_EPISODIC, info_ref_space=3, info_ref_id=5,
            context_tag=100, round_id=7)
    row = backend.select("memory_item", where={"space_id": mem.space_id, "local_id": 2})[0]
    assert row["seg_type"] == SEG_EPISODIC
    assert row["info_ref_space"] == 3
    assert row["info_ref_id"] == 5
    assert row["context_tag"] == 100
    assert row["round_id"] == 7
    assert row["content_hash"] == 789


# ============ 11a MemorySpace 生产实例化 ============

def test_make_train_context_instantiates_memory():
    """11a：make_train_context → ctx.memory_read/memory_interact 非 None + 两 space_id 独立（两层物理分开）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    assert ctx.memory_read is not None
    assert ctx.memory_interact is not None
    assert ctx.memory_read.space_id != ctx.memory_interact.space_id   # 两层物理分开铁律
    assert ctx.memory_read.space_id != ctx.core_space.space_id
    b.close()


def test_build_space_ctx_training_none():
    """11a bit-identical 守卫：_build_space_ctx 训练期 SpaceContext.memory_read 守 None + memory_active False。

    守 observe:93 bit-identical（训练期核心养洁净）·memory_read 进 SpaceContext 是 Stage 5/训练后阅读层的事。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    sctx = _build_space_ctx(ctx)
    assert sctx.memory_read is None        # 训练期守 None（不破 observe）
    assert sctx.memory_interact is None
    assert sctx.memory_active is False     # memory_active 不动（Stage 5 独立刀）
    b.close()


# ============ 11d reward_propagate 落点② 真写 ============

def test_propagate_positive_writes_episodic(core_mem):
    """11d：reward>0 → memory_item 写 1 条 SEG_EPISODIC·info_ref=sink 两列·content_hash=0。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, 1, es, b, memory_read=mem)
    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})
    assert len(rows) == 1
    assert rows[0]["info_ref_space"] == c[0]
    assert rows[0]["info_ref_id"] == c[1]
    assert rows[0]["content_hash"] == 0          # 留 0 占位（无 surface）
    assert rows[0]["context_tag"] == pack_ctx_code(*_CTX_TAG)


def test_propagate_negative_writes_negative(core_mem):
    """11d：reward<0 → memory_item 写 SEG_NEGATIVE（死路负经验）。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, -5, es, b, memory_read=mem)
    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_NEGATIVE})
    assert len(rows) == 1
    # SEG_EPISODIC 不写（reward<0）
    assert len(b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})) == 0


def test_propagate_zero_no_write(core_mem):
    """11d：reward==0（judge veto·非经验信号）→ 不写 memory_item（守记忆洁净）。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, 0, es, b, memory_read=mem)
    rows = b.select("memory_item", where={"space_id": mem.space_id})
    assert len(rows) == 0    # veto 不污染记忆


def test_language_scalar_reward_writes_no_long_term_object(core_mem):
    """M-00：语言 scalar reward 不写 edge、aggregate 或 memory_item。"""
    from pure_integer_ai.storage.experience_count import read_experience_count
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    before = b.select("edge")

    _prop(pr, 1, es, b, memory_read=mem, ctx_tag=_LANGUAGE_CTX_TAG)

    assert b.select("edge") == before
    assert read_experience_count(
        b, a, ctx_code=pack_ctx_code(*_LANGUAGE_CTX_TAG)) is None
    assert b.select("memory_item", where={"space_id": mem.space_id}) == []


def test_propagate_no_sink_no_write(core_mem):
    """11d：sink=None → 不写（无概念锚·info_ref 无着落）。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=None)   # 无 sink
    _prop(pr, 1, es, b, memory_read=mem)
    rows = b.select("memory_item", where={"space_id": mem.space_id})
    assert len(rows) == 0


def test_propagate_memory_none_degrade(core_mem):
    """11d bit-identical：memory_read=None → 不写 + 不抛（bare fixture 退化·既有行为不变）。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, 1, es, b, memory_read=None)    # 不抛
    rows = b.select("memory_item", where={"space_id": mem.space_id})
    assert len(rows) == 0    # memory_read=None 不写


def test_info_ref_single_sink_not_chain(core_mem):
    """11d 反重复：info_ref 单 sink 两列·非 struct_unit_refs 链（experience_count 落点① 已聚合概念集）。

    memory_item 是 episode 单条记录·info_ref=sink（episode 终点目标）·struct_unit_refs 概念维聚合
    已由 experience_count 落点① 负责（concept_targets=CAUSES 端点+sink+struct_unit_refs·阶段2 已活）·
    memory_item 不重复。一 episode 一条 memory_item（非每概念一条）。
    """
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    struct1 = ci.ensure("struct1", space_id=sid)
    struct2 = ci.ensure("struct2", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    pr.path.struct_unit_refs = [struct1, struct2]   # 多 struct refs·不进 memory_item
    _prop(pr, 1, es, b, memory_read=mem)
    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})
    assert len(rows) == 1                          # 一 episode 一条·非每 struct ref 一条
    assert rows[0]["info_ref_space"] == c[0]       # info_ref = sink 单点·非 struct 链
    assert rows[0]["info_ref_id"] == c[1]


def test_propagate_round_id_from_workmem(core_mem):
    """11d：memory_item round_id 从 workmem.round_id（G5 回溯时序锚）。"""
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    wm = WorkMemory(round_id=42)
    _prop(pr, 1, es, b, memory_read=mem, workmem=wm)
    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})
    assert rows[0]["round_id"] == 42


# ============ bit-identical + 反 theater ============

def test_bit_identical_experience_count_feed_alive(core_mem):
    """11d 反 theater 对照：memory_read=None 时落点① experience_count feed 仍活（11d 不破既有）。

    memory_read=None → 落点② 不写 memory_item·但落点① experience_count feed（阶段2 已活）bit-identical。
    证明 11d 是叠加（memory_read 非 None 才写②）非替换。
    """
    from pure_integer_ai.storage.experience_count import read_experience_count
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, 1, es, b, memory_read=None)   # memory_read=None
    # 落点① experience_count feed 仍活（bit-identical·a 端点 e_sn=1 e_tn=1）
    got = read_experience_count(b, a, ctx_code=pack_ctx_code(*_CTX_TAG))
    assert got == (0, 1, 1)


def test_write_is_real_insert_not_theater(core_mem):
    """11d 反 theater 主锚：memory_read 非 None + reward>0 → memory_item 真有数据（no-op→真 insert 行为变）。

    对照 test_propagate_memory_none_degrade（memory_read=None 不写）：同 path 同 reward·
    memory_read 非 None 写 1 条·None 写 0 条 = 真行为变（非 theater 标签）。
    消费者 defer（G5 闸/tri_space 中环）诚实标在 docstring·非本测试范围。
    """
    b, sid, mem, es, ci = core_mem
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    # memory_read=None：0 条
    _prop(pr, 1, es, b, memory_read=None)
    assert len(b.select("memory_item", where={"space_id": mem.space_id})) == 0
    # memory_read 非 None：1 条（真 insert）
    _prop(pr, 1, es, b, memory_read=mem)
    assert len(b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})) == 1
