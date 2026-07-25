"""#732 G5-C 记忆项延迟晋升闸测试（反 theater e2e + record_use 接线 + bit-identical）。

覆盖（doc/重来_任务0732_G5晋升闸_设计.md §五反 theater e2e + §三决断2 方案 d）：
  - good vs bad（反 theater 核心·reward>0 consolidate vs reward<0 不 consolidate·status 不同）
  - 假触发防护（空表 + count=0 → 0 consolidate·防 0/0 假达标）
  - mutation 敏感性（monkeypatch record_use sc 恒 0 / 阈值极高 → 0 consolidate·防 hardcode True）
  - record_use 接线 e2e（经 reward_propagate 落点②·验 count/sc 累加 + consolidate flip）
  - bit-identical（DictBackend + SQLiteBackend 两 backend 同 fixture·consolidate 决策一致）

铁律：纯整数（count/sc/率×1000 全整）/ MUTABLE_MONOTONE（consolidate status flip 单向·count/sc 单调增）/
  不写死（PROMOTE_MEM_* oracle 标）/ 不纸面闭合（G5-C 闸真落 code·record_use 接线 + caller 侧 sum 聚合）/
  bit-identical（落点② 不改·record_use 接线不破既有测·sorted(info_key) 守迭代序）。

★反 theater 诚实边界：G5-C 闸真消费 memory_item（record_use 接线 count/sc 累加 + caller 侧 sum 聚合 +
  consolidate flip）·非 theater。good vs bad status 不同（consolidate vs 不 consolidate）= 非假触发。
  ⑧整体仍 FAIL（⑧b defer·非半拉子美化）。G5-C reward 统计非语义正确（#479 墙·stable≠correct）。
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
    MemorySpace, SEG_EPISODIC, SEG_NEGATIVE, STATUS_EXPERIENCE, STATUS_CONSOLIDATED,
)
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.experience_count import register_experience_count, pack_ctx_code
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, INTENT_QUESTION, TERMINAL_REACHED_SINK,
    DOMAIN_MATH, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.training.promote import (
    promote_memory_consolidate, PROMOTE_MEM_REWARD_NUM, PROMOTE_MEM_REWARD_DEN, PROMOTE_MEM_FREQ_MIN,
)


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """bootstrap（memory_item 表 core=True 已注册）+ register experience_count。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def mem_setup():
    """DictBackend + core 空间 + memory_read 空间 + EdgeStore + ConceptIndex。

    返 (backend, core_sid, mem, es, ci)——建 concept + memory_item + propagate。
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


def _prop(pr, reward, es, b, *, memory_read=None):
    propagate_reward(pr, [], reward, _CTX_TAG, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b,
                     memory_read=memory_read)


def _put_episodic(mem, sid, c_lid, *, lid):
    """建 1 行 SEG_EPISODIC memory_item（info_ref=sink）+ record_use(success=True) → count=1/sc=2。"""
    mem.put(lid, 0, seg_type=SEG_EPISODIC, info_ref_space=sid, info_ref_id=c_lid,
            context_tag=pack_ctx_code(*_CTX_TAG), round_id=0)
    mem.record_use(lid, success=True)


def _put_negative(mem, sid, c_lid, *, lid):
    """建 1 行 SEG_NEGATIVE memory_item + record_use(success=False) → count=1/sc=0。"""
    mem.put(lid, 0, seg_type=SEG_NEGATIVE, info_ref_space=sid, info_ref_id=c_lid,
            context_tag=pack_ctx_code(*_CTX_TAG), round_id=0)
    mem.record_use(lid, success=False)


def _consolidated_count(b, mem):
    return len(b.select("memory_item", where={"space_id": mem.space_id, "status": STATUS_CONSOLIDATED}))


# ============ 测试1：good vs bad（反 theater 核心） ============

def test_g5c_good_vs_bad(mem_setup):
    """反 theater 核心：reward>0（SEG_EPISODIC·sc=2 each）consolidate vs reward<0（SEG_NEGATIVE·sc=0）不 consolidate。

    good：3 行 SEG_EPISODIC + record_use(success=True) → sum(count)=3·sum(sc)=6 → cross_ge(6,3,1,1)=6≥3 True
      + sum(count)=3≥FREQ_MIN=3 → consolidate flip 3 行。
    bad：3 行 SEG_NEGATIVE + record_use(success=False) → sum(count)=3·sum(sc)=0 → cross_ge(0,3,1,1)=0≥3 False
      → 不 consolidate。
    **good vs bad status 不同 → 非假触发**（反 theater 锚点）。
    """
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)

    # good：3 行 SEG_EPISODIC
    for i in range(3):
        _put_episodic(mem, sid, c[1], lid=mem.new_local_id())
    n_good = promote_memory_consolidate(b, mem)
    assert n_good == 3
    assert _consolidated_count(b, mem) == 3

    # bad：3 行 SEG_NEGATIVE（同 info_ref·sc=0）
    for i in range(3):
        _put_negative(mem, sid, c[1], lid=mem.new_local_id())
    n_bad = promote_memory_consolidate(b, mem)
    assert n_bad == 0   # sum_sc=0 < sum_count=3·cross_ge(0,3,1,1)=False
    # bad 行仍 EXPERIENCE（未 consolidate）
    assert _consolidated_count(b, mem) == 3   # 仅 good 3 行·bad 0 行


# ============ 测试2：假触发防护（空表 + count=0） ============

def test_g5c_empty_table_no_false_trigger(mem_setup):
    """空表 → 0 consolidate（无候选·防假触发）。"""
    b, sid, mem, es, ci = mem_setup
    n = promote_memory_consolidate(b, mem)
    assert n == 0
    assert _consolidated_count(b, mem) == 0


def test_g5c_zero_count_no_false_trigger(mem_setup):
    """有行但 count=0（record_use 未调·模拟接线前）→ 0 consolidate（sum_count=0 < FREQ_MIN=3·防 0/0 假达标）。"""
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)
    # put 不 record_use → count=0/sc=0
    for i in range(3):
        mem.put(mem.new_local_id(), 0, seg_type=SEG_EPISODIC,
                info_ref_space=sid, info_ref_id=c[1])
    n = promote_memory_consolidate(b, mem)
    assert n == 0   # sum_count=0 < FREQ_MIN=3
    assert _consolidated_count(b, mem) == 0


# ============ 测试3：mutation 敏感性（防 hardcode True） ============

def test_g5c_mutation_record_use_zero_sc(mem_setup, monkeypatch):
    """monkeypatch record_use 让 success_count 恒 0 → good case 0 consolidate（防 hardcode 达标）。"""
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)

    # monkeypatch record_use：count++ 但 success_count 不动（恒 0）
    def no_sc_record_use(self, local_id, *, success):
        self.backend.update("memory_item",
                            where={"space_id": self.space_id, "local_id": local_id},
                            set_={"count": ("+=", 1)})   # success_count 不动

    monkeypatch.setattr(MemorySpace, "record_use", no_sc_record_use)

    # 建 3 行 good + record_use（sc 恒 0）
    for i in range(3):
        lid = mem.new_local_id()
        mem.put(lid, 0, seg_type=SEG_EPISODIC, info_ref_space=sid, info_ref_id=c[1])
        mem.record_use(lid, success=True)   # sc 恒 0
    n = promote_memory_consolidate(b, mem)
    assert n == 0   # sum_sc=0 < sum_count=3·不达·防 hardcode True


def test_g5c_mutation_high_threshold(mem_setup, monkeypatch):
    """monkeypatch PROMOTE_MEM_REWARD_NUM=999（极高阈值）→ good case 0 consolidate（防 hardcode True）。"""
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)

    import pure_integer_ai.training.promote as promote_mod
    monkeypatch.setattr(promote_mod, "PROMOTE_MEM_REWARD_NUM", 999)

    # 建 3 行 good + record_use（sc=2 each）
    for i in range(3):
        _put_episodic(mem, sid, c[1], lid=mem.new_local_id())
    n = promote_memory_consolidate(b, mem)
    assert n == 0   # cross_ge(6,3,999,1)=6≥3*999 False·极高阈值不达


def test_g5c_mutation_freq_min(mem_setup, monkeypatch):
    """monkeypatch PROMOTE_MEM_FREQ_MIN=999（极高频次）→ good case（3 行）0 consolidate（防 hardcode True）。"""
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)

    import pure_integer_ai.training.promote as promote_mod
    monkeypatch.setattr(promote_mod, "PROMOTE_MEM_FREQ_MIN", 999)

    # 建 3 行 good + record_use（sc=2 each·sum_count=3）
    for i in range(3):
        _put_episodic(mem, sid, c[1], lid=mem.new_local_id())
    n = promote_memory_consolidate(b, mem)
    assert n == 0   # sum_count=3 < FREQ_MIN=999·不达


# ============ 测试4：record_use 接线 e2e（经 reward_propagate） ============

def test_g5c_record_use_wiring_e2e(mem_setup):
    """record_use 接线 e2e：3 次 reward_propagate（reward>0·同 sink）→ 3 行 memory_item + count=1/sc=2 each
    → promote_memory_consolidate → 3 CONSOLIDATED。

    验 #732 record_use 接线（reward_propagate:265 if 块内 put 后 record_use）真累加 count/sc。
    反 theater：reward==0 不写（test_propagate_zero_no_write 已验·此处验 record_use 累加 + consolidate flip）。
    """
    b, sid, mem, es, ci = mem_setup
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)

    # 3 次 reward>0 propagate（同 sink·同 info_ref）
    for _ in range(3):
        _prop(pr, 1, es, b, memory_read=mem)

    # 验 record_use 接线：3 行 memory_item·每行 count=1/sc=2
    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_EPISODIC})
    assert len(rows) == 3
    for row in rows:
        assert row["count"] == 1
        assert row["success_count"] == 2
        assert row["status"] == STATUS_EXPERIENCE

    # promote_memory_consolidate → 3 CONSOLIDATED（sum_count=3·sum_sc=6·cross_ge(6,3,1,1)=True）
    n = promote_memory_consolidate(b, mem)
    assert n == 3
    assert _consolidated_count(b, mem) == 3


def test_g5c_record_use_negative_no_consolidate(mem_setup):
    """record_use 接线 e2e 反 theater：3 次 reward<0 propagate（SEG_NEGATIVE·sc=0）→ 不 consolidate。

    bad case：reward<0 → SEG_NEGATIVE + record_use(success=False) → sc=0 → 不 consolidate。
    **good vs bad status 不同 → 非假触发**（reward>0 consolidate vs reward<0 不 consolidate）。
    """
    b, sid, mem, es, ci = mem_setup
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)

    # 3 次 reward<0 propagate
    for _ in range(3):
        _prop(pr, -1, es, b, memory_read=mem)

    rows = b.select("memory_item", where={"space_id": mem.space_id, "seg_type": SEG_NEGATIVE})
    assert len(rows) == 3
    for row in rows:
        assert row["count"] == 1
        assert row["success_count"] == 0   # success=False·sc 不加

    n = promote_memory_consolidate(b, mem)
    assert n == 0   # sum_sc=0 < sum_count=3·不达
    assert _consolidated_count(b, mem) == 0


# ============ 测试5：bit-identical（两 backend 同 fixture） ============

def test_g5c_bit_identical_dict_sqlite():
    """bit-identical：DictBackend + SQLiteBackend 两 backend 同 fixture → consolidate 决策一致。

    验 #732 索引扩（info_ref_space, info_ref_id）+ sorted(info_key) 守跨 backend 一致。
    """
    results = {}
    for be_kind in ("dict", "sqlite"):
        b = DictBackend() if be_kind == "dict" else SQLiteBackend(":memory:")
        try:
            bootstrap(b)
            register_experience_count(b)
            reg = SpaceRegistry(b)
            sp = AbstractSpace.create(reg, "core")
            mem = MemorySpace.create(reg, "memory_read")
            ci = ConceptIndex(b)
            c = ci.ensure("cherry", space_id=sp.space_id)
            # 3 行 good + 1 行 bad（同 info_ref）
            for _ in range(3):
                _put_episodic(mem, sp.space_id, c[1], lid=mem.new_local_id())
            _put_negative(mem, sp.space_id, c[1], lid=mem.new_local_id())
            n = promote_memory_consolidate(b, mem)
            rows = b.select("memory_item", where={"space_id": mem.space_id})
            # 审2 P2-3：验 count/sc 跨 backend 一致（record_use += 1/2 应跨 backend 同）
            sum_count = sum(int(r.get("count", 0) or 0) for r in rows)
            sum_sc = sum(int(r.get("success_count", 0) or 0) for r in rows)
            results[be_kind] = {
                "consolidate_count": n,
                "total_rows": len(rows),
                "consolidated": _consolidated_count(b, mem),
                "experience": len(b.select("memory_item", where={
                    "space_id": mem.space_id, "status": STATUS_EXPERIENCE})),
                "sum_count": sum_count,
                "sum_success_count": sum_sc,
            }
        finally:
            b.close()
    assert results["dict"] == results["sqlite"]   # 跨 backend 一致


def test_g5c_bit_identical_idempotent(mem_setup):
    """bit-identical 幂等：同 fixture 连续调 promote_memory_consolidate 两次 → 第二次 0（已 CONSOLIDATED 跳过）。"""
    b, sid, mem, es, ci = mem_setup
    c = ci.ensure("cherry", space_id=sid)
    for _ in range(3):
        _put_episodic(mem, sid, c[1], lid=mem.new_local_id())
    n1 = promote_memory_consolidate(b, mem)
    assert n1 == 3
    n2 = promote_memory_consolidate(b, mem)   # 第二次·已 CONSOLIDATED 跳过
    assert n2 == 0   # 幂等
    assert _consolidated_count(b, mem) == 3
