"""Stage 1 验收门测试：§十五 9 决策全落 + 三空间物理分开 + 纯整数 + discipline。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 1 验收门）：
  - edge schema 12+ 列齐（决策2 D1 宽表）
  - 三空间物理分开复制（决策1 + C5 per-space dump）
  - 记忆两层 dump 跨会话（A4 session_id）
  - per-space dump 独立文件（C5）
  - HotCache 接 prod 无 TypeError（决策7 必修 defer_indexes）
  - MUTABLE_MONOTONE 只增 / append-only 拒 DELETE（决策6/8 discipline）
  - guard_write 守核心表 / audit append-only 链式（决策6/8）
  - 纯整数（_validate_row 拒 float/str 入核心）
  - 确定性有序读（A10·两 backend 同 order_by 同序）
"""
from __future__ import annotations

import os
import tempfile

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.backend import (
    DictBackend, SQLiteBackend, TYPE_INT, register_extension_table,
)
from pure_integer_ai.storage.node_store import (
    NodeStore, register_node_tables, NODE_CONCEPT, TIER_PRIMARY, TIER_SHADOW,
)
from pure_integer_ai.storage.edge_store import (
    EdgeStore, register_edge_table, EDGE_COLUMNS, DEFAULT_STRENGTH,
    SOURCE_CONCEPTNET, SOURCE_BARE_TEXT, EPI_STRUCTURED, SUBTYPE_PURE_ALIAS,
)
from pure_integer_ai.storage.spaces.registry import (
    SpaceRegistry, register_space_table, SPACE_TYPE_CORE, SPACE_TYPE_MEMORY,
    SPACE_TYPE_COMPANION,
)
from pure_integer_ai.storage.spaces.memory_space import (
    MemorySpace, register_memory_table, STATUS_EXPERIENCE, STATUS_CONSOLIDATED,
)
from pure_integer_ai.storage.spaces.companion import CompanionSpace, register_companion_table
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.audit import PersistedAuditLog, register_audit_table
from pure_integer_ai.storage.hot_cache import HotCache
from pure_integer_ai.storage.cold_store import ColdStore, PageRequest
from pure_integer_ai.storage import paths
from pure_integer_ai.crosscut.determinism.audit_event import AuditLog
from pure_integer_ai.crosscut.guards.float_guard import FloatViolation


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    if request.param == "dict":
        b = DictBackend()
    else:
        b = SQLiteBackend(":memory:")
    bootstrap(b)
    yield b
    b.close()


# ---- edge schema D1 ----

def test_edge_schema_d1_columns():
    """决策2 D1 宽表：12+ 列齐（含 source/epistemic_origin/subtype/tier/order_index/role/
    memory_time_attach/base_strength·weight_p/q 不保留）。"""
    cols = {c for c, _ in EDGE_COLUMNS}
    required = {
        "space_id_from", "local_id_from", "space_id_to", "local_id_to",
        "edge_type", "strength", "base_strength", "belief_p", "belief_q",
        "sn", "tn", "tier", "source", "epistemic_origin", "subtype",
        "order_index", "role", "memory_time_attach", "content_version",
    }
    assert required <= cols, f"缺列: {required - cols}"
    # weight_p/weight_q 不保留（决策4）
    assert "weight_p" not in cols and "weight_q" not in cols
    assert len(cols) >= 19


# ---- 纯整数守卫 ----

def test_core_table_rejects_float(backend):
    ns = NodeStore(backend)
    ns.put(1, 1, node_type=NODE_CONCEPT)
    # update strength 用 float 应拒
    with pytest.raises(FloatViolation):
        backend.update("concept_node",
                       where={"space_id": 1, "local_id": 1},
                       set_={"tier": 1.5})


def test_core_table_rejects_str(backend):
    """核心表拒 str（文本入伴随库·守'文本不入核心'）。"""
    with pytest.raises(Exception):
        backend.insert("concept_node", {
            "space_id": 1, "local_id": 1, "type": NODE_CONCEPT,
            "born_granularity": 0, "version_head": 0,
            "tier": 1, "bogus_str": "x",
        })


def test_companion_accepts_text(backend):
    """伴随库 text_assoc 接受 TEXT（非整数合法·决策1）。"""
    reg = SpaceRegistry(backend)
    comp = CompanionSpace.create(reg, "comp1")
    aid = comp.put_text("原输入文本", meta=1)
    assert aid == 1
    items = comp.all_items()
    assert len(items) == 1
    assert items[0]["text"] == "原输入文本"


# ---- discipline: append-only / MUTABLE_MONOTONE ----

def test_append_only_rejects_delete_core(backend):
    """核心表 DELETE 拒（append-only·核心永不删）。"""
    ns = NodeStore(backend)
    ns.put(1, 1, node_type=NODE_CONCEPT)
    with pytest.raises(disc.AppendOnlyViolation):
        backend.delete("concept_node", {"space_id": 1, "local_id": 1})


def test_append_only_rejects_update_appendonly_table(backend):
    """APPEND_ONLY 核心表 UPDATE 拒（def_array 只增）。"""
    backend.insert("def_array", {
        "space_id": 1, "local_id": 1, "order_index": 0,
        "ref_space_id": 1, "ref_local_id": 2,
    })
    with pytest.raises(disc.AppendOnlyViolation):
        backend.update("def_array", {"space_id": 1}, {"order_index": 5})


def test_edge_mutable_monotone_strength(backend):
    """edge MUTABLE_MONOTONE：strength 可 update·delta<0 拒。"""
    es = EdgeStore(backend)
    es.add(space_id_from=1, local_id_from=1, space_id_to=1, local_id_to=2,
           edge_type=EDGE_PRECEDES, strength=5, source=SOURCE_CONCEPTNET, tier=TIER_PRIMARY)
    es.add_strength(space_id_from=1, local_id_from=1, space_id_to=1,
                    local_id_to=2, edge_type=EDGE_PRECEDES, delta=3)
    rows = es.query_from(1, 1)
    assert rows[0]["strength"] == 8
    # base_strength 不变（reward 不调·H4 守 base_strength_unchanged）
    assert rows[0]["base_strength"] == 5
    with pytest.raises(disc.MonotoneViolation):
        es.add_strength(space_id_from=1, local_id_from=1, space_id_to=1,
                        local_id_to=2, edge_type=EDGE_PRECEDES, delta=-1)


def test_edge_type_gate_rejects_unregistered(backend):
    """C9-bis 完备性 #1：edge_type 须在权威表登记·废类型(砍/降字段)/未登记值拒。"""
    es = EdgeStore(backend)
    # 合法 live 类型接纳
    es.add(space_id_from=1, local_id_from=1, space_id_to=1, local_id_to=2,
           edge_type=EDGE_CAUSES, strength=5, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    # 设计 E 砍/降字段类型拒（CONTAINS=2 已砍非 edge_type·ROLE=4 降字段·REALIZES=8 降台账）
    for cut_type in (2, 4, 8):
        with pytest.raises(ValueError):
            es.add(space_id_from=1, local_id_from=1, space_id_to=1, local_id_to=3,
                   edge_type=cut_type, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    # 未登记整数拒
    with pytest.raises(ValueError):
        es.add(space_id_from=1, local_id_from=1, space_id_to=1, local_id_to=3,
               edge_type=99, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)


def test_build_quarantine_link_cross_space_edge(backend):
    """C9-ter：QUARANTINE_LINK 跨 space 检疫边·字段契约（SHADOW/strength=0/SOURCE_QUARANTINE/不接反传）。"""
    from pure_integer_ai.storage.spaces.companion import build_quarantine_link
    from pure_integer_ai.storage.edge_store import SOURCE_QUARANTINE
    from pure_integer_ai.storage.edge_types import EDGE_QUARANTINE_LINK
    es = EdgeStore(backend)
    build_quarantine_link(es, from_companion=(3, 5), to_memory=(2, 9))
    rows = es.query_from(3, 5, EDGE_QUARANTINE_LINK)
    assert len(rows) == 1
    e = rows[0]
    assert e["space_id_to"] == 2 and e["local_id_to"] == 9   # 跨 space（伴随 3 → 记忆 2）
    assert e["source"] == SOURCE_QUARANTINE
    assert e["tier"] == TIER_SHADOW        # 检疫留档非已验证语义边·不进默认 A1/PR
    assert e["strength"] == 0 and e["sn"] == 0 and e["tn"] == 0   # 非学习对象·不接 reward 反传
    assert e["memory_time_attach"] is None  # 跨 space 结构关联非记忆时序经验
    assert e["subtype"] is None and e["order_index"] is None and e["role"] is None


def test_node_tier_monotone(backend):
    """节点 tier MUTABLE_MONOTONE·只升不降。"""
    ns = NodeStore(backend)
    ns.put(1, 1, node_type=NODE_CONCEPT, tier=TIER_SHADOW)
    ns.set_tier(1, 1, TIER_PRIMARY)
    assert ns.get(1, 1)["tier"] == TIER_PRIMARY
    with pytest.raises(disc.MonotoneViolation):
        ns.set_tier(1, 1, TIER_SHADOW)  # 降级拒


def test_memory_status_flip_monotone(backend):
    """memory_item status EXPERIENCE→CONSOLIDATED 单向 flip。"""
    register_memory_table(backend) if "memory_item" not in [t for t in getattr(backend, "_tables", {})] else None
    ms = MemorySpace.__new__(MemorySpace)
    ms.backend = backend
    ms.space_id = 5
    ms.registry = None
    ms.put(1, 123456)
    ms.consolidate(1)
    assert ms.backend.select("memory_item", where={"space_id": 5, "local_id": 1})[0]["status"] == STATUS_CONSOLIDATED
    # 已巩固幂等
    ms.consolidate(1)


# ---- 三空间 + A5 space_name_hash ----

def test_three_spaces_registered(backend):
    """决策1：三空间可注册·A5 type_hash/name_hash 落表。"""
    reg = SpaceRegistry(backend)
    core = AbstractSpace.create(reg, "core")
    mem = MemorySpace.create(reg, "memory_read")
    comp = CompanionSpace.create(reg, "comp1")
    spaces = reg.all_spaces()
    types = {s["type"] for s in spaces}
    assert types == {SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION}
    # A5 hash 非零
    assert all(s["type_hash"] != 0 and s["name_hash"] != 0 for s in spaces)
    # 文本名不入核心 space 表（只 hash）
    assert "name" not in spaces[0]


def test_space_registry_recovers_watermark_and_reuses_identity(backend):
    """重建 registry 后同身份复用，新增空间继续递增且不碰撞。"""
    first = AbstractSpace.create(SpaceRegistry(backend), "core")
    same = AbstractSpace.create(SpaceRegistry(backend), "core")
    second = AbstractSpace.create(SpaceRegistry(backend), "secondary")

    assert same.space_id == first.space_id
    assert second.space_id == first.space_id + 1
    assert [row["space_id"] for row in backend.select(
        "space", order_by="space_id")] == [first.space_id, second.space_id]


def test_space_registry_interleaved_instances_do_not_collide(backend):
    """两个长期存活的 registry 交错注册时也必须读取持久层最新水位。"""
    left = SpaceRegistry(backend)
    right = SpaceRegistry(backend)

    first = AbstractSpace.create(left, "left")
    second = AbstractSpace.create(right, "right")
    third = AbstractSpace.create(left, "third")

    assert (first.space_id, second.space_id, third.space_id) == (1, 2, 3)


def test_companion_lookup_by_hash(backend):
    """伴随库经 hash 反查文本（纯整数热路径先 hash 再反查·不扫全文本）。"""
    reg = SpaceRegistry(backend)
    comp = CompanionSpace.create(reg, "comp1")
    aid = comp.put_text("hello world")
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    th = Hasher("pure_integer_ai.companion.v1").h63("hello world")
    found = comp.lookup_by_hash(th)
    assert len(found) == 1
    assert found[0]["assoc_id"] == aid


# ---- per-space dump (C5) + 三空间物理分开 ----

def test_per_space_dump_separate_files(backend):
    """C5：per-space dump 独立文件·三空间物理分开。"""
    reg = SpaceRegistry(backend)
    core = AbstractSpace.create(reg, "core")  # space_id=1
    mem1 = MemorySpace.create(reg, "mem_read")  # space_id=2
    mem2 = MemorySpace.create(reg, "mem_interact")  # space_id=3（两层物理分开）

    # 各 space 建节点
    ns = NodeStore(backend)
    ns.put(1, 1, node_type=NODE_CONCEPT)
    ns.put(2, 1, node_type=NODE_CONCEPT)
    ns.put(3, 1, node_type=NODE_CONCEPT)

    with tempfile.TemporaryDirectory() as td:
        # 各 space 独立 dump
        for sid in (1, 2, 3):
            rows = backend.select("concept_node", where={"space_id": sid})
            d = paths.ensure_run_dir(td, "run_001")
            with open(paths.space_dump_path(td, "run_001", sid), "w") as f:
                f.write(repr(rows))
        # 三文件独立存在
        dumped = paths.list_space_dumps(td, "run_001")
        assert dumped == [1, 2, 3]
        # 物理分开：删 space 2 的 dump 不影响 1/3
        os.remove(paths.space_dump_path(td, "run_001", 2))
        assert paths.list_space_dumps(td, "run_001") == [1, 3]


def test_per_space_filter_edge_cross_space(backend):
    """C5 filter：跨 space 边在两端 space dump 各留一份（非跨 space 移动）。"""
    es = EdgeStore(backend)
    es.add(space_id_from=1, local_id_from=1, space_id_to=2, local_id_to=1,
           edge_type=EDGE_PRECEDES, source=SOURCE_CONCEPTNET, tier=TIER_PRIMARY)
    all_edges = backend.select("edge")
    in_s1 = paths.filter_rows_for_space(all_edges, 1)
    in_s2 = paths.filter_rows_for_space(all_edges, 2)
    in_s3 = paths.filter_rows_for_space(all_edges, 3)
    assert len(in_s1) == 1 and len(in_s2) == 1  # 跨 space 边两端各留
    assert len(in_s3) == 0


# ---- HotCache 决策7 必修 ----

def test_hot_cache_defer_indexes_no_typeerror(backend):
    """决策7必修：ensure_index 带 defer_indexes kwarg·接 prod 无 TypeError。"""
    hc = HotCache(backend, capacity=8)
    # 旧 bug：缺 defer_indexes kwarg → 传 True 直接 TypeError。现已修。
    hc.ensure_index("edge", ("source",), defer_indexes=True)
    hc.ensure_index("edge", ("tier",), defer_indexes=False)


def test_hot_cache_caches_and_invalidates(backend):
    """HotCache 机制：经 hc 写透传失效同表缓存（接线 cognition 热路径 defer Stage 3+）。"""
    hc = HotCache(backend, capacity=8)
    hc.insert("concept_node", {
        "space_id": 1, "local_id": 1, "type": NODE_CONCEPT,
        "born_granularity": 0, "version_head": 0,
        "tier": TIER_SHADOW,
    })
    r1 = hc.select("concept_node", where={"space_id": 1})
    assert len(r1) == 1
    # 再查命中缓存（拷贝·不改缓存）
    r2 = hc.select("concept_node", where={"space_id": 1})
    assert r2 == r1
    # 写透传失效：经 hc 再插一条→同表缓存失效→重查返 2
    hc.insert("concept_node", {
        "space_id": 1, "local_id": 2, "type": NODE_CONCEPT,
        "born_granularity": 0, "version_head": 0,
        "tier": TIER_SHADOW,
    })
    r3 = hc.select("concept_node", where={"space_id": 1})
    assert len(r3) == 2


# ---- audit append-only 链式 ----

def test_audit_persist_and_verify_chain(backend):
    """audit_event append-only 链式·持久化后验链·rebuild 一致。"""
    log = AuditLog()
    log.append("observe", {"a": 1})
    log.append("reward", {"r": 5})
    log.append("promote", {"id": 7})
    assert log.verify_chain() is True

    pal = PersistedAuditLog(backend)
    n = pal.persist(log)
    assert n == 3
    # 持久化链验链
    assert pal.verify_persisted_chain() is True
    # rebuild 一致（event_hash 序列）
    rebuilt = pal.rebuild_chain()
    assert rebuilt.event_hash_sequence() == log.event_hash_sequence()
    assert rebuilt.verify_chain() is True


def test_audit_append_only_rejects_update(backend):
    """audit_event 是 APPEND_ONLY 核心表·UPDATE/DELETE 拒。"""
    backend.insert("audit_event", {
        "seq": 1, "op": 99, "payload_hash": 0, "prev_hash": 0, "event_hash": 7,
    })
    with pytest.raises(disc.AppendOnlyViolation):
        backend.update("audit_event", {"seq": 1}, {"op": 100})
    with pytest.raises(disc.AppendOnlyViolation):
        backend.delete("audit_event", {"seq": 1})


# ---- register_extension_table（L1 迁移·非核心表） ----

def test_register_extension_table(backend):
    """非核心扩展表注册（L1 迁移·decision8 残债收口）。"""
    register_extension_table(backend, "my_ext",
                             [("id", TYPE_INT), ("val", TYPE_INT)],
                             discipline=disc.DISC_MUTABLE_MONOTONE,
                             indexes=[("id",)])
    backend.insert("my_ext", {"id": 1, "val": 10})
    # 非核心表·DISC_MUTABLE_MONOTONE·update 放行
    backend.update("my_ext", {"id": 1}, {"val": ("+=", 5)})
    rows = backend.select("my_ext", where={"id": 1})
    assert rows[0]["val"] == 15
    # 非核心表 delete 仍受纪律（DISC_MUTABLE_MONOTONE 允许 delete？check_write: delete 非核心按 discipline）
    # MUTABLE_MONOTONE delete 放行（非核心·非 append-only）
    n = backend.delete("my_ext", {"id": 1})
    assert n == 1


# ---- 确定性有序读 A10（两 backend 同 order_by 同序） ----

def test_order_by_consistent_across_backends():
    """A10：同 order_by 两 backend 返同序（续训可复现基础）。"""
    rows_spec = [(3,), (1,), (2,), (1,)]
    seqs = []
    for B in (DictBackend, SQLiteBackend):
        b = B() if B is DictBackend else SQLiteBackend(":memory:")
        bootstrap(b)
        for i, (v,) in enumerate(rows_spec):
            b.insert("concept_node", {
                "space_id": 1, "local_id": i + 1, "type": NODE_CONCEPT,
                "born_granularity": 0, "version_head": 0,
                "tier": v,
            })
        got = [r["tier"] for r in b.select("concept_node", order_by="tier")]
        seqs.append(got)
        b.close()
    assert seqs[0] == seqs[1]  # 两 backend 同序


# ---- cold_store 骨架 ----

def test_cold_store_skeleton():
    cs = ColdStore()
    cs.archive_to_cold([{"x": 1}, {"x": 2}, {"x": 3}])
    assert cs.archived_count() == 3
    page = cs.page(PageRequest(space_id=1, offset=0, limit=2))
    assert len(page.rows) == 2 and page.has_more is True
    page2 = cs.page(PageRequest(space_id=1, offset=2, limit=2))
    assert len(page2.rows) == 1 and page2.has_more is False
