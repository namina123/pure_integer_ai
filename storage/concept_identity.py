"""storage.concept_identity — 概念身份索引持久化（跨 run _index 重建·Task #475·§8.7-idx）。

ConceptIndex._index 是 in-memory run-scoped（concept_index.py:43）·load_run 还原 concept_node 行
但**不重建 _index** → 载入算子不可 inline（_try_inline_learned.lookup 返 None）+ observe 续训后建
重复概念点（latent corrupt）。本表持久化 (space_id, local_id, content_hash) → ConceptIndex lazy
扫表重建 _index（跨 run identity 闭环）。

**为何独立表非 concept_node 加列**（决策3·§8.7-idx）：concept_node 是**纯整数**核心表（决策3·身份靠
content_hash·无 hash/text 列）。加 content_hash 列 = core 迁移 + backfill = 触决策3（破坏纯整数纪律）。
独立 core=False 扩展表（同 composes_attr/op_confidence 范式）**不触决策3**·concept_node 保持纯整数。

**content_hash 是 surface 派生**（Hasher.h63·deterministic）·持久化是重建机制非语义变更。表
DISC_APPEND_ONLY（identity 写一次 per concept·(space_id, local_id) 唯一·hash 不变）。

铁律：纯整数（content_hash h63 + local_id 全 int）/ 确定性（无重复 hash·序无关 bit-identical）/
单向依赖（L0 storage·cognition L4 ConceptIndex 读写皆向下）/ APPEND_ONLY（写一次·不 update/delete）/
不写死（schema 元定义列）/ 决策3 守（concept_node 纯整数不动）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT, register_extension_table

CONCEPT_IDENTITY_TABLE = "concept_identity"

_CONCEPT_IDENTITY_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("content_hash", TYPE_INT),
]
_CONCEPT_IDENTITY_INDEXES = [
    ("space_id", "local_id"),      # PK：per concept 写一次（ensure 幂等门）
    ("space_id", "content_hash"),  # lazy 重建扫：hash → local_id
]


def register_concept_identity(backend: StorageBackend) -> None:
    """注册 concept_identity 扩展表（core=False·DISC_APPEND_ONLY·启动/用前调·幂等）。"""
    register_extension_table(backend, CONCEPT_IDENTITY_TABLE,
                             _CONCEPT_IDENTITY_COLUMNS,
                             disc.DISC_APPEND_ONLY, _CONCEPT_IDENTITY_INDEXES)


def load_space_identity(backend: StorageBackend,
                        space_id: int) -> dict[int, int]:
    """扫某 space 的 concept_identity → {content_hash: local_id}（ConceptIndex lazy 重建用）。

    返空 dict = 表未注册（bare fixture·caller 退既 有 ensure 建 _index·向后兼容）或该 space 无行。
    **确定性**：同 space 内 (content_hash → local_id) 唯一（ensure dedup·无重复 hash）·故 select
    序无关·bit-identical。setdefault first-wins 是防御（理论无重复）。
    """
    assert_int(space_id, _where="load_space_identity.space_id")
    try:
        rows = backend.select(CONCEPT_IDENTITY_TABLE, where={"space_id": space_id})
    except KeyError:
        return {}   # 表未注册（bare fixture 未 register_concept_identity）·向后兼容
    space_map: dict[int, int] = {}
    for r in rows:
        space_map.setdefault(r["content_hash"], r["local_id"])   # first-wins·无重复 hash 故序无关
    return space_map


def record_concept_identity(backend: StorageBackend, *,
                            space_id: int, local_id: int,
                            content_hash: int) -> None:
    """持久化 (space_id, local_id, content_hash)·跨 run _index 重建（ensure 新建概念点后调·best-effort）。

    幂等：(space_id, local_id) 已有 → skip（APPEND_ONLY·identity 写一次）。表未注册（bare fixture）
    → KeyError 静默 skip（向后兼容·_index 仍由 ensure 内存建）。
    """
    assert_int(space_id, local_id, content_hash,
               _where="record_concept_identity")
    try:
        existing = backend.select(CONCEPT_IDENTITY_TABLE, where={
            "space_id": space_id, "local_id": local_id,
        }, limit=1)
    except KeyError:
        return   # 表未注册（bare fixture）·向后兼容 skip
    if existing:
        return   # 幂等：该概念点 identity 已持久化（APPEND_ONLY·不重写）
    backend.insert(CONCEPT_IDENTITY_TABLE, {
        "space_id": space_id, "local_id": local_id,
        "content_hash": content_hash,
    })
