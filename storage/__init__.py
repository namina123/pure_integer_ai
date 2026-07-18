"""storage — 存储抽象层（依赖 crosscut·无上层依赖·§十五 9 决策全落）。

模块：
  discipline    DISC 纪律枚举 + 违例 + CORE_TABLES + register_extension_table（L1 迁移）
  backend       StorageBackend 协议（无 raw SQL）+ DictBackend + SQLiteBackend（首版宿主）
  node_store    concept_node（决策3扩列）+ def_array + assoc_table
  edge_store    edge 统一宽表 D1（决策2 + 决策4 weight_p/q 不保留）
  spaces/       三空间（决策1）+ A5 space_name_hash
  audit         audit_event 表（挂 crosscut.audit_event 链）
  hot_cache     HotCache 查询结果 LRU（决策7·修 defer_indexes 必修）
  cold_store    ColdStore 真分页骨架（defer·决策7）
  paths         per-space dump 路径（C5·三空间物理分开）

铁律（§7.3 + §十五决策8）：cognition 层只经 backend 抽象访问·绝不写 raw SQL；
纯整数（_validate_row 拒 float）；MUTABLE_MONOTONE / append-only；确定性有序读（A10）。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import StorageBackend


def bootstrap(backend: StorageBackend) -> None:
    """注册全部核心表 + 伴随扩展表（启动调一次·幂等）。

    顺序：space/id_pool 经 SpaceRegistry 用前先建；此处建表 schema。
    cognition 层后续只经 backend 抽象访问·绝不写 raw SQL。
    """
    from pure_integer_ai.storage.spaces.registry import register_space_table
    from pure_integer_ai.storage.node_store import register_node_tables
    from pure_integer_ai.storage.edge_store import register_edge_table
    from pure_integer_ai.storage.spaces.memory_space import register_memory_table
    from pure_integer_ai.storage.spaces.companion import register_companion_table
    from pure_integer_ai.storage.audit import register_audit_table

    register_space_table(backend)
    register_node_tables(backend)
    register_edge_table(backend)
    register_memory_table(backend)
    register_companion_table(backend)
    register_audit_table(backend)


__all__ = ["bootstrap"]

