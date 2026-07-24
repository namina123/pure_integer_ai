"""storage — 存储抽象层（依赖 crosscut·无上层依赖·§十五 9 决策全落）。

模块：
  discipline    DISC 纪律枚举 + 违例 + CORE_TABLES + register_extension_table（L1 迁移）
  backend       StorageBackend 协议（无 raw SQL）+ DictBackend + SQLiteBackend（首版宿主）
  node_store    concept_node（决策3扩列）+ def_array + assoc_table
  edge_store    edge 统一宽表 D1（决策2 + 决策4 weight_p/q 不保留）
  spaces/       三空间（决策1）+ A5 space_name_hash
  audit         audit_event 表（挂 crosscut.audit_event 链）
  hot_cache     HotCache 查询结果 LRU（决策7·修 defer_indexes 必修）
  cold_store    K-02 前兼容保留的旧列表占位，不构成真分页
  paths         per-space dump 路径（C5·三空间物理分开）
  segment_dependency  run/location/增量 segment 共用的完整依赖身份
  sealed_segment  开放 hot delta、纯整数记录和不可变段
  segment_repository  capability 协商后的 seal-last 物理对象仓库
  segment_commit  首次发布、迁移和 compaction 共用提交阶段
  segment_cache  page-in/prefetch 与 clean/dirty 淘汰
  tiered_segment_store  location epoch、稳定续页和 reader 回收屏障
  recovery_protocol  schema、segment、manifest、迁移和故障注入协议
  recovery_package   终 dump 原子发布、幂等加载和失败回滚

铁律（§7.3 + §十五决策8）：cognition 层只经 backend 抽象访问·绝不写 raw SQL；
纯整数（_validate_row 拒 float）；MUTABLE_MONOTONE / append-only；确定性有序读（A10）。
"""
from __future__ import annotations

from typing import Callable

from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.placement import TemperatureProfile
from pure_integer_ai.storage.segment_repository import BackendObjectRepository
from pure_integer_ai.storage.storage_role import StorageRoleRegistry
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore


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
    from pure_integer_ai.storage.assertion_identity import (
        register_assertion_identity_tables,
    )
    from pure_integer_ai.storage.assertion_record import (
        register_assertion_record_tables,
    )
    from pure_integer_ai.storage.graph_object import register_graph_object_table
    from pure_integer_ai.storage.graph_statement import (
        register_graph_statement_table,
    )
    from pure_integer_ai.storage.memory_overlay import (
        register_memory_overlay_table,
    )
    from pure_integer_ai.storage.memory_event import register_memory_event_table
    from pure_integer_ai.storage.memory_aggregate import (
        register_memory_aggregate_tables,
    )
    from pure_integer_ai.storage.memory_batch import register_memory_batch_table
    from pure_integer_ai.storage.occurrence import register_occurrence_tables
    from pure_integer_ai.storage.source_record import register_source_record_table
    from pure_integer_ai.storage.span import register_span_tables
    from pure_integer_ai.storage.training_candidate_event import (
        register_training_candidate_event_tables,
    )
    from pure_integer_ai.storage.curriculum_mastery import (
        register_curriculum_mastery_tables,
    )

    register_space_table(backend)
    register_node_tables(backend)
    register_edge_table(backend)
    register_memory_table(backend)
    register_companion_table(backend)
    register_audit_table(backend)
    register_assertion_identity_tables(backend)
    register_assertion_record_tables(backend)
    register_graph_object_table(backend)
    register_graph_statement_table(backend)
    register_memory_overlay_table(backend)
    register_memory_event_table(backend)
    register_memory_aggregate_tables(backend)
    register_memory_batch_table(backend)
    register_source_record_table(backend)
    register_occurrence_tables(backend)
    register_span_tables(backend)
    register_training_candidate_event_tables(backend)
    register_curriculum_mastery_tables(backend)


def build_storage_role_registry() -> StorageRoleRegistry:
    """构造当前已接入的存储角色注册表，不创建全局可变状态。"""
    from pure_integer_ai.storage.memory_aggregate import (
        MEMORY_AGGREGATE_STORAGE_DESCRIPTOR,
    )
    from pure_integer_ai.storage.memory_event import (
        MEMORY_EVENT_STORAGE_DESCRIPTOR,
    )
    from pure_integer_ai.storage.memory_forget import (
        MEMORY_FORGET_COMMIT_DESCRIPTOR,
        MEMORY_FORGET_SET_DESCRIPTOR,
    )
    from pure_integer_ai.storage.memory_query_projection import (
        MEMORY_QUERY_PROJECTION_DESCRIPTOR,
    )
    from pure_integer_ai.storage.memory_batch import (
        MEMORY_BATCH_ACTIVATION_DESCRIPTOR,
        MEMORY_BATCH_CORE_DEPENDENCY,
        MEMORY_BATCH_EVENT_DESCRIPTOR,
        MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR,
        MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR,
        MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR,
        MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR,
        MEMORY_BATCH_ROLLBACK_DESCRIPTOR,
        MEMORY_BATCH_SOURCE_DEPENDENCY,
    )

    registry = StorageRoleRegistry()
    registry.register(MEMORY_EVENT_STORAGE_DESCRIPTOR)
    registry.register(MEMORY_AGGREGATE_STORAGE_DESCRIPTOR)
    registry.register(MEMORY_BATCH_CORE_DEPENDENCY)
    registry.register(MEMORY_BATCH_SOURCE_DEPENDENCY)
    registry.register(MEMORY_BATCH_EVENT_DESCRIPTOR)
    registry.register(MEMORY_BATCH_ACTIVATION_DESCRIPTOR)
    registry.register(MEMORY_BATCH_ROLLBACK_DESCRIPTOR)
    registry.register(MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR)
    registry.register(MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR)
    registry.register(MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR)
    registry.register(MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR)
    registry.register(MEMORY_FORGET_SET_DESCRIPTOR)
    registry.register(MEMORY_FORGET_COMMIT_DESCRIPTOR)
    registry.register(MEMORY_QUERY_PROJECTION_DESCRIPTOR)
    return registry


def build_tiered_segment_store(
        backend: StorageBackend,
        registry: StorageRoleRegistry,
        temperature_profile: TemperatureProfile,
        *,
        index_key_fn: Callable[[int, tuple[int, ...]], int] | None = None,
        ) -> TieredSegmentStore:
    """从后端能力、角色注册表和注入温层构造可恢复 K-02 存储入口。"""
    repository = BackendObjectRepository(
        backend,
        index_key_fn=index_key_fn,
    )
    return TieredSegmentStore(
        repository,
        registry,
        temperature_profile,
    )


__all__ = [
    "bootstrap",
    "build_storage_role_registry",
    "build_tiered_segment_store",
]
