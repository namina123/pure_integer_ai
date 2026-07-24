"""M-10 独立 Memory/Companion 增量恢复包的投影、依赖与按键合并。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
)
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
    MEMORY_HYPOTHESIS_DIRTY_TABLE,
    MEMORY_HYPOTHESIS_EVENT_TABLE,
    MEMORY_HYPOTHESIS_SOURCE_TABLE,
)
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
    MEMORY_EVENT_BATCH_LINK_TABLE,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_PART_TABLE,
    MEMORY_EVENT_TABLE,
)
from pure_integer_ai.storage.memory_overlay import MEMORY_OVERLAY_TABLE
from pure_integer_ai.storage.recovery_package import (
    inspect_recovery_package,
    merge_recovery_package,
    publish_recovery_package,
)
from pure_integer_ai.storage.recovery_protocol import (
    RecoveryDependency,
    RecoveryFaultInjector,
    RecoveryIntegrityError,
    RecoveryLoadResult,
    RecoveryManifest,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import (
    SEGMENT_OBJECT_PART_TABLE,
    SEGMENT_OBJECT_RESERVATION_TABLE,
    SEGMENT_OBJECT_SEAL_TABLE,
    SEGMENT_OBJECT_TOMBSTONE_TABLE,
)
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_COMPANION,
    SPACE_TYPE_MEMORY,
)


MEMORY_RECOVERY_FORMAT_VERSION = 1
MEMORY_RECOVERY_VERSION_KEY = (20260723, 10, 1)

_SPACE_TABLE = "space"

_MEMORY_PRIMARY_TABLES = frozenset({
    MEMORY_OVERLAY_TABLE,
    MEMORY_EVENT_TABLE,
    MEMORY_EVENT_PART_TABLE,
    MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
    MEMORY_HYPOTHESIS_SOURCE_TABLE,
    MEMORY_HYPOTHESIS_EVENT_TABLE,
    MEMORY_HYPOTHESIS_DIRTY_TABLE,
    MEMORY_EVENT_BATCH_LINK_TABLE,
})
_SEGMENT_TABLES = frozenset({
    SEGMENT_OBJECT_RESERVATION_TABLE,
    SEGMENT_OBJECT_PART_TABLE,
    SEGMENT_OBJECT_SEAL_TABLE,
    SEGMENT_OBJECT_TOMBSTONE_TABLE,
})
_PACKAGE_TABLES = (
    _SPACE_TABLE,
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    MEMORY_OVERLAY_TABLE,
    MEMORY_EVENT_TABLE,
    MEMORY_EVENT_PART_TABLE,
    MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
    MEMORY_HYPOTHESIS_SOURCE_TABLE,
    MEMORY_HYPOTHESIS_EVENT_TABLE,
    MEMORY_HYPOTHESIS_DIRTY_TABLE,
    MEMORY_EVENT_BATCH_LINK_TABLE,
    SOURCE_RECORD_TABLE,
    TEXT_ASSOC_TABLE,
    SEGMENT_OBJECT_RESERVATION_TABLE,
    SEGMENT_OBJECT_PART_TABLE,
    SEGMENT_OBJECT_SEAL_TABLE,
    SEGMENT_OBJECT_TOMBSTONE_TABLE,
)


@dataclass(frozen=True)
class MemoryRecoveryProjection:
    """独立包完整 Memory/Companion 空间和全局 identity/source 闭包。"""

    memory_space_ids: tuple[int, ...]
    companion_space_ids: tuple[int, ...]
    source_hashes: frozenset[int]
    identity_rows: frozenset[tuple[int, int]]


def recovery_manifest_dependency(
        manifest: RecoveryManifest,
        ) -> SegmentDependency:
    """把一个已封印 Core 基线 manifest 转为 Memory 包可核验依赖。"""
    if not isinstance(manifest, RecoveryManifest):
        raise TypeError("manifest 必须是 RecoveryManifest")
    payload = json.dumps(
        manifest.to_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SegmentDependency(
        MEMORY_BATCH_CORE_DEPENDENCY_KEY,
        (
            MEMORY_RECOVERY_FORMAT_VERSION,
            manifest.format_version,
            manifest.publish_epoch,
            len(manifest.version_key),
            *manifest.version_key,
        ),
        tuple(hashlib.sha256(payload).digest()),
    )


def _space_rows(backend) -> dict[int, dict[str, Any]]:
    """读取并核验运行时空间 id 唯一映射。"""
    result: dict[int, dict[str, Any]] = {}
    for row in backend.select(_SPACE_TABLE, where=None):
        space_id = row.get("space_id")
        if type(space_id) is not int or space_id <= 0 or space_id in result:
            raise RecoveryIntegrityError("space 表包含非法或重复 space_id")
        result[space_id] = row
    return result


def _identity_closure(
        backend,
        initial_hashes: set[int],
        ) -> frozenset[tuple[int, int]]:
    """沿 identity header parent_hash 收集恢复所需 header/part 闭包。"""
    rows = backend.select(IDENTITY_HEADER_TABLE, where=None)
    selected: set[tuple[int, int]] = set()
    pending = set(initial_hashes)
    changed = True
    while changed:
        changed = False
        for row in rows:
            identity_hash = row.get("identity_hash")
            identity_kind = row.get("identity_kind")
            parent_hash = row.get("parent_hash")
            if (identity_hash not in pending
                    or (identity_kind, identity_hash) in selected):
                continue
            if (type(identity_kind) is not int
                    or type(identity_hash) is not int
                    or type(parent_hash) is not int):
                raise RecoveryIntegrityError("identity header 含非法整数列")
            selected.add((identity_kind, identity_hash))
            if parent_hash > 0 and parent_hash not in pending:
                pending.add(parent_hash)
                changed = True
    return frozenset(selected)


def build_memory_recovery_projection(
        backend,
        *,
        memory_space_ids: tuple[int, ...],
        companion_space_ids: tuple[int, ...],
        ) -> MemoryRecoveryProjection:
    """核验完整空间集，并计算 SourceRecord 和 identity 最小闭包。"""
    if (not isinstance(memory_space_ids, tuple)
            or not memory_space_ids
            or any(type(item) is not int or item <= 0
                   for item in memory_space_ids)
            or len(set(memory_space_ids)) != len(memory_space_ids)):
        raise ValueError("memory_space_ids 必须是非空唯一正整数 tuple")
    if (not isinstance(companion_space_ids, tuple)
            or not companion_space_ids
            or any(type(item) is not int or item <= 0
                   for item in companion_space_ids)
            or len(set(companion_space_ids)) != len(companion_space_ids)):
        raise ValueError("companion_space_ids 必须是非空唯一正整数 tuple")
    spaces = _space_rows(backend)
    for space_id in memory_space_ids:
        if spaces.get(space_id, {}).get("type") != SPACE_TYPE_MEMORY:
            raise RecoveryIntegrityError("Memory 包声明了非 Memory 空间")
    companion_identities = set()
    for space_id in companion_space_ids:
        row = spaces.get(space_id, {})
        if row.get("type") != SPACE_TYPE_COMPANION:
            raise RecoveryIntegrityError("Memory 包声明了非 Companion 空间")
        companion_identities.add((row["type_hash"], row["name_hash"]))
    all_memory_spaces = {
        space_id for space_id, row in spaces.items()
        if row.get("type") == SPACE_TYPE_MEMORY
    }
    all_companion_spaces = {
        space_id for space_id, row in spaces.items()
        if row.get("type") == SPACE_TYPE_COMPANION
    }
    if set(memory_space_ids) != all_memory_spaces:
        raise RecoveryIntegrityError(
            "共享 K-02 manifest 要求独立包包含完整 Memory 空间集")
    if set(companion_space_ids) != all_companion_spaces:
        raise RecoveryIntegrityError(
            "共享 K-02 manifest 要求独立包包含完整 Companion 空间集")

    source_hashes = {
        row["source_hash"]
        for row in backend.select(SOURCE_RECORD_TABLE, where=None)
        if (row.get("companion_type_hash"), row.get("companion_name_hash"))
        in companion_identities
    }
    identity_hashes = set(source_hashes)
    memory_ids = set(memory_space_ids)
    for row in backend.select(MEMORY_EVENT_TABLE, where=None):
        if row.get("space_id") not in memory_ids:
            continue
        for column in (
                "scope_hash",
                "timestamp_hash",
                "clock_hash",
                "timeline_timestamp_hash"):
            value = row.get(column)
            if type(value) is not int or value <= 0:
                raise RecoveryIntegrityError(
                    f"memory_event 缺少合法 {column}")
            identity_hashes.add(value)
    for row in backend.select(MEMORY_OVERLAY_TABLE, where=None):
        if row.get("space_id") not in memory_ids:
            continue
        for column in ("identity_hash", "scope_hash"):
            value = row.get(column)
            if type(value) is not int or value <= 0:
                raise RecoveryIntegrityError(
                    f"memory_overlay 缺少合法 {column}")
            identity_hashes.add(value)
    return MemoryRecoveryProjection(
        tuple(sorted(memory_space_ids)),
        tuple(sorted(companion_space_ids)),
        frozenset(source_hashes),
        _identity_closure(backend, identity_hashes),
    )


def _selector(projection: MemoryRecoveryProjection):
    """构造只选择投影闭包的 V-03 row selector。"""
    selected_spaces = set(
        (*projection.memory_space_ids, *projection.companion_space_ids))
    memory_spaces = set(projection.memory_space_ids)
    companion_spaces = set(projection.companion_space_ids)

    def select(table: str, row: dict[str, Any]) -> bool:
        """按表的显式 owner 列选择行，不从路径或后端类型推断。"""
        if table == _SPACE_TABLE:
            return row.get("space_id") in selected_spaces
        if table == IDENTITY_HEADER_TABLE:
            return (row.get("identity_kind"), row.get("identity_hash")) in (
                projection.identity_rows)
        if table == IDENTITY_PART_TABLE:
            return (row.get("identity_kind"), row.get("identity_hash")) in (
                projection.identity_rows)
        if table == SOURCE_RECORD_TABLE:
            return row.get("source_hash") in projection.source_hashes
        if table == TEXT_ASSOC_TABLE:
            return row.get("space_id") in companion_spaces
        if table in _MEMORY_PRIMARY_TABLES:
            return row.get("space_id") in memory_spaces
        if table in _SEGMENT_TABLES:
            return True
        raise RecoveryIntegrityError(f"Memory package 出现未注册表 {table}")

    return select


def publish_memory_recovery_package(
        backend,
        run_dir: str,
        run_id: str,
        *,
        memory_space_ids: tuple[int, ...],
        companion_space_ids: tuple[int, ...],
        core_dependency: RecoveryDependency,
        publish_epoch: int = 1,
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryManifest:
    """发布只含 Memory/Companion/Source 闭包并显式依赖 Core 的增量包。"""
    if (not isinstance(core_dependency, SegmentDependency)
            or core_dependency.descriptor_key
            != MEMORY_BATCH_CORE_DEPENDENCY_KEY):
        raise ValueError("Memory package 缺少合法 Core dependency")
    schema = backend.schema_snapshot()
    missing = tuple(table for table in _PACKAGE_TABLES if table not in schema)
    if missing:
        raise RecoveryIntegrityError(
            f"Memory package 缺少已注册表: {missing}")
    for table in _PACKAGE_TABLES:
        if not schema[table].get("recovery_key"):
            raise RecoveryIntegrityError(
                f"Memory package 表 {table} 未声明 recovery_key")
    projection = build_memory_recovery_projection(
        backend,
        memory_space_ids=memory_space_ids,
        companion_space_ids=companion_space_ids,
    )
    return publish_recovery_package(
        backend,
        run_dir,
        run_id,
        spaces=tuple(sorted((
            *projection.memory_space_ids,
            *projection.companion_space_ids,
        ))),
        tables=_PACKAGE_TABLES,
        version_key=MEMORY_RECOVERY_VERSION_KEY,
        dependencies=(core_dependency,),
        publish_epoch=publish_epoch,
        row_selector=_selector(projection),
        fault_injector=fault_injector,
    )


def load_memory_recovery_package(
        backend,
        run_dir: str,
        run_id: str,
        *,
        available_dependencies: tuple[RecoveryDependency, ...],
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryLoadResult:
    """核验目标已有 Core dependency 后，按 recovery_key 合并独立 Memory 包。"""
    if (not isinstance(available_dependencies, tuple)
            or any(not isinstance(item, SegmentDependency)
                   for item in available_dependencies)):
        raise TypeError("available_dependencies 类型错误")
    package = inspect_recovery_package(
        run_dir,
        run_id,
        expected_version_key=MEMORY_RECOVERY_VERSION_KEY,
    )
    available = set(available_dependencies)
    missing = tuple(
        dependency for dependency in package.manifest.dependencies
        if dependency not in available
    )
    if missing:
        raise RecoveryIntegrityError(
            f"Memory package 缺少目标 Core dependency: {missing}")
    return merge_recovery_package(
        backend,
        run_dir,
        run_id,
        expected_version_key=MEMORY_RECOVERY_VERSION_KEY,
        expected_dependencies=package.manifest.dependencies,
        fault_injector=fault_injector,
    )


__all__ = [
    "MEMORY_RECOVERY_FORMAT_VERSION",
    "MEMORY_RECOVERY_VERSION_KEY",
    "MemoryRecoveryProjection",
    "build_memory_recovery_projection",
    "load_memory_recovery_package",
    "publish_memory_recovery_package",
    "recovery_manifest_dependency",
]
