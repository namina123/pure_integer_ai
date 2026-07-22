"""旧 memory_item/experience_count 到显式 LEGACY_IMPORT 事件的无损映射。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.shared.identity import OwnerScope, VersionBundle
from pure_integer_ai.cognition.shared.memory_event import (
    LEGACY_TABLE_EXPERIENCE_COUNT,
    LEGACY_TABLE_MEMORY_ITEM,
    MEMORY_EVENT_LEGACY_IMPORT,
    MEMORY_OBJECT_LEGACY_IMPORT,
    LegacyImportPayload,
    MemoryEvent,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    ScopeIdentity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_MEMORY_ITEM_FIELDS = (
    "space_id",
    "local_id",
    "content_hash",
    "status",
    "session_id",
    "count",
    "success_count",
    "seg_type",
    "info_ref_space",
    "info_ref_id",
    "context_tag",
    "round_id",
)

_EXPERIENCE_COUNT_FIELDS = (
    "space_id",
    "local_id",
    "ctx_code",
    "speaker_code",
    "base_freq",
    "e_sn",
    "e_tn",
    "observe_tn",
)


def _encode_fields(row: dict[str, Any], fields: tuple[str, ...], *,
                   where: str) -> tuple[int, ...]:
    """按冻结旧 schema 顺序编码 NULL 标签和值，拒绝缺列和非整数内容。"""
    if not isinstance(row, dict):
        raise TypeError(f"{where} 必须是行 dict")
    missing = tuple(field for field in fields if field not in row)
    if missing:
        raise ValueError(f"{where} 缺少字段 {missing}")
    extras = tuple(sorted(set(row) - set(fields)))
    if extras:
        raise ValueError(f"{where} 含未知字段 {extras}")
    result: list[int] = []
    for field in fields:
        value = row[field]
        if value is None:
            result.extend((1, 0))
            continue
        assert_int(value, _where=f"{where}.{field}")
        if type(value) is not int:
            raise ValueError(f"{where}.{field} 必须是严格整数或 None")
        result.extend((0, value))
    return tuple(result)


def legacy_memory_item_payload(
        row: dict[str, Any], imported_at: LogicalTimestamp,
        ) -> LegacyImportPayload:
    """把一条旧 memory_item 原字段映射为不带新语义的导入载荷。"""
    values = _encode_fields(row, _MEMORY_ITEM_FIELDS, where="memory_item")
    return LegacyImportPayload(
        LEGACY_TABLE_MEMORY_ITEM,
        (row["space_id"], row["local_id"]),
        values,
        imported_at,
    )


def legacy_experience_count_payload(
        row: dict[str, Any], imported_at: LogicalTimestamp,
        ) -> LegacyImportPayload:
    """把一条旧 experience_count 原字段映射为不带新语义的导入载荷。"""
    values = _encode_fields(
        row, _EXPERIENCE_COUNT_FIELDS, where="experience_count")
    return LegacyImportPayload(
        LEGACY_TABLE_EXPERIENCE_COUNT,
        (
            row["space_id"], row["local_id"],
            row["ctx_code"], row["speaker_code"],
        ),
        values,
        imported_at,
    )


def append_legacy_import(
        event_log: MemoryEventLog,
        payload: LegacyImportPayload,
        *, scope: ScopeIdentity,
        owner: OwnerScope,
        versions: VersionBundle,
        ) -> MaterializedMemoryEvent:
    """显式追加 legacy import；只保存原字段，不创建 Hypothesis、Evidence 或 Use。"""
    if not isinstance(event_log, MemoryEventLog):
        raise TypeError("event_log 必须是 MemoryEventLog")
    if not isinstance(payload, LegacyImportPayload):
        raise TypeError("payload 必须是 LegacyImportPayload")
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("scope 必须是 ScopeIdentity")
    if scope.owner != owner or scope.versions != versions:
        raise ValueError("legacy import scope 与 owner/version 不一致")
    ref = memory_object_ref(
        event_log.memory_space_identity,
        MEMORY_OBJECT_LEGACY_IMPORT,
        payload.stable_key(),
        owner=owner,
        versions=versions,
    )
    return event_log.append(MemoryEvent(
        MEMORY_EVENT_LEGACY_IMPORT,
        ref,
        scope,
        payload,
    ))


__all__ = [
    "append_legacy_import",
    "legacy_experience_count_payload",
    "legacy_memory_item_payload",
]
