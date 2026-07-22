"""一等 occurrence 的位置详情和候选端点兼容索引。

Occurrence 本体身份仍由 graph_object 保存；本模块只保存可回源的位置、scope、说话人
端点和候选端点索引。typed 关系由上层 GraphOntology 写图，legacy 候选只在本表显式标 0。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.telemetry import telemetry_scope_if_active


OCCURRENCE_TABLE = "occurrence"
OCCURRENCE_CANDIDATE_TABLE = "occurrence_candidate"

OCCURRENCE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("source_hash", TYPE_INT),
    ("scope_hash", TYPE_INT),
    ("start", TYPE_INT),
    ("end", TYPE_INT),
    ("ordinal", TYPE_INT),
    ("segment_index", TYPE_INT),
    ("local_index", TYPE_INT),
    ("document_index", TYPE_INT),
    ("parser_version", TYPE_INT),
    ("speaker_object_kind", TYPE_INT),
    ("speaker_space_id", TYPE_INT),
    ("speaker_local_id", TYPE_INT),
]
OCCURRENCE_CANDIDATE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("candidate_ordinal", TYPE_INT),
    ("candidate_object_kind", TYPE_INT),
    ("candidate_space_id", TYPE_INT),
    ("candidate_local_id", TYPE_INT),
]


class OccurrenceStorageIntegrityError(RuntimeError):
    """Occurrence 详情或候选桥出现冲突、重复和非法端点。"""


@dataclass(frozen=True, order=True)
class OccurrenceStorageRecord:
    """一个 occurrence 的来源、位置、scope 和可选说话人详情。"""

    space_id: int
    local_id: int
    source_hash: int
    scope_hash: int
    start: int
    end: int
    ordinal: int
    segment_index: int
    local_index: int
    document_index: int
    parser_version: int
    speaker_object_kind: int = 0
    speaker_space_id: int = 0
    speaker_local_id: int = 0


@dataclass(frozen=True, order=True)
class OccurrenceCandidateStorage:
    """Occurrence 到 typed 或显式 legacy 候选端点的有序索引。"""

    space_id: int
    local_id: int
    candidate_ordinal: int
    candidate_object_kind: int
    candidate_space_id: int
    candidate_local_id: int


def register_occurrence_tables(backend: StorageBackend) -> None:
    """注册 occurrence 详情和候选桥两张 append-only 核心表。"""
    backend.register_table(
        OCCURRENCE_TABLE,
        OCCURRENCE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("space_id", "local_id"),
            ("source_hash",),
            ("scope_hash",),
        ],
        core=True,
    )
    backend.register_table(
        OCCURRENCE_CANDIDATE_TABLE,
        OCCURRENCE_CANDIDATE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("space_id", "local_id"),
            ("candidate_space_id", "candidate_local_id"),
        ],
        core=True,
    )


def _validate_occurrence(record: OccurrenceStorageRecord) -> None:
    """校验 occurrence 编址、span、位置和 speaker 端点完整性。"""
    values = tuple(record.__dict__.values())
    assert_int(*values, _where="OccurrenceStorageRecord")
    if any(type(value) is not int for value in values):
        raise ValueError("OccurrenceStorageRecord 必须使用严格整数")
    if min(record.space_id, record.local_id, record.source_hash,
           record.scope_hash) <= 0:
        raise ValueError("occurrence 编址、来源和 scope hash 必须为正")
    if (record.start < 0 or record.end < record.start or record.ordinal < 0
            or record.segment_index < 0 or record.local_index < 0
            or record.document_index < 0 or record.parser_version < 0):
        raise ValueError("occurrence span、位置或 parser version 非法")
    speaker = (
        record.speaker_object_kind,
        record.speaker_space_id,
        record.speaker_local_id,
    )
    if speaker != (0, 0, 0) and min(speaker) <= 0:
        raise ValueError("speaker 端点必须全零或全为正整数")


def _validate_candidate(record: OccurrenceCandidateStorage) -> None:
    """校验候选桥；object_kind=0 只表示显式 legacy 节点。"""
    values = tuple(record.__dict__.values())
    assert_int(*values, _where="OccurrenceCandidateStorage")
    if any(type(value) is not int for value in values):
        raise ValueError("OccurrenceCandidateStorage 必须使用严格整数")
    if (record.space_id <= 0 or record.local_id <= 0
            or record.candidate_ordinal < 0
            or record.candidate_object_kind < 0
            or record.candidate_space_id <= 0
            or record.candidate_local_id <= 0):
        raise ValueError("occurrence candidate 编址、顺序或类型非法")


class OccurrenceStore:
    """维护 occurrence 详情和候选桥的幂等 append-only 写入。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    def add(self, record: OccurrenceStorageRecord) -> OccurrenceStorageRecord:
        """幂等追加 occurrence 详情，同身份异内容拒绝。"""
        if not isinstance(record, OccurrenceStorageRecord):
            raise TypeError("OccurrenceStore.add 需要 OccurrenceStorageRecord")
        _validate_occurrence(record)
        with telemetry_scope_if_active(
                occurrence_key=(record.space_id, record.local_id),
                query="occurrence.add"):
            rows = self._backend.select(OCCURRENCE_TABLE, where={
                "space_id": record.space_id,
                "local_id": record.local_id,
            })
            if rows:
                if (len(rows) != 1
                        or self._occurrence_from_row(rows[0]) != record):
                    raise OccurrenceStorageIntegrityError(
                        "occurrence 身份已绑定冲突详情")
                return record
            self._backend.insert(OCCURRENCE_TABLE, dict(record.__dict__))
            return self.read(record.space_id, record.local_id)

    def read(self, space_id: int, local_id: int) -> OccurrenceStorageRecord:
        """按 occurrence 图节点回读唯一详情。"""
        assert_int(space_id, local_id, _where="OccurrenceStore.read")
        with telemetry_scope_if_active(
                occurrence_key=(space_id, local_id),
                query="occurrence.read"):
            rows = self._backend.select(OCCURRENCE_TABLE, where={
                "space_id": space_id,
                "local_id": local_id,
            })
            if len(rows) != 1:
                raise OccurrenceStorageIntegrityError("occurrence 没有唯一详情记录")
            return self._occurrence_from_row(rows[0])

    def add_candidate(
            self, record: OccurrenceCandidateStorage
            ) -> OccurrenceCandidateStorage:
        """追加候选桥，同 ordinal 或同端点发生冲突时 fail closed。"""
        if not isinstance(record, OccurrenceCandidateStorage):
            raise TypeError("add_candidate 需要 OccurrenceCandidateStorage")
        _validate_candidate(record)
        with telemetry_scope_if_active(
                occurrence_key=(record.space_id, record.local_id),
                query="occurrence.add_candidate"):
            rows = self._backend.select(OCCURRENCE_CANDIDATE_TABLE, where={
                "space_id": record.space_id,
                "local_id": record.local_id,
            })
            existing = tuple(self._candidate_from_row(row) for row in rows)
            if record in existing:
                return record
            endpoint = (
                record.candidate_object_kind,
                record.candidate_space_id,
                record.candidate_local_id,
            )
            for item in existing:
                item_endpoint = (
                    item.candidate_object_kind,
                    item.candidate_space_id,
                    item.candidate_local_id,
                )
                if (item.candidate_ordinal == record.candidate_ordinal
                        or item_endpoint == endpoint):
                    raise OccurrenceStorageIntegrityError(
                        "occurrence candidate 顺序或端点冲突")
            self._backend.insert(
                OCCURRENCE_CANDIDATE_TABLE, dict(record.__dict__))
            return record

    def candidates(self, space_id: int, local_id: int
                   ) -> tuple[OccurrenceCandidateStorage, ...]:
        """按候选 ordinal 返回 occurrence 的全部 typed/legacy 端点。"""
        assert_int(space_id, local_id, _where="OccurrenceStore.candidates")
        with telemetry_scope_if_active(
                occurrence_key=(space_id, local_id),
                query="occurrence.candidates"):
            return tuple(sorted(
                (self._candidate_from_row(row) for row in self._backend.select(
                    OCCURRENCE_CANDIDATE_TABLE,
                    where={"space_id": space_id, "local_id": local_id},
                )),
                key=lambda item: item.candidate_ordinal,
            ))

    def occurrence_count(self) -> int:
        """返回当前后端的唯一 occurrence 详情行数。"""
        return len(self._backend.select(OCCURRENCE_TABLE, where=None))

    @staticmethod
    def _occurrence_from_row(row) -> OccurrenceStorageRecord:
        """把存储行恢复为经校验的 occurrence 详情。"""
        record = OccurrenceStorageRecord(**row)
        _validate_occurrence(record)
        return record

    @staticmethod
    def _candidate_from_row(row) -> OccurrenceCandidateStorage:
        """把存储行恢复为经校验的候选桥。"""
        record = OccurrenceCandidateStorage(**row)
        _validate_candidate(record)
        return record


__all__ = [
    "OCCURRENCE_CANDIDATE_TABLE",
    "OCCURRENCE_TABLE",
    "OccurrenceCandidateStorage",
    "OccurrenceStorageIntegrityError",
    "OccurrenceStorageRecord",
    "OccurrenceStore",
    "register_occurrence_tables",
]
