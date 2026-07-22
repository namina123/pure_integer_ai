"""Memory Hypothesis 派生聚合、来源索引和 dirty queue 的物理表。

这些表只服务查询与增量重建，不能替代 ``memory_event`` 真源。它们注册为可删除
扩展表，允许在事件不变时整体丢弃并重建。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.backend import register_extension_table


MEMORY_HYPOTHESIS_AGGREGATE_TABLE = "memory_hypothesis_aggregate"
MEMORY_HYPOTHESIS_SOURCE_TABLE = "memory_hypothesis_source_index"
MEMORY_HYPOTHESIS_EVENT_TABLE = "memory_hypothesis_event_index"
MEMORY_HYPOTHESIS_DIRTY_TABLE = "memory_hypothesis_dirty"

MEMORY_HYPOTHESIS_AGGREGATE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("hypothesis_hash", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
    ("hypothesis_kind_hash", TYPE_INT),
    ("competition_hash", TYPE_INT),
    ("context_hash", TYPE_INT),
    ("source_hash", TYPE_INT),
    ("created_seq", TYPE_INT),
    ("last_observed_seq", TYPE_INT),
    ("last_supported_seq", TYPE_INT),
    ("last_refuted_seq", TYPE_INT),
    ("last_used_seq", TYPE_INT),
    ("support_count", TYPE_INT),
    ("contradict_count", TYPE_INT),
    ("unknown_count", TYPE_INT),
    ("independent_source_count", TYPE_INT),
    ("support_source_count", TYPE_INT),
    ("contradict_source_count", TYPE_INT),
    ("use_count", TYPE_INT),
    ("retention_state", TYPE_INT),
    ("lifecycle_state", TYPE_INT),
    ("evidence_state", TYPE_INT),
]

MEMORY_HYPOTHESIS_SOURCE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("hypothesis_hash", TYPE_INT),
    ("source_hash", TYPE_INT),
    ("stance", TYPE_INT),
    ("first_observed_seq", TYPE_INT),
    ("last_observed_seq", TYPE_INT),
    ("evidence_count", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
    *((f"source_key_{index:02d}", TYPE_INT) for index in range(11)),
]

MEMORY_HYPOTHESIS_EVENT_COLUMNS = [
    ("space_id", TYPE_INT),
    ("hypothesis_hash", TYPE_INT),
    ("event_hash", TYPE_INT),
    ("event_kind", TYPE_INT),
    ("event_object_hash", TYPE_INT),
    ("event_seq", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
]

MEMORY_HYPOTHESIS_DIRTY_COLUMNS = [
    ("space_id", TYPE_INT),
    ("hypothesis_hash", TYPE_INT),
    ("reason_event_hash", TYPE_INT),
    ("dirty_seq", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
]

MEMORY_HYPOTHESIS_AGGREGATE_INDEXES = [
    ("space_id", "hypothesis_hash"),
    ("space_id", "owner_visibility", "owner_tenant_id", "owner_user_id", "owner_session_id"),
    ("space_id", "hypothesis_kind_hash", "context_hash", "evidence_state"),
    ("space_id", "evidence_state", "lifecycle_state", "retention_state"),
    ("space_id", "created_seq"),
    ("space_id", "last_observed_seq"),
    ("space_id", "last_used_seq"),
]

MEMORY_HYPOTHESIS_SOURCE_INDEXES = [
    ("space_id", "source_hash"),
    ("space_id", "hypothesis_hash"),
    ("space_id", "stance", "source_hash"),
]

MEMORY_HYPOTHESIS_EVENT_INDEXES = [
    ("space_id", "hypothesis_hash"),
    ("space_id", "event_hash"),
    ("space_id", "event_kind", "hypothesis_hash"),
]

MEMORY_HYPOTHESIS_DIRTY_INDEXES = [
    ("space_id", "hypothesis_hash"),
    ("space_id", "dirty_seq"),
    ("space_id", "owner_visibility", "owner_tenant_id", "owner_user_id", "owner_session_id"),
]


class MemoryAggregateIntegrityError(RuntimeError):
    """派生 Memory 表出现重复、漂移或不完整行。"""


def register_memory_aggregate_tables(backend: StorageBackend) -> None:
    """注册可删除的 aggregate、来源索引、事件反向索引和 dirty queue。"""
    register_extension_table(
        backend,
        MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
        MEMORY_HYPOTHESIS_AGGREGATE_COLUMNS,
        disc.DISC_NONE,
        MEMORY_HYPOTHESIS_AGGREGATE_INDEXES,
    )
    register_extension_table(
        backend,
        MEMORY_HYPOTHESIS_SOURCE_TABLE,
        MEMORY_HYPOTHESIS_SOURCE_COLUMNS,
        disc.DISC_NONE,
        MEMORY_HYPOTHESIS_SOURCE_INDEXES,
    )
    register_extension_table(
        backend,
        MEMORY_HYPOTHESIS_EVENT_TABLE,
        MEMORY_HYPOTHESIS_EVENT_COLUMNS,
        disc.DISC_NONE,
        MEMORY_HYPOTHESIS_EVENT_INDEXES,
    )
    register_extension_table(
        backend,
        MEMORY_HYPOTHESIS_DIRTY_TABLE,
        MEMORY_HYPOTHESIS_DIRTY_COLUMNS,
        disc.DISC_NONE,
        MEMORY_HYPOTHESIS_DIRTY_INDEXES,
    )


def _strict(value: int, *, where: str, positive: bool = False) -> int:
    """校验派生行中的纯整数，并按字段要求限制为正数。"""
    assert_int(value, _where=where)
    if type(value) is not int:
        raise ValueError(f"{where} 必须是严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须是正整数")
    if not positive and value < 0:
        raise ValueError(f"{where} 不得为负数")
    return value


def _owner(value: tuple[int, int, int, int], *, where: str) -> tuple[int, int, int, int]:
    """校验派生行的 owner 四元组。"""
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{where} 必须是四整数 tuple")
    return tuple(
        _strict(item, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class MemoryHypothesisAggregateRecord:
    """一个 Hypothesis 当前派生状态的固定整数快照。"""

    space_id: int
    hypothesis_hash: int
    owner_key: tuple[int, int, int, int]
    hypothesis_kind_hash: int
    competition_hash: int
    context_hash: int
    source_hash: int
    created_seq: int
    last_observed_seq: int
    last_supported_seq: int
    last_refuted_seq: int
    last_used_seq: int
    support_count: int
    contradict_count: int
    unknown_count: int
    independent_source_count: int
    support_source_count: int
    contradict_source_count: int
    use_count: int
    retention_state: int
    lifecycle_state: int
    evidence_state: int

    def __post_init__(self) -> None:
        """校验 aggregate 的 hash、owner、计数和状态字段。"""
        _strict(self.space_id, where="aggregate.space_id", positive=True)
        _strict(self.hypothesis_hash, where="aggregate.hypothesis_hash", positive=True)
        _owner(self.owner_key, where="aggregate.owner_key")
        for name, value in (
                ("hypothesis_kind_hash", self.hypothesis_kind_hash),
                ("competition_hash", self.competition_hash),
                ("context_hash", self.context_hash),
                ("source_hash", self.source_hash)):
            _strict(value, where=f"aggregate.{name}", positive=True)
        for name, value in (
                ("created_seq", self.created_seq),
                ("last_observed_seq", self.last_observed_seq),
                ("last_supported_seq", self.last_supported_seq),
                ("last_refuted_seq", self.last_refuted_seq),
                ("last_used_seq", self.last_used_seq),
                ("support_count", self.support_count),
                ("contradict_count", self.contradict_count),
                ("unknown_count", self.unknown_count),
                ("independent_source_count", self.independent_source_count),
                ("support_source_count", self.support_source_count),
                ("contradict_source_count", self.contradict_source_count),
                ("use_count", self.use_count),
                ("retention_state", self.retention_state),
                ("lifecycle_state", self.lifecycle_state),
                ("evidence_state", self.evidence_state)):
            _strict(value, where=f"aggregate.{name}")

    def to_row(self) -> dict[str, int]:
        """把 aggregate 快照投影为扩展表行。"""
        return {
            "space_id": self.space_id,
            "hypothesis_hash": self.hypothesis_hash,
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
            "hypothesis_kind_hash": self.hypothesis_kind_hash,
            "competition_hash": self.competition_hash,
            "context_hash": self.context_hash,
            "source_hash": self.source_hash,
            "created_seq": self.created_seq,
            "last_observed_seq": self.last_observed_seq,
            "last_supported_seq": self.last_supported_seq,
            "last_refuted_seq": self.last_refuted_seq,
            "last_used_seq": self.last_used_seq,
            "support_count": self.support_count,
            "contradict_count": self.contradict_count,
            "unknown_count": self.unknown_count,
            "independent_source_count": self.independent_source_count,
            "support_source_count": self.support_source_count,
            "contradict_source_count": self.contradict_source_count,
            "use_count": self.use_count,
            "retention_state": self.retention_state,
            "lifecycle_state": self.lifecycle_state,
            "evidence_state": self.evidence_state,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryHypothesisAggregateRecord":
        """从扩展表行恢复 aggregate，缺列或损坏值直接失败。"""
        try:
            return cls(
                row["space_id"],
                row["hypothesis_hash"],
                (
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                ),
                row["hypothesis_kind_hash"], row["competition_hash"],
                row["context_hash"], row["source_hash"],
                row["created_seq"], row["last_observed_seq"],
                row["last_supported_seq"], row["last_refuted_seq"],
                row["last_used_seq"], row["support_count"],
                row["contradict_count"], row["unknown_count"],
                row["independent_source_count"], row["support_source_count"],
                row["contradict_source_count"], row["use_count"],
                row["retention_state"], row["lifecycle_state"],
                row["evidence_state"],
            )
        except KeyError as exc:
            raise MemoryAggregateIntegrityError(
                f"aggregate 行缺少字段 {exc.args[0]}") from exc


@dataclass(frozen=True)
class MemoryHypothesisSourceRecord:
    """一个 Hypothesis 在一个来源和立场上的活动证据统计。"""

    space_id: int
    hypothesis_hash: int
    source_hash: int
    stance: int
    first_observed_seq: int
    last_observed_seq: int
    evidence_count: int
    owner_key: tuple[int, int, int, int]
    source_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """校验来源派生行的身份、计数和 owner。"""
        _strict(self.space_id, where="source.space_id", positive=True)
        _strict(self.hypothesis_hash, where="source.hypothesis_hash", positive=True)
        _strict(self.source_hash, where="source.source_hash", positive=True)
        _strict(self.stance, where="source.stance", positive=True)
        for name, value in (
                ("first_observed_seq", self.first_observed_seq),
                ("last_observed_seq", self.last_observed_seq),
                ("evidence_count", self.evidence_count)):
            _strict(value, where=f"source.{name}")
        if self.evidence_count <= 0:
            raise ValueError("source.evidence_count 必须为正数")
        _owner(self.owner_key, where="source.owner_key")
        if len(self.source_key) != 11 or any(
                type(value) is not int for value in self.source_key):
            raise ValueError("source.source_key 必须是 11 项严格整数键")

    def to_row(self) -> dict[str, int]:
        """把来源统计投影为扩展表行。"""
        return {
            "space_id": self.space_id,
            "hypothesis_hash": self.hypothesis_hash,
            "source_hash": self.source_hash,
            "stance": self.stance,
            "first_observed_seq": self.first_observed_seq,
            "last_observed_seq": self.last_observed_seq,
            "evidence_count": self.evidence_count,
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
            **{
                f"source_key_{index:02d}": value
                for index, value in enumerate(self.source_key)
            },
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryHypothesisSourceRecord":
        """从来源索引行恢复统计，缺列直接失败。"""
        try:
            return cls(
                row["space_id"], row["hypothesis_hash"], row["source_hash"],
                row["stance"], row["first_observed_seq"],
                row["last_observed_seq"], row["evidence_count"],
                (
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                ),
                tuple(row[f"source_key_{index:02d}"] for index in range(11)),
            )
        except KeyError as exc:
            raise MemoryAggregateIntegrityError(
                f"source index 行缺少字段 {exc.args[0]}") from exc


@dataclass(frozen=True)
class MemoryHypothesisEventIndexRecord:
    """一个事件与其影响 Hypothesis 的反向索引行。"""

    space_id: int
    hypothesis_hash: int
    event_hash: int
    event_kind: int
    event_object_hash: int
    event_seq: int
    owner_key: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """校验事件反向索引的非空 hash、序号和 owner。"""
        for name, value in (
                ("space_id", self.space_id),
                ("hypothesis_hash", self.hypothesis_hash),
                ("event_hash", self.event_hash),
                ("event_kind", self.event_kind),
                ("event_object_hash", self.event_object_hash)):
            _strict(value, where=f"event_index.{name}", positive=True)
        _strict(self.event_seq, where="event_index.event_seq")
        _owner(self.owner_key, where="event_index.owner_key")

    def to_row(self) -> dict[str, int]:
        """把事件反向索引投影为扩展表行。"""
        return {
            "space_id": self.space_id,
            "hypothesis_hash": self.hypothesis_hash,
            "event_hash": self.event_hash,
            "event_kind": self.event_kind,
            "event_object_hash": self.event_object_hash,
            "event_seq": self.event_seq,
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryHypothesisEventIndexRecord":
        """从事件反向索引行恢复记录。"""
        try:
            return cls(
                row["space_id"], row["hypothesis_hash"], row["event_hash"],
                row["event_kind"], row["event_object_hash"], row["event_seq"],
                (
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                ),
            )
        except KeyError as exc:
            raise MemoryAggregateIntegrityError(
                f"event index 行缺少字段 {exc.args[0]}") from exc


@dataclass(frozen=True)
class MemoryHypothesisDirtyRecord:
    """待重建 Hypothesis 的唯一 dirty queue 行。"""

    space_id: int
    hypothesis_hash: int
    reason_event_hash: int
    dirty_seq: int
    owner_key: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """校验 dirty key、原因事件和逻辑序号。"""
        _strict(self.space_id, where="dirty.space_id", positive=True)
        _strict(self.hypothesis_hash, where="dirty.hypothesis_hash", positive=True)
        _strict(self.reason_event_hash, where="dirty.reason_event_hash", positive=True)
        _strict(self.dirty_seq, where="dirty.dirty_seq")
        _owner(self.owner_key, where="dirty.owner_key")

    def to_row(self) -> dict[str, int]:
        """把 dirty 记录投影为扩展表行。"""
        return {
            "space_id": self.space_id,
            "hypothesis_hash": self.hypothesis_hash,
            "reason_event_hash": self.reason_event_hash,
            "dirty_seq": self.dirty_seq,
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryHypothesisDirtyRecord":
        """从 dirty queue 行恢复记录。"""
        try:
            return cls(
                row["space_id"], row["hypothesis_hash"],
                row["reason_event_hash"], row["dirty_seq"],
                (
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                ),
            )
        except KeyError as exc:
            raise MemoryAggregateIntegrityError(
                f"dirty queue 行缺少字段 {exc.args[0]}") from exc


class MemoryAggregateStore:
    """四张派生表的严格读写边界。"""

    def __init__(self, backend: StorageBackend, space_id: int) -> None:
        """绑定 backend 和一个 Memory 空间。"""
        _strict(space_id, where="MemoryAggregateStore.space_id", positive=True)
        self.backend = backend
        self.space_id = space_id

    def replace_aggregate(self, record: MemoryHypothesisAggregateRecord) -> None:
        """以一行新快照替换旧 aggregate，不触碰事件真源。"""
        if record.space_id != self.space_id:
            raise ValueError("aggregate 空间漂移")
        self.backend.delete(
            MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": record.hypothesis_hash},
        )
        self.backend.insert(MEMORY_HYPOTHESIS_AGGREGATE_TABLE, record.to_row())

    def read_aggregate(self, hypothesis_hash: int) -> MemoryHypothesisAggregateRecord | None:
        """按 Hypothesis hash 读取唯一 aggregate，重复行直接失败。"""
        _strict(hypothesis_hash, where="read_aggregate.hypothesis_hash", positive=True)
        rows = self.backend.select(
            MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": hypothesis_hash},
        )
        if len(rows) > 1:
            raise MemoryAggregateIntegrityError("同一 Hypothesis 存在重复 aggregate")
        return None if not rows else MemoryHypothesisAggregateRecord.from_row(rows[0])

    def list_aggregates(self) -> tuple[MemoryHypothesisAggregateRecord, ...]:
        """按固定插入投影读取当前空间全部 aggregate。"""
        return tuple(
            MemoryHypothesisAggregateRecord.from_row(row)
            for row in self.backend.select(
                MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
                {"space_id": self.space_id},
            )
        )

    def replace_sources(self, records: tuple[MemoryHypothesisSourceRecord, ...],
                       hypothesis_hash: int) -> None:
        """替换一个 Hypothesis 的来源索引行。"""
        _strict(hypothesis_hash, where="replace_sources.hypothesis_hash", positive=True)
        self.delete_sources(hypothesis_hash)
        for record in records:
            if record.space_id != self.space_id or record.hypothesis_hash != hypothesis_hash:
                raise ValueError("source index 空间或 Hypothesis 漂移")
            self.backend.insert(MEMORY_HYPOTHESIS_SOURCE_TABLE, record.to_row())

    def delete_sources(self, hypothesis_hash: int) -> None:
        """删除一个 Hypothesis 的来源派生行。"""
        _strict(hypothesis_hash, where="delete_sources.hypothesis_hash", positive=True)
        self.backend.delete(
            MEMORY_HYPOTHESIS_SOURCE_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": hypothesis_hash},
        )

    def replace_events(self, records: tuple[MemoryHypothesisEventIndexRecord, ...],
                       hypothesis_hash: int) -> None:
        """替换一个 Hypothesis 的事件反向索引行。"""
        _strict(hypothesis_hash, where="replace_events.hypothesis_hash", positive=True)
        self.delete_events(hypothesis_hash)
        for record in records:
            if record.space_id != self.space_id or record.hypothesis_hash != hypothesis_hash:
                raise ValueError("event index 空间或 Hypothesis 漂移")
            self.backend.insert(MEMORY_HYPOTHESIS_EVENT_TABLE, record.to_row())

    def delete_events(self, hypothesis_hash: int) -> None:
        """删除一个 Hypothesis 的事件反向索引行。"""
        _strict(hypothesis_hash, where="delete_events.hypothesis_hash", positive=True)
        self.backend.delete(
            MEMORY_HYPOTHESIS_EVENT_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": hypothesis_hash},
        )

    def list_events(self, hypothesis_hash: int) -> tuple[MemoryHypothesisEventIndexRecord, ...]:
        """按 Hypothesis hash 读取事件反向索引，并拒绝重复 event。"""
        _strict(hypothesis_hash, where="list_events.hypothesis_hash", positive=True)
        records = tuple(
            MemoryHypothesisEventIndexRecord.from_row(row)
            for row in self.backend.select(
                MEMORY_HYPOTHESIS_EVENT_TABLE,
                {"space_id": self.space_id, "hypothesis_hash": hypothesis_hash},
                order_by="event_seq",
            )
        )
        hashes = tuple(record.event_hash for record in records)
        if len(hashes) != len(set(hashes)):
            raise MemoryAggregateIntegrityError("Hypothesis event index 含重复 event")
        return records

    def index_event(self, record: MemoryHypothesisEventIndexRecord) -> None:
        """幂等追加事件反向索引，碰撞时拒绝静默覆盖。"""
        if record.space_id != self.space_id:
            raise ValueError("event index 空间漂移")
        rows = self.backend.select(
            MEMORY_HYPOTHESIS_EVENT_TABLE,
            {"space_id": self.space_id, "event_hash": record.event_hash},
        )
        if len(rows) > 1:
            raise MemoryAggregateIntegrityError("同一事件存在重复 Hypothesis index")
        if rows:
            existing = MemoryHypothesisEventIndexRecord.from_row(rows[0])
            if existing != record:
                raise MemoryAggregateIntegrityError("event index 命中不同 Hypothesis")
            return
        self.backend.insert(MEMORY_HYPOTHESIS_EVENT_TABLE, record.to_row())

    def enqueue_dirty(self, record: MemoryHypothesisDirtyRecord) -> None:
        """按 Hypothesis 幂等合并 dirty 原因，只保留最新逻辑序。"""
        if record.space_id != self.space_id:
            raise ValueError("dirty queue 空间漂移")
        rows = self.backend.select(
            MEMORY_HYPOTHESIS_DIRTY_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": record.hypothesis_hash},
        )
        if len(rows) > 1:
            raise MemoryAggregateIntegrityError("同一 Hypothesis 存在重复 dirty 行")
        if not rows:
            self.backend.insert(MEMORY_HYPOTHESIS_DIRTY_TABLE, record.to_row())
            return
        existing = MemoryHypothesisDirtyRecord.from_row(rows[0])
        if existing.owner_key != record.owner_key:
            raise MemoryAggregateIntegrityError("dirty owner 漂移")
        if (record.dirty_seq, record.reason_event_hash) > (
                existing.dirty_seq, existing.reason_event_hash):
            self.backend.update(
                MEMORY_HYPOTHESIS_DIRTY_TABLE,
                {"space_id": self.space_id, "hypothesis_hash": record.hypothesis_hash},
                {
                    "reason_event_hash": record.reason_event_hash,
                    "dirty_seq": record.dirty_seq,
                },
            )

    def list_dirty(self) -> tuple[MemoryHypothesisDirtyRecord, ...]:
        """按 dirty 序和 Hypothesis hash 返回当前空间待处理键。"""
        records = [
            MemoryHypothesisDirtyRecord.from_row(row)
            for row in self.backend.select(
                MEMORY_HYPOTHESIS_DIRTY_TABLE,
                {"space_id": self.space_id},
            )
        ]
        return tuple(sorted(records, key=lambda item: (
            item.dirty_seq, item.hypothesis_hash)))

    def delete_dirty(self, hypothesis_hash: int) -> None:
        """删除一个已成功处理的 dirty 键。"""
        _strict(hypothesis_hash, where="delete_dirty.hypothesis_hash", positive=True)
        self.backend.delete(
            MEMORY_HYPOTHESIS_DIRTY_TABLE,
            {"space_id": self.space_id, "hypothesis_hash": hypothesis_hash},
        )

    def clear_all(self) -> None:
        """删除当前空间所有派生行，供全量 rebuild 使用。"""
        for table in (
                MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
                MEMORY_HYPOTHESIS_SOURCE_TABLE,
                MEMORY_HYPOTHESIS_EVENT_TABLE,
                MEMORY_HYPOTHESIS_DIRTY_TABLE):
            self.backend.delete(table, {"space_id": self.space_id})


__all__ = [
    "MEMORY_HYPOTHESIS_AGGREGATE_TABLE",
    "MEMORY_HYPOTHESIS_SOURCE_TABLE",
    "MEMORY_HYPOTHESIS_EVENT_TABLE",
    "MEMORY_HYPOTHESIS_DIRTY_TABLE",
    "MemoryAggregateIntegrityError",
    "MemoryHypothesisAggregateRecord",
    "MemoryHypothesisSourceRecord",
    "MemoryHypothesisEventIndexRecord",
    "MemoryHypothesisDirtyRecord",
    "MemoryAggregateStore",
    "register_memory_aggregate_tables",
]
