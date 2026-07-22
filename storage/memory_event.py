"""Memory 事件信封的 append-only 物理记录。

完整事件与对象键由共享 identity registry 的外部恢复器从本表和固定宽度 payload
chunk 重建；本表保存可索引信封，chunk 只保存一份 payload，二者在每次读写时双向
核验。aggregate、当前态和检索评分不属于本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


MEMORY_EVENT_TABLE = "memory_event"
MEMORY_EVENT_PART_TABLE = "memory_event_part"
MEMORY_EVENT_CHUNK_WIDTH = 16

MEMORY_EVENT_COLUMNS = [
    ("event_hash", TYPE_INT),
    ("object_hash", TYPE_INT),
    ("space_id", TYPE_INT),
    ("memory_space_type", TYPE_INT),
    ("memory_space_type_hash", TYPE_INT),
    ("memory_space_name_hash", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
    ("event_kind", TYPE_INT),
    ("object_kind", TYPE_INT),
    ("payload_size", TYPE_INT),
    ("scope_hash", TYPE_INT),
    ("timestamp_hash", TYPE_INT),
    ("clock_hash", TYPE_INT),
    ("event_seq", TYPE_INT),
    ("created_seq", TYPE_INT),
    ("observed_seq", TYPE_INT),
    ("used_seq", TYPE_INT),
]

MEMORY_EVENT_PART_COLUMNS = [
    ("event_hash", TYPE_INT),
    ("space_id", TYPE_INT),
    ("chunk_index", TYPE_INT),
    ("part_size", TYPE_INT),
    *((f"part_{index:02d}", TYPE_INT)
      for index in range(MEMORY_EVENT_CHUNK_WIDTH)),
]

MEMORY_EVENT_PART_INDEXES = [
    ("event_hash",),
    ("event_hash", "chunk_index"),
    ("space_id", "event_hash"),
]

MEMORY_EVENT_INDEXES = [
    ("event_hash",),
    ("object_hash",),
    ("space_id", "event_kind"),
    ("space_id", "object_kind"),
    ("space_id", "owner_tenant_id", "owner_user_id", "owner_session_id"),
    ("space_id", "created_seq"),
    ("space_id", "observed_seq"),
    ("space_id", "used_seq"),
]


class MemoryEventIntegrityError(RuntimeError):
    """Memory 事件信封、完整身份或引用链不一致。"""


def register_memory_event_table(backend: StorageBackend) -> None:
    """注册 Memory 事件信封和固定宽度 payload chunk 核心表。"""
    backend.register_table(
        MEMORY_EVENT_TABLE,
        MEMORY_EVENT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        MEMORY_EVENT_INDEXES,
        core=True,
    )
    backend.register_table(
        MEMORY_EVENT_PART_TABLE,
        MEMORY_EVENT_PART_COLUMNS,
        disc.DISC_APPEND_ONLY,
        MEMORY_EVENT_PART_INDEXES,
        core=True,
    )


def _strict_int(value: int, *, where: str,
                positive: bool = False, nonnegative: bool = False) -> int:
    """校验事件物理字段为严格整数，并按需限制范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


@dataclass(frozen=True)
class MemoryEventRecord:
    """一个事件的固定物理索引信封，不承载 payload 语义。"""

    event_hash: int
    object_hash: int
    space_id: int
    memory_space_key: tuple[int, int, int]
    owner_key: tuple[int, int, int, int]
    event_kind: int
    object_kind: int
    payload_size: int
    scope_hash: int
    timestamp_hash: int
    clock_hash: int
    event_seq: int
    created_seq: int
    observed_seq: int
    used_seq: int

    def __post_init__(self) -> None:
        """核验固定键宽度、正整数索引和唯一非零时间轴。"""
        for name, value in (
                ("event_hash", self.event_hash),
                ("object_hash", self.object_hash),
                ("space_id", self.space_id),
                ("event_kind", self.event_kind),
                ("object_kind", self.object_kind),
                ("scope_hash", self.scope_hash),
                ("timestamp_hash", self.timestamp_hash),
                ("clock_hash", self.clock_hash),
                ("event_seq", self.event_seq)):
            _strict_int(value, where=f"MemoryEventRecord.{name}", positive=True)
        _strict_int(
            self.payload_size,
            where="MemoryEventRecord.payload_size",
            positive=True,
        )
        if (not isinstance(self.memory_space_key, tuple)
                or len(self.memory_space_key) != 3):
            raise ValueError("memory_space_key 必须是三整数键")
        if not isinstance(self.owner_key, tuple) or len(self.owner_key) != 4:
            raise ValueError("owner_key 必须是四整数键")
        for name, values in (
                ("memory_space_key", self.memory_space_key),
                ("owner_key", self.owner_key)):
            for index, value in enumerate(values):
                _strict_int(
                    value, where=f"MemoryEventRecord.{name}[{index}]",
                    nonnegative=True)
        axes = (self.created_seq, self.observed_seq, self.used_seq)
        for index, value in enumerate(axes):
            _strict_int(
                value, where=f"MemoryEventRecord.time_axis[{index}]",
                nonnegative=True)
        nonzero = tuple(value for value in axes if value > 0)
        if nonzero != (self.event_seq,):
            raise ValueError("事件必须且只能在一个时间轴保存 event_seq")

    def to_row(self) -> dict[str, int]:
        """把信封投影为 backend 行。"""
        return {
            "event_hash": self.event_hash,
            "object_hash": self.object_hash,
            "space_id": self.space_id,
            "memory_space_type": self.memory_space_key[0],
            "memory_space_type_hash": self.memory_space_key[1],
            "memory_space_name_hash": self.memory_space_key[2],
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
            "event_kind": self.event_kind,
            "object_kind": self.object_kind,
            "payload_size": self.payload_size,
            "scope_hash": self.scope_hash,
            "timestamp_hash": self.timestamp_hash,
            "clock_hash": self.clock_hash,
            "event_seq": self.event_seq,
            "created_seq": self.created_seq,
            "observed_seq": self.observed_seq,
            "used_seq": self.used_seq,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryEventRecord":
        """从 backend 行恢复固定信封，缺列和损坏字段直接失败。"""
        try:
            return cls(
                row["event_hash"],
                row["object_hash"],
                row["space_id"],
                (
                    row["memory_space_type"],
                    row["memory_space_type_hash"],
                    row["memory_space_name_hash"],
                ),
                (
                    row["owner_tenant_id"],
                    row["owner_user_id"],
                    row["owner_session_id"],
                    row["owner_visibility"],
                ),
                row["event_kind"],
                row["object_kind"],
                row["payload_size"],
                row["scope_hash"],
                row["timestamp_hash"],
                row["clock_hash"],
                row["event_seq"],
                row["created_seq"],
                row["observed_seq"],
                row["used_seq"],
            )
        except KeyError as exc:
            raise MemoryEventIntegrityError(
                f"Memory event 行缺少字段 {exc.args[0]}") from exc


class MemoryEventRecordStore:
    """Memory 事件固定信封的严格 append/read/query facade。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定一个已注册 memory_event 表的 backend。"""
        self.backend = backend
        self._by_hash: dict[int, MemoryEventRecord] = {}

    def add(self, record: MemoryEventRecord,
            payload_key: tuple[int, ...]) -> MemoryEventRecord:
        """幂等追加信封和单份 chunk payload，重复、半写或碰撞失败。"""
        if not isinstance(record, MemoryEventRecord):
            raise TypeError("record 必须是 MemoryEventRecord")
        self._validate_payload_key(record, payload_key)
        rows = self.backend.select(
            MEMORY_EVENT_TABLE,
            where={"event_hash": record.event_hash},
        )
        if rows:
            if len(rows) != 1:
                raise MemoryEventIntegrityError("同一 event_hash 存在重复事件行")
            existing = MemoryEventRecord.from_row(rows[0])
            if existing != record:
                raise MemoryEventIntegrityError("event_hash 命中不同事件信封")
            if self.read_payload(record.event_hash) != payload_key:
                raise MemoryEventIntegrityError("event_hash 命中不同 payload")
            self._by_hash[record.event_hash] = existing
            return existing
        if self.backend.select(
                MEMORY_EVENT_PART_TABLE,
                where={"event_hash": record.event_hash}):
            raise MemoryEventIntegrityError("事件信封缺失但存在孤儿 payload chunk")
        self._append_payload(record, payload_key)
        self.backend.insert(MEMORY_EVENT_TABLE, record.to_row())
        restored = self.read(record.event_hash)
        if (restored != record
                or self.read_payload(record.event_hash) != payload_key):
            raise MemoryEventIntegrityError("Memory event 写后核验失败")
        return restored

    def read_payload(self, event_hash: int) -> tuple[int, ...]:
        """按 event_hash 顺序恢复唯一 chunk 流，并核验大小、索引和填充位。"""
        record = self.read(event_hash)
        rows = self.backend.select(
            MEMORY_EVENT_PART_TABLE,
            where={"event_hash": event_hash},
            order_by="chunk_index",
        )
        expected_chunks = (
            record.payload_size + MEMORY_EVENT_CHUNK_WIDTH - 1
        ) // MEMORY_EVENT_CHUNK_WIDTH
        if len(rows) != expected_chunks:
            raise MemoryEventIntegrityError("Memory event payload chunk 数量不完整")
        result: list[int] = []
        for expected_index, row in enumerate(rows):
            try:
                if (row["space_id"] != record.space_id
                        or row["chunk_index"] != expected_index):
                    raise MemoryEventIntegrityError(
                        "Memory event payload chunk 空间或序号漂移")
                part_size = row["part_size"]
                _strict_int(
                    part_size,
                    where="MemoryEventPart.part_size",
                    positive=True,
                )
                if part_size > MEMORY_EVENT_CHUNK_WIDTH:
                    raise MemoryEventIntegrityError(
                        "Memory event payload chunk 宽度非法")
                values = tuple(
                    row[f"part_{index:02d}"]
                    for index in range(MEMORY_EVENT_CHUNK_WIDTH)
                )
            except KeyError as exc:
                raise MemoryEventIntegrityError(
                    f"Memory event payload chunk 缺字段 {exc.args[0]}") from exc
            for index, value in enumerate(values):
                _strict_int(
                    value,
                    where=f"MemoryEventPart.value[{index}]",
                )
            if any(value != 0 for value in values[part_size:]):
                raise MemoryEventIntegrityError(
                    "Memory event payload chunk 填充位必须为零")
            result.extend(values[:part_size])
        payload = tuple(result)
        if len(payload) != record.payload_size:
            raise MemoryEventIntegrityError("Memory event payload_size 不一致")
        return payload

    def read(self, event_hash: int) -> MemoryEventRecord:
        """按 event_hash 读取唯一信封；缓存不能掩盖物理重复或损坏。"""
        _strict_int(event_hash, where="MemoryEventRecordStore.event_hash", positive=True)
        rows = self.backend.select(
            MEMORY_EVENT_TABLE, where={"event_hash": event_hash})
        if len(rows) != 1:
            raise MemoryEventIntegrityError("event_hash 没有唯一 Memory event 行")
        restored = MemoryEventRecord.from_row(rows[0])
        cached = self._by_hash.get(event_hash)
        if cached is not None and cached != restored:
            raise MemoryEventIntegrityError("Memory event 物理行在缓存后发生漂移")
        self._by_hash[event_hash] = restored
        return restored

    def query(self, *, space_id: int,
              event_kind: int | None = None,
              object_kind: int | None = None,
              object_hash: int | None = None) -> tuple[MemoryEventRecord, ...]:
        """按 Memory 空间和可选种类/对象 hash 读取全部信封。"""
        _strict_int(space_id, where="MemoryEventRecordStore.space_id", positive=True)
        where = {"space_id": space_id}
        for column, value in (
                ("event_kind", event_kind),
                ("object_kind", object_kind),
                ("object_hash", object_hash)):
            if value is not None:
                _strict_int(value, where=f"MemoryEventRecordStore.{column}", positive=True)
                where[column] = value
        records = tuple(
            MemoryEventRecord.from_row(row)
            for row in self.backend.select(MEMORY_EVENT_TABLE, where=where)
        )
        hashes = tuple(record.event_hash for record in records)
        if len(set(hashes)) != len(hashes):
            raise MemoryEventIntegrityError("Memory event 查询包含重复 event_hash")
        for record in records:
            cached = self._by_hash.get(record.event_hash)
            if cached is not None and cached != record:
                raise MemoryEventIntegrityError("Memory event 查询发现缓存后漂移")
            self._by_hash[record.event_hash] = record
        return records

    def rows_for_object(self, object_hash: int) -> tuple[MemoryEventRecord, ...]:
        """跨 Memory 空间读取某对象的全部事件信封，供引用完整性核验。"""
        _strict_int(
            object_hash, where="MemoryEventRecordStore.object_hash", positive=True)
        records = tuple(
            MemoryEventRecord.from_row(row)
            for row in self.backend.select(
                MEMORY_EVENT_TABLE, where={"object_hash": object_hash})
        )
        hashes = tuple(record.event_hash for record in records)
        if len(set(hashes)) != len(hashes):
            raise MemoryEventIntegrityError("Memory 对象包含重复 event_hash")
        return records

    def clear_runtime_caches(self) -> None:
        """清空信封缓存，供 dump/load 或损坏对抗重新核验。"""
        self._by_hash.clear()

    @staticmethod
    def _validate_payload_key(record: MemoryEventRecord,
                              payload_key: tuple[int, ...]) -> None:
        """核验 payload 是与信封声明大小一致的非空严格整数键。"""
        if (not isinstance(payload_key, tuple)
                or len(payload_key) != record.payload_size
                or not payload_key):
            raise ValueError("Memory event payload_key 大小非法")
        assert_int(*payload_key, _where="MemoryEvent.payload_key")
        if any(type(value) is not int for value in payload_key):
            raise ValueError("Memory event payload_key 必须使用严格整数")

    def _append_payload(self, record: MemoryEventRecord,
                        payload_key: tuple[int, ...]) -> None:
        """把 payload 按固定纯整数宽度追加，最后一块以零填充。"""
        for chunk_index, start in enumerate(
                range(0, len(payload_key), MEMORY_EVENT_CHUNK_WIDTH)):
            values = payload_key[start:start + MEMORY_EVENT_CHUNK_WIDTH]
            padded = (*values, *(0 for _ in range(
                MEMORY_EVENT_CHUNK_WIDTH - len(values))))
            row = {
                "event_hash": record.event_hash,
                "space_id": record.space_id,
                "chunk_index": chunk_index,
                "part_size": len(values),
            }
            row.update({
                f"part_{index:02d}": value
                for index, value in enumerate(padded)
            })
            self.backend.insert(MEMORY_EVENT_PART_TABLE, row)


__all__ = [
    "MEMORY_EVENT_TABLE",
    "MEMORY_EVENT_PART_TABLE",
    "MemoryEventIntegrityError",
    "MemoryEventRecord",
    "MemoryEventRecordStore",
    "register_memory_event_table",
]
