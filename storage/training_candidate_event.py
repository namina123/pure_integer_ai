"""断奶训练候选历史的 append-only Core 物理记录。

该日志只保存训练期 H-00/H-04 事件，不属于断奶后 Memory。完整协议键和事件键
只保存一份固定宽度 chunk，信封中的 hash 仅用于索引，每次读取都会回验完整内容。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


TRAINING_CANDIDATE_EVENT_TABLE = "training_candidate_event"
TRAINING_CANDIDATE_EVENT_PART_TABLE = "training_candidate_event_part"
TRAINING_CANDIDATE_EVENT_CHUNK_WIDTH = 16

TRAINING_CANDIDATE_EVENT_COLUMNS = [
    ("event_hash", TYPE_INT),
    ("space_id", TYPE_INT),
    ("protocol_hash", TYPE_INT),
    ("event_kind", TYPE_INT),
    ("event_seq", TYPE_INT),
    ("original_size", TYPE_INT),
    ("payload_size", TYPE_INT),
]

TRAINING_CANDIDATE_EVENT_PART_COLUMNS = [
    ("event_hash", TYPE_INT),
    ("space_id", TYPE_INT),
    ("chunk_index", TYPE_INT),
    ("part_size", TYPE_INT),
    *((f"part_{index:02d}", TYPE_INT)
      for index in range(TRAINING_CANDIDATE_EVENT_CHUNK_WIDTH)),
]


class TrainingCandidateEventIntegrityError(RuntimeError):
    """训练候选事件信封、协议或完整 payload 不一致。"""


_INTEGER_STREAM_CODEC_VERSION = 1
_INTEGER_STREAM_WORD_BYTES = 7


def encode_integer_stream(values: tuple[int, ...]) -> tuple[int, ...]:
    """用 zigzag varint 和 56-bit word 可逆压缩严格整数流。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError("待编码整数流必须是非空 tuple")
    assert_int(*values, _where="training candidate integer stream")
    if any(type(value) is not int for value in values):
        raise ValueError("待编码整数流必须使用严格整数")
    encoded = bytearray()
    for value in values:
        unsigned = value * 2 if value >= 0 else (-value * 2) - 1
        while unsigned >= 128:
            encoded.append((unsigned & 127) | 128)
            unsigned >>= 7
        encoded.append(unsigned)
    words = []
    width = _INTEGER_STREAM_WORD_BYTES
    for start in range(0, len(encoded), width):
        word = 0
        for offset, value in enumerate(encoded[start:start + width]):
            word |= value << (offset * 8)
        words.append(word)
    return (
        _INTEGER_STREAM_CODEC_VERSION,
        len(values),
        len(encoded),
        *words,
    )


def decode_integer_stream(values: tuple[int, ...]) -> tuple[int, ...]:
    """解码整数流并拒绝版本、填充位、varint 和原始长度损坏。"""
    if (not isinstance(values, tuple) or len(values) < 4
            or values[0] != _INTEGER_STREAM_CODEC_VERSION):
        raise TrainingCandidateEventIntegrityError(
            "训练候选整数流编码版本非法")
    original_size, byte_size = values[1:3]
    if (type(original_size) is not int or original_size <= 0
            or type(byte_size) is not int or byte_size <= 0):
        raise TrainingCandidateEventIntegrityError(
            "训练候选整数流长度非法")
    expected_words = (
        byte_size + _INTEGER_STREAM_WORD_BYTES - 1
    ) // _INTEGER_STREAM_WORD_BYTES
    if len(values) != expected_words + 3:
        raise TrainingCandidateEventIntegrityError(
            "训练候选整数流 word 数量不完整")
    encoded = bytearray()
    for word_index, word in enumerate(values[3:]):
        if type(word) is not int or word < 0 or word >= (1 << 56):
            raise TrainingCandidateEventIntegrityError(
                "训练候选整数流 word 超出 56-bit 范围")
        available = min(
            _INTEGER_STREAM_WORD_BYTES,
            byte_size - word_index * _INTEGER_STREAM_WORD_BYTES,
        )
        for offset in range(available):
            encoded.append((word >> (offset * 8)) & 255)
        if word >> (available * 8):
            raise TrainingCandidateEventIntegrityError(
                "训练候选整数流尾 word 填充位非零")
    result = []
    cursor = 0
    while cursor < len(encoded):
        unsigned = 0
        shift = 0
        while True:
            if cursor >= len(encoded):
                raise TrainingCandidateEventIntegrityError(
                    "训练候选整数流 varint 被截断")
            byte = encoded[cursor]
            cursor += 1
            unsigned |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
        result.append(
            unsigned // 2 if unsigned % 2 == 0
            else -((unsigned + 1) // 2))
    if len(result) != original_size:
        raise TrainingCandidateEventIntegrityError(
            "训练候选整数流原始长度不一致")
    restored = tuple(result)
    if encode_integer_stream(restored) != values:
        raise TrainingCandidateEventIntegrityError(
            "训练候选整数流不是规范编码")
    return restored


def register_training_candidate_event_tables(backend: StorageBackend) -> None:
    """注册训练候选事件信封和固定宽度 payload chunk 核心表。"""
    backend.register_table(
        TRAINING_CANDIDATE_EVENT_TABLE,
        TRAINING_CANDIDATE_EVENT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("event_hash",),
            ("space_id", "protocol_hash"),
            ("space_id", "protocol_hash", "event_kind"),
        ],
        core=True,
    )
    backend.register_table(
        TRAINING_CANDIDATE_EVENT_PART_TABLE,
        TRAINING_CANDIDATE_EVENT_PART_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("event_hash",),
            ("event_hash", "chunk_index"),
            ("space_id", "event_hash"),
        ],
        core=True,
    )


def _strict_int(value: int, *, label: str, positive: bool = False,
                nonnegative: bool = False) -> int:
    """核验物理信封字段为严格整数，并按需限制取值范围。"""
    if type(value) is not int:
        assert_int(value, _where=label)
        raise ValueError(f"{label} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{label} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{label} 必须为非负整数")
    return value


@dataclass(frozen=True)
class TrainingCandidateEventRecord:
    """一条训练候选事件的固定索引信封。"""

    event_hash: int
    space_id: int
    protocol_hash: int
    event_kind: int
    event_seq: int
    original_size: int
    payload_size: int

    def __post_init__(self) -> None:
        """核验信封索引和 payload 大小。"""
        for label, value in (
                ("event_hash", self.event_hash),
                ("space_id", self.space_id),
                ("protocol_hash", self.protocol_hash),
                ("event_kind", self.event_kind),
                ("original_size", self.original_size),
                ("payload_size", self.payload_size)):
            _strict_int(
                value,
                label=f"TrainingCandidateEventRecord.{label}",
                positive=True,
            )
        _strict_int(
            self.event_seq,
            label="TrainingCandidateEventRecord.event_seq",
            nonnegative=True,
        )

    def to_row(self) -> dict[str, int]:
        """把固定信封转换为 backend 行。"""
        return {
            "event_hash": self.event_hash,
            "space_id": self.space_id,
            "protocol_hash": self.protocol_hash,
            "event_kind": self.event_kind,
            "event_seq": self.event_seq,
            "original_size": self.original_size,
            "payload_size": self.payload_size,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TrainingCandidateEventRecord":
        """从 backend 行恢复固定信封，缺列或损坏字段直接失败。"""
        try:
            return cls(
                row["event_hash"],
                row["space_id"],
                row["protocol_hash"],
                row["event_kind"],
                row["event_seq"],
                row["original_size"],
                row["payload_size"],
            )
        except KeyError as exc:
            raise TrainingCandidateEventIntegrityError(
                f"训练候选事件行缺少字段 {exc.args[0]}") from exc


class TrainingCandidateEventRecordStore:
    """训练候选事件信封和完整整数 payload 的严格存取入口。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定已经注册训练候选事件表的 backend。"""
        self.backend = backend

    def add(
            self,
            record: TrainingCandidateEventRecord,
            payload: tuple[int, ...],
            ) -> TrainingCandidateEventRecord:
        """幂等追加事件；重复、hash 碰撞和任一种半写状态均失败。"""
        if not isinstance(record, TrainingCandidateEventRecord):
            raise TypeError("record 必须是 TrainingCandidateEventRecord")
        self._validate_payload(record, payload)
        rows = self.backend.select(
            TRAINING_CANDIDATE_EVENT_TABLE,
            where={"event_hash": record.event_hash},
        )
        if rows:
            if len(rows) != 1:
                raise TrainingCandidateEventIntegrityError(
                    "同一训练候选 event_hash 存在重复信封")
            restored = TrainingCandidateEventRecord.from_row(rows[0])
            if restored != record or self.read_payload(record) != payload:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选 event_hash 命中不同事件")
            return restored
        if self.backend.select(
                TRAINING_CANDIDATE_EVENT_PART_TABLE,
                where={"event_hash": record.event_hash}):
            raise TrainingCandidateEventIntegrityError(
                "训练候选事件信封缺失但存在孤儿 chunk")
        self._append_payload(record, payload)
        self.backend.insert(
            TRAINING_CANDIDATE_EVENT_TABLE,
            record.to_row(),
        )
        restored = self.read(record.event_hash)
        if restored != record or self.read_payload(restored) != payload:
            raise TrainingCandidateEventIntegrityError(
                "训练候选事件写后核验失败")
        return restored

    def read(self, event_hash: int) -> TrainingCandidateEventRecord:
        """按 event_hash 读取唯一事件信封。"""
        _strict_int(
            event_hash,
            label="TrainingCandidateEventRecordStore.event_hash",
            positive=True,
        )
        rows = self.backend.select(
            TRAINING_CANDIDATE_EVENT_TABLE,
            where={"event_hash": event_hash},
        )
        if len(rows) != 1:
            raise TrainingCandidateEventIntegrityError(
                "event_hash 没有唯一训练候选事件信封")
        return TrainingCandidateEventRecord.from_row(rows[0])

    def query(
            self,
            *,
            space_id: int,
            protocol_hash: int,
            event_kind: int | None = None,
            ) -> tuple[TrainingCandidateEventRecord, ...]:
        """按 Core 空间、完整协议 hash 和可选事件种类读取信封。"""
        _strict_int(space_id, label="training history space_id", positive=True)
        _strict_int(
            protocol_hash,
            label="training history protocol_hash",
            positive=True,
        )
        where = {
            "space_id": space_id,
            "protocol_hash": protocol_hash,
        }
        if event_kind is not None:
            _strict_int(
                event_kind,
                label="training history event_kind",
                positive=True,
            )
            where["event_kind"] = event_kind
        records = tuple(
            TrainingCandidateEventRecord.from_row(row)
            for row in self.backend.select(
                TRAINING_CANDIDATE_EVENT_TABLE,
                where=where,
            )
        )
        hashes = tuple(item.event_hash for item in records)
        if len(set(hashes)) != len(hashes):
            raise TrainingCandidateEventIntegrityError(
                "训练候选历史查询返回重复 event_hash")
        return records

    def read_payload(
            self,
            record: TrainingCandidateEventRecord,
            ) -> tuple[int, ...]:
        """按 chunk 顺序恢复完整 payload，并核验空间、长度和填充位。"""
        if not isinstance(record, TrainingCandidateEventRecord):
            raise TypeError("record 必须是 TrainingCandidateEventRecord")
        rows = self.backend.select(
            TRAINING_CANDIDATE_EVENT_PART_TABLE,
            where={"event_hash": record.event_hash},
            order_by="chunk_index",
        )
        width = TRAINING_CANDIDATE_EVENT_CHUNK_WIDTH
        expected = (record.payload_size + width - 1) // width
        if len(rows) != expected:
            raise TrainingCandidateEventIntegrityError(
                "训练候选事件 payload chunk 数量不完整")
        result: list[int] = []
        for expected_index, row in enumerate(rows):
            try:
                if (row["space_id"] != record.space_id
                        or row["chunk_index"] != expected_index):
                    raise TrainingCandidateEventIntegrityError(
                        "训练候选 chunk 空间或顺序漂移")
                part_size = row["part_size"]
                _strict_int(
                    part_size,
                    label="training candidate chunk part_size",
                    positive=True,
                )
                if part_size > width:
                    raise TrainingCandidateEventIntegrityError(
                        "训练候选 chunk 宽度非法")
                values = tuple(
                    row[f"part_{index:02d}"] for index in range(width))
            except KeyError as exc:
                raise TrainingCandidateEventIntegrityError(
                    f"训练候选 chunk 缺字段 {exc.args[0]}") from exc
            assert_int(*values, _where="training candidate event chunk")
            if any(type(value) is not int for value in values):
                raise TrainingCandidateEventIntegrityError(
                    "训练候选 chunk 必须使用严格整数")
            if any(value != 0 for value in values[part_size:]):
                raise TrainingCandidateEventIntegrityError(
                    "训练候选 chunk 填充位必须为零")
            result.extend(values[:part_size])
        payload = tuple(result)
        if len(payload) != record.payload_size:
            raise TrainingCandidateEventIntegrityError(
                "训练候选事件 payload_size 不一致")
        return payload

    @staticmethod
    def _validate_payload(
            record: TrainingCandidateEventRecord,
            payload: tuple[int, ...],
            ) -> None:
        """核验 payload 与信封声明大小一致且只含严格整数。"""
        if (not isinstance(payload, tuple)
                or not payload
                or len(payload) != record.payload_size):
            raise ValueError("训练候选事件 payload 大小非法")
        assert_int(*payload, _where="training candidate event payload")
        if any(type(value) is not int for value in payload):
            raise ValueError("训练候选事件 payload 必须使用严格整数")

    def _append_payload(
            self,
            record: TrainingCandidateEventRecord,
            payload: tuple[int, ...],
            ) -> None:
        """把 payload 分为固定宽度纯整数 chunk 追加到物理表。"""
        width = TRAINING_CANDIDATE_EVENT_CHUNK_WIDTH
        for chunk_index, start in enumerate(range(0, len(payload), width)):
            values = payload[start:start + width]
            padded = (*values, *(0 for _ in range(width - len(values))))
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
            self.backend.insert(TRAINING_CANDIDATE_EVENT_PART_TABLE, row)


__all__ = [
    "TRAINING_CANDIDATE_EVENT_PART_TABLE",
    "TRAINING_CANDIDATE_EVENT_TABLE",
    "TrainingCandidateEventIntegrityError",
    "TrainingCandidateEventRecord",
    "TrainingCandidateEventRecordStore",
    "decode_integer_stream",
    "encode_integer_stream",
    "register_training_candidate_event_tables",
]
