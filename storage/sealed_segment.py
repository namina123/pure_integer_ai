"""开放热 delta 和不可变 sealed segment 的纯整数领域协议。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.segment_dependency import (
    SegmentDependency,
    canonical_dependencies,
)


SEALED_SEGMENT_FORMAT_VERSION = 1


class SegmentIntegrityError(RuntimeError):
    """段身份、记录顺序、校验和或规范编码不一致。"""


class SegmentBudgetExceeded(RuntimeError):
    """开放 delta 或单次读取超过调用方注入的对象/字节预算。"""


@dataclass(frozen=True)
class SegmentBudget:
    """开放 delta 或分页读取使用的对象数和规范字节硬预算。"""

    object_limit: int
    byte_limit: int

    def __post_init__(self) -> None:
        """要求两个预算均为正严格整数。"""
        if (type(self.object_limit) is not int or self.object_limit <= 0
                or type(self.byte_limit) is not int or self.byte_limit <= 0):
            raise ValueError("segment budget 必须使用正严格整数")


@dataclass(frozen=True, order=True)
class SegmentRecord:
    """sealed segment 中一条完整稳定键和纯整数载荷记录。"""

    record_key: tuple[int, ...]
    payload: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验记录身份非空，并允许业务载荷为空。"""
        strict_integer_tuple(self.record_key, label="segment record_key")
        strict_integer_tuple(
            self.payload, label="segment record payload", empty=True)

    def integer_stream(self) -> tuple[int, ...]:
        """返回记录身份和载荷的完整分帧整数流。"""
        result: list[int] = []
        pack_key(result, self.record_key)
        pack_key(result, self.payload)
        return tuple(result)

    def size_bytes(self) -> int:
        """返回该记录规范整数编码的实际字节数。"""
        return len(encode_integer_tuple(self.integer_stream()))


@dataclass(frozen=True)
class SealedSegment:
    """一个按完整稳定键排序、封存后不可变的物理 segment。"""

    descriptor_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    version_key: tuple[int, ...]
    dependencies: tuple[SegmentDependency, ...]
    read_fence: int
    records: tuple[SegmentRecord, ...]

    def __post_init__(self) -> None:
        """核验段身份、依赖、读屏障和记录的唯一严格顺序。"""
        strict_integer_tuple(
            self.descriptor_key, label="sealed segment descriptor_key")
        strict_integer_tuple(
            self.segment_key, label="sealed segment segment_key")
        strict_integer_tuple(
            self.version_key, label="sealed segment version_key")
        dependencies = canonical_dependencies(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        if type(self.read_fence) is not int or self.read_fence < 0:
            raise ValueError("sealed segment read_fence 必须是非负严格整数")
        if (not isinstance(self.records, tuple)
                or any(not isinstance(item, SegmentRecord)
                       for item in self.records)
                or not self.records):
            raise ValueError("sealed segment records 必须是非空 SegmentRecord tuple")
        ordered = tuple(sorted(self.records, key=lambda item: item.record_key))
        if len({item.record_key for item in ordered}) != len(ordered):
            raise ValueError("sealed segment record_key 不得重复")
        object.__setattr__(self, "records", ordered)

    @property
    def lower_key(self) -> tuple[int, ...]:
        """返回段内最小完整稳定键。"""
        return self.records[0].record_key

    @property
    def upper_key(self) -> tuple[int, ...]:
        """返回段内最大完整稳定键。"""
        return self.records[-1].record_key

    @property
    def checksum_key(self) -> tuple[int, ...]:
        """返回规范段字节的完整 SHA-256 字节整数键。"""
        return tuple(hashlib.sha256(self.to_bytes()).digest())

    @property
    def size_bytes(self) -> int:
        """返回规范段编码的实际字节数。"""
        return len(self.to_bytes())

    def integer_stream(self) -> tuple[int, ...]:
        """形成版本化、可逆且不依赖对象 repr 的规范整数流。"""
        result: list[int] = [SEALED_SEGMENT_FORMAT_VERSION]
        pack_key(result, self.descriptor_key)
        pack_key(result, self.segment_key)
        pack_key(result, self.version_key)
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            pack_key(result, dependency.descriptor_key)
            pack_key(result, dependency.version_key)
            pack_key(result, dependency.checksum_key)
        result.extend((self.read_fence, len(self.records)))
        for record in self.records:
            pack_key(result, record.record_key)
            pack_key(result, record.payload)
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把 segment 编为确定性规范字节，供任意介质封存。"""
        return encode_integer_tuple(self.integer_stream())

    @classmethod
    def from_bytes(cls, data: bytes) -> "SealedSegment":
        """从规范字节恢复 segment，并重建全部身份与排序不变量。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            version = reader.read_positive(label="sealed segment format")
            if version != SEALED_SEGMENT_FORMAT_VERSION:
                raise SegmentIntegrityError("sealed segment format 不兼容")
            descriptor_key = reader.read_key(label="sealed descriptor_key")
            segment_key = reader.read_key(label="sealed segment_key")
            version_key = reader.read_key(label="sealed version_key")
            dependency_count = reader.read_nonnegative(
                label="sealed dependency_count")
            dependencies = []
            for _ in range(dependency_count):
                dependencies.append(SegmentDependency(
                    reader.read_key(label="sealed dependency descriptor"),
                    reader.read_key(label="sealed dependency version"),
                    reader.read_key(label="sealed dependency checksum"),
                ))
            read_fence = reader.read_nonnegative(label="sealed read_fence")
            record_count = reader.read_positive(label="sealed record_count")
            records = []
            for _ in range(record_count):
                records.append(SegmentRecord(
                    reader.read_key(label="sealed record_key"),
                    reader.read_key(
                        label="sealed record payload", empty=True),
                ))
            reader.finish()
            return cls(
                descriptor_key,
                segment_key,
                version_key,
                tuple(dependencies),
                read_fence,
                tuple(records),
            )
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, SegmentIntegrityError):
                raise
            raise SegmentIntegrityError("sealed segment 编码损坏") from exc


class OpenHotDelta:
    """按注入预算收集尚未发布的记录，并批量封存为 segment。"""

    def __init__(
            self,
            descriptor_key: tuple[int, ...],
            version_key: tuple[int, ...],
            dependencies: tuple[SegmentDependency, ...],
            budget: SegmentBudget,
            ) -> None:
        """绑定逻辑描述、版本、依赖和硬预算，不创建物理位置。"""
        self.descriptor_key = strict_integer_tuple(
            descriptor_key, label="hot delta descriptor_key")
        self.version_key = strict_integer_tuple(
            version_key, label="hot delta version_key")
        self.dependencies = canonical_dependencies(dependencies)
        if not isinstance(budget, SegmentBudget):
            raise TypeError("hot delta budget 类型错误")
        self.budget = budget
        self._records: dict[tuple[int, ...], SegmentRecord] = {}
        self._size_bytes = 0

    @property
    def object_count(self) -> int:
        """返回当前尚未封存的唯一记录数。"""
        return len(self._records)

    @property
    def size_bytes(self) -> int:
        """返回当前记录规范编码字节数之和。"""
        return self._size_bytes

    def append(self, record: SegmentRecord) -> bool:
        """幂等追加一条记录；身份漂移或预算越界时不改变 delta。"""
        if not isinstance(record, SegmentRecord):
            raise TypeError("hot delta record 类型错误")
        previous = self._records.get(record.record_key)
        if previous is not None:
            if previous != record:
                raise SegmentIntegrityError("hot delta 同键载荷漂移")
            return False
        size = record.size_bytes()
        if len(self._records) + 1 > self.budget.object_limit:
            raise SegmentBudgetExceeded("hot delta 超过对象数预算")
        if self._size_bytes + size > self.budget.byte_limit:
            raise SegmentBudgetExceeded("hot delta 超过字节预算")
        self._records[record.record_key] = record
        self._size_bytes += size
        return True

    def seal(
            self,
            segment_key: tuple[int, ...],
            read_fence: int,
            ) -> SealedSegment:
        """从当前批次形成不可变段，发布确认前仍保留开放 delta。"""
        if not self._records:
            raise ValueError("空 hot delta 不得封存")
        return SealedSegment(
            self.descriptor_key,
            strict_integer_tuple(segment_key, label="hot delta segment_key"),
            self.version_key,
            self.dependencies,
            read_fence,
            tuple(self._records.values()),
        )

    def acknowledge(self, segment: SealedSegment) -> None:
        """在段已完整发布后清空与该段逐条一致的开放 delta。"""
        if not isinstance(segment, SealedSegment):
            raise TypeError("acknowledged segment 类型错误")
        if (segment.descriptor_key != self.descriptor_key
                or segment.version_key != self.version_key
                or segment.dependencies != self.dependencies
                or segment.records != tuple(sorted(
                    self._records.values(), key=lambda item: item.record_key))):
            raise SegmentIntegrityError("发布确认的 segment 与 hot delta 漂移")
        self._records.clear()
        self._size_bytes = 0


__all__ = [
    "OpenHotDelta",
    "SEALED_SEGMENT_FORMAT_VERSION",
    "SealedSegment",
    "SegmentBudget",
    "SegmentBudgetExceeded",
    "SegmentIntegrityError",
    "SegmentRecord",
]
