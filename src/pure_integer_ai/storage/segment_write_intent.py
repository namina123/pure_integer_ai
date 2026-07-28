"""K-02 target segment 写前的轻量、可恢复 intent。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.segment_commit import (
    MIGRATION_PHASE_PREPARED,
    MigrationCommitRecord,
)


SEGMENT_WRITE_INTENT_FORMAT_VERSION = 1


class SegmentWriteIntentIntegrityError(RuntimeError):
    """segment write intent 编码、身份或 prepared 绑定不完整。"""


@dataclass(frozen=True)
class SegmentWriteIntent:
    """在大 payload 写入前持久化的最小 target 回收身份。"""

    migration_key: tuple[int, ...]
    descriptor_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    target_tier_key: tuple[int, ...]
    version_key: tuple[int, ...]
    checksum_key: tuple[int, ...]
    read_fence: int

    def __post_init__(self) -> None:
        for label, value in (
                ("migration_key", self.migration_key),
                ("descriptor_key", self.descriptor_key),
                ("segment_key", self.segment_key),
                ("target_tier_key", self.target_tier_key),
                ("version_key", self.version_key),
                ("checksum_key", self.checksum_key)):
            strict_integer_tuple(value, label=f"segment write intent {label}")
        if type(self.read_fence) is not int or self.read_fence < 0:
            raise ValueError("segment write intent read_fence 必须是非负严格整数")

    @classmethod
    def from_prepared(
            cls,
            record: MigrationCommitRecord,
            ) -> "SegmentWriteIntent":
        """从尚未写 target 的 PREPARED 内容裁出最小 intent。"""
        if not isinstance(record, MigrationCommitRecord):
            raise TypeError("segment write intent prepared 类型错误")
        if record.phase != MIGRATION_PHASE_PREPARED:
            raise SegmentWriteIntentIntegrityError(
                "segment write intent 只能绑定 PREPARED 内容")
        return cls(
            record.migration_key,
            record.descriptor_key,
            record.segment_key,
            record.target_tier_key,
            record.version_key,
            record.checksum_key,
            record.read_fence,
        )

    def identity_key(self) -> tuple[int, ...]:
        """返回同一 migration 唯一的 intent 对象身份。"""
        return (len(self.migration_key), *self.migration_key)

    def target_copy_key(self) -> tuple[int, ...]:
        """返回独立编码的 target tier 与 segment 键。"""
        result: list[int] = []
        pack_key(result, self.target_tier_key)
        pack_key(result, self.segment_key)
        return tuple(result)

    def to_bytes(self) -> bytes:
        """编码全部 target 校验字段，不复制源段或 manifest payload。"""
        result = [SEGMENT_WRITE_INTENT_FORMAT_VERSION]
        for value in (
                self.migration_key,
                self.descriptor_key,
                self.segment_key,
                self.target_tier_key,
                self.version_key,
                self.checksum_key):
            pack_key(result, value)
        result.append(self.read_fence)
        return encode_integer_tuple(tuple(result))

    @classmethod
    def from_bytes(cls, data: bytes) -> "SegmentWriteIntent":
        """严格恢复 intent，拒绝未知版本、尾字段和整数损坏。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            version = reader.read_positive(label="segment write intent format")
            if version != SEGMENT_WRITE_INTENT_FORMAT_VERSION:
                raise SegmentWriteIntentIntegrityError(
                    "segment write intent format 不兼容")
            result = cls(
                reader.read_key(label="segment write intent migration"),
                reader.read_key(label="segment write intent descriptor"),
                reader.read_key(label="segment write intent segment"),
                reader.read_key(label="segment write intent target tier"),
                reader.read_key(label="segment write intent version"),
                reader.read_key(label="segment write intent checksum"),
                reader.read_nonnegative(label="segment write intent read fence"),
            )
            reader.finish()
            return result
        except (IntegerCodecError, TypeError, ValueError) as error:
            if isinstance(error, SegmentWriteIntentIntegrityError):
                raise
            raise SegmentWriteIntentIntegrityError(
                "segment write intent 编码损坏") from error


__all__ = [
    "SEGMENT_WRITE_INTENT_FORMAT_VERSION",
    "SegmentWriteIntent",
    "SegmentWriteIntentIntegrityError",
]
