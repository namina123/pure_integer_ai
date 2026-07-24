"""K-02 可重建 segment 的无目标释放提交与恢复记录。"""
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
    MIGRATION_PHASE_ABORTED,
    MIGRATION_PHASE_PREPARED,
    MIGRATION_PHASE_PUBLISHED,
    MIGRATION_PHASE_RECLAIMED,
    SegmentCopyReference,
)


SEGMENT_RELEASE_FORMAT_VERSION = 1


class SegmentReleaseIntegrityError(RuntimeError):
    """可重建段释放阶段、源副本或 manifest epoch 不一致。"""


@dataclass(frozen=True)
class SegmentReleaseCommitRecord:
    """一次无目标 segment 释放在某阶段的完整 append-only 恢复记录。"""

    release_key: tuple[int, ...]
    phase: int
    descriptor_key: tuple[int, ...]
    source_copies: tuple[SegmentCopyReference, ...]
    previous_epoch: int
    publish_epoch: int
    manifest_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验释放身份、阶段、非空源副本集和连续发布 epoch。"""
        strict_integer_tuple(self.release_key, label="segment release key")
        strict_integer_tuple(
            self.descriptor_key, label="segment release descriptor_key")
        strict_integer_tuple(
            self.manifest_key, label="segment release manifest_key")
        if self.phase not in {
                MIGRATION_PHASE_PREPARED,
                MIGRATION_PHASE_PUBLISHED,
                MIGRATION_PHASE_RECLAIMED,
                MIGRATION_PHASE_ABORTED}:
            raise ValueError("segment release phase 未注册")
        if (not isinstance(self.source_copies, tuple)
                or not self.source_copies
                or any(not isinstance(item, SegmentCopyReference)
                       for item in self.source_copies)):
            raise TypeError("segment release source_copies 必须是非空 tuple")
        sources = tuple(sorted(self.source_copies))
        if len(set(sources)) != len(sources):
            raise ValueError("segment release source_copies 不得重复")
        object.__setattr__(self, "source_copies", sources)
        if type(self.previous_epoch) is not int or self.previous_epoch <= 0:
            raise ValueError("segment release previous_epoch 必须是正严格整数")
        if (type(self.publish_epoch) is not int
                or self.publish_epoch != self.previous_epoch + 1):
            raise ValueError("segment release publish_epoch 必须紧随 previous_epoch")

    def with_phase(self, phase: int) -> "SegmentReleaseCommitRecord":
        """复制同一释放事实并只推进 append-only 阶段。"""
        return SegmentReleaseCommitRecord(
            self.release_key,
            phase,
            self.descriptor_key,
            self.source_copies,
            self.previous_epoch,
            self.publish_epoch,
            self.manifest_key,
        )

    def identity_key(self) -> tuple[int, ...]:
        """返回该释放阶段在对象仓库中的完整唯一身份。"""
        return len(self.release_key), *self.release_key, self.phase

    def base_key(self) -> tuple[int, ...]:
        """返回忽略阶段后的全部释放事实，用于阶段链漂移核验。"""
        result: list[int] = []
        self._pack_common(result)
        result.extend((self.previous_epoch, self.publish_epoch))
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把释放阶段记录编码为版本化规范整数字节。"""
        result = [SEGMENT_RELEASE_FORMAT_VERSION, self.phase]
        self._pack_common(result)
        result.extend((self.previous_epoch, self.publish_epoch))
        return encode_integer_tuple(tuple(result))

    def _pack_common(self, result: list[int]) -> None:
        """按稳定字段顺序写入身份、描述、manifest 和全部源副本。"""
        for value in (
                self.release_key,
                self.descriptor_key,
                self.manifest_key):
            pack_key(result, value)
        result.append(len(self.source_copies))
        for source in self.source_copies:
            pack_key(result, source.segment_key)
            pack_key(result, source.tier_key)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SegmentReleaseCommitRecord":
        """从规范字节恢复释放阶段，并拒绝未知版本和尾字段。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            version = reader.read_positive(label="segment release format")
            if version != SEGMENT_RELEASE_FORMAT_VERSION:
                raise SegmentReleaseIntegrityError(
                    "segment release format 不兼容")
            phase = reader.read_positive(label="segment release phase")
            release_key = reader.read_key(label="segment release key")
            descriptor_key = reader.read_key(
                label="segment release descriptor")
            manifest_key = reader.read_key(label="segment release manifest")
            source_count = reader.read_positive(
                label="segment release source_count")
            sources = tuple(SegmentCopyReference(
                reader.read_key(label="segment release source segment"),
                reader.read_key(label="segment release source tier"),
            ) for _ in range(source_count))
            previous_epoch = reader.read_positive(
                label="segment release previous_epoch")
            publish_epoch = reader.read_positive(
                label="segment release publish_epoch")
            reader.finish()
            return cls(
                release_key,
                phase,
                descriptor_key,
                sources,
                previous_epoch,
                publish_epoch,
                manifest_key,
            )
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, SegmentReleaseIntegrityError):
                raise
            raise SegmentReleaseIntegrityError(
                "segment release 编码损坏") from exc


def validate_release_chain(
        records: tuple[SegmentReleaseCommitRecord, ...],
        ) -> tuple[SegmentReleaseCommitRecord, ...]:
    """规范化同一释放阶段链并拒绝跳阶段、分叉或事实漂移。"""
    if (not isinstance(records, tuple) or not records
            or any(not isinstance(item, SegmentReleaseCommitRecord)
                   for item in records)):
        raise TypeError("segment release chain 必须是非空记录 tuple")
    if len({item.release_key for item in records}) != 1:
        raise SegmentReleaseIntegrityError("release chain 混入多个 release")
    if len({item.base_key() for item in records}) != 1:
        raise SegmentReleaseIntegrityError("release chain 释放事实发生漂移")
    by_phase = {item.phase: item for item in records}
    if len(by_phase) != len(records):
        raise SegmentReleaseIntegrityError("release chain 重复 phase")
    if MIGRATION_PHASE_PREPARED not in by_phase:
        raise SegmentReleaseIntegrityError("release chain 缺少 prepared")
    phases = set(by_phase)
    valid = (
        {MIGRATION_PHASE_PREPARED},
        {MIGRATION_PHASE_PREPARED, MIGRATION_PHASE_PUBLISHED},
        {
            MIGRATION_PHASE_PREPARED,
            MIGRATION_PHASE_PUBLISHED,
            MIGRATION_PHASE_RECLAIMED,
        },
        {MIGRATION_PHASE_PREPARED, MIGRATION_PHASE_ABORTED},
    )
    if phases not in valid:
        raise SegmentReleaseIntegrityError("release chain 阶段组合非法")
    return tuple(sorted(records, key=lambda item: item.phase))


__all__ = [
    "SEGMENT_RELEASE_FORMAT_VERSION",
    "SegmentReleaseCommitRecord",
    "SegmentReleaseIntegrityError",
    "validate_release_chain",
]
