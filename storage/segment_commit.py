"""K-02 段替换提交阶段和可恢复 append-only 记录。"""
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


SEGMENT_COMMIT_FORMAT_VERSION = 1

MIGRATION_PHASE_PREPARED = 1
MIGRATION_PHASE_PUBLISHED = 2
MIGRATION_PHASE_RECLAIMED = 3
MIGRATION_PHASE_ABORTED = 4

_MIGRATION_PHASES = {
    MIGRATION_PHASE_PREPARED,
    MIGRATION_PHASE_PUBLISHED,
    MIGRATION_PHASE_RECLAIMED,
    MIGRATION_PHASE_ABORTED,
}


class SegmentCommitIntegrityError(RuntimeError):
    """段替换提交阶段、对象身份或 epoch 链不一致。"""


@dataclass(frozen=True, order=True)
class SegmentCopyReference:
    """发布下一 epoch 前一个待替换的 segment 物理副本。"""

    segment_key: tuple[int, ...]
    tier_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验源 segment 和温层均使用完整稳定整数键。"""
        strict_integer_tuple(
            self.segment_key, label="source copy segment_key")
        strict_integer_tuple(self.tier_key, label="source copy tier_key")


@dataclass(frozen=True)
class MigrationCommitRecord:
    """一次首次发布、迁移或 compaction 在某阶段的完整恢复记录。"""

    migration_key: tuple[int, ...]
    phase: int
    descriptor_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    source_copies: tuple[SegmentCopyReference, ...]
    target_tier_key: tuple[int, ...]
    version_key: tuple[int, ...]
    checksum_key: tuple[int, ...]
    read_fence: int
    previous_epoch: int
    publish_epoch: int
    manifest_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验替换身份、源副本集、目标、校验和和连续 epoch。"""
        strict_integer_tuple(
            self.migration_key, label="migration commit migration_key")
        if type(self.phase) is not int or self.phase not in _MIGRATION_PHASES:
            raise ValueError("migration commit phase 未注册")
        for label, value in (
                ("descriptor_key", self.descriptor_key),
                ("segment_key", self.segment_key),
                ("target_tier_key", self.target_tier_key),
                ("version_key", self.version_key),
                ("checksum_key", self.checksum_key),
                ("manifest_key", self.manifest_key)):
            strict_integer_tuple(value, label=f"migration commit {label}")
        if (not isinstance(self.source_copies, tuple)
                or any(not isinstance(item, SegmentCopyReference)
                       for item in self.source_copies)):
            raise TypeError("migration source_copies 类型错误")
        sources = tuple(sorted(self.source_copies))
        if len(set(sources)) != len(sources):
            raise ValueError("migration source_copies 不得重复")
        target = SegmentCopyReference(self.segment_key, self.target_tier_key)
        if target in sources:
            raise ValueError("migration target 不得与任一源副本完全相同")
        object.__setattr__(self, "source_copies", sources)
        if type(self.read_fence) is not int or self.read_fence < 0:
            raise ValueError("migration read_fence 必须是非负严格整数")
        if type(self.previous_epoch) is not int or self.previous_epoch < 0:
            raise ValueError("migration previous_epoch 必须是非负严格整数")
        if (type(self.publish_epoch) is not int
                or self.publish_epoch != self.previous_epoch + 1):
            raise ValueError("migration publish_epoch 必须紧随 previous_epoch")

    def with_phase(self, phase: int) -> "MigrationCommitRecord":
        """复制同一段替换事实并只推进到指定 append-only 阶段。"""
        return MigrationCommitRecord(
            self.migration_key,
            phase,
            self.descriptor_key,
            self.segment_key,
            self.source_copies,
            self.target_tier_key,
            self.version_key,
            self.checksum_key,
            self.read_fence,
            self.previous_epoch,
            self.publish_epoch,
            self.manifest_key,
        )

    def identity_key(self) -> tuple[int, ...]:
        """返回该段替换阶段在对象仓库中的完整唯一身份。"""
        return (len(self.migration_key), *self.migration_key, self.phase)

    def base_key(self) -> tuple[int, ...]:
        """返回忽略阶段后的全部替换事实，用于阶段链漂移核验。"""
        result: list[int] = []
        self._pack_common(result)
        result.extend((
            self.read_fence,
            self.previous_epoch,
            self.publish_epoch,
        ))
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把段替换阶段记录编码为版本化规范整数字节。"""
        result: list[int] = [SEGMENT_COMMIT_FORMAT_VERSION, self.phase]
        self._pack_common(result)
        result.extend((
            self.read_fence,
            self.previous_epoch,
            self.publish_epoch,
        ))
        return encode_integer_tuple(tuple(result))

    def _pack_common(self, result: list[int]) -> None:
        """按稳定字段顺序写入身份、目标、版本、校验和和全部源副本。"""
        for value in (
                self.migration_key,
                self.descriptor_key,
                self.segment_key,
                self.target_tier_key,
                self.version_key,
                self.checksum_key,
                self.manifest_key):
            pack_key(result, value)
        result.append(len(self.source_copies))
        for source in self.source_copies:
            pack_key(result, source.segment_key)
            pack_key(result, source.tier_key)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MigrationCommitRecord":
        """从规范字节恢复段替换阶段，并拒绝未知尾字段和格式。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            version = reader.read_positive(label="migration commit format")
            if version != SEGMENT_COMMIT_FORMAT_VERSION:
                raise SegmentCommitIntegrityError(
                    "migration commit format 不兼容")
            phase = reader.read_positive(label="migration commit phase")
            migration_key = reader.read_key(label="migration key")
            descriptor_key = reader.read_key(label="migration descriptor")
            segment_key = reader.read_key(label="migration segment")
            target = reader.read_key(label="migration target tier")
            version_key = reader.read_key(label="migration version")
            checksum_key = reader.read_key(label="migration checksum")
            manifest_key = reader.read_key(label="migration manifest")
            source_count = reader.read_nonnegative(
                label="migration source_count")
            sources = []
            for _ in range(source_count):
                sources.append(SegmentCopyReference(
                    reader.read_key(label="migration source segment"),
                    reader.read_key(label="migration source tier"),
                ))
            read_fence = reader.read_nonnegative(label="migration read_fence")
            previous_epoch = reader.read_nonnegative(
                label="migration previous_epoch")
            publish_epoch = reader.read_positive(
                label="migration publish_epoch")
            reader.finish()
            return cls(
                migration_key,
                phase,
                descriptor_key,
                segment_key,
                tuple(sources),
                target,
                version_key,
                checksum_key,
                read_fence,
                previous_epoch,
                publish_epoch,
                manifest_key,
            )
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, SegmentCommitIntegrityError):
                raise
            raise SegmentCommitIntegrityError(
                "migration commit 编码损坏") from exc


def validate_commit_chain(
        records: tuple[MigrationCommitRecord, ...],
        ) -> tuple[MigrationCommitRecord, ...]:
    """规范化同一段替换阶段链并拒绝跳阶段、分叉或事实漂移。"""
    if (not isinstance(records, tuple) or not records
            or any(not isinstance(item, MigrationCommitRecord)
                   for item in records)):
        raise TypeError("migration commit chain 必须是非空记录 tuple")
    migration_keys = {item.migration_key for item in records}
    if len(migration_keys) != 1:
        raise SegmentCommitIntegrityError("commit chain 混入多个 migration")
    base_keys = {item.base_key() for item in records}
    if len(base_keys) != 1:
        raise SegmentCommitIntegrityError("commit chain 替换事实发生漂移")
    by_phase = {item.phase: item for item in records}
    if len(by_phase) != len(records):
        raise SegmentCommitIntegrityError("commit chain 重复 phase")
    if MIGRATION_PHASE_PREPARED not in by_phase:
        raise SegmentCommitIntegrityError("commit chain 缺少 prepared")
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
        raise SegmentCommitIntegrityError("commit chain 阶段组合非法")
    return tuple(sorted(records, key=lambda item: item.phase))


__all__ = [
    "MIGRATION_PHASE_ABORTED",
    "MIGRATION_PHASE_PREPARED",
    "MIGRATION_PHASE_PUBLISHED",
    "MIGRATION_PHASE_RECLAIMED",
    "MigrationCommitRecord",
    "SEGMENT_COMMIT_FORMAT_VERSION",
    "SegmentCommitIntegrityError",
    "SegmentCopyReference",
    "validate_commit_chain",
]
