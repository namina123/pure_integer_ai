"""Memory 摄入批次的 K-02 staged event、activation、rollback 与事件映射。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
from typing import Iterator

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.integer_codec import (
    IntegerStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.sealed_segment import (
    SealedSegment,
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.storage_role import (
    STORAGE_ACCESS_APPEND_ONLY,
    STORAGE_ACCESS_INDEXED_READ,
    STORAGE_ROLE_AUTHORITATIVE,
    StorageRoleDescriptor,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.storage.source_record import SourceRecordStorage


MEMORY_BATCH_FORMAT_VERSION = 1

MEMORY_BATCH_CORE_DEPENDENCY_KEY = (1, 1, 4)
MEMORY_BATCH_SOURCE_DEPENDENCY_KEY = (1, 1, 5)
MEMORY_BATCH_EVENT_DESCRIPTOR_KEY = (1, 1, 6)
MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY = (1, 1, 7)
MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY = (1, 1, 8)
MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY = (1, 1, 9)
MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY = (1, 1, 10)
MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY = (1, 1, 11)
MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY = (1, 1, 12)

MEMORY_BATCH_CORE_DEPENDENCY = StorageRoleDescriptor(
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_INDEXED_READ,),
)
MEMORY_BATCH_SOURCE_DEPENDENCY = StorageRoleDescriptor(
    MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_INDEXED_READ,),
)
MEMORY_BATCH_EVENT_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(
        MEMORY_BATCH_CORE_DEPENDENCY_KEY,
        MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
    ),
)
MEMORY_BATCH_ACTIVATION_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,),
)
MEMORY_BATCH_ROLLBACK_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,),
)
MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
)
MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,),
)
MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,),
)
MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,),
)

MEMORY_EVENT_BATCH_LINK_TABLE = "memory_event_batch_link"

_STAGE_SEGMENT_TAG = 2026072301
_ACTIVATION_SEGMENT_TAG = 2026072302
_ROLLBACK_SEGMENT_TAG = 2026072303
_GROUP_INTENT_SEGMENT_TAG = 2026072304
_GROUP_MEMBERSHIP_SEGMENT_TAG = 2026072305
_GROUP_COMMIT_SEGMENT_TAG = 2026072306
_GROUP_ROLLBACK_SEGMENT_TAG = 2026072307
_STAGE_MANIFEST_TAG = 2026072311
_ACTIVATION_MANIFEST_TAG = 2026072312
_ROLLBACK_MANIFEST_TAG = 2026072313
_GROUP_INTENT_MANIFEST_TAG = 2026072314
_GROUP_MEMBERSHIP_MANIFEST_TAG = 2026072315
_GROUP_COMMIT_MANIFEST_TAG = 2026072316
_GROUP_ROLLBACK_MANIFEST_TAG = 2026072317
_STAGE_MIGRATION_TAG = 2026072321
_ACTIVATION_MIGRATION_TAG = 2026072322
_ROLLBACK_MIGRATION_TAG = 2026072323
_GROUP_INTENT_MIGRATION_TAG = 2026072324
_GROUP_MEMBERSHIP_MIGRATION_TAG = 2026072325
_GROUP_COMMIT_MIGRATION_TAG = 2026072326
_GROUP_ROLLBACK_MIGRATION_TAG = 2026072327
_READER_TAG = 2026072331
_RANGE_END = 1 << 63

_BATCH_HASHER = Hasher("pure_integer_ai.memory_batch.v1")


class MemoryBatchIntegrityError(RuntimeError):
    """批次身份、receipt、事件映射或 K-02 段出现不一致。"""


def register_memory_batch_table(backend: StorageBackend) -> None:
    """注册 event 到摄入单元的多对多 append-only 完整映射。"""
    backend.register_table(
        MEMORY_EVENT_BATCH_LINK_TABLE,
        [
            ("batch_hash", TYPE_INT),
            ("batch_id", TYPE_INT),
            ("event_hash", TYPE_INT),
            ("space_id", TYPE_INT),
            ("event_ordinal", TYPE_INT),
        ],
        disc.DISC_APPEND_ONLY,
        [
            ("batch_hash",),
            ("batch_id", "batch_hash"),
            ("event_hash",),
            ("space_id", "batch_hash"),
            ("space_id", "batch_id"),
        ],
        core=True,
        recovery_key=("batch_hash", "event_hash"),
    )


def memory_batch_hash(
        space_key: tuple[int, ...],
        source_key: tuple[int, ...],
        batch_id: int,
        ) -> int:
    """从稳定 Memory 空间、完整来源和来源批次生成可回验非零索引。"""
    space = strict_integer_tuple(space_key, label="memory batch space_key")
    source = strict_integer_tuple(source_key, label="memory batch source_key")
    assert_int(batch_id, _where="memory batch batch_id")
    if type(batch_id) is not int or batch_id <= 0:
        raise ValueError("memory batch batch_id 必须是正严格整数")
    value = _BATCH_HASHER.h63((space, source, batch_id))
    return value if value > 0 else 1


def source_record_dependency(
        record: SourceRecordStorage,
        ) -> SegmentDependency:
    """从完整 SourceRecord 与 Companion 绑定形成摄入单元的内容依赖。"""
    if not isinstance(record, SourceRecordStorage):
        raise TypeError("record 必须是 SourceRecordStorage")
    if not record.metadata_complete:
        raise MemoryBatchIntegrityError("source dependency 缺少完整 Companion metadata")
    digest = hashlib.sha256()
    integer_values = (
        *record.source_key,
        record.text_hash,
        record.codepoint_count,
        record.batch_id,
        record.companion_type_hash,
        record.companion_name_hash,
        record.companion_assoc_id,
    )
    for value in integer_values:
        digest.update(int(value).to_bytes(16, "big", signed=True))
    for text in (record.license_id, record.raw_text):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return SegmentDependency(
        MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
        (
            MEMORY_BATCH_FORMAT_VERSION,
            record.corpus_version,
            record.parser_version,
            record.batch_id,
        ),
        tuple(digest.digest()),
    )


@dataclass(frozen=True, order=True)
class MemoryEventBatchLink:
    """一个物理事件对摄入单元的完整引用。"""

    batch_hash: int
    batch_id: int
    event_hash: int
    space_id: int
    event_ordinal: int

    def __post_init__(self) -> None:
        """核验批次、事件、空间和批内序号。"""
        for name, value in (
                ("batch_hash", self.batch_hash),
                ("batch_id", self.batch_id),
                ("event_hash", self.event_hash),
                ("space_id", self.space_id)):
            assert_int(value, _where=f"MemoryEventBatchLink.{name}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正严格整数")
        assert_int(
            self.event_ordinal,
            _where="MemoryEventBatchLink.event_ordinal",
        )
        if type(self.event_ordinal) is not int or self.event_ordinal < 0:
            raise ValueError("event_ordinal 必须是非负严格整数")

    def to_row(self) -> dict[str, int]:
        """转为 batch link 物理行。"""
        return {
            "batch_hash": self.batch_hash,
            "batch_id": self.batch_id,
            "event_hash": self.event_hash,
            "space_id": self.space_id,
            "event_ordinal": self.event_ordinal,
        }

    @classmethod
    def from_row(cls, row: dict[str, int]) -> "MemoryEventBatchLink":
        """从物理行恢复并重新核验 batch link。"""
        try:
            return cls(
                row["batch_hash"],
                row["batch_id"],
                row["event_hash"],
                row["space_id"],
                row["event_ordinal"],
            )
        except KeyError as exc:
            raise MemoryBatchIntegrityError(
                f"batch link 缺少字段 {exc.args[0]}") from exc


class MemoryEventBatchLinkStore:
    """维护 event 与一个或多个摄入单元之间的严格 append-only 映射。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定已注册 batch link 表的后端。"""
        self.backend = backend

    def add(self, link: MemoryEventBatchLink) -> MemoryEventBatchLink:
        """幂等追加映射，同 batch/event 的空间或序号漂移直接失败。"""
        if not isinstance(link, MemoryEventBatchLink):
            raise TypeError("link 必须是 MemoryEventBatchLink")
        rows = self.backend.select(
            MEMORY_EVENT_BATCH_LINK_TABLE,
            where={
                "batch_hash": link.batch_hash,
                "event_hash": link.event_hash,
            },
        )
        if len(rows) > 1:
            raise MemoryBatchIntegrityError("同一 batch/event 存在重复 link")
        if rows:
            existing = MemoryEventBatchLink.from_row(rows[0])
            if existing != link:
                raise MemoryBatchIntegrityError("同一 batch/event link 内容漂移")
            return existing
        ordinal_rows = self.backend.select(
            MEMORY_EVENT_BATCH_LINK_TABLE,
            where={
                "batch_hash": link.batch_hash,
                "event_ordinal": link.event_ordinal,
            },
        )
        if ordinal_rows:
            raise MemoryBatchIntegrityError("同一 batch ordinal 指向多个事件")
        self.backend.insert(MEMORY_EVENT_BATCH_LINK_TABLE, link.to_row())
        return link

    def for_event(self, event_hash: int) -> tuple[MemoryEventBatchLink, ...]:
        """返回引用一个物理事件的全部摄入单元。"""
        assert_int(event_hash, _where="batch links event_hash")
        if type(event_hash) is not int or event_hash <= 0:
            raise ValueError("event_hash 必须是正严格整数")
        records = tuple(
            MemoryEventBatchLink.from_row(row)
            for row in self.backend.select(
                MEMORY_EVENT_BATCH_LINK_TABLE,
                where={"event_hash": event_hash},
            )
        )
        batches = tuple(item.batch_hash for item in records)
        if len(set(batches)) != len(batches):
            raise MemoryBatchIntegrityError("event link 含重复 batch")
        return tuple(sorted(records))

    def for_batch(self, batch_hash: int) -> tuple[MemoryEventBatchLink, ...]:
        """按批内序返回一个摄入单元的全部事件映射。"""
        assert_int(batch_hash, _where="batch links batch_hash")
        if type(batch_hash) is not int or batch_hash <= 0:
            raise ValueError("batch_hash 必须是正严格整数")
        records = tuple(
            MemoryEventBatchLink.from_row(row)
            for row in self.backend.select(
                MEMORY_EVENT_BATCH_LINK_TABLE,
                where={"batch_hash": batch_hash},
            )
        )
        ordinals = tuple(item.event_ordinal for item in records)
        if len(set(ordinals)) != len(ordinals):
            raise MemoryBatchIntegrityError("batch link 含重复 ordinal")
        batch_ids = {item.batch_id for item in records}
        if len(batch_ids) > 1:
            raise MemoryBatchIntegrityError("同一 batch_hash 绑定多个 batch_id")
        return tuple(sorted(records, key=lambda item: item.event_ordinal))

    def hashes_for_source_batch(
            self,
            batch_id: int,
            *,
            space_id: int | None = None,
            ) -> tuple[int, ...]:
        """通过 append-only 索引返回来源批次涉及的唯一摄入单元。"""
        assert_int(batch_id, _where="batch links batch_id")
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("batch_id 必须是正严格整数")
        where = {"batch_id": batch_id}
        if space_id is not None:
            assert_int(space_id, _where="batch links space_id")
            if type(space_id) is not int or space_id <= 0:
                raise ValueError("space_id 必须是正严格整数")
            where["space_id"] = space_id
        records = tuple(
            MemoryEventBatchLink.from_row(row)
            for row in self.backend.select(
                MEMORY_EVENT_BATCH_LINK_TABLE,
                where=where,
            )
        )
        return tuple(sorted({item.batch_hash for item in records}))


@dataclass(frozen=True)
class StagedMemoryBatch:
    """一个已由 K-02 封存但尚不必可见的摄入事件单元。"""

    batch_hash: int
    space_key: tuple[int, ...]
    source_key: tuple[int, ...]
    batch_id: int
    event_keys: tuple[tuple[int, ...], ...]
    dependencies: tuple[SegmentDependency, ...]

    def __post_init__(self) -> None:
        """核验完整批次身份、事件顺序和两项外部依赖。"""
        expected = memory_batch_hash(
            self.space_key,
            self.source_key,
            self.batch_id,
        )
        if self.batch_hash != expected:
            raise MemoryBatchIntegrityError("batch_hash 与完整批次键不一致")
        if (not isinstance(self.event_keys, tuple)
                or not self.event_keys):
            raise ValueError("staged batch 必须包含事件")
        normalized = tuple(
            strict_integer_tuple(item, label="staged event_key")
            for item in self.event_keys
        )
        if len(set(normalized)) != len(normalized):
            raise MemoryBatchIntegrityError("staged batch 含重复事件")
        object.__setattr__(self, "event_keys", normalized)
        if (not isinstance(self.dependencies, tuple)
                or len(self.dependencies) != 2
                or tuple(item.descriptor_key for item in self.dependencies)
                != (
                    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
                    MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
                )):
            raise ValueError("staged batch 必须声明 Core 和 Source 两项依赖")

    @property
    def event_count(self) -> int:
        """返回批次内不可变事件数量。"""
        return len(self.event_keys)

    def to_segment(self) -> SealedSegment:
        """把批次 metadata 和有序事件键编码为一个 K-02 sealed segment。"""
        metadata: list[int] = [MEMORY_BATCH_FORMAT_VERSION]
        pack_key(metadata, self.space_key)
        pack_key(metadata, self.source_key)
        metadata.extend((self.batch_id, self.event_count))
        records = [SegmentRecord((self.batch_hash, 0), tuple(metadata))]
        records.extend(
            SegmentRecord((self.batch_hash, ordinal), event_key)
            for ordinal, event_key in enumerate(self.event_keys, start=1)
        )
        return SealedSegment(
            MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
            (_STAGE_SEGMENT_TAG, self.batch_hash),
            (MEMORY_BATCH_FORMAT_VERSION, self.batch_hash),
            self.dependencies,
            self.event_count,
            tuple(records),
        )

    @classmethod
    def from_segment(cls, segment: SealedSegment) -> "StagedMemoryBatch":
        """从已核验 K-02 segment 恢复批次 metadata 与事件稳定键。"""
        if segment.descriptor_key != MEMORY_BATCH_EVENT_DESCRIPTOR_KEY:
            raise MemoryBatchIntegrityError("staged segment descriptor 漂移")
        if (len(segment.segment_key) != 2
                or segment.segment_key[0] != _STAGE_SEGMENT_TAG):
            raise MemoryBatchIntegrityError("staged segment_key 非法")
        batch_hash = segment.segment_key[1]
        expected_keys = tuple(
            (batch_hash, ordinal)
            for ordinal in range(len(segment.records))
        )
        actual_keys = tuple(item.record_key for item in segment.records)
        if actual_keys != expected_keys:
            raise MemoryBatchIntegrityError("staged segment 批内序不连续")
        reader = IntegerStreamReader(segment.records[0].payload)
        version = reader.read_positive(label="memory batch format")
        if version != MEMORY_BATCH_FORMAT_VERSION:
            raise MemoryBatchIntegrityError("memory batch format 不兼容")
        space_key = reader.read_key(label="memory batch space_key")
        source_key = reader.read_key(label="memory batch source_key")
        batch_id = reader.read_positive(label="memory batch batch_id")
        event_count = reader.read_positive(label="memory batch event_count")
        reader.finish()
        if event_count != len(segment.records) - 1:
            raise MemoryBatchIntegrityError("staged event_count 与记录数不一致")
        return cls(
            batch_hash,
            space_key,
            source_key,
            batch_id,
            tuple(item.payload for item in segment.records[1:]),
            segment.dependencies,
        )


class MemoryBatchReceiptStore:
    """使用同一 K-02 store 发布和读取 staged、activation 与 rollback receipt。"""

    def __init__(
            self,
            store: TieredSegmentStore,
            *,
            tier_key: tuple[int, ...],
            read_budget: SegmentBudget,
            ) -> None:
        """绑定注入温层和有界读取预算，不推断介质或容量。"""
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("store 必须是 TieredSegmentStore")
        if not isinstance(read_budget, SegmentBudget):
            raise TypeError("read_budget 必须是 SegmentBudget")
        self.store = store
        self.tier_key = strict_integer_tuple(
            tier_key, label="memory batch tier_key")
        if not store.temperature_profile.has(self.tier_key):
            raise ValueError("memory batch tier 不属于 K-02 profile")
        self.read_budget = read_budget

    def stage(self, batch: StagedMemoryBatch) -> SealedSegment:
        """发布完整事件段；同批同内容幂等，内容漂移由 K-02 拒绝。"""
        if not isinstance(batch, StagedMemoryBatch):
            raise TypeError("batch 必须是 StagedMemoryBatch")
        segment = batch.to_segment()
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_STAGE_MANIFEST_TAG, batch.batch_hash),
            migration_key=(_STAGE_MIGRATION_TAG, batch.batch_hash),
        )
        return self._segment_for(
            MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def activate(self, batch: StagedMemoryBatch) -> SealedSegment:
        """在事件和派生投影完整后发布 activation，作为正式可见点。"""
        staged = self._segment_for(
            MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
            (_STAGE_SEGMENT_TAG, batch.batch_hash),
        )
        if StagedMemoryBatch.from_segment(staged) != batch:
            raise MemoryBatchIntegrityError("activation 对应 staged batch 漂移")
        segment = SealedSegment(
            MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
            (_ACTIVATION_SEGMENT_TAG, batch.batch_hash),
            (MEMORY_BATCH_FORMAT_VERSION, batch.batch_hash),
            (SegmentDependency(
                MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
                staged.version_key,
                staged.checksum_key,
            ),),
            batch.event_count,
            (SegmentRecord(
                (batch.batch_hash,),
                (MEMORY_BATCH_FORMAT_VERSION, batch.event_count),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_ACTIVATION_MANIFEST_TAG, batch.batch_hash),
            migration_key=(_ACTIVATION_MIGRATION_TAG, batch.batch_hash),
        )
        return self._segment_for(
            MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def rollback(self, batch_hash: int) -> SealedSegment:
        """为已 activation 单元追加 rollback receipt，不删除 staged event。"""
        activation = self._segment_for(
            MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
            (_ACTIVATION_SEGMENT_TAG, batch_hash),
        )
        record = activation.records[0]
        segment = SealedSegment(
            MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
            (_ROLLBACK_SEGMENT_TAG, batch_hash),
            (MEMORY_BATCH_FORMAT_VERSION, batch_hash),
            (SegmentDependency(
                MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
                activation.version_key,
                activation.checksum_key,
            ),),
            activation.read_fence,
            (SegmentRecord(
                (batch_hash,),
                (MEMORY_BATCH_FORMAT_VERSION, record.payload[1]),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_ROLLBACK_MANIFEST_TAG, batch_hash),
            migration_key=(_ROLLBACK_MIGRATION_TAG, batch_hash),
        )
        return self._segment_for(
            MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def begin_group(self, batch_id: int) -> SealedSegment:
        """发布多单元来源批次 intent，使随后单元在组提交前保持隐藏。"""
        self._validate_batch_id(batch_id)
        segment = SealedSegment(
            MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
            (_GROUP_INTENT_SEGMENT_TAG, batch_id),
            (MEMORY_BATCH_FORMAT_VERSION, batch_id),
            (),
            0,
            (SegmentRecord(
                (batch_id,),
                (MEMORY_BATCH_FORMAT_VERSION, batch_id),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_GROUP_INTENT_MANIFEST_TAG, batch_id),
            migration_key=(_GROUP_INTENT_MIGRATION_TAG, batch_id),
        )
        return self._segment_for(
            MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def finalize_group(
            self,
            batch_id: int,
            unit_hashes: tuple[int, ...],
            ) -> SealedSegment:
        """封存组内完整单元清单，并核验每个单元已独立 activation。"""
        self._validate_batch_id(batch_id)
        members = self._validate_group_members(unit_hashes)
        if self.is_group_rolled_back(batch_id):
            raise MemoryBatchIntegrityError("已 rollback 来源批次不得封存成员")
        intent = self._require_group_intent(batch_id)
        for batch_hash in members:
            staged = self.staged(batch_hash)
            if staged is None or staged.batch_id != batch_id:
                raise MemoryBatchIntegrityError("组成员未绑定当前来源批次")
            if (not self.is_active(batch_hash)
                    or self.is_rolled_back(batch_hash)):
                raise MemoryBatchIntegrityError("组成员尚未形成活动单元")
        segment = SealedSegment(
            MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
            (_GROUP_MEMBERSHIP_SEGMENT_TAG, batch_id),
            (MEMORY_BATCH_FORMAT_VERSION, batch_id, len(members)),
            (SegmentDependency(
                MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
                intent.version_key,
                intent.checksum_key,
            ),),
            len(members),
            tuple(
                SegmentRecord((batch_id, ordinal), (batch_hash,))
                for ordinal, batch_hash in enumerate(members)
            ),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_GROUP_MEMBERSHIP_MANIFEST_TAG, batch_id),
            migration_key=(_GROUP_MEMBERSHIP_MIGRATION_TAG, batch_id),
        )
        restored = self._segment_for(
            MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
            segment.segment_key,
        )
        if self._members_from_segment(restored, batch_id) != members:
            raise MemoryBatchIntegrityError("来源批次成员段恢复后漂移")
        return restored

    def commit_group(self, batch_id: int) -> SealedSegment:
        """在组投影完整后发布唯一组提交点。"""
        self._validate_batch_id(batch_id)
        if self.is_group_rolled_back(batch_id):
            raise MemoryBatchIntegrityError("已 rollback 来源批次不得提交")
        membership = self._require_group_membership(batch_id)
        members = self._members_from_segment(membership, batch_id)
        for batch_hash in members:
            if (not self.is_active(batch_hash)
                    or self.is_rolled_back(batch_hash)):
                raise MemoryBatchIntegrityError("提交时组成员不再活动")
        segment = SealedSegment(
            MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
            (_GROUP_COMMIT_SEGMENT_TAG, batch_id),
            (MEMORY_BATCH_FORMAT_VERSION, batch_id),
            (SegmentDependency(
                MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
                membership.version_key,
                membership.checksum_key,
            ),),
            len(members),
            (SegmentRecord(
                (batch_id,),
                (MEMORY_BATCH_FORMAT_VERSION, len(members)),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_GROUP_COMMIT_MANIFEST_TAG, batch_id),
            migration_key=(_GROUP_COMMIT_MIGRATION_TAG, batch_id),
        )
        return self._segment_for(
            MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def rollback_group(self, batch_id: int) -> SealedSegment:
        """发布单一组回滚可见点，原子隐藏全部已 activation 单元。"""
        self._validate_batch_id(batch_id)
        intent = self._require_group_intent(batch_id)
        segment = SealedSegment(
            MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
            (_GROUP_ROLLBACK_SEGMENT_TAG, batch_id),
            (MEMORY_BATCH_FORMAT_VERSION, batch_id),
            (SegmentDependency(
                MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
                intent.version_key,
                intent.checksum_key,
            ),),
            0,
            (SegmentRecord(
                (batch_id,),
                (MEMORY_BATCH_FORMAT_VERSION, batch_id),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_GROUP_ROLLBACK_MANIFEST_TAG, batch_id),
            migration_key=(_GROUP_ROLLBACK_MIGRATION_TAG, batch_id),
        )
        return self._segment_for(
            MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def staged(self, batch_hash: int) -> StagedMemoryBatch | None:
        """读取指定批次 staged segment；不存在时返回空。"""
        segment = self._optional_segment(
            MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
            (_STAGE_SEGMENT_TAG, batch_hash),
        )
        return None if segment is None else StagedMemoryBatch.from_segment(
            segment)

    def staged_batches(self) -> tuple[StagedMemoryBatch, ...]:
        """按 batch hash 读取全部 staged 单元，供启动恢复逐个 roll-forward。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            return ()
        result = []
        for entry in manifest.entries:
            if entry.descriptor_key != MEMORY_BATCH_EVENT_DESCRIPTOR_KEY:
                continue
            if (len(entry.segment_key) != 2
                    or entry.segment_key[0] != _STAGE_SEGMENT_TAG):
                raise MemoryBatchIntegrityError("staged manifest segment_key 非法")
            result.append(StagedMemoryBatch.from_segment(
                self._segment_for(entry.descriptor_key, entry.segment_key)))
        return tuple(sorted(result, key=lambda item: item.batch_hash))

    def group_batch_ids(self) -> tuple[int, ...]:
        """返回全部已发布 intent 的来源批次编号。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            return ()
        result = []
        for entry in manifest.entries:
            if entry.descriptor_key != MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY:
                continue
            if (len(entry.segment_key) != 2
                    or entry.segment_key[0] != _GROUP_INTENT_SEGMENT_TAG):
                raise MemoryBatchIntegrityError("group intent segment_key 非法")
            result.append(entry.segment_key[1])
        if len(set(result)) != len(result):
            raise MemoryBatchIntegrityError("来源批次 intent 重复")
        return tuple(sorted(result))

    def has_group_intent(self, batch_id: int) -> bool:
        """判断来源批次是否已进入组级提交流程。"""
        self._validate_batch_id(batch_id)
        return self._optional_segment(
            MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
            (_GROUP_INTENT_SEGMENT_TAG, batch_id),
        ) is not None

    def group_members(self, batch_id: int) -> tuple[int, ...] | None:
        """返回已封存组成员；组尚未封存时返回空。"""
        self._validate_batch_id(batch_id)
        segment = self._optional_segment(
            MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
            (_GROUP_MEMBERSHIP_SEGMENT_TAG, batch_id),
        )
        if segment is None:
            return None
        return self._members_from_segment(segment, batch_id)

    def is_group_committed(self, batch_id: int) -> bool:
        """核验指定来源批次是否已有完整组提交 receipt。"""
        self._validate_batch_id(batch_id)
        committed = self._has_receipt(
            MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
            (_GROUP_COMMIT_SEGMENT_TAG, batch_id),
            batch_id,
        )
        if committed and self.group_members(batch_id) is None:
            raise MemoryBatchIntegrityError("组提交 receipt 缺少成员段")
        return committed

    def is_group_rolled_back(self, batch_id: int) -> bool:
        """核验指定来源批次是否已有组回滚 receipt。"""
        self._validate_batch_id(batch_id)
        rolled_back = self._has_receipt(
            MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
            (_GROUP_ROLLBACK_SEGMENT_TAG, batch_id),
            batch_id,
        )
        if rolled_back and not self.has_group_intent(batch_id):
            raise MemoryBatchIntegrityError("组回滚 receipt 缺少 intent")
        return rolled_back

    def group_allows(self, batch_id: int, batch_hash: int) -> bool:
        """判断组级状态是否允许一个活动单元进入正式视图。"""
        self._validate_batch_id(batch_id)
        if not self.has_group_intent(batch_id):
            return True
        if self.is_group_rolled_back(batch_id):
            return False
        if not self.is_group_committed(batch_id):
            return False
        members = self.group_members(batch_id)
        if members is None:
            raise MemoryBatchIntegrityError("已提交来源批次缺少成员")
        return batch_hash in members

    def group_is_pending(self, batch_id: int) -> bool:
        """判断来源批次已开始但尚未 commit 或 rollback。"""
        self._validate_batch_id(batch_id)
        if not self.has_group_intent(batch_id):
            return False
        committed = self.is_group_committed(batch_id)
        rolled_back = self.is_group_rolled_back(batch_id)
        return not committed and not rolled_back

    def is_active(self, batch_hash: int) -> bool:
        """核验指定 batch 是否存在完整 activation receipt。"""
        return self._has_receipt(
            MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
            (_ACTIVATION_SEGMENT_TAG, batch_hash),
            batch_hash,
        )

    def is_rolled_back(self, batch_hash: int) -> bool:
        """核验指定 batch 是否存在完整 rollback receipt。"""
        return self._has_receipt(
            MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
            (_ROLLBACK_SEGMENT_TAG, batch_hash),
            batch_hash,
        )

    def _has_receipt(
            self,
            descriptor_key: tuple[int, ...],
            segment_key: tuple[int, ...],
            batch_hash: int,
            ) -> bool:
        """读取并核验单记录 receipt，避免只看 manifest 范围猜状态。"""
        segment = self._optional_segment(descriptor_key, segment_key)
        if segment is None:
            return False
        if (len(segment.records) != 1
                or segment.records[0].record_key != (batch_hash,)
                or segment.records[0].payload[0]
                != MEMORY_BATCH_FORMAT_VERSION):
            raise MemoryBatchIntegrityError("memory batch receipt 内容非法")
        return True

    def _require_group_intent(self, batch_id: int) -> SealedSegment:
        """读取唯一组 intent，不存在时拒绝继续。"""
        segment = self._optional_segment(
            MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
            (_GROUP_INTENT_SEGMENT_TAG, batch_id),
        )
        if segment is None:
            raise MemoryBatchIntegrityError("来源批次缺少 group intent")
        return segment

    def _require_group_membership(self, batch_id: int) -> SealedSegment:
        """读取唯一组成员段，不存在时拒绝提交。"""
        segment = self._optional_segment(
            MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
            (_GROUP_MEMBERSHIP_SEGMENT_TAG, batch_id),
        )
        if segment is None:
            raise MemoryBatchIntegrityError("来源批次尚未封存成员")
        return segment

    @staticmethod
    def _validate_batch_id(batch_id: int) -> None:
        """核验来源批次编号是正严格整数。"""
        assert_int(batch_id, _where="memory group batch_id")
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("group batch_id 必须是正严格整数")

    @staticmethod
    def _validate_group_members(
            unit_hashes: tuple[int, ...],
            ) -> tuple[int, ...]:
        """核验并规范化非空、唯一、稳定排序的组成员。"""
        if not isinstance(unit_hashes, tuple) or not unit_hashes:
            raise TypeError("unit_hashes 必须是非空 tuple")
        assert_int(*unit_hashes, _where="memory group unit_hashes")
        if any(type(item) is not int or item <= 0 for item in unit_hashes):
            raise ValueError("unit_hashes 必须是正严格整数")
        members = tuple(sorted(unit_hashes))
        if len(set(members)) != len(members):
            raise MemoryBatchIntegrityError("来源批次含重复摄入单元")
        return members

    def _members_from_segment(
            self,
            segment: SealedSegment,
            batch_id: int,
            ) -> tuple[int, ...]:
        """从已核验成员段恢复稳定单元清单。"""
        if (segment.descriptor_key
                != MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY
                or segment.segment_key
                != (_GROUP_MEMBERSHIP_SEGMENT_TAG, batch_id)):
            raise MemoryBatchIntegrityError("来源批次成员段身份漂移")
        expected_keys = tuple(
            (batch_id, ordinal) for ordinal in range(len(segment.records)))
        if tuple(item.record_key for item in segment.records) != expected_keys:
            raise MemoryBatchIntegrityError("来源批次成员序号不连续")
        if any(len(item.payload) != 1 for item in segment.records):
            raise MemoryBatchIntegrityError("来源批次成员记录格式非法")
        members = self._validate_group_members(tuple(
            item.payload[0] for item in segment.records))
        if segment.read_fence != len(members):
            raise MemoryBatchIntegrityError("来源批次成员 read fence 漂移")
        return members

    def _optional_segment(
            self,
            descriptor_key: tuple[int, ...],
            segment_key: tuple[int, ...],
            ) -> SealedSegment | None:
        """从当前 manifest 查找唯一 segment，不存在时返回空。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            return None
        matches = tuple(
            entry for entry in manifest.entries
            if (entry.descriptor_key == descriptor_key
                and entry.segment_key == segment_key)
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise MemoryBatchIntegrityError("manifest 含重复 batch segment")
        return self._segment_for(descriptor_key, segment_key)

    def _segment_for(
            self,
            descriptor_key: tuple[int, ...],
            segment_key: tuple[int, ...],
            ) -> SealedSegment:
        """按 manifest entry 的完整版本、依赖和校验恢复一个 segment。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            raise MemoryBatchIntegrityError("没有可读取的 K-02 manifest")
        matches = tuple(
            entry for entry in manifest.entries
            if (entry.descriptor_key == descriptor_key
                and entry.segment_key == segment_key)
        )
        if len(matches) != 1:
            raise MemoryBatchIntegrityError("batch segment 没有唯一 manifest entry")
        entry = matches[0]
        reader = self.store.open_reader(
            (_READER_TAG, descriptor_key[-1], segment_key[-1]),
            descriptor_key,
        )
        records: list[SegmentRecord] = []
        continuation = None
        try:
            while True:
                page = reader.page(
                    budget=self.read_budget,
                    lower_key=entry.key_range.lower_key,
                    upper_key=entry.key_range.upper_key,
                    continuation=continuation,
                )
                records.extend(page.records)
                if not page.has_more:
                    break
                continuation = page.continuation
        finally:
            reader.close()
        segment = SealedSegment(
            entry.descriptor_key,
            entry.segment_key,
            entry.version_key,
            entry.dependencies,
            entry.read_fence,
            tuple(records),
        )
        if segment.checksum_key != entry.checksum_key:
            raise MemoryBatchIntegrityError("batch segment checksum 漂移")
        return segment


class MemoryBatchVisibility:
    """把 event link 与 K-02 receipt 组合为正式事件可见性判定。"""

    def __init__(
            self,
            links: MemoryEventBatchLinkStore,
            receipts: MemoryBatchReceiptStore,
            ) -> None:
        """绑定同一 backend 的 link 表和 receipt store。"""
        if not isinstance(links, MemoryEventBatchLinkStore):
            raise TypeError("links 类型错误")
        repository_backend = getattr(receipts.store.repository, "backend", None)
        if (repository_backend is not None
                and links.backend is not repository_backend):
            raise ValueError("batch link 与 K-02 receipt backend 不一致")
        self.links = links
        self.receipts = receipts
        self._status_epoch = 0
        self._status_cache: dict[int, tuple[bool, bool, int]] = {}
        self._group_cache: dict[
            int,
            tuple[bool, bool, bool, tuple[int, ...] | None],
        ] = {}
        self._preview_batches: ContextVar[tuple[int, ...]] = ContextVar(
            "memory_batch_preview", default=())
        self._suppressed_batches: ContextVar[tuple[int, ...]] = ContextVar(
            "memory_batch_suppressed", default=())

    @contextmanager
    def preview(self, batch_hash: int) -> Iterator[None]:
        """仅在当前调用链把 staged batch 视为可见，供 activation 前重建投影。"""
        current = self._preview_batches.get()
        token = self._preview_batches.set((*current, batch_hash))
        try:
            yield
        finally:
            self._preview_batches.reset(token)

    @contextmanager
    def preview_many(self, batch_hashes: tuple[int, ...]) -> Iterator[None]:
        """仅在当前调用链预览一组已 activation 单元。"""
        if not isinstance(batch_hashes, tuple) or not batch_hashes:
            raise TypeError("batch_hashes 必须是非空 tuple")
        current = self._preview_batches.get()
        token = self._preview_batches.set((*current, *batch_hashes))
        try:
            yield
        finally:
            self._preview_batches.reset(token)

    @contextmanager
    def suppress(self, batch_hash: int) -> Iterator[None]:
        """仅在当前调用链隐藏 active batch，供 rollback 前构造剩余投影。"""
        current = self._suppressed_batches.get()
        token = self._suppressed_batches.set((*current, batch_hash))
        try:
            yield
        finally:
            self._suppressed_batches.reset(token)

    def event_is_visible(self, event_hash: int, *, space_id: int) -> bool:
        """无 link 的 legacy event 直接可见；有 link 时要求至少一个活动单元。"""
        links = self.links.for_event(event_hash)
        if not links:
            return True
        if any(item.space_id != space_id for item in links):
            raise MemoryBatchIntegrityError("event link 跨 Memory 空间漂移")
        preview = set(self._preview_batches.get())
        suppressed = set(self._suppressed_batches.get())
        for link in links:
            if link.batch_hash in suppressed:
                continue
            if link.batch_hash in preview:
                return True
            active, rolled_back, batch_id = self._status(link.batch_hash)
            if (active and not rolled_back
                    and self._group_allows(batch_id, link.batch_hash)):
                return True
        return False

    def event_is_available_to_batch(
            self,
            event_hash: int,
            *,
            space_id: int,
            batch_hash: int | None,
            ) -> bool:
        """写入校验可读取正式事件和当前批次已映射事件，拒绝其他 pending。"""
        if self.event_is_visible(event_hash, space_id=space_id):
            return True
        if batch_hash is None:
            return False
        return any(
            item.batch_hash == batch_hash and item.space_id == space_id
            for item in self.links.for_event(event_hash)
        )

    def batch_is_active(self, batch_hash: int) -> bool:
        """返回 batch activation 已发布且尚未 rollback 的正式状态。"""
        active, rolled_back, batch_id = self._status(batch_hash)
        return (active and not rolled_back
                and self._group_allows(batch_id, batch_hash))

    def state_epoch(self) -> int:
        """返回缓存失效用全局位置 epoch，不得用作投影语义状态。"""
        manifest = self.receipts.store.current_manifest()
        return 0 if manifest is None else manifest.publish_epoch

    def state_key(self) -> tuple[int, ...]:
        """返回只由 batch/group 可见性描述内容决定的完整逻辑状态。"""
        return self.receipts.store.descriptor_state_key((
            MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
            MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
            MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
            MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
            MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
            MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
            MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
        ))

    def group_is_pending(self, batch_id: int) -> bool:
        """返回来源批次是否正处于组提交前的隐藏状态。"""
        has_intent, committed, rolled_back, _ = self._group_status(batch_id)
        return has_intent and not committed and not rolled_back

    def _status(self, batch_hash: int) -> tuple[bool, bool, int]:
        """按当前 K-02 epoch 缓存 activation 与 rollback receipt 状态。"""
        manifest = self.receipts.store.current_manifest()
        epoch = 0 if manifest is None else manifest.publish_epoch
        if epoch != self._status_epoch:
            self._status_epoch = epoch
            self._status_cache.clear()
            self._group_cache.clear()
        cached = self._status_cache.get(batch_hash)
        if cached is not None:
            return cached
        staged = self.receipts.staged(batch_hash)
        if staged is None:
            raise MemoryBatchIntegrityError("event link 缺少 staged batch")
        status = (
            self.receipts.is_active(batch_hash),
            self.receipts.is_rolled_back(batch_hash),
            staged.batch_id,
        )
        if status[1] and not status[0]:
            raise MemoryBatchIntegrityError("rollback receipt 缺少 activation")
        self._status_cache[batch_hash] = status
        return status

    def _group_allows(self, batch_id: int, batch_hash: int) -> bool:
        """按缓存的组状态核验单元是否属于正式提交成员。"""
        has_intent, committed, rolled_back, members = self._group_status(
            batch_id)
        if not has_intent:
            return True
        if rolled_back or not committed:
            return False
        if members is None:
            raise MemoryBatchIntegrityError("已提交来源批次缺少成员")
        return batch_hash in members

    def _group_status(
            self,
            batch_id: int,
            ) -> tuple[bool, bool, bool, tuple[int, ...] | None]:
        """按当前 manifest epoch 缓存组提交、回滚和成员状态。"""
        manifest = self.receipts.store.current_manifest()
        epoch = 0 if manifest is None else manifest.publish_epoch
        if epoch != self._status_epoch:
            self._status_epoch = epoch
            self._status_cache.clear()
            self._group_cache.clear()
        cached = self._group_cache.get(batch_id)
        if cached is not None:
            return cached
        has_intent = self.receipts.has_group_intent(batch_id)
        status = (
            has_intent,
            self.receipts.is_group_committed(batch_id) if has_intent else False,
            self.receipts.is_group_rolled_back(batch_id) if has_intent else False,
            self.receipts.group_members(batch_id) if has_intent else None,
        )
        self._group_cache[batch_id] = status
        return status


__all__ = [
    "MEMORY_BATCH_ACTIVATION_DESCRIPTOR",
    "MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY",
    "MEMORY_BATCH_CORE_DEPENDENCY",
    "MEMORY_BATCH_CORE_DEPENDENCY_KEY",
    "MEMORY_BATCH_EVENT_DESCRIPTOR",
    "MEMORY_BATCH_EVENT_DESCRIPTOR_KEY",
    "MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR",
    "MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY",
    "MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR",
    "MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY",
    "MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR",
    "MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY",
    "MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR",
    "MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY",
    "MEMORY_BATCH_ROLLBACK_DESCRIPTOR",
    "MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY",
    "MEMORY_BATCH_SOURCE_DEPENDENCY",
    "MEMORY_BATCH_SOURCE_DEPENDENCY_KEY",
    "MEMORY_EVENT_BATCH_LINK_TABLE",
    "MemoryBatchIntegrityError",
    "MemoryBatchReceiptStore",
    "MemoryBatchVisibility",
    "MemoryEventBatchLink",
    "MemoryEventBatchLinkStore",
    "StagedMemoryBatch",
    "memory_batch_hash",
    "register_memory_batch_table",
    "source_record_dependency",
]
