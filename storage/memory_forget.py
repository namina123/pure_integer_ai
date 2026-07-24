"""Memory 全对象遗忘目标集、提交 receipt 和统一可见性。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
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


MEMORY_FORGET_FORMAT_VERSION = 1

MEMORY_FORGET_SET_DESCRIPTOR_KEY = (1, 1, 13)
MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY = (1, 1, 14)

MEMORY_FORGET_SET_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_FORGET_SET_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
)
MEMORY_FORGET_COMMIT_DESCRIPTOR = StorageRoleDescriptor(
    MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
    dependency_keys=(MEMORY_FORGET_SET_DESCRIPTOR_KEY,),
)

FORGET_TARGET_EVENT = 1
FORGET_TARGET_OVERLAY = 2
FORGET_TARGET_SOURCE = 3
FORGET_TARGET_COMPANION = 4
_TARGET_KINDS = frozenset({
    FORGET_TARGET_EVENT,
    FORGET_TARGET_OVERLAY,
    FORGET_TARGET_SOURCE,
    FORGET_TARGET_COMPANION,
})
_OWNER_KEY_SIZE = 4
_OWNED_OBJECT_PREFIX_SIZE = 2
_SOURCE_OWNER_START = 3

_SET_SEGMENT_TAG = 2026072341
_COMMIT_SEGMENT_TAG = 2026072342
_SET_MANIFEST_TAG = 2026072351
_COMMIT_MANIFEST_TAG = 2026072352
_SET_MIGRATION_TAG = 2026072361
_COMMIT_MIGRATION_TAG = 2026072362
_READER_TAG = 2026072371

_FORGET_HASHER = Hasher("pure_integer_ai.memory_forget.v1")


class MemoryForgetIntegrityError(RuntimeError):
    """遗忘目标集、提交 receipt 或完整键出现不一致。"""


@dataclass(frozen=True, order=True)
class MemoryForgetTarget:
    """一个需要从正式 Memory 视图隐藏的完整对象目标。"""

    target_kind: int
    target_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验目标种类和非空严格整数完整键。"""
        assert_int(self.target_kind, _where="MemoryForgetTarget.kind")
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError("forget target_kind 未注册")
        object.__setattr__(
            self,
            "target_key",
            strict_integer_tuple(
                self.target_key, label="memory forget target_key"),
        )
        expected_sizes = {
            FORGET_TARGET_EVENT: _OWNED_OBJECT_PREFIX_SIZE + _OWNER_KEY_SIZE,
            FORGET_TARGET_OVERLAY: _OWNED_OBJECT_PREFIX_SIZE + _OWNER_KEY_SIZE,
            FORGET_TARGET_SOURCE: 11,
            FORGET_TARGET_COMPANION: 3,
        }
        if len(self.target_key) != expected_sizes[self.target_kind]:
            raise ValueError("forget target_key 长度与目标种类不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回带种类和长度分帧的稳定目标键。"""
        return self.target_kind, len(self.target_key), *self.target_key

    def owner_key(self) -> tuple[int, ...] | None:
        """返回目标携带的完整 owner 键；Companion 物理项本身没有 owner。"""
        if self.target_kind in {FORGET_TARGET_EVENT, FORGET_TARGET_OVERLAY}:
            return self.target_key[_OWNED_OBJECT_PREFIX_SIZE:]
        if self.target_kind == FORGET_TARGET_SOURCE:
            end = _SOURCE_OWNER_START + _OWNER_KEY_SIZE
            return self.target_key[_SOURCE_OWNER_START:end]
        return None


@dataclass(frozen=True)
class StagedMemoryForget:
    """一个已封存但尚不必对正式 reader 生效的遗忘操作。"""

    operation_hash: int
    owner_key: tuple[int, ...]
    selection_kind: int
    reason_key: tuple[int, ...]
    targets: tuple[MemoryForgetTarget, ...]

    def __post_init__(self) -> None:
        """核验操作身份、owner、选择方式、理由和非空目标闭包。"""
        owner_key = strict_integer_tuple(
            self.owner_key, label="memory forget owner_key")
        if len(owner_key) != 4:
            raise ValueError("memory forget owner_key 长度非法")
        object.__setattr__(self, "owner_key", owner_key)
        assert_int(self.selection_kind, _where="memory forget selection_kind")
        if type(self.selection_kind) is not int or self.selection_kind <= 0:
            raise ValueError("selection_kind 必须是正严格整数")
        object.__setattr__(
            self,
            "reason_key",
            strict_integer_tuple(
                self.reason_key, label="memory forget reason_key"),
        )
        if (not isinstance(self.targets, tuple)
                or not self.targets
                or any(not isinstance(item, MemoryForgetTarget)
                       for item in self.targets)):
            raise TypeError("forget targets 必须是非空目标 tuple")
        normalized = tuple(sorted(self.targets))
        if len(set(normalized)) != len(normalized):
            raise MemoryForgetIntegrityError("forget targets 含重复完整键")
        object.__setattr__(self, "targets", normalized)
        expected = memory_forget_hash(
            owner_key,
            self.selection_kind,
            self.reason_key,
            normalized,
        )
        if self.operation_hash != expected:
            raise MemoryForgetIntegrityError("forget operation_hash 漂移")

    def to_segment(self) -> SealedSegment:
        """把遗忘 metadata 和完整目标集编码为 K-02 sealed segment。"""
        metadata: list[int] = [MEMORY_FORGET_FORMAT_VERSION]
        pack_key(metadata, self.owner_key)
        metadata.append(self.selection_kind)
        pack_key(metadata, self.reason_key)
        metadata.append(len(self.targets))
        records = [SegmentRecord(
            (self.operation_hash, 0), tuple(metadata))]
        for ordinal, target in enumerate(self.targets, start=1):
            payload = [target.target_kind]
            pack_key(payload, target.target_key)
            records.append(SegmentRecord(
                (self.operation_hash, ordinal), tuple(payload)))
        return SealedSegment(
            MEMORY_FORGET_SET_DESCRIPTOR_KEY,
            (_SET_SEGMENT_TAG, self.operation_hash),
            (MEMORY_FORGET_FORMAT_VERSION, self.operation_hash),
            (),
            len(self.targets),
            tuple(records),
        )

    @classmethod
    def from_segment(cls, segment: SealedSegment) -> "StagedMemoryForget":
        """从已核验 K-02 segment 恢复遗忘操作。"""
        if segment.descriptor_key != MEMORY_FORGET_SET_DESCRIPTOR_KEY:
            raise MemoryForgetIntegrityError("forget set descriptor 漂移")
        if (len(segment.segment_key) != 2
                or segment.segment_key[0] != _SET_SEGMENT_TAG):
            raise MemoryForgetIntegrityError("forget set segment_key 非法")
        operation_hash = segment.segment_key[1]
        expected_keys = tuple(
            (operation_hash, ordinal)
            for ordinal in range(len(segment.records))
        )
        if tuple(item.record_key for item in segment.records) != expected_keys:
            raise MemoryForgetIntegrityError("forget set 记录序号不连续")
        reader = IntegerStreamReader(segment.records[0].payload)
        version = reader.read_positive(label="memory forget format")
        if version != MEMORY_FORGET_FORMAT_VERSION:
            raise MemoryForgetIntegrityError("memory forget format 不兼容")
        owner_key = reader.read_key(label="memory forget owner")
        selection_kind = reader.read_positive(
            label="memory forget selection")
        reason_key = reader.read_key(label="memory forget reason")
        target_count = reader.read_positive(label="memory forget target count")
        reader.finish()
        targets = []
        for record in segment.records[1:]:
            target_reader = IntegerStreamReader(record.payload)
            target_kind = target_reader.read_positive(
                label="memory forget target kind")
            target_key = target_reader.read_key(
                label="memory forget target key")
            target_reader.finish()
            targets.append(MemoryForgetTarget(target_kind, target_key))
        if target_count != len(targets) or segment.read_fence != target_count:
            raise MemoryForgetIntegrityError("forget target count 漂移")
        return cls(
            operation_hash,
            owner_key,
            selection_kind,
            reason_key,
            tuple(targets),
        )


def memory_forget_hash(
        owner_key: tuple[int, ...],
        selection_kind: int,
        reason_key: tuple[int, ...],
        targets: tuple[MemoryForgetTarget, ...],
        ) -> int:
    """从目标 owner、选择、理由和完整闭包生成稳定非零操作索引。"""
    owner = strict_integer_tuple(owner_key, label="forget hash owner")
    reason = strict_integer_tuple(reason_key, label="forget hash reason")
    assert_int(selection_kind, _where="forget hash selection")
    if type(selection_kind) is not int or selection_kind <= 0:
        raise ValueError("forget hash selection 必须是正严格整数")
    if (not isinstance(targets, tuple)
            or not targets
            or any(not isinstance(item, MemoryForgetTarget)
                   for item in targets)):
        raise TypeError("forget hash targets 必须是非空 tuple")
    value = _FORGET_HASHER.h63((
        owner,
        selection_kind,
        reason,
        tuple(item.stable_key() for item in sorted(targets)),
    ))
    return value if value > 0 else 1


class MemoryForgetStore:
    """使用同一 K-02 store 发布和读取 forget set 与 commit receipt。"""

    def __init__(
            self,
            store: TieredSegmentStore,
            *,
            tier_key: tuple[int, ...],
            read_budget: SegmentBudget,
            write_budget: SegmentBudget,
            ) -> None:
        """绑定注入式温层和读写预算。"""
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("store 必须是 TieredSegmentStore")
        if (not isinstance(read_budget, SegmentBudget)
                or not isinstance(write_budget, SegmentBudget)):
            raise TypeError("forget 读写预算必须是 SegmentBudget")
        self.store = store
        self.tier_key = strict_integer_tuple(
            tier_key, label="memory forget tier_key")
        if not store.temperature_profile.has(self.tier_key):
            raise ValueError("memory forget tier 不属于 K-02 profile")
        self.read_budget = read_budget
        self.write_budget = write_budget

    def stage(self, operation: StagedMemoryForget) -> SealedSegment:
        """发布完整遗忘目标集；精确重放幂等，内容漂移拒绝。"""
        if not isinstance(operation, StagedMemoryForget):
            raise TypeError("operation 必须是 StagedMemoryForget")
        segment = operation.to_segment()
        if len(segment.records) > self.write_budget.object_limit:
            raise MemoryForgetIntegrityError("forget set 超过对象数预算")
        if segment.size_bytes > self.write_budget.byte_limit:
            raise MemoryForgetIntegrityError("forget set 超过字节预算")
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_SET_MANIFEST_TAG, operation.operation_hash),
            migration_key=(_SET_MIGRATION_TAG, operation.operation_hash),
        )
        restored = self._segment_for(
            MEMORY_FORGET_SET_DESCRIPTOR_KEY,
            segment.segment_key,
        )
        if StagedMemoryForget.from_segment(restored) != operation:
            raise MemoryForgetIntegrityError("forget set 恢复后漂移")
        return restored

    def commit(self, operation: StagedMemoryForget) -> SealedSegment:
        """在派生投影完成后发布 forget commit 唯一可见点。"""
        staged = self._segment_for(
            MEMORY_FORGET_SET_DESCRIPTOR_KEY,
            (_SET_SEGMENT_TAG, operation.operation_hash),
        )
        if StagedMemoryForget.from_segment(staged) != operation:
            raise MemoryForgetIntegrityError("forget commit 对应 set 漂移")
        segment = SealedSegment(
            MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY,
            (_COMMIT_SEGMENT_TAG, operation.operation_hash),
            (MEMORY_FORGET_FORMAT_VERSION, operation.operation_hash),
            (SegmentDependency(
                MEMORY_FORGET_SET_DESCRIPTOR_KEY,
                staged.version_key,
                staged.checksum_key,
            ),),
            len(operation.targets),
            (SegmentRecord(
                (operation.operation_hash,),
                (MEMORY_FORGET_FORMAT_VERSION, len(operation.targets)),
            ),),
        )
        self.store.publish_segment(
            segment,
            tier_key=self.tier_key,
            manifest_key=(_COMMIT_MANIFEST_TAG, operation.operation_hash),
            migration_key=(_COMMIT_MIGRATION_TAG, operation.operation_hash),
        )
        return self._segment_for(
            MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY,
            segment.segment_key,
        )

    def staged(self, operation_hash: int) -> StagedMemoryForget | None:
        """读取指定遗忘目标集，不存在时返回空。"""
        self._validate_operation_hash(operation_hash)
        segment = self._optional_segment(
            MEMORY_FORGET_SET_DESCRIPTOR_KEY,
            (_SET_SEGMENT_TAG, operation_hash),
        )
        return None if segment is None else StagedMemoryForget.from_segment(
            segment)

    def staged_operations(self) -> tuple[StagedMemoryForget, ...]:
        """按 operation hash 返回全部已封存遗忘操作。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            return ()
        result = []
        for entry in manifest.entries:
            if entry.descriptor_key != MEMORY_FORGET_SET_DESCRIPTOR_KEY:
                continue
            if (len(entry.segment_key) != 2
                    or entry.segment_key[0] != _SET_SEGMENT_TAG):
                raise MemoryForgetIntegrityError(
                    "forget manifest segment_key 非法")
            result.append(StagedMemoryForget.from_segment(
                self._segment_for(entry.descriptor_key, entry.segment_key)))
        return tuple(sorted(result, key=lambda item: item.operation_hash))

    def is_committed(self, operation_hash: int) -> bool:
        """核验指定遗忘操作是否已有完整 commit receipt。"""
        self._validate_operation_hash(operation_hash)
        segment = self._optional_segment(
            MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY,
            (_COMMIT_SEGMENT_TAG, operation_hash),
        )
        if segment is None:
            return False
        if (len(segment.records) != 1
                or segment.records[0].record_key != (operation_hash,)
                or segment.records[0].payload[0]
                != MEMORY_FORGET_FORMAT_VERSION):
            raise MemoryForgetIntegrityError("forget commit receipt 非法")
        return True

    @staticmethod
    def _validate_operation_hash(operation_hash: int) -> None:
        """核验 operation hash 是正严格整数。"""
        assert_int(operation_hash, _where="memory forget operation_hash")
        if type(operation_hash) is not int or operation_hash <= 0:
            raise ValueError("operation_hash 必须是正严格整数")

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
            raise MemoryForgetIntegrityError("manifest 含重复 forget segment")
        return self._segment_for(descriptor_key, segment_key)

    def _segment_for(
            self,
            descriptor_key: tuple[int, ...],
            segment_key: tuple[int, ...],
            ) -> SealedSegment:
        """按 manifest entry 的完整版本、依赖和校验恢复 segment。"""
        manifest = self.store.current_manifest()
        if manifest is None:
            raise MemoryForgetIntegrityError("没有可读取的 K-02 manifest")
        matches = tuple(
            entry for entry in manifest.entries
            if (entry.descriptor_key == descriptor_key
                and entry.segment_key == segment_key)
        )
        if len(matches) != 1:
            raise MemoryForgetIntegrityError(
                "forget segment 没有唯一 manifest entry")
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
            raise MemoryForgetIntegrityError("forget segment checksum 漂移")
        return segment


class MemoryForgetVisibility:
    """把已 commit forget set 合并为事件、关系、来源和伴随可见性。"""

    def __init__(self, store: MemoryForgetStore) -> None:
        """绑定遗忘存储并创建按 manifest epoch 失效的目标缓存。"""
        if not isinstance(store, MemoryForgetStore):
            raise TypeError("store 必须是 MemoryForgetStore")
        self.store = store
        self._status_epoch = 0
        self._committed_targets: frozenset[MemoryForgetTarget] = frozenset()
        self._staged_by_hash: dict[int, StagedMemoryForget] = {}
        self._preview_operations: ContextVar[tuple[int, ...]] = ContextVar(
            "memory_forget_preview", default=())

    @contextmanager
    def preview(self, operation_hash: int) -> Iterator[None]:
        """仅在当前调用链预览一个 staged 遗忘集，供 commit 前重建投影。"""
        current = self._preview_operations.get()
        token = self._preview_operations.set((*current, operation_hash))
        try:
            yield
        finally:
            self._preview_operations.reset(token)

    def event_is_forgotten(
            self,
            space_id: int,
            event_hash: int,
            owner_key: tuple[int, ...],
            ) -> bool:
        """判断一个 Memory event 物理身份是否已遗忘。"""
        return self.is_forgotten(MemoryForgetTarget(
            FORGET_TARGET_EVENT, (space_id, event_hash, *owner_key)))

    def overlay_is_forgotten(
            self,
            space_id: int,
            identity_hash: int,
            owner_key: tuple[int, ...],
            ) -> bool:
        """判断一个 Memory overlay 物理身份是否已遗忘。"""
        return self.is_forgotten(MemoryForgetTarget(
            FORGET_TARGET_OVERLAY,
            (space_id, identity_hash, *owner_key),
        ))

    def source_is_forgotten(self, source_key: tuple[int, ...]) -> bool:
        """判断一个完整 SourceRef 是否已遗忘。"""
        return self.is_forgotten(MemoryForgetTarget(
            FORGET_TARGET_SOURCE, source_key))

    def companion_is_forgotten(self, assoc_key: tuple[int, ...]) -> bool:
        """判断一个稳定 Companion assoc 是否已遗忘。"""
        return self.is_forgotten(MemoryForgetTarget(
            FORGET_TARGET_COMPANION, assoc_key))

    def is_forgotten(self, target: MemoryForgetTarget) -> bool:
        """核验目标是否属于已提交或当前预览的遗忘集。"""
        if not isinstance(target, MemoryForgetTarget):
            raise TypeError("target 必须是 MemoryForgetTarget")
        self._refresh()
        if target in self._committed_targets:
            return True
        for operation_hash in self._preview_operations.get():
            operation = self._staged_by_hash.get(operation_hash)
            if operation is None:
                operation = self.store.staged(operation_hash)
            if operation is None:
                raise MemoryForgetIntegrityError("预览遗忘操作缺少 staged set")
            if target in operation.targets:
                return True
        return False

    def state_epoch(self) -> int:
        """返回缓存失效用全局位置 epoch，不得用作投影语义状态。"""
        manifest = self.store.store.current_manifest()
        return 0 if manifest is None else manifest.publish_epoch

    def state_key(self) -> tuple[int, ...]:
        """返回只由遗忘集合与提交描述内容决定的完整逻辑状态。"""
        return self.store.store.descriptor_state_key((
            MEMORY_FORGET_SET_DESCRIPTOR_KEY,
            MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY,
        ))

    def _refresh(self) -> None:
        """manifest epoch 变化时重建已 commit 目标集合。"""
        manifest = self.store.store.current_manifest()
        epoch = 0 if manifest is None else manifest.publish_epoch
        if epoch == self._status_epoch:
            return
        operations = self.store.staged_operations()
        self._staged_by_hash = {
            item.operation_hash: item for item in operations}
        committed = set()
        for operation in operations:
            if self.store.is_committed(operation.operation_hash):
                committed.update(operation.targets)
        self._committed_targets = frozenset(committed)
        self._status_epoch = epoch


__all__ = [
    "FORGET_TARGET_COMPANION",
    "FORGET_TARGET_EVENT",
    "FORGET_TARGET_OVERLAY",
    "FORGET_TARGET_SOURCE",
    "MEMORY_FORGET_COMMIT_DESCRIPTOR",
    "MEMORY_FORGET_COMMIT_DESCRIPTOR_KEY",
    "MEMORY_FORGET_SET_DESCRIPTOR",
    "MEMORY_FORGET_SET_DESCRIPTOR_KEY",
    "MemoryForgetIntegrityError",
    "MemoryForgetStore",
    "MemoryForgetTarget",
    "MemoryForgetVisibility",
    "StagedMemoryForget",
    "memory_forget_hash",
]
