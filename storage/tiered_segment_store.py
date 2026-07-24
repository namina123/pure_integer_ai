"""K-02 分层 segment 发布、迁移、稳定分页和 reader epoch 回收。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.storage.integer_codec import (
    IntegerStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.location_manifest import (
    LocationManifest,
    LocationManifestEntry,
    LocationManifestLedger,
    ManifestIntegrityError,
    ManifestKeyRange,
)
from pure_integer_ai.storage.placement import TemperatureProfile
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SealedSegment,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentIntegrityError,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_cache import SegmentPageCache
from pure_integer_ai.storage.segment_commit import (
    MIGRATION_PHASE_ABORTED,
    MIGRATION_PHASE_PREPARED,
    MIGRATION_PHASE_PUBLISHED,
    MIGRATION_PHASE_RECLAIMED,
    MigrationCommitRecord,
    SegmentCommitIntegrityError,
    SegmentCopyReference,
    validate_commit_chain,
)
from pure_integer_ai.storage.segment_repository import (
    AppendOnlyObjectRepository,
    OBJECT_KIND_LOCATION_MANIFEST,
    OBJECT_KIND_MIGRATION_COMMIT,
    OBJECT_KIND_SEGMENT,
    OBJECT_KIND_SEGMENT_RELEASE,
    SegmentRepositoryError,
    SegmentRepositoryFaultInjector,
    hit_repository_fault,
)
from pure_integer_ai.storage.segment_release import (
    SegmentReleaseCommitRecord,
    SegmentReleaseIntegrityError,
    validate_release_chain,
)
from pure_integer_ai.storage.storage_role import (
    STORAGE_ROLE_REBUILDABLE,
    StorageRoleRegistry,
)


FAULT_MIGRATION_AFTER_TARGET_WRITE = 101
FAULT_MIGRATION_AFTER_TARGET_VERIFY = 102
FAULT_MIGRATION_AFTER_PREPARE = 103
FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH = 104
FAULT_MIGRATION_AFTER_READER_SWITCH = 105
FAULT_MIGRATION_BEFORE_SOURCE_RECLAIM = 106
FAULT_MIGRATION_AFTER_SOURCE_RECLAIM = 107

FAULT_RELEASE_AFTER_PREPARE = 201
FAULT_RELEASE_AFTER_MANIFEST_PUBLISH = 202
FAULT_RELEASE_BEFORE_SOURCE_RECLAIM = 203
FAULT_RELEASE_AFTER_SOURCE_RECLAIM = 204

DESCRIPTOR_STATE_KEY_VERSION = 1


class TieredSegmentStoreError(RuntimeError):
    """分层段发布、恢复、分页或 reader epoch 状态不一致。"""


def segment_copy_identity(
        tier_key: tuple[int, ...], segment_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """返回一个 segment 在指定物理温层中的完整副本身份。"""
    result: list[int] = []
    pack_key(result, strict_integer_tuple(tier_key, label="segment copy tier"))
    pack_key(
        result,
        strict_integer_tuple(segment_key, label="segment copy segment"),
    )
    return tuple(result)


def parse_segment_copy_identity(
        identity_key: tuple[int, ...],
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """从对象仓库身份恢复温层和 segment 完整键。"""
    reader = IntegerStreamReader(identity_key)
    tier_key = reader.read_key(label="segment copy tier")
    segment_key = reader.read_key(label="segment copy segment")
    reader.finish()
    return tier_key, segment_key


def manifest_object_identity(
        manifest: LocationManifest,
        ) -> tuple[int, ...]:
    """返回 location manifest 的 epoch 与完整 manifest 键身份。"""
    if not isinstance(manifest, LocationManifest):
        raise TypeError("manifest object identity 类型错误")
    result = [manifest.publish_epoch]
    pack_key(result, manifest.manifest_key)
    return tuple(result)


def parse_manifest_object_identity(
        identity_key: tuple[int, ...],
        ) -> tuple[int, tuple[int, ...]]:
    """从对象仓库身份恢复 manifest 发布 epoch 和完整键。"""
    reader = IntegerStreamReader(identity_key)
    epoch = reader.read_positive(label="manifest object epoch")
    manifest_key = reader.read_key(label="manifest object key")
    reader.finish()
    return epoch, manifest_key


@dataclass(frozen=True)
class ReaderLease:
    """一个固定在不可变 location manifest epoch 的 reader 租约。"""

    reader_key: tuple[int, ...]
    publish_epoch: int

    def __post_init__(self) -> None:
        """核验 reader 完整身份和正发布 epoch。"""
        strict_integer_tuple(self.reader_key, label="reader lease reader_key")
        if type(self.publish_epoch) is not int or self.publish_epoch <= 0:
            raise ValueError("reader lease publish_epoch 必须是正严格整数")


class ReaderEpochRegistry:
    """用显式 reader 身份维护进程内旧 epoch 回收屏障。"""

    def __init__(self) -> None:
        """创建没有模块级全局状态的空 reader 注册表。"""
        self._leases: dict[tuple[int, ...], ReaderLease] = {}

    def acquire(
            self, reader_key: tuple[int, ...], publish_epoch: int,
            ) -> ReaderLease:
        """为新 reader 固定 epoch；重复活动身份拒绝覆盖。"""
        lease = ReaderLease(reader_key, publish_epoch)
        if lease.reader_key in self._leases:
            raise TieredSegmentStoreError("reader_key 已有活动租约")
        self._leases[lease.reader_key] = lease
        return lease

    def release(self, lease: ReaderLease) -> None:
        """释放与注册状态逐字段一致的 reader 租约。"""
        if not isinstance(lease, ReaderLease):
            raise TypeError("reader lease 类型错误")
        previous = self._leases.get(lease.reader_key)
        if previous != lease:
            raise TieredSegmentStoreError("reader lease 未注册或 epoch 漂移")
        del self._leases[lease.reader_key]

    def has_readers(self, publish_epoch: int) -> bool:
        """判断指定旧 epoch 是否仍有活动 reader。"""
        if type(publish_epoch) is not int or publish_epoch <= 0:
            raise ValueError("reader epoch 必须是正严格整数")
        return any(
            item.publish_epoch == publish_epoch
            for item in self._leases.values()
        )

    def has_readers_at_or_before(self, publish_epoch: int) -> bool:
        """判断迁移前任一历史 epoch 是否仍可能引用待回收位置。"""
        if type(publish_epoch) is not int or publish_epoch <= 0:
            raise ValueError("reader barrier epoch 必须是正严格整数")
        return any(
            item.publish_epoch <= publish_epoch
            for item in self._leases.values()
        )

    def snapshot(self) -> tuple[ReaderLease, ...]:
        """按 reader 完整身份返回活动租约快照。"""
        return tuple(self._leases[key] for key in sorted(self._leases))


@dataclass(frozen=True)
class ContinuationToken:
    """绑定完整 read view、查询范围和最后稳定键的续页令牌。"""

    reader_key: tuple[int, ...]
    manifest_state_key: tuple[int, ...]
    descriptor_key: tuple[int, ...]
    lower_key: tuple[int, ...] | None
    upper_key: tuple[int, ...] | None
    after_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验续页身份、完整 manifest 状态、范围和最后键。"""
        for label, value in (
                ("reader_key", self.reader_key),
                ("manifest_state_key", self.manifest_state_key),
                ("descriptor_key", self.descriptor_key),
                ("after_key", self.after_key)):
            strict_integer_tuple(value, label=f"continuation {label}")
        if self.lower_key is not None:
            strict_integer_tuple(
                self.lower_key, label="continuation lower_key")
        if self.upper_key is not None:
            strict_integer_tuple(
                self.upper_key, label="continuation upper_key")
        if (self.lower_key is not None and self.upper_key is not None
                and self.lower_key > self.upper_key):
            raise ValueError("continuation range 不能反向")


@dataclass(frozen=True)
class StablePageResult:
    """一次 bounded range read 的记录、read view 和后续令牌。"""

    records: tuple[SegmentRecord, ...]
    publish_epoch: int
    continuation: ContinuationToken | None

    @property
    def has_more(self) -> bool:
        """判断当前 read view 是否还有未消费稳定键。"""
        return self.continuation is not None


class BoundedSegmentReader:
    """固定读取一个 manifest epoch 和逻辑 descriptor 的有界 reader。"""

    def __init__(
            self,
            store: "TieredSegmentStore",
            lease: ReaderLease,
            manifest: LocationManifest,
            descriptor_key: tuple[int, ...],
            ) -> None:
        """绑定 store、活动租约、不可变 manifest 和逻辑描述。"""
        self._store = store
        self.lease = lease
        self.manifest = manifest
        self.descriptor_key = strict_integer_tuple(
            descriptor_key, label="bounded reader descriptor_key")
        self._closed = False

    def page(
            self,
            *,
            budget: SegmentBudget,
            lower_key: tuple[int, ...] | None = None,
            upper_key: tuple[int, ...] | None = None,
            continuation: ContinuationToken | None = None,
            ) -> StablePageResult:
        """按完整稳定键和固定 read fence 读取一页，不使用裸 offset。"""
        self._ensure_open()
        if not isinstance(budget, SegmentBudget):
            raise TypeError("bounded page budget 类型错误")
        lower = None if lower_key is None else strict_integer_tuple(
            lower_key, label="bounded page lower_key")
        upper = None if upper_key is None else strict_integer_tuple(
            upper_key, label="bounded page upper_key")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("bounded page range 不能反向")
        after = None
        if continuation is not None:
            self._validate_continuation(continuation, lower, upper)
            after = continuation.after_key
        selected: list[SegmentRecord] = []
        selected_bytes = 0
        seen: set[tuple[int, ...]] = set()
        for entry in self._matching_entries(lower, upper, after):
            segment = self._store._load_segment(entry)
            for record in segment.records:
                if lower is not None and record.record_key < lower:
                    continue
                if after is not None and record.record_key <= after:
                    continue
                if upper is not None and record.record_key > upper:
                    return StablePageResult(
                        tuple(selected), self.manifest.publish_epoch, None)
                if record.record_key in seen:
                    raise TieredSegmentStoreError(
                        "同一 read view 出现重复 canonical record_key")
                size = record.size_bytes()
                if (len(selected) >= budget.object_limit
                        or selected_bytes + size > budget.byte_limit):
                    if not selected:
                        raise SegmentBudgetExceeded(
                            "单条 segment record 超过分页字节预算")
                    return StablePageResult(
                        tuple(selected),
                        self.manifest.publish_epoch,
                        self._continuation(lower, upper, selected[-1].record_key),
                    )
                seen.add(record.record_key)
                selected.append(record)
                selected_bytes += size
        return StablePageResult(
            tuple(selected), self.manifest.publish_epoch, None)

    def prefetch(
            self,
            cache: SegmentPageCache,
            *,
            budget: SegmentBudget,
            lower_key: tuple[int, ...] | None = None,
            upper_key: tuple[int, ...] | None = None,
            continuation: ContinuationToken | None = None,
            ) -> StablePageResult:
        """读取一页并按同一预算协议装入 query-scoped clean 热集。"""
        if not isinstance(cache, SegmentPageCache):
            raise TypeError("bounded prefetch cache 类型错误")
        result = self.page(
            budget=budget,
            lower_key=lower_key,
            upper_key=upper_key,
            continuation=continuation,
        )
        cache.prefetch(self.descriptor_key, result.records)
        return result

    def close(self) -> None:
        """释放 reader epoch，并触发已满足屏障的旧位置回收。"""
        if self._closed:
            return
        self._store._close_reader(self.lease)
        self._closed = True

    def _matching_entries(
            self,
            lower: tuple[int, ...] | None,
            upper: tuple[int, ...] | None,
            after: tuple[int, ...] | None,
            ) -> tuple[LocationManifestEntry, ...]:
        """只选择 descriptor 和键范围可能命中的 manifest 段。"""
        start = after if after is not None else lower
        result = []
        for entry in self.manifest.entries:
            if entry.descriptor_key != self.descriptor_key:
                continue
            if start is not None and entry.key_range.upper_key <= start:
                if lower is not None and after is None and entry.key_range.upper_key == start:
                    pass
                else:
                    continue
            if upper is not None and entry.key_range.lower_key > upper:
                continue
            result.append(entry)
        return tuple(result)

    def _continuation(
            self,
            lower: tuple[int, ...] | None,
            upper: tuple[int, ...] | None,
            after: tuple[int, ...],
            ) -> ContinuationToken:
        """形成绑定完整 read view 和范围的稳定续页令牌。"""
        return ContinuationToken(
            self.lease.reader_key,
            self.manifest.stable_key(),
            self.descriptor_key,
            lower,
            upper,
            after,
        )

    def _validate_continuation(
            self,
            token: ContinuationToken,
            lower: tuple[int, ...] | None,
            upper: tuple[int, ...] | None,
            ) -> None:
        """拒绝跨 reader、manifest、descriptor 或范围重放续页令牌。"""
        if not isinstance(token, ContinuationToken):
            raise TypeError("continuation token 类型错误")
        if (token.reader_key != self.lease.reader_key
                or token.manifest_state_key != self.manifest.stable_key()
                or token.descriptor_key != self.descriptor_key
                or token.lower_key != lower
                or token.upper_key != upper):
            raise TieredSegmentStoreError("continuation token read view 漂移")

    def _ensure_open(self) -> None:
        """拒绝在 reader 租约释放后继续读取。"""
        if self._closed:
            raise TieredSegmentStoreError("bounded reader 已关闭")


class TieredSegmentStore:
    """协调 sealed object、location ledger、迁移提交和 reader epoch。"""

    def __init__(
            self,
            repository: AppendOnlyObjectRepository,
            registry: StorageRoleRegistry,
            temperature_profile: TemperatureProfile,
            ) -> None:
        """从介质恢复 manifests/commits，完成 roll-forward/rollback 和孤儿清理。"""
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("tiered segment repository 协议错误")
        if not isinstance(registry, StorageRoleRegistry):
            raise TypeError("tiered segment registry 类型错误")
        if not isinstance(temperature_profile, TemperatureProfile):
            raise TypeError("tiered segment temperature_profile 类型错误")
        self.repository = repository
        self.registry = registry
        self.temperature_profile = temperature_profile
        self.ledger = LocationManifestLedger(registry, temperature_profile)
        self.reader_epochs = ReaderEpochRegistry()
        self._commits: dict[
            tuple[int, ...], tuple[MigrationCommitRecord, ...]
        ] = {}
        self._releases: dict[
            tuple[int, ...], tuple[SegmentReleaseCommitRecord, ...]
        ] = {}
        self._load_manifests()
        self._load_commits()
        self._load_releases()
        self._recover_commits()
        self._recover_releases()
        self._reclaim_unreferenced_segments()

    def current_manifest(self) -> LocationManifest | None:
        """返回当前完整 location epoch；尚未发布时返回 None。"""
        return self.ledger.current()

    def recover_pending_operations(self) -> None:
        """重载持久化阶段记录，推进未完成迁移/释放并回收未引用段。"""
        self._load_manifests()
        self._load_commits()
        self._load_releases()
        self._recover_commits()
        self._recover_releases()
        self._reclaim_unreferenced_segments()

    def descriptor_state_key(
            self,
            descriptor_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[int, ...]:
        """返回指定逻辑描述的完整内容状态，排除无关发布和物理温层变化。"""
        if (not isinstance(descriptor_keys, tuple)
                or not descriptor_keys):
            raise TypeError("descriptor state keys 必须是非空 tuple")
        descriptors = tuple(sorted(strict_integer_tuple(
            item, label="descriptor state descriptor_key")
            for item in descriptor_keys))
        if len(set(descriptors)) != len(descriptors):
            raise ValueError("descriptor state keys 不得重复")
        current = self.ledger.current()
        selected = tuple(
            entry for entry in (() if current is None else current.entries)
            if entry.descriptor_key in descriptors
        )
        result = [DESCRIPTOR_STATE_KEY_VERSION, len(descriptors)]
        for descriptor in descriptors:
            pack_key(result, descriptor)
        result.append(len(selected))
        for entry in selected:
            for value in (
                    entry.descriptor_key,
                    entry.segment_key,
                    entry.key_range.lower_key,
                    entry.key_range.upper_key,
                    entry.version_key,
                    entry.checksum_key):
                pack_key(result, value)
            result.append(len(entry.dependencies))
            for dependency in entry.dependencies:
                pack_key(result, dependency.descriptor_key)
                pack_key(result, dependency.version_key)
                pack_key(result, dependency.checksum_key)
            result.append(entry.read_fence)
        return tuple(result)

    def publish_delta(
            self,
            delta: OpenHotDelta,
            *,
            segment_key: tuple[int, ...],
            tier_key: tuple[int, ...],
            read_fence: int,
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> LocationManifest:
        """批量封存开放 hot delta，完整发布后才确认并清空 delta。"""
        if not isinstance(delta, OpenHotDelta):
            raise TypeError("publish delta 类型错误")
        segment = delta.seal(segment_key, read_fence)
        manifest = self.publish_segment(
            segment,
            tier_key=tier_key,
            manifest_key=manifest_key,
            migration_key=migration_key,
            fault_injector=fault_injector,
        )
        delta.acknowledge(segment)
        return manifest

    def publish_segment(
            self,
            segment: SealedSegment,
            *,
            tier_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> LocationManifest:
        """把新 sealed segment 作为非重叠 canonical 范围发布到下一 epoch。"""
        if not isinstance(segment, SealedSegment):
            raise TypeError("publish segment 类型错误")
        target_tier = self._validate_segment_target(segment, tier_key)
        current = self.ledger.current()
        existing = self._entry_by_segment(current, segment.segment_key)
        if existing is not None:
            self._validate_entry_segment(existing, segment, target_tier)
            return current
        previous_epoch = 0 if current is None else current.publish_epoch
        prepared = MigrationCommitRecord(
            strict_integer_tuple(
                migration_key, label="publish segment migration_key"),
            MIGRATION_PHASE_PREPARED,
            segment.descriptor_key,
            segment.segment_key,
            (),
            target_tier,
            segment.version_key,
            segment.checksum_key,
            segment.read_fence,
            previous_epoch,
            previous_epoch + 1,
            strict_integer_tuple(
                manifest_key, label="publish segment manifest_key"),
        )
        return self._execute_prepared(
            prepared,
            segment,
            fault_injector=fault_injector,
        )

    def migrate(
            self,
            segment_key: tuple[int, ...],
            *,
            target_tier_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> LocationManifest:
        """复制核验一个已发布段，切换新 epoch 后等待旧 reader 再回收。"""
        current = self.ledger.current()
        if current is None:
            raise TieredSegmentStoreError("没有已发布 segment 可迁移")
        key = strict_integer_tuple(segment_key, label="migrate segment_key")
        entry = self._entry_by_segment(current, key)
        if entry is None:
            raise KeyError(f"当前 manifest 不含 segment: {key}")
        target = strict_integer_tuple(
            target_tier_key, label="migrate target_tier_key")
        if not self.temperature_profile.has(target):
            raise TieredSegmentStoreError("迁移目标温层未注册")
        if entry.tier_key == target:
            raise TieredSegmentStoreError("迁移目标与当前温层相同")
        segment = self._load_segment(entry)
        prepared = MigrationCommitRecord(
            strict_integer_tuple(migration_key, label="migrate migration_key"),
            MIGRATION_PHASE_PREPARED,
            entry.descriptor_key,
            entry.segment_key,
            (SegmentCopyReference(entry.segment_key, entry.tier_key),),
            target,
            entry.version_key,
            entry.checksum_key,
            entry.read_fence,
            current.publish_epoch,
            current.publish_epoch + 1,
            strict_integer_tuple(manifest_key, label="migrate manifest_key"),
        )
        return self._execute_prepared(
            prepared,
            segment,
            fault_injector=fault_injector,
        )

    def compact(
            self,
            source_segment_keys: tuple[tuple[int, ...], ...],
            *,
            target_segment_key: tuple[int, ...],
            target_tier_key: tuple[int, ...],
            version_key: tuple[int, ...],
            read_fence: int,
            budget: SegmentBudget,
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> LocationManifest:
        """在独立对象/字节预算内合并多个段，并原子替换全部源副本。"""
        current = self.ledger.current()
        if current is None:
            raise TieredSegmentStoreError("没有已发布 segment 可 compaction")
        if (not isinstance(source_segment_keys, tuple)
                or len(source_segment_keys) < 2):
            raise ValueError("compaction 至少需要两个源 segment")
        source_keys = tuple(strict_integer_tuple(
            item, label="compaction source segment_key")
            for item in source_segment_keys)
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("compaction source segment_key 不得重复")
        target_key = strict_integer_tuple(
            target_segment_key, label="compaction target segment_key")
        if self._entry_by_segment(current, target_key) is not None:
            raise TieredSegmentStoreError("compaction target segment_key 已发布")
        target_tier = strict_integer_tuple(
            target_tier_key, label="compaction target_tier_key")
        if not self.temperature_profile.has(target_tier):
            raise TieredSegmentStoreError("compaction target tier 未注册")
        if not isinstance(budget, SegmentBudget):
            raise TypeError("compaction budget 类型错误")
        source_entries = []
        for source_key in source_keys:
            entry = self._entry_by_segment(current, source_key)
            if entry is None:
                raise KeyError(f"compaction 源 segment 不存在: {source_key}")
            source_entries.append(entry)
        descriptors = {item.descriptor_key for item in source_entries}
        if len(descriptors) != 1:
            raise TieredSegmentStoreError("compaction 源 segment 跨 descriptor")
        records: list[SegmentRecord] = []
        record_keys: set[tuple[int, ...]] = set()
        size_bytes = 0
        dependencies = None
        for entry in source_entries:
            segment = self._load_segment(entry)
            if dependencies is None:
                dependencies = segment.dependencies
            elif segment.dependencies != dependencies:
                raise TieredSegmentStoreError("compaction 源依赖漂移")
            for record in segment.records:
                if record.record_key in record_keys:
                    raise TieredSegmentStoreError(
                        "compaction 源段出现重复 canonical key")
                next_size = size_bytes + record.size_bytes()
                if len(records) + 1 > budget.object_limit:
                    raise SegmentBudgetExceeded("compaction 超过对象数预算")
                if next_size > budget.byte_limit:
                    raise SegmentBudgetExceeded("compaction 超过字节预算")
                record_keys.add(record.record_key)
                records.append(record)
                size_bytes = next_size
        if dependencies is None:
            raise AssertionError("compaction 源段为空")
        segment = SealedSegment(
            source_entries[0].descriptor_key,
            target_key,
            strict_integer_tuple(version_key, label="compaction version_key"),
            dependencies,
            read_fence,
            tuple(records),
        )
        if segment.size_bytes > budget.byte_limit:
            raise SegmentBudgetExceeded("compaction 段封装后超过字节预算")
        self._validate_segment_target(segment, target_tier)
        prepared = MigrationCommitRecord(
            strict_integer_tuple(
                migration_key, label="compaction migration_key"),
            MIGRATION_PHASE_PREPARED,
            segment.descriptor_key,
            segment.segment_key,
            tuple(SegmentCopyReference(
                item.segment_key, item.tier_key) for item in source_entries),
            target_tier,
            segment.version_key,
            segment.checksum_key,
            segment.read_fence,
            current.publish_epoch,
            current.publish_epoch + 1,
            strict_integer_tuple(
                manifest_key, label="compaction manifest_key"),
        )
        return self._execute_prepared(
            prepared,
            segment,
            fault_injector=fault_injector,
        )

    def release_rebuildable_segments(
            self,
            segment_keys: tuple[tuple[int, ...], ...],
            *,
            release_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> LocationManifest:
        """发布无目标新 epoch，并在 reader barrier 后释放可重建源段。"""
        current = self.ledger.current()
        if current is None:
            raise TieredSegmentStoreError("没有已发布 segment 可释放")
        if (not isinstance(segment_keys, tuple) or not segment_keys):
            raise TypeError("release segment_keys 必须是非空 tuple")
        keys = tuple(strict_integer_tuple(
            item, label="release segment_key") for item in segment_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("release segment_keys 不得重复")
        entries = []
        for key in keys:
            entry = self._entry_by_segment(current, key)
            if entry is None:
                raise KeyError(f"release 源 segment 不存在: {key}")
            entries.append(entry)
        descriptors = {item.descriptor_key for item in entries}
        if len(descriptors) != 1:
            raise TieredSegmentStoreError("一次 release 不得跨 descriptor")
        descriptor_key = next(iter(descriptors))
        descriptor = self.registry.get(descriptor_key)
        if descriptor.role != STORAGE_ROLE_REBUILDABLE:
            raise TieredSegmentStoreError("只有可重建存储角色允许无目标 release")
        selected = {item.segment_key for item in entries}
        for entry in current.entries:
            if entry.segment_key in selected:
                continue
            if any(dependency.descriptor_key == descriptor_key
                   for dependency in entry.dependencies):
                raise TieredSegmentStoreError(
                    "仍有已发布 segment 依赖待释放 descriptor")
        sources = tuple(SegmentCopyReference(
            item.segment_key, item.tier_key) for item in entries)
        prepared = SegmentReleaseCommitRecord(
            strict_integer_tuple(release_key, label="segment release_key"),
            MIGRATION_PHASE_PREPARED,
            descriptor_key,
            sources,
            current.publish_epoch,
            current.publish_epoch + 1,
            strict_integer_tuple(
                manifest_key, label="segment release manifest_key"),
        )
        self._validate_release_sources(prepared, current)
        return self._execute_release(
            prepared, fault_injector=fault_injector)

    def open_reader(
            self,
            reader_key: tuple[int, ...],
            descriptor_key: tuple[int, ...],
            ) -> BoundedSegmentReader:
        """让新 reader 固定在当前完整 epoch，并绑定一个已注册逻辑描述。"""
        manifest = self.ledger.current()
        if manifest is None:
            raise TieredSegmentStoreError("没有已发布 manifest 可读取")
        descriptor = strict_integer_tuple(
            descriptor_key, label="open reader descriptor_key")
        self.registry.get(descriptor)
        lease = self.reader_epochs.acquire(reader_key, manifest.publish_epoch)
        return BoundedSegmentReader(self, lease, manifest, descriptor)

    def reclaim_ready(
            self,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> int:
        """回收所有已发布且旧 epoch reader 已退出的源位置。"""
        completed = 0
        for migration_key in sorted(self._commits):
            chain = validate_commit_chain(self._commits[migration_key])
            phases = {item.phase for item in chain}
            if (MIGRATION_PHASE_PUBLISHED not in phases
                    or MIGRATION_PHASE_RECLAIMED in phases
                    or MIGRATION_PHASE_ABORTED in phases):
                continue
            prepared = chain[0]
            if self._complete_reclaim(
                    prepared, fault_injector=fault_injector):
                completed += 1
        for release_key in sorted(self._releases):
            chain = validate_release_chain(self._releases[release_key])
            phases = {item.phase for item in chain}
            if (MIGRATION_PHASE_PUBLISHED not in phases
                    or MIGRATION_PHASE_RECLAIMED in phases
                    or MIGRATION_PHASE_ABORTED in phases):
                continue
            if self._complete_release_reclaim(
                    chain[0], fault_injector=fault_injector):
                completed += 1
        return completed

    def _execute_release(
            self,
            prepared: SegmentReleaseCommitRecord,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None,
            ) -> LocationManifest:
        """执行 prepared、manifest 发布、reader 切换和有屏障物理释放。"""
        self._append_release(prepared)
        hit_repository_fault(
            fault_injector,
            FAULT_RELEASE_AFTER_PREPARE,
            {"release_key": prepared.release_key},
        )
        manifest = self._build_release_manifest(prepared)
        self._persist_manifest(manifest, fault_injector=fault_injector)
        self.ledger.append(manifest)
        hit_repository_fault(
            fault_injector,
            FAULT_RELEASE_AFTER_MANIFEST_PUBLISH,
            {"release_key": prepared.release_key},
        )
        self._append_release(prepared.with_phase(MIGRATION_PHASE_PUBLISHED))
        self._complete_release_reclaim(
            prepared, fault_injector=fault_injector)
        return manifest

    def _recover_releases(self) -> None:
        """启动时把未完成 release 确定性 roll-forward 到发布或回收完成。"""
        for release_key in sorted(tuple(self._releases)):
            chain = validate_release_chain(self._releases[release_key])
            phases = {item.phase for item in chain}
            prepared = chain[0]
            if (MIGRATION_PHASE_RECLAIMED in phases
                    or MIGRATION_PHASE_ABORTED in phases):
                continue
            if MIGRATION_PHASE_PUBLISHED not in phases:
                manifest = self._manifest_for_release(prepared)
                if manifest is None:
                    current = self.ledger.current()
                    if (current is None
                            or current.publish_epoch
                            != prepared.previous_epoch):
                        raise SegmentReleaseIntegrityError(
                            "release recovery 的 previous epoch 已被越过")
                    self._validate_release_sources(prepared, current)
                    manifest = self._build_release_manifest(prepared)
                    self._persist_manifest(manifest)
                    self.ledger.append(manifest)
                self._append_release(
                    prepared.with_phase(MIGRATION_PHASE_PUBLISHED))
            self._complete_release_reclaim(prepared)

    def _complete_release_reclaim(
            self,
            prepared: SegmentReleaseCommitRecord,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> bool:
        """旧 epoch 无 reader 后物理回收全部源副本并登记 reclaimed。"""
        chain = validate_release_chain(self._releases[prepared.release_key])
        phases = {item.phase for item in chain}
        if MIGRATION_PHASE_RECLAIMED in phases:
            return False
        if MIGRATION_PHASE_PUBLISHED not in phases:
            return False
        if self.reader_epochs.has_readers_at_or_before(
                prepared.previous_epoch):
            return False
        hit_repository_fault(
            fault_injector,
            FAULT_RELEASE_BEFORE_SOURCE_RECLAIM,
            {
                "release_key": prepared.release_key,
                "source_count": len(prepared.source_copies),
            },
        )
        for source in prepared.source_copies:
            self.repository.reclaim(
                OBJECT_KIND_SEGMENT,
                segment_copy_identity(source.tier_key, source.segment_key),
                fault_injector=fault_injector,
            )
        hit_repository_fault(
            fault_injector,
            FAULT_RELEASE_AFTER_SOURCE_RECLAIM,
            {
                "release_key": prepared.release_key,
                "source_count": len(prepared.source_copies),
            },
        )
        self._append_release(
            prepared.with_phase(MIGRATION_PHASE_RECLAIMED))
        return True

    def _build_release_manifest(
            self,
            prepared: SegmentReleaseCommitRecord,
            ) -> LocationManifest:
        """从当前 epoch 删除精确源副本，并原样克隆其余位置记录。"""
        current = self.ledger.current()
        if (current is None
                or current.publish_epoch != prepared.previous_epoch):
            existing = self._manifest_for_release(prepared)
            if existing is not None:
                return existing
            raise TieredSegmentStoreError("构造 release manifest 时 epoch 漂移")
        sources = set(prepared.source_copies)
        removed: set[SegmentCopyReference] = set()
        entries = []
        for entry in current.entries:
            source = SegmentCopyReference(entry.segment_key, entry.tier_key)
            if source in sources:
                if entry.descriptor_key != prepared.descriptor_key:
                    raise TieredSegmentStoreError(
                        "release 源 descriptor 与提交记录漂移")
                removed.add(source)
                continue
            entries.append(self._clone_entry(entry, prepared.publish_epoch))
        if removed != sources:
            raise TieredSegmentStoreError("当前 manifest 缺少 release 源副本")
        return LocationManifest(
            prepared.manifest_key,
            self.temperature_profile.profile_key,
            prepared.publish_epoch,
            prepared.previous_epoch,
            tuple(entries),
        )

    def _validate_release_sources(
            self,
            prepared: SegmentReleaseCommitRecord,
            manifest: LocationManifest,
            ) -> None:
        """核验待释放源仍由当前 manifest 唯一引用且物理副本完整可读。"""
        entries = {
            SegmentCopyReference(item.segment_key, item.tier_key): item
            for item in manifest.entries
        }
        for source in prepared.source_copies:
            entry = entries.get(source)
            if entry is None or entry.descriptor_key != prepared.descriptor_key:
                raise TieredSegmentStoreError("release 源副本不在当前 manifest")
            self._load_segment(entry)

    def _execute_prepared(
            self,
            prepared: MigrationCommitRecord,
            segment: SealedSegment,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None,
            ) -> LocationManifest:
        """执行写目标、核验、prepared、发布、切换和有屏障回收。"""
        target_identity = segment_copy_identity(
            prepared.target_tier_key, prepared.segment_key)
        self.repository.put(
            OBJECT_KIND_SEGMENT,
            target_identity,
            segment.to_bytes(),
            fault_injector=fault_injector,
        )
        hit_repository_fault(
            fault_injector,
            FAULT_MIGRATION_AFTER_TARGET_WRITE,
            {"migration_key": prepared.migration_key},
        )
        restored = self._read_segment_copy(
            prepared.target_tier_key, prepared.segment_key)
        self._validate_commit_segment(prepared, restored)
        hit_repository_fault(
            fault_injector,
            FAULT_MIGRATION_AFTER_TARGET_VERIFY,
            {"migration_key": prepared.migration_key},
        )
        self._append_commit(prepared)
        hit_repository_fault(
            fault_injector,
            FAULT_MIGRATION_AFTER_PREPARE,
            {"migration_key": prepared.migration_key},
        )
        manifest = self._build_manifest(prepared, restored)
        self._persist_manifest(manifest, fault_injector=fault_injector)
        self.ledger.append(manifest)
        hit_repository_fault(
            fault_injector,
            FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH,
            {"migration_key": prepared.migration_key},
        )
        self._append_commit(prepared.with_phase(MIGRATION_PHASE_PUBLISHED))
        hit_repository_fault(
            fault_injector,
            FAULT_MIGRATION_AFTER_READER_SWITCH,
            {"migration_key": prepared.migration_key},
        )
        self._complete_reclaim(prepared, fault_injector=fault_injector)
        return manifest

    def _recover_commits(self) -> None:
        """启动时对 prepared/published 迁移确定性 roll-forward 或 rollback。"""
        for migration_key in sorted(tuple(self._commits)):
            chain = validate_commit_chain(self._commits[migration_key])
            phases = {item.phase for item in chain}
            prepared = chain[0]
            if (MIGRATION_PHASE_RECLAIMED in phases
                    or MIGRATION_PHASE_ABORTED in phases):
                continue
            if MIGRATION_PHASE_PUBLISHED not in phases:
                manifest = self._manifest_for_commit(prepared)
                if manifest is None:
                    current = self.ledger.current()
                    current_epoch = 0 if current is None else current.publish_epoch
                    if current_epoch != prepared.previous_epoch:
                        raise SegmentCommitIntegrityError(
                            "prepared recovery 的 previous epoch 已被越过")
                    try:
                        segment = self._read_segment_copy(
                            prepared.target_tier_key,
                            prepared.segment_key,
                        )
                        self._validate_commit_segment(prepared, segment)
                    except (KeyError, SegmentIntegrityError,
                            SegmentRepositoryError,
                            TieredSegmentStoreError):
                        self.repository.reclaim(
                            OBJECT_KIND_SEGMENT,
                            segment_copy_identity(
                                prepared.target_tier_key,
                                prepared.segment_key,
                            ),
                        )
                        self._append_commit(
                            prepared.with_phase(MIGRATION_PHASE_ABORTED))
                        continue
                    manifest = self._build_manifest(prepared, segment)
                    self._persist_manifest(manifest)
                    self.ledger.append(manifest)
                self._append_commit(
                    prepared.with_phase(MIGRATION_PHASE_PUBLISHED))
            self._complete_reclaim(prepared)

    def _complete_reclaim(
            self,
            prepared: MigrationCommitRecord,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> bool:
        """在旧 epoch 无 reader 后回收源副本并发布 reclaimed 阶段。"""
        chain = validate_commit_chain(self._commits[prepared.migration_key])
        phases = {item.phase for item in chain}
        if MIGRATION_PHASE_RECLAIMED in phases:
            return False
        if MIGRATION_PHASE_PUBLISHED not in phases:
            return False
        if (prepared.previous_epoch > 0
                and self.reader_epochs.has_readers_at_or_before(
                    prepared.previous_epoch)):
            return False
        if prepared.source_copies:
            hit_repository_fault(
                fault_injector,
                FAULT_MIGRATION_BEFORE_SOURCE_RECLAIM,
                {
                    "migration_key": prepared.migration_key,
                    "source_count": len(prepared.source_copies),
                },
            )
            for source in prepared.source_copies:
                self.repository.reclaim(
                    OBJECT_KIND_SEGMENT,
                    segment_copy_identity(
                        source.tier_key,
                        source.segment_key,
                    ),
                    fault_injector=fault_injector,
                )
            hit_repository_fault(
                fault_injector,
                FAULT_MIGRATION_AFTER_SOURCE_RECLAIM,
                {
                    "migration_key": prepared.migration_key,
                    "source_count": len(prepared.source_copies),
                },
            )
        self._append_commit(prepared.with_phase(MIGRATION_PHASE_RECLAIMED))
        return True

    def _build_manifest(
            self,
            prepared: MigrationCommitRecord,
            segment: SealedSegment,
            ) -> LocationManifest:
        """从当前完整 epoch 和已核验目标段构造下一 canonical manifest。"""
        current = self.ledger.current()
        current_epoch = 0 if current is None else current.publish_epoch
        if current_epoch != prepared.previous_epoch:
            existing = self._manifest_for_commit(prepared)
            if existing is not None:
                return existing
            raise TieredSegmentStoreError("构造 manifest 时 previous epoch 漂移")
        entries = []
        source_copies = set(prepared.source_copies)
        replaced: set[SegmentCopyReference] = set()
        for entry in () if current is None else current.entries:
            source = SegmentCopyReference(entry.segment_key, entry.tier_key)
            if source in source_copies:
                if entry.descriptor_key != prepared.descriptor_key:
                    raise TieredSegmentStoreError(
                        "段替换源 descriptor 与 commit 漂移")
                replaced.add(source)
                continue
            entries.append(self._clone_entry(entry, prepared.publish_epoch))
        if replaced != source_copies:
            raise TieredSegmentStoreError("当前 manifest 缺少段替换源副本")
        entries.append(self._entry_from_segment(
            segment,
            prepared.target_tier_key,
            prepared.publish_epoch,
        ))
        return LocationManifest(
            prepared.manifest_key,
            self.temperature_profile.profile_key,
            prepared.publish_epoch,
            None if prepared.previous_epoch == 0 else prepared.previous_epoch,
            tuple(entries),
        )

    def _entry_from_segment(
            self,
            segment: SealedSegment,
            tier_key: tuple[int, ...],
            publish_epoch: int,
            ) -> LocationManifestEntry:
        """从已核验 sealed segment 形成一个新 epoch 的位置 entry。"""
        return LocationManifestEntry(
            segment.descriptor_key,
            segment.segment_key,
            tier_key,
            ManifestKeyRange(segment.lower_key, segment.upper_key),
            segment.version_key,
            segment.checksum_key,
            segment.dependencies,
            segment.read_fence,
            publish_epoch,
        )

    def _clone_entry(
            self, entry: LocationManifestEntry, publish_epoch: int,
            ) -> LocationManifestEntry:
        """把未迁移 entry 原样带入下一发布 epoch。"""
        return LocationManifestEntry(
            entry.descriptor_key,
            entry.segment_key,
            entry.tier_key,
            entry.key_range,
            entry.version_key,
            entry.checksum_key,
            entry.dependencies,
            entry.read_fence,
            publish_epoch,
        )

    def _persist_manifest(
            self,
            manifest: LocationManifest,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> None:
        """以 seal-last 对象发布 location manifest，完整身份重放幂等。"""
        self.repository.put(
            OBJECT_KIND_LOCATION_MANIFEST,
            manifest_object_identity(manifest),
            manifest.to_bytes(),
            fault_injector=fault_injector,
        )

    def _append_commit(self, record: MigrationCommitRecord) -> None:
        """幂等持久化一个迁移阶段，并更新内存阶段链。"""
        self.repository.put(
            OBJECT_KIND_MIGRATION_COMMIT,
            record.identity_key(),
            record.to_bytes(),
        )
        previous = self._commits.get(record.migration_key, ())
        if any(item.phase == record.phase for item in previous):
            matched = next(item for item in previous if item.phase == record.phase)
            if matched != record:
                raise SegmentCommitIntegrityError("同一 migration phase 内容漂移")
            return
        updated = validate_commit_chain((*previous, record))
        self._commits[record.migration_key] = updated

    def _append_release(self, record: SegmentReleaseCommitRecord) -> None:
        """幂等持久化一个释放阶段，并更新内存阶段链。"""
        self.repository.put(
            OBJECT_KIND_SEGMENT_RELEASE,
            record.identity_key(),
            record.to_bytes(),
        )
        previous = self._releases.get(record.release_key, ())
        if any(item.phase == record.phase for item in previous):
            matched = next(
                item for item in previous if item.phase == record.phase)
            if matched != record:
                raise SegmentReleaseIntegrityError(
                    "同一 release phase 内容漂移")
            return
        self._releases[record.release_key] = validate_release_chain(
            (*previous, record))

    def _load_manifests(self) -> None:
        """从对象仓库恢复全部 manifest epoch 并重放到 K-01 ledger。"""
        manifests = []
        for descriptor in self.repository.list_kind(
                OBJECT_KIND_LOCATION_MANIFEST):
            payload = self.repository.get(
                OBJECT_KIND_LOCATION_MANIFEST,
                descriptor.identity_key,
            )
            manifest = LocationManifest.from_bytes(payload)
            if manifest_object_identity(manifest) != descriptor.identity_key:
                raise ManifestIntegrityError("manifest 对象身份与 payload 漂移")
            manifests.append(manifest)
        for manifest in sorted(manifests, key=lambda item: item.publish_epoch):
            self.ledger.append(manifest)

    def _load_commits(self) -> None:
        """从对象仓库恢复全部迁移阶段并按 migration 分组核验。"""
        grouped: dict[tuple[int, ...], list[MigrationCommitRecord]] = {}
        for descriptor in self.repository.list_kind(
                OBJECT_KIND_MIGRATION_COMMIT):
            payload = self.repository.get(
                OBJECT_KIND_MIGRATION_COMMIT,
                descriptor.identity_key,
            )
            record = MigrationCommitRecord.from_bytes(payload)
            if record.identity_key() != descriptor.identity_key:
                raise SegmentCommitIntegrityError(
                    "migration commit 对象身份与 payload 漂移")
            grouped.setdefault(record.migration_key, []).append(record)
        self._commits = {
            key: validate_commit_chain(tuple(records))
            for key, records in grouped.items()
        }

    def _load_releases(self) -> None:
        """从对象仓库恢复全部 release 阶段链并逐条核验。"""
        grouped: dict[
            tuple[int, ...], list[SegmentReleaseCommitRecord]
        ] = {}
        for descriptor in self.repository.list_kind(
                OBJECT_KIND_SEGMENT_RELEASE):
            payload = self.repository.get(
                OBJECT_KIND_SEGMENT_RELEASE,
                descriptor.identity_key,
            )
            record = SegmentReleaseCommitRecord.from_bytes(payload)
            if descriptor.identity_key != record.identity_key():
                raise SegmentReleaseIntegrityError(
                    "release 对象 identity 与 payload 漂移")
            grouped.setdefault(record.release_key, []).append(record)
        self._releases = {
            key: validate_release_chain(tuple(records))
            for key, records in grouped.items()
        }

    def _manifest_for_commit(
            self, prepared: MigrationCommitRecord,
            ) -> LocationManifest | None:
        """读取并核验迁移声明的发布 epoch，尚未发布时返回 None。"""
        try:
            manifest = self.ledger.get(prepared.publish_epoch)
        except KeyError:
            return None
        if manifest.manifest_key != prepared.manifest_key:
            raise SegmentCommitIntegrityError("commit 指向的 manifest_key 漂移")
        return manifest

    def _manifest_for_release(
            self,
            prepared: SegmentReleaseCommitRecord,
            ) -> LocationManifest | None:
        """读取并核验 release 声明的发布 epoch，尚未发布时返回空。"""
        try:
            manifest = self.ledger.get(prepared.publish_epoch)
        except KeyError:
            return None
        if manifest.manifest_key != prepared.manifest_key:
            raise SegmentReleaseIntegrityError(
                "release 指向的 manifest_key 漂移")
        return manifest

    def _validate_segment_target(
            self,
            segment: SealedSegment,
            tier_key: tuple[int, ...],
            ) -> tuple[int, ...]:
        """核验温层、角色和段依赖声明，不读取业务 lifecycle。"""
        target = strict_integer_tuple(tier_key, label="segment target tier")
        if not self.temperature_profile.has(target):
            raise TieredSegmentStoreError("segment target tier 未注册")
        descriptor = self.registry.get(segment.descriptor_key)
        dependencies = tuple(
            item.descriptor_key for item in segment.dependencies)
        if dependencies != descriptor.dependency_keys:
            raise TieredSegmentStoreError("sealed segment 依赖与角色声明漂移")
        return target

    def _load_segment(self, entry: LocationManifestEntry) -> SealedSegment:
        """按 manifest 物理位置读取段，并核验全部 entry 字段。"""
        segment = self._read_segment_copy(entry.tier_key, entry.segment_key)
        self._validate_entry_segment(entry, segment, entry.tier_key)
        return segment

    def _read_segment_copy(
            self,
            tier_key: tuple[int, ...],
            segment_key: tuple[int, ...],
            ) -> SealedSegment:
        """从对象仓库读取一个指定温层副本并恢复 sealed segment。"""
        payload = self.repository.get(
            OBJECT_KIND_SEGMENT,
            segment_copy_identity(tier_key, segment_key),
        )
        return SealedSegment.from_bytes(payload)

    def _validate_entry_segment(
            self,
            entry: LocationManifestEntry,
            segment: SealedSegment,
            tier_key: tuple[int, ...],
            ) -> None:
        """核验 location entry 与实际 sealed segment 逐字段一致。"""
        if (entry.tier_key != tier_key
                or entry.descriptor_key != segment.descriptor_key
                or entry.segment_key != segment.segment_key
                or entry.key_range != ManifestKeyRange(
                    segment.lower_key, segment.upper_key)
                or entry.version_key != segment.version_key
                or entry.checksum_key != segment.checksum_key
                or entry.dependencies != segment.dependencies
                or entry.read_fence != segment.read_fence):
            raise TieredSegmentStoreError("location entry 与 sealed segment 漂移")

    def _validate_commit_segment(
            self,
            prepared: MigrationCommitRecord,
            segment: SealedSegment,
            ) -> None:
        """核验 prepared 记录和目标 segment 的全部逻辑字段。"""
        if (prepared.descriptor_key != segment.descriptor_key
                or prepared.segment_key != segment.segment_key
                or prepared.version_key != segment.version_key
                or prepared.checksum_key != segment.checksum_key
                or prepared.read_fence != segment.read_fence):
            raise TieredSegmentStoreError("prepared commit 与目标 segment 漂移")

    def _entry_by_segment(
            self,
            manifest: LocationManifest | None,
            segment_key: tuple[int, ...],
            ) -> LocationManifestEntry | None:
        """按完整 segment_key 读取唯一当前 entry。"""
        if manifest is None:
            return None
        matches = [
            item for item in manifest.entries
            if item.segment_key == segment_key
        ]
        if len(matches) > 1:
            raise TieredSegmentStoreError("manifest 重复 segment_key")
        return None if not matches else matches[0]

    def _close_reader(self, lease: ReaderLease) -> None:
        """释放 reader 租约并尝试推进所有等待中的回收。"""
        self.reader_epochs.release(lease)
        self.reclaim_ready()

    def _reclaim_unreferenced_segments(self) -> None:
        """回收不属于当前 epoch 或任何活动 reader 固定 epoch 的孤儿副本。"""
        current = self.ledger.current()
        referenced = set()
        if current is not None:
            referenced_epochs = {current.publish_epoch}
            referenced_epochs.update(
                item.publish_epoch for item in self.reader_epochs.snapshot())
            for epoch in referenced_epochs:
                manifest = self.ledger.get(epoch)
                referenced.update(
                    segment_copy_identity(entry.tier_key, entry.segment_key)
                    for entry in manifest.entries
                )
        for descriptor in self.repository.list_kind(OBJECT_KIND_SEGMENT):
            parse_segment_copy_identity(descriptor.identity_key)
            if descriptor.identity_key not in referenced:
                self.repository.reclaim(
                    OBJECT_KIND_SEGMENT, descriptor.identity_key)


__all__ = [
    "BoundedSegmentReader",
    "ContinuationToken",
    "FAULT_RELEASE_AFTER_MANIFEST_PUBLISH",
    "FAULT_RELEASE_AFTER_PREPARE",
    "FAULT_RELEASE_AFTER_SOURCE_RECLAIM",
    "FAULT_RELEASE_BEFORE_SOURCE_RECLAIM",
    "FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH",
    "FAULT_MIGRATION_AFTER_READER_SWITCH",
    "FAULT_MIGRATION_AFTER_SOURCE_RECLAIM",
    "FAULT_MIGRATION_AFTER_TARGET_VERIFY",
    "FAULT_MIGRATION_AFTER_TARGET_WRITE",
    "FAULT_MIGRATION_AFTER_PREPARE",
    "FAULT_MIGRATION_BEFORE_SOURCE_RECLAIM",
    "ReaderEpochRegistry",
    "ReaderLease",
    "StablePageResult",
    "TieredSegmentStore",
    "TieredSegmentStoreError",
    "manifest_object_identity",
    "parse_manifest_object_identity",
    "parse_segment_copy_identity",
    "segment_copy_identity",
]
