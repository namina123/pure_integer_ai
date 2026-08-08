"""K-04 查询索引基于 Memory source state 的有界增量维护。"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Iterator

from pure_integer_ai.cognition.shared.identity import OwnerScope
from pure_integer_ai.cognition.shared.memory_event import LIFECYCLE_ACTIVE
from pure_integer_ai.cognition.shared.memory_hot_set import (
    decode_memory_candidate_payload,
    ExactMemoryProjectionPlanner,
    encode_memory_candidate_payload,
    MemoryProjectionIndexEntry,
    memory_query_index_record_key,
    visible_owner_keys,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryCandidateBundle,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    MemoryProjectionPublication,
    MemoryProjectionSegment,
    MemoryQueryIndexChange,
    MemoryQueryIndexPartition,
    MemoryQueryIndexProjectionManifest,
    MemoryQueryIndexRun,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
    MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
    MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_HYPOTHESIS_EVENT_TABLE,
)
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.tiered_segment_store import (
    DESCRIPTOR_STATE_KEY_VERSION,
    TieredSegmentStore,
)


MEMORY_QUERY_INDEX_MAINTENANCE_KEY_VERSION = 1
MEMORY_QUERY_INDEX_MAINTENANCE_SEGMENT_OBJECT = 1
MEMORY_QUERY_INDEX_MAINTENANCE_MANIFEST_OBJECT = 2
MEMORY_QUERY_INDEX_MAINTENANCE_MIGRATION_OBJECT = 3
MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_OBJECT = 4
MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_MANIFEST_OBJECT = 5
MEMORY_QUERY_INDEX_MAINTENANCE_READER_OBJECT = 6

_BATCH_APPEND_DESCRIPTORS = frozenset((
    MEMORY_BATCH_EVENT_DESCRIPTOR_KEY,
    MEMORY_BATCH_ACTIVATION_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_INTENT_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_MEMBERSHIP_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_COMMIT_DESCRIPTOR_KEY,
))
_BATCH_ROLLBACK_DESCRIPTORS = frozenset((
    MEMORY_BATCH_ROLLBACK_DESCRIPTOR_KEY,
    MEMORY_BATCH_GROUP_ROLLBACK_DESCRIPTOR_KEY,
))


# object-model: exception
class MemoryQueryIndexMaintenanceError(RuntimeError):
    """增量索引 source state、候选状态或物理发布不闭合。"""


# object-model: exception
class MemoryQueryIndexRebuildRequired(MemoryQueryIndexMaintenanceError):
    """变化超出有界增量证明，调用方必须执行全量重建。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MemoryQueryIndexMaintenancePolicy:
    """一次维护的事件页、变化候选和不可压实 run 硬上限。"""

    event_page_limit: int
    changed_candidate_limit: int
    max_run_count: int = 8
    compaction_page_limit: int = 4_096

    def __post_init__(self) -> None:
        """要求读取边界为正整数，run 上限至少可容纳基线和增量。"""
        if (type(self.event_page_limit) is not int
                or self.event_page_limit <= 0
                or type(self.changed_candidate_limit) is not int
                or self.changed_candidate_limit <= 0
                or type(self.max_run_count) is not int
                or self.max_run_count <= 1
                or type(self.compaction_page_limit) is not int
                or self.compaction_page_limit <= 0):
            raise ValueError(
                "query index maintenance policy 边界非法或 run 上限小于 2")


def _framed_key(
        namespace: tuple[int, ...],
        source_state_key: tuple[int, ...],
        planner_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """形成不会跨 source state 或 planner 复用的固定分帧代键。"""
    result = [MEMORY_QUERY_INDEX_MAINTENANCE_KEY_VERSION]
    for value, label in (
            (namespace, "query index maintenance namespace"),
            (source_state_key, "query index maintenance source state"),
            (planner_key, "query index maintenance planner")):
        pack_key(result, strict_integer_tuple(value, label=label))
    return tuple(result)


def _object_key(
        generation_key: tuple[int, ...],
        object_kind: int,
        ordinal: int,
        ) -> tuple[int, ...]:
    """形成一个增量代内不重用的物理对象键。"""
    generation = strict_integer_tuple(
        generation_key, label="query index maintenance generation")
    if type(object_kind) is not int or object_kind <= 0:
        raise ValueError("query index maintenance object kind 非法")
    if type(ordinal) is not int or ordinal <= 0:
        raise ValueError("query index maintenance ordinal 非法")
    return (
        MEMORY_QUERY_INDEX_MAINTENANCE_KEY_VERSION,
        object_kind,
        len(generation),
        *generation,
        ordinal,
    )


def _source_components(
        state_key: tuple[int, ...],
        ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """拆分 projection state 的物理水位、batch 和 forget 状态。"""
    reader = IntegerStreamReader(state_key)
    fence = reader.read_nonnegative(label="query index source fence")
    batch_state = reader.read_key(label="query index batch state")
    forget_state = reader.read_key(label="query index forget state")
    reader.finish()
    return fence, batch_state, forget_state


def _descriptor_state_entries(
        state_key: tuple[int, ...],
        ) -> tuple[
            tuple[tuple[int, ...], ...],
            frozenset[tuple[object, ...]],
        ]:
    """恢复 descriptor_state_key 的声明集合和无 epoch 内容条目。"""
    if state_key == (0,):
        return (), frozenset()
    reader = IntegerStreamReader(state_key)
    version = reader.read_positive(label="descriptor state version")
    if version != DESCRIPTOR_STATE_KEY_VERSION:
        raise MemoryQueryIndexMaintenanceError(
            "descriptor state version 未注册")
    descriptor_count = reader.read_positive(label="descriptor state count")
    descriptors = tuple(
        reader.read_key(label=f"descriptor state descriptor[{index}]")
        for index in range(descriptor_count)
    )
    if tuple(sorted(set(descriptors))) != descriptors:
        raise MemoryQueryIndexMaintenanceError(
            "descriptor state 声明未唯一排序")
    entry_count = reader.read_nonnegative(label="descriptor state entry count")
    entries: set[tuple[object, ...]] = set()
    for index in range(entry_count):
        descriptor = reader.read_key(label=f"descriptor entry[{index}].descriptor")
        segment = reader.read_key(label=f"descriptor entry[{index}].segment")
        lower = reader.read_key(label=f"descriptor entry[{index}].lower")
        upper = reader.read_key(label=f"descriptor entry[{index}].upper")
        version = reader.read_key(label=f"descriptor entry[{index}].version")
        checksum = reader.read_key(label=f"descriptor entry[{index}].checksum")
        dependency_count = reader.read_nonnegative(
            label=f"descriptor entry[{index}].dependency count")
        dependencies = tuple((
            reader.read_key(label=f"descriptor entry[{index}].dependency[{dep}].descriptor"),
            reader.read_key(label=f"descriptor entry[{index}].dependency[{dep}].version"),
            reader.read_key(label=f"descriptor entry[{index}].dependency[{dep}].checksum"),
        ) for dep in range(dependency_count))
        read_fence = reader.read_nonnegative(
            label=f"descriptor entry[{index}].read fence")
        entries.add((
            descriptor,
            segment,
            lower,
            upper,
            version,
            checksum,
            dependencies,
            read_fence,
        ))
    reader.finish()
    if len(entries) != entry_count:
        raise MemoryQueryIndexMaintenanceError(
            "descriptor state 含重复内容条目")
    return descriptors, frozenset(entries)


def _batch_transition_is_append_only(
        previous_state: tuple[int, ...],
        current_state: tuple[int, ...],
        ) -> bool:
    """只接受旧内容完整保留且没有新增 rollback descriptor 的批次变化。"""
    if previous_state == current_state:
        return True
    previous_descriptors, previous_entries = _descriptor_state_entries(
        previous_state)
    current_descriptors, current_entries = _descriptor_state_entries(
        current_state)
    known_descriptors = (
        _BATCH_APPEND_DESCRIPTORS | _BATCH_ROLLBACK_DESCRIPTORS)
    if (set(current_descriptors) != known_descriptors
            or (previous_descriptors
                and previous_descriptors != current_descriptors)):
        return False
    if not previous_entries.issubset(current_entries):
        return False
    new_entries = current_entries - previous_entries
    new_descriptors = {item[0] for item in new_entries}
    return (
        not new_descriptors & _BATCH_ROLLBACK_DESCRIPTORS
        and new_descriptors.issubset(_BATCH_APPEND_DESCRIPTORS)
    )


# object-model: lifecycle; owner=maintenance-call; cleanup=scope-end
class MemoryQueryIndexMaintainer:
    """把已封存基线索引推进到当前 Memory source state。"""

    def __init__(
            self,
            resolver: MemoryOverlayResolver,
            store: TieredSegmentStore,
            ) -> None:
        """绑定只读 Memory 派生和可重建 K-02 store。"""
        if not isinstance(resolver, MemoryOverlayResolver):
            raise TypeError("query index maintainer resolver 类型错误")
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("query index maintainer store 类型错误")
        self.resolver = resolver
        self.store = store

    def maintain(
            self,
            previous: MemoryQueryIndexProjectionManifest,
            *,
            publication: MemoryProjectionPublication,
            policy: MemoryQueryIndexMaintenancePolicy,
            ) -> MemoryQueryIndexProjectionManifest:
        """只读取 source tail，发布变化候选 run，并返回当前状态清单。"""
        if not isinstance(previous, MemoryQueryIndexProjectionManifest):
            raise TypeError("query index previous manifest 类型错误")
        if not isinstance(publication, MemoryProjectionPublication):
            raise TypeError("query index maintenance publication 类型错误")
        if not isinstance(policy, MemoryQueryIndexMaintenancePolicy):
            raise TypeError("query index maintenance policy 类型错误")
        previous.validate_store(self.store)
        planner = self.resolver.score_provider
        if not isinstance(planner, ExactMemoryProjectionPlanner):
            raise TypeError("query index maintainer scorer 未实现精确 planner")
        planner_key = strict_integer_tuple(
            planner.state_key(), label="query index maintenance planner state")
        if planner_key != previous.planner_key:
            raise MemoryQueryIndexRebuildRequired("planner state 已变化")
        storage = previous.storage
        event_log = self.resolver.aggregates.event_log
        if (storage.memory_space != event_log.memory_space_identity
                or storage.memory_space_id != event_log.memory_space_id):
            raise ValueError("query index maintainer Memory 空间漂移")
        self.resolver.aggregates.require_clean(access=storage.access)
        current_state = event_log.projection_state_key()
        previous_state = storage.source_state_key
        if current_state == previous_state:
            return previous
        current_fence, current_batch, current_forget = _source_components(
            current_state)
        previous_fence, previous_batch, previous_forget = _source_components(
            previous_state)
        if (current_fence <= previous_fence
                or current_forget != previous_forget
                or not _batch_transition_is_append_only(
                    previous_batch, current_batch)
                or not event_log.physical_tail_is_fully_visible(
                    previous_fence,
                    page_limit=policy.event_page_limit,
                )):
            raise MemoryQueryIndexRebuildRequired(
                "source tail 不是完整可见且无 rollback/forget 的严格追加")

        changed_hashes = self._changed_hypothesis_hashes(
            previous_fence, policy)
        partitions = {
            (item.hypothesis_kind_hash, item.owner_key): item
            for item in previous.partitions
        }
        accepted_by_hash: dict[int, set[tuple[int, ...]]] = {}
        for kind in storage.hypothesis_kinds:
            kind_hash = self.resolver.aggregates.hypothesis_kind_hash(kind)
            accepted_by_hash.setdefault(kind_hash, set()).add(kind)
        runs = previous.query_runs()
        changes: list[MemoryQueryIndexChange] = []
        records: list[SegmentRecord] = []
        generation_key = _framed_key(
            publication.publication_key, current_state, planner_key)
        for hypothesis_hash in changed_hashes:
            aggregate = self.resolver.aggregates.store.read_aggregate(
                hypothesis_hash)
            if aggregate is None:
                raise MemoryQueryIndexRebuildRequired(
                    "变化 Hypothesis 缺少当前 aggregate")
            accepted_kinds = accepted_by_hash.get(
                aggregate.hypothesis_kind_hash)
            if (accepted_kinds is None
                    or not storage.access.can_read(OwnerScope(
                        *aggregate.owner_key))):
                continue
            bundle = self.resolver.load_bundle(
                aggregate, access=storage.access)
            if bundle.hypothesis.hypothesis_kind not in accepted_kinds:
                continue
            partition_key = (
                aggregate.hypothesis_kind_hash, aggregate.owner_key)
            previous_active, previous_records = self._previous_candidate_state(
                hypothesis_hash,
                partition_key,
                previous_fence,
                partitions,
                runs,
            )
            active = int(aggregate.lifecycle_state == LIFECYCLE_ACTIVE)
            entries = () if not active else planner.index_entries(bundle)
            if active and (not isinstance(entries, tuple) or not entries):
                raise TypeError("query index planner entries 必须是非空 tuple")
            current_records = len(entries)
            if not previous_active and not active:
                continue
            changes.append(MemoryQueryIndexChange(
                hypothesis_hash,
                aggregate.hypothesis_kind_hash,
                aggregate.owner_key,
                active,
                current_records,
            ))
            if active:
                payload = encode_memory_candidate_payload(bundle)
                for entry in entries:
                    records.append(SegmentRecord(
                        memory_query_index_record_key(
                            generation_key,
                            aggregate,
                            entry,
                        ),
                        payload,
                    ))
            candidate_delta = active - previous_active
            record_delta = current_records - previous_records
            current_partition = partitions.get(partition_key)
            if current_partition is None:
                if candidate_delta <= 0 or record_delta <= 0:
                    raise MemoryQueryIndexMaintenanceError(
                        "新分区没有正候选增量")
                partitions[partition_key] = MemoryQueryIndexPartition(
                    partition_key[0],
                    partition_key[1],
                    candidate_delta,
                    record_delta,
                )
            else:
                next_candidates = (
                    current_partition.candidate_count + candidate_delta)
                next_records = (
                    current_partition.index_record_count + record_delta)
                if (next_candidates < 0
                        or next_records < 0
                        or next_records < next_candidates):
                    raise MemoryQueryIndexMaintenanceError(
                        "增量维护使分区计数越界")
                if next_candidates == 0:
                    if next_records != 0:
                        raise MemoryQueryIndexMaintenanceError(
                            "空分区仍残留查询索引记录")
                    del partitions[partition_key]
                else:
                    partitions[partition_key] = MemoryQueryIndexPartition(
                        current_partition.hypothesis_kind_hash,
                        current_partition.owner_key,
                        next_candidates,
                        next_records,
                    )

        records = sorted(records, key=lambda item: item.record_key)
        if len({item.record_key for item in records}) != len(records):
            raise MemoryQueryIndexMaintenanceError(
                "增量 planner 生成了重复 canonical record key")
        attempted: list[tuple[int, ...]] = []
        summaries: tuple[MemoryProjectionSegment, ...] = ()
        current_manifest = self.store.current_manifest()
        preexisting = frozenset(
            () if current_manifest is None else (
                item.segment_key for item in current_manifest.entries))
        try:
            summaries = self._publish_records(
                tuple(records),
                generation_key,
                current_fence,
                publication,
                attempted,
                preexisting,
            )
            self.resolver.aggregates.require_clean(access=storage.access)
            if event_log.projection_state_key() != current_state:
                raise MemoryQueryIndexMaintenanceError(
                    "增量发布期间 Memory source state 已变化")
            current = self.store.current_manifest()
            publish_epoch = 0 if current is None else current.publish_epoch
            if changes:
                run_storage = MemoryCandidateProjectionManifest(
                    generation_key,
                    storage.memory_space,
                    storage.memory_space_id,
                    storage.access,
                    storage.hypothesis_kinds,
                    current_fence,
                    current_state,
                    publication.version_key,
                    publication.dependencies,
                    summaries,
                    publish_epoch,
                )
                runs = (*runs, MemoryQueryIndexRun(
                    run_storage, tuple(changes)))
            metadata_key = _framed_key(
                (*publication.publication_key, 0),
                current_state,
                planner_key,
            )
            metadata = MemoryCandidateProjectionManifest(
                metadata_key,
                storage.memory_space,
                storage.memory_space_id,
                storage.access,
                storage.hypothesis_kinds,
                current_fence,
                current_state,
                publication.version_key,
                publication.dependencies,
                (),
                publish_epoch,
            )
            result = MemoryQueryIndexProjectionManifest(
                metadata,
                planner_key,
                tuple(partitions.values()),
                runs,
            )
            result.validate_store(self.store)
            if len(runs) > policy.max_run_count:
                return self.compact(
                    result,
                    publication=publication,
                    policy=policy,
                )
            return result
        except Exception:
            self._cleanup_new_segments(tuple(attempted), generation_key)
            raise

    def build_initial(
            self,
            *,
            access: MemoryAccessContext,
            hypothesis_kinds: tuple[tuple[int, ...], ...],
            publication: MemoryProjectionPublication,
            policy: MemoryQueryIndexMaintenancePolicy,
            ) -> MemoryQueryIndexProjectionManifest:
        """从真实 aggregate 流构建有界临时 run，再压成单一初始索引。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("query index initial access 类型错误")
        if not isinstance(publication, MemoryProjectionPublication):
            raise TypeError("query index initial publication 类型错误")
        if not isinstance(policy, MemoryQueryIndexMaintenancePolicy):
            raise TypeError("query index initial policy 类型错误")
        kinds = tuple(sorted(
            strict_integer_tuple(
                item, label="query index initial hypothesis kind")
            for item in hypothesis_kinds
        ))
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("query index initial kinds 必须非空且唯一")
        planner = self.resolver.score_provider
        if not isinstance(planner, ExactMemoryProjectionPlanner):
            raise TypeError("query index initial scorer 未实现精确 planner")
        planner_key = strict_integer_tuple(
            planner.state_key(), label="query index initial planner state")
        event_log = self.resolver.aggregates.event_log
        self.resolver.aggregates.require_clean(access=access)
        source_state = event_log.projection_state_key()
        source_fence = source_state[0]
        master_generation = _framed_key(
            (*publication.publication_key, 3), source_state, planner_key)
        kind_by_hash: dict[int, set[tuple[int, ...]]] = {}
        for kind in kinds:
            kind_hash = self.resolver.aggregates.hypothesis_kind_hash(kind)
            kind_by_hash.setdefault(kind_hash, set()).add(kind)
        partitions: dict[
            tuple[int, tuple[int, int, int, int]], tuple[int, int]
        ] = {}
        runs: list[MemoryQueryIndexRun] = []
        chunk: list[tuple[
            MemoryCandidateBundle,
            tuple[MemoryProjectionIndexEntry, ...],
            tuple[int, ...],
        ]] = []
        attempted: list[tuple[int, ...]] = []
        current_manifest = self.store.current_manifest()
        preexisting = frozenset(
            () if current_manifest is None else (
                item.segment_key for item in current_manifest.entries))

        def flush_chunk() -> None:
            """把一个候选页编码为独立排序 run，并清空页内引用。"""
            if not chunk:
                return
            ordinal = len(runs) + 1
            generation_key = _framed_key(
                (*publication.publication_key, 3, ordinal),
                source_state,
                planner_key,
            )
            records: list[SegmentRecord] = []
            changes: list[MemoryQueryIndexChange] = []
            for bundle, entries, payload in chunk:
                aggregate = bundle.aggregate
                for entry in entries:
                    records.append(SegmentRecord(
                        memory_query_index_record_key(
                            generation_key, aggregate, entry),
                        payload,
                    ))
                changes.append(MemoryQueryIndexChange(
                    aggregate.hypothesis_hash,
                    aggregate.hypothesis_kind_hash,
                    aggregate.owner_key,
                    1,
                    len(entries),
                ))
            records.sort(key=lambda item: item.record_key)
            if len({item.record_key for item in records}) != len(records):
                raise MemoryQueryIndexMaintenanceError(
                    "初始 query-index planner 生成重复 record key")
            summaries = self._publish_records(
                tuple(records),
                generation_key,
                source_fence,
                publication,
                attempted,
                preexisting,
            )
            current = self.store.current_manifest()
            run_storage = MemoryCandidateProjectionManifest(
                generation_key,
                event_log.memory_space_identity,
                event_log.memory_space_id,
                access,
                kinds,
                source_fence,
                source_state,
                publication.version_key,
                publication.dependencies,
                summaries,
                0 if current is None else current.publish_epoch,
            )
            runs.append(MemoryQueryIndexRun(
                run_storage, tuple(changes)))
            chunk.clear()

        try:
            for kind_hash in sorted(kind_by_hash):
                accepted_kinds = kind_by_hash[kind_hash]
                for owner_key in visible_owner_keys(access):
                    partition_key = (kind_hash, owner_key)
                    candidate_count = 0
                    record_count = 0
                    for aggregate in (
                            self.resolver.aggregates.store
                            .iter_aggregates_by_kind_owner(
                                kind_hash,
                                owner_key,
                                page_limit=publication.backend_page_limit,
                            )):
                        if aggregate.lifecycle_state != LIFECYCLE_ACTIVE:
                            continue
                        bundle = self.resolver.load_bundle(
                            aggregate, access=access)
                        if bundle.hypothesis.hypothesis_kind not in accepted_kinds:
                            continue
                        entries = planner.index_entries(bundle)
                        if not isinstance(entries, tuple) or not entries:
                            raise TypeError(
                                "query index planner entries 必须是非空 tuple")
                        payload = encode_memory_candidate_payload(bundle)
                        chunk.append((bundle, entries, payload))
                        candidate_count += 1
                        record_count += len(entries)
                        if len(chunk) >= policy.compaction_page_limit:
                            flush_chunk()
                    if candidate_count:
                        partitions[partition_key] = (
                            candidate_count, record_count)
            flush_chunk()
            self.resolver.aggregates.require_clean(access=access)
            if event_log.projection_state_key() != source_state:
                raise MemoryQueryIndexMaintenanceError(
                    "初始索引构建期间 Memory source state 已变化")
            if not runs:
                empty_storage = MemoryCandidateProjectionManifest(
                    master_generation,
                    event_log.memory_space_identity,
                    event_log.memory_space_id,
                    access,
                    kinds,
                    source_fence,
                    source_state,
                    publication.version_key,
                    publication.dependencies,
                    (),
                    0 if current_manifest is None
                    else current_manifest.publish_epoch,
                )
                runs.append(MemoryQueryIndexRun(empty_storage, ()))
            current = self.store.current_manifest()
            metadata = MemoryCandidateProjectionManifest(
                master_generation,
                event_log.memory_space_identity,
                event_log.memory_space_id,
                access,
                kinds,
                source_fence,
                source_state,
                publication.version_key,
                publication.dependencies,
                (),
                0 if current is None else current.publish_epoch,
            )
            initial = MemoryQueryIndexProjectionManifest(
                metadata,
                planner_key,
                tuple(
                    MemoryQueryIndexPartition(
                        key[0], key[1], values[0], values[1])
                    for key, values in partitions.items()
                ),
                tuple(runs),
            )
            initial.validate_store(self.store)
            if len(runs) > 1:
                return self.compact(
                    initial,
                    publication=publication,
                    policy=policy,
                )
            return initial
        except Exception:
            self._cleanup_new_segments(
                tuple(attempted), master_generation)
            raise

    def compact(
            self,
            previous: MemoryQueryIndexProjectionManifest,
            *,
            publication: MemoryProjectionPublication,
            policy: MemoryQueryIndexMaintenancePolicy,
            ) -> MemoryQueryIndexProjectionManifest:
        """合并多个 query-index run，按当前 source state 释放旧物理段。"""
        if not isinstance(previous, MemoryQueryIndexProjectionManifest):
            raise TypeError("query index compaction previous manifest 类型错误")
        if not isinstance(publication, MemoryProjectionPublication):
            raise TypeError("query index compaction publication 类型错误")
        if not isinstance(policy, MemoryQueryIndexMaintenancePolicy):
            raise TypeError("query index compaction policy 类型错误")
        previous.validate_store(self.store)
        planner = self.resolver.score_provider
        if not isinstance(planner, ExactMemoryProjectionPlanner):
            raise TypeError("query index compactor scorer 未实现精确 planner")
        planner_key = strict_integer_tuple(
            planner.state_key(), label="query index compaction planner state")
        if planner_key != previous.planner_key:
            raise MemoryQueryIndexRebuildRequired("planner state 已变化")
        runs = previous.query_runs()
        if len(runs) <= 1:
            return previous
        storage = previous.storage
        event_log = self.resolver.aggregates.event_log
        if (storage.memory_space != event_log.memory_space_identity
                or storage.memory_space_id != event_log.memory_space_id):
            raise ValueError("query index compactor Memory 空间漂移")
        self.resolver.aggregates.require_clean(access=storage.access)
        current_state = event_log.projection_state_key()
        if current_state != storage.source_state_key:
            raise MemoryQueryIndexRebuildRequired(
                "压实前 source state 已变化，必须先维护增量")
        current_fence = current_state[0]
        generation_key = _framed_key(
            (*publication.publication_key, 1), current_state, planner_key)
        top_storage_key = _framed_key(
            (*publication.publication_key, 2), current_state, planner_key)
        later_invalidated: list[frozenset[int]] = [frozenset()] * len(runs)
        changed_after: set[int] = set()
        for run_index in range(len(runs) - 1, -1, -1):
            later_invalidated[run_index] = frozenset(changed_after)
            changed_after.update(runs[run_index].invalidated_hypothesis_hashes)

        attempted: list[tuple[int, ...]] = []
        old_segment_keys = tuple(
            segment.segment_key
            for run in runs
            for segment in run.storage.segments
        )
        read_budget = SegmentBudget(
            policy.compaction_page_limit,
            max(publication.segment_budget.byte_limit, 1_000_000),
        )
        current_manifest = self.store.current_manifest()
        preexisting = frozenset(
            () if current_manifest is None else (
                item.segment_key for item in current_manifest.entries))
        try:
            merged = heapq.merge(*tuple(
                self._iter_compaction_run_records(
                    run,
                    later_invalidated[index],
                    planner,
                    generation_key,
                    _object_key(
                        generation_key,
                        MEMORY_QUERY_INDEX_MAINTENANCE_READER_OBJECT,
                        index + 1,
                    ),
                    read_budget,
                )
                for index, run in enumerate(runs)
            ), key=lambda item: item.record_key)
            candidate_states: dict[
                int, tuple[int, tuple[int, int, int, int], int]
            ] = {}
            previous_key: tuple[int, ...] | None = None

            def counted_records() -> Iterator[SegmentRecord]:
                """校验压实输入严格递增并统计每个候选的索引记录数。"""
                nonlocal previous_key
                for record in merged:
                    if previous_key is not None and record.record_key <= previous_key:
                        raise MemoryQueryIndexMaintenanceError(
                            "压实输入 query-index record 未严格递增")
                    previous_key = record.record_key
                    bundle = decode_memory_candidate_payload(record)
                    aggregate = bundle.aggregate
                    if aggregate.lifecycle_state != LIFECYCLE_ACTIVE:
                        raise MemoryQueryIndexMaintenanceError(
                            "压实输入含 inactive query-index candidate")
                    existing = candidate_states.get(aggregate.hypothesis_hash)
                    state = (
                        aggregate.hypothesis_kind_hash,
                        aggregate.owner_key,
                        1 if existing is None else existing[2] + 1,
                    )
                    if existing is not None and existing[:2] != state[:2]:
                        raise MemoryQueryIndexMaintenanceError(
                            "压实候选在 run 间发生 kind/owner 漂移")
                    candidate_states[aggregate.hypothesis_hash] = state
                    yield record

            summaries = self._publish_record_stream(
                counted_records(),
                generation_key,
                current_fence,
                publication,
                attempted,
                preexisting,
            )
            if len(candidate_states) != previous.candidate_count:
                raise MemoryQueryIndexMaintenanceError(
                    "压实候选数与逻辑分区不闭合")
            if sum(item[2] for item in candidate_states.values()) != sum(
                    item.index_record_count for item in previous.partitions):
                raise MemoryQueryIndexMaintenanceError(
                    "压实记录数与逻辑分区不闭合")
            self.resolver.aggregates.require_clean(access=storage.access)
            if event_log.projection_state_key() != current_state:
                raise MemoryQueryIndexMaintenanceError(
                    "压实发布期间 Memory source state 已变化")
            current = self.store.current_manifest()
            publish_epoch = 0 if current is None else current.publish_epoch
            compacted_storage = MemoryCandidateProjectionManifest(
                generation_key,
                storage.memory_space,
                storage.memory_space_id,
                storage.access,
                storage.hypothesis_kinds,
                current_fence,
                current_state,
                publication.version_key,
                publication.dependencies,
                summaries,
                publish_epoch,
            )
            changes = tuple(
                MemoryQueryIndexChange(
                    hypothesis_hash,
                    values[0],
                    values[1],
                    1,
                    values[2],
                )
                for hypothesis_hash, values in sorted(candidate_states.items())
            )
            compacted_run = MemoryQueryIndexRun(compacted_storage, changes)
            metadata = MemoryCandidateProjectionManifest(
                top_storage_key,
                storage.memory_space,
                storage.memory_space_id,
                storage.access,
                storage.hypothesis_kinds,
                current_fence,
                current_state,
                publication.version_key,
                publication.dependencies,
                (),
                publish_epoch,
            )
            result = MemoryQueryIndexProjectionManifest(
                metadata,
                planner_key,
                previous.partitions,
                (compacted_run,),
            )
            result.validate_store(self.store)
            if old_segment_keys:
                self.store.release_rebuildable_segments(
                    old_segment_keys,
                    release_key=_object_key(
                        generation_key,
                        MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_OBJECT,
                        1,
                    ),
                    manifest_key=_object_key(
                        generation_key,
                        MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_MANIFEST_OBJECT,
                        1,
                    ),
                )
            return result
        except Exception:
            self._cleanup_new_segments(tuple(attempted), generation_key)
            raise

    def _iter_compaction_run_records(
            self,
            run: MemoryQueryIndexRun,
            invalidated: frozenset[int],
            planner: ExactMemoryProjectionPlanner,
            generation_key: tuple[int, ...],
            reader_key: tuple[int, ...],
            budget: SegmentBudget,
            ) -> Iterator[SegmentRecord]:
        """读取单个 run 的段并过滤其后已失效候选，保持键序输出。"""
        reader = self.store.open_reader(
            reader_key,
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        )
        validated: dict[
            int, dict[tuple[int, ...], tuple[int, ...]]
        ] = {}
        try:
            for summary in run.storage.segments:
                continuation = None
                read_count = 0
                while True:
                    page = reader.page(
                        budget=budget,
                        lower_key=summary.lower_key,
                        upper_key=summary.upper_key,
                        continuation=continuation,
                    )
                    for record in page.records:
                        read_count += 1
                        bundle = decode_memory_candidate_payload(record)
                        aggregate = bundle.aggregate
                        hypothesis_hash = aggregate.hypothesis_hash
                        if hypothesis_hash in invalidated:
                            continue
                        mapping = validated.get(hypothesis_hash)
                        if mapping is None:
                            mapping = {
                                memory_query_index_record_key(
                                    run.storage.projection_key,
                                    aggregate,
                                    entry,
                                ): memory_query_index_record_key(
                                    generation_key,
                                    aggregate,
                                    entry,
                                )
                                for entry in planner.index_entries(bundle)
                            }
                            validated[hypothesis_hash] = mapping
                        rewritten_key = mapping.get(record.record_key)
                        if rewritten_key is None:
                            raise MemoryQueryIndexMaintenanceError(
                                "压实输入 record 未绑定当前 planner entry")
                        yield SegmentRecord(rewritten_key, record.payload)
                    continuation = page.continuation
                    if continuation is None:
                        break
                if read_count != summary.record_count:
                    raise MemoryQueryIndexMaintenanceError(
                        "压实读取 segment 记录数与摘要不闭合")
        finally:
            reader.close()

    def _changed_hypothesis_hashes(
            self,
            previous_fence: int,
            policy: MemoryQueryIndexMaintenancePolicy,
            ) -> tuple[int, ...]:
        """分页读取 source tail 的 Hypothesis 反向索引并有界去重。"""
        after = previous_fence
        changed: set[int] = set()
        while True:
            rows = self.resolver.aggregates.event_log.backend.select(
                MEMORY_HYPOTHESIS_EVENT_TABLE,
                {"space_id": self.resolver.aggregates.event_log.memory_space_id},
                where_gt={"event_seq": after},
                order_by="event_seq",
                limit=policy.event_page_limit,
            )
            if not rows:
                break
            for row in rows:
                event_seq = row["event_seq"]
                hypothesis_hash = row["hypothesis_hash"]
                if (type(event_seq) is not int or event_seq <= after
                        or type(hypothesis_hash) is not int
                        or hypothesis_hash <= 0):
                    raise MemoryQueryIndexMaintenanceError(
                        "Hypothesis event tail 顺序或身份损坏")
                after = event_seq
                changed.add(hypothesis_hash)
                if len(changed) > policy.changed_candidate_limit:
                    raise MemoryQueryIndexRebuildRequired(
                        "变化候选数超过增量维护硬上限")
            if len(rows) < policy.event_page_limit:
                break
        return tuple(sorted(changed))

    def _previous_candidate_state(
            self,
            hypothesis_hash: int,
            partition_key: tuple[int, tuple[int, int, int, int]],
            previous_fence: int,
            partitions: dict[
                tuple[int, tuple[int, int, int, int]],
                MemoryQueryIndexPartition,
            ],
            runs: tuple[MemoryQueryIndexRun, ...],
            ) -> tuple[int, int]:
        """由最近变化状态或基线事件水位恢复候选上一资格。"""
        for run in reversed(runs):
            for change in run.changes:
                if change.hypothesis_hash != hypothesis_hash:
                    continue
                if (change.hypothesis_kind_hash, change.owner_key) != partition_key:
                    raise MemoryQueryIndexMaintenanceError(
                        "候选在增量 run 间发生分区漂移")
                return change.active, change.index_record_count
        events = self.resolver.aggregates.store.list_events(hypothesis_hash)
        if not events:
            raise MemoryQueryIndexMaintenanceError(
                "变化候选缺少 Hypothesis 事件反向索引")
        if min(item.event_seq for item in events) > previous_fence:
            return 0, 0
        partition = partitions.get(partition_key)
        if partition is None:
            return 0, 0
        if partition.index_record_count % partition.candidate_count:
            raise MemoryQueryIndexRebuildRequired(
                "基线分区不是固定每候选索引记录数")
        return 1, partition.index_record_count // partition.candidate_count

    def _publish_record_stream(
            self,
            records: Iterable[SegmentRecord],
            generation_key: tuple[int, ...],
            source_fence: int,
            publication: MemoryProjectionPublication,
            attempted: list[tuple[int, ...]],
            preexisting: frozenset[tuple[int, ...]],
            ) -> tuple[MemoryProjectionSegment, ...]:
        """把已严格排序的压实流按预算封段，避免重新累积全部 payload。"""
        summaries: list[MemoryProjectionSegment] = []
        delta = OpenHotDelta(
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            publication.version_key,
            publication.dependencies,
            publication.segment_budget,
        )
        previous_key: tuple[int, ...] | None = None

        def flush() -> None:
            nonlocal delta
            if delta.object_count == 0:
                return
            ordinal = len(summaries) + 1
            segment_key = _object_key(
                generation_key,
                MEMORY_QUERY_INDEX_MAINTENANCE_SEGMENT_OBJECT,
                ordinal,
            )
            segment = delta.seal(segment_key, source_fence)
            if segment_key not in preexisting:
                attempted.append(segment_key)
            self.store.publish_segment(
                segment,
                tier_key=publication.tier_key,
                manifest_key=_object_key(
                    generation_key,
                    MEMORY_QUERY_INDEX_MAINTENANCE_MANIFEST_OBJECT,
                    ordinal,
                ),
                migration_key=_object_key(
                    generation_key,
                    MEMORY_QUERY_INDEX_MAINTENANCE_MIGRATION_OBJECT,
                    ordinal,
                ),
            )
            delta.acknowledge(segment)
            summaries.append(MemoryProjectionSegment(
                segment.segment_key,
                segment.lower_key,
                segment.upper_key,
                segment.checksum_key,
                len(segment.records),
                segment.size_bytes,
            ))
            delta = OpenHotDelta(
                MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
                publication.version_key,
                publication.dependencies,
                publication.segment_budget,
            )

        for record in records:
            if previous_key is not None and record.record_key <= previous_key:
                raise MemoryQueryIndexMaintenanceError(
                    "压实输出 record 未严格递增")
            previous_key = record.record_key
            try:
                delta.append(record)
            except SegmentBudgetExceeded:
                flush()
                delta.append(record)
        flush()
        return tuple(summaries)

    def _publish_records(
            self,
            records: tuple[SegmentRecord, ...],
            generation_key: tuple[int, ...],
            source_fence: int,
            publication: MemoryProjectionPublication,
            attempted: list[tuple[int, ...]],
            preexisting: frozenset[tuple[int, ...]],
            ) -> tuple[MemoryProjectionSegment, ...]:
        """按 canonical 顺序把有界变化记录封为不重叠 segment。"""
        if not records:
            return ()
        summaries: list[MemoryProjectionSegment] = []
        delta = OpenHotDelta(
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            publication.version_key,
            publication.dependencies,
            publication.segment_budget,
        )

        def flush() -> None:
            nonlocal delta
            if delta.object_count == 0:
                return
            ordinal = len(summaries) + 1
            segment_key = _object_key(
                generation_key,
                MEMORY_QUERY_INDEX_MAINTENANCE_SEGMENT_OBJECT,
                ordinal,
            )
            segment = delta.seal(segment_key, source_fence)
            if segment_key not in preexisting:
                attempted.append(segment_key)
            self.store.publish_segment(
                segment,
                tier_key=publication.tier_key,
                manifest_key=_object_key(
                    generation_key,
                    MEMORY_QUERY_INDEX_MAINTENANCE_MANIFEST_OBJECT,
                    ordinal,
                ),
                migration_key=_object_key(
                    generation_key,
                    MEMORY_QUERY_INDEX_MAINTENANCE_MIGRATION_OBJECT,
                    ordinal,
                ),
            )
            delta.acknowledge(segment)
            summaries.append(MemoryProjectionSegment(
                segment.segment_key,
                segment.lower_key,
                segment.upper_key,
                segment.checksum_key,
                len(segment.records),
                segment.size_bytes,
            ))
            delta = OpenHotDelta(
                MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
                publication.version_key,
                publication.dependencies,
                publication.segment_budget,
            )

        for record in records:
            try:
                delta.append(record)
            except SegmentBudgetExceeded:
                flush()
                delta.append(record)
        flush()
        return tuple(summaries)

    def _cleanup_new_segments(
            self,
            segment_keys: tuple[tuple[int, ...], ...],
            generation_key: tuple[int, ...],
            ) -> None:
        """失败时只释放本次已经进入 location manifest 的新段。"""
        if not segment_keys:
            return
        self.store.recover_pending_operations()
        current = self.store.current_manifest()
        if current is None:
            return
        current_keys = {item.segment_key for item in current.entries}
        selected = tuple(key for key in segment_keys if key in current_keys)
        if not selected:
            return
        self.store.release_rebuildable_segments(
            selected,
            release_key=_object_key(
                generation_key,
                MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_OBJECT,
                1,
            ),
            manifest_key=_object_key(
                generation_key,
                MEMORY_QUERY_INDEX_MAINTENANCE_RELEASE_MANIFEST_OBJECT,
                1,
            ),
        )


__all__ = [
    "MemoryQueryIndexMaintainer",
    "MemoryQueryIndexMaintenanceError",
    "MemoryQueryIndexMaintenancePolicy",
    "MemoryQueryIndexRebuildRequired",
]
