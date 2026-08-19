"""K-04 Memory 候选投影发布和 query-scoped 有界 resolver 接线。"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.memory_hot_set import (
    BoundedCandidateSelectionPolicy,
    ExactMemoryProjectionPlanner,
    MemoryProjectionQueryGroup,
    decode_memory_candidate,
    decode_memory_candidate_payload,
    encode_memory_candidate,
    matches_memory_filter,
    memory_candidate_record_key,
    memory_candidate_scan_range,
    memory_query_index_record_key,
    memory_query_index_scan_range,
    visible_owner_keys,
)
from pure_integer_ai.cognition.shared.memory_event import (
    LIFECYCLE_ACTIVE,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.identity import OwnerScope
from pure_integer_ai.cognition.shared.memory_maintenance import (
    MemoryMaintenanceAssessment,
    MemoryPlacementHint,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryActivationRequest,
    MemoryQueryCompilation,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryAggregateFilter,
    MemoryResolution,
    RESOLUTION_ORIGIN_MEMORY,
    ResolvedCandidateSet,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.edge_budget import (
    EdgeBudgetProfile,
    EdgeBudgetReport,
    EdgeMetricObservation,
)
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR,
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentBudgetExceeded,
)
from pure_integer_ai.storage.segment_dependency import (
    SegmentDependency,
    canonical_dependencies,
)
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetMetrics,
    QueryHotSetPolicy,
    QuerySegmentHotSet,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.storage.spaces.registry import SpaceIdentity
from pure_integer_ai.experiments.train_context import TrainContext


MEMORY_PROJECTION_MANIFEST_VERSION = 1
MEMORY_PROJECTION_OBJECT_KEY_VERSION = 1
MEMORY_PROJECTION_SEGMENT_OBJECT = 1
MEMORY_PROJECTION_LOCATION_MANIFEST_OBJECT = 2
MEMORY_PROJECTION_MIGRATION_OBJECT = 3
MEMORY_PROJECTION_CLEANUP_RELEASE_OBJECT = 4
MEMORY_PROJECTION_CLEANUP_MANIFEST_OBJECT = 5
MEMORY_PROJECTION_PLACEMENT_MANIFEST_OBJECT = 6
MEMORY_PROJECTION_PLACEMENT_MIGRATION_OBJECT = 7
MEMORY_PROJECTION_PLACEMENT_READER_OBJECT = 8
MEMORY_PROJECTION_GENERATION_KEY_VERSION = 1
MEMORY_PROJECTION_PLACEMENT_PLAN_VERSION = 1
MEMORY_QUERY_INDEX_MANIFEST_VERSION = 2
MEMORY_QUERY_INDEX_LEGACY_MANIFEST_VERSION = 1


class MemoryProjectionError(RuntimeError):
    """Memory 候选投影发布、恢复或 source fence 校验失败。"""


@dataclass(frozen=True, order=True)
class MemoryProjectionSegment:
    """候选投影中一个已发布 segment 的稳定校验摘要。"""

    segment_key: tuple[int, ...]
    lower_key: tuple[int, ...]
    upper_key: tuple[int, ...]
    checksum_key: tuple[int, ...]
    record_count: int
    size_bytes: int

    def __post_init__(self) -> None:
        """核验 segment 完整身份、闭区间、校验和和物理规模。"""
        for label, value in (
                ("segment_key", self.segment_key),
                ("lower_key", self.lower_key),
                ("upper_key", self.upper_key),
                ("checksum_key", self.checksum_key)):
            strict_integer_tuple(value, label=f"memory projection {label}")
        if self.lower_key > self.upper_key:
            raise ValueError("memory projection segment range 不能反向")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise ValueError("memory projection record_count 必须是正严格整数")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("memory projection size_bytes 必须是正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 segment 摘要的完整稳定键。"""
        result: list[int] = []
        for value in (
                self.segment_key,
                self.lower_key,
                self.upper_key,
                self.checksum_key):
            pack_key(result, value)
        result.extend((self.record_count, self.size_bytes))
        return tuple(result)


@dataclass(frozen=True)
class MemoryCandidateProjectionManifest:
    """绑定 ACL、Memory 空间、源 fence、依赖和全部候选段的冻结投影。"""

    projection_key: tuple[int, ...]
    memory_space: SpaceIdentity
    memory_space_id: int
    access: MemoryAccessContext
    hypothesis_kinds: tuple[tuple[int, ...], ...]
    source_fence: int
    source_state_key: tuple[int, ...]
    version_key: tuple[int, ...]
    dependencies: tuple[SegmentDependency, ...]
    segments: tuple[MemoryProjectionSegment, ...]
    publish_epoch: int

    def __post_init__(self) -> None:
        """核验投影身份、ACL、依赖闭合、段不重叠和发布 epoch。"""
        strict_integer_tuple(
            self.projection_key, label="memory projection projection_key")
        if not isinstance(self.memory_space, SpaceIdentity):
            raise TypeError("memory projection memory_space 类型错误")
        if type(self.memory_space_id) is not int or self.memory_space_id <= 0:
            raise ValueError("memory projection memory_space_id 必须是正严格整数")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("memory projection access 类型错误")
        if (not isinstance(self.hypothesis_kinds, tuple)
                or not self.hypothesis_kinds):
            raise ValueError("memory projection hypothesis_kinds 不得为空")
        kinds = tuple(sorted(
            strict_integer_tuple(item, label="memory projection hypothesis_kind")
            for item in self.hypothesis_kinds
        ))
        if len(set(kinds)) != len(kinds):
            raise ValueError("memory projection hypothesis_kinds 不得重复")
        object.__setattr__(self, "hypothesis_kinds", kinds)
        if type(self.source_fence) is not int or self.source_fence < 0:
            raise ValueError("memory projection source_fence 必须是非负严格整数")
        source_state = strict_integer_tuple(
            self.source_state_key,
            label="memory projection source_state_key",
        )
        if not source_state:
            raise ValueError("memory projection source_state_key 不得为空")
        if source_state[0] != self.source_fence:
            raise ValueError("memory projection source_state_key 与 fence 漂移")
        strict_integer_tuple(
            self.version_key, label="memory projection version_key")
        dependencies = canonical_dependencies(self.dependencies)
        if tuple(item.descriptor_key for item in dependencies) != (
                MEMORY_QUERY_PROJECTION_DESCRIPTOR.dependency_keys):
            raise ValueError("memory projection dependencies 与存储角色漂移")
        object.__setattr__(self, "dependencies", dependencies)
        if (not isinstance(self.segments, tuple)
                or any(not isinstance(item, MemoryProjectionSegment)
                       for item in self.segments)):
            raise TypeError("memory projection segments 类型错误")
        segments = tuple(sorted(
            self.segments, key=lambda item: (item.lower_key, item.segment_key)))
        if len({item.segment_key for item in segments}) != len(segments):
            raise ValueError("memory projection segment_key 不得重复")
        for previous, current in zip(segments, segments[1:]):
            if previous.upper_key >= current.lower_key:
                raise ValueError("memory projection segment range 重叠")
        object.__setattr__(self, "segments", segments)
        if type(self.publish_epoch) is not int or self.publish_epoch < 0:
            raise ValueError("memory projection publish_epoch 必须是非负严格整数")
        if segments and self.publish_epoch <= 0:
            raise ValueError("非空 memory projection 必须绑定正发布 epoch")

    @property
    def record_count(self) -> int:
        """返回投影全部 segment 的候选记录数。"""
        return sum(item.record_count for item in self.segments)

    def stable_key(self) -> tuple[int, ...]:
        """返回可跨重启保存和严格恢复的完整投影 manifest。"""
        result = [MEMORY_PROJECTION_MANIFEST_VERSION]
        pack_key(result, self.projection_key)
        pack_key(result, self.memory_space.stable_key())
        result.append(self.memory_space_id)
        pack_key(result, self.access.stable_key())
        result.append(len(self.hypothesis_kinds))
        for kind in self.hypothesis_kinds:
            pack_key(result, kind)
        result.append(self.source_fence)
        pack_key(result, self.source_state_key)
        pack_key(result, self.version_key)
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            pack_key(result, dependency.descriptor_key)
            pack_key(result, dependency.version_key)
            pack_key(result, dependency.checksum_key)
        result.append(len(self.segments))
        for segment in self.segments:
            pack_key(result, segment.stable_key())
        result.append(self.publish_epoch)
        return tuple(result)

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "MemoryCandidateProjectionManifest":
        """从持久化纯整数键恢复投影 manifest，拒绝截断和未知版本。"""
        reader = IntegerStreamReader(key)
        version = reader.read_positive(label="memory projection manifest version")
        if version != MEMORY_PROJECTION_MANIFEST_VERSION:
            raise MemoryProjectionError("memory projection manifest 版本未注册")
        projection_key = reader.read_key(label="memory projection key")
        memory_space = SpaceIdentity(*reader.read_key(
            label="memory projection memory_space"))
        memory_space_id = reader.read_positive(
            label="memory projection memory_space_id")
        access = MemoryAccessContext(*reader.read_key(
            label="memory projection access"))
        kind_count = reader.read_positive(label="memory projection kind_count")
        kinds = tuple(reader.read_key(
            label=f"memory projection kind[{index}]")
            for index in range(kind_count))
        source_fence = reader.read_nonnegative(
            label="memory projection source_fence")
        source_state_key = reader.read_key(
            label="memory projection source_state_key")
        version_key = reader.read_key(label="memory projection version_key")
        dependency_count = reader.read_positive(
            label="memory projection dependency_count")
        dependencies = tuple(SegmentDependency(
            reader.read_key(label=f"memory projection dependency[{index}].descriptor"),
            reader.read_key(label=f"memory projection dependency[{index}].version"),
            reader.read_key(label=f"memory projection dependency[{index}].checksum"),
        ) for index in range(dependency_count))
        segment_count = reader.read_nonnegative(
            label="memory projection segment_count")
        segments = []
        for index in range(segment_count):
            segment_reader = IntegerStreamReader(reader.read_key(
                label=f"memory projection segment[{index}]"))
            segments.append(MemoryProjectionSegment(
                segment_reader.read_key(label="projection segment key"),
                segment_reader.read_key(label="projection segment lower"),
                segment_reader.read_key(label="projection segment upper"),
                segment_reader.read_key(label="projection segment checksum"),
                segment_reader.read_positive(label="projection segment records"),
                segment_reader.read_positive(label="projection segment bytes"),
            ))
            segment_reader.finish()
        publish_epoch = reader.read_nonnegative(
            label="memory projection publish_epoch")
        reader.finish()
        return cls(
            projection_key,
            memory_space,
            memory_space_id,
            access,
            kinds,
            source_fence,
            source_state_key,
            version_key,
            dependencies,
            tuple(segments),
            publish_epoch,
        )

    def validate_store(self, store: TieredSegmentStore) -> None:
        """核验当前 location manifest 仍完整包含本投影的全部 segment。"""
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("memory projection store 类型错误")
        current = store.current_manifest()
        if not self.segments:
            return
        if current is None or current.publish_epoch < self.publish_epoch:
            raise MemoryProjectionError("location manifest 早于候选投影发布 epoch")
        entries = {item.segment_key: item for item in current.entries}
        for summary in self.segments:
            entry = entries.get(summary.segment_key)
            if entry is None:
                raise MemoryProjectionError("location manifest 缺少候选投影 segment")
            if (entry.descriptor_key != MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
                    or entry.version_key != self.version_key
                    or entry.checksum_key != summary.checksum_key
                    or entry.dependencies != self.dependencies
                    or entry.read_fence != self.source_fence
                    or entry.key_range.lower_key != summary.lower_key
                    or entry.key_range.upper_key != summary.upper_key):
                raise MemoryProjectionError("候选投影 segment 与 location entry 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, order=True, slots=True)
class MemoryQueryIndexPartition:
    """精确查询索引中一个 kind/owner 分区的逻辑候选数。"""

    hypothesis_kind_hash: int
    owner_key: tuple[int, int, int, int]
    candidate_count: int
    index_record_count: int

    def __post_init__(self) -> None:
        """核验派生 kind hash、完整 owner 和正候选数。"""
        if type(self.hypothesis_kind_hash) is not int or self.hypothesis_kind_hash <= 0:
            raise ValueError("query index hypothesis_kind_hash 必须是正严格整数")
        object.__setattr__(self, "owner_key", OwnerScope(*self.owner_key).stable_key())
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("query index candidate_count 必须是正严格整数")
        if (type(self.index_record_count) is not int
                or self.index_record_count < self.candidate_count):
            raise ValueError("query index record_count 不得小于逻辑候选数")

    def stable_key(self) -> tuple[int, ...]:
        """返回分区身份和逻辑候选数。"""
        return (
            self.hypothesis_kind_hash,
            *self.owner_key,
            self.candidate_count,
            self.index_record_count,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, order=True, slots=True)
class MemoryQueryIndexChange:
    """一个增量 run 对候选当前资格和索引记录数的最终裁决。"""

    hypothesis_hash: int
    hypothesis_kind_hash: int
    owner_key: tuple[int, int, int, int]
    active: int
    index_record_count: int

    def __post_init__(self) -> None:
        """核验候选身份、分区、二值资格和当前记录数。"""
        if type(self.hypothesis_hash) is not int or self.hypothesis_hash <= 0:
            raise ValueError("query index change hypothesis_hash 必须是正严格整数")
        if (type(self.hypothesis_kind_hash) is not int
                or self.hypothesis_kind_hash <= 0):
            raise ValueError("query index change kind hash 必须是正严格整数")
        object.__setattr__(
            self, "owner_key", OwnerScope(*self.owner_key).stable_key())
        if self.active not in (0, 1) or type(self.active) is not int:
            raise ValueError("query index change active 必须是严格 0/1")
        if (type(self.index_record_count) is not int
                or self.index_record_count < 0
                or bool(self.index_record_count) != bool(self.active)):
            raise ValueError("query index change 记录数与 active 不闭合")

    def stable_key(self) -> tuple[int, ...]:
        """返回变化候选的完整纯整数状态。"""
        return (
            self.hypothesis_hash,
            self.hypothesis_kind_hash,
            *self.owner_key,
            self.active,
            self.index_record_count,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MemoryQueryIndexRun:
    """一个不可变查询索引 run 及其覆盖的候选变化集合。"""

    storage: MemoryCandidateProjectionManifest
    changes: tuple[MemoryQueryIndexChange, ...]

    def __post_init__(self) -> None:
        """核验 run 存储、唯一变化身份和 source fence。"""
        if not isinstance(self.storage, MemoryCandidateProjectionManifest):
            raise TypeError("query index run storage 类型错误")
        if (not isinstance(self.changes, tuple)
                or any(not isinstance(item, MemoryQueryIndexChange)
                       for item in self.changes)):
            raise TypeError("query index run changes 必须是 change tuple")
        changes = tuple(sorted(self.changes))
        if len({item.hypothesis_hash for item in changes}) != len(changes):
            raise ValueError("query index run 变化候选不得重复")
        object.__setattr__(self, "changes", changes)

    @property
    def invalidated_hypothesis_hashes(self) -> tuple[int, ...]:
        """返回本 run 使更早版本失效的全部候选身份。"""
        return tuple(item.hypothesis_hash for item in self.changes)

    def stable_key(self) -> tuple[int, ...]:
        """返回可重启恢复的 run 存储和变化状态。"""
        result: list[int] = []
        pack_key(result, self.storage.stable_key())
        result.append(len(self.changes))
        for change in self.changes:
            pack_key(result, change.stable_key())
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "MemoryQueryIndexRun":
        """从严格整数键恢复一个增量 run。"""
        reader = IntegerStreamReader(key)
        storage = MemoryCandidateProjectionManifest.from_stable_key(
            reader.read_key(label="query index run storage"))
        count = reader.read_nonnegative(label="query index run change count")
        changes = []
        for index in range(count):
            values = reader.read_key(label=f"query index run change[{index}]")
            if len(values) != 8:
                raise MemoryProjectionError("query index run change 长度漂移")
            changes.append(MemoryQueryIndexChange(
                values[0],
                values[1],
                tuple(values[2:6]),
                values[6],
                values[7],
            ))
        reader.finish()
        return cls(storage, tuple(changes))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class MemoryQueryIndexProjectionManifest:
    """绑定基础 Memory source state、planner 和双入口索引段的冻结清单。"""

    storage: MemoryCandidateProjectionManifest
    planner_key: tuple[int, ...]
    partitions: tuple[MemoryQueryIndexPartition, ...]
    runs: tuple[MemoryQueryIndexRun, ...] = ()

    def __post_init__(self) -> None:
        """核验存储清单、planner 身份和唯一分区计数。"""
        if not isinstance(self.storage, MemoryCandidateProjectionManifest):
            raise TypeError("query index storage manifest 类型错误")
        strict_integer_tuple(
            self.planner_key, label="query index planner_key")
        if (not isinstance(self.partitions, tuple)
                or any(not isinstance(item, MemoryQueryIndexPartition)
                       for item in self.partitions)):
            raise TypeError("query index partitions 必须是 partition tuple")
        if not self.partitions and not self.runs:
            raise ValueError("空 query index 只能由增量 run 保留失效历史")
        partitions = tuple(sorted(self.partitions))
        keys = tuple(
            (item.hypothesis_kind_hash, item.owner_key)
            for item in partitions
        )
        if len(set(keys)) != len(keys):
            raise ValueError("query index kind/owner 分区不得重复")
        if sum(item.index_record_count for item in partitions) != (
                self.storage.record_count) and not self.runs:
            raise ValueError("query index 分区记录数与冷段总数不闭合")
        object.__setattr__(self, "partitions", partitions)
        if (not isinstance(self.runs, tuple)
                or any(not isinstance(item, MemoryQueryIndexRun)
                       for item in self.runs)):
            raise TypeError("query index runs 必须是 run tuple")
        if self.runs:
            if self.storage.segments:
                raise ValueError("增量 query index 顶层 storage 不得重复承载 run 段")
            previous_fence = -1
            closed_states: set[tuple[int, ...]] = set()
            active_state: tuple[int, ...] | None = None
            active_state_hashes: set[int] = set()
            for run in self.runs:
                run_storage = run.storage
                if (run_storage.memory_space != self.storage.memory_space
                        or run_storage.memory_space_id
                        != self.storage.memory_space_id
                        or run_storage.access != self.storage.access
                        or run_storage.hypothesis_kinds
                        != self.storage.hypothesis_kinds
                        or run_storage.version_key != self.storage.version_key
                        or run_storage.dependencies != self.storage.dependencies):
                    raise ValueError("query index run 空间、ACL、kind、版本或依赖漂移")
                if (run_storage.source_fence < previous_fence
                        or run_storage.source_fence > self.storage.source_fence):
                    raise ValueError("query index run source fence 顺序漂移")
                state = run_storage.source_state_key
                hashes = set(run.invalidated_hypothesis_hashes)
                if state != active_state:
                    if state in closed_states:
                        raise ValueError("query index run source state 发生回返")
                    if active_state is not None:
                        closed_states.add(active_state)
                    active_state = state
                    active_state_hashes = set()
                elif not hashes:
                    raise ValueError(
                        "同 source state 的 query index spill run 必须声明候选")
                if active_state_hashes & hashes:
                    raise ValueError(
                        "同 source state 的 query index spill run 候选重叠")
                active_state_hashes.update(hashes)
                previous_fence = run_storage.source_fence

    @property
    def candidate_count(self) -> int:
        """返回全部 kind/owner 分区的逻辑候选总数。"""
        return sum(item.candidate_count for item in self.partitions)

    def partition_count(
            self,
            hypothesis_kind_hash: int,
            owner_key: tuple[int, int, int, int],
            ) -> int:
        """返回一个精确 kind/owner 分区的逻辑候选数。"""
        for item in self.partitions:
            if (item.hypothesis_kind_hash == hypothesis_kind_hash
                    and item.owner_key == owner_key):
                return item.candidate_count
        return 0

    def query_runs(self) -> tuple[MemoryQueryIndexRun, ...]:
        """返回按 source fence 排序的物理 run；v1 清单映射为单基线 run。"""
        if self.runs:
            return self.runs
        return (MemoryQueryIndexRun(self.storage, ()),)

    def stable_key(self) -> tuple[int, ...]:
        """返回可跨重启恢复的存储、planner 和分区完整键。"""
        version = (
            MEMORY_QUERY_INDEX_MANIFEST_VERSION
            if self.runs else MEMORY_QUERY_INDEX_LEGACY_MANIFEST_VERSION)
        result = [version]
        pack_key(result, self.storage.stable_key())
        pack_key(result, self.planner_key)
        result.append(len(self.partitions))
        for item in self.partitions:
            pack_key(result, item.stable_key())
        if self.runs:
            result.append(len(self.runs))
            for run in self.runs:
                pack_key(result, run.stable_key())
        return tuple(result)

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "MemoryQueryIndexProjectionManifest":
        """从纯整数键严格恢复查询索引清单。"""
        reader = IntegerStreamReader(key)
        version = reader.read_positive(label="query index manifest version")
        if version not in (
                MEMORY_QUERY_INDEX_LEGACY_MANIFEST_VERSION,
                MEMORY_QUERY_INDEX_MANIFEST_VERSION):
            raise MemoryProjectionError("query index manifest 版本未注册")
        storage = MemoryCandidateProjectionManifest.from_stable_key(
            reader.read_key(label="query index storage manifest"))
        planner_key = reader.read_key(label="query index planner_key")
        count = (
            reader.read_positive(label="query index partition count")
            if version == MEMORY_QUERY_INDEX_LEGACY_MANIFEST_VERSION
            else reader.read_nonnegative(label="query index partition count")
        )
        partitions = []
        for index in range(count):
            values = reader.read_key(label=f"query index partition[{index}]")
            if len(values) != 7:
                raise MemoryProjectionError("query index partition 长度漂移")
            partitions.append(MemoryQueryIndexPartition(
                values[0],
                tuple(values[1:5]),
                values[5],
                values[6],
            ))
        runs = ()
        if version == MEMORY_QUERY_INDEX_MANIFEST_VERSION:
            run_count = reader.read_positive(label="query index run count")
            runs = tuple(MemoryQueryIndexRun.from_stable_key(
                reader.read_key(label=f"query index run[{index}]"))
                for index in range(run_count))
        reader.finish()
        return cls(storage, planner_key, tuple(partitions), runs)

    def validate_store(self, store: TieredSegmentStore) -> None:
        """委托底层冻结投影核验 location manifest。"""
        self.storage.validate_store(store)
        for run in self.runs:
            run.storage.validate_store(store)


@dataclass(frozen=True)
class MemoryProjectionPublication:
    """由调用方注入的投影发布身份、温层、版本、依赖和双层预算。"""

    publication_key: tuple[int, ...]
    tier_key: tuple[int, ...]
    version_key: tuple[int, ...]
    dependencies: tuple[SegmentDependency, ...]
    segment_budget: SegmentBudget
    backend_page_limit: int

    def __post_init__(self) -> None:
        """核验发布身份、目标温层、依赖和有界源读取配置。"""
        strict_integer_tuple(
            self.publication_key, label="memory projection publication_key")
        strict_integer_tuple(self.tier_key, label="memory projection tier_key")
        strict_integer_tuple(
            self.version_key, label="memory projection publication version_key")
        dependencies = canonical_dependencies(self.dependencies)
        if tuple(item.descriptor_key for item in dependencies) != (
                MEMORY_QUERY_PROJECTION_DESCRIPTOR.dependency_keys):
            raise ValueError("memory projection publication dependencies 漂移")
        object.__setattr__(self, "dependencies", dependencies)
        if not isinstance(self.segment_budget, SegmentBudget):
            raise TypeError("memory projection segment_budget 类型错误")
        if type(self.backend_page_limit) is not int or self.backend_page_limit <= 0:
            raise ValueError("memory projection backend_page_limit 必须是正严格整数")


@dataclass(frozen=True)
class MemoryProjectionPlacementPublication:
    """由调用方注入的 placement 执行命名空间和精确记录查找预算。"""

    publication_key: tuple[int, ...]
    lookup_budget: SegmentBudget

    def __post_init__(self) -> None:
        """核验 placement 命名空间和单记录冷页预算。"""
        strict_integer_tuple(
            self.publication_key,
            label="memory projection placement publication_key",
        )
        if not isinstance(self.lookup_budget, SegmentBudget):
            raise TypeError("memory projection placement lookup_budget 类型错误")


@dataclass(frozen=True, order=True)
class MemoryProjectionPlacementDirective:
    """由对象级 hint 归并出的一个 segment 目标温层及完整理由集。"""

    segment_key: tuple[int, ...]
    target_tier_key: tuple[int, ...]
    hint_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验 segment、目标温层和非空唯一 hint 稳定键。"""
        strict_integer_tuple(
            self.segment_key, label="projection placement segment_key")
        strict_integer_tuple(
            self.target_tier_key, label="projection placement target_tier_key")
        if (not isinstance(self.hint_keys, tuple)
                or not self.hint_keys):
            raise ValueError("projection placement hint_keys 不得为空")
        hints = tuple(sorted(strict_integer_tuple(
            item, label="projection placement hint_key")
            for item in self.hint_keys))
        if len(set(hints)) != len(hints):
            raise ValueError("projection placement hint_keys 不得重复")
        object.__setattr__(self, "hint_keys", hints)

    def stable_key(self) -> tuple[int, ...]:
        """返回 segment、目标 tier 和全部对象级理由的稳定键。"""
        result: list[int] = []
        pack_key(result, self.segment_key)
        pack_key(result, self.target_tier_key)
        result.append(len(self.hint_keys))
        for key in self.hint_keys:
            pack_key(result, key)
        return tuple(result)


@dataclass(frozen=True)
class MemoryProjectionPlacementReceipt:
    """一次 placement plan 的稳定身份、最终指令和完成 manifest epoch。"""

    plan_key: tuple[int, ...]
    projection_key: tuple[int, ...]
    directives: tuple[MemoryProjectionPlacementDirective, ...]
    publish_epoch: int

    def __post_init__(self) -> None:
        """核验 plan、投影代、唯一指令和正发布 epoch。"""
        strict_integer_tuple(
            self.plan_key, label="projection placement receipt plan_key")
        strict_integer_tuple(
            self.projection_key,
            label="projection placement receipt projection_key",
        )
        if (not isinstance(self.directives, tuple)
                or not self.directives
                or any(not isinstance(item, MemoryProjectionPlacementDirective)
                       for item in self.directives)):
            raise TypeError("projection placement directives 必须是非空 tuple")
        directives = tuple(sorted(self.directives))
        if len({item.segment_key for item in directives}) != len(directives):
            raise ValueError("projection placement segment 指令不得重复")
        object.__setattr__(self, "directives", directives)
        if type(self.publish_epoch) is not int or self.publish_epoch <= 0:
            raise ValueError("projection placement publish_epoch 必须是正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回可用于审计和重试比对的完整 placement receipt。"""
        result: list[int] = []
        pack_key(result, self.plan_key)
        pack_key(result, self.projection_key)
        result.append(len(self.directives))
        for directive in self.directives:
            pack_key(result, directive.stable_key())
        result.append(self.publish_epoch)
        return tuple(result)


def _projection_object_key(
        publication_key: tuple[int, ...],
        object_kind: int,
        ordinal: int,
        ) -> tuple[int, ...]:
    """形成带对象类别和 ordinal 的无歧义 K-04 发布对象键。"""
    if type(object_kind) is not int or object_kind <= 0:
        raise ValueError("memory projection object_kind 必须是正严格整数")
    if type(ordinal) is not int or ordinal <= 0:
        raise ValueError("memory projection ordinal 必须是正严格整数")
    publication = strict_integer_tuple(
        publication_key, label="memory projection object publication_key")
    return (
        MEMORY_PROJECTION_OBJECT_KEY_VERSION,
        object_kind,
        len(publication),
        *publication,
        ordinal,
    )


def _projection_generation_key(
        base_key: tuple[int, ...],
        source_state_key: tuple[int, ...],
        version_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """把调用方命名空间、完整源状态和版本组成无重叠投影代身份。"""
    result = [MEMORY_PROJECTION_GENERATION_KEY_VERSION]
    for value, label in (
            (base_key, "projection generation base_key"),
            (source_state_key, "projection generation source_state_key"),
            (version_key, "projection generation version_key")):
        pack_key(result, strict_integer_tuple(value, label=label))
    return tuple(result)


def _projection_placement_plan_key(
        publication_key: tuple[int, ...],
        projection_key: tuple[int, ...],
        hints: tuple[MemoryPlacementHint, ...],
        ) -> tuple[int, ...]:
    """把执行命名空间、投影代和完整 hint 集组成确定性迁移计划身份。"""
    result = [MEMORY_PROJECTION_PLACEMENT_PLAN_VERSION]
    pack_key(result, strict_integer_tuple(
        publication_key, label="projection placement plan publication_key"))
    pack_key(result, strict_integer_tuple(
        projection_key, label="projection placement plan projection_key"))
    result.append(len(hints))
    for hint in hints:
        pack_key(result, hint.stable_key())
    return tuple(result)


class MemoryCandidateProjectionPublisher:
    """从干净 M-04 索引流式形成不可变候选 segment 并发布 location epoch。"""

    def __init__(
            self,
            resolver: MemoryOverlayResolver,
            store: TieredSegmentStore,
            ) -> None:
        """绑定全热恢复边界和 K-02 store，不持有写入 Memory event 的能力。"""
        if not isinstance(resolver, MemoryOverlayResolver):
            raise TypeError("memory projection resolver 类型错误")
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("memory projection store 类型错误")
        self.resolver = resolver
        self.store = store

    def publish(
            self,
            projection_key: tuple[int, ...],
            *,
            access: MemoryAccessContext,
            hypothesis_kinds: tuple[tuple[int, ...], ...],
            publication: MemoryProjectionPublication,
            ) -> MemoryCandidateProjectionManifest:
        """按 kind/owner 稳定分区发布投影，并在前后核验源 timeline fence。"""
        projection = strict_integer_tuple(
            projection_key, label="memory projection publish projection_key")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("memory projection publish access 类型错误")
        if not isinstance(publication, MemoryProjectionPublication):
            raise TypeError("memory projection publication 类型错误")
        kinds = tuple(sorted(
            strict_integer_tuple(item, label="memory projection publish kind")
            for item in hypothesis_kinds
        ))
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("memory projection publish kinds 必须非空且唯一")
        self.resolver.aggregates.require_clean(access=access)
        source_state_key = self._source_state_key()
        source_fence = source_state_key[0]
        generation_key = _projection_generation_key(
            projection,
            source_state_key,
            publication.version_key,
        )
        publication_generation_key = _projection_generation_key(
            publication.publication_key,
            source_state_key,
            publication.version_key,
        )
        kind_by_hash: dict[int, set[tuple[int, ...]]] = {}
        for kind in kinds:
            kind_hash = self.resolver.aggregates.hypothesis_kind_hash(kind)
            kind_by_hash.setdefault(kind_hash, set()).add(kind)
        summaries: list[MemoryProjectionSegment] = []
        attempted_segment_keys: list[tuple[int, ...]] = []
        ordinal = 0
        try:
            for kind_hash in sorted(kind_by_hash):
                accepted_kinds = kind_by_hash[kind_hash]
                for owner_key in visible_owner_keys(access):
                    delta = OpenHotDelta(
                        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
                        publication.version_key,
                        publication.dependencies,
                        publication.segment_budget,
                    )
                    for aggregate in self.resolver.aggregates.store.iter_aggregates_by_kind_owner(
                            kind_hash,
                            owner_key,
                            page_limit=publication.backend_page_limit):
                        bundle = self.resolver.load_bundle(aggregate, access=access)
                        if bundle.hypothesis.hypothesis_kind not in accepted_kinds:
                            continue
                        record = encode_memory_candidate(generation_key, bundle)
                        try:
                            delta.append(record)
                        except SegmentBudgetExceeded:
                            if delta.object_count == 0:
                                raise
                            ordinal += 1
                            attempted_segment_keys.append(_projection_object_key(
                                publication_generation_key,
                                MEMORY_PROJECTION_SEGMENT_OBJECT,
                                ordinal,
                            ))
                            summaries.append(self._publish_delta(
                                delta,
                                publication,
                                publication_generation_key,
                                source_fence,
                                ordinal,
                            ))
                            delta = OpenHotDelta(
                                MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
                                publication.version_key,
                                publication.dependencies,
                                publication.segment_budget,
                            )
                            delta.append(record)
                    if delta.object_count:
                        ordinal += 1
                        attempted_segment_keys.append(_projection_object_key(
                            publication_generation_key,
                            MEMORY_PROJECTION_SEGMENT_OBJECT,
                            ordinal,
                        ))
                        summaries.append(self._publish_delta(
                            delta,
                            publication,
                            publication_generation_key,
                            source_fence,
                            ordinal,
                        ))
            self.resolver.aggregates.require_clean(access=access)
            if self._source_state_key() != source_state_key:
                raise MemoryProjectionError(
                    "投影扫描期间 Memory/visibility epoch 已变化")
            current = self.store.current_manifest()
            manifest = MemoryCandidateProjectionManifest(
                generation_key,
                self.resolver.aggregates.event_log.memory_space_identity,
                self.resolver.aggregates.event_log.memory_space_id,
                access,
                kinds,
                source_fence,
                source_state_key,
                publication.version_key,
                publication.dependencies,
                tuple(summaries),
                0 if current is None else current.publish_epoch,
            )
            manifest.validate_store(self.store)
            return manifest
        except Exception:
            try:
                self._cleanup_attempted_generation(
                    tuple(attempted_segment_keys),
                    publication_generation_key,
                )
            except Exception as cleanup_error:
                raise MemoryProjectionError(
                    "候选投影发布失败，且部分 generation 自动清理失败"
                ) from cleanup_error
            raise

    def _publish_delta(
            self,
            delta: OpenHotDelta,
            publication: MemoryProjectionPublication,
            publication_generation_key: tuple[int, ...],
            source_fence: int,
            ordinal: int,
            ) -> MemoryProjectionSegment:
        """封存并幂等发布一个有界分区页，返回其完整校验摘要。"""
        segment_key = _projection_object_key(
            publication_generation_key,
            MEMORY_PROJECTION_SEGMENT_OBJECT,
            ordinal,
        )
        segment = delta.seal(segment_key, source_fence)
        self.store.publish_segment(
            segment,
            tier_key=publication.tier_key,
            manifest_key=_projection_object_key(
                publication_generation_key,
                MEMORY_PROJECTION_LOCATION_MANIFEST_OBJECT,
                ordinal,
            ),
            migration_key=_projection_object_key(
                publication_generation_key,
                MEMORY_PROJECTION_MIGRATION_OBJECT,
                ordinal,
            ),
        )
        delta.acknowledge(segment)
        return MemoryProjectionSegment(
            segment.segment_key,
            segment.lower_key,
            segment.upper_key,
            segment.checksum_key,
            len(segment.records),
            segment.size_bytes,
        )

    def _source_state_key(self) -> tuple[int, ...]:
        """读取物理事件水位及 batch/forget 可见性 epoch。"""
        return self.resolver.aggregates.event_log.projection_state_key()

    def _cleanup_attempted_generation(
            self,
            attempted_segment_keys: tuple[tuple[int, ...], ...],
            publication_generation_key: tuple[int, ...],
            ) -> None:
        """恢复底层中间态，并精确释放本次失败 generation 已发布的段。"""
        if not attempted_segment_keys:
            return
        self.store.recover_pending_operations()
        current = self.store.current_manifest()
        if current is None:
            return
        attempted = set(attempted_segment_keys)
        entries = tuple(
            item for item in current.entries
            if item.segment_key in attempted
        )
        if not entries:
            return
        if any(item.descriptor_key != MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
               for item in entries):
            raise MemoryProjectionError(
                "失败 generation 的 segment 身份与其他 descriptor 冲突")
        self.store.release_rebuildable_segments(
            tuple(item.segment_key for item in entries),
            release_key=_projection_object_key(
                publication_generation_key,
                MEMORY_PROJECTION_CLEANUP_RELEASE_OBJECT,
                1,
            ),
            manifest_key=_projection_object_key(
                publication_generation_key,
                MEMORY_PROJECTION_CLEANUP_MANIFEST_OBJECT,
                1,
            ),
        )

    def apply_placement_hints(
            self,
            projection: MemoryCandidateProjectionManifest,
            hints: tuple[MemoryPlacementHint, ...],
            *,
            publication: MemoryProjectionPlacementPublication,
            ) -> MemoryProjectionPlacementReceipt:
        """把对象级 M-09 hint 精确定位并归并为可恢复的 segment 迁移。"""
        if not isinstance(projection, MemoryCandidateProjectionManifest):
            raise TypeError("projection placement manifest 类型错误")
        if (not isinstance(hints, tuple)
                or not hints
                or any(not isinstance(item, MemoryPlacementHint)
                       for item in hints)):
            raise TypeError("projection placement hints 必须是非空 tuple")
        if not isinstance(publication, MemoryProjectionPlacementPublication):
            raise TypeError("projection placement publication 类型错误")
        if (projection.memory_space
                != self.resolver.aggregates.event_log.memory_space_identity
                or projection.memory_space_id
                != self.resolver.aggregates.event_log.memory_space_id):
            raise ValueError("placement projection 属于其他 Memory 空间")
        normalized = tuple(sorted(hints, key=lambda item: item.stable_key()))
        hint_keys = tuple(item.stable_key() for item in normalized)
        if len(set(hint_keys)) != len(hint_keys):
            raise ValueError("projection placement hints 不得重复")
        if len({item.object_key for item in normalized}) != len(normalized):
            raise ValueError("同一 Memory 对象不得提交多个 placement hint")

        self.store.recover_pending_operations()
        self._require_projection_fresh(projection)
        projection.validate_store(self.store)
        plan_key = _projection_placement_plan_key(
            publication.publication_key,
            projection.projection_key,
            normalized,
        )
        directives = self._placement_directives(
            projection,
            normalized,
            plan_key,
            publication.lookup_budget,
        )
        self._require_projection_fresh(projection)
        try:
            for ordinal, directive in enumerate(directives, start=1):
                current = self.store.current_manifest()
                if current is None:
                    raise MemoryProjectionError(
                        "placement 执行期间 location manifest 消失")
                entry = next((
                    item for item in current.entries
                    if item.segment_key == directive.segment_key
                ), None)
                if entry is None:
                    raise MemoryProjectionError(
                        "placement 执行期间目标 segment 消失")
                if entry.descriptor_key != MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY:
                    raise MemoryProjectionError(
                        "placement 目标 segment descriptor 漂移")
                if entry.tier_key == directive.target_tier_key:
                    continue
                self.store.migrate(
                    directive.segment_key,
                    target_tier_key=directive.target_tier_key,
                    manifest_key=_projection_object_key(
                        plan_key,
                        MEMORY_PROJECTION_PLACEMENT_MANIFEST_OBJECT,
                        ordinal,
                    ),
                    migration_key=_projection_object_key(
                        plan_key,
                        MEMORY_PROJECTION_PLACEMENT_MIGRATION_OBJECT,
                        ordinal,
                    ),
                )
        except Exception:
            self.store.recover_pending_operations()
            raise
        self._require_projection_fresh(projection)
        projection.validate_store(self.store)
        current = self.store.current_manifest()
        if current is None:
            raise MemoryProjectionError("placement 完成后缺少 location manifest")
        return MemoryProjectionPlacementReceipt(
            plan_key,
            projection.projection_key,
            directives,
            current.publish_epoch,
        )

    def _placement_directives(
            self,
            projection: MemoryCandidateProjectionManifest,
            hints: tuple[MemoryPlacementHint, ...],
            plan_key: tuple[int, ...],
            lookup_budget: SegmentBudget,
            ) -> tuple[MemoryProjectionPlacementDirective, ...]:
        """回读每个 hint 的精确候选，并把一致目标归并到所属 segment。"""
        lower_keys = tuple(item.lower_key for item in projection.segments)
        grouped: dict[
            tuple[int, ...], tuple[tuple[int, ...], list[tuple[int, ...]]]
        ] = {}
        reader = self.store.open_reader(
            _projection_object_key(
                plan_key,
                MEMORY_PROJECTION_PLACEMENT_READER_OBJECT,
                1,
            ),
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        )
        try:
            for hint in hints:
                if hint.descriptor_key != MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY:
                    raise MemoryProjectionError(
                        "M-09 hint descriptor 不是当前候选投影")
                if (hint.temperature_profile_key
                        != self.store.temperature_profile.profile_key):
                    raise MemoryProjectionError(
                        "M-09 hint temperature profile 与 K-02 store 漂移")
                if not self.store.temperature_profile.has(
                        hint.preferred_tier_key):
                    raise MemoryProjectionError(
                        "M-09 hint preferred tier 未注册")
                if hint.as_of_seq != projection.source_fence:
                    raise MemoryProjectionError(
                        "M-09 hint as_of 与投影 source fence 漂移")
                reference = MemoryObjectRef.from_stable_key(hint.object_key)
                if (reference.memory_space != projection.memory_space
                        or not projection.access.can_read(reference.owner)):
                    raise MemoryProjectionError(
                        "M-09 hint 对象不属于投影空间或 ACL")
                self.resolver.aggregates.require_hypothesis_clean(
                    reference, access=projection.access)
                aggregate = self.resolver.aggregates.read(
                    reference, access=projection.access)
                if aggregate is None:
                    raise MemoryProjectionError(
                        "M-09 hint 找不到当前 Hypothesis aggregate")
                record_key = memory_candidate_record_key(
                    projection.projection_key, aggregate)
                segment = self._segment_for_record_key(
                    projection.segments, lower_keys, record_key)
                page = reader.page(
                    budget=lookup_budget,
                    lower_key=record_key,
                    upper_key=record_key,
                )
                if page.has_more or len(page.records) != 1:
                    raise MemoryProjectionError(
                        "M-09 hint 未唯一定位到投影 typed record")
                bundle = decode_memory_candidate(
                    projection.projection_key, page.records[0])
                if bundle.hypothesis_ref != reference:
                    raise MemoryProjectionError(
                        "M-09 hint 定位记录与完整对象引用漂移")
                existing = grouped.get(segment.segment_key)
                hint_key = hint.stable_key()
                if existing is None:
                    grouped[segment.segment_key] = (
                        hint.preferred_tier_key, [hint_key])
                else:
                    target, keys = existing
                    if target != hint.preferred_tier_key:
                        raise MemoryProjectionError(
                            "同一 projection segment 收到冲突目标 tier")
                    keys.append(hint_key)
        finally:
            reader.close()
        return tuple(MemoryProjectionPlacementDirective(
            segment_key,
            target,
            tuple(keys),
        ) for segment_key, (target, keys) in sorted(grouped.items()))

    def _segment_for_record_key(
            self,
            segments: tuple[MemoryProjectionSegment, ...],
            lower_keys: tuple[tuple[int, ...], ...],
            record_key: tuple[int, ...],
            ) -> MemoryProjectionSegment:
        """用已排序闭区间定位唯一 segment，拒绝范围缺失或重叠。"""
        index = bisect_right(lower_keys, record_key) - 1
        if index < 0:
            raise MemoryProjectionError("M-09 hint record 早于全部投影范围")
        segment = segments[index]
        if record_key > segment.upper_key:
            raise MemoryProjectionError("M-09 hint record 不在投影 segment 范围")
        return segment

    def _require_projection_fresh(
            self,
            projection: MemoryCandidateProjectionManifest,
            ) -> None:
        """要求 placement 处理期间投影仍绑定当前完整 Memory source state。"""
        if self._source_state_key() != projection.source_state_key:
            raise MemoryProjectionError(
                "M-09 placement hint 所属候选投影已经失效")

    def release(
            self,
            projection: MemoryCandidateProjectionManifest,
            *,
            release_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            ) -> None:
        """按冻结 manifest 精确释放一个旧投影代，并保留 K-02 reader 屏障。"""
        if not isinstance(projection, MemoryCandidateProjectionManifest):
            raise TypeError("release memory projection 类型错误")
        if (projection.memory_space
                != self.resolver.aggregates.event_log.memory_space_identity
                or projection.memory_space_id
                != self.resolver.aggregates.event_log.memory_space_id):
            raise ValueError("待释放 projection 属于其他 Memory 空间")
        if not projection.segments:
            return
        projection.validate_store(self.store)
        self.store.release_rebuildable_segments(
            tuple(item.segment_key for item in projection.segments),
            release_key=release_key,
            manifest_key=manifest_key,
        )


class MemoryHotSetRuntime:
    """把 M-07 resolver 切换到版本化冷投影和 query-local 有界热集。"""

    def __init__(
            self,
            ctx: TrainContext,
            resolver: MemoryOverlayResolver,
            store: TieredSegmentStore,
            projection: MemoryCandidateProjectionManifest,
            policy: QueryHotSetPolicy,
            ) -> None:
        """绑定当前上下文、同一 resolver、投影 manifest 和设备热集策略。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("memory hot-set ctx 类型错误")
        if not isinstance(resolver, MemoryOverlayResolver):
            raise TypeError("memory hot-set resolver 类型错误")
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("memory hot-set store 类型错误")
        if not isinstance(projection, MemoryCandidateProjectionManifest):
            raise TypeError("memory hot-set projection 类型错误")
        if not isinstance(policy, QueryHotSetPolicy):
            raise TypeError("memory hot-set policy 类型错误")
        if ctx.memory_resolver_runtime is None:
            raise ValueError("安装 K-04 前必须先安装 M-07 runtime")
        if all(item is not resolver
               for item in ctx.memory_resolver_runtime.resolvers):
            raise ValueError("K-04 resolver 不属于当前 M-07 联邦")
        if ctx.tiered_segment_store is not store:
            raise ValueError("K-04 store 不是当前 TrainContext tiered store")
        if projection.memory_space != resolver.aggregates.event_log.memory_space_identity:
            raise ValueError("K-04 projection 属于其他 Memory 空间")
        if projection.memory_space_id != resolver.aggregates.event_log.memory_space_id:
            raise ValueError("K-04 projection 物理 Memory space_id 漂移")
        if not isinstance(resolver.diversity_policy, BoundedCandidateSelectionPolicy):
            raise TypeError("受限 K-04 profile 要求有界流式选择协议")
        projection.validate_store(store)
        self._ctx = ctx
        self.resolver = resolver
        self.store = store
        self.projection = projection
        self.policy = policy
        self._hot_set: QuerySegmentHotSet | None = None
        self._compilation_key: tuple[int, ...] | None = None
        self._resolution: MemoryResolution | None = None
        self._last_metrics: QueryHotSetMetrics | None = None
        self._last_considered_count: int | None = None
        self._last_selected_candidate_keys: tuple[tuple[int, ...], ...] | None = None
        self._query_index_projection: MemoryQueryIndexProjectionManifest | None = None

    def resolve(self, compilation: MemoryQueryCompilation) -> MemoryResolution:
        """从固定源 fence 的候选页流式评分并返回与全热策略等价的结果。"""
        if not isinstance(compilation, MemoryQueryCompilation):
            raise TypeError("memory hot-set compilation 类型错误")
        active_scope = self._ctx.work_memory.active_query_scope
        if active_scope is None or active_scope != compilation.current.scope:
            raise ValueError("memory hot-set compilation 不属于活动 query")
        if compilation.memory_space != self.projection.memory_space:
            raise ValueError("memory hot-set compilation 属于其他 Memory 空间")
        if compilation.access != self.projection.access:
            raise PermissionError("memory hot-set projection ACL 与 query 不一致")
        current_key = compilation.stable_key()
        if self._resolution is not None:
            if self._compilation_key != current_key:
                raise RuntimeError("同一 query 不得切换 Memory compilation")
            return self._resolution
        self._require_fresh_projection()
        self.projection.validate_store(self.store)
        active_projection = (
            self.projection
            if self._query_index_projection is None
            else self._query_index_projection.storage
        )
        active_projection.validate_store(self.store)
        indexed_segments = (
            ()
            if self._query_index_projection is None
            else tuple(
                segment
                for run in self._query_index_projection.query_runs()
                for segment in run.storage.segments
            )
        )
        if active_projection.segments or indexed_segments:
            reader_key = (
                MEMORY_PROJECTION_MANIFEST_VERSION,
                *active_scope.stable_key(),
                len(active_projection.projection_key),
                *active_projection.projection_key,
            )
            self._hot_set = QuerySegmentHotSet(
                self.store,
                reader_key=reader_key,
                descriptor_key=MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
                policy=self.policy,
            )
        self._ctx.work_memory.register_query_resource(self)
        sets = tuple(self._resolve_request(request)
                     for request in compilation.requests)
        self._resolution = MemoryResolution(compilation, sets)
        self._last_considered_count = sum(
            item.considered_count for item in sets)
        self._last_selected_candidate_keys = tuple(
            (
                request_ordinal,
                candidate_ordinal,
                candidate.origin_kind,
                len(candidate.candidate_key),
                *candidate.candidate_key,
            )
            for request_ordinal, candidate_set in enumerate(sets, start=1)
            for candidate_ordinal, candidate in enumerate(
                candidate_set.candidates, start=1)
        )
        self._compilation_key = current_key
        return self._resolution

    def close(self) -> None:
        """由 WorkMemory query 生命周期释放全部候选 pin、cache 和 reader lease。"""
        if self._hot_set is not None:
            self._hot_set.close()
            self._last_metrics = self._hot_set.metrics()
        self._hot_set = None
        self._compilation_key = None
        self._resolution = None

    def metrics(self) -> QueryHotSetMetrics | None:
        """返回活动 query 或最近一次已关闭 query 的物理计数。"""
        if self._hot_set is not None:
            return self._hot_set.metrics()
        return self._last_metrics

    def considered_count(self) -> int | None:
        """返回当前或最近完成 query 在本 Memory 空间的逻辑候选考虑数。"""
        if self._resolution is not None:
            return sum(item.considered_count for item in self._resolution.sets)
        return self._last_considered_count

    def selected_candidate_keys(self) -> tuple[tuple[int, ...], ...] | None:
        """返回当前或最近 query 逐 request、逐序位的候选完整身份。"""
        return self._last_selected_candidate_keys

    def query_index_projection(
            self,
            ) -> MemoryQueryIndexProjectionManifest | None:
        """返回当前只读精确查询索引清单；未安装时为空。"""
        return self._query_index_projection

    def exact_index_available_for(
            self,
            requests: tuple[MemoryActivationRequest, ...],
            ) -> bool:
        """仅在本批 requests 全部命中 exact-index 前提时返回证明资格。"""
        if (not isinstance(requests, tuple)
                or not requests
                or any(not isinstance(item, MemoryActivationRequest)
                       for item in requests)):
            raise TypeError("exact index requests 必须是非空 request tuple")
        if self._query_index_projection is None:
            return False
        planner = self.resolver.score_provider
        if not isinstance(planner, ExactMemoryProjectionPlanner):
            return False
        return all(self._filters(request) == (MemoryAggregateFilter(),)
                   for request in requests)

    def query_resources_closed(self) -> bool:
        """返回当前 query 的 reader、热集和缓存结果是否已经全部释放。"""
        return (
            self._hot_set is None
            and self._compilation_key is None
            and self._resolution is None
        )

    def replace_projection(
            self,
            projection: MemoryCandidateProjectionManifest,
            ) -> None:
        """在无活动 query 时切换到同 ACL/kind 的已核验新一代投影。"""
        if not isinstance(projection, MemoryCandidateProjectionManifest):
            raise TypeError("replace projection 类型错误")
        if (self._ctx.work_memory.active_query_scope is not None
                or self._hot_set is not None
                or self._resolution is not None):
            raise RuntimeError("活动 query 期间不得替换 Memory 候选投影")
        if (projection.memory_space != self.projection.memory_space
                or projection.memory_space_id != self.projection.memory_space_id
                or projection.access != self.projection.access
                or projection.hypothesis_kinds
                != self.projection.hypothesis_kinds):
            raise ValueError("新旧 Memory 候选投影的空间、ACL 或 kind 漂移")
        if (self.resolver.aggregates.event_log.projection_state_key()
                != projection.source_state_key):
            raise MemoryProjectionError("待切换 Memory 候选投影已经失效")
        projection.validate_store(self.store)
        self.projection = projection
        if (self._query_index_projection is not None
                and (self._query_index_projection.storage.source_state_key
                     != projection.source_state_key
                     or self._query_index_projection.storage.memory_space
                     != projection.memory_space)):
            self._query_index_projection = None
        self._last_metrics = None
        self._last_considered_count = None
        self._last_selected_candidate_keys = None

    def replace_query_index(
            self,
            projection: MemoryQueryIndexProjectionManifest | None,
            ) -> None:
        """在无活动 query 时安装或移除同 source state 的精确查询索引。"""
        if (self._ctx.work_memory.active_query_scope is not None
                or self._hot_set is not None
                or self._resolution is not None):
            raise RuntimeError("活动 query 期间不得替换 Memory 查询索引")
        if projection is not None:
            if not isinstance(projection, MemoryQueryIndexProjectionManifest):
                raise TypeError("replace query index projection 类型错误")
            storage = projection.storage
            if (storage.memory_space != self.projection.memory_space
                    or storage.memory_space_id != self.projection.memory_space_id
                    or storage.access != self.projection.access
                    or storage.hypothesis_kinds
                    != self.projection.hypothesis_kinds):
                raise ValueError("查询索引与基础候选投影的空间、ACL 或 kind 漂移")
            if (self.resolver.aggregates.event_log.projection_state_key()
                    != storage.source_state_key):
                raise ValueError("查询索引 source state 已经失效")
            planner = self.resolver.score_provider
            if not isinstance(planner, ExactMemoryProjectionPlanner):
                raise TypeError("当前 scorer 未实现精确投影 planner")
            planner_key = strict_integer_tuple(
                planner.state_key(), label="query index live planner_key")
            if planner_key != projection.planner_key:
                raise ValueError("查询索引 planner 与当前 scorer 状态漂移")
            projection.validate_store(self.store)
        elif (self.resolver.aggregates.event_log.projection_state_key()
              != self.projection.source_state_key):
            raise MemoryProjectionError(
                "基础候选投影失效时不得移除当前查询索引")
        self._query_index_projection = projection
        self._last_metrics = None
        self._last_considered_count = None
        self._last_selected_candidate_keys = None

    def replace_indexed_projection(
            self,
            base_projection: MemoryCandidateProjectionManifest,
            query_index: MemoryQueryIndexProjectionManifest,
            ) -> None:
        """原子恢复同范围旧基线和绑定当前 source state 的增量索引。"""
        if (self._ctx.work_memory.active_query_scope is not None
                or self._hot_set is not None
                or self._resolution is not None):
            raise RuntimeError("活动 query 期间不得恢复 Memory 增量索引")
        if not isinstance(base_projection, MemoryCandidateProjectionManifest):
            raise TypeError("indexed base projection 类型错误")
        if not isinstance(query_index, MemoryQueryIndexProjectionManifest):
            raise TypeError("indexed query projection 类型错误")
        current = self.projection
        index_storage = query_index.storage
        for candidate in (base_projection, index_storage):
            if (candidate.memory_space != current.memory_space
                    or candidate.memory_space_id != current.memory_space_id
                    or candidate.access != current.access
                    or candidate.hypothesis_kinds != current.hypothesis_kinds):
                raise ValueError("增量索引恢复的空间、ACL 或 kind 漂移")
        if (self.resolver.aggregates.event_log.projection_state_key()
                != index_storage.source_state_key):
            raise MemoryProjectionError("待恢复增量索引 source state 已失效")
        planner = self.resolver.score_provider
        if (not isinstance(planner, ExactMemoryProjectionPlanner)
                or strict_integer_tuple(
                    planner.state_key(), label="indexed restore planner state")
                != query_index.planner_key):
            raise ValueError("待恢复增量索引 planner 状态漂移")
        base_projection.validate_store(self.store)
        query_index.validate_store(self.store)
        self.projection = base_projection
        self._query_index_projection = query_index
        self._last_metrics = None
        self._last_considered_count = None
        self._last_selected_candidate_keys = None

    def replace_policy(self, policy: QueryHotSetPolicy) -> None:
        """在无活动 query 时替换同语义的设备热集预算和预取策略。"""
        if not isinstance(policy, QueryHotSetPolicy):
            raise TypeError("replace memory hot-set policy 类型错误")
        if (self._ctx.work_memory.active_query_scope is not None
                or self._hot_set is not None
                or self._resolution is not None):
            raise RuntimeError("活动 query 期间不得替换 Memory hot-set policy")
        self.policy = policy
        self._last_metrics = None
        self._last_considered_count = None
        self._last_selected_candidate_keys = None

    def apply_maintenance_assessments(
            self,
            assessments: tuple[MemoryMaintenanceAssessment, ...],
            *,
            publication: MemoryProjectionPlacementPublication,
            ) -> MemoryProjectionPlacementReceipt | None:
        """消费当前 M-09 assessment 的 placement hint，并执行纯物理投影迁移。"""
        if (not isinstance(assessments, tuple)
                or not assessments
                or any(not isinstance(item, MemoryMaintenanceAssessment)
                       for item in assessments)):
            raise TypeError("K-04 maintenance assessments 必须是非空 tuple")
        maintenance = self._ctx.memory_maintenance_runtime
        if maintenance is None:
            raise RuntimeError("消费 M-09 placement hint 前必须安装维护 runtime")
        service = maintenance.service
        if service.aggregates is not self.resolver.aggregates:
            raise ValueError("M-09 与 K-04 未绑定同一 Memory aggregate")
        if service.storage_roles is not self.store.registry:
            raise ValueError("M-09 与 K-04 未绑定同一 storage role registry")
        if service.temperature_profile != self.store.temperature_profile:
            raise ValueError("M-09 与 K-04 temperature profile 漂移")
        placement_policy_key = strict_integer_tuple(
            service.placement_policy.state_key(),
            label="K-04 M-09 placement policy state_key",
        )
        hints = tuple(
            hint
            for assessment in assessments
            for hint in assessment.placement_hints
        )
        if not hints:
            return None
        if any(item.policy_key != placement_policy_key for item in hints):
            raise ValueError("M-09 assessment placement policy 身份漂移")
        publisher = MemoryCandidateProjectionPublisher(
            self.resolver, self.store)
        return publisher.apply_placement_hints(
            self.projection,
            hints,
            publication=publication,
        )

    def evaluate_edge_budget(
            self,
            profile: EdgeBudgetProfile,
            *,
            external_observations: tuple[EdgeMetricObservation, ...] = (),
            ) -> EdgeBudgetReport:
        """合并 query 真实计数与外部物理探针，并执行逐维硬预算。"""
        if not isinstance(profile, EdgeBudgetProfile):
            raise TypeError("memory hot-set edge profile 类型错误")
        if (not isinstance(external_observations, tuple)
                or any(not isinstance(item, EdgeMetricObservation)
                       for item in external_observations)):
            raise TypeError("external edge observations 必须是 tuple")
        metrics = self.metrics()
        if metrics is None:
            raise RuntimeError("尚无完成或活动 query 可供边缘预算评估")
        return profile.evaluate((
            *metrics.observations(),
            *external_observations,
        ))

    def state_key(self) -> tuple[int, ...]:
        """返回投影、resolver 和热集策略的完整装配状态。"""
        result = [MEMORY_PROJECTION_MANIFEST_VERSION]
        for value in (
                self.resolver.state_key(),
                self.projection.stable_key(),
                (
                    self.policy.cache_budget.object_limit,
                    self.policy.cache_budget.byte_limit,
                    self.policy.page_budget.object_limit,
                    self.policy.page_budget.byte_limit,
                    self.policy.fault_report_limit,
                ),
                self.policy.prefetch_state_key()):
            pack_key(result, value)
        if self._query_index_projection is not None:
            pack_key(result, self._query_index_projection.stable_key())
        return tuple(result)

    def clone_for_context(self, ctx: TrainContext) -> "MemoryHotSetRuntime":
        """为 V-06 重绑 clone 的 resolver/store，并复用不可变投影和物理策略。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("memory hot-set clone ctx 类型错误")
        if ctx.memory_resolver_runtime is None or ctx.tiered_segment_store is None:
            raise ValueError("评测 clone 缺少 M-07 或 tiered store")
        matches = tuple(
            item for item in ctx.memory_resolver_runtime.resolvers
            if (item.aggregates.event_log.memory_space_identity
                == self.resolver.aggregates.event_log.memory_space_identity)
        )
        if len(matches) != 1:
            raise ValueError("评测 clone 缺少唯一同空间 M-07 resolver")
        cloned = MemoryHotSetRuntime(
            ctx,
            matches[0],
            ctx.tiered_segment_store,
            self.projection,
            self.policy,
        )
        if self._query_index_projection is not None:
            cloned.replace_query_index(self._query_index_projection)
        if cloned.state_key() != self.state_key():
            raise ValueError("memory hot-set clone 改变装配状态")
        return cloned

    def _resolve_request(
            self,
            request: MemoryActivationRequest,
            ) -> ResolvedCandidateSet:
        """按 kind/owner 范围流式恢复候选，并用有界策略维持精确 Top-K。"""
        if request.hypothesis_kind not in self.projection.hypothesis_kinds:
            raise MemoryProjectionError("投影未覆盖当前 Hypothesis kind")
        policy = self.resolver.diversity_policy
        if not isinstance(policy, BoundedCandidateSelectionPolicy):
            raise TypeError("resolver diversity policy 不再满足有界协议")
        if self._hot_set is None and self.projection.segments:
            raise RuntimeError("memory hot-set reader 尚未打开")
        filters = self._filters(request)
        if (self._query_index_projection is not None
                and filters == (MemoryAggregateFilter(),)):
            return self._resolve_indexed_request(request)

        def pin(record_key: tuple[int, ...]) -> None:
            """把进入有界候选集的冷记录固定在当前 query cache。"""
            if self._hot_set is None:
                raise RuntimeError("空投影不能 pin Memory 记录")
            self._hot_set.pin(record_key)

        def unpin(record_key: tuple[int, ...]) -> None:
            """在候选被更优项替换时立即释放其冷记录。"""
            if self._hot_set is None:
                raise RuntimeError("空投影不能 unpin Memory 记录")
            self._hot_set.unpin(record_key)

        accumulator = policy.new_accumulator(
            request,
            request.budget,
            pin=pin,
            unpin=unpin,
        )
        considered = 0
        for candidate in self.resolver.core_candidates(request):
            accumulator.offer(candidate, None)
            considered += 1
        kind_hash = self.resolver.aggregates.hypothesis_kind_hash(
            request.hypothesis_kind)
        if self._hot_set is not None:
            for owner_key in visible_owner_keys(request.access):
                lower, upper = memory_candidate_scan_range(
                    self.projection.projection_key,
                    kind_hash,
                    owner_key,
                )
                for cached in self._hot_set.iter_range(
                        lower_key=lower, upper_key=upper):
                    bundle = decode_memory_candidate(
                        self.projection.projection_key, cached.record)
                    if bundle.hypothesis.hypothesis_kind != request.hypothesis_kind:
                        continue
                    if bundle.aggregate.lifecycle_state != LIFECYCLE_ACTIVE:
                        continue
                    if not any(matches_memory_filter(bundle, item)
                               for item in filters):
                        continue
                    candidate = self.resolver.candidate_from_bundle(
                        request, bundle)
                    if candidate.origin_kind != RESOLUTION_ORIGIN_MEMORY:
                        raise RuntimeError("候选投影恢复出非 Memory origin")
                    accumulator.offer(candidate, cached.record.record_key)
                    considered += 1
        selected = accumulator.finish()
        if len(selected) != min(request.budget, considered):
            raise ValueError("有界选择策略未返回精确 Top-K 数量")
        return ResolvedCandidateSet(request, selected, considered)

    def _resolve_indexed_request(
            self,
            request: MemoryActivationRequest,
            ) -> ResolvedCandidateSet:
        """按 planner 完备组读取有界前缀，并保留完整分区逻辑考虑数。"""
        projection = self._query_index_projection
        if projection is None:
            raise RuntimeError("indexed request 缺少查询索引")
        if self._hot_set is None:
            raise RuntimeError("indexed request reader 尚未打开")
        planner = self.resolver.score_provider
        if not isinstance(planner, ExactMemoryProjectionPlanner):
            raise TypeError("indexed request scorer 未实现精确 planner")
        if strict_integer_tuple(
                planner.state_key(), label="indexed request planner_key"
                ) != projection.planner_key:
            raise MemoryProjectionError("indexed request planner 状态漂移")
        groups = planner.query_groups(request)
        if (not isinstance(groups, tuple)
                or not groups
                or any(not isinstance(item, MemoryProjectionQueryGroup)
                       for item in groups)):
            raise TypeError("精确 planner query_groups 返回类型错误")
        group_keys = tuple(item.group_key for item in groups)
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("精确 planner query group 不得重复")
        policy = self.resolver.diversity_policy
        if not isinstance(policy, BoundedCandidateSelectionPolicy):
            raise TypeError("indexed resolver diversity policy 不满足有界协议")

        def pin(record_key: tuple[int, ...]) -> None:
            """持有进入精确 Top-K 的查询索引记录。"""
            self._hot_set.pin(record_key)

        def unpin(record_key: tuple[int, ...]) -> None:
            """释放被更优候选替换的查询索引记录。"""
            self._hot_set.unpin(record_key)

        accumulator = policy.new_accumulator(
            request,
            request.budget,
            pin=pin,
            unpin=unpin,
        )
        core_candidates = self.resolver.core_candidates(request)
        seen = set()
        for candidate in core_candidates:
            accumulator.offer(candidate, None)
            seen.add((candidate.origin_kind, candidate.candidate_key))
        kind_hash = self.resolver.aggregates.hypothesis_kind_hash(
            request.hypothesis_kind)
        runs = projection.query_runs()
        later_invalidated: list[frozenset[int]] = [frozenset()] * len(runs)
        changed_after: set[int] = set()
        for run_index in range(len(runs) - 1, -1, -1):
            later_invalidated[run_index] = frozenset(changed_after)
            changed_after.update(runs[run_index].invalidated_hypothesis_hashes)
        logical_memory_count = 0
        for owner_key in visible_owner_keys(request.access):
            logical_memory_count += projection.partition_count(
                kind_hash, owner_key)
            for group in groups:
                for run_index, run in enumerate(runs):
                    invalidated = later_invalidated[run_index]
                    lower, upper = memory_query_index_scan_range(
                        run.storage.projection_key,
                        kind_hash,
                        owner_key,
                        group.group_key,
                    )
                    consumed = 0
                    physical_limit = group.limit + len(invalidated)
                    for cached in self._hot_set.iter_range(
                            lower_key=lower, upper_key=upper):
                        if consumed >= physical_limit:
                            break
                        consumed += 1
                        bundle = decode_memory_candidate_payload(cached.record)
                        if bundle.aggregate.hypothesis_hash in invalidated:
                            continue
                        if (bundle.hypothesis.hypothesis_kind
                                != request.hypothesis_kind
                                or bundle.aggregate.lifecycle_state
                                != LIFECYCLE_ACTIVE):
                            raise MemoryProjectionError(
                                "查询索引含错误 kind 或 inactive 候选")
                        entries = planner.index_entries(bundle)
                        expected = {
                            memory_query_index_record_key(
                                run.storage.projection_key,
                                bundle.aggregate,
                                entry,
                            )
                            for entry in entries
                            if entry.group_key == group.group_key
                        }
                        if cached.record.record_key not in expected:
                            raise MemoryProjectionError(
                                "查询索引 record key 与 planner entry 漂移")
                        candidate = self.resolver.candidate_from_bundle(
                            request, bundle)
                        candidate_key = (
                            candidate.origin_kind, candidate.candidate_key)
                        if candidate_key in seen:
                            continue
                        seen.add(candidate_key)
                        accumulator.offer(candidate, cached.record.record_key)
        considered = len(core_candidates) + logical_memory_count
        selected = accumulator.finish()
        if len(selected) != min(request.budget, considered):
            raise MemoryProjectionError("精确查询索引未恢复足量 Top-K")
        return ResolvedCandidateSet(request, selected, considered)

    def _filters(
            self,
            request: MemoryActivationRequest,
            ) -> tuple[MemoryAggregateFilter, ...]:
        """读取并核验与 M-07 全热路径相同的 OR-of-AND 过滤分支。"""
        filters = self.resolver.index_filter_provider.filters(request)
        if not isinstance(filters, tuple) or not filters:
            raise ValueError("Memory index filter provider 必须返回非空 tuple")
        if any(not isinstance(item, MemoryAggregateFilter) for item in filters):
            raise TypeError("Memory index filter provider 返回错误过滤分支")
        keys = tuple(item.stable_key() for item in filters)
        if len(set(keys)) != len(keys):
            raise ValueError("Memory index filter provider 不得返回重复分支")
        return filters

    def _require_fresh_projection(self) -> None:
        """要求新 query 使用的候选投影仍等于当前 Memory timeline 水位。"""
        current_state = (
            self.resolver.aggregates.event_log.projection_state_key())
        if self._query_index_projection is None:
            if current_state != self.projection.source_state_key:
                raise MemoryProjectionError(
                    "Memory 候选投影已因 event/batch/forget epoch 变化失效")
            return
        if (self._query_index_projection.storage.source_state_key
                != current_state):
            raise MemoryProjectionError("Memory 查询索引 source state 已失效")


def memory_hot_set_runtimes(
        ctx: TrainContext,
        ) -> tuple[MemoryHotSetRuntime, ...]:
    """返回按稳定 Memory 空间身份排序的全部 K-04 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("memory hot-set runtimes ctx 类型错误")
    candidates = (
        ctx.memory_read_hot_set_runtime,
        ctx.memory_interact_hot_set_runtime,
        ctx.memory_hot_set_runtime,
    )
    unique: dict[int, MemoryHotSetRuntime] = {}
    for item in candidates:
        if item is None:
            continue
        if not isinstance(item, MemoryHotSetRuntime):
            raise TypeError("TrainContext K-04 槽位类型错误")
        unique[id(item)] = item
    return tuple(sorted(
        unique.values(),
        key=lambda item: (
            item.resolver.aggregates.event_log.memory_space_identity.stable_key()),
    ))


def memory_hot_set_runtime_for(
        ctx: TrainContext,
        resolver: MemoryOverlayResolver,
        ) -> MemoryHotSetRuntime | None:
    """按 resolver 实例返回其独立 K-04 runtime，不跨 Memory 空间回退。"""
    if not isinstance(resolver, MemoryOverlayResolver):
        raise TypeError("memory hot-set lookup resolver 类型错误")
    matches = tuple(
        item for item in memory_hot_set_runtimes(ctx)
        if item.resolver is resolver
    )
    if len(matches) > 1:
        raise RuntimeError("同一 M-07 resolver 绑定了多个 K-04 runtime")
    return None if not matches else matches[0]


def memory_hot_sets_closed(ctx: TrainContext) -> bool:
    """核验当前上下文全部逐空间 K-04 query 资源均已关闭。"""
    return all(item.query_resources_closed()
               for item in memory_hot_set_runtimes(ctx))


def install_memory_hot_set_runtime(
        ctx: TrainContext,
        projection: MemoryCandidateProjectionManifest,
        policy: QueryHotSetPolicy,
        *,
        resolver: MemoryOverlayResolver | None = None,
        ) -> MemoryHotSetRuntime:
    """在 M-07/K-02 上为一个明确 Memory 空间安装唯一 K-04 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install memory hot-set ctx 类型错误")
    if ctx.memory_resolver_runtime is None or ctx.tiered_segment_store is None:
        raise ValueError("安装 K-04 前必须先安装 M-07 和 K-02 store")
    selected = ctx.memory_resolver_runtime.resolver if resolver is None else resolver
    if not isinstance(selected, MemoryOverlayResolver):
        raise TypeError("install memory hot-set resolver 类型错误")
    if all(item is not selected for item in ctx.memory_resolver_runtime.resolvers):
        raise ValueError("install memory hot-set resolver 不属于当前 M-07 联邦")
    if selected.aggregates is ctx.memory_read_aggregates:
        owner_field = "memory_read_hot_set_runtime"
    elif selected.aggregates is ctx.memory_interact_aggregates:
        owner_field = "memory_interact_hot_set_runtime"
    else:
        raise ValueError("install memory hot-set resolver aggregate 不属于当前上下文")
    if getattr(ctx, owner_field) is not None:
        raise ValueError("该 Memory 空间已安装 hot-set runtime")
    primary = selected is ctx.memory_resolver_runtime.resolver
    if primary and ctx.memory_hot_set_runtime is not None:
        raise ValueError("TrainContext 已安装主 Memory hot-set runtime")
    runtime = MemoryHotSetRuntime(
        ctx,
        selected,
        ctx.tiered_segment_store,
        projection,
        policy,
    )
    setattr(ctx, owner_field, runtime)
    if primary:
        ctx.memory_hot_set_runtime = runtime
    return runtime


__all__ = [
    "MemoryCandidateProjectionManifest",
    "MemoryCandidateProjectionPublisher",
    "MemoryHotSetRuntime",
    "MemoryProjectionError",
    "MemoryProjectionPlacementDirective",
    "MemoryProjectionPlacementPublication",
    "MemoryProjectionPlacementReceipt",
    "MemoryProjectionPublication",
    "MemoryProjectionSegment",
    "MemoryQueryIndexChange",
    "MemoryQueryIndexPartition",
    "MemoryQueryIndexProjectionManifest",
    "MemoryQueryIndexRun",
    "install_memory_hot_set_runtime",
    "memory_hot_set_runtime_for",
    "memory_hot_set_runtimes",
    "memory_hot_sets_closed",
]
