"""K-03 稳定逻辑分片、worker 局部增量和确定性 barrier 合并。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.location_manifest import (
    LocationManifest,
    LocationManifestEntry,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SealedSegment,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_dependency import (
    SegmentDependency,
    canonical_dependencies,
)
from pure_integer_ai.storage.segment_repository import (
    AppendOnlyObjectRepository,
    SegmentRepositoryFaultInjector,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore


TRAINING_SHARD_FORMAT_VERSION = 1
OBJECT_KIND_TRAINING_WORKER_ARTIFACT = 4
OBJECT_KIND_TRAINING_BARRIER_RECEIPT = 5

FAULT_TRAINING_BARRIER_AFTER_MERGE = 1
FAULT_TRAINING_BARRIER_AFTER_PUBLISH = 2
FAULT_TRAINING_BARRIER_AFTER_RECEIPT = 3


class TrainingShardIntegrityError(RuntimeError):
    """训练 manifest、分片、增量、身份或 barrier 状态发生漂移。"""


class TrainingShardConflictError(TrainingShardIntegrityError):
    """同一 merge identity 收到不一致内容。"""


@runtime_checkable
class TrainingBarrierFaultInjector(Protocol):
    """在 K-03 barrier 可见性边界注入故障。"""

    def hit(self, point: int, context: dict[str, int]) -> None:
        """观察边界，需要中断时由实现抛出异常。"""
        ...


def _hit(
        injector: TrainingBarrierFaultInjector | None,
        point: int,
        context: dict[str, int],
        ) -> None:
    """调用可选 barrier 故障注入器并复制上下文。"""
    if injector is None:
        return
    if not isinstance(injector, TrainingBarrierFaultInjector):
        raise TypeError("training barrier fault injector 协议错误")
    injector.hit(point, dict(context))


def _key(
        value: tuple[int, ...], *, label: str, empty: bool = False,
        ) -> tuple[int, ...]:
    """核验开放协议键为严格整数 tuple。"""
    return strict_integer_tuple(value, label=label, empty=empty)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    """返回一个带长度边界的整数键。"""
    return (len(values), *values)


def _digest_key(values: tuple[int, ...]) -> tuple[int, ...]:
    """返回规范整数流的完整 SHA-256 字节键。"""
    return tuple(hashlib.sha256(encode_integer_tuple(values)).digest())


def training_base_manifest_state_key(
        manifest: LocationManifest | None,
        ) -> tuple[int, ...]:
    """返回紧凑且绑定完整身份与内容校验的 location manifest 状态键。"""
    result = [TRAINING_SHARD_FORMAT_VERSION]
    if manifest is None:
        result.append(0)
    else:
        result.append(1)
        pack_key(result, manifest.manifest_key)
        pack_key(result, manifest.profile_key)
        result.extend((
            manifest.publish_epoch,
            0 if manifest.previous_epoch is None else manifest.previous_epoch,
            len(manifest.entries),
        ))
        pack_key(result, _digest_key(manifest.stable_key()))
    return tuple(result)


@dataclass(frozen=True, order=True)
class TrainingManifestEntry:
    """冻结训练输入的来源、scope、顺序和内容校验身份。"""

    input_key: tuple[int, ...]
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    course_seq: int
    source_seq: int
    checksum_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验输入身份和两个冻结逻辑序。"""
        for label, value in (
                ("input_key", self.input_key),
                ("source_key", self.source_key),
                ("scope_key", self.scope_key),
                ("checksum_key", self.checksum_key)):
            _key(value, label=f"TrainingManifestEntry.{label}")
        if (type(self.course_seq) is not int or self.course_seq < 0
                or type(self.source_seq) is not int or self.source_seq < 0):
            raise ValueError("训练 manifest 顺序必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回输入项的完整稳定键。"""
        return (
            *_pack(self.input_key),
            *_pack(self.source_key),
            *_pack(self.scope_key),
            self.course_seq,
            self.source_seq,
            *_pack(self.checksum_key),
        )


@dataclass(frozen=True)
class FrozenTrainingManifest:
    """一次训练 barrier 使用的冻结 source/course 最小 manifest。"""

    manifest_key: tuple[int, ...]
    version_key: tuple[int, ...]
    entries: tuple[TrainingManifestEntry, ...]

    def __post_init__(self) -> None:
        """规范输入顺序并拒绝身份、课程序和来源序重复。"""
        _key(self.manifest_key, label="FrozenTrainingManifest.manifest_key")
        _key(self.version_key, label="FrozenTrainingManifest.version_key")
        if (not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(item, TrainingManifestEntry)
                       for item in self.entries)):
            raise ValueError("冻结训练 manifest 必须包含输入项")
        ordered = tuple(sorted(self.entries, key=lambda item: (
            item.course_seq,
            item.source_seq,
            item.source_key,
            item.scope_key,
            item.input_key,
        )))
        if len({item.input_key for item in ordered}) != len(ordered):
            raise ValueError("训练 manifest input_key 不得重复")
        order_keys = tuple(
            (item.course_seq, item.source_seq, item.input_key)
            for item in ordered
        )
        if len(set(order_keys)) != len(order_keys):
            raise ValueError("训练 manifest 逻辑顺序身份不得重复")
        object.__setattr__(self, "entries", ordered)

    def entry(self, input_key: tuple[int, ...]) -> TrainingManifestEntry:
        """按完整输入身份读取唯一冻结项。"""
        key = _key(input_key, label="training manifest input lookup")
        matches = tuple(item for item in self.entries if item.input_key == key)
        if len(matches) != 1:
            raise KeyError(f"训练 manifest 缺少唯一 input_key: {key}")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 全部内容的规范稳定键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.manifest_key)
        pack_key(result, self.version_key)
        result.append(len(self.entries))
        for entry in self.entries:
            key = entry.stable_key()
            result.extend((len(key), *key))
        return tuple(result)

    @property
    def checksum_key(self) -> tuple[int, ...]:
        """返回冻结 manifest 的完整内容校验键。"""
        return _digest_key(self.stable_key())

    def state_key(self) -> tuple[int, ...]:
        """返回身份、版本、规模和内容校验组成的紧凑状态键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.manifest_key)
        pack_key(result, self.version_key)
        result.append(len(self.entries))
        pack_key(result, self.checksum_key)
        return tuple(result)


@dataclass(frozen=True, order=True)
class LogicalTrainingShard:
    """与执行 worker 数无关的一个逻辑输入分片。"""

    shard_key: tuple[int, ...]
    input_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验 shard 身份及其中唯一输入集合。"""
        _key(self.shard_key, label="LogicalTrainingShard.shard_key")
        if not isinstance(self.input_keys, tuple) or not self.input_keys:
            raise ValueError("逻辑 shard 必须包含输入")
        normalized = tuple(
            _key(value, label="LogicalTrainingShard.input_key")
            for value in self.input_keys
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("逻辑 shard input_key 不得重复")


@dataclass(frozen=True)
class TrainingWorkerAssignment:
    """仅用于调度的 worker 到逻辑 shard 分配。"""

    worker_index: int
    shard_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验 worker 序号和 shard 键。"""
        if type(self.worker_index) is not int or self.worker_index < 0:
            raise ValueError("worker_index 必须是非负严格整数")
        if not isinstance(self.shard_keys, tuple):
            raise TypeError("worker shard_keys 必须是 tuple")
        for value in self.shard_keys:
            _key(value, label="TrainingWorkerAssignment.shard_key")


@dataclass(frozen=True)
class LogicalShardPlan:
    """冻结逻辑 shard 身份并把 worker 数降为纯调度参数。"""

    plan_key: tuple[int, ...]
    manifest_key: tuple[int, ...]
    shards: tuple[LogicalTrainingShard, ...]

    def __post_init__(self) -> None:
        """规范 shard 顺序并拒绝 shard 或输入重复。"""
        _key(self.plan_key, label="LogicalShardPlan.plan_key")
        _key(self.manifest_key, label="LogicalShardPlan.manifest_key")
        if (not isinstance(self.shards, tuple) or not self.shards
                or any(not isinstance(item, LogicalTrainingShard)
                       for item in self.shards)):
            raise ValueError("逻辑 shard plan 必须包含 shard")
        ordered = tuple(sorted(self.shards, key=lambda item: item.shard_key))
        if len({item.shard_key for item in ordered}) != len(ordered):
            raise ValueError("逻辑 shard_key 不得重复")
        flattened = tuple(
            input_key for shard in ordered for input_key in shard.input_keys)
        if len(set(flattened)) != len(flattened):
            raise ValueError("同一输入不得跨逻辑 shard 重复")
        object.__setattr__(self, "shards", ordered)

    def validate_manifest(self, manifest: FrozenTrainingManifest) -> None:
        """要求 shard 计划恰好覆盖冻结 manifest 的全部输入。"""
        if not isinstance(manifest, FrozenTrainingManifest):
            raise TypeError("manifest 必须是 FrozenTrainingManifest")
        if self.manifest_key != manifest.manifest_key:
            raise TrainingShardIntegrityError("shard plan 绑定的 manifest 漂移")
        planned = {
            input_key for shard in self.shards for input_key in shard.input_keys
        }
        expected = {item.input_key for item in manifest.entries}
        if planned != expected:
            raise TrainingShardIntegrityError("shard plan 未完整且唯一覆盖 manifest")

    def shard(self, shard_key: tuple[int, ...]) -> LogicalTrainingShard:
        """按完整键读取唯一逻辑 shard。"""
        key = _key(shard_key, label="logical shard lookup")
        matches = tuple(item for item in self.shards if item.shard_key == key)
        if len(matches) != 1:
            raise KeyError(f"逻辑 shard 不存在: {key}")
        return matches[0]

    def assignments(self, worker_count: int) -> tuple[TrainingWorkerAssignment, ...]:
        """确定性分配 shard；worker 数不进入 shard 或记录身份。"""
        if type(worker_count) is not int or worker_count <= 0:
            raise ValueError("worker_count 必须是正严格整数")
        buckets: list[list[tuple[int, ...]]] = [
            [] for _ in range(worker_count)
        ]
        for index, shard in enumerate(self.shards):
            buckets[index % worker_count].append(shard.shard_key)
        return tuple(
            TrainingWorkerAssignment(index, tuple(values))
            for index, values in enumerate(buckets)
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回与 worker 数无关的完整 plan 键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.plan_key)
        pack_key(result, self.manifest_key)
        result.append(len(self.shards))
        for shard in self.shards:
            pack_key(result, shard.shard_key)
            result.append(len(shard.input_keys))
            for input_key in shard.input_keys:
                pack_key(result, input_key)
        return tuple(result)

    @property
    def checksum_key(self) -> tuple[int, ...]:
        """返回完整逻辑 shard plan 的内容校验键。"""
        return _digest_key(self.stable_key())

    def state_key(self) -> tuple[int, ...]:
        """返回计划身份、manifest 身份、规模和内容校验状态键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.plan_key)
        pack_key(result, self.manifest_key)
        result.append(len(self.shards))
        pack_key(result, self.checksum_key)
        return tuple(result)


@dataclass(frozen=True, order=True)
class TrainingAllocationFloor:
    """一个分配 scope 在 base fence 上的已占用 local id 水位。"""

    allocation_scope_key: tuple[int, ...]
    local_id_floor: int

    def __post_init__(self) -> None:
        """核验分配 scope 和非负水位。"""
        _key(
            self.allocation_scope_key,
            label="TrainingAllocationFloor.allocation_scope_key",
        )
        if type(self.local_id_floor) is not int or self.local_id_floor < 0:
            raise ValueError("local_id_floor 必须是非负严格整数")


@dataclass(frozen=True)
class TrainingBaseReadFence:
    """worker 共读 base 的 manifest、版本、resolver 和逻辑水位快照。"""

    manifest_epoch: int
    manifest_state_key: tuple[int, ...]
    version_key: tuple[int, ...]
    identity_resolver_key: tuple[int, ...]
    allocation_floors: tuple[TrainingAllocationFloor, ...]
    timeline_floor: int

    def __post_init__(self) -> None:
        """核验 base fence 完整且各分配 scope 唯一。"""
        if type(self.manifest_epoch) is not int or self.manifest_epoch < 0:
            raise ValueError("base manifest_epoch 必须是非负严格整数")
        for label, value in (
                ("manifest_state_key", self.manifest_state_key),
                ("version_key", self.version_key),
                ("identity_resolver_key", self.identity_resolver_key)):
            _key(value, label=f"TrainingBaseReadFence.{label}")
        if (not isinstance(self.allocation_floors, tuple)
                or any(not isinstance(item, TrainingAllocationFloor)
                       for item in self.allocation_floors)):
            raise TypeError("allocation_floors 必须是 TrainingAllocationFloor tuple")
        ordered = tuple(sorted(self.allocation_floors))
        if len({item.allocation_scope_key for item in ordered}) != len(ordered):
            raise ValueError("allocation floor scope 不得重复")
        object.__setattr__(self, "allocation_floors", ordered)
        if type(self.timeline_floor) is not int or self.timeline_floor < 0:
            raise ValueError("timeline_floor 必须是非负严格整数")

    def floor(self, allocation_scope_key: tuple[int, ...]) -> int:
        """读取一个已冻结分配 scope 的 local id 水位。"""
        key = _key(allocation_scope_key, label="allocation floor lookup")
        matches = tuple(
            item.local_id_floor for item in self.allocation_floors
            if item.allocation_scope_key == key
        )
        if len(matches) != 1:
            raise TrainingShardIntegrityError("base fence 缺少唯一 allocation floor")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回 base fence 的完整稳定键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION, self.manifest_epoch]
        pack_key(result, self.manifest_state_key)
        pack_key(result, self.version_key)
        pack_key(result, self.identity_resolver_key)
        result.append(len(self.allocation_floors))
        for item in self.allocation_floors:
            pack_key(result, item.allocation_scope_key)
            result.append(item.local_id_floor)
        result.append(self.timeline_floor)
        return tuple(result)


@dataclass(frozen=True, order=True)
class TrainingExternalReference:
    """merge 前仅使用分配 scope 和完整外部身份表达引用。"""

    allocation_scope_key: tuple[int, ...]
    external_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验引用的两个开放身份键。"""
        _key(
            self.allocation_scope_key,
            label="TrainingExternalReference.allocation_scope_key",
        )
        _key(self.external_key, label="TrainingExternalReference.external_key")


@dataclass(frozen=True)
class TrainingDeltaRecord:
    """worker 产生的无共享 local id、无完成时序依赖的逻辑增量。"""

    input_key: tuple[int, ...]
    merge_key: tuple[int, ...]
    allocation_scope_key: tuple[int, ...]
    external_object_key: tuple[int, ...]
    references: tuple[TrainingExternalReference, ...]
    logical_seq: int
    payload: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 merge identity、可选对象分配和外部引用。"""
        _key(self.input_key, label="TrainingDeltaRecord.input_key")
        _key(self.merge_key, label="TrainingDeltaRecord.merge_key")
        _key(
            self.allocation_scope_key,
            label="TrainingDeltaRecord.allocation_scope_key",
            empty=True,
        )
        _key(
            self.external_object_key,
            label="TrainingDeltaRecord.external_object_key",
            empty=True,
        )
        if bool(self.allocation_scope_key) != bool(self.external_object_key):
            raise ValueError("对象分配 scope 与 external key 必须同时存在或同时为空")
        if (not isinstance(self.references, tuple)
                or any(not isinstance(item, TrainingExternalReference)
                       for item in self.references)):
            raise TypeError("references 必须是 TrainingExternalReference tuple")
        if len(set(self.references)) != len(self.references):
            raise ValueError("同一增量记录引用不得重复")
        if type(self.logical_seq) is not int or self.logical_seq < 0:
            raise ValueError("logical_seq 必须是非负严格整数")
        _key(self.payload, label="TrainingDeltaRecord.payload", empty=True)

    def content_key(self) -> tuple[int, ...]:
        """返回跨重复投递比较时不含来源顺序的逻辑内容键。"""
        result: list[int] = []
        pack_key(result, self.allocation_scope_key)
        pack_key(result, self.external_object_key)
        result.append(len(self.references))
        for reference in self.references:
            pack_key(result, reference.allocation_scope_key)
            pack_key(result, reference.external_key)
        pack_key(result, self.payload)
        return tuple(result)

    def to_segment_record(
            self,
            entry: TrainingManifestEntry,
            ) -> SegmentRecord:
        """把逻辑增量编码为 worker sealed segment 记录。"""
        if not isinstance(entry, TrainingManifestEntry):
            raise TypeError("entry 必须是 TrainingManifestEntry")
        if entry.input_key != self.input_key:
            raise TrainingShardIntegrityError("增量 input_key 与 manifest entry 漂移")
        record_key = [
            TRAINING_SHARD_FORMAT_VERSION,
            entry.course_seq,
            entry.source_seq,
        ]
        for value in (
                entry.input_key,
                entry.source_key,
                entry.scope_key,
                self.merge_key):
            pack_key(record_key, value)
        record_key.append(self.logical_seq)
        payload = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(payload, self.allocation_scope_key)
        pack_key(payload, self.external_object_key)
        payload.append(len(self.references))
        for reference in self.references:
            pack_key(payload, reference.allocation_scope_key)
            pack_key(payload, reference.external_key)
        pack_key(payload, self.payload)
        return SegmentRecord(tuple(record_key), tuple(payload))

    @classmethod
    def from_segment_record(
            cls,
            record: SegmentRecord,
            manifest: FrozenTrainingManifest,
            ) -> tuple[TrainingDeltaRecord, TrainingManifestEntry]:
        """从 worker segment 恢复增量，并回验冻结来源和顺序。"""
        if not isinstance(record, SegmentRecord):
            raise TypeError("record 必须是 SegmentRecord")
        if not isinstance(manifest, FrozenTrainingManifest):
            raise TypeError("manifest 必须是 FrozenTrainingManifest")
        try:
            key_reader = IntegerStreamReader(record.record_key)
            if key_reader.read_positive(
                    label="training delta key version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("training delta key 版本未知")
            course_seq = key_reader.read_nonnegative(label="training delta course_seq")
            source_seq = key_reader.read_nonnegative(label="training delta source_seq")
            input_key = key_reader.read_key(label="training delta input_key")
            source_key = key_reader.read_key(label="training delta source_key")
            scope_key = key_reader.read_key(label="training delta scope_key")
            merge_key = key_reader.read_key(label="training delta merge_key")
            logical_seq = key_reader.read_nonnegative(label="training delta logical_seq")
            key_reader.finish()
            entry = manifest.entry(input_key)
            if (entry.course_seq != course_seq
                    or entry.source_seq != source_seq
                    or entry.source_key != source_key
                    or entry.scope_key != scope_key):
                raise TrainingShardIntegrityError("worker delta 的 manifest 投影漂移")
            payload_reader = IntegerStreamReader(record.payload)
            if payload_reader.read_positive(
                    label="training delta payload version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("training delta payload 版本未知")
            allocation_scope_key = payload_reader.read_key(
                label="training delta allocation scope", empty=True)
            external_object_key = payload_reader.read_key(
                label="training delta external object", empty=True)
            reference_count = payload_reader.read_nonnegative(
                label="training delta reference count")
            references = []
            for _ in range(reference_count):
                references.append(TrainingExternalReference(
                    payload_reader.read_key(
                        label="training delta reference scope"),
                    payload_reader.read_key(
                        label="training delta reference external"),
                ))
            payload = payload_reader.read_key(
                label="training delta business payload", empty=True)
            payload_reader.finish()
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, TrainingShardIntegrityError):
                raise
            raise TrainingShardIntegrityError("worker delta record 编码损坏") from exc
        return cls(
            input_key,
            merge_key,
            allocation_scope_key,
            external_object_key,
            tuple(references),
            logical_seq,
            payload,
        ), entry


def worker_segment_key(
        barrier_key: tuple[int, ...], shard_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """从 barrier 和逻辑 shard 派生与 worker 数无关的 segment 键。"""
    result = [TRAINING_SHARD_FORMAT_VERSION]
    pack_key(result, _key(barrier_key, label="worker barrier_key"))
    pack_key(result, _key(shard_key, label="worker shard_key"))
    return tuple(result)


def worker_artifact_identity(
        *,
        manifest_key: tuple[int, ...],
        plan_key: tuple[int, ...],
        execution_key: tuple[int, ...],
        barrier_key: tuple[int, ...],
        shard_key: tuple[int, ...],
        base_fence_key: tuple[int, ...],
        descriptor_key: tuple[int, ...],
        segment_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    """返回 worker artifact 槽位身份，载荷漂移由仓库拒绝。"""
    result = [TRAINING_SHARD_FORMAT_VERSION]
    for value in (
            manifest_key, plan_key, execution_key, barrier_key, shard_key,
            base_fence_key, descriptor_key, segment_key):
        pack_key(result, _key(value, label="worker artifact identity field"))
    return tuple(result)


@dataclass(frozen=True)
class WorkerDeltaArtifact:
    """一个逻辑 shard 已完成的可恢复 worker 增量或空完成凭据。"""

    manifest_key: tuple[int, ...]
    manifest_state_key: tuple[int, ...]
    plan_key: tuple[int, ...]
    plan_state_key: tuple[int, ...]
    producer_key: tuple[int, ...]
    execution_key: tuple[int, ...]
    barrier_key: tuple[int, ...]
    shard_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    descriptor_key: tuple[int, ...]
    version_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    dependencies: tuple[SegmentDependency, ...]
    segment: SealedSegment | None

    def __post_init__(self) -> None:
        """核验 artifact 元数据与可选 sealed segment 完全一致。"""
        for label, value in (
                ("manifest_key", self.manifest_key),
                ("manifest_state_key", self.manifest_state_key),
                ("plan_key", self.plan_key),
                ("plan_state_key", self.plan_state_key),
                ("producer_key", self.producer_key),
                ("execution_key", self.execution_key),
                ("barrier_key", self.barrier_key),
                ("shard_key", self.shard_key),
                ("base_fence_key", self.base_fence_key),
                ("descriptor_key", self.descriptor_key),
                ("version_key", self.version_key),
                ("segment_key", self.segment_key)):
            _key(value, label=f"WorkerDeltaArtifact.{label}")
        dependencies = canonical_dependencies(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        if self.segment is not None:
            if not isinstance(self.segment, SealedSegment):
                raise TypeError("artifact segment 类型错误")
            if (self.segment.descriptor_key != self.descriptor_key
                    or self.segment.segment_key != self.segment_key
                    or self.segment.version_key != self.version_key
                    or self.segment.dependencies != dependencies):
                raise TrainingShardIntegrityError("artifact segment 身份漂移")

    @property
    def identity_key(self) -> tuple[int, ...]:
        """返回 append-only worker artifact 的完整槽位身份。"""
        return worker_artifact_identity(
            manifest_key=self.manifest_key,
            plan_key=self.plan_key,
            execution_key=self.execution_key,
            barrier_key=self.barrier_key,
            shard_key=self.shard_key,
            base_fence_key=self.base_fence_key,
            descriptor_key=self.descriptor_key,
            segment_key=self.segment_key,
        )

    def to_bytes(self) -> bytes:
        """把空或非空 worker artifact 编为规范整数流。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        for value in (
                self.manifest_key, self.manifest_state_key,
                self.plan_key, self.plan_state_key,
                self.producer_key, self.execution_key,
                self.barrier_key, self.shard_key,
                self.base_fence_key, self.descriptor_key, self.version_key,
                self.segment_key):
            pack_key(result, value)
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            pack_key(result, dependency.descriptor_key)
            pack_key(result, dependency.version_key)
            pack_key(result, dependency.checksum_key)
        if self.segment is None:
            result.append(0)
        else:
            payload = self.segment.to_bytes()
            result.extend((1, len(payload), *payload))
        return encode_integer_tuple(tuple(result))

    @classmethod
    def from_bytes(cls, data: bytes) -> WorkerDeltaArtifact:
        """恢复 worker artifact，并重新核验 segment 身份。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            if reader.read_positive(
                    label="worker artifact version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("worker artifact 版本未知")
            fields = tuple(
                reader.read_key(label="worker artifact identity field")
                for _ in range(12)
            )
            dependency_count = reader.read_nonnegative(
                label="worker artifact dependency count")
            dependencies = tuple(
                SegmentDependency(
                    reader.read_key(
                        label="worker artifact dependency descriptor"),
                    reader.read_key(
                        label="worker artifact dependency version"),
                    reader.read_key(
                        label="worker artifact dependency checksum"),
                )
                for _ in range(dependency_count)
            )
            has_segment = reader.read_nonnegative(label="worker artifact has_segment")
            if has_segment not in {0, 1}:
                raise TrainingShardIntegrityError("worker artifact segment 标志非法")
            segment = None
            if has_segment:
                size = reader.read_positive(label="worker artifact segment size")
                payload = []
                for _ in range(size):
                    value = reader.read_nonnegative(
                        label="worker artifact segment byte")
                    if value > 255:
                        raise TrainingShardIntegrityError("worker artifact byte 越界")
                    payload.append(value)
                segment = SealedSegment.from_bytes(bytes(payload))
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, TrainingShardIntegrityError):
                raise
            raise TrainingShardIntegrityError("worker artifact 编码损坏") from exc
        return cls(*fields, dependencies, segment)

    def persist(
            self,
            repository: AppendOnlyObjectRepository,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> None:
        """按稳定槽位幂等持久化 artifact，重试载荷漂移必须失败。"""
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("worker artifact repository 协议错误")
        repository.put(
            OBJECT_KIND_TRAINING_WORKER_ARTIFACT,
            self.identity_key,
            self.to_bytes(),
            fault_injector=fault_injector,
        )

    @classmethod
    def restore(
            cls,
            repository: AppendOnlyObjectRepository,
            identity_key: tuple[int, ...],
            ) -> WorkerDeltaArtifact:
        """按完整槽位读取并核验 worker artifact。"""
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("worker artifact repository 协议错误")
        artifact = cls.from_bytes(repository.get(
            OBJECT_KIND_TRAINING_WORKER_ARTIFACT,
            _key(identity_key, label="worker artifact restore identity"),
        ))
        if artifact.identity_key != identity_key:
            raise TrainingShardIntegrityError("worker artifact payload 与身份漂移")
        return artifact


class WorkerLocalDelta:
    """只服务一个逻辑 shard 的预算化开放增量。"""

    def __init__(
            self,
            *,
            manifest: FrozenTrainingManifest,
            shard_plan: LogicalShardPlan,
            shard: LogicalTrainingShard,
            producer_key: tuple[int, ...],
            execution_key: tuple[int, ...],
            barrier_key: tuple[int, ...],
            base_fence: TrainingBaseReadFence,
            descriptor_key: tuple[int, ...],
            version_key: tuple[int, ...],
            dependencies: tuple[SegmentDependency, ...],
            budget: SegmentBudget,
            ) -> None:
        """绑定冻结输入、逻辑 shard、base fence 和 worker 热预算。"""
        if not isinstance(manifest, FrozenTrainingManifest):
            raise TypeError("manifest 必须是 FrozenTrainingManifest")
        if not isinstance(shard_plan, LogicalShardPlan):
            raise TypeError("shard_plan 必须是 LogicalShardPlan")
        shard_plan.validate_manifest(manifest)
        if not isinstance(shard, LogicalTrainingShard):
            raise TypeError("shard 必须是 LogicalTrainingShard")
        if shard_plan.shard(shard.shard_key) != shard:
            raise TrainingShardIntegrityError("worker shard 与冻结 plan 漂移")
        if not isinstance(base_fence, TrainingBaseReadFence):
            raise TypeError("base_fence 必须是 TrainingBaseReadFence")
        self.manifest = manifest
        self.shard_plan = shard_plan
        self.shard = shard
        self.producer_key = _key(
            producer_key, label="worker delta producer_key")
        self.execution_key = _key(execution_key, label="worker delta execution_key")
        self.barrier_key = _key(barrier_key, label="worker delta barrier_key")
        self.base_fence = base_fence
        self.descriptor_key = _key(
            descriptor_key, label="worker delta descriptor_key")
        self.version_key = _key(version_key, label="worker delta version_key")
        self.dependencies = canonical_dependencies(dependencies)
        self.delta = OpenHotDelta(
            self.descriptor_key,
            self.version_key,
            self.dependencies,
            budget,
        )
        self._input_keys = frozenset(shard.input_keys)

    def append(self, record: TrainingDeltaRecord) -> bool:
        """追加属于当前逻辑 shard 的增量记录。"""
        if not isinstance(record, TrainingDeltaRecord):
            raise TypeError("record 必须是 TrainingDeltaRecord")
        if record.input_key not in self._input_keys:
            raise TrainingShardIntegrityError("worker 不得写其他逻辑 shard 的输入")
        entry = self.manifest.entry(record.input_key)
        return self.delta.append(record.to_segment_record(entry))

    def seal(self) -> WorkerDeltaArtifact:
        """形成可持久化 artifact；空 delta 形成独立完成凭据。"""
        segment_key = worker_segment_key(
            self.barrier_key,
            self.shard.shard_key,
        )
        segment = None
        if self.delta.object_count:
            segment = self.delta.seal(
                segment_key,
                self.base_fence.manifest_epoch,
            )
        return WorkerDeltaArtifact(
            self.manifest.manifest_key,
            self.manifest.state_key(),
            self.shard_plan.plan_key,
            self.shard_plan.state_key(),
            self.producer_key,
            self.execution_key,
            self.barrier_key,
            self.shard.shard_key,
            self.base_fence.stable_key(),
            self.descriptor_key,
            self.version_key,
            segment_key,
            self.dependencies,
            segment,
        )


@runtime_checkable
class TrainingIdentityResolver(Protocol):
    """按 base read fence 查询既有完整外部身份。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 resolver 数据与逻辑的完整稳定键。"""
        ...

    def resolve(
            self,
            allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        """返回既有正 local id，未命中时返回 None。"""
        ...


@dataclass(frozen=True, order=True)
class TrainingIdentityAssignment:
    """barrier merge 后确定的外部身份到 local id 映射。"""

    allocation_scope_key: tuple[int, ...]
    external_key: tuple[int, ...]
    local_id: int
    existed_in_base: bool

    def __post_init__(self) -> None:
        """核验映射身份、正 local id 和来源标志。"""
        _key(
            self.allocation_scope_key,
            label="TrainingIdentityAssignment.allocation_scope_key",
        )
        _key(self.external_key, label="TrainingIdentityAssignment.external_key")
        if type(self.local_id) is not int or self.local_id <= 0:
            raise ValueError("assigned local_id 必须是正严格整数")
        if type(self.existed_in_base) is not bool:
            raise TypeError("existed_in_base 必须是 bool")


@dataclass(frozen=True, order=True)
class ResolvedTrainingReference:
    """merge 后保留分配 scope 的正 local-id 引用。"""

    allocation_scope_key: tuple[int, ...]
    local_id: int

    def __post_init__(self) -> None:
        """核验引用 scope 和其中的正 local id。"""
        _key(
            self.allocation_scope_key,
            label="ResolvedTrainingReference.allocation_scope_key",
        )
        if type(self.local_id) is not int or self.local_id <= 0:
            raise ValueError("resolved reference local_id 必须是正严格整数")


@dataclass(frozen=True)
class MergedTrainingRecord:
    """已分配 local id、重映射引用和统一 timeline 的 canonical 记录。"""

    barrier_key: tuple[int, ...]
    merge_key: tuple[int, ...]
    input_key: tuple[int, ...]
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    timeline_seq: int
    allocation_scope_key: tuple[int, ...]
    external_object_key: tuple[int, ...]
    assigned_local_id: int
    resolved_references: tuple[ResolvedTrainingReference, ...]
    payload: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 canonical 身份、timeline 和重映射整数。"""
        for label, value in (
                ("barrier_key", self.barrier_key),
                ("merge_key", self.merge_key),
                ("input_key", self.input_key),
                ("source_key", self.source_key),
                ("scope_key", self.scope_key)):
            _key(value, label=f"MergedTrainingRecord.{label}")
        if type(self.timeline_seq) is not int or self.timeline_seq <= 0:
            raise ValueError("merged timeline_seq 必须是正严格整数")
        if type(self.assigned_local_id) is not int or self.assigned_local_id < 0:
            raise ValueError("assigned_local_id 必须是非负严格整数")
        _key(
            self.allocation_scope_key,
            label="MergedTrainingRecord.allocation_scope_key",
            empty=True,
        )
        _key(
            self.external_object_key,
            label="MergedTrainingRecord.external_object_key",
            empty=True,
        )
        has_assignment = bool(self.allocation_scope_key)
        if (has_assignment != bool(self.external_object_key)
                or has_assignment != (self.assigned_local_id > 0)):
            raise ValueError("canonical 外部身份与 local id 字段组合非法")
        if (not isinstance(self.resolved_references, tuple)
                or any(not isinstance(item, ResolvedTrainingReference)
                       for item in self.resolved_references)):
            raise TypeError("resolved_references 类型错误")
        _key(self.payload, label="MergedTrainingRecord.payload", empty=True)

    def to_segment_record(self) -> SegmentRecord:
        """编码为最终 canonical barrier segment 记录。"""
        record_key = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(record_key, self.barrier_key)
        pack_key(record_key, self.scope_key)
        pack_key(record_key, self.merge_key)
        record_key.append(self.timeline_seq)
        payload = [TRAINING_SHARD_FORMAT_VERSION]
        for value in (self.input_key, self.source_key):
            pack_key(payload, value)
        pack_key(payload, self.allocation_scope_key)
        pack_key(payload, self.external_object_key)
        payload.extend((self.assigned_local_id, len(self.resolved_references)))
        for reference in self.resolved_references:
            pack_key(payload, reference.allocation_scope_key)
            payload.append(reference.local_id)
        pack_key(payload, self.payload)
        return SegmentRecord(tuple(record_key), tuple(payload))

    @classmethod
    def from_segment_record(cls, record: SegmentRecord) -> MergedTrainingRecord:
        """从 canonical segment 记录恢复可重建的完整身份和引用。"""
        if not isinstance(record, SegmentRecord):
            raise TypeError("record 必须是 SegmentRecord")
        try:
            key_reader = IntegerStreamReader(record.record_key)
            if key_reader.read_positive(
                    label="merged training key version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("merged training key 版本未知")
            barrier_key = key_reader.read_key(label="merged training barrier_key")
            scope_key = key_reader.read_key(label="merged training scope_key")
            merge_key = key_reader.read_key(label="merged training merge_key")
            timeline_seq = key_reader.read_positive(
                label="merged training timeline_seq")
            key_reader.finish()
            payload_reader = IntegerStreamReader(record.payload)
            if payload_reader.read_positive(
                    label="merged training payload version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("merged training payload 版本未知")
            input_key = payload_reader.read_key(label="merged training input_key")
            source_key = payload_reader.read_key(label="merged training source_key")
            allocation_scope_key = payload_reader.read_key(
                label="merged training allocation scope", empty=True)
            external_object_key = payload_reader.read_key(
                label="merged training external object", empty=True)
            assigned_local_id = payload_reader.read_nonnegative(
                label="merged training assigned local id")
            reference_count = payload_reader.read_nonnegative(
                label="merged training reference count")
            references = tuple(
                ResolvedTrainingReference(
                    payload_reader.read_key(
                        label="merged training reference scope"),
                    payload_reader.read_positive(
                        label="merged training reference local id"),
                )
                for _ in range(reference_count)
            )
            payload = payload_reader.read_key(
                label="merged training business payload", empty=True)
            payload_reader.finish()
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, TrainingShardIntegrityError):
                raise
            raise TrainingShardIntegrityError(
                "merged training record 编码损坏") from exc
        return cls(
            barrier_key,
            merge_key,
            input_key,
            source_key,
            scope_key,
            timeline_seq,
            allocation_scope_key,
            external_object_key,
            assigned_local_id,
            references,
            payload,
        )


@dataclass(frozen=True)
class TrainingBarrierMetrics:
    """不读墙钟的 barrier 对象数、冷热字节和去重计量。"""

    logical_shards: int
    worker_artifacts: int
    empty_artifacts: int
    raw_records: int
    merged_records: int
    duplicate_records: int
    worker_segment_bytes: int
    canonical_segment_bytes: int


@dataclass(frozen=True)
class TrainingBarrierResult:
    """一次完整 merge 的 canonical segment、身份映射和资源报告。"""

    barrier_key: tuple[int, ...]
    execution_key: tuple[int, ...]
    context_key: tuple[int, ...]
    segment: SealedSegment | None
    assignments: tuple[TrainingIdentityAssignment, ...]
    metrics: TrainingBarrierMetrics

    def __post_init__(self) -> None:
        """核验 barrier 身份、可选 segment 和确定性 assignment 顺序。"""
        _key(self.barrier_key, label="TrainingBarrierResult.barrier_key")
        _key(self.execution_key, label="TrainingBarrierResult.execution_key")
        _key(self.context_key, label="TrainingBarrierResult.context_key")
        if self.segment is not None and not isinstance(self.segment, SealedSegment):
            raise TypeError("barrier result segment 类型错误")
        if (not isinstance(self.assignments, tuple)
                or any(not isinstance(item, TrainingIdentityAssignment)
                       for item in self.assignments)
                or tuple(sorted(self.assignments)) != self.assignments):
            raise ValueError("barrier assignments 必须规范排序")
        if not isinstance(self.metrics, TrainingBarrierMetrics):
            raise TypeError("barrier metrics 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回与 worker 数和完成顺序无关的结果键。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.barrier_key)
        pack_key(result, self.execution_key)
        pack_key(result, self.context_key)
        if self.segment is None:
            result.append(0)
        else:
            result.append(1)
            pack_key(result, self.segment.checksum_key)
        result.append(len(self.assignments))
        for item in self.assignments:
            pack_key(result, item.allocation_scope_key)
            pack_key(result, item.external_key)
            result.extend((item.local_id, int(item.existed_in_base)))
        return tuple(result)


@dataclass(frozen=True)
class TrainingBarrierPublishReceipt:
    """barrier 已完整发布或空提交的唯一可恢复 receipt。"""

    barrier_key: tuple[int, ...]
    execution_key: tuple[int, ...]
    result_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    segment_checksum_key: tuple[int, ...]
    manifest_epoch: int
    manifest_state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 receipt 身份及空提交和非空提交的字段组合。"""
        for label, value, empty in (
                ("barrier_key", self.barrier_key, False),
                ("execution_key", self.execution_key, False),
                ("result_key", self.result_key, False),
                ("segment_key", self.segment_key, True),
                ("segment_checksum_key", self.segment_checksum_key, True),
                ("manifest_state_key", self.manifest_state_key, True)):
            _key(value, label=f"TrainingBarrierPublishReceipt.{label}", empty=empty)
        if type(self.manifest_epoch) is not int or self.manifest_epoch < 0:
            raise ValueError("receipt manifest_epoch 必须是非负严格整数")
        has_segment = bool(self.segment_key)
        if (has_segment != bool(self.segment_checksum_key)
                or has_segment != bool(self.manifest_state_key)
                or has_segment != (self.manifest_epoch > 0)):
            raise ValueError("receipt 空提交与 manifest 字段组合非法")

    @property
    def identity_key(self) -> tuple[int, ...]:
        """返回 barrier receipt 的稳定槽位身份。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(result, self.barrier_key)
        pack_key(result, self.execution_key)
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把 receipt 编为规范整数流。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        for value in (
                self.barrier_key, self.execution_key, self.result_key,
                self.segment_key, self.segment_checksum_key,
                self.manifest_state_key):
            pack_key(result, value)
        result.append(self.manifest_epoch)
        return encode_integer_tuple(tuple(result))

    @classmethod
    def from_bytes(cls, data: bytes) -> TrainingBarrierPublishReceipt:
        """从规范字节恢复 barrier receipt。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            if reader.read_positive(
                    label="barrier receipt version") != TRAINING_SHARD_FORMAT_VERSION:
                raise TrainingShardIntegrityError("barrier receipt 版本未知")
            fields = tuple(
                reader.read_key(
                    label="barrier receipt key field",
                    empty=index >= 3,
                )
                for index in range(6)
            )
            epoch = reader.read_nonnegative(label="barrier receipt manifest epoch")
            reader.finish()
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, TrainingShardIntegrityError):
                raise
            raise TrainingShardIntegrityError("barrier receipt 编码损坏") from exc
        return cls(*fields[:5], epoch, fields[5])


class TrainingBarrierCoordinator:
    """在单协调器内核对 shard、去重冲突、分配 id 并发布 barrier。"""

    def __init__(
            self,
            *,
            manifest: FrozenTrainingManifest,
            shard_plan: LogicalShardPlan,
            base_fence: TrainingBaseReadFence,
            identity_resolver: TrainingIdentityResolver,
            producer_key: tuple[int, ...],
            execution_key: tuple[int, ...],
            barrier_key: tuple[int, ...],
            descriptor_key: tuple[int, ...],
            version_key: tuple[int, ...],
            dependencies: tuple[SegmentDependency, ...],
            output_budget: SegmentBudget,
            output_segment_key: tuple[int, ...],
            ) -> None:
        """绑定冻结输入、base fence、resolver 和单 barrier 输出预算。"""
        if not isinstance(manifest, FrozenTrainingManifest):
            raise TypeError("manifest 必须是 FrozenTrainingManifest")
        if not isinstance(shard_plan, LogicalShardPlan):
            raise TypeError("shard_plan 必须是 LogicalShardPlan")
        shard_plan.validate_manifest(manifest)
        if not isinstance(base_fence, TrainingBaseReadFence):
            raise TypeError("base_fence 必须是 TrainingBaseReadFence")
        if not isinstance(identity_resolver, TrainingIdentityResolver):
            raise TypeError("identity_resolver 协议错误")
        if not isinstance(output_budget, SegmentBudget):
            raise TypeError("output_budget 必须是 SegmentBudget")
        self.manifest = manifest
        self.shard_plan = shard_plan
        self.base_fence = base_fence
        self.identity_resolver = identity_resolver
        self.producer_key = _key(
            producer_key, label="coordinator producer_key")
        self.execution_key = _key(execution_key, label="coordinator execution_key")
        self.barrier_key = _key(barrier_key, label="coordinator barrier_key")
        self.descriptor_key = _key(
            descriptor_key, label="coordinator descriptor_key")
        self.version_key = _key(version_key, label="coordinator version_key")
        self.dependencies = canonical_dependencies(dependencies)
        self.output_budget = output_budget
        self.output_segment_key = _key(
            output_segment_key, label="coordinator output_segment_key")

    def _resolver_key(self) -> tuple[int, ...]:
        """读取并核验 resolver 当前稳定键。"""
        return _key(
            self.identity_resolver.state_key(),
            label="TrainingIdentityResolver.state_key",
        )

    def _context_key(self) -> tuple[int, ...]:
        """返回 barrier 结果必须绑定的全部冻结逻辑上下文。"""
        result = [TRAINING_SHARD_FORMAT_VERSION]
        for value in (
                self.manifest.state_key(),
                self.shard_plan.state_key(),
                self.base_fence.stable_key(),
                self.producer_key,
                self.descriptor_key,
                self.version_key,
                self.output_segment_key):
            pack_key(result, value)
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            pack_key(result, dependency.descriptor_key)
            pack_key(result, dependency.version_key)
            pack_key(result, dependency.checksum_key)
        return tuple(result)

    def merge(
            self,
            artifacts: tuple[WorkerDeltaArtifact, ...],
            ) -> TrainingBarrierResult:
        """确定性合并完整 shard artifact，缺失、漂移或冲突均失败。"""
        if (not isinstance(artifacts, tuple)
                or any(not isinstance(item, WorkerDeltaArtifact)
                       for item in artifacts)):
            raise TypeError("artifacts 必须是 WorkerDeltaArtifact tuple")
        if self._resolver_key() != self.base_fence.identity_resolver_key:
            raise TrainingShardIntegrityError("base identity resolver state 漂移")
        artifact_by_shard: dict[tuple[int, ...], WorkerDeltaArtifact] = {}
        for artifact in artifacts:
            self.validate_artifact(artifact)
            previous = artifact_by_shard.get(artifact.shard_key)
            if previous is not None and previous != artifact:
                raise TrainingShardConflictError("同一 shard 重复投递内容漂移")
            artifact_by_shard[artifact.shard_key] = artifact
        expected_shards = {item.shard_key for item in self.shard_plan.shards}
        if set(artifact_by_shard) != expected_shards:
            raise TrainingShardIntegrityError("barrier 缺少或夹带逻辑 shard artifact")
        return self._merge_artifact_stream(
            (
                (shard, artifact_by_shard[shard.shard_key])
                for shard in self.shard_plan.shards
            ),
            worker_artifact_count=len(artifact_by_shard),
        )

    def merge_stream(
            self,
            artifact_loader: Callable[
                [LogicalTrainingShard], WorkerDeltaArtifact
            ],
            ) -> TrainingBarrierResult:
        """按 plan 顺序逐 shard 冷读 artifact，避免全量 segment 常驻内存。"""
        if not callable(artifact_loader):
            raise TypeError("artifact_loader 必须可调用")
        if self._resolver_key() != self.base_fence.identity_resolver_key:
            raise TrainingShardIntegrityError("base identity resolver state 漂移")

        def loaded_artifacts():
            """逐 shard 加载并立即交给 merge，不保留已消费 segment。"""
            for shard in self.shard_plan.shards:
                artifact = artifact_loader(shard)
                if not isinstance(artifact, WorkerDeltaArtifact):
                    raise TypeError("artifact_loader 返回值类型错误")
                if artifact.shard_key != shard.shard_key:
                    raise TrainingShardIntegrityError(
                        "artifact_loader 返回了其他逻辑 shard")
                self.validate_artifact(artifact)
                yield shard, artifact
                artifact = None

        return self._merge_artifact_stream(
            loaded_artifacts(),
            worker_artifact_count=len(self.shard_plan.shards),
        )

    def _merge_artifact_stream(
            self,
            artifacts,
            *,
            worker_artifact_count: int,
            ) -> TrainingBarrierResult:
        """流式读取 artifact，并只保留受输出对象预算约束的唯一记录。"""
        unique: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[TrainingDeltaRecord, TrainingManifestEntry],
        ] = {}
        worker_segment_bytes = 0
        empty_artifacts = 0
        raw_records = 0
        duplicate_records = 0
        for shard, artifact in artifacts:
            if artifact.segment is None:
                empty_artifacts += 1
                artifact = None
                continue
            worker_segment_bytes += artifact.segment.size_bytes
            for segment_record in artifact.segment.records:
                raw_records += 1
                delta_record, entry = TrainingDeltaRecord.from_segment_record(
                    segment_record,
                    self.manifest,
                )
                if delta_record.input_key not in shard.input_keys:
                    raise TrainingShardIntegrityError("artifact 包含其他 shard 输入")
                merge_identity = (entry.scope_key, delta_record.merge_key)
                previous = unique.get(merge_identity)
                if previous is None:
                    if len(unique) >= self.output_budget.object_limit:
                        raise SegmentBudgetExceeded(
                            "barrier 唯一记录超过 canonical 对象预算")
                    unique[merge_identity] = (delta_record, entry)
                    continue
                previous_record, previous_entry = previous
                if previous_record.content_key() != delta_record.content_key():
                    raise TrainingShardConflictError("同一 merge identity 内容冲突")
                duplicate_records += 1
                if self._sort_key(delta_record, entry) < self._sort_key(
                        previous_record, previous_entry):
                    unique[merge_identity] = (delta_record, entry)
            artifact = None
        ordered = tuple(sorted(
            unique.values(),
            key=lambda item: self._sort_key(item[0], item[1]),
        ))
        assignments, assignment_map = self._assign_identities(ordered)
        output_delta = OpenHotDelta(
            self.descriptor_key,
            self.version_key,
            self.dependencies,
            self.output_budget,
        )
        for ordinal, (delta_record, entry) in enumerate(ordered, start=1):
            assigned_local_id = 0
            if delta_record.allocation_scope_key:
                assigned_local_id = assignment_map[
                    (delta_record.allocation_scope_key,
                     delta_record.external_object_key)
                ]
            resolved_references = tuple(
                self._resolve_reference(reference, assignment_map)
                for reference in delta_record.references
            )
            merged = MergedTrainingRecord(
                self.barrier_key,
                delta_record.merge_key,
                entry.input_key,
                entry.source_key,
                entry.scope_key,
                self.base_fence.timeline_floor + ordinal,
                delta_record.allocation_scope_key,
                delta_record.external_object_key,
                assigned_local_id,
                resolved_references,
                delta_record.payload,
            )
            output_delta.append(merged.to_segment_record())
        segment = None
        if output_delta.object_count:
            segment = output_delta.seal(
                self.output_segment_key,
                self.base_fence.manifest_epoch,
            )
        if self._resolver_key() != self.base_fence.identity_resolver_key:
            raise TrainingShardIntegrityError("merge 期间 identity resolver state 漂移")
        metrics = TrainingBarrierMetrics(
            logical_shards=len(self.shard_plan.shards),
            worker_artifacts=worker_artifact_count,
            empty_artifacts=empty_artifacts,
            raw_records=raw_records,
            merged_records=len(ordered),
            duplicate_records=duplicate_records,
            worker_segment_bytes=worker_segment_bytes,
            canonical_segment_bytes=0 if segment is None else segment.size_bytes,
        )
        return TrainingBarrierResult(
            self.barrier_key,
            self.execution_key,
            self._context_key(),
            segment,
            assignments,
            metrics,
        )

    def publish(
            self,
            result: TrainingBarrierResult,
            *,
            store: TieredSegmentStore,
            receipt_repository: AppendOnlyObjectRepository,
            tier_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            barrier_fault_injector: TrainingBarrierFaultInjector | None = None,
            repository_fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> TrainingBarrierPublishReceipt:
        """发布唯一 canonical segment，最后写 barrier receipt 供崩溃恢复。"""
        if not isinstance(result, TrainingBarrierResult):
            raise TypeError("result 必须是 TrainingBarrierResult")
        if (result.barrier_key != self.barrier_key
                or result.execution_key != self.execution_key
                or result.context_key != self._context_key()):
            raise TrainingShardIntegrityError("barrier result 身份漂移")
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("store 必须是 TieredSegmentStore")
        if not isinstance(receipt_repository, AppendOnlyObjectRepository):
            raise TypeError("receipt_repository 协议错误")
        existing = self._optional_receipt(receipt_repository)
        if existing is not None:
            self._validate_receipt(existing, result, store)
            return existing
        _hit(barrier_fault_injector, FAULT_TRAINING_BARRIER_AFTER_MERGE, {
            "merged_records": result.metrics.merged_records,
        })
        manifest = self._published_result_manifest(result, store)
        if manifest is None:
            self._validate_current_base(store.current_manifest())
            if self._resolver_key() != self.base_fence.identity_resolver_key:
                raise TrainingShardIntegrityError(
                    "barrier 发布前 identity resolver state 漂移")
        if result.segment is not None:
            if manifest is None:
                manifest = store.publish_segment(
                    result.segment,
                    tier_key=_key(tier_key, label="training publish tier_key"),
                    manifest_key=_key(
                        manifest_key, label="training publish manifest_key"),
                    migration_key=_key(
                        migration_key, label="training publish migration_key"),
                    fault_injector=repository_fault_injector,
                )
        _hit(barrier_fault_injector, FAULT_TRAINING_BARRIER_AFTER_PUBLISH, {
            "manifest_epoch": 0 if manifest is None else manifest.publish_epoch,
        })
        if self._resolver_key() != self.base_fence.identity_resolver_key:
            raise TrainingShardIntegrityError(
                "barrier receipt 前 identity resolver state 漂移")
        receipt = self._receipt(result, manifest)
        receipt_repository.put(
            OBJECT_KIND_TRAINING_BARRIER_RECEIPT,
            receipt.identity_key,
            receipt.to_bytes(),
            fault_injector=repository_fault_injector,
        )
        _hit(barrier_fault_injector, FAULT_TRAINING_BARRIER_AFTER_RECEIPT, {
            "manifest_epoch": receipt.manifest_epoch,
        })
        return receipt

    def validate_artifact(self, artifact: WorkerDeltaArtifact) -> None:
        """核验 artifact 属于当前 manifest、plan、barrier 和 base fence。"""
        if not isinstance(artifact, WorkerDeltaArtifact):
            raise TypeError("artifact 必须是 WorkerDeltaArtifact")
        expected = (
            self.manifest.manifest_key,
            self.manifest.state_key(),
            self.shard_plan.plan_key,
            self.shard_plan.state_key(),
            self.producer_key,
            self.execution_key,
            self.barrier_key,
            self.base_fence.stable_key(),
            self.descriptor_key,
            self.version_key,
            self.dependencies,
        )
        actual = (
            artifact.manifest_key,
            artifact.manifest_state_key,
            artifact.plan_key,
            artifact.plan_state_key,
            artifact.producer_key,
            artifact.execution_key,
            artifact.barrier_key,
            artifact.base_fence_key,
            artifact.descriptor_key,
            artifact.version_key,
            artifact.dependencies,
        )
        if actual != expected:
            raise TrainingShardIntegrityError("worker artifact 上下文漂移")
        expected_segment_key = worker_segment_key(
            self.barrier_key,
            artifact.shard_key,
        )
        if artifact.segment_key != expected_segment_key:
            raise TrainingShardIntegrityError("worker artifact segment_key 漂移")
        self.shard_plan.shard(artifact.shard_key)
        if artifact.segment is not None:
            if artifact.segment.read_fence != self.base_fence.manifest_epoch:
                raise TrainingShardIntegrityError("worker segment 版本、依赖或 read fence 漂移")

    @staticmethod
    def _sort_key(
            record: TrainingDeltaRecord,
            entry: TrainingManifestEntry,
            ) -> tuple:
        """返回与完成顺序无关的 source/scope/identity/逻辑序排序键。"""
        return (
            entry.course_seq,
            entry.source_seq,
            entry.source_key,
            entry.scope_key,
            record.merge_key,
            record.logical_seq,
            entry.input_key,
        )

    def _assign_identities(
            self,
            ordered: tuple[tuple[TrainingDeltaRecord, TrainingManifestEntry], ...],
            ) -> tuple[
                tuple[TrainingIdentityAssignment, ...],
                dict[tuple[tuple[int, ...], tuple[int, ...]], int],
            ]:
        """按外部身份稳定序复用 base id 或从冻结 floor 分配新 id。"""
        identities = sorted({
            (record.allocation_scope_key, record.external_object_key)
            for record, _ in ordered if record.allocation_scope_key
        })
        result = []
        mapping = {}
        next_by_scope: dict[tuple[int, ...], int] = {}
        for scope_key, external_key in identities:
            floor = self.base_fence.floor(scope_key)
            existing = self.identity_resolver.resolve(scope_key, external_key)
            if existing is not None:
                if type(existing) is not int or not 0 < existing <= floor:
                    raise TrainingShardIntegrityError("base resolver local id 超出冻结 floor")
                local_id = existing
                existed = True
            else:
                next_id = next_by_scope.get(scope_key, floor) + 1
                next_by_scope[scope_key] = next_id
                local_id = next_id
                existed = False
            mapping[(scope_key, external_key)] = local_id
            result.append(TrainingIdentityAssignment(
                scope_key,
                external_key,
                local_id,
                existed,
            ))
        return tuple(sorted(result)), mapping

    def _resolve_reference(
            self,
            reference: TrainingExternalReference,
            assignment_map: dict[
                tuple[tuple[int, ...], tuple[int, ...]], int
            ],
            ) -> ResolvedTrainingReference:
        """把外部引用解析到本 barrier 新分配或 base 既有 local id。"""
        key = (reference.allocation_scope_key, reference.external_key)
        assigned = assignment_map.get(key)
        if assigned is not None:
            return ResolvedTrainingReference(
                reference.allocation_scope_key,
                assigned,
            )
        existing = self.identity_resolver.resolve(*key)
        if existing is None:
            raise TrainingShardIntegrityError("外部引用在 base 和当前 barrier 均未解析")
        floor = self.base_fence.floor(reference.allocation_scope_key)
        if type(existing) is not int or not 0 < existing <= floor:
            raise TrainingShardIntegrityError("外部引用 local id 超出冻结 floor")
        return ResolvedTrainingReference(
            reference.allocation_scope_key,
            existing,
        )

    def _validate_current_base(
            self,
            current_manifest: LocationManifest | None,
            ) -> None:
        """要求新提交仍位于冻结 base manifest，避免重复分配 local id。"""
        current_epoch = (
            0 if current_manifest is None else current_manifest.publish_epoch)
        if (current_epoch != self.base_fence.manifest_epoch
                or training_base_manifest_state_key(current_manifest)
                != self.base_fence.manifest_state_key):
            raise TrainingShardIntegrityError(
                "barrier base manifest 已推进，必须重建 read fence")

    def _published_result_manifest(
            self,
            result: TrainingBarrierResult,
            store: TieredSegmentStore,
            ) -> LocationManifest | None:
        """识别 publish 后 receipt 前崩溃留下的精确 canonical segment。"""
        if result.segment is None:
            return None
        current = store.current_manifest()
        if current is None:
            return None
        entries = tuple(
            entry for entry in current.entries
            if entry.segment_key == result.segment.segment_key
        )
        if not entries:
            return None
        if len(entries) != 1:
            raise TrainingShardIntegrityError(
                "当前 manifest 重复包含 barrier segment")
        self._validate_manifest_entry(entries[0], result.segment)
        return current

    @staticmethod
    def _validate_manifest_entry(
            entry: LocationManifestEntry,
            segment: SealedSegment,
            ) -> None:
        """核验 manifest entry 完整指向给定 canonical segment。"""
        if (entry.descriptor_key != segment.descriptor_key
                or entry.segment_key != segment.segment_key
                or entry.version_key != segment.version_key
                or entry.checksum_key != segment.checksum_key
                or entry.read_fence != segment.read_fence
                or entry.key_range.lower_key != segment.lower_key
                or entry.key_range.upper_key != segment.upper_key
                or tuple(
                    (
                        dependency.descriptor_key,
                        dependency.version_key,
                        dependency.checksum_key,
                    )
                    for dependency in entry.dependencies
                ) != tuple(
                    (
                        dependency.descriptor_key,
                        dependency.version_key,
                        dependency.checksum_key,
                    )
                    for dependency in segment.dependencies
                )):
            raise TrainingShardIntegrityError(
                "manifest entry 与 canonical barrier segment 漂移")

    def _receipt(
            self,
            result: TrainingBarrierResult,
            manifest: LocationManifest | None,
            ) -> TrainingBarrierPublishReceipt:
        """从 merge 结果和可选新 manifest 构造正式 receipt。"""
        result_key = _digest_key(result.stable_key())
        if result.segment is None:
            if manifest is not None:
                raise TrainingShardIntegrityError("空 barrier 不得发布新 manifest")
            return TrainingBarrierPublishReceipt(
                self.barrier_key,
                self.execution_key,
                result_key,
                (),
                (),
                0,
                (),
            )
        if manifest is None:
            raise TrainingShardIntegrityError("非空 barrier 缺少发布 manifest")
        entries = tuple(
            item for item in manifest.entries
            if item.segment_key == result.segment.segment_key
        )
        if len(entries) != 1:
            raise TrainingShardIntegrityError("发布 manifest 缺少唯一 barrier segment")
        entry = entries[0]
        if (entry.checksum_key != result.segment.checksum_key
                or entry.descriptor_key != result.segment.descriptor_key):
            raise TrainingShardIntegrityError("发布 manifest 的 barrier segment 漂移")
        return TrainingBarrierPublishReceipt(
            self.barrier_key,
            self.execution_key,
            result_key,
            result.segment.segment_key,
            result.segment.checksum_key,
            manifest.publish_epoch,
            training_base_manifest_state_key(manifest),
        )

    def _optional_receipt(
            self,
            repository: AppendOnlyObjectRepository,
            ) -> TrainingBarrierPublishReceipt | None:
        """读取已发布 receipt；不存在时返回 None。"""
        identity = [TRAINING_SHARD_FORMAT_VERSION]
        pack_key(identity, self.barrier_key)
        pack_key(identity, self.execution_key)
        try:
            payload = repository.get(
                OBJECT_KIND_TRAINING_BARRIER_RECEIPT,
                tuple(identity),
            )
        except KeyError:
            return None
        receipt = TrainingBarrierPublishReceipt.from_bytes(payload)
        if receipt.identity_key != tuple(identity):
            raise TrainingShardIntegrityError("barrier receipt 身份漂移")
        return receipt

    def _validate_receipt(
            self,
            receipt: TrainingBarrierPublishReceipt,
            result: TrainingBarrierResult,
            store: TieredSegmentStore,
            ) -> None:
        """核验重放 receipt 与当前结果和已发布 manifest 一致。"""
        if (receipt.barrier_key != self.barrier_key
                or receipt.execution_key != self.execution_key):
            raise TrainingShardIntegrityError("barrier receipt 身份漂移")
        if receipt.result_key != _digest_key(result.stable_key()):
            raise TrainingShardIntegrityError("barrier receipt 与重放结果漂移")
        if result.segment is None:
            if receipt.segment_key or receipt.manifest_epoch:
                raise TrainingShardIntegrityError("空 barrier receipt 漂移")
            return
        if (receipt.segment_key != result.segment.segment_key
                or receipt.segment_checksum_key != result.segment.checksum_key):
            raise TrainingShardIntegrityError("barrier receipt segment 漂移")
        current_manifest = store.current_manifest()
        if current_manifest is None:
            raise TrainingShardIntegrityError("barrier receipt 存在但 manifest 缺失")
        if current_manifest.publish_epoch < receipt.manifest_epoch:
            raise TrainingShardIntegrityError("当前 manifest 早于 barrier receipt")
        try:
            historical = store.ledger.get(receipt.manifest_epoch)
        except KeyError as exc:
            raise TrainingShardIntegrityError(
                "barrier receipt 对应的历史 manifest 缺失") from exc
        if (training_base_manifest_state_key(historical)
                != receipt.manifest_state_key):
            raise TrainingShardIntegrityError("barrier receipt manifest state 漂移")
        entries = tuple(
            entry for entry in historical.entries
            if entry.segment_key == result.segment.segment_key
        )
        if len(entries) != 1:
            raise TrainingShardIntegrityError(
                "barrier receipt 历史 manifest 缺少唯一 segment")
        self._validate_manifest_entry(entries[0], result.segment)


__all__ = [
    "FAULT_TRAINING_BARRIER_AFTER_MERGE",
    "FAULT_TRAINING_BARRIER_AFTER_PUBLISH",
    "FAULT_TRAINING_BARRIER_AFTER_RECEIPT",
    "FrozenTrainingManifest",
    "LogicalShardPlan",
    "LogicalTrainingShard",
    "MergedTrainingRecord",
    "OBJECT_KIND_TRAINING_BARRIER_RECEIPT",
    "OBJECT_KIND_TRAINING_WORKER_ARTIFACT",
    "ResolvedTrainingReference",
    "TrainingAllocationFloor",
    "TrainingBarrierCoordinator",
    "TrainingBarrierFaultInjector",
    "TrainingBarrierMetrics",
    "TrainingBarrierPublishReceipt",
    "TrainingBarrierResult",
    "TrainingBaseReadFence",
    "TrainingDeltaRecord",
    "TrainingExternalReference",
    "TrainingIdentityAssignment",
    "TrainingIdentityResolver",
    "TrainingManifestEntry",
    "TrainingShardConflictError",
    "TrainingShardIntegrityError",
    "TrainingWorkerAssignment",
    "WorkerDeltaArtifact",
    "WorkerLocalDelta",
    "worker_artifact_identity",
    "worker_segment_key",
    "training_base_manifest_state_key",
]
