"""把 W-02 三 owner train records 接到 K-03 稳定 typed shard/merge。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_w01_shards import SQLiteObjectRepository
from pure_integer_ai.experiments.ph2_w02_contract import (
    W02FrozenContext,
    W02RunRequest,
    W02TrainingPayload,
)
from pure_integer_ai.experiments.ph2_w02_faults import (
    W02FaultPoint,
    hit_w02_fault,
)
from pure_integer_ai.experiments.training_shard_runtime import (
    FAULT_TRAINING_RUNTIME_AFTER_PARTIAL_ARTIFACT,
    FAULT_TRAINING_RUNTIME_BEFORE_FIRST_SHARD,
    FAULT_TRAINING_RUNTIME_BEFORE_MERGE,
    SharedTrainingArtifactRepository,
    TrainingShardRuntime,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.storage_role import (
    STORAGE_ACCESS_APPEND_ONLY,
    STORAGE_ACCESS_INDEXED_READ,
    STORAGE_ROLE_AUTHORITATIVE,
    StorageRoleDescriptor,
    StorageRoleRegistry,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.training.sharded_delta import (
    OBJECT_KIND_TRAINING_BARRIER_RECEIPT,
    FrozenTrainingManifest,
    LogicalShardPlan,
    LogicalTrainingShard,
    MergedTrainingRecord,
    TrainingBarrierCoordinator,
    TrainingBaseReadFence,
    TrainingDeltaRecord,
    TrainingManifestEntry,
    training_base_manifest_state_key,
)


W02_STORAGE_DESCRIPTOR_KEY = (20260729, 102, 1)
W02_STORAGE_DESCRIPTOR = StorageRoleDescriptor(
    W02_STORAGE_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
)
W02_TEMPERATURE_PROFILE = TemperatureProfile(
    (20260729, 102, 2),
    (
        TemperatureTier((20260729, 102, 3), 0),
        TemperatureTier((20260729, 102, 4), 1),
    ),
)
W02_HOT_TIER_KEY = (20260729, 102, 3)
_RECORD_SOURCE = 1
_RECORD_OBSERVATION = 2
_RECORD_TEACHER = 3


class W02ShardError(RuntimeError):
    """W-02 typed shard 输入、canonical merge 或 record readback 漂移。"""


def _key(name: str, value: object = ()) -> tuple[int, ...]:
    """把开放名字和规范结构压成稳定 SHA-256 byte key。"""
    return tuple(hashlib.sha256(canonical_json_bytes({
        "name": name,
        "value": value,
    })).digest())


def _record_code(record: object) -> int:
    """返回三种训练 owner record 的稳定类型码。"""
    if isinstance(record, SourceRefRecord):
        return _RECORD_SOURCE
    if isinstance(record, ObservationRecord):
        return _RECORD_OBSERVATION
    if isinstance(record, TeacherEvidenceRecord):
        return _RECORD_TEACHER
    raise W02ShardError("W-02 shard 混入非 train record")


def _input_key(record: object) -> tuple[int, ...]:
    """从 owner kind 和完整 Dataset stable key 构造输入身份。"""
    stable = getattr(record, "stable_key", None)
    components = getattr(stable, "components", None)
    if (not isinstance(components, tuple) or not components
            or any(type(item) is not int for item in components)):
        raise W02ShardError("W-02 record stable key 非法")
    return _record_code(record), len(components), *components


def _source_key(record: object) -> tuple[int, ...]:
    """返回 SourceRef 自身或 Observation/Evidence 引用的完整来源键。"""
    source = (record.stable_key if isinstance(record, SourceRefRecord)
              else record.source_ref_key)
    return source.components


def _ordered_records(payload: W02TrainingPayload) -> tuple[object, ...]:
    """按 owner、课程逻辑序和 stable key 形成 worker 无关输入序。"""
    sources = tuple(sorted(payload.source_refs, key=lambda item: item.stable_key))
    observations = tuple(sorted(
        payload.observations,
        key=lambda item: (item.substage, item.logical_order, item.stable_key),
    ))
    teachers = tuple(sorted(
        payload.teacher_evidence,
        key=lambda item: (item.observation_key, item.stable_key),
    ))
    return (*sources, *observations, *teachers)


@dataclass(frozen=True)
class _FrozenInputs:
    """K-03 manifest/plan 与按 input key 索引的规范 record bytes。"""

    manifest: FrozenTrainingManifest
    plan: LogicalShardPlan
    payload_by_input: dict[tuple[int, ...], tuple[int, ...]]


def _frozen_inputs(
        context: W02FrozenContext,
        payload: W02TrainingPayload,
        ) -> _FrozenInputs:
    """把 76 条 record 冻结为恰好 16 个由 record key 决定的逻辑 shard。"""
    records = _ordered_records(payload)
    entries = []
    encoded = {}
    ordered_inputs = []
    for ordinal, record in enumerate(records):
        input_key = _input_key(record)
        record_bytes = canonical_json_bytes(record.to_dict())
        if input_key in encoded:
            raise W02ShardError("W-02 shard input key 重复")
        encoded[input_key] = tuple(record_bytes)
        ordered_inputs.append(input_key)
        entries.append(TrainingManifestEntry(
            input_key,
            _source_key(record),
            context.stable_key(),
            ordinal,
            ordinal,
            _key("W02_RECORD_CHECKSUM", list(record_bytes)),
        ))
    if len(records) != 76:
        raise W02ShardError("W-02 frozen train record count 必须为 76")
    manifest = FrozenTrainingManifest(
        _key("W02_FROZEN_TRAIN_MANIFEST", context.stable_key()),
        context.stable_key(),
        tuple(entries),
    )
    buckets: list[list[tuple[int, ...]]] = [[] for _ in range(16)]
    for index, input_key in enumerate(sorted(ordered_inputs)):
        buckets[index % 16].append(input_key)
    shards = tuple(LogicalTrainingShard(
        _key("W02_LOGICAL_SHARD", {
            "context": list(context.stable_key()),
            "ordinal": ordinal,
        }),
        tuple(bucket),
    ) for ordinal, bucket in enumerate(buckets))
    plan = LogicalShardPlan(
        _key("W02_LOGICAL_SHARD_PLAN", context.stable_key()),
        manifest.manifest_key,
        shards,
    )
    return _FrozenInputs(manifest, plan, encoded)


class _Resolver:
    """W-02 typed delta 不提前分配 Core local id 的冻结 resolver。"""

    def __init__(self, state_key: tuple[int, ...]) -> None:
        self._state_key = state_key

    def state_key(self) -> tuple[int, ...]:
        """返回绑定 D-03/context 的空 resolver 身份。"""
        return self._state_key

    def resolve(
            self, allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        """W-02 worker 只传 typed record，不解析或分配 Core 对象。"""
        return None


class _Producer:
    """把每个冻结 record 的完整规范 JSON bytes 写进 worker delta。"""

    def __init__(self, payload_by_input: dict[tuple[int, ...], tuple[int, ...]]) -> None:
        self._payload_by_input = payload_by_input

    def produce(self, request, emit) -> None:
        """按 shard request 顺序发出每个 record 的唯一 typed delta。"""
        for entry in request.entries:
            emit(TrainingDeltaRecord(
                entry.input_key,
                entry.input_key,
                (),
                (),
                (),
                entry.course_seq,
                self._payload_by_input[entry.input_key],
            ))


class _ProducerProvider:
    """为所有物理 worker 提供共享只读输入和无状态 producer。"""

    def __init__(
            self,
            state_key: tuple[int, ...],
            payload_by_input: dict[tuple[int, ...], tuple[int, ...]],
            ) -> None:
        self._state_key = state_key
        self._payload_by_input = payload_by_input

    def state_key(self) -> tuple[int, ...]:
        """返回 producer 逻辑与 payload identity 的冻结键。"""
        return self._state_key

    def producer_for(self, shard_key: tuple[int, ...]) -> _Producer:
        """返回不拥有共享写状态的轻量 producer。"""
        if not isinstance(shard_key, tuple) or not shard_key:
            raise W02ShardError("W-02 producer shard key 非法")
        return _Producer(self._payload_by_input)


class _LifecycleFaultInjector:
    """把 K-03 三个生命周期点映射到 D-03 W-02 故障键。"""

    _POINTS = {
        FAULT_TRAINING_RUNTIME_BEFORE_FIRST_SHARD:
            W02FaultPoint.BEFORE_FIRST_SHARD,
        FAULT_TRAINING_RUNTIME_AFTER_PARTIAL_ARTIFACT:
            W02FaultPoint.AFTER_PARTIAL_SHARD,
        FAULT_TRAINING_RUNTIME_BEFORE_MERGE:
            W02FaultPoint.BEFORE_MERGE_PREVIEW,
    }

    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """同一运行只在选定生命周期点中断一次。"""
        mapped = self._POINTS.get(point)
        if mapped is None or self.triggered or self.selected != mapped:
            return
        self.triggered = True
        hit_w02_fault(self.selected, mapped)


@dataclass(frozen=True)
class W02ShardResult:
    """K-03 返回的 canonical train payload、逻辑摘要和资源计数。"""

    payload: W02TrainingPayload
    artifact_digest: str
    barrier_result_key: tuple[int, ...]
    receipt_key: tuple[int, ...]
    logical_shards: int
    merged_records: int
    merge_publication_count: int
    canonical_artifact_bytes: int
    resource_report: dict[str, int]

    def preview_payload(self) -> dict[str, object]:
        """形成 transaction preview 的 worker 无关逻辑摘要。"""
        return {
            "artifact_digest": self.artifact_digest,
            "barrier_result_key": list(self.barrier_result_key),
            "logical_shards": self.logical_shards,
            "merge_publication_count": self.merge_publication_count,
            "merged_records": self.merged_records,
            "receipt_key": list(self.receipt_key),
        }


def _read_merged_payload(result, expected: W02TrainingPayload) -> W02TrainingPayload:
    """从 coordinator canonical segment 恢复 76 条 record 并核完整输入 identity。"""
    segment = result.barrier_result.segment
    if segment is None:
        raise W02ShardError("W-02 merge 不得产生空 segment")
    records = []
    for item in segment.records:
        merged = MergedTrainingRecord.from_segment_record(item)
        try:
            payload_bytes = bytes(merged.payload)
            value = parse_canonical_json_bytes(
                payload_bytes, require_object=True)
            assert isinstance(value, dict)
            record = record_from_dict(value)
        except (TypeError, ValueError, KeyError) as exc:
            raise W02ShardError("W-02 merged record payload 损坏") from exc
        if _input_key(record) != merged.input_key:
            raise W02ShardError("W-02 merged record 与 input identity 漂移")
        records.append(record)
    payload = W02TrainingPayload(
        tuple(item for item in records if isinstance(item, SourceRefRecord)),
        tuple(item for item in records if isinstance(item, ObservationRecord)),
        tuple(item for item in records if isinstance(item, TeacherEvidenceRecord)),
    )
    expected_bytes = tuple(sorted(
        canonical_json_bytes(item.to_dict())
        for item in _ordered_records(expected)))
    actual_bytes = tuple(sorted(
        canonical_json_bytes(item.to_dict())
        for item in _ordered_records(payload)))
    if actual_bytes != expected_bytes:
        raise W02ShardError("W-02 canonical merge 增删或改写 train record")
    return payload


def run_w02_training_shards(
        context: W02FrozenContext,
        request: W02RunRequest,
        payload: W02TrainingPayload,
        sqlite_path: str | Path,
        *,
        fault_point: str | None = None,
        ) -> W02ShardResult:
    """执行或恢复 16 typed shard，经唯一 coordinator merge 后回读 payload。"""
    frozen = _frozen_inputs(context, payload)
    base_path = Path(sqlite_path).resolve()
    artifact_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.worker{base_path.suffix}"))
    receipt_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.barrier{base_path.suffix}"))
    segment_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.segments{base_path.suffix}"))
    registry = StorageRoleRegistry()
    registry.register(W02_STORAGE_DESCRIPTOR)
    store = TieredSegmentStore(
        segment_repository, registry, W02_TEMPERATURE_PROFILE)
    resolver = _Resolver(_key(
        "W02_EMPTY_IDENTITY_RESOLVER", context.stable_key()))
    provider = _ProducerProvider(
        _key("W02_TYPED_RECORD_PRODUCER", {
            "context": list(context.stable_key()),
            "manifest": list(frozen.manifest.state_key()),
        }),
        frozen.payload_by_input,
    )
    base_fence = TrainingBaseReadFence(
        0,
        training_base_manifest_state_key(None),
        request.base_fence_key,
        resolver.state_key(),
        (),
        0,
    )
    coordinator = TrainingBarrierCoordinator(
        manifest=frozen.manifest,
        shard_plan=frozen.plan,
        base_fence=base_fence,
        identity_resolver=resolver,
        producer_key=provider.state_key(),
        execution_key=request.execution_identity_key(),
        barrier_key=_key("W02_BARRIER", {
            "context": list(context.stable_key()),
            "execution": list(request.execution_identity_key()),
        }),
        descriptor_key=W02_STORAGE_DESCRIPTOR_KEY,
        version_key=context.stable_key(),
        dependencies=(),
        output_budget=SegmentBudget(
            context.resource_budget["max_records"],
            context.resource_budget["max_payload_bytes"],
        ),
        output_segment_key=_key("W02_CANONICAL_TRAIN_SEGMENT", {
            "context": list(context.stable_key()),
            "execution": list(request.execution_identity_key()),
        }),
    )
    runtime = TrainingShardRuntime(
        coordinator,
        provider,
        SharedTrainingArtifactRepository(artifact_repository),
        SegmentBudget(
            context.resource_budget["max_records"],
            context.resource_budget["max_payload_bytes"],
        ),
    )
    result = runtime.run(
        request.worker_count,
        store=store,
        receipt_repository=receipt_repository,
        tier_key=W02_HOT_TIER_KEY,
        manifest_key=_key("W02_LOCATION_MANIFEST", context.stable_key()),
        migration_key=_key("W02_LOCATION_MIGRATION", context.stable_key()),
        runtime_fault_injector=_LifecycleFaultInjector(fault_point),
    )
    merged_payload = _read_merged_payload(result, payload)
    segment = result.barrier_result.segment
    assert segment is not None
    artifact_bytes = segment.to_bytes() + result.receipt.to_bytes()
    publication_count = len(receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT))
    if publication_count != 1:
        raise W02ShardError("W-02 barrier receipt 必须唯一")
    metrics = result.metrics
    barrier_metrics = result.barrier_result.metrics
    return W02ShardResult(
        merged_payload,
        hashlib.sha256(artifact_bytes).hexdigest(),
        result.barrier_result.stable_key(),
        result.receipt.result_key,
        metrics.logical_shards,
        barrier_metrics.merged_records,
        publication_count,
        len(artifact_bytes),
        {
            "canonical_segment_bytes": barrier_metrics.canonical_segment_bytes,
            "in_flight_shard_limit": metrics.in_flight_shard_limit,
            "logical_shards": metrics.logical_shards,
            "merged_records": barrier_metrics.merged_records,
            "produced_shards": metrics.produced_shards,
            "raw_records": barrier_metrics.raw_records,
            "requested_workers": metrics.requested_workers,
            "restored_shards": metrics.restored_shards,
            "sealed_cold_bytes": metrics.sealed_cold_bytes,
            "worker_byte_limit": metrics.worker_byte_limit,
            "worker_object_limit": metrics.worker_object_limit,
        },
    )


__all__ = [
    "W02ShardError",
    "W02ShardResult",
    "run_w02_training_shards",
]
