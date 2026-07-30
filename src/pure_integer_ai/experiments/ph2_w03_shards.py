"""W-03 的 163 条 train record 到 16 个稳定 typed shard。"""
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
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03FrozenContext,
    W03RunRequest,
)
from pure_integer_ai.experiments.ph2_w03_faults import (
    W03FaultPoint,
    hit_w03_fault,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
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


W03_STORAGE_DESCRIPTOR_KEY = (20260730, 103, 1)
W03_STORAGE_DESCRIPTOR = StorageRoleDescriptor(
    W03_STORAGE_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
)
W03_TEMPERATURE_PROFILE = TemperatureProfile(
    (20260730, 103, 2),
    (
        TemperatureTier((20260730, 103, 3), 0),
        TemperatureTier((20260730, 103, 4), 1),
    ),
)
W03_HOT_TIER_KEY = (20260730, 103, 3)
_RECORD_SOURCE = 1
_RECORD_OBSERVATION = 2
_RECORD_TEACHER = 3


class W03ShardError(RuntimeError):
    """W-03 typed shard 输入、merge 或 record readback 漂移。"""


def _key(name: str, value: object = ()) -> tuple[int, ...]:
    return tuple(hashlib.sha256(canonical_json_bytes({
        "name": name,
        "value": value,
    })).digest())


def _record_code(record: object) -> int:
    if isinstance(record, SourceRefRecord):
        return _RECORD_SOURCE
    if isinstance(record, ObservationRecord):
        return _RECORD_OBSERVATION
    if isinstance(record, TeacherEvidenceRecord):
        return _RECORD_TEACHER
    raise W03ShardError("W-03 shard 混入非 train record")


def _input_key(record: object) -> tuple[int, ...]:
    stable = getattr(record, "stable_key", None)
    components = getattr(stable, "components", None)
    if (not isinstance(components, tuple) or not components
            or any(type(item) is not int for item in components)):
        raise W03ShardError("W-03 record stable key 非法")
    return _record_code(record), len(components), *components


def _source_key(record: object) -> tuple[int, ...]:
    source = (
        record.stable_key
        if isinstance(record, SourceRefRecord)
        else record.source_ref_key
    )
    return source.components


def _ordered_records(payload: W03TrainingPayload) -> tuple[object, ...]:
    sources = tuple(sorted(payload.source_refs, key=lambda item: item.stable_key))
    observations = tuple(sorted(
        payload.observations,
        key=lambda item: (item.w_stage, item.substage, item.logical_order,
                          item.stable_key),
    ))
    teachers = tuple(sorted(
        payload.teacher_evidence,
        key=lambda item: (item.observation_key, item.stable_key),
    ))
    return (*sources, *observations, *teachers)


@dataclass(frozen=True)
class _FrozenInputs:
    manifest: FrozenTrainingManifest
    plan: LogicalShardPlan
    payload_by_input: dict[tuple[int, ...], tuple[int, ...]]


def _frozen_inputs(
        context: W03FrozenContext,
        payload: W03TrainingPayload,
        ) -> _FrozenInputs:
    records = _ordered_records(payload)
    if len(records) != 163:
        raise W03ShardError("W-03 frozen train record count 必须为 163")
    entries = []
    encoded = {}
    ordered_inputs = []
    for ordinal, record in enumerate(records):
        input_key = _input_key(record)
        record_bytes = canonical_json_bytes(record.to_dict())
        if input_key in encoded:
            raise W03ShardError("W-03 shard input key 重复")
        encoded[input_key] = tuple(record_bytes)
        ordered_inputs.append(input_key)
        entries.append(TrainingManifestEntry(
            input_key,
            _source_key(record),
            context.stable_key(),
            ordinal,
            ordinal,
            _key("W03_RECORD_CHECKSUM", list(record_bytes)),
        ))
    manifest = FrozenTrainingManifest(
        _key("W03_FROZEN_TRAIN_MANIFEST", context.stable_key()),
        context.stable_key(),
        tuple(entries),
    )
    buckets: list[list[tuple[int, ...]]] = [
        [] for _ in range(context.logical_shard_count)
    ]
    for index, input_key in enumerate(sorted(ordered_inputs)):
        buckets[index % context.logical_shard_count].append(input_key)
    shards = tuple(LogicalTrainingShard(
        _key("W03_LOGICAL_SHARD", {
            "context": list(context.stable_key()),
            "ordinal": ordinal,
        }),
        tuple(bucket),
    ) for ordinal, bucket in enumerate(buckets))
    return _FrozenInputs(
        manifest,
        LogicalShardPlan(
            _key("W03_LOGICAL_SHARD_PLAN", context.stable_key()),
            manifest.manifest_key,
            shards,
        ),
        encoded,
    )


class _Resolver:
    def __init__(self, state_key: tuple[int, ...]) -> None:
        self._state_key = state_key

    def state_key(self) -> tuple[int, ...]:
        return self._state_key

    def resolve(
            self,
            allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        return None


class _Producer:
    def __init__(self, payload_by_input) -> None:
        self._payload_by_input = payload_by_input

    def produce(self, request, emit) -> None:
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
    def __init__(self, state_key, payload_by_input) -> None:
        self._state_key = state_key
        self._payload_by_input = payload_by_input

    def state_key(self) -> tuple[int, ...]:
        return self._state_key

    def producer_for(self, shard_key: tuple[int, ...]) -> _Producer:
        if not isinstance(shard_key, tuple) or not shard_key:
            raise W03ShardError("W-03 producer shard key 非法")
        return _Producer(self._payload_by_input)


class _LifecycleFaultInjector:
    _POINTS = {
        FAULT_TRAINING_RUNTIME_BEFORE_FIRST_SHARD:
            W03FaultPoint.BEFORE_FIRST_SHARD,
        FAULT_TRAINING_RUNTIME_AFTER_PARTIAL_ARTIFACT:
            W03FaultPoint.AFTER_PARTIAL_SHARD,
        FAULT_TRAINING_RUNTIME_BEFORE_MERGE:
            W03FaultPoint.BEFORE_MERGE_PREVIEW,
    }

    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        mapped = self._POINTS.get(point)
        if mapped is None or self.triggered or self.selected != mapped.value:
            return
        self.triggered = True
        hit_w03_fault(self.selected, mapped)


@dataclass(frozen=True)
class W03ShardResult:
    payload: W03TrainingPayload
    artifact_digest: str
    barrier_result_key: tuple[int, ...]
    receipt_key: tuple[int, ...]
    logical_shards: int
    merged_records: int
    merge_publication_count: int
    canonical_artifact_bytes: int
    resource_report: dict[str, int]

    def preview_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.artifact_digest,
            "barrier_result_key": list(self.barrier_result_key),
            "logical_shards": self.logical_shards,
            "merge_publication_count": self.merge_publication_count,
            "merged_records": self.merged_records,
            "receipt_key": list(self.receipt_key),
        }


def _read_merged_payload(result, expected: W03TrainingPayload) -> W03TrainingPayload:
    segment = result.barrier_result.segment
    if segment is None:
        raise W03ShardError("W-03 merge 不得产生空 segment")
    records = []
    for item in segment.records:
        merged = MergedTrainingRecord.from_segment_record(item)
        try:
            value = parse_canonical_json_bytes(
                bytes(merged.payload), require_object=True)
            assert isinstance(value, dict)
            record = record_from_dict(value)
        except (TypeError, ValueError, KeyError) as exc:
            raise W03ShardError("W-03 merged record payload 损坏") from exc
        if _input_key(record) != merged.input_key:
            raise W03ShardError("W-03 merged record 与 input identity 漂移")
        records.append(record)
    payload = W03TrainingPayload(
        tuple(item for item in records if isinstance(item, SourceRefRecord)),
        tuple(item for item in records if isinstance(item, ObservationRecord)),
        tuple(item for item in records if isinstance(item, TeacherEvidenceRecord)),
    )
    expected_bytes = tuple(sorted(
        canonical_json_bytes(item.to_dict())
        for item in _ordered_records(expected)
    ))
    actual_bytes = tuple(sorted(
        canonical_json_bytes(item.to_dict())
        for item in _ordered_records(payload)
    ))
    if actual_bytes != expected_bytes:
        raise W03ShardError("W-03 canonical merge 增删或改写 train record")
    return payload


def run_w03_training_shards(
        context: W03FrozenContext,
        request: W03RunRequest,
        payload: W03TrainingPayload,
        sqlite_path: str | Path,
        *,
        fault_point: str | None = None,
        ) -> W03ShardResult:
    frozen = _frozen_inputs(context, payload)
    base_path = Path(sqlite_path).resolve()
    artifact_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.w03-worker{base_path.suffix}"))
    receipt_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.w03-barrier{base_path.suffix}"))
    segment_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.w03-segments{base_path.suffix}"))
    registry = StorageRoleRegistry()
    registry.register(W03_STORAGE_DESCRIPTOR)
    store = TieredSegmentStore(
        segment_repository, registry, W03_TEMPERATURE_PROFILE)
    resolver = _Resolver(_key(
        "W03_EMPTY_IDENTITY_RESOLVER", context.stable_key()))
    provider = _ProducerProvider(
        _key("W03_TYPED_RECORD_PRODUCER", {
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
        barrier_key=_key("W03_BARRIER", {
            "context": list(context.stable_key()),
            "execution": list(request.execution_identity_key()),
        }),
        descriptor_key=W03_STORAGE_DESCRIPTOR_KEY,
        version_key=context.stable_key(),
        dependencies=(),
        output_budget=SegmentBudget(
            context.resource_budget["max_records"],
            context.resource_budget["max_payload_bytes"],
        ),
        output_segment_key=_key("W03_CANONICAL_TRAIN_SEGMENT", {
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
        tier_key=W03_HOT_TIER_KEY,
        manifest_key=_key("W03_LOCATION_MANIFEST", context.stable_key()),
        migration_key=_key("W03_LOCATION_MIGRATION", context.stable_key()),
        runtime_fault_injector=_LifecycleFaultInjector(fault_point),
    )
    merged_payload = _read_merged_payload(result, payload)
    segment = result.barrier_result.segment
    assert segment is not None
    artifact_bytes = segment.to_bytes() + result.receipt.to_bytes()
    publication_count = len(receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT))
    if publication_count != 1:
        raise W03ShardError("W-03 barrier receipt 必须唯一")
    metrics = result.metrics
    barrier_metrics = result.barrier_result.metrics
    return W03ShardResult(
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
    "W03ShardError",
    "W03ShardResult",
    "run_w03_training_shards",
]
