"""把十六项 W-01 协议输入薄装配到 K-03 稳定 shard/merge。"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w01_contract import (
    W01FrozenContext,
    W01RunRequest,
)
from pure_integer_ai.experiments.ph2_w01_faults import (
    W01FaultPoint,
    hit_w01_fault,
)
from pure_integer_ai.experiments.training_shard_runtime import (
    FAULT_TRAINING_RUNTIME_AFTER_PARTIAL_ARTIFACT,
    FAULT_TRAINING_RUNTIME_BEFORE_FIRST_SHARD,
    FAULT_TRAINING_RUNTIME_BEFORE_MERGE,
    SharedTrainingArtifactRepository,
    TrainingShardRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_repository import BackendObjectRepository
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
    TrainingBarrierCoordinator,
    TrainingBaseReadFence,
    TrainingDeltaRecord,
    TrainingManifestEntry,
    training_base_manifest_state_key,
)


W01_PROTOCOL_STORAGE_DESCRIPTOR_KEY = (20260729, 101, 1)
W01_PROTOCOL_STORAGE_DESCRIPTOR = StorageRoleDescriptor(
    W01_PROTOCOL_STORAGE_DESCRIPTOR_KEY,
    STORAGE_ROLE_AUTHORITATIVE,
    (STORAGE_ACCESS_APPEND_ONLY, STORAGE_ACCESS_INDEXED_READ),
)
W01_PROTOCOL_TEMPERATURE_PROFILE = TemperatureProfile(
    (20260729, 101, 2),
    (
        TemperatureTier((20260729, 101, 3), 0),
        TemperatureTier((20260729, 101, 4), 1),
    ),
)
W01_PROTOCOL_HOT_TIER_KEY = (20260729, 101, 3)


def _key(name: str, value: object = ()) -> tuple[int, ...]:
    """从开放名字和结构值派生稳定纯整数键。"""
    return tuple(hashlib.sha256(canonical_json_bytes({
        "name": name,
        "value": value,
    })).digest())


class SQLiteObjectRepository:
    """串行化每次 SQLite 对象操作，连接只在线程内存活。"""

    def __init__(self, path: str | Path) -> None:
        """绑定共享 SQLite 文件；锁只约束当前进程的物理写。"""
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _call(self, method: str, *args, **kwargs):
        """在锁内用 fresh connection 调用对象仓库并立即关闭。"""
        with self._lock:
            backend = SQLiteBackend(str(self.path))
            try:
                repository = BackendObjectRepository(backend)
                return getattr(repository, method)(*args, **kwargs)
            finally:
                backend.close()

    def put(self, object_kind, identity_key, payload, *, fault_injector=None):
        """幂等 seal-last 发布一个对象。"""
        return self._call(
            "put", object_kind, identity_key, payload,
            fault_injector=fault_injector)

    def get(self, object_kind, identity_key):
        """按完整身份读取一个已封存对象。"""
        return self._call("get", object_kind, identity_key)

    def list_kind(self, object_kind):
        """按发布序列出指定对象类型。"""
        return self._call("list_kind", object_kind)

    def reclaim(self, object_kind, identity_key, *, fault_injector=None):
        """委托对象仓库执行显式回收协议。"""
        return self._call(
            "reclaim", object_kind, identity_key,
            fault_injector=fault_injector)


class _ProtocolResolver:
    """W-01 不分配语义对象，只提供冻结的空 base resolver。"""

    def __init__(self, state_key: tuple[int, ...]) -> None:
        self._state_key = state_key

    def state_key(self) -> tuple[int, ...]:
        """返回 D-03/context 绑定的空 resolver 身份。"""
        return self._state_key

    def resolve(
            self,
            allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        """W-01 不产生或解析任何语言对象 local id。"""
        return None


class _ProtocolProducer:
    """把协议 identity 写成非语义、无对象分配的 K-03 delta。"""

    def __init__(self, payload_by_input: dict[tuple[int, ...], tuple[int, ...]]) -> None:
        self._payload_by_input = payload_by_input

    def produce(self, request, emit) -> None:
        """按冻结 entry 顺序逐项发出协议校验记录。"""
        for entry in request.entries:
            index = entry.course_seq + 1
            emit(TrainingDeltaRecord(
                entry.input_key,
                (20260729, 101, index),
                (),
                (),
                (),
                entry.source_seq,
                self._payload_by_input[entry.input_key],
            ))


class _ProtocolProducerProvider:
    """为所有 shard 提供无共享可变状态的协议 producer。"""

    def __init__(
            self,
            state_key: tuple[int, ...],
            payload_by_input: dict[tuple[int, ...], tuple[int, ...]],
            ) -> None:
        self._state_key = state_key
        self._payload_by_input = payload_by_input

    def state_key(self) -> tuple[int, ...]:
        """返回 producer 逻辑和 D-03 输入的稳定身份。"""
        return self._state_key

    def producer_for(self, shard_key: tuple[int, ...]) -> _ProtocolProducer:
        """返回只读共享映射的独立轻量 producer。"""
        return _ProtocolProducer(self._payload_by_input)


class _LifecycleFaultInjector:
    """把通用 K-03 生命周期点映射到 D-03 W-01 故障键。"""

    _POINTS = {
        FAULT_TRAINING_RUNTIME_BEFORE_FIRST_SHARD:
            W01FaultPoint.BEFORE_FIRST_SHARD,
        FAULT_TRAINING_RUNTIME_AFTER_PARTIAL_ARTIFACT:
            W01FaultPoint.AFTER_PARTIAL_SHARD,
        FAULT_TRAINING_RUNTIME_BEFORE_MERGE:
            W01FaultPoint.BEFORE_MERGE_PREVIEW,
    }

    def __init__(self, selected: str | None) -> None:
        self.selected = selected
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """同一实例只允许选定故障点中断一次。"""
        mapped = self._POINTS.get(point)
        if mapped is None or self.triggered or self.selected != mapped:
            return
        self.triggered = True
        hit_w01_fault(self.selected, mapped)


@dataclass(frozen=True)
class W01ShardResult:
    """K-03 子设施返回的 worker 无关逻辑摘要和资源观测。"""

    artifact_digest: str
    barrier_result_key: tuple[int, ...]
    receipt_key: tuple[int, ...]
    logical_shards: int
    merged_records: int
    merge_publication_count: int
    canonical_artifact_bytes: int
    resource_report: dict[str, int]

    def preview_payload(self) -> dict[str, object]:
        """形成事务 preview 绑定的规范逻辑结果。"""
        return {
            "artifact_digest": self.artifact_digest,
            "barrier_result_key": list(self.barrier_result_key),
            "logical_shards": self.logical_shards,
            "merge_publication_count": self.merge_publication_count,
            "merged_records": self.merged_records,
            "receipt_key": list(self.receipt_key),
        }


def _frozen_manifest(
        context: W01FrozenContext,
        ) -> tuple[FrozenTrainingManifest, LogicalShardPlan,
                   dict[tuple[int, ...], tuple[int, ...]]]:
    """把十六项协议输入冻结为十六个 worker 数无关逻辑 shard。"""
    entries = []
    shards = []
    payloads = {}
    for index, item in enumerate(context.protocol_inputs, start=1):
        input_key = (20260729, 101, index)
        payloads[input_key] = item.identity_key
        entries.append(TrainingManifestEntry(
            input_key,
            item.identity_key,
            context.stable_key(),
            index - 1,
            index - 1,
            item.identity_key,
        ))
        shards.append(LogicalTrainingShard(
            (20260729, 102, index),
            (input_key,),
        ))
    manifest = FrozenTrainingManifest(
        _key("W01_FROZEN_PROTOCOL_MANIFEST", context.stable_key()),
        context.stable_key(),
        tuple(entries),
    )
    plan = LogicalShardPlan(
        _key("W01_LOGICAL_SHARD_PLAN", context.stable_key()),
        manifest.manifest_key,
        tuple(shards),
    )
    return manifest, plan, payloads


def run_w01_protocol_shards(
        context: W01FrozenContext,
        request: W01RunRequest,
        sqlite_path: str | Path,
        *,
        fault_point: str | None = None,
        ) -> W01ShardResult:
    """用持久 SQLite artifact 执行/恢复十六 shard 并单点稳定 merge。"""
    manifest, plan, payloads = _frozen_manifest(context)
    base_path = Path(sqlite_path).resolve()
    artifact_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.worker{base_path.suffix}"))
    receipt_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.barrier{base_path.suffix}"))
    segment_repository = SQLiteObjectRepository(
        base_path.with_name(f"{base_path.stem}.segments{base_path.suffix}"))
    registry = StorageRoleRegistry()
    registry.register(W01_PROTOCOL_STORAGE_DESCRIPTOR)
    store = TieredSegmentStore(
        segment_repository,
        registry,
        W01_PROTOCOL_TEMPERATURE_PROFILE,
    )
    resolver = _ProtocolResolver(_key(
        "W01_EMPTY_IDENTITY_RESOLVER", context.stable_key()))
    provider = _ProtocolProducerProvider(
        _key("W01_PROTOCOL_PRODUCER", context.stable_key()),
        payloads,
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
        manifest=manifest,
        shard_plan=plan,
        base_fence=base_fence,
        identity_resolver=resolver,
        producer_key=provider.state_key(),
        execution_key=request.execution_identity_key(),
        barrier_key=_key("W01_BARRIER", {
            "context": list(context.stable_key()),
            "execution": list(request.execution_identity_key()),
        }),
        descriptor_key=W01_PROTOCOL_STORAGE_DESCRIPTOR_KEY,
        version_key=context.stable_key(),
        dependencies=(),
        output_budget=SegmentBudget(
            context.logical_shard_count,
            context.resource_budget["max_payload_bytes"],
        ),
        output_segment_key=_key("W01_CANONICAL_PROTOCOL_SEGMENT", {
            "context": list(context.stable_key()),
            "execution": list(request.execution_identity_key()),
        }),
    )
    runtime = TrainingShardRuntime(
        coordinator,
        provider,
        SharedTrainingArtifactRepository(artifact_repository),
        SegmentBudget(
            context.logical_shard_count,
            context.resource_budget["max_payload_bytes"],
        ),
    )
    result = runtime.run(
        request.worker_count,
        store=store,
        receipt_repository=receipt_repository,
        tier_key=W01_PROTOCOL_HOT_TIER_KEY,
        manifest_key=_key("W01_LOCATION_MANIFEST", context.stable_key()),
        migration_key=_key("W01_LOCATION_MIGRATION", context.stable_key()),
        runtime_fault_injector=_LifecycleFaultInjector(fault_point),
    )
    if result.barrier_result.segment is None:
        raise RuntimeError("W-01 protocol shard merge 不得产生空 segment")
    artifact_bytes = (
        result.barrier_result.segment.to_bytes() + result.receipt.to_bytes())
    artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
    publication_count = len(receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT))
    if publication_count != 1:
        raise RuntimeError("W-01 barrier receipt 必须唯一")
    metrics = result.metrics
    return W01ShardResult(
        artifact_digest=artifact_digest,
        barrier_result_key=result.barrier_result.stable_key(),
        receipt_key=result.receipt.result_key,
        logical_shards=metrics.logical_shards,
        merged_records=result.barrier_result.metrics.merged_records,
        merge_publication_count=publication_count,
        canonical_artifact_bytes=len(artifact_bytes),
        resource_report={
            "in_flight_shard_limit": metrics.in_flight_shard_limit,
            "logical_shards": metrics.logical_shards,
            "produced_shards": metrics.produced_shards,
            "requested_workers": metrics.requested_workers,
            "restored_shards": metrics.restored_shards,
            "sealed_cold_bytes": metrics.sealed_cold_bytes,
            "worker_byte_limit": metrics.worker_byte_limit,
            "worker_object_limit": metrics.worker_object_limit,
        },
    )


__all__ = [
    "SQLiteObjectRepository",
    "W01_PROTOCOL_STORAGE_DESCRIPTOR",
    "W01_PROTOCOL_STORAGE_DESCRIPTOR_KEY",
    "W01ShardResult",
    "run_w01_protocol_shards",
]
