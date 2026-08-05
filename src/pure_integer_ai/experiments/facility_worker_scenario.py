"""用生产 K-03 owner 执行 F-01 多 worker 确定性场景。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.training_shard_runtime import (
    SharedTrainingArtifactRepository,
    TrainingShardRuntime,
)
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_repository import InMemoryObjectRepository
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.training.sharded_delta import (
    FrozenTrainingManifest,
    LogicalShardPlan,
    LogicalTrainingShard,
    TrainingAllocationFloor,
    TrainingBarrierCoordinator,
    TrainingBaseReadFence,
    TrainingDeltaRecord,
    TrainingExternalReference,
    TrainingManifestEntry,
    training_base_manifest_state_key,
)


_PROFILE = TemperatureProfile(
    (54_000, 1),
    (
        TemperatureTier((54_000, 1), 0),
        TemperatureTier((54_000, 2), 1),
    ),
)
_HOT = (54_000, 1)
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_VERSION = (54_001, 1)
_PRODUCER_KEY = (54_002, 1)
_EXECUTION_KEY = (54_003, 1)
_BARRIER_KEY = (54_004, 1)
_ALLOCATION_SCOPE = (54_005, 1)


class _Resolver:
    """为 barrier 冻结一个既有 base identity 映射。"""

    def __init__(self) -> None:
        self.mapping = {(_ALLOCATION_SCOPE, (50,)): 5}

    def state_key(self) -> tuple[int, ...]:
        """返回外部身份 resolver 的完整逻辑版本。"""
        return 54_006, 1

    def resolve(
            self,
            allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        """按完整 allocation scope 和外部键查询 base local id。"""
        return self.mapping.get((allocation_scope_key, external_key))


class _Producer:
    """从冻结输入映射流式产生一个逻辑 shard 的增量。"""

    def __init__(
            self,
            records: dict[tuple[int, ...], tuple[TrainingDeltaRecord, ...]],
            ) -> None:
        self.records = records

    def produce(self, request: Any, emit: Any) -> None:
        """按 manifest entry 顺序发出当前 shard 的全部记录。"""
        for entry in request.entries:
            for record in self.records.get(entry.input_key, ()):
                emit(record)


class _ProducerProvider:
    """为每个逻辑 shard 提供可并行的独立 producer。"""

    def __init__(
            self,
            records: dict[tuple[int, ...], tuple[TrainingDeltaRecord, ...]],
            ) -> None:
        self.records = records

    def state_key(self) -> tuple[int, ...]:
        """返回与调度和 worker 数无关的生产逻辑键。"""
        return _PRODUCER_KEY

    def producer_for(self, shard_key: tuple[int, ...]) -> _Producer:
        """为一个已冻结 shard 返回不共享可变状态的 producer。"""
        if not isinstance(shard_key, tuple) or not shard_key:
            raise ValueError("F-01 K-03 shard_key 非法")
        return _Producer(self.records)


def _manifest() -> FrozenTrainingManifest:
    """构造四项稳定 source/course 输入 manifest。"""
    entries = tuple(
        TrainingManifestEntry(
            (index,),
            (100 + index,),
            (54_010, 1),
            0,
            index - 1,
            (54_011, index),
        )
        for index in range(1, 5)
    )
    return FrozenTrainingManifest((54_012, 1), (54_013, 1), entries)


def _plan(manifest: FrozenTrainingManifest) -> LogicalShardPlan:
    """把四项输入冻结为四个不依赖 worker 数的逻辑 shard。"""
    return LogicalShardPlan(
        (54_014, 1),
        manifest.manifest_key,
        tuple(
            LogicalTrainingShard((54_015, index), ((index,),))
            for index in range(1, 5)
        ),
    )


def _records() -> dict[tuple[int, ...], tuple[TrainingDeltaRecord, ...]]:
    """构造重复、base id、新 id 和跨 shard 引用混合的真实增量。"""
    first = TrainingDeltaRecord(
        (1,),
        (1,),
        _ALLOCATION_SCOPE,
        (100,),
        (TrainingExternalReference(_ALLOCATION_SCOPE, (200,)),),
        0,
        (1,),
    )
    duplicate = TrainingDeltaRecord(
        (3,),
        (1,),
        _ALLOCATION_SCOPE,
        (100,),
        (TrainingExternalReference(_ALLOCATION_SCOPE, (200,)),),
        0,
        (1,),
    )
    return {
        (1,): (first,),
        (2,): (TrainingDeltaRecord(
            (2,), (2,), _ALLOCATION_SCOPE, (200,), (), 0, (2,)),),
        (3,): (duplicate,),
        (4,): (TrainingDeltaRecord(
            (4,),
            (3,),
            _ALLOCATION_SCOPE,
            (50,),
            (TrainingExternalReference(_ALLOCATION_SCOPE, (100,)),),
            0,
            (3,),
        ),),
    }


def _coordinator(
        manifest: FrozenTrainingManifest,
        plan: LogicalShardPlan,
        resolver: _Resolver,
        store: TieredSegmentStore,
        ) -> TrainingBarrierCoordinator:
    """从全新 store 冻结 base fence 并建立唯一 barrier coordinator。"""
    current = store.current_manifest()
    fence = TrainingBaseReadFence(
        0 if current is None else current.publish_epoch,
        training_base_manifest_state_key(current),
        (54_016, 1),
        resolver.state_key(),
        (TrainingAllocationFloor(_ALLOCATION_SCOPE, 10),),
        20,
    )
    return TrainingBarrierCoordinator(
        manifest=manifest,
        shard_plan=plan,
        base_fence=fence,
        identity_resolver=resolver,
        producer_key=_PRODUCER_KEY,
        execution_key=_EXECUTION_KEY,
        barrier_key=_BARRIER_KEY,
        descriptor_key=_DESCRIPTOR,
        version_key=_VERSION,
        dependencies=(),
        output_budget=SegmentBudget(16, 32_768),
        output_segment_key=(54_017, *_BARRIER_KEY),
    )


def published_worker_bytes(worker_count: int) -> tuple[Any, ...]:
    """在全新介质运行 K-03 并返回全部正式 canonical 发布字节。"""
    if worker_count not in (1, 2, 4):
        raise ValueError("F-01 K-03 worker_count 必须是 1/2/4")
    manifest = _manifest()
    plan = _plan(manifest)
    resolver = _Resolver()
    store = TieredSegmentStore(
        InMemoryObjectRepository(),
        build_storage_role_registry(),
        _PROFILE,
    )
    artifact_repository = InMemoryObjectRepository()
    receipt_repository = InMemoryObjectRepository()
    runtime = TrainingShardRuntime(
        _coordinator(manifest, plan, resolver, store),
        _ProducerProvider(_records()),
        SharedTrainingArtifactRepository(artifact_repository),
        SegmentBudget(16, 32_768),
    )
    result = runtime.run(
        worker_count,
        store=store,
        receipt_repository=receipt_repository,
        tier_key=_HOT,
        manifest_key=(54_018, 1),
        migration_key=(54_019, 1),
    )
    current = store.current_manifest()
    if current is None:
        raise RuntimeError("F-01 K-03 未发布 store manifest")
    return (
        result.barrier_result.segment.to_bytes(),
        result.receipt.to_bytes(),
        current.to_bytes(),
        result.barrier_result.stable_key(),
    )


__all__ = ["published_worker_bytes"]
