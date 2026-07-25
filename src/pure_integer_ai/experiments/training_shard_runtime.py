"""K-03 稳定逻辑分片、可恢复 worker artifact 和单协调器运行时。"""
from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pure_integer_ai.experiments.train_execution import (
    TelemetryClock,
    sample_working_set,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_repository import (
    AppendOnlyObjectRepository,
    SegmentRepositoryFaultInjector,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.training.sharded_delta import (
    LogicalTrainingShard,
    TrainingBarrierCoordinator,
    TrainingBarrierFaultInjector,
    TrainingBarrierPublishReceipt,
    TrainingBarrierResult,
    TrainingDeltaRecord,
    TrainingManifestEntry,
    TrainingShardIntegrityError,
    WorkerDeltaArtifact,
    WorkerLocalDelta,
    worker_artifact_identity,
    worker_segment_key,
)


@dataclass(frozen=True)
class TrainingShardWorkRequest:
    """交给注入式 producer 的冻结 shard 输入和 base read fence。"""

    shard: LogicalTrainingShard
    entries: tuple[TrainingManifestEntry, ...]
    base_fence_key: tuple[int, ...]
    execution_key: tuple[int, ...]
    barrier_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求请求恰好覆盖逻辑 shard，且输入顺序来自冻结 manifest。"""
        if not isinstance(self.shard, LogicalTrainingShard):
            raise TypeError("work request shard 类型错误")
        if (not isinstance(self.entries, tuple)
                or any(not isinstance(item, TrainingManifestEntry)
                       for item in self.entries)):
            raise TypeError("work request entries 类型错误")
        if ({item.input_key for item in self.entries}
                != set(self.shard.input_keys)):
            raise TrainingShardIntegrityError(
                "work request 未恰好覆盖逻辑 shard")


@runtime_checkable
class TrainingShardProducer(Protocol):
    """把一个冻结逻辑 shard 流式写入隔离 worker delta。"""

    def produce(
            self,
            request: TrainingShardWorkRequest,
            emit: Callable[[TrainingDeltaRecord], bool],
            ) -> None:
        """读取请求对应输入，并逐条调用 emit 写入逻辑增量。"""
        ...


@runtime_checkable
class TrainingShardProducerProvider(Protocol):
    """提供完整逻辑版本键和可并行使用的 shard producer。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 producer 逻辑、解析规则和依赖的完整稳定键。"""
        ...

    def producer_for(
            self,
            shard_key: tuple[int, ...],
            ) -> TrainingShardProducer:
        """为一个逻辑 shard 返回独立或线程安全的 producer。"""
        ...


@runtime_checkable
class TrainingArtifactRepositoryProvider(Protocol):
    """为逻辑 shard 提供窄 append-only artifact 仓库。"""

    def repository_for(
            self,
            shard_key: tuple[int, ...],
            ) -> AppendOnlyObjectRepository:
        """返回保存该 shard artifact 的仓库，不复制完整训练 backend。"""
        ...


@dataclass(frozen=True)
class SharedTrainingArtifactRepository:
    """让多个逻辑 shard 共用一个已注入的窄对象仓库。"""

    repository: AppendOnlyObjectRepository

    def __post_init__(self) -> None:
        """核验共享对象实现 append-only repository 协议。"""
        if not isinstance(self.repository, AppendOnlyObjectRepository):
            raise TypeError("共享 artifact repository 协议错误")

    def repository_for(
            self,
            shard_key: tuple[int, ...],
            ) -> AppendOnlyObjectRepository:
        """忽略物理分配细节，返回同一窄对象仓库。"""
        if not isinstance(shard_key, tuple) or not shard_key:
            raise ValueError("artifact shard_key 必须是非空 tuple")
        return self.repository


@dataclass(frozen=True)
class TrainingShardRuntimeMetrics:
    """不读墙钟的恢复、调度和 worker 热上界计量。"""

    requested_workers: int
    logical_shards: int
    restored_shards: int
    produced_shards: int
    in_flight_shard_limit: int
    worker_object_limit: int
    worker_byte_limit: int
    elapsed_ns: int
    peak_working_set_bytes: int
    sealed_cold_bytes: int


class TrainingShardResourceBudgetExceeded(RuntimeError):
    """K-03 吞吐、工作集、冷写或写放大超过调用方预注册预算。"""


@dataclass(frozen=True)
class TrainingShardResourceBudget:
    """由设备/实验 profile 注入的 K-03 多维资源硬预算。"""

    minimum_raw_records_per_window: int
    throughput_window_ns: int
    peak_working_set_byte_limit: int
    sealed_cold_byte_limit: int
    write_amplification_numerator_limit: int
    write_amplification_denominator: int

    def __post_init__(self) -> None:
        """核验吞吐可为零，其余窗口和上限必须为正严格整数。"""
        if (type(self.minimum_raw_records_per_window) is not int
                or self.minimum_raw_records_per_window < 0):
            raise ValueError("minimum_raw_records_per_window 必须是非负严格整数")
        for label, value in (
                ("throughput_window_ns", self.throughput_window_ns),
                ("peak_working_set_byte_limit", self.peak_working_set_byte_limit),
                ("sealed_cold_byte_limit", self.sealed_cold_byte_limit),
                ("write_amplification_numerator_limit",
                 self.write_amplification_numerator_limit),
                ("write_amplification_denominator",
                 self.write_amplification_denominator)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} 必须是正严格整数")

    def validate(
            self,
            metrics: TrainingShardRuntimeMetrics,
            barrier_result: TrainingBarrierResult,
            ) -> None:
        """逐维硬验资源报告，不把不同维度压成可抵消总分。"""
        if metrics.peak_working_set_bytes > self.peak_working_set_byte_limit:
            raise TrainingShardResourceBudgetExceeded(
                "K-03 峰值工作集超过预注册预算")
        if metrics.sealed_cold_bytes > self.sealed_cold_byte_limit:
            raise TrainingShardResourceBudgetExceeded(
                "K-03 sealed cold bytes 超过预注册预算")
        minimum = self.minimum_raw_records_per_window
        if minimum:
            if metrics.elapsed_ns <= 0:
                raise TrainingShardResourceBudgetExceeded(
                    "K-03 吞吐预算启用但 elapsed_ns 不可用")
            if (barrier_result.metrics.raw_records * self.throughput_window_ns
                    < minimum * metrics.elapsed_ns):
                raise TrainingShardResourceBudgetExceeded(
                    "K-03 训练吞吐低于预注册预算")
        canonical_bytes = barrier_result.metrics.canonical_segment_bytes
        if (canonical_bytes > 0
                and metrics.sealed_cold_bytes
                * self.write_amplification_denominator
                > canonical_bytes
                * self.write_amplification_numerator_limit):
            raise TrainingShardResourceBudgetExceeded(
                "K-03 逻辑写放大超过预注册预算")


@dataclass(frozen=True)
class TrainingArtifactReference:
    """已冷落盘 worker artifact 的轻量身份、校验和和尺寸引用。"""

    shard_key: tuple[int, ...]
    identity_key: tuple[int, ...]
    segment_checksum_key: tuple[int, ...]
    segment_size_bytes: int
    artifact_size_bytes: int

    @classmethod
    def from_artifact(
            cls,
            artifact: WorkerDeltaArtifact,
            ) -> TrainingArtifactReference:
        """从已核验 artifact 提取不持有 segment 记录的轻量引用。"""
        if not isinstance(artifact, WorkerDeltaArtifact):
            raise TypeError("artifact reference 来源类型错误")
        if artifact.segment is None:
            return cls(
                artifact.shard_key,
                artifact.identity_key,
                (),
                0,
                len(artifact.to_bytes()),
            )
        return cls(
            artifact.shard_key,
            artifact.identity_key,
            artifact.segment.checksum_key,
            artifact.segment.size_bytes,
            len(artifact.to_bytes()),
        )

    @property
    def has_segment(self) -> bool:
        """返回该 shard 是否产生了非空 worker segment。"""
        return bool(self.segment_checksum_key)


@dataclass(frozen=True)
class TrainingShardRunResult:
    """一次完整 shard 运行的 artifact、merge、receipt 和资源计量。"""

    artifact_references: tuple[TrainingArtifactReference, ...]
    barrier_result: TrainingBarrierResult
    receipt: TrainingBarrierPublishReceipt
    metrics: TrainingShardRuntimeMetrics


ExecutorFactory = Callable[[int], Executor]


def _thread_executor(worker_count: int) -> Executor:
    """按调用方 worker 数创建只影响调度的标准线程执行器。"""
    return ThreadPoolExecutor(max_workers=worker_count)


class TrainingShardRuntime:
    """先恢复或生成全部 shard artifact，再确定性 merge 和发布。"""

    def __init__(
            self,
            coordinator: TrainingBarrierCoordinator,
            producer_provider: TrainingShardProducerProvider,
            artifact_repositories: TrainingArtifactRepositoryProvider,
            worker_budget: SegmentBudget,
            *,
            executor_factory: ExecutorFactory = _thread_executor,
            telemetry_clock_ns: Callable[[], int] | None = None,
            working_set_source: Callable[[], int] | None = None,
            resource_budget: TrainingShardResourceBudget | None = None,
            ) -> None:
        """绑定单协调器、注入式 producer、窄仓库和每 shard 热预算。"""
        if not isinstance(coordinator, TrainingBarrierCoordinator):
            raise TypeError("coordinator 必须是 TrainingBarrierCoordinator")
        if not isinstance(producer_provider, TrainingShardProducerProvider):
            raise TypeError("producer_provider 协议错误")
        if not isinstance(
                artifact_repositories, TrainingArtifactRepositoryProvider):
            raise TypeError("artifact_repositories 协议错误")
        if not isinstance(worker_budget, SegmentBudget):
            raise TypeError("worker_budget 必须是 SegmentBudget")
        if not callable(executor_factory):
            raise TypeError("executor_factory 必须可调用")
        if resource_budget is not None:
            if not isinstance(resource_budget, TrainingShardResourceBudget):
                raise TypeError("resource_budget 类型错误")
            if telemetry_clock_ns is None or working_set_source is None:
                raise ValueError("启用 K-03 资源预算时必须注入时钟和工作集源")
        self.coordinator = coordinator
        self.producer_provider = producer_provider
        self.artifact_repositories = artifact_repositories
        self.worker_budget = worker_budget
        self.executor_factory = executor_factory
        self.telemetry_clock = TelemetryClock(telemetry_clock_ns)
        self.working_set_source = working_set_source
        self.resource_budget = resource_budget
        self._bound_producer_key = self._producer_key()
        if self._bound_producer_key != coordinator.producer_key:
            raise TrainingShardIntegrityError(
                "producer provider 与 coordinator 逻辑版本漂移")

    def _producer_key(self) -> tuple[int, ...]:
        """读取并核验 provider 当前稳定键。"""
        key = self.producer_provider.state_key()
        if (not isinstance(key, tuple) or not key
                or any(type(item) is not int for item in key)):
            raise TrainingShardIntegrityError(
                "producer provider state_key 必须是非空严格整数 tuple")
        return key

    def _require_producer_stable(self) -> None:
        """拒绝运行期间漂移的 producer 逻辑。"""
        if self._producer_key() != self._bound_producer_key:
            raise TrainingShardIntegrityError("producer provider state_key 已漂移")

    def _repository(
            self,
            shard_key: tuple[int, ...],
            ) -> AppendOnlyObjectRepository:
        """读取一个 shard 的窄仓库并核验协议。"""
        repository = self.artifact_repositories.repository_for(shard_key)
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("artifact repository provider 返回值协议错误")
        return repository

    def _artifact_identity(
            self,
            shard: LogicalTrainingShard,
            ) -> tuple[int, ...]:
        """构造与 worker 数和完成顺序无关的 artifact 槽位身份。"""
        coordinator = self.coordinator
        return worker_artifact_identity(
            manifest_key=coordinator.manifest.manifest_key,
            plan_key=coordinator.shard_plan.plan_key,
            execution_key=coordinator.execution_key,
            barrier_key=coordinator.barrier_key,
            shard_key=shard.shard_key,
            base_fence_key=coordinator.base_fence.stable_key(),
            descriptor_key=coordinator.descriptor_key,
            segment_key=worker_segment_key(
                coordinator.barrier_key,
                shard.shard_key,
            ),
        )

    def _restore(
            self,
            shard: LogicalTrainingShard,
            repository: AppendOnlyObjectRepository,
            ) -> WorkerDeltaArtifact | None:
        """恢复一个完整 artifact；槽位不存在时返回 None。"""
        try:
            artifact = WorkerDeltaArtifact.restore(
                repository,
                self._artifact_identity(shard),
            )
        except KeyError:
            return None
        self.coordinator.validate_artifact(artifact)
        return artifact

    def _request(
            self,
            shard: LogicalTrainingShard,
            ) -> TrainingShardWorkRequest:
        """按冻结 manifest 顺序形成一个 shard 的只读请求。"""
        coordinator = self.coordinator
        input_keys = frozenset(shard.input_keys)
        entries = tuple(
            entry for entry in coordinator.manifest.entries
            if entry.input_key in input_keys
        )
        return TrainingShardWorkRequest(
            shard,
            entries,
            coordinator.base_fence.stable_key(),
            coordinator.execution_key,
            coordinator.barrier_key,
        )

    def _produce(
            self,
            shard: LogicalTrainingShard,
            fault_injector: SegmentRepositoryFaultInjector | None,
            ) -> WorkerDeltaArtifact:
        """在隔离热 delta 中生成、封存并持久化一个 shard。"""
        self._require_producer_stable()
        repository = self._repository(shard.shard_key)
        producer = self.producer_provider.producer_for(shard.shard_key)
        if not isinstance(producer, TrainingShardProducer):
            raise TypeError("producer_for 返回值协议错误")
        coordinator = self.coordinator
        delta = WorkerLocalDelta(
            manifest=coordinator.manifest,
            shard_plan=coordinator.shard_plan,
            shard=shard,
            producer_key=self._bound_producer_key,
            execution_key=coordinator.execution_key,
            barrier_key=coordinator.barrier_key,
            base_fence=coordinator.base_fence,
            descriptor_key=coordinator.descriptor_key,
            version_key=coordinator.version_key,
            dependencies=coordinator.dependencies,
            budget=self.worker_budget,
        )
        returned = producer.produce(self._request(shard), delta.append)
        if returned is not None:
            raise TypeError("TrainingShardProducer.produce 必须返回 None")
        self._require_producer_stable()
        artifact = delta.seal()
        artifact.persist(repository, fault_injector=fault_injector)
        self.coordinator.validate_artifact(artifact)
        return artifact

    def run(
            self,
            worker_count: int,
            *,
            store: TieredSegmentStore,
            receipt_repository: AppendOnlyObjectRepository,
            tier_key: tuple[int, ...],
            manifest_key: tuple[int, ...],
            migration_key: tuple[int, ...],
            artifact_fault_injector: SegmentRepositoryFaultInjector | None = None,
            barrier_fault_injector: TrainingBarrierFaultInjector | None = None,
            repository_fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> TrainingShardRunResult:
        """完成全部 shard 后单点 merge/publish；任一失败都不形成 receipt。"""
        if type(worker_count) is not int or worker_count <= 0:
            raise ValueError("worker_count 必须是正严格整数")
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("store 必须是 TieredSegmentStore")
        if not isinstance(receipt_repository, AppendOnlyObjectRepository):
            raise TypeError("receipt_repository 协议错误")
        started_ns = self.telemetry_clock.now_ns()
        peak_working_set = sample_working_set(self.working_set_source)

        def sample_peak() -> None:
            """在主要边界提升本次运行的工作集峰值采样。"""
            nonlocal peak_working_set
            peak_working_set = max(
                peak_working_set,
                sample_working_set(self.working_set_source),
            )

        self._require_producer_stable()
        references: dict[tuple[int, ...], TrainingArtifactReference] = {}
        missing = []
        for shard in self.coordinator.shard_plan.shards:
            repository = self._repository(shard.shard_key)
            artifact = self._restore(shard, repository)
            if artifact is None:
                missing.append(shard)
            else:
                references[shard.shard_key] = (
                    TrainingArtifactReference.from_artifact(artifact))
            sample_peak()
        restored_count = len(references)
        if missing:
            executor = self.executor_factory(worker_count)
            if not isinstance(executor, Executor):
                raise TypeError("executor_factory 必须返回 Executor")
            futures = {
                executor.submit(
                    self._produce,
                    shard,
                    artifact_fault_injector,
                ): shard
                for shard in missing
            }
            try:
                for future in as_completed(futures):
                    shard = futures[future]
                    artifact = future.result()
                    references[shard.shard_key] = (
                        TrainingArtifactReference.from_artifact(artifact))
                    sample_peak()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        self._require_producer_stable()
        ordered_references = tuple(
            references[shard.shard_key]
            for shard in self.coordinator.shard_plan.shards
        )

        def load_artifact(shard: LogicalTrainingShard) -> WorkerDeltaArtifact:
            """在 merge 时逐 shard 从冷仓恢复，不保留已消费 segment。"""
            artifact = self._restore(
                shard,
                self._repository(shard.shard_key),
            )
            if artifact is None:
                raise TrainingShardIntegrityError(
                    "barrier merge 前 worker artifact 消失")
            reference = TrainingArtifactReference.from_artifact(artifact)
            if reference != references[shard.shard_key]:
                raise TrainingShardIntegrityError(
                    "barrier merge 前 worker artifact 内容漂移")
            return artifact

        barrier_result = self.coordinator.merge_stream(load_artifact)
        sample_peak()
        receipt = self.coordinator.publish(
            barrier_result,
            store=store,
            receipt_repository=receipt_repository,
            tier_key=tier_key,
            manifest_key=manifest_key,
            migration_key=migration_key,
            barrier_fault_injector=barrier_fault_injector,
            repository_fault_injector=repository_fault_injector,
        )
        sample_peak()
        elapsed_ns = self.telemetry_clock.now_ns() - started_ns
        if elapsed_ns < 0:
            raise TrainingShardIntegrityError("K-03 注入时钟发生倒退")
        sealed_cold_bytes = (
            sum(item.artifact_size_bytes for item in ordered_references)
            + barrier_result.metrics.canonical_segment_bytes
            + len(receipt.to_bytes())
        )
        metrics = TrainingShardRuntimeMetrics(
            worker_count,
            len(self.coordinator.shard_plan.shards),
            restored_count,
            len(missing),
            min(worker_count, len(missing)),
            self.worker_budget.object_limit,
            self.worker_budget.byte_limit,
            elapsed_ns,
            peak_working_set,
            sealed_cold_bytes,
        )
        if self.resource_budget is not None:
            self.resource_budget.validate(metrics, barrier_result)
        return TrainingShardRunResult(
            ordered_references,
            barrier_result,
            receipt,
            metrics,
        )


__all__ = [
    "ExecutorFactory",
    "SharedTrainingArtifactRepository",
    "TrainingArtifactReference",
    "TrainingArtifactRepositoryProvider",
    "TrainingShardProducer",
    "TrainingShardProducerProvider",
    "TrainingShardResourceBudget",
    "TrainingShardResourceBudgetExceeded",
    "TrainingShardRunResult",
    "TrainingShardRuntime",
    "TrainingShardRuntimeMetrics",
    "TrainingShardWorkRequest",
]
