"""PW-01 双 Memory 完整回答的可重启整数性能曲线。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any, Callable

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_PROVISIONAL,
)
from pure_integer_ai.cognition.shared.memory_event import (
    LIFECYCLE_ACTIVE,
    MEMORY_OBJECT_HYPOTHESIS,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryCandidateBundle,
)
from pure_integer_ai.cognition.shared.memory_hot_set import (
    encode_memory_candidate,
    encode_memory_candidate_payload,
    memory_query_index_record_key,
)
from pure_integer_ai.cognition.shared.post_weaning import (
    PostWeaningIntakeRequest,
)
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _ACCESS,
    _COLD,
    _DEPENDENCIES,
    _NoPrefetch,
    _close_outer_lifecycle,
    _complete_source,
    _observation,
    _post_weaning_manifest,
    _refresh_projection,
    _restore_runtime,
    prepare_facility_context,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    MemoryProjectionSegment,
    MemoryQueryIndexPartition,
    MemoryQueryIndexProjectionManifest,
    memory_hot_set_runtimes,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
)
from pure_integer_ai.experiments.pw01_controlled_reading import (
    PW01ControlledReadingParser,
    PW01_HYPOTHESIS_KIND,
    build_pw01_question_dialogue,
    install_pw01_controlled_query,
    pw01_source,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.experiments.v02_run_store import (
    HostMonotonicClock,
    HostProcessMemory,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


PW01_PERFORMANCE_SCALES = (3_200, 12_800, 51_200)
_PERFORMANCE_VERSION_KEY = (20260807, 91, 1)
_PERFORMANCE_BATCH_ID = 2026080791
_PERFORMANCE_LINEAGE_ID = 91
_PERFORMANCE_SEGMENT_OBJECT_LIMIT = 512
# 冷查询只取有界 Top-K；小段限制整段反序列化的读取放大。
_PERFORMANCE_INDEX_SEGMENT_OBJECT_LIMIT = 512


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW01PerformanceMeasurement:
    """一次 fresh 或 restart 完整回答的纯整数物理测量。"""

    elapsed_ns: int
    end_to_end_elapsed_ns: int
    memory_resolve_elapsed_ns: int
    dialogue_build_elapsed_ns: int
    question_operation_elapsed_ns: int
    generation_elapsed_ns: int
    considered_candidates: int
    partition_rows_scanned: int
    page_faults: int
    page_in_records: int
    cold_read_bytes: int
    rss_before_bytes: int
    rss_after_memory_query_bytes: int
    rss_memory_query_endpoint_max_bytes: int
    rss_after_bytes: int
    rss_process_peak_before_bytes: int
    rss_peak_bytes: int
    database_bytes: int
    answer_complete: int
    source_exact: int

    def __post_init__(self) -> None:
        """拒绝负数、布尔伪整数和未完成的性能点。"""
        for name, value in self.as_dict().items():
            if type(value) is not int or value < 0:
                raise ValueError(f"PW-01 performance {name} 必须是非负严格整数")
        if self.answer_complete != 1 or self.source_exact != 1:
            raise ValueError("PW-01 performance 不能记录不正确回答")
        if self.end_to_end_elapsed_ns != (
                self.dialogue_build_elapsed_ns
                + self.question_operation_elapsed_ns):
            raise ValueError("PW-01 performance 完整时延分账不闭合")
        if self.elapsed_ns != self.question_operation_elapsed_ns:
            raise ValueError("PW-01 elapsed 与 question operation 漂移")
        if self.memory_resolve_elapsed_ns > self.question_operation_elapsed_ns:
            raise ValueError("PW-01 Memory resolve 时延超过 question operation")
        if self.generation_elapsed_ns > self.question_operation_elapsed_ns:
            raise ValueError("PW-01 generation 时延超过 question operation")
        if self.rss_peak_bytes < max(
                self.rss_before_bytes,
                self.rss_after_memory_query_bytes,
                self.rss_memory_query_endpoint_max_bytes,
                self.rss_after_bytes,
                self.rss_process_peak_before_bytes):
            raise ValueError("PW-01 performance 进程峰值工作集不闭合")
        if self.rss_memory_query_endpoint_max_bytes != max(
                self.rss_before_bytes,
                self.rss_after_memory_query_bytes):
            raise ValueError("PW-01 query RSS 端点上界不闭合")

    def as_dict(self) -> dict[str, int]:
        """返回可规范 JSON 发布的字段映射。"""
        return {
            "elapsed_ns": self.elapsed_ns,
            "end_to_end_elapsed_ns": self.end_to_end_elapsed_ns,
            "memory_resolve_elapsed_ns": self.memory_resolve_elapsed_ns,
            "dialogue_build_elapsed_ns": self.dialogue_build_elapsed_ns,
            "question_operation_elapsed_ns": self.question_operation_elapsed_ns,
            "generation_elapsed_ns": self.generation_elapsed_ns,
            "considered_candidates": self.considered_candidates,
            "partition_rows_scanned": self.partition_rows_scanned,
            "page_faults": self.page_faults,
            "page_in_records": self.page_in_records,
            "cold_read_bytes": self.cold_read_bytes,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_memory_query_bytes": self.rss_after_memory_query_bytes,
            "rss_memory_query_endpoint_max_bytes": (
                self.rss_memory_query_endpoint_max_bytes),
            "rss_after_bytes": self.rss_after_bytes,
            "rss_process_peak_before_bytes": (
                self.rss_process_peak_before_bytes),
            "rss_peak_bytes": self.rss_peak_bytes,
            "database_bytes": self.database_bytes,
            "answer_complete": self.answer_complete,
            "source_exact": self.source_exact,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW01PerformanceScalePoint:
    """一个总候选规模的 fresh/restart 成对测量。"""

    total_projection_records: int
    read_projection_records: int
    interact_projection_records: int
    candidate_projection_build_elapsed_ns: int
    query_index_build_elapsed_ns: int
    build_rss_before_bytes: int
    candidate_projection_rss_after_bytes: int
    query_index_rss_after_bytes: int
    fresh: PW01PerformanceMeasurement
    restart: PW01PerformanceMeasurement

    def __post_init__(self) -> None:
        """核验规模分账和成对测量类型。"""
        for name, value in (
                ("total_projection_records", self.total_projection_records),
                ("read_projection_records", self.read_projection_records),
                ("interact_projection_records", self.interact_projection_records)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"PW-01 scale {name} 必须是正严格整数")
        if (self.read_projection_records + self.interact_projection_records
                != self.total_projection_records):
            raise ValueError("PW-01 scale 双 Memory 记录分账不闭合")
        for name, value in (
                ("candidate_projection_build_elapsed_ns",
                 self.candidate_projection_build_elapsed_ns),
                ("query_index_build_elapsed_ns",
                 self.query_index_build_elapsed_ns),
                ("build_rss_before_bytes", self.build_rss_before_bytes),
                ("candidate_projection_rss_after_bytes",
                 self.candidate_projection_rss_after_bytes),
                ("query_index_rss_after_bytes",
                 self.query_index_rss_after_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"PW-01 scale {name} 必须是非负严格整数")
        if (self.candidate_projection_build_elapsed_ns <= 0
                or min(
                    self.build_rss_before_bytes,
                    self.candidate_projection_rss_after_bytes,
                    self.query_index_rss_after_bytes,
                ) <= 0):
            raise ValueError("PW-01 scale 构建时延或 RSS 端点未闭合")
        if (not isinstance(self.fresh, PW01PerformanceMeasurement)
                or not isinstance(self.restart, PW01PerformanceMeasurement)):
            raise TypeError("PW-01 scale measurement 类型错误")

    def as_dict(self) -> dict[str, object]:
        """返回成对曲线点的规范字段。"""
        return {
            "total_projection_records": self.total_projection_records,
            "read_projection_records": self.read_projection_records,
            "interact_projection_records": self.interact_projection_records,
            "candidate_projection_build_elapsed_ns": (
                self.candidate_projection_build_elapsed_ns),
            "query_index_build_elapsed_ns": self.query_index_build_elapsed_ns,
            "build_rss_before_bytes": self.build_rss_before_bytes,
            "candidate_projection_rss_after_bytes": (
                self.candidate_projection_rss_after_bytes),
            "query_index_rss_after_bytes": self.query_index_rss_after_bytes,
            "fresh": self.fresh.as_dict(),
            "restart": self.restart.as_dict(),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class PW01PerformanceReport:
    """按严格递增规模保存的完整回答基线报告。"""

    schema_version: int
    scales: tuple[int, ...]
    points: tuple[PW01PerformanceScalePoint, ...]

    def __post_init__(self) -> None:
        """核验版本、规模顺序和点覆盖完全一致。"""
        if self.schema_version != 4:
            raise ValueError("PW-01 performance report 版本未注册")
        if (not isinstance(self.scales, tuple)
                or any(type(item) is not int or item <= 0
                       for item in self.scales)
                or tuple(sorted(set(self.scales))) != self.scales):
            raise ValueError("PW-01 performance scales 必须严格递增且唯一")
        if (not isinstance(self.points, tuple)
                or any(not isinstance(item, PW01PerformanceScalePoint)
                       for item in self.points)):
            raise TypeError("PW-01 performance points 类型错误")
        if tuple(item.total_projection_records for item in self.points) != self.scales:
            raise ValueError("PW-01 performance points 未覆盖预注册规模")

    def as_dict(self) -> dict[str, object]:
        """返回不含路径、时钟文本或浮点数的公开报告。"""
        return {
            "schema_version": self.schema_version,
            "scales": list(self.scales),
            "points": [item.as_dict() for item in self.points],
        }


def _performance_policy(*, indexed: bool) -> QueryHotSetPolicy:
    """返回全扫基线或索引路径各自固定的有界设备读取策略。"""
    if type(indexed) is not bool:
        raise TypeError("PW-01 performance indexed policy 必须是 bool")
    cache_records = 16 if indexed else 128
    page_records = 8 if indexed else 64
    return QueryHotSetPolicy(
        SegmentBudget(cache_records, 16_000_000),
        SegmentBudget(page_records, 8_000_000),
        _NoPrefetch(),
        16,
    )


def _database_size_bytes(path: Path) -> int:
    """合计 SQLite 主文件及当前 WAL/SHM 的真实磁盘字节。"""
    return sum(
        item.stat().st_size
        for item in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if item.exists()
    )


def _working_set_sample() -> tuple[int, int]:
    """读取当前及不可复位的进程峰值工作集，拒绝端点伪装成峰值。"""
    sample = HostProcessMemory()()
    current = sample.get("current_working_set_bytes", 0)
    peak = sample.get("process_peak_working_set_bytes", 0)
    if (type(current) is not int or current <= 0
            or type(peak) is not int or peak < current):
        raise RuntimeError("PW-01 performance 无法读取闭合的进程工作集")
    return current, peak


# object-model: lifecycle; owner=measurement-call; cleanup=scope-end
class _MeasuredGenerationExecutor:
    """只在一次性能测量内代理 typed generator 并记录 execute 时延。"""

    def __init__(
            self,
            delegate: Any,
            clock_ns: Callable[[], int],
            ) -> None:
        """绑定原 executor 和同一 host monotonic clock。"""
        if not callable(getattr(delegate, "execute", None)):
            raise TypeError("PW-01 measured generator 缺少 execute")
        if not callable(clock_ns):
            raise TypeError("PW-01 measured generator clock 不可调用")
        self.delegate = delegate
        self.clock_ns = clock_ns
        self.elapsed_ns: int | None = None

    def execute(self, request: Any) -> Any:
        """委托唯一一次 generation，并保存严格正整数时延。"""
        if self.elapsed_ns is not None:
            raise RuntimeError("PW-01 同次 question 重复调用 generator")
        started = self.clock_ns()
        result = self.delegate.execute(request)
        elapsed = self.clock_ns() - started
        if type(elapsed) is not int or elapsed <= 0:
            raise RuntimeError("PW-01 generation clock 未返回正整数耗时")
        self.elapsed_ns = elapsed
        return result


def _read_resolver(ctx: TrainContext) -> Any:
    """返回联邦中唯一绑定 memory_read aggregate 的 resolver。"""
    matches = tuple(
        item for item in ctx.memory_resolver_runtime.resolvers
        if item.aggregates is ctx.memory_read_aggregates
    )
    if len(matches) != 1:
        raise RuntimeError("PW-01 performance 缺少唯一 reading resolver")
    return matches[0]


def _target_bundle(ctx: TrainContext) -> MemoryCandidateBundle:
    """从真实已摄入来源恢复唯一目标候选完整 bundle。"""
    source = pw01_source(parser_version=1)
    records = ctx.memory_read_aggregates.query(
        access=_ACCESS,
        hypothesis_kind=PW01_HYPOTHESIS_KIND,
        source=source,
    )
    if len(records) != 1:
        raise RuntimeError("PW-01 performance 缺少唯一真实阅读目标")
    return _read_resolver(ctx).load_bundle(records[0], access=_ACCESS)


def _synthetic_bundle(
        target: MemoryCandidateBundle,
        ordinal: int,
        ) -> MemoryCandidateBundle:
    """构造不带来源、不能完成 held-out 的只读负载候选。"""
    if type(ordinal) is not int or ordinal <= 0:
        raise ValueError("PW-01 synthetic ordinal 必须是正严格整数")
    hypothesis = HypothesisKey(
        target.hypothesis.hypothesis_kind,
        (1_000_000 + ordinal,),
        target.hypothesis.competition_key,
        target.hypothesis.scope,
        target.hypothesis.observation,
    )
    hypothesis_ref = MemoryObjectRef(
        target.hypothesis_ref.memory_space,
        target.hypothesis_ref.owner,
        target.hypothesis_ref.versions,
        MEMORY_OBJECT_HYPOTHESIS,
        hypothesis.stable_key(),
    )
    aggregate = replace(
        target.aggregate,
        hypothesis_hash=target.aggregate.hypothesis_hash + ordinal,
        created_seq=ordinal,
        last_observed_seq=0,
        last_supported_seq=0,
        last_refuted_seq=0,
        last_used_seq=0,
        support_count=0,
        contradict_count=0,
        unknown_count=0,
        independent_source_count=0,
        support_source_count=0,
        contradict_source_count=0,
        use_count=0,
        lifecycle_state=LIFECYCLE_ACTIVE,
        evidence_state=MEMORY_EVIDENCE_PROVISIONAL,
    )
    return MemoryCandidateBundle(
        hypothesis_ref,
        hypothesis,
        aggregate,
        (),
        (),
    )


def _publish_synthetic_read_projection(
        ctx: TrainContext,
        record_count: int,
        ) -> MemoryCandidateProjectionManifest:
    """发布一个真实目标加合成负载的只读候选投影。"""
    if type(record_count) is not int or record_count <= 0:
        raise ValueError("PW-01 performance record_count 必须是正严格整数")
    resolver = _read_resolver(ctx)
    target = _target_bundle(ctx)
    source_state_key = resolver.aggregates.event_log.projection_state_key()
    source_fence = source_state_key[0]
    projection_key = (
        20260807,
        91,
        record_count,
        source_fence,
    )
    summaries: list[MemoryProjectionSegment] = []
    delta = OpenHotDelta(
        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
        _PERFORMANCE_VERSION_KEY,
        _DEPENDENCIES,
        SegmentBudget(_PERFORMANCE_SEGMENT_OBJECT_LIMIT, 8_000_000),
    )

    def flush() -> None:
        """封存并发布当前非空 delta。"""
        nonlocal delta
        if delta.object_count == 0:
            return
        ordinal = len(summaries) + 1
        segment = delta.seal(
            (20260807, 92, record_count, ordinal),
            source_fence,
        )
        ctx.tiered_segment_store.publish_segment(
            segment,
            tier_key=_COLD,
            manifest_key=(20260807, 93, record_count, ordinal),
            migration_key=(20260807, 94, record_count, ordinal),
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
            _PERFORMANCE_VERSION_KEY,
            _DEPENDENCIES,
            SegmentBudget(_PERFORMANCE_SEGMENT_OBJECT_LIMIT, 8_000_000),
        )

    appended = 0
    for bundle in chain(
            (target,),
            (_synthetic_bundle(target, ordinal)
             for ordinal in range(1, record_count))):
        record = encode_memory_candidate(projection_key, bundle)
        try:
            delta.append(record)
        except SegmentBudgetExceeded:
            flush()
            delta.append(record)
        appended += 1
    flush()
    if appended != record_count:
        raise RuntimeError("PW-01 synthetic projection 记录数未完整生成")
    current = ctx.tiered_segment_store.current_manifest()
    if current is None:
        raise RuntimeError("PW-01 synthetic projection 没有 location manifest")
    manifest = MemoryCandidateProjectionManifest(
        projection_key,
        resolver.aggregates.event_log.memory_space_identity,
        resolver.aggregates.event_log.memory_space_id,
        _ACCESS,
        ctx.memory_read_hot_set_runtime.projection.hypothesis_kinds,
        source_fence,
        source_state_key,
        _PERFORMANCE_VERSION_KEY,
        _DEPENDENCIES,
        tuple(summaries),
        current.publish_epoch,
    )
    manifest.validate_store(ctx.tiered_segment_store)
    if manifest.record_count != record_count:
        raise RuntimeError("PW-01 synthetic projection 记录数漂移")
    return manifest


def _publish_synthetic_query_index(
        ctx: TrainContext,
        candidate_count: int,
        ) -> MemoryQueryIndexProjectionManifest:
    """为唯一 candidate 负载发布 exact/fallback 双入口查询索引。

    候选只构造和编码一次；两个索引入口分别保留独立 delta，避免
    exact/fallback canonical range 交叉，同时消除旧实现的双遍生成成本。
    """
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("PW-01 query index candidate_count 必须是正严格整数")
    resolver = _read_resolver(ctx)
    planner = resolver.score_provider
    target = _target_bundle(ctx)
    source_state_key = resolver.aggregates.event_log.projection_state_key()
    source_fence = source_state_key[0]
    projection_key = (
        20260807,
        95,
        candidate_count,
        source_fence,
    )
    summaries: list[MemoryProjectionSegment] = []
    deltas = [
        OpenHotDelta(
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            _PERFORMANCE_VERSION_KEY,
            _DEPENDENCIES,
            SegmentBudget(
                _PERFORMANCE_INDEX_SEGMENT_OBJECT_LIMIT,
                16_000_000,
            ),
        )
        for _ in range(2)
    ]
    segment_ordinals = [0, 0]

    def flush(entry_ordinal: int) -> None:
        """封存并发布一个入口组当前非空的查询索引 delta。"""
        delta = deltas[entry_ordinal]
        if delta.object_count == 0:
            return
        segment_ordinals[entry_ordinal] += 1
        ordinal = segment_ordinals[entry_ordinal]
        segment = delta.seal(
            (20260807, 96, candidate_count, entry_ordinal, ordinal),
            source_fence,
        )
        ctx.tiered_segment_store.publish_segment(
            segment,
            tier_key=_COLD,
            manifest_key=(20260807, 97, candidate_count, entry_ordinal, ordinal),
            migration_key=(20260807, 98, candidate_count, entry_ordinal, ordinal),
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
        deltas[entry_ordinal] = OpenHotDelta(
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            _PERFORMANCE_VERSION_KEY,
            _DEPENDENCIES,
            SegmentBudget(
                _PERFORMANCE_INDEX_SEGMENT_OBJECT_LIMIT,
                16_000_000,
            ),
        )

    appended = 0
    # exact 与 fallback 的范围必须分 delta；候选本身和 payload 只计算一次。
    for ordinal in range(candidate_count):
        bundle = target if ordinal == 0 else _synthetic_bundle(target, ordinal)
        payload = encode_memory_candidate_payload(bundle)
        entries = planner.index_entries(bundle)
        if len(entries) != 2:
            raise RuntimeError("PW-01 query index planner 入口数漂移")
        for entry_ordinal, entry in enumerate(entries):
            record = SegmentRecord(
                memory_query_index_record_key(
                    projection_key,
                    bundle.aggregate,
                    entry,
                ),
                payload,
            )
            delta = deltas[entry_ordinal]
            try:
                if not delta.append(record):
                    raise RuntimeError("PW-01 query index 出现重复记录身份")
            except SegmentBudgetExceeded:
                flush(entry_ordinal)
                if not deltas[entry_ordinal].append(record):
                    raise RuntimeError("PW-01 query index 出现跨段重复记录身份")
            appended += 1
    flush(0)
    flush(1)
    if appended != candidate_count * 2:
        raise RuntimeError("PW-01 query index 双入口记录数不闭合")
    current = ctx.tiered_segment_store.current_manifest()
    if current is None:
        raise RuntimeError("PW-01 query index 没有 location manifest")
    storage = MemoryCandidateProjectionManifest(
        projection_key,
        resolver.aggregates.event_log.memory_space_identity,
        resolver.aggregates.event_log.memory_space_id,
        _ACCESS,
        ctx.memory_read_hot_set_runtime.projection.hypothesis_kinds,
        source_fence,
        source_state_key,
        _PERFORMANCE_VERSION_KEY,
        _DEPENDENCIES,
        tuple(summaries),
        current.publish_epoch,
    )
    result = MemoryQueryIndexProjectionManifest(
        storage,
        planner.state_key(),
        (MemoryQueryIndexPartition(
            target.aggregate.hypothesis_kind_hash,
            target.aggregate.owner_key,
            candidate_count,
            candidate_count * 2,
        ),),
    )
    result.validate_store(ctx.tiered_segment_store)
    return result


def _learn_target(ctx: TrainContext) -> None:
    """经正式 reading route 摄入唯一可完成 held-out 的真实来源。"""
    routes, manifest = _post_weaning_manifest(ctx, ctx.f01_source)
    learned_source = pw01_source(parser_version=1)
    request = PostWeaningIntakeRequest(
        routes.reading,
        learned_source,
        "PW-01 performance controlled source",
        "CC0-1.0",
        _PERFORMANCE_BATCH_ID,
        parser=PW01ControlledReadingParser(
            learned_source,
            EVIDENCE_SUPPORT,
            _PERFORMANCE_LINEAGE_ID,
        ),
        trace=(20260807, 91, 1),
    )
    operation = PostWeaningDryRunRuntime(ctx, manifest).run_intake(request)
    if not operation.report.core_unchanged:
        raise RuntimeError("PW-01 performance 摄入改变了 Core")
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)


def _measure_answer(
        ctx: TrainContext,
        database_path: Path,
        source: Any,
        observation: Any,
        *,
        clock_ns: Callable[[], int] | None = None,
        ) -> PW01PerformanceMeasurement:
    """执行一次不含投影构建的完整问答并汇总双热集物理计数。"""
    _close_outer_lifecycle(ctx)
    clock = HostMonotonicClock() if clock_ns is None else clock_ns
    if not callable(clock):
        raise TypeError("PW-01 performance clock_ns 必须可调用")
    fixture = None
    original_generator = None
    measured_generator = None
    resolve_elapsed_values: list[int] = []
    query_rss_values: list[int] = []

    def record_resolve_elapsed(value: int) -> None:
        """接收正式 question 内唯一一次 federated Memory resolver 时延。"""
        if resolve_elapsed_values:
            raise RuntimeError("PW-01 同次回答重复记录 Memory resolve")
        if type(value) is not int or value <= 0:
            raise ValueError("PW-01 Memory resolve elapsed 必须是正严格整数")
        resolve_elapsed_values.append(value)
        query_rss, _ = _working_set_sample()
        query_rss_values.append(query_rss)

    try:
        repository = SourceRecordRepository(ctx.backend)
        for ordinal, trace in enumerate(
                _target_bundle(ctx).source_traces, start=1):
            _complete_source(repository, trace, ordinal)
        dialogue_build_started = clock()
        fixture, dialogue = build_pw01_question_dialogue(
            ctx,
            source,
            observation,
            source_records_ready=True,
        )
        dialogue_built = clock()
        dialogue_build_elapsed = dialogue_built - dialogue_build_started
        _, manifest = _post_weaning_manifest(ctx, source)
        original_generator = dialogue.runtime.generator
        measured_generator = _MeasuredGenerationExecutor(
            original_generator, clock)
        dialogue.runtime.generator = measured_generator
        rss_before, process_peak_before = _working_set_sample()
        operation_started = clock()
        try:
            with ctx.memory_resolver_runtime.measure_resolve(
                    clock, record_resolve_elapsed):
                operation = PostWeaningDryRunRuntime(
                    ctx, manifest).run_question(dialogue, fixture.request)
        finally:
            dialogue.runtime.generator = original_generator
        operation_finished = clock()
        question_operation_elapsed = operation_finished - operation_started
        end_to_end_elapsed = dialogue_build_elapsed + question_operation_elapsed
        elapsed = question_operation_elapsed
        if type(elapsed) is not int or elapsed <= 0:
            raise RuntimeError("PW-01 performance clock 未返回正整数耗时")
        if (type(dialogue_build_elapsed) is not int
                or dialogue_build_elapsed <= 0
                or type(question_operation_elapsed) is not int
                or question_operation_elapsed <= 0):
            raise RuntimeError("PW-01 performance phase clock 未闭合")
        if (len(resolve_elapsed_values) != 1 or len(query_rss_values) != 1):
            raise RuntimeError("PW-01 performance 缺少唯一正式 Memory resolve 测量")
        if (measured_generator.elapsed_ns is None
                or measured_generator.elapsed_ns <= 0):
            raise RuntimeError("PW-01 performance 缺少 generation 时延")
        rss_after, process_peak_after = _working_set_sample()
        if process_peak_after < process_peak_before:
            raise RuntimeError("PW-01 performance 进程峰值工作集发生回退")
        hot_sets = memory_hot_set_runtimes(ctx)
        metrics = tuple(item.metrics() for item in hot_sets)
        if any(item is None for item in metrics):
            raise RuntimeError("PW-01 performance 缺少双热集 query metrics")
        considered = tuple(item.considered_count() for item in hot_sets)
        if any(item is None for item in considered):
            raise RuntimeError("PW-01 performance 缺少候选考虑数")
        question = operation.result.question
        exact = int({item.trace.source for item in operation.result.sources} == {
            pw01_source(parser_version=1)})
        return PW01PerformanceMeasurement(
            elapsed,
            end_to_end_elapsed,
            resolve_elapsed_values[0],
            dialogue_build_elapsed,
            question_operation_elapsed,
            measured_generator.elapsed_ns,
            sum(item for item in considered if item is not None),
            sum(item.page_in_records for item in metrics if item is not None),
            sum(item.page_faults for item in metrics if item is not None),
            sum(item.page_in_records for item in metrics if item is not None),
            sum(item.cold_read_bytes for item in metrics if item is not None),
            rss_before,
            query_rss_values[0],
            max(rss_before, query_rss_values[0]),
            rss_after,
            process_peak_before,
            process_peak_after,
            _database_size_bytes(database_path),
            int(question.complete),
            exact,
        )
    finally:
        if fixture is not None:
            fixture.close()
        _close_outer_lifecycle(ctx)


def _prepare_fresh(
        database_path: Path,
        *,
        use_query_index: bool,
        ) -> tuple[TrainContext, Any, Any]:
    """创建新 SQLite、安装完整设施并摄入真实目标。"""
    if database_path.exists():
        raise FileExistsError(f"PW-01 performance database 已存在: {database_path}")
    backend = SQLiteBackend(str(database_path))
    ctx = make_train_context(backend, companion=True)
    prepare_facility_context(ctx)
    install_pw01_controlled_query(ctx)
    ctx.memory_read_hot_set_runtime.replace_policy(
        _performance_policy(indexed=use_query_index))
    _learn_target(ctx)
    return ctx, ctx.f01_source, ctx.f01_observation


def _restore(
        database_path: Path,
        primary_projection_key: tuple[int, ...],
        read_projection_key: tuple[int, ...],
        query_index_key: tuple[int, ...] | None,
        observation_ref: MemoryObjectRef,
        ) -> tuple[TrainContext, Any, Any]:
    """真重开 SQLite 并恢复同一双投影回答环境。"""
    backend = SQLiteBackend(str(database_path))
    ctx, source, _ = _restore_runtime(backend, primary_projection_key)
    install_pw01_controlled_query(ctx)
    ctx.memory_read_hot_set_runtime.replace_policy(
        _performance_policy(indexed=query_index_key is not None))
    read_projection = MemoryCandidateProjectionManifest.from_stable_key(
        read_projection_key)
    read_projection.validate_store(ctx.tiered_segment_store)
    if query_index_key is None:
        ctx.memory_read_hot_set_runtime.replace_projection(read_projection)
    else:
        query_index = MemoryQueryIndexProjectionManifest.from_stable_key(
            query_index_key)
        query_index.validate_store(ctx.tiered_segment_store)
        ctx.memory_read_hot_set_runtime.replace_indexed_projection(
            read_projection, query_index)
    return ctx, source, _observation(ctx, observation_ref)


def run_pw01_dual_memory_scale_curve(
        database_path: str | Path,
        *,
        scales: tuple[int, ...] = PW01_PERFORMANCE_SCALES,
        use_query_index: bool = False,
        ) -> PW01PerformanceReport:
    """执行预注册三档 fresh/restart 完整回答基线。"""
    normalized = tuple(scales)
    if type(use_query_index) is not bool:
        raise TypeError("PW-01 performance use_query_index 必须是 bool")
    if (not normalized
            or tuple(sorted(set(normalized))) != normalized
            or any(type(item) is not int or item <= 3 for item in normalized)):
        raise ValueError("PW-01 performance scales 必须严格递增、唯一且大于 3")
    path = Path(database_path).resolve()
    ctx, source, observation = _prepare_fresh(
        path, use_query_index=use_query_index)
    observation_ref = observation.event.object_ref
    points: list[PW01PerformanceScalePoint] = []
    clock = HostMonotonicClock()
    try:
        for total_records in normalized:
            primary = _refresh_projection(ctx)
            interact_count = primary.record_count
            read_count = total_records - interact_count
            if read_count <= 0:
                raise ValueError("PW-01 performance 总规模小于 interaction 投影")
            build_rss_before, _ = _working_set_sample()
            candidate_build_started = clock()
            read_projection = _publish_synthetic_read_projection(
                ctx, read_count)
            candidate_build_elapsed = clock() - candidate_build_started
            candidate_rss_after, _ = _working_set_sample()
            ctx.memory_read_hot_set_runtime.replace_projection(read_projection)
            query_index = None
            query_index_build_elapsed = 0
            if use_query_index:
                index_build_started = clock()
                query_index = _publish_synthetic_query_index(
                    ctx, read_count)
                query_index_build_elapsed = clock() - index_build_started
                ctx.memory_read_hot_set_runtime.replace_query_index(
                    query_index)
            query_index_rss_after, _ = _working_set_sample()
            fresh = _measure_answer(
                ctx, path, source, observation, clock_ns=clock)

            primary = _refresh_projection(ctx)
            ctx.backend.commit()
            primary_key = primary.stable_key()
            read_key = read_projection.stable_key()
            query_index_key = (
                None if query_index is None else query_index.stable_key())
            ctx.backend.close()

            ctx, source, observation = _restore(
                path,
                primary_key,
                read_key,
                query_index_key,
                observation_ref,
            )
            restart = _measure_answer(
                ctx, path, source, observation, clock_ns=clock)
            points.append(PW01PerformanceScalePoint(
                total_records,
                read_count,
                interact_count,
                candidate_build_elapsed,
                query_index_build_elapsed,
                build_rss_before,
                candidate_rss_after,
                query_index_rss_after,
                fresh,
                restart,
            ))
        return PW01PerformanceReport(4, normalized, tuple(points))
    finally:
        ctx.backend.close()


__all__ = [
    "PW01_PERFORMANCE_SCALES",
    "PW01PerformanceMeasurement",
    "PW01PerformanceReport",
    "PW01PerformanceScalePoint",
    "run_pw01_dual_memory_scale_curve",
]
