"""K-04 Memory 候选冷投影、范围规划和有界稳定 Top-K 协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    SourceRef,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateRecord,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryActivationRequest
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryAggregateFilter,
    MemoryCandidateBundle,
    MemorySourceTrace,
    ResolvedCandidate,
    SourceDiversityAssessment,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerStreamReader,
    pack_key,
    strict_integer_tuple,
)
from pure_integer_ai.storage.sealed_segment import SegmentRecord


MEMORY_CANDIDATE_PROJECTION_FORMAT_VERSION = 1
STABLE_TOP_K_POLICY_VERSION = 1


class MemoryHotSetIntegrityError(RuntimeError):
    """候选投影的完整身份、范围或 payload 发生漂移。"""


def visible_owner_keys(
        access: MemoryAccessContext,
        ) -> tuple[tuple[int, int, int, int], ...]:
    """返回当前 ACL 可见的精确 owner 分区，不生成通配范围。"""
    if not isinstance(access, MemoryAccessContext):
        raise TypeError("visible owner access 类型错误")
    owners = [OwnerScope(0, 0, 0, VISIBILITY_GLOBAL).stable_key()]
    if access.tenant_id > 0:
        owners.append(OwnerScope(
            access.tenant_id, 0, 0, VISIBILITY_TENANT).stable_key())
    if access.user_id > 0:
        owners.append(OwnerScope(
            access.tenant_id,
            access.user_id,
            0,
            VISIBILITY_USER,
        ).stable_key())
    if access.session_id > 0:
        owners.append(OwnerScope(
            access.tenant_id,
            access.user_id,
            access.session_id,
            VISIBILITY_SESSION,
        ).stable_key())
    return tuple(sorted(owners))


def memory_candidate_record_key(
        projection_key: tuple[int, ...],
        aggregate: MemoryHypothesisAggregateRecord,
        ) -> tuple[int, ...]:
    """形成完整投影、kind 索引、owner 和 Hypothesis hash 的冷记录键。"""
    projection = strict_integer_tuple(
        projection_key, label="memory candidate projection_key")
    if not isinstance(aggregate, MemoryHypothesisAggregateRecord):
        raise TypeError("memory candidate aggregate 类型错误")
    return (
        len(projection),
        *projection,
        aggregate.hypothesis_kind_hash,
        *aggregate.owner_key,
        aggregate.hypothesis_hash,
    )


def memory_candidate_scan_range(
        projection_key: tuple[int, ...],
        hypothesis_kind_hash: int,
        owner_key: tuple[int, int, int, int],
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """返回一个 kind/owner 分区的前缀闭区间，不依赖整数最大值哨兵。"""
    projection = strict_integer_tuple(
        projection_key, label="memory candidate scan projection_key")
    if type(hypothesis_kind_hash) is not int or hypothesis_kind_hash <= 0:
        raise ValueError("hypothesis_kind_hash 必须是正严格整数")
    owner = OwnerScope(*owner_key).stable_key()
    lower = (
        len(projection),
        *projection,
        hypothesis_kind_hash,
        *owner,
    )
    upper = (*lower[:-1], lower[-1] + 1)
    return lower, upper


def encode_memory_candidate(
        projection_key: tuple[int, ...],
        bundle: MemoryCandidateBundle,
        ) -> SegmentRecord:
    """把完整 typed 候选编码为 K-02 纯整数 canonical record。"""
    if not isinstance(bundle, MemoryCandidateBundle):
        raise TypeError("encode memory candidate bundle 类型错误")
    aggregate = bundle.aggregate
    payload: list[int] = [MEMORY_CANDIDATE_PROJECTION_FORMAT_VERSION]
    pack_key(payload, bundle.hypothesis_ref.stable_key())
    pack_key(payload, bundle.hypothesis.stable_key())
    payload.extend((
        aggregate.space_id,
        aggregate.hypothesis_hash,
        *aggregate.owner_key,
        aggregate.hypothesis_kind_hash,
        aggregate.competition_hash,
        aggregate.context_hash,
        aggregate.source_hash,
        aggregate.created_seq,
        aggregate.last_observed_seq,
        aggregate.last_supported_seq,
        aggregate.last_refuted_seq,
        aggregate.last_used_seq,
        aggregate.support_count,
        aggregate.contradict_count,
        aggregate.unknown_count,
        aggregate.independent_source_count,
        aggregate.support_source_count,
        aggregate.contradict_source_count,
        aggregate.use_count,
        aggregate.retention_state,
        aggregate.lifecycle_state,
        aggregate.evidence_state,
        len(bundle.source_traces),
    ))
    for trace in bundle.source_traces:
        pack_key(payload, trace.source.stable_key())
        payload.extend((
            trace.stance,
            trace.first_observed_seq,
            trace.last_observed_seq,
            trace.evidence_count,
        ))
    return SegmentRecord(
        memory_candidate_record_key(projection_key, aggregate),
        tuple(payload),
    )


def decode_memory_candidate(
        projection_key: tuple[int, ...],
        record: SegmentRecord,
        ) -> MemoryCandidateBundle:
    """从冷页恢复完整 typed 候选，并逐层核验 record key 与 payload。"""
    projection = strict_integer_tuple(
        projection_key, label="decode memory candidate projection_key")
    if not isinstance(record, SegmentRecord):
        raise TypeError("decode memory candidate record 类型错误")
    reader = IntegerStreamReader(record.payload)
    version = reader.read_positive(label="memory candidate format version")
    if version != MEMORY_CANDIDATE_PROJECTION_FORMAT_VERSION:
        raise MemoryHotSetIntegrityError("Memory candidate 投影版本未注册")
    hypothesis_ref = MemoryObjectRef.from_stable_key(
        reader.read_key(label="memory candidate hypothesis_ref"))
    hypothesis = HypothesisKey.from_stable_key(
        reader.read_key(label="memory candidate hypothesis"))
    values = tuple(
        reader.read_nonnegative(label=f"memory candidate aggregate[{index}]")
        for index in range(25)
    )
    aggregate = MemoryHypothesisAggregateRecord(
        values[0],
        values[1],
        tuple(values[2:6]),
        *values[6:25],
    )
    trace_count = reader.read_nonnegative(label="memory candidate trace_count")
    traces = []
    for index in range(trace_count):
        source = SourceRef.from_stable_key(reader.read_key(
            label=f"memory candidate trace[{index}].source"))
        traces.append(MemorySourceTrace(
            source,
            reader.read_positive(label=f"memory candidate trace[{index}].stance"),
            reader.read_nonnegative(
                label=f"memory candidate trace[{index}].first_observed_seq"),
            reader.read_nonnegative(
                label=f"memory candidate trace[{index}].last_observed_seq"),
            reader.read_positive(
                label=f"memory candidate trace[{index}].evidence_count"),
        ))
    reader.finish()
    normalized_traces = tuple(sorted(
        traces, key=lambda item: item.stable_key()))
    sources_by_key = {
        item.source.stable_key(): item.source for item in normalized_traces
    }
    bundle = MemoryCandidateBundle(
        hypothesis_ref,
        hypothesis,
        aggregate,
        tuple(sources_by_key[key] for key in sorted(sources_by_key)),
        normalized_traces,
    )
    expected = memory_candidate_record_key(projection, aggregate)
    if record.record_key != expected:
        raise MemoryHotSetIntegrityError("Memory candidate record key 与 payload 漂移")
    return bundle


def matches_memory_filter(
        bundle: MemoryCandidateBundle,
        branch: MemoryAggregateFilter,
        ) -> bool:
    """对冷页 typed bundle 执行与 M-07 OR-of-AND 分支相同的完整条件。"""
    if not isinstance(bundle, MemoryCandidateBundle):
        raise TypeError("memory filter bundle 类型错误")
    if not isinstance(branch, MemoryAggregateFilter):
        raise TypeError("memory filter branch 类型错误")
    aggregate = bundle.aggregate
    if branch.context is not None and bundle.hypothesis.scope.stable_key() != branch.context:
        return False
    if branch.source is not None and branch.source not in bundle.sources:
        return False
    for actual, expected in (
            (aggregate.evidence_state, branch.evidence_state),
            (aggregate.lifecycle_state, branch.lifecycle_state),
            (aggregate.retention_state, branch.retention_state)):
        if expected is not None and actual != expected:
            return False
    return True


def resolved_candidate_order_key(
        candidate: ResolvedCandidate,
        ) -> tuple[object, ...]:
    """返回与 M-07 全热排序完全相同的候选顺序键。"""
    if not isinstance(candidate, ResolvedCandidate):
        raise TypeError("resolved candidate order 类型错误")
    return (
        -candidate.score,
        candidate.competition_key,
        candidate.candidate_key,
        candidate.origin_kind,
    )


PinRecord = Callable[[tuple[int, ...]], None]
UnpinRecord = Callable[[tuple[int, ...]], None]


@runtime_checkable
class BoundedCandidateAccumulator(Protocol):
    """以显式有界状态接收候选并维护精确选择结果。"""

    def offer(
            self,
            candidate: ResolvedCandidate,
            record_key: tuple[int, ...] | None,
            ) -> None:
        """接收一个 Core 或 Memory 候选及其可选冷记录身份。"""
        ...

    def finish(self) -> tuple[ResolvedCandidate, ...]:
        """返回与同策略全热执行一致的最终有序候选。"""
        ...


@runtime_checkable
class BoundedCandidateSelectionPolicy(Protocol):
    """同时服务 M-07 全热基线和 K-04 流式热集的选择协议。"""

    def select(
            self,
            request: MemoryActivationRequest,
            candidates: tuple[ResolvedCandidate, ...],
            budget: int,
            ) -> tuple[ResolvedCandidate, ...]:
        """在全热基线中选择候选。"""
        ...

    def new_accumulator(
            self,
            request: MemoryActivationRequest,
            budget: int,
            *,
            pin: PinRecord,
            unpin: UnpinRecord,
            ) -> BoundedCandidateAccumulator:
        """创建状态上限只依赖 budget 的流式选择器。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回选择协议及其预算语义版本。"""
        ...


@dataclass
class _HeldCandidate:
    """流式 Top-K 内一个候选和已配对 pin 的冷记录身份。"""

    candidate: ResolvedCandidate
    record_key: tuple[int, ...] | None


class _StableTopKAccumulator:
    """只保留 budget 个最优候选，并同步 pin/unpin 被保留冷记录。"""

    def __init__(
            self,
            budget: int,
            *,
            pin: PinRecord,
            unpin: UnpinRecord,
            ) -> None:
        """绑定正预算和查询 cache 的配对持有回调。"""
        if type(budget) is not int or budget <= 0:
            raise ValueError("stable Top-K budget 必须是正严格整数")
        if not callable(pin) or not callable(unpin):
            raise TypeError("stable Top-K pin/unpin 必须可调用")
        self.budget = budget
        self.pin = pin
        self.unpin = unpin
        self._items: list[_HeldCandidate] = []

    def offer(
            self,
            candidate: ResolvedCandidate,
            record_key: tuple[int, ...] | None,
            ) -> None:
        """只在候选进入当前 Top-K 时持有记录，被替换时立即释放。"""
        if not isinstance(candidate, ResolvedCandidate):
            raise TypeError("stable Top-K candidate 类型错误")
        if record_key is not None:
            strict_integer_tuple(record_key, label="stable Top-K record_key")
        held = _HeldCandidate(candidate, record_key)
        if len(self._items) >= self.budget:
            worst = max(
                self._items,
                key=lambda item: resolved_candidate_order_key(item.candidate),
            )
            if (resolved_candidate_order_key(candidate)
                    >= resolved_candidate_order_key(worst.candidate)):
                return
        if record_key is not None:
            self.pin(record_key)
        self._items.append(held)
        self._items.sort(key=lambda item: resolved_candidate_order_key(
            item.candidate))
        if len(self._items) > self.budget:
            removed = self._items.pop()
            if removed.record_key is not None:
                self.unpin(removed.record_key)

    def finish(self) -> tuple[ResolvedCandidate, ...]:
        """按 M-07 稳定顺序返回当前全部保留候选。"""
        return tuple(item.candidate for item in self._items)


class StableTopKSourcePolicy:
    """不改分、只稳定取前 K 的可流式来源策略。"""

    def assess(self, request, hypothesis, aggregate, sources):
        """保留真实独立来源数，不增加任何领域偏好分。"""
        del request, hypothesis, aggregate
        return SourceDiversityAssessment(len(sources), 0, ())

    def select(self, request, candidates, budget):
        """全热基线直接选择已按 M-07 顺序排列的前 K 项。"""
        del request
        if type(budget) is not int or budget < 0:
            raise ValueError("stable Top-K select budget 必须是非负严格整数")
        return tuple(candidates[:budget])

    def new_accumulator(
            self,
            request: MemoryActivationRequest,
            budget: int,
            *,
            pin: PinRecord,
            unpin: UnpinRecord,
            ) -> BoundedCandidateAccumulator:
        """为一个 request 创建只保留 budget 项的精确流式选择器。"""
        if not isinstance(request, MemoryActivationRequest):
            raise TypeError("stable Top-K request 类型错误")
        return _StableTopKAccumulator(
            budget,
            pin=pin,
            unpin=unpin,
        )

    def state_key(self) -> tuple[int, ...]:
        """返回稳定 Top-K 选择协议版本。"""
        return (STABLE_TOP_K_POLICY_VERSION,)


__all__ = [
    "BoundedCandidateAccumulator",
    "BoundedCandidateSelectionPolicy",
    "MEMORY_CANDIDATE_PROJECTION_FORMAT_VERSION",
    "MemoryHotSetIntegrityError",
    "StableTopKSourcePolicy",
    "decode_memory_candidate",
    "encode_memory_candidate",
    "matches_memory_filter",
    "memory_candidate_record_key",
    "memory_candidate_scan_range",
    "resolved_candidate_order_key",
    "visible_owner_keys",
]
