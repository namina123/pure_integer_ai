"""M-07 Core/Memory 仲裁使用的纯协议、策略接口和 typed trace。

本模块不读取存储，也不执行排序。Core 基线、索引过滤、整数评分和来源多样性均由
调用方注入；只读身份恢复与 Top-K 执行位于 ``memory_resolver_engine``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import SourceRef, TypedRef
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateRecord,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryActivationRequest,
    MemoryQueryCompilation,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


RESOLUTION_ORIGIN_CORE = 1
RESOLUTION_ORIGIN_MEMORY = 2
MEMORY_RESOLVER_PROTOCOL_VERSION = 1


def _strict_tuple(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验 resolver 使用的开放整数键，禁止 bool、字符串和浮点混入。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if not value and not allow_empty:
        raise ValueError(f"{label} 不得为空")
    if value:
        assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放键增加长度边界，避免 resolver trace 拼接产生歧义。"""
    return len(value), *value


def _aggregate_key(
        aggregate: MemoryHypothesisAggregateRecord,
        ) -> tuple[int, ...]:
    """展开 M-04 派生快照，使冲突、时序和计数进入 resolver trace。"""
    return (
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
    )


@dataclass(frozen=True)
class ActivationScoreReason:
    """由注入评分器提供的一条可回溯整数评分理由。"""

    reason_key: tuple[int, ...]
    value: int

    def __post_init__(self) -> None:
        """校验理由身份和分值，确保 trace 不依赖浮点或文字。"""
        _strict_tuple(self.reason_key, label="ActivationScoreReason.reason_key")
        assert_int(self.value, _where="ActivationScoreReason.value")
        if type(self.value) is not int:
            raise ValueError("ActivationScoreReason.value 必须是严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回理由键和值的完整稳定表示。"""
        return (*_packed(self.reason_key), self.value)


@dataclass(frozen=True)
class ActivationScore:
    """一个 Core 或 Memory 候选的注入式整数基础评分及理由。"""

    value: int
    reasons: tuple[ActivationScoreReason, ...]

    def __post_init__(self) -> None:
        """校验评分和理由集合，避免无理由分数进入仲裁结果。"""
        assert_int(self.value, _where="ActivationScore.value")
        if type(self.value) is not int:
            raise ValueError("ActivationScore.value 必须是严格整数")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("ActivationScore.reasons 必须为非空 tuple")
        if any(not isinstance(item, ActivationScoreReason)
               for item in self.reasons):
            raise TypeError("ActivationScore.reasons 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回评分值和所有理由的稳定键。"""
        result = [self.value, len(self.reasons)]
        for reason in self.reasons:
            result.extend(_packed(reason.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SourceDiversityAssessment:
    """由调用方注入的来源多样性判断及整数调整，不改写原始来源统计。"""

    distinct_source_count: int
    adjustment: int
    reasons: tuple[ActivationScoreReason, ...]

    def __post_init__(self) -> None:
        """校验来源数量、调整值和多样性理由。"""
        assert_int(
            self.distinct_source_count,
            self.adjustment,
            _where="SourceDiversityAssessment",
        )
        if type(self.distinct_source_count) is not int:
            raise ValueError("distinct_source_count 必须是严格整数")
        if self.distinct_source_count < 0:
            raise ValueError("distinct_source_count 不得为负数")
        if type(self.adjustment) is not int:
            raise ValueError("adjustment 必须是严格整数")
        if not isinstance(self.reasons, tuple) or any(
                not isinstance(item, ActivationScoreReason)
                for item in self.reasons):
            raise TypeError("SourceDiversityAssessment.reasons 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源数量、调整值和多样性理由的稳定键。"""
        result = [self.distinct_source_count, self.adjustment, len(self.reasons)]
        for reason in self.reasons:
            result.extend(_packed(reason.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemorySourceTrace:
    """一个 Memory 候选的完整来源、立场和活动证据统计。"""

    source: SourceRef
    stance: int
    first_observed_seq: int
    last_observed_seq: int
    evidence_count: int

    def __post_init__(self) -> None:
        """校验来源分账的对象身份、立场和逻辑序计数。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("MemorySourceTrace.source 必须是 SourceRef")
        assert_int(
            self.stance,
            self.first_observed_seq,
            self.last_observed_seq,
            self.evidence_count,
            _where="MemorySourceTrace",
        )
        if any(type(value) is not int for value in (
                self.stance,
                self.first_observed_seq,
                self.last_observed_seq,
                self.evidence_count)):
            raise ValueError("MemorySourceTrace 必须使用严格整数")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("MemorySourceTrace.stance 未注册")
        if self.first_observed_seq < 0:
            raise ValueError("MemorySourceTrace.first_observed_seq 不得为负")
        if self.last_observed_seq < self.first_observed_seq:
            raise ValueError("MemorySourceTrace 观察逻辑序倒退")
        if self.evidence_count <= 0:
            raise ValueError("MemorySourceTrace.evidence_count 必须为正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、立场、逻辑序和证据数的完整稳定键。"""
        return (
            *_packed(self.source.stable_key()),
            self.stance,
            self.first_observed_seq,
            self.last_observed_seq,
            self.evidence_count,
        )


def _aggregate_stable_key(
        aggregate: MemoryHypothesisAggregateRecord,
        ) -> tuple[int, ...]:
    """把固定字段 aggregate 投影为无歧义纯整数状态键。"""
    if not isinstance(aggregate, MemoryHypothesisAggregateRecord):
        raise TypeError("aggregate 类型错误")
    return (
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
    )


@dataclass(frozen=True)
class MemoryCandidateBundle:
    """可由全热索引或 K-04 冷页恢复的完整 Memory 候选。"""

    hypothesis_ref: MemoryObjectRef
    hypothesis: HypothesisKey
    aggregate: MemoryHypothesisAggregateRecord
    sources: tuple[SourceRef, ...]
    source_traces: tuple[MemorySourceTrace, ...]

    def __post_init__(self) -> None:
        """核验引用、声明、aggregate 和来源分账形成同一完整候选。"""
        if not isinstance(self.hypothesis_ref, MemoryObjectRef):
            raise TypeError("MemoryCandidateBundle.hypothesis_ref 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("MemoryCandidateBundle.hypothesis 类型错误")
        if not isinstance(self.aggregate, MemoryHypothesisAggregateRecord):
            raise TypeError("MemoryCandidateBundle.aggregate 类型错误")
        if self.hypothesis_ref.object_key != self.hypothesis.stable_key():
            raise ValueError("Memory candidate 引用与 Hypothesis 声明漂移")
        if (self.hypothesis_ref.owner != self.hypothesis.observation.owner
                or self.hypothesis_ref.versions
                != self.hypothesis.observation.versions):
            raise ValueError("Memory candidate 引用 owner/version 与声明漂移")
        if self.aggregate.owner_key != self.hypothesis_ref.owner.stable_key():
            raise ValueError("Memory candidate aggregate owner 漂移")
        if (not isinstance(self.sources, tuple)
                or any(not isinstance(item, SourceRef) for item in self.sources)):
            raise TypeError("MemoryCandidateBundle.sources 类型错误")
        source_keys = tuple(item.stable_key() for item in self.sources)
        if source_keys != tuple(sorted(set(source_keys))):
            raise ValueError("Memory candidate sources 必须唯一稳定排序")
        if (not isinstance(self.source_traces, tuple)
                or any(not isinstance(item, MemorySourceTrace)
                       for item in self.source_traces)):
            raise TypeError("MemoryCandidateBundle.source_traces 类型错误")
        trace_keys = tuple(item.stable_key() for item in self.source_traces)
        if trace_keys != tuple(sorted(set(trace_keys))):
            raise ValueError("Memory candidate source traces 必须唯一稳定排序")
        traced_sources = tuple(sorted({
            item.source.stable_key() for item in self.source_traces
        }))
        if traced_sources != source_keys:
            raise ValueError("Memory candidate 来源集合与分账不一致")
        if self.aggregate.independent_source_count != len(self.sources):
            raise ValueError("Memory candidate 独立来源数与 aggregate 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整引用、声明、aggregate 和来源分账状态键。"""
        result = [
            MEMORY_RESOLVER_PROTOCOL_VERSION,
            *_packed(self.hypothesis_ref.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *_packed(_aggregate_stable_key(self.aggregate)),
            len(self.sources),
        ]
        for source in self.sources:
            result.extend(_packed(source.stable_key()))
        result.append(len(self.source_traces))
        for trace in self.source_traces:
            result.extend(_packed(trace.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class CoreBaselineCandidate:
    """Core 基线提供方返回的完整 typed 候选，不复制 Core 身份。"""

    core_ref: TypedRef
    competition_key: tuple[int, ...]
    score: ActivationScore
    sources: tuple[SourceRef, ...] = ()

    def __post_init__(self) -> None:
        """校验 Core 引用、竞争组、评分和来源均为完整对象。"""
        if not isinstance(self.core_ref, TypedRef):
            raise TypeError("CoreBaselineCandidate.core_ref 必须是 TypedRef")
        _strict_tuple(
            self.competition_key,
            label="CoreBaselineCandidate.competition_key",
        )
        if not isinstance(self.score, ActivationScore):
            raise TypeError("CoreBaselineCandidate.score 类型错误")
        _validate_sources(self.sources, label="CoreBaselineCandidate.sources")
        object.__setattr__(self, "sources", tuple(sorted(
            self.sources,
            key=lambda item: item.stable_key(),
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回 Core 候选的完整引用和注入字段稳定键。"""
        result = [
            RESOLUTION_ORIGIN_CORE,
            *_packed(self.core_ref.stable_key()),
            *_packed(self.competition_key),
            *_packed(self.score.stable_key()),
            len(self.sources),
        ]
        for source in self.sources:
            result.extend(_packed(source.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryAggregateFilter:
    """一个注入式 M-04 索引过滤分支；多个分支按 OR 合并并去重。"""

    context: tuple[int, ...] | None = None
    source: SourceRef | None = None
    evidence_state: int | None = None
    lifecycle_state: int | None = None
    retention_state: int | None = None

    def __post_init__(self) -> None:
        """校验可选完整 context/source 和三个正整数派生状态。"""
        if self.context is not None:
            _strict_tuple(
                self.context,
                label="MemoryAggregateFilter.context",
            )
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("MemoryAggregateFilter.source 必须是 SourceRef")
        for name, value in (
                ("evidence_state", self.evidence_state),
                ("lifecycle_state", self.lifecycle_state),
                ("retention_state", self.retention_state)):
            if value is None:
                continue
            assert_int(value, _where=f"MemoryAggregateFilter.{name}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"MemoryAggregateFilter.{name} 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部可选索引条件的无歧义稳定键。"""
        return (
            *_packed(() if self.context is None else self.context),
            *_packed(() if self.source is None else self.source.stable_key()),
            0 if self.evidence_state is None else self.evidence_state,
            0 if self.lifecycle_state is None else self.lifecycle_state,
            0 if self.retention_state is None else self.retention_state,
        )


class CoreBaselineProvider(Protocol):
    """按当前 request 提供只读 Core 基线候选。"""

    def candidates(
            self,
            request: MemoryActivationRequest,
            ) -> tuple[CoreBaselineCandidate, ...]:
        """返回当前 query 可见的 Core 基线候选，不得写入任何存储。"""


class MemoryIndexFilterProvider(Protocol):
    """把当前 request 编译为一个或多个 M-04 索引过滤分支。"""

    def filters(
            self,
            request: MemoryActivationRequest,
            ) -> tuple[MemoryAggregateFilter, ...]:
        """返回非空 OR 分支；每个分支内部条件按 AND 查询。"""


class MemoryScoreProvider(Protocol):
    """按完整 Memory 候选计算注入式整数评分。"""

    def score(
            self,
            request: MemoryActivationRequest,
            hypothesis: HypothesisKey,
            aggregate: MemoryHypothesisAggregateRecord,
            sources: tuple[SourceRef, ...],
            ) -> ActivationScore:
        """返回 Memory 候选基础分和可回溯理由，不修改 aggregate。"""


class SourceDiversityPolicy(Protocol):
    """按当前 request 和完整来源集合计算来源多样性调整。"""

    def assess(
            self,
            request: MemoryActivationRequest,
            hypothesis: HypothesisKey,
            aggregate: MemoryHypothesisAggregateRecord,
            sources: tuple[SourceRef, ...],
        ) -> SourceDiversityAssessment:
        """返回来源调整和理由，不把同源重复计为独立来源。"""

    def select(
            self,
            request: MemoryActivationRequest,
            candidates: tuple[ResolvedCandidate, ...],
            budget: int,
            ) -> tuple[ResolvedCandidate, ...]:
        """从基础分有序候选中选择恰好 Top-K，并执行跨候选来源多样化。"""


def _validate_sources(
        sources: tuple[SourceRef, ...], *, label: str,
        ) -> tuple[SourceRef, ...]:
    """校验来源集合为去重、完整和稳定排序前的 SourceRef tuple。"""
    if not isinstance(sources, tuple):
        raise TypeError(f"{label} 必须是 SourceRef tuple")
    if any(not isinstance(item, SourceRef) for item in sources):
        raise TypeError(f"{label} 包含非 SourceRef")
    keys = tuple(item.stable_key() for item in sources)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} 不得重复同一 SourceRef")
    return sources


@dataclass(frozen=True)
class ResolvedCandidate:
    """一个保留 Core/Memory 身份、scope、来源和评分理由的 typed 结果。"""

    origin_kind: int
    candidate_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    query_scope: ScopeIdentity
    candidate_scope: ScopeIdentity | None
    core_ref: TypedRef | None
    memory_ref: MemoryObjectRef | None
    hypothesis: HypothesisKey | None
    aggregate: MemoryHypothesisAggregateRecord | None
    sources: tuple[SourceRef, ...]
    memory_source_traces: tuple[MemorySourceTrace, ...]
    score: int
    score_reasons: tuple[ActivationScoreReason, ...]
    diversity: SourceDiversityAssessment | None

    def __post_init__(self) -> None:
        """核验候选来源分支互斥，禁止用 hash 或半成品对象冒充身份。"""
        if self.origin_kind not in {
                RESOLUTION_ORIGIN_CORE, RESOLUTION_ORIGIN_MEMORY}:
            raise ValueError("ResolvedCandidate.origin_kind 未注册")
        _strict_tuple(self.candidate_key, label="ResolvedCandidate.candidate_key")
        _strict_tuple(
            self.competition_key,
            label="ResolvedCandidate.competition_key",
        )
        if not isinstance(self.query_scope, ScopeIdentity):
            raise TypeError("ResolvedCandidate.query_scope 必须是 ScopeIdentity")
        if self.candidate_scope is not None and not isinstance(
                self.candidate_scope, ScopeIdentity):
            raise TypeError("ResolvedCandidate.candidate_scope 类型错误")
        assert_int(self.score, _where="ResolvedCandidate.score")
        if type(self.score) is not int:
            raise ValueError("ResolvedCandidate.score 必须是严格整数")
        if not isinstance(self.score_reasons, tuple) or not self.score_reasons:
            raise ValueError("ResolvedCandidate.score_reasons 不得为空")
        if any(not isinstance(item, ActivationScoreReason)
               for item in self.score_reasons):
            raise TypeError("ResolvedCandidate.score_reasons 类型错误")
        _validate_sources(self.sources, label="ResolvedCandidate.sources")
        canonical_sources = tuple(sorted(
            self.sources,
            key=lambda item: item.stable_key(),
        ))
        if self.sources != canonical_sources:
            raise ValueError("ResolvedCandidate.sources 必须按完整身份稳定排序")
        if not isinstance(self.memory_source_traces, tuple) or any(
                not isinstance(item, MemorySourceTrace)
                for item in self.memory_source_traces):
            raise TypeError("ResolvedCandidate.memory_source_traces 类型错误")
        trace_keys = tuple(
            item.stable_key() for item in self.memory_source_traces)
        if trace_keys != tuple(sorted(trace_keys)):
            raise ValueError("Memory 来源分账必须按完整稳定键排序")
        if len(set(trace_keys)) != len(trace_keys):
            raise ValueError("Memory 来源分账不得重复")
        if self.origin_kind == RESOLUTION_ORIGIN_CORE:
            if not isinstance(self.core_ref, TypedRef):
                raise TypeError("Core 候选必须保留 TypedRef")
            if any(item is not None for item in (
                    self.memory_ref, self.hypothesis, self.aggregate,
                    self.diversity)):
                raise ValueError("Core 候选不得携带 Memory 派生字段")
            if self.memory_source_traces:
                raise ValueError("Core 候选不得携带 Memory 来源分账")
            if self.candidate_key != self.core_ref.stable_key():
                raise ValueError("Core 候选 candidate_key 与 TypedRef 不一致")
        else:
            if not isinstance(self.memory_ref, MemoryObjectRef):
                raise TypeError("Memory 候选必须保留 MemoryObjectRef")
            if not isinstance(self.hypothesis, HypothesisKey):
                raise TypeError("Memory 候选必须保留 HypothesisKey")
            if not isinstance(
                    self.aggregate, MemoryHypothesisAggregateRecord):
                raise TypeError("Memory 候选必须保留 aggregate")
            if not isinstance(self.diversity, SourceDiversityAssessment):
                raise TypeError("Memory 候选必须保留来源多样性评估")
            if self.candidate_key != self.memory_ref.stable_key():
                raise ValueError("Memory 候选 candidate_key 与完整引用不一致")
            trace_sources = {
                item.source.stable_key() for item in self.memory_source_traces}
            if trace_sources != {
                    item.stable_key() for item in self.sources}:
                raise ValueError("Memory 来源集合与来源分账不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回结果中全部权威身份和派生 trace 的确定性键。"""
        result = [
            MEMORY_RESOLVER_PROTOCOL_VERSION,
            self.origin_kind,
            *_packed(self.candidate_key),
            *_packed(self.competition_key),
            *_packed(self.query_scope.stable_key()),
            *_packed(() if self.candidate_scope is None
                     else self.candidate_scope.stable_key()),
            len(self.sources),
        ]
        for source in self.sources:
            result.extend(_packed(source.stable_key()))
        result.append(len(self.memory_source_traces))
        for source_trace in self.memory_source_traces:
            result.extend(_packed(source_trace.stable_key()))
        result.extend(_packed(
            () if self.aggregate is None else _aggregate_key(self.aggregate)))
        result.extend((self.score, len(self.score_reasons)))
        for reason in self.score_reasons:
            result.extend(_packed(reason.stable_key()))
        if self.diversity is None:
            result.extend((0,))
        else:
            result.extend((1, *_packed(self.diversity.stable_key())))
        return tuple(result)


@dataclass(frozen=True)
class ResolvedCandidateSet:
    """一个 activation request 的确定性 Top-K 结果，按 request 独立限额。"""

    request: MemoryActivationRequest
    candidates: tuple[ResolvedCandidate, ...]
    considered_count: int

    def __post_init__(self) -> None:
        """校验结果只包含当前 request 的候选且没有重复身份。"""
        if not isinstance(self.request, MemoryActivationRequest):
            raise TypeError("ResolvedCandidateSet.request 类型错误")
        if not isinstance(self.candidates, tuple) or any(
                not isinstance(item, ResolvedCandidate)
                for item in self.candidates):
            raise TypeError("ResolvedCandidateSet.candidates 类型错误")
        assert_int(
            self.considered_count,
            _where="ResolvedCandidateSet.considered_count",
        )
        if type(self.considered_count) is not int or self.considered_count < 0:
            raise ValueError("considered_count 必须是非负严格整数")
        keys = tuple(
            (item.origin_kind, item.candidate_key)
            for item in self.candidates
        )
        if len(set(keys)) != len(keys):
            raise ValueError("ResolvedCandidateSet 不得重复候选身份")
        if len(self.candidates) > self.request.budget:
            raise ValueError("ResolvedCandidateSet 超出 request budget")
        for candidate in self.candidates:
            if candidate.query_scope != self.request.scope:
                raise ValueError("ResolvedCandidateSet 含其他 query scope")

    def stable_key(self) -> tuple[int, ...]:
        """返回 request、考虑数和 Top-K 候选的完整稳定键。"""
        result = [
            *_packed(self.request.stable_key()),
            self.considered_count,
            len(self.candidates),
        ]
        for candidate in self.candidates:
            result.extend(_packed(candidate.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryResolution:
    """一次 compilation 的分 request 仲裁结果，供后续 AttractorState 消费。"""

    compilation: MemoryQueryCompilation
    sets: tuple[ResolvedCandidateSet, ...]

    def __post_init__(self) -> None:
        """核验结果覆盖 compilation 的每个 request 且顺序保持一致。"""
        if not isinstance(self.compilation, MemoryQueryCompilation):
            raise TypeError("MemoryResolution.compilation 类型错误")
        if not isinstance(self.sets, tuple) or any(
                not isinstance(item, ResolvedCandidateSet)
                for item in self.sets):
            raise TypeError("MemoryResolution.sets 类型错误")
        requests = tuple(item.request for item in self.sets)
        if requests != self.compilation.requests:
            raise ValueError("MemoryResolution 未逐项覆盖 compilation requests")

    def stable_key(self) -> tuple[int, ...]:
        """返回 compilation 和所有 request 结果的稳定键。"""
        result = [
            MEMORY_RESOLVER_PROTOCOL_VERSION,
            *_packed(self.compilation.stable_key()),
            len(self.sets),
        ]
        for item in self.sets:
            result.extend(_packed(item.stable_key()))
        return tuple(result)



__all__ = [
    "ActivationScore",
    "ActivationScoreReason",
    "CoreBaselineCandidate",
    "CoreBaselineProvider",
    "MEMORY_RESOLVER_PROTOCOL_VERSION",
    "MemoryAggregateFilter",
    "MemoryCandidateBundle",
    "MemoryIndexFilterProvider",
    "MemoryResolution",
    "MemorySourceTrace",
    "MemoryScoreProvider",
    "RESOLUTION_ORIGIN_CORE",
    "RESOLUTION_ORIGIN_MEMORY",
    "ResolvedCandidate",
    "ResolvedCandidateSet",
    "SourceDiversityAssessment",
    "SourceDiversityPolicy",
]
