"""M-07 Core/Memory resolver 的只读索引执行和确定性仲裁引擎。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
    MemoryHypothesisAggregateRecord,
    MemoryHypothesisSourceRecord,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_HYPOTHESIS,
    HypothesisPayload,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    CoreIdentityCatalog,
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryActivationRequest,
    MemoryQueryCompilation,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    MEMORY_RESOLVER_PROTOCOL_VERSION,
    ActivationScore,
    CoreBaselineCandidate,
    CoreBaselineProvider,
    MemoryAggregateFilter,
    MemoryCandidateBundle,
    MemoryIndexFilterProvider,
    MemoryResolution,
    MemoryScoreProvider,
    MemorySourceTrace,
    RESOLUTION_ORIGIN_CORE,
    RESOLUTION_ORIGIN_MEMORY,
    ResolvedCandidate,
    ResolvedCandidateSet,
    SourceDiversityAssessment,
    SourceDiversityPolicy,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给组件和空间状态键增加长度边界，避免拼接歧义。"""
    return len(value), *value


def _component_state_key(component: object) -> tuple[int, ...]:
    """读取注入组件的非空版本状态键，禁止未版本化评分策略。"""
    method = getattr(component, "state_key", None)
    if not callable(method):
        raise TypeError("resolver 注入组件必须提供 state_key")
    value = method()
    if not isinstance(value, tuple) or not value:
        raise ValueError("resolver component state_key 必须是非空整数 tuple")
    assert_int(*value, _where="resolver component state_key")
    if any(type(item) is not int for item in value):
        raise ValueError("resolver component state_key 必须使用严格整数")
    return value


class MemoryOverlayResolver:
    """只读合并 Core 基线和 Memory Hypothesis 的确定性 resolver。"""

    def __init__(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            core_identities: CoreIdentityCatalog,
            baseline_provider: CoreBaselineProvider,
            index_filter_provider: MemoryIndexFilterProvider,
            score_provider: MemoryScoreProvider,
            diversity_policy: SourceDiversityPolicy,
            ) -> None:
        """绑定存储读取边界和四个注入策略，不取得任何可写 Memory 入口。"""
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("aggregates 类型错误")
        if not isinstance(core_identities, CoreIdentityCatalog):
            raise TypeError("core_identities 类型错误")
        for name, provider, methods in (
                ("baseline_provider", baseline_provider, ("candidates",)),
                ("index_filter_provider", index_filter_provider, ("filters",)),
                ("score_provider", score_provider, ("score",)),
                ("diversity_policy", diversity_policy, ("assess", "select"))):
            for method in methods:
                if not callable(getattr(provider, method, None)):
                    raise TypeError(f"{name} 缺少 {method} 注入协议方法")
            if not callable(getattr(provider, "state_key", None)):
                raise TypeError(f"{name} 缺少 state_key 版本协议")
            _component_state_key(provider)
        if aggregates.event_log.backend is not core_identities.backend:
            raise ValueError("resolver 的 aggregate 与 Core catalog 不得跨 backend")
        self.aggregates = aggregates
        self.core_identities = core_identities
        self.baseline_provider = baseline_provider
        self.index_filter_provider = index_filter_provider
        self.score_provider = score_provider
        self.diversity_policy = diversity_policy

    def resolve(self, compilation: MemoryQueryCompilation) -> MemoryResolution:
        """按每个分型 request 独立检索、评分、去重和 Top-K，不写任何状态。"""
        if not isinstance(compilation, MemoryQueryCompilation):
            raise TypeError("compilation 必须是 MemoryQueryCompilation")
        if compilation.memory_space != self.aggregates.event_log.memory_space_identity:
            raise ValueError("compilation Memory 空间与 resolver 不一致")
        self.aggregates.require_clean(access=compilation.access)
        sets = tuple(self._resolve_request(request)
                     for request in compilation.requests)
        self.aggregates.require_clean(access=compilation.access)
        return MemoryResolution(compilation, sets)

    def clone_for_aggregates(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            core_identities: CoreIdentityCatalog,
            *,
            baseline_provider: CoreBaselineProvider | None = None,
            index_filter_provider: MemoryIndexFilterProvider | None = None,
            score_provider: MemoryScoreProvider | None = None,
            diversity_policy: SourceDiversityPolicy | None = None,
            ) -> "MemoryOverlayResolver":
        """为 V-06 重绑独立 aggregate/Core facade，不共享宿主存储边界。"""
        return MemoryOverlayResolver(
            aggregates,
            core_identities,
            self.baseline_provider if baseline_provider is None
            else baseline_provider,
            self.index_filter_provider if index_filter_provider is None
            else index_filter_provider,
            self.score_provider if score_provider is None else score_provider,
            self.diversity_policy if diversity_policy is None
            else diversity_policy,
        )

    def state_key(self) -> tuple[int, ...]:
        """返回 resolver 协议和注入组件状态，供隔离检查读取。"""
        return (
            MEMORY_RESOLVER_PROTOCOL_VERSION,
            *_packed(
                self.aggregates.event_log.memory_space_identity.stable_key()),
            *_packed(_component_state_key(self.baseline_provider)),
            *_packed(_component_state_key(self.index_filter_provider)),
            *_packed(_component_state_key(self.score_provider)),
            *_packed(_component_state_key(self.diversity_policy)),
        )

    def _resolve_request(
            self,
            request: MemoryActivationRequest,
            ) -> ResolvedCandidateSet:
        """执行单个 request 的 kind 预过滤、候选恢复和确定性 Top-K。"""
        if request.memory_space != self.aggregates.event_log.memory_space_identity:
            raise ValueError("activation request Memory 空间与 resolver 不一致")
        aggregate_rows = self._prefilter_aggregates(request)
        candidates: dict[tuple[int, tuple[int, ...]], ResolvedCandidate] = {}
        for candidate in self.core_candidates(request):
            self._insert_candidate(candidates, candidate)
        for aggregate in aggregate_rows:
            candidate = self._memory_candidate(request, aggregate)
            self._insert_candidate(candidates, candidate)
        ordered = tuple(sorted(
            candidates.values(),
            key=lambda item: (
                -item.score,
                item.competition_key,
                item.candidate_key,
                item.origin_kind,
            ),
        ))
        selected = self.diversity_policy.select(
            request,
            ordered,
            min(request.budget, len(ordered)),
        )
        self._validate_selection(ordered, selected, request.budget)
        return ResolvedCandidateSet(
            request,
            selected,
            len(ordered),
        )

    def _prefilter_aggregates(
            self,
            request: MemoryActivationRequest,
            ) -> tuple[MemoryHypothesisAggregateRecord, ...]:
        """执行注入式 OR-of-AND 索引查询，并按完整 aggregate 身份去重。"""
        filters = self.index_filter_provider.filters(request)
        if not isinstance(filters, tuple) or not filters:
            raise ValueError("Memory index filter provider 必须返回非空 tuple")
        if any(not isinstance(item, MemoryAggregateFilter) for item in filters):
            raise TypeError("Memory index filter provider 返回了错误过滤分支")
        filter_keys = tuple(item.stable_key() for item in filters)
        if len(set(filter_keys)) != len(filter_keys):
            raise ValueError("Memory index filter provider 不得返回重复分支")
        records: dict[int, MemoryHypothesisAggregateRecord] = {}
        for item in filters:
            rows = self.aggregates.query(
                access=request.access,
                hypothesis_kind=request.hypothesis_kind,
                context=item.context,
                evidence_state=item.evidence_state,
                lifecycle_state=item.lifecycle_state,
                retention_state=item.retention_state,
                source=item.source,
            )
            for record in rows:
                previous = records.get(record.hypothesis_hash)
                if previous is not None and previous != record:
                    raise RuntimeError("索引过滤分支命中漂移 aggregate")
                records[record.hypothesis_hash] = record
        return tuple(records[key] for key in sorted(records))

    def core_candidates(
            self,
            request: MemoryActivationRequest,
            ) -> tuple[ResolvedCandidate, ...]:
        """恢复并核验当前 request 的全部只读 Core 基线候选。"""
        if not isinstance(request, MemoryActivationRequest):
            raise TypeError("core candidates request 类型错误")
        baselines = self.baseline_provider.candidates(request)
        if not isinstance(baselines, tuple):
            raise TypeError("Core baseline provider 必须返回 tuple")
        result: dict[tuple[int, tuple[int, ...]], ResolvedCandidate] = {}
        for baseline in baselines:
            self._insert_candidate(
                result, self._core_candidate(request, baseline))
        return tuple(result[key] for key in sorted(result))

    def _core_candidate(
            self,
            request: MemoryActivationRequest,
            baseline: CoreBaselineCandidate,
            ) -> ResolvedCandidate:
        """核验并包装一个 Core 基线，不从它复制或创建概念身份。"""
        if not isinstance(baseline, CoreBaselineCandidate):
            raise TypeError("Core baseline provider 返回了错误候选")
        identity = self.core_identities.identity_of(baseline.core_ref)
        if not request.access.can_read(identity.owner):
            raise PermissionError("Core baseline 对象超出当前 ACL")
        for source in baseline.sources:
            if not request.access.can_read(source.owner):
                raise PermissionError("Core baseline 来源超出当前 ACL")
        return ResolvedCandidate(
            RESOLUTION_ORIGIN_CORE,
            baseline.core_ref.stable_key(),
            baseline.competition_key,
            request.scope,
            None,
            baseline.core_ref,
            None,
            None,
            None,
            baseline.sources,
            (),
            baseline.score.value,
            baseline.score.reasons,
            None,
        )

    @staticmethod
    def _validate_selection(
            candidates: tuple[ResolvedCandidate, ...],
            selected: tuple[ResolvedCandidate, ...],
            budget: int,
            ) -> None:
        """阻止多样性策略伪造、重复、改写或少选 Top-K 候选。"""
        if not isinstance(selected, tuple) or any(
                not isinstance(item, ResolvedCandidate)
                for item in selected):
            raise TypeError("来源多样性 select 必须返回 ResolvedCandidate tuple")
        expected_count = min(budget, len(candidates))
        if len(selected) != expected_count:
            raise ValueError("来源多样性 select 必须恰好返回 Top-K")
        available = {
            (item.origin_kind, item.candidate_key): item
            for item in candidates
        }
        selected_keys = tuple(
            (item.origin_kind, item.candidate_key)
            for item in selected
        )
        if len(set(selected_keys)) != len(selected_keys):
            raise ValueError("来源多样性 select 不得重复候选")
        for key, item in zip(selected_keys, selected):
            if available.get(key) != item:
                raise ValueError("来源多样性 select 返回了伪造或改写候选")

    def _memory_candidate(
            self,
            request: MemoryActivationRequest,
            aggregate: MemoryHypothesisAggregateRecord,
            ) -> ResolvedCandidate:
        """恢复完整 Memory Hypothesis、来源和冲突派生，并应用注入评分。"""
        bundle = self.load_bundle(aggregate, access=request.access)
        return self.candidate_from_bundle(request, bundle)

    def load_bundle(
            self,
            aggregate: MemoryHypothesisAggregateRecord,
            *,
            access: MemoryAccessContext,
            ) -> MemoryCandidateBundle:
        """从全热 M-04/event 索引恢复一个可独立分页保存的完整候选。"""
        if not isinstance(aggregate, MemoryHypothesisAggregateRecord):
            raise TypeError("load bundle aggregate 类型错误")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("load bundle access 类型错误")
        hypothesis_ref = self.aggregates.hypothesis_ref_for_aggregate(
            aggregate,
            access=access,
        )
        if hypothesis_ref is None:
            raise RuntimeError("可见 aggregate 无法恢复完整 Hypothesis 引用")
        hypothesis = self._read_hypothesis(hypothesis_ref, access)
        sources, source_traces = self._read_sources(
            hypothesis_ref, aggregate, access)
        return MemoryCandidateBundle(
            hypothesis_ref,
            hypothesis,
            aggregate,
            sources,
            source_traces,
        )

    def candidate_from_bundle(
            self,
            request: MemoryActivationRequest,
            bundle: MemoryCandidateBundle,
            ) -> ResolvedCandidate:
        """对已恢复 typed bundle 复用 M-07 评分、多样性和输出构造。"""
        if not isinstance(request, MemoryActivationRequest):
            raise TypeError("candidate bundle request 类型错误")
        if not isinstance(bundle, MemoryCandidateBundle):
            raise TypeError("candidate bundle 类型错误")
        hypothesis_ref = bundle.hypothesis_ref
        hypothesis = bundle.hypothesis
        aggregate = bundle.aggregate
        sources = bundle.sources
        source_traces = bundle.source_traces
        if hypothesis_ref.memory_space != request.memory_space:
            raise ValueError("Memory candidate bundle 属于其他空间")
        if hypothesis.hypothesis_kind != request.hypothesis_kind:
            raise ValueError("Memory candidate bundle Hypothesis kind 漂移")
        if not request.access.can_read(hypothesis_ref.owner):
            raise PermissionError("Memory candidate bundle owner 超出当前 ACL")
        if any(not request.access.can_read(source.owner) for source in sources):
            raise PermissionError("Memory candidate bundle 来源超出当前 ACL")
        score = self.score_provider.score(
            request, hypothesis, aggregate, sources)
        diversity = self.diversity_policy.assess(
            request, hypothesis, aggregate, sources)
        if not isinstance(score, ActivationScore):
            raise TypeError("Memory score provider 返回了错误评分")
        if not isinstance(diversity, SourceDiversityAssessment):
            raise TypeError("来源多样性策略返回了错误评估")
        expected_source_count = len(sources)
        if aggregate.independent_source_count != expected_source_count:
            raise RuntimeError(
                "aggregate independent_source_count 与完整来源索引不一致")
        if diversity.distinct_source_count != expected_source_count:
            raise ValueError(
                "来源多样性策略的 distinct_source_count 与完整来源不一致")
        return ResolvedCandidate(
            RESOLUTION_ORIGIN_MEMORY,
            hypothesis_ref.stable_key(),
            hypothesis.competition_key,
            request.scope,
            hypothesis.scope,
            None,
            hypothesis_ref,
            hypothesis,
            aggregate,
            sources,
            source_traces,
            score.value + diversity.adjustment,
            score.reasons + diversity.reasons,
            diversity,
        )

    def _read_hypothesis(
            self,
            hypothesis_ref: MemoryObjectRef,
            access: MemoryAccessContext,
            ) -> HypothesisKey:
        """通过事件日志公开查询恢复声明，拒绝 hash-only 或 payload 漂移。"""
        entries = self.aggregates.event_log.query(
            access=access,
            event_kind=MEMORY_EVENT_HYPOTHESIS,
            object_ref=hypothesis_ref,
        )
        if len(entries) != 1:
            raise RuntimeError("Memory Hypothesis 没有唯一可见声明")
        event = entries[0].event
        if not isinstance(event.payload, HypothesisPayload):
            raise RuntimeError("Memory Hypothesis 声明 payload 类型漂移")
        hypothesis = HypothesisKey.from_stable_key(hypothesis_ref.object_key)
        if event.object_ref != hypothesis_ref or event.payload.hypothesis != hypothesis:
            raise RuntimeError("Memory Hypothesis 完整身份与声明不一致")
        return hypothesis

    def _read_sources(
            self,
            hypothesis_ref: MemoryObjectRef,
            aggregate: MemoryHypothesisAggregateRecord,
            access: MemoryAccessContext,
            ) -> tuple[tuple[SourceRef, ...], tuple[MemorySourceTrace, ...]]:
        """恢复来源索引的完整 SourceRef，并核验 hash、ACL 和去重。"""
        source_records = self.aggregates.sources(
            hypothesis_ref,
            access=access,
        )
        by_key: dict[tuple[int, ...], SourceRef] = {}
        traces: list[MemorySourceTrace] = []
        for record in source_records:
            source = self._source_from_record(record, aggregate)
            if not access.can_read(source.owner):
                raise PermissionError("Memory 候选来源超出当前 ACL")
            by_key[source.stable_key()] = source
            traces.append(MemorySourceTrace(
                source,
                record.stance,
                record.first_observed_seq,
                record.last_observed_seq,
                record.evidence_count,
            ))
        return (
            tuple(by_key[key] for key in sorted(by_key)),
            tuple(sorted(traces, key=lambda item: item.stable_key())),
        )

    def _source_from_record(
            self,
            record: MemoryHypothesisSourceRecord,
            aggregate: MemoryHypothesisAggregateRecord,
            ) -> SourceRef:
        """从来源索引行恢复 SourceRef，并阻断 source hash 或 owner 漂移。"""
        if not isinstance(record, MemoryHypothesisSourceRecord):
            raise TypeError("source index 返回了错误记录")
        if (record.space_id != aggregate.space_id
                or record.hypothesis_hash != aggregate.hypothesis_hash):
            raise RuntimeError("source index 指向其他 Hypothesis")
        if record.owner_key != aggregate.owner_key:
            raise RuntimeError("source index owner 与 aggregate 漂移")
        source = SourceRef.from_stable_key(record.source_key)
        if self.aggregates.source_hash(source) != record.source_hash:
            raise RuntimeError("source index hash 与完整 SourceRef 漂移")
        return source

    @staticmethod
    def _insert_candidate(
            candidates: dict[tuple[int, tuple[int, ...]], ResolvedCandidate],
            candidate: ResolvedCandidate,
            ) -> None:
        """按完整候选身份去重，发现同身份内容漂移时立即失败。"""
        key = (candidate.origin_kind, candidate.candidate_key)
        previous = candidates.get(key)
        if previous is not None and previous != candidate:
            raise RuntimeError("resolver 命中重复候选但内容不同")
        candidates[key] = candidate


__all__ = ["MemoryOverlayResolver"]
