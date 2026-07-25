"""M-07 Core/Memory overlay resolver 的检索、仲裁和隔离对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    OwnerScope,
    SourceRef,
    VISIBILITY_SESSION,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_CONFLICTED,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryQueryDefinition,
    MemoryQueryProtocol,
    MemoryQueryRoles,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScore,
    ActivationScoreReason,
    CoreBaselineCandidate,
    MemoryAggregateFilter,
    RESOLUTION_ORIGIN_CORE,
    RESOLUTION_ORIGIN_MEMORY,
    SourceDiversityAssessment,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.memory_resolver_runtime import (
    install_memory_resolver_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_aggregate import (
    MemoryAggregateIntegrityError,
)

from test_m03_memory_event import _core_refs, _source as _memory_source
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source as _query_source,
)


def _instruction(source: SourceRef, value: int) -> ObjectIdentity:
    """构造测试协议使用的一等 MinimalInstruction。"""
    return minimal_instruction_identity(
        (value,), owner=source.owner, versions=source.versions)


def _protocol(
        source: SourceRef,
        *,
        first_budget: int = 3,
        ) -> MemoryQueryProtocol:
    """构造两个独立 Hypothesis kind 和预算，验证分型 Top-K 不串账。"""
    roles = MemoryQueryRoles(*(
        _instruction(source, value)
        for value in range(8101, 8109)
    ))
    return MemoryQueryProtocol(
        roles,
        (
            MemoryQueryDefinition(
                _instruction(source, 8201),
                (7201,),
                (roles.occurrence, roles.domain),
                first_budget,
            ),
            MemoryQueryDefinition(
                _instruction(source, 8202),
                (7202,),
                (roles.span, roles.intent),
                1,
            ),
        ),
    )


def _hypothesis(
        source: SourceRef,
        *,
        kind: tuple[int, ...],
        candidate: int,
        competition: int,
        ) -> HypothesisKey:
    """构造来源化 Memory 候选，候选值由测试课程注入。"""
    return HypothesisKey(
        kind,
        (candidate,),
        (competition,),
        document_scope(source),
        source,
    )


class _BaselineProvider:
    """测试用只读 Core 基线提供方。"""

    def __init__(self, core_ref) -> None:
        """绑定一个已存在的 Core typed ref。"""
        self.core_ref = core_ref

    def candidates(self, request):
        """为每个 request 返回同一 Core 身份，但使用该 request 的竞争组。"""
        return (CoreBaselineCandidate(
            self.core_ref,
            (8300, *request.hypothesis_kind),
            ActivationScore(
                50,
                (ActivationScoreReason(
                    (8301, *request.hypothesis_kind), 50),),
            ),
        ),)

    def clone_for_context(self, ctx):
        """为评测上下文创建独立 provider 对象，Core 引用保持同一身份。"""
        del ctx
        return _BaselineProvider(self.core_ref)

    def state_key(self):
        """返回 provider 的完整 Core 引用状态键。"""
        return (1, *self.core_ref.stable_key())


class _CurrentContextScorer:
    """测试用分型整数评分器，使当前文档稳定改变 winner。"""

    def score(self, request, hypothesis, aggregate, sources):
        """当前文档命中 candidate 时给高分，否则按支持事件数给较低分。"""
        del sources
        candidate = hypothesis.candidate_key[0]
        value = (
            1000
            if candidate == request.source.document_id
            else aggregate.support_count * 100
        )
        return ActivationScore(
            value,
            (ActivationScoreReason(
                (8401, request.source.document_id, candidate), value),),
        )

    def clone_for_context(self, ctx):
        """返回无共享可变状态的新评分器。"""
        del ctx
        return _CurrentContextScorer()

    def state_key(self):
        """返回评分器协议版本。"""
        return (1,)


class _IndexFilterProvider:
    """测试用 M-04 索引过滤器，可选择限制为一个完整来源。"""

    def __init__(self, source: SourceRef | None = None) -> None:
        """保存可选来源条件；空条件仍保留 request kind 预过滤。"""
        self.source = source

    def filters(self, request):
        """返回单个 AND 分支，kind 始终由 resolver 从 request 注入。"""
        del request
        return (MemoryAggregateFilter(source=self.source),)

    def clone_for_context(self, ctx):
        """返回无共享可变状态的新索引过滤器。"""
        del ctx
        return _IndexFilterProvider(self.source)

    def state_key(self):
        """返回过滤器协议版本和可选完整来源身份。"""
        return (
            1,
            0 if self.source is None else 1,
            *(() if self.source is None else self.source.stable_key()),
        )


class _DistinctSourcePolicy:
    """测试用来源多样性策略，只按完整 SourceRef 去重数调整整数分。"""

    def assess(self, request, hypothesis, aggregate, sources, source_traces):
        """每个独立来源增加十分，并保留独立理由。"""
        del request, hypothesis, aggregate, sources
        source_count = len({item.source_cluster_key for item in source_traces})
        adjustment = source_count * 10
        return SourceDiversityAssessment(
            source_count,
            adjustment,
            (ActivationScoreReason((8501, source_count), adjustment),),
        )

    def select(self, request, candidates, budget):
        """先选能引入新来源的候选，再按基础分顺序填满剩余预算。"""
        del request
        selected = []
        pending = []
        seen_sources = set()
        for candidate in candidates:
            source_keys = {
                trace.source_cluster_key
                for trace in candidate.memory_source_traces}
            if source_keys - seen_sources:
                selected.append(candidate)
                seen_sources.update(source_keys)
            else:
                pending.append(candidate)
            if len(selected) == budget:
                return tuple(selected)
        selected.extend(pending[:budget - len(selected)])
        return tuple(selected)

    def clone_for_context(self, ctx):
        """返回无共享可变状态的新多样性策略。"""
        del ctx
        return _DistinctSourcePolicy()

    def state_key(self):
        """返回来源策略协议版本。"""
        return (1,)


def _install_resolver(ctx, protocol, core_ref, *, source_filter=None):
    """安装同一 aggregate 上的 M-06 compiler 和 M-07 resolver runtime。"""
    query_runtime = install_memory_query_runtime(
        ctx,
        protocol,
        aggregates=ctx.memory_interact_aggregates,
    )
    resolver = MemoryOverlayResolver(
        ctx.memory_interact_aggregates,
        ctx.core_identity_catalog,
        _BaselineProvider(core_ref),
        _IndexFilterProvider(source_filter),
        _CurrentContextScorer(),
        _DistinctSourcePolicy(),
    )
    resolver_runtime = install_memory_resolver_runtime(ctx, resolver)
    return query_runtime, resolver_runtime


def _seed_memory(ctx):
    """写入两个同 kind 候选和一个第二 kind 候选，并形成冲突与同源重复。"""
    source_a = _memory_source(source_id=10, document_id=11)
    source_b = _memory_source(source_id=20, document_id=12)
    first = _hypothesis(
        source_a, kind=(7201,), candidate=1, competition=8601)
    second = _hypothesis(
        source_b, kind=(7201,), candidate=2, competition=8601)
    other_kind = _hypothesis(
        source_a, kind=(7202,), candidate=3, competition=8602)
    ledger = HypothesisLedger(
        MemoryHypothesisEventSink(ctx.memory_interact_events))
    for hypothesis in (first, second, other_kind):
        ledger.register(hypothesis)
    ledger.append_evidence(EvidenceRecord(
        1, first, EVIDENCE_SUPPORT, (8701,), source_a, 1))
    ledger.append_evidence(EvidenceRecord(
        2, first, EVIDENCE_REFUTE, (8702,), source_b, 2))
    for evidence_id in range(3, 8):
        ledger.append_evidence(EvidenceRecord(
            evidence_id,
            second,
            EVIDENCE_SUPPORT,
            (8700 + evidence_id,),
            source_b,
            evidence_id,
        ))
    ledger.append_evidence(EvidenceRecord(
        8, other_kind, EVIDENCE_SUPPORT, (8708,), source_a, 8))
    access = MemoryAccessContext(1, 2, 3)
    ctx.memory_interact_aggregates.rebuild_dirty(access=access)
    return first, second, other_kind


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_resolver_preserves_conflict_sources_and_deduplicates_events(
        backend_type):
    """Core 与 Memory 同场仲裁，冲突和来源保留，同源五事件只占一个候选。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        core_ref = _core_refs(ctx)[1]
        first, second, other_kind = _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, _protocol(query_source), core_ref)
        scope = _open_query(ctx, query_source)
        current = _current(ctx, query_source, scope)
        compilation = query_runtime.compile(
            current,
            access=MemoryAccessContext(1, 2, 3),
        )
        before = backend.snapshot()

        resolution = resolver_runtime.resolve(compilation)

        assert backend.snapshot() == before
        assert len(resolution.sets) == 2
        first_set, second_set = resolution.sets
        assert first_set.considered_count == 3
        assert len(first_set.candidates) == 3
        assert first_set.candidates[0].hypothesis == first
        assert sum(
            item.hypothesis == second for item in first_set.candidates) == 1
        conflict = next(
            item for item in first_set.candidates
            if item.hypothesis == first)
        assert conflict.origin_kind == RESOLUTION_ORIGIN_MEMORY
        assert conflict.aggregate.evidence_state == MEMORY_EVIDENCE_CONFLICTED
        assert len(conflict.sources) == 2
        assert {item.stance for item in conflict.memory_source_traces} == {
            EVIDENCE_SUPPORT,
            EVIDENCE_REFUTE,
        }
        assert conflict.candidate_scope == first.scope
        assert conflict.query_scope == scope
        assert len(conflict.score_reasons) == 2
        baseline = next(
            item for item in first_set.candidates
            if item.origin_kind == RESOLUTION_ORIGIN_CORE)
        assert baseline.core_ref == core_ref
        assert baseline.memory_ref is None
        assert second_set.considered_count == 2
        assert len(second_set.candidates) == 1
        assert second_set.candidates[0].hypothesis == other_kind
        assert resolver_runtime.resolve(
            compilation).stable_key() == resolution.stable_key()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_current_context_changes_winner_without_previous_reward_or_direction():
    """只改变当前来源和 query scope 即改变 winner，WorkMemory 历史残留不参与评分。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first, second, _ = _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, _protocol(query_source), _core_refs(ctx)[1])
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )
        ctx.work_memory.replay_candidates[:] = [(999,)]
        ctx.work_memory.pr_vector[(1, 2)] = 9999
        first_winner = resolver_runtime.resolve(
            compilation).sets[0].candidates[0]
        assert first_winner.hypothesis == first
        _close_query(ctx)

        changed_source = _query_source(document_id=2)
        changed_scope = _open_query(ctx, changed_source, local_id=2)
        changed = query_runtime.compile(
            _current(ctx, changed_source, changed_scope, ordinal=1),
            access=MemoryAccessContext(1, 2, 3),
        )
        second_winner = resolver_runtime.resolve(
            changed).sets[0].candidates[0]
        assert second_winner.hypothesis == second
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_injected_source_filter_reduces_candidates_before_event_restore():
    """来源过滤先走 M-04 索引，未命中候选不进入完整事件恢复和评分。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first, second, _ = _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        source_filter = _memory_source(source_id=10, document_id=11)
        query_runtime, resolver_runtime = _install_resolver(
            ctx,
            _protocol(query_source),
            _core_refs(ctx)[1],
            source_filter=source_filter,
        )
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )

        first_set = resolver_runtime.resolve(compilation).sets[0]

        assert first_set.considered_count == 2
        assert any(item.hypothesis == first for item in first_set.candidates)
        assert all(item.hypothesis != second for item in first_set.candidates)
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_selection_policy_prevents_one_source_from_filling_top_k():
    """集合级多样性可让新来源候选进入 Top-K，而同源高分候选留在考虑集。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first, second, _ = _seed_memory(ctx)
        source_c = _memory_source(source_id=30, document_id=13)
        third = _hypothesis(
            source_c,
            kind=(7201,),
            candidate=4,
            competition=8601,
        )
        ledger = HypothesisLedger(
            MemoryHypothesisEventSink(ctx.memory_interact_events))
        ledger.register(third)
        ledger.append_evidence(EvidenceRecord(
            100, third, EVIDENCE_SUPPORT, (8800,), source_c, 10))
        ctx.memory_interact_aggregates.rebuild_dirty(
            access=MemoryAccessContext(1, 2, 3))
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx,
            _protocol(query_source, first_budget=2),
            _core_refs(ctx)[1],
        )
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )

        first_set = resolver_runtime.resolve(compilation).sets[0]

        assert first_set.considered_count == 4
        assert tuple(item.hypothesis for item in first_set.candidates) == (
            first,
            third,
        )
        assert all(item.hypothesis != second for item in first_set.candidates)
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_aggregate_public_ref_reader_checks_acl_space_and_owner_drift():
    """M-07 只能经公共 ACL 接口恢复引用，跨空间或 owner 漂移均失败。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first, _, _ = _seed_memory(ctx)
        access = MemoryAccessContext(1, 2, 3)
        records = ctx.memory_interact_aggregates.query(
            access=access,
            hypothesis_kind=first.hypothesis_kind,
        )
        aggregate = next(
            item for item in records
            if ctx.memory_interact_aggregates.hypothesis_ref_for_aggregate(
                item, access=access).object_key == first.stable_key()
        )
        ref = ctx.memory_interact_aggregates.hypothesis_ref_for_aggregate(
            aggregate, access=access)
        assert ref is not None and ref.object_key == first.stable_key()
        assert ctx.memory_interact_aggregates.hypothesis_ref_for_aggregate(
            aggregate,
            access=MemoryAccessContext(1, 2, 99),
        ) is None
        with pytest.raises(ValueError, match="其他 Memory 空间"):
            ctx.memory_interact_aggregates.hypothesis_ref_for_aggregate(
                replace(aggregate, space_id=999), access=access)
        with pytest.raises(MemoryAggregateIntegrityError, match="owner"):
            ctx.memory_interact_aggregates.hypothesis_ref_for_aggregate(
                replace(
                    aggregate,
                    owner_key=GLOBAL_OWNER_SCOPE.stable_key(),
                ),
                access=access,
            )
    finally:
        backend.close()


def test_core_baseline_object_owner_must_be_visible_to_request_acl():
    """存在于 Core 的另一 session 对象仍不可作为当前 request baseline 泄露。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        private_ref = ctx.graph_ontology.materialize(concept_identity(
            (8999,),
            owner=OwnerScope(1, 2, 99, VISIBILITY_SESSION),
            versions=VersionBundle(),
        ))
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, _protocol(query_source), private_ref)
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )
        before = backend.snapshot()

        with pytest.raises(PermissionError, match="Core baseline 对象"):
            resolver_runtime.resolve(compilation)
        assert backend.snapshot() == before
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_resolver_rejects_stale_aggregate_until_dirty_rebuild_finishes():
    """新 Evidence 产生 dirty 后不得读取旧冲突快照，显式重建后才能继续。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first, _, _ = _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, _protocol(query_source), _core_refs(ctx)[1])
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = sink.load_ledger(
            access=MemoryAccessContext(1, 2, 3),
            hypotheses=(first,),
            attach_sink=True,
        )
        ledger.append_evidence(EvidenceRecord(
            99,
            first,
            EVIDENCE_SUPPORT,
            (8799,),
            _memory_source(source_id=30, document_id=13),
            99,
        ))

        with pytest.raises(MemoryAggregateIntegrityError, match="dirty"):
            resolver_runtime.resolve(compilation)

        ctx.memory_interact_aggregates.rebuild_dirty(
            access=MemoryAccessContext(1, 2, 3))
        resolution = resolver_runtime.resolve(compilation)
        resolved = next(
            item for item in resolution.sets[0].candidates
            if item.hypothesis == first)
        assert len(resolved.sources) == 3
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_v06_clone_rebinds_resolver_facades_and_injected_components():
    """V-06 resolver 使用克隆 aggregate/Core 和策略对象，不共享宿主 facade。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        query_source = _query_source(document_id=1)
        _, runtime = _install_resolver(
            ctx, _protocol(query_source), _core_refs(ctx)[1])
        with isolated_evaluation(ctx, label="m07-memory-resolver") as cloned:
            cloned_runtime = cloned.memory_resolver_runtime
            assert cloned_runtime is not runtime
            assert (cloned_runtime.resolver.aggregates
                    is cloned.memory_interact_aggregates)
            assert (cloned_runtime.resolver.aggregates
                    is not runtime.resolver.aggregates)
            assert (cloned_runtime.resolver.core_identities
                    is cloned.core_identity_catalog)
            assert (cloned_runtime.resolver.baseline_provider
                    is not runtime.resolver.baseline_provider)
            assert cloned_runtime.state_key() == runtime.state_key()
    finally:
        backend.close()
