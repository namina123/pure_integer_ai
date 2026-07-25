"""G-05 一等 Memory generation Evidence 与 SourceRecord 分账测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MemoryLinkedRef,
    UseOutcomePayload,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationEvidence,
    MemoryGenerationSource,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
)
from pure_integer_ai.experiments.memory_generation_runtime import (
    MemoryAwareQuestionDialogueRuntime,
    MemoryQuestionSelectionCommitter,
    ResolvedMemoryQuestionExecutor,
)
from pure_integer_ai.experiments.memory_generation_outcome_runtime import (
    MemoryGenerationOutcomeProtocol,
    MemoryGenerationOutcomeRoute,
    MemoryGenerationOutcomeValueRoute,
    MemoryQuestionOutcomeCommitter,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationReport,
)
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)
from tests.test_a10_attractor_state import _instruction, _setup
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
)
from tests.test_m06_memory_query import _close_query
from tests.test_m08_memory_use import (
    _append_observation,
    _core_refs,
    _events,
)


def _complete_source(repository, trace, ordinal: int):
    """为一条 M-07 来源分账建立完整 SourceRecord 并返回纯整数投影。"""
    record = repository.put_complete(
        trace.source.stable_key(),
        f"来源{ordinal}",
        metadata=SourceRecordMetadata(
            "fixture-license",
            ordinal,
            100 + ordinal,
            200 + ordinal,
            300 + ordinal,
        ),
    )
    return MemoryGenerationSource.from_record(trace, record)


class _MemoryQuestionExecutor:
    """把预先完成来源绑定的 Memory generation candidate 接到统一问题 route。"""

    def __init__(self, candidate, reason) -> None:
        """绑定唯一候选和注入式执行原因。"""
        self.candidate = candidate
        self.reason = reason

    def execute(self, query):
        """返回同次 query 的 Memory candidate，不读取 expected 或答案文字。"""
        return QuestionExecutionResult(
            query,
            self.reason,
            (self.candidate,),
            (19810, *query.route.stable_key()),
        )


class _EmptyQuestionExecutor:
    """返回真实执行过但没有候选的 Memory OFF 对照 route。"""

    def __init__(self, reason) -> None:
        """绑定注入式空结果原因。"""
        self.reason = reason

    def execute(self, query):
        """返回同次 query 的空候选结果，不调用 Memory runtime。"""
        return QuestionExecutionResult(query, self.reason, (), (19811, 1))


class _UseBeforePostcheckMapper:
    """核验 M-08 Use 已提交后才把同次 generation 交给 G-04 mapper。"""

    def __init__(self, ctx, delegate) -> None:
        """绑定 Memory 事件 owner 和既有 postcheck mapper。"""
        self.ctx = ctx
        self.delegate = delegate
        self.calls = 0

    def build(self, request, query, result, generation):
        """要求当前 query 已有实际 Use，再委托来源化 G-04 请求构造。"""
        assert _events(self.ctx, MEMORY_EVENT_USE)
        self.calls += 1
        return self.delegate.build(request, query, result, generation)


def _memory_candidate(repository, activation, source, ordinal: int = 1):
    """从 A-10 activation 和完整 SourceRecord 建立 G-00 Memory 候选。"""
    sources = tuple(
        _complete_source(repository, trace, index)
        for index, trace in enumerate(
            activation.candidate.memory_source_traces,
            start=ordinal,
        )
    )
    evidence = MemoryGenerationEvidence(
        activation.candidate,
        activation.obligation.proposition,
        _instruction(source, 19801 + ordinal),
        (19802, ordinal),
        sources,
    )
    candidate = GenerationCandidate(
        activation.obligation.proposition,
        evidence.state,
        source,
        activation.candidate.query_scope,
        (),
        memory_evidence=(evidence,),
    )
    return candidate, evidence


def _outcome_protocol(source, postchecker, *, complete: bool = True):
    """按 G-04 注入维度建立逐维 outcome 路由，不在 runtime 写死结果语义。"""
    pairs = (
        (APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),
        (APPLICABILITY_APPLICABLE, VERDICT_REFUTE),
        (APPLICABILITY_APPLICABLE, VERDICT_UNKNOWN),
        (APPLICABILITY_APPLICABLE, VERDICT_CONFLICTED),
        (APPLICABILITY_NOT_APPLICABLE, VERDICT_UNKNOWN),
        (APPLICABILITY_UNKNOWN, VERDICT_UNKNOWN),
    )
    routes = []
    for route_index, (dimension, verifier) in enumerate(
            postchecker.protocol.bindings(), start=1):
        selected_pairs = pairs if complete else ((999_001, 999_002),)
        values = tuple(
            MemoryGenerationOutcomeValueRoute(
                applicability,
                verdict,
                MemoryLinkedRef.object(_instruction(
                    source,
                    19_900 + route_index * 10 + value_index,
                )),
            )
            for value_index, (applicability, verdict) in enumerate(
                selected_pairs,
                start=1,
            )
        )
        routes.append(MemoryGenerationOutcomeRoute(
            dimension,
            verifier,
            MemoryLinkedRef.object(_instruction(
                source,
                19_800 + route_index,
            )),
            values,
        ))
    return MemoryGenerationOutcomeProtocol(tuple(routes))


def _core_graph_snapshot(backend, core_space_id: int):
    """冻结 Core 节点、定义、关联和宽边行，排除预期的 Memory/SourceRecord 写入。"""
    snapshot = backend.snapshot()
    return (
        tuple(row for row in snapshot["concept_node"]
              if row["space_id"] == core_space_id),
        tuple(row for row in snapshot["def_array"]
              if row["space_id"] == core_space_id),
        tuple(row for row in snapshot["assoc_table"]
              if row["space_id"] == core_space_id),
        tuple(row for row in snapshot["edge"]
              if (row["space_id_from"] == core_space_id
                  or row["space_id_to"] == core_space_id)),
    )


def test_memory_evidence_keeps_resolved_identity_sources_and_target_binding():
    """Memory 候选不伪装 H-00 Evidence，仍可形成带来源分账的 G-00 候选。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        assert activation.candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
        repository = SourceRecordRepository(backend)
        sources = tuple(
            _complete_source(repository, trace, ordinal)
            for ordinal, trace in enumerate(
                activation.candidate.memory_source_traces,
                start=1,
            )
        )
        evidence = MemoryGenerationEvidence(
            activation.candidate,
            activation.obligation.proposition,
            _instruction(source, 19801),
            (19802, 1),
            sources,
        )

        candidate = GenerationCandidate(
            activation.obligation.proposition,
            evidence.state,
            source,
            state.scope,
            (),
            memory_evidence=(evidence,),
        )

        assert candidate.evidence == ()
        assert candidate.memory_evidence == (evidence,)
        assert candidate.hypotheses == (activation.candidate.hypothesis,)
        assert candidate.citation_sources == tuple(
            item.trace.source for item in sources)
        assert all(item.source_record_hash > 0 for item in sources)
        assert all(item.companion_assoc_id > 0 for item in sources)
    finally:
        _close_query(ctx)
        backend.close()


def test_memory_evidence_rejects_missing_complete_source_record_metadata():
    """缺许可或 Companion 绑定的 SourceRecord 不得进入 Memory 回答证据。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, _, attractor, _, compilation, goals = setup
    try:
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        trace = activation.candidate.memory_source_traces[0]
        record = SourceRecordRepository(backend).put(
            trace.source.stable_key(),
            "不完整来源",
        )

        with pytest.raises(ValueError, match="完整许可"):
            MemoryGenerationSource.from_record(trace, record)
    finally:
        _close_query(ctx)
        backend.close()


def test_selected_memory_answer_commits_frontier_and_writes_exact_use():
    """成功 surface 后仅把 G-01 已选 frontier 项提交 consumed 并写唯一 Use。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    off_fixture = None
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        candidate, evidence = _memory_candidate(
            SourceRecordRepository(backend), activation, source)
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19820),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19821)),
            trace_prefix=(19822, 1),
        )
        fixture = _question_fixture(
            executor_factory=lambda route: _MemoryQuestionExecutor(
                candidate, _instruction(source, 19823)),
            world=(source, state.scope, activation.obligation.proposition),
            selection_committer=committer,
        )

        run = fixture.runtime.run(fixture.request)

        assert run.complete
        assert run.selection_commit is not None
        assert len(run.selection_commit.commits) == 1
        commit = run.selection_commit.commits[0]
        assert commit.evidence_key == evidence.stable_key()
        assert state.processing_traces()[0].decision.disposition == (
            state.protocol.consumed)
        uses = _events(ctx, MEMORY_EVENT_USE)
        assert len(uses) == 1
        assert uses[0].event.object_ref == commit.use_ref
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_unselected_memory_candidate_keeps_agenda_and_writes_no_use():
    """G-01 返回 unknown 时 Memory 候选虽已检索并入 agenda，仍不得形成 Use。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        candidate, _ = _memory_candidate(
            SourceRecordRepository(backend), activation, source)
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19830),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19831)),
            trace_prefix=(19832, 1),
        )
        fixture = _question_fixture(
            required=LogicEvidenceState(
                not candidate.state.support,
                not candidate.state.refute,
            ),
            executor_factory=lambda route: _MemoryQuestionExecutor(
                candidate, _instruction(source, 19833)),
            world=(source, state.scope, activation.obligation.proposition),
            selection_committer=committer,
        )

        run = fixture.runtime.run(fixture.request)

        assert run.complete
        assert run.selection.selected_candidate_keys == ()
        assert run.selection_commit is not None
        assert run.selection_commit.commits == ()
        assert state.processing_traces() == ()
        assert _events(ctx, MEMORY_EVENT_USE) == ()
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_non_frontier_memory_selection_fails_before_consumption_or_use():
    """选择非 frontier Memory 项必须在 A-10 和 M-08 都零写时 fail closed。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        frontier = state.frontier()
        assert len(frontier) >= 2
        candidate, _ = _memory_candidate(
            SourceRecordRepository(backend), frontier[1], source, ordinal=10)
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19840),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19841)),
            trace_prefix=(19842, 1),
        )
        fixture = _question_fixture(
            executor_factory=lambda route: _MemoryQuestionExecutor(
                candidate, _instruction(source, 19843)),
            world=(source, state.scope, frontier[1].obligation.proposition),
            selection_committer=committer,
        )

        with pytest.raises(ValueError, match="frontier"):
            fixture.runtime.run(fixture.request)

        assert state.processing_traces() == ()
        assert _events(ctx, MEMORY_EVENT_USE) == ()
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_dialogue_caller_runs_memory_chain_backtraces_sources_and_closes_query():
    """完整 caller 应执行 M-06 至 G-04 前链、回查来源并在返回前关闭 query。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        repository = SourceRecordRepository(backend)
        resolution = ctx.memory_resolver_runtime.resolve(compilation)
        traces = {
            trace.source.stable_key(): trace
            for candidate_set in resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
            for trace in candidate.memory_source_traces
        }
        for ordinal, trace in enumerate(
                (traces[key] for key in sorted(traces)), start=1):
            _complete_source(repository, trace, ordinal)
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        current = compilation.current
        ctx.work_memory.end_query()
        target = goals[1].proposition

        off_fixture = _question_fixture(
            executor_factory=lambda route: _EmptyQuestionExecutor(
                _instruction(source, 19849)),
            world=(source, current.scope, target),
        )
        off_dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            off_fixture.runtime,
            trace_prefix=(19849, 2),
            source_records=repository,
        )
        before_core = _core_graph_snapshot(backend, ctx.core_space.space_id)

        off_run = off_dialogue.run(off_fixture.request)

        assert off_run.question.selection.selected_candidate_keys == ()
        assert off_run.sources == ()
        assert _events(ctx, MEMORY_EVENT_USE) == ()

        executor = ResolvedMemoryQuestionExecutor(
            ctx,
            current,
            MemoryAccessContext(1, 2, 3),
            goals,
            executed_reason=_instruction(source, 19850),
            binding_reason=_instruction(source, 19851),
            trace_prefix=(19852, 1),
            source_records=repository,
        )
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19853),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19854)),
            trace_prefix=(19855, 1),
        )
        mapper, postchecker, _, _, _ = _postcheck_owners()
        ordered_mapper = _UseBeforePostcheckMapper(ctx, mapper)
        outcome_committer = MemoryQuestionOutcomeCommitter(
            ctx.memory_use_runtime,
            _outcome_protocol(source, postchecker),
            trace_prefix=(19855, 2),
        )
        fixture = _question_fixture(
            executor_factory=lambda route: executor,
            world=(source, current.scope, target),
            selection_committer=committer,
            postcheck_mapper=ordered_mapper,
            postchecker=postchecker,
            outcome_committer=outcome_committer,
        )
        dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            fixture.runtime,
            trace_prefix=(19856, 1),
            source_records=repository,
        )

        run = dialogue.run(fixture.request)

        assert run.question.complete
        assert run.question.status != off_run.question.status
        assert run.question.selection_commit is not None
        assert run.question.selection_commit.commits
        assert run.question.postcheck is not None
        assert run.question.postcheck.complete
        assert run.question.outcome_commit is not None
        assert run.question.outcome_commit.outcomes
        assert ordered_mapper.calls == 1
        assert run.sources
        for item in run.sources:
            record = repository.read(item.source_record_hash)
            assert record.source_key == item.trace.source.stable_key()
            assert record.metadata_complete
        assert _events(ctx, MEMORY_EVENT_USE)
        outcome_events = _events(ctx, MEMORY_EVENT_USE_OUTCOME)
        assert len(outcome_events) == len(
            run.question.outcome_commit.outcomes)
        for outcome in run.question.outcome_commit.outcomes:
            matches = tuple(
                item for item in outcome_events
                if (item.event.object_ref == outcome.use_ref
                    and isinstance(item.event.payload, UseOutcomePayload)
                    and item.event.payload.outcome_kind == outcome.outcome_kind)
            )
            assert len(matches) == 1
            assert matches[0].event.payload.outcome_ref == outcome.outcome_ref
            assert matches[0].event.payload.outcome_trace_key == outcome.signal_key
        before_replay = tuple(
            item.event.stable_key() for item in outcome_events)
        question = run.question
        replay = outcome_committer.commit(
            question.request,
            question.query,
            question.query_result,
            question.selection,
            question.generation,
            question.selection_commit,
            question.postcheck,
        )
        assert replay == question.outcome_commit
        assert tuple(
            item.event.stable_key()
            for item in _events(ctx, MEMORY_EVENT_USE_OUTCOME)
        ) == before_replay
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.attractor_state is None
        assert _core_graph_snapshot(backend, ctx.core_space.space_id) == before_core
    finally:
        if fixture is not None:
            fixture.close()
        if off_fixture is not None:
            off_fixture.close()
        _close_query(ctx)
        backend.close()


def test_memory_outcome_missing_value_route_fails_before_first_outcome():
    """缺任一 G-04 结果映射时可保留真实 Use，但不得写部分 outcome。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        candidate, _ = _memory_candidate(
            SourceRecordRepository(backend), activation, source)
        selection_committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19_860),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19_861)),
            trace_prefix=(19_862, 1),
        )
        mapper, postchecker, _, _, _ = _postcheck_owners()
        outcome_committer = MemoryQuestionOutcomeCommitter(
            ctx.memory_use_runtime,
            _outcome_protocol(source, postchecker, complete=False),
            trace_prefix=(19_863, 1),
        )
        fixture = _question_fixture(
            executor_factory=lambda route: _MemoryQuestionExecutor(
                candidate, _instruction(source, 19_864)),
            world=(source, state.scope, activation.obligation.proposition),
            selection_committer=selection_committer,
            postcheck_mapper=mapper,
            postchecker=postchecker,
            outcome_committer=outcome_committer,
        )

        with pytest.raises(ValueError, match="缺少唯一 Memory outcome 映射"):
            fixture.runtime.run(fixture.request)

        assert len(_events(ctx, MEMORY_EVENT_USE)) == 1
        assert _events(ctx, MEMORY_EVENT_USE_OUTCOME) == ()
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_memory_outcome_rejects_foreign_claim_and_keeps_all_uses_unattributed():
    """G-04 claim 漂移到当前 generation 外时不得归因给任何已采用 Use。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        activation = state.next_activation()
        assert activation is not None
        candidate, _ = _memory_candidate(
            SourceRecordRepository(backend), activation, source)
        selection_committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19_870),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19_871)),
            trace_prefix=(19_872, 1),
        )
        mapper, postchecker, _, _, _ = _postcheck_owners()
        fixture = _question_fixture(
            executor_factory=lambda route: _MemoryQuestionExecutor(
                candidate, _instruction(source, 19_873)),
            world=(source, state.scope, activation.obligation.proposition),
            selection_committer=selection_committer,
            postcheck_mapper=mapper,
            postchecker=postchecker,
        )
        run = fixture.runtime.run(fixture.request)
        assert run.selection_commit is not None
        assert run.postcheck is not None
        results = list(run.postcheck.report.results)
        candidate_key = run.selection_commit.commits[0].candidate_key
        for index, verification in enumerate(results):
            if candidate_key in verification.claim_keys:
                results[index] = replace(
                    verification,
                    claim_keys=tuple(
                        (19_999, item_index)
                        if claim == candidate_key else claim
                        for item_index, claim in enumerate(
                            verification.claim_keys,
                            start=1,
                        )
                    ),
                )
        bad_postcheck = replace(
            run.postcheck,
            report=VerificationReport(True, tuple(results)),
        )
        outcome_committer = MemoryQuestionOutcomeCommitter(
            ctx.memory_use_runtime,
            _outcome_protocol(source, postchecker),
            trace_prefix=(19_874, 1),
        )

        with pytest.raises(ValueError, match="没有 G-04 claimed"):
            outcome_committer.commit(
                run.request,
                run.query,
                run.query_result,
                run.selection,
                run.generation,
                run.selection_commit,
                bad_postcheck,
            )

        assert len(_events(ctx, MEMORY_EVENT_USE)) == 1
        assert _events(ctx, MEMORY_EVENT_USE_OUTCOME) == ()
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_generation_exception_closes_query_and_writes_no_memory_use(monkeypatch):
    """M-07/A-10 后 generation 异常不得调用 committer，并须清理 query 状态。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    fixture = None
    try:
        repository = SourceRecordRepository(backend)
        resolution = ctx.memory_resolver_runtime.resolve(compilation)
        traces = {
            trace.source.stable_key(): trace
            for candidate_set in resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
            for trace in candidate.memory_source_traces
        }
        for ordinal, trace in enumerate(
                (traces[key] for key in sorted(traces)), start=1):
            _complete_source(repository, trace, ordinal)
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        current = compilation.current
        ctx.work_memory.end_query()
        executor = ResolvedMemoryQuestionExecutor(
            ctx,
            current,
            MemoryAccessContext(1, 2, 3),
            goals,
            executed_reason=_instruction(source, 19860),
            binding_reason=_instruction(source, 19861),
            trace_prefix=(19862, 1),
            source_records=repository,
        )
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 19863),
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 19864)),
            trace_prefix=(19865, 1),
        )
        fixture = _question_fixture(
            executor_factory=lambda route: executor,
            world=(source, current.scope, goals[1].proposition),
            selection_committer=committer,
        )
        dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            fixture.runtime,
            trace_prefix=(19866, 1),
            source_records=repository,
        )

        def fail_generation(request):
            """模拟 G-02/G-03 失败，不返回可供 committer 采用的 surface。"""
            del request
            raise RuntimeError("generation failed")

        monkeypatch.setattr(
            fixture.runtime.generator,
            "execute",
            fail_generation,
        )

        with pytest.raises(RuntimeError, match="generation failed"):
            dialogue.run(fixture.request)

        assert _events(ctx, MEMORY_EVENT_USE) == ()
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.attractor_state is None
    finally:
        if fixture is not None:
            fixture.close()
        _close_query(ctx)
        backend.close()


def test_memory_dialogue_runs_in_v06_clone_without_host_write():
    """G-05 在 V-06 独立 backend 内可写 Use，宿主 backend 与生命周期保持不变。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    question_fixture = None
    try:
        repository = SourceRecordRepository(backend)
        resolution = ctx.memory_resolver_runtime.resolve(compilation)
        traces = {
            trace.source.stable_key(): trace
            for candidate_set in resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
            for trace in candidate.memory_source_traces
        }
        for ordinal, trace in enumerate(
                (traces[key] for key in sorted(traces)), start=1):
            _complete_source(repository, trace, ordinal)
        observation = _append_observation(ctx, source, _core_refs(ctx))
        install_memory_use_runtime(ctx)
        current = compilation.current
        host_backend = backend.snapshot()
        host_lifecycle = (
            ctx.work_memory.active_session_scope,
            ctx.work_memory.active_document_scope,
            ctx.work_memory.active_episode_scope,
            ctx.work_memory.active_query_scope,
        )

        with isolated_evaluation(ctx, label="g05-memory-dialogue") as eval_ctx:
            eval_ctx.work_memory.end_session()
            episode = current.scope.parent
            assert episode is not None and episode.parent is not None
            document = episode.parent
            assert document.parent is not None
            eval_ctx.work_memory.begin_session(document.parent)
            eval_ctx.work_memory.begin_document(document)
            eval_ctx.work_memory.begin_episode(episode)
            eval_repository = SourceRecordRepository(eval_ctx.backend)
            executor = ResolvedMemoryQuestionExecutor(
                eval_ctx,
                current,
                MemoryAccessContext(1, 2, 3),
                goals,
                executed_reason=_instruction(source, 19870),
                binding_reason=_instruction(source, 19871),
                trace_prefix=(19872, 1),
                source_records=eval_repository,
            )
            committer = MemoryQuestionSelectionCommitter(
                eval_ctx,
                consumer=_instruction(source, 19873),
                input_observation_ref=observation.event.object_ref,
                influence_kind=MemoryLinkedRef.object(
                    _instruction(source, 19874)),
                trace_prefix=(19875, 1),
            )
            mapper, postchecker, _, _, _ = _postcheck_owners()
            outcome_committer = MemoryQuestionOutcomeCommitter(
                eval_ctx.memory_use_runtime,
                _outcome_protocol(source, postchecker),
                trace_prefix=(19875, 2),
            )
            question_fixture = _question_fixture(
                executor_factory=lambda route: executor,
                world=(source, current.scope, goals[1].proposition),
                selection_committer=committer,
                postcheck_mapper=mapper,
                postchecker=postchecker,
                outcome_committer=outcome_committer,
            )
            dialogue = MemoryAwareQuestionDialogueRuntime(
                eval_ctx,
                question_fixture.runtime,
                trace_prefix=(19876, 1),
                source_records=eval_repository,
            )

            run = dialogue.run(question_fixture.request)

            assert run.question.complete
            assert run.question.outcome_commit is not None
            assert run.question.outcome_commit.outcomes
            clone_uses = eval_ctx.memory_interact_events.query(
                access=MemoryAccessContext(1, 2, 3),
                event_kind=MEMORY_EVENT_USE,
            )
            assert clone_uses
            clone_outcomes = eval_ctx.memory_interact_events.query(
                access=MemoryAccessContext(1, 2, 3),
                event_kind=MEMORY_EVENT_USE_OUTCOME,
            )
            assert len(clone_outcomes) == len(
                run.question.outcome_commit.outcomes)
            assert eval_ctx.work_memory.active_query_scope is None

        assert backend.snapshot() == host_backend
        assert _events(ctx, MEMORY_EVENT_USE) == ()
        assert _events(ctx, MEMORY_EVENT_USE_OUTCOME) == ()
        assert (
            ctx.work_memory.active_session_scope,
            ctx.work_memory.active_document_scope,
            ctx.work_memory.active_episode_scope,
            ctx.work_memory.active_query_scope,
        ) == host_lifecycle
    finally:
        if question_fixture is not None:
            question_fixture.close()
        _close_query(ctx)
        backend.close()
