"""M-08 真实 Memory Use、延迟结果归因和评测隔离测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorConsumptionDecision,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    EpisodePayload,
    MemoryLinkedRef,
    UseOutcomePayload,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MemoryEventIntegrityError,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningBudget
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)

from test_a10_attractor_state import (
    _close_query,
    _instruction,
    _planner,
    _setup,
)
from test_m03_memory_event import _append_observation, _core_refs


_ACCESS = MemoryAccessContext(1, 2, 3)


def _events(ctx, event_kind, *, object_ref=None):
    """读取测试 owner 可见的指定 Memory 事件。"""
    return ctx.memory_interact_events.query(
        access=_ACCESS,
        event_kind=event_kind,
        object_ref=object_ref,
    )


def _consume_head(runtime, source):
    """经真实 S-05 consumer 消费当前 frontier head 并返回 processing trace。"""
    _, _, consumer = _planner(source)
    result = runtime.consume_reasoning(consumer, ReasoningBudget(1, 0, 0))
    assert result is not None
    state = runtime._ctx.work_memory.require_attractor_state()
    return state.processing_traces()[-1]


def _query_time(state, seq: int) -> LogicalTimestamp:
    """在当前 query 的同一逻辑时钟上构造后续时间。"""
    return LogicalTimestamp(state.current_timestamp.clock, seq)


def test_consumed_frontier_writes_exact_use_and_delayed_outcome_only():
    """两个真实消费各写一个 Use，延迟结果只落到目标 Use。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        memory_use = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)

        assert _events(ctx, MEMORY_EVENT_USE) == ()
        assert state.processing_traces() == ()

        first_trace = _consume_head(attractor, source)
        first = memory_use.record_selection_use(
            first_trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 9201)),
            used_at=_query_time(state, 2),
        )
        second_trace = _consume_head(attractor, source)
        second = memory_use.record_selection_use(
            second_trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 9202)),
            used_at=_query_time(state, 3),
        )

        assert first.processing.activation.candidate.memory_ref != (
            second.processing.activation.candidate.memory_ref)
        assert len(_events(ctx, MEMORY_EVENT_USE)) == 2
        first_payload = first.use.event.payload
        assert isinstance(first_payload, UsePayload)
        assert first_payload.memory_ref == (
            first_trace.activation.candidate.memory_ref)
        assert first_payload.decision_trace_key == first_trace.stable_key()
        assert first_payload.query_kind == MemoryLinkedRef.object(
            first_trace.activation.request.query_kind)
        assert first_payload.context_key
        episode = first.episode.event.payload
        assert isinstance(episode, EpisodePayload)
        assert episode.selected_path_ref == MemoryLinkedRef.memory(
            first_payload.memory_ref)
        assert len(episode.candidate_refs) == 2

        aggregate_index = ctx.memory_interact_aggregates
        aggregate_index.rebuild_dirty(access=_ACCESS)
        for result in (first, second):
            aggregate = aggregate_index.read(
                result.use.event.payload.memory_ref,
                access=_ACCESS,
            )
            assert aggregate is not None and aggregate.use_count == 1

        outcome_kind = MemoryLinkedRef.object(_instruction(source, 9211))
        outcome = memory_use.record_outcome(
            first.use.event.object_ref,
            scope=state.scope,
            outcome_kind=outcome_kind,
            outcome_ref=None,
            observed_at=_query_time(state, 4),
        )
        payload = outcome.event.payload
        assert isinstance(payload, UseOutcomePayload)
        assert UseOutcomePayload.from_stable_key(
            payload.stable_key()) == payload
        traced_payload = replace(payload, outcome_trace_key=(9253, 1, 2, 3))
        assert UseOutcomePayload.from_stable_key(
            traced_payload.stable_key()) == traced_payload
        assert traced_payload.stable_key()[:-5] == payload.stable_key()
        assert payload.target_ref == first.use.event.object_ref
        assert payload.decision_trace_key == first_payload.decision_trace_key
        assert payload.query_kind == first_payload.query_kind
        assert payload.context_key == first_payload.context_key
        assert len(_events(
            ctx,
            MEMORY_EVENT_USE_OUTCOME,
            object_ref=first.use.event.object_ref,
        )) == 1
        assert _events(
            ctx,
            MEMORY_EVENT_USE_OUTCOME,
            object_ref=second.use.event.object_ref,
        ) == ()

        with pytest.raises(MemoryEventIntegrityError, match="同类延迟结果"):
            memory_use.record_outcome(
                first.use.event.object_ref,
                scope=state.scope,
                outcome_kind=outcome_kind,
                outcome_ref=None,
                observed_at=_query_time(state, 5),
            )
    finally:
        _close_query(ctx)
        backend.close()


def test_suspended_or_unprocessed_candidate_never_writes_use():
    """召回和入 agenda 不写 Use，暂停的真实处理也不能冒充采用。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        memory_use = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        head = state.next_activation()
        assert head is not None
        assert _events(ctx, MEMORY_EVENT_USE) == ()

        decision = AttractorConsumptionDecision(
            head.identity_key(),
            _instruction(source, 9221),
            state.protocol.suspended,
            (9222,),
        )
        trace = state.commit_consumption(decision)
        with pytest.raises(ValueError, match="consumed"):
            memory_use.record_selection_use(
                trace,
                input_observation_ref=observation.event.object_ref,
                influence_kind=MemoryLinkedRef.object(
                    _instruction(source, 9223)),
                used_at=_query_time(state, 2),
            )
        assert _events(ctx, MEMORY_EVENT_USE) == ()
    finally:
        _close_query(ctx)
        backend.close()


def test_selection_use_is_idempotent_and_rejects_competing_attribution():
    """同一 processing 可稳定重试，但不能改换影响类型或逻辑时间。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        memory_use = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        trace = _consume_head(attractor, source)
        influence = MemoryLinkedRef.object(_instruction(source, 9231))
        first = memory_use.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=influence,
            used_at=_query_time(state, 2),
        )
        repeated = memory_use.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=influence,
            used_at=_query_time(state, 2),
        )
        assert repeated == first
        assert len(_events(ctx, MEMORY_EVENT_USE)) == 1

        with pytest.raises(MemoryEventIntegrityError):
            memory_use.record_selection_use(
                trace,
                input_observation_ref=observation.event.object_ref,
                influence_kind=MemoryLinkedRef.object(
                    _instruction(source, 9232)),
                used_at=_query_time(state, 2),
            )
        assert len(_events(ctx, MEMORY_EVENT_USE)) == 1
    finally:
        _close_query(ctx)
        backend.close()


def test_incremental_selection_use_never_scans_all_use_events(monkeypatch):
    """每次新增 Use 只能走对象 identity 索引，不得扫描全空间历史。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        memory_use = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        trace = _consume_head(attractor, source)
        original_query = memory_use.event_log.query

        def guarded_query(**kwargs):
            """拒绝无对象边界的 Use 查询，放行 Observation 定点恢复。"""
            if (kwargs.get("event_kind") == MEMORY_EVENT_USE
                    and kwargs.get("object_ref") is None):
                raise AssertionError("增量 Use 提交扫描了全部历史")
            return original_query(**kwargs)

        monkeypatch.setattr(memory_use.event_log, "query", guarded_query)
        result = memory_use.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 9241)),
            used_at=_query_time(state, 2),
        )
        assert result.use.event.payload.memory_ref == (
            trace.activation.candidate.memory_ref)
    finally:
        _close_query(ctx)
        backend.close()


def test_v06_clones_memory_use_runtime_and_keeps_outcome_in_sandbox():
    """评测 clone 的延迟结果只写沙箱 event log，不污染宿主 Memory。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        host_runtime = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        trace = _consume_head(attractor, source)
        use = host_runtime.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 9251)),
            used_at=_query_time(state, 2),
        )
        use_ref = use.use.event.object_ref
        query_scope = state.scope
        _close_query(ctx)
        host_before = backend.snapshot()

        with isolated_evaluation(ctx, label="m08-outcome") as eval_ctx:
            cloned = eval_ctx.memory_use_runtime
            assert cloned is not host_runtime
            assert cloned.state_key() == host_runtime.state_key()
            clone_before = eval_ctx.backend.snapshot()
            cloned.record_outcome(
                use_ref,
                scope=query_scope,
                outcome_kind=MemoryLinkedRef.object(
                    _instruction(source, 9252)),
                outcome_ref=None,
                observed_at=LogicalTimestamp(
                    trace.activation.request.logical_timestamp.clock,
                    3,
                ),
            )
            assert eval_ctx.backend.snapshot() != clone_before
            assert len(_events(
                eval_ctx,
                MEMORY_EVENT_USE_OUTCOME,
                object_ref=use_ref,
            )) == 1
            assert _events(
                ctx,
                MEMORY_EVENT_USE_OUTCOME,
                object_ref=use_ref,
            ) == ()

        assert backend.snapshot() == host_before
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()
