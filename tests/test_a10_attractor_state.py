"""A-10 query-scoped AttractorState、agenda 消费和局部重算测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.attractor_reasoning import (
    ReasoningAgendaConsumer,
)
from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivationProposal,
    AttractorBudget,
    AttractorConsumptionDecision,
    AttractorContextUpdate,
    AttractorDependency,
    AttractorProtocol,
    AttractorRecomputeDecision,
    AttractorScoreReason,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
    STATE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningBudget,
    ReasoningObligation,
    ReasoningPlanner,
    ReasoningTerminationProtocol,
)
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.attractor_runtime import (
    install_attractor_runtime,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend

from test_m03_memory_event import _core_refs
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source as _query_source,
)
from test_m07_memory_resolver import (
    _install_resolver,
    _protocol as _memory_protocol,
    _seed_memory,
)


_SUPPORT = LogicEvidenceState(True, False)
_UNKNOWN = LogicEvidenceState(False, False)


def _instruction(source: SourceRef, value: int) -> ObjectIdentity:
    """构造测试注入的一等最小指令。"""
    return minimal_instruction_identity(
        (value,), owner=source.owner, versions=source.versions)


def _attractor_protocol(source: SourceRef) -> AttractorProtocol:
    """构造互异 agenda 生命周期状态身份。"""
    return AttractorProtocol(*tuple(
        _instruction(source, value) for value in range(9101, 9105)))


def _binding_failures(source: SourceRef) -> BindingFailureProtocol:
    """构造 S-03 要求的九个互异失败原因。"""
    return BindingFailureProtocol(*tuple(
        _instruction(source, value) for value in range(9111, 9120)))


def _reasoning_protocol(source: SourceRef) -> ReasoningTerminationProtocol:
    """构造 S-05 evaluate 指令和五个终止原因。"""
    return ReasoningTerminationProtocol(*tuple(
        _instruction(source, value) for value in range(9121, 9127)))


def _goals(
        source: SourceRef,
        scope,
        *,
        count: int = 2,
        ) -> tuple[ReasoningObligation, ...]:
    """构造属于当前 query 的 typed 目标，不预置任何真值。"""
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (9130, ordinal)),
            concept_identity(
                (9131, ordinal),
                owner=source.owner,
                versions=source.versions,
            ),
            occurrence_identity(
                source,
                start=ordinal,
                end=ordinal + 1,
                ordinal=ordinal,
            ),
            context_scope_identity(source, (9132, ordinal)),
            (),
        )
        for ordinal in range(1, count + 1)
    )
    templates = tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity(
                (9133, ordinal),
                owner=source.owner,
                versions=source.versions,
            ),
        )
        for ordinal, definition in enumerate(definitions, start=1)
    )
    graph = PropositionTemplateGraph(templates)
    substituter = PropositionSubstituter(SubstitutionProtocol(
        _instruction(source, 9134),
        _binding_failures(source),
    ))
    return tuple(
        ReasoningObligation(
            substituter.substitute(
                definition.proposition, graph, BindingEnvironment()),
            _SUPPORT,
            source,
            scope,
        )
        for definition in definitions
    )


class _GoalMapper:
    """把两个 Memory 候选映射到两个当前目标，并注入 query 相关方向分。"""

    def __init__(self, *, prefer_matching_document: bool) -> None:
        """选择按当前 document 匹配或固定偏向第二目标。"""
        self.prefer_matching_document = prefer_matching_document

    def project(self, request, candidate, obligations):
        """只投影目标 kind 的 Memory 候选，Core 和其他 kind 不进入 agenda。"""
        if candidate.hypothesis is None:
            return ()
        if request.hypothesis_kind != (7201,):
            return ()
        candidate_id = candidate.hypothesis.candidate_key[0]
        if candidate_id not in {1, 2}:
            return ()
        obligation = obligations[candidate_id - 1]
        dependency = AttractorDependency(
            request.query_kind, candidate.hypothesis)
        if self.prefer_matching_document:
            adjustment = (
                5000 if candidate_id == request.source.document_id else 0)
        else:
            adjustment = 5000 if candidate_id == 2 else 0
        reason = AttractorScoreReason(
            _instruction(request.source, 9140 + candidate_id),
            adjustment,
            (dependency,),
        )
        return (AttractorActivationProposal(
            _instruction(request.source, 9143),
            obligation,
            adjustment,
            (reason,),
            (dependency,),
        ),)

    def clone_for_context(self, ctx):
        """为 V-06 返回无共享可变状态的新 mapper。"""
        del ctx
        return _GoalMapper(
            prefer_matching_document=self.prefer_matching_document)

    def state_key(self):
        """返回 mapper 版本和方向模式。"""
        return 1, int(self.prefer_matching_document)


class _SupersedeChanged:
    """把依赖命中的未执行焦点替代，用于验证局部重算边界。"""

    def __init__(self) -> None:
        """记录被实际重算的 activation 身份。"""
        self.calls = []

    def recompute(self, activation, update):
        """只返回同一 activation 的低分 superseded 快照。"""
        self.calls.append(activation.identity_key())
        reason = AttractorScoreReason(
            update.reason,
            -9000,
            update.changed_dependencies,
        )
        return AttractorRecomputeDecision(
            -9000,
            (reason,),
            activation.dependencies,
            _PROTOCOLS[activation.request.source].superseded,
        )

    def clone_for_context(self, ctx):
        """为 V-06 返回独立调用记录。"""
        del ctx
        return _SupersedeChanged()

    def state_key(self):
        """返回局部重算策略版本。"""
        return (1,)


class _UnknownEvaluator:
    """记录 S-05 真正处理顺序，并始终返回 unknown。"""

    def __init__(self) -> None:
        """初始化空调用顺序。"""
        self.calls = []

    def evaluate(self, obligation):
        """返回与目标 source/scope 对齐的 unknown，拒绝显著性伪造真值。"""
        self.calls.append(obligation)
        return LogicEvaluation(
            obligation.proposition,
            _UNKNOWN,
            obligation.source,
            obligation.scope,
        )


class _EmptyRetriever:
    """不提供规则候选，使测试只观察目标 agenda 顺序。"""

    def retrieve(self, obligation):
        """返回空候选，不把 M-07 命中直接解释为推理规则。"""
        del obligation
        return ()


class _NeverVerifier:
    """无规则候选时不应调用的 verifier。"""

    def verify(self, candidate, premises):
        """拒绝任何意外逻辑验证调用。"""
        del candidate, premises
        raise AssertionError("空候选不应调用 verifier")


class _MarkProcessed:
    """把完整执行过的 S-05 结果标为 processed，不声称目标成立。"""

    def disposition(self, activation, result, protocol):
        """只表示 consumer 已处理，结果仍保留原始 unknown 四态。"""
        del activation, result
        return protocol.consumed

    def state_key(self):
        """返回消费策略版本。"""
        return (1,)


_PROTOCOLS: dict[SourceRef, AttractorProtocol] = {}


def _install_a10(
        ctx,
        source: SourceRef,
        *,
        prefer_matching_document: bool,
        max_agenda_entries: int = 2,
        max_consumptions: int = 4,
        ):
    """安装测试 A-10 runtime，并保存策略所需的注入协议引用。"""
    protocol = _attractor_protocol(source)
    _PROTOCOLS[source] = protocol
    strategy = _SupersedeChanged()
    runtime = install_attractor_runtime(
        ctx,
        protocol,
        AttractorBudget(max_agenda_entries, max_consumptions, 4),
        _GoalMapper(
            prefer_matching_document=prefer_matching_document),
        strategy,
    )
    return runtime, strategy


def _planner(source: SourceRef):
    """构造会记录真实目标处理顺序的 S-05 planner 和 A-10 consumer。"""
    evaluator = _UnknownEvaluator()
    planner = ReasoningPlanner(
        evaluator,
        _EmptyRetriever(),
        _NeverVerifier(),
        _reasoning_protocol(source),
    )
    consumer = ReasoningAgendaConsumer(
        planner,
        _PROTOCOLS[source],
        _instruction(source, 9150),
        _MarkProcessed(),
    )
    return evaluator, planner, consumer


def _setup(
        *,
        document_id: int = 1,
        prefer_matching_document: bool = False,
        max_consumptions: int = 4,
        ):
    """建立真实 M-06/M-07/A-10 测试上下文并打开当前 query。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    _seed_memory(ctx)
    source = _query_source(document_id=document_id)
    query_runtime, resolver_runtime = _install_resolver(
        ctx,
        _memory_protocol(source),
        _core_refs(ctx)[1],
    )
    del resolver_runtime
    attractor_runtime, strategy = _install_a10(
        ctx,
        source,
        prefer_matching_document=prefer_matching_document,
        max_consumptions=max_consumptions,
    )
    scope = _open_query(ctx, source)
    compilation = query_runtime.compile(
        _current(ctx, source, scope),
        access=MemoryAccessContext(1, 2, 3),
    )
    goals = _goals(source, scope)
    return (
        backend,
        ctx,
        source,
        attractor_runtime,
        strategy,
        compilation,
        goals,
    )


def test_real_m07_to_a10_to_s05_consumer_changes_processing_order_without_truth():
    """A-10 真消费改变 S-05 目标顺序，但高方向分不能把 unknown 变成支持。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, runtime, _, compilation, goals = setup
    try:
        evaluator, planner, consumer = _planner(source)
        planner.plan(goals[0], ReasoningBudget(1, 0, 0))
        assert evaluator.calls == [goals[0]]
        evaluator.calls.clear()
        before = backend.snapshot()

        state = runtime.resolve_and_activate(compilation, goals)
        result = runtime.consume_reasoning(
            consumer, ReasoningBudget(1, 0, 0))

        assert result is not None
        assert evaluator.calls == [goals[1]]
        assert result.result.state.status == STATE_UNKNOWN
        assert result.result.goal_satisfied is False
        assert state.processing_traces()[0].activation.obligation == goals[1]
        assert state.processing_traces()[0].decision.decision_trace_key == (
            result.result.stable_key())
        assert ctx.work_memory.pr_vector == {}
        assert backend.snapshot() == before
    finally:
        _close_query(ctx)
        backend.close()


def test_same_memory_different_current_query_changes_agenda_and_clears_old_state():
    """长期 Memory 不变时，当前 query document 改变可稳定改变首个 activation。"""
    setup = _setup(document_id=1, prefer_matching_document=True)
    backend, ctx, _, runtime, _, compilation, goals = setup
    try:
        first_state = runtime.resolve_and_activate(compilation, goals)
        first = first_state.next_activation()
        assert first is not None
        assert first.candidate.hypothesis.candidate_key == (1,)
        first_key = first_state.stable_key()
        _close_query(ctx)
        assert ctx.work_memory.attractor_state is None

        source = _query_source(document_id=2)
        scope = _open_query(ctx, source, local_id=2)
        changed = ctx.memory_query_runtime.compile(
            _current(ctx, source, scope, ordinal=1),
            access=MemoryAccessContext(1, 2, 3),
        )
        changed_goals = _goals(source, scope)
        second_state = runtime.resolve_and_activate(changed, changed_goals)
        second = second_state.next_activation()

        assert second is not None
        assert second.candidate.hypothesis.candidate_key == (2,)
        assert second_state.stable_key() != first_key
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_later_context_only_recomputes_matching_open_item_and_never_rewrites_consumed():
    """后文只替代依赖命中的旧焦点，已执行项在后续更新中保持不可变。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, runtime, strategy, compilation, goals = setup
    try:
        state = runtime.resolve_and_activate(compilation, goals)
        old_focus = state.next_activation()
        assert old_focus.candidate.hypothesis.candidate_key == (2,)
        old_dependency = old_focus.dependencies[0]
        clock = compilation.current.logical_timestamp.clock
        update = AttractorContextUpdate(
            state.scope,
            LogicalTimestamp(clock, 2),
            _instruction(source, 9160),
            (old_dependency,),
        )

        trace = runtime.apply_update(update)

        assert len(trace.recomputed) == 1
        assert trace.recomputed[0].before == old_focus
        assert trace.recomputed[0].after.status == state.protocol.superseded
        assert strategy.calls == [old_focus.identity_key()]
        remaining = state.next_activation()
        assert remaining.candidate.hypothesis.candidate_key == (1,)

        _, _, consumer = _planner(source)
        runtime.consume_reasoning(consumer, ReasoningBudget(1, 0, 0))
        consumed_key = remaining.identity_key()
        calls_before = tuple(strategy.calls)
        immutable_update = AttractorContextUpdate(
            state.scope,
            LogicalTimestamp(clock, 3),
            _instruction(source, 9161),
            (remaining.dependencies[0],),
        )
        immutable_trace = runtime.apply_update(immutable_update)

        assert immutable_trace.recomputed == ()
        assert immutable_trace.immutable_activation_keys == (consumed_key,)
        assert tuple(strategy.calls) == calls_before
        consumed = next(
            item for item in state.activations()
            if item.identity_key() == consumed_key)
        assert consumed.status == state.protocol.consumed
    finally:
        _close_query(ctx)
        backend.close()


def test_agenda_budget_records_dropped_projection_without_fabricating_candidate():
    """agenda 预算只截取 mapper 已提交项，并保留未进入项的身份边界。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _seed_memory(ctx)
        source = _query_source()
        query_runtime, _ = _install_resolver(
            ctx, _memory_protocol(source), _core_refs(ctx)[1])
        protocol = _attractor_protocol(source)
        _PROTOCOLS[source] = protocol
        runtime = install_attractor_runtime(
            ctx,
            protocol,
            AttractorBudget(1, 1, 1),
            _GoalMapper(prefer_matching_document=False),
            _SupersedeChanged(),
        )
        scope = _open_query(ctx, source)
        compilation = query_runtime.compile(
            _current(ctx, source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )
        state = runtime.resolve_and_activate(
            compilation, _goals(source, scope))

        assert state.considered_activation_count == 2
        assert len(state.activations()) == 1
        assert len(state.dropped_activation_keys) == 1
        selected_and_dropped = {
            state.activations()[0].identity_key(),
            state.dropped_activation_keys[0],
        }
        assert len(selected_and_dropped) == 2
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_consumption_budget_fails_before_second_planner_execution():
    """处理预算耗尽必须在调用 S-05 前失败，不能产生未登记执行。"""
    setup = _setup(
        prefer_matching_document=False,
        max_consumptions=1,
    )
    backend, ctx, source, runtime, _, compilation, goals = setup
    try:
        evaluator, _, consumer = _planner(source)
        runtime.resolve_and_activate(compilation, goals)
        runtime.consume_reasoning(consumer, ReasoningBudget(1, 0, 0))
        calls_after_first = tuple(evaluator.calls)

        with pytest.raises(RuntimeError, match="预算"):
            runtime.consume_reasoning(consumer, ReasoningBudget(1, 0, 0))

        assert tuple(evaluator.calls) == calls_after_first
        assert len(ctx.work_memory.require_attractor_state().processing_traces()) == 1
    finally:
        _close_query(ctx)
        backend.close()


def test_state_rejects_consumption_of_non_head_agenda_item():
    """公共状态机也必须拒绝绕过 frontier head 直接提交低优先级项。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, runtime, _, compilation, goals = setup
    try:
        state = runtime.resolve_and_activate(compilation, goals)
        frontier = state.frontier()
        assert len(frontier) == 2
        bypass = AttractorConsumptionDecision(
            frontier[1].identity_key(),
            _instruction(source, 9170),
            state.protocol.consumed,
            (9171,),
        )

        with pytest.raises(ValueError, match="frontier head"):
            state.commit_consumption(bypass)

        assert state.processing_traces() == ()
        assert state.next_activation() == frontier[0]
    finally:
        _close_query(ctx)
        backend.close()


def test_v06_clones_runtime_components_and_does_not_share_query_state():
    """评测 clone 重绑 A-10 组件，宿主和 clone 不共享 AttractorState。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _seed_memory(ctx)
        source = _query_source()
        _install_resolver(ctx, _memory_protocol(source), _core_refs(ctx)[1])
        runtime, _ = _install_a10(
            ctx, source, prefer_matching_document=True)
        before = backend.snapshot()

        with isolated_evaluation(ctx, label="a10-clone") as eval_ctx:
            cloned = eval_ctx.attractor_runtime
            assert cloned is not runtime
            assert cloned.mapper is not runtime.mapper
            assert cloned.recompute_strategy is not runtime.recompute_strategy
            assert cloned.state_key() == runtime.state_key()
            assert eval_ctx.work_memory.attractor_state is None

        assert ctx.work_memory.attractor_state is None
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_work_memory_rejects_cross_query_state_and_query_end_clears_it():
    """AttractorState 不能安装到其他 owner/query，当前 query 结束后必须清空。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, _, runtime, _, compilation, goals = setup
    other_backend = DictBackend()
    other_ctx = make_train_context(other_backend)
    try:
        state = runtime.resolve_and_activate(compilation, goals)
        other_source = _query_source(document_id=9)
        _open_query(other_ctx, other_source, local_id=9)
        with pytest.raises(ValueError, match="当前 query"):
            other_ctx.work_memory.install_attractor_state(state)

        _close_query(ctx)
        assert ctx.work_memory.attractor_state is None
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        if other_ctx.work_memory.active_query_scope is not None:
            _close_query(other_ctx)
        backend.close()
        other_backend.close()
