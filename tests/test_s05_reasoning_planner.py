"""S-05 多步目标分解、竞争分支、预算和 trace 测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.process.reasoning_candidate_adapter import (
    DagPathCandidateProvider,
    PRCandidateProvider,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicAtomEvidence,
    LogicEvaluation,
    LogicEvidenceState,
    LogicExecutor,
    LogicFailureProtocol,
    LogicOperatorRegistry,
    MappingAtomEvidenceResolver,
    STATE_CONFLICTED,
    STATE_PROVISIONAL,
    STATE_REFUTED,
    STATE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.reasoning_planner import (
    CompositeCandidateRetriever,
    LogicObligationEvaluator,
    ReasoningBudget,
    ReasoningCandidate,
    ReasoningObligation,
    ReasoningPlanner,
    ReasoningTerminationProtocol,
    RuleVerification,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    ExactTypeCompatibilityResolver,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.shared.types import (
    PathResult,
    TERMINAL_REACHED_SINK,
)
from pure_integer_ai.storage.backend import DictBackend


_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)
_U = LogicEvidenceState(False, False)
_B = LogicEvidenceState(True, True)


def _source(document_id: int = 1) -> SourceRef:
    """构造测试来源，document_id 用于制造来源漂移反例。"""
    return SourceRef(
        9801, 9802, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _binding_failures(seed: int = 9810) -> BindingFailureProtocol:
    """注入九个互异 S-03 failure reason。"""
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _logic_failures(seed: int = 9820) -> LogicFailureProtocol:
    """注入九个互异 S-04 failure reason。"""
    return LogicFailureProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 10)
    ))


def _planning_protocol(seed: int = 9830) -> ReasoningTerminationProtocol:
    """注入 S-05 evaluate 指令和五个互异终止原因。"""
    return ReasoningTerminationProtocol(*tuple(
        minimal_instruction_identity((seed, ordinal))
        for ordinal in range(1, 7)
    ))


def _hypothesis(
        source: SourceRef,
        scope,
        key: int,
        ) -> HypothesisKey:
    """构造与当前 source/scope 对齐的候选或证据 Hypothesis。"""
    return HypothesisKey(
        (9840, 1), (9840, key), (9840, 2), scope, source)


def _world(count: int = 6):
    """构造若干无真值原子 template 及其 S-03 bound views。"""
    source = _source()
    scope = document_scope(source)
    failures = _binding_failures()
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (9850, key)),
            concept_identity((9851, key)),
            occurrence_identity(
                source, start=key, end=key + 1, ordinal=0),
            context_scope_identity(source, (9852, key)),
            (),
        )
        for key in range(1, count + 1)
    )
    templates = tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity((9853, key)),
        )
        for key, definition in enumerate(definitions, start=1)
    )
    graph = PropositionTemplateGraph(templates)
    protocol = SubstitutionProtocol(
        minimal_instruction_identity((9854, 1)), failures)
    substituter = PropositionSubstituter(protocol)
    bound = tuple(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    return source, scope, failures, graph, protocol, bound


class _MappingEvaluator:
    """按 Proposition template 注入直接四态 Evidence。"""

    def __init__(self, states) -> None:
        self._states = states
        self.calls: list[ObjectIdentity] = []

    def evaluate(self, obligation):
        """为非 unknown 直接状态附带独立 Evidence 和 Hypothesis。"""
        self.calls.append(obligation.proposition.template)
        state, evidence_id = self._states.get(
            obligation.proposition.template, (_U, 0))
        evidence_ids = ()
        hypotheses = ()
        if state.status != STATE_UNKNOWN:
            evidence_ids = (
                (evidence_id, evidence_id + 1000)
                if state.status == STATE_CONFLICTED
                else (evidence_id,)
            )
            hypotheses = (_hypothesis(
                obligation.source, obligation.scope, evidence_id),)
        return LogicEvaluation(
            proposition=obligation.proposition,
            state=state,
            source=obligation.source,
            scope=obligation.scope,
            evidence_ids=evidence_ids,
            hypotheses=hypotheses,
        )


class _MappingRetriever:
    """按完整目标 template 返回全部注入竞争候选。"""

    def __init__(self, candidates) -> None:
        self._candidates = candidates
        self.calls: list[ObjectIdentity] = []

    def retrieve(self, obligation):
        """不做 rank 选优，直接返回当前目标的完整候选 tuple。"""
        self.calls.append(obligation.proposition.template)
        return self._candidates.get(obligation.proposition.template, ())


class _MappingVerifier:
    """按 candidate Hypothesis 注入受限规则验证结果。"""

    def __init__(self, states) -> None:
        self._states = states
        self.calls: list[HypothesisKey] = []

    def verify(self, candidate, premises):
        """仅在全部所需 premise 已满足时返回注入状态，否则 unknown。"""
        self.calls.append(candidate.hypothesis)
        state, evidence_id = self._states[candidate.hypothesis]
        if any(not result.goal_satisfied for result in premises):
            state, evidence_id = _U, 0
        return RuleVerification(
            candidate.hypothesis,
            state,
            candidate.conclusion.source,
            candidate.conclusion.scope,
            () if state.status == STATE_UNKNOWN else (evidence_id,),
            () if state.status == STATE_UNKNOWN else (
                _hypothesis(
                    candidate.conclusion.source,
                    candidate.conclusion.scope,
                    evidence_id,
                ),
            ),
        )


class _NeverVerifier:
    """无候选测试使用的 verifier，意外调用即失败。"""

    def verify(self, candidate, premises):
        """拒绝无候选路径上的意外规则验证。"""
        del candidate, premises
        raise AssertionError("不应调用 verifier")


class _Provider:
    """CompositeCandidateRetriever 测试使用的固定候选 provider。"""

    def __init__(self, candidates) -> None:
        self._candidates = candidates

    def retrieve(self, obligation):
        """返回固定候选，组合器负责目标一致性与去重。"""
        del obligation
        return self._candidates


def _obligation(bound, source, scope, required=_T):
    """构造默认要求 support 的 typed obligation。"""
    return ReasoningObligation(bound, required, source, scope)


def _candidate(
        conclusion,
        premises,
        key: int,
        *,
        assumption: bool = False,
        ) -> ReasoningCandidate:
    """构造带规则、指令、Hypothesis 和 Evidence 的候选应用。"""
    hypothesis = _hypothesis(conclusion.source, conclusion.scope, key)
    assumptions = (
        (_hypothesis(conclusion.source, conclusion.scope, key + 500),)
        if assumption else ()
    )
    return ReasoningCandidate(
        conclusion,
        tuple(premises),
        structure_concept_identity((9860, key)),
        minimal_instruction_identity((9861, key)),
        hypothesis,
        (key + 100,),
        assumptions,
    )


def _planner(evaluator, retriever, verifier, protocol=None):
    """构造不持有 query activation 的测试 planner。"""
    return ReasoningPlanner(
        evaluator,
        retriever,
        verifier,
        protocol or _planning_protocol(),
    )


def test_budget_and_obligation_reject_invalid_shapes():
    """预算只收严格整数，obligation 不能要求空证据方向。"""
    source, scope, _, _, _, bound = _world(1)
    with pytest.raises(ValueError):
        ReasoningBudget(0, 1, 1)
    with pytest.raises(ValueError):
        ReasoningBudget(1, -1, 1)
    with pytest.raises(ValueError):
        ReasoningBudget(1, 1, -1)
    with pytest.raises(ValueError):
        ReasoningBudget(True, 1, 1)
    with pytest.raises(ValueError):
        _obligation(bound[0], source, scope, _U)


def test_two_step_unseen_chain_combines_rules_with_source_bearing_trace():
    """合成未见命题可经 leaf→mid→goal 两次规则验证形成支持。"""
    source, scope, _, _, _, bound = _world(3)
    goal = _obligation(bound[0], source, scope)
    middle = _obligation(bound[1], source, scope)
    leaf = _obligation(bound[2], source, scope)
    middle_candidate = _candidate(middle, (leaf,), 1, assumption=True)
    goal_candidate = _candidate(goal, (middle,), 2)
    evaluator = _MappingEvaluator({leaf.proposition.template: (_T, 11)})
    retriever = _MappingRetriever({
        goal.proposition.template: (goal_candidate,),
        middle.proposition.template: (middle_candidate,),
    })
    verifier = _MappingVerifier({
        middle_candidate.hypothesis: (_T, 21),
        goal_candidate.hypothesis: (_T, 22),
    })

    result = _planner(evaluator, retriever, verifier).plan(
        goal, ReasoningBudget(10, 10, 5))

    assert result.state == _T
    assert result.goal_satisfied is True
    assert result.complete is True
    assert result.unresolved == ()
    assert result.evaluations_used == 3
    assert result.expansions_used == 2
    assert [step.ordinal for step in result.steps] == [1, 2, 3, 4, 5]
    rule_steps = [step for step in result.steps if step.rule is not None]
    assert [step.rule for step in rule_steps] == [
        middle_candidate.rule,
        goal_candidate.rule,
    ]
    assert all(step.source == source and step.scope == scope
               for step in result.steps)
    assert all(step.logic_evaluation is not None
               for step in result.steps if step.rule is None)
    assert all(step.logic_evaluation is None for step in rule_steps)
    assert middle_candidate.hypothesis in result.hypotheses
    assert middle_candidate.assumptions[0] in result.hypotheses
    assert goal_candidate.hypothesis in result.hypotheses
    assert {11, 21, 22, 101, 102} <= set(result.evidence_ids)


def test_competing_support_and_refute_candidates_form_conflicted_result():
    """Planner 必须验证全部竞争分支，不能选首个支持后遗忘反驳。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    support = _candidate(goal, (), 10)
    refute = _candidate(goal, (), 11)
    verifier = _MappingVerifier({
        support.hypothesis: (_T, 31),
        refute.hypothesis: (_F, 32),
    })
    result = _planner(
        _MappingEvaluator({}),
        _MappingRetriever({goal.proposition.template: (support, refute)}),
        verifier,
    ).plan(goal, ReasoningBudget(5, 5, 2))

    assert result.state == _B
    assert result.goal_satisfied is True
    assert len(result.branches) == 2
    assert {branch.state for branch in result.branches} == {_T, _F}
    assert set(verifier.calls) == {support.hypothesis, refute.hypothesis}
    assert support.hypothesis in result.hypotheses
    assert refute.hypothesis in result.hypotheses


def test_direct_support_does_not_hide_derived_refutation():
    """直接 Evidence 已支持时仍探索候选反证，最终保留 conflicted。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    refute = _candidate(goal, (), 20)
    result = _planner(
        _MappingEvaluator({goal.proposition.template: (_T, 41)}),
        _MappingRetriever({goal.proposition.template: (refute,)}),
        _MappingVerifier({refute.hypothesis: (_F, 42)}),
    ).plan(goal, ReasoningBudget(5, 5, 2))

    assert result.state == _B
    assert result.evidence_ids == tuple(sorted((41, 120, 42)))


def test_budget_exhaustion_before_competition_closes_returns_unknown():
    """只展开到中间目标时不能沿已见支持猜结论，根结果必须 unknown。"""
    source, scope, _, _, _, bound = _world(3)
    goal = _obligation(bound[0], source, scope)
    middle = _obligation(bound[1], source, scope)
    leaf = _obligation(bound[2], source, scope)
    goal_candidate = _candidate(goal, (middle,), 30)
    middle_candidate = _candidate(middle, (leaf,), 31)
    protocol = _planning_protocol()
    verifier = _MappingVerifier({
        goal_candidate.hypothesis: (_T, 51),
        middle_candidate.hypothesis: (_T, 52),
    })
    result = _planner(
        _MappingEvaluator({leaf.proposition.template: (_T, 53)}),
        _MappingRetriever({
            goal.proposition.template: (goal_candidate,),
            middle.proposition.template: (middle_candidate,),
        }),
        verifier,
        protocol,
    ).plan(goal, ReasoningBudget(10, 1, 5))

    assert result.state == _U
    assert result.complete is False
    assert result.termination == protocol.budget_exhausted
    assert goal in result.unresolved
    assert middle in result.unresolved
    assert result.expansions_used == 1
    assert verifier.calls == []
    assert goal_candidate.hypothesis in result.hypotheses
    assert goal_candidate.evidence_ids[0] in result.evidence_ids


def test_evaluation_budget_exhaustion_returns_unknown_without_verification():
    """子目标尚未直接求值就耗尽 evaluation budget 时不得调用 verifier。"""
    source, scope, _, _, _, bound = _world(2)
    goal = _obligation(bound[0], source, scope)
    premise = _obligation(bound[1], source, scope)
    candidate = _candidate(goal, (premise,), 40)
    protocol = _planning_protocol()
    verifier = _MappingVerifier({candidate.hypothesis: (_T, 61)})
    result = _planner(
        _MappingEvaluator({premise.proposition.template: (_T, 62)}),
        _MappingRetriever({goal.proposition.template: (candidate,)}),
        verifier,
        protocol,
    ).plan(goal, ReasoningBudget(1, 2, 2))

    assert result.state == _U
    assert result.termination == protocol.budget_exhausted
    assert result.evaluations_used == 1
    assert verifier.calls == []


def test_depth_budget_blocks_expansion_but_keeps_direct_trace():
    """max_depth=0 只允许根直接求值，存在候选时按预算未闭合处理。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    candidate = _candidate(goal, (), 50)
    protocol = _planning_protocol()
    result = _planner(
        _MappingEvaluator({goal.proposition.template: (_T, 71)}),
        _MappingRetriever({goal.proposition.template: (candidate,)}),
        _MappingVerifier({candidate.hypothesis: (_T, 72)}),
        protocol,
    ).plan(goal, ReasoningBudget(2, 2, 0))

    assert result.state == _U
    assert result.complete is False
    assert result.termination == protocol.budget_exhausted
    assert len(result.steps) == 1
    assert result.steps[0].state == _T


def test_cycle_is_local_unknown_and_trace_records_active_stack():
    """A→B→A 循环返回结构化 unknown，不造成递归溢出或跨运行污染。"""
    source, scope, _, _, _, bound = _world(2)
    first = _obligation(bound[0], source, scope)
    second = _obligation(bound[1], source, scope)
    first_candidate = _candidate(first, (second,), 60)
    second_candidate = _candidate(second, (first,), 61)
    protocol = _planning_protocol()
    planner = _planner(
        _MappingEvaluator({}),
        _MappingRetriever({
            first.proposition.template: (first_candidate,),
            second.proposition.template: (second_candidate,),
        }),
        _MappingVerifier({
            first_candidate.hypothesis: (_T, 81),
            second_candidate.hypothesis: (_T, 82),
        }),
        protocol,
    )

    first_run = planner.plan(first, ReasoningBudget(10, 10, 5))
    second_run = planner.plan(first, ReasoningBudget(10, 10, 5))

    assert first_run.state == _U
    assert first_run.complete is True
    assert first in first_run.unresolved
    assert second in first_run.unresolved
    cycle_steps = [
        step for step in first_run.steps
        if step.instruction == protocol.cycle
    ]
    assert len(cycle_steps) == 1
    assert len(cycle_steps[0].stack_keys) == 2
    assert first_run.stable_key() == second_run.stable_key()


def test_path_candidate_reaching_sink_is_not_success_without_verification():
    """零前提候选可代表已到 sink 的路径，但 verifier unknown 时目标仍 unknown。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    path_candidate = _candidate(goal, (), 70)
    protocol = _planning_protocol()
    result = _planner(
        _MappingEvaluator({}),
        _MappingRetriever({goal.proposition.template: (path_candidate,)}),
        _MappingVerifier({path_candidate.hypothesis: (_U, 0)}),
        protocol,
    ).plan(goal, ReasoningBudget(2, 2, 1))

    assert result.state == _U
    assert result.goal_satisfied is False
    assert result.termination == protocol.unresolved
    assert result.branches[0].termination == protocol.verifier_unknown
    assert path_candidate.evidence_ids[0] in result.evidence_ids


def test_refute_obligation_uses_typed_required_evidence_direction():
    """同一 refuted 状态满足反驳目标而不满足支持目标。"""
    source, scope, _, _, _, bound = _world(1)
    support_goal = _obligation(bound[0], source, scope, _T)
    refute_goal = _obligation(bound[0], source, scope, _F)
    evaluator = _MappingEvaluator({bound[0].template: (_F, 91)})
    empty = _MappingRetriever({})

    support = _planner(evaluator, empty, _NeverVerifier()).plan(
        support_goal, ReasoningBudget(2, 0, 0))
    refute = _planner(evaluator, empty, _NeverVerifier()).plan(
        refute_goal, ReasoningBudget(2, 0, 0))

    assert support.state == _F
    assert support.goal_satisfied is False
    assert refute.state == _F
    assert refute.goal_satisfied is True


def test_cross_source_premise_keeps_its_own_source_and_scope_in_trace():
    """跨文档规则可引用其他来源前提，trace 不把前提归属改写成结论来源。"""
    source, scope, _, _, _, bound = _world(1)
    other_source = _source(2)
    other_scope = document_scope(other_source)
    other_failures = _binding_failures(9870)
    other_definition = AtomicPropositionDefinition(
        proposition_identity(other_source, (9871, 1)),
        concept_identity((9871, 2)),
        occurrence_identity(
            other_source, start=1, end=2, ordinal=0),
        context_scope_identity(other_source, (9871, 3)),
        (),
    )
    other_graph = PropositionTemplateGraph((ScopedPropositionTemplate(
        other_definition,
        structure_concept_identity((9871, 4)),
    ),))
    other_bound = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((9871, 5)), other_failures,
    )).substitute(
        other_definition.proposition,
        other_graph,
        BindingEnvironment(),
    )
    goal = _obligation(bound[0], source, scope)
    premise = _obligation(other_bound, other_source, other_scope)
    candidate = _candidate(goal, (premise,), 75)
    result = _planner(
        _MappingEvaluator({premise.proposition.template: (_T, 95)}),
        _MappingRetriever({goal.proposition.template: (candidate,)}),
        _MappingVerifier({candidate.hypothesis: (_T, 96)}),
    ).plan(goal, ReasoningBudget(4, 2, 2))

    assert result.state == _T
    premise_steps = [
        step for step in result.steps
        if step.obligation == premise
    ]
    assert len(premise_steps) == 1
    assert premise_steps[0].source == other_source
    assert premise_steps[0].scope == other_scope
    assert result.steps[-1].source == source
    assert result.steps[-1].scope == scope


def test_composite_retriever_deduplicates_without_rank_selection():
    """不同路径设施返回同一候选时只去重，不按 provider 顺序或 rank 选义。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    first = _candidate(goal, (), 80)
    second = _candidate(goal, (), 81)
    retriever = CompositeCandidateRetriever((
        _Provider((first, second)),
        _Provider((second, first)),
    ))

    candidates = retriever.retrieve(goal)
    assert set(candidates) == {first, second}
    assert len(candidates) == 2
    assert candidates == tuple(sorted(
        candidates, key=lambda item: item.stable_key()))


def test_dag_path_and_pr_adapters_only_submit_typed_candidates():
    """sink 与 PR snapshot 经 mapper 只形成候选，不在 adapter 内产生逻辑结论。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    path_candidate = _candidate(goal, (), 85)
    pr_candidate = _candidate(goal, (), 86)
    path_result = PathResult(
        terminal=TERMINAL_REACHED_SINK,
        sink=(1, 2),
    )

    class _PathMapper:
        """测试 mapper 显式接收 PathResult 后返回 typed candidate。"""

        def map_candidates(self, obligation, result):
            """确认 adapter 未改写 terminal/sink。"""
            assert obligation == goal
            assert result is path_result
            assert result.terminal == TERMINAL_REACHED_SINK
            return (path_candidate,)

    class _PRSource:
        """模拟 A3PRWrapper 的 query-scoped snapshot 接口。"""

        def __init__(self):
            self.values = {(1, 2): (3, 4)}

        def snapshot(self):
            """返回 owner 快照；adapter 必须复制后再交 mapper。"""
            return self.values

    pr_source = _PRSource()

    class _PRMapper:
        """测试 mapper 只把启发式节点映射成 typed candidate。"""

        def map_candidates(self, obligation, snapshot):
            """修改副本以验证 PR owner 状态不被 mapper 污染。"""
            assert obligation == goal
            assert snapshot == {(1, 2): (3, 4)}
            snapshot.clear()
            return (pr_candidate,)

    combined = CompositeCandidateRetriever((
        DagPathCandidateProvider(path_result, _PathMapper()),
        PRCandidateProvider(pr_source, _PRMapper()),
    )).retrieve(goal)

    assert set(combined) == {path_candidate, pr_candidate}
    assert pr_source.values == {(1, 2): (3, 4)}
    assert all(isinstance(item, ReasoningCandidate) for item in combined)


def test_candidate_for_other_goal_and_evaluator_scope_drift_are_rejected():
    """候选结论或直接求值 scope 漂移必须在规划边界失败。"""
    source, scope, _, _, _, bound = _world(2)
    goal = _obligation(bound[0], source, scope)
    other = _obligation(bound[1], source, scope)
    wrong = _candidate(other, (), 90)
    planner = _planner(
        _MappingEvaluator({}),
        _MappingRetriever({goal.proposition.template: (wrong,)}),
        _NeverVerifier(),
    )
    with pytest.raises(ValueError, match="conclusion"):
        planner.plan(goal, ReasoningBudget(2, 2, 1))

    other_scope = query_scope(2, parent=scope)

    class _DriftEvaluator:
        """固定返回其他 query scope，用于验证直接求值边界。"""

        def evaluate(self, obligation):
            """返回命题相同但 scope 错误的结果。"""
            return LogicEvaluation(
                obligation.proposition, _T, source, other_scope,
                evidence_ids=(1,),
            )

    with pytest.raises(ValueError, match="source/scope"):
        _planner(
            _DriftEvaluator(), _MappingRetriever({}), _NeverVerifier(),
        ).plan(goal, ReasoningBudget(2, 0, 0))


def test_logic_obligation_evaluator_calls_real_s04_without_backend_writes():
    """S-05 adapter 经真实 S-04 原子 Evidence 求值，规划前后 Backend 不变。"""
    source, scope, failures, graph, protocol, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    hypothesis = _hypothesis(source, scope, 100)
    evidence = LogicAtomEvidence(
        bound[0].template,
        _T,
        source,
        scope,
        hypothesis,
        (101,),
    )
    executor = LogicExecutor(
        LogicOperatorRegistry(()),
        MappingAtomEvidenceResolver((evidence,)),
        _logic_failures(),
        protocol,
        ExactTypeCompatibilityResolver(),
        failures,
    )
    evaluator = LogicObligationEvaluator(
        executor, graph, BindingEnvironment())
    backend = DictBackend()
    try:
        before = backend.snapshot()
        result = _planner(
            evaluator, _MappingRetriever({}), _NeverVerifier(),
        ).plan(goal, ReasoningBudget(2, 0, 0))
        assert result.state == _T
        assert result.evidence_ids == (101,)
        assert result.hypotheses == (hypothesis,)
        assert result.steps[0].logic_evaluation is not None
        assert result.steps[0].logic_evaluation.evidence_ids == (101,)
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_rule_verification_source_drift_is_rejected():
    """InferenceVerifier 不能把其他 source 的规则结果注入当前分支。"""
    source, scope, _, _, _, bound = _world(1)
    goal = _obligation(bound[0], source, scope)
    candidate = _candidate(goal, (), 110)
    other_source = _source(2)
    other_scope = document_scope(other_source)

    class _DriftVerifier:
        """返回其他来源但内部自洽的 RuleVerification。"""

        def verify(self, candidate, premises):
            """忽略前提并制造 source 漂移。"""
            del premises
            return RuleVerification(
                candidate.hypothesis,
                _T,
                other_source,
                other_scope,
                (112,),
            )

    with pytest.raises(ValueError, match="source"):
        _planner(
            _MappingEvaluator({}),
            _MappingRetriever({goal.proposition.template: (candidate,)}),
            _DriftVerifier(),
        ).plan(goal, ReasoningBudget(2, 2, 1))
