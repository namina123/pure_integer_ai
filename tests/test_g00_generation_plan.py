"""G-00 typed 六层生成计划、失败传播和防 PR 捷径测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationLayerResult,
    GenerationPlan,
    GenerationPlanProtocol,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    VISIBILITY_SESSION,
    OwnerScope,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningBudget,
    ReasoningObligation,
    ReasoningPlanResult,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    document_scope,
    make_scope,
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
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)


_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)


def _source(document_id: int = 1) -> SourceRef:
    """构造共享 owner/version 的测试来源。"""
    return SourceRef(
        9911, 9912, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _bound_world(count: int = 2):
    """构造若干来源化原子命题及其不可物化 bound view。"""
    source = _source()
    scope = query_scope(9, parent=document_scope(source))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((9913, index))
        for index in range(1, 10)
    ))
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (9914, index)),
            concept_identity((9915, index)),
            occurrence_identity(
                source, start=index, end=index + 1, ordinal=0),
            context_scope_identity(source, (9916, index)),
            (),
        )
        for index in range(1, count + 1)
    )
    templates = tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity((9917, index)),
        )
        for index, definition in enumerate(definitions, start=1)
    )
    graph = PropositionTemplateGraph(templates)
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((9918, 1)), failures))
    bound = tuple(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    return source, scope, bound


def _protocol(seed: int = 9920) -> GenerationPlanProtocol:
    """注入六层、三种结果和下游阻断原因身份。"""
    identities = tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 11)
    )
    return GenerationPlanProtocol(*identities)


def _hypothesis(source: SourceRef, scope, key: int) -> HypothesisKey:
    """构造与当前 query scope 对齐的 Evidence Hypothesis。"""
    return HypothesisKey(
        (9921, 1), (9921, key), (9921, 2), scope, source)


def _evidence(
        source: SourceRef,
        scope,
        evidence_id: int,
        stance: int = EVIDENCE_SUPPORT,
        ) -> EvidenceRecord:
    """构造一条来源化 Evidence 事件。"""
    return EvidenceRecord(
        evidence_id,
        _hypothesis(source, scope, evidence_id),
        stance,
        (9922, stance),
        source,
        evidence_id,
    )


def _candidate(
        bound,
        source: SourceRef,
        scope,
        evidence_id: int,
        *,
        state: LogicEvidenceState = _T,
        reasoning: ReasoningPlanResult | None = None,
        ) -> GenerationCandidate:
    """按目标四态构造支持、反驳或冲突 Evidence 候选。"""
    evidence = []
    if state.support:
        evidence.append(_evidence(source, scope, evidence_id, EVIDENCE_SUPPORT))
    if state.refute:
        evidence.append(_evidence(
            source, scope, evidence_id + 100, EVIDENCE_REFUTE))
    return GenerationCandidate(
        bound, state, source, scope, tuple(evidence), reasoning)


def _request(count: int = 1):
    """构造默认 support 回答目标和若干 typed 候选。"""
    source, scope, bound = _bound_world(max(2, count))
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((9923, 1)),
        bound[0],
        _T,
        source,
        scope,
    )
    candidates = tuple(
        _candidate(bound[index], source, scope, 100 + index)
        for index in range(count)
    )
    return GenerationPlanningRequest(goal, candidates), source, scope, bound


class _LayerResolver:
    """按注入结果返回当前层 decision，并记录实际调用顺序。"""

    def __init__(
            self,
            layer,
            outcome,
            reason,
            *,
            selected=(),
            payload=(),
            trace=(1,),
            ) -> None:
        self.layer = layer
        self.outcome = outcome
        self.reason = reason
        self.selected = selected
        self.payload = payload
        self.trace = trace
        self.calls: list[tuple] = []

    def resolve(self, request, prior):
        """返回固定 decision，测试 planner 的核验和失败传播。"""
        self.calls.append(tuple(item.layer for item in prior))
        return GenerationLayerDecision(
            self.layer,
            self.outcome,
            self.reason,
            self.selected,
            self.payload,
            self.trace,
        )


def _planner(
        protocol: GenerationPlanProtocol,
        *,
        failed_index: int | None = None,
        selected=(),
        reverse: bool = False,
        ):
    """构造六层 resolver 和可打乱注册顺序的 planner。"""
    resolvers = []
    registrations = []
    for index, layer in enumerate(protocol.layers()):
        outcome = (
            protocol.failed if failed_index == index else protocol.complete)
        resolver = _LayerResolver(
            layer,
            outcome,
            minimal_instruction_identity((9924, index + 1)),
            selected=selected if index == 1 else (),
            payload=(index + 1,),
            trace=(9925, index + 1),
        )
        resolvers.append(resolver)
        registrations.append(GenerationLayerRegistration(layer, resolver))
    if reverse:
        registrations.reverse()
    return GenerationPlanner(protocol, registrations), tuple(resolvers)


def test_g00_executes_all_six_layers_in_protocol_order() -> None:
    """注册输入顺序不影响六层执行顺序，完整计划保留采用和上游输入指纹。"""
    request, _, _, _ = _request()
    protocol = _protocol()
    selected = (request.candidate_keys()[0],)
    planner, resolvers = _planner(protocol, selected=selected, reverse=True)

    plan = planner.plan(request)

    assert plan.complete
    assert tuple(item.layer for item in plan.layers) == protocol.layers()
    assert plan.layers[1].selected_candidate_keys == selected
    assert all(len(resolver.calls) == 1 for resolver in resolvers)
    assert resolvers[0].calls == [()]
    assert resolvers[-1].calls[0] == protocol.layers()[:-1]
    assert len({item.input_key for item in plan.layers}) == 6


def test_g00_failure_blocks_all_downstream_resolvers() -> None:
    """首个 failed 层真实执行，后续层全部 blocked 且 resolver 零调用。"""
    request, _, _, _ = _request()
    protocol = _protocol(9930)
    planner, resolvers = _planner(protocol, failed_index=2)

    plan = planner.plan(request)

    assert not plan.complete
    assert tuple(item.outcome for item in plan.layers) == (
        protocol.complete,
        protocol.complete,
        protocol.failed,
        protocol.blocked,
        protocol.blocked,
        protocol.blocked,
    )
    assert tuple(item.executed for item in plan.layers) == (
        True, True, True, False, False, False)
    assert [len(resolver.calls) for resolver in resolvers] == [1, 1, 1, 0, 0, 0]
    assert all(item.reason == protocol.downstream_blocked
               for item in plan.layers[3:])


def test_g00_rejects_layer_or_candidate_substitution() -> None:
    """resolver 不能返回其他层，也不能采用请求之外的伪造候选键。"""
    request, _, _, _ = _request()
    protocol = _protocol(9940)
    planner, resolvers = _planner(protocol)
    resolvers[0].layer = protocol.content_layer
    with pytest.raises(ValueError, match="其他 layer"):
        planner.plan(request)

    planner, resolvers = _planner(protocol)
    resolvers[0].selected = ((9941, 1),)
    with pytest.raises(ValueError, match="请求之外"):
        planner.plan(request)


def test_g00_candidate_requires_evidence_state_and_reasoning_closure() -> None:
    """候选四态必须由所携 Evidence 重建，reasoning 引用也必须完整随候选进入。"""
    request, source, scope, bound = _request()
    support = request.candidates[0].evidence[0]
    with pytest.raises(ValueError, match="四态"):
        GenerationCandidate(
            bound[0], _F, source, scope, (support,))

    obligation = ReasoningObligation(bound[0], _T, source, scope)
    reasoning = ReasoningPlanResult(
        obligation,
        _T,
        True,
        minimal_instruction_identity((9942, 1)),
        ReasoningBudget(1, 0, 0),
        1,
        0,
        evidence_ids=(999,),
        hypotheses=(_hypothesis(source, scope, 999),),
    )
    with pytest.raises(ValueError, match="全部 Evidence"):
        _candidate(bound[0], source, scope, 200, reasoning=reasoning)

    old_support = _evidence(source, scope, 301, EVIDENCE_SUPPORT)
    new_refute = EvidenceRecord(
        302,
        old_support.hypothesis,
        EVIDENCE_REFUTE,
        (9942, 2),
        source,
        302,
        supersedes_evidence_id=301,
    )
    with pytest.raises(ValueError, match="已被替代"):
        GenerationCandidate(
            bound[0], LogicEvidenceState(True, True), source, scope,
            (old_support, new_refute),
        )


def test_g00_request_is_candidate_order_independent() -> None:
    """候选集合按完整稳定键规范化，输入排列不改变请求或计划键。"""
    request, source, scope, bound = _request(2)
    reversed_request = GenerationPlanningRequest(
        request.goal, tuple(reversed(request.candidates)))
    protocol = _protocol(9950)
    planner, _ = _planner(protocol)

    assert request.stable_key() == reversed_request.stable_key()
    assert planner.plan(request).stable_key() == planner.plan(
        reversed_request).stable_key()

    other_scope = query_scope(10, parent=document_scope(source))
    with pytest.raises(ValueError, match="当前 query scope"):
        GenerationPlanningRequest(
            request.goal,
            (_candidate(bound[0], source, other_scope, 300),),
        )


def test_g00_keeps_source_provenance_separate_from_evaluation_query_scope():
    """评测 query 可使用独立 owner，但命题和 Evidence 仍须绑定原 SourceRef。"""
    _, source, source_scope, bound = _request()
    evaluation_scope = make_scope(
        SCOPE_QUERY,
        71,
        owner=OwnerScope(1, 1, 71, VISIBILITY_SESSION),
        versions=source.versions,
    )
    support = _evidence(source, source_scope, 701, EVIDENCE_SUPPORT)
    candidate = GenerationCandidate(
        bound[0], _T, source, evaluation_scope, (support,))
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((9949, 1)),
        bound[0],
        _T,
        source,
        evaluation_scope,
    )

    request = GenerationPlanningRequest(goal, (candidate,))

    assert request.goal.scope == evaluation_scope
    assert request.candidates[0].evidence[0].hypothesis.scope == source_scope
    other_source = _source(72)
    bad_evidence = _evidence(
        other_source,
        document_scope(other_source),
        702,
        EVIDENCE_SUPPORT,
    )
    with pytest.raises(ValueError, match="跨来源"):
        GenerationCandidate(
            bound[0], _T, source, evaluation_scope, (bad_evidence,))


def test_g00_accepts_only_exact_aggregate_candidate_provenance() -> None:
    """aggregate H-00 Evidence 必须由完整候选定义精确指回当前 Proposition。"""
    _, source, _source_scope, bound = _request()
    aggregate_source = _source(73)
    aggregate_scope = document_scope(aggregate_source)
    definition = EvidenceCandidateDefinition(
        bound[0].template,
        (9950, 1),
        (CandidateBinding(
            concept_identity((9950, 2)),
            bound[0].predicate,
        ),),
        (source,),
    )
    hypothesis = HypothesisKey(
        (9950, 3),
        definition.stable_key(),
        definition.competition_key,
        aggregate_scope,
        aggregate_source,
    )
    evidence = EvidenceRecord(
        703,
        hypothesis,
        EVIDENCE_SUPPORT,
        (9950, 4),
        _source(74),
        1,
    )
    candidate = GenerationCandidate(
        bound[0], _T, source, query_scope(2), (evidence,))

    assert candidate.evidence == (evidence,)
    assert candidate.evidence[0].hypothesis.observation == aggregate_source

    other = EvidenceCandidateDefinition(
        bound[1].template,
        definition.competition_key,
        definition.bindings,
        definition.forming_sources,
    )
    wrong = EvidenceRecord(
        704,
        HypothesisKey(
            hypothesis.hypothesis_kind,
            other.stable_key(),
            hypothesis.competition_key,
            hypothesis.scope,
            hypothesis.observation,
        ),
        EVIDENCE_SUPPORT,
        (9950, 5),
        _source(75),
        2,
    )
    with pytest.raises(ValueError, match="其他 Proposition"):
        GenerationCandidate(
            bound[0], _T, source, query_scope(3), (wrong,))


def test_g00_requires_exactly_six_registered_layers() -> None:
    """缺层、重复层和协议身份重用均在执行前失败。"""
    protocol = _protocol(9960)
    planner, resolvers = _planner(protocol)
    del planner
    registrations = tuple(
        GenerationLayerRegistration(layer, resolver)
        for layer, resolver in zip(protocol.layers(), resolvers)
    )
    with pytest.raises(ValueError, match="六层"):
        GenerationPlanner(protocol, registrations[:-1])
    with pytest.raises(ValueError, match="重复注册"):
        GenerationPlanner(protocol, registrations + (registrations[0],))
    identities = tuple(
        minimal_instruction_identity((9961, index))
        for index in range(1, 10)
    )
    with pytest.raises(ValueError, match="互不相同"):
        GenerationPlanProtocol(*identities, identities[-1])


def test_g00_rejects_naked_inputs_and_tampered_blocked_plan() -> None:
    """planner 不接受 PR/路径替身，计划对象也拒绝失败后伪执行下游层。"""
    request, _, _, _ = _request()
    protocol = _protocol(9970)
    planner, _ = _planner(protocol, failed_index=0)
    with pytest.raises(TypeError, match="GenerationPlanningRequest"):
        planner.plan(object())

    plan = planner.plan(request)
    tampered = list(plan.layers)
    blocked = tampered[1]
    tampered[1] = GenerationLayerResult(
        blocked.layer,
        protocol.blocked,
        blocked.reason,
        True,
        blocked.input_key,
        trace=blocked.trace,
    )
    with pytest.raises(ValueError, match="blocked"):
        GenerationPlan(request, protocol, tuple(tampered))
