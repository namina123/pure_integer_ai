"""G-01 回答立场、内容选择、Artifact 边界和双层交叉核验测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactAuthority,
    ArtifactInvocation,
    ArtifactSchema,
    FormalArtifact,
    FormalArtifactDefinition,
    artifact_identity,
)
from pure_integer_ai.cognition.shared.formal_artifact_bridge import (
    ArtifactExecutionObservation,
    ArtifactInvocationResult,
    ArtifactVerificationObservation,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
    AnswerContentSelector,
    ContentArtifactAttachment,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationPlanProtocol,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
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
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)


_T = LogicEvidenceState(True, False)
_F = LogicEvidenceState(False, True)
_U = LogicEvidenceState(False, False)


def _world(count: int = 2):
    """构造来源、query scope 和若干不可物化 bound proposition。"""
    source = SourceRef(
        9981, 9982, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = query_scope(1, parent=document_scope(source))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((9983, index))
        for index in range(1, 10)
    ))
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (9984, index)),
            concept_identity((9985, index)),
            occurrence_identity(
                source, start=index, end=index + 1, ordinal=0),
            context_scope_identity(source, (9986, index)),
            (),
        )
        for index in range(1, count + 1)
    )
    graph = PropositionTemplateGraph(tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity((9987, index)),
        )
        for index, definition in enumerate(definitions, start=1)
    ))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((9988, 1)), failures))
    bound = tuple(
        substituter.substitute(
            definition.proposition, graph, BindingEnvironment())
        for definition in definitions
    )
    return source, scope, bound


def _plan_protocol(seed: int = 9990) -> GenerationPlanProtocol:
    """注入 G-00 六层和结果身份。"""
    return GenerationPlanProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 11)
    ))


def _content_protocol(seed: int = 10000) -> AnswerContentProtocol:
    """注入五种回答立场身份。"""
    return AnswerContentProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 6)
    ))


def _evidence(source, scope, evidence_id: int, stance: int) -> EvidenceRecord:
    """构造一条与 query scope 对齐的 Evidence。"""
    hypothesis = HypothesisKey(
        (10010, 1),
        (10010, evidence_id),
        (10010, 2),
        scope,
        source,
    )
    return EvidenceRecord(
        evidence_id,
        hypothesis,
        stance,
        (10011, stance),
        source,
        evidence_id,
    )


def _candidate(bound, source, scope, evidence_id: int, state) -> GenerationCandidate:
    """按目标四态生成支持、反驳、unknown 或 conflicted 候选。"""
    evidence = []
    if state.support:
        evidence.append(_evidence(
            source, scope, evidence_id, EVIDENCE_SUPPORT))
    if state.refute:
        evidence.append(_evidence(
            source, scope, evidence_id + 100, EVIDENCE_REFUTE))
    if not evidence:
        evidence.append(_evidence(
            source, scope, evidence_id, EVIDENCE_UNKNOWN))
    return GenerationCandidate(
        bound, state, source, scope, tuple(evidence))


def _request(states=(_T,), proposition_indexes=None):
    """构造 support 目标和调用方指定的候选四态/命题分布。"""
    source, scope, bound = _world(max(2, len(states)))
    if proposition_indexes is None:
        proposition_indexes = tuple(range(len(states)))
    candidates = tuple(
        _candidate(
            bound[prop_index], source, scope, 200 + index * 10, state)
        for index, (state, prop_index) in enumerate(
            zip(states, proposition_indexes))
    )
    goal = AnswerGenerationGoal(
        minimal_instruction_identity((10012, 1)),
        bound[0],
        _T,
        source,
        scope,
    )
    return GenerationPlanningRequest(goal, candidates), source, scope, bound


def _artifact(
        candidate: GenerationCandidate,
        *,
        accepted: bool | None = True,
        ) -> ArtifactInvocationResult:
    """构造无参数形式调用，独立保留 execution/verifier 和语言 unknown。"""
    source = candidate.source
    scope = candidate.scope
    value_type = concept_identity((10020, 1))
    proof_type = concept_identity((10020, 2))
    program_type = concept_identity((10020, 3))
    unit = concept_identity((10021, 1))
    program_kind = concept_identity((10022, 1))
    value_kind = concept_identity((10022, 2))
    proof_kind = concept_identity((10022, 3))
    value_schema = ArtifactSchema(value_type, unit)
    proof_schema = ArtifactSchema(proof_type, unit)
    program_schema = ArtifactSchema(program_type, unit)
    executor = ArtifactAuthority(
        concept_identity((10023, 1)), concept_identity((10023, 2)))
    verifier = ArtifactAuthority(
        concept_identity((10024, 1)), concept_identity((10024, 2)))

    def artifact(kind, schema, declaration, payload, artifact_scope):
        """构造完整 identity 与载荷一致的测试 Artifact。"""
        return FormalArtifact(
            artifact_identity(
                source, kind, schema, declaration, payload, artifact_scope),
            kind,
            schema,
            source,
            payload,
            artifact_scope,
        )

    program = artifact(program_kind, program_schema, (1,), (101,), None)
    definition = FormalArtifactDefinition(
        program,
        (),
        value_kind,
        value_schema,
        proof_kind,
        proof_schema,
        executor,
        verifier,
    )
    invocation = ArtifactInvocation(
        candidate.proposition.template,
        definition,
        (),
        source,
        scope,
        (10025, 1),
    )
    execution = ArtifactExecutionObservation(
        executor, source, scope, True, (7,), (10026, 1))
    value = artifact(value_kind, value_schema, (2,), (7,), scope)
    proof = artifact(proof_kind, proof_schema, (3,), (1,), scope)
    verification = ArtifactVerificationObservation(
        verifier, source, scope, accepted, (1,), (10027, 1))
    return ArtifactInvocationResult(
        invocation,
        (),
        execution,
        verification,
        value,
        proof,
        _U,
    )


class _Policy:
    """返回固定或按调用次序变化的 typed 内容决策。"""

    def __init__(self, decisions) -> None:
        self.decisions = (
            list(decisions) if isinstance(decisions, list) else [decisions])
        self.calls = 0

    def select(self, request, artifacts):
        """不读取 surface/token，只返回测试注入的 policy 结果。"""
        del request, artifacts
        index = min(self.calls, len(self.decisions) - 1)
        self.calls += 1
        return self.decisions[index]


def _decision(stance, selected=(), artifacts=(), seed: int = 1):
    """构造带显式 reason 和 trace 的 policy decision。"""
    return AnswerContentDecision(
        stance,
        minimal_instruction_identity((10030, seed)),
        selected,
        artifacts,
        (10031, seed),
    )


class _CompleteResolver:
    """G-01 测试中为后四层提供无领域语义的完整占位执行。"""

    def __init__(self, layer, protocol, seed: int) -> None:
        self.layer = layer
        self.protocol = protocol
        self.seed = seed

    def resolve(self, request, prior):
        """只证明 G-00 可继续调度，不冒充 G-02/G-03 领域实现。"""
        del request, prior
        return GenerationLayerDecision(
            self.layer,
            self.protocol.complete,
            minimal_instruction_identity((10032, self.seed)),
            payload=(self.seed,),
            trace=(10033, self.seed),
        )


def test_g01_answer_selects_supported_content_and_verified_artifact() -> None:
    """answer 必须由独立 Evidence 支持，成功 Artifact 只作为已选命题附件。"""
    request, _, _, _ = _request()
    candidate_key = request.candidate_keys()[0]
    attachment = ContentArtifactAttachment(
        candidate_key, _artifact(request.candidates[0]))
    protocol = _content_protocol()
    policy = _Policy(_decision(
        protocol.answer,
        (candidate_key,),
        (attachment.stable_key(),),
    ))
    selector = AnswerContentSelector(protocol, policy)

    selection = selector.select(request, (attachment,))

    assert selection.stance == protocol.answer
    assert selection.selected_candidate_keys == (candidate_key,)
    assert selection.selected_artifact_keys == (attachment.stable_key(),)
    assert request.candidates[0].state == _T


def test_g01_artifact_cannot_create_language_support_or_hide_failure() -> None:
    """语言 Evidence unknown 时成功 Artifact 也不能 answer，失败 Artifact 更不能被采用。"""
    request, _, _, _ = _request(states=(_U,))
    candidate_key = request.candidate_keys()[0]
    success = ContentArtifactAttachment(
        candidate_key, _artifact(request.candidates[0]))
    protocol = _content_protocol(10040)
    selector = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.answer,
        (candidate_key,),
        (success.stable_key(),),
    )))
    with pytest.raises(ValueError, match="Evidence|满足目标"):
        selector.select(request, (success,))

    supported, _, _, _ = _request()
    supported_key = supported.candidate_keys()[0]
    failed_result = replace(
        _artifact(supported.candidates[0]),
        verification=replace(
            _artifact(supported.candidates[0]).verification,
            accepted=False,
        ),
    )
    failed = ContentArtifactAttachment(supported_key, failed_result)
    selector = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.answer,
        (supported_key,),
        (failed.stable_key(),),
        seed=2,
    )))
    with pytest.raises(ValueError, match="验证成功"):
        selector.select(supported, (failed,))


def test_g01_conflict_cannot_be_hidden_by_omitting_refute_candidate() -> None:
    """同命题 support/refute 必须显示 conflict，policy 不能只选 support 后 answer。"""
    request, _, _, _ = _request(
        states=(_T, _F), proposition_indexes=(0, 0))
    support_key, refute_key = request.candidate_keys()
    protocol = _content_protocol(10050)

    hidden = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.answer, (support_key,))))
    with pytest.raises(ValueError, match="冲突"):
        hidden.select(request)

    conflict = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.conflict, (support_key, refute_key), seed=2)))
    selection = conflict.select(request)
    assert selection.stance == protocol.conflict
    assert len(selection.selected_candidate_keys) == 2


def test_g01_clarify_unknown_and_refuse_keep_distinct_semantics() -> None:
    """clarify 保留多命题，unknown 不隐藏答案，refuse 不改候选四态。"""
    ambiguous, _, _, _ = _request(states=(_T, _T))
    protocol = _content_protocol(10060)
    clarify = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.clarify, ambiguous.candidate_keys())))
    assert clarify.select(ambiguous).stance == protocol.clarify

    private_pick = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.answer, (ambiguous.candidate_keys()[0],), seed=6)))
    with pytest.raises(ValueError, match="多解"):
        private_pick.select(ambiguous)

    one_key = ambiguous.candidate_keys()[0]
    invalid = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.clarify, (one_key,), seed=2)))
    with pytest.raises(ValueError, match="完整多解"):
        invalid.select(ambiguous)

    unknown_request, _, _, _ = _request(states=(_U,))
    unknown = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.unknown, unknown_request.candidate_keys(), seed=3)))
    selection = unknown.select(unknown_request)
    assert selection.stance == protocol.unknown
    assert selection.trace

    hidden_answer = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.unknown, (), seed=4)))
    with pytest.raises(ValueError, match="存在可回答"):
        hidden_answer.select(ambiguous)

    before = ambiguous.candidates[0].state
    refuse = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.refuse, (one_key,), seed=5)))
    assert refuse.select(ambiguous).stance == protocol.refuse
    assert ambiguous.candidates[0].state == before


def test_g01_stance_and_content_layers_recompute_same_selection() -> None:
    """G-00 前两层独立重算同一选择；policy 漂移时 content 层 fail closed。"""
    request, _, _, _ = _request()
    key = request.candidate_keys()[0]
    plan_protocol = _plan_protocol()
    content_protocol = _content_protocol(10070)
    selector = AnswerContentSelector(content_protocol, _Policy(_decision(
        content_protocol.answer, (key,))))
    registrations = [
        GenerationLayerRegistration(
            plan_protocol.stance_layer,
            GenerationStanceLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.content_layer,
            GenerationContentLayerResolver(plan_protocol, selector),
        ),
    ]
    for index, layer in enumerate(plan_protocol.layers()[2:], start=1):
        registrations.append(GenerationLayerRegistration(
            layer, _CompleteResolver(layer, plan_protocol, index)))
    plan = GenerationPlanner(plan_protocol, registrations).plan(request)
    assert plan.complete
    assert plan.layers[0].payload == plan.layers[1].payload

    drifting_policy = _Policy([
        _decision(content_protocol.answer, (key,), seed=2),
        _decision(content_protocol.refuse, (key,), seed=3),
    ])
    drifting_selector = AnswerContentSelector(
        content_protocol, drifting_policy)
    registrations[0] = GenerationLayerRegistration(
        plan_protocol.stance_layer,
        GenerationStanceLayerResolver(plan_protocol, drifting_selector),
    )
    registrations[1] = GenerationLayerRegistration(
        plan_protocol.content_layer,
        GenerationContentLayerResolver(plan_protocol, drifting_selector),
    )
    with pytest.raises(ValueError, match="不一致"):
        GenerationPlanner(plan_protocol, registrations).plan(request)


def test_g01_rejects_artifact_scope_or_candidate_drift() -> None:
    """Artifact attachment 不能替换 Proposition，也不能挂到其他候选。"""
    request, _, _, _ = _request(states=(_T, _T))
    first, second = request.candidates
    drift = ContentArtifactAttachment(
        second.stable_key(), _artifact(first))
    protocol = _content_protocol(10080)
    selector = AnswerContentSelector(protocol, _Policy(_decision(
        protocol.answer, (second.stable_key(),))))
    with pytest.raises(ValueError, match="Proposition"):
        selector.select(request, (drift,))
