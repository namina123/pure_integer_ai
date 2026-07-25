"""F-00 显式事实、unknown、冲突和 unsupported 首个真实生成纵切。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerRegistration,
    GenerationPlanner,
)
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActDiscourseRouter,
    ResponseActGenerationRegistry,
    ResponseActGenerationTemplate,
    ResponseActPropositionRouter,
    ResponseActSyntaxRouter,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscoursePlan,
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationStructurePlanner,
    GenerationSyntaxLayerResolver,
    PlannedSentence,
    PropositionPlan,
    PropositionSlotFiller,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    FactQuestionExecutor,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureSlotValue,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorPropositionMapper,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)
from tests.test_g02_generation_structure_plan import (
    _plan_protocol,
    _world,
)
from tests.test_g03_generation_surface import (
    _alias_fixture,
    _directives,
    _manual_execution,
    _surface_protocol,
)


_BASE = 19300
_REQUIRED = LogicEvidenceState(True, False)


class _AnswerDiscourseMapper:
    """为单个显式事实保留唯一命题节点，不从容器顺序生成多句。"""

    def plan(self, selection) -> DiscoursePlan:
        """把唯一已选候选投影为无依赖篇章节点。"""
        keys = selection.selected_candidate_keys
        if len(keys) != 1:
            raise ValueError("测试事实 mapper 只接受唯一已选命题")
        return DiscoursePlan(
            selection.stable_key(),
            keys,
            (),
            (),
            (),
            require_unique_order=True,
        )


class _AnswerSyntaxMapper:
    """用注入的单槽语言结构把已选事实命题送入 S-07。"""

    def __init__(
            self,
            sentence,
            slot: StructureSlotDefinition,
            boundary,
            linearization_reason,
            ) -> None:
        """绑定测试课程提供的句式、槽、边界和顺序原因。"""
        self.sentence = sentence
        self.slot = slot
        self.boundary = boundary
        self.linearization_reason = linearization_reason

    def plan(self, selection, discourse, propositions) -> SyntaxPlan:
        """把唯一命题本体绑定到实际槽值，并保留完整候选归因。"""
        keys = selection.selected_candidate_keys
        if len(keys) != 1 or discourse.topological_order != keys:
            raise ValueError("测试事实 syntax 只接受唯一有序命题")
        planned = {item.candidate_key: item for item in propositions.propositions}
        proposition = planned[keys[0]].proposition
        value = StructureSlotValue(self.slot.slot, proposition.template)
        sentence = PlannedSentence(
            self.sentence,
            self.slot.structure,
            0,
            keys,
            (self.slot,),
            (value,),
            (PropositionSlotFiller(keys[0], proposition, value),),
            self.boundary,
            selection.request.goal.source,
            selection.request.goal.scope,
        )
        obligation = SyntaxLinearizationObligation(
            self.sentence,
            self.slot.structure,
            (value,),
            (),
            (),
            self.linearization_reason,
            selection.request.goal.source,
            selection.request.goal.scope,
        )
        return SyntaxPlan(
            selection.stable_key(),
            (sentence,),
            (),
            (obligation,),
        )


class _SurfaceRequestBuilder:
    """把动态 G-02 结构接到真实顺序执行和 R-01 surface runtime。"""

    def __init__(self, protocol) -> None:
        """绑定注入式 G-03 动作和失败原因。"""
        self.protocol = protocol

    def build(self, structure):
        """从当前结构建立同次顺序执行和逐槽 surface 指令。"""
        from pure_integer_ai.cognition.shared.generation_surface import (
            GenerationSurfaceRequest,
        )
        return GenerationSurfaceRequest(
            self.protocol,
            structure,
            _manual_execution(structure),
            structure.selection.request.goal.target_branch,
            _directives(structure, self.protocol),
        )


@dataclass
class _Fixture:
    """保存问答 runtime、请求、H-00 ledger 和可关闭的 R-01 owner。"""

    runtime: QuestionAnswerRuntime
    request: QuestionRequest
    ledger: HypothesisLedger
    alias: object
    renderer: UnicodeRepresentationRenderer
    content: AnswerContentProtocol
    evidence: tuple[EvidenceRecord, ...]

    def close(self) -> None:
        """关闭 R-01 测试后端。"""
        self.alias.close()


def _content_protocol() -> AnswerContentProtocol:
    """注入互异的五类回答 stance。"""
    return AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE + 1, index))
        for index in range(1, 6)
    ))


def _fixture(
        *stances: int,
        world=None,
        executor_factory=None,
        required: LogicEvidenceState = _REQUIRED,
        answer_text: str = "事实",
        target_branch=None,
        query_kind=None,
        intent=None,
        goal_kind=None,
        selection_committer=None,
        postcheck_mapper=None,
        postchecker=None,
        outcome_committer=None,
        ) -> _Fixture:
    """建立可注入 typed 查询 owner 的 G-00 至 G-03 与 Unicode renderer 纵切。"""
    if world is None:
        source, response_scope, targets = _world(1)
        target = targets[0]
    else:
        source, response_scope, target = world
    if stances and executor_factory is not None:
        raise ValueError("外部 question executor 不得同时写本地事实 Evidence")
    evidence_scope = document_scope(source)
    branch = (
        language_branch_identity((_BASE + 2, 1))
        if target_branch is None else target_branch
    )
    content = _content_protocol()
    ledger = HypothesisLedger()
    records = ()
    hypothesis_kind = (_BASE + 3, 1)
    if stances:
        hypothesis = ledger.register(HypothesisKey(
            hypothesis_kind,
            target.template.stable_key(),
            (_BASE + 3, 2),
            evidence_scope,
            source,
        ))
        records = tuple(
            ledger.append_evidence(EvidenceRecord(
                index,
                hypothesis,
                stance,
                (_BASE + 3, 10 + index),
                source,
                index,
            ))
            for index, stance in enumerate(stances, start=1)
        )

    request = QuestionRequest(
        (minimal_instruction_identity((_BASE + 4, 1))
         if query_kind is None else query_kind),
        (minimal_instruction_identity((_BASE + 4, 2))
         if intent is None else intent),
        (minimal_instruction_identity((_BASE + 4, 3))
         if goal_kind is None else goal_kind),
        target,
        required,
        evidence_scope,
        response_scope,
        (_BASE + 4, 4),
        branch,
    )
    policy = EvidenceAnswerPolicy(
        content,
        EvidenceAnswerPolicyProtocol(*tuple(
            minimal_instruction_identity((_BASE + 5, index))
            for index in range(1, 5)
        )),
    )
    selector = AnswerContentSelector(content, policy)

    answer_structure = structure_concept_identity((_BASE + 6, 1))
    answer_slot = StructureSlotDefinition(
        answer_structure,
        structure_concept_identity((_BASE + 6, 2)),
        role_identity((_BASE + 6, 3)),
        concept_identity((_BASE + 6, 4)),
    )
    response_structure = structure_concept_identity((_BASE + 7, 1))
    response_slot = StructureSlotDefinition(
        response_structure,
        structure_concept_identity((_BASE + 7, 2)),
        role_identity((_BASE + 7, 3)),
        concept_identity((_BASE + 7, 4)),
    )
    registry = ResponseActGenerationRegistry((
        ResponseActGenerationTemplate(
            branch,
            content.unknown,
            structure_concept_identity((_BASE + 7, 5)),
            response_slot,
            minimal_instruction_identity((_BASE + 7, 6)),
            minimal_instruction_identity((_BASE + 7, 7)),
        ),
    ))
    structure_planner = GenerationStructurePlanner(
        ResponseActDiscourseRouter(_AnswerDiscourseMapper(), registry),
        ResponseActPropositionRouter(
            LanguageConnectorPropositionMapper(), registry),
        ResponseActSyntaxRouter(
            _AnswerSyntaxMapper(
                structure_concept_identity((_BASE + 6, 5)),
                answer_slot,
                minimal_instruction_identity((_BASE + 6, 6)),
                minimal_instruction_identity((_BASE + 6, 7)),
            ),
            registry,
        ),
    )
    planner_protocol = _plan_protocol(_BASE + 8)
    structure_protocol = GenerationStructureLayerProtocol(*tuple(
        minimal_instruction_identity((_BASE + 9, index))
        for index in range(1, 4)
    ))
    surface_protocol = _surface_protocol(_BASE + 10)
    family = (_BASE + 11, 1)
    alias = _alias_fixture(branch, (
        (target.template,
         representation_identity(
             family, tuple(ord(char) for char in answer_text))),
        (content.unknown,
         representation_identity(family, tuple(ord(char) for char in "未知"))),
    ))
    surface_runtime = GenerationSurfaceRuntime(alias.runtime)
    registrations = (
        GenerationLayerRegistration(
            planner_protocol.stance_layer,
            GenerationStanceLayerResolver(planner_protocol, selector),
        ),
        GenerationLayerRegistration(
            planner_protocol.content_layer,
            GenerationContentLayerResolver(planner_protocol, selector),
        ),
        GenerationLayerRegistration(
            planner_protocol.discourse_layer,
            GenerationDiscourseLayerResolver(
                planner_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            planner_protocol.proposition_layer,
            GenerationPropositionLayerResolver(
                planner_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            planner_protocol.syntax_layer,
            GenerationSyntaxLayerResolver(
                planner_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            planner_protocol.surface_layer,
            GenerationSurfaceLayerResolver(
                planner_protocol,
                selector,
                structure_planner,
                _SurfaceRequestBuilder(surface_protocol),
                surface_runtime,
                commit=False,
            ),
        ),
    )
    renderer = UnicodeRepresentationRenderer(
        family,
        minimal_instruction_identity((_BASE + 12, 1)),
    )
    generator = TypedGenerationExecutor(
        GenerationPlanner(planner_protocol, registrations),
        renderer,
        surface_runtime,
    )
    route = minimal_instruction_identity((_BASE + 13, 1))
    question_executor = (
        FactQuestionExecutor(
            ledger,
            route=route,
            hypothesis_kind=hypothesis_kind,
            executed_reason=minimal_instruction_identity(
                (_BASE + 13, 2)),
        )
        if executor_factory is None else executor_factory(route)
    )
    runtime = QuestionAnswerRuntime(
        QuestionAnswerProtocol(*tuple(
            minimal_instruction_identity((_BASE + 14, index))
            for index in range(1, 4)
        )),
        (QuestionRouteRegistration(
            request.query_kind,
            route,
            question_executor,
        ),),
        selector,
        generator,
        selection_committer=selection_committer,
        postcheck_mapper=postcheck_mapper,
        postchecker=postchecker,
        outcome_committer=outcome_committer,
    )
    return _Fixture(
        runtime, request, ledger, alias, renderer, content, records)


def _rendered_text(fixture: _Fixture, run) -> str:
    """从同次 typed generation 读取 Unicode renderer 输出。"""
    assert run.generation is not None
    assert run.generation.rendered is not None
    return fixture.renderer.text(run.generation.rendered)


def test_f00_explicit_fact_reads_active_h00_evidence_and_generates_answer():
    """显式事实必须由 H-00 active Evidence 查询并走完整生成纵切。"""
    fixture = _fixture(EVIDENCE_SUPPORT)
    try:
        before = fixture.ledger.state_key()

        run = fixture.runtime.run(fixture.request)

        assert run.complete
        assert run.status == fixture.content.answer
        assert run.query_result.candidates[0].evidence == fixture.evidence
        assert run.selection.selected_candidate_keys
        assert _rendered_text(fixture, run) == "事实"
        assert fixture.ledger.state_key() == before
    finally:
        fixture.close()


@pytest.mark.parametrize("stances", [(), (EVIDENCE_UNKNOWN,)])
def test_f00_unknown_uses_stance_structure_instead_of_echoing_target(stances):
    """无 Evidence 或仅 unknown Evidence 时必须生成 response-act，不复述目标。"""
    fixture = _fixture(*stances)
    try:
        before = fixture.ledger.state_key()

        run = fixture.runtime.run(fixture.request)

        assert run.complete
        assert run.status == fixture.content.unknown
        assert run.selection.selected_candidate_keys == ()
        structure = run.generation.preview.request.structure
        sentence = structure.syntax.sentences[0]
        assert sentence.response_act == fixture.content.unknown
        assert sentence.values[0].filler == fixture.content.unknown
        assert _rendered_text(fixture, run) == "未知"
        assert fixture.ledger.state_key() == before
    finally:
        fixture.close()


def test_f00_unsupported_query_kind_never_calls_generation(monkeypatch):
    """未注册 query kind 必须 typed unsupported，且生成器不可被调用。"""
    fixture = _fixture(EVIDENCE_SUPPORT)
    try:
        unsupported = QuestionRequest(
            minimal_instruction_identity((_BASE + 20, 1)),
            fixture.request.intent,
            fixture.request.goal_kind,
            fixture.request.target,
            fixture.request.required,
            fixture.request.evidence_scope,
            fixture.request.response_scope,
            (_BASE + 20, 2),
            fixture.request.target_branch,
        )

        def forbidden(_request):
            """任何调用都表示 unsupported 分支错误回退到生成器。"""
            raise AssertionError("unsupported 不得调用生成器")

        monkeypatch.setattr(fixture.runtime.generator, "execute", forbidden)
        before = fixture.ledger.state_key()

        run = fixture.runtime.run(unsupported)

        assert run.status == fixture.runtime.protocol.unsupported_status
        assert run.query is None
        assert run.generation is None
        assert not run.complete
        assert fixture.ledger.state_key() == before
    finally:
        fixture.close()


def test_f00_conflicting_evidence_cannot_be_selected_as_answer():
    """同一命题 support/refute 冲突必须保持 conflict，不得按稳定序私选答案。"""
    fixture = _fixture(EVIDENCE_SUPPORT, EVIDENCE_REFUTE)
    try:
        before = fixture.ledger.state_key()

        run = fixture.runtime.run(fixture.request)

        assert run.complete
        assert run.status == fixture.content.conflict
        assert run.selection.stance == fixture.content.conflict
        assert run.selection.selected_candidate_keys
        assert run.status != fixture.content.answer
        assert fixture.ledger.state_key() == before
    finally:
        fixture.close()
