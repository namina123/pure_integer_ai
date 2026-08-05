"""为 F-01 生产设施演练装配 typed 问答、生成和同次复核。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionSelector,
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
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
    GenerationPlanProtocol,
    GenerationPlanner,
)
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActDiscourseRouter,
    ResponseActGenerationRegistry,
    ResponseActGenerationTemplate,
    ResponseActPropositionRouter,
    ResponseActSyntaxRouter,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionRequest,
    SentenceStructureExecution,
    SentenceStructureExecutionBudget,
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
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
    GenerationSurfaceRequest,
    SurfaceSlotDirective,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisLedger
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
    RelationClosureField,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_CONSUMER_ACCEPTED,
    StructureOrderLinearizationResult,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorPropositionMapper,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
    VerificationEvaluation,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


_BASE = 19300
_ALIAS_BASE = 13200
_POSTCHECK_BASE = 13600
_POSTCHECK_MAPPER_BASE = 19700


def _source(source_id: int) -> SourceRef:
    """构造 relation closure 使用的独立公开夹具来源。"""
    return SourceRef(131, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


def _semantic_graph(ontology: Any) -> SemanticGraph:
    """安装可从相同 ontology 重建的原子命题 predicate 协议。"""
    identities = tuple(
        relation_concept_identity((8100, ordinal))
        for ordinal in range(1, 7)
    )
    refs = tuple(ontology.materialize(item) for item in identities)
    return SemanticGraph(ontology, AtomicPropositionPredicates(*refs))


def _projection_protocol() -> CandidateProjectionProtocol:
    """构造互不复用字段的候选 lifecycle 图协议。"""
    values = tuple(concept_identity((8200, ordinal)) for ordinal in range(13))
    return CandidateProjectionProtocol(*values, (8201, 1))


def _evidence_protocol() -> EvidenceCandidateProtocol:
    """构造要求两个独立形成来源的候选 aggregate 协议。"""
    aggregate = _source(900)
    return EvidenceCandidateProtocol(
        (8300, 1),
        (8300, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )


def _verifier() -> IndependentObjectVerifier:
    """构造只消费显式 reveal 的三态对象 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((8400, 1)),
        (8400, 2),
        (8400, 3),
        (8400, 4),
        (8400, 5),
    ))


def _relation_protocol() -> RelationClosureProtocol:
    """构造 relation/schema 两个互异候选字段。"""
    return RelationClosureProtocol(
        RelationClosureField(concept_identity((8500, 1))),
        RelationClosureField(concept_identity((8500, 2))),
    )


def _candidate_runtime(
        graph: CandidateProjectionGraph,
        ) -> CandidateLearningRuntime:
    """把共享候选 engine、verifier 和图装成真实 relation owner。"""
    return CandidateLearningRuntime(
        EvidenceCandidateEngine(_evidence_protocol()),
        graph,
        _verifier(),
        CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
    )


def _relation_definition(
        source: SourceRef,
        *,
        family: int,
        relation: ObjectIdentity,
        schema_identity: ObjectIdentity,
        role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        ) -> tuple[AtomicPropositionDefinition, RelationSchema]:
    """按实际 filler 类型构造开放 n 元 relation fact。"""
    schema = RelationSchema(
        schema_identity,
        relation,
        tuple(
            RelationSlotSchema(
                role,
                frozenset({filler.object_kind}),
                1,
                1,
            )
            for role, filler in role_fillers
        ),
    )
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (family, 1)),
        relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (family, 2)),
        tuple(AtomicRoleBinding(role, filler) for role, filler in role_fillers),
    )
    return definition, schema


def _recognition(
        spec: RelationClosureCandidateSpec,
        source_id: int,
        ) -> RelationClosureRecognitionInput:
    """为一个 relation spec 构造独立 support reveal。"""
    observation = _source(source_id)
    anchor = occurrence_identity(observation, start=0, end=1, ordinal=0)
    proposition = spec.proposition.proposition
    return RelationClosureRecognitionInput(
        proposition,
        observation,
        document_scope(observation),
        ProtocolKey((9200, source_id)),
        (9201, source_id),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            document_scope(observation),
            (9201, source_id),
            _source(700 + source_id),
            supported_targets=(proposition,),
            trace=(9202, source_id),
        ),
    )


@dataclass
class _AliasFixture:
    """持有设施生成场景的 active relation owner 和图。"""

    backend: DictBackend
    runtime: AliasRelationRuntime

    def close(self) -> None:
        """关闭 alias 场景的独立内存后端。"""
        self.backend.close()


def _alias_fixture(
        branch: ObjectIdentity,
        realizations: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        ) -> _AliasFixture:
    """为权威对象建立 active realizes fact 和唯一 surface route。"""
    if branch.object_kind != OBJECT_LANGUAGE_BRANCH:
        raise ValueError("设施 branch 必须是 LanguageBranch")
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = _candidate_runtime(candidate_graph)
    closure_protocol = _relation_protocol()
    alias_relation = concept_identity((_ALIAS_BASE + 1, 1))
    refers_relation = concept_identity((_ALIAS_BASE + 1, 2))
    realizes_relation = concept_identity((_ALIAS_BASE + 1, 3))
    alias_roles = (
        role_identity((_ALIAS_BASE + 2, 1)),
        role_identity((_ALIAS_BASE + 2, 2)),
    )
    refers_roles = (
        role_identity((_ALIAS_BASE + 2, 3)),
        role_identity((_ALIAS_BASE + 2, 4)),
    )
    realizes_roles = tuple(
        role_identity((_ALIAS_BASE + 2, value))
        for value in range(5, 8)
    )
    alias_schema_identity = structure_concept_identity((_ALIAS_BASE + 3, 1))
    refers_schema_identity = structure_concept_identity((_ALIAS_BASE + 3, 2))
    realizes_schema_identity = structure_concept_identity((_ALIAS_BASE + 3, 3))
    alias_schema = RelationSchema(
        alias_schema_identity,
        alias_relation,
        (
            RelationSlotSchema(
                alias_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(
                alias_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    refers_schema = RelationSchema(
        refers_schema_identity,
        refers_relation,
        (
            RelationSlotSchema(
                refers_roles[0], frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(
                refers_roles[1], frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    realizes_schema = RelationSchema(
        realizes_schema_identity,
        realizes_relation,
        (
            RelationSlotSchema(
                realizes_roles[0],
                frozenset(origin.object_kind for origin, _ in realizations),
                1,
                1,
            ),
            RelationSlotSchema(
                realizes_roles[1],
                frozenset({realizations[0][1].object_kind}),
                1,
                1,
            ),
            RelationSlotSchema(
                realizes_roles[2], frozenset({branch.object_kind}), 1, 1),
        ),
    )
    definitions: list[tuple[AtomicPropositionDefinition, RelationSchema]] = []
    for index, (origin, representation) in enumerate(realizations, start=1):
        definition, _ = _relation_definition(
            _source(_ALIAS_BASE + 40 + index),
            family=_ALIAS_BASE + 200 + index,
            relation=realizes_relation,
            schema_identity=realizes_schema_identity,
            role_fillers=(
                (realizes_roles[0], origin),
                (realizes_roles[1], representation),
                (realizes_roles[2], branch),
            ),
        )
        definitions.append((definition, realizes_schema))
        semantic_graph.define_atomic(
            definition,
            scope=document_scope(definition.source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        closure_protocol,
        (alias_schema, refers_schema, realizes_schema),
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        closure_protocol,
    )
    for index, (definition, schema) in enumerate(definitions, start=1):
        spec = RelationClosureCandidateSpec(
            definition,
            schema,
            (_ALIAS_BASE + 300, index),
            (_source(_ALIAS_BASE + 301), _source(_ALIAS_BASE + 302)),
        )
        closure.form(spec)
        trace = closure.recognize(_recognition(spec, _ALIAS_BASE + 400 + index))
        if trace.active_fact is None:
            raise RuntimeError("设施 alias relation 未形成 active fact")
    protocol = AliasResolutionProtocol(
        alias_relation,
        (alias_schema_identity,),
        *alias_roles,
        minimal_instruction_identity((_ALIAS_BASE + 4, 1)),
        refers_relation,
        (refers_schema_identity,),
        *refers_roles,
        minimal_instruction_identity((_ALIAS_BASE + 4, 2)),
        realizes_relation,
        (realizes_schema_identity,),
        *realizes_roles,
        minimal_instruction_identity((_ALIAS_BASE + 4, 3)),
        minimal_instruction_identity((_ALIAS_BASE + 4, 4)),
        minimal_instruction_identity((_ALIAS_BASE + 4, 5)),
        minimal_instruction_identity((_ALIAS_BASE + 4, 6)),
    )
    return _AliasFixture(
        backend,
        AliasRelationRuntime(closure, AliasResolutionSelector(protocol)),
    )


def _world() -> tuple[SourceRef, Any, Any]:
    """构造来源、query scope 和一个已绑定 proposition。"""
    source = SourceRef(10101, 10102, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = query_scope(1, parent=document_scope(source))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((10103, index))
        for index in range(1, 10)
    ))
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (10104, 1)),
        concept_identity((10105, 1)),
        occurrence_identity(source, start=1, end=2, ordinal=0),
        context_scope_identity(source, (10106, 1)),
        (),
    )
    graph = PropositionTemplateGraph((ScopedPropositionTemplate(
        definition,
        structure_concept_identity((10107, 1)),
    ),))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((10108, 1)),
        failures,
    ))
    bound = substituter.substitute(
        definition.proposition,
        graph,
        BindingEnvironment(),
    )
    return source, scope, bound


def _plan_protocol(seed: int) -> GenerationPlanProtocol:
    """注入 G-00 六层和结果身份。"""
    return GenerationPlanProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 11)
    ))


def _surface_protocol(seed: int) -> GenerationSurfaceProtocol:
    """构造互异 emit/silent 动作和七类 surface 原因。"""
    return GenerationSurfaceProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 10)
    ))


def _manual_execution(structure: Any) -> GenerationStructureExecutionPlan:
    """为无 active constraint 的单句计划构造真实 accepted 结果。"""
    sentence = structure.syntax.sentences[0]
    obligation = structure.syntax.linearization[0]
    budget = StructureOrderSearchBudget(20)
    request = GenerationStructureExecutionRequest(
        structure.syntax,
        (SentenceStructureExecutionBudget(sentence.sentence, budget),),
    )
    result = StructureOrderLinearizationResult(
        ORDER_CONSUMER_ACCEPTED,
        obligation.values,
        (),
        (),
        1,
    )
    return GenerationStructureExecutionPlan(
        request,
        (SentenceStructureExecution(
            obligation,
            sentence.slots,
            (),
            budget,
            result,
        ),),
    )


def _directives(
        structure: Any,
        protocol: GenerationSurfaceProtocol,
        ) -> tuple[SurfaceSlotDirective, ...]:
    """为全部 planned slot 注入唯一 surface 采用身份和预算。"""
    directives = []
    ordinal = 0
    for sentence in structure.syntax.sentences:
        for value in sentence.values:
            ordinal += 1
            directives.append(SurfaceSlotDirective(
                sentence=sentence.sentence,
                slot=value.slot,
                action=protocol.emit_action,
                instruction=minimal_instruction_identity(
                    (_ALIAS_BASE + 520, ordinal)),
                trace=(_ALIAS_BASE + 521, ordinal),
                surface_prefix_steps=(),
                surface_budget=AliasRouteSearchBudget(30, 30, 30),
                surface_use_key=(_ALIAS_BASE + 522, ordinal),
            ))
    return tuple(directives)


class _AnswerDiscourseMapper:
    """把唯一已选候选投影为无依赖篇章节点。"""

    def plan(self, selection: Any) -> DiscoursePlan:
        """保留唯一命题节点，不从容器顺序生成额外语句。"""
        keys = selection.selected_candidate_keys
        if len(keys) != 1:
            raise ValueError("设施问答 mapper 只接受唯一候选")
        return DiscoursePlan(
            selection.stable_key(),
            keys,
            (),
            (),
            (),
            require_unique_order=True,
        )


class _AnswerSyntaxMapper:
    """用显式单槽结构把已选命题送入结构顺序执行。"""

    def __init__(
            self,
            sentence: ObjectIdentity,
            slot: StructureSlotDefinition,
            boundary: ObjectIdentity,
            reason: ObjectIdentity,
            ) -> None:
        self.sentence = sentence
        self.slot = slot
        self.boundary = boundary
        self.reason = reason

    def plan(
            self,
            selection: Any,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
            ) -> SyntaxPlan:
        """绑定唯一命题 template，并保留候选归因和线性化义务。"""
        keys = selection.selected_candidate_keys
        if len(keys) != 1 or discourse.topological_order != keys:
            raise ValueError("设施 syntax 只接受唯一有序命题")
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
            self.reason,
            selection.request.goal.source,
            selection.request.goal.scope,
        )
        return SyntaxPlan(selection.stable_key(), (sentence,), (), (obligation,))


class _SurfaceRequestBuilder:
    """把动态结构接到真实顺序结果和 typed surface 请求。"""

    def __init__(self, protocol: GenerationSurfaceProtocol) -> None:
        self.protocol = protocol

    def build(self, structure: Any) -> GenerationSurfaceRequest:
        """从当前结构建立同次执行和逐槽 surface 指令。"""
        return GenerationSurfaceRequest(
            self.protocol,
            structure,
            _manual_execution(structure),
            structure.selection.request.goal.target_branch,
            _directives(structure, self.protocol),
        )


@dataclass
class FacilityQuestionFixture:
    """保存设施问答 runtime、请求和可关闭 alias owner。"""

    runtime: QuestionAnswerRuntime
    request: QuestionRequest
    alias: _AliasFixture

    def close(self) -> None:
        """关闭本次问答使用的独立 alias 后端。"""
        self.alias.close()


def _content_protocol() -> AnswerContentProtocol:
    """注入互异的五类回答 stance。"""
    return AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE + 1, index))
        for index in range(1, 6)
    ))


def build_question_fixture(
        *,
        world: tuple[SourceRef, Any, Any] | None = None,
        executor_factory: Any,
        selection_committer: Any = None,
        postcheck_mapper: Any = None,
        postchecker: Any = None,
        outcome_committer: Any = None,
        ) -> FacilityQuestionFixture:
    """用 typed query owner 装配 G-00 至 G-04 的单命题纵切。"""
    source, response_scope, target = _world() if world is None else world
    branch = language_branch_identity((_BASE + 2, 1))
    content = _content_protocol()
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(*tuple(
                minimal_instruction_identity((_BASE + 5, index))
                for index in range(1, 5)
            )),
        ),
    )
    request = QuestionRequest(
        minimal_instruction_identity((_BASE + 4, 1)),
        minimal_instruction_identity((_BASE + 4, 2)),
        minimal_instruction_identity((_BASE + 4, 3)),
        target,
        LogicEvidenceState(True, False),
        document_scope(source),
        response_scope,
        (_BASE + 4, 4),
        branch,
    )
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
    registry = ResponseActGenerationRegistry((ResponseActGenerationTemplate(
        branch,
        content.unknown,
        structure_concept_identity((_BASE + 7, 5)),
        response_slot,
        minimal_instruction_identity((_BASE + 7, 6)),
        minimal_instruction_identity((_BASE + 7, 7)),
    ),))
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
    alias = _alias_fixture(branch, ((
        target.template,
        representation_identity(family, tuple(ord(char) for char in "设施")),
    ),))
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
    runtime = QuestionAnswerRuntime(
        QuestionAnswerProtocol(*tuple(
            minimal_instruction_identity((_BASE + 14, index))
            for index in range(1, 4)
        )),
        (QuestionRouteRegistration(
            request.query_kind,
            route,
            executor_factory(route),
        ),),
        selector,
        generator,
        selection_committer=selection_committer,
        postcheck_mapper=postcheck_mapper,
        postchecker=postchecker,
        outcome_committer=outcome_committer,
    )
    return FacilityQuestionFixture(runtime, request, alias)


def _postcheck_protocol() -> GenerationPostcheckProtocol:
    """构造六维复核键和全部内置失败原因。"""
    keys = tuple(
        ProtocolKey((_POSTCHECK_BASE + 1, index))
        for index in range(1, 13)
    )
    reasons = tuple(
        minimal_instruction_identity((_POSTCHECK_BASE + 2, index))
        for index in range(1, 16)
    )
    return GenerationPostcheckProtocol(*keys, *reasons)


class _ExecutionParser:
    """只按 mapper 预登记的同次 execution 请求返回 typed 观察。"""

    def __init__(self) -> None:
        self._observations: dict[tuple[int, ...], Any] = {}

    def record(self, execution: Any, *, cited_sources: tuple[Any, ...]) -> None:
        """从实际 execution 构造并登记受限 surface 观察。"""
        from pure_integer_ai.cognition.shared.generation_verification import (
            GenerationSurfaceObservation,
            GenerationSurfaceParseRequest,
            RecoveredGenerationProposition,
        )

        planned = execution.surface.preview.request.structure.propositions.propositions
        goal = execution.plan.request.goal
        structure = execution.surface.preview.request.structure
        observation = GenerationSurfaceObservation(
            GenerationSurfaceParseRequest.from_execution(execution).stable_key(),
            execution.representations,
            goal.target_branch,
            structure.selection.stance,
            goal.source,
            goal.scope,
            tuple(
                RecoveredGenerationProposition(
                    item.candidate_key,
                    item.proposition,
                    item.source,
                    item.scope,
                    (_POSTCHECK_BASE + 4, index),
                )
                for index, item in enumerate(planned, start=1)
            ),
            (),
            cited_sources,
            structure.syntax.stable_key(),
            (),
            (_POSTCHECK_BASE + 5, 1),
        )
        key = GenerationSurfaceParseRequest.from_execution(execution).stable_key()
        self._observations[key] = observation

    def parse(self, request: Any) -> Any:
        """返回同次请求观察，未知请求显式分型为 parse failure。"""
        from pure_integer_ai.cognition.shared.generation_verification import (
            GenerationSurfaceParseResult,
        )

        observation = self._observations.get(request.stable_key())
        if observation is None:
            return GenerationSurfaceParseResult(
                minimal_instruction_identity((_POSTCHECK_BASE + 16, 3)),
                (_POSTCHECK_BASE + 16, 4),
            )
        return GenerationSurfaceParseResult(
            minimal_instruction_identity((_POSTCHECK_BASE + 16, 1)),
            (_POSTCHECK_BASE + 16, 2),
            observation,
        )


class _StaticVerifier:
    """为结构或来源维度返回固定 support 的独立 verifier。"""

    def __init__(self, marker: int) -> None:
        self.marker = marker

    def verify(self, request: Any) -> VerificationEvaluation:
        """把同次 execution 作为 claim，并绑定 generation goal 归属。"""
        goal = request.postcheck.execution.plan.request.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT,
            (request.postcheck.execution.stable_key(),),
            detail=(_POSTCHECK_BASE + 6, self.marker),
            source=goal.source,
            scope=goal.scope,
        )


class _RecordingPostcheckMapper:
    """登记实际 renderer 输出，再委托生产 Evidence postcheck mapper。"""

    def __init__(self, parser: _ExecutionParser) -> None:
        self.parser = parser
        self.delegate = EvidenceQuestionPostcheckMapper(
            (_POSTCHECK_MAPPER_BASE + 1, 1),
            citation_required=True,
            trust_required=True,
        )

    def build(
            self,
            request: Any,
            query: Any,
            result: Any,
            generation: Any,
            ) -> Any:
        """从被选择候选读取实际 citation 来源并建立同次复核请求。"""
        selected = set(
            generation.surface.preview.request.structure
            .selection.selected_candidate_keys
        )
        cited_sources = tuple(sorted({
            source
            for candidate in generation.plan.request.candidates
            if candidate.stable_key() in selected
            for source in candidate.citation_sources
        }, key=SourceRef.stable_key))
        self.parser.record(generation, cited_sources=cited_sources)
        return self.delegate.build(request, query, result, generation)


def build_postcheck_owners() -> tuple[Any, GenerationPostcheckRuntime]:
    """装配六维 G-04 runtime 和受限同次 execution parser。"""
    parser = _ExecutionParser()
    runtime = GenerationPostcheckRuntime(
        _postcheck_protocol(),
        parser,
        _StaticVerifier(1),
        _StaticVerifier(2),
    )
    return _RecordingPostcheckMapper(parser), runtime


__all__ = [
    "FacilityQuestionFixture",
    "build_postcheck_owners",
    "build_question_fixture",
]
