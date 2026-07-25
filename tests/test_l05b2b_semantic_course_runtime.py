"""L-05B2B 正式语义课程到 GenerationPlanningRequest 的接线测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest
import pure_integer_ai.experiments.round_runtime as round_module

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EVIDENCE_SUPPORT,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    document_scope,
    episode_scope,
    make_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerDecision,
    GenerationLayerRegistration,
    GenerationPlanProtocol,
    GenerationPlanner,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationStructurePlanner,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    context_scope_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.shared.types import (
    Episode,
    InputPayload,
    MODALITY_LANGUAGE,
    ObserveResult,
    STAGE_TRAINING,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    LocalSemanticRef,
    SemanticBindingSpec,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
    SemanticFillerSpec,
    SemanticObjectSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRequestDecision,
    ProductionGenerationRuntime,
    install_production_generation_runtime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    EvaluationIsolationError,
    isolated_evaluation,
)
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseDecision,
    LanguageSemanticCourseProtocol,
    LanguageSemanticLesson,
    SemanticCourseEvidenceSpec,
    SemanticCourseTemplateScope,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    SemanticCourseGenerationRequestMapper,
    install_language_semantic_course_runtime,
)
from pure_integer_ai.experiments.language_semantic_query import (
    LanguageSemanticQueryDecision,
    LanguageSemanticQueryProtocol,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.capability_exam import (
    project_g_attribution,
    project_layer0,
)
from pure_integer_ai.experiments.formal_train import FormalTrainResult
from pure_integer_ai.experiments.metrics import MetricsCollector
from pure_integer_ai.experiments.preflight_runtime import _pre_flight_impl
from pure_integer_ai.experiments.round_runtime import (
    DefaultRoundRunner,
    RoundResult,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_dispatch import (
    VERIFY_ROUTE_COMPARISON,
    VERIFY_ROUTE_NUMERIC,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationReport,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from tests.test_g02_generation_structure_plan import (
    _Policy,
    _content_protocol,
    _discourse,
    _plan_protocol,
    _propositions,
)
from tests.test_g03_generation_surface import (
    _StaticSurfaceBuilder,
    _alias_fixture,
    _structure_plan,
    _surface_protocol,
    _templates,
)
from tests.test_g04_generation_postcheck import (
    _ExecutionParser,
    _ProductionPostcheckMapper,
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from tests.test_l05b2_typed_production_generation import _forbid_legacy
from tests.test_l05b2b_language_generation_connector import (
    _connector,
    _execution_planner,
)
from tests.test_s07_structure_order import (
    _active_plan,
    _domain,
    _graphs,
    _learning_protocol,
    _pattern,
)


_BASE = 15100
_PROP = LocalSemanticRef(OBJECT_PROPOSITION, (2,))
_PROP_SECOND = LocalSemanticRef(OBJECT_PROPOSITION, (3,))
_EPISODE_HASHER = Hasher("formal_train.episode_scope.v1")


class _CourseMapper:
    """按测试来源返回显式 lesson，并记录收到的无词面输入。"""

    def __init__(self, decision: LanguageSemanticCourseDecision) -> None:
        self.decision = decision
        self.inputs = []

    def map(self, input_value):
        """保存 typed 输入并返回当前课程决定。"""
        self.inputs.append(input_value)
        if input_value.read_only:
            return replace(self.decision, lesson=None)
        return self.decision

    def clone_for_evaluation(self):
        """复制决定和调用历史，后续修改互不共享。"""
        cloned = _CourseMapper(self.decision)
        cloned.inputs = list(self.inputs)
        return cloned

    def state_key(self):
        """返回调用来源和 typed anchor 数量，不保存对象地址。"""
        return tuple((
            item.source.stable_key(),
            len(item.occurrences),
            len(item.spans),
            len(item.active_senses),
            item.read_only,
        ) for item in self.inputs)


class _UnusedLayerResolver:
    """为无请求评测构造合法 executor；实际调用即表示 fail-closed 失效。"""

    def resolve(self, request, prior):
        """无请求路径不得进入任一 G-00 layer。"""
        del request, prior
        raise AssertionError("typed 无请求评测不得执行 generation layer")


class _UnusedRenderer:
    """无请求评测使用的 renderer 哨兵。"""

    def render(self, representations):
        """无请求路径不得产生 Representation。"""
        del representations
        raise AssertionError("typed 无请求评测不得调用 renderer")


class _SemanticNoRequestFactory:
    """为 V-06 重建 semantic mapper 和全新无状态 executor。"""

    def __init__(self, key=(_BASE + 40, 1)) -> None:
        self.key = tuple(key)

    def build(self, ctx):
        """在当前上下文建立只消费同次 semantic course run 的 owner。"""
        del ctx
        identities = tuple(
            minimal_instruction_identity((self.key[0], ordinal))
            for ordinal in range(1, 11)
        )
        protocol = GenerationPlanProtocol(*identities)
        resolver = _UnusedLayerResolver()
        planner = GenerationPlanner(protocol, tuple(
            GenerationLayerRegistration(layer, resolver)
            for layer in protocol.layers()
        ))
        return ProductionGenerationRuntime(
            SemanticCourseGenerationRequestMapper(),
            TypedGenerationExecutor(planner, _UnusedRenderer()),
        )

    def clone_for_evaluation(self):
        """返回不共享对象身份的同配置 factory。"""
        return _SemanticNoRequestFactory(self.key)

    def state_key(self):
        """返回完整注入键。"""
        return self.key


class _MissingEvaluationProtocolFactory:
    """能装配 typed owner，但故意不提供 V-06 clone/state 协议。"""

    def __init__(self) -> None:
        self.delegate = _SemanticNoRequestFactory()

    def build(self, ctx):
        """复用合法 owner，隔离失败只能来自 factory 协议缺失。"""
        return self.delegate.build(ctx)


class _DriftingEvaluationFactory(_SemanticNoRequestFactory):
    """故意让评测 clone 改变注入键，验证配置漂移被拒绝。"""

    def clone_for_evaluation(self):
        """返回带不同键的 factory，模拟沙箱装配悄然漂移。"""
        return _SemanticNoRequestFactory((self.key[0], self.key[1] + 1))


class _DynamicDiscourseMapper:
    """从当前课程选择实时建立 G-02 discourse，不缓存预制请求。"""

    def plan(self, selection):
        """逐次按当前 selected candidate 构造篇章计划。"""
        return _discourse(selection)


class _DynamicPropositionMapper:
    """从当前课程选择实时建立 Evidence 守恒的 PropositionPlan。"""

    def plan(self, selection, discourse):
        """核验篇章属于当前选择后投影全部命题。"""
        if discourse != _discourse(selection):
            raise ValueError("动态 proposition mapper 收到漂移的 discourse")
        return _propositions(selection)


class _DynamicSyntaxMapper:
    """为当前双命题课程建立注入式句法结构和线性化义务。"""

    def __init__(self, branch) -> None:
        self.branch = branch

    def plan(self, selection, discourse, propositions):
        """只按 typed selection 重建结构，不读取 token 或 legacy 顺序。"""
        structure = _structure_plan(
            self.branch,
            request=selection.request,
        )
        if structure.discourse != discourse:
            raise ValueError("动态 syntax mapper 收到漂移的 discourse")
        if structure.propositions != propositions:
            raise ValueError("动态 syntax mapper 收到漂移的 proposition plan")
        return structure.syntax


class _ReadQueryMapper:
    """只在 recovered 输入恰有一个 ground 候选时显式选择 exact Hypothesis。"""

    def __init__(self) -> None:
        self.inputs = []

    def map(self, input_value):
        """保存不可变输入，并对唯一 ground 候选形成 query 决策。"""
        self.inputs.append(input_value)
        if len(input_value.candidates) != 1:
            return LanguageSemanticQueryDecision(
                minimal_instruction_identity((_BASE + 90, 1)),
                (_BASE + 90, len(self.inputs)),
            )
        candidate = input_value.candidates[0]
        if not candidate.ground:
            return LanguageSemanticQueryDecision(
                minimal_instruction_identity((_BASE + 90, 2)),
                (_BASE + 90, len(self.inputs)),
            )
        return LanguageSemanticQueryDecision(
            minimal_instruction_identity((_BASE + 90, 3)),
            (_BASE + 90, len(self.inputs)),
            candidate.hypothesis,
            (candidate.hypothesis,),
            minimal_instruction_identity((_BASE + 90, 4)),
            LogicEvidenceState(True, False),
            language_branch_identity((_BASE + 12, 1)),
        )

    def clone_for_evaluation(self):
        """复制既有调用历史，后续 query 状态互不共享。"""
        cloned = _ReadQueryMapper()
        cloned.inputs = list(self.inputs)
        return cloned

    def state_key(self):
        """返回已处理输入的候选数量和来源，不保存对象地址。"""
        return tuple(
            (item.current.source.stable_key(), len(item.candidates))
            for item in self.inputs
        )


class _NonGroundSelectingQueryMapper:
    """故意选择含 Variable 的候选，验证 runtime 归一为 typed 无请求。"""

    def __init__(self) -> None:
        self.inputs = []

    def map(self, input_value):
        """无视 ground 标志选择首项，模拟越权或错误 query mapper。"""
        self.inputs.append(input_value)
        if not input_value.candidates:
            return LanguageSemanticQueryDecision(
                minimal_instruction_identity((_BASE + 93, 1)),
                (_BASE + 93, len(self.inputs)),
            )
        candidate = input_value.candidates[0]
        return LanguageSemanticQueryDecision(
            minimal_instruction_identity((_BASE + 93, 2)),
            (_BASE + 93, len(self.inputs)),
            candidate.hypothesis,
            (candidate.hypothesis,),
            minimal_instruction_identity((_BASE + 93, 3)),
            LogicEvidenceState(True, False),
            language_branch_identity((_BASE + 93, 4)),
        )

    def clone_for_evaluation(self):
        """复制调用历史，避免评测读取后污染宿主 mapper。"""
        cloned = _NonGroundSelectingQueryMapper()
        cloned.inputs = list(self.inputs)
        return cloned

    def state_key(self):
        """返回每次输入的候选数和 ground 分布。"""
        return tuple(
            (len(item.candidates), tuple(
                candidate.ground for candidate in item.candidates))
            for item in self.inputs
        )


class _FailFirstLayerResolver:
    """让 typed planner 在首层显式失败，以证明请求已消费且 renderer 不运行。"""

    def __init__(self, protocol) -> None:
        self.protocol = protocol

    def resolve(self, request, prior):
        """核验 recovered 请求非空后返回 typed failed。"""
        assert request.candidates
        assert not prior
        return GenerationLayerDecision(
            self.protocol.stance_layer,
            self.protocol.failed,
            minimal_instruction_identity((_BASE + 91, 1)),
            trace=(_BASE + 91, 2),
        )


class _RecoveringFailFactory:
    """为 V-06 重建消费 recovered 请求但不调用 renderer 的 typed owner。"""

    def __init__(self, key=(_BASE + 92, 1)) -> None:
        self.key = tuple(key)

    def build(self, ctx):
        """在当前上下文建立首层 typed failure owner。"""
        del ctx
        identities = tuple(
            minimal_instruction_identity((self.key[0], ordinal))
            for ordinal in range(1, 11)
        )
        protocol = GenerationPlanProtocol(*identities)
        registrations = [GenerationLayerRegistration(
            protocol.stance_layer,
            _FailFirstLayerResolver(protocol),
        )]
        registrations.extend(
            GenerationLayerRegistration(layer, _UnusedLayerResolver())
            for layer in protocol.layers()[1:]
        )
        return ProductionGenerationRuntime(
            SemanticCourseGenerationRequestMapper(),
            TypedGenerationExecutor(
                GenerationPlanner(protocol, tuple(registrations)),
                _UnusedRenderer(),
            ),
        )

    def clone_for_evaluation(self):
        """返回同配置且不共享运行 owner 的 factory。"""
        return _RecoveringFailFactory(self.key)

    def state_key(self):
        """返回完整注入键。"""
        return self.key


def _source(document_id: int = 1) -> SourceRef:
    """建立带稳定 owner/version 的课程来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 1,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _failures() -> BindingFailureProtocol:
    """为 S-03 注入九个互异的失败原因。"""
    return BindingFailureProtocol(*tuple(
        minimal_instruction_identity((_BASE + 2, ordinal))
        for ordinal in range(1, 10)
    ))


def _lesson(source, span) -> LanguageSemanticLesson:
    """构造单命题理论课程，predicate、Role 和结构均由 fixture 注入。"""
    scope = document_scope(source)
    upstream = HypothesisKey(
        (_BASE + 3, 1),
        (_BASE + 3, 2),
        (_BASE + 3, 3),
        scope,
        source,
    )
    entity = SemanticObjectSpec(OBJECT_ENTITY, (1,))
    proposition = SemanticPropositionSpec(
        (2,),
        (_BASE + 4, 1),
        concept_identity((_BASE + 5, 1)),
        structure_concept_identity((_BASE + 6, 1)),
        (SemanticBindingSpec(
            role_identity((_BASE + 7, 1)),
            SemanticFillerSpec(local_ref=entity.local_ref),
        ),),
    )
    plan = SemanticBuildPlan(
        upstream,
        (_BASE + 8, 1),
        (entity,),
        (proposition,),
    )
    evidence = SemanticCourseEvidenceSpec(
        _PROP,
        _BASE + 9,
        EVIDENCE_SUPPORT,
        (_BASE + 10, 1),
        source,
        1,
        (_BASE + 10, 2),
    )
    return LanguageSemanticLesson(
        span,
        plan,
        (evidence,),
        (SemanticCourseTemplateScope(
            _PROP,
            context_scope_identity(source, (_BASE + 13, 1)),
        ),),
        BindingEnvironment(),
        _PROP,
        (_PROP,),
        minimal_instruction_identity((_BASE + 11, 1)),
        LogicEvidenceState(True, False),
        language_branch_identity((_BASE + 12, 1)),
    )


def _non_ground_lesson(source, anchor) -> LanguageSemanticLesson:
    """构造父命题嵌套含 Variable 子 template 的训练命题。"""
    lesson = _lesson(source, anchor)
    binder = binder_identity(source, (_BASE + 94, 1))
    variable = variable_identity(
        binder,
        (_BASE + 94, 2),
        concept_identity((_BASE + 94, 3)),
    )
    parent = replace(
        lesson.plan.propositions[0],
        bindings=(SemanticBindingSpec(
            role_identity((_BASE + 94, 4)),
            SemanticFillerSpec(local_ref=_PROP_SECOND),
        ),),
    )
    child = SemanticPropositionSpec(
        _PROP_SECOND.local_key,
        (_BASE + 94, 5),
        concept_identity((_BASE + 94, 6)),
        structure_concept_identity((_BASE + 94, 7)),
        (SemanticBindingSpec(
            role_identity((_BASE + 94, 8)),
            SemanticFillerSpec(external=variable),
        ),),
    )
    plan = replace(
        lesson.plan,
        objects=(),
        propositions=(parent, child),
    )
    return replace(
        lesson,
        plan=plan,
        template_scopes=(
            lesson.template_scopes[0],
            SemanticCourseTemplateScope(
                _PROP_SECOND,
                context_scope_identity(source, (_BASE + 94, 9)),
                (binder,),
            ),
        ),
    )


def _dual_lesson(source, anchor) -> LanguageSemanticLesson:
    """构造两个独立命题，供正式 G-02 句法与 G-03 双槽输出消费。"""
    scope = document_scope(source)
    upstream = HypothesisKey(
        (_BASE + 60, 1),
        (_BASE + 60, 2),
        (_BASE + 60, 3),
        scope,
        source,
    )
    entities = tuple(
        SemanticObjectSpec(OBJECT_ENTITY, (ordinal,))
        for ordinal in (1, 2)
    )
    proposition_refs = (_PROP, _PROP_SECOND)
    propositions = tuple(
        SemanticPropositionSpec(
            local_ref.local_key,
            (_BASE + 61, ordinal),
            concept_identity((_BASE + 62, ordinal)),
            structure_concept_identity((_BASE + 63, ordinal)),
            (SemanticBindingSpec(
                role_identity((_BASE + 64, ordinal)),
                SemanticFillerSpec(local_ref=entity.local_ref),
            ),),
        )
        for ordinal, (local_ref, entity)
        in enumerate(zip(proposition_refs, entities), start=1)
    )
    plan = SemanticBuildPlan(
        upstream,
        (_BASE + 65, 1),
        entities,
        propositions,
    )
    evidence = tuple(
        SemanticCourseEvidenceSpec(
            local_ref,
            _BASE + 70 + ordinal,
            EVIDENCE_SUPPORT,
            (_BASE + 66, ordinal),
            source,
            ordinal,
            (_BASE + 67, ordinal),
        )
        for ordinal, local_ref in enumerate(proposition_refs, start=1)
    )
    return LanguageSemanticLesson(
        anchor,
        plan,
        evidence,
        tuple(
            SemanticCourseTemplateScope(
                local_ref,
                context_scope_identity(source, (_BASE + 69, ordinal)),
            )
            for ordinal, local_ref in enumerate(proposition_refs, start=1)
        ),
        BindingEnvironment(),
        _PROP,
        proposition_refs,
        minimal_instruction_identity((_BASE + 68, 1)),
        LogicEvidenceState(True, False),
        language_branch_identity((_BASE + 69, 1)),
    )


def _protocol(mapper) -> LanguageSemanticCourseProtocol:
    """构造全部开放 predicate、builder、substitution 和来源元数据协议。"""
    predicates = tuple(
        concept_identity((_BASE + 20, ordinal))
        for ordinal in range(1, 12)
    )
    return LanguageSemanticCourseProtocol(
        SemanticBuilderProtocol(
            minimal_instruction_identity((_BASE + 21, 1)),
            (_BASE + 21, 2),
        ),
        predicates[:6],
        predicates[6:9],
        predicates[9:],
        SubstitutionProtocol(
            minimal_instruction_identity((_BASE + 22, 1)),
            _failures(),
        ),
        mapper,
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
        1,
        (_BASE + 23, 1),
    )


def _fixture(query_protocol=None, *, raw_text: str = "甲乙"):
    """装配同一图上的 occurrence/span、课程 runtime 和正式输入。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    occurrence_index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((_BASE + 30, 1), (_BASE + 30, 2)),
    )
    span_index = SpanIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        SpanProtocol(
            (_BASE + 31, 1),
            (_BASE + 31, 2),
            (_BASE + 31, 3),
            (_BASE + 31, 4),
        ),
        occurrence_index,
    )
    ctx.occurrence_index = occurrence_index
    ctx.span_index = span_index
    source = _source()
    scope = document_scope(source)
    occurrence = occurrence_index.record(
        source=source,
        raw_text=raw_text,
        scope=scope,
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
    ).occurrence
    span = span_index.ensure_ref(
        source=source,
        raw_text=raw_text,
        scope=scope,
        members=((0, 2),),
    )
    lesson = _lesson(source, span)
    decision = LanguageSemanticCourseDecision(
        minimal_instruction_identity((_BASE + 32, 1)),
        (_BASE + 32, 2),
        lesson,
    )
    mapper = _CourseMapper(decision)
    runtime = install_language_semantic_course_runtime(
        ctx, _protocol(mapper), query_protocol)
    runtime_scope = episode_scope(1, parent=scope)
    payload = InputPayload(
        [],
        SOURCE_BARE_TEXT,
        STAGE_TRAINING,
        modality=MODALITY_LANGUAGE,
        scope_identity=runtime_scope,
        source_ref=source,
        occurrence_scope_identity=scope,
    )
    item = CollectedItem(
        source=SOURCE_BARE_TEXT,
        source_ref=source,
        modality=MODALITY_LANGUAGE,
    )
    observed = ObserveResult(
        occurrence_refs=[occurrence],
        span_refs=[span],
    )
    return backend, ctx, runtime, mapper, item, payload, observed


def _full_generation_runtime(request):
    """为课程请求装配真实 G-00 至 G-04 测试链，并返回可检查的 owner。"""
    branch = request.goal.target_branch
    if branch is None:
        raise ValueError("正式全链 fixture 需要目标 LanguageBranch")
    structure = _structure_plan(branch, request=request)
    first, second = _templates(structure)
    family = (_BASE + 80, 1)
    alias = _alias_fixture(
        branch,
        (
            (first, representation_identity(family, (0x7532,))),
            (second, representation_identity(family, (0x4E59,))),
        ),
    )
    plan_protocol = _plan_protocol(_BASE + 81)
    content_protocol = _content_protocol()
    selector = AnswerContentSelector(
        content_protocol,
        _Policy(content_protocol),
    )
    structure_planner = GenerationStructurePlanner(
        _DynamicDiscourseMapper(),
        _DynamicPropositionMapper(),
        _DynamicSyntaxMapper(branch),
    )
    structure_protocol = GenerationStructureLayerProtocol(*tuple(
        minimal_instruction_identity((_BASE + 83, ordinal))
        for ordinal in range(1, 4)
    ))
    surface_protocol = _surface_protocol(_BASE + 84)
    surface_runtime = GenerationSurfaceRuntime(alias.runtime)
    registrations = (
        GenerationLayerRegistration(
            plan_protocol.stance_layer,
            GenerationStanceLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.content_layer,
            GenerationContentLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.discourse_layer,
            GenerationDiscourseLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.proposition_layer,
            GenerationPropositionLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.syntax_layer,
            GenerationSyntaxLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.surface_layer,
            GenerationSurfaceLayerResolver(
                plan_protocol,
                selector,
                structure_planner,
                _StaticSurfaceBuilder(surface_protocol),
                surface_runtime,
                commit=False,
            ),
        ),
    )
    renderer = UnicodeRepresentationRenderer(
        family,
        minimal_instruction_identity((_BASE + 85, 1)),
    )
    executor = TypedGenerationExecutor(
        GenerationPlanner(plan_protocol, registrations),
        renderer,
        surface_runtime,
    )
    postcheck_protocol = _postcheck_protocol()
    parser = _ExecutionParser()
    postcheck_mapper = _ProductionPostcheckMapper(parser=parser)
    postchecker = GenerationPostcheckRuntime(
        postcheck_protocol,
        parser,
        _StaticVerifier(VERDICT_SUPPORT, 1),
        _StaticVerifier(VERDICT_SUPPORT, 2),
    )
    runtime = ProductionGenerationRuntime(
        SemanticCourseGenerationRequestMapper(),
        executor,
        postcheck_mapper=postcheck_mapper,
        postchecker=postchecker,
    )
    return runtime, renderer, alias, postcheck_protocol, parser, postcheck_mapper


def _connector_generation_runtime(request, backend):
    """从单命题课程请求装配真实 connector、S-07、R-01 和 G-04 owner。"""
    branch = request.goal.target_branch
    if branch is None:
        raise ValueError("connector formal fixture 需要目标 LanguageBranch")
    if len(request.candidates) != 1:
        raise ValueError("connector formal fixture 只接受单命题请求")
    candidate = request.candidates[0]
    if len(candidate.proposition.bindings) != 1:
        raise ValueError("connector formal fixture 需要一个显式 Role binding")
    binding = candidate.proposition.bindings[0]
    if not isinstance(binding.filler, ObjectIdentity):
        raise ValueError("connector formal fixture 暂不接受嵌套命题 filler")

    content_protocol = _content_protocol()
    selector = AnswerContentSelector(
        content_protocol,
        _Policy(content_protocol),
    )
    selection = selector.select(request, ())
    graphs = _graphs(backend)
    engine = OrderHypothesisEngine(_learning_protocol(source_id=_BASE + 180))
    domain = replace(_domain(variant=180, slot_count=3), language=branch)
    promoter = OrderConstraintPromoter(
        engine,
        graphs.order_graph,
        graphs.lifecycle,
    )
    promotion = _active_plan(
        engine,
        promoter,
        domain,
        _pattern(domain, first=1, second=2, kind=180),
        event=180,
        instance=180,
    )
    connector = _connector(promotion, selection, binding.role)
    structure_planner = connector.structure_planner()

    family = (_BASE + 181, 1)
    predicate_representation = representation_identity(family, (0x7532,))
    filler_representation = representation_identity(family, (0x4E59,))
    alias = _alias_fixture(
        branch,
        (
            (candidate.proposition.predicate, predicate_representation),
            (binding.filler, filler_representation),
        ),
    )
    plan_protocol = _plan_protocol(_BASE + 182)
    structure_protocol = GenerationStructureLayerProtocol(*tuple(
        minimal_instruction_identity((_BASE + 183, ordinal))
        for ordinal in range(1, 4)
    ))
    surface_runtime = GenerationSurfaceRuntime(alias.runtime)
    surface_builder = connector.surface_request_builder(
        _execution_planner(graphs, promotion))
    registrations = (
        GenerationLayerRegistration(
            plan_protocol.stance_layer,
            GenerationStanceLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.content_layer,
            GenerationContentLayerResolver(plan_protocol, selector),
        ),
        GenerationLayerRegistration(
            plan_protocol.discourse_layer,
            GenerationDiscourseLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.proposition_layer,
            GenerationPropositionLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.syntax_layer,
            GenerationSyntaxLayerResolver(
                plan_protocol,
                structure_protocol,
                selector,
                structure_planner,
            ),
        ),
        GenerationLayerRegistration(
            plan_protocol.surface_layer,
            GenerationSurfaceLayerResolver(
                plan_protocol,
                selector,
                structure_planner,
                surface_builder,
                surface_runtime,
                commit=False,
            ),
        ),
    )
    renderer = UnicodeRepresentationRenderer(
        family,
        minimal_instruction_identity((_BASE + 184, 1)),
    )
    executor = TypedGenerationExecutor(
        GenerationPlanner(plan_protocol, registrations),
        renderer,
        surface_runtime,
    )
    postcheck_protocol = _postcheck_protocol()
    parser = _ExecutionParser()
    postcheck_mapper = _ProductionPostcheckMapper(parser=parser)
    postchecker = GenerationPostcheckRuntime(
        postcheck_protocol,
        parser,
        _StaticVerifier(VERDICT_SUPPORT, 1),
        _StaticVerifier(VERDICT_SUPPORT, 2),
    )
    runtime = ProductionGenerationRuntime(
        SemanticCourseGenerationRequestMapper(),
        executor,
        postcheck_mapper=postcheck_mapper,
        postchecker=postchecker,
    )
    return (
        runtime,
        renderer,
        alias,
        promotion,
        predicate_representation,
        filler_representation,
        graphs,
        connector,
    )


def test_semantic_course_forms_graph_bound_view_and_generation_request():
    """正式课程从 typed anchor 形成 S-02/S-03/G-00 产物且不读取词面序列。"""
    backend, ctx, runtime, mapper, item, payload, observed = _fixture()
    try:
        run = runtime.process(ctx, item, payload, observed)
        observed.semantic_course_run = run

        assert run.build is not None
        assert run.materialized is not None
        assert run.request is not None
        assert run.request.goal.proposition.template == (
            run.build.propositions[0].definition.proposition)
        assert run.request.goal.target_branch == (
            mapper.decision.lesson.target_branch)
        assert len(run.request.candidates) == 1
        assert run.request.candidates[0].state.status == EPISTEMIC_SUPPORTED
        assert run.request.candidates[0].evidence == run.evidence
        template_scope = runtime.template_scopes.read(
            run.build.propositions[0].definition.proposition)
        assert template_scope.definition.scope == (
            mapper.decision.lesson.template_scopes[0].scope)
        assert template_scope.definition.introduced_binders == ()
        assert len(template_scope.assertion_hashes) == 1
        assert mapper.inputs[0].occurrences == tuple(observed.occurrence_refs)
        assert mapper.inputs[0].spans == tuple(observed.span_refs)
        assert not hasattr(mapper.inputs[0], "tokens")
        assert not hasattr(mapper.inputs[0], "raw_text")

        production = SemanticCourseGenerationRequestMapper().build(
            ctx, item, payload, observed)
        assert isinstance(production, ProductionGenerationRequestDecision)
        assert production.request == run.request
    finally:
        backend.close()


def test_semantic_course_stage3_requires_typed_generation_before_write(
        monkeypatch):
    """S-02 已接通而 G-00 至 G-04 缺失时不得回退旧语言输出。"""
    backend, ctx, _runtime, mapper, item, _payload, _observed = _fixture()
    try:
        before = backend.snapshot()
        _forbid_legacy(monkeypatch)

        with pytest.raises(RuntimeError, match="S-02 语义课程.*typed generation"):
            DefaultRoundRunner().run_round_full(
                ctx,
                item,
                STAGE3_REWARD,
                1,
            )

        assert backend.snapshot() == before
        assert mapper.inputs == []
        assert ctx.work_memory.active_session_scope is None
        assert ctx.work_memory.active_document_scope is None
        assert ctx.work_memory.active_episode_scope is None
    finally:
        backend.close()


def test_semantic_course_observation_disables_legacy_sequence_writer(
        monkeypatch):
    """S-02 观察阶段也必须关闭 role_seq/token_seq 的兼容写入口。"""
    backend, ctx, _runtime, _mapper, item, _payload, _observed = _fixture()
    try:
        calls = []

        def observe_stub(*_args, **kwargs):
            """记录 round 传给 observe 的旧序写开关，避免本测试复用课程材料。"""
            calls.append(kwargs["write_legacy_language_sequences"])
            return ObserveResult()

        monkeypatch.setattr(round_module, "observe", observe_stub)
        monkeypatch.setattr(
            round_module,
            "_run_item_semantic_course",
            lambda *_args: None,
        )
        item = replace(item, tokens=["甲"], role_seq=[1])

        result = DefaultRoundRunner().run_round_full(
            ctx,
            item,
            STAGE_TRAINING,
            1,
        )

        assert result.episodes() == ()
        assert calls == [False]
    finally:
        backend.close()


def test_typed_owner_merges_all_legacy_verifiers_without_episode_leak(
        monkeypatch):
    """typed 语言 round 保留全部 route 结果，但不得向外泄漏标量 Episode。"""
    backend, ctx, _runtime, _mapper, item, _payload, _observed = _fixture()
    try:
        ctx.language_semantic_course_runtime = None
        generation = _SemanticNoRequestFactory(
            (_BASE + 41, 1)).build(ctx)

        class _NoRequestMapper:
            """在 verifier 完成后返回显式 typed 无请求，不依赖课程副作用。"""

            def build(self, *_args):
                """形成来源无关的无请求决定，确保测试继续进入 typed episode。"""
                return ProductionGenerationRequestDecision(
                    minimal_instruction_identity((_BASE + 41, 2)),
                    (_BASE + 41, 3),
                )

        generation._mapper = _NoRequestMapper()
        ctx.language_generation_runtime = generation
        item = replace(item, tokens=["甲", "乙"], raw_text="甲乙")
        runner = DefaultRoundRunner()
        calls = []
        monkeypatch.setattr(
            round_module,
            "select_verification_routes",
            lambda *_args: (
                VERIFY_ROUTE_COMPARISON,
                VERIFY_ROUTE_NUMERIC,
            ),
        )

        def route_adapter(
                route, _ctx, _item, _raw, _obs, route_round_id):
            """模拟仍返回旧 artifact 的领域 adapter，检查 typed 边界会截断它。"""
            calls.append(route)
            reward = 1 if route == VERIFY_ROUTE_NUMERIC else 0
            return RoundResult(episode=Episode(
                episode_id=route_round_id,
                run_id=route_round_id,
                reward=reward,
            ))

        monkeypatch.setattr(runner, "_run_verification_route", route_adapter)
        result = runner.run_round_full(
            ctx,
            item,
            STAGE3_REWARD,
            1,
        )

        assert calls == [VERIFY_ROUTE_NUMERIC, VERIFY_ROUTE_COMPARISON]
        report = result.verification_report
        assert report is not None
        assert [entry.verdict for entry in report.results] == [
            VERDICT_SUPPORT,
            VERDICT_REFUTE,
        ]
        typed = result.typed_episode
        assert isinstance(typed, TypedLanguageEpisode)
        assert typed.supplemental_verification is report
        assert [signal.verdict for signal in typed.signals] == [
            VERDICT_SUPPORT,
            VERDICT_REFUTE,
        ]
        assert result.episode is None
        assert result.episodes() == (typed,)
        assert all(isinstance(entry.artifact, RoundResult)
                   for entry in report.results)
        assert len(ctx.verification_reports) == 1
        assert all(entry.artifact is None
                   for entry in ctx.verification_reports[0].results)

        drifted = VerificationReport(
            report.read_only,
            (
                replace(report.results[0], source=_source(999)),
                report.results[1],
            ),
        )
        with pytest.raises(ValueError, match="supplemental verifier 来源漂移"):
            TypedLanguageEpisode.from_production(
                typed.round_id,
                typed.source,
                typed.scope,
                typed.production,
                read_only=typed.read_only,
                supplemental_verification=drifted,
            )
    finally:
        backend.close()


def test_semantic_course_requires_explicit_scope_for_every_proposition():
    """缺失空 scope 声明也必须在 S-02 图和 H-00 写入前失败。"""
    backend, ctx, runtime, mapper, item, payload, observed = _fixture()
    try:
        mapper.decision = replace(
            mapper.decision,
            lesson=replace(mapper.decision.lesson, template_scopes=()),
        )
        before_graph = backend.snapshot()
        before_ledger = runtime.ledger.state_key()

        with pytest.raises(ValueError, match="完整声明每个 Proposition scope"):
            runtime.process(ctx, item, payload, observed)

        assert backend.snapshot() == before_graph
        assert runtime.ledger.state_key() == before_ledger
    finally:
        backend.close()


def test_formal_round_runs_semantic_course_through_g00_to_g04_without_legacy(
        monkeypatch, tmp_path):
    """正式 round 只用同次语义课程请求完成 G-00..G-04，并保持来源/query 双 scope。"""
    backend, ctx, semantic_runtime, mapper, _, payload, observed = _fixture()
    alias = None
    try:
        source = payload.source_ref
        lesson = _dual_lesson(source, observed.occurrence_refs[0])
        mapper.decision = LanguageSemanticCourseDecision(
            minimal_instruction_identity((_BASE + 86, 1)),
            (_BASE + 86, 2),
            lesson,
        )
        episode_id = _EPISODE_HASHER.h63((STAGE3_REWARD, 1)) or 1
        formal_scope = episode_scope(
            episode_id,
            parent=document_scope(source),
        )
        prepared = semantic_runtime.process(
            ctx,
            CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                source=source.source_kind,
                source_ref=source,
                modality=MODALITY_LANGUAGE,
            ),
            replace(payload, scope_identity=formal_scope),
            observed,
        )
        assert prepared.request is not None
        runtime, renderer, alias, postcheck_protocol, parser, postcheck_mapper = (
            _full_generation_runtime(prepared.request)
        )
        ctx.language_generation_runtime = runtime
        _forbid_legacy(monkeypatch)
        item = CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            source=source.source_kind,
            source_ref=source,
            modality=MODALITY_LANGUAGE,
        )
        before_precedes = tuple(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES}))
        before_def_array = tuple(backend.select("def_array", where=None))
        before_alias = alias.runtime.state_key()

        result = DefaultRoundRunner().run_round_full(
            ctx,
            item,
            STAGE3_REWARD,
            1,
        )

        production = result.output
        course_run = ctx.language_semantic_course_reports[-1]
        assert result.episode is None
        assert result.dag_path is None
        assert production.decision.request == course_run.request
        assert production.execution is not None
        assert production.complete
        assert all(layer.executed for layer in production.execution.plan.layers)
        assert all(
            layer.outcome == production.execution.plan.protocol.complete
            for layer in production.execution.plan.layers
        )
        assert renderer.text(production.execution.rendered) == "甲乙"
        assert production.postcheck is not None
        assert production.postcheck_complete is True
        typed_episode = result.typed_episode
        assert isinstance(typed_episode, TypedLanguageEpisode)
        assert result.episodes() == (typed_episode,)
        assert typed_episode.production is production
        assert typed_episode.generation_complete is True
        assert typed_episode.postcheck_complete is True
        assert len(typed_episode.signals) == 6
        assert tuple(
            (item.dimension, item.verifier)
            for item in typed_episode.signals
        ) == tuple(
            (item.dimension, item.verifier)
            for item in production.postcheck.report.results
        )
        assert not hasattr(typed_episode, "reward")
        with MetricsCollector(str(tmp_path / "typed-metrics.jsonl")) as metrics:
            measured = metrics.record_round(
                1,
                STAGE3_REWARD,
                result.episodes(),
                graph_size=1,
                causes_coverage=0,
                promote_count=0,
                oov_promote_count=0,
            )
        assert measured.episode_count == 0
        assert measured.reward_pos == 0
        assert measured.conduction_rate == 0
        assert measured.typed_episode_count == 1
        assert measured.typed_generation_complete == 1
        assert measured.typed_postcheck_complete == 1
        assert sum((
            measured.typed_signal_support,
            measured.typed_signal_refute,
            measured.typed_signal_unknown,
            measured.typed_signal_conflicted,
            measured.typed_signal_not_applicable,
            measured.typed_signal_applicability_unknown,
        )) == 6
        class _FixedTypedRunner:
            """让 preflight 消费同一 typed episode，隔离重复 generation 干扰。"""

            def run_round_many(self, *_args):
                """按 preflight 调度协议返回不可标量化 episode。"""
                return (typed_episode,)

        preflight = _pre_flight_impl(
            ctx,
            [item],
            rounds=1,
            runner=_FixedTypedRunner(),
        )
        assert preflight.metrics_signal is True
        assert preflight.reward_gate_ok is True
        assert preflight.detail["conduction_rate"] == 0
        assert preflight.detail["legacy_episode_count"] == 0
        assert preflight.detail["typed_episode_count"] == 1
        assert preflight.detail["typed_generation_complete"] == 1
        assert preflight.detail["typed_postcheck_complete"] == 1
        assert preflight.detail["typed_signal_count"] == 6
        capability_result = FormalTrainResult(
            run_id="typed-only",
            episodes=[typed_episode],
        )
        assert project_layer0(capability_result)["total"] == 0
        assert all(
            cell["total"] == 0
            for dimension in project_g_attribution(capability_result).values()
            for cell in dimension.values()
        )
        assert production.postcheck.report.dimension_results(
            postcheck_protocol.source_dimension)
        assert production.decision.request.goal.source == source
        assert production.decision.request.goal.scope == query_scope(
            1, parent=formal_scope)
        assert all(
            candidate.source == source
            and candidate.scope == production.decision.request.goal.scope
            for candidate in production.decision.request.candidates
        )
        assert alias.runtime.state_key() != before_alias
        assert parser.calls == 1
        assert postcheck_mapper.calls == 1
        assert tuple(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES})) == before_precedes
        assert tuple(backend.select("def_array", where=None)) == before_def_array
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.active_generation_scope is None
    finally:
        if alias is not None:
            alias.close()
        backend.close()


def test_formal_round_routes_semantic_course_through_connector_s07_and_r01(
        monkeypatch):
    """单命题正式 round 经 connector 消费 active S-07，并原子提交 R-01 Use。"""
    backend, ctx, semantic_runtime, mapper, _, payload, observed = _fixture()
    alias = None
    try:
        source = payload.source_ref
        episode_id = _EPISODE_HASHER.h63((STAGE3_REWARD, 1)) or 1
        formal_scope = episode_scope(
            episode_id,
            parent=document_scope(source),
        )
        mapper.decision = LanguageSemanticCourseDecision(
            minimal_instruction_identity((_BASE + 185, 1)),
            (_BASE + 185, 2),
            _lesson(source, observed.occurrence_refs[0]),
        )
        item = CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            source=source.source_kind,
            source_ref=source,
            modality=MODALITY_LANGUAGE,
        )
        prepared = semantic_runtime.process(
            ctx,
            item,
            replace(payload, scope_identity=formal_scope),
            observed,
        )
        assert prepared.request is not None
        (
            runtime,
            renderer,
            alias,
            promotion,
            predicate_representation,
            filler_representation,
            _connector_graphs,
            _connector_owner,
        ) = _connector_generation_runtime(prepared.request, backend)
        ctx.language_generation_runtime = runtime
        _forbid_legacy(monkeypatch)
        before_alias = alias.runtime.state_key()
        before_precedes = tuple(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES}))
        before_def_array = tuple(backend.select("def_array", where=None))

        result = DefaultRoundRunner().run_round_full(
            ctx,
            item,
            STAGE3_REWARD,
            1,
        )

        production = result.output
        course_run = ctx.language_semantic_course_reports[-1]
        assert production.decision.request == course_run.request
        assert production.execution is not None
        assert production.complete is True
        assert production.postcheck_complete is True
        assert isinstance(result.typed_episode, TypedLanguageEpisode)
        assert result.episodes() == (result.typed_episode,)
        surface = production.execution.surface
        assert surface is not None
        structure = surface.preview.request.structure
        planned = structure.propositions.propositions
        assert len(planned) == 1
        candidate = course_run.request.candidates[0]
        assert planned[0].proposition == candidate.proposition
        assert planned[0].evidence == candidate.evidence
        sentence_execution = surface.preview.request.execution.sentences[0]
        assert tuple(
            item.constraint.definition.constraint
            for item in sentence_execution.active_constraints
        ) == (promotion.constraint.constraint,)
        binding = candidate.proposition.bindings[0]
        emitted_fillers = tuple(
            value.filler
            for _, value in surface.preview.request.ordered_values()
            if value.filler in {
                candidate.proposition.predicate,
                binding.filler,
            }
        )
        assert emitted_fillers == (
            binding.filler,
            candidate.proposition.predicate,
        )
        assert production.representations == (
            filler_representation,
            predicate_representation,
        )
        assert renderer.text(production.execution.rendered) == "乙甲"
        assert len(surface.adoptions) == 2
        assert alias.runtime.state_key() != before_alias
        assert tuple(backend.select(
            "edge", where={"edge_type": EDGE_PRECEDES})) == before_precedes
        assert tuple(backend.select("def_array", where=None)) == before_def_array
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.active_generation_scope is None
    finally:
        if alias is not None:
            alias.close()
        backend.close()


def test_replayed_lesson_is_graph_idempotent_and_keeps_full_active_evidence():
    """重复课程不新增图事件，G-00 请求仍携带账本中的完整 active Evidence。"""
    backend, ctx, runtime, _, item, payload, observed = _fixture()
    try:
        first = runtime.process(ctx, item, payload, observed)
        graph_after_first = backend.snapshot()
        ledger_after_first = runtime.ledger.state_key()

        second = runtime.process(ctx, item, payload, observed)

        assert backend.snapshot() == graph_after_first
        assert runtime.ledger.state_key() == ledger_after_first
        assert second.request.candidates[0].evidence == first.evidence
        assert second.request.stable_key() == first.request.stable_key()
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["anchor", "evidence"])
def test_invalid_course_mapping_fails_before_semantic_graph_or_ledger_write(
        failure):
    """外部 anchor 或局部 Evidence 归属错误时不得留下部分命题与 H-00 事件。"""
    backend, ctx, runtime, mapper, item, payload, observed = _fixture()
    try:
        baseline_graph = backend.snapshot()
        baseline_ledger = runtime.ledger.state_key()
        lesson = mapper.decision.lesson
        if failure == "anchor":
            lesson = replace(lesson, root_anchor=observed.occurrence_refs[0])
            observed.occurrence_refs.clear()
        else:
            bad_local = LocalSemanticRef(OBJECT_PROPOSITION, (999,))
            lesson = replace(
                lesson,
                evidence=(replace(
                    lesson.evidence[0], proposition=bad_local),),
            )
        mapper.decision = replace(mapper.decision, lesson=lesson)

        with pytest.raises(ValueError, match=(
                "root anchor" if failure == "anchor" else "plan 外")):
            runtime.process(ctx, item, payload, observed)

        assert backend.snapshot() == baseline_graph
        assert runtime.ledger.state_key() == baseline_ledger
    finally:
        backend.close()


def test_semantic_course_protocol_clones_mapper_and_ledger_without_sharing():
    """V-06 克隆复用图协议但课程状态和 H-00 账本互不共享。"""
    backend, ctx, runtime, _, item, payload, observed = _fixture()
    clone_backend = DictBackend()
    try:
        runtime.process(ctx, item, payload, observed)
        clone_ctx = make_train_context(clone_backend)
        clone_ctx.occurrence_index = OccurrenceIndex(
            clone_ctx.graph_ontology,
            clone_ctx.scoped_identity_store,
            ctx.occurrence_index.protocol,
        )
        clone_ctx.span_index = SpanIndex(
            clone_ctx.graph_ontology,
            clone_ctx.scoped_identity_store,
            ctx.span_index.protocol,
            clone_ctx.occurrence_index,
        )
        cloned = runtime.clone_for_context(clone_ctx)

        assert cloned.state_key() == runtime.state_key()
        assert cloned.ledger is not runtime.ledger
        assert cloned.protocol.mapper is not runtime.protocol.mapper
    finally:
        backend.close()
        clone_backend.close()


def test_v06_rebuilds_typed_owner_and_read_only_no_request_never_executes():
    """评测 owner 与来源分离，课程不重放 Evidence，typed owner 仍保持在位。"""
    backend, ctx, runtime, _, item, payload, observed = _fixture()
    try:
        runtime.process(ctx, item, payload, observed)
        factory = _SemanticNoRequestFactory()
        install_production_generation_runtime(ctx, factory)
        host_ledger = runtime.ledger.state_key()

        with isolated_evaluation(ctx, label="semantic-no-request") as eval_ctx:
            assert eval_ctx.language_generation_runtime is not None
            assert eval_ctx.language_generation_runtime_factory is not factory
            eval_document = make_scope(
                SCOPE_DOCUMENT,
                81,
                owner=eval_ctx.scope_owner,
            )
            eval_episode = episode_scope(82, parent=eval_document)
            eval_payload = replace(payload, scope_identity=eval_episode)
            eval_observed = replace(observed, semantic_course_run=None)
            run = eval_ctx.language_semantic_course_runtime.process(
                eval_ctx, item, eval_payload, eval_observed)
            eval_observed.semantic_course_run = run
            assert run.input_value.source == payload.source_ref
            assert run.input_value.runtime_scope.owner == eval_ctx.scope_owner
            assert run.decision.lesson is None
            assert run.request is None
            assert eval_ctx.language_semantic_course_runtime.ledger.state_key() == (
                host_ledger)

            work_memory = eval_ctx.work_memory
            work_memory.begin_document(eval_document)
            work_memory.begin_episode(eval_episode, round_id=82)
            try:
                production = eval_ctx.language_generation_runtime.run(
                    eval_ctx, item, eval_payload, eval_observed)
                assert production.decision.request is None
                assert production.execution is None
            finally:
                work_memory.end_episode()
                work_memory.end_document()
    finally:
        backend.close()


def test_v06_read_only_query_recovers_ground_request_and_never_uses_legacy(
        monkeypatch):
    """评测从训练图和 H-00 恢复请求，保留来源并使用独立 query owner。"""
    query_mapper = _ReadQueryMapper()
    backend, ctx, runtime, mapper, _, payload, observed = _fixture(
        LanguageSemanticQueryProtocol(query_mapper))
    try:
        mapper.decision = replace(
            mapper.decision,
            lesson=_lesson(payload.source_ref, observed.occurrence_refs[0]),
        )
        training = runtime.process(
            ctx,
            CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                source=payload.source_ref.source_kind,
                source_ref=payload.source_ref,
                modality=MODALITY_LANGUAGE,
            ),
            payload,
            observed,
        )
        assert training.request is not None
        host_ledger = runtime.ledger.state_key()
        install_production_generation_runtime(ctx, _RecoveringFailFactory())
        _forbid_legacy(monkeypatch)

        with isolated_evaluation(ctx, label="semantic-ground-recovery") as eval_ctx:
            before_precedes = tuple(eval_ctx.backend.select(
                "edge", where={"edge_type": EDGE_PRECEDES}))
            before_def_array = tuple(eval_ctx.backend.select(
                "def_array", where=None))
            item = CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                source=payload.source_ref.source_kind,
                source_ref=payload.source_ref,
                modality=MODALITY_LANGUAGE,
            )

            result = DefaultRoundRunner().run_round_full(
                eval_ctx,
                item,
                STAGE3_REWARD,
                3,
            )

            course_run = eval_ctx.language_semantic_course_reports[-1]
            assert course_run.decision.lesson is None
            assert course_run.recovery is not None
            assert len(course_run.recovery.input_value.candidates) == 1
            assert course_run.request is not None
            assert course_run.request.goal.source == payload.source_ref
            assert course_run.request.goal.scope.owner == eval_ctx.scope_owner
            assert all(
                evidence.hypothesis.observation == payload.source_ref
                for candidate in course_run.request.candidates
                for evidence in candidate.evidence
            )
            assert result.episode is None
            assert result.dag_path is None
            assert result.output.execution is not None
            assert result.output.complete is False
            assert result.output.decision.request == course_run.request
            assert eval_ctx.language_semantic_course_runtime.ledger.state_key() == (
                host_ledger)
            assert tuple(eval_ctx.backend.select(
                "edge", where={"edge_type": EDGE_PRECEDES})) == before_precedes
            assert tuple(eval_ctx.backend.select(
                "def_array", where=None)) == before_def_array

        assert runtime.ledger.state_key() == host_ledger
        assert query_mapper.inputs == []
    finally:
        backend.close()


def test_v06_scoped_variable_query_recovers_request_without_writing(
        monkeypatch):
    """完整 template scope 允许 Variable 请求，且执行与恢复都不写回知识。"""
    query_mapper = _NonGroundSelectingQueryMapper()
    backend, ctx, runtime, mapper, _, payload, observed = _fixture(
        LanguageSemanticQueryProtocol(query_mapper))
    try:
        mapper.decision = replace(
            mapper.decision,
            lesson=_non_ground_lesson(
                payload.source_ref, observed.occurrence_refs[0]),
        )
        training = runtime.process(
            ctx,
            CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                source=payload.source_ref.source_kind,
                source_ref=payload.source_ref,
                modality=MODALITY_LANGUAGE,
            ),
            payload,
            observed,
        )
        assert training.request is not None
        host_ledger = runtime.ledger.state_key()
        install_production_generation_runtime(ctx, _RecoveringFailFactory())
        _forbid_legacy(monkeypatch)

        with isolated_evaluation(ctx, label="semantic-non-ground") as eval_ctx:
            eval_document = make_scope(
                SCOPE_DOCUMENT,
                94,
                owner=eval_ctx.scope_owner,
            )
            eval_episode = episode_scope(95, parent=eval_document)
            eval_payload = replace(payload, scope_identity=eval_episode)
            eval_observed = replace(observed, semantic_course_run=None)
            item = CollectedItem(
                tokens=["甲", "乙"],
                raw_text="甲乙",
                source=payload.source_ref.source_kind,
                source_ref=payload.source_ref,
                modality=MODALITY_LANGUAGE,
            )
            before_graph = eval_ctx.backend.snapshot()
            course_run = eval_ctx.language_semantic_course_runtime.process(
                eval_ctx, item, eval_payload, eval_observed)
            eval_observed.semantic_course_run = course_run
            assert course_run.recovery is not None
            assert len(course_run.recovery.input_value.candidates) == 1
            assert course_run.recovery.input_value.candidates[0].ground is False
            assert course_run.recovery.input_value.candidates[0].recoverable is True
            nested = training.build.propositions[1].definition.proposition
            restored_scope = (
                eval_ctx.language_semantic_course_runtime.template_scopes.read(
                    nested)
            )
            assert restored_scope.definition.introduced_binders
            assert course_run.recovery.decision.goal is not None
            assert course_run.request is not None
            work_memory = eval_ctx.work_memory
            work_memory.begin_document(eval_document)
            work_memory.begin_episode(eval_episode, round_id=95)
            try:
                production = eval_ctx.language_generation_runtime.run(
                    eval_ctx, item, eval_payload, eval_observed)
            finally:
                work_memory.end_episode()
                work_memory.end_document()
            assert production.decision.request == course_run.request
            assert production.execution is not None
            assert production.complete is False
            assert eval_ctx.language_semantic_course_runtime.ledger.state_key() == (
                host_ledger)
            assert eval_ctx.backend.snapshot() == before_graph

        assert runtime.ledger.state_key() == host_ledger
        assert query_mapper.inputs == []
    finally:
        backend.close()


def test_v06_rejects_generation_factory_without_clone_and_state_protocol():
    """typed owner 缺评测重建协议时必须分型失败，不能共享或回退 legacy。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        install_production_generation_runtime(
            ctx,
            _MissingEvaluationProtocolFactory(),
        )

        with pytest.raises(EvaluationIsolationError, match="clone/state"):
            with isolated_evaluation(ctx, label="missing-generation-protocol"):
                raise AssertionError("缺协议的 typed owner 不得进入评测沙箱")
    finally:
        backend.close()


def test_v06_rejects_generation_factory_clone_configuration_drift():
    """评测 factory 改变任一装配键时必须在 build 前失败。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        install_production_generation_runtime(
            ctx,
            _DriftingEvaluationFactory(),
        )

        with pytest.raises(EvaluationIsolationError, match="改变了装配状态"):
            with isolated_evaluation(ctx, label="drifting-generation-factory"):
                raise AssertionError("配置漂移的 typed owner 不得进入评测沙箱")
    finally:
        backend.close()
