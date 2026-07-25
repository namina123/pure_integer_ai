"""L-05B2B 默认 connector 课程和真实 production builder 对抗测试。"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionSelector,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    semantic_source,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    document_scope,
    episode_scope,
    make_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
    representation_parts,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseGraph,
    RelationUseGraphProtocol,
    RelationUseOwner,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderConsumer,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import SpanProtocol
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
)
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRequestDecision,
    ProductionGenerationRun,
    install_production_generation_runtime,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.generation_postcheck_course import (
    GenerationPostcheckRuntimeBinding,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.language_generation_connector_factory import (
    DefaultLanguageConnectorProductionRuntimeBuilder,
    LanguageConnectorProductionComponents,
    LanguageConnectorProductionFactory,
)
from pure_integer_ai.experiments.language_generation_connector_stage4 import (
    LanguageConnectorSignalRoute,
    LanguageConnectorStage4Policy,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.language_generation_connector import (
    BoundPropositionDiscourseDeclaration,
    BoundPropositionDiscourseDeclarations,
    BoundPropositionDiscourseDependency,
    LanguageConnectorDiscourseDeclaration,
    StaticLanguageConnectorDiscourseDeclarations,
)
from pure_integer_ai.experiments.language_generation_course import (
    LanguageConnectorCourseRecognition,
    LanguageConnectorCourseTemplate,
    LanguageGenerationCourseError,
    LanguageGenerationCourseLoader,
    LanguageGenerationCourseManifest,
)
from pure_integer_ai.experiments.precedence_relation_runtime import (
    PrecedenceRelationProtocol,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_TABLE,
)

from tests.test_g02_generation_structure_plan import (
    _Policy,
    _content_protocol,
    _plan_protocol,
    _request,
    _selection,
)
from tests.test_g04_generation_postcheck import (
    _ExecutionParser,
    _ProductionPostcheckMapper,
    _StaticVerifier,
    _protocol as _postcheck_protocol,
    _source_requirements,
)
from tests.test_h05_language_candidate import _reveal, _source
from tests.test_l05b2b_language_generation_connector import (
    _connector,
    _execution_planner,
    _selection_with_role,
)
from tests.test_l05b2b_language_generation_connector_candidate import (
    _factory_fixture,
    _production_stage4_policy,
    _recognize,
    _replacement_template,
)
from tests.test_l05b2b_semantic_course_runtime import (
    _protocol as _semantic_course_protocol,
)
from tests.test_r06_precedence_relation_runtime import (
    _Course as _PrecedenceCourse,
    _domain as _precedence_domain,
)
from tests.test_r00_relation_closure import (
    _candidate_runtime as _relation_candidate_runtime,
    _projection_protocol as _relation_projection_protocol,
    _r01_fixture,
    _relation_protocol,
    _semantic_graph,
)
from tests.test_r01_alias_relation_course import _manifest as _relation_manifest
from tests.test_s07_structure_order import (
    _active_plan,
    _domain,
    _learning_protocol,
    _pattern,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)


_BASE = 16600


def _relation_use_protocol(variant: int) -> RelationUseGraphProtocol:
    """按组件版本构造注入式 PH2 Core Use 图协议。"""
    identities = tuple(
        concept_identity((_BASE + 700 + variant, ordinal))
        for ordinal in range(1, 9)
    )
    return RelationUseGraphProtocol(
        *identities,
        (_BASE + 700 + variant, 9),
    )


def _connector_branches(manifest) -> tuple:
    """按完整身份返回课程 connector 模板覆盖的全部 LanguageBranch。"""
    return tuple(sorted(
        {item.template.language_branch for item in manifest.templates},
        key=lambda item: item.stable_key(),
    ))


class _ComponentFactory:
    """为每个目标 context 重建空 R-01 owner 和独立 G-01/G-04 组件。"""

    def __init__(
            self, alias_protocol, relation_schemas, *, variant: int,
            include_alias: bool = True,
            include_postcheck: bool = True,
            renderer_family: tuple[int, ...] | None = None) -> None:
        """冻结可克隆组件协议，并允许测试注入实际 Representation 表示族。"""
        self.alias_protocol = alias_protocol
        self.relation_schemas = tuple(relation_schemas)
        self.variant = variant
        self.include_alias = include_alias
        self.include_postcheck = include_postcheck
        self.renderer_family = (
            (_BASE + 300 + variant,)
            if renderer_family is None else tuple(renderer_family))
        self.components = []

    def build(self, ctx):
        """在当前图建立可选兼容 R-01 与完整 G-01/G-04 辅助组件。"""
        alias = None
        if self.include_alias:
            semantic_graph = _semantic_graph(ctx.graph_ontology)
            candidate_graph = CandidateProjectionGraph(
                ctx.graph_ontology,
                _relation_projection_protocol(),
            )
            candidate_runtime = _relation_candidate_runtime(candidate_graph)
            relation_protocol = _relation_protocol()
            consumer = ActiveRelationClosureConsumer(
                semantic_graph,
                candidate_graph,
                relation_protocol,
                self.relation_schemas,
                engine=candidate_runtime.engine,
            )
            closure = RelationClosureRuntime(
                candidate_runtime,
                semantic_graph,
                consumer,
                relation_protocol,
                RelationUseOwner(
                    RelationUseGraph(
                        ctx.graph_ontology,
                        _relation_use_protocol(self.variant),
                    ),
                    RelationUseWriteMetadata(
                        SOURCE_BARE_TEXT,
                        EPI_STRUCTURED,
                        content_version=self.variant,
                    ),
                ),
            )
            alias = AliasRelationRuntime(
                closure,
                AliasResolutionSelector(self.alias_protocol),
            )
        content = _content_protocol(_BASE + self.variant)
        parser = _ExecutionParser() if self.include_postcheck else None
        postcheck = None
        postcheck_mapper = None
        if parser is not None:
            postcheck = GenerationPostcheckRuntime(
                _postcheck_protocol(),
                parser,
                _StaticVerifier(VERDICT_SUPPORT, 1),
                _StaticVerifier(VERDICT_SUPPORT, 2),
            )
            postcheck_mapper = _ProductionPostcheckMapper(parser=parser)
        components = LanguageConnectorProductionComponents(
            AnswerContentSelector(content, _Policy(content)),
            _plan_protocol(_BASE + 100 + self.variant),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE + 200 + self.variant, ordinal))
                for ordinal in range(1, 4)
            )),
            alias,
            UnicodeRepresentationRenderer(
                self.renderer_family,
                minimal_instruction_identity(
                    (_BASE + 301 + self.variant,)),
            ),
            postcheck_mapper,
            postcheck,
        )
        self.components.append(components)
        return components

    def clone_for_evaluation(self):
        """复制不可变协议，运行组件只在后续 build 时新建。"""
        return _ComponentFactory(
            self.alias_protocol,
            self.relation_schemas,
            variant=self.variant,
            include_alias=self.include_alias,
            include_postcheck=self.include_postcheck,
            renderer_family=self.renderer_family,
        )

    def state_key(self):
        """返回 R-01、schema 和本测试版本键。"""
        return (
            self.alias_protocol,
            self.relation_schemas,
            self.variant,
            self.include_alias,
            self.include_postcheck,
            self.renderer_family,
            _relation_use_protocol(self.variant).stable_key(),
        )


class _PostcheckFactory:
    """为默认 builder 测试重建独立 G-04 mapper/parser/verifier owner。"""

    def __init__(self, variant: int, branches=()) -> None:
        self.variant = variant
        self._branches = tuple(branches)
        self.bindings = []

    def build(self, _ctx):
        """建立一套不共享 parser 状态的 G-04 production 绑定。"""
        parser = _ExecutionParser()
        binding = GenerationPostcheckRuntimeBinding(
            _ProductionPostcheckMapper(parser=parser),
            GenerationPostcheckRuntime(
                _postcheck_protocol(),
                parser,
                _StaticVerifier(VERDICT_SUPPORT, 1),
                _StaticVerifier(VERDICT_SUPPORT, 2),
            ),
        )
        self.bindings.append(binding)
        return binding

    def clone_for_evaluation(self):
        """复制配置但不共享已建立的 G-04 owner。"""
        return _PostcheckFactory(self.variant, self._branches)

    def branches(self):
        """返回当前测试 G-04 owner 声明覆盖的语言分支。"""
        return self._branches

    def state_key(self):
        """返回测试 G-04 factory 的固定配置键。"""
        return (
            _BASE + 600,
            self.variant,
            tuple(item.stable_key() for item in self._branches),
        )


class _PostcheckLoader:
    """给 formal 入口测试提供独立 G-04 factory 和版本报告。"""

    def __init__(self, factory, report) -> None:
        self.factory = factory
        self.report = report

    def load(self, _ctx):
        """返回已预备的测试 G-04 课程装配结果。"""
        return SimpleNamespace(factory=self.factory, report=self.report)


class _UnusedSemanticCourseMapper:
    """为无训练轮次的默认入口测试提供可克隆 semantic 课程协议。"""

    def map(self, _input_value):
        """启动装配不应读取语义课程输入。"""
        raise AssertionError("无训练轮次时不得调用 semantic course mapper")

    def clone_for_evaluation(self):
        """返回不共享运行状态的同配置 mapper。"""
        return _UnusedSemanticCourseMapper()

    def state_key(self):
        """返回测试 mapper 的固定协议键。"""
        return (_BASE + 500,)


def _course_fixture(backend, *, variant: int, support: bool = True):
    """建立已有 active S-07、但 connector 尚未写图的版本化课程。"""
    graphs, promotion, connector, predicates, candidates = _factory_fixture(
        backend,
        variant=variant,
    )
    execution = _execution_planner(graphs, promotion)
    graphs.context.precedence_relation_runtime = SimpleNamespace(
        order_graph=graphs.order_graph,
        lifecycle=graphs.lifecycle,
        consumer=execution._consumer,
    )
    template = connector.registry.templates[0]
    observation = _source(_BASE + variant)
    revealed = _reveal(
        observation,
        event_key=(_BASE + 1, variant),
        supported=(template.connector,) if support else (
            concept_identity((_BASE + 2, variant)),),
    )
    entry = LanguageConnectorCourseTemplate(
        template,
        (_source(_BASE + 10 + variant), _source(_BASE + 20 + variant)),
        promotion.constraint.scope,
        1,
        (LanguageConnectorCourseRecognition(
            template.connector,
            (template.connector,),
            revealed,
            10,
            11,
            12,
        ),),
    )
    manifest = LanguageGenerationCourseManifest(
        1,
        (_BASE, variant),
        predicates,
        candidates.learning.graph.protocol,
        candidates.protocol,
        candidates.learning.engine.protocol,
        candidates.learning.verifier.protocol,
        candidates.learning.metadata,
        connector.registry.value_protocol,
        connector.surface_protocol,
        connector.runtime_policy,
        _production_stage4_policy(variant=variant),
        (entry,),
    )
    return graphs, manifest


def _matching_execution_course_fixture(
        backend, relation_fixture, *, variant: int):
    """构造与版本化 R-01 realizes 事实同键的 connector 课程和单命题请求。"""
    graphs, promotion, _connector_value, predicates, candidates = (
        _factory_fixture(backend, variant=variant)
    )
    graphs.context.precedence_relation_runtime = SimpleNamespace(
        order_graph=graphs.order_graph,
        lifecycle=graphs.lifecycle,
        consumer=_execution_planner(graphs, promotion)._consumer,
    )
    role = promotion.slots[2].role
    selection = _selection_with_role(
        relation_fixture.objects["branch"],
        role,
        relation_fixture.objects["b"],
    )
    selected = selection.request.candidates[0]
    proposition = replace(
        selected.proposition,
        predicate=relation_fixture.objects["a"],
    )
    request = GenerationPlanningRequest(
        replace(
            selection.request.goal,
            proposition=proposition,
            target_branch=relation_fixture.objects["branch"],
        ),
        (replace(selected, proposition=proposition),),
    )
    selection, _unused_first, _unused_second = _selection(request)
    connector = _connector(promotion, selection, role)
    template = connector.registry.templates[0]
    observation = _source(_BASE + variant)
    entry = LanguageConnectorCourseTemplate(
        template,
        (_source(_BASE + 10 + variant), _source(_BASE + 20 + variant)),
        promotion.constraint.scope,
        1,
        (LanguageConnectorCourseRecognition(
            template.connector,
            (template.connector,),
            _reveal(
                observation,
                event_key=(_BASE + 1, variant),
                supported=(template.connector,),
            ),
            10,
            11,
            12,
        ),),
    )
    postcheck_protocol = _postcheck_protocol()
    base_policy = _production_stage4_policy(variant=variant)
    stage4_policy = LanguageConnectorStage4Policy(
        (LanguageConnectorSignalRoute(
            postcheck_protocol.proposition_dimension,
            postcheck_protocol.proposition_verifier,
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        ),),
        base_policy.verifier_source,
        base_policy.event_namespace,
        base_policy.active_purpose,
        base_policy.trial_purpose,
    )
    manifest = LanguageGenerationCourseManifest(
        1,
        (_BASE, variant),
        predicates,
        candidates.learning.graph.protocol,
        candidates.protocol,
        candidates.learning.engine.protocol,
        candidates.learning.verifier.protocol,
        candidates.learning.metadata,
        connector.registry.value_protocol,
        connector.surface_protocol,
        connector.runtime_policy,
        stage4_policy,
        (entry,),
    )
    return graphs, manifest, request


def _matching_multi_execution_course_fixture(
        backend, relation_fixture, *, variant: int):
    """构造可跨 V-06 query scope 绑定的双命题默认课程和篇章声明。"""
    graphs, promotion, _connector_value, predicates, candidates = (
        _factory_fixture(backend, variant=variant)
    )
    graphs.context.precedence_relation_runtime = SimpleNamespace(
        order_graph=graphs.order_graph,
        lifecycle=graphs.lifecycle,
        consumer=_execution_planner(graphs, promotion)._consumer,
    )
    role = promotion.slots[2].role
    initial_request, _unused = _request(count=2)
    first_raw, second_raw = initial_request.candidates
    first_proposition = replace(
        first_raw.proposition,
        predicate=relation_fixture.objects["a"],
        bindings=(BoundRoleBinding(role, relation_fixture.objects["b"]),),
    )
    first = replace(first_raw, proposition=first_proposition)
    second_proposition = replace(
        second_raw.proposition,
        predicate=relation_fixture.objects["a"],
        bindings=(BoundRoleBinding(role, relation_fixture.objects["b"]),),
    )
    second = replace(second_raw, proposition=second_proposition)
    request = GenerationPlanningRequest(
        replace(
            initial_request.goal,
            proposition=first_proposition,
            target_branch=relation_fixture.objects["branch"],
        ),
        (first, second),
    )
    selection, _unused_first, _unused_second = _selection(request)
    first_candidate, second_candidate = selection.request.candidates
    connector = _connector(promotion, selection, role)
    first_template = connector.registry.templates[0]
    second_template = replace(
        _replacement_template(first_template, variant=_BASE + variant),
        sentence=structure_concept_identity((_BASE + 871, variant, 1)),
        proposition_structure=second_candidate.proposition.structure,
        predicate=second_candidate.proposition.predicate,
    )
    first_observation = _source(_BASE + 872 + variant)
    second_observation = _source(_BASE + 873 + variant)
    first_entry = LanguageConnectorCourseTemplate(
        first_template,
        (_source(_BASE + 874 + variant), _source(_BASE + 875 + variant)),
        promotion.constraint.scope,
        1,
        (LanguageConnectorCourseRecognition(
            first_template.connector,
            (first_template.connector,),
            _reveal(
                first_observation,
                event_key=(_BASE + 876, variant, 1),
                supported=(first_template.connector,),
            ),
            10,
            11,
            12,
        ),),
    )
    second_entry = LanguageConnectorCourseTemplate(
        second_template,
        (_source(_BASE + 877 + variant), _source(_BASE + 878 + variant)),
        promotion.constraint.scope,
        20,
        (LanguageConnectorCourseRecognition(
            second_template.connector,
            (second_template.connector,),
            _reveal(
                second_observation,
                event_key=(_BASE + 879, variant, 1),
                supported=(second_template.connector,),
            ),
            20,
            21,
            22,
        ),),
    )
    base_policy = _production_stage4_policy(variant=variant)
    stage4_policy = LanguageConnectorStage4Policy(
        (LanguageConnectorSignalRoute(
            _postcheck_protocol().proposition_dimension,
            _postcheck_protocol().proposition_verifier,
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        ),),
        base_policy.verifier_source,
        base_policy.event_namespace,
        base_policy.active_purpose,
        base_policy.trial_purpose,
    )
    discourse_declarations = BoundPropositionDiscourseDeclarations((
        BoundPropositionDiscourseDeclaration(
            (first_candidate.proposition, second_candidate.proposition),
            (BoundPropositionDiscourseDependency(
                first_candidate.proposition,
                second_candidate.proposition,
                structure_concept_identity((_BASE + 880, variant, 1)),
                minimal_instruction_identity((_BASE + 880, variant, 2)),
                (_BASE + 880, variant, 3),
            ),),
            _source(_BASE + 881 + variant),
            (_BASE + 880, variant, 4),
        ),
    ))
    runtime_policy = replace(
        connector.runtime_policy,
        templates=(
            connector.runtime_policy.templates[0],
            replace(
                connector.runtime_policy.templates[0],
                connector=second_template.connector,
            ),
        ),
    )
    manifest = LanguageGenerationCourseManifest(
        1,
        (_BASE, variant, 2),
        predicates,
        candidates.learning.graph.protocol,
        candidates.protocol,
        candidates.learning.engine.protocol,
        candidates.learning.verifier.protocol,
        candidates.learning.metadata,
        connector.registry.value_protocol,
        connector.surface_protocol,
        runtime_policy,
        stage4_policy,
        (first_entry, second_entry),
        discourse_declarations=discourse_declarations,
    )
    return graphs, manifest, request


def _request_in_query_scope(
        request: GenerationPlanningRequest,
        scope,
        ) -> GenerationPlanningRequest:
    """只替换本次运行边界，保留知识命题、Evidence 与目标来源不变。"""
    return GenerationPlanningRequest(
        replace(request.goal, scope=scope),
        tuple(replace(item, scope=scope) for item in request.candidates),
    )


def _episode_from_installed_runtime(
        runtime, request: GenerationPlanningRequest, *,
        episode_scope_value, round_id: int) -> TypedLanguageEpisode:
    """直接执行已由 factory 装配的六层 runtime，并生成同次 G-04 episode。"""
    execution = runtime._executor.execute(request)
    if not execution.complete:
        raise AssertionError("默认 production factory 未形成完整 typed surface")
    postchecker = runtime._postchecker
    if postchecker is None or not isinstance(postchecker.parser, _ExecutionParser):
        raise AssertionError("默认 production factory 缺少可登记的 G-04 parser")
    postchecker.parser.record(execution)
    postcheck = postchecker.run(GenerationPostcheckRequest(
        execution,
        (),
        _source_requirements(execution),
    ))
    production = ProductionGenerationRun(
        ProductionGenerationRequestDecision(
            minimal_instruction_identity((_BASE + 850, round_id, 1)),
            (_BASE + 850, round_id, 2),
            request,
        ),
        execution,
        postcheck,
    )
    return TypedLanguageEpisode.from_production(
        round_id,
        request.goal.source,
        episode_scope_value,
        production,
        read_only=False,
    )


def _install_cloned_precedence_runtime(graphs, target_ctx) -> None:
    """在 V-06 目标图重建 S-07/R-06 facade，避免复用宿主运行对象。"""
    source_ontology = graphs.context.graph_ontology
    target_ontology = target_ctx.graph_ontology
    order_identities = tuple(
        source_ontology.identity_of(item)
        for item in graphs.order_graph.predicates.refs()
    )
    cloned_order = StructureOrderGraph(
        target_ontology,
        StructureOrderGraphPredicates(*tuple(
            target_ontology.resolve(item)
            for item in order_identities
        )),
    )
    source_lifecycle = graphs.lifecycle
    lifecycle_identities = tuple(
        source_ontology.identity_of(item)
        for item in source_lifecycle.protocol.predicate_refs()
    )
    cloned_lifecycle = StructureOrderLifecycleGraph(
        cloned_order,
        StructureOrderLifecycleProtocol(
            *tuple(
                target_ontology.resolve(item)
                for item in lifecycle_identities
            ),
            *source_lifecycle.protocol.state_identities(),
            *source_lifecycle.protocol.kind_identities(),
            source_lifecycle.protocol.event_namespace_key,
        ),
    )
    source_consumer = graphs.context.precedence_relation_runtime.consumer
    target_ctx.precedence_relation_runtime = SimpleNamespace(
        order_graph=cloned_order,
        lifecycle=cloned_lifecycle,
        consumer=StructureOrderConsumer(
            cloned_lifecycle,
            source_consumer.resolver,
            source_consumer.protocol,
        ),
    )


def _resumed_precedence_protocol(graphs) -> PrecedenceRelationProtocol:
    """用已落 S-07 图的开放身份重建正式 R-06 facade。"""
    ontology = graphs.context.graph_ontology
    lifecycle = graphs.lifecycle.protocol
    consumer = graphs.context.precedence_relation_runtime.consumer
    return PrecedenceRelationProtocol(
        _learning_protocol(),
        tuple(
            ontology.identity_of(item)
            for item in graphs.order_graph.predicates.refs()
        ),
        tuple(
            ontology.identity_of(item)
            for item in lifecycle.predicate_refs()
        ),
        (*lifecycle.state_identities(), *lifecycle.kind_identities()),
        lifecycle.event_namespace_key,
        consumer.protocol,
        provenance_kind=_BASE + 501,
        qualifiers=(_BASE + 502,),
    )


def test_course_loader_is_idempotent_and_restores_core_history():
    """同一内容锁重复加载零增长，新 context 从 Core 恢复同一 active 理论。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=301)
        loader = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        )
        first = loader.load(graphs.context)
        first_snapshot = backend.snapshot()
        first_training_rows = backend.count(TRAINING_CANDIDATE_EVENT_TABLE)

        second = loader.load(graphs.context)
        assert backend.snapshot() == first_snapshot
        assert second.report == first.report
        assert second.candidates.active_templates() == (
            manifest.templates[0].template,)
        assert first_training_rows > 0
        assert backend.count(MEMORY_EVENT_TABLE) == 0

        restarted = make_train_context(backend)
        ontology = restarted.graph_ontology
        order_identities = tuple(
            graphs.context.graph_ontology.identity_of(item)
            for item in graphs.order_graph.predicates.refs()
        )
        restarted_order = StructureOrderGraph(
            ontology,
            StructureOrderGraphPredicates(*tuple(
                ontology.resolve(item) for item in order_identities
            )),
        )
        restarted.precedence_relation_runtime = SimpleNamespace(
            order_graph=restarted_order)
        restored = loader.load(restarted)
        assert restored.report == first.report
        assert backend.snapshot() == first_snapshot
    finally:
        backend.close()


def test_course_locks_discourse_declarations_and_clones_scheduled_provider():
    """版本化课程必须内容锁声明读取器，并让 scheduled V-06 使用独立 provider。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        graphs, manifest = _course_fixture(backend, variant=315)
        declarations = StaticLanguageConnectorDiscourseDeclarations((
            LanguageConnectorDiscourseDeclaration(
                ((_BASE + 801, 315),),
                (),
                _source(_BASE + 802),
                (_BASE + 803, 315),
            ),
        ))
        configured = replace(
            manifest,
            discourse_declarations=declarations,
        )
        assert configured.sha256() != manifest.sha256()
        loaded = LanguageGenerationCourseLoader(
            configured,
            configured.sha256(),
        ).load(graphs.context)
        host = loaded.connector_factory.build(graphs.context)

        cloned_backend = clone_backend(backend)
        cloned_context = make_train_context(cloned_backend)
        cloned_factory = loaded.connector_factory.clone_for_evaluation()
        cloned = cloned_factory.build(cloned_context)

        assert host.connector.discourse_declarations is declarations
        assert cloned.connector.discourse_declarations is not declarations
        assert cloned.connector.discourse_declarations.state_key() == (
            declarations.state_key())
        assert cloned_factory.state_key() == loaded.connector_factory.state_key()
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def test_course_hash_fails_before_write_and_unknown_remains_forming():
    """内容锁漂移保持零写，独立 unknown 则只形成可隔离 trial 的 forming。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=302)
        baseline = backend.snapshot()
        with pytest.raises(LanguageGenerationCourseError, match="哈希漂移"):
            LanguageGenerationCourseLoader(
                manifest,
                "0" * 64,
            ).load(graphs.context)
        assert backend.snapshot() == baseline

        _other_graphs, inactive = _course_fixture(
            backend,
            variant=303,
            support=False,
        )
        loaded = LanguageGenerationCourseLoader(
            inactive,
            inactive.sha256(),
        ).load(_other_graphs.context)
        assert loaded.candidates.active_templates() == ()
        assert loaded.candidates.trial_template_hypotheses() == ((
            inactive.templates[0].template,
            loaded.candidates.learning.hypothesis_for_candidate(
                inactive.templates[0].template.connector),
        ),)
    finally:
        backend.close()


def test_course_allows_versioned_forming_template_without_recognition():
    """版本化理论可只登记 forming，loader 不得伪造 support 或 active 投影。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=314)
        entry = replace(manifest.templates[0], recognitions=())
        forming = replace(manifest, templates=(entry,))

        loaded = LanguageGenerationCourseLoader(
            forming,
            forming.sha256(),
        ).load(graphs.context)

        assert loaded.report.active_count == 0
        assert loaded.candidates.active_templates() == ()
        trials = loaded.candidates.trial_template_hypotheses()
        assert len(trials) == 1

        _recognize(
            loaded.candidates,
            trials[0][1],
            entry.template,
            source_id=950,
            event=950,
            stance="support",
        )
        resumed = LanguageGenerationCourseLoader(
            forming,
            forming.sha256(),
        ).load(graphs.context)
        assert resumed.candidates.active_templates() == (entry.template,)
        assert resumed.candidates.trial_template_hypotheses() == ()
    finally:
        backend.close()


def test_course_schema_and_runtime_policy_drift_are_rejected():
    """未知 schema 在加载期拒绝，理论与运行策略未双向覆盖时 manifest 不成立。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=305)
        unsupported = replace(manifest, schema_version=2)
        baseline = backend.snapshot()
        with pytest.raises(LanguageGenerationCourseError, match="版本不受支持"):
            LanguageGenerationCourseLoader(
                unsupported,
                unsupported.sha256(),
            ).load(graphs.context)
        assert backend.snapshot() == baseline

        policy_item = manifest.runtime_policy.templates[0]
        drifted_policy = replace(
            manifest.runtime_policy,
            templates=(replace(
                policy_item,
                connector=structure_concept_identity((_BASE + 400, 305)),
            ),),
        )
        with pytest.raises(ValueError, match="未双向覆盖"):
            replace(manifest, runtime_policy=drifted_policy)
    finally:
        backend.close()


def test_course_loader_supports_multiple_templates_with_shared_protocol():
    """同一版本化协议可加载多套理论，逻辑序并列也不得串扰候选历史。"""
    backend = DictBackend()
    try:
        graphs, first = _course_fixture(backend, variant=307)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=308, slot_count=3)
        promotion = _active_plan(
            engine,
            OrderConstraintPromoter(
                engine,
                graphs.order_graph,
                graphs.lifecycle,
            ),
            domain,
            _pattern(domain, first=1, second=2, kind=308),
            event=308,
            instance=308,
        )
        role = promotion.slots[2].role
        selection = _selection_with_role(
            domain.language,
            role,
            concept_identity((_BASE + 599, 1)),
        )
        second_connector_runtime = _connector(promotion, selection, role)
        second_template = second_connector_runtime.registry.templates[0]
        second_connector = structure_concept_identity((_BASE + 600, 1))
        second_template = replace(
            second_template,
            connector=second_connector,
            predicate=concept_identity((_BASE + 600, 2)),
            sentence=structure_concept_identity((_BASE + 600, 3)),
            constraint_set=structure_concept_identity((_BASE + 600, 4)),
            context_set=structure_concept_identity((_BASE + 600, 5)),
            bindings=tuple(
                replace(
                    item,
                    binding=structure_concept_identity(
                        (_BASE + 601, ordinal)),
                )
                for ordinal, item in enumerate(
                    second_template.bindings, start=1)
            ),
            surface=tuple(
                replace(
                    item,
                    directive=structure_concept_identity(
                        (_BASE + 602, ordinal, 1)),
                    prefix_route=structure_concept_identity(
                        (_BASE + 602, ordinal, 2)),
                )
                for ordinal, item in enumerate(
                    second_template.surface, start=1)
            ),
        )
        observation = _source(_BASE + 603)
        second_entry = LanguageConnectorCourseTemplate(
            second_template,
            (_source(_BASE + 604), _source(_BASE + 605)),
            promotion.constraint.scope,
            1,
            (LanguageConnectorCourseRecognition(
                second_connector,
                (second_connector,),
                _reveal(
                    observation,
                    event_key=(_BASE + 606, 1),
                    supported=(second_connector,),
                ),
                10,
                11,
                12,
            ),),
        )
        combined_policy = replace(
            first.runtime_policy,
            templates=(
                *first.runtime_policy.templates,
                replace(
                    second_connector_runtime.runtime_policy.templates[0],
                    connector=second_connector,
                ),
            ),
        )
        manifest = replace(
            first,
            course_version=(_BASE, 307, 308),
            runtime_policy=combined_policy,
            templates=(*first.templates, second_entry),
        )

        loaded = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)

        assert loaded.report.template_count == 2
        assert loaded.report.active_count == 2
        assert loaded.candidates.active_templates() == tuple(sorted(
            (item.template for item in manifest.templates),
            key=lambda item: item.connector.stable_key(),
        ))
        assert backend.count(MEMORY_EVENT_TABLE) == 0
    finally:
        backend.close()


def test_formal_train_rejects_partial_or_conflicting_default_course_config(
        tmp_path):
    """默认 loader/component 必须成对，且不能与直接 factory 或缺失上游并存。"""
    backend = DictBackend()
    try:
        base = FormalTrainConfig(
            str(tmp_path),
            "connector-course-config",
            active_training_stages=(),
            persist_graph_dump=False,
        )
        with pytest.raises(ValueError, match="必须成对配置"):
            formal_train(
                replace(base, language_generation_course_loader=object()),
                [],
                backend=backend,
            )
        with pytest.raises(ValueError, match="R-01 课程 loader 必须配套"):
            formal_train(
                replace(
                    base,
                    language_alias_relation_course_loader=object(),
                ),
                [],
                backend=backend,
            )
        with pytest.raises(ValueError, match="G-04 课程 loader 必须配套"):
            formal_train(
                replace(
                    base,
                    language_generation_postcheck_course_loader=object(),
                ),
                [],
                backend=backend,
            )
        with pytest.raises(ValueError, match="互斥"):
            formal_train(
                replace(
                    base,
                    language_generation_runtime_factory=object(),
                    language_generation_course_loader=object(),
                    language_generation_component_factory=object(),
                ),
                [],
                backend=backend,
            )
        with pytest.raises(ValueError, match="semantic course"):
            formal_train(
                replace(
                    base,
                    language_generation_course_loader=object(),
                    language_generation_component_factory=object(),
                ),
                [],
                backend=backend,
            )
    finally:
        backend.close()


def test_default_builder_installs_real_six_layer_and_stage4_owners():
    """默认入口不借测试 binding，真实组装 connector、S-07、R-01、G-04 和 stage4。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    try:
        graphs, manifest = _course_fixture(backend, variant=304)
        loaded = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=304,
        )
        factory = LanguageConnectorProductionFactory(
            loaded.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory),
            loaded.stage4_policy,
        )

        runtime = install_production_generation_runtime(
            graphs.context,
            factory,
        )
        host_components = component_factory.components[-1]
        assert len(runtime._executor._planner._registrations) == 6
        assert runtime._postchecker is not None
        assert graphs.context.language_generation_stage4_runtime is not None
        cloned_factory = factory.clone_for_evaluation()
        assert cloned_factory.state_key() == factory.state_key()

        cloned_backend = clone_backend(backend)
        try:
            cloned_ctx = make_train_context(cloned_backend)
            source_ontology = graphs.context.graph_ontology
            target_ontology = cloned_ctx.graph_ontology
            order_identities = tuple(
                source_ontology.identity_of(item)
                for item in graphs.order_graph.predicates.refs()
            )
            cloned_order = StructureOrderGraph(
                target_ontology,
                StructureOrderGraphPredicates(*tuple(
                    target_ontology.resolve(item)
                    for item in order_identities
                )),
            )
            source_lifecycle = graphs.lifecycle
            lifecycle_identities = tuple(
                source_ontology.identity_of(item)
                for item in source_lifecycle.protocol.predicate_refs()
            )
            cloned_lifecycle = StructureOrderLifecycleGraph(
                cloned_order,
                StructureOrderLifecycleProtocol(
                    *tuple(target_ontology.resolve(item)
                           for item in lifecycle_identities),
                    *source_lifecycle.protocol.state_identities(),
                    *source_lifecycle.protocol.kind_identities(),
                    source_lifecycle.protocol.event_namespace_key,
                ),
            )
            source_consumer = graphs.context.precedence_relation_runtime.consumer
            cloned_consumer = StructureOrderConsumer(
                cloned_lifecycle,
                source_consumer.resolver,
                source_consumer.protocol,
            )
            cloned_ctx.precedence_relation_runtime = SimpleNamespace(
                order_graph=cloned_order,
                lifecycle=cloned_lifecycle,
                consumer=cloned_consumer,
            )
            cloned_installation = cloned_factory.build_installation(cloned_ctx)
            cloned_components = (
                cloned_factory._runtime_builder._component_factory
                .components[-1]
            )
            assert cloned_installation.runtime is not runtime
            assert cloned_installation.stage4_runtime is not (
                graphs.context.language_generation_stage4_runtime)
            assert cloned_installation.stage4_runtime.candidates.definition_graph \
                .ontology is target_ontology
            assert cloned_components.alias is not host_components.alias
            assert cloned_components.postchecker is not host_components.postchecker
            assert cloned_components.postcheck_mapper is not (
                host_components.postcheck_mapper)
        finally:
            cloned_backend.close()
    finally:
        relation_fixture.close()
        backend.close()


def test_default_factory_v06_clone_executes_core_use_and_stage4_without_host_write():
    """默认 factory 在 V-06 clone 完整执行 G-00 至 G-04、R-01 Use 与 stage4。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    cloned_backend = None
    try:
        graphs, manifest, base_request = _matching_execution_course_fixture(
            backend,
            relation_fixture,
            variant=316,
        )
        loaded_connector = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)
        relation_manifest = _relation_manifest(variant=316)
        loaded_relation = AliasRelationCourseLoader(
            relation_manifest,
            relation_manifest.sha256(),
        ).load(graphs.context)
        representation_family, _content = representation_parts(
            relation_fixture.objects["rep1"])
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=316,
            include_alias=False,
            renderer_family=representation_family,
        )
        factory = LanguageConnectorProductionFactory(
            loaded_connector.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                loaded_relation.factory,
            ),
            loaded_connector.stage4_policy,
        )

        host_precedence = graphs.context.precedence_relation_runtime
        graphs.context.precedence_relation_runtime = None
        try:
            cloned_backend = clone_backend(backend)
            cloned_ctx = clone_train_context(
                graphs.context,
                cloned_backend,
                label="default-factory-core-use-success",
            )
        finally:
            graphs.context.precedence_relation_runtime = host_precedence
        if cloned_ctx.scope_owner is None:
            raise RuntimeError("V-06 clone 缺少独立评测 owner")
        _install_cloned_precedence_runtime(graphs, cloned_ctx)

        host_installation = factory.build_installation(graphs.context)
        cloned_factory = factory.clone_for_evaluation()
        cloned_installation = cloned_factory.build_installation(cloned_ctx)
        source = base_request.goal.source
        host_episode_scope = episode_scope(
            _BASE + 860,
            parent=document_scope(source),
        )
        host_request = _request_in_query_scope(
            base_request,
            query_scope(1, parent=host_episode_scope),
        )
        host_episode = _episode_from_installed_runtime(
            host_installation.runtime,
            host_request,
            episode_scope_value=host_episode_scope,
            round_id=_BASE + 861,
        )
        assert host_installation.stage4_runtime.apply((host_episode,)).complete
        host_before_clone_execution = backend.snapshot()

        eval_document = make_scope(
            SCOPE_DOCUMENT,
            _BASE + 862,
            owner=cloned_ctx.scope_owner,
        )
        eval_episode_scope = episode_scope(
            _BASE + 863,
            parent=eval_document,
        )
        eval_query_scope = query_scope(1, parent=eval_episode_scope)
        assert eval_query_scope.owner == cloned_ctx.scope_owner
        eval_request = _request_in_query_scope(base_request, eval_query_scope)
        eval_episode = _episode_from_installed_runtime(
            cloned_installation.runtime,
            eval_request,
            episode_scope_value=eval_episode_scope,
            round_id=_BASE + 864,
        )
        clone_training_rows_before_stage4 = cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE)
        report = cloned_installation.stage4_runtime.apply((eval_episode,))
        clone_training_rows_after_stage4 = cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE)

        assert eval_episode.production.complete is True
        assert report.complete is True
        assert report.changed_count == 1
        assert clone_training_rows_after_stage4 > clone_training_rows_before_stage4
        assert cloned_installation.stage4_runtime.apply((eval_episode,)) == report
        assert cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE) == clone_training_rows_after_stage4
        assert backend.snapshot() == host_before_clone_execution
        clone_alias = cloned_factory._runtime_builder._relation_factory.build(
            cloned_ctx)
        use_owner = clone_alias.closure.use_owner
        assert use_owner is not None
        materialized_uses = use_owner.history()
        assert materialized_uses
        assert all(
            item.definition.context.source == source
            and item.definition.context.scope == eval_query_scope
            and semantic_source(item.event) == source
            for item in materialized_uses
        )
        assert all(
            statement.assertion.scope == eval_query_scope
            for item in materialized_uses
            for statement in clone_alias.closure.semantic_graph.ontology.statements(
                subject=item.event_ref)
        )
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        relation_fixture.close()
        backend.close()


def test_default_factory_v06_clone_executes_multi_connector_without_host_write():
    """默认 factory 在 V-06 clone 重绑双命题篇章声明并隔离写入。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    cloned_backend = None
    try:
        graphs, manifest, base_request = _matching_multi_execution_course_fixture(
            backend,
            relation_fixture,
            variant=317,
        )
        loaded_connector = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)
        relation_manifest = _relation_manifest(variant=317)
        loaded_relation = AliasRelationCourseLoader(
            relation_manifest,
            relation_manifest.sha256(),
        ).load(graphs.context)
        representation_family, _content = representation_parts(
            relation_fixture.objects["rep1"])
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=317,
            include_alias=False,
            renderer_family=representation_family,
        )
        factory = LanguageConnectorProductionFactory(
            loaded_connector.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                loaded_relation.factory,
            ),
            loaded_connector.stage4_policy,
        )

        host_precedence = graphs.context.precedence_relation_runtime
        graphs.context.precedence_relation_runtime = None
        try:
            cloned_backend = clone_backend(backend)
            cloned_ctx = clone_train_context(
                graphs.context,
                cloned_backend,
                label="default-factory-multi-core-use-success",
            )
        finally:
            graphs.context.precedence_relation_runtime = host_precedence
        if cloned_ctx.scope_owner is None:
            raise RuntimeError("多命题 V-06 clone 缺少独立评测 owner")
        _install_cloned_precedence_runtime(graphs, cloned_ctx)

        host_installation = factory.build_installation(graphs.context)
        cloned_factory = factory.clone_for_evaluation()
        cloned_installation = cloned_factory.build_installation(cloned_ctx)
        source = base_request.goal.source
        host_episode_scope = episode_scope(
            _BASE + 882,
            parent=document_scope(source),
        )
        host_request = _request_in_query_scope(
            base_request,
            query_scope(1, parent=host_episode_scope),
        )
        host_episode = _episode_from_installed_runtime(
            host_installation.runtime,
            host_request,
            episode_scope_value=host_episode_scope,
            round_id=_BASE + 883,
        )
        host_report = host_installation.stage4_runtime.apply((host_episode,))
        assert host_report.complete is True
        assert len(host_report.outcomes) == 2
        host_before_clone_execution = backend.snapshot()

        eval_document = make_scope(
            SCOPE_DOCUMENT,
            _BASE + 884,
            owner=cloned_ctx.scope_owner,
        )
        eval_episode_scope = episode_scope(
            _BASE + 885,
            parent=eval_document,
        )
        eval_query_scope = query_scope(1, parent=eval_episode_scope)
        assert eval_query_scope.owner == cloned_ctx.scope_owner
        eval_request = _request_in_query_scope(base_request, eval_query_scope)
        eval_episode = _episode_from_installed_runtime(
            cloned_installation.runtime,
            eval_request,
            episode_scope_value=eval_episode_scope,
            round_id=_BASE + 886,
        )
        execution = eval_episode.production.execution
        assert execution is not None and execution.surface is not None
        surface_request = execution.surface.preview.request
        instances = tuple(
            item.instance for item in surface_request.structure.syntax.sentences)
        assert len(instances) == 2
        assert len(set(instances)) == 2
        assert len(surface_request.sentence_attributions) == 2
        assert len({item.hypothesis for item in surface_request.sentence_attributions}) == 2
        clone_training_rows_before_stage4 = cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE)
        report = cloned_installation.stage4_runtime.apply((eval_episode,))
        clone_training_rows_after_stage4 = cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE)

        assert eval_episode.production.complete is True
        assert report.complete is True
        assert report.changed_count == 2
        assert len(report.outcomes) == 2
        assert clone_training_rows_after_stage4 > clone_training_rows_before_stage4
        assert cloned_installation.stage4_runtime.apply((eval_episode,)) == report
        assert cloned_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE) == clone_training_rows_after_stage4
        assert backend.snapshot() == host_before_clone_execution
        clone_alias = cloned_factory._runtime_builder._relation_factory.build(
            cloned_ctx)
        use_owner = clone_alias.closure.use_owner
        assert use_owner is not None
        materialized_uses = use_owner.history()
        assert materialized_uses
        assert all(
            item.definition.context.source == source
            and item.definition.context.scope == eval_query_scope
            and semantic_source(item.event) == source
            for item in materialized_uses
        )
        assert {
            item.definition.context.sentence_instance_key
            for item in materialized_uses
        } == {item.stable_key() for item in instances}
        assert {
            item.definition.context.connector_hypothesis
            for item in materialized_uses
        } == {
            item.hypothesis for item in surface_request.sentence_attributions
        }
        assert all(
            statement.assertion.scope == eval_query_scope
            for item in materialized_uses
            for statement in clone_alias.closure.semantic_graph.ontology.statements(
                subject=item.event_ref)
        )
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        relation_fixture.close()
        backend.close()


def test_default_builder_uses_versioned_relation_course_owner():
    """默认 builder 可由版本化 R-01 课程提供 owner，辅助组件不再现场造关系图。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=309)
        relation_manifest = _relation_manifest(variant=9)
        loaded_relation = AliasRelationCourseLoader(
            relation_manifest,
            relation_manifest.sha256(),
        ).load(graphs.context)
        loaded_connector = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_manifest.alias_protocol,
            relation_manifest.schemas,
            variant=309,
            include_alias=False,
        )
        factory = LanguageConnectorProductionFactory(
            loaded_connector.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                loaded_relation.factory,
            ),
            loaded_connector.stage4_policy,
        )

        factory.build_installation(graphs.context)
        binding = factory._runtime_builder

        assert component_factory.components[-1].alias is None
        assert binding._relation_factory.build(graphs.context) is (
            loaded_relation.alias)
        assert binding.state_key() == factory.clone_for_evaluation() \
            ._runtime_builder.state_key()
    finally:
        backend.close()


def test_default_builder_rejects_course_and_component_alias_owners():
    """版本化课程与辅助组件不得同时声明两套 R-01 owner。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=310)
        relation_manifest = _relation_manifest(variant=10)
        loaded_relation = AliasRelationCourseLoader(
            relation_manifest,
            relation_manifest.sha256(),
        ).load(graphs.context)
        loaded_connector = LanguageGenerationCourseLoader(
            manifest,
            manifest.sha256(),
        ).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_manifest.alias_protocol,
            relation_manifest.schemas,
            variant=310,
            include_alias=True,
        )
        factory = LanguageConnectorProductionFactory(
            loaded_connector.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                loaded_relation.factory,
            ),
            loaded_connector.stage4_policy,
        )

        with pytest.raises(ValueError, match="不得同时配置"):
            factory.build_installation(graphs.context)
    finally:
        backend.close()


def test_default_builder_uses_external_postcheck_factory_owner():
    """独立 G-04 factory 接管复核 owner，辅助组件不得再现场提供第二套。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    try:
        graphs, manifest = _course_fixture(backend, variant=311)
        loaded = LanguageGenerationCourseLoader(
            manifest, manifest.sha256()).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=311,
            include_postcheck=False,
        )
        postcheck_factory = _PostcheckFactory(
            311, _connector_branches(manifest))
        factory = LanguageConnectorProductionFactory(
            loaded.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                postcheck_factory=postcheck_factory,
            ),
            loaded.stage4_policy,
        )

        installation = factory.build_installation(graphs.context)

        assert component_factory.components[-1].postcheck_mapper is None
        assert component_factory.components[-1].postchecker is None
        assert installation.runtime._postcheck_mapper is (
            postcheck_factory.bindings[-1].mapper)
        assert installation.runtime._postchecker is (
            postcheck_factory.bindings[-1].runtime)
        cloned = factory.clone_for_evaluation()
        assert cloned.state_key() == factory.state_key()
        assert cloned._runtime_builder._postcheck_factory is not (
            postcheck_factory)
    finally:
        relation_fixture.close()
        backend.close()


def test_default_builder_rejects_course_and_component_postcheck_owners():
    """课程 G-04 与辅助组件 G-04 不得同时成为 production owner。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    try:
        graphs, manifest = _course_fixture(backend, variant=312)
        loaded = LanguageGenerationCourseLoader(
            manifest, manifest.sha256()).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=312,
            include_postcheck=True,
        )
        factory = LanguageConnectorProductionFactory(
            loaded.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                postcheck_factory=_PostcheckFactory(
                    312, _connector_branches(manifest)),
            ),
            loaded.stage4_policy,
        )

        with pytest.raises(ValueError, match="G-04 owner 不得同时配置"):
            factory.build_installation(graphs.context)
    finally:
        relation_fixture.close()
        backend.close()


def test_default_builder_rejects_postcheck_course_branch_gap():
    """G-04 课程漏掉 connector 分支时必须在 runtime 创建前失败。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    try:
        graphs, manifest = _course_fixture(backend, variant=313)
        loaded = LanguageGenerationCourseLoader(
            manifest, manifest.sha256()).load(graphs.context)
        component_factory = _ComponentFactory(
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            variant=313,
            include_postcheck=False,
        )
        factory = LanguageConnectorProductionFactory(
            loaded.connector_factory,
            DefaultLanguageConnectorProductionRuntimeBuilder(
                component_factory,
                postcheck_factory=_PostcheckFactory(313),
            ),
            loaded.stage4_policy,
        )

        with pytest.raises(ValueError, match="全部 LanguageBranch"):
            factory.build_installation(graphs.context)
        assert not factory._runtime_builder._postcheck_factory.bindings
    finally:
        relation_fixture.close()
        backend.close()


def test_formal_train_default_course_entry_installs_resumed_production_owner(
        tmp_path):
    """真实 formal 入口从 S-07、R-01 Core 课程装配六层生成和 stage4。"""
    backend = DictBackend()
    try:
        graphs, manifest = _course_fixture(backend, variant=306)
        loader = LanguageGenerationCourseLoader(manifest, manifest.sha256())
        relation_manifest = _relation_manifest(variant=6)
        relation_loader = AliasRelationCourseLoader(
            relation_manifest,
            relation_manifest.sha256(),
        )
        component_factory = _ComponentFactory(
            relation_manifest.alias_protocol,
            relation_manifest.schemas,
            variant=306,
            include_alias=False,
            include_postcheck=False,
        )
        postcheck_report = SimpleNamespace(manifest_sha256="g04-course")
        postcheck_factory = _PostcheckFactory(
            306, _connector_branches(manifest))
        result = formal_train(
            FormalTrainConfig(
                str(tmp_path),
                "connector-course-default-entry",
                active_training_stages=(),
                persist_graph_dump=False,
                language_occurrence_protocol=OccurrenceProtocol(
                    (_BASE + 510, 1)),
                language_occurrence_order_protocol=OccurrenceOrderProtocol(
                    (_BASE + 510, 2)),
                language_span_protocol=SegmentationSpanProtocol(
                    SpanProtocol(*tuple(
                        (_BASE + 510, ordinal) for ordinal in range(3, 7)
                    )),
                    (_BASE + 510, 7),
                    (_BASE + 510, 8),
                    (_BASE + 510, 9),
                ),
                language_semantic_course_protocol=(
                    _semantic_course_protocol(
                        _UnusedSemanticCourseMapper())),
                language_precedence_protocol=(
                    _resumed_precedence_protocol(graphs)),
                language_precedence_course=(
                    _PrecedenceCourse(_precedence_domain())),
                language_generation_course_loader=loader,
                language_alias_relation_course_loader=relation_loader,
                language_generation_postcheck_course_loader=(
                    _PostcheckLoader(postcheck_factory, postcheck_report)),
                language_generation_component_factory=component_factory,
            ),
            [],
            backend=backend,
        )

        report = result.language_generation_course_report
        assert report is not None
        assert report.manifest_sha256 == manifest.sha256()
        assert report.active_count == len(manifest.templates)
        assert result.alias_relation_course_report.manifest_sha256 == (
            relation_manifest.sha256())
        assert result.alias_relation_course_report.active_count == len(
            relation_manifest.entries)
        assert result.language_generation_postcheck_course_report is (
            postcheck_report)
        assert len(component_factory.components) == 1
        assert component_factory.components[0].alias is None
        assert component_factory.components[0].postchecker is None
        assert len(postcheck_factory.bindings) == 1
        assert backend.count(MEMORY_EVENT_TABLE) == 0
    finally:
        backend.close()
