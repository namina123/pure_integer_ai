"""从冻结训练图恢复可执行的语言生成 connector。

本模块只读取发布 SQLite 中的一等 LanguageBranch、connector statement 和
S-07 生命周期。课程、candidate pack、问答表和自然语言词表不在依赖边界内；
运行策略只按已冻结的整数协议版本重建，不从对象排序猜测语言含义。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlanner,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationLayerRegistration,
    GenerationPlanner,
    GenerationPlanningRequest,
    GenerationPlanProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePreview,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.generation_observed_surface import (
    ObservedGenerationSpan, ObservedSpanRealizationRule, ObservedSurfaceProposal,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
    StructureOrderSearchBudget,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
    render_generation_preview,
    representation_parts,
)
from pure_integer_ai.cognition.shared.identity import representation_identity
from pure_integer_ai.cognition.shared.semantic_object import (
    role_identity,
    semantic_source,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
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
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureField,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionProposal,
    AliasResolutionResult,
    AliasRouteSearchBudget,
    AliasResolutionSelector,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingCandidateHistoryLog,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.storage.training_candidate_event import (
    TrainingCandidateEventRecordStore,
    decode_integer_stream,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSurfaceRuntimePolicy,
    LanguageConnectorTemplateRuntimePolicy,
    LanguageConnectorValueProtocol,
    LanguageGenerationConnector,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerOrderRequirement,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    GroundedAnswerOrderSemanticsResolver,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationGenerationInput,
    ActiveRelationSurface,
    GraphRelationGeneration,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.assertion_record import ASSERTION_RECORD_TABLE
from pure_integer_ai.storage.graph_statement import GRAPH_STATEMENT_TABLE
from pure_integer_ai.experiments.relation_generation_protocol import (
    RELATION_CONNECTOR_PROFILE, restore_ordinal,
)


_DIALOGUE_GENERATION_NAMESPACE = (21405, 1)
_GROUNDED_CONNECTOR_NAMESPACE = 20916
_GENERIC_CONNECTOR_NAMESPACE = 91563
_GROUNDED_ORDER_NAMESPACE = 20940
_ALIAS_BUDGET = AliasRouteSearchBudget(32, 32, 32)


# object-model: exception
class TrainedGenerationConnectorError(RuntimeError):
    """发布图缺少 connector 执行字段或协议身份发生漂移。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RecoveredGenerationConnector:
    """一个从发布图恢复并可执行 G-02/S-07 的 connector 组件。"""

    template: LanguageGenerationConnectorTemplate
    connector: LanguageGenerationConnector
    lifecycle: StructureOrderLifecycleGraph
    execution_planner: GenerationStructureExecutionPlanner
    alias: AliasRelationRuntime | None = None

    def surface_for(
            self,
            origin: ObjectIdentity,
            *,
            budget: AliasRouteSearchBudget = AliasRouteSearchBudget(32, 32, 32),
            ) -> AliasResolutionProposal:
        """只读执行 R-01 surface route，返回完整 Representation proposal。"""
        if self.alias is None:
            raise TrainedGenerationConnectorError("connector 缺少 R-01 alias runtime")
        return self.alias.preview_surface(
            origin,
            self.template.language_branch,
            budget=budget,
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _RecoveredBranch:
    """一个 LanguageBranch 的图 facade 和已恢复 connector 集。"""

    branch: ObjectIdentity
    lifecycle: StructureOrderLifecycleGraph
    definition_graph: LanguageGenerationConnectorGraph
    templates: tuple[LanguageGenerationConnectorTemplate, ...]
    frame_templates: tuple[tuple[tuple[int, ...], LanguageGenerationConnectorTemplate], ...] = ()
    response_connectors: tuple = ()


def _instruction_series(
        branch: ObjectIdentity, group: int, count: int,
        ) -> tuple[ObjectIdentity, ...]:
    """按发布协议恢复分支内最小指令身份，不物化缺失对象。"""
    return tuple(
        minimal_instruction_identity(
            (*_DIALOGUE_GENERATION_NAMESPACE, group, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, count + 1)
    )


def _resolve_all(ctx: TrainContext, identities, *, label: str):
    """只读恢复一组协议对象；任一缺失均报告字段级错误。"""
    refs = tuple(ctx.graph_ontology.resolve(item) for item in identities)
    missing = tuple(
        item.stable_key() for item, ref in zip(identities, refs, strict=True)
        if ref is None
    )
    if missing:
        raise TrainedGenerationConnectorError(
            f"发布图缺少 {label} 对象: {missing}")
    return refs


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class _RecoveredGenerationProtocols:
    """从同一 LanguageBranch 恢复的 G-00/G-01/G-02 与 renderer 协议。"""

    content: AnswerContentProtocol
    selector: AnswerContentSelector
    plan: GenerationPlanProtocol
    structure: GenerationStructureLayerProtocol
    renderer: ObjectIdentity


def _generation_protocols(
        ctx: TrainContext,
        branch: ObjectIdentity,
        *, retain_unknown_candidates: bool = False,
        ) -> _RecoveredGenerationProtocols:
    """按 connector 同版整数合同恢复训练期使用的生成协议。"""
    content_values = _instruction_series(branch, 10, 5)
    policy_values = _instruction_series(branch, 11, 4)
    plan_values = _instruction_series(branch, 12, 10)
    structure_values = _instruction_series(branch, 13, 3)
    renderer_values = _instruction_series(branch, 18, 1)
    # 这些 MinimalInstruction 是 run-local 控制协议，训练时和 G-03 一样
    # 嵌入 connector 稳定合同而不要求独立物化成图节点。语言内容、槽位与
    # 顺序仍必须来自下方已恢复的 connector、S-07 和 R-01 图。
    del ctx
    content = AnswerContentProtocol(*content_values)
    return _RecoveredGenerationProtocols(
        content,
        AnswerContentSelector(
            content,
            EvidenceAnswerPolicy(
                content,
                EvidenceAnswerPolicyProtocol(*policy_values),
                retain_unknown_candidates=retain_unknown_candidates,
            ),
        ),
        GenerationPlanProtocol(*plan_values),
        GenerationStructureLayerProtocol(*structure_values),
        renderer_values[0],
    )


def _plan_generation(request, protocols, structure_planner, surface_builder, surface_runtime):
    """共用原 G-00 六层调度；事实与来源化对话只注入不同 typed mapper。"""
    registrations = (
        (protocols.plan.stance_layer, GenerationStanceLayerResolver(protocols.plan, protocols.selector)),
        (protocols.plan.content_layer, GenerationContentLayerResolver(protocols.plan, protocols.selector)),
        (protocols.plan.discourse_layer, GenerationDiscourseLayerResolver(
            protocols.plan, protocols.structure, protocols.selector, structure_planner)),
        (protocols.plan.proposition_layer, GenerationPropositionLayerResolver(
            protocols.plan, protocols.structure, protocols.selector, structure_planner)),
        (protocols.plan.syntax_layer, GenerationSyntaxLayerResolver(
            protocols.plan, protocols.structure, protocols.selector, structure_planner)),
        (protocols.plan.surface_layer, GenerationSurfaceLayerResolver(
            protocols.plan, protocols.selector, structure_planner, surface_builder, surface_runtime, commit=False)),
    )
    return GenerationPlanner(protocols.plan, tuple(GenerationLayerRegistration(layer, resolver)
                                                  for layer, resolver in registrations)).plan(request)


def _bound_relation_proposition(
        source: ActiveRelationGenerationInput,
        template: LanguageGenerationConnectorTemplate,
        instruction: ObjectIdentity,
        ) -> BoundProposition:
    """把同一 Core 原子命题投影到已学 connector 的生成 predicate/structure。"""
    atomic = source.proposition
    definition = atomic.definition
    return BoundProposition(
        definition.proposition,
        instruction,
        template.predicate,
        template.proposition_structure,
        definition.source_anchor,
        definition.context,
        (),
        tuple(BoundRoleBinding(
            item.role,
            item.filler,
            item.ordinal,
        ) for item in definition.bindings),
        (),
    )


def _surface_protocol(branch: ObjectIdentity) -> GenerationSurfaceProtocol:
    """恢复与训练阶段相同的 G-03 纯整数协议。"""
    return GenerationSurfaceProtocol(*_instruction_series(branch, 14, 9))


def _value_protocol(branch: ObjectIdentity) -> LanguageConnectorValueProtocol:
    """恢复 connector 四类槽值读取指令。"""
    return LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity(
            (_GROUNDED_CONNECTOR_NAMESPACE, 1, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 5)
    ))


def _lifecycle(
        ctx: TrainContext, branch: ObjectIdentity,
        ) -> StructureOrderLifecycleGraph:
    """从图内 25 个 S-07 predicate 和六个状态身份重建生命周期 facade。"""
    predicates = tuple(
        concept_identity(
            (*_DIALOGUE_GENERATION_NAMESPACE, 30, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 26)
    )
    refs = _resolve_all(ctx, predicates, label="S-07 predicate")
    states = tuple(
        concept_identity(
            (*_DIALOGUE_GENERATION_NAMESPACE, 31, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 7)
    )
    _resolve_all(ctx, states, label="S-07 lifecycle state")
    order_graph = StructureOrderGraph(
        ctx.graph_ontology,
        StructureOrderGraphPredicates(*refs[:19]),
    )
    return StructureOrderLifecycleGraph(
        order_graph,
        StructureOrderLifecycleProtocol(
            *refs[19:],
            *states,
            (*_DIALOGUE_GENERATION_NAMESPACE, 32, 1),
        ),
    )


def _definition_graph(
        ctx: TrainContext,
        branch: ObjectIdentity,
        lifecycle: StructureOrderLifecycleGraph,
        ) -> LanguageGenerationConnectorGraph:
    """恢复 21 个 connector predicate 并建立只读理论图 facade。"""
    identities = tuple(
        concept_identity(
            (*_DIALOGUE_GENERATION_NAMESPACE, 72, index),
            owner=branch.owner,
            versions=branch.versions,
        )
        for index in range(1, 22)
    )
    refs = _resolve_all(ctx, identities, label="connector predicate")
    ordinals = tuple(sorted({restore_ordinal(ctx.graph_ontology.identity_of(item.object), branch)
                             for item in ctx.graph_ontology.statements(predicate=refs[9])
                             if ctx.graph_ontology.identity_of(item.object).owner == branch.owner
                             and ctx.graph_ontology.identity_of(item.object).versions == branch.versions},
                            key=lambda item: item.ordinal))
    return LanguageGenerationConnectorGraph(
        ctx.graph_ontology,
        lifecycle.order_graph,
        LanguageConnectorGraphPredicates(*refs),
        replace(_value_protocol(branch), ordinals=ordinals),
    )


def _pattern_key(
        template: LanguageGenerationConnectorTemplate,
        ) -> tuple[tuple[int, ...], int]:
    """按 grounded connector v1 身份布局恢复 pattern key 与 id。"""
    components = template.connector.components
    grounded = (
        len(components) == 5
        and components[0] == _GROUNDED_CONNECTOR_NAMESPACE
        and components[1] == 2
        and components[4] == 1
    )
    generic = (
        len(components) == 5
        and components[0] == _GENERIC_CONNECTOR_NAMESPACE
        and components[1] == 1
        and components[4] == 1
    )
    if ((not grounded and not generic)
            or components[2] <= 0
            or components[3] <= 0):
        raise TrainedGenerationConnectorError(
            "connector 根不是受支持的 grounded connector v1 身份")
    return components[:4], components[2]


def _slot_ordinal(identity: ObjectIdentity) -> int:
    """从 grounded surface-structure v1 slot 身份恢复声明序。"""
    values = identity.components
    if (len(values) != 6
            or values[0] not in (
                _GROUNDED_CONNECTOR_NAMESPACE, _GENERIC_CONNECTOR_NAMESPACE)
            or values[1] != (
                3 if values[0] == _GROUNDED_CONNECTOR_NAMESPACE else 2)
            or values[4] != 10
            or values[5] <= 0):
        raise TrainedGenerationConnectorError(
            "connector slot 缺少 grounded v1 ordinal")
    return values[5]


def _constraint_ordinal(identity: ObjectIdentity) -> int:
    """从 grounded S-07 constraint 身份恢复相邻 part 序。"""
    values = identity.components
    if (len(values) != 6
            or values[0] not in (
                _GROUNDED_CONNECTOR_NAMESPACE, _GENERIC_CONNECTOR_NAMESPACE)
            or values[1] != (
                3 if values[0] == _GROUNDED_CONNECTOR_NAMESPACE else 2)
            or values[4] != 60
            or values[5] <= 0):
        raise TrainedGenerationConnectorError(
            "connector constraint 缺少 grounded v1 ordinal")
    return values[5]


def _runtime_policy(
        template: LanguageGenerationConnectorTemplate,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> LanguageGenerationConnectorRuntimePolicy:
    """按训练期 ``_variant`` 合同确定性重建非语义搜索策略。"""
    pattern_key, _pattern_id = _pattern_key(template)
    directives = {item.slot: item for item in template.surface}
    surface = []
    for slot in template.slots:
        ordinal = _slot_ordinal(slot.slot)
        directive = directives[slot.slot]
        if directive.action not in surface_protocol.actions():
            raise TrainedGenerationConnectorError(
                "connector surface action 不属于分支 G-03 协议")
        surface.append(LanguageConnectorSurfaceRuntimePolicy(
            slot.slot,
            (*pattern_key, 50, ordinal),
            (_ALIAS_BUDGET
             if directive.action == surface_protocol.emit_action else None),
            (*pattern_key, 51, ordinal),
        ))
    return LanguageGenerationConnectorRuntimePolicy(
        (*pattern_key, 70),
        StructureOrderSearchBudget(max(16, len(template.slots) ** 2 * 2)),
        (LanguageConnectorTemplateRuntimePolicy(
            template.connector, tuple(surface)),),
    )


def _theory_id(template: LanguageGenerationConnectorTemplate) -> int:
    """按训练期 S-07 合同恢复 connector 理论整数身份。"""
    fingerprint = integer_tuple_fingerprint(
        template.connector.stable_key(),
        domain="grounded.answer.order.theory.v1",
    )
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _execution_planner(
        template: LanguageGenerationConnectorTemplate,
        lifecycle: StructureOrderLifecycleGraph,
        ) -> GenerationStructureExecutionPlanner:
    """从模板槽序和 active S-07 constraint 重建通用线性化消费者。"""
    _pattern_components, pattern_id = _pattern_key(template)
    slots = tuple(sorted(
        template.slots,
        key=lambda item: _slot_ordinal(item.slot),
    ))
    if tuple(_slot_ordinal(item.slot) for item in slots) != tuple(
            range(1, len(slots) + 1)):
        raise TrainedGenerationConnectorError(
            "connector slot ordinal 不连续")
    constraints = tuple(sorted(
        template.constraints,
        key=_constraint_ordinal,
    ))
    if len(constraints) != max(0, len(slots) - 1):
        raise TrainedGenerationConnectorError(
            "connector S-07 constraint 未逐相邻 slot 覆盖")
    if tuple(_constraint_ordinal(item) for item in constraints) != tuple(
            range(1, len(constraints) + 1)):
        raise TrainedGenerationConnectorError(
            "connector constraint ordinal 不连续")
    requirements = tuple(
        GroundedAnswerOrderRequirement(
            constraint,
            before.slot,
            after.slot,
        )
        for constraint, (before, after) in zip(
            constraints,
            zip(slots, slots[1:]),
            strict=True,
        )
    )
    theory_id = _theory_id(template)
    owner = template.language_branch.owner
    versions = template.language_branch.versions
    order_namespace = (
        _GROUNDED_ORDER_NAMESPACE
        if template.connector.components[0] == _GROUNDED_CONNECTOR_NAMESPACE
        else _GENERIC_CONNECTOR_NAMESPACE
    )
    consumer_prefix = (order_namespace, 2, pattern_id, theory_id)
    consumer_protocol = StructureOrderConsumerProtocol(*tuple(
        minimal_instruction_identity(
            (*consumer_prefix, index),
            owner=owner,
            versions=versions,
        )
        for index in range(1, 8)
    ))
    applicable_reason = minimal_instruction_identity(
        (order_namespace, 3, pattern_id, theory_id, 4),
        owner=owner,
        versions=versions,
    )
    consumer = StructureOrderConsumer(
        lifecycle,
        GroundedAnswerOrderSemanticsResolver(
            requirements, applicable_reason),
        consumer_protocol,
    )
    return GenerationStructureExecutionPlanner(lifecycle, consumer)


class TrainedGenerationConnectorRuntime:
    """拥有只读 SQLite 句柄，并按需匹配发布图内 connector。"""

    def __init__(
            self,
            database: str | Path,
            *,
            shared_relation_runtime: TrainedRelationGraphRuntime | None = None,
            relation_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
            ) -> None:
        """恢复有限 branch/connector 元数据，可借用同库 Core 只读 owner。

        借用模式只共享物理连接、TrainContext 和图对象缓存；connector 仍从
        同一冻结 SQLite 独立恢复自己的 branch/生命周期状态。借用者关闭时
        不得关闭 Core owner，避免同一模型在严格终端中被完整恢复两次。
        """
        path = Path(database).resolve()
        if not path.is_file():
            raise ValueError("训练图 SQLite 不存在")
        self.path = path
        self._owns_backend = shared_relation_runtime is None
        self.backend = (
            SQLiteBackend(str(path), read_only=True)
            if self._owns_backend else None)
        try:
            if shared_relation_runtime is None:
                self.context = make_train_context(self.backend)
            else:
                if not isinstance(
                        shared_relation_runtime, TrainedRelationGraphRuntime):
                    raise TypeError("shared_relation_runtime 类型错误")
                if shared_relation_runtime.path != path:
                    raise ValueError("共享 Core owner 必须属于同一训练 SQLite")
                backend = getattr(shared_relation_runtime, "backend", None)
                if (not isinstance(backend, SQLiteBackend)
                        or not backend.read_only):
                    raise ValueError("共享 Core owner 必须仍持有只读 SQLite")
                self.backend = backend
                self.context = shared_relation_runtime.context
            if (type(relation_pairs) is not tuple
                    or any(type(item) is not tuple or len(item) != 2
                           or type(item[0]) is not tuple
                           or type(item[1]) is not tuple
                           for item in relation_pairs)):
                raise TypeError("relation_pairs 必须是 (structure_key, predicate_key) tuple")
            self._branches = self._restore_branches(
                relation_pairs=relation_pairs)
            # Alias/R-01 closure is large (it contains the complete historical
            # candidate ledger).  Keep protocol discovery cheap at startup and
            # materialize the read-only closure only when a connector actually
            # requests a surface route.  This does not change the route, graph,
            # or evidence used; it only defers an unused restore.
            self._alias_protocols = self._discover_alias_protocols()
            self._alias_runtimes: dict[ObjectIdentity, AliasRelationRuntime] = {}
        except BaseException:
            if self._owns_backend:
                self.backend.close()
            raise

    def close(self) -> None:
        """关闭自有 SQLite；借用的 Core owner 始终由其拥有者关闭。"""
        backend = getattr(self, "backend", None)
        if backend is not None and self._owns_backend:
            backend.close()
        self.backend = None

    def __enter__(self) -> "TrainedGenerationConnectorRuntime":
        """返回当前只读 owner。"""
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        """退出生命周期时关闭 SQLite。"""
        self.close()

    @property
    def template_count(self) -> int:
        """返回已从发布图恢复的 connector 模板数。"""
        return sum(len(item.templates) for item in self._branches)

    def templates(self) -> tuple[LanguageGenerationConnectorTemplate, ...]:
        """返回有限 connector 元数据，不加载来源正文或全图。"""
        return tuple(
            template
            for branch in self._branches
            for template in branch.templates
        )

    def connector_for(
            self,
            proposition: BoundProposition,
            *,
            branch: ObjectIdentity | None = None,
            connector_identity: ObjectIdentity | None = None,
            ) -> RecoveredGenerationConnector:
        """按分支、predicate 和结构精确匹配唯一可执行 connector。"""
        if not isinstance(proposition, BoundProposition):
            raise TypeError("connector query proposition 必须是 BoundProposition")
        if branch is not None and (
                not isinstance(branch, ObjectIdentity)
                or branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise TypeError("connector query branch 必须是 LanguageBranch")
        matches = []
        for owner in self._branches:
            if branch is not None and owner.branch != branch:
                continue
            for template in owner.templates:
                if (template.proposition_structure == proposition.structure
                        and template.predicate == proposition.predicate
                        and (connector_identity is None or template.connector == connector_identity)):
                    matches.append((owner, template))
        if len(matches) != 1:
            raise TrainedGenerationConnectorError(
                "发布图没有唯一匹配 connector: "
                f"count={len(matches)}")
        owner, template = matches[0]
        surface = _surface_protocol(owner.branch)
        policy = _runtime_policy(template, surface)
        connector = LanguageGenerationConnector(
            LanguageGenerationConnectorRegistry(
                owner.definition_graph.value_protocol,
                (template,),
            ),
            policy,
            surface,
        )
        return RecoveredGenerationConnector(
            template,
            connector,
            owner.lifecycle,
            _execution_planner(template, owner.lifecycle),
            self._alias_runtime_for(owner.branch),
        )

    def alias_runtime(self, branch: ObjectIdentity) -> AliasRelationRuntime:
        """返回与发布图 LanguageBranch 对应的只读 R-01 runtime。"""
        return self._alias_runtime_for(branch)

    def response_connectors(self, branch: ObjectIdentity | None = None) -> tuple:
        """返回训练图中的全部对话动作竞争结构，不按失败回退或私选模板。"""
        return tuple(item for owner in self._branches if branch is None or owner.branch == branch
                     for item in owner.response_connectors)

    def _role_template(self, fact: ActiveRelationSurface) -> LanguageGenerationConnectorTemplate:
        matches = tuple(template for owner in self._branches for frame_key, template in owner.frame_templates
                        if frame_key == fact.proposition.stable_key() and template.predicate == fact.predicate)
        if len(matches) != 1:
            raise TrainedGenerationConnectorError(
                f"当前图命题没有唯一角色级 connector: count={len(matches)}")
        return matches[0]

    def generation_template_for(
            self, fact: ActiveRelationSurface,
            ) -> LanguageGenerationConnectorTemplate:
        """返回 active fact 的唯一已训练角色级模板，供同图结构桥复用。"""
        if not isinstance(fact, ActiveRelationSurface):
            raise TypeError("generation template query 需要 ActiveRelationSurface")
        return self._role_template(fact)

    def observed_role_surfaces(
            self, source: ActiveRelationGenerationInput, fact: ActiveRelationSurface,
            values: tuple[ObservedGenerationSpan, ...],
            ) -> tuple[ObservedSurfaceProposal, ...]:
        """Recover role-reader language evidence without creating Memory-to-Core facts."""
        if source.proposition.definition.proposition != fact.proposition:
            raise TrainedGenerationConnectorError("observed role frame and Core source differ")
        if (type(values) is not tuple or len(values) != len(fact.bindings)
                or any(not isinstance(item, ObservedGenerationSpan) for item in values)
                or tuple(item.ordinal for item in values) != tuple(range(len(values)))):
            raise TrainedGenerationConnectorError("observed roles must cover the whole trained frame")
        template = self._role_template(fact)
        protocol = next(owner.definition_graph.value_protocol for owner in self._branches
                        if owner.branch == template.language_branch)
        alias = self._alias_runtime_for(template.language_branch)
        source_ref = semantic_source(fact.proposition)
        evidence = tuple(sorted(item.stable_key() for item in source.evidence))
        results = []
        for value, binding in zip(values, fact.bindings, strict=True):
            if value.role != binding.role:
                raise TrainedGenerationConnectorError("observed role identity differs from the frame")
            typed = tuple(item for item in source.proposition.definition.bindings
                          if item.role == binding.role and item.filler == binding.filler)
            if len(typed) != 1:
                raise TrainedGenerationConnectorError("observed role has no unique trained RoleBinding")
            readers = tuple(item for item in template.bindings
                            if item.source == protocol.role_filler_source and item.role == binding.role
                            and protocol.ordinal_value(item.ordinal) == typed[0].ordinal)
            if len(readers) != 1:
                raise TrainedGenerationConnectorError("observed role has no unique trained read instruction")
            emitters = tuple(item for item in template.surface if item.slot == readers[0].slot)
            if (len(emitters) != 1
                    or emitters[0].action != _surface_protocol(template.language_branch).emit_action
                    or emitters[0].surface_prefix_steps):
                raise TrainedGenerationConnectorError("observed role has no trained direct emit directive")
            route = alias.preview_surface(binding.filler, template.language_branch,
                                          budget=_ALIAS_BUDGET, allowed_prefix_steps=())
            if (route.result.selected is None or representation_parts(route.result.selected.value)[1]
                    != tuple(map(ord, binding.surface))):
                raise TrainedGenerationConnectorError("trained role realization does not prove its Span")
            rule = ObservedSpanRealizationRule(
                template.language_branch, binding.role, value.ordinal, readers[0].source,
                span_identity(source_ref, members=((binding.start, binding.end),)), route,
                template.stable_key(), evidence,
            )
            results.append(ObservedSurfaceProposal(value, (rule,)))
        return tuple(results)

    def generate_relation(
            self,
            source: ActiveRelationGenerationInput,
            fact: ActiveRelationSurface,
            ) -> GraphRelationGeneration:
        """复用 G-00/G-01/G-02/S-07/G-03/R-01 按真实角色与语言原子生成。

        未物化角色级 connector 的旧模型明确失败，不使用整句命题表示或来源正文。
        """
        if not isinstance(source, ActiveRelationGenerationInput):
            raise TypeError("relation generation source 类型错误")
        if not isinstance(fact, ActiveRelationSurface):
            raise TypeError("relation generation fact 类型错误")
        if source.proposition.definition.proposition != fact.proposition:
            raise TrainedGenerationConnectorError(
                "relation generation Core 命题与查询事实漂移")
        template = self._role_template(fact)
        protocols = _generation_protocols(
            self.context, template.language_branch)
        bound = _bound_relation_proposition(
            source, template, protocols.content.answer)
        recovered = self.connector_for(
            bound, branch=template.language_branch, connector_identity=template.connector)
        values = recovered.connector.registry.values(template, bound)
        value_protocol = recovered.connector.registry.value_protocol
        if (any(item.source == value_protocol.proposition_source for item in template.bindings)
                or not any(item.source == value_protocol.role_filler_source for item in template.bindings)):
            raise TrainedGenerationConnectorError("角色级 connector 不得包含整句命题槽")
        expected_representations = {}
        realization_routes = []
        for value in values:
            proposal = recovered.surface_for(value.filler)
            realization_routes.append(proposal.stable_key())
            options = tuple(option.value for option in proposal.result.options)
            if len(options) != 1:
                raise TrainedGenerationConnectorError("角色或语言原子的 R-01 表示未唯一闭合")
            expected_representations[value.slot] = options[0]
        state = LogicEvidenceState.from_status(
            source.snapshot.epistemic_status)
        candidate = GenerationCandidate(
            bound,
            state,
            semantic_source(bound.template),
            source.proposition.scope,
            source.evidence,
        )
        request = GenerationPlanningRequest(
            AnswerGenerationGoal(
                protocols.content.answer,
                bound,
                LogicEvidenceState(True, False),
                candidate.source,
                candidate.scope,
                template.language_branch,
            ),
            (candidate,),
        )
        structure_planner = recovered.connector.structure_planner()
        surface_runtime = GenerationSurfaceRuntime(
            recovered.alias,
            expected_representations=expected_representations,
        )
        surface_builder = recovered.connector.surface_request_builder(
            recovered.execution_planner)
        plan = _plan_generation(request, protocols, structure_planner, surface_builder, surface_runtime)
        preview = plan.layers[-1].artifact
        if (not plan.complete
                or not isinstance(preview, GenerationSurfacePreview)
                or not preview.complete):
            reason = plan.layers[-1].reason.stable_key()
            raise TrainedGenerationConnectorError(
                "发布图 typed generation 未完成: "
                f"reason={reason}")
        families = {
            representation_parts(item)[0]
            for item in preview.representations
        }
        if len(families) != 1:
            raise TrainedGenerationConnectorError(
                "typed generation Representation 表示族不唯一")
        renderer = UnicodeRepresentationRenderer(
            next(iter(families)), protocols.renderer)
        rendered = render_generation_preview(preview, renderer)
        surface = renderer.text(rendered)
        if not surface.strip():
            raise TrainedGenerationConnectorError(
                "typed generation 产生空表层")
        trace_parts = (plan.stable_key(), rendered.stable_key(), *realization_routes)
        trace = (2, len(trace_parts), *(v for key in trace_parts for v in (len(key), *key)))
        return GraphRelationGeneration(
            surface,
            source.proposition.definition.proposition,
            fact.source_hash,
            len(preview.slots),
            template.connector,
            preview.representations,
            trace,
            structure_plan=preview.request.structure,
        )

    def _alias_runtime_for(self, branch: ObjectIdentity) -> AliasRelationRuntime:
        runtime = self._alias_runtimes.get(branch)
        if runtime is None:
            runtime = self._restore_alias_runtime_for(branch)
            self._alias_runtimes[branch] = runtime
        if not isinstance(runtime, AliasRelationRuntime):
            raise TrainedGenerationConnectorError(
                "LanguageBranch 没有唯一 R-01 runtime")
        return runtime

    def _restore_alias_runtime_for(
            self, branch: ObjectIdentity,
            ) -> AliasRelationRuntime:
        """按需恢复单个分支的完整 R-01 active closure。"""
        matches = tuple(
            item for item in self._alias_protocols
            if branch == self._branches[0].branch
        )
        if len(matches) != 1:
            raise TrainedGenerationConnectorError(
                f"LanguageBranch 没有唯一 R-01 protocol: count={len(matches)}")
        prefix, source, scope = matches[0]
        return self._build_alias_runtime(
            prefix, source, scope, branch, projection_only=True)

    def _restore_alias_runtimes(
            self,
            ) -> tuple[tuple[ObjectIdentity, AliasRelationRuntime], ...]:
        """从发布 Core 图及训练候选历史恢复只读 R-01 closure。

        R-01 的协议稳定键随 candidate manifest 一并写入训练历史；这里仅解码
        整数协议并读取现有 active 投影，不调用课程 loader，也不向 SQLite 写入。
        """
        protocols = self._discover_alias_protocols()
        result = []
        for prefix, source, scope in protocols:
            result.append((self._branches[0].branch,
                           self._build_alias_runtime(
                               prefix, source, scope,
                               self._branches[0].branch,
                               projection_only=True)))
        return tuple(result)

    def _build_alias_runtime(
            self, prefix: tuple[int, ...], source: SourceRef,
            scope: ScopeIdentity, branch: ObjectIdentity,
            *, projection_only: bool = False,
            ) -> AliasRelationRuntime:
        """构造一个协议分支的 R-01 closure。

        发布查询使用 projection_only：active relation 事实直接来自当前
        CandidateProjectionGraph 与 SemanticGraph，保留 CandidateProjectionEvent
        中的 Evidence/decision/route 归因，但不重放完整候选历史 ledger。
        """
        def concept(*suffix: int) -> ObjectIdentity:
            return concept_identity(
                (*prefix, *suffix), owner=source.owner,
                versions=source.versions)

        def structure(*suffix: int) -> ObjectIdentity:
            return ObjectIdentity(
                14, (*prefix, *suffix), source.owner, source.versions)

        def role(*suffix: int) -> ObjectIdentity:
            return role_identity(
                (*prefix, *suffix), owner=source.owner,
                versions=source.versions)

        def instruction(*suffix: int) -> ObjectIdentity:
            return minimal_instruction_identity(
                (*prefix, *suffix), owner=source.owner,
                versions=source.versions)

        projection = CandidateProjectionProtocol(
            *(concept(2, index) for index in range(1, 14)),
            (*prefix, 2, 20),
        )
        learning = EvidenceCandidateProtocol(
            (*prefix, 3, 1), (*prefix, 3, 2), source, scope, 1)
        verifier = IndependentVerifierProtocol(
            concept(4, 1), (*prefix, 4, 2), (*prefix, 4, 3),
            (*prefix, 4, 4), (*prefix, 4, 5))
        relation = RelationClosureProtocol(
            RelationClosureField(concept(5, 1)),
            RelationClosureField(concept(5, 2)),
        )
        alias_relation, refers_relation, realizes_relation = (
            concept(6, index) for index in range(1, 4))
        alias_roles = (role(7, 1), role(7, 2))
        refers_roles = (role(7, 3), role(7, 4))
        realizes_roles = (role(7, 5), role(7, 6), role(7, 7))
        schemas = (
            RelationSchema(
                structure(8, 1), alias_relation,
                tuple(RelationSlotSchema(item, frozenset({7}), 1, 1)
                      for item in alias_roles)),
            RelationSchema(
                structure(8, 2), refers_relation,
                tuple(RelationSlotSchema(item, frozenset({7}), 1, 1)
                      for item in refers_roles)),
            RelationSchema(
                structure(8, 3), realizes_relation,
                (
                    RelationSlotSchema(
                        realizes_roles[0], frozenset(range(3, 24)), 1, 1),
                    RelationSlotSchema(
                        realizes_roles[1], frozenset({13}), 1, 1),
                    RelationSlotSchema(
                        realizes_roles[2], frozenset({11}), 1, 1),
                )),
        )
        alias_protocol = AliasResolutionProtocol(
            alias_relation, (schemas[0].schema,), *alias_roles,
            instruction(9, 1), refers_relation, (schemas[1].schema,),
            *refers_roles, instruction(9, 2), realizes_relation,
            (schemas[2].schema,), *realizes_roles, instruction(9, 3),
            instruction(9, 4), instruction(9, 5), instruction(9, 6))
        candidate_graph = CandidateProjectionGraph(
            self.context.graph_ontology, projection, read_only=True)
        semantic = SemanticGraph(
            self.context.graph_ontology,
            AtomicPropositionPredicates(*(
                self.context.graph_ontology.resolve(concept(1, index))
                for index in range(1, 7))),
        )
        if projection_only:
            consumer = ActiveRelationClosureConsumer(
                semantic, candidate_graph, relation, schemas, engine=None)
            closure = RelationClosureRuntime(
                None, semantic, consumer, relation, None)
            return AliasRelationRuntime(
                closure, AliasResolutionSelector(alias_protocol))
        history_protocol = TrainingHypothesisHistoryProtocol(
            (*prefix, 40), (*prefix, 3, 1), source, scope)
        learning_runtime = CandidateLearningRuntime.restore_for_training_graph(
            learning,
            candidate_graph,
            IndependentObjectVerifier(verifier),
            CandidateProjectionMetadata(1, 2, content_version=1),
            self.context.training_candidate_history,
            history_protocol,
        )
        consumer = ActiveRelationClosureConsumer(
            semantic, candidate_graph, relation, schemas,
            engine=learning_runtime.engine)
        closure = RelationClosureRuntime(
            learning_runtime, semantic, consumer, relation, None)
        return AliasRelationRuntime(
            closure, AliasResolutionSelector(alias_protocol))

    def _discover_alias_protocols(
            self,
            ) -> tuple[tuple[tuple[int, ...], SourceRef, ScopeIdentity], ...]:
        """从训练历史整数信封发现 candidate alias manifest profile。"""
        history = self.context.training_candidate_history
        if not isinstance(history, TrainingCandidateHistoryLog):
            raise TrainedGenerationConnectorError("发布图缺少训练候选历史")
        store = TrainingCandidateEventRecordStore(self.backend)
        rows = self.backend.select("training_candidate_event")
        result = []
        for row in rows:
            record = store.read(row["event_hash"])
            envelope = decode_integer_stream(store.read_payload(record))
            values = envelope[2:2 + envelope[1]]
            if len(values) < 4 or values[0] != 2:
                continue
            cursor = 1
            parts = []
            valid = True
            for _ in range(4):
                if cursor >= len(values):
                    valid = False
                    break
                size = values[cursor]
                cursor += 1
                if size <= 0 or cursor + size > len(values):
                    valid = False
                    break
                parts.append(tuple(values[cursor:cursor + size]))
                cursor += size
            if not valid or cursor != len(values):
                continue
            namespace, hypothesis_kind, source_key, scope_key = parts
            if (len(namespace) < 6 or namespace[0] != 22020
                    or namespace[-1] != 40
                    or hypothesis_kind != (*namespace[:-1], 3, 1)):
                continue
            try:
                source = SourceRef.from_stable_key(source_key)
                scope = ScopeIdentity.from_stable_key(scope_key)
            except (TypeError, ValueError):
                continue
            item = (tuple(namespace[:-1]), source, scope)
            if item not in result:
                result.append(item)
        return tuple(result)

    def _restore_branches(
            self,
            *,
            relation_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
            ) -> tuple[_RecoveredBranch, ...]:
        """按对象类型稀疏发现分支，再沿 connector_language 入边恢复根。

        Strict terminal 只需要当前 active Core relation 对应的 frame
        connector。传入 relation_pairs 时先用 predicate/structure 两个图边
        做整数索引筛选，再完整恢复命中的 connector；response connector
        仍逐槽恢复。完整发布审计不传该参数，保留全量闭合校验。
        """
        from pure_integer_ai.experiments.response_generation_graph import recover_response_connectors
        rows = self.backend.select(
            GRAPH_OBJECT_TABLE,
            where={"object_kind": OBJECT_LANGUAGE_BRANCH},
            order_by="local_id",
        )
        if not rows:
            raise TrainedGenerationConnectorError(
                "发布图没有 LanguageBranch")
        result = []
        ontology = self.context.graph_ontology
        for row in rows:
            ref = ontology.typed_ref_for_node(
                row["space_id"], row["local_id"])
            branch = ontology.identity_of(ref)
            lifecycle = _lifecycle(self.context, branch)
            graph = _definition_graph(
                self.context, branch, lifecycle)
            if relation_pairs:
                # Use the physical integer statement index for the broad
                # connector predicates. Calling ontology.statements here would
                # materialize hundreds of thousands of graph objects before
                # the active Core predicate filter can run.
                wanted_predicates = frozenset(item[1] for item in relation_pairs)
                language_hash = ontology.identity_hash_of(
                    graph.predicates.connector_language)
                # assertion_record is the canonical complete projection;
                # graph_statement is only a partial physical compatibility
                # index in this release and cannot be used for discovery.
                language_rows = self.backend.select(
                    ASSERTION_RECORD_TABLE,
                    where={
                        "relation_kind": language_hash,
                        "object_object_kind": ref.object_kind,
                        "object_space_id": ref.space_id,
                        "object_local_id": ref.local_id,
                    })
                root_nodes = tuple(sorted({
                    (row["subject_object_kind"], row["subject_space_id"],
                     row["subject_local_id"])
                    for row in language_rows
                }))
                connector_hash = ontology.identity_hash_of(
                    graph.predicates.connector_predicate)
                # Filter the connector-predicate projection by the exact
                # integer target identities in SQLite.  The old path loaded
                # every connector-predicate assertion for the branch and
                # only then discarded unrelated predicates in Python; on a
                # broad graph this copied more than a million rows during
                # startup and made generation look like a model-size issue.
                predicate_rows = []
                wanted_targets = set()
                for predicate_key in wanted_predicates:
                    predicate_identity = ObjectIdentity.from_stable_key(
                        predicate_key)
                    predicate_ref = ontology.resolve(predicate_identity)
                    if predicate_ref is None:
                        raise TrainedGenerationConnectorError(
                            "active relation predicate 不在生成图中")
                    target = (
                        predicate_ref.object_kind,
                        predicate_ref.space_id,
                        predicate_ref.local_id,
                    )
                    if target in wanted_targets:
                        continue
                    wanted_targets.add(target)
                    predicate_rows.extend(self.backend.select(
                        ASSERTION_RECORD_TABLE,
                        where={
                            "relation_kind": connector_hash,
                            "object_object_kind": target[0],
                            "object_space_id": target[1],
                            "object_local_id": target[2],
                        }))
                predicates = {
                    (item["subject_object_kind"], item["subject_space_id"],
                     item["subject_local_id"]):
                    ontology.identity_of(ontology.typed_ref_for_node(
                        item["object_space_id"], item["object_local_id"])).stable_key()
                    for item in predicate_rows
                    if (item["subject_object_kind"], item["subject_space_id"],
                        item["subject_local_id"]) in root_nodes}
                selected_roots = tuple(
                    ontology.identity_of(ontology.typed_ref_for_node(
                        node[1], node[2]))
                    for node in root_nodes
                    if predicates.get(node) in wanted_predicates)
                definitions = tuple(graph.read(root)
                                    for root in selected_roots)
            else:
                links = ontology.statements(
                    predicate=graph.predicates.connector_language,
                    object_ref=ref,
                )
                link_roots = tuple(ontology.identity_of(item.subject) for item in links)
                definitions = tuple(graph.read(root) for root in link_roots)
            responses = recover_response_connectors(graph, branch)
            response_roots = {item.template.connector for item in responses}
            templates = tuple(sorted(
                (item.definition for item in definitions if item.definition.connector not in response_roots),
                key=lambda item: item.connector.stable_key(),
            ))
            frame_templates = []
            for item in definitions:
                key = item.qualifiers
                if key[:2] != (RELATION_CONNECTOR_PROFILE, 1):
                    continue
                if len(key) < 4 or key[2] <= 0 or key[2] + 3 >= len(key):
                    raise TrainedGenerationConnectorError("角色级 connector 来源记录被截断")
                frame_templates.append((key[3:3 + key[2]], item.definition))
            if templates or responses:
                result.append(_RecoveredBranch(
                    branch, lifecycle, graph, templates, tuple(sorted(
                        frame_templates, key=lambda item: (item[0], item[1].connector.stable_key()))), responses))
        if not result:
            raise TrainedGenerationConnectorError(
                "发布图没有 connector_language 根")
        return tuple(result)


__all__ = [
    "RecoveredGenerationConnector",
    "TrainedGenerationConnectorError",
    "TrainedGenerationConnectorRuntime",
]
