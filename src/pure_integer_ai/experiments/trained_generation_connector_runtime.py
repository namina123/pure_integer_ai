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
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
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
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE


_DIALOGUE_GENERATION_NAMESPACE = (21405, 1)
_GROUNDED_CONNECTOR_NAMESPACE = 20916
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
            ),
        ),
        GenerationPlanProtocol(*plan_values),
        GenerationStructureLayerProtocol(*structure_values),
        renderer_values[0],
    )


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
    return LanguageGenerationConnectorGraph(
        ctx.graph_ontology,
        lifecycle.order_graph,
        LanguageConnectorGraphPredicates(*refs),
        _value_protocol(branch),
    )


def _pattern_key(
        template: LanguageGenerationConnectorTemplate,
        ) -> tuple[tuple[int, ...], int]:
    """按 grounded connector v1 身份布局恢复 pattern key 与 id。"""
    components = template.connector.components
    if (len(components) != 5
            or components[0] != _GROUNDED_CONNECTOR_NAMESPACE
            or components[1] != 2
            or components[2] <= 0
            or components[3] <= 0
            or components[4] != 1):
        raise TrainedGenerationConnectorError(
            "connector 根不是受支持的 grounded connector v1 身份")
    return components[:4], components[2]


def _slot_ordinal(identity: ObjectIdentity) -> int:
    """从 grounded surface-structure v1 slot 身份恢复声明序。"""
    values = identity.components
    if (len(values) != 6
            or values[0] != _GROUNDED_CONNECTOR_NAMESPACE
            or values[1] != 3
            or values[4] != 10
            or values[5] <= 0):
        raise TrainedGenerationConnectorError(
            "connector slot 缺少 grounded v1 ordinal")
    return values[5]


def _constraint_ordinal(identity: ObjectIdentity) -> int:
    """从 grounded S-07 constraint 身份恢复相邻 part 序。"""
    values = identity.components
    if (len(values) != 6
            or values[0] != _GROUNDED_CONNECTOR_NAMESPACE
            or values[1] != 3
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


def _claim_surface_template(
        template: LanguageGenerationConnectorTemplate,
        value_protocol: LanguageConnectorValueProtocol,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> LanguageGenerationConnectorTemplate:
    """把完整 Core claim 的 Representation 设为唯一可见槽。

    Core relation generation 已从同一来源 Span 包络物化完整命题
    Representation。若继续发射 connector 中的常量、predicate 或 role
    filler 槽，就会把训练句式的 literal 再包在完整 claim 外，形成诸如
    ``档案显示，<claim>`` 的错误输出。这里不删除或改写任何图内容，而是
    按图内 binding source 将非 proposition 槽声明为 S-07 silent；命题槽
    仍由 R-01 唯一恢复其完整 Representation。
    """
    bindings = {item.slot: item for item in template.bindings}
    changed = []
    for directive in template.surface:
        binding = bindings.get(directive.slot)
        if binding is None:
            raise TrainedGenerationConnectorError(
                "connector surface slot 缺少 binding")
        if binding.source != value_protocol.proposition_source:
            changed.append(replace(
                directive,
                action=surface_protocol.silent_action,
            ))
        else:
            changed.append(directive)
    return replace(template, surface=tuple(changed))


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
    consumer_prefix = (
        _GROUNDED_ORDER_NAMESPACE, 2, pattern_id, theory_id)
    consumer_protocol = StructureOrderConsumerProtocol(*tuple(
        minimal_instruction_identity(
            (*consumer_prefix, index),
            owner=owner,
            versions=versions,
        )
        for index in range(1, 8)
    ))
    applicable_reason = minimal_instruction_identity(
        (_GROUNDED_ORDER_NAMESPACE, 3, pattern_id, theory_id, 4),
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

    def __init__(self, database: str | Path) -> None:
        """只读打开模型，并恢复有限 branch/connector 元数据。"""
        path = Path(database).resolve()
        if not path.is_file():
            raise ValueError("训练图 SQLite 不存在")
        self.path = path
        self.backend = SQLiteBackend(str(path), read_only=True)
        try:
            self.context = make_train_context(self.backend)
            self._branches = self._restore_branches()
            self._alias_runtimes = self._restore_alias_runtimes()
        except BaseException:
            self.backend.close()
            raise

    def close(self) -> None:
        """关闭只读 SQLite；重复关闭不改变模型。"""
        backend = getattr(self, "backend", None)
        if backend is not None:
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
                        and template.predicate == proposition.predicate):
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

    def generate_relation(
            self,
            source: ActiveRelationGenerationInput,
            fact: ActiveRelationSurface,
            ) -> GraphRelationGeneration:
        """执行 G-00/G-01/G-02/S-07/G-03/R-01 并渲染 Core 命题。"""
        if not isinstance(source, ActiveRelationGenerationInput):
            raise TypeError("relation generation source 类型错误")
        if not isinstance(fact, ActiveRelationSurface):
            raise TypeError("relation generation fact 类型错误")
        if source.proposition.definition.proposition != fact.proposition:
            raise TrainedGenerationConnectorError(
                "relation generation Core 命题与查询事实漂移")
        templates = self.templates()
        if len(templates) != 1:
            raise TrainedGenerationConnectorError(
                "当前发布图没有唯一默认 LanguageBranch connector: "
                f"count={len(templates)}")
        template = templates[0]
        protocols = _generation_protocols(
            self.context, template.language_branch)
        bound = _bound_relation_proposition(
            source, template, protocols.content.answer)
        recovered = self.connector_for(
            bound, branch=template.language_branch)
        values = recovered.connector.registry.values(template, bound)
        proposition_slots = tuple(
            item.slot for item in values if item.filler == bound.template)
        if len(proposition_slots) != 1:
            raise TrainedGenerationConnectorError(
                "connector 命题本体槽位未唯一解析")
        envelope_start = min(
            fact.cue_start,
            *(item.start for item in fact.bindings),
        )
        envelope_end = max(
            fact.cue_end,
            *(item.end for item in fact.bindings),
        )
        if (envelope_start < 0 or envelope_end <= envelope_start
                or envelope_end > len(fact.evidence_surface)):
            raise TrainedGenerationConnectorError(
                "Core relation 表层包络越界")
        claim = fact.evidence_surface[envelope_start:envelope_end]
        realizes = recovered.alias.closure.consumer.lookup_relation(
            recovered.alias.selector.protocol.realizes_relation)
        representations = tuple(
            binding.filler
            for item in realizes
            for binding in item.proposition.bindings
            if binding.filler.object_kind == OBJECT_REPRESENTATION
        )
        families = {representation_parts(item)[0] for item in representations}
        if len(families) != 1:
            raise TrainedGenerationConnectorError(
                "R-01 Representation family 不唯一")
        expected_representation = representation_identity(
            next(iter(families)),
            tuple(ord(character) for character in claim),
            owner=template.language_branch.owner,
            versions=template.language_branch.versions,
        )
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
            expected_representations={
                proposition_slots[0]: expected_representation,
            },
        )
        surface_builder = recovered.connector.surface_request_builder(
            recovered.execution_planner)
        plan = GenerationPlanner(
            protocols.plan,
            (
                GenerationLayerRegistration(
                    protocols.plan.stance_layer,
                    GenerationStanceLayerResolver(
                        protocols.plan, protocols.selector),
                ),
                GenerationLayerRegistration(
                    protocols.plan.content_layer,
                    GenerationContentLayerResolver(
                        protocols.plan, protocols.selector),
                ),
                GenerationLayerRegistration(
                    protocols.plan.discourse_layer,
                    GenerationDiscourseLayerResolver(
                        protocols.plan,
                        protocols.structure,
                        protocols.selector,
                        structure_planner,
                    ),
                ),
                GenerationLayerRegistration(
                    protocols.plan.proposition_layer,
                    GenerationPropositionLayerResolver(
                        protocols.plan,
                        protocols.structure,
                        protocols.selector,
                        structure_planner,
                    ),
                ),
                GenerationLayerRegistration(
                    protocols.plan.syntax_layer,
                    GenerationSyntaxLayerResolver(
                        protocols.plan,
                        protocols.structure,
                        protocols.selector,
                        structure_planner,
                    ),
                ),
                GenerationLayerRegistration(
                    protocols.plan.surface_layer,
                    GenerationSurfaceLayerResolver(
                        protocols.plan,
                        protocols.selector,
                        structure_planner,
                        surface_builder,
                        surface_runtime,
                        commit=False,
                    ),
                ),
            ),
        ).plan(request)
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
        trace = integer_tuple_fingerprint(
            (
                *plan.stable_key(),
                *rendered.stable_key(),
            ),
            domain="trained.relation.typed.generation.v1",
        )
        return GraphRelationGeneration(
            surface,
            source.proposition.definition.proposition,
            fact.source_hash,
            len(preview.slots),
            template.connector,
            preview.representations,
            trace,
        )

    def _alias_runtime_for(self, branch: ObjectIdentity) -> AliasRelationRuntime:
        matches = tuple(
            runtime for owner, runtime in self._alias_runtimes
            if owner == branch
        )
        if len(matches) != 1:
            raise TrainedGenerationConnectorError(
                f"LanguageBranch 没有唯一 R-01 runtime: count={len(matches)}")
        return matches[0]

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
            semantic = SemanticGraph(
                self.context.graph_ontology,
                AtomicPropositionPredicates(*(
                    self.context.graph_ontology.resolve(concept(1, index))
                    for index in range(1, 7))),
            )
            consumer = ActiveRelationClosureConsumer(
                semantic, candidate_graph, relation, schemas,
                engine=learning_runtime.engine)
            closure = RelationClosureRuntime(
                learning_runtime, semantic, consumer, relation, None)
            result.append((
                self._branches[0].branch,
                AliasRelationRuntime(
                    closure, AliasResolutionSelector(alias_protocol)),
            ))
        if not result:
            raise TrainedGenerationConnectorError(
                "发布图没有可恢复的 R-01 alias protocol")
        return tuple(result)

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

    def _restore_branches(self) -> tuple[_RecoveredBranch, ...]:
        """按对象类型稀疏发现分支，再沿 connector_language 入边恢复根。"""
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
            links = ontology.statements(
                predicate=graph.predicates.connector_language,
                object_ref=ref,
            )
            value_protocol = graph.value_protocol
            surface_protocol = _surface_protocol(branch)
            templates = tuple(sorted(
                (_claim_surface_template(
                    graph.read(ontology.identity_of(item.subject)).definition,
                    value_protocol,
                    surface_protocol,
                ) for item in links),
                key=lambda item: item.connector.stable_key(),
            ))
            if templates:
                result.append(_RecoveredBranch(
                    branch, lifecycle, graph, templates))
        if not result:
            raise TrainedGenerationConnectorError(
                "发布图没有 connector_language 根")
        return tuple(result)


__all__ = [
    "RecoveredGenerationConnector",
    "TrainedGenerationConnectorError",
    "TrainedGenerationConnectorRuntime",
]
