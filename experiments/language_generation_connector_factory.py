"""从当前图和 active 候选重建 connector 运行组件。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.generation_content import (
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
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlanner,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderConsumer,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationInstallation,
    ProductionGenerationRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorDiscourseDeclarationProvider,
    LanguageGenerationConnector,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
)
from pure_integer_ai.experiments.language_generation_connector_candidate import (
    CANDIDATE_PERSISTENCE_TRAINING,
    LanguageConnectorCandidateRuntime,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.experiments.language_generation_connector_scheduler import (
    ScheduledLanguageGenerationConnectorRegistry,
)
from pure_integer_ai.experiments.language_generation_connector_stage4 import (
    LanguageConnectorStage4Policy,
    LanguageConnectorStage4Runtime,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    SemanticCourseGenerationRequestMapper,
)
from pure_integer_ai.experiments.train_context import TrainContext


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键添加长度边界。"""
    return len(key), *key


def _predicate_identities(ontology, refs) -> tuple[ObjectIdentity, ...]:
    """把来源 facade 的运行引用冻结为可跨 backend 恢复的图身份。"""
    identities = tuple(ontology.identity_of(ref) for ref in refs)
    if len(set(identities)) != len(identities):
        raise ValueError("connector factory predicate 身份不得重复")
    return identities


def _resolve_predicates(ontology, identities, *, label: str):
    """在目标图只读恢复完整 predicate 集，缺任一对象时 fail closed。"""
    refs = tuple(ontology.resolve(identity) for identity in identities)
    if any(ref is None for ref in refs):
        raise RuntimeError(f"{label} predicate 未在当前图完整恢复")
    return refs


def _require_discourse_declarations(
        declarations: LanguageConnectorDiscourseDeclarationProvider | None,
        ) -> LanguageConnectorDiscourseDeclarationProvider | None:
    """核验篇章声明读取器能在 V-06 中独立克隆，禁止复用宿主可变状态。"""
    if declarations is None:
        return None
    if any(not hasattr(declarations, method) for method in (
            "declaration", "state_key", "clone_for_evaluation")):
        raise TypeError("connector discourse declaration provider 协议不完整")
    return declarations


def _clone_discourse_declarations(
        declarations: LanguageConnectorDiscourseDeclarationProvider | None,
        ) -> LanguageConnectorDiscourseDeclarationProvider | None:
    """克隆声明读取器并核验内容锁不漂移且不与宿主共用实例。"""
    declarations = _require_discourse_declarations(declarations)
    if declarations is None:
        return None
    cloned = _require_discourse_declarations(
        declarations.clone_for_evaluation())
    if cloned is declarations:
        raise RuntimeError("connector discourse declaration provider 不得复用宿主实例")
    if cloned.state_key() != declarations.state_key():
        raise RuntimeError("connector discourse declaration provider clone 配置漂移")
    return cloned


def _rebuild_candidate_owner(
        ctx: TrainContext,
        candidates: LanguageConnectorCandidateRuntime,
        order_predicates: tuple[ObjectIdentity, ...],
        connector_predicates: tuple[ObjectIdentity, ...],
        ) -> tuple[LanguageConnectorCandidateRuntime, StructureOrderGraph]:
    """在当前 context 重建两图 facade，并按 ontology 决定复用或克隆 owner。"""
    ontology = ctx.graph_ontology
    order_graph = StructureOrderGraph(
        ontology,
        StructureOrderGraphPredicates(*_resolve_predicates(
            ontology,
            order_predicates,
            label="S-07",
        )),
    )
    definition_graph = LanguageGenerationConnectorGraph(
        ontology,
        order_graph,
        LanguageConnectorGraphPredicates(*_resolve_predicates(
            ontology,
            connector_predicates,
            label="connector",
        )),
        candidates.definition_graph.value_protocol,
    )
    candidate_graph = CandidateProjectionGraph(
        ontology,
        candidates.learning.graph.protocol,
    )
    if candidates.memory_enabled:
        source = candidates.learning.engine.protocol.aggregate_source
        event_log = (
            ctx.memory_interact_events
            if source.owner.visibility == VISIBILITY_SESSION
            else ctx.memory_read_events
        )
        if event_log is None:
            raise RuntimeError("connector M-03 恢复缺少目标 Memory event log")
        if (ontology is candidates.definition_graph.ontology
                and candidates.memory_event_log is event_log):
            return candidates, candidates.definition_graph.order_graph
        return candidates.restore_for_graphs(
            definition_graph,
            candidate_graph,
            event_log,
        ), order_graph
    if candidates.persistence_kind == CANDIDATE_PERSISTENCE_TRAINING:
        history = ctx.training_candidate_history
        if history is None:
            raise RuntimeError("connector 恢复缺少目标 Core 训练历史")
        if (ontology is candidates.definition_graph.ontology
                and candidates.training_history is history):
            return candidates, candidates.definition_graph.order_graph
        return candidates.restore_for_training_graphs(
            definition_graph,
            candidate_graph,
            history,
        ), order_graph
    if ontology is candidates.definition_graph.ontology:
        return candidates, candidates.definition_graph.order_graph
    return candidates.clone_for_graphs(
        definition_graph,
        candidate_graph,
    ), order_graph


@dataclass(frozen=True)
class ActiveLanguageConnectorAssembly:
    """当前图的 active connector、候选 owner 和 S-07 定义 facade。"""

    connector: LanguageGenerationConnector
    candidates: LanguageConnectorCandidateRuntime
    order_graph: StructureOrderGraph

    def __post_init__(self) -> None:
        """核验 active connector、候选 owner 与 S-07 facade 属于同次装配。"""
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("connector assembly connector 类型错误")
        if not isinstance(self.candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("connector assembly candidates 类型错误")
        if not isinstance(self.order_graph, StructureOrderGraph):
            raise TypeError("connector assembly order graph 类型错误")
        if self.candidates.definition_graph.order_graph is not self.order_graph:
            raise ValueError("connector assembly 理论图未绑定同一 S-07 facade")

    def state_key(self) -> tuple:
        """返回 active 理论、候选 owner 和运行策略的完整装配状态。"""
        return (
            self.connector.stable_key(),
            self.candidates.state_key(),
            self.order_graph.ontology.space_identity.stable_key(),
        )


class ActiveLanguageConnectorFactory:
    """用跨图身份协议在宿主或 V-06 上重建 active connector。"""

    def __init__(
            self,
            candidates: LanguageConnectorCandidateRuntime,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            surface_protocol: GenerationSurfaceProtocol,
            production_purpose: ObjectIdentity,
            discourse_declarations: LanguageConnectorDiscourseDeclarationProvider
            | None = None,
            ) -> None:
        """保存候选协议、运行策略、purpose 和来源化篇章声明读取器。"""
        if not isinstance(candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("connector factory candidates 类型错误")
        if not isinstance(
                runtime_policy, LanguageGenerationConnectorRuntimePolicy):
            raise TypeError("connector factory runtime policy 类型错误")
        if not isinstance(surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("connector factory surface protocol 类型错误")
        if (not isinstance(production_purpose, ObjectIdentity)
                or production_purpose.object_kind
                != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("connector factory production purpose 类型错误")
        source_ontology = candidates.definition_graph.ontology
        self._candidates = candidates
        self._runtime_policy = runtime_policy
        self._surface_protocol = surface_protocol
        self._production_purpose = production_purpose
        self._discourse_declarations = _require_discourse_declarations(
            discourse_declarations)
        self._order_predicates = _predicate_identities(
            source_ontology,
            candidates.definition_graph.order_graph.predicates.refs(),
        )
        self._connector_predicates = _predicate_identities(
            source_ontology,
            candidates.definition_graph.predicates.refs(),
        )

    def build(self, ctx: TrainContext) -> ActiveLanguageConnectorAssembly:
        """从当前 context 的真实图恢复 active 理论并核验运行策略全覆盖。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("connector factory ctx 类型错误")
        active_candidates, order_graph = _rebuild_candidate_owner(
            ctx,
            self._candidates,
            self._order_predicates,
            self._connector_predicates,
        )
        connector = LanguageGenerationConnector(
            active_candidates.active_registry(),
            self._runtime_policy,
            self._surface_protocol,
            tuple(
                GenerationSurfaceAttribution(
                    template.connector,
                    active_candidates.learning.hypothesis_for_candidate(
                        template.connector),
                    self._production_purpose,
                )
                for template in active_candidates.active_templates()
            ),
            self._discourse_declarations,
        )
        return ActiveLanguageConnectorAssembly(
            connector,
            active_candidates,
            order_graph,
        )

    def clone_for_evaluation(self) -> "ActiveLanguageConnectorFactory":
        """复制 H-00/H-04 owner，评测 build 后不共享可变候选历史。"""
        cloned_candidates = self._candidates.clone_for_graphs(
            self._candidates.definition_graph,
            self._candidates.learning.graph,
        )
        return ActiveLanguageConnectorFactory(
            cloned_candidates,
            self._runtime_policy,
            self._surface_protocol,
            self._production_purpose,
            _clone_discourse_declarations(self._discourse_declarations),
        )

    def state_key(self) -> tuple:
        """返回跨图 predicate、候选状态和 run-local 策略配置。"""
        return (
            tuple(item.stable_key() for item in self._order_predicates),
            tuple(item.stable_key() for item in self._connector_predicates),
            self._candidates.state_key(),
            self._runtime_policy.stable_key(),
            self._surface_protocol.stable_key(),
            self._production_purpose.stable_key(),
            (() if self._discourse_declarations is None else (
                self._discourse_declarations.state_key())),
        )


@dataclass(frozen=True)
class TrialLanguageConnectorAssembly:
    """只暴露一个 forming Hypothesis 的隔离 trial connector。"""

    connector: LanguageGenerationConnector
    candidates: LanguageConnectorCandidateRuntime
    order_graph: StructureOrderGraph
    hypothesis: HypothesisKey

    def __post_init__(self) -> None:
        """核验 trial connector 只绑定一个 exact Hypothesis 和同一 S-07 facade。"""
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("trial connector assembly connector 类型错误")
        if not isinstance(self.candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("trial connector assembly candidates 类型错误")
        if not isinstance(self.order_graph, StructureOrderGraph):
            raise TypeError("trial connector assembly order graph 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("trial connector assembly hypothesis 类型错误")
        if self.candidates.definition_graph.order_graph is not self.order_graph:
            raise ValueError("trial connector 未绑定同一 S-07 facade")

    def state_key(self) -> tuple:
        """返回 trial 理论、exact Hypothesis 和隔离 owner 状态。"""
        return (
            self.connector.stable_key(),
            self.candidates.state_key(),
            self.order_graph.ontology.space_identity.stable_key(),
            self.hypothesis.stable_key(),
        )


class TrialLanguageConnectorFactory:
    """为调用方指定的 exact forming Hypothesis 建立非 active trial 组件。"""

    def __init__(
            self,
            candidates: LanguageConnectorCandidateRuntime,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            surface_protocol: GenerationSurfaceProtocol,
            hypothesis: HypothesisKey,
            trial_purpose: ObjectIdentity,
            discourse_declarations: LanguageConnectorDiscourseDeclarationProvider
            | None = None,
            ) -> None:
        """保存 forming 候选、exact Hypothesis、purpose 和篇章声明读取器。"""
        if not isinstance(candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("trial connector factory candidates 类型错误")
        if not isinstance(
                runtime_policy, LanguageGenerationConnectorRuntimePolicy):
            raise TypeError("trial connector factory runtime policy 类型错误")
        if not isinstance(surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("trial connector factory surface protocol 类型错误")
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("trial connector factory hypothesis 类型错误")
        if (not isinstance(trial_purpose, ObjectIdentity)
                or trial_purpose.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("trial connector factory purpose 类型错误")
        source_ontology = candidates.definition_graph.ontology
        self._candidates = candidates
        self._runtime_policy = runtime_policy
        self._surface_protocol = surface_protocol
        self._hypothesis = hypothesis
        self._trial_purpose = trial_purpose
        self._discourse_declarations = _require_discourse_declarations(
            discourse_declarations)
        self._order_predicates = _predicate_identities(
            source_ontology,
            candidates.definition_graph.order_graph.predicates.refs(),
        )
        self._connector_predicates = _predicate_identities(
            source_ontology,
            candidates.definition_graph.predicates.refs(),
        )

    def build(self, ctx: TrainContext) -> TrialLanguageConnectorAssembly:
        """恢复 exact forming 理论，并建立不进入普通 active registry 的组件。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("trial connector factory ctx 类型错误")
        candidates, order_graph = _rebuild_candidate_owner(
            ctx,
            self._candidates,
            self._order_predicates,
            self._connector_predicates,
        )
        template = candidates.trial_template(self._hypothesis)
        policies = tuple(
            item for item in self._runtime_policy.templates
            if item.connector == template.connector
        )
        if len(policies) != 1:
            raise ValueError("trial connector 缺唯一运行策略")
        runtime_policy = replace(self._runtime_policy, templates=policies)
        connector = LanguageGenerationConnector(
            LanguageGenerationConnectorRegistry(
                candidates.definition_graph.value_protocol,
                (template,),
            ),
            runtime_policy,
            self._surface_protocol,
            (GenerationSurfaceAttribution(
                template.connector,
                self._hypothesis,
                self._trial_purpose,
            ),),
            self._discourse_declarations,
        )
        return TrialLanguageConnectorAssembly(
            connector,
            candidates,
            order_graph,
            self._hypothesis,
        )

    def clone_for_evaluation(self) -> "TrialLanguageConnectorFactory":
        """复制 owner；评测若调用 trial 仍保持独立图和 H-00 状态。"""
        cloned_candidates = self._candidates.clone_for_graphs(
            self._candidates.definition_graph,
            self._candidates.learning.graph,
        )
        return TrialLanguageConnectorFactory(
            cloned_candidates,
            self._runtime_policy,
            self._surface_protocol,
            self._hypothesis,
            self._trial_purpose,
            _clone_discourse_declarations(self._discourse_declarations),
        )

    def state_key(self) -> tuple:
        """返回图协议、exact Hypothesis、运行策略和 trial purpose。"""
        return (
            tuple(item.stable_key() for item in self._order_predicates),
            tuple(item.stable_key() for item in self._connector_predicates),
            self._candidates.state_key(),
            self._runtime_policy.stable_key(),
            self._surface_protocol.stable_key(),
            self._hypothesis.stable_key(),
            self._trial_purpose.stable_key(),
            (() if self._discourse_declarations is None else (
                self._discourse_declarations.state_key())),
        )


@dataclass(frozen=True)
class ScheduledLanguageConnectorAssembly:
    """保存同一启动快照中的 active 与 exact forming trial connector。"""

    connector: LanguageGenerationConnector
    candidates: LanguageConnectorCandidateRuntime
    order_graph: StructureOrderGraph

    def __post_init__(self) -> None:
        """核验 scheduled registry、候选 owner 和 S-07 facade 属于同次装配。"""
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("scheduled connector assembly connector 类型错误")
        if not isinstance(
                self.connector.registry,
                ScheduledLanguageGenerationConnectorRegistry):
            raise TypeError("scheduled connector assembly 缺少动态调度 registry")
        if not isinstance(self.candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("scheduled connector assembly candidates 类型错误")
        if not isinstance(self.order_graph, StructureOrderGraph):
            raise TypeError("scheduled connector assembly order graph 类型错误")
        if self.candidates.definition_graph.order_graph is not self.order_graph:
            raise ValueError("scheduled connector 未绑定同一 S-07 facade")

    def state_key(self) -> tuple:
        """返回调度快照、候选 owner 和 S-07 图身份。"""
        return (
            self.connector.stable_key(),
            self.candidates.state_key(),
            self.order_graph.ontology.space_identity.stable_key(),
        )


class ScheduledLanguageConnectorFactory(ActiveLanguageConnectorFactory):
    """从当前 owner 一次恢复 active/forming 索引并建立混合 production connector。"""

    def __init__(
            self,
            candidates: LanguageConnectorCandidateRuntime,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            surface_protocol: GenerationSurfaceProtocol,
            active_purpose: ObjectIdentity,
            trial_purpose: ObjectIdentity,
            discourse_declarations: LanguageConnectorDiscourseDeclarationProvider
            | None = None,
            ) -> None:
        """保存候选恢复协议、全部运行策略和互异的 active/trial purpose。"""
        super().__init__(
            candidates,
            runtime_policy,
            surface_protocol,
            active_purpose,
            discourse_declarations,
        )
        if (not isinstance(trial_purpose, ObjectIdentity)
                or trial_purpose.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("scheduled connector trial purpose 类型错误")
        if active_purpose == trial_purpose:
            raise ValueError("scheduled connector active/trial purpose 必须互异")
        self._trial_purpose = trial_purpose

    def build(self, ctx: TrainContext) -> ScheduledLanguageConnectorAssembly:
        """恢复启动快照并建立 active 优先、唯一 forming 才 trial 的局部索引。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("scheduled connector factory ctx 类型错误")
        candidates, order_graph = _rebuild_candidate_owner(
            ctx,
            self._candidates,
            self._order_predicates,
            self._connector_predicates,
        )
        active_entries = candidates.active_template_hypotheses()
        trial_entries = candidates.trial_template_hypotheses()
        registry = ScheduledLanguageGenerationConnectorRegistry(
            candidates.definition_graph.value_protocol,
            active_entries,
            trial_entries,
        )
        connectors = {item.connector for item in registry.templates}
        policies = tuple(
            item for item in self._runtime_policy.templates
            if item.connector in connectors
        )
        runtime_policy = replace(self._runtime_policy, templates=policies)
        attributions = tuple(
            GenerationSurfaceAttribution(
                template.connector,
                hypothesis,
                self._production_purpose,
            )
            for template, hypothesis in registry.active_entries
        ) + tuple(
            GenerationSurfaceAttribution(
                template.connector,
                hypothesis,
                self._trial_purpose,
            )
            for template, hypothesis in registry.trial_entries
        )
        connector = LanguageGenerationConnector(
            registry,
            runtime_policy,
            self._surface_protocol,
            attributions,
            self._discourse_declarations,
        )
        return ScheduledLanguageConnectorAssembly(
            connector,
            candidates,
            order_graph,
        )

    def clone_for_evaluation(self) -> "ScheduledLanguageConnectorFactory":
        """复制候选 owner，评测 build 时从 clone 图重建独立调度索引。"""
        cloned_candidates = self._candidates.clone_for_graphs(
            self._candidates.definition_graph,
            self._candidates.learning.graph,
        )
        return ScheduledLanguageConnectorFactory(
            cloned_candidates,
            self._runtime_policy,
            self._surface_protocol,
            self._production_purpose,
            self._trial_purpose,
            _clone_discourse_declarations(self._discourse_declarations),
        )

    def state_key(self) -> tuple:
        """返回跨图协议、候选快照、运行策略和两个 purpose。"""
        return *super().state_key(), self._trial_purpose.stable_key()


class LanguageConnectorAssemblyFactory(Protocol):
    """重建一个 active 或 exact trial connector assembly。"""

    def build(
            self,
            ctx: TrainContext,
            ) -> (
                ActiveLanguageConnectorAssembly
                | TrialLanguageConnectorAssembly
                | ScheduledLanguageConnectorAssembly):
        """返回绑定当前 context 图的 connector 组件。"""
        ...

    def clone_for_evaluation(self) -> "LanguageConnectorAssemblyFactory":
        """返回不共享候选 owner 的评测 factory。"""
        ...

    def state_key(self) -> tuple:
        """返回图协议、候选和策略的完整状态。"""
        ...


@dataclass(frozen=True)
class LanguageConnectorProductionRuntimeBinding:
    """证明 production runtime 使用同一 connector、S-07 lifecycle 和 R-01 owner。"""

    runtime: ProductionGenerationRuntime
    connector: LanguageGenerationConnector
    order_lifecycle: StructureOrderLifecycleGraph
    alias: AliasRelationRuntime

    def __post_init__(self) -> None:
        """核验 production binding 显式携带四类正确运行 owner。"""
        if not isinstance(self.runtime, ProductionGenerationRuntime):
            raise TypeError("connector production runtime 类型错误")
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("connector production connector 类型错误")
        if not isinstance(self.order_lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("connector production S-07 lifecycle 类型错误")
        if not isinstance(self.alias, AliasRelationRuntime):
            raise TypeError("connector production R-01 owner 类型错误")


class LanguageConnectorProductionRuntimeBuilder(Protocol):
    """用已恢复 connector 组件装配完整 G-00 至 G-04 runtime binding。"""

    def build(
            self,
            ctx: TrainContext,
            assembly: ActiveLanguageConnectorAssembly
            | TrialLanguageConnectorAssembly
            | ScheduledLanguageConnectorAssembly,
            ) -> LanguageConnectorProductionRuntimeBinding:
        """返回显式携带 connector、S-07 和 R-01 owner 的 runtime binding。"""
        ...

    def clone_for_evaluation(
            self,
            ) -> "LanguageConnectorProductionRuntimeBuilder":
        """复制 mapper、verifier、renderer 和关系 owner 配置。"""
        ...

    def state_key(self) -> tuple:
        """返回 builder 的完整可比较配置和可变状态。"""
        ...


@dataclass(frozen=True)
class LanguageConnectorProductionComponents:
    """保存一次 context 独占的 G-01、可选 R-01、renderer 和 G-04 组件。"""

    selector: AnswerContentSelector
    plan_protocol: GenerationPlanProtocol
    structure_protocol: GenerationStructureLayerProtocol
    alias: AliasRelationRuntime | None
    renderer: object
    postcheck_mapper: object | None
    postchecker: GenerationPostcheckRuntime | None

    def __post_init__(self) -> None:
        """核验一次 context 独占组件具备完整的六层与复核协议。"""
        if not isinstance(self.selector, AnswerContentSelector):
            raise TypeError("connector production selector 类型错误")
        if not isinstance(self.plan_protocol, GenerationPlanProtocol):
            raise TypeError("connector production plan protocol 类型错误")
        if not isinstance(
                self.structure_protocol, GenerationStructureLayerProtocol):
            raise TypeError("connector production structure protocol 类型错误")
        if self.alias is not None and not isinstance(
                self.alias, AliasRelationRuntime):
            raise TypeError("connector production alias 类型错误")
        if not hasattr(self.renderer, "render"):
            raise TypeError("connector production renderer 必须实现 render")
        if (self.postcheck_mapper is None) != (self.postchecker is None):
            raise ValueError("connector production G-04 mapper/runtime 必须成对配置")
        if (self.postcheck_mapper is not None
                and not hasattr(self.postcheck_mapper, "build")):
            raise TypeError("connector production G-04 mapper 必须实现 build")
        if (self.postchecker is not None
                and not isinstance(self.postchecker, GenerationPostcheckRuntime)):
            raise TypeError("connector production G-04 runtime 类型错误")


class LanguageConnectorProductionComponentFactory(Protocol):
    """为宿主或 V-06 context 重建不共享可变状态的运行组件。"""

    def build(
            self,
            ctx: TrainContext,
            ) -> LanguageConnectorProductionComponents:
        """返回绑定当前图和当前评测 owner 的全部运行组件。"""
        ...

    def clone_for_evaluation(
            self,
            ) -> "LanguageConnectorProductionComponentFactory":
        """复制策略配置并清空调用、Use、parser 和 verifier 状态。"""
        ...

    def state_key(self) -> tuple:
        """返回组件协议、预算和版本的完整可比较键。"""
        ...


class DefaultLanguageConnectorProductionRuntimeBuilder:
    """把当前 connector、S-07、R-01 和注入组件装成真实 G-00 至 G-04。"""

    def __init__(
            self,
            component_factory: LanguageConnectorProductionComponentFactory,
            relation_factory=None,
            postcheck_factory=None,
            ) -> None:
        """绑定辅助组件，并可由独立版本化课程提供 R-01 和 G-04 owner。"""
        if any(not hasattr(component_factory, method) for method in (
                "build", "clone_for_evaluation", "state_key")):
            raise TypeError("connector production component factory 协议不完整")
        if (relation_factory is not None
                and any(not hasattr(relation_factory, method) for method in (
                    "build", "clone_for_evaluation", "state_key"))):
            raise TypeError("connector production relation factory 协议不完整")
        if (postcheck_factory is not None
                and any(not hasattr(postcheck_factory, method) for method in (
                    "build", "branches", "clone_for_evaluation", "state_key"))):
            raise TypeError("connector production G-04 factory 协议不完整")
        self._component_factory = component_factory
        self._relation_factory = relation_factory
        self._postcheck_factory = postcheck_factory

    def build(
            self,
            ctx: TrainContext,
            assembly: ActiveLanguageConnectorAssembly
            | TrialLanguageConnectorAssembly
            | ScheduledLanguageConnectorAssembly,
            ) -> LanguageConnectorProductionRuntimeBinding:
        """按同一 owner 装配六层 planner、延迟提交 surface 和 G-04 复核。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("connector production builder ctx 类型错误")
        if not isinstance(assembly, (
                ActiveLanguageConnectorAssembly,
                TrialLanguageConnectorAssembly,
                ScheduledLanguageConnectorAssembly)):
            raise TypeError("connector production builder assembly 类型错误")
        precedence = ctx.precedence_relation_runtime
        if precedence is None:
            raise RuntimeError("connector production 缺少 R-06/S-07 runtime")
        lifecycle = getattr(precedence, "lifecycle", None)
        consumer = getattr(precedence, "consumer", None)
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise RuntimeError("connector production 缺少 S-07 lifecycle owner")
        if not isinstance(consumer, StructureOrderConsumer):
            raise RuntimeError("connector production 缺少 S-07 consumer owner")
        source_order = lifecycle.order_graph
        if (source_order.ontology is not ctx.graph_ontology
                or assembly.order_graph.ontology is not ctx.graph_ontology
                or source_order.predicates.refs()
                != assembly.order_graph.predicates.refs()):
            raise ValueError("connector production S-07 图身份与课程理论不一致")
        if source_order is not assembly.order_graph:
            lifecycle = StructureOrderLifecycleGraph(
                assembly.order_graph,
                lifecycle.protocol,
            )
            consumer = StructureOrderConsumer(
                lifecycle,
                consumer.resolver,
                consumer.protocol,
            )
        components = self._component_factory.build(ctx)
        if not isinstance(components, LanguageConnectorProductionComponents):
            raise TypeError("connector production component factory 返回类型错误")
        alias = components.alias
        if self._relation_factory is not None:
            if alias is not None:
                raise ValueError("课程 R-01 factory 与组件 alias owner 不得同时配置")
            alias = self._relation_factory.build(ctx)
        if not isinstance(alias, AliasRelationRuntime):
            raise RuntimeError("connector production 缺少 R-01 owner")
        closure = alias.closure
        if (closure.semantic_graph.ontology is not ctx.graph_ontology
                or closure.candidate_runtime.graph.ontology
                is not ctx.graph_ontology):
            raise ValueError("connector production R-01 组件未绑定当前图")
        if closure.use_owner is None:
            raise RuntimeError(
                "connector production R-01 必须配置 PH2 Core Use owner")

        postcheck_mapper = components.postcheck_mapper
        postchecker = components.postchecker
        if self._postcheck_factory is not None:
            if postcheck_mapper is not None or postchecker is not None:
                raise ValueError("课程 G-04 factory 与组件 G-04 owner 不得同时配置")
            connector_branches = frozenset(
                template.language_branch
                for template in assembly.connector.registry.templates
            )
            postcheck_branches = frozenset(self._postcheck_factory.branches())
            if not connector_branches.issubset(postcheck_branches):
                raise ValueError("G-04 课程未覆盖 connector 的全部 LanguageBranch")
            postcheck_binding = self._postcheck_factory.build(ctx)
            postcheck_mapper = getattr(postcheck_binding, "mapper", None)
            postchecker = getattr(postcheck_binding, "runtime", None)
        if postcheck_mapper is None or postchecker is None:
            raise RuntimeError("connector production 缺少 G-04 owner")
        if not hasattr(postcheck_mapper, "build"):
            raise TypeError("connector production G-04 mapper 必须实现 build")
        if not isinstance(postchecker, GenerationPostcheckRuntime):
            raise TypeError("connector production G-04 runtime 类型错误")

        connector = assembly.connector
        structure_planner = connector.structure_planner()
        execution_planner = GenerationStructureExecutionPlanner(
            lifecycle,
            consumer,
        )
        surface_runtime = GenerationSurfaceRuntime(alias)
        surface_builder = connector.surface_request_builder(
            execution_planner)
        protocol = components.plan_protocol
        structure_protocol = components.structure_protocol
        selector = components.selector
        registrations = (
            GenerationLayerRegistration(
                protocol.stance_layer,
                GenerationStanceLayerResolver(protocol, selector),
            ),
            GenerationLayerRegistration(
                protocol.content_layer,
                GenerationContentLayerResolver(protocol, selector),
            ),
            GenerationLayerRegistration(
                protocol.discourse_layer,
                GenerationDiscourseLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.proposition_layer,
                GenerationPropositionLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.syntax_layer,
                GenerationSyntaxLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.surface_layer,
                GenerationSurfaceLayerResolver(
                    protocol,
                    selector,
                    structure_planner,
                    surface_builder,
                    surface_runtime,
                    commit=False,
                ),
            ),
        )
        executor = TypedGenerationExecutor(
            GenerationPlanner(protocol, registrations),
            components.renderer,
            surface_runtime,
        )
        runtime = ProductionGenerationRuntime(
            SemanticCourseGenerationRequestMapper(),
            executor,
            postcheck_mapper=postcheck_mapper,
            postchecker=postchecker,
        )
        return LanguageConnectorProductionRuntimeBinding(
            runtime,
            connector,
            lifecycle,
            alias,
        )

    def clone_for_evaluation(
            self,
            ) -> "DefaultLanguageConnectorProductionRuntimeBuilder":
        """为 V-06 复制组件 factory，禁止共享 Use、parser 或 verifier 状态。"""
        return DefaultLanguageConnectorProductionRuntimeBuilder(
            self._component_factory.clone_for_evaluation(),
            None if self._relation_factory is None else (
                self._relation_factory.clone_for_evaluation()),
            None if self._postcheck_factory is None else (
                self._postcheck_factory.clone_for_evaluation()),
        )

    def state_key(self) -> tuple:
        """返回组件 factory 的完整配置键。"""
        return (
            self._component_factory.state_key(),
            () if self._relation_factory is None else (
                self._relation_factory.state_key()),
            () if self._postcheck_factory is None else (
                self._postcheck_factory.state_key()),
        )


class LanguageConnectorProductionFactory:
    """原子装配 connector production runtime 与共享候选 owner 的 typed stage4。"""

    def __init__(
            self,
            connector_factory: LanguageConnectorAssemblyFactory,
            runtime_builder: LanguageConnectorProductionRuntimeBuilder,
            stage4_policy: LanguageConnectorStage4Policy,
            ) -> None:
        """绑定 connector、production builder 和同次 stage4 策略。"""
        for label, value, methods in (
                ("connector factory", connector_factory,
                 ("build", "clone_for_evaluation", "state_key")),
                ("runtime builder", runtime_builder,
                 ("build", "clone_for_evaluation", "state_key"))):
            if any(not hasattr(value, method) for method in methods):
                raise TypeError(f"{label} 协议不完整")
        if not isinstance(stage4_policy, LanguageConnectorStage4Policy):
            raise TypeError("connector production stage4 policy 类型错误")
        self._connector_factory = connector_factory
        self._runtime_builder = runtime_builder
        self._stage4_policy = stage4_policy

    def _expected_attributions(
            self,
            assembly: ActiveLanguageConnectorAssembly
            | TrialLanguageConnectorAssembly
            | ScheduledLanguageConnectorAssembly,
            ) -> tuple[GenerationSurfaceAttribution, ...]:
        """从装配快照重建 exact theory/Hypothesis/purpose 归属集合。"""
        if isinstance(assembly, ActiveLanguageConnectorAssembly):
            entries = tuple(
                (
                    template,
                    assembly.candidates.learning.hypothesis_for_candidate(
                        template.connector),
                    self._stage4_policy.active_purpose,
                )
                for template in assembly.connector.registry.templates
            )
        elif isinstance(assembly, TrialLanguageConnectorAssembly):
            entries = ((
                assembly.connector.registry.templates[0],
                assembly.hypothesis,
                self._stage4_policy.trial_purpose,
            ),)
        else:
            registry = assembly.connector.registry
            if not isinstance(
                    registry, ScheduledLanguageGenerationConnectorRegistry):
                raise TypeError("scheduled assembly registry 类型错误")
            entries = tuple(
                (template, hypothesis, self._stage4_policy.active_purpose)
                for template, hypothesis in registry.active_entries
            ) + tuple(
                (template, hypothesis, self._stage4_policy.trial_purpose)
                for template, hypothesis in registry.trial_entries
            )
        return tuple(sorted(
            (
                GenerationSurfaceAttribution(
                    template.connector,
                    hypothesis,
                    purpose,
                )
                for template, hypothesis, purpose in entries
            ),
            key=lambda item: item.stable_key(),
        ))

    def build_installation(
            self,
            ctx: TrainContext,
            ) -> ProductionGenerationInstallation:
        """一次性重建生产组件，并双向核验 S-07/R-01 当前图归属。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("connector production factory ctx 类型错误")
        assembly = self._connector_factory.build(ctx)
        if not isinstance(assembly, (
                ActiveLanguageConnectorAssembly,
                TrialLanguageConnectorAssembly,
                ScheduledLanguageConnectorAssembly)):
            raise TypeError("connector assembly factory 返回类型错误")
        binding = self._runtime_builder.build(ctx, assembly)
        if not isinstance(binding, LanguageConnectorProductionRuntimeBinding):
            raise TypeError("connector runtime builder 返回类型错误")
        if binding.connector is not assembly.connector:
            raise ValueError("production runtime 未使用本次恢复的 connector")
        attributions = tuple(sorted(
            assembly.connector.attribution_mapper.attributions.values(),
            key=lambda item: item.stable_key(),
        ))
        if attributions != self._expected_attributions(assembly):
            raise ValueError("production connector purpose 与 stage4 policy 不一致")
        if binding.order_lifecycle.order_graph is not assembly.order_graph:
            raise ValueError("production runtime 未使用本次恢复的 S-07 lifecycle")
        if (binding.alias.closure.semantic_graph.ontology
                is not ctx.graph_ontology):
            raise ValueError("production runtime R-01 owner 未绑定当前 context 图")
        if (assembly.candidates.definition_graph.ontology
                is not ctx.graph_ontology):
            raise ValueError("production stage4 候选 owner 未绑定当前 context 图")
        return ProductionGenerationInstallation(
            binding.runtime,
            LanguageConnectorStage4Runtime(
                assembly.candidates,
                self._stage4_policy,
            ),
        )

    def build(self, ctx: TrainContext) -> ProductionGenerationRuntime:
        """兼容基础 factory 协议；正式安装应复用同次 installation。"""
        return self.build_installation(ctx).runtime

    def clone_for_evaluation(self) -> "LanguageConnectorProductionFactory":
        """同时复制 connector owner 与 G-00 至 G-04 runtime builder。"""
        return LanguageConnectorProductionFactory(
            self._connector_factory.clone_for_evaluation(),
            self._runtime_builder.clone_for_evaluation(),
            self._stage4_policy,
        )

    def state_key(self) -> tuple:
        """返回 connector、production builder 和 stage4 policy 完整状态。"""
        return (
            self._connector_factory.state_key(),
            self._runtime_builder.state_key(),
            self._stage4_policy.stable_key(),
        )


__all__ = [
    "ActiveLanguageConnectorAssembly",
    "ActiveLanguageConnectorFactory",
    "DefaultLanguageConnectorProductionRuntimeBuilder",
    "LanguageConnectorAssemblyFactory",
    "LanguageConnectorProductionComponentFactory",
    "LanguageConnectorProductionComponents",
    "LanguageConnectorProductionFactory",
    "LanguageConnectorProductionRuntimeBinding",
    "LanguageConnectorProductionRuntimeBuilder",
    "ScheduledLanguageConnectorAssembly",
    "ScheduledLanguageConnectorFactory",
    "TrialLanguageConnectorAssembly",
    "TrialLanguageConnectorFactory",
]
