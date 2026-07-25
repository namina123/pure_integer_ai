"""R-08 逻辑候选 H-05 采用闸与 S-04 执行编排。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
    CandidateRecognitionRequest,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicDerivedEvidenceBundle,
    LogicDerivedEvidenceSeed,
    LogicOperatorAdoption,
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
    LogicOperatorExecutionUse,
    build_logic_derived_evidence,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    AtomEvidenceResolver,
    LogicExecutor,
    LogicFailureProtocol,
    LogicOperatorRegistry,
    ModalResolver,
    QuantifierResolver,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    BoundProposition,
    PropositionTemplateGraph,
    SubstitutionProtocol,
    TypeCompatibilityResolver,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.train_context import TrainContext


class LogicClosureConflictError(RuntimeError):
    """同一 StructureConcept 同时出现多个可采用逻辑解释。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验课程、builder 和执行路由使用非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为 builder 状态中的可变长稳定键增加长度前缀。"""
    return len(value), *value


@dataclass(frozen=True)
class LogicClosureFormationPlan:
    """声明一个逻辑作用候选及其 forming 起始逻辑序。"""

    spec: LogicOperatorCandidateSpec
    timestamp_base: int = 0

    def __post_init__(self) -> None:
        """拒绝无类型候选和负数或布尔型 forming 逻辑序。"""
        if not isinstance(self.spec, LogicOperatorCandidateSpec):
            raise TypeError("logic formation spec 类型错误")
        assert_int(self.timestamp_base, _where="LogicClosureFormationPlan")
        if type(self.timestamp_base) is not int or self.timestamp_base < 0:
            raise ValueError("logic formation timestamp_base 必须为非负严格整数")


@dataclass(frozen=True)
class LogicClosureRecognitionPlan:
    """声明一次不含宿主时间戳的独立逻辑候选核验。"""

    candidate: ObjectIdentity
    observation: SourceRef
    event_key: tuple[int, ...]
    visible_inputs: tuple[ObjectIdentity, ...]
    revealed: RevealedObjectObservation
    scorers: tuple = ()
    archive_refuted: bool = False
    replacement: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验来源、事件、候选和 reveal 路由，不接受隐式 predicted。"""
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("logic recognition candidate 类型错误")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("logic recognition observation 类型错误")
        _strict_key(self.event_key, label="logic recognition event_key")
        if not isinstance(self.visible_inputs, tuple):
            raise TypeError("logic recognition visible_inputs 必须是 tuple")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.visible_inputs):
            raise TypeError("logic recognition visible_inputs 元素类型错误")
        if not isinstance(self.revealed, RevealedObjectObservation):
            raise TypeError("logic recognition revealed 类型错误")
        if (self.revealed.observation != self.observation
                or self.revealed.event_key != self.event_key):
            raise ValueError("logic recognition reveal 替换了来源或事件")
        if not isinstance(self.scorers, tuple):
            raise TypeError("logic recognition scorers 必须是 tuple")
        if type(self.archive_refuted) is not bool:
            raise TypeError("logic recognition archive_refuted 必须是 bool")
        if (self.replacement is not None
                and not isinstance(self.replacement, ObjectIdentity)):
            raise TypeError("logic recognition replacement 类型错误")

    def route_key(self) -> tuple:
        """返回同轮 recognition 去重使用的完整对象路由。"""
        return self.candidate, self.observation, self.event_key


@dataclass(frozen=True)
class LogicClosureExecutionPlan:
    """声明一次 typed 逻辑执行及其独立 provisional Evidence 形成。"""

    root: BoundProposition
    use_key: tuple[int, ...]
    source: SourceRef
    graph: PropositionTemplateGraph
    environment: BindingEnvironment
    atom_resolver: AtomEvidenceResolver
    failures: LogicFailureProtocol
    substitution: SubstitutionProtocol
    type_resolver: TypeCompatibilityResolver
    binding_failures: BindingFailureProtocol
    hypothesis: HypothesisKey
    evidence_seeds: tuple[LogicDerivedEvidenceSeed, ...]
    quantifier_resolver: QuantifierResolver | None = None
    modal_resolver: ModalResolver | None = None
    inherited_binders: tuple = ()

    def __post_init__(self) -> None:
        """核验执行依赖均显式注入，禁止 course 以答案或表层文字替代。"""
        if not isinstance(self.root, BoundProposition):
            raise TypeError("logic execution root 类型错误")
        _strict_key(self.use_key, label="logic execution use_key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("logic execution source 类型错误")
        if not isinstance(self.graph, PropositionTemplateGraph):
            raise TypeError("logic execution graph 类型错误")
        if not isinstance(self.environment, BindingEnvironment):
            raise TypeError("logic execution environment 类型错误")
        if not hasattr(self.atom_resolver, "resolve"):
            raise TypeError("logic execution atom_resolver 缺少 resolve")
        if not isinstance(self.failures, LogicFailureProtocol):
            raise TypeError("logic execution failures 类型错误")
        if not isinstance(self.substitution, SubstitutionProtocol):
            raise TypeError("logic execution substitution 类型错误")
        if not hasattr(self.type_resolver, "resolve"):
            raise TypeError("logic execution type_resolver 缺少 resolve")
        if not isinstance(self.binding_failures, BindingFailureProtocol):
            raise TypeError("logic execution binding_failures 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("logic execution hypothesis 类型错误")
        if (not isinstance(self.evidence_seeds, tuple)
                or not self.evidence_seeds
                or any(not isinstance(item, LogicDerivedEvidenceSeed)
                       for item in self.evidence_seeds)):
            raise TypeError("logic execution evidence_seeds 类型错误")
        if (self.quantifier_resolver is not None
                and not hasattr(self.quantifier_resolver, "resolve")):
            raise TypeError("logic execution quantifier_resolver 缺少 resolve")
        if (self.modal_resolver is not None
                and not hasattr(self.modal_resolver, "resolve")):
            raise TypeError("logic execution modal_resolver 缺少 resolve")
        if not isinstance(self.inherited_binders, tuple):
            raise TypeError("logic execution inherited_binders 必须是 tuple")


@dataclass(frozen=True)
class LogicClosureRoundRequest:
    """一个来源 scope 中的逻辑候选学习和 typed 执行请求。"""

    scope: ScopeIdentity
    formations: tuple[LogicClosureFormationPlan, ...] = ()
    recognitions: tuple[LogicClosureRecognitionPlan, ...] = ()
    executions: tuple[LogicClosureExecutionPlan, ...] = ()

    def __post_init__(self) -> None:
        """全量拒绝错 scope、重复候选、重复路由和重复 Evidence id。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("logic round scope 类型错误")
        groups = (
            (self.formations, LogicClosureFormationPlan, "formations"),
            (self.recognitions, LogicClosureRecognitionPlan, "recognitions"),
            (self.executions, LogicClosureExecutionPlan, "executions"),
        )
        for values, expected, label in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"logic round {label} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"logic round {label} 元素类型错误")
        candidates = tuple(item.spec.candidate for item in self.formations)
        if len(set(candidates)) != len(candidates):
            raise ValueError("同一 logic round 不得重复 forming 候选")
        routes = tuple(item.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("同一 logic round 不得重复 recognition 路由")
        if any(item.revealed.scope != self.scope
               for item in self.recognitions):
            raise ValueError("logic recognition 必须绑定当前 round scope")
        use_keys = tuple(item.use_key for item in self.executions)
        if len(set(use_keys)) != len(use_keys):
            raise ValueError("同一 logic round 不得重复 execution use_key")
        evidence_ids = tuple(
            seed.evidence_id
            for item in self.executions
            for seed in item.evidence_seeds
        )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("同一 logic round 不得重复派生 Evidence id")
        for item in self.executions:
            if item.hypothesis.scope != self.scope:
                raise ValueError("logic execution Hypothesis 必须绑定当前 scope")
            if item.hypothesis.observation != item.source:
                raise ValueError("logic execution Hypothesis 必须绑定当前来源")
            if self.scope.source is not None and self.scope.source != item.source:
                raise ValueError("logic execution source 必须匹配来源化 round scope")


@dataclass(frozen=True)
class LogicClosureRoundReport:
    """保存一次生产逻辑轮的 forming、核验和 G-00 派生候选。"""

    scope: ScopeIdentity
    read_only: bool
    formations: tuple[HypothesisKey, ...]
    recognitions: tuple[CandidateLearningOutcome, ...]
    executions: tuple[LogicDerivedEvidenceBundle, ...]

    def __post_init__(self) -> None:
        """核验报告容器，read-only 轮不得携带任何 H-05 写入。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("logic report scope 类型错误")
        if type(self.read_only) is not bool:
            raise TypeError("logic report read_only 必须是 bool")
        checks = (
            (self.formations, HypothesisKey),
            (self.recognitions, CandidateLearningOutcome),
            (self.executions, LogicDerivedEvidenceBundle),
        )
        for values, expected in checks:
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, expected) for item in values)):
                raise TypeError("logic report 字段类型错误")
        if self.read_only and (self.formations or self.recognitions):
            raise ValueError("read-only logic report 不得含 H-05 写入")


@dataclass(frozen=True)
class LogicOperatorRegistrySnapshot:
    """当前 H-05 active 候选投影出的 S-04 registry 和冲突结构。"""

    registry: LogicOperatorRegistry
    adoptions: tuple[LogicOperatorAdoption, ...]
    conflicted_structures: tuple

    def __post_init__(self) -> None:
        """核验 registry 快照只携带 typed adoption 和结构身份。"""
        if not isinstance(self.registry, LogicOperatorRegistry):
            raise TypeError("logic registry snapshot registry 类型错误")
        if any(not isinstance(item, LogicOperatorAdoption)
               for item in self.adoptions):
            raise TypeError("logic registry snapshot adoption 类型错误")


class LogicClosureRuntime:
    """复用一个 H-05 owner 管理全部逻辑作用候选并执行 adopted 子集。"""

    def __init__(
            self,
            candidate_runtime: CandidateLearningRuntime,
            protocol: LogicOperatorCandidateProtocol,
            specs: tuple[LogicOperatorCandidateSpec, ...] = (),
            ) -> None:
        """绑定候选 owner、图协议和调用方可恢复的执行 adapter。"""
        if not isinstance(candidate_runtime, CandidateLearningRuntime):
            raise TypeError("logic candidate_runtime 类型错误")
        if not isinstance(protocol, LogicOperatorCandidateProtocol):
            raise TypeError("logic candidate protocol 类型错误")
        if any(not isinstance(item, LogicOperatorCandidateSpec)
               for item in specs):
            raise TypeError("logic specs 元素类型错误")
        candidates = tuple(item.candidate for item in specs)
        if len(set(candidates)) != len(candidates):
            raise ValueError("logic specs 不得重复候选 Proposition")
        self.candidate_runtime = candidate_runtime
        self.protocol = protocol
        self._specs = {item.candidate: item for item in specs}

    def form(
            self, spec: LogicOperatorCandidateSpec, *,
            timestamp_base: int = 0,
            ):
        """登记逻辑候选 Proposition 和 forming Evidence，不直接激活 handler。"""
        return self.form_many(((spec, timestamp_base),))[0]

    def form_many(
            self,
            requests: tuple[tuple[LogicOperatorCandidateSpec, int], ...],
            ) -> tuple[HypothesisKey, ...]:
        """整批预检后登记逻辑候选，失败时不修改 runtime spec 目录。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("logic form_many requests 必须是非空 tuple")
        pending = dict(self._specs)
        definitions = []
        candidates = []
        for item in requests:
            if (not isinstance(item, tuple) or len(item) != 2
                    or not isinstance(item[0], LogicOperatorCandidateSpec)):
                raise TypeError("logic form_many request 类型错误")
            spec, timestamp_base = item
            assert_int(timestamp_base, _where="LogicClosureRuntime.form_many")
            if type(timestamp_base) is not int or timestamp_base < 0:
                raise ValueError("logic forming 逻辑序必须为非负严格整数")
            prior = pending.get(spec.candidate)
            if prior is not None and prior != spec:
                raise ValueError("同一逻辑候选 Proposition 绑定了不同执行定义")
            pending[spec.candidate] = spec
            candidates.append(spec.candidate)
            definitions.append((
                spec.candidate_definition(self.protocol),
                timestamp_base,
            ))
        if len(set(candidates)) != len(candidates):
            raise ValueError("logic form_many 不得重复候选 Proposition")
        hypotheses = self.candidate_runtime.register_many(tuple(definitions))
        self._specs = pending
        return hypotheses

    def specs(self) -> tuple[LogicOperatorCandidateSpec, ...]:
        """返回当前 owner 已知的候选 spec，按一等候选身份稳定排序。"""
        return tuple(sorted(
            self._specs.values(),
            key=lambda item: item.candidate.stable_key(),
        ))

    def state_key(self) -> tuple:
        """返回 H-05 owner、逻辑协议和候选 spec 的完整隔离键。"""
        return (
            self.protocol.stable_key(),
            self.candidate_runtime.state_key(),
            tuple(item.stable_key(self.protocol) for item in self.specs()),
        )

    def clone_for_graph(
            self, graph: CandidateProjectionGraph,
            ) -> "LogicClosureRuntime":
        """在隔离图上复制 H-05 owner，不共享训练侧候选可变状态。"""
        if not isinstance(graph, CandidateProjectionGraph):
            raise TypeError("logic clone graph 类型错误")
        return LogicClosureRuntime(
            self.candidate_runtime.clone_for_graph(graph),
            self.protocol,
            self.specs(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "LogicClosureRuntime":
        """在评测 TrainContext 的独立图上复制完整逻辑候选 owner。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("logic clone ctx 类型错误")
        graph = CandidateProjectionGraph(
            ctx.graph_ontology,
            self.candidate_runtime.graph.protocol,
        )
        return self.clone_for_graph(graph)

    def recognize(
            self, request: CandidateRecognitionRequest,
            ) -> CandidateLearningOutcome:
        """提交独立 reveal、H-04 decision 和候选图 lifecycle 投影。"""
        if not isinstance(request, CandidateRecognitionRequest):
            raise TypeError("logic recognition request 类型错误")
        return self.candidate_runtime.recognize_many((request,))[0]

    def recognize_many(
            self, requests: tuple[CandidateRecognitionRequest, ...],
            ) -> tuple[CandidateLearningOutcome, ...]:
        """整批提交已由生产 runtime 分配逻辑序的候选核验。"""
        if (not isinstance(requests, tuple) or not requests
                or any(not isinstance(item, CandidateRecognitionRequest)
                       for item in requests)):
            raise TypeError("logic recognize_many requests 类型错误")
        return self.candidate_runtime.recognize_many(requests)

    def adoption(
            self, spec: LogicOperatorCandidateSpec,
            ) -> LogicOperatorAdoption | None:
        """读取 active+supported+adopted 候选及当前有效 Evidence。"""
        if not isinstance(spec, LogicOperatorCandidateSpec):
            raise TypeError("logic adoption spec 类型错误")
        expected = spec.candidate_definition(self.protocol)
        try:
            hypothesis = self.candidate_runtime.hypothesis_for_candidate(
                spec.candidate)
        except KeyError:
            return None
        if self.candidate_runtime.engine.definition(hypothesis) != expected:
            raise ValueError("logic candidate H-05 定义与执行 spec 不一致")
        active = self.candidate_runtime.engine.active(hypothesis)
        if active is None:
            return None
        candidate_ref = self.candidate_runtime.graph.ontology.resolve(
            spec.candidate)
        if candidate_ref is None:
            raise ValueError("active logic candidate 未物化到候选图")
        projection = self.candidate_runtime.graph.project(candidate_ref)
        if projection.state != self.candidate_runtime.graph.protocol.active_state:
            raise ValueError("H-04 active 与逻辑候选图状态不一致")
        active_ids = frozenset((
            *active.snapshot.support_evidence_ids,
            *active.snapshot.refute_evidence_ids,
            *active.snapshot.unknown_evidence_ids,
        ))
        evidence = tuple(
            item for item in self.candidate_runtime.engine.ledger.evidence_history(
                hypothesis)
            if item.evidence_id in active_ids
        )
        if {item.evidence_id for item in evidence} != active_ids:
            raise ValueError("logic adoption 未恢复全部当前有效 Evidence")
        return LogicOperatorAdoption(
            spec,
            hypothesis,
            evidence,
            active.decision,
            projection,
        )

    def registry_snapshot(self) -> LogicOperatorRegistrySnapshot:
        """只把唯一 active 解释装入 S-04 registry，竞争冲突保持 fail closed。"""
        grouped: dict[object, list[LogicOperatorAdoption]] = {}
        for spec in sorted(
                self._specs.values(),
                key=lambda item: item.candidate.stable_key()):
            adoption = self.adoption(spec)
            if adoption is None:
                continue
            grouped.setdefault(spec.definition.structure, []).append(adoption)
        accepted = []
        conflicts = []
        for structure in sorted(grouped, key=lambda item: item.stable_key()):
            candidates = grouped[structure]
            if len(candidates) == 1:
                accepted.append(candidates[0])
            else:
                conflicts.append(structure)
        registry = LogicOperatorRegistry(tuple(
            item.spec.definition for item in accepted))
        return LogicOperatorRegistrySnapshot(
            registry,
            tuple(accepted),
            tuple(conflicts),
        )

    def execute(
            self,
            root: BoundProposition,
            *,
            use_key: tuple[int, ...],
            source,
            scope: ScopeIdentity,
            graph: PropositionTemplateGraph,
            environment: BindingEnvironment,
            atom_resolver: AtomEvidenceResolver,
            failures: LogicFailureProtocol,
            substitution: SubstitutionProtocol,
            type_resolver: TypeCompatibilityResolver,
            binding_failures: BindingFailureProtocol,
            quantifier_resolver: QuantifierResolver | None = None,
            modal_resolver: ModalResolver | None = None,
            inherited_binders: tuple = (),
            ) -> LogicOperatorExecutionUse:
        """用当前 active registry 执行根命题并记录实际采用候选。"""
        snapshot = self.registry_snapshot()
        executor = LogicExecutor(
            snapshot.registry,
            atom_resolver,
            failures,
            substitution,
            type_resolver,
            binding_failures,
        )
        evaluation = executor.evaluate(
            root,
            source=source,
            scope=scope,
            graph=graph,
            environment=environment,
            quantifier_resolver=quantifier_resolver,
            modal_resolver=modal_resolver,
            inherited_binders=inherited_binders,
        )
        used = {item.operator for item in evaluation.derivation}
        adoptions = tuple(
            item for item in snapshot.adoptions
            if item.spec.definition.structure in used
        )
        return LogicOperatorExecutionUse(
            use_key,
            evaluation,
            adoptions,
            snapshot.conflicted_structures,
        )


@dataclass(frozen=True)
class TrainingLogicClosureRuntimeBuilder:
    """用完整注入协议从 Core 训练历史恢复或创建 R-08 候选 owner。"""

    builder_key: tuple[int, ...]
    learning_protocol: EvidenceCandidateProtocol
    projection_protocol: CandidateProjectionProtocol
    verifier_protocol: IndependentVerifierProtocol
    projection_metadata: CandidateProjectionMetadata
    operator_protocol: LogicOperatorCandidateProtocol
    history_protocol: TrainingHypothesisHistoryProtocol
    specs: tuple[LogicOperatorCandidateSpec, ...]

    def __post_init__(self) -> None:
        """核验历史边界、图协议和 handler 目录均由调用方完整注入。"""
        _strict_key(self.builder_key, label="logic training builder_key")
        if not isinstance(self.learning_protocol, EvidenceCandidateProtocol):
            raise TypeError("logic training learning_protocol 类型错误")
        if not isinstance(self.projection_protocol, CandidateProjectionProtocol):
            raise TypeError("logic training projection_protocol 类型错误")
        if not isinstance(self.verifier_protocol, IndependentVerifierProtocol):
            raise TypeError("logic training verifier_protocol 类型错误")
        if not isinstance(self.projection_metadata, CandidateProjectionMetadata):
            raise TypeError("logic training projection_metadata 类型错误")
        if not isinstance(self.operator_protocol, LogicOperatorCandidateProtocol):
            raise TypeError("logic training operator_protocol 类型错误")
        if not isinstance(
                self.history_protocol, TrainingHypothesisHistoryProtocol):
            raise TypeError("logic training history_protocol 类型错误")
        if (not isinstance(self.specs, tuple)
                or any(not isinstance(item, LogicOperatorCandidateSpec)
                       for item in self.specs)):
            raise TypeError("logic training specs 类型错误")
        candidates = tuple(item.candidate for item in self.specs)
        if len(set(candidates)) != len(candidates):
            raise ValueError("logic training specs 不得重复候选")
        learning = self.learning_protocol
        history = self.history_protocol
        if (history.hypothesis_kind != learning.hypothesis_kind_key
                or history.aggregate_source != learning.aggregate_source
                or history.aggregate_scope != learning.aggregate_scope):
            raise ValueError("logic training history 与 H-00 aggregate 协议不一致")

    def build(self, ctx: TrainContext) -> LogicClosureRuntime:
        """在当前 Core 图和训练历史上恢复可继续追加的逻辑候选 owner。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("logic training builder ctx 类型错误")
        history = ctx.training_candidate_history
        if history is None:
            raise ValueError("logic training builder 缺少 Core 候选历史")
        graph = CandidateProjectionGraph(
            ctx.graph_ontology,
            self.projection_protocol,
        )
        candidate_runtime = CandidateLearningRuntime.restore_for_training_graph(
            self.learning_protocol,
            graph,
            IndependentObjectVerifier(self.verifier_protocol),
            self.projection_metadata,
            history,
            self.history_protocol,
        )
        expected = {
            item.candidate_definition(self.operator_protocol)
            for item in self.specs
        }
        restored = set(candidate_runtime.engine.definitions())
        if not restored.issubset(expected):
            raise ValueError("logic Core 历史包含当前 builder 未声明的候选")
        for definition in expected - restored:
            hypothesis = definition.hypothesis(self.learning_protocol)
            candidate_ref = ctx.graph_ontology.resolve(definition.candidate)
            has_lifecycle = (
                candidate_ref is not None and bool(graph.history(candidate_ref))
            )
            if (ctx.graph_ontology.resolve(hypothesis.object_identity())
                    is not None or has_lifecycle):
                raise ValueError("logic 候选图存在但 Core 历史缺失")
        return LogicClosureRuntime(
            candidate_runtime,
            self.operator_protocol,
            self.specs,
        )

    def clone_for_evaluation(self) -> "TrainingLogicClosureRuntimeBuilder":
        """返回不携带 backend、ledger 或图 facade 的冻结 builder 副本。"""
        return TrainingLogicClosureRuntimeBuilder(
            self.builder_key,
            self.learning_protocol,
            self.projection_protocol,
            self.verifier_protocol,
            self.projection_metadata,
            self.operator_protocol,
            self.history_protocol,
            self.specs,
        )

    def state_key(self) -> tuple[int, ...]:
        """返回全部协议、元数据、历史边界和 handler 版本声明的稳定键。"""
        learning = self.learning_protocol
        projection = self.projection_protocol
        verifier = self.verifier_protocol
        metadata = self.projection_metadata
        result = [1, *_packed(self.builder_key)]
        result.extend(_packed(learning.hypothesis_kind_key))
        result.extend(_packed(learning.formation_reason_key))
        result.extend(_packed(learning.aggregate_source.stable_key()))
        result.extend(_packed(learning.aggregate_scope.stable_key()))
        result.append(learning.minimum_forming_sources)
        for identity in (
                *projection.predicate_identities(),
                *projection.state_identities(),
                *projection.kind_identities()):
            result.extend(_packed(identity.stable_key()))
        result.extend(_packed(projection.event_namespace_key))
        result.extend(_packed(verifier.authority.stable_key()))
        result.extend(_packed(verifier.authority_version))
        result.extend(_packed(verifier.support_reason_key))
        result.extend(_packed(verifier.refute_reason_key))
        result.extend(_packed(verifier.unknown_reason_key))
        result.extend((
            metadata.provenance_kind,
            metadata.epistemic_origin,
            metadata.content_version,
            len(metadata.qualifiers),
            *metadata.qualifiers,
        ))
        result.extend(_packed(self.operator_protocol.stable_key()))
        result.extend(_packed(self.history_protocol.stable_key()))
        ordered_specs = tuple(sorted(
            self.specs,
            key=lambda item: item.candidate.stable_key(),
        ))
        result.append(len(ordered_specs))
        for item in ordered_specs:
            result.extend(_packed(item.stable_key(self.operator_protocol)))
        return tuple(result)


@runtime_checkable
class LogicClosureRuntimeBuilder(Protocol):
    """由项目课程注入完整 H-05 owner、图协议和逻辑 adapter。"""

    def build(self, ctx: TrainContext) -> LogicClosureRuntime:
        """在指定 TrainContext 图上构造 R-08 owner。"""
        ...

    def clone_for_evaluation(self) -> "LogicClosureRuntimeBuilder":
        """返回不共享宿主可变状态的评测 builder。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回协议、resolver 和课程版本的完整整数键。"""
        ...


@runtime_checkable
class LogicClosureCourse(Protocol):
    """把来源 scope 映射为 typed 逻辑候选学习或执行请求。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> LogicClosureRoundRequest:
        """返回当前训练或 held-out 来源的 R-08 请求。"""
        ...

    def clone_for_evaluation(self) -> "LogicClosureCourse":
        """返回不共享可变游标或宿主引用的评测课程副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程 mapper、来源策略和版本的完整整数键。"""
        ...


class LogicClosureCourseRuntime:
    """让 formal round 只提交 scope，由课程完成 R-08 学习和执行。"""

    def __init__(
            self,
            ctx: TrainContext,
            owner: LogicClosureRuntime,
            builder: LogicClosureRuntimeBuilder,
            course: LogicClosureCourse,
            ) -> None:
        """绑定当前 context、唯一逻辑 owner、builder 和可克隆课程。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("logic course ctx 类型错误")
        if not isinstance(owner, LogicClosureRuntime):
            raise TypeError("logic course owner 类型错误")
        if not isinstance(builder, LogicClosureRuntimeBuilder):
            raise TypeError("logic builder 协议不完整")
        if not isinstance(course, LogicClosureCourse):
            raise TypeError("logic course 协议不完整")
        _strict_key(builder.state_key(), label="LogicClosureRuntimeBuilder")
        _strict_key(course.state_key(), label="LogicClosureCourse")
        self.ctx = ctx
        self.owner = owner
        self.builder = builder
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> LogicClosureRoundReport:
        """先核验完整课程请求，再按 forming、recognition、execution 提交。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("logic process scope 类型错误")
        if type(read_only) is not bool:
            raise TypeError("logic process read_only 必须是 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, LogicClosureRoundRequest):
            raise TypeError("logic course.request 返回类型错误")
        if request.scope != scope:
            raise ValueError("logic course.request 替换了 round scope")
        if read_only and (request.formations or request.recognitions):
            raise ValueError("read-only logic 请求不得形成或核验候选")

        formation_candidates = {
            item.spec.candidate for item in request.formations
        }
        prospective_hypotheses: dict[ObjectIdentity, HypothesisKey] = {}
        formation_definitions = tuple(
            (
                item.spec.candidate_definition(self.owner.protocol),
                item.timestamp_base,
            )
            for item in request.formations
        )
        for item, (definition, _timestamp_base) in zip(
                request.formations, formation_definitions, strict=True):
            prospective_hypotheses[item.spec.candidate] = (
                definition.hypothesis(
                    self.owner.candidate_runtime.engine.protocol))
        for item in request.recognitions:
            if item.candidate in formation_candidates:
                pass
            else:
                try:
                    prospective_hypotheses[item.candidate] = (
                        self.owner.candidate_runtime.hypothesis_for_candidate(
                            item.candidate))
                except KeyError as exc:
                    raise ValueError(
                        "logic recognition 候选缺少 forming") from exc
            if item.replacement is not None:
                if item.replacement in formation_candidates:
                    continue
                try:
                    prospective_hypotheses[item.replacement] = (
                        self.owner.candidate_runtime.hypothesis_for_candidate(
                            item.replacement))
                except KeyError as exc:
                    raise ValueError(
                        "logic recognition replacement 缺少 forming") from exc

        formation_requests = tuple(
            (item.spec, item.timestamp_base) for item in request.formations)
        formation_end = max((
            item.timestamp_base + len(item.spec.forming_sources) - 1
            for item in request.formations
        ), default=-1)
        recognition_requests = ()
        if request.recognitions:
            next_timestamp = self.owner.candidate_runtime.next_timestamps(1)[0]
            start = max(next_timestamp, formation_end + 1)
            timestamps = tuple(range(
                start,
                start + len(request.recognitions) * 3,
            ))
            built = []
            for index, item in enumerate(request.recognitions):
                hypothesis = prospective_hypotheses[item.candidate]
                replacement = None
                if item.replacement is not None:
                    replacement = prospective_hypotheses[item.replacement]
                built.append(CandidateRecognitionRequest(
                    hypothesis,
                    item.observation,
                    scope,
                    item.event_key,
                    item.visible_inputs,
                    item.candidate,
                    item.revealed,
                    timestamps[index * 3],
                    timestamps[index * 3 + 1],
                    timestamps[index * 3 + 2],
                    item.scorers,
                    item.archive_refuted,
                    replacement,
                ))
            recognition_requests = tuple(built)
        self._preflight_learning(
            formation_definitions,
            recognition_requests,
        )
        formations = (
            self.owner.form_many(formation_requests)
            if formation_requests else ()
        )
        expected_formations = tuple(
            prospective_hypotheses[item.spec.candidate]
            for item in request.formations
        )
        if formations != expected_formations:
            raise RuntimeError("logic forming 预检与正式 Hypothesis 不一致")
        recognitions = (
            self.owner.recognize_many(recognition_requests)
            if recognition_requests else ()
        )

        executions = []
        for item in request.executions:
            execution = self.owner.execute(
                item.root,
                use_key=item.use_key,
                source=item.source,
                scope=scope,
                graph=item.graph,
                environment=item.environment,
                atom_resolver=item.atom_resolver,
                failures=item.failures,
                substitution=item.substitution,
                type_resolver=item.type_resolver,
                binding_failures=item.binding_failures,
                quantifier_resolver=item.quantifier_resolver,
                modal_resolver=item.modal_resolver,
                inherited_binders=item.inherited_binders,
            )
            executions.append(build_logic_derived_evidence(
                execution,
                item.hypothesis,
                item.evidence_seeds,
            ))
        return LogicClosureRoundReport(
            scope,
            read_only,
            formations,
            recognitions,
            tuple(executions),
        )

    def _preflight_learning(
            self,
            formations: tuple[tuple[object, int], ...],
            recognitions: tuple[CandidateRecognitionRequest, ...],
            ) -> None:
        """在首写前用 owner clone 预演 forming、reveal 和 H-04 决策。"""
        if not formations and not recognitions:
            return
        probe = self.owner.candidate_runtime.engine.clone()
        if formations:
            probe.register_many(formations)
        for request in recognitions:
            prediction = probe.predict(
                request.hypothesis,
                observation=request.observation,
                scope=request.scope,
                event_key=request.event_key,
                visible_inputs=request.visible_inputs,
                predicted=request.predicted,
            )
            verification = self.owner.candidate_runtime.verifier.verify(
                prediction,
                request.revealed,
            )
            probe.reveal(
                prediction,
                verification,
                timestamp_seq=request.timestamp_seq,
            )
            probe.resolve(
                request.hypothesis,
                timestamp_seq=request.resolve_timestamp_seq,
                scorers=request.scorers,
                archive_refuted=request.archive_refuted,
                replacement=request.replacement,
            )

    def clone_for_context(
            self, ctx: TrainContext,
            ) -> "LogicClosureCourseRuntime":
        """在评测 context 上复制 owner、builder 和课程，隔离 H-05 写状态。"""
        cloned_builder = self.builder.clone_for_evaluation()
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_builder, LogicClosureRuntimeBuilder):
            raise TypeError("logic builder clone 协议不完整")
        if not isinstance(cloned_course, LogicClosureCourse):
            raise TypeError("logic course clone 协议不完整")
        if cloned_builder.state_key() != self.builder.state_key():
            raise ValueError("logic builder clone 改变协议状态")
        if cloned_course.state_key() != self.course.state_key():
            raise ValueError("logic course clone 改变课程状态")
        return LogicClosureCourseRuntime(
            ctx,
            self.owner.clone_for_context(ctx),
            cloned_builder,
            cloned_course,
        )

    def state_key(self) -> tuple:
        """返回 builder、课程和 H-05 owner 的完整隔离状态。"""
        return (
            self.builder.state_key(),
            self.course.state_key(),
            self.owner.state_key(),
        )


def install_logic_closure_runtime(
        ctx: TrainContext,
        builder: LogicClosureRuntimeBuilder,
        course: LogicClosureCourse,
        ) -> LogicClosureCourseRuntime:
    """在 TrainContext 上安装成对注入且默认关闭的 R-08 生产 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install logic closure ctx 类型错误")
    if not isinstance(builder, LogicClosureRuntimeBuilder):
        raise TypeError("logic builder 协议不完整")
    if not isinstance(course, LogicClosureCourse):
        raise TypeError("logic course 协议不完整")
    if getattr(ctx, "logic_closure_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 logic closure runtime")
    _strict_key(builder.state_key(), label="LogicClosureRuntimeBuilder")
    _strict_key(course.state_key(), label="LogicClosureCourse")
    owner = builder.build(ctx)
    if not isinstance(owner, LogicClosureRuntime):
        raise TypeError("logic builder.build 返回类型错误")
    if owner.candidate_runtime.graph.ontology is not ctx.graph_ontology:
        raise ValueError("logic owner 未绑定当前 TrainContext 图")
    runtime = LogicClosureCourseRuntime(ctx, owner, builder, course)
    ctx.logic_closure_runtime = runtime
    return runtime


__all__ = [
    "LogicClosureCourse",
    "LogicClosureCourseRuntime",
    "LogicClosureConflictError",
    "LogicClosureExecutionPlan",
    "LogicClosureFormationPlan",
    "LogicClosureRecognitionPlan",
    "LogicClosureRoundReport",
    "LogicClosureRoundRequest",
    "LogicClosureRuntime",
    "LogicClosureRuntimeBuilder",
    "LogicOperatorRegistrySnapshot",
    "TrainingLogicClosureRuntimeBuilder",
    "install_logic_closure_runtime",
]
