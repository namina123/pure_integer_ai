"""用 H-00/H-04 和通用候选投影管理 connector 理论生命周期。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_UNKNOWN,
    LIFECYCLE_ACTIVE,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingCandidateHistoryLog,
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageGenerationConnectorGraph,
)


class LanguageConnectorCandidateError(RuntimeError):
    """connector 理论、H-00 候选定义和图 lifecycle 投影不一致。"""


CANDIDATE_PERSISTENCE_VOLATILE = 0
CANDIDATE_PERSISTENCE_TRAINING = 1
CANDIDATE_PERSISTENCE_MEMORY = 2
_CANDIDATE_PERSISTENCE_KINDS = frozenset({
    CANDIDATE_PERSISTENCE_VOLATILE,
    CANDIDATE_PERSISTENCE_TRAINING,
    CANDIDATE_PERSISTENCE_MEMORY,
})


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键添加长度边界。"""
    return len(value), *value


@dataclass(frozen=True)
class LanguageConnectorCandidateProtocol:
    """注入候选枚举 predicate、理论投影 predicate、kind 和竞争命名空间。"""

    candidate_kind_predicate: ObjectIdentity
    definition_member_predicate: ObjectIdentity
    candidate_kind: ObjectIdentity
    competition_namespace: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, value in (
                ("candidate kind predicate", self.candidate_kind_predicate),
                ("definition member predicate",
                 self.definition_member_predicate)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"connector {label} 必须是 ObjectIdentity")
            if value.object_kind != OBJECT_CONCEPT:
                raise ValueError(f"connector {label} 必须是一等 Concept")
        if self.candidate_kind_predicate == self.definition_member_predicate:
            raise ValueError("connector candidate predicate 必须互异")
        if not isinstance(self.candidate_kind, ObjectIdentity):
            raise TypeError("connector candidate kind 必须是一等对象")
        _strict_key(
            self.competition_namespace,
            label="connector competition namespace",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回候选 kind、两个 predicate 和竞争命名空间。"""
        return (
            *_packed(self.candidate_kind_predicate.stable_key()),
            *_packed(self.definition_member_predicate.stable_key()),
            *_packed(self.candidate_kind.stable_key()),
            *_packed(self.competition_namespace),
        )


class LanguageConnectorCandidateMapper:
    """把 connector 权威理论映射为 H-00 可验证的派生候选定义。"""

    def __init__(self, protocol: LanguageConnectorCandidateProtocol) -> None:
        if not isinstance(protocol, LanguageConnectorCandidateProtocol):
            raise TypeError("connector candidate mapper protocol 类型错误")
        self.protocol = protocol

    def definition(
            self,
            template: LanguageGenerationConnectorTemplate,
            forming_sources: tuple[SourceRef, ...],
            ) -> EvidenceCandidateDefinition:
        """构造完整理论成员投影；ordinal 只用于派生完整性索引。"""
        if not isinstance(template, LanguageGenerationConnectorTemplate):
            raise TypeError("connector candidate template 类型错误")
        if not isinstance(forming_sources, tuple) or any(
                not isinstance(item, SourceRef) for item in forming_sources):
            raise TypeError("connector forming sources 必须是 SourceRef tuple")
        members = self._theory_members(template)
        bindings = [CandidateBinding(
            self.protocol.candidate_kind_predicate,
            self.protocol.candidate_kind,
            0,
        )]
        bindings.extend(
            CandidateBinding(
                self.protocol.definition_member_predicate,
                member,
                ordinal,
            )
            for ordinal, member in enumerate(members)
        )
        return EvidenceCandidateDefinition(
            template.connector,
            self.competition_key(template),
            tuple(bindings),
            forming_sources,
        )

    def competition_key(
            self,
            template: LanguageGenerationConnectorTemplate,
            ) -> tuple[int, ...]:
        """按 branch、Proposition structure 和 predicate 建精确竞争边界。"""
        return (
            *self.protocol.competition_namespace,
            *_packed(template.language_branch.stable_key()),
            *_packed(template.proposition_structure.stable_key()),
            *_packed(template.predicate.stable_key()),
        )

    @staticmethod
    def _theory_members(
            template: LanguageGenerationConnectorTemplate,
            ) -> tuple[ObjectIdentity, ...]:
        """展平全部一等理论对象，供 H-00 身份校验而非执行语义读取。"""
        members: list[ObjectIdentity] = [
            template.language_branch,
            template.proposition_structure,
            template.predicate,
            template.sentence,
            template.structure,
        ]
        for slot in template.slots:
            members.extend((
                slot.structure,
                slot.slot,
                slot.role,
                slot.value_type,
            ))
        for binding in template.bindings:
            members.extend((
                binding.binding,
                binding.slot,
                binding.source,
            ))
            members.extend(item for item in (
                binding.role,
                binding.ordinal,
                binding.constant,
            ) if item is not None)
        members.append(template.constraint_set)
        members.extend(template.constraints)
        members.append(template.context_set)
        members.extend(template.context)
        members.extend((template.boundary, template.linearization_reason))
        for surface in template.surface:
            members.extend((
                surface.directive,
                surface.slot,
                surface.action,
                surface.instruction,
                surface.prefix_route,
                *surface.surface_prefix_steps,
            ))
        return tuple(members)


class LanguageConnectorCandidateRuntime:
    """连接权威理论图、H-00/H-04 owner 和 active registry 重建。"""

    def __init__(
            self,
            definition_graph: LanguageGenerationConnectorGraph,
            learning: CandidateLearningRuntime,
            protocol: LanguageConnectorCandidateProtocol,
            *, persistence_kind: int | None = None,
            ) -> None:
        if not isinstance(
                definition_graph, LanguageGenerationConnectorGraph):
            raise TypeError("connector candidate definition graph 类型错误")
        if not isinstance(learning, CandidateLearningRuntime):
            raise TypeError("connector candidate learning runtime 类型错误")
        if learning.graph.ontology is not definition_graph.ontology:
            raise ValueError("connector 理论和 lifecycle 必须共享 GraphOntology")
        if not isinstance(protocol, LanguageConnectorCandidateProtocol):
            raise TypeError("connector candidate protocol 类型错误")
        sink = learning.engine.ledger.event_sink
        if isinstance(sink, MemoryHypothesisEventSink):
            inferred_kind = CANDIDATE_PERSISTENCE_MEMORY
        elif isinstance(sink, TrainingHypothesisEventSink):
            inferred_kind = CANDIDATE_PERSISTENCE_TRAINING
        elif sink is None:
            inferred_kind = CANDIDATE_PERSISTENCE_VOLATILE
        else:
            raise TypeError("connector 候选 event sink 类型未注册")
        if persistence_kind is None:
            persistence_kind = inferred_kind
        if (type(persistence_kind) is not int
                or persistence_kind not in _CANDIDATE_PERSISTENCE_KINDS):
            raise ValueError("connector persistence kind 未注册")
        if sink is not None and persistence_kind != inferred_kind:
            raise ValueError("connector persistence kind 与 event sink 不一致")
        training_protocol = self._history_protocol(protocol, learning)
        if (isinstance(sink, TrainingHypothesisEventSink)
                and sink.protocol != training_protocol):
            raise ValueError("connector Core 训练历史协议与候选 owner 不一致")
        self.definition_graph = definition_graph
        self.learning = learning
        self.protocol = protocol
        self.mapper = LanguageConnectorCandidateMapper(protocol)
        self._persistence_kind = persistence_kind
        self._training_history_protocol = training_protocol

    def register(
            self,
            template: LanguageGenerationConnectorTemplate,
            forming_sources: tuple[SourceRef, ...],
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            timestamp_base: int = 0,
            ) -> HypothesisKey:
        """双侧零写预检后物化理论并登记 forming，故障不得形成 active 采用。"""
        definition = self.mapper.definition(template, forming_sources)
        self.definition_graph.preflight(
            template,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        expected_hypothesis = self.learning.preflight_register(
            definition,
            timestamp_base=timestamp_base,
        )
        self.definition_graph.materialize(
            template,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        hypothesis = self.learning.register(
            definition,
            timestamp_base=timestamp_base,
        )
        if hypothesis != expected_hypothesis:
            raise LanguageConnectorCandidateError(
                "connector 双侧预检与正式登记身份不一致")
        self._validate_definition_projection(definition)
        return hypothesis

    def recognize(
            self,
            hypothesis: HypothesisKey,
            *,
            observation: SourceRef,
            scope: ScopeIdentity,
            event_key: tuple[int, ...],
            visible_inputs: tuple[ObjectIdentity, ...],
            predicted: ObjectIdentity,
            revealed: RevealedObjectObservation,
            timestamp_seq: int,
            resolve_timestamp_seq: int,
            projection_timestamp_seq: int,
            scorers=(),
            archive_refuted: bool = False,
            replacement: HypothesisKey | None = None,
            ) -> CandidateLearningOutcome:
        """提交独立揭示并在写后核验 lifecycle 仍绑定同一权威理论。"""
        outcome = self.learning.recognize(
            hypothesis,
            observation=observation,
            scope=scope,
            event_key=event_key,
            visible_inputs=visible_inputs,
            predicted=predicted,
            revealed=revealed,
            timestamp_seq=timestamp_seq,
            resolve_timestamp_seq=resolve_timestamp_seq,
            projection_timestamp_seq=projection_timestamp_seq,
            scorers=scorers,
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        if outcome.projection is not None:
            self._validated_projection_template(outcome.projection)
        return outcome

    def active_templates(self) -> tuple[LanguageGenerationConnectorTemplate, ...]:
        """只从 active 图投影恢复理论，缺失、多义或删边均拒绝采用。"""
        projections = self.learning.graph.active_for_binding(CandidateBinding(
            self.protocol.candidate_kind_predicate,
            self.protocol.candidate_kind,
            0,
        ))
        templates = tuple(self._validated_template(item) for item in projections)
        if len({item.connector for item in templates}) != len(templates):
            raise LanguageConnectorCandidateError(
                "active connector 投影重复同一理论身份")
        return tuple(sorted(
            templates,
            key=lambda item: item.connector.stable_key(),
        ))

    def active_template_hypotheses(
            self,
            ) -> tuple[
                tuple[LanguageGenerationConnectorTemplate, HypothesisKey], ...
            ]:
        """返回启动时 active 理论及其 exact Hypothesis，不从模板键反推身份。"""
        return tuple(
            (
                template,
                self.learning.hypothesis_for_candidate(template.connector),
            )
            for template in self.active_templates()
        )

    def active_registry(self) -> LanguageGenerationConnectorRegistry:
        """用当前 active 理论重建 registry；无 active 候选时 fail closed。"""
        templates = self.active_templates()
        if not templates:
            raise LanguageConnectorCandidateError(
                "当前没有 active connector 理论")
        return LanguageGenerationConnectorRegistry(
            self.definition_graph.value_protocol,
            templates,
        )

    def trial_template(
            self,
            hypothesis: HypothesisKey,
            ) -> LanguageGenerationConnectorTemplate:
        """只为 exact forming Hypothesis 返回隔离 trial 理论。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("connector trial hypothesis 类型错误")
        definition = self.learning.engine.definition(hypothesis)
        registered = self.learning.hypothesis_for_candidate(
            definition.candidate)
        if registered != hypothesis:
            raise LanguageConnectorCandidateError(
                "connector trial Hypothesis 未绑定当前候选 owner")
        projection = self.learning.lifecycle_projection_if_available(
            definition.candidate)
        if projection is not None:
            raise LanguageConnectorCandidateError(
                "已有 lifecycle Event 的 connector 不得进入 trial registry")
        snapshot = self.learning.engine.ledger.snapshot(hypothesis)
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or snapshot.epistemic_status != EPISTEMIC_UNKNOWN
                or self.learning.engine.active(hypothesis) is not None):
            raise LanguageConnectorCandidateError(
                "connector trial 只接受未采用且未归档的 forming Hypothesis")
        restored = self.definition_graph.read(definition.candidate).definition
        expected = self.mapper.definition(restored, definition.forming_sources)
        if expected != definition:
            raise LanguageConnectorCandidateError(
                "connector trial 与权威理论或 H-00 定义不一致")
        return restored

    def trial_template_hypotheses(
            self,
            ) -> tuple[
                tuple[LanguageGenerationConnectorTemplate, HypothesisKey], ...
            ]:
        """一次扫描恢复全部合法 forming trial，退出或已有投影的候选不进入索引。"""
        result = []
        for definition in self.learning.engine.definitions():
            hypothesis = self.learning.hypothesis_for_candidate(
                definition.candidate)
            projection = self.learning.lifecycle_projection_if_available(
                definition.candidate)
            if projection is not None:
                continue
            snapshot = self.learning.engine.ledger.snapshot(hypothesis)
            if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                    or snapshot.epistemic_status != EPISTEMIC_UNKNOWN
                    or self.learning.engine.active(hypothesis) is not None):
                raise LanguageConnectorCandidateError(
                    "无 lifecycle 投影的 connector 不是合法 forming 状态")
            result.append((self.trial_template(hypothesis), hypothesis))
        return tuple(sorted(
            result,
            key=lambda item: item[1].stable_key(),
        ))

    def clone_for_graphs(
            self,
            definition_graph: LanguageGenerationConnectorGraph,
            candidate_graph: CandidateProjectionGraph,
            ) -> "LanguageConnectorCandidateRuntime":
        """复制 H-00/H-04 owner 并绑定 V-06 隔离后的两个图 facade。"""
        if not isinstance(candidate_graph, CandidateProjectionGraph):
            raise TypeError("connector clone candidate graph 类型错误")
        return LanguageConnectorCandidateRuntime(
            definition_graph,
            self.learning.clone_for_graph(candidate_graph),
            self.protocol,
            persistence_kind=self._persistence_kind,
        )

    @property
    def persistence_kind(self) -> int:
        """返回进程内、Core 训练历史或断奶后 Memory 三种持久化来源。"""
        return self._persistence_kind

    @property
    def memory_enabled(self) -> bool:
        """兼容返回该 owner 是否明确绑定断奶后 M-03 Memory。"""
        return self._persistence_kind == CANDIDATE_PERSISTENCE_MEMORY

    @property
    def training_history_protocol(self) -> TrainingHypothesisHistoryProtocol:
        """返回与候选 aggregate 精确绑定的 Core 训练历史协议。"""
        return self._training_history_protocol

    @property
    def training_history(self) -> TrainingCandidateHistoryLog | None:
        """返回当前绑定的 Core 训练历史；待重绑定 clone 返回空。"""
        sink = self.learning.engine.ledger.event_sink
        return sink.history if isinstance(
            sink, TrainingHypothesisEventSink) else None

    @property
    def memory_event_log(self) -> MemoryEventLog | None:
        """返回当前绑定的 M-03 event log；待重绑定 clone 返回 None。"""
        sink = self.learning.engine.ledger.event_sink
        return sink.event_log if isinstance(
            sink, MemoryHypothesisEventSink) else None

    def restore_for_graphs(
            self,
            definition_graph: LanguageGenerationConnectorGraph,
            candidate_graph: CandidateProjectionGraph,
            event_log: MemoryEventLog,
            ) -> "LanguageConnectorCandidateRuntime":
        """按 connector 协议过滤 M-03 历史，并绑定当前图和独立 event log。"""
        if not isinstance(definition_graph, LanguageGenerationConnectorGraph):
            raise TypeError("restore definition_graph 类型错误")
        if not isinstance(candidate_graph, CandidateProjectionGraph):
            raise TypeError("restore candidate_graph 类型错误")
        if definition_graph.ontology is not candidate_graph.ontology:
            raise ValueError("connector 恢复的定义图和候选图必须共享 ontology")
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("restore event_log 类型错误")
        sink = MemoryHypothesisEventSink(event_log)
        aggregate_source = self.learning.engine.protocol.aggregate_source
        owner = aggregate_source.owner
        access = MemoryAccessContext(
            owner.tenant_id, owner.user_id, owner.session_id)
        hypotheses = tuple(
            item for item in sink.hypotheses(access=access)
            if (item.hypothesis_kind
                == self.learning.engine.protocol.hypothesis_kind_key
                and item.scope
                == self.learning.engine.protocol.aggregate_scope
                and item.observation == aggregate_source)
        )
        definitions: list[EvidenceCandidateDefinition] = []
        for hypothesis in hypotheses:
            try:
                definition = EvidenceCandidateDefinition.from_stable_key(
                    hypothesis.candidate_key)
            except (TypeError, ValueError) as exc:
                raise LanguageConnectorCandidateError(
                    "M-03 connector Hypothesis 无法恢复候选定义") from exc
            if (definition.hypothesis(self.learning.engine.protocol)
                    != hypothesis):
                raise LanguageConnectorCandidateError(
                    "M-03 connector Hypothesis 与候选协议不一致")
            materialized = candidate_graph.read_definition(hypothesis)
            if materialized.definition != definition:
                raise LanguageConnectorCandidateError(
                    "M-03 connector 定义与候选图不一致")
            definitions.append(definition)
        ledger = sink.load_ledger(
            access=access,
            hypotheses=hypotheses,
            attach_sink=True,
        )
        decisions = sink.load_decisions(
            access=access,
            hypotheses=hypotheses,
        )
        engine = EvidenceCandidateEngine.from_history(
            self.learning.engine.protocol,
            definitions=tuple(definitions),
            ledger=ledger,
            decisions=decisions,
        )
        learning = CandidateLearningRuntime.from_history(
            engine,
            candidate_graph,
            self.learning.verifier,
            self.learning.metadata,
        )
        restored = LanguageConnectorCandidateRuntime(
            definition_graph,
            learning,
            self.protocol,
            persistence_kind=CANDIDATE_PERSISTENCE_MEMORY,
        )
        for definition in definitions:
            restored._validate_definition_projection(definition)
        return restored

    def restore_for_training_graphs(
            self,
            definition_graph: LanguageGenerationConnectorGraph,
            candidate_graph: CandidateProjectionGraph,
            history: TrainingCandidateHistoryLog,
            ) -> "LanguageConnectorCandidateRuntime":
        """从 Core 训练历史和候选图恢复 H-00/H-04，并绑定后续追加。"""
        if not isinstance(definition_graph, LanguageGenerationConnectorGraph):
            raise TypeError("training restore definition_graph 类型错误")
        if not isinstance(candidate_graph, CandidateProjectionGraph):
            raise TypeError("training restore candidate_graph 类型错误")
        if definition_graph.ontology is not candidate_graph.ontology:
            raise ValueError("Core 训练恢复的定义图和候选图必须共享 ontology")
        if not isinstance(history, TrainingCandidateHistoryLog):
            raise TypeError("training restore history 类型错误")
        sink = TrainingHypothesisEventSink(
            history,
            self._training_history_protocol,
        )
        hypotheses = sink.hypotheses()
        definitions: list[EvidenceCandidateDefinition] = []
        for hypothesis in hypotheses:
            try:
                definition = EvidenceCandidateDefinition.from_stable_key(
                    hypothesis.candidate_key)
            except (TypeError, ValueError) as exc:
                raise LanguageConnectorCandidateError(
                    "Core 训练历史无法恢复 connector 候选定义") from exc
            if (definition.hypothesis(self.learning.engine.protocol)
                    != hypothesis):
                raise LanguageConnectorCandidateError(
                    "Core 训练 Hypothesis 与候选协议不一致")
            materialized = candidate_graph.read_definition(hypothesis)
            if materialized.definition != definition:
                raise LanguageConnectorCandidateError(
                    "Core 训练候选定义与候选图不一致")
            definitions.append(definition)
        ledger = sink.load_ledger(attach_sink=True)
        engine = EvidenceCandidateEngine.from_history(
            self.learning.engine.protocol,
            definitions=tuple(definitions),
            ledger=ledger,
            decisions=sink.load_decisions(),
        )
        learning = CandidateLearningRuntime.from_history(
            engine,
            candidate_graph,
            self.learning.verifier,
            self.learning.metadata,
        )
        restored = LanguageConnectorCandidateRuntime(
            definition_graph,
            learning,
            self.protocol,
            persistence_kind=CANDIDATE_PERSISTENCE_TRAINING,
        )
        for definition in definitions:
            restored._validate_definition_projection(definition)
        return restored

    def state_key(self) -> tuple:
        """返回协议、H-00/H-04 owner 和当前 active 理论的完整状态。"""
        return (
            self.protocol.stable_key(),
            self._persistence_kind,
            self._training_history_protocol.stable_key(),
            self.definition_graph.value_protocol.stable_key(),
            self.learning.state_key(),
            tuple(item.stable_key() for item in self.active_templates()),
        )

    @staticmethod
    def _history_protocol(
            protocol: LanguageConnectorCandidateProtocol,
            learning: CandidateLearningRuntime,
            ) -> TrainingHypothesisHistoryProtocol:
        """由 connector 和 aggregate 协议派生不含物理 backend 的历史边界。"""
        aggregate = learning.engine.protocol
        return TrainingHypothesisHistoryProtocol(
            protocol.stable_key(),
            aggregate.hypothesis_kind_key,
            aggregate.aggregate_source,
            aggregate.aggregate_scope,
        )

    def _validate_definition_projection(
            self,
            definition: EvidenceCandidateDefinition,
            ) -> None:
        """核验 forming 图定义和 connector 权威理论逐字段一致。"""
        materialized = self.learning.graph.read_definition(
            definition.hypothesis(self.learning.engine.protocol))
        if materialized.definition != definition:
            raise LanguageConnectorCandidateError(
                "connector forming 投影替换了候选定义")
        restored = self.definition_graph.read(definition.candidate).definition
        expected = self.mapper.definition(
            restored,
            definition.forming_sources,
        )
        if expected != definition:
            raise LanguageConnectorCandidateError(
                "connector forming 投影与权威理论不一致")

    def _validated_template(
            self,
            projection: CandidateGraphProjection,
            ) -> LanguageGenerationConnectorTemplate:
        """从 active Event 恢复模板并与 H-00 完整候选定义双向比对。"""
        if projection.state != self.learning.graph.protocol.active_state:
            raise LanguageConnectorCandidateError(
                "connector consumer 只能采用 active 投影")
        return self._validated_projection_template(projection)

    def _validated_projection_template(
            self,
            projection: CandidateGraphProjection,
            ) -> LanguageGenerationConnectorTemplate:
        """核验任意 lifecycle 状态仍绑定同一 H-00 定义和权威理论。"""
        candidate = projection.candidate.definition
        hypothesis = candidate.hypothesis(self.learning.engine.protocol)
        try:
            registered = self.learning.hypothesis_for_candidate(
                candidate.candidate)
        except (KeyError, RuntimeError) as exc:
            raise LanguageConnectorCandidateError(
                "connector 图投影没有可续写的 H-00 owner 历史") from exc
        if (registered != hypothesis
                or projection.candidate.hypothesis != hypothesis):
            raise LanguageConnectorCandidateError(
                "connector 图投影与 H-00 Hypothesis 身份不一致")
        active = self.learning.engine.active(hypothesis)
        projected_active = (
            projection.state == self.learning.graph.protocol.active_state)
        if projected_active != (active is not None):
            raise LanguageConnectorCandidateError(
                "connector 图当前状态与 H-00/H-04 采用态不一致")
        if (active is not None
                and projection.history[-1].definition.decision_key
                != active.decision.stable_key()):
            raise LanguageConnectorCandidateError(
                "connector active 图未引用 H-04 当前采用决策")
        restored = self.definition_graph.read(candidate.candidate).definition
        expected = self.mapper.definition(
            restored,
            candidate.forming_sources,
        )
        if expected != candidate:
            raise LanguageConnectorCandidateError(
                "active connector lifecycle 与权威理论不一致")
        return restored


__all__ = [
    "CANDIDATE_PERSISTENCE_MEMORY",
    "CANDIDATE_PERSISTENCE_TRAINING",
    "CANDIDATE_PERSISTENCE_VOLATILE",
    "LanguageConnectorCandidateError",
    "LanguageConnectorCandidateMapper",
    "LanguageConnectorCandidateProtocol",
    "LanguageConnectorCandidateRuntime",
]
