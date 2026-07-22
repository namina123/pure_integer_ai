"""R-00 关系闭环的 forming、recognition、消费、审计和规模报告编排。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningReport,
    CandidateLearningRuntime,
    CandidateRecognitionRequest,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    ActiveRelationClosureFact,
    RelationClosureCandidateSpec,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseContext,
    RelationUseDefinition,
    RelationUseOwner,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import SemanticGraph
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey


class RelationClosureIncompleteError(RuntimeError):
    """关系候选尚未通过 writer、Evidence、resolver、投影和消费全链。"""


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验编排路由、评测 partition 和 use 使用的非空整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长完整键增加长度前缀，避免拼接歧义。"""
    return len(value), *value


def _validate_observation_anchor(
        anchor: ObjectIdentity, observation: SourceRef) -> None:
    """按 Occurrence/Span 权威 identity codec 核验 anchor 来源和完整布局。"""
    source_key = observation.stable_key()
    if anchor.components[:len(source_key)] != source_key:
        raise ValueError("observation_anchor 与 recognition 来源不一致")
    payload = anchor.components[len(source_key):]
    if anchor.object_kind == OBJECT_OCCURRENCE:
        if len(payload) != 3:
            raise ValueError("Occurrence anchor identity 布局非法")
        restored = occurrence_identity(
            observation,
            start=payload[0],
            end=payload[1],
            ordinal=payload[2],
        )
    elif anchor.object_kind == OBJECT_SPAN:
        if len(payload) < 4:
            raise ValueError("Span anchor identity 布局非法")
        ordinal, member_count, *member_values = payload
        if member_count <= 0 or len(member_values) != member_count * 2:
            raise ValueError("Span anchor member 布局非法")
        members = tuple(
            (member_values[index], member_values[index + 1])
            for index in range(0, len(member_values), 2)
        )
        restored = span_identity(
            observation,
            members=members,
            ordinal=ordinal,
        )
    else:
        raise ValueError("observation_anchor 必须是 Occurrence 或 Span")
    if restored != anchor:
        raise ValueError("observation_anchor 不能通过权威 identity codec 重建")


@dataclass(frozen=True)
class RelationClosureRecognitionInput:
    """一次来源化 relation prediction/reveal 的完整正式输入。"""

    proposition: ObjectIdentity
    observation: SourceRef
    scope: ScopeIdentity
    partition: ProtocolKey
    event_key: tuple[int, ...]
    observation_anchor: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]
    revealed: RevealedObjectObservation
    archive_refuted: bool = False
    replacement: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("proposition 必须是 ObjectIdentity")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("observation 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if self.scope.source != self.observation:
            raise ValueError("scope 必须绑定 recognition observation")
        if not isinstance(self.partition, ProtocolKey):
            raise TypeError("partition 必须是 V-00 ProtocolKey")
        _strict_key(self.event_key, where="event_key")
        if not isinstance(self.observation_anchor, ObjectIdentity):
            raise TypeError("observation_anchor 必须是 ObjectIdentity")
        if self.observation_anchor.object_kind not in {
                OBJECT_OCCURRENCE, OBJECT_SPAN}:
            raise ValueError("observation_anchor 必须是 Occurrence 或 Span")
        _validate_observation_anchor(
            self.observation_anchor, self.observation)
        if not isinstance(self.visible_inputs, tuple) or not self.visible_inputs:
            raise ValueError("visible_inputs 必须是非空 ObjectIdentity tuple")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.visible_inputs):
            raise TypeError("visible_inputs 只能包含 ObjectIdentity")
        if self.observation_anchor not in self.visible_inputs:
            raise ValueError("visible_inputs 必须保留 observation_anchor")
        if not isinstance(self.revealed, RevealedObjectObservation):
            raise TypeError("revealed 必须是 RevealedObjectObservation")
        if (self.revealed.observation != self.observation
                or self.revealed.scope != self.scope
                or self.revealed.event_key != self.event_key):
            raise ValueError("revealed 与 recognition 路由不一致")
        if type(self.archive_refuted) is not bool:
            raise TypeError("archive_refuted 必须是 bool")
        if self.replacement is not None and not isinstance(
                self.replacement, ObjectIdentity):
            raise TypeError("replacement 必须是 ObjectIdentity 或 None")

    def route_key(self) -> tuple:
        """返回同一候选、来源和事件的幂等 recognition 路由。"""
        return self.proposition, self.observation, self.event_key

    def stable_key(self) -> tuple[int, ...]:
        """展开 held-out partition、可见输入和 reveal 的完整审计键。"""
        values: list[int] = [
            *_pack(self.proposition.stable_key()),
            *_pack(self.observation.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.partition.stable_key()),
            *_pack(self.event_key),
            *_pack(self.observation_anchor.stable_key()),
            len(self.visible_inputs),
        ]
        for item in self.visible_inputs:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.revealed.verifier_source.stable_key()))
        values.append(len(self.revealed.supported_targets))
        for item in self.revealed.supported_targets:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.revealed.refuted_targets))
        for item in self.revealed.refuted_targets:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.revealed.trace))
        values.append(int(self.archive_refuted))
        replacement = (
            () if self.replacement is None else self.replacement.stable_key())
        values.extend(_pack(replacement))
        return tuple(values)


@dataclass(frozen=True)
class RelationClosureFormationTrace:
    """一次 typed relation forming 的 spec、候选定义和 H-05 Hypothesis。"""

    spec: RelationClosureCandidateSpec
    hypothesis: HypothesisKey


@dataclass(frozen=True)
class RelationClosureRecognitionTrace:
    """一次 recognition 的输入、Evidence/H-04 结果和可选 active 消费投影。"""

    input: RelationClosureRecognitionInput
    outcome: CandidateLearningOutcome
    active_fact: ActiveRelationClosureFact | None


@dataclass(frozen=True)
class RelationClosureUse:
    """一个消费者实际采用事实时留下的内存级完整归因。"""

    use_key: tuple[int, ...]
    proposition: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    read_only_recovered: bool
    context: RelationUseContext | None = None
    event: ObjectIdentity | None = None

    def route_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回完整 context 加局部 use_key；PH1 无 context 时保留兼容路由。"""
        context_key = () if self.context is None else self.context.stable_key()
        return context_key, self.use_key

    def to_definition(self) -> RelationUseDefinition:
        """把已采用事实转换为可写入 PH2 Core 图的完整 Use 定义。"""
        if self.context is None:
            raise RelationClosureIncompleteError(
                "配置 Core Use owner 后必须提供完整 RelationUseContext")
        return RelationUseDefinition(
            self.use_key,
            self.context,
            self.proposition,
            self.hypothesis,
            self.evidence_keys,
            self.decision_key,
            self.read_only_recovered,
        )


@dataclass(frozen=True)
class RelationClosureAudit:
    """单候选 writer 到 consumer 的分环节审计，不以计数替代闭环。"""

    proposition: ObjectIdentity
    writer_defined: bool
    recognition_evidence: bool
    resolver_adopted: bool
    active_projection: bool
    consumer_used: bool

    @property
    def complete(self) -> bool:
        """只有五个独立环节全部存在时才认定闭环完整。"""
        return all((
            self.writer_defined,
            self.recognition_evidence,
            self.resolver_adopted,
            self.active_projection,
            self.consumer_used,
        ))

    @property
    def missing(self) -> tuple[str, ...]:
        """稳定列出缺失环节，供故障消融和验收报告定位。"""
        checks = (
            ("writer", self.writer_defined),
            ("evidence", self.recognition_evidence),
            ("resolver", self.resolver_adopted),
            ("projection", self.active_projection),
            ("consumer", self.consumer_used),
        )
        return tuple(name for name, present in checks if not present)


@dataclass(frozen=True)
class RelationClosurePerformanceCounters:
    """由宿主采样的绝对整数性能计数，不内置阈值或预算。"""

    backend_queries: int
    backend_rows: int
    graph_objects: int
    graph_statements: int

    def __post_init__(self) -> None:
        assert_int(
            self.backend_queries,
            self.backend_rows,
            self.graph_objects,
            self.graph_statements,
            _where="RelationClosurePerformanceCounters",
        )
        if any(type(item) is not int or item < 0 for item in (
                self.backend_queries,
                self.backend_rows,
                self.graph_objects,
                self.graph_statements)):
            raise ValueError("性能绝对计数必须是非负严格整数")


@dataclass(frozen=True)
class RelationClosurePerformanceWindow:
    """一次关系闭环测量窗口的耗时和后端/图增长整数差值。"""

    elapsed_ns: int
    backend_queries: int
    backend_rows: int
    graph_objects: int
    graph_statements: int

    def __post_init__(self) -> None:
        assert_int(
            self.elapsed_ns,
            self.backend_queries,
            self.backend_rows,
            self.graph_objects,
            self.graph_statements,
            _where="RelationClosurePerformanceWindow",
        )
        if any(type(item) is not int or item < 0 for item in (
                self.elapsed_ns,
                self.backend_queries,
                self.backend_rows,
                self.graph_objects,
                self.graph_statements)):
            raise ValueError("性能窗口必须使用非负严格整数")

    @classmethod
    def between(
            cls, before: RelationClosurePerformanceCounters,
            after: RelationClosurePerformanceCounters, *,
            elapsed_ns: int) -> "RelationClosurePerformanceWindow":
        """由两次绝对采样构造增长窗口，任一计数回退均拒绝。"""
        if not isinstance(before, RelationClosurePerformanceCounters):
            raise TypeError("before 必须是 RelationClosurePerformanceCounters")
        if not isinstance(after, RelationClosurePerformanceCounters):
            raise TypeError("after 必须是 RelationClosurePerformanceCounters")
        values = (
            after.backend_queries - before.backend_queries,
            after.backend_rows - before.backend_rows,
            after.graph_objects - before.graph_objects,
            after.graph_statements - before.graph_statements,
        )
        if any(item < 0 for item in values):
            raise ValueError("性能绝对计数不得在同一测量窗口内回退")
        return cls(elapsed_ns, *values)


@dataclass(frozen=True)
class RelationClosureReport:
    """关系闭环的流程计数、Evidence 分型、消费数和性能窗口模板。"""

    candidate: CandidateLearningReport
    formation_count: int
    recognition_count: int
    support_count: int
    refute_count: int
    unknown_count: int
    consumer_use_count: int
    performance: RelationClosurePerformanceWindow


class RelationClosureRuntime:
    """复用 H-05 owner，统一 relation forming、recognition、消费和审计。"""

    def __init__(
            self, candidate_runtime: CandidateLearningRuntime,
            semantic_graph: SemanticGraph,
            consumer: ActiveRelationClosureConsumer,
            protocol: RelationClosureProtocol,
            use_owner: RelationUseOwner | None = None) -> None:
        """绑定同一候选、语义图和消费者，并可从 PH2 Core 图恢复 Use owner。"""
        if not isinstance(candidate_runtime, CandidateLearningRuntime):
            raise TypeError("candidate_runtime 必须是 CandidateLearningRuntime")
        if not isinstance(semantic_graph, SemanticGraph):
            raise TypeError("semantic_graph 必须是 SemanticGraph")
        if not isinstance(consumer, ActiveRelationClosureConsumer):
            raise TypeError("consumer 必须是 ActiveRelationClosureConsumer")
        if not isinstance(protocol, RelationClosureProtocol):
            raise TypeError("protocol 必须是 RelationClosureProtocol")
        if candidate_runtime.graph is not consumer.candidate_graph:
            raise ValueError("candidate runtime 与 consumer 必须共享候选图")
        if semantic_graph is not consumer.semantic_graph:
            raise ValueError("runtime 与 consumer 必须共享 SemanticGraph")
        if consumer.engine is not candidate_runtime.engine:
            raise ValueError("live consumer 必须绑定同一 H-05 owner")
        if consumer.protocol != protocol:
            raise ValueError("runtime 与 consumer 的关系字段协议不一致")
        if use_owner is not None:
            if not isinstance(use_owner, RelationUseOwner):
                raise TypeError("use_owner 必须是 RelationUseOwner 或 None")
            if use_owner.graph.ontology is not semantic_graph.ontology:
                raise ValueError("Core Use owner 必须绑定 relation SemanticGraph")
        self.candidate_runtime = candidate_runtime
        self.semantic_graph = semantic_graph
        self.consumer = consumer
        self.protocol = protocol
        self.use_owner = use_owner
        self._formations: dict[
            ObjectIdentity, RelationClosureFormationTrace] = {}
        self._recognitions: dict[
            tuple, RelationClosureRecognitionTrace] = {}
        self._uses: dict[
            tuple[tuple[int, ...], tuple[int, ...]], RelationClosureUse
        ] = {}
        if use_owner is not None:
            for materialized in use_owner.history():
                definition = materialized.definition
                use = RelationClosureUse(
                    definition.use_key,
                    definition.proposition,
                    definition.hypothesis,
                    definition.evidence_keys,
                    definition.decision_key,
                    definition.read_only_recovered,
                    definition.context,
                    materialized.event,
                )
                route = use.route_key()
                existing = self._uses.get(route)
                if existing is not None and existing != use:
                    raise RelationClosureIncompleteError(
                        "恢复的 Core Use 路由发生竞争")
                self._uses[route] = use

    def form(
            self, spec: RelationClosureCandidateSpec, *,
            timestamp_base: int = 0,
            ) -> RelationClosureFormationTrace:
        """核验 S-00 定义后按显式逻辑序登记 forming unknown，不重写真值。"""
        return self.form_many(((spec, timestamp_base),))[0]

    def form_many(
            self,
            requests: tuple[tuple[RelationClosureCandidateSpec, int], ...],
            ) -> tuple[RelationClosureFormationTrace, ...]:
        """整批核验 S-00 后登记 relation forming，避免逐候选复制完整 H-00 owner。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("relation form_many requests 必须是非空 tuple")
        normalized = []
        pending = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("relation form_many request 必须是 spec/timestamp 对")
            spec, timestamp_base = request
            if not isinstance(spec, RelationClosureCandidateSpec):
                raise TypeError("spec 必须是 RelationClosureCandidateSpec")
            assert_int(timestamp_base, _where="relation forming timestamp_base")
            if type(timestamp_base) is not int or timestamp_base < 0:
                raise ValueError(
                    "relation forming timestamp_base 必须为非负严格整数")
            proposition_ref = self.semantic_graph.ontology.resolve(
                spec.proposition.proposition)
            if proposition_ref is None:
                raise RelationClosureIncompleteError(
                    "relation forming 前必须已有 S-00 原子命题定义")
            restored = self.semantic_graph.read_atomic(proposition_ref)
            if restored.definition != spec.proposition:
                raise RelationClosureIncompleteError(
                    "relation forming 输入与 SemanticGraph 定义不一致")
            existing = self._formations.get(spec.proposition.proposition)
            if existing is not None:
                if existing.spec != spec:
                    raise RelationClosureIncompleteError(
                        "同一 Proposition 已绑定不同 relation closure spec")
                normalized.append(existing)
                continue
            definition = spec.candidate_definition(self.protocol)
            pending.append((len(normalized), spec, definition, timestamp_base))
            normalized.append(None)
        propositions = tuple(item[1].proposition.proposition for item in pending)
        if len(set(propositions)) != len(propositions):
            raise RelationClosureIncompleteError(
                "同批 relation forming 不得重复 Proposition")
        if pending:
            hypotheses = self.candidate_runtime.register_many(tuple(
                (definition, timestamp_base)
                for _index, _spec, definition, timestamp_base in pending
            ))
            for (index, spec, _definition, _timestamp), hypothesis in zip(
                    pending, hypotheses, strict=True):
                trace = RelationClosureFormationTrace(spec, hypothesis)
                self._formations[spec.proposition.proposition] = trace
                normalized[index] = trace
        return tuple(normalized)

    def recognize(
            self, input_value: RelationClosureRecognitionInput,
            ) -> RelationClosureRecognitionTrace:
        """自动分配下一段逻辑序，再执行独立 reveal、H-04 和 active 投影。"""
        timestamps = self.candidate_runtime.next_timestamps(3)
        return self.recognize_at(
            input_value,
            timestamp_seq=timestamps[0],
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
        )

    def recognize_at(
            self,
            input_value: RelationClosureRecognitionInput,
            *,
            timestamp_seq: int,
            resolve_timestamp_seq: int,
            projection_timestamp_seq: int,
            ) -> RelationClosureRecognitionTrace:
        """按课程注入的三段逻辑序执行可重放 relation prediction/reveal。"""
        return self.recognize_many_at(((
            input_value,
            timestamp_seq,
            resolve_timestamp_seq,
            projection_timestamp_seq,
        ),))[0]

    def recognize_many_at(
            self,
            requests: tuple[
                tuple[RelationClosureRecognitionInput, int, int, int], ...],
            ) -> tuple[RelationClosureRecognitionTrace, ...]:
        """整批预演并提交 relation recognition，保持 manifest 顺序和幂等路由。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("relation recognize_many_at requests 必须是非空 tuple")
        normalized = []
        pending = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 4:
                raise TypeError("relation recognition request 必须含输入和三段逻辑序")
            input_value, timestamp_seq, resolve_seq, projection_seq = request
            if not isinstance(input_value, RelationClosureRecognitionInput):
                raise TypeError("input_value 必须是 RelationClosureRecognitionInput")
            formation = self._formations.get(input_value.proposition)
            if formation is None:
                raise RelationClosureIncompleteError(
                    "recognition 前必须完成 relation forming writer")
            route = input_value.route_key()
            existing = self._recognitions.get(route)
            if existing is not None:
                if existing.input != input_value:
                    raise RelationClosureIncompleteError(
                        "同一 recognition 路由绑定了不同输入")
                normalized.append(existing)
                continue
            replacement = None
            if input_value.replacement is not None:
                replacement = self.candidate_runtime.hypothesis_for_candidate(
                    input_value.replacement)
            candidate_request = CandidateRecognitionRequest(
                formation.hypothesis,
                input_value.observation,
                input_value.scope,
                input_value.event_key,
                input_value.visible_inputs,
                input_value.proposition,
                input_value.revealed,
                timestamp_seq,
                resolve_seq,
                projection_seq,
                archive_refuted=input_value.archive_refuted,
                replacement=replacement,
            )
            pending.append((len(normalized), input_value, candidate_request))
            normalized.append(None)
        routes = tuple(item[1].route_key() for item in pending)
        if len(set(routes)) != len(routes):
            raise RelationClosureIncompleteError(
                "同批 relation recognition 路由不得重复")
        if pending:
            outcomes = self.candidate_runtime.recognize_many(tuple(
                item[2] for item in pending))
            for (index, input_value, _request), outcome in zip(
                    pending, outcomes, strict=True):
                facts = self.consumer.lookup_proposition(
                    input_value.proposition)
                active_fact = facts[0] if len(facts) == 1 else None
                trace = RelationClosureRecognitionTrace(
                    input_value,
                    outcome,
                    active_fact,
                )
                self._recognitions[input_value.route_key()] = trace
                normalized[index] = trace
        return tuple(normalized)

    def consume(
            self, proposition: ObjectIdentity, *,
            use_key: tuple[int, ...],
            context: RelationUseContext | None = None) -> RelationClosureUse:
        """通过正式 typed consumer 采用事实，并记录 Evidence/H-04 使用归因。"""
        return self.consume_many(
            ((proposition, use_key),), context=context)[0]

    def consume_many(
            self,
            requests: tuple[tuple[ObjectIdentity, tuple[int, ...]], ...],
            *,
            context: RelationUseContext | None = None,
            ) -> tuple[RelationClosureUse, ...]:
        """全量预检多事实采用，任一冲突时不留下部分 use ledger。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("consume_many requests 必须是非空 tuple")
        if context is not None and not isinstance(context, RelationUseContext):
            raise TypeError("consume_many context 类型错误")
        if self.use_owner is not None and context is None:
            raise RelationClosureIncompleteError(
                "配置 Core Use owner 后必须提供完整 RelationUseContext")
        prepared: list[RelationClosureUse] = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("consume_many request 必须是 proposition/use_key 对")
            proposition, use_key = request
            if not isinstance(proposition, ObjectIdentity):
                raise TypeError("consume_many proposition 类型错误")
            key = _strict_key(
                use_key, where="RelationClosureRuntime.use_key")
            fact = self.consumer.require_proposition(proposition)
            prepared.append(RelationClosureUse(
                key,
                proposition,
                fact.hypothesis,
                fact.evidence_keys,
                fact.decision_key,
                fact.read_only_recovered,
                context,
            ))
        routes = tuple(item.route_key() for item in prepared)
        if len(set(routes)) != len(routes):
            raise RelationClosureIncompleteError(
                "同批 relation Use 路由不得重复")
        for use in prepared:
            existing = self._uses.get(use.route_key())
            if existing is not None and not self._same_use(existing, use):
                raise RelationClosureIncompleteError(
                    "同一 relation Use 路由已绑定不同关系事实")
        committed = tuple(prepared)
        if self.use_owner is not None:
            materialized = self.use_owner.append_many(tuple(
                item.to_definition() for item in prepared))
            committed = tuple(
                RelationClosureUse(
                    item.definition.use_key,
                    item.definition.proposition,
                    item.definition.hypothesis,
                    item.definition.evidence_keys,
                    item.definition.decision_key,
                    item.definition.read_only_recovered,
                    item.definition.context,
                    item.event,
                )
                for item in materialized
            )
        for use in committed:
            self._uses[use.route_key()] = use
        return committed

    def audit(
            self, spec: RelationClosureCandidateSpec,
            ) -> RelationClosureAudit:
        """分别核验 writer、recognition Evidence、resolver、投影和实际消费。"""
        if not isinstance(spec, RelationClosureCandidateSpec):
            raise TypeError("spec 必须是 RelationClosureCandidateSpec")
        definition = spec.candidate_definition(self.protocol)
        hypothesis = definition.hypothesis(
            self.candidate_runtime.engine.protocol)
        writer_defined = False
        try:
            writer_defined = (
                self.candidate_runtime.graph.read_definition(
                    hypothesis).definition == definition)
        except (KeyError, RuntimeError, ValueError):
            writer_defined = False
        recognition_evidence = any(
            trace.input.proposition == spec.proposition.proposition
            for trace in self._recognitions.values()
        )
        resolver_adopted = False
        try:
            resolver_adopted = (
                self.candidate_runtime.engine.active(hypothesis) is not None)
        except KeyError:
            resolver_adopted = False
        active_projection = bool(
            self.consumer.lookup_proposition(spec.proposition.proposition))
        consumer_used = any(
            use.proposition == spec.proposition.proposition
            for use in self._uses.values()
        )
        return RelationClosureAudit(
            spec.proposition.proposition,
            writer_defined,
            recognition_evidence,
            resolver_adopted,
            active_projection,
            consumer_used,
        )

    def require_complete(
            self, spec: RelationClosureCandidateSpec) -> RelationClosureAudit:
        """要求候选五环完整；任一断开时报告具体缺口并失败。"""
        audit = self.audit(spec)
        if not audit.complete:
            raise RelationClosureIncompleteError(
                "关系闭环缺失环节: " + ",".join(audit.missing))
        return audit

    def report(
            self, performance: RelationClosurePerformanceWindow,
            ) -> RelationClosureReport:
        """生成不含掌握结论的全链计数和整数性能报告。"""
        if not isinstance(performance, RelationClosurePerformanceWindow):
            raise TypeError("performance 必须是 RelationClosurePerformanceWindow")
        stances = tuple(
            trace.outcome.verification.stance
            for trace in self._recognitions.values()
        )
        return RelationClosureReport(
            self.candidate_runtime.report(),
            len(self._formations),
            len(self._recognitions),
            stances.count(EVIDENCE_SUPPORT),
            stances.count(EVIDENCE_REFUTE),
            stances.count(EVIDENCE_UNKNOWN),
            len(self._uses),
            performance,
        )

    def clone_for_evaluation(
            self, semantic_graph: SemanticGraph,
            candidate_graph: CandidateProjectionGraph,
            ) -> "RelationClosureRuntime":
        """复制 H-05 owner 和编排账并绑定隔离图，供 held-out 写隔离。"""
        cloned_candidate = self.candidate_runtime.clone_for_graph(
            candidate_graph)
        cloned_consumer = self.consumer.clone_for_graphs(
            semantic_graph,
            candidate_graph,
            engine=cloned_candidate.engine,
        )
        cloned_use_owner = None
        if self.use_owner is not None:
            cloned_use_owner = self.use_owner.clone_for_ontology(
                semantic_graph.ontology)
        cloned = RelationClosureRuntime(
            cloned_candidate,
            semantic_graph,
            cloned_consumer,
            self.protocol,
            cloned_use_owner,
        )
        cloned._formations = dict(self._formations)
        cloned._recognitions = dict(self._recognitions)
        if self.use_owner is None:
            cloned._uses = dict(self._uses)
        return cloned

    def state_key(self) -> tuple:
        """返回 owner、forming、recognition 和 use 的完整隔离状态键。"""
        formations = tuple(sorted(
            (
                proposition.stable_key(),
                trace.spec.candidate_definition(self.protocol).stable_key(),
                trace.hypothesis.stable_key(),
            )
            for proposition, trace in self._formations.items()
        ))
        recognitions = tuple(sorted(
            trace.input.stable_key()
            for trace in self._recognitions.values()
        ))
        uses = tuple(sorted(
            (
                use.use_key,
                use.proposition.stable_key(),
                use.hypothesis.stable_key(),
                use.evidence_keys,
                use.decision_key,
                use.read_only_recovered,
                None if use.context is None else use.context.stable_key(),
                None if use.event is None else use.event.stable_key(),
            )
            for use in self._uses.values()
        ))
        return (
            self.candidate_runtime.state_key(),
            formations,
            recognitions,
            uses,
            None if self.use_owner is None else self.use_owner.state_key(),
        )

    @staticmethod
    def _same_use(left: RelationClosureUse, right: RelationClosureUse) -> bool:
        """比较 Use 领域内容并忽略仅由 Core 图恢复得到的 Event 引用。"""
        return (
            left.use_key == right.use_key
            and left.proposition == right.proposition
            and left.hypothesis == right.hypothesis
            and left.evidence_keys == right.evidence_keys
            and left.decision_key == right.decision_key
            and left.read_only_recovered == right.read_only_recovered
            and left.context == right.context
        )


__all__ = [
    "RelationClosureAudit",
    "RelationClosureFormationTrace",
    "RelationClosureIncompleteError",
    "RelationClosurePerformanceCounters",
    "RelationClosurePerformanceWindow",
    "RelationClosureRecognitionInput",
    "RelationClosureRecognitionTrace",
    "RelationClosureReport",
    "RelationClosureRuntime",
    "RelationClosureUse",
]
