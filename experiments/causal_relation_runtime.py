"""R-07 typed CAUSES 的独立核验、R-00 提交和执行消费编排。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.causal_execution import (
    CAUSAL_EXECUTION_CONFLICTED,
    CAUSAL_EXECUTION_PREDICTED,
    CAUSAL_TEMPORAL_ACCEPTED,
    CAUSAL_TEMPORAL_CONFLICTED,
    CAUSAL_TEMPORAL_REJECTED,
    CausalEndpointEvaluation,
    CausalEndpointProtocol,
    CausalExecutionResult,
    CausalExecutor,
    CausalTemporalAssessment,
    CausalTemporalResolver,
    causal_endpoints,
    validate_temporal_assessment,
)
from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    EventTimeFactIndex,
    EventTimeVerificationResult,
    EventTimeVerifier,
    ResolvedEventTimeRelation,
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
)
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import SemanticGraph
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
    RelationClosureUse,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationEffect,
    VerificationEvaluation,
    VerifierRegistration,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键添加长度前缀，避免拼接歧义。"""
    return len(value), *value


@dataclass(frozen=True)
class CausalVerificationProtocol:
    """声明 causal R-09 维度、verifier 和 Evidence effect 类型。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    evidence_target_kind: ProtocolKey

    def __post_init__(self) -> None:
        """核验 causal dimension、verifier 和 effect 类型均为开放协议键。"""
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("causal dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("causal verifier 必须是 ProtocolKey")
        if not isinstance(self.evidence_target_kind, ProtocolKey):
            raise TypeError("causal evidence target 必须是 ProtocolKey")


@dataclass(frozen=True)
class CausalIndependentWitness:
    """与 forming、候选图和当前 observation 分离的 causal 三态 witness。"""

    stance: int
    verifier_source: SourceRef
    input_objects: tuple[ObjectIdentity, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 witness 三态、独立来源、typed 输入和核验 trace。"""
        assert_int(self.stance, _where="CausalIndependentWitness.stance")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("causal witness stance 未注册")
        if not isinstance(self.verifier_source, SourceRef):
            raise TypeError("causal verifier_source 必须是 SourceRef")
        if not isinstance(self.input_objects, tuple) or not self.input_objects:
            raise ValueError("causal witness 必须保留非空 typed 输入")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.input_objects):
            raise TypeError("causal witness 输入必须是 ObjectIdentity")
        if len(set(self.input_objects)) != len(self.input_objects):
            raise ValueError("causal witness 输入不得重复")
        _strict_key(self.trace, where="CausalIndependentWitness.trace")


@dataclass(frozen=True)
class CausalVerificationRequest:
    """一次 causal prediction、独立 witness 和 event-time 约束请求。"""

    spec: RelationClosureCandidateSpec
    observation: SourceRef
    scope: ScopeIdentity
    partition: ProtocolKey
    event_key: tuple[int, ...]
    observation_anchor: ObjectIdentity
    visible_inputs: tuple[ObjectIdentity, ...]
    witness: CausalIndependentWitness
    temporal: EventTimeVerificationRequest
    archive_refuted: bool = False
    replacement: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验 recognition、witness、时间请求和来源 scope 的完整契约。"""
        if not isinstance(self.spec, RelationClosureCandidateSpec):
            raise TypeError("causal spec 必须是 RelationClosureCandidateSpec")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("causal observation 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("causal scope 必须是 ScopeIdentity")
        if self.scope.source != self.observation:
            raise ValueError("causal scope 必须绑定当前 observation")
        if not isinstance(self.partition, ProtocolKey):
            raise TypeError("causal partition 必须是 ProtocolKey")
        _strict_key(self.event_key, where="CausalVerificationRequest.event_key")
        if not isinstance(self.observation_anchor, ObjectIdentity):
            raise TypeError("causal observation_anchor 必须是 ObjectIdentity")
        if self.observation_anchor.object_kind not in {
                OBJECT_OCCURRENCE, OBJECT_SPAN}:
            raise ValueError("causal anchor 必须是 Occurrence 或 Span")
        if not isinstance(self.visible_inputs, tuple) or not self.visible_inputs:
            raise ValueError("causal visible_inputs 必须是非空 tuple")
        if any(not isinstance(item, ObjectIdentity)
               for item in self.visible_inputs):
            raise TypeError("causal visible_inputs 类型非法")
        if self.observation_anchor not in self.visible_inputs:
            raise ValueError("causal visible_inputs 必须保留 observation anchor")
        if not isinstance(self.witness, CausalIndependentWitness):
            raise TypeError("causal witness 类型非法")
        if not isinstance(self.temporal, EventTimeVerificationRequest):
            raise TypeError("causal temporal request 类型非法")
        if type(self.archive_refuted) is not bool:
            raise TypeError("archive_refuted 必须是 bool")
        if self.replacement is not None and not isinstance(
                self.replacement, ObjectIdentity):
            raise TypeError("replacement 必须是 ObjectIdentity 或 None")


def _causal_reveal_trace(
        request: CausalVerificationRequest,
        result: EventTimeVerificationResult,
        assessment: CausalTemporalAssessment,
        ) -> tuple[int, ...]:
    """组合 witness、时间 assertion 和 temporal reason 的完整整数 trace。"""
    values: list[int] = [
        len(request.witness.trace),
        *request.witness.trace,
        result.status,
        len(result.fact_set.facts),
    ]
    values.extend(fact.assertion_hash for fact in result.fact_set.facts)
    reason = assessment.reason.stable_key()
    values.extend((
        assessment.status,
        len(reason),
        *reason,
        len(assessment.trace),
        *assessment.trace,
        len(assessment.supporting_assertion_hashes),
        *assessment.supporting_assertion_hashes,
    ))
    return tuple(values)


@dataclass(frozen=True)
class CausalVerificationArtifact:
    """保存 causal 纯评估、待提交 recognition 和时间归因。"""

    request: CausalVerificationRequest
    temporal_result: EventTimeVerificationResult
    temporal_assessment: CausalTemporalAssessment
    evidence_stance: int
    verdict: int
    recognition: RelationClosureRecognitionInput
    effect: VerificationEffect

    def __post_init__(self) -> None:
        """主动核验纯评估、recognition、verdict 和 effect 没有漂移。"""
        if not isinstance(self.request, CausalVerificationRequest):
            raise TypeError("causal artifact request 类型非法")
        if not isinstance(self.temporal_result, EventTimeVerificationResult):
            raise TypeError("causal artifact temporal_result 类型非法")
        if not isinstance(self.temporal_assessment, CausalTemporalAssessment):
            raise TypeError("causal artifact temporal_assessment 类型非法")
        validate_temporal_assessment(
            self.temporal_result,
            self.temporal_assessment,
        )
        assert_int(
            self.evidence_stance,
            self.verdict,
            _where="CausalVerificationArtifact",
        )
        expected_stance, expected_verdict = _combine_causal_evidence(
            self.request.witness.stance,
            self.temporal_assessment.status,
        )
        if (self.evidence_stance != expected_stance
                or self.verdict != expected_verdict):
            raise ValueError("causal artifact stance/verdict 与纯评估输入不一致")
        if not isinstance(self.recognition, RelationClosureRecognitionInput):
            raise TypeError("causal artifact recognition 类型非法")
        request = self.request
        if (
            self.recognition.proposition
            != request.spec.proposition.proposition
            or self.recognition.observation != request.observation
            or self.recognition.scope != request.scope
            or self.recognition.partition != request.partition
            or self.recognition.event_key != request.event_key
            or self.recognition.observation_anchor
            != request.observation_anchor
            or self.recognition.visible_inputs != request.visible_inputs
            or self.recognition.archive_refuted != request.archive_refuted
            or self.recognition.replacement != request.replacement
        ):
            raise ValueError("causal artifact recognition 与 request 不一致")
        target = request.spec.proposition.proposition
        expected_supported = (
            (target,) if self.evidence_stance == EVIDENCE_SUPPORT else ())
        expected_refuted = (
            (target,) if self.evidence_stance == EVIDENCE_REFUTE else ())
        if (
            self.recognition.revealed.verifier_source
            != request.witness.verifier_source
            or self.recognition.revealed.supported_targets
            != expected_supported
            or self.recognition.revealed.refuted_targets != expected_refuted
            or self.recognition.revealed.trace != _causal_reveal_trace(
                request,
                self.temporal_result,
                self.temporal_assessment,
            )
        ):
            raise ValueError("causal artifact reveal 与 Evidence 归因不一致")
        if self.temporal_result.fact_set.scope != request.temporal.scope:
            raise ValueError("causal artifact event-time scope 与 request 不一致")
        requested_relations = tuple(sorted(
            request.temporal.relations,
            key=ObjectIdentity.stable_key,
        ))
        if self.temporal_result.fact_set.relations != requested_relations:
            raise ValueError("causal artifact event-time relation 集与 request 不一致")
        if not isinstance(self.effect, VerificationEffect):
            raise TypeError("causal artifact effect 类型非法")
        if self.effect.target_key != self.recognition.stable_key():
            raise ValueError("causal artifact effect 未绑定 recognition")


@dataclass(frozen=True)
class CausalGenerationUse:
    """生成采用 causal prediction 时保留的 relation 和执行归因。"""

    use_key: tuple[int, ...]
    proposition: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    execution_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验生成采用保存完整 relation、Evidence、decision 和执行键。"""
        _strict_key(self.use_key, where="CausalGenerationUse.use_key")
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("generation proposition 必须是 ObjectIdentity")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("generation hypothesis 必须是 HypothesisKey")
        if not isinstance(self.evidence_keys, tuple):
            raise TypeError("generation evidence_keys 必须是 tuple")
        for index, key in enumerate(self.evidence_keys):
            _strict_key(key, where=f"generation evidence_keys[{index}]")
        if not isinstance(self.decision_key, tuple):
            raise TypeError("generation decision_key 必须是 tuple")
        _strict_key(self.decision_key, where="generation decision_key")
        _strict_key(self.execution_key, where="generation execution_key")


@dataclass(frozen=True)
class CausalExecutionTrace:
    """一次 typed relation use、执行结果和可选生成采用的完整报告。"""

    relation_use: RelationClosureUse
    execution: CausalExecutionResult
    generation_use: CausalGenerationUse | None

    def __post_init__(self) -> None:
        """核验 relation use、执行事实和生成采用引用同一 active 关系。"""
        if not isinstance(self.relation_use, RelationClosureUse):
            raise TypeError("causal trace relation_use 类型非法")
        if not isinstance(self.execution, CausalExecutionResult):
            raise TypeError("causal trace execution 类型非法")
        fact = self.execution.fact
        if (
            self.relation_use.proposition != fact.proposition.proposition
            or self.relation_use.hypothesis != fact.hypothesis
            or self.relation_use.evidence_keys != fact.evidence_keys
            or self.relation_use.decision_key != fact.decision_key
        ):
            raise ValueError("causal trace relation use 与 execution 事实不一致")
        if self.generation_use is not None:
            if not isinstance(self.generation_use, CausalGenerationUse):
                raise TypeError("causal trace generation_use 类型非法")
            if (
                self.generation_use.proposition != self.relation_use.proposition
                or self.generation_use.hypothesis != self.relation_use.hypothesis
                or self.generation_use.evidence_keys
                != self.relation_use.evidence_keys
                or self.generation_use.decision_key
                != self.relation_use.decision_key
                or self.generation_use.execution_key
                != self.execution.stable_key()
            ):
                raise ValueError("causal generation use 与 execution 归因不一致")


def _combine_causal_evidence(
        witness_stance: int,
        temporal_status: int,
        ) -> tuple[int, int]:
    """组合独立 witness 与必要时间约束，不允许时间单独产生 support。"""
    if temporal_status == CAUSAL_TEMPORAL_CONFLICTED:
        return EVIDENCE_UNKNOWN, VERDICT_CONFLICTED
    if witness_stance == EVIDENCE_REFUTE:
        return EVIDENCE_REFUTE, VERDICT_REFUTE
    if witness_stance != EVIDENCE_SUPPORT:
        return EVIDENCE_UNKNOWN, VERDICT_UNKNOWN
    if temporal_status == CAUSAL_TEMPORAL_ACCEPTED:
        return EVIDENCE_SUPPORT, VERDICT_SUPPORT
    if temporal_status == CAUSAL_TEMPORAL_REJECTED:
        return EVIDENCE_REFUTE, VERDICT_REFUTE
    return EVIDENCE_UNKNOWN, VERDICT_UNKNOWN


class CausalVerificationAdapter:
    """把独立 witness 与 event-time 约束组合为 R-09 causal 维度。"""

    def __init__(
            self,
            relation_runtime: RelationClosureRuntime,
            event_time_verifier: EventTimeVerifier,
            temporal_resolver: CausalTemporalResolver,
            endpoints: CausalEndpointProtocol,
            protocol: CausalVerificationProtocol,
            ) -> None:
        """绑定唯一 R-00 owner、event-time verifier 和 causal R-09 协议。"""
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("relation_runtime 必须是 RelationClosureRuntime")
        if not isinstance(event_time_verifier, EventTimeVerifier):
            raise TypeError("event_time_verifier 必须是 EventTimeVerifier")
        if not callable(getattr(temporal_resolver, "resolve", None)):
            raise TypeError("temporal_resolver 必须实现 resolve")
        if not isinstance(endpoints, CausalEndpointProtocol):
            raise TypeError("endpoints 必须是 CausalEndpointProtocol")
        if not isinstance(protocol, CausalVerificationProtocol):
            raise TypeError("protocol 必须是 CausalVerificationProtocol")
        self.relation_runtime = relation_runtime
        self.event_time_verifier = event_time_verifier
        self.temporal_resolver = temporal_resolver
        self.endpoints = endpoints
        self.protocol = protocol

    def registration(self) -> VerifierRegistration:
        """返回 causal 独立维度注册，并只授权本维 Evidence effect。"""
        return VerifierRegistration(
            dimension=self.protocol.dimension,
            verifier=self.protocol.verifier,
            applies=self._applies,
            evaluate=self._evaluate,
            allowed_target_kinds=(self.protocol.evidence_target_kind,),
            commit=self._commit,
        )

    @staticmethod
    def _applies(request: object) -> bool:
        """只有完整 causal verification request 才适用。"""
        return isinstance(request, CausalVerificationRequest)

    def _evaluate(self, request: object) -> VerificationEvaluation:
        """纯评估 witness 和时间约束，不读取 active CAUSES 或提交 Evidence。"""
        if not isinstance(request, CausalVerificationRequest):
            raise TypeError("causal verifier 收到错误 request")
        self._validate_independence(request)
        if self.endpoints.relation in request.temporal.relations:
            raise ValueError("CAUSES relation 不得冒充 event-time relation")
        cause, effect = causal_endpoints(
            request.spec.proposition,
            self.endpoints,
        )
        temporal_result = self.event_time_verifier.verify(
            request.temporal.relations,
            scope=request.temporal.scope,
        )
        temporal_assessment = self.temporal_resolver.resolve(
            cause,
            effect,
            temporal_result,
        )
        temporal_assessment = validate_temporal_assessment(
            temporal_result,
            temporal_assessment,
        )
        self.validate_temporal_support(
            cause,
            effect,
            temporal_result,
            temporal_assessment,
        )
        evidence_stance, verdict = self._combine(
            request.witness.stance,
            temporal_assessment.status,
        )
        proposition = request.spec.proposition.proposition
        supported = (proposition,) if evidence_stance == EVIDENCE_SUPPORT else ()
        refuted = (proposition,) if evidence_stance == EVIDENCE_REFUTE else ()
        reveal = RevealedObjectObservation(
            request.observation,
            request.scope,
            request.event_key,
            request.witness.verifier_source,
            supported_targets=supported,
            refuted_targets=refuted,
            trace=self._reveal_trace(
                request,
                temporal_result,
                temporal_assessment,
            ),
        )
        recognition = RelationClosureRecognitionInput(
            proposition,
            request.observation,
            request.scope,
            request.partition,
            request.event_key,
            request.observation_anchor,
            request.visible_inputs,
            reveal,
            archive_refuted=request.archive_refuted,
            replacement=request.replacement,
        )
        effect_value = VerificationEffect(
            self.protocol.dimension,
            self.protocol.evidence_target_kind,
            recognition.stable_key(),
        )
        artifact = CausalVerificationArtifact(
            request,
            temporal_result,
            temporal_assessment,
            evidence_stance,
            verdict,
            recognition,
            effect_value,
        )
        return VerificationEvaluation(
            verdict,
            claim_keys=(proposition.stable_key(),),
            proposed_effects=(effect_value,),
            detail=(
                request.witness.stance,
                temporal_result.status,
                temporal_assessment.status,
                evidence_stance,
            ),
            source=request.witness.verifier_source,
            scope=request.scope,
            artifact=artifact,
        )

    def _commit(
            self, evaluation: VerificationEvaluation,
            ) -> tuple[VerificationEffect, ...]:
        """提交已纯评估的 R-00 recognition，并核验写入 stance 未漂移。"""
        artifact = evaluation.artifact
        if not isinstance(artifact, CausalVerificationArtifact):
            raise TypeError("causal commit 缺少 CausalVerificationArtifact")
        target = artifact.request.spec.proposition.proposition
        if (
            evaluation.verdict != artifact.verdict
            or evaluation.claim_keys != (target.stable_key(),)
            or evaluation.proposed_effects != (artifact.effect,)
            or evaluation.source != artifact.request.witness.verifier_source
            or evaluation.scope != artifact.request.scope
        ):
            raise ValueError("causal commit evaluation 与 artifact 不一致")
        if (
            artifact.effect.dimension != self.protocol.dimension
            or artifact.effect.target_kind
            != self.protocol.evidence_target_kind
        ):
            raise ValueError("causal commit effect 维度或目标类型不一致")
        trace = self.relation_runtime.recognize(artifact.recognition)
        if trace.outcome.verification.stance != artifact.evidence_stance:
            raise RuntimeError("causal Evidence commit stance 与纯评估不一致")
        return (artifact.effect,)

    @staticmethod
    def _combine(witness_stance: int, temporal_status: int) -> tuple[int, int]:
        """组合独立 witness 与必要时间约束，不允许时间单独产生 support。"""
        return _combine_causal_evidence(witness_stance, temporal_status)

    def validate_temporal_support(
            self,
            cause: ObjectIdentity,
            effect: ObjectIdentity,
            result: EventTimeVerificationResult,
            assessment: CausalTemporalAssessment,
            ) -> None:
        """要求 accepted/rejected 所引 assertion 形成对应方向的完整路径。"""
        if assessment.status not in {
                CAUSAL_TEMPORAL_ACCEPTED,
                CAUSAL_TEMPORAL_REJECTED,
        }:
            return
        start, target = (
            (cause, effect)
            if assessment.status == CAUSAL_TEMPORAL_ACCEPTED
            else (effect, cause)
        )
        if not self._has_temporal_path(
                start,
                target,
                result,
                set(assessment.supporting_assertion_hashes),
        ):
            raise ValueError("temporal 裁决引用的 assertion 未形成完整方向路径")

    def _has_temporal_path(
            self,
            start: ObjectIdentity,
            target: ObjectIdentity,
            result: EventTimeVerificationResult,
            assertion_hashes: set[int],
            ) -> bool:
        """仅用裁决显式引用的事实重建同序压缩图并检查可达性。"""
        ontology = self.event_time_verifier.facts.ontology
        parent: dict[ObjectIdentity, ObjectIdentity] = {}

        def find(value: ObjectIdentity) -> ObjectIdentity:
            """返回所引同序事实的确定性并查集根。"""
            current = parent.setdefault(value, value)
            if current != value:
                parent[value] = find(current)
            return parent[value]

        def union(left: ObjectIdentity, right: ObjectIdentity) -> None:
            """按稳定键合并所引同序端点。"""
            first = find(left)
            second = find(right)
            if first == second:
                return
            if first.stable_key() <= second.stable_key():
                parent[second] = first
            else:
                parent[first] = second

        decoded = []
        for fact in result.fact_set.facts:
            if fact.assertion_hash not in assertion_hashes:
                continue
            relation = ontology.identity_of(fact.statement.predicate)
            resolved = self.event_time_verifier.resolver.resolve(relation)
            if not isinstance(resolved, ResolvedEventTimeRelation):
                raise TypeError("event-time resolver 返回类型错误")
            if resolved.relation != relation:
                raise ValueError("event-time resolver 替换了 relation 身份")
            first = ontology.identity_of(fact.statement.subject)
            second = ontology.identity_of(fact.statement.object)
            find(first)
            find(second)
            decoded.append((first, second, resolved.direction))
            if resolved.direction == EVENT_TIME_SAME:
                union(first, second)

        find(start)
        find(target)
        outgoing: dict[ObjectIdentity, set[ObjectIdentity]] = {}
        for first, second, direction in decoded:
            if direction in {EVENT_TIME_SAME, EVENT_TIME_DIRECTION_UNKNOWN}:
                continue
            if direction == EVENT_TIME_BEFORE:
                before, after = first, second
            elif direction == EVENT_TIME_AFTER:
                before, after = second, first
            else:
                raise ValueError("event-time resolver direction 未注册")
            outgoing.setdefault(find(before), set()).add(find(after))
        pending = [find(start)]
        expected = find(target)
        visited: set[ObjectIdentity] = set()
        while pending:
            current = pending.pop()
            if current == expected:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(sorted(
                outgoing.get(current, ()),
                key=ObjectIdentity.stable_key,
                reverse=True,
            ))
        return False

    @staticmethod
    def _validate_independence(request: CausalVerificationRequest) -> None:
        """拒绝 forming、候选来源或当前 observation 充当 causal witness。"""
        witness_source = request.witness.verifier_source
        forbidden_sources = {
            request.observation,
            request.spec.proposition.source,
            *request.spec.forming_sources,
        }
        if witness_source in forbidden_sources:
            raise ValueError("causal witness source 必须与 forming/observation 分离")
        if request.spec.proposition.proposition in request.witness.input_objects:
            raise ValueError("causal witness 不得读取待核验 Proposition 自身")

    @staticmethod
    def _reveal_trace(
            request: CausalVerificationRequest,
            result: EventTimeVerificationResult,
            assessment: CausalTemporalAssessment,
            ) -> tuple[int, ...]:
        """返回 witness、时间 assertion 和 temporal reason 的完整整数 trace。"""
        return _causal_reveal_trace(request, result, assessment)


class CausalRelationRuntime:
    """集中管理 causal forming、R-09 adapter、执行 use 和生成采用。"""

    def __init__(
            self,
            semantic_graph: SemanticGraph,
            relation_runtime: RelationClosureRuntime,
            event_time_facts: EventTimeFactIndex,
            event_time_verifier: EventTimeVerifier,
            executor: CausalExecutor,
            verification_protocol: CausalVerificationProtocol,
            ) -> None:
        """装配共享语义图、R-00 owner、event-time facade 和执行器。"""
        if not isinstance(semantic_graph, SemanticGraph):
            raise TypeError("semantic_graph 必须是 SemanticGraph")
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("relation_runtime 必须是 RelationClosureRuntime")
        if relation_runtime.semantic_graph is not semantic_graph:
            raise ValueError("causal runtime 必须共享 R-00 SemanticGraph")
        if not isinstance(event_time_facts, EventTimeFactIndex):
            raise TypeError("event_time_facts 必须是 EventTimeFactIndex")
        if not isinstance(event_time_verifier, EventTimeVerifier):
            raise TypeError("event_time_verifier 必须是 EventTimeVerifier")
        if event_time_verifier.facts is not event_time_facts:
            raise ValueError("event-time verifier 必须绑定同一 typed facade")
        if not isinstance(executor, CausalExecutor):
            raise TypeError("executor 必须是 CausalExecutor")
        if not isinstance(verification_protocol, CausalVerificationProtocol):
            raise TypeError("verification_protocol 类型非法")
        self.semantic_graph = semantic_graph
        self.relation_runtime = relation_runtime
        self.event_time_facts = event_time_facts
        self.event_time_verifier = event_time_verifier
        self.executor = executor
        self.verification_protocol = verification_protocol
        self.adapter = CausalVerificationAdapter(
            relation_runtime,
            event_time_verifier,
            executor.temporal_resolver,
            executor.protocol,
            verification_protocol,
        )
        self._execution_traces: dict[tuple[int, ...], CausalExecutionTrace] = {}
        self._generation_uses: dict[tuple[int, ...], CausalGenerationUse] = {}

    def form(
            self,
            spec: RelationClosureCandidateSpec,
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ):
        """先物化 S-00 typed CAUSES，再登记 R-00 forming unknown。"""
        if not isinstance(spec, RelationClosureCandidateSpec):
            raise TypeError("spec 必须是 RelationClosureCandidateSpec")
        self.semantic_graph.define_atomic(
            spec.proposition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return self.relation_runtime.form(spec)

    def registration(self) -> VerifierRegistration:
        """返回当前 causal runtime 绑定的 R-09 注册项。"""
        return self.adapter.registration()

    def execute(
            self,
            proposition: ObjectIdentity,
            temporal: EventTimeVerificationRequest,
            cause: CausalEndpointEvaluation,
            effect: CausalEndpointEvaluation,
            *,
            use_key: tuple[int, ...],
            generation_use_key: tuple[int, ...] | None = None,
            ) -> CausalExecutionTrace:
        """消费 active CAUSES，运行 provisional effect prediction 并可记录生成采用。"""
        key = _strict_key(use_key, where="causal execution use_key")
        if not isinstance(temporal, EventTimeVerificationRequest):
            raise TypeError("temporal 必须是 EventTimeVerificationRequest")
        fact = self.relation_runtime.consumer.require_proposition(proposition)
        temporal_result = self.event_time_verifier.verify(
            temporal.relations,
            scope=temporal.scope,
        )
        result = self.executor.execute(
            fact,
            temporal_result,
            cause,
            effect,
        )
        self.adapter.validate_temporal_support(
            cause.endpoint,
            effect.endpoint,
            temporal_result,
            result.temporal_assessment,
        )
        generation_use = None
        if generation_use_key is not None:
            generation_use = self._preflight_generation(
                result,
                generation_use_key,
            )
        existing = self._execution_traces.get(key)
        if existing is not None:
            if (
                existing.execution != result
                or existing.generation_use != generation_use
            ):
                raise RuntimeError("同一 causal execution use_key 绑定了不同输入")
            return existing
        relation_use = self.relation_runtime.consume(
            proposition,
            use_key=key,
        )
        if generation_use is not None:
            self._generation_uses[generation_use.use_key] = generation_use
        trace = CausalExecutionTrace(
            relation_use,
            result,
            generation_use,
        )
        self._execution_traces[key] = trace
        return trace

    def _preflight_generation(
            self,
            result: CausalExecutionResult,
            use_key: tuple[int, ...],
            ) -> CausalGenerationUse:
        """在写 relation use 前校验并构造 causal 生成采用。"""
        key = _strict_key(use_key, where="causal generation use_key")
        if (not result.predicted_effect
                or result.status not in {
                    CAUSAL_EXECUTION_PREDICTED,
                    CAUSAL_EXECUTION_CONFLICTED,
                }):
            raise ValueError("未知或时间拒绝结果不得进入 causal 生成采用")
        use = CausalGenerationUse(
            key,
            result.fact.proposition.proposition,
            result.fact.hypothesis,
            result.fact.evidence_keys,
            result.fact.decision_key,
            result.stable_key(),
        )
        existing = self._generation_uses.get(key)
        if existing is not None and existing != use:
            raise RuntimeError("同一 causal generation use_key 绑定了不同结果")
        return existing or use

    def clone_for_evaluation(
            self,
            semantic_graph: SemanticGraph,
            candidate_graph: CandidateProjectionGraph,
            event_time_facts: EventTimeFactIndex,
            ) -> "CausalRelationRuntime":
        """复制 R-00 owner 并在克隆图上重建 event-time 和 causal facade。"""
        cloned_relation = self.relation_runtime.clone_for_evaluation(
            semantic_graph,
            candidate_graph,
        )
        cloned_event_verifier = EventTimeVerifier(
            event_time_facts,
            self.event_time_verifier.resolver,
        )
        cloned_executor = CausalExecutor(
            self.executor.protocol,
            self.executor.temporal_resolver,
        )
        cloned = CausalRelationRuntime(
            semantic_graph,
            cloned_relation,
            event_time_facts,
            cloned_event_verifier,
            cloned_executor,
            self.verification_protocol,
        )
        cloned._execution_traces = dict(self._execution_traces)
        cloned._generation_uses = dict(self._generation_uses)
        return cloned

    def clone_for_context(self, ctx) -> "CausalRelationRuntime":
        """在评测 TrainContext 的图和 scoped registry 上重建全部 typed facade。"""
        if self.semantic_graph.ontology is not self.event_time_facts.ontology:
            raise ValueError("causal runtime 的语义图与 event-time 图必须一致")
        predicate_identities = tuple(
            self.semantic_graph.ontology.identity_of(ref)
            for ref in self.semantic_graph.predicates.refs()
        )
        cloned_semantic = SemanticGraph(
            ctx.graph_ontology,
            AtomicPropositionPredicates(*tuple(
                ctx.graph_ontology.materialize(identity)
                for identity in predicate_identities
            )),
        )
        cloned_candidates = CandidateProjectionGraph(
            ctx.graph_ontology,
            self.relation_runtime.candidate_runtime.graph.protocol,
        )
        from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
        cloned_event_time = EventTimeFactIndex(OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        ))
        return self.clone_for_evaluation(
            cloned_semantic,
            cloned_candidates,
            cloned_event_time,
        )

    def state_key(self) -> tuple:
        """返回 relation owner、执行 trace 和生成采用的完整隔离状态。"""
        executions = tuple(sorted(
            (key, trace.execution.stable_key())
            for key, trace in self._execution_traces.items()
        ))
        generations = tuple(sorted(
            (
                key,
                use.proposition.stable_key(),
                use.hypothesis.stable_key(),
                use.evidence_keys,
                use.decision_key,
                use.execution_key,
            )
            for key, use in self._generation_uses.items()
        ))
        return self.relation_runtime.state_key(), executions, generations


__all__ = [
    "CausalExecutionTrace",
    "CausalGenerationUse",
    "CausalIndependentWitness",
    "CausalRelationRuntime",
    "CausalVerificationAdapter",
    "CausalVerificationArtifact",
    "CausalVerificationProtocol",
    "CausalVerificationRequest",
]
