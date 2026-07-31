"""LC-16 carrier-neutral 候选投影运行时。

本模块直接消费 :class:`CarrierProjectionInput`，复用 H-05 候选引擎、
H-04 resolver 与 CandidateProjectionGraph。它只闭合候选生命周期和
ArtifactSemanticProjection，不把方向占位输出冒充理解、推理或生成结果。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
    ArtifactSemanticProjection,
    make_artifact_semantic_projection,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionInput,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


PROJECTION_HYPOTHESIS_KIND = (16617830, 1)
_DIRECTIONS = (
    PROJECTION_UNDERSTANDING,
    PROJECTION_REASONING,
    PROJECTION_GENERATION,
)


class CarrierProjectionRuntimeError(RuntimeError):
    """共享 feature、输入上下文或候选定义未严格闭合。"""


@dataclass(frozen=True)
class CarrierProjectionProtocol:
    """候选图、独立 verifier 与共享 projection binding 的注入身份。"""

    lifecycle: CandidateProjectionProtocol
    evidence: EvidenceCandidateProtocol
    verifier: IndependentVerifierProtocol
    carrier_feature_predicate: ObjectIdentity
    semantic_object_predicate: ObjectIdentity
    projection_kind_predicate: ObjectIdentity
    projection_hypothesis_kind: tuple[int, ...] = PROJECTION_HYPOTHESIS_KIND

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, CandidateProjectionProtocol):
            raise TypeError("carrier lifecycle protocol 类型非法")
        if not isinstance(self.evidence, EvidenceCandidateProtocol):
            raise TypeError("carrier evidence protocol 类型非法")
        if not isinstance(self.verifier, IndependentVerifierProtocol):
            raise TypeError("carrier verifier protocol 类型非法")
        for name in (
                "carrier_feature_predicate",
                "semantic_object_predicate",
                "projection_kind_predicate"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"{name} 必须是 ObjectIdentity")
        if (not isinstance(self.projection_hypothesis_kind, tuple)
                or not self.projection_hypothesis_kind
                or any(type(item) is not int
                       for item in self.projection_hypothesis_kind)):
            raise ValueError("projection hypothesis kind 必须是非空整数 tuple")


@dataclass(frozen=True)
class CarrierProjectionSpec:
    """数据声明的 carrier feature 到共享语义对象候选。"""

    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    projection_kind: ObjectIdentity
    semantic_object: ObjectIdentity
    directions: tuple[int, ...]
    feature_identities: tuple[ObjectIdentity, ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        for name in ("candidate", "projection_kind", "semantic_object"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"carrier {name} 类型非法")
        if (not isinstance(self.competition_key, tuple)
                or not self.competition_key
                or any(type(item) is not int for item in self.competition_key)):
            raise ValueError("carrier competition_key 必须是非空整数 tuple")
        if (not isinstance(self.directions, tuple)
                or self.directions != tuple(sorted(set(self.directions)))
                or not self.directions
                or set(self.directions) - set(_DIRECTIONS)):
            raise ValueError("carrier directions 必须是已登记的有序非空集合")
        if (not isinstance(self.feature_identities, tuple)
                or not self.feature_identities
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.feature_identities)
                or self.feature_identities != tuple(sorted(
                    set(self.feature_identities), key=ObjectIdentity.stable_key))):
            raise ValueError("carrier feature_identities 必须排序去重且非空")
        if (not isinstance(self.forming_sources, tuple)
                or len(self.forming_sources) < 2
                or any(not isinstance(item, SourceRef)
                       for item in self.forming_sources)
                or len(set(self.forming_sources)) != len(self.forming_sources)):
            raise ValueError("carrier forming_sources 必须是至少两个独立来源")

    def definition(
            self, protocol: CarrierProjectionProtocol,
            ) -> EvidenceCandidateDefinition:
        """构造 H-05 定义，feature binding 保留完整一等对象身份。"""
        bindings = [
            CandidateBinding(
                protocol.semantic_object_predicate,
                self.semantic_object,
                0,
                CANDIDATE_AS_SUBJECT,
            ),
            CandidateBinding(
                protocol.projection_kind_predicate,
                self.projection_kind,
                0,
                CANDIDATE_AS_SUBJECT,
            ),
        ]
        bindings.extend(
            CandidateBinding(
                protocol.carrier_feature_predicate,
                feature,
                ordinal,
                CANDIDATE_AS_SUBJECT,
            )
            for ordinal, feature in enumerate(self.feature_identities)
        )
        return EvidenceCandidateDefinition(
            self.candidate,
            self.competition_key,
            tuple(bindings),
            self.forming_sources,
        )


@dataclass(frozen=True)
class CarrierProjectionTrace:
    """一次 mapper input 到候选生命周期和 artifact projection 的回执。"""

    spec: CarrierProjectionSpec
    carrier_input: CarrierProjectionInput
    outcome: CandidateLearningOutcome
    projection: ArtifactSemanticProjection | None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CarrierProjectionSpec):
            raise TypeError("carrier projection trace spec 类型非法")
        if not isinstance(self.carrier_input, CarrierProjectionInput):
            raise TypeError("carrier projection trace input 类型非法")
        if not isinstance(self.outcome, CandidateLearningOutcome):
            raise TypeError("carrier projection trace outcome 类型非法")
        prediction = self.outcome.prediction
        if (prediction.observation != self.carrier_input.source
                or prediction.scope != self.carrier_input.scope
                or prediction.event_key != self.carrier_input.input_key
                or prediction.visible_inputs != self.carrier_input.visible_inputs
                or prediction.predicted != self.spec.semantic_object):
            raise CarrierProjectionRuntimeError(
                "carrier projection trace prediction context 漂移")
        if self.projection is None:
            return
        if not isinstance(self.projection, ArtifactSemanticProjection):
            raise TypeError("carrier projection trace projection 类型非法")
        if (self.projection.source != self.carrier_input.source
                or self.projection.scope != self.carrier_input.scope
                or self.projection.envelope_identity
                != self.carrier_input.envelope.identity
                or self.projection.anchor_identities
                != self.carrier_input.anchor_identities
                or self.projection.structure_node_identities
                != self.carrier_input.structure_node_identities):
            raise CarrierProjectionRuntimeError("artifact projection local context 漂移")
        if (self.projection.projection_kind != self.spec.projection_kind
                or self.projection.semantic_object != self.spec.semantic_object
                or self.projection.directions != self.spec.directions):
            raise CarrierProjectionRuntimeError("artifact projection spec 漂移")


def default_carrier_projection_protocol() -> CarrierProjectionProtocol:
    """返回九载体共享、全身份注入且不带标签语义表的协议。"""
    lifecycle = CandidateProjectionProtocol(
        *(concept_identity((16617800, item)) for item in range(13)),
        (16617801, 1),
    )
    aggregate = SourceRef(
        16617810, 999, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    evidence = EvidenceCandidateProtocol(
        (16617802, 1),
        (16617802, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )
    verifier = IndependentVerifierProtocol(
        concept_identity((16617803, 1)),
        (16617803, 2),
        (16617803, 3),
        (16617803, 4),
        (16617803, 5),
    )
    return CarrierProjectionProtocol(
        lifecycle,
        evidence,
        verifier,
        concept_identity((16617804, 1)),
        concept_identity((16617804, 2)),
        concept_identity((16617804, 3)),
    )


class CarrierProjectionRuntime:
    """九类载体共享的候选学习和 artifact projection 生命周期。"""

    def __init__(
            self, backend, *,
            protocol: CarrierProjectionProtocol | None = None,
            ) -> None:
        self.protocol = (default_carrier_projection_protocol()
                         if protocol is None else protocol)
        if not isinstance(self.protocol, CarrierProjectionProtocol):
            raise TypeError("carrier projection protocol 类型非法")
        context = make_train_context(backend)
        self.graph = CandidateProjectionGraph(
            context.graph_ontology, self.protocol.lifecycle)
        self.learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(self.protocol.evidence),
            self.graph,
            IndependentObjectVerifier(self.protocol.verifier),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        self._specs: dict[ObjectIdentity, CarrierProjectionSpec] = {}
        self._hypotheses: dict[ObjectIdentity, HypothesisKey] = {}

    def register(
            self, spec: CarrierProjectionSpec, *, timestamp_base: int = 0,
            ) -> HypothesisKey:
        if not isinstance(spec, CarrierProjectionSpec):
            raise TypeError("carrier projection spec 类型非法")
        prior = self._specs.get(spec.candidate)
        if prior is not None:
            if prior != spec:
                raise CarrierProjectionRuntimeError("同一 carrier candidate 定义漂移")
            return self._hypotheses[spec.candidate]
        hypothesis = self.learning.register(
            spec.definition(self.protocol), timestamp_base=timestamp_base)
        self._specs[spec.candidate] = spec
        self._hypotheses[spec.candidate] = hypothesis
        return hypothesis

    def learn(
            self,
            spec: CarrierProjectionSpec,
            carrier_input: CarrierProjectionInput,
            *,
            revealed: RevealedObjectObservation,
            replacement_candidate: ObjectIdentity | None = None,
            ) -> CarrierProjectionTrace:
        """核验 mapper 输入后执行 prediction、reveal、H-04 与图同步。"""
        if not isinstance(spec, CarrierProjectionSpec):
            raise TypeError("carrier projection spec 类型非法")
        if not isinstance(carrier_input, CarrierProjectionInput):
            raise TypeError("carrier projection input 类型非法")
        if spec.feature_identities != carrier_input.feature_identities:
            raise CarrierProjectionRuntimeError(
                "candidate feature bindings 与 input features 不精确对齐")
        if not isinstance(revealed, RevealedObjectObservation):
            raise TypeError("carrier projection reveal 类型非法")
        if (revealed.observation != carrier_input.source
                or revealed.scope != carrier_input.scope
                or revealed.event_key != carrier_input.input_key):
            raise CarrierProjectionRuntimeError(
                "reveal 必须精确绑定 mapper input source/scope/key")
        definition = spec.definition(self.protocol)
        feature_values = tuple(sorted(
            (item.value for item in definition.bindings
             if item.predicate == self.protocol.carrier_feature_predicate),
            key=ObjectIdentity.stable_key,
        ))
        if feature_values != carrier_input.feature_identities:
            raise CarrierProjectionRuntimeError(
                "H-05 definition feature bindings 与 mapper input 漂移")
        prior = self._specs.get(spec.candidate)
        if prior is not None and prior != spec:
            raise CarrierProjectionRuntimeError("carrier candidate spec 漂移")
        replacement = None
        if replacement_candidate is not None:
            replacement = self._hypotheses.get(replacement_candidate)
            if replacement is None:
                raise KeyError("replacement candidate 尚未登记")
        if spec.candidate not in self._specs:
            self.register(
                spec, timestamp_base=self.learning.next_timestamps(1)[0])
        timestamps = self.learning.next_timestamps(3)
        outcome = self.learning.recognize(
            self._hypotheses[spec.candidate],
            observation=carrier_input.source,
            scope=carrier_input.scope,
            event_key=carrier_input.input_key,
            visible_inputs=carrier_input.visible_inputs,
            predicted=spec.semantic_object,
            revealed=revealed,
            timestamp_seq=timestamps[0],
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
            replacement=replacement,
        )
        projection = self._materialize_projection(
            spec, carrier_input, outcome)
        return CarrierProjectionTrace(spec, carrier_input, outcome, projection)

    def _materialize_projection(
            self,
            spec: CarrierProjectionSpec,
            carrier_input: CarrierProjectionInput,
            outcome: CandidateLearningOutcome,
            ) -> ArtifactSemanticProjection | None:
        if outcome.projection is None:
            return None
        local_hypothesis = HypothesisKey(
            self.protocol.projection_hypothesis_kind,
            spec.definition(self.protocol).stable_key(),
            spec.competition_key,
            carrier_input.scope,
            carrier_input.source,
        )
        if not outcome.projection.history:
            raise CarrierProjectionRuntimeError(
                "candidate lifecycle projection 缺少图内历史")
        projected_evidence = tuple(
            EvidenceRecord.from_stable_key(item)
            for item in outcome.projection.history[-1].definition.evidence_keys
        )
        local_evidence = tuple(sorted((
            EvidenceRecord(
                item.evidence_id,
                local_hypothesis,
                item.stance,
                item.reason_key,
                item.source,
                item.timestamp_seq,
                item.payload,
            )
            for item in projected_evidence
        ), key=EvidenceRecord.stable_key))
        return make_artifact_semantic_projection(
            envelope_identity=carrier_input.envelope.identity,
            source=carrier_input.source,
            scope=carrier_input.scope,
            anchor_identities=carrier_input.anchor_identities,
            structure_node_identities=carrier_input.structure_node_identities,
            projection_kind=spec.projection_kind,
            semantic_object=spec.semantic_object,
            lifecycle_state=outcome.projection.state,
            hypothesis=local_hypothesis,
            evidence=local_evidence,
            directions=spec.directions,
            projection_key=carrier_input.input_key,
        )

    def retained_projection(
            self, candidate: ObjectIdentity,
            ) -> CandidateGraphProjection:
        """清缓存或恢复后只读返回既有候选 lifecycle 投影。"""
        return self.learning.projection_for_candidate(candidate)


__all__ = [
    "CarrierProjectionProtocol",
    "CarrierProjectionRuntime",
    "CarrierProjectionRuntimeError",
    "CarrierProjectionSpec",
    "CarrierProjectionTrace",
    "PROJECTION_HYPOTHESIS_KIND",
    "default_carrier_projection_protocol",
]
