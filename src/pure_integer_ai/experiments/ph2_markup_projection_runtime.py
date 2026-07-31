"""LC-16 Markdown/HTML 共享候选投影与三向 consumer 纵切。

载体 adapter 只提供 raw/tree observation。本模块复用 H-05 候选引擎，
把数据提供的结构候选经独立 reveal 后投影为共享 ArtifactSemanticProjection。
它不读取 Memory/Companion，不调用 teacher，也不把标签名写成语义表。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


PROJECTION_HYPOTHESIS_KIND = (16617030, 1)
CONSUMER_UNDERSTANDING = 1
CONSUMER_REASONING = 2
CONSUMER_GENERATION = 3
_DIRECTIONS = (
    PROJECTION_UNDERSTANDING,
    PROJECTION_REASONING,
    PROJECTION_GENERATION,
)


class MarkupProjectionRuntimeError(RuntimeError):
    """共享 carrier projection 或 consumer 合同不闭合。"""


@runtime_checkable
class CarrierMaterialization(Protocol):
    """Markdown/HTML adapter 共同提供的最小只读视图。"""

    sources: tuple[SourceRef, ...]
    scopes: tuple[Any, ...]
    envelopes: tuple[Any, ...]
    anchors: tuple[Any, ...]
    structure_nodes: tuple[Any, ...]


@dataclass(frozen=True)
class MarkupProjectionProtocol:
    """候选图、独立 verifier 和共享 projection kind 的注入身份。"""

    lifecycle: CandidateProjectionProtocol
    evidence: EvidenceCandidateProtocol
    verifier: IndependentVerifierProtocol
    structure_feature_predicate: ObjectIdentity
    semantic_object_predicate: ObjectIdentity
    projection_kind_predicate: ObjectIdentity
    projection_hypothesis_kind: tuple[int, ...] = PROJECTION_HYPOTHESIS_KIND

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, CandidateProjectionProtocol):
            raise TypeError("markup lifecycle protocol 类型非法")
        if not isinstance(self.evidence, EvidenceCandidateProtocol):
            raise TypeError("markup evidence protocol 类型非法")
        if not isinstance(self.verifier, IndependentVerifierProtocol):
            raise TypeError("markup verifier protocol 类型非法")
        for name in (
                "structure_feature_predicate",
                "semantic_object_predicate",
                "projection_kind_predicate"):
            value = getattr(self, name)
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"{name} 必须是 ObjectIdentity")
        if not self.projection_hypothesis_kind:
            raise ValueError("projection hypothesis kind 不能为空")


@dataclass(frozen=True)
class MarkupProjectionSpec:
    """由数据提供的候选映射；不包含固定 Markdown/HTML 标签枚举。"""

    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    projection_kind: ObjectIdentity
    semantic_object: ObjectIdentity
    directions: tuple[int, ...]
    structure_features: tuple[ObjectIdentity, ...]
    forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("markup candidate 类型非法")
        if not isinstance(self.projection_kind, ObjectIdentity):
            raise TypeError("markup projection_kind 类型非法")
        if not isinstance(self.semantic_object, ObjectIdentity):
            raise TypeError("markup semantic_object 类型非法")
        if (not isinstance(self.directions, tuple)
                or self.directions != tuple(sorted(set(self.directions)))
                or not self.directions
                or set(self.directions) - set(_DIRECTIONS)):
            raise ValueError("markup directions 必须是已登记的有序非空集合")
        if (not isinstance(self.structure_features, tuple)
                or not self.structure_features
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.structure_features)):
            raise ValueError("markup structure_features 必须为非空对象 tuple")
        if (not isinstance(self.forming_sources, tuple)
                or len(self.forming_sources) < 2
                or any(not isinstance(item, SourceRef)
                       for item in self.forming_sources)
                or len(set(self.forming_sources)) != len(self.forming_sources)):
            raise ValueError("markup forming_sources 必须是至少两个独立来源")
        if not self.competition_key:
            raise ValueError("markup competition_key 不能为空")

    def definition(self, protocol: MarkupProjectionProtocol
                   ) -> EvidenceCandidateDefinition:
        """把开放数据映射为共享 H-05 候选定义。"""
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
                protocol.structure_feature_predicate,
                feature,
                ordinal,
                CANDIDATE_AS_SUBJECT,
            )
            for ordinal, feature in enumerate(self.structure_features))
        return EvidenceCandidateDefinition(
            self.candidate,
            self.competition_key,
            tuple(bindings),
            self.forming_sources,
        )


@dataclass(frozen=True)
class MarkupProjectionUse:
    """一个方向 consumer 的只读使用记录。"""

    projection: ArtifactSemanticProjection
    direction: int
    consumer: ObjectIdentity
    output: ObjectIdentity
    accepted: bool
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.projection, ArtifactSemanticProjection):
            raise TypeError("projection use 必须绑定 ArtifactSemanticProjection")
        if self.direction not in _DIRECTIONS:
            raise ValueError("projection use direction 未登记")
        if self.direction not in self.projection.directions:
            raise ValueError("projection use 越过 projection directions")
        if not isinstance(self.consumer, ObjectIdentity):
            raise TypeError("projection consumer 类型非法")
        if not isinstance(self.output, ObjectIdentity):
            raise TypeError("projection use output 类型非法")
        if type(self.accepted) is not bool:
            raise TypeError("projection use accepted 必须是 bool")
        if not isinstance(self.trace, tuple) or any(
                type(item) is not int for item in self.trace):
            raise ValueError("projection use trace 必须是整数 tuple")

    def stable_key(self) -> tuple[int, ...]:
        projection_key = self.projection.identity.stable_key()
        consumer_key = self.consumer.stable_key()
        output_key = self.output.stable_key()
        return (
            1,
            len(projection_key),
            *projection_key,
            self.direction,
            len(consumer_key),
            *consumer_key,
            len(output_key),
            *output_key,
            int(self.accepted),
            len(self.trace),
            *self.trace,
        )


@dataclass(frozen=True)
class MarkupProjectionOutcome:
    """延迟结果对一个精确 projection Use 的来源化归因。"""

    use: MarkupProjectionUse
    accepted: bool
    source: SourceRef
    outcome_key: tuple[int, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.use, MarkupProjectionUse):
            raise TypeError("markup outcome use 类型非法")
        if type(self.accepted) is not bool:
            raise TypeError("markup outcome accepted 必须是 bool")
        if not isinstance(self.source, SourceRef):
            raise TypeError("markup outcome source 类型非法")
        for name in ("outcome_key", "trace"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int for item in value)):
                raise ValueError(f"markup outcome {name} 必须是非空整数 tuple")

    def stable_key(self) -> tuple[int, ...]:
        use_key = self.use.stable_key()
        source_key = self.source.stable_key()
        return (
            1,
            len(use_key),
            *use_key,
            int(self.accepted),
            len(source_key),
            *source_key,
            len(self.outcome_key),
            *self.outcome_key,
            len(self.trace),
            *self.trace,
        )


@dataclass(frozen=True)
class MarkupProjectionTrace:
    """一次 Observation→Evidence→projection→consumer 的完整回执。"""

    spec: MarkupProjectionSpec
    observation: SourceRef
    outcome: CandidateLearningOutcome
    carrier_projection: ArtifactSemanticProjection | None
    uses: tuple[MarkupProjectionUse, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, MarkupProjectionSpec):
            raise TypeError("projection trace spec 类型非法")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("projection trace observation 类型非法")
        if not isinstance(self.outcome, CandidateLearningOutcome):
            raise TypeError("projection trace outcome 类型非法")
        if self.carrier_projection is None and self.uses:
            raise ValueError("无 projection 不得产生 consumer use")
        if self.carrier_projection is not None:
            if (self.carrier_projection.semantic_object
                    != self.spec.semantic_object):
                raise ValueError("projection semantic object 与 spec 漂移")
            if (self.carrier_projection.source != self.observation):
                raise ValueError("projection source 与 trace observation 漂移")
        if not isinstance(self.uses, tuple):
            raise TypeError("projection uses 必须是 tuple")


def default_markup_projection_protocol() -> MarkupProjectionProtocol:
    """返回纵切使用的全注入、无标签语义表的公开协议。"""
    lifecycle_values = tuple(concept_identity((16617000, item))
                             for item in range(13))
    lifecycle = CandidateProjectionProtocol(
        *lifecycle_values,
        (16617001, 1),
    )
    aggregate = SourceRef(
        16617010, 999, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    evidence = EvidenceCandidateProtocol(
        (16617002, 1),
        (16617002, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )
    verifier = IndependentVerifierProtocol(
        concept_identity((16617003, 1)),
        (16617003, 2),
        (16617003, 3),
        (16617003, 4),
        (16617003, 5),
    )
    return MarkupProjectionProtocol(
        lifecycle,
        evidence,
        verifier,
        concept_identity((16617004, 1)),
        concept_identity((16617004, 2)),
        concept_identity((16617004, 3)),
    )


class MarkupProjectionRuntime:
    """共享 Markdown/HTML 候选学习、projection 物化和三向只读 consumer。"""

    def __init__(self, backend, *, protocol: MarkupProjectionProtocol | None = None):
        self.protocol = (default_markup_projection_protocol()
                         if protocol is None else protocol)
        if not isinstance(self.protocol, MarkupProjectionProtocol):
            raise TypeError("markup protocol 类型非法")
        context = make_train_context(backend)
        self.graph = CandidateProjectionGraph(
            context.graph_ontology, self.protocol.lifecycle)
        self.learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(self.protocol.evidence),
            self.graph,
            IndependentObjectVerifier(self.protocol.verifier),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        self._specs: dict[ObjectIdentity, MarkupProjectionSpec] = {}
        self._hypotheses: dict[ObjectIdentity, HypothesisKey] = {}

    def register(self, spec: MarkupProjectionSpec, *, timestamp_base: int = 0
                 ) -> HypothesisKey:
        if not isinstance(spec, MarkupProjectionSpec):
            raise TypeError("markup spec 类型非法")
        prior = self._specs.get(spec.candidate)
        if prior is not None:
            if prior != spec:
                raise MarkupProjectionRuntimeError(
                    "同一 markup candidate 定义发生漂移")
            return self._hypotheses[spec.candidate]
        definition = spec.definition(self.protocol)
        hypothesis = self.learning.register(
            definition, timestamp_base=timestamp_base)
        self._specs[spec.candidate] = spec
        self._hypotheses[spec.candidate] = hypothesis
        return hypothesis

    def learn(
            self,
            spec: MarkupProjectionSpec,
            materialization: CarrierMaterialization,
            *,
            node_indices: tuple[int, ...],
            event_key: tuple[int, ...],
            revealed: RevealedObjectObservation,
            replacement_candidate: ObjectIdentity | None = None,
            envelope_index: int = 0,
            ) -> MarkupProjectionTrace:
        """提交一个 carrier observation，并在 active/superseded 时物化 projection。"""
        if not isinstance(materialization, CarrierMaterialization):
            raise TypeError("markup materialization 未实现共同 carrier 视图")
        if spec.candidate not in self._specs:
            self.register(spec, timestamp_base=self.learning.next_timestamps(1)[0])
        if self._specs[spec.candidate] != spec:
            raise MarkupProjectionRuntimeError("markup candidate spec 漂移")
        if (type(envelope_index) is not int or not
                0 <= envelope_index < len(materialization.envelopes)):
            raise ValueError("markup envelope_index 越界")
        if not isinstance(node_indices, tuple) or not node_indices:
            raise ValueError("markup node_indices 必须非空")
        source = materialization.sources[envelope_index]
        scope = materialization.scopes[envelope_index]
        envelope = materialization.envelopes[envelope_index]
        nodes = tuple(materialization.structure_nodes[index]
                      for index in node_indices)
        if any(item.envelope_identity != envelope.identity for item in nodes):
            raise ValueError("markup node 不属于指定 envelope")
        if revealed.observation != source or revealed.scope != scope:
            raise ValueError("reveal 必须绑定 carrier observation")
        hypothesis = self._hypotheses[spec.candidate]
        replacement = (None if replacement_candidate is None else
                       self._hypotheses.get(replacement_candidate))
        if replacement_candidate is not None and replacement is None:
            raise KeyError("replacement candidate 尚未登记")
        timestamps = self.learning.next_timestamps(3)
        outcome = self.learning.recognize(
            hypothesis,
            observation=source,
            scope=scope,
            event_key=event_key,
            visible_inputs=tuple(item.identity for item in nodes),
            predicted=spec.semantic_object,
            revealed=revealed,
            timestamp_seq=timestamps[0],
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
            replacement=replacement,
        )
        carrier_projection = None
        if outcome.projection is not None:
            local_hypothesis = HypothesisKey(
                self.protocol.projection_hypothesis_kind,
                spec.definition(self.protocol).stable_key(),
                spec.competition_key,
                scope,
                source,
            )
            local_evidence = EvidenceRecord(
                outcome.evidence.evidence_id,
                local_hypothesis,
                outcome.evidence.stance,
                outcome.evidence.reason_key,
                outcome.evidence.source,
                outcome.evidence.timestamp_seq,
                outcome.evidence.payload,
            )
            carrier_projection = make_artifact_semantic_projection(
                envelope_identity=envelope.identity,
                source=source,
                scope=scope,
                anchor_identities=tuple(sorted(
                    (item.anchor_identity for item in nodes),
                    key=ObjectIdentity.stable_key,
                )),
                structure_node_identities=tuple(sorted(
                    (item.identity for item in nodes),
                    key=ObjectIdentity.stable_key,
                )),
                projection_kind=spec.projection_kind,
                semantic_object=spec.semantic_object,
                lifecycle_state=outcome.projection.state,
                hypothesis=local_hypothesis,
                evidence=(local_evidence,),
                directions=spec.directions,
                projection_key=event_key,
            )
        uses = () if carrier_projection is None else self._consume(
            carrier_projection)
        return MarkupProjectionTrace(
            spec, source, outcome, carrier_projection, uses)

    def _consume(self, projection: ArtifactSemanticProjection
                 ) -> tuple[MarkupProjectionUse, ...]:
        if projection.lifecycle_state != self.protocol.lifecycle.active_state:
            return ()
        result = []
        for direction, consumer_key in zip(
                _DIRECTIONS,
                (CONSUMER_UNDERSTANDING,
                 CONSUMER_REASONING,
                 CONSUMER_GENERATION),
                strict=True):
            if direction not in projection.directions:
                continue
            consumer = concept_identity((16617005, consumer_key))
            if direction == PROJECTION_UNDERSTANDING:
                output = projection.semantic_object
            elif direction == PROJECTION_REASONING:
                output = projection.hypothesis.object_identity()
            else:
                output = projection.envelope_identity
            accepted = direction in projection.directions
            if direction == PROJECTION_GENERATION:
                accepted = accepted and bool(projection.structure_node_identities)
            result.append(MarkupProjectionUse(
                projection,
                direction,
                consumer,
                output,
                accepted,
                (projection.projection_key[0], consumer_key),
            ))
        return tuple(result)

    def record_outcome(
            self,
            use: MarkupProjectionUse,
            *,
            accepted: bool,
            source: SourceRef,
            outcome_key: tuple[int, ...],
            trace: tuple[int, ...],
            ) -> MarkupProjectionOutcome:
        """记录一个不可变、精确指向 Use 的外部结果，不直接改候选。"""
        return MarkupProjectionOutcome(
            use, accepted, source, outcome_key, trace)

    def reveal_from_outcome(
            self,
            materialization: CarrierMaterialization,
            *,
            event_key: tuple[int, ...],
            outcome: MarkupProjectionOutcome,
            envelope_index: int = 0,
            ) -> RevealedObjectObservation:
        """把外部 Use outcome 转为独立 reveal，后续仍由 H-05 决定状态。"""
        if not isinstance(materialization, CarrierMaterialization):
            raise TypeError("outcome reveal materialization 类型非法")
        if not isinstance(outcome, MarkupProjectionOutcome):
            raise TypeError("outcome reveal 类型非法")
        source = materialization.sources[envelope_index]
        scope = materialization.scopes[envelope_index]
        target = outcome.use.projection.semantic_object
        return RevealedObjectObservation(
            source,
            scope,
            event_key,
            outcome.source,
            (target,) if outcome.accepted else (),
            () if outcome.accepted else (target,),
            (*outcome.outcome_key, *outcome.trace),
        )

    def retained_projection(
            self, candidate: ObjectIdentity) -> CandidateGraphProjection:
        """清缓存或恢复后只读返回图内 lifecycle 投影。"""
        return self.learning.projection_for_candidate(candidate)


__all__ = [
    "CONSUMER_GENERATION",
    "CONSUMER_REASONING",
    "CONSUMER_UNDERSTANDING",
    "CarrierMaterialization",
    "MarkupProjectionProtocol",
    "MarkupProjectionOutcome",
    "MarkupProjectionRuntime",
    "MarkupProjectionRuntimeError",
    "MarkupProjectionSpec",
    "MarkupProjectionTrace",
    "MarkupProjectionUse",
    "default_markup_projection_protocol",
]
