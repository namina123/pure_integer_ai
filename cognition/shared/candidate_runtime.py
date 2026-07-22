"""H-05 候选内核、独立 verifier、H-04 和图投影的薄编排层。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
    CandidateProjectionGraph,
    EvidenceCandidateProjector,
    MaterializedCandidateDefinition,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidatePrediction,
    CandidateVerification,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class CandidateHistoryUnavailableError(RuntimeError):
    """图中已有候选生命周期，但 M-03 前缺少可继续追加的 H-00 历史。"""


@dataclass(frozen=True)
class CandidateProjectionMetadata:
    """候选定义和 lifecycle statement 使用的注入式 assertion 元数据。"""

    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("projection qualifiers 必须是整数 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="CandidateProjectionMetadata",
        )
        if type(self.provenance_kind) is not int or self.provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(self.epistemic_origin) is not int or self.epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(self.content_version) is not int or self.content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if any(type(item) is not int for item in self.qualifiers):
            raise ValueError("qualifiers 必须使用严格整数")

    def kwargs(self) -> dict:
        """返回投影 facade 接受的统一关键字参数。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }


@dataclass(frozen=True)
class CandidateLearningOutcome:
    """一次 recognition 的冻结预测、揭示、Evidence、决策和图状态。"""

    prediction: CandidatePrediction
    verification: CandidateVerification
    evidence: EvidenceRecord
    decision: ResolverDecision
    projection: CandidateGraphProjection | None


@dataclass(frozen=True)
class CandidateLearningReport:
    """从当前 owner 和图历史派生的全链计数，不把计数当掌握证明。"""

    candidate_count: int
    prediction_count: int
    evidence_count: int
    decision_count: int
    projection_event_count: int
    active_projection_count: int


class CandidateLearningRuntime:
    """编排 forming、prediction、independent reveal、H-04 和图状态同步。"""

    def __init__(
            self, engine: EvidenceCandidateEngine,
            graph: CandidateProjectionGraph,
            verifier: IndependentObjectVerifier,
            metadata: CandidateProjectionMetadata) -> None:
        if not isinstance(engine, EvidenceCandidateEngine):
            raise TypeError("engine 必须是 EvidenceCandidateEngine")
        if not isinstance(graph, CandidateProjectionGraph):
            raise TypeError("graph 必须是 CandidateProjectionGraph")
        if not isinstance(verifier, IndependentObjectVerifier):
            raise TypeError("verifier 必须是 IndependentObjectVerifier")
        if not isinstance(metadata, CandidateProjectionMetadata):
            raise TypeError("metadata 必须是 CandidateProjectionMetadata")
        self.engine = engine
        self.graph = graph
        self.verifier = verifier
        self.metadata = metadata
        self.projector = EvidenceCandidateProjector(engine, graph)
        self._hypotheses: set[HypothesisKey] = set()
        self._candidate_hypotheses: dict[ObjectIdentity, HypothesisKey] = {}
        self._predictions: set[tuple[int, ...]] = set()
        self._logical_clock = 0

    @classmethod
    def from_history(
            cls, engine: EvidenceCandidateEngine,
            graph: CandidateProjectionGraph,
            verifier: IndependentObjectVerifier,
            metadata: CandidateProjectionMetadata,
            ) -> "CandidateLearningRuntime":
        """从已恢复 H-00/H-04 和候选图重建运行 owner，不恢复半成品 prediction。"""
        runtime = cls(engine, graph, verifier, metadata)
        hypotheses = engine.ledger.hypotheses()
        candidate_hypotheses: dict[ObjectIdentity, HypothesisKey] = {}
        logical_clock = 0
        for hypothesis in hypotheses:
            definition = engine.definition(hypothesis)
            materialized = graph.read_definition(hypothesis)
            if materialized.definition != definition:
                raise CandidateHistoryUnavailableError(
                    "M-03 候选定义与图恢复定义不一致")
            prior = candidate_hypotheses.get(definition.candidate)
            if prior is not None and prior != hypothesis:
                raise CandidateHistoryUnavailableError(
                    "恢复历史中同一候选绑定多个 Hypothesis")
            candidate_hypotheses[definition.candidate] = hypothesis
            evidence = engine.ledger.evidence_history(hypothesis)
            transitions = engine.ledger.transition_history(hypothesis)
            logical_clock = max(
                logical_clock,
                *(item.timestamp_seq for item in evidence),
                *(item.timestamp_seq for item in transitions),
            )
            decision_by_id = {
                item.decision_id: item
                for item in engine.resolver.decision_history(hypothesis)
            }
            logical_clock = max((
                logical_clock,
                *(item.timestamp_seq for item in decision_by_id.values()),
            ))
            candidate_ref = graph.ontology.resolve(definition.candidate)
            history = () if candidate_ref is None else graph.history(candidate_ref)
            for event in history:
                projected = event.definition
                decision = ResolverDecision.from_stable_key(
                    projected.decision_key)
                if decision_by_id.get(decision.decision_id) != decision:
                    raise CandidateHistoryUnavailableError(
                        "候选图引用的 H-04 decision 未从 M-03 完整恢复")
                trace = decision.candidate(hypothesis)
                active_ids = frozenset((
                    *trace.after.support_evidence_ids,
                    *trace.after.refute_evidence_ids,
                    *trace.after.unknown_evidence_ids,
                ))
                projected_evidence = tuple(
                    EvidenceRecord.from_stable_key(item)
                    for item in projected.evidence_keys)
                if ({item.evidence_id for item in projected_evidence}
                        != active_ids
                        or any(item.hypothesis != hypothesis
                               for item in projected_evidence)):
                    raise CandidateHistoryUnavailableError(
                        "候选图 Evidence key 与 H-04 decision 快照不一致")
                ledger_by_id = {
                    item.evidence_id: item for item in evidence
                }
                if any(ledger_by_id.get(item.evidence_id) != item
                       for item in projected_evidence):
                    raise CandidateHistoryUnavailableError(
                        "候选图 Evidence key 未在 M-03 ledger 无损恢复")
                logical_clock = max(
                    logical_clock,
                    projected.timestamp_seq,
                    decision.timestamp_seq,
                )
            active = engine.active(hypothesis)
            if not history:
                if active is not None:
                    raise CandidateHistoryUnavailableError(
                        "M-03 当前 adopted 候选缺少 active 图投影")
                continue
            projection = graph.project(candidate_ref)
            if active is not None:
                latest = history[-1].definition
                if (projection.state != graph.protocol.active_state
                        or latest.decision_key
                        != active.decision.stable_key()):
                    raise CandidateHistoryUnavailableError(
                        "M-03 当前 adopted 决策与 active 图投影不同步")
            elif projection.state == graph.protocol.active_state:
                raise CandidateHistoryUnavailableError(
                    "候选图仍为 active，但 M-03 当前状态已不可采用")
            snapshot = engine.ledger.snapshot(hypothesis)
            graph_superseded = (
                projection.state == graph.protocol.superseded_state)
            if ((snapshot.lifecycle == LIFECYCLE_SUPERSEDED)
                    != graph_superseded):
                raise CandidateHistoryUnavailableError(
                    "M-03 supersede 状态与候选图当前投影不一致")
        runtime._hypotheses = set(hypotheses)
        runtime._candidate_hypotheses = candidate_hypotheses
        runtime._predictions = {
            item.stable_key() for item in engine.predictions()
        }
        runtime._logical_clock = logical_clock
        return runtime

    def preflight_register(
            self, definition: EvidenceCandidateDefinition, *,
            timestamp_base: int = 0) -> HypothesisKey:
        """零写核验 H-00 forming、运行 owner 和候选图定义可共同登记。"""
        probe = self.engine.clone()
        hypothesis = probe.register(
            definition, timestamp_base=timestamp_base)
        prior = self._candidate_hypotheses.get(definition.candidate)
        if prior is not None and prior != hypothesis:
            raise RuntimeError("同一候选对象绑定了不同 Hypothesis")
        candidate_ref = self.graph.ontology.resolve(definition.candidate)
        if prior is None:
            hypothesis_ref = self.graph.ontology.resolve(
                hypothesis.object_identity())
            if hypothesis_ref is not None:
                restored = self.graph.read_definition(hypothesis)
                if restored.definition != definition:
                    raise RuntimeError("恢复候选图定义与 forming 输入不一致")
                raise CandidateHistoryUnavailableError(
                    "候选图已恢复但 H-00/H-04 历史尚未由 M-03 恢复，禁止伪续写")
            if (candidate_ref is not None
                    and self.graph.history(candidate_ref)):
                raise RuntimeError("候选已有 lifecycle，但对应 Hypothesis 图对象缺失")
        self.graph.preflight_definition(
            definition,
            hypothesis,
            **self.metadata.kwargs(),
        )
        return hypothesis

    def register(
            self, definition: EvidenceCandidateDefinition, *,
            timestamp_base: int = 0) -> HypothesisKey:
        """先用 clone 预检 forming，再写图定义和正式 unknown Evidence。"""
        hypothesis = self.preflight_register(
            definition, timestamp_base=timestamp_base)
        self.graph.define(
            definition,
            hypothesis,
            **self.metadata.kwargs(),
        )
        committed = self.engine.register(
            definition, timestamp_base=timestamp_base)
        if committed != hypothesis:
            raise RuntimeError("候选 clone 预检与正式登记身份不一致")
        self._hypotheses.add(hypothesis)
        prior = self._candidate_hypotheses.get(definition.candidate)
        if prior is not None and prior != hypothesis:
            raise RuntimeError("候选登记后 owner 身份发生漂移")
        self._candidate_hypotheses[definition.candidate] = hypothesis
        self._logical_clock = max(
            self._logical_clock,
            timestamp_base + max(len(definition.forming_sources) - 1, 0),
        )
        return hypothesis

    def recognize(
            self, hypothesis: HypothesisKey, *, observation: SourceRef,
            scope: ScopeIdentity, event_key: tuple[int, ...],
            visible_inputs: tuple[ObjectIdentity, ...],
            predicted: ObjectIdentity,
            revealed: RevealedObjectObservation,
            timestamp_seq: int, resolve_timestamp_seq: int,
            projection_timestamp_seq: int,
            scorers=(), archive_refuted: bool = False,
            replacement: HypothesisKey | None = None,
            ) -> CandidateLearningOutcome:
        """完整执行 prediction 先行、独立 reveal、H-04 和 typed 图同步。"""
        prediction = self.engine.predict(
            hypothesis,
            observation=observation,
            scope=scope,
            event_key=event_key,
            visible_inputs=visible_inputs,
            predicted=predicted,
        )
        verification = self.verifier.verify(prediction, revealed)
        probe = self.engine.clone()
        probe_prediction = probe.predict(
            hypothesis,
            observation=observation,
            scope=scope,
            event_key=event_key,
            visible_inputs=visible_inputs,
            predicted=predicted,
        )
        probe.reveal(
            probe_prediction,
            verification,
            timestamp_seq=timestamp_seq,
        )
        probe.resolve(
            hypothesis,
            timestamp_seq=resolve_timestamp_seq,
            scorers=scorers,
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        evidence = self.engine.reveal(
            prediction,
            verification,
            timestamp_seq=timestamp_seq,
        )
        decision = self.engine.resolve(
            hypothesis,
            timestamp_seq=resolve_timestamp_seq,
            scorers=scorers,
            archive_refuted=archive_refuted,
            replacement=replacement,
        )
        projections = self.sync_competition(
            hypothesis,
            timestamp_seq=projection_timestamp_seq,
        )
        projection = next((
            item for candidate, item in projections
            if candidate == hypothesis
        ), None)
        self._hypotheses.add(hypothesis)
        self._predictions.add(prediction.stable_key())
        projection_written = any(
            item is not None for _candidate, item in projections)
        self._logical_clock = max(
            self._logical_clock,
            timestamp_seq,
            resolve_timestamp_seq,
            projection_timestamp_seq if projection_written else 0,
        )
        return CandidateLearningOutcome(
            prediction,
            verification,
            evidence,
            decision,
            projection,
        )

    def sync_competition(
            self, hypothesis: HypothesisKey, *, timestamp_seq: int,
            ) -> tuple[tuple[HypothesisKey, CandidateGraphProjection | None], ...]:
        """先同步同组可采用候选，再同步降级/替代者，避免 decision 局部陈旧。"""
        snapshots = self.engine.ledger.competition(hypothesis)
        ordered = tuple(sorted(
            (item.hypothesis for item in snapshots),
            key=lambda item: (
                self.engine.active(item) is None,
                item.stable_key(),
            ),
        ))
        results = tuple(
            (candidate, self.sync(candidate, timestamp_seq=timestamp_seq))
            for candidate in ordered
        )
        self._hypotheses.update(ordered)
        return tuple(sorted(
            results,
            key=lambda item: item[0].stable_key(),
        ))

    def sync(
            self, hypothesis: HypothesisKey, *, timestamp_seq: int,
            ) -> CandidateGraphProjection | None:
        """把 H-00/H-04 当前状态保守同步为 promotion/refresh/demotion/supersede。"""
        definition = self.engine.definition(hypothesis)
        active = self.engine.active(hypothesis)
        candidate = self.graph.ontology.resolve(definition.candidate)
        history = () if candidate is None else self.graph.history(candidate)
        if active is not None:
            return self.projector.promote(
                hypothesis,
                timestamp_seq=timestamp_seq,
                **self.metadata.kwargs(),
            )
        if not history:
            return None
        projection = self.graph.project(candidate)
        if projection.state != self.graph.protocol.active_state:
            return projection
        snapshot = self.engine.ledger.snapshot(hypothesis)
        if snapshot.lifecycle == LIFECYCLE_SUPERSEDED:
            transitions = self.engine.ledger.transition_history(hypothesis)
            if not transitions or transitions[-1].replacement is None:
                raise RuntimeError("superseded H-00 候选缺 replacement transition")
            return self.projector.supersede(
                hypothesis,
                transitions[-1].replacement,
                timestamp_seq=timestamp_seq,
                **self.metadata.kwargs(),
            )
        return self.projector.demote(
            hypothesis,
            timestamp_seq=timestamp_seq,
            **self.metadata.kwargs(),
        )

    def report(self) -> CandidateLearningReport:
        """按已登记 Hypothesis 去重派生当前训练上下文的全链计数。"""
        evidence_count = 0
        decisions: set[int] = set()
        projection_events = 0
        active_count = 0
        for hypothesis in sorted(
                self._hypotheses, key=HypothesisKey.stable_key):
            evidence_count += len(self.engine.ledger.evidence_history(hypothesis))
            decisions.update(
                item.decision_id
                for item in self.engine.resolver.decision_history(hypothesis))
            definition = self.engine.definition(hypothesis)
            candidate = self.graph.ontology.resolve(definition.candidate)
            if candidate is None:
                continue
            history = self.graph.history(candidate)
            projection_events += len(history)
            if (history
                    and self.graph.project(candidate).state
                    == self.graph.protocol.active_state):
                active_count += 1
        return CandidateLearningReport(
            len(self._hypotheses),
            len(self._predictions),
            evidence_count,
            len(decisions),
            projection_events,
            active_count,
        )

    def hypothesis_for_candidate(
            self, candidate: ObjectIdentity) -> HypothesisKey:
        """按完整一等候选身份返回本上下文 owner 中已登记的 Hypothesis。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        hypothesis = self._candidate_hypotheses.get(candidate)
        if hypothesis is None:
            candidate_ref = self.graph.ontology.resolve(candidate)
            if (candidate_ref is not None
                    and self.graph.history(candidate_ref)):
                raise CandidateHistoryUnavailableError(
                    "候选只有图内恢复投影，M-03 前不得追加新 Evidence")
            raise KeyError("候选对象未在当前 H-05 owner 中登记")
        return hypothesis

    def read_only_projection(
            self, definition: EvidenceCandidateDefinition,
            ) -> CandidateGraphProjection:
        """核验图恢复定义并返回只读投影，不重建或伪造 H-00 历史。"""
        materialized = self.read_only_definition(definition)
        projection = self.graph.project(materialized.candidate)
        return projection

    def read_only_definition(
            self, definition: EvidenceCandidateDefinition,
            ) -> MaterializedCandidateDefinition:
        """核验恢复的 forming 定义，不要求候选已经产生 lifecycle Event。"""
        if not isinstance(definition, EvidenceCandidateDefinition):
            raise TypeError("definition 必须是 EvidenceCandidateDefinition")
        probe = self.engine.clone()
        hypothesis = probe.register(definition)
        materialized = self.graph.read_definition(hypothesis)
        if materialized.definition != definition:
            raise RuntimeError("恢复候选图定义与课程 mapper 输出不一致")
        return materialized

    def projection_for_candidate(
            self, candidate: ObjectIdentity,
            ) -> CandidateGraphProjection:
        """按完整候选身份只读恢复当前图投影，缺定义或生命周期时 fail closed。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        candidate_ref = self.graph.ontology.resolve(candidate)
        if candidate_ref is None:
            raise KeyError("候选尚未写入投影图")
        projection = self.graph.project(candidate_ref)
        if projection.candidate.definition.candidate != candidate:
            raise RuntimeError("候选图投影身份不一致")
        return projection

    def lifecycle_projection_if_available(
            self, candidate: ObjectIdentity,
            ) -> CandidateGraphProjection | None:
        """只读恢复 lifecycle；仅有 forming 定义时返回 None，损坏历史继续抛错。"""
        if not isinstance(candidate, ObjectIdentity):
            raise TypeError("candidate 必须是 ObjectIdentity")
        candidate_ref = self.graph.ontology.resolve(candidate)
        if candidate_ref is None:
            raise KeyError("候选尚未写入投影图")
        if not self.graph.history(candidate_ref):
            return None
        return self.projection_for_candidate(candidate)

    def next_timestamps(self, count: int) -> tuple[int, ...]:
        """预览下一段严格递增逻辑序；只有成功写入才推进运行时水位。"""
        assert_int(count, _where="CandidateLearningRuntime.next_timestamps")
        if type(count) is not int or count <= 0:
            raise ValueError("timestamp count 必须为严格正整数")
        start = self._logical_clock + 1
        return tuple(range(start, start + count))

    def state_key(self) -> tuple:
        """返回候选 owner、已编排 Hypothesis 和 prediction 的完整隔离状态。"""
        return (
            self.engine.state_key(),
            tuple(sorted(
                item.stable_key() for item in self._hypotheses)),
            tuple(sorted(
                (candidate.stable_key(), hypothesis.stable_key())
                for candidate, hypothesis
                in self._candidate_hypotheses.items())),
            tuple(sorted(self._predictions)),
            self._logical_clock,
        )

    def clone_for_graph(
            self, graph: CandidateProjectionGraph) -> "CandidateLearningRuntime":
        """复制 H-00/H-04 owner 并绑定隔离图，供 held-out 上下文写隔离。"""
        cloned = CandidateLearningRuntime(
            self.engine.clone(),
            graph,
            self.verifier,
            self.metadata,
        )
        cloned._hypotheses = set(self._hypotheses)
        cloned._candidate_hypotheses = dict(self._candidate_hypotheses)
        cloned._predictions = set(self._predictions)
        cloned._logical_clock = self._logical_clock
        return cloned


__all__ = [
    "CandidateHistoryUnavailableError",
    "CandidateLearningOutcome",
    "CandidateLearningReport",
    "CandidateLearningRuntime",
    "CandidateProjectionMetadata",
]
