"""W06-R02 的薄组装入口与确定性有界报告。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w06_adapter import W06TypedAdapterOutput
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r02_contract import (
    W06R02ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r02_generation import (
    W06R02GenerationRuntime,
    generation_request_for_candidate,
    query_for_candidate,
)
from pure_integer_ai.experiments.ph2_w06_r02_query import W06R02QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r02_shared import (
    W06R02View,
    slice_w06_r02_adapter,
    w06_r02_set_protocol,
)
from pure_integer_ai.experiments.set_relation_runtime import (
    SetRelationEndpointResolver,
)


@dataclass(frozen=True)
class W06R02RuntimeReport:
    """R02 当前 truth、集合 proof、三向消费和资源摘要。"""

    relation_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    active_projection_digest: tuple[int, ...]
    candidate_count: int
    rejection_count: int
    active_count: int
    refuted_count: int
    conflict_count: int
    superseded_count: int
    understanding_resolution_count: int
    understanding_use_count: int
    reasoning_resolution_count: int
    reasoning_use_count: int
    generation_choice_count: int
    generation_use_count: int
    generation_outcome_count: int
    derived_query_count: int
    consumed_premise_count: int
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_relation_claim_count: int = 0
    w06_started: int = 0


class W06R02Runtime:
    """组合共享 view、独立 U/R 历史和 Generation runtime。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            endpoint_resolver: SetRelationEndpointResolver,
            *,
            protocol: W06R02ConsumerProtocol = W06R02ConsumerProtocol(),
            ) -> None:
        self.view = W06R02View(
            learning, adapter, protocol, endpoint_resolver)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = self.view.candidates
        self.understanding = W06R02QueryRuntime(
            self.view, "UNDERSTANDING")
        self.reasoning = W06R02QueryRuntime(self.view, "REASONING")
        self.generation = W06R02GenerationRuntime(
            self.view, self.understanding)

    def resolve_understanding(self, request):
        return self.understanding.resolve(request)

    def adopt_understanding(self, resolution):
        return self.understanding.adopt(resolution)

    def verify_understanding(self, use):
        return self.understanding.verify(use)

    def resolve_reasoning(self, request):
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        return self.reasoning.adopt(resolution)

    def verify_reasoning(self, use):
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        return self.generation.adopt(choice, option_key)

    def verify_generation(self, use):
        return self.generation.verify(use)

    def state_key(self) -> tuple:
        assert self.learning.closure is not None
        return (
            self.learning.closure.state_key(),
            self.protocol.stable_key(),
            self.view.set_protocol.stable_key(),
            self.view.endpoint_resolver.state_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W06R02RuntimeReport:
        """汇总 R02 truth、来源/Evidence、proof Use 与资源实际值。"""
        snapshots = tuple(
            (item, self.learning.snapshot_for(item.proposition.proposition))
            for item in self.candidates
        )
        relation_value = [
            {
                "directionality": item.directionality,
                "proposition": list(item.proposition.proposition.stable_key()),
                "relation_family": item.relation_family,
                "schema": list(item.schema.schema.stable_key()),
            }
            for item, _snapshot in snapshots
        ]
        evidence_value = [
            {
                "evidence": [list(record.stable_key())
                             for record in snapshot.evidence],
                "proposition": list(item.proposition.proposition.stable_key()),
                "source": list(item.source_ref.stable_key()),
            }
            for item, snapshot in snapshots
        ]
        active_value = [
            {
                "decision": list(snapshot.active_fact.decision_key),
                "evidence": [list(key)
                             for key in snapshot.active_fact.evidence_keys],
                "proposition": list(item.proposition.proposition.stable_key()),
            }
            for item, snapshot in snapshots
            if snapshot.active_fact is not None
        ]
        query_resolutions = (
            *self.understanding.resolutions,
            *self.reasoning.resolutions,
        )
        query_uses = (*self.understanding.uses, *self.reasoning.uses)
        return W06R02RuntimeReport(
            digest_value(relation_value),
            digest_value(evidence_value),
            digest_value(active_value),
            len(snapshots),
            len(self.adapter.rejections),
            sum(snapshot.active_fact is not None for _item, snapshot in snapshots),
            sum(snapshot.snapshot.epistemic_status == EPISTEMIC_REFUTED
                for _item, snapshot in snapshots),
            sum(
                snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
                and snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
                for _item, snapshot in snapshots),
            sum(snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
                for _item, snapshot in snapshots),
            len(self.understanding.resolutions),
            len(self.understanding.uses),
            len(self.reasoning.resolutions),
            len(self.reasoning.uses),
            len(self.generation.choices),
            len(self.generation.uses),
            len(self.generation.outcomes),
            sum(len(item.propositions) > 1 for item in query_resolutions),
            sum(len(item.relation_uses) for item in (
                *query_uses, *self.generation.uses)),
        )


__all__ = [
    "W06R02Runtime",
    "W06R02RuntimeReport",
    "generation_request_for_candidate",
    "query_for_candidate",
    "slice_w06_r02_adapter",
    "w06_r02_set_protocol",
]
