"""W06-R07 的薄组装入口与确定性有界报告。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w06_adapter import W06TypedAdapterOutput
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r07_generation import (
    W06R07GenerationRuntime,
    generation_request_for_candidate,
    query_for_candidate,
)
from pure_integer_ai.experiments.ph2_w06_r07_query import W06R07QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r07_shared import (
    W06R07View,
    candidate_causal_protocol,
    slice_w06_r07_adapter,
)


@dataclass(frozen=True)
class W06R07RuntimeReport:
    """R07 truth、witness、三向消费和隔离资源摘要。"""

    relation_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    active_projection_digest: tuple[int, ...]
    candidate_count: int
    rejection_count: int
    active_count: int
    refuted_count: int
    unknown_count: int
    conflict_count: int
    superseded_count: int
    current_witness_count: int
    understanding_resolution_count: int
    understanding_use_count: int
    reasoning_resolution_count: int
    reasoning_use_count: int
    generation_choice_count: int
    generation_use_count: int
    generation_outcome_count: int
    consumed_premise_count: int
    effect_execution_count: int = 0
    event_time_fact_write_count: int = 0
    causal_implies_event_time_fact: int = 0
    precedence_implies_causation: int = 0
    temporal_support_sufficient: int = 0
    occurrence_order_consumed: int = 0
    structure_order_consumed: int = 0
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_relation_claim_count: int = 0
    w06_started: int = 0


class W06R07Runtime:
    """组合共享 direct CAUSES view、独立 U/R 历史与 Generation runtime。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            endpoint_resolver,
            *,
            protocol: W06R07ConsumerProtocol = W06R07ConsumerProtocol(),
            ) -> None:
        self.view = W06R07View(
            learning, adapter, protocol, endpoint_resolver)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = self.view.candidates
        self.understanding = W06R07QueryRuntime(
            self.view, "UNDERSTANDING")
        self.reasoning = W06R07QueryRuntime(self.view, "REASONING")
        self.generation = W06R07GenerationRuntime(
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

    def query_for_candidate(self, candidate, *, request_key, budget=None):
        kwargs = {"request_key": request_key}
        if budget is not None:
            kwargs["budget"] = budget
        return query_for_candidate(
            candidate, self.view.endpoint_resolver, **kwargs)

    def state_key(self) -> tuple:
        return (
            self.view.state_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W06R07RuntimeReport:
        snapshots = tuple(
            (item, self.learning.snapshot_for(item.proposition.proposition))
            for item in self.candidates
        )
        relation_value = []
        evidence_value = []
        active_value = []
        witness_count = 0
        for item, snapshot in snapshots:
            endpoint_protocol = candidate_causal_protocol(item)
            witnesses = self.view.witness_accounts(item)
            witness_count += len(witnesses)
            relation_value.append({
                "canonical_endpoints": [
                    list(value.stable_key())
                    for value in self.view.endpoints_for(item)
                ],
                "directionality": item.directionality,
                "execution_instruction": list(
                    endpoint_protocol.execution_instruction.stable_key()),
                "proposition": list(item.proposition.proposition.stable_key()),
                "relation": list(endpoint_protocol.relation.stable_key()),
                "relation_family": item.relation_family,
                "schema": list(item.schema.schema.stable_key()),
            })
            evidence_value.append({
                "evidence": [list(record.stable_key())
                             for record in snapshot.evidence],
                "proposition": list(item.proposition.proposition.stable_key()),
                "source": list(item.source_ref.stable_key()),
                "witnesses": [list(value.stable_key()) for value in witnesses],
            })
            if snapshot.active_fact is not None:
                active_value.append({
                    "canonical_endpoints": [
                        list(value.stable_key())
                        for value in self.view.endpoints_for(item)
                    ],
                    "decision": list(snapshot.active_fact.decision_key),
                    "evidence": [list(key)
                                 for key in snapshot.active_fact.evidence_keys],
                    "proposition": list(item.proposition.proposition.stable_key()),
                    "witnesses": [list(value.stable_key())
                                  for value in witnesses],
                })
        query_uses = (*self.understanding.uses, *self.reasoning.uses)
        return W06R07RuntimeReport(
            digest_value(relation_value),
            digest_value(evidence_value),
            digest_value(active_value),
            len(snapshots),
            len(self.adapter.rejections),
            sum(snapshot.active_fact is not None
                for _item, snapshot in snapshots),
            sum(snapshot.snapshot.epistemic_status == EPISTEMIC_REFUTED
                for _item, snapshot in snapshots),
            sum(snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
                for _item, snapshot in snapshots),
            sum(
                snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
                and snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
                for _item, snapshot in snapshots),
            sum(snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
                for _item, snapshot in snapshots),
            witness_count,
            len(self.understanding.resolutions),
            len(self.understanding.uses),
            len(self.reasoning.resolutions),
            len(self.reasoning.uses),
            len(self.generation.choices),
            len(self.generation.uses),
            len(self.generation.outcomes),
            sum(len(item.relation_uses) for item in (
                *query_uses, *self.generation.uses)),
        )


__all__ = [
    "W06R07Runtime",
    "W06R07RuntimeReport",
    "generation_request_for_candidate",
    "query_for_candidate",
    "slice_w06_r07_adapter",
]
