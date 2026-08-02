"""W06-R03 的薄组装入口与确定性有界报告。"""
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
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r03_generation import (
    W06R03GenerationRuntime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w06_r03_reasoning import (
    W06R03ReasoningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    W06R03View,
    slice_w06_r03_adapter,
    w06_r03_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r03_understanding import (
    W06R03UnderstandingRuntime,
)


@dataclass(frozen=True)
class W06R03RuntimeReport:
    """R03 当前 PROPERTY truth、三向消费和资源的确定性摘要。"""

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
    consumed_premise_count: int
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_relation_claim_count: int = 0
    w06_started: int = 0


class W06R03Runtime:
    """组合共享 PROPERTY view 与 U/R/G 三个独立 consumer。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            *,
            protocol: W06R03ConsumerProtocol = W06R03ConsumerProtocol(),
            ) -> None:
        self.view = W06R03View(learning, adapter, protocol)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = self.view.candidates
        self.understanding = W06R03UnderstandingRuntime(self.view)
        self.reasoning = W06R03ReasoningRuntime(self.view)
        self.generation = W06R03GenerationRuntime(
            self.view, self.reasoning)

    def resolve_understanding(self, request):
        """转发一次有状态 Understanding 查询。"""
        return self.understanding.resolve(request)

    def adopt_understanding(self, resolution):
        """转发一次 UNIQUE Understanding 采用。"""
        return self.understanding.adopt(resolution)

    def verify_understanding(self, use):
        """转发一次 Understanding exact Use 重验。"""
        return self.understanding.verify(use)

    def resolve_reasoning(self, request):
        """转发一次有状态 PROPERTY claim 裁决。"""
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        """转发一次 SUPPORTED Reasoning 采用。"""
        return self.reasoning.adopt(resolution)

    def verify_reasoning(self, use):
        """转发一次 Reasoning exact Use 重验。"""
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        """转发一次 PROPERTY-structure generation choice。"""
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        """转发一次 generation option 采用。"""
        return self.generation.adopt(choice, option_key)

    def verify_generation(self, use):
        """转发一次独立 Generation postcheck。"""
        return self.generation.verify(use)

    def state_key(self) -> tuple:
        """返回 learning truth、协议、resolver 和三方向 runtime 状态。"""
        assert self.learning.closure is not None
        return (
            self.learning.closure.state_key(),
            self.protocol.stable_key(),
            self.view.property_protocol.stable_key(),
            self.view.intensity_resolver.state_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W06R03RuntimeReport:
        """汇总 R03 truth、来源/Evidence、active projection 与资源实际值。"""
        snapshots = tuple(
            (item, self.learning.snapshot_for(item.proposition.proposition))
            for item in self.candidates
        )
        relation_value = [
            {
                "claim": list(self.view.claim_for(item).stable_key()),
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
                "claim": list(self.view.claim_for(item).stable_key()),
                "decision": list(snapshot.active_fact.decision_key),
                "evidence": [list(key)
                             for key in snapshot.active_fact.evidence_keys],
                "proposition": list(item.proposition.proposition.stable_key()),
            }
            for item, snapshot in snapshots
            if snapshot.active_fact is not None
        ]
        relation_uses = (
            *(item.relation_uses for item in self.understanding.uses),
            *(item.relation_uses for item in self.reasoning.uses),
            *(item.relation_uses for item in self.generation.uses),
        )
        return W06R03RuntimeReport(
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
            sum(len(items) for items in relation_uses),
        )


__all__ = [
    "W06R03Runtime",
    "W06R03RuntimeReport",
    "generation_request_for_candidate",
    "slice_w06_r03_adapter",
    "w06_r03_language_branch",
]
