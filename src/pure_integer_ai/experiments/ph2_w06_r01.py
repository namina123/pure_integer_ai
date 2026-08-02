"""W06-R01 的薄组装入口与确定性有界报告。"""
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
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r01_generation import (
    W06R01GenerationRuntime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w06_r01_reasoning import (
    W06R01ReasoningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r01_shared import (
    W06R01View,
    slice_w06_r01_adapter,
    w06_r01_alias_protocol,
    w06_r01_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r01_understanding import (
    W06R01UnderstandingRuntime,
)


@dataclass(frozen=True)
class W06R01RuntimeReport:
    """R01 当前 relation truth、三向消费和资源的确定性摘要。"""

    relation_digest: tuple[int, ...]
    source_evidence_digest: tuple[int, ...]
    active_projection_digest: tuple[int, ...]
    candidate_count: int
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
    explored_state_count: int
    considered_fact_count: int
    route_count: int
    private_read_count: int = 0
    formal_guard_read_count: int = 0
    future_relation_claim_count: int = 0
    w06_started: int = 0


class W06R01Runtime:
    """组合共享 view 与三个方向 runtime，不承载各方向算法。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            *,
            protocol: W06R01ConsumerProtocol = W06R01ConsumerProtocol(),
            ) -> None:
        self.view = W06R01View(learning, adapter, protocol)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = self.view.candidates
        self.understanding = W06R01UnderstandingRuntime(self.view)
        self.reasoning = W06R01ReasoningRuntime(self.view)
        self.generation = W06R01GenerationRuntime(
            self.view, self.understanding)

    @property
    def understanding_resolutions(self):
        """兼容返回 Understanding resolution 历史。"""
        return self.understanding.resolutions

    @property
    def reasoning_resolutions(self):
        """兼容返回 Reasoning resolution 历史。"""
        return self.reasoning.resolutions

    @property
    def generation_choices(self):
        """兼容返回 Generation choice 历史。"""
        return self.generation.choices

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
        """转发一次有状态 typed relation adjudication。"""
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        """转发一次 SUPPORTED Reasoning 采用。"""
        return self.reasoning.adopt(resolution)

    def verify_reasoning(self, use):
        """转发一次 Reasoning exact Use 重验。"""
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        """转发一次 relation-structure generation choice。"""
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        """转发一次 generation option 采用。"""
        return self.generation.adopt(choice, option_key)

    def verify_generation(self, use):
        """转发一次独立 Generation postcheck。"""
        return self.generation.verify(use)

    def state_key(self) -> tuple:
        """返回 learning truth 与三方向 runtime 的完整有界状态。"""
        assert self.learning.closure is not None
        return (
            self.learning.closure.state_key(),
            self.protocol.stable_key(),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W06R01RuntimeReport:
        """汇总 R01 truth、来源/Evidence、active projection 与资源实际值。"""
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
        discoveries = tuple(
            item.proposal.discovery
            for item in self.understanding.resolutions
            if item.proposal is not None
        )
        return W06R01RuntimeReport(
            digest_value(relation_value),
            digest_value(evidence_value),
            digest_value(active_value),
            len(snapshots),
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
            sum(item.explored_states for item in discoveries),
            sum(len(item.considered_facts) for item in discoveries),
            sum(len(item.routes) for item in discoveries),
        )


__all__ = [
    "W06R01Runtime",
    "W06R01RuntimeReport",
    "generation_request_for_candidate",
    "slice_w06_r01_adapter",
    "w06_r01_alias_protocol",
    "w06_r01_language_branch",
]
