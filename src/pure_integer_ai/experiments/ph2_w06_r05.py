"""W06-R05 的薄组装入口与确定性有界报告。"""
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
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w06_r05_generation import (
    W06R05GenerationRuntime,
    generation_request_for_candidate,
    query_for_candidate,
)
from pure_integer_ai.experiments.ph2_w06_r05_query import W06R05QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r05_shared import (
    W06R05View,
    slice_w06_r05_adapter,
    w06_r05_symmetric_protocols,
)


@dataclass(frozen=True)
class W06R05RuntimeReport:
    """R05 当前 truth、双 channel、三向消费和资源摘要。"""

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


class W06R05Runtime:
    """组合共享 view、独立 U/R 历史和 Generation runtime。"""

    def __init__(
            self,
            learning: W06RelationLearningRuntime,
            adapter: W06TypedAdapterOutput,
            *,
            protocol: W06R05ConsumerProtocol = W06R05ConsumerProtocol(),
            ) -> None:
        """绑定 R05 learning、双 channel protocol 和三向 consumer。"""
        self.view = W06R05View(learning, adapter, protocol)
        self.learning = learning
        self.adapter = adapter
        self.protocol = protocol
        self.candidates = self.view.candidates
        self.understanding = W06R05QueryRuntime(
            self.view, "UNDERSTANDING")
        self.reasoning = W06R05QueryRuntime(self.view, "REASONING")
        self.generation = W06R05GenerationRuntime(
            self.view, self.understanding)

    def resolve_understanding(self, request):
        """转发一次有状态 Understanding 查询。"""
        return self.understanding.resolve(request)

    def adopt_understanding(self, resolution):
        """转发一次 Understanding SUPPORTED 采用。"""
        return self.understanding.adopt(resolution)

    def verify_understanding(self, use):
        """转发一次 Understanding exact Use 重验。"""
        return self.understanding.verify(use)

    def resolve_reasoning(self, request):
        """转发一次有状态 Reasoning 查询。"""
        return self.reasoning.resolve(request)

    def adopt_reasoning(self, resolution):
        """转发一次 Reasoning SUPPORTED 采用。"""
        return self.reasoning.adopt(resolution)

    def verify_reasoning(self, use):
        """转发一次 Reasoning exact Use 重验。"""
        return self.reasoning.verify(use)

    def choose_generation(self, request):
        """转发一次 pair/channel-structure generation choice。"""
        return self.generation.choose(request)

    def adopt_generation(self, choice, option_key):
        """转发一次 generation option 采用。"""
        return self.generation.adopt(choice, option_key)

    def verify_generation(self, use):
        """转发一次独立 Generation postcheck。"""
        return self.generation.verify(use)

    def state_key(self) -> tuple:
        """返回 learning truth、协议和三方向 runtime 状态。"""
        assert self.learning.closure is not None
        return (
            self.learning.closure.state_key(),
            self.protocol.stable_key(),
            tuple(item.stable_key()
                  for item in w06_r05_symmetric_protocols(self.adapter)),
            self.understanding.state_key(),
            self.reasoning.state_key(),
            self.generation.state_key(),
        )

    def report(self) -> W06R05RuntimeReport:
        """汇总 R05 truth、来源/Evidence、Use 与资源实际值。"""
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
        query_uses = (*self.understanding.uses, *self.reasoning.uses)
        return W06R05RuntimeReport(
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
            sum(len(item.relation_uses) for item in (
                *query_uses, *self.generation.uses)),
        )


__all__ = [
    "W06R05Runtime",
    "W06R05RuntimeReport",
    "generation_request_for_candidate",
    "query_for_candidate",
    "slice_w06_r05_adapter",
    "w06_r05_symmetric_protocols",
]
