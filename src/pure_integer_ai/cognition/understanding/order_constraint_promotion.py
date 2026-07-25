"""把 H-06 顺序 Hypothesis 晋升为 S-07 一等结构约束。

本模块只核验 H-00 snapshot、已提交 H-04 ResolverDecision、H-06 完整模式映射和
append-only lifecycle。modality、constraint、Role、参数及语言含义全部由调用方计划
提供，晋升器不会按频次、名称或位置猜测。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.structure_order import (
    MaterializedStructureOrderConstraint,
    StructureOrderConstraintDefinition,
    StructureOrderGraph,
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    MaterializedStructureOrderLifecycleEvent,
    StructureOrderLifecycleGraph,
)


class OrderConstraintPromotionError(RuntimeError):
    """顺序 Hypothesis、resolver decision 或结构映射不能支持本次转换。"""


@dataclass(frozen=True)
class StructureOrderPromotionPlan:
    """调用方显式提供的 slot schema 和 constraint 图定义。"""

    slots: tuple[StructureSlotDefinition, ...]
    constraint: StructureOrderConstraintDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.slots, tuple) or not self.slots:
            raise ValueError("promotion plan slots 必须是非空 tuple")
        if any(not isinstance(item, StructureSlotDefinition)
               for item in self.slots):
            raise TypeError("promotion plan slots 类型错误")
        if not isinstance(self.constraint, StructureOrderConstraintDefinition):
            raise TypeError("promotion plan constraint 类型错误")


@dataclass(frozen=True)
class StructureOrderPromotionResult:
    """一次结构定义写入和对应 lifecycle Event 的联合结果。"""

    constraint: MaterializedStructureOrderConstraint
    lifecycle: MaterializedStructureOrderLifecycleEvent


class OrderConstraintPromoter:
    """连接 H-04/H-06 与 S-07 图 facade，但不承担语言规则映射。"""

    def __init__(
            self, engine: OrderHypothesisEngine,
            order_graph: StructureOrderGraph,
            lifecycle: StructureOrderLifecycleGraph) -> None:
        if not isinstance(engine, OrderHypothesisEngine):
            raise TypeError("engine 必须是 OrderHypothesisEngine")
        if not isinstance(order_graph, StructureOrderGraph):
            raise TypeError("order_graph 必须是 StructureOrderGraph")
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("lifecycle 必须是 StructureOrderLifecycleGraph")
        if lifecycle.order_graph is not order_graph:
            raise ValueError("lifecycle 与 promoter 必须共享 StructureOrderGraph")
        self.engine = engine
        self.order_graph = order_graph
        self.lifecycle = lifecycle

    def promote(
            self, plan: StructureOrderPromotionPlan,
            decision: ResolverDecision, *,
            timestamp_seq: int,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> StructureOrderPromotionResult:
        """只把 active+supported 且被已提交 H-04 决策采用的模式设为 active。"""
        if not isinstance(plan, StructureOrderPromotionPlan):
            raise TypeError("plan 必须是 StructureOrderPromotionPlan")
        definition = plan.constraint
        pattern = self.engine.pattern_for_hypothesis(definition.hypothesis)
        self._validate_mapping(pattern, plan)
        snapshot = self.engine.ledger.snapshot(definition.hypothesis)
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or snapshot.epistemic_status != EPISTEMIC_SUPPORTED):
            raise OrderConstraintPromotionError(
                "只有 active+supported 顺序 Hypothesis 可以晋升")
        self._validate_decision(
            definition.hypothesis,
            decision,
            require_adopted=True,
        )
        if timestamp_seq < decision.timestamp_seq:
            raise OrderConstraintPromotionError(
                "晋升逻辑序不得早于 resolver decision")
        evidence = self._active_evidence(definition.hypothesis)
        self._validate_event_time(evidence, timestamp_seq)

        materialized = self.order_graph.define_constraint(
            plan.slots,
            definition,
            scope=definition.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        projection = self.lifecycle.project(materialized.constraint)
        if projection.state == self.lifecycle.protocol.superseded_state:
            raise OrderConstraintPromotionError(
                "superseded constraint 不得再次晋升")
        if projection.state == self.lifecycle.protocol.active_state:
            if not projection.history:
                raise OrderConstraintPromotionError("active 投影缺少 lifecycle Event")
            return StructureOrderPromotionResult(
                materialized,
                projection.history[-1],
            )
        event = self.lifecycle.make_event(
            definition.constraint,
            event_kind=self.lifecycle.protocol.promotion_kind,
            from_state=self.lifecycle.protocol.inactive_state,
            to_state=self.lifecycle.protocol.active_state,
            hypothesis=definition.hypothesis,
            evidence_keys=tuple(item.stable_key() for item in evidence),
            decision_key=decision.stable_key(),
            timestamp_seq=timestamp_seq,
        )
        lifecycle = self.lifecycle.append(
            event,
            scope=definition.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return StructureOrderPromotionResult(materialized, lifecycle)

    def demote(
            self, constraint: ObjectIdentity,
            decision: ResolverDecision, *,
            timestamp_seq: int,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedStructureOrderLifecycleEvent:
        """在 Hypothesis 失去 supported 或 adopted 后追加 active→inactive Event。"""
        constraint_ref = self.order_graph.ontology.resolve(constraint)
        if constraint_ref is None:
            raise OrderConstraintPromotionError("constraint 尚未定义")
        projection = self.lifecycle.project(constraint_ref)
        if projection.state != self.lifecycle.protocol.active_state:
            raise OrderConstraintPromotionError("只有 active constraint 可以降级")
        definition = projection.constraint.definition
        self.engine.pattern_for_hypothesis(definition.hypothesis)
        snapshot = self.engine.ledger.snapshot(definition.hypothesis)
        adopted = self._validate_decision(
            definition.hypothesis,
            decision,
            require_adopted=None,
        )
        if (snapshot.lifecycle == LIFECYCLE_ACTIVE
                and snapshot.epistemic_status == EPISTEMIC_SUPPORTED
                and adopted):
            raise OrderConstraintPromotionError(
                "仍为 active+supported+adopted 的 constraint 不得降级")
        if timestamp_seq < decision.timestamp_seq:
            raise OrderConstraintPromotionError(
                "降级逻辑序不得早于 resolver decision")
        evidence = self._active_evidence(definition.hypothesis)
        self._validate_event_time(evidence, timestamp_seq)
        event = self.lifecycle.make_event(
            definition.constraint,
            event_kind=self.lifecycle.protocol.demotion_kind,
            from_state=self.lifecycle.protocol.active_state,
            to_state=self.lifecycle.protocol.inactive_state,
            hypothesis=definition.hypothesis,
            evidence_keys=tuple(item.stable_key() for item in evidence),
            decision_key=decision.stable_key(),
            timestamp_seq=timestamp_seq,
        )
        return self.lifecycle.append(
            event,
            scope=definition.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    def supersede(
            self, rejected_constraint: ObjectIdentity,
            replacement_constraint: ObjectIdentity,
            decision: ResolverDecision, *,
            timestamp_seq: int,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedStructureOrderLifecycleEvent:
        """核验 H-00 replacement 后追加 terminal supersede Event。"""
        rejected_ref = self.order_graph.ontology.resolve(rejected_constraint)
        replacement_ref = self.order_graph.ontology.resolve(
            replacement_constraint)
        if rejected_ref is None or replacement_ref is None:
            raise OrderConstraintPromotionError(
                "rejected 和 replacement constraint 必须先定义")
        rejected = self.lifecycle.project(rejected_ref)
        replacement = self.lifecycle.project(replacement_ref)
        if rejected.state != self.lifecycle.protocol.active_state:
            raise OrderConstraintPromotionError(
                "只有 active constraint 可以 supersede")
        if replacement.state != self.lifecycle.protocol.active_state:
            raise OrderConstraintPromotionError(
                "replacement constraint 必须已晋升 active")
        old_hypothesis = rejected.constraint.definition.hypothesis
        new_hypothesis = replacement.constraint.definition.hypothesis
        self.engine.pattern_for_hypothesis(old_hypothesis)
        self.engine.pattern_for_hypothesis(new_hypothesis)
        old_snapshot = self.engine.ledger.snapshot(old_hypothesis)
        if old_snapshot.lifecycle != LIFECYCLE_SUPERSEDED:
            raise OrderConstraintPromotionError(
                "H-00 rejected Hypothesis 尚未 superseded")
        transitions = self.engine.ledger.transition_history(old_hypothesis)
        matching = tuple(
            item for item in transitions
            if (item.to_state == LIFECYCLE_SUPERSEDED
                and item.replacement == new_hypothesis)
        )
        if len(matching) != 1:
            raise OrderConstraintPromotionError(
                "H-00 没有唯一指向 replacement 的 supersede 事件")
        self._validate_decision(
            old_hypothesis,
            decision,
            require_adopted=False,
        )
        self._validate_decision(
            new_hypothesis,
            decision,
            require_adopted=True,
        )
        if timestamp_seq < decision.timestamp_seq:
            raise OrderConstraintPromotionError(
                "supersede 逻辑序不得早于 resolver decision")
        evidence = self._active_evidence(old_hypothesis)
        self._validate_event_time(evidence, timestamp_seq)
        event = self.lifecycle.make_event(
            rejected.constraint.definition.constraint,
            event_kind=self.lifecycle.protocol.supersede_kind,
            from_state=self.lifecycle.protocol.active_state,
            to_state=self.lifecycle.protocol.superseded_state,
            hypothesis=old_hypothesis,
            evidence_keys=tuple(item.stable_key() for item in evidence),
            decision_key=decision.stable_key(),
            timestamp_seq=timestamp_seq,
            replacement=replacement.constraint.definition.constraint,
        )
        return self.lifecycle.append(
            event,
            scope=rejected.constraint.definition.scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    @staticmethod
    def _validate_mapping(
            pattern: OrderPattern,
            plan: StructureOrderPromotionPlan) -> None:
        """逐字段核验 S-07 定义没有替换 H-06 模式身份或 slot。"""
        definition = plan.constraint
        actual = (
            definition.language_branch,
            definition.order_kind,
            definition.structure_family,
            definition.structure,
            definition.first_slot,
            definition.second_slot,
            definition.constraint_kind,
            definition.context,
            tuple(sorted(
                definition.conditions,
                key=ObjectIdentity.stable_key,
            )),
        )
        expected = (
            pattern.language_branch,
            pattern.order_kind,
            pattern.structure_family,
            pattern.structure_candidate,
            pattern.first_slot,
            pattern.second_slot,
            pattern.constraint,
            pattern.context,
            tuple(sorted(
                pattern.conditions,
                key=ObjectIdentity.stable_key,
            )),
        )
        if actual != expected:
            raise OrderConstraintPromotionError(
                "promotion plan 与 H-06 OrderPattern 完整映射不一致")

    def _validate_decision(
            self, hypothesis: HypothesisKey,
            decision: ResolverDecision, *,
            require_adopted: bool | None) -> bool:
        """核验 decision 已提交、包含当前 snapshot，并按要求采用或排除候选。"""
        if not isinstance(decision, ResolverDecision):
            raise TypeError("decision 必须是 ResolverDecision")
        history = self.engine.resolver.decision_history(hypothesis)
        if not any(item == decision for item in history):
            raise OrderConstraintPromotionError(
                "resolver decision 未提交到当前 H-04 owner")
        try:
            trace = decision.candidate(hypothesis)
        except KeyError as exc:
            raise OrderConstraintPromotionError(
                "resolver decision 不包含目标 Hypothesis") from exc
        current = self.engine.ledger.snapshot(hypothesis)
        if trace.after != current:
            raise OrderConstraintPromotionError(
                "resolver decision 已被后续 Evidence 或 lifecycle 事件陈旧化")
        adopted = hypothesis in decision.adopted_hypotheses
        if require_adopted is True and not adopted:
            raise OrderConstraintPromotionError("Hypothesis 未被 resolver adopted")
        if require_adopted is False and adopted:
            raise OrderConstraintPromotionError(
                "rejected Hypothesis 仍被 resolver adopted")
        return adopted

    def _active_evidence(
            self, hypothesis: HypothesisKey) -> tuple[EvidenceRecord, ...]:
        """从 snapshot 的当前 Evidence id 恢复完整事件，排除已被替代旧证据。"""
        snapshot = self.engine.ledger.snapshot(hypothesis)
        active_ids = frozenset((
            *snapshot.support_evidence_ids,
            *snapshot.refute_evidence_ids,
            *snapshot.unknown_evidence_ids,
        ))
        evidence = tuple(
            item for item in self.engine.ledger.evidence_history(hypothesis)
            if item.evidence_id in active_ids
        )
        if not evidence or {item.evidence_id for item in evidence} != active_ids:
            raise OrderConstraintPromotionError(
                "无法完整恢复当前 active Evidence")
        return tuple(sorted(
            evidence, key=lambda item: (item.timestamp_seq, item.evidence_id)))

    @staticmethod
    def _validate_event_time(
            evidence: tuple[EvidenceRecord, ...], timestamp_seq: int) -> None:
        """禁止 lifecycle Event 的逻辑序早于任一当前 Evidence。"""
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        if any(item.timestamp_seq > timestamp_seq for item in evidence):
            raise OrderConstraintPromotionError(
                "lifecycle Event 逻辑序不得早于当前 Evidence")


__all__ = [
    "OrderConstraintPromoter",
    "OrderConstraintPromotionError",
    "StructureOrderPromotionPlan",
    "StructureOrderPromotionResult",
]
