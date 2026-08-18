"""为 ``CONFLICT_SET`` connector 建立独立的 H-06/S-07 顺序课程。

每个句子只晋升一条由公开 typed course 明确给出的
``proposition-slot -> claim-slot`` 约束。课程使用显式 aggregate source/scope，
不读取 private label，也不复用旧 grounded-answer order owner。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderHypothesisEngine,
    OrderLearningProtocol,
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_APPLICABLE,
    ResolvedStructureOrderConstraint,
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
    StructureOrderSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlanner,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
    StructureOrderPromotionPlan,
    StructureOrderPromotionResult,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector import (
    ConflictSetConnectorCompilation,
    ConflictSetSentenceCompilation,
)


class ConflictSetOrderError(ValueError):
    """CONFLICT_SET 顺序课程无法形成或恢复 active constraint。"""


@dataclass(frozen=True, slots=True)
class ConflictSetOrderSemanticsResolver:
    """只按已编译 sentence requirement 恢复 silent-before-claim 方向。"""

    sentences: tuple[ConflictSetSentenceCompilation, ...]
    applicable_reason: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.sentences, tuple) or not self.sentences
                or any(not isinstance(item, ConflictSetSentenceCompilation)
                       for item in self.sentences)):
            raise TypeError("conflict order sentences 非空且类型正确")
        if (not isinstance(self.applicable_reason, ObjectIdentity)
                or self.applicable_reason.object_kind
                != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("conflict order applicable reason 类型错误")
        constraints = tuple(item.order_constraint for item in self.sentences)
        if len(set(constraints)) != len(constraints):
            raise ConflictSetOrderError("CONFLICT_SET order constraint 不得重复")

    def resolve(
            self,
            definition: StructureOrderConstraintDefinition,
            context: tuple[ObjectIdentity, ...],
            ) -> ResolvedStructureOrderConstraint:
        """拒绝从 slot 稳定排序猜方向，只接受 compiler 的显式方向。"""
        del context
        matches = tuple(
            item for item in self.sentences
            if item.order_constraint == definition.constraint
        )
        if len(matches) != 1:
            raise ConflictSetOrderError("order constraint 未唯一绑定 sentence")
        item = matches[0]
        if {definition.first_slot, definition.second_slot} != {
                item.proposition_slot, item.claim_slot}:
            raise ConflictSetOrderError("order constraint slot pair 漂移")
        return ResolvedStructureOrderConstraint(
            definition.constraint,
            ORDER_APPLICABLE,
            item.proposition_slot,
            item.claim_slot,
            True,
            False,
            0,
            0,
            None,
            self.applicable_reason,
        )


@dataclass(frozen=True, slots=True)
class ConflictSetOrderInstallation:
    """CONFLICT_SET 的 H-06 Evidence、active S-07 和执行 planner。"""

    compilation: ConflictSetConnectorCompilation
    lifecycle: StructureOrderLifecycleGraph
    engine: OrderHypothesisEngine
    execution_planner: GenerationStructureExecutionPlanner
    promotions: tuple[StructureOrderPromotionResult, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.compilation, ConflictSetConnectorCompilation):
            raise TypeError("conflict order compilation 类型错误")
        if not isinstance(self.lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("conflict order lifecycle 类型错误")
        if not isinstance(self.engine, OrderHypothesisEngine):
            raise TypeError("conflict order engine 类型错误")
        if not isinstance(
                self.execution_planner,
                GenerationStructureExecutionPlanner):
            raise TypeError("conflict order execution planner 类型错误")
        if (not isinstance(self.promotions, tuple)
                or len(self.promotions) != len(self.compilation.sentences)):
            raise ConflictSetOrderError("order promotions 未覆盖全部 sentence")
        if self.evidence_count != len(self.promotions):
            raise ConflictSetOrderError("order Evidence 数量漂移")


def _concept(
        branch: ObjectIdentity,
        key: tuple[int, ...],
        ) -> ObjectIdentity:
    """在 connector branch owner/version 内建立课程 Concept。"""
    return concept_identity(key, owner=branch.owner, versions=branch.versions)


def _structure(
        branch: ObjectIdentity,
        key: tuple[int, ...],
        ) -> ObjectIdentity:
    """在 connector branch owner/version 内建立课程 StructureConcept。"""
    return structure_concept_identity(
        key, owner=branch.owner, versions=branch.versions)


def _instruction(
        branch: ObjectIdentity,
        key: tuple[int, ...],
        ) -> ObjectIdentity:
    """在 connector branch owner/version 内建立课程 MinimalInstruction。"""
    return minimal_instruction_identity(
        key, owner=branch.owner, versions=branch.versions)


def install_conflict_set_order_course(
        compilation: ConflictSetConnectorCompilation,
        lifecycle: StructureOrderLifecycleGraph,
        aggregate_source: SourceRef,
        aggregate_scope: ScopeIdentity,
        namespace: tuple[int, ...],
        ) -> ConflictSetOrderInstallation:
    """将 compiler 的句级 order contract 晋升为 active S-07 runtime。"""
    if not isinstance(compilation, ConflictSetConnectorCompilation):
        raise TypeError("conflict order compilation 类型错误")
    if not isinstance(lifecycle, StructureOrderLifecycleGraph):
        raise TypeError("conflict order lifecycle 类型错误")
    if not isinstance(aggregate_source, SourceRef):
        raise TypeError("conflict order aggregate_source 类型错误")
    if not isinstance(aggregate_scope, ScopeIdentity):
        raise TypeError("conflict order aggregate_scope 类型错误")
    if aggregate_scope.source != aggregate_source:
        raise ConflictSetOrderError("aggregate scope/source 不一致")
    if (not isinstance(namespace, tuple) or not namespace
            or any(type(item) is not int for item in namespace)):
        raise ConflictSetOrderError("order namespace 必须是非空整数 tuple")

    branch = compilation.language_branch
    if (aggregate_source.owner != branch.owner
            or aggregate_source.versions != branch.versions):
        raise ConflictSetOrderError(
            "aggregate source 与 language branch owner/version 不一致")
    protocol = OrderLearningProtocol(
        (*namespace, 1, 1),
        (*namespace, 1, 2),
        (*namespace, 1, 3),
        (*namespace, 1, 4),
        (*namespace, 1, 5),
        aggregate_source,
        aggregate_scope,
    )
    engine = OrderHypothesisEngine(protocol)
    promoter = OrderConstraintPromoter(
        engine, lifecycle.order_graph, lifecycle)
    order_kind = _concept(branch, (*namespace, 2, 1))
    modality = _concept(branch, (*namespace, 2, 2))
    context = _concept(branch, (*namespace, 2, 3))
    structure_family = _structure(branch, (*namespace, 2, 4))
    resolver_reason = _instruction(branch, (*namespace, 2, 5))
    promotions = []
    timestamp = 1
    for index, sentence in enumerate(compilation.sentences, start=1):
        first_slot, second_slot = sorted(
            (sentence.proposition_slot, sentence.claim_slot),
            key=ObjectIdentity.stable_key,
        )
        constraint_kind = _concept(branch, (*namespace, 8, index))
        pattern = OrderPattern(
            branch,
            order_kind,
            structure_family,
            sentence.template.structure,
            first_slot,
            second_slot,
            constraint_kind,
            context,
            (),
        )
        proposition_is_first = sentence.proposition_slot == first_slot
        occurrence_start = index * 2
        observation = OrderObservation(
            aggregate_source,
            aggregate_scope,
            (*namespace, 3, index),
            branch,
            structure_family,
            sentence.template.structure,
            first_slot,
            second_slot,
            context,
            (),
            occurrence_identity(
                aggregate_source,
                start=occurrence_start,
                end=occurrence_start + 1,
                ordinal=0,
            ),
            occurrence_identity(
                aggregate_source,
                start=occurrence_start + 1,
                end=occurrence_start + 2,
                ordinal=0,
            ),
            0 if proposition_is_first else 1,
            1 if proposition_is_first else 0,
            (*namespace, 4, index),
        )
        engine.accumulate(
            pattern,
            observation,
            lambda _pattern, _observation: OrderAssessment(
                EVIDENCE_SUPPORT, (*namespace, 5, index)),
            timestamp_seq=timestamp,
        )
        timestamp += 1
        decision = engine.resolve(pattern, timestamp_seq=timestamp)
        timestamp += 1
        definition = StructureOrderConstraintDefinition(
            sentence.order_constraint,
            branch,
            structure_family,
            sentence.template.structure,
            first_slot,
            second_slot,
            order_kind,
            constraint_kind,
            modality,
            context,
            (),
            (),
            (),
            engine.hypothesis_for(pattern),
        )
        promotions.append(promoter.promote(
            StructureOrderPromotionPlan(sentence.template.slots, definition),
            decision,
            timestamp_seq=timestamp,
            provenance_kind=namespace[0],
            qualifiers=(*namespace, 6, index),
        ))
        timestamp += 1
    resolver = ConflictSetOrderSemanticsResolver(
        compilation.sentences, resolver_reason)
    consumer_protocol = StructureOrderConsumerProtocol(*tuple(
        _instruction(branch, (*namespace, 7, index))
        for index in range(1, 8)
    ))
    consumer = StructureOrderConsumer(lifecycle, resolver, consumer_protocol)
    planner = GenerationStructureExecutionPlanner(lifecycle, consumer)
    return ConflictSetOrderInstallation(
        compilation,
        lifecycle,
        engine,
        planner,
        tuple(promotions),
        len(promotions),
    )


__all__ = [
    "ConflictSetOrderError",
    "ConflictSetOrderInstallation",
    "ConflictSetOrderSemanticsResolver",
    "install_conflict_set_order_course",
]
