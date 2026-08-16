"""把已学 grounded-answer part 顺序提升为 run-local S-07 约束。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlanner,
)
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderHypothesisEngine,
    OrderLearningProtocol,
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_APPLICABLE,
    ResolvedStructureOrderConstraint,
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
    StructureOrderPromotionPlan,
    StructureOrderPromotionResult,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorVariant,
    GroundedAnswerOrderRequirement,
)


_NAMESPACE = 20940


# object-model: exception
class GroundedAnswerOrderError(ValueError):
    """已学 pattern 不能形成来源化 H-06/S-07 顺序课程。"""


def _theory_id(variant: GroundedAnswerConnectorVariant) -> int:
    """从 connector 理论而非具体目标命题生成稳定课程身份。"""
    fingerprint = integer_tuple_fingerprint(
        variant.template.connector.stable_key(),
        domain="grounded.answer.order.theory.v1",
    )
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _identity(
        key: tuple[int, ...],
        variant: GroundedAnswerConnectorVariant,
        *,
        instruction: bool = False,
        ) -> ObjectIdentity:
    """在 connector owner/version 内建立课程专用概念或指令。"""
    branch = variant.template.language_branch
    factory = minimal_instruction_identity if instruction else concept_identity
    return factory(key, owner=branch.owner, versions=branch.versions)


def _learning_protocol(
        variant: GroundedAnswerConnectorVariant,
        ) -> OrderLearningProtocol:
    """为一个显式 pattern variant 建立独立聚合来源和 H-06 协议。"""
    branch = variant.template.language_branch
    theory_id = _theory_id(variant)
    prefix = (_NAMESPACE, 1, variant.option.pattern_id, theory_id)
    source = SourceRef(
        _NAMESPACE,
        theory_id,
        variant.option.pattern_id,
        branch.owner,
        branch.versions,
    )
    return OrderLearningProtocol(
        (*prefix, 1),
        (*prefix, 2),
        (*prefix, 3),
        (*prefix, 4),
        (*prefix, 5),
        source,
        document_scope(source),
    )


def _consumer_protocol(
        variant: GroundedAnswerConnectorVariant,
        ) -> StructureOrderConsumerProtocol:
    """建立当前 pattern 独占的 S-07 消费失败分型。"""
    prefix = (_NAMESPACE, 2, variant.option.pattern_id, _theory_id(variant))
    return StructureOrderConsumerProtocol(*tuple(
        _identity((*prefix, index), variant, instruction=True)
        for index in range(1, 8)
    ))


# object-model: resolver; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerOrderSemanticsResolver:
    """把当前 pattern 的 active constraint 恢复为相邻 part 必要顺序。"""

    requirements: tuple[GroundedAnswerOrderRequirement, ...]
    applicable_reason: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.requirements, tuple)
                or any(not isinstance(item, GroundedAnswerOrderRequirement)
                       for item in self.requirements)):
            raise TypeError("grounded order requirements 类型错误")
        if len({item.constraint for item in self.requirements}) != len(
                self.requirements):
            raise GroundedAnswerOrderError("grounded order constraint 重复")
        if (not isinstance(self.applicable_reason, ObjectIdentity)
                or self.applicable_reason.object_kind
                != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("grounded order applicable reason 类型错误")

    def resolve(
            self,
            definition: StructureOrderConstraintDefinition,
            context: tuple[ObjectIdentity, ...],
            ) -> ResolvedStructureOrderConstraint:
        """按 constraint identity 精确恢复方向，不从 slot 排序猜方向。"""
        if not isinstance(definition, StructureOrderConstraintDefinition):
            raise TypeError("grounded order definition 类型错误")
        if (not isinstance(context, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in context)):
            raise TypeError("grounded order context 类型错误")
        matches = tuple(
            item for item in self.requirements
            if item.constraint == definition.constraint)
        if len(matches) != 1:
            raise GroundedAnswerOrderError(
                "active S-07 constraint 不属于当前 grounded pattern")
        requirement = matches[0]
        if {requirement.before_slot, requirement.after_slot} != {
                definition.first_slot, definition.second_slot}:
            raise GroundedAnswerOrderError(
                "grounded order requirement 与 S-07 slot pair 漂移")
        return ResolvedStructureOrderConstraint(
            definition.constraint,
            ORDER_APPLICABLE,
            requirement.before_slot,
            requirement.after_slot,
            True,
            False,
            0,
            0,
            None,
            self.applicable_reason,
        )


# object-model: runtime-bundle; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerOrderInstallation:
    """一个 variant 的 H-06 Evidence、active S-07 与执行 planner。"""

    variant: GroundedAnswerConnectorVariant
    engine: OrderHypothesisEngine
    lifecycle: StructureOrderLifecycleGraph
    consumer: StructureOrderConsumer
    execution_planner: GenerationStructureExecutionPlanner
    promotions: tuple[StructureOrderPromotionResult, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.variant, GroundedAnswerConnectorVariant):
            raise TypeError("grounded order variant 类型错误")
        if not isinstance(self.engine, OrderHypothesisEngine):
            raise TypeError("grounded order engine 类型错误")
        if not isinstance(self.lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("grounded order lifecycle 类型错误")
        if not isinstance(self.consumer, StructureOrderConsumer):
            raise TypeError("grounded order consumer 类型错误")
        if not isinstance(
                self.execution_planner, GenerationStructureExecutionPlanner):
            raise TypeError("grounded order execution planner 类型错误")
        if (not isinstance(self.promotions, tuple)
                or any(not isinstance(item, StructureOrderPromotionResult)
                       for item in self.promotions)):
            raise TypeError("grounded order promotions 类型错误")
        if len(self.promotions) != len(self.variant.order_requirements):
            raise GroundedAnswerOrderError("grounded order 未逐约束晋升")
        expected_evidence = (
            len(self.variant.order_requirements)
            * len(self.variant.option.support_teacher_keys)
        )
        if type(self.evidence_count) is not int or self.evidence_count != (
                expected_evidence):
            raise GroundedAnswerOrderError("grounded order Evidence 数量漂移")


def _pattern(
        variant: GroundedAnswerConnectorVariant,
        requirement: GroundedAnswerOrderRequirement,
        order_kind: ObjectIdentity,
        constraint_kind: ObjectIdentity,
        ) -> OrderPattern:
    """规范化 slot pair；真实前后方向由 observation 和 resolver 保留。"""
    first, second = sorted(
        (requirement.before_slot, requirement.after_slot),
        key=ObjectIdentity.stable_key,
    )
    return OrderPattern(
        variant.template.language_branch,
        order_kind,
        variant.template.proposition_structure,
        variant.template.structure,
        first,
        second,
        constraint_kind,
        variant.template.connector,
        (),
    )


def install_grounded_answer_order_course(
        variant: GroundedAnswerConnectorVariant,
        lifecycle: StructureOrderLifecycleGraph,
        ) -> GroundedAnswerOrderInstallation:
    """从形成 teacher Evidence 重放相邻 part 顺序并晋升 active S-07。"""
    if not isinstance(variant, GroundedAnswerConnectorVariant):
        raise TypeError("grounded order variant 类型错误")
    if not isinstance(lifecycle, StructureOrderLifecycleGraph):
        raise TypeError("grounded order lifecycle 类型错误")
    if not variant.order_requirements:
        raise GroundedAnswerOrderError("单 part pattern 不需要 S-07 顺序课程")
    learning = _learning_protocol(variant)
    engine = OrderHypothesisEngine(learning)
    promoter = OrderConstraintPromoter(
        engine, lifecycle.order_graph, lifecycle)
    prefix = (_NAMESPACE, 3, variant.option.pattern_id, _theory_id(variant))
    order_kind = _identity((*prefix, 1), variant)
    constraint_kind = _identity((*prefix, 2), variant)
    modality = _identity((*prefix, 3), variant)
    for value in (order_kind, constraint_kind, modality):
        if value.object_kind != OBJECT_CONCEPT:
            raise GroundedAnswerOrderError("grounded order concept 类型漂移")

    promotions = []
    timestamp = 1
    evidence_count = 0
    for requirement_index, requirement in enumerate(
            variant.order_requirements, start=1):
        pattern = _pattern(
            variant, requirement, order_kind, constraint_kind)
        before_is_first = requirement.before_slot == pattern.first_slot
        for teacher_index, teacher_key in enumerate(
                variant.option.support_teacher_keys, start=1):
            occurrence_base = (
                (requirement_index - 1)
                * len(variant.option.support_teacher_keys)
                + teacher_index
            ) * 2
            observation = OrderObservation(
                learning.aggregate_source,
                learning.aggregate_scope,
                (*prefix, 10, requirement_index, teacher_index,
                 len(teacher_key), *teacher_key),
                pattern.language_branch,
                pattern.structure_family,
                pattern.structure_candidate,
                pattern.first_slot,
                pattern.second_slot,
                pattern.context,
                pattern.conditions,
                occurrence_identity(
                    learning.aggregate_source,
                    start=occurrence_base,
                    end=occurrence_base + 1,
                    ordinal=0,
                ),
                occurrence_identity(
                    learning.aggregate_source,
                    start=occurrence_base + 1,
                    end=occurrence_base + 2,
                    ordinal=0,
                ),
                0 if before_is_first else 1,
                1 if before_is_first else 0,
                (variant.option.pattern_id, requirement_index, teacher_index),
            )
            detail = (*prefix, 11, requirement_index, teacher_index)
            engine.accumulate(
                pattern,
                observation,
                lambda _pattern, _observation, detail=detail: OrderAssessment(
                    EVIDENCE_SUPPORT, detail),
                timestamp_seq=timestamp,
            )
            timestamp += 1
            evidence_count += 1
        decision = engine.resolve(pattern, timestamp_seq=timestamp)
        timestamp += 1
        definition = StructureOrderConstraintDefinition(
            requirement.constraint,
            pattern.language_branch,
            pattern.structure_family,
            pattern.structure_candidate,
            pattern.first_slot,
            pattern.second_slot,
            pattern.order_kind,
            pattern.constraint,
            modality,
            pattern.context,
            (),
            (),
            (),
            engine.hypothesis_for(pattern),
        )
        promotions.append(promoter.promote(
            StructureOrderPromotionPlan(variant.template.slots, definition),
            decision,
            timestamp_seq=timestamp,
            provenance_kind=_NAMESPACE,
            qualifiers=(variant.option.pattern_id, requirement_index),
        ))
        timestamp += 1

    resolver = GroundedAnswerOrderSemanticsResolver(
        variant.order_requirements,
        _identity((*prefix, 4), variant, instruction=True),
    )
    consumer = StructureOrderConsumer(
        lifecycle, resolver, _consumer_protocol(variant))
    planner = GenerationStructureExecutionPlanner(lifecycle, consumer)
    return GroundedAnswerOrderInstallation(
        variant,
        engine,
        lifecycle,
        consumer,
        planner,
        tuple(promotions),
        evidence_count,
    )


__all__ = [
    "GroundedAnswerOrderError",
    "GroundedAnswerOrderInstallation",
    "GroundedAnswerOrderSemanticsResolver",
    "install_grounded_answer_order_course",
]
