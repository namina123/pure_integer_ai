"""为 grounded reference 双句 compilation 安装来源化 H-06/S-07 顺序。"""
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
    GroundedAnswerOrderRequirement,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
    GroundedAnswerReferenceSentenceCompilation,
)


_NAMESPACE = 20980


# object-model: exception
class GroundedAnswerReferenceOrderError(ValueError):
    """双句 compilation 不能形成完整 active S-07 课程。"""


def _theory_id(compilation: GroundedAnswerReferenceCompilation) -> int:
    """从 connector 理论与 strategy 形成稳定课程身份。"""
    fingerprint = integer_tuple_fingerprint(
        compilation.connector.registry.stable_key(),
        domain="grounded.answer.reference.order.theory.v1",
    )
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _identity(
        compilation: GroundedAnswerReferenceCompilation,
        key: tuple[int, ...],
        *, instruction: bool = False,
        ) -> ObjectIdentity:
    """在目标 LanguageBranch owner/version 内建立课程身份。"""
    branch = compilation.connector.registry.templates[0].language_branch
    factory = minimal_instruction_identity if instruction else concept_identity
    return factory(key, owner=branch.owner, versions=branch.versions)


def _learning_protocol(
        compilation: GroundedAnswerReferenceCompilation,
        ) -> OrderLearningProtocol:
    """建立 reference compilation 独占的 H-06 aggregate owner。"""
    branch = compilation.connector.registry.templates[0].language_branch
    theory = _theory_id(compilation)
    source = SourceRef(
        _NAMESPACE,
        theory,
        len(compilation.forming_teacher_keys),
        branch.owner,
        branch.versions,
    )
    prefix = (_NAMESPACE, 1, theory)
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
        compilation: GroundedAnswerReferenceCompilation,
        ) -> StructureOrderConsumerProtocol:
    """建立多 template 共享但 run-local 独占的消费失败分型。"""
    prefix = (_NAMESPACE, 2, _theory_id(compilation))
    return StructureOrderConsumerProtocol(*tuple(
        _identity(compilation, (*prefix, index), instruction=True)
        for index in range(1, 8)
    ))


# object-model: resolver; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceOrderResolver:
    """按 constraint identity 恢复每句相邻 part 的真实方向。"""

    requirements: tuple[GroundedAnswerOrderRequirement, ...]
    applicable_reason: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.requirements, tuple)
                or not self.requirements
                or any(not isinstance(item, GroundedAnswerOrderRequirement)
                       for item in self.requirements)):
            raise TypeError("reference order requirements 必须非空")
        if len({item.constraint for item in self.requirements}) != len(
                self.requirements):
            raise GroundedAnswerReferenceOrderError(
                "reference order constraint 重复")
        if (not isinstance(self.applicable_reason, ObjectIdentity)
                or self.applicable_reason.object_kind
                != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("reference order applicable reason 类型错误")

    def resolve(
            self,
            definition: StructureOrderConstraintDefinition,
            context: tuple[ObjectIdentity, ...],
            ) -> ResolvedStructureOrderConstraint:
        """精确恢复一个 active constraint，不按 slot key 猜方向。"""
        del context
        matches = tuple(
            item for item in self.requirements
            if item.constraint == definition.constraint)
        if len(matches) != 1:
            raise GroundedAnswerReferenceOrderError(
                "active constraint 不属于 reference compilation")
        requirement = matches[0]
        if {requirement.before_slot, requirement.after_slot} != {
                definition.first_slot, definition.second_slot}:
            raise GroundedAnswerReferenceOrderError(
                "reference requirement 与 constraint slot pair 漂移")
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
class GroundedAnswerReferenceOrderInstallation:
    """多句课程的 active constraints 与共享 execution planner。"""

    compilation: GroundedAnswerReferenceCompilation
    engine: OrderHypothesisEngine
    lifecycle: StructureOrderLifecycleGraph
    consumer: StructureOrderConsumer
    execution_planner: GenerationStructureExecutionPlanner
    promotions: tuple[StructureOrderPromotionResult, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        if not isinstance(
                self.compilation, GroundedAnswerReferenceCompilation):
            raise TypeError("reference order compilation 类型错误")
        if not isinstance(self.engine, OrderHypothesisEngine):
            raise TypeError("reference order engine 类型错误")
        if not isinstance(self.lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("reference order lifecycle 类型错误")
        if not isinstance(self.consumer, StructureOrderConsumer):
            raise TypeError("reference order consumer 类型错误")
        if not isinstance(
                self.execution_planner, GenerationStructureExecutionPlanner):
            raise TypeError("reference order execution planner 类型错误")
        requirement_count = sum(
            len(item.orders) for item in self.compilation.sentences)
        if len(self.promotions) != requirement_count:
            raise GroundedAnswerReferenceOrderError(
                "reference order 未逐约束晋升")
        expected_evidence = (
            requirement_count
            * len(self.compilation.forming_teacher_keys)
        )
        if self.evidence_count != expected_evidence:
            raise GroundedAnswerReferenceOrderError(
                "reference order Evidence 数量漂移")


def _pattern(
        compilation: GroundedAnswerReferenceCompilation,
        sentence: GroundedAnswerReferenceSentenceCompilation,
        requirement: GroundedAnswerOrderRequirement,
        order_kind: ObjectIdentity,
        constraint_kind: ObjectIdentity,
        ) -> OrderPattern:
    """为一个 sentence template 构造相邻 slot pair pattern。"""
    first, second = sorted(
        (requirement.before_slot, requirement.after_slot),
        key=ObjectIdentity.stable_key,
    )
    template = sentence.template
    return OrderPattern(
        template.language_branch,
        order_kind,
        template.proposition_structure,
        template.structure,
        first,
        second,
        constraint_kind,
        template.connector,
        (),
    )


def install_grounded_answer_reference_order(
        compilation: GroundedAnswerReferenceCompilation,
        lifecycle: StructureOrderLifecycleGraph,
        ) -> GroundedAnswerReferenceOrderInstallation:
    """从显式 forming teacher keys 晋升全部句内相邻顺序。"""
    if not isinstance(compilation, GroundedAnswerReferenceCompilation):
        raise TypeError("reference order compilation 类型错误")
    if not isinstance(lifecycle, StructureOrderLifecycleGraph):
        raise TypeError("reference order lifecycle 类型错误")
    requirements = tuple(
        requirement
        for sentence in compilation.sentences
        for requirement in sentence.orders
    )
    if not requirements:
        raise GroundedAnswerReferenceOrderError(
            "reference compilation 缺少句内顺序义务")
    learning = _learning_protocol(compilation)
    engine = OrderHypothesisEngine(learning)
    promoter = OrderConstraintPromoter(
        engine, lifecycle.order_graph, lifecycle)
    theory = _theory_id(compilation)
    prefix = (_NAMESPACE, 3, theory)
    order_kind = _identity(compilation, (*prefix, 1))
    constraint_kind = _identity(compilation, (*prefix, 2))
    modality = _identity(compilation, (*prefix, 3))
    if any(value.object_kind != OBJECT_CONCEPT for value in (
            order_kind, constraint_kind, modality)):
        raise GroundedAnswerReferenceOrderError(
            "reference order concept 类型漂移")
    promotions = []
    evidence_count = 0
    timestamp = 1
    requirement_index = 0
    for sentence_index, sentence in enumerate(
            compilation.sentences, start=1):
        for requirement in sentence.orders:
            requirement_index += 1
            pattern = _pattern(
                compilation,
                sentence,
                requirement,
                order_kind,
                constraint_kind,
            )
            before_is_first = requirement.before_slot == pattern.first_slot
            for teacher_index, teacher_key in enumerate(
                    compilation.forming_teacher_keys, start=1):
                occurrence_base = (
                    (requirement_index - 1)
                    * len(compilation.forming_teacher_keys)
                    + teacher_index
                ) * 2
                observation = OrderObservation(
                    learning.aggregate_source,
                    learning.aggregate_scope,
                    (*prefix, 10, sentence_index, requirement_index,
                     teacher_index, len(teacher_key), *teacher_key),
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
                    (theory, sentence_index, requirement_index,
                     teacher_index),
                )
                detail = (*prefix, 11, requirement_index, teacher_index)
                engine.accumulate(
                    pattern,
                    observation,
                    lambda _pattern, _observation, detail=detail: (
                        OrderAssessment(EVIDENCE_SUPPORT, detail)),
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
                StructureOrderPromotionPlan(
                    sentence.template.slots, definition),
                decision,
                timestamp_seq=timestamp,
                provenance_kind=_NAMESPACE,
                qualifiers=(theory, sentence_index, requirement_index),
            ))
            timestamp += 1
    resolver = GroundedAnswerReferenceOrderResolver(
        requirements,
        _identity(compilation, (*prefix, 4), instruction=True),
    )
    consumer = StructureOrderConsumer(
        lifecycle, resolver, _consumer_protocol(compilation))
    planner = GenerationStructureExecutionPlanner(lifecycle, consumer)
    return GroundedAnswerReferenceOrderInstallation(
        compilation,
        engine,
        lifecycle,
        consumer,
        planner,
        tuple(promotions),
        evidence_count,
    )


__all__ = [
    "GroundedAnswerReferenceOrderError",
    "GroundedAnswerReferenceOrderInstallation",
    "GroundedAnswerReferenceOrderResolver",
    "install_grounded_answer_reference_order",
]
