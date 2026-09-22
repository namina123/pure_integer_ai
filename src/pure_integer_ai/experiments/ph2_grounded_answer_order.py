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
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity, document_scope
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


@dataclass(frozen=True, slots=True)
class SourceGenerationOrderCourse:
    """来源化顺序；零宽成员必须是静默上下文或外部图 filler。"""

    template: object
    source: SourceRef
    scope: ScopeIdentity
    positions: tuple[tuple[int, int], ...]
    qualifiers: tuple[int, ...]
    provenance_kind: int
    silent_slots: tuple[ObjectIdentity, ...] = ()
    external_slots: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        """核验真实来源、完整槽序、静默成员及整数形成证据。"""
        from pure_integer_ai.experiments.language_generation_connector import (
            LanguageGenerationConnectorTemplate,
        )
        if not isinstance(self.template, LanguageGenerationConnectorTemplate):
            raise TypeError("生成顺序课程必须持有完整 connector")
        if not isinstance(self.source, SourceRef) or not isinstance(self.scope, ScopeIdentity):
            raise TypeError("生成顺序课程来源类型错误")
        if self.scope.owner != self.source.owner or self.scope.versions != self.source.versions:
            raise ValueError("生成顺序课程来源与作用域不同")
        if (type(self.provenance_kind) is not int or self.provenance_kind <= 0
                or type(self.qualifiers) is not tuple or not self.qualifiers
                or any(type(v) is not int for v in self.qualifiers)):
            raise ValueError("生成顺序课程必须保留纯整数证据")
        count = len(self.template.slots)
        ordered_slots = tuple(sorted(
            self.template.slots, key=lambda item: item.slot.components[-1]))
        all_slots = {item.slot for item in ordered_slots}
        for label, declared in (
                ("静默", self.silent_slots),
                ("外部图 filler", self.external_slots)):
            if (type(declared) is not tuple
                    or any(not isinstance(item, ObjectIdentity)
                           for item in declared)
                    or len(set(declared)) != len(declared)
                    or not set(declared) <= all_slots):
                raise ValueError(f"生成顺序课程{label}成员声明非法")
        if (set(self.silent_slots) & set(self.external_slots)
                or len(self.silent_slots) >= count):
            raise ValueError("生成顺序课程零宽成员类别冲突")
        if (type(self.positions) is not tuple or len(self.positions) != count
                or len(self.template.constraints) != count - 1
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(v) is not int for v in pair)
                       or pair[0] < 0 or pair[0] > pair[1] for pair in self.positions)):
            raise ValueError("生成顺序课程区间或约束未完整覆盖成员")
        zero_width = set(self.silent_slots) | set(self.external_slots)
        if any(
                ((slot.slot in zero_width) != (position[0] == position[1]))
                for slot, position in zip(
                    ordered_slots, self.positions, strict=True)):
            raise ValueError(
                "只有显式静默上下文或外部图 filler 可使用零宽来源坐标")
        if any(first[1] > second[0] for first, second in zip(self.positions, self.positions[1:])):
            raise ValueError("生成顺序课程成员互相覆盖")
        if [item.slot.components[-1] for item in ordered_slots] != list(range(1, count + 1)):
            raise ValueError("生成顺序课程缺少连续成员序")


def install_source_generation_order(course: SourceGenerationOrderCourse,
                                    lifecycle: StructureOrderLifecycleGraph) -> int:
    """复用 H-06/H-04/S-07 消费实际来源区间；不创建占位事实或缩减证据。"""
    if not isinstance(course, SourceGenerationOrderCourse):
        raise TypeError("来源化生成顺序课程类型错误")
    template = course.template
    original = course.source
    scope = course.scope
    prefix = (course.provenance_kind, 3, *template.connector.components)
    learning = OrderLearningProtocol(*(tuple((*prefix, index)) for index in range(1, 6)),
                                      original, scope)
    engine = OrderHypothesisEngine(learning)
    promoter = OrderConstraintPromoter(engine, lifecycle.order_graph, lifecycle)
    metadata = dict(owner=template.language_branch.owner, versions=template.language_branch.versions)
    order_kind, constraint_kind, modality = (concept_identity((*prefix, 10, index), **metadata)
                                             for index in range(1, 4))
    timestamp = 1
    count = 0
    slots = tuple(sorted(template.slots, key=lambda item: item.slot.components[-1]))
    constraints = tuple(sorted(template.constraints, key=lambda item: item.components[-1]))
    for index, (before, after, constraint) in enumerate(zip(
            slots, slots[1:], constraints), 1):
        first, second = sorted((before.slot, after.slot), key=ObjectIdentity.stable_key)
        before_position, after_position = course.positions[index - 1:index + 1]
        if before_position[1] > after_position[0]:
            raise GroundedAnswerOrderError("训练框架成员相互覆盖，不能晋升顺序")
        positions = {before.slot: before_position, after.slot: after_position}
        pattern = OrderPattern(template.language_branch, order_kind, template.proposition_structure,
                               template.structure, first, second, constraint_kind,
                               template.context_set)
        # 多个外部 graph filler 可以合法共享同一零宽 source span；
        # occurrence identity 仍须按槽序分开，避免把两个图角色误判为同一 occurrence。
        first_ordinal = index if positions[first] == positions[second] else 0
        second_ordinal = index + 1 if positions[first] == positions[second] else 0
        observation = OrderObservation(
            original, scope, (*prefix, 11, index), template.language_branch,
            template.proposition_structure, template.structure, first, second,
            template.context_set, (),
            occurrence_identity(original, start=positions[first][0], end=positions[first][1], ordinal=first_ordinal),
            occurrence_identity(original, start=positions[second][0], end=positions[second][1], ordinal=second_ordinal),
            positions[first][0], positions[second][0],
            (*course.qualifiers, index, *before_position, *after_position))
        engine.accumulate(pattern, observation, lambda _pattern, _observation: OrderAssessment(
            EVIDENCE_SUPPORT, observation.stable_key()), timestamp_seq=timestamp)
        timestamp += 1
        decision = engine.resolve(pattern, timestamp_seq=timestamp)
        timestamp += 1
        definition = StructureOrderConstraintDefinition(
            constraint, pattern.language_branch, pattern.structure_family,
            pattern.structure_candidate, first, second, order_kind, constraint_kind, modality,
            pattern.context, (), (), (), engine.hypothesis_for(pattern))
        promoter.promote(StructureOrderPromotionPlan(slots, definition), decision,
                         timestamp_seq=timestamp, provenance_kind=course.provenance_kind,
                         content_version=1, qualifiers=course.qualifiers)
        timestamp += 1
        count += 1
    return count


def install_relation_frame_order(course, lifecycle: StructureOrderLifecycleGraph) -> int:
    """保留既有关系课程合同及身份，委托同一个来源化顺序学习器。"""
    from pure_integer_ai.experiments.relation_generation_structure import (
        RELATION_CONNECTOR_PROFILE, RelationConnectorCourse,
    )
    if not isinstance(course, RelationConnectorCourse):
        raise TypeError("角色顺序课程类型错误")
    source = course.source.proposition.definition.source
    return install_source_generation_order(SourceGenerationOrderCourse(
        course.template, source, document_scope(source), course.positions,
        course.qualifiers, RELATION_CONNECTOR_PROFILE), lifecycle)


__all__ = [
    "GroundedAnswerOrderError",
    "GroundedAnswerOrderInstallation",
    "GroundedAnswerOrderSemanticsResolver",
    "install_grounded_answer_order_course",
    "install_relation_frame_order",
    "SourceGenerationOrderCourse",
    "install_source_generation_order",
]
