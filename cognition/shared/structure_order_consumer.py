"""S-07 active typed 结构顺序约束的解析和线性化消费者。

消费者只执行 resolver 已解释的适用三态、方向、必要性、偏好权重和整数间距界。
具体语言、modality、constraint、condition、exception 与参数意义均不在本模块写死。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderProjection,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


ORDER_APPLICABLE = 1
ORDER_NOT_APPLICABLE = 2
ORDER_APPLICABILITY_UNKNOWN = 3
_APPLICABILITY_STATES = frozenset({
    ORDER_APPLICABLE,
    ORDER_NOT_APPLICABLE,
    ORDER_APPLICABILITY_UNKNOWN,
})

ORDER_CONSUMER_ACCEPTED = 1
ORDER_CONSUMER_REJECTED = 2
ORDER_CONSUMER_UNKNOWN = 3
_CONSUMER_STATES = frozenset({
    ORDER_CONSUMER_ACCEPTED,
    ORDER_CONSUMER_REJECTED,
    ORDER_CONSUMER_UNKNOWN,
})


def _reason(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """校验 consumer reason 使用一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{where} 必须是 MinimalInstruction")
    return value


@dataclass(frozen=True)
class StructureOrderConsumerProtocol:
    """消费者通用失败分型使用的注入式 reason。"""

    invalid_assignment: ObjectIdentity
    applicability_unknown: ObjectIdentity
    missing_slot: ObjectIdentity
    constraint_cycle: ObjectIdentity
    constraint_conflict: ObjectIdentity
    constraint_violation: ObjectIdentity
    budget_exhausted: ObjectIdentity

    def __post_init__(self) -> None:
        reasons = (
            self.invalid_assignment,
            self.applicability_unknown,
            self.missing_slot,
            self.constraint_cycle,
            self.constraint_conflict,
            self.constraint_violation,
            self.budget_exhausted,
        )
        for index, value in enumerate(reasons):
            _reason(value, where=f"StructureOrderConsumerProtocol[{index}]")
        if len({item.stable_key() for item in reasons}) != len(reasons):
            raise ValueError("consumer reason 必须互不相同")


@dataclass(frozen=True)
class StructureOrderSearchBudget:
    """局部偏序搜索允许访问的前缀状态数。"""

    max_states: int

    def __post_init__(self) -> None:
        assert_int(self.max_states, _where="StructureOrderSearchBudget.max_states")
        if type(self.max_states) is not int or self.max_states <= 0:
            raise ValueError("max_states 必须为严格正整数")


@dataclass(frozen=True)
class ResolvedStructureOrderConstraint:
    """semantics resolver 为一次 query 产生的通用可执行约束。"""

    constraint: ObjectIdentity
    applicability: int
    before_slot: ObjectIdentity
    after_slot: ObjectIdentity
    enforced: bool
    allow_missing: bool
    preference_weight: int
    minimum_gap: int
    maximum_gap: int | None
    reason: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.constraint, ObjectIdentity)
                or self.constraint.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise ValueError("resolved constraint 必须是 StructureConcept")
        for name, value in (
                ("before_slot", self.before_slot),
                ("after_slot", self.after_slot)):
            if (not isinstance(value, ObjectIdentity)
                    or value.object_kind != OBJECT_STRUCTURE_CONCEPT):
                raise ValueError(f"{name} 必须是 StructureConcept")
        assert_int(
            self.applicability,
            self.preference_weight,
            self.minimum_gap,
            _where="ResolvedStructureOrderConstraint",
        )
        if self.applicability not in _APPLICABILITY_STATES:
            raise ValueError("applicability 未注册")
        if type(self.enforced) is not bool or type(self.allow_missing) is not bool:
            raise TypeError("enforced/allow_missing 必须是 bool")
        if type(self.preference_weight) is not int or self.preference_weight < 0:
            raise ValueError("preference_weight 必须为非负严格整数")
        if type(self.minimum_gap) is not int or self.minimum_gap < 0:
            raise ValueError("minimum_gap 必须为非负严格整数")
        if self.maximum_gap is not None:
            assert_int(self.maximum_gap, _where="maximum_gap")
            if (type(self.maximum_gap) is not int
                    or self.maximum_gap < self.minimum_gap):
                raise ValueError("maximum_gap 必须不小于 minimum_gap")
        if self.before_slot == self.after_slot:
            raise ValueError("resolved constraint 的两个 slot 必须不同")
        _reason(self.reason, where="ResolvedStructureOrderConstraint.reason")


class StructureOrderSemanticsResolver(Protocol):
    """把图内约束身份解释为一次 query 的通用控制参数。"""

    def resolve(
            self, definition: StructureOrderConstraintDefinition,
            context: tuple[ObjectIdentity, ...],
            ) -> ResolvedStructureOrderConstraint: ...


@dataclass(frozen=True)
class StructureSlotValue:
    """一个结构 slot 在本次解析或生成中的实际 filler。"""

    slot: ObjectIdentity
    filler: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.slot, ObjectIdentity)
                or self.slot.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise ValueError("slot value 的 slot 必须是 StructureConcept")
        if not isinstance(self.filler, ObjectIdentity):
            raise TypeError("slot value 的 filler 必须是 ObjectIdentity")


@dataclass(frozen=True)
class PositionedStructureSlotValue:
    """解析输入中的 slot filler 和局部整数位置。"""

    value: StructureSlotValue
    position: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, StructureSlotValue):
            raise TypeError("value 必须是 StructureSlotValue")
        assert_int(self.position, _where="PositionedStructureSlotValue.position")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("position 必须为非负严格整数")


@dataclass(frozen=True)
class StructureOrderConstraintEvaluation:
    """一个 resolved constraint 在给定 slot 序上的适用和满足结果。"""

    resolved: ResolvedStructureOrderConstraint
    satisfied: bool | None


@dataclass(frozen=True)
class StructureOrderParseResult:
    """typed parse 顺序验收结果，不产生 Proposition 真值。"""

    status: int
    assignments: tuple[PositionedStructureSlotValue, ...]
    evaluations: tuple[StructureOrderConstraintEvaluation, ...]
    reasons: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if self.status not in _CONSUMER_STATES:
            raise ValueError("parse result status 未注册")


@dataclass(frozen=True)
class StructureOrderLinearizationResult:
    """typed generation 的线性扩展、评估和预算使用结果。"""

    status: int
    values: tuple[StructureSlotValue, ...]
    evaluations: tuple[StructureOrderConstraintEvaluation, ...]
    reasons: tuple[ObjectIdentity, ...]
    explored_states: int

    def __post_init__(self) -> None:
        if self.status not in _CONSUMER_STATES:
            raise ValueError("linearization result status 未注册")
        assert_int(self.explored_states, _where="explored_states")
        if type(self.explored_states) is not int or self.explored_states < 0:
            raise ValueError("explored_states 必须为非负严格整数")


@dataclass(frozen=True)
class _SearchResult:
    """内部搜索结果，区分无解和预算未完成。"""

    order: tuple[ObjectIdentity, ...]
    explored_states: int
    exhausted: bool


class StructureOrderConsumer:
    """从 active S-07 projection 执行解析验证和生成线性化。"""

    def __init__(
            self, lifecycle: StructureOrderLifecycleGraph,
            resolver: StructureOrderSemanticsResolver,
            protocol: StructureOrderConsumerProtocol) -> None:
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("lifecycle 必须是 StructureOrderLifecycleGraph")
        if not callable(getattr(resolver, "resolve", None)):
            raise TypeError("resolver 必须实现 resolve")
        if not isinstance(protocol, StructureOrderConsumerProtocol):
            raise TypeError("protocol 必须是 StructureOrderConsumerProtocol")
        self.lifecycle = lifecycle
        self.resolver = resolver
        self.protocol = protocol

    def parse(
            self, structure: TypedRef,
            assignments: tuple[PositionedStructureSlotValue, ...], *,
            context: tuple[ObjectIdentity, ...],
            budget: StructureOrderSearchBudget,
            ) -> StructureOrderParseResult:
        """验证观测局部序；模型自身冲突返回 unknown，真实违反才 rejected。"""
        checked_context = self._validate_context(context)
        if not isinstance(budget, StructureOrderSearchBudget):
            raise TypeError("budget 必须是 StructureOrderSearchBudget")
        active = self.lifecycle.active_constraints(structure)
        resolved, unknown_reasons = self._resolve(active, checked_context)
        invalid = self._validate_positioned_assignments(structure, assignments)
        if invalid is not None:
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                assignments,
                (),
                (invalid,),
            )
        if unknown_reasons:
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                assignments,
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                unknown_reasons,
            )
        ordered_assignments = tuple(sorted(
            assignments, key=lambda item: item.position))
        slots = tuple(item.value.slot for item in ordered_assignments)
        missing = self._missing_required(slots, resolved)
        if missing:
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                ordered_assignments,
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.missing_slot,),
            )
        applicable = tuple(
            item for item in resolved
            if item.applicability == ORDER_APPLICABLE
        )
        if self._has_required_cycle(slots, applicable):
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                ordered_assignments,
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.constraint_cycle,),
            )
        search = self._search(
            slots, applicable, budget, optimize_preferences=False)
        if search.exhausted:
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                ordered_assignments,
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.budget_exhausted,),
            )
        if not search.order:
            return StructureOrderParseResult(
                ORDER_CONSUMER_UNKNOWN,
                ordered_assignments,
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.constraint_conflict,),
            )
        positions = {
            item.value.slot: item.position for item in ordered_assignments
        }
        evaluations = self._evaluate_positions(positions, resolved)
        violated = any(
            item.resolved.enforced and item.satisfied is False
            for item in evaluations
        )
        if violated:
            return StructureOrderParseResult(
                ORDER_CONSUMER_REJECTED,
                ordered_assignments,
                evaluations,
                (self.protocol.constraint_violation,),
            )
        return StructureOrderParseResult(
            ORDER_CONSUMER_ACCEPTED,
            ordered_assignments,
            evaluations,
            (),
        )

    def linearize(
            self, structure: TypedRef,
            values: tuple[StructureSlotValue, ...], *,
            context: tuple[ObjectIdentity, ...],
            budget: StructureOrderSearchBudget,
            ) -> StructureOrderLinearizationResult:
        """在调用方基序上寻找满足必要约束且偏好最优的局部线性扩展。"""
        checked_context = self._validate_context(context)
        if not isinstance(budget, StructureOrderSearchBudget):
            raise TypeError("budget 必须是 StructureOrderSearchBudget")
        active = self.lifecycle.active_constraints(structure)
        resolved, unknown_reasons = self._resolve(active, checked_context)
        invalid = self._validate_values(structure, values)
        if invalid is not None:
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN, (), (), (invalid,), 0)
        if unknown_reasons:
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN,
                (),
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                unknown_reasons,
                0,
            )
        slots = tuple(item.slot for item in values)
        if self._missing_required(slots, resolved):
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN,
                (),
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.missing_slot,),
                0,
            )
        applicable = tuple(
            item for item in resolved
            if item.applicability == ORDER_APPLICABLE
        )
        if self._has_required_cycle(slots, applicable):
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN,
                (),
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.constraint_cycle,),
                0,
            )
        search = self._search(
            slots, applicable, budget, optimize_preferences=True)
        if search.exhausted:
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN,
                (),
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.budget_exhausted,),
                search.explored_states,
            )
        if not search.order:
            return StructureOrderLinearizationResult(
                ORDER_CONSUMER_UNKNOWN,
                (),
                tuple(StructureOrderConstraintEvaluation(item, None)
                      for item in resolved),
                (self.protocol.constraint_conflict,),
                search.explored_states,
            )
        by_slot = {item.slot: item for item in values}
        ordered = tuple(by_slot[item] for item in search.order)
        return StructureOrderLinearizationResult(
            ORDER_CONSUMER_ACCEPTED,
            ordered,
            self._evaluate(search.order, resolved),
            (),
            search.explored_states,
        )

    def _resolve(
            self, projections: tuple[StructureOrderProjection, ...],
            context: tuple[ObjectIdentity, ...],
            ) -> tuple[
                tuple[ResolvedStructureOrderConstraint, ...],
                tuple[ObjectIdentity, ...],
                ]:
        """调用注入 resolver 并核验输出仍指向同一 constraint 和 slot pair。"""
        resolved: list[ResolvedStructureOrderConstraint] = []
        unknown_reasons: list[ObjectIdentity] = []
        for projection in projections:
            definition = projection.constraint.definition
            item = self.resolver.resolve(definition, context)
            if not isinstance(item, ResolvedStructureOrderConstraint):
                raise TypeError("semantics resolver 返回类型错误")
            if item.constraint != definition.constraint:
                raise ValueError("semantics resolver 替换了 constraint 身份")
            if {item.before_slot, item.after_slot} != {
                    definition.first_slot, definition.second_slot}:
                raise ValueError("semantics resolver 替换了 constraint slot pair")
            resolved.append(item)
            if item.applicability == ORDER_APPLICABILITY_UNKNOWN:
                unknown_reasons.extend((
                    self.protocol.applicability_unknown,
                    item.reason,
                ))
        return (
            tuple(resolved),
            tuple(dict.fromkeys(unknown_reasons)),
        )

    def _validate_positioned_assignments(
            self, structure: TypedRef,
            assignments: tuple[PositionedStructureSlotValue, ...],
            ) -> ObjectIdentity | None:
        """核验 parse slot/position 唯一且全部属于目标 StructureConcept。"""
        if not isinstance(assignments, tuple) or not assignments:
            return self.protocol.invalid_assignment
        if any(not isinstance(item, PositionedStructureSlotValue)
               for item in assignments):
            raise TypeError("assignments 类型错误")
        slots = tuple(item.value for item in assignments)
        invalid = self._validate_values(structure, slots)
        if invalid is not None:
            return invalid
        positions = tuple(item.position for item in assignments)
        if len(set(positions)) != len(positions):
            return self.protocol.invalid_assignment
        return None

    def _validate_values(
            self, structure: TypedRef,
            values: tuple[StructureSlotValue, ...],
            ) -> ObjectIdentity | None:
        """核验 slot value 唯一且引用结构当前图定义中的成员。"""
        if not isinstance(values, tuple) or not values:
            return self.protocol.invalid_assignment
        if any(not isinstance(item, StructureSlotValue) for item in values):
            raise TypeError("values 类型错误")
        slots = tuple(item.slot for item in values)
        if len(set(slots)) != len(slots):
            return self.protocol.invalid_assignment
        materialized = self.lifecycle.order_graph.read_structure(structure)
        known = {item.definition.slot for item in materialized.slots}
        if any(item not in known for item in slots):
            return self.protocol.invalid_assignment
        return None

    @staticmethod
    def _validate_context(
            context: tuple[ObjectIdentity, ...],
            ) -> tuple[ObjectIdentity, ...]:
        """校验显式 query context 是无重复的一等对象集合。"""
        if not isinstance(context, tuple):
            raise TypeError("context 必须是 ObjectIdentity tuple")
        if any(not isinstance(item, ObjectIdentity) for item in context):
            raise TypeError("context 必须由 ObjectIdentity 组成")
        if len(set(context)) != len(context):
            raise ValueError("context 不得重复")
        return context

    @staticmethod
    def _missing_required(
            slots: tuple[ObjectIdentity, ...],
            constraints: tuple[ResolvedStructureOrderConstraint, ...],
            ) -> bool:
        """判断 applicable enforced constraint 是否缺少不允许省略的端点。"""
        present = frozenset(slots)
        return any(
            item.applicability == ORDER_APPLICABLE
            and item.enforced
            and not item.allow_missing
            and (item.before_slot not in present or item.after_slot not in present)
            for item in constraints
        )

    @staticmethod
    def _has_required_cycle(
            slots: tuple[ObjectIdentity, ...],
            constraints: tuple[ResolvedStructureOrderConstraint, ...],
            ) -> bool:
        """仅沿当前存在端点的 enforced precedence 检查局部环。"""
        present = frozenset(slots)
        successors: dict[ObjectIdentity, set[ObjectIdentity]] = {
            item: set() for item in slots
        }
        for item in constraints:
            if (not item.enforced
                    or item.before_slot not in present
                    or item.after_slot not in present):
                continue
            successors[item.before_slot].add(item.after_slot)
        visiting: set[ObjectIdentity] = set()
        visited: set[ObjectIdentity] = set()

        def visit(slot: ObjectIdentity) -> bool:
            """深度检查当前 slot 是否回到递归路径。"""
            if slot in visiting:
                return True
            if slot in visited:
                return False
            visiting.add(slot)
            for target in successors[slot]:
                if visit(target):
                    return True
            visiting.remove(slot)
            visited.add(slot)
            return False

        return any(visit(slot) for slot in slots if slot not in visited)

    def _search(
            self, base_order: tuple[ObjectIdentity, ...],
            constraints: tuple[ResolvedStructureOrderConstraint, ...],
            budget: StructureOrderSearchBudget,
            *, optimize_preferences: bool,
            ) -> _SearchResult:
        """按调用方基序搜索可行扩展，并按调用目的决定是否证明偏好最优。"""
        if type(optimize_preferences) is not bool:
            raise TypeError("optimize_preferences 必须是 bool")
        present = frozenset(base_order)
        required = tuple(
            item for item in constraints
            if (item.enforced
                and item.before_slot in present
                and item.after_slot in present)
        )
        preferred = tuple(
            item for item in constraints
            if (optimize_preferences
                and item.preference_weight > 0
                and item.before_slot in present
                and item.after_slot in present)
        )
        predecessors: dict[ObjectIdentity, set[ObjectIdentity]] = {
            item: set() for item in base_order
        }
        for item in required:
            predecessors[item.after_slot].add(item.before_slot)
        explored = 0
        exhausted = False
        best: tuple[ObjectIdentity, ...] = ()
        best_score = -1
        stop_after_first = not preferred

        def walk(prefix: tuple[ObjectIdentity, ...],
                 remaining: tuple[ObjectIdentity, ...]) -> None:
            """按基序展开满足 precedence 的候选，预算耗尽后不返回未证最优结果。"""
            nonlocal explored, exhausted, best, best_score
            if exhausted or (stop_after_first and best):
                return
            if explored >= budget.max_states:
                exhausted = True
                return
            explored += 1
            if not remaining:
                if not all(self._satisfied(prefix, item) for item in required):
                    return
                score = sum(
                    item.preference_weight
                    for item in preferred
                    if self._satisfied(prefix, item)
                )
                if score > best_score:
                    best = prefix
                    best_score = score
                return
            placed = frozenset(prefix)
            for index, slot in enumerate(remaining):
                if not predecessors[slot].issubset(placed):
                    continue
                walk(
                    (*prefix, slot),
                    remaining[:index] + remaining[index + 1:],
                )
                if exhausted or (stop_after_first and best):
                    return

        walk((), base_order)
        return _SearchResult(() if exhausted else best, explored, exhausted)

    @staticmethod
    def _satisfied(
            order: tuple[ObjectIdentity, ...],
            constraint: ResolvedStructureOrderConstraint,
            ) -> bool:
        """按 resolved 方向和间距界判断一个完整局部序。"""
        positions = {slot: index for index, slot in enumerate(order)}
        if (constraint.before_slot not in positions
                or constraint.after_slot not in positions):
            return constraint.allow_missing
        before = positions[constraint.before_slot]
        after = positions[constraint.after_slot]
        if before >= after:
            return False
        gap = after - before - 1
        if gap < constraint.minimum_gap:
            return False
        if (constraint.maximum_gap is not None
                and gap > constraint.maximum_gap):
            return False
        return True

    def _evaluate(
            self, order: tuple[ObjectIdentity, ...],
            constraints: tuple[ResolvedStructureOrderConstraint, ...],
            ) -> tuple[StructureOrderConstraintEvaluation, ...]:
        """保留 applicable、skipped 和 unknown constraint 的逐项消费结果。"""
        evaluations: list[StructureOrderConstraintEvaluation] = []
        for item in constraints:
            if item.applicability == ORDER_APPLICABLE:
                satisfied = self._satisfied(order, item)
            elif item.applicability == ORDER_NOT_APPLICABLE:
                satisfied = None
            else:
                satisfied = None
            evaluations.append(StructureOrderConstraintEvaluation(
                item, satisfied))
        return tuple(evaluations)

    @staticmethod
    def _satisfied_positions(
            positions: dict[ObjectIdentity, int],
            constraint: ResolvedStructureOrderConstraint,
            ) -> bool:
        """按解析输入的真实局部位置判断方向和间距，保留未赋值位置空隙。"""
        if (constraint.before_slot not in positions
                or constraint.after_slot not in positions):
            return constraint.allow_missing
        before = positions[constraint.before_slot]
        after = positions[constraint.after_slot]
        if before >= after:
            return False
        gap = after - before - 1
        if gap < constraint.minimum_gap:
            return False
        if (constraint.maximum_gap is not None
                and gap > constraint.maximum_gap):
            return False
        return True

    def _evaluate_positions(
            self, positions: dict[ObjectIdentity, int],
            constraints: tuple[ResolvedStructureOrderConstraint, ...],
            ) -> tuple[StructureOrderConstraintEvaluation, ...]:
        """逐项记录 resolved constraint 在真实解析位置上的满足状态。"""
        evaluations: list[StructureOrderConstraintEvaluation] = []
        for item in constraints:
            satisfied = (
                self._satisfied_positions(positions, item)
                if item.applicability == ORDER_APPLICABLE
                else None
            )
            evaluations.append(StructureOrderConstraintEvaluation(
                item, satisfied))
        return tuple(evaluations)


__all__ = [
    "ORDER_APPLICABILITY_UNKNOWN",
    "ORDER_APPLICABLE",
    "ORDER_CONSUMER_ACCEPTED",
    "ORDER_CONSUMER_REJECTED",
    "ORDER_CONSUMER_UNKNOWN",
    "ORDER_NOT_APPLICABLE",
    "PositionedStructureSlotValue",
    "ResolvedStructureOrderConstraint",
    "StructureOrderConstraintEvaluation",
    "StructureOrderConsumer",
    "StructureOrderConsumerProtocol",
    "StructureOrderLinearizationResult",
    "StructureOrderParseResult",
    "StructureOrderSearchBudget",
    "StructureOrderSemanticsResolver",
    "StructureSlotValue",
]
