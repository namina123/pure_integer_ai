"""L-05B1：把 G-02 句法义务接到 active S-07 结构顺序消费者。

本模块只执行 typed schema、active lifecycle 和局部线性化守恒。它不读取旧
EDGE_PRECEDES、def_array、role_seq/token_seq 或 dag_path，也不选择词形和 surface。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_structure_plan import (
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_CONSUMER_ACCEPTED,
    ResolvedStructureOrderConstraint,
    StructureOrderConstraintEvaluation,
    StructureOrderConsumer,
    StructureOrderLinearizationResult,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderProjection,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _require_structure(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """核验句子和结构身份均为一等 StructureConcept。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
        raise ValueError(f"{label} 必须是 StructureConcept")
    return identity


def _slot_key(definition: StructureSlotDefinition) -> tuple[int, ...]:
    """返回 graph slot 的 structure、slot、Role 和 value type 完整键。"""
    if not isinstance(definition, StructureSlotDefinition):
        raise TypeError("slot definition 类型错误")
    return (
        *_packed(definition.structure.stable_key()),
        *_packed(definition.slot.stable_key()),
        *_packed(definition.role.stable_key()),
        *_packed(definition.value_type.stable_key()),
    )


def _value_key(value: StructureSlotValue) -> tuple[int, ...]:
    """返回实际 slot value 的完整对象键。"""
    if not isinstance(value, StructureSlotValue):
        raise TypeError("slot value 类型错误")
    return (
        *_packed(value.slot.stable_key()),
        *_packed(value.filler.stable_key()),
    )


def _resolved_key(
        resolved: ResolvedStructureOrderConstraint,
        ) -> tuple[int, ...]:
    """序列化 resolver 已解释的通用顺序控制参数。"""
    maximum = (
        (0,)
        if resolved.maximum_gap is None
        else (1, resolved.maximum_gap)
    )
    return (
        *_packed(resolved.constraint.stable_key()),
        resolved.applicability,
        *_packed(resolved.before_slot.stable_key()),
        *_packed(resolved.after_slot.stable_key()),
        1 if resolved.enforced else 0,
        1 if resolved.allow_missing else 0,
        resolved.preference_weight,
        resolved.minimum_gap,
        *maximum,
        *_packed(resolved.reason.stable_key()),
    )


def _evaluation_key(
        evaluation: StructureOrderConstraintEvaluation,
        ) -> tuple[int, ...]:
    """保存一个 active constraint 的 resolved 参数和满足三态。"""
    satisfied = (
        0 if evaluation.satisfied is None
        else 2 if evaluation.satisfied else 1
    )
    return (*_packed(_resolved_key(evaluation.resolved)), satisfied)


def _projection_key(projection: StructureOrderProjection) -> tuple[int, ...]:
    """保存 active constraint 的 H-06 来源和完整 lifecycle 事件链。"""
    definition = projection.constraint.definition
    result = [
        *_packed(definition.constraint.stable_key()),
        *_packed(definition.hypothesis.stable_key()),
        *_packed(projection.state.stable_key()),
        0 if projection.replacement is None else 1,
    ]
    if projection.replacement is not None:
        result.extend(_packed(projection.replacement.stable_key()))
    result.append(len(projection.history))
    for materialized in projection.history:
        event = materialized.definition
        result.extend(_packed((
            *_packed(event.event.stable_key()),
            *_packed(event.event_kind.stable_key()),
            *_packed(event.from_state.stable_key()),
            *_packed(event.to_state.stable_key()),
            *_packed(event.hypothesis.stable_key()),
            len(event.evidence_keys),
            *(value for key in event.evidence_keys for value in _packed(key)),
            *_packed(event.decision_key),
            event.timestamp_seq,
            0 if event.replacement is None else 1,
            *(() if event.replacement is None
              else _packed(event.replacement.stable_key())),
        )))
    return tuple(result)


def _linearization_key(
        result: StructureOrderLinearizationResult,
        ) -> tuple[int, ...]:
    """返回有序 value、逐约束评估、原因和搜索预算使用键。"""
    values: list[int] = [result.status, len(result.values)]
    for value in result.values:
        values.extend(_packed(_value_key(value)))
    values.append(len(result.evaluations))
    for evaluation in result.evaluations:
        values.extend(_packed(_evaluation_key(evaluation)))
    values.append(len(result.reasons))
    for reason in result.reasons:
        values.extend(_packed(reason.stable_key()))
    values.append(result.explored_states)
    return tuple(values)


@dataclass(frozen=True)
class SentenceStructureExecutionBudget:
    """调用方为一个 G-02 sentence 注入的 S-07 搜索预算。"""

    sentence: ObjectIdentity
    budget: StructureOrderSearchBudget

    def __post_init__(self) -> None:
        _require_structure(self.sentence, label="execution budget sentence")
        if not isinstance(self.budget, StructureOrderSearchBudget):
            raise TypeError("execution budget 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回句子和严格整数搜索上限。"""
        return (*_packed(self.sentence.stable_key()), self.budget.max_states)


@dataclass(frozen=True)
class GenerationStructureExecutionRequest:
    """完整 G-02 SyntaxPlan 及其逐句注入预算。"""

    syntax: SyntaxPlan
    budgets: tuple[SentenceStructureExecutionBudget, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.syntax, SyntaxPlan):
            raise TypeError("execution request syntax 类型错误")
        if not isinstance(self.budgets, tuple) or any(
                not isinstance(item, SentenceStructureExecutionBudget)
                for item in self.budgets):
            raise TypeError("execution request budgets 类型错误")
        sentence_order = tuple(item.sentence for item in self.syntax.sentences)
        budget_sentences = tuple(item.sentence for item in self.budgets)
        if len(set(budget_sentences)) != len(budget_sentences):
            raise ValueError("同一 sentence 不得重复注入搜索预算")
        if set(budget_sentences) != set(sentence_order):
            raise ValueError("每个 planned sentence 必须恰有一个搜索预算")
        ordinal = {
            sentence.sentence: sentence.ordinal
            for sentence in self.syntax.sentences
        }
        object.__setattr__(self, "budgets", tuple(sorted(
            self.budgets, key=lambda item: ordinal[item.sentence])))

    def stable_key(self) -> tuple[int, ...]:
        """返回 SyntaxPlan 和全部逐句预算完整键。"""
        result = [*_packed(self.syntax.stable_key()), len(self.budgets)]
        for budget in self.budgets:
            result.extend(_packed(budget.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SentenceStructureExecution:
    """一个句子从 graph schema、active constraint 到线性化的完整 trace。"""

    obligation: SyntaxLinearizationObligation
    graph_slots: tuple[StructureSlotDefinition, ...]
    active_constraints: tuple[StructureOrderProjection, ...]
    budget: StructureOrderSearchBudget
    result: StructureOrderLinearizationResult

    def __post_init__(self) -> None:
        if not isinstance(self.obligation, SyntaxLinearizationObligation):
            raise TypeError("sentence execution obligation 类型错误")
        if not isinstance(self.graph_slots, tuple) or not self.graph_slots:
            raise ValueError("sentence execution 必须保存 graph slot schema")
        if any(not isinstance(item, StructureSlotDefinition)
               for item in self.graph_slots):
            raise TypeError("sentence execution graph_slots 类型错误")
        if any(item.structure != self.obligation.structure
               for item in self.graph_slots):
            raise ValueError("graph slot schema 混入其他 StructureConcept")
        if not isinstance(self.active_constraints, tuple) or any(
                not isinstance(item, StructureOrderProjection)
                for item in self.active_constraints):
            raise TypeError("sentence execution active_constraints 类型错误")
        if not isinstance(self.budget, StructureOrderSearchBudget):
            raise TypeError("sentence execution budget 类型错误")
        if not isinstance(self.result, StructureOrderLinearizationResult):
            raise TypeError("sentence execution result 类型错误")
        active = tuple(
            item.constraint.definition.constraint
            for item in self.active_constraints
        )
        if active != self.obligation.constraints:
            raise ValueError("obligation constraint 与 active projection 不一致")
        evaluated = tuple(
            item.resolved.constraint for item in self.result.evaluations)
        if evaluated != active:
            raise ValueError("线性化 evaluation 未完整覆盖 active constraint")
        if self.result.status == ORDER_CONSUMER_ACCEPTED:
            if (len(self.result.values) != len(self.obligation.values)
                    or set(self.result.values) != set(self.obligation.values)):
                raise ValueError("accepted 线性化必须守恒全部输入 slot value")
        object.__setattr__(self, "graph_slots", tuple(sorted(
            self.graph_slots, key=lambda item: item.slot.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回 obligation、graph schema、lifecycle trace、预算和结果键。"""
        result = [
            *_packed(self.obligation.stable_key()),
            len(self.graph_slots),
        ]
        for slot in self.graph_slots:
            result.extend(_packed(_slot_key(slot)))
        result.append(len(self.active_constraints))
        for projection in self.active_constraints:
            result.extend(_packed(_projection_key(projection)))
        result.append(self.budget.max_states)
        result.extend(_packed(_linearization_key(self.result)))
        return tuple(result)


@dataclass(frozen=True)
class GenerationStructureExecutionPlan:
    """G-02 SyntaxPlan 经 active S-07 消费后的逐句只读执行结果。"""

    request: GenerationStructureExecutionRequest
    sentences: tuple[SentenceStructureExecution, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationStructureExecutionRequest):
            raise TypeError("structure execution request 类型错误")
        if not isinstance(self.sentences, tuple) or any(
                not isinstance(item, SentenceStructureExecution)
                for item in self.sentences):
            raise TypeError("structure execution sentences 类型错误")
        identities = tuple(item.obligation.sentence for item in self.sentences)
        if len(set(identities)) != len(identities):
            raise ValueError("structure execution sentence 不得重复")
        expected = tuple(
            item.sentence for item in self.request.syntax.sentences)
        if identities != expected:
            raise ValueError("structure execution 必须逐点覆盖 SyntaxPlan sentence")

    @property
    def complete(self) -> bool:
        """仅当所有句子均被 S-07 接受时返回真。"""
        return all(
            item.result.status == ORDER_CONSUMER_ACCEPTED
            for item in self.sentences
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回请求和全部句级 active projection 消费结果。"""
        result = [*_packed(self.request.stable_key()), len(self.sentences)]
        for sentence in self.sentences:
            result.extend(_packed(sentence.stable_key()))
        return tuple(result)


class GenerationStructureExecutionPlanner:
    """只读消费 G-02 obligation，并拒绝与 active S-07 图状态漂移。"""

    def __init__(
            self,
            lifecycle: StructureOrderLifecycleGraph,
            consumer: StructureOrderConsumer,
            ) -> None:
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("execution planner lifecycle 类型错误")
        if not isinstance(consumer, StructureOrderConsumer):
            raise TypeError("execution planner consumer 类型错误")
        if consumer.lifecycle is not lifecycle:
            raise ValueError("execution planner 必须共享同一个 lifecycle")
        self._lifecycle = lifecycle
        self._consumer = consumer

    def execute(
            self,
            request: GenerationStructureExecutionRequest,
            ) -> GenerationStructureExecutionPlan:
        """按 sentence ordinal 消费 active schema，不读取或回退到旧顺序链。"""
        if not isinstance(request, GenerationStructureExecutionRequest):
            raise TypeError("execution planner request 类型错误")
        syntax = request.syntax
        budgets = {item.sentence: item.budget for item in request.budgets}
        obligations = {
            item.sentence: item for item in syntax.linearization
        }
        executions: list[SentenceStructureExecution] = []
        ontology = self._lifecycle.order_graph.ontology
        for sentence in syntax.sentences:
            obligation = obligations[sentence.sentence]
            structure_ref = ontology.resolve(obligation.structure)
            if structure_ref is None:
                raise ValueError("G-02 structure 尚未在 S-07 图中定义")
            schema = self._lifecycle.order_graph.read_structure(structure_ref)
            graph_slots = tuple(item.definition for item in schema.slots)
            by_slot = {item.slot: item for item in graph_slots}
            for planned in sentence.slots:
                if by_slot.get(planned.slot) != planned:
                    raise ValueError("G-02 sentence slot 与 active graph schema 不一致")
            projections = self._lifecycle.active_constraints(structure_ref)
            active = tuple(
                item.constraint.definition.constraint for item in projections)
            if active != obligation.constraints:
                raise ValueError("G-02 obligation 未精确声明 active S-07 constraint")
            result = self._consumer.linearize(
                structure_ref,
                obligation.values,
                context=obligation.context,
                budget=budgets[sentence.sentence],
            )
            executions.append(SentenceStructureExecution(
                obligation,
                graph_slots,
                projections,
                budgets[sentence.sentence],
                result,
            ))
        return GenerationStructureExecutionPlan(
            request, tuple(executions))


__all__ = [
    "GenerationStructureExecutionPlan",
    "GenerationStructureExecutionPlanner",
    "GenerationStructureExecutionRequest",
    "SentenceStructureExecution",
    "SentenceStructureExecutionBudget",
]
