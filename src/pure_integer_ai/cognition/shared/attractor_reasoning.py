"""A-10 agenda 到 S-05 ReasoningPlanner 的真实消费边界。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivation,
    AttractorConsumptionDecision,
    AttractorProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningBudget,
    ReasoningPlanResult,
    ReasoningPlanner,
)


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(value), *value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """要求 consumer 身份来自图内最小指令。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


class ReasoningConsumptionStrategy(Protocol):
    """根据完整 S-05 结果决定已消费或暂停，不改变逻辑状态。"""

    def disposition(
            self,
            activation: AttractorActivation,
            result: ReasoningPlanResult,
            protocol: AttractorProtocol,
            ) -> ObjectIdentity:
        """只返回 consumed 或 suspended 图内状态。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回消费决策协议版本和配置键。"""
        ...


@dataclass(frozen=True)
class ReasoningAgendaConsumption:
    """一次 agenda 处理的 S-05 结果和 A-10 提交决策。"""

    decision: AttractorConsumptionDecision
    result: ReasoningPlanResult

    def __post_init__(self) -> None:
        """核验下游 trace 精确链接当前完整规划结果。"""
        if not isinstance(self.decision, AttractorConsumptionDecision):
            raise TypeError("ReasoningAgendaConsumption.decision 类型错误")
        if not isinstance(self.result, ReasoningPlanResult):
            raise TypeError("ReasoningAgendaConsumption.result 类型错误")
        if self.decision.decision_trace_key != self.result.stable_key():
            raise ValueError("agenda decision 未链接当前 ReasoningPlanResult")

    def stable_key(self) -> tuple[int, ...]:
        """返回 A-10 决策和 S-05 完整 trace 的稳定键。"""
        return (
            *_packed(self.decision.stable_key()),
            *_packed(self.result.stable_key()),
        )


class ReasoningAgendaConsumer:
    """按 agenda 选中的 obligation 调用 S-05，不参与评分或真值裁决。"""

    def __init__(
            self,
            planner: ReasoningPlanner,
            protocol: AttractorProtocol,
            consumer_instruction: ObjectIdentity,
            strategy: ReasoningConsumptionStrategy,
            ) -> None:
        """绑定纯 planner、A-10 状态协议和注入式消费决策。"""
        if not isinstance(planner, ReasoningPlanner):
            raise TypeError("planner 必须是 ReasoningPlanner")
        if not isinstance(protocol, AttractorProtocol):
            raise TypeError("protocol 必须是 AttractorProtocol")
        _require_instruction(
            consumer_instruction,
            label="ReasoningAgendaConsumer.consumer_instruction",
        )
        if not hasattr(strategy, "disposition"):
            raise TypeError("strategy 必须实现 disposition")
        state_key = getattr(strategy, "state_key", None)
        if not callable(state_key):
            raise TypeError("strategy 必须实现 state_key")
        strategy_key = state_key()
        if (not isinstance(strategy_key, tuple)
                or not strategy_key
                or any(type(item) is not int for item in strategy_key)):
            raise ValueError("ReasoningConsumptionStrategy.state_key 非法")
        self._planner = planner
        self.protocol = protocol
        self.consumer_instruction = consumer_instruction
        self.strategy = strategy
        self._strategy_key = strategy_key

    def consume(
            self,
            activation: AttractorActivation,
            budget: ReasoningBudget,
            ) -> ReasoningAgendaConsumption:
        """执行选中目标并返回可提交决策，planner 异常时不改变 AttractorState。"""
        if not isinstance(activation, AttractorActivation):
            raise TypeError("activation 必须是 AttractorActivation")
        if activation.status != self.protocol.agenda:
            raise ValueError("ReasoningAgendaConsumer 只能处理 agenda 状态")
        if not isinstance(budget, ReasoningBudget):
            raise TypeError("budget 必须是 ReasoningBudget")
        result = self._planner.plan(activation.obligation, budget)
        disposition = self.strategy.disposition(
            activation, result, self.protocol)
        if disposition not in {
                self.protocol.consumed, self.protocol.suspended}:
            raise ValueError("ReasoningConsumptionStrategy 返回非法状态")
        decision = AttractorConsumptionDecision(
            activation.identity_key(),
            self.consumer_instruction,
            disposition,
            result.stable_key(),
        )
        return ReasoningAgendaConsumption(decision, result)

    def state_key(self) -> tuple[int, ...]:
        """返回 consumer 指令、状态协议和消费策略配置键。"""
        return (
            1,
            *_packed(self.consumer_instruction.stable_key()),
            *_packed(self.protocol.stable_key()),
            *_packed(self._strategy_key),
        )


__all__ = [
    "ReasoningAgendaConsumer",
    "ReasoningAgendaConsumption",
    "ReasoningConsumptionStrategy",
]
