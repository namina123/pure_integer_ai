"""A-10 AttractorState 的 query 生命周期、M-07 接线和消费入口。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.attractor_reasoning import (
    ReasoningAgendaConsumer,
    ReasoningAgendaConsumption,
)
from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivationMapper,
    AttractorBudget,
    AttractorContextUpdate,
    AttractorProtocol,
    AttractorRecomputeStrategy,
    AttractorState,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryQueryCompilation
from pure_integer_ai.cognition.shared.memory_resolver import MemoryResolution
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningBudget,
    ReasoningObligation,
)
from pure_integer_ai.experiments.train_context import TrainContext


class AttractorRuntime:
    """把 query-scoped A-10 状态安装到当前 TrainContext 并提供真实消费者入口。"""

    def __init__(
            self,
            ctx: TrainContext,
            protocol: AttractorProtocol,
            budget: AttractorBudget,
            mapper: AttractorActivationMapper,
            recompute_strategy: AttractorRecomputeStrategy,
            ) -> None:
        """绑定 M-07 所属上下文和两个注入式方向组件，不持有长期 Memory。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(protocol, AttractorProtocol):
            raise TypeError("protocol 必须是 AttractorProtocol")
        if not isinstance(budget, AttractorBudget):
            raise TypeError("budget 必须是 AttractorBudget")
        if not hasattr(mapper, "project") or not hasattr(mapper, "state_key"):
            raise TypeError("mapper 缺少 A-10 协议")
        if (not hasattr(recompute_strategy, "recompute")
                or not hasattr(recompute_strategy, "state_key")):
            raise TypeError("recompute_strategy 缺少 A-10 协议")
        if ctx.memory_resolver_runtime is None:
            raise ValueError("安装 A-10 前必须先安装 M-07 resolver runtime")
        if ctx.attractor_runtime is not None:
            raise ValueError("TrainContext 已安装 A-10 runtime")
        self._ctx = ctx
        self.protocol = protocol
        self.budget = budget
        self.mapper = mapper
        self.recompute_strategy = recompute_strategy

    def resolve_and_activate(
            self,
            compilation: MemoryQueryCompilation,
            obligations: tuple[ReasoningObligation, ...],
            ) -> AttractorState:
        """沿 M-07 resolver 到 A-10 agenda 完成当前 query 的真实接线。"""
        if not isinstance(compilation, MemoryQueryCompilation):
            raise TypeError("compilation 必须是 MemoryQueryCompilation")
        if self._ctx.work_memory.active_query_scope != compilation.current.scope:
            raise ValueError("compilation 不属于当前活动 query")
        resolution = self._ctx.memory_resolver_runtime.resolve(compilation)
        return self.activate_resolution(resolution, obligations)

    def activate_resolution(
            self,
            resolution: MemoryResolution,
            obligations: tuple[ReasoningObligation, ...],
            ) -> AttractorState:
        """把已验证的 M-07 结果投影成唯一当前 query AttractorState。"""
        if not isinstance(resolution, MemoryResolution):
            raise TypeError("resolution 必须是 MemoryResolution")
        if self._ctx.work_memory.active_query_scope != (
                resolution.compilation.current.scope):
            raise ValueError("resolution 不属于当前活动 query")
        state = AttractorState(
            resolution.compilation.current.scope,
            resolution.compilation.current.source,
            resolution.compilation.current.logical_timestamp,
            self.protocol,
            self.budget,
        )
        state.activate(resolution, obligations, self.mapper)
        self._ctx.work_memory.install_attractor_state(state)
        return state

    def apply_update(self, update: AttractorContextUpdate):
        """将后文 typed 变化交给当前 query 的局部重算协议。"""
        state = self._ctx.work_memory.require_attractor_state()
        return state.apply_update(update, self.recompute_strategy)

    def consume_reasoning(
            self,
            consumer: ReasoningAgendaConsumer,
            budget: ReasoningBudget,
            ) -> ReasoningAgendaConsumption | None:
        """取一个 agenda 项交给 S-05，并在成功返回后原子记录处理边界。"""
        if not isinstance(consumer, ReasoningAgendaConsumer):
            raise TypeError("consumer 必须是 ReasoningAgendaConsumer")
        if consumer.protocol != self.protocol:
            raise ValueError("consumer 与 A-10 protocol 不一致")
        state = self._ctx.work_memory.require_attractor_state()
        activation = state.next_activation()
        if activation is None:
            return None
        if state.remaining_consumptions <= 0:
            raise RuntimeError("A-10 consumption 预算已耗尽")
        result = consumer.consume(activation, budget)
        state.commit_consumption(result.decision)
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回 runtime 配置键，不包含当前 query 的可变 agenda。"""
        mapper_key = self.mapper.state_key()
        strategy_key = self.recompute_strategy.state_key()
        if (not isinstance(mapper_key, tuple)
                or not mapper_key
                or any(type(item) is not int for item in mapper_key)):
            raise ValueError("A-10 mapper state_key 非法")
        if (not isinstance(strategy_key, tuple)
                or not strategy_key
                or any(type(item) is not int for item in strategy_key)):
            raise ValueError("A-10 recompute state_key 非法")
        return (
            1,
            *self.protocol.stable_key(),
            *self.budget.stable_key(),
            len(mapper_key),
            *mapper_key,
            len(strategy_key),
            *strategy_key,
        )

    def clone_for_context(self, ctx: TrainContext) -> "AttractorRuntime":
        """为 V-06 重绑独立上下文，禁止复用可变 mapper 或重算组件。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        mapper = _clone_component(self.mapper, ctx)
        strategy = _clone_component(self.recompute_strategy, ctx)
        cloned = AttractorRuntime(
            ctx, self.protocol, self.budget, mapper, strategy)
        if cloned.state_key() != self.state_key():
            raise ValueError("A-10 runtime clone 改变了注入协议状态")
        return cloned


def _clone_component(component: object, ctx: TrainContext) -> object:
    """调用组件 clone_for_context，拒绝在 V-06 中共享可变方向状态。"""
    clone = getattr(component, "clone_for_context", None)
    if not callable(clone):
        raise TypeError("A-10 组件必须实现 clone_for_context")
    return clone(ctx)


def install_attractor_runtime(
        ctx: TrainContext,
        protocol: AttractorProtocol,
        budget: AttractorBudget,
        mapper: AttractorActivationMapper,
        recompute_strategy: AttractorRecomputeStrategy,
        ) -> AttractorRuntime:
    """在已装配 M-07 的 TrainContext 上安装唯一 A-10 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    runtime = AttractorRuntime(
        ctx, protocol, budget, mapper, recompute_strategy)
    ctx.attractor_runtime = runtime
    return runtime


__all__ = [
    "AttractorRuntime",
    "install_attractor_runtime",
]
