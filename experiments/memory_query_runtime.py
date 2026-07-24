"""M-06 当前输入 Memory query compiler 的 TrainContext 安装与生命周期接线。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryCurrentQuery,
    MemoryQueryCompilation,
    MemoryQueryCompiler,
    MemoryQueryProtocol,
)
from pure_integer_ai.experiments.train_context import TrainContext


class MemoryQueryRuntime:
    """把一个 M-04 aggregate 目标与 A-09 当前 query 生命周期绑定。"""

    def __init__(
            self,
            ctx: TrainContext,
            compiler: MemoryQueryCompiler,
            ) -> None:
        """核验 compiler 只绑定当前上下文拥有的一个 Memory aggregate。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(compiler, MemoryQueryCompiler):
            raise TypeError("compiler 必须是 MemoryQueryCompiler")
        if all(compiler.aggregates is not item for item in (
                ctx.memory_read_aggregates, ctx.memory_interact_aggregates)):
            raise ValueError("Memory query compiler 未绑定当前 TrainContext aggregate")
        self._ctx = ctx
        self.compiler = compiler

    def compile(
            self,
            current: MemoryCurrentQuery,
            *,
            access: MemoryAccessContext,
            ) -> MemoryQueryCompilation:
        """在活动 query scope 内编译请求，拒绝脱离 A-09 生命周期的调用。"""
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("current 必须是 MemoryCurrentQuery")
        active_scope = self._ctx.work_memory.active_query_scope
        if active_scope is None:
            raise RuntimeError("Memory query runtime 需要活动 WorkMemory query scope")
        if active_scope != current.scope:
            raise ValueError("Memory query 输入 scope 与活动 WorkMemory query 不一致")
        return self.compiler.compile(current, access=access)

    def clone_for_context(self, ctx: TrainContext) -> "MemoryQueryRuntime":
        """为 V-06 独立上下文重绑同 identity 的 aggregate，绝不共享宿主 facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        matches = tuple(
            item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
            )
            if (isinstance(item, MemoryHypothesisAggregateIndex)
                and item.event_log.memory_space_identity
                == self.compiler.memory_space)
        )
        if len(matches) != 1:
            raise ValueError("评测上下文缺少唯一同 identity Memory aggregate")
        return MemoryQueryRuntime(
            ctx,
            self.compiler.clone_for_aggregates(matches[0]),
        )

    def state_key(self) -> tuple[int, ...]:
        """返回不可变 query 协议与目标空间状态，供隔离断言读取。"""
        return self.compiler.state_key()


def install_memory_query_runtime(
        ctx: TrainContext,
        protocol: MemoryQueryProtocol,
        *,
        aggregates: MemoryHypothesisAggregateIndex,
        ) -> MemoryQueryRuntime:
    """在 TrainContext 安装一个显式协议驱动的当前输入 query runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(protocol, MemoryQueryProtocol):
        raise TypeError("protocol 必须是 MemoryQueryProtocol")
    if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
        raise TypeError("aggregates 必须是 MemoryHypothesisAggregateIndex")
    if ctx.memory_query_runtime is not None:
        raise ValueError("TrainContext 已安装 Memory query runtime")
    if all(aggregates is not item for item in (
            ctx.memory_read_aggregates, ctx.memory_interact_aggregates)):
        raise ValueError("Memory query runtime 必须绑定当前 TrainContext aggregate")
    runtime = MemoryQueryRuntime(
        ctx,
        MemoryQueryCompiler(aggregates, protocol),
    )
    ctx.memory_query_runtime = runtime
    return runtime


__all__ = [
    "MemoryQueryRuntime",
    "install_memory_query_runtime",
]
