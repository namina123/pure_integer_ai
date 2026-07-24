"""M-07 resolver 的 TrainContext 生命周期和 V-06 隔离装配。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryQueryCompilation
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryResolution,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.experiments.train_context import TrainContext


class MemoryResolverRuntime:
    """把只读 resolver 约束在当前 TrainContext 的活动 query 生命周期内。"""

    def __init__(
            self,
            ctx: TrainContext,
            resolver: MemoryOverlayResolver,
            ) -> None:
        """绑定上下文、M-06 query runtime 和同空间 aggregate/Core facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(resolver, MemoryOverlayResolver):
            raise TypeError("resolver 必须是 MemoryOverlayResolver")
        if ctx.memory_query_runtime is None:
            raise ValueError("安装 M-07 前必须先安装 M-06 query runtime")
        if resolver.aggregates is not ctx.memory_query_runtime.compiler.aggregates:
            raise ValueError("M-07 resolver 与 M-06 compiler 未绑定同一 aggregate")
        if resolver.core_identities is not ctx.core_identity_catalog:
            raise ValueError("M-07 resolver 未绑定当前 TrainContext Core catalog")
        if all(resolver.aggregates is not item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
        )):
            raise ValueError("M-07 resolver aggregate 不属于当前 TrainContext")
        self._ctx = ctx
        self.resolver = resolver

    def resolve(self, compilation: MemoryQueryCompilation) -> MemoryResolution:
        """在活动 query scope 内执行只读仲裁，拒绝陈旧或跨上下文 compilation。"""
        if not isinstance(compilation, MemoryQueryCompilation):
            raise TypeError("compilation 必须是 MemoryQueryCompilation")
        active_scope = self._ctx.work_memory.active_query_scope
        if active_scope is None:
            raise RuntimeError("Memory resolver runtime 需要活动 WorkMemory query scope")
        if compilation.current.scope != active_scope:
            raise ValueError("Memory resolution 输入与活动 WorkMemory query 不一致")
        if compilation.memory_space != self.resolver.aggregates.event_log.memory_space_identity:
            raise ValueError("Memory resolution compilation 属于其他 Memory 空间")
        if self._ctx.memory_hot_set_runtime is not None:
            return self._ctx.memory_hot_set_runtime.resolve(compilation)
        return self.resolver.resolve(compilation)

    def clone_for_context(self, ctx: TrainContext) -> "MemoryResolverRuntime":
        """为 V-06 克隆重绑独立存储 facade，并按组件协议复制注入策略。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        matches = tuple(
            item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
            )
            if (isinstance(item, MemoryHypothesisAggregateIndex)
                and item.event_log.memory_space_identity
                == self.resolver.aggregates.event_log.memory_space_identity)
        )
        if len(matches) != 1:
            raise ValueError("评测上下文缺少唯一同 identity Memory aggregate")
        if ctx.core_identity_catalog is None:
            raise ValueError("评测上下文缺少 Core identity catalog")
        cloned = self.resolver.clone_for_aggregates(
            matches[0],
            ctx.core_identity_catalog,
            baseline_provider=_clone_component(
                self.resolver.baseline_provider, ctx),
            index_filter_provider=_clone_component(
                self.resolver.index_filter_provider, ctx),
            score_provider=_clone_component(
                self.resolver.score_provider, ctx),
            diversity_policy=_clone_component(
                self.resolver.diversity_policy, ctx),
        )
        if cloned.state_key() != self.resolver.state_key():
            raise ValueError("M-07 resolver 克隆改变了注入协议状态")
        return MemoryResolverRuntime(ctx, cloned)

    def state_key(self) -> tuple[int, ...]:
        """返回 resolver 的只读协议状态，供 V-06 宿主污染检查。"""
        return self.resolver.state_key()


def _clone_component(component: object, ctx: TrainContext) -> object:
    """调用组件可选 clone_for_context 协议，否则复用声明为只读的组件。"""
    clone = getattr(component, "clone_for_context", None)
    if clone is None:
        return component
    if not callable(clone):
        raise TypeError("resolver 组件 clone_for_context 必须可调用")
    return clone(ctx)


def install_memory_resolver_runtime(
        ctx: TrainContext,
        resolver: MemoryOverlayResolver,
        ) -> MemoryResolverRuntime:
    """在已安装 M-06 的 TrainContext 上安装唯一 M-07 resolver runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(resolver, MemoryOverlayResolver):
        raise TypeError("resolver 必须是 MemoryOverlayResolver")
    if ctx.memory_resolver_runtime is not None:
        raise ValueError("TrainContext 已安装 Memory resolver runtime")
    runtime = MemoryResolverRuntime(ctx, resolver)
    ctx.memory_resolver_runtime = runtime
    return runtime


__all__ = [
    "MemoryResolverRuntime",
    "install_memory_resolver_runtime",
]
