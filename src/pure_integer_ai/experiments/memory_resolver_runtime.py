"""M-07 resolver 的 TrainContext 生命周期和 V-06 隔离装配。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_HYPOTHESIS,
    require_memory_object_kind,
)
from pure_integer_ai.cognition.shared.memory_query import (
    FederatedMemoryQueryCompilation,
    MemoryQueryCompilation,
    MemoryQueryCompiler,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    FederatedMemoryResolution,
    MemoryResolution,
    ResolvedCandidateSet,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.memory_query_runtime import MemoryQueryRuntime


class MemoryResolverRuntime:
    """把只读 resolver 约束在当前 TrainContext 的活动 query 生命周期内。"""

    def __init__(
            self,
            ctx: TrainContext,
            resolver: MemoryOverlayResolver,
            routes: tuple[object, ...] = (),
            additional_resolvers: tuple[MemoryOverlayResolver, ...] = (),
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
        if (not isinstance(additional_resolvers, tuple)
                or any(not isinstance(item, MemoryOverlayResolver)
                       for item in additional_resolvers)):
            raise TypeError("additional_resolvers 必须是 MemoryOverlayResolver tuple")
        compilers_by_space = {
            item.memory_space.stable_key(): item
            for item in ctx.memory_query_runtime.compilers
        }
        resolvers = (resolver, *additional_resolvers)
        resolver_spaces = tuple(
            item.aggregates.event_log.memory_space_identity.stable_key()
            for item in resolvers
        )
        if len(set(resolver_spaces)) != len(resolver_spaces):
            raise ValueError("联邦 M-07 resolver 不得重复 Memory 空间")
        if set(resolver_spaces) != set(compilers_by_space):
            raise ValueError("联邦 M-06/M-07 Memory 空间集合不一致")
        for item in additional_resolvers:
            space_key = item.aggregates.event_log.memory_space_identity.stable_key()
            if item.aggregates is not compilers_by_space[space_key].aggregates:
                raise ValueError("附加 M-07 resolver 与 compiler aggregate 漂移")
            if item.core_identities is not ctx.core_identity_catalog:
                raise ValueError("附加 M-07 resolver 未绑定当前 Core catalog")
        self.additional_resolvers = tuple(sorted(
            additional_resolvers,
            key=lambda item: (
                item.aggregates.event_log.memory_space_identity.stable_key()),
        ))
        self._routes: dict[int, object] = {}
        if not isinstance(routes, tuple):
            raise TypeError("M-07 routes 必须是 tuple")
        for route in routes:
            self.register_route(route)

    @property
    def resolvers(self) -> tuple[MemoryOverlayResolver, ...]:
        """返回主 resolver 和全部隔离空间 resolver。"""
        return self.resolver, *self.additional_resolvers

    def register_route(self, route: object) -> None:
        """按公开 Memory object kind 注册唯一只读 resolver route。"""
        object_kind = getattr(route, "memory_object_kind", None)
        require_memory_object_kind(
            object_kind, where="Memory resolver route object kind")
        if object_kind == MEMORY_OBJECT_HYPOTHESIS:
            raise ValueError("Hypothesis route 由 M-07 主 resolver 承担")
        if not callable(getattr(route, "resolve", None)):
            raise TypeError("Memory resolver route 缺少 resolve")
        state_key = getattr(route, "state_key", None)
        if not callable(state_key) or not state_key():
            raise TypeError("Memory resolver route 缺少非空 state_key")
        if object_kind in self._routes:
            raise ValueError("同一 Memory object kind 已注册 resolver route")
        self._routes[object_kind] = route

    def resolve(
            self,
            compilation: MemoryQueryCompilation | FederatedMemoryQueryCompilation,
            ) -> MemoryResolution | FederatedMemoryResolution:
        """在活动 query scope 内执行只读仲裁，拒绝陈旧或跨上下文 compilation。"""
        if isinstance(compilation, FederatedMemoryQueryCompilation):
            active_scope = self._ctx.work_memory.active_query_scope
            if active_scope is None or compilation.current.scope != active_scope:
                raise ValueError("联邦 Memory resolution 与活动 query 不一致")
            by_space = {
                item.aggregates.event_log.memory_space_identity.stable_key(): item
                for item in self.resolvers
            }
            resolutions = []
            for child in compilation.compilations:
                resolver = by_space.get(child.memory_space.stable_key())
                if resolver is None:
                    raise ValueError("联邦 compilation 含未安装 Memory 空间")
                resolutions.append(self._resolve_single(
                    child,
                    resolver,
                    allow_typed_routes=resolver is self.resolver,
                ))
            return FederatedMemoryResolution(
                compilation, tuple(resolutions))
        if not isinstance(compilation, MemoryQueryCompilation):
            raise TypeError("compilation 必须是 MemoryQueryCompilation")
        return self._resolve_single(
            compilation, self.resolver, allow_typed_routes=True)

    def _resolve_single(
            self,
            compilation: MemoryQueryCompilation,
            resolver: MemoryOverlayResolver,
            *,
            allow_typed_routes: bool,
            ) -> MemoryResolution:
        """在指定空间执行一次仲裁；附加空间首片只开放 Hypothesis。"""
        active_scope = self._ctx.work_memory.active_query_scope
        if active_scope is None:
            raise RuntimeError("Memory resolver runtime 需要活动 WorkMemory query scope")
        if compilation.current.scope != active_scope:
            raise ValueError("Memory resolution 输入与活动 WorkMemory query 不一致")
        if compilation.memory_space != resolver.aggregates.event_log.memory_space_identity:
            raise ValueError("Memory resolution compilation 属于其他 Memory 空间")
        if all(request.memory_object_kind == MEMORY_OBJECT_HYPOTHESIS
               for request in compilation.requests):
            return self._resolve_hypotheses(compilation, resolver)
        if not allow_typed_routes:
            raise ValueError("附加 Memory 空间尚未安装逐空间 typed resolver route")
        resolved: dict[tuple[int, ...], ResolvedCandidateSet] = {}
        hypothesis_requests = tuple(
            request for request in compilation.requests
            if request.memory_object_kind == MEMORY_OBJECT_HYPOTHESIS
        )
        if hypothesis_requests:
            hypothesis_compilation = MemoryQueryCompilation(
                compilation.current,
                compilation.access,
                compilation.memory_space,
                hypothesis_requests,
            )
            for item in self._resolve_hypotheses(
                    hypothesis_compilation, resolver).sets:
                resolved[item.request.stable_key()] = item
        for request in compilation.requests:
            if request.memory_object_kind == MEMORY_OBJECT_HYPOTHESIS:
                continue
            route = self._routes.get(request.memory_object_kind)
            if route is None:
                raise ValueError("当前 Memory object kind 未安装 resolver route")
            item = route.resolve(request)
            if not isinstance(item, ResolvedCandidateSet):
                raise TypeError("Memory resolver route 返回类型错误")
            if item.request != request:
                raise ValueError("Memory resolver route 替换了 request")
            resolved[request.stable_key()] = item
        return MemoryResolution(
            compilation,
            tuple(resolved[request.stable_key()]
                  for request in compilation.requests),
        )

    def _resolve_hypotheses(
            self,
            compilation: MemoryQueryCompilation,
            resolver: MemoryOverlayResolver,
            ) -> MemoryResolution:
        """保持原 Hypothesis/hot-set 路径，供全量和混合 request 复用。"""
        if (resolver is self.resolver
                and self._ctx.memory_hot_set_runtime is not None):
            return self._ctx.memory_hot_set_runtime.resolve(compilation)
        return resolver.resolve(compilation)

    def clone_for_context(self, ctx: TrainContext) -> "MemoryResolverRuntime":
        """为 V-06 克隆重绑独立存储 facade，并按组件协议复制注入策略。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        available = tuple(
            item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
            )
            if isinstance(item, MemoryHypothesisAggregateIndex)
        )
        if ctx.core_identity_catalog is None:
            raise ValueError("评测上下文缺少 Core identity catalog")

        def clone(source: MemoryOverlayResolver) -> MemoryOverlayResolver:
            """按空间身份重绑一个 resolver 和全部注入策略。"""
            matches = tuple(
                item for item in available
                if item.event_log.memory_space_identity
                == source.aggregates.event_log.memory_space_identity
            )
            if len(matches) != 1:
                raise ValueError("评测上下文缺少唯一同 identity Memory aggregate")
            result = source.clone_for_aggregates(
                matches[0],
                ctx.core_identity_catalog,
                baseline_provider=_clone_component(
                    source.baseline_provider, ctx),
                index_filter_provider=_clone_component(
                    source.index_filter_provider, ctx),
                score_provider=_clone_component(source.score_provider, ctx),
                diversity_policy=_clone_component(
                    source.diversity_policy, ctx),
            )
            if result.state_key() != source.state_key():
                raise ValueError("M-07 resolver 克隆改变了注入协议状态")
            return result

        cloned = clone(self.resolver)
        additional = tuple(clone(item) for item in self.additional_resolvers)
        routes = tuple(
            _clone_component(self._routes[key], ctx)
            for key in sorted(self._routes)
        )
        return MemoryResolverRuntime(ctx, cloned, routes, additional)

    def state_key(self) -> tuple[int, ...]:
        """返回 resolver 的只读协议状态，供 V-06 宿主污染检查。"""
        if not self._routes and not self.additional_resolvers:
            return self.resolver.state_key()
        result = [3, len(self.resolvers)]
        for resolver in sorted(
                self.resolvers,
                key=lambda item: (
                    item.aggregates.event_log.memory_space_identity.stable_key())):
            key = resolver.state_key()
            result.extend((len(key), *key))
        result.append(len(self._routes))
        for object_kind in sorted(self._routes):
            route_key = self._routes[object_kind].state_key()
            result.extend((object_kind, len(route_key), *route_key))
        return tuple(result)


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
        *,
        routes: tuple[object, ...] = (),
        ) -> MemoryResolverRuntime:
    """在已安装 M-06 的 TrainContext 上安装唯一 M-07 resolver runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(resolver, MemoryOverlayResolver):
        raise TypeError("resolver 必须是 MemoryOverlayResolver")
    if ctx.memory_resolver_runtime is not None:
        raise ValueError("TrainContext 已安装 Memory resolver runtime")
    runtime = MemoryResolverRuntime(ctx, resolver, routes)
    ctx.memory_resolver_runtime = runtime
    return runtime


def federate_hypothesis_memory_runtimes(
        ctx: TrainContext,
        aggregates: MemoryHypothesisAggregateIndex,
        ) -> MemoryResolverRuntime:
    """原子升级既有 M-06/M-07，使同一 Hypothesis query 覆盖第二空间。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
        raise TypeError("aggregates 必须是 MemoryHypothesisAggregateIndex")
    if ctx.work_memory.active_query_scope is not None:
        raise RuntimeError("活动 query 中不得改变 Memory 联邦成员")
    query_runtime = ctx.memory_query_runtime
    resolver_runtime = ctx.memory_resolver_runtime
    if (not isinstance(query_runtime, MemoryQueryRuntime)
            or not isinstance(resolver_runtime, MemoryResolverRuntime)):
        raise ValueError("联邦升级前必须安装 M-06/M-07")
    if query_runtime.additional_compilers or resolver_runtime.additional_resolvers:
        raise ValueError("首片只允许从单空间升级一次双 Memory 联邦")
    if aggregates in tuple(
            item.aggregates for item in query_runtime.compilers):
        raise ValueError("目标 aggregate 已属于当前 Memory query")
    if all(aggregates is not item for item in (
            ctx.memory_read_aggregates, ctx.memory_interact_aggregates)):
        raise ValueError("目标 aggregate 不属于当前 TrainContext")

    compiler = MemoryQueryCompiler(
        aggregates, query_runtime.compiler.protocol)
    peer = resolver_runtime.resolver.clone_for_aggregates(
        aggregates,
        ctx.core_identity_catalog,
        baseline_provider=_clone_component(
            resolver_runtime.resolver.baseline_provider, ctx),
        index_filter_provider=_clone_component(
            resolver_runtime.resolver.index_filter_provider, ctx),
        score_provider=_clone_component(
            resolver_runtime.resolver.score_provider, ctx),
        diversity_policy=_clone_component(
            resolver_runtime.resolver.diversity_policy, ctx),
    )
    replacement_query = MemoryQueryRuntime(
        ctx, query_runtime.compiler, (compiler,))
    routes = tuple(
        resolver_runtime._routes[key]
        for key in sorted(resolver_runtime._routes)
    )
    original_query = ctx.memory_query_runtime
    original_resolver = ctx.memory_resolver_runtime
    try:
        ctx.memory_query_runtime = replacement_query
        replacement_resolver = MemoryResolverRuntime(
            ctx,
            resolver_runtime.resolver,
            routes,
            (peer,),
        )
        ctx.memory_resolver_runtime = replacement_resolver
        if ctx.memory_use_runtime is not None:
            ctx.memory_use_runtime.register_candidate_event_log(
                aggregates.event_log)
    except BaseException:
        ctx.memory_query_runtime = original_query
        ctx.memory_resolver_runtime = original_resolver
        raise
    return replacement_resolver


__all__ = [
    "MemoryResolverRuntime",
    "federate_hypothesis_memory_runtimes",
    "install_memory_resolver_runtime",
]
