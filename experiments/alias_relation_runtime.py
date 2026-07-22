"""R-01 typed alias/refers/realizes 选择与 R-00 consumer use 归因编排。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    ActiveAliasRouteFinder,
    AliasResolutionProposal,
    AliasResolutionResult,
    AliasResolutionSelector,
    AliasRouteSearchBudget,
    ReferenceRouteDiscovery,
    SurfaceRouteDiscovery,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureFact,
)
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
    RelationClosureUse,
)


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验消费 use key 是非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


@dataclass(frozen=True)
class AliasResolutionUse:
    """一次唯一选择及其全部 R-00 active fact 采用账。"""

    use_key: tuple[int, ...]
    result: AliasResolutionResult
    discovery: ReferenceRouteDiscovery | SurfaceRouteDiscovery
    relation_uses: tuple[RelationClosureUse, ...]
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """核验唯一选择、全部 active fact 归因和 query context 精确一致。"""
        _strict_key(self.use_key, label="alias resolution use key")
        AliasResolutionProposal(self.result, self.discovery)
        if not isinstance(self.relation_uses, tuple) or any(
                not isinstance(item, RelationClosureUse)
                for item in self.relation_uses):
            raise TypeError("alias resolution relation_uses 类型错误")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("alias resolution context 类型错误")
        if any(item.context != self.context for item in self.relation_uses):
            raise ValueError("alias resolution 与 relation use context 不一致")
        selected = self.result.selected
        expected = {}
        if selected is not None:
            expected = {
                step.fact.proposition.proposition: step.fact
                for route in selected.routes
                for step in route.steps
            }
        actual = {item.proposition: item for item in self.relation_uses}
        if (set(actual) != set(expected)
                or len(actual) != len(self.relation_uses)):
            raise ValueError("alias resolution use 未精确覆盖 selected active fact")
        for proposition, use in actual.items():
            fact = expected[proposition]
            if (use.hypothesis != fact.hypothesis
                    or use.evidence_keys != fact.evidence_keys
                    or use.decision_key != fact.decision_key
                    or use.read_only_recovered != fact.read_only_recovered):
                raise ValueError("alias resolution use 的 active 归因已陈旧")

    def stable_key(self) -> tuple[int, ...]:
        """返回 query 选择和全部 R-00 Evidence/H-04 采用归因。"""
        result = [
            *_packed(self.use_key),
            *_packed(self.result.stable_key()),
            *_packed(self.discovery.stable_key()),
            len(self.relation_uses),
        ]
        for use in self.relation_uses:
            result.extend(_packed((
                *_packed(use.use_key),
                *_packed(use.proposition.stable_key()),
                *_packed(use.hypothesis.stable_key()),
                len(use.evidence_keys),
                *(value for key in use.evidence_keys for value in _packed(key)),
                *_packed(use.decision_key),
                1 if use.read_only_recovered else 0,
                *_packed(() if use.event is None else use.event.stable_key()),
            )))
        result.extend(_packed(
            () if self.context is None else self.context.stable_key()))
        return tuple(result)

    def route_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回完整 context 加局部 alias use_key 的幂等路由。"""
        context_key = () if self.context is None else self.context.stable_key()
        return context_key, self.use_key


class AliasRelationRuntime:
    """执行 typed 选择，并仅为唯一结果提交 R-00 consumer use。"""

    def __init__(
            self,
            closure: RelationClosureRuntime,
            selector: AliasResolutionSelector,
            ) -> None:
        """绑定 R-00 closure 和 route selector，并建立 context-scoped Use 索引。"""
        if not isinstance(closure, RelationClosureRuntime):
            raise TypeError("alias runtime closure 类型错误")
        if not isinstance(selector, AliasResolutionSelector):
            raise TypeError("alias runtime selector 类型错误")
        self.closure = closure
        self.selector = selector
        self.route_finder = ActiveAliasRouteFinder(
            closure.consumer, selector)
        self._uses: dict[
            tuple[tuple[int, ...], tuple[int, ...]], AliasResolutionUse
        ] = {}

    def resolve_reference(
            self,
            origin: ObjectIdentity,
            *,
            target_kinds: tuple[int, ...],
            budget: AliasRouteSearchBudget,
            use_key: tuple[int, ...],
            context: RelationUseContext | None = None,
            ) -> AliasResolutionUse:
        """从全部 active alias/refers fact 解析方向同指并保留歧义。"""
        proposal = self.preview_reference(
            origin, target_kinds=target_kinds, budget=budget)
        return self.commit_many(
            ((proposal, use_key),), context=context)[0]

    def select_surface(
            self,
            origin: ObjectIdentity,
            branch: ObjectIdentity,
            *,
            budget: AliasRouteSearchBudget,
            use_key: tuple[int, ...],
            allowed_prefix_steps: tuple[ObjectIdentity, ...] | None = None,
            context: RelationUseContext | None = None,
            ) -> AliasResolutionUse:
        """按注入前缀策略选择目标分支词形；多词形保持 ambiguous。"""
        proposal = self.preview_surface(
            origin,
            branch,
            budget=budget,
            allowed_prefix_steps=allowed_prefix_steps,
        )
        return self.commit_many(
            ((proposal, use_key),), context=context)[0]

    def preview_reference(
            self,
            origin: ObjectIdentity,
            *,
            target_kinds: tuple[int, ...],
            budget: AliasRouteSearchBudget,
            ) -> AliasResolutionProposal:
        """完整发现方向同指候选但不提交任何 alias 或 relation use。"""
        discovery = self.route_finder.discover_reference(
            origin, target_kinds, budget)
        return AliasResolutionProposal(
            self.selector.resolve_reference(discovery.query()), discovery)

    def preview_surface(
            self,
            origin: ObjectIdentity,
            branch: ObjectIdentity,
            *,
            budget: AliasRouteSearchBudget,
            allowed_prefix_steps: tuple[ObjectIdentity, ...] | None = None,
            ) -> AliasResolutionProposal:
        """按注入前缀策略完整发现目标分支词形但不提交采用账。"""
        discovery = self.route_finder.discover_surface(
            origin, branch, budget, allowed_prefix_steps)
        return AliasResolutionProposal(
            self.selector.select_surface(discovery.query()), discovery)

    def commit_many(
            self,
            requests: tuple[
                tuple[AliasResolutionProposal, tuple[int, ...]], ...],
            *,
            context: RelationUseContext | None = None,
            ) -> tuple[AliasResolutionUse, ...]:
        """全量预检多项选择，再原子提交其全部 R-00 fact use 和 alias use。"""
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("alias commit_many requests 必须是非空 tuple")
        if context is not None and not isinstance(context, RelationUseContext):
            raise TypeError("alias commit_many context 类型错误")
        if self.closure.use_owner is not None and context is None:
            raise ValueError("配置 Core Use owner 后 alias 必须提供完整 context")
        normalized: list[tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[int, ...], AliasResolutionProposal,
        ]] = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("alias commit request 必须是 proposal/use_key 对")
            proposal, use_key = request
            if not isinstance(proposal, AliasResolutionProposal):
                raise TypeError("alias commit proposal 类型错误")
            if proposal.result.protocol != self.selector.protocol:
                raise ValueError("alias proposal 使用了其他 runtime protocol")
            key = _strict_key(use_key, label="AliasRelationRuntime.use_key")
            route = (
                () if context is None else context.stable_key(),
                key,
            )
            normalized.append((route, key, proposal))
        routes = tuple(item[0] for item in normalized)
        if len(set(routes)) != len(routes):
            raise ValueError("同批 alias Use 路由不得重复")

        prepared: dict[
            tuple[tuple[int, ...], tuple[int, ...]], AliasResolutionUse
        ] = {}
        relation_requests: list[
            tuple[ObjectIdentity, tuple[int, ...]]
        ] = []
        pending: list[tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[int, ...], AliasResolutionProposal, int, int,
        ]] = []
        for route, key, proposal in normalized:
            existing = self._uses.get(route)
            if existing is not None:
                if (existing.result != proposal.result
                        or existing.discovery != proposal.discovery
                        or existing.context != context):
                    raise ValueError("同一 alias Use 路由已绑定不同选择结果")
                prepared[route] = existing
                continue
            facts = self._selected_facts(proposal.result)
            start = len(relation_requests)
            for ordinal, proposition in enumerate(sorted(
                    facts, key=ObjectIdentity.stable_key)):
                relation_key = (
                    *_packed(key),
                    ordinal,
                    *_packed(proposition.stable_key()),
                )
                relation_requests.append((proposition, relation_key))
            pending.append((
                route, key, proposal, start, len(relation_requests) - start))

        committed: tuple[RelationClosureUse, ...] = ()
        if relation_requests:
            committed = self.closure.consume_many(
                tuple(relation_requests), context=context)
        for route, key, proposal, start, count in pending:
            prepared[route] = AliasResolutionUse(
                key,
                proposal.result,
                proposal.discovery,
                committed[start:start + count],
                context,
            )
        for route, _, _ in normalized:
            if route not in self._uses:
                self._uses[route] = prepared[route]
        return tuple(prepared[route] for route, _, _ in normalized)

    def clone_for_runtime(
            self,
            closure: RelationClosureRuntime,
            ) -> "AliasRelationRuntime":
        """把相同 route protocol 绑定到 R-00 held-out 克隆。"""
        cloned = AliasRelationRuntime(closure, AliasResolutionSelector(
            self.selector.protocol))
        if closure.use_owner is None:
            cloned._uses = dict(self._uses)
        return cloned

    def state_key(self) -> tuple:
        """返回 R-00 owner 和全部 alias/reference/surface 采用状态。"""
        return (
            self.closure.state_key(),
            tuple(
                self._uses[key].stable_key()
                for key in sorted(self._uses)
            ),
        )

    @staticmethod
    def _selected_facts(
            result: AliasResolutionResult,
            ) -> dict[ObjectIdentity, ActiveRelationClosureFact]:
        """按 Proposition 汇总唯一选项使用的 active fact，并核验重复一致。"""
        selected = result.selected
        facts = {}
        if selected is None:
            return facts
        for route in selected.routes:
            for step in route.steps:
                proposition = step.fact.proposition.proposition
                existing = facts.get(proposition)
                if existing is not None and existing != step.fact:
                    raise ValueError("同一 Proposition route 使用了冲突 active fact")
                facts[proposition] = step.fact
        return facts


__all__ = [
    "AliasRelationRuntime",
    "AliasResolutionUse",
]
