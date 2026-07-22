"""Event/Proposition 时间事实的分型读写与冲突核验。

关系本身及其 before/after/same/unknown 执行含义均由调用方注入。该模块不读取
Occurrence、Span、token 位置、共享 PRECEDES 或 CAUSES，也不从无事实状态补出唯一序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.order_facts import (
    OrderFact,
    OrderFactIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    ScopeIdentity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


EVENT_TIME_EMPTY = 0
EVENT_TIME_CONSISTENT = 1
EVENT_TIME_CONFLICTED = 2
EVENT_TIME_UNKNOWN = 3

EVENT_TIME_BEFORE = 1
EVENT_TIME_AFTER = 2
EVENT_TIME_SAME = 3
EVENT_TIME_DIRECTION_UNKNOWN = 4
_DIRECTIONS = frozenset({
    EVENT_TIME_BEFORE,
    EVENT_TIME_AFTER,
    EVENT_TIME_SAME,
    EVENT_TIME_DIRECTION_UNKNOWN,
})
_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    return value


def _endpoint(
        value: ObjectIdentity, *, where: str,
        ) -> ObjectIdentity:
    """限定时间事实端点为来源化 Event 或 Proposition。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind not in _ENDPOINT_KINDS:
        raise ValueError(f"{where} 必须是 Event 或 Proposition")
    return value


def _strongly_connected_components(
        nodes: set[ObjectIdentity],
        outgoing: dict[ObjectIdentity, set[ObjectIdentity]],
        ) -> tuple[tuple[ObjectIdentity, ...], ...]:
    """在线性时间内确定性计算方向图的强连通分量。"""
    adjacency = {
        node: tuple(sorted(
            outgoing.get(node, ()),
            key=ObjectIdentity.stable_key,
        ))
        for node in nodes
    }
    index: dict[ObjectIdentity, int] = {}
    low: dict[ObjectIdentity, int] = {}
    active: set[ObjectIdentity] = set()
    stack: list[ObjectIdentity] = []
    components: list[tuple[ObjectIdentity, ...]] = []
    next_index = 0
    for start in sorted(nodes, key=ObjectIdentity.stable_key):
        if start in index:
            continue
        work: list[tuple[ObjectIdentity, int]] = [(start, 0)]
        index[start] = next_index
        low[start] = next_index
        next_index += 1
        stack.append(start)
        active.add(start)
        while work:
            node, neighbor_index = work[-1]
            neighbors = adjacency[node]
            if neighbor_index < len(neighbors):
                neighbor = neighbors[neighbor_index]
                work[-1] = (node, neighbor_index + 1)
                if neighbor not in index:
                    index[neighbor] = next_index
                    low[neighbor] = next_index
                    next_index += 1
                    stack.append(neighbor)
                    active.add(neighbor)
                    work.append((neighbor, 0))
                elif neighbor in active:
                    low[node] = min(low[node], index[neighbor])
                continue
            if low[node] == index[node]:
                component: list[ObjectIdentity] = []
                while True:
                    member = stack.pop()
                    active.remove(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(
                    component,
                    key=ObjectIdentity.stable_key,
                )))
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return tuple(sorted(
        components,
        key=lambda component: tuple(
            item.stable_key() for item in component),
    ))


@dataclass(frozen=True)
class ResolvedEventTimeRelation:
    """把一个图内时间 relation 解释为最小方向执行协议。"""

    relation: ObjectIdentity
    direction: int
    detail_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.relation, ObjectIdentity):
            raise TypeError("resolved relation 必须是 ObjectIdentity")
        assert_int(self.direction, _where="ResolvedEventTimeRelation.direction")
        if self.direction not in _DIRECTIONS:
            raise ValueError("event-time direction 未注册")
        _strict_key(
            self.detail_key,
            where="ResolvedEventTimeRelation.detail_key",
        )


class EventTimeRelationResolver(Protocol):
    """把开放 relation 概念解释为一次核验使用的方向语义。"""

    def resolve(
            self, relation: ObjectIdentity,
            ) -> ResolvedEventTimeRelation: ...


@dataclass(frozen=True)
class EventTimeFactSet:
    """一个精确 scope 下按 relation 聚合的已核验时间事实。"""

    scope: ScopeIdentity
    relations: tuple[ObjectIdentity, ...]
    facts: tuple[OrderFact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("EventTimeFactSet.scope 必须是 ScopeIdentity")
        if not isinstance(self.relations, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.relations):
            raise TypeError("EventTimeFactSet.relations 类型非法")
        if not isinstance(self.facts, tuple) or any(
                not isinstance(item, OrderFact) for item in self.facts):
            raise TypeError("EventTimeFactSet.facts 类型非法")


@dataclass(frozen=True)
class EventTimeVerificationResult:
    """保留原始事实、规范方向、同序分组和冲突来源的核验结果。"""

    status: int
    fact_set: EventTimeFactSet
    before_edges: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    same_groups: tuple[tuple[ObjectIdentity, ...], ...]
    unknown_relations: tuple[ObjectIdentity, ...]
    conflict_assertion_hashes: tuple[int, ...]
    detail_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        assert_int(self.status, _where="EventTimeVerificationResult.status")
        if self.status not in {
                EVENT_TIME_EMPTY,
                EVENT_TIME_CONSISTENT,
                EVENT_TIME_CONFLICTED,
                EVENT_TIME_UNKNOWN}:
            raise ValueError("event-time verification status 未注册")
        if not isinstance(self.fact_set, EventTimeFactSet):
            raise TypeError("fact_set 必须是 EventTimeFactSet")
        if not isinstance(self.before_edges, tuple) or any(
                not isinstance(edge, tuple) or len(edge) != 2
                or any(not isinstance(item, ObjectIdentity) for item in edge)
                for edge in self.before_edges):
            raise TypeError("before_edges 类型非法")
        if not isinstance(self.same_groups, tuple) or any(
                not isinstance(group, tuple) or len(group) < 2
                or any(not isinstance(item, ObjectIdentity) for item in group)
                for group in self.same_groups):
            raise TypeError("same_groups 类型非法")
        if not isinstance(self.unknown_relations, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.unknown_relations):
            raise TypeError("unknown_relations 类型非法")
        if not isinstance(self.conflict_assertion_hashes, tuple):
            raise TypeError("conflict_assertion_hashes 必须是 tuple")
        assert_int(
            *self.conflict_assertion_hashes,
            _where="EventTimeVerificationResult.conflicts",
        )
        if not isinstance(self.detail_keys, tuple):
            raise TypeError("detail_keys 必须是 tuple")
        for index, key in enumerate(self.detail_keys):
            _strict_key(key, where=f"detail_keys[{index}]")


class EventTimeFactIndex:
    """在通用 OrderFactIndex 上提供 Event/Proposition 专用 typed facade。"""

    def __init__(self, facts: OrderFactIndex) -> None:
        if not isinstance(facts, OrderFactIndex):
            raise TypeError("facts 必须是 OrderFactIndex")
        self.facts = facts

    @property
    def ontology(self):
        """返回当前 facade 绑定的权威图。"""
        return self.facts.ontology

    def record(
            self,
            relation: ObjectIdentity,
            subject: ObjectIdentity,
            object_identity: ObjectIdentity,
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> OrderFact:
        """物化来源化端点并追加一条精确 scope 的 typed 时间事实。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("event-time scope 必须是 ScopeIdentity")
        first = _endpoint(subject, where="event-time subject")
        second = _endpoint(object_identity, where="event-time object")
        if first == second:
            raise ValueError("event-time 事实不得形成自环")
        if first.owner != second.owner or first.owner != scope.owner:
            raise ValueError("event-time 端点和 scope owner 必须一致")
        subject_ref = self.ontology.materialize(first)
        object_ref = self.ontology.materialize(second)
        return self.facts.record(
            relation,
            subject_ref,
            object_ref,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    def read(
            self,
            relations: tuple[ObjectIdentity, ...],
            *,
            scope: ScopeIdentity,
            active_only: bool = True,
            ) -> EventTimeFactSet:
        """按调用方 relation 集和精确 scope 返回全部已核验时间事实。"""
        if not isinstance(relations, tuple) or not relations:
            raise ValueError("event-time relations 必须是非空 tuple")
        if any(not isinstance(item, ObjectIdentity) for item in relations):
            raise TypeError("event-time relations 只能包含 ObjectIdentity")
        ordered_relations = tuple(sorted(
            dict.fromkeys(relations), key=ObjectIdentity.stable_key))
        if len(ordered_relations) != len(relations):
            raise ValueError("event-time relations 不得重复")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("event-time scope 必须是 ScopeIdentity")
        collected: list[OrderFact] = []
        for relation in ordered_relations:
            for fact in self.facts.facts(
                    relation, scope=scope, active_only=active_only):
                self._validate_fact(fact, relation=relation, scope=scope)
                collected.append(fact)
        collected.sort(key=OrderFactIndex._sort_key)
        return EventTimeFactSet(scope, ordered_relations, tuple(collected))

    def clone_for_context(
            self, facts: OrderFactIndex,
            ) -> "EventTimeFactIndex":
        """在克隆图的 OrderFactIndex 上重建独立 typed facade。"""
        return EventTimeFactIndex(facts)

    def supersede(
            self,
            old: OrderFact,
            new: OrderFact,
            timestamp: LogicalTimestamp,
            ) -> int:
        """重新核验两条 typed 时间事实后追加替代事件，不删除旧 statement。"""
        if not isinstance(old, OrderFact) or not isinstance(new, OrderFact):
            raise TypeError("event-time supersede 需要两个 OrderFact")
        old_relation = self.ontology.identity_of(old.statement.predicate)
        new_relation = self.ontology.identity_of(new.statement.predicate)
        if old_relation != new_relation:
            raise ValueError("不同 event-time relation 不得互相替代")
        self._validate_fact(
            old,
            relation=old_relation,
            scope=old.statement.assertion.scope,
        )
        self._validate_fact(
            new,
            relation=new_relation,
            scope=new.statement.assertion.scope,
        )
        return self.facts.supersede(old, new, timestamp)

    def _validate_fact(
            self, fact: OrderFact, *, relation: ObjectIdentity,
            scope: ScopeIdentity,
            ) -> None:
        """核验读回 statement 的 relation、scope 和端点类型未被污染。"""
        if fact.statement.assertion.scope != scope:
            raise ValueError("event-time reader 读到跨 scope 事实")
        predicate_identity = self.ontology.identity_of(
            fact.statement.predicate)
        if predicate_identity != relation:
            raise ValueError("event-time reader 读到错误 relation")
        first = self.ontology.identity_of(fact.statement.subject)
        second = self.ontology.identity_of(fact.statement.object)
        _endpoint(first, where="event-time fact subject")
        _endpoint(second, where="event-time fact object")
        if first.owner != second.owner or first.owner != scope.owner:
            raise ValueError("event-time 事实端点与 scope owner 不一致")


class EventTimeVerifier:
    """把 scoped typed 时间事实规范化并保留未知与冲突。"""

    def __init__(
            self, facts: EventTimeFactIndex,
            resolver: EventTimeRelationResolver) -> None:
        if not isinstance(facts, EventTimeFactIndex):
            raise TypeError("facts 必须是 EventTimeFactIndex")
        if not callable(getattr(resolver, "resolve", None)):
            raise TypeError("resolver 必须实现 resolve")
        self.facts = facts
        self.resolver = resolver

    def verify(
            self,
            relations: tuple[ObjectIdentity, ...],
            *,
            scope: ScopeIdentity,
            ) -> EventTimeVerificationResult:
        """核验同序压缩后的方向图，不把一致性提升为现实真值。"""
        fact_set = self.facts.read(relations, scope=scope)
        if not fact_set.facts:
            return EventTimeVerificationResult(
                EVENT_TIME_EMPTY,
                fact_set,
                (),
                (),
                (),
                (),
                (),
            )

        relation_resolutions: dict[
            ObjectIdentity, ResolvedEventTimeRelation] = {}
        detail_keys: list[tuple[int, ...]] = []
        used_relations = {
            self.facts.ontology.identity_of(fact.statement.predicate)
            for fact in fact_set.facts
        }
        for relation in sorted(
                used_relations, key=ObjectIdentity.stable_key):
            resolved = self.resolver.resolve(relation)
            if not isinstance(resolved, ResolvedEventTimeRelation):
                raise TypeError("event-time resolver 返回类型错误")
            if resolved.relation != relation:
                raise ValueError("event-time resolver 替换了 relation 身份")
            relation_resolutions[relation] = resolved
            detail_keys.append(resolved.detail_key)

        endpoints: dict[ObjectIdentity, ObjectIdentity] = {}

        def find(value: ObjectIdentity) -> ObjectIdentity:
            """返回同序并查集根，并执行确定性路径压缩。"""
            parent = endpoints.setdefault(value, value)
            if parent != value:
                endpoints[value] = find(parent)
            return endpoints[value]

        def union(left: ObjectIdentity, right: ObjectIdentity) -> None:
            """按完整稳定键合并同序端点，避免调用顺序改变根。"""
            first = find(left)
            second = find(right)
            if first == second:
                return
            if first.stable_key() <= second.stable_key():
                endpoints[second] = first
            else:
                endpoints[first] = second

        decoded: list[
            tuple[OrderFact, ObjectIdentity, ObjectIdentity, int]
        ] = []
        unknown_relations: dict[ObjectIdentity, None] = {}
        for fact in fact_set.facts:
            relation = self.facts.ontology.identity_of(
                fact.statement.predicate)
            first = self.facts.ontology.identity_of(
                fact.statement.subject)
            second = self.facts.ontology.identity_of(
                fact.statement.object)
            direction = relation_resolutions[relation].direction
            decoded.append((fact, first, second, direction))
            find(first)
            find(second)
            if direction == EVENT_TIME_SAME:
                union(first, second)
            elif direction == EVENT_TIME_DIRECTION_UNKNOWN:
                unknown_relations[relation] = None

        groups: dict[ObjectIdentity, list[ObjectIdentity]] = {}
        for endpoint in sorted(endpoints, key=ObjectIdentity.stable_key):
            groups.setdefault(find(endpoint), []).append(endpoint)
        same_groups = tuple(sorted(
            (tuple(values) for values in groups.values() if len(values) > 1),
            key=lambda values: tuple(
                item.stable_key() for item in values),
        ))

        normalized: dict[
            tuple[ObjectIdentity, ObjectIdentity], set[int]
        ] = {}
        conflict_hashes: set[int] = set()
        for fact, first, second, direction in decoded:
            if direction in {EVENT_TIME_SAME, EVENT_TIME_DIRECTION_UNKNOWN}:
                continue
            before, after = (
                (first, second)
                if direction == EVENT_TIME_BEFORE
                else (second, first)
            )
            edge = (find(before), find(after))
            if edge[0] == edge[1]:
                conflict_hashes.add(fact.assertion_hash)
            normalized.setdefault(edge, set()).add(fact.assertion_hash)

        nodes = {item for edge in normalized for item in edge}
        outgoing: dict[ObjectIdentity, set[ObjectIdentity]] = {
            node: set() for node in nodes
        }
        for before, after in normalized:
            outgoing[before].add(after)
        components = _strongly_connected_components(nodes, outgoing)
        component_by_node = {
            node: component_index
            for component_index, component in enumerate(components)
            for node in component
        }
        component_sizes = tuple(len(component) for component in components)
        for edge, hashes in normalized.items():
            component_index = component_by_node[edge[0]]
            if (component_index == component_by_node[edge[1]]
                    and (component_sizes[component_index] > 1
                         or edge[0] == edge[1])):
                conflict_hashes.update(hashes)

        before_edges = tuple(sorted(
            normalized,
            key=lambda edge: (
                edge[0].stable_key(), edge[1].stable_key()),
        ))
        unknown = tuple(sorted(
            unknown_relations, key=ObjectIdentity.stable_key))
        if conflict_hashes:
            status = EVENT_TIME_CONFLICTED
        elif unknown:
            status = EVENT_TIME_UNKNOWN
        else:
            status = EVENT_TIME_CONSISTENT
        return EventTimeVerificationResult(
            status,
            fact_set,
            before_edges,
            same_groups,
            unknown,
            tuple(sorted(conflict_hashes)),
            tuple(sorted(set(detail_keys))),
        )


__all__ = [
    "EVENT_TIME_AFTER",
    "EVENT_TIME_BEFORE",
    "EVENT_TIME_CONFLICTED",
    "EVENT_TIME_CONSISTENT",
    "EVENT_TIME_DIRECTION_UNKNOWN",
    "EVENT_TIME_EMPTY",
    "EVENT_TIME_SAME",
    "EVENT_TIME_UNKNOWN",
    "EventTimeFactIndex",
    "EventTimeFactSet",
    "EventTimeRelationResolver",
    "EventTimeVerificationResult",
    "EventTimeVerifier",
    "ResolvedEventTimeRelation",
]
