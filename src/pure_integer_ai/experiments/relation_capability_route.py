"""训练后 relation capability route 的纯整数桥接合同。

该模块只把 active Core predicate 映射到已冻结的 D-02C relation kind，并
为同一次 QueryState 提供 route binding/evidence 所需的可逆整数身份。它
不读取课程正文、不执行 legacy mapper，也不把能力存在升级为事实支持。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
)


ROUTE_VERSION = 91540
ROUTE_NAMESPACE = 91541
ROUTE_BINDING_FIELD = 1
ROUTE_EVIDENCE_FIELD = 2
ROUTE_PRODUCER_FIELD = 3
ROUTE_CONSUMER_FIELD = 4
ROUTE_GENERATION_FIELD = 5

# D-02C relation kinds are stable integer coordinates. Adding an unknown
# predicate must fail closed instead of silently gaining a production route.
REGISTERED_RELATION_KINDS = tuple(range(1, 15))


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} 必须是 tuple")
    if not value:
        raise ValueError(f"{label} 不能为空")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} 只能包含非负严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


@dataclass(frozen=True, slots=True)
class RelationCapabilityRoute:
    """一个 active relation 的四段能力闭合记录。"""

    relation_kind: int
    predicate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    role_keys: tuple[tuple[int, ...], ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    generation_registered: int

    def __post_init__(self) -> None:
        if (type(self.relation_kind) is not int
                or self.relation_kind not in REGISTERED_RELATION_KINDS):
            raise ValueError("RelationCapabilityRoute.relation_kind 未注册")
        for label in (
                "predicate_key", "proposition_key", "source_ref",
                "scope_key"):
            _key(getattr(self, label),
                 label=f"RelationCapabilityRoute.{label}")
        if (type(self.role_keys) is not tuple or not self.role_keys
                or any(type(item) is not tuple or not item
                       for item in self.role_keys)
                or self.role_keys != tuple(sorted(set(self.role_keys)))):
            raise ValueError(
                "RelationCapabilityRoute.role_keys 必须排序去重")
        if self.generation_registered not in {0, 1}:
            raise ValueError(
                "RelationCapabilityRoute.generation_registered 必须是 0 或 1")
        expected = authored_relation_identity(
            self.relation_kind).stable_key()
        if self.predicate_key != expected:
            raise ValueError("relation kind 与 predicate identity 不一致")

    @property
    def binding_key(self) -> tuple[int, ...]:
        return (ROUTE_NAMESPACE, ROUTE_BINDING_FIELD, self.relation_kind)

    @property
    def producer_key(self) -> tuple[int, ...]:
        return (ROUTE_NAMESPACE, ROUTE_PRODUCER_FIELD, self.relation_kind)

    @property
    def consumer_key(self) -> tuple[int, ...]:
        return (ROUTE_NAMESPACE, ROUTE_CONSUMER_FIELD, self.relation_kind)

    @property
    def generation_key(self) -> tuple[int, ...]:
        return (
            ROUTE_NAMESPACE,
            ROUTE_GENERATION_FIELD,
            self.relation_kind,
            self.generation_registered,
        )

    @property
    def evidence_key(self) -> tuple[int, ...]:
        return (
            ROUTE_VERSION,
            ROUTE_EVIDENCE_FIELD,
            *_pack(self.stable_key()),
        )

    def stable_key(self) -> tuple[int, ...]:
        result = [
            ROUTE_VERSION,
            self.relation_kind,
            *_pack(self.predicate_key),
            *_pack(self.proposition_key),
            *_pack(self.source_ref),
            *_pack(self.scope_key),
            self.generation_registered,
            len(self.role_keys),
        ]
        for role in self.role_keys:
            result.extend(_pack(role))
        return tuple(result)

    def trace(self) -> dict[str, object]:
        """仅投影整数 route，供共同查询 trace 使用。"""
        return {
            "route": list(self.stable_key()),
            "relation_kind": self.relation_kind,
            "predicate": list(self.predicate_key),
            "proposition": list(self.proposition_key),
            "roles": [list(item) for item in self.role_keys],
            "source_ref": list(self.source_ref),
            "scope": list(self.scope_key),
            "producer": list(self.producer_key),
            "consumer": list(self.consumer_key),
            "generation": list(self.generation_key),
            "generation_registered": self.generation_registered,
        }


class RelationCapabilityRouteRegistry:
    """按 active relation facts 建立确定性 route registry。"""

    def __init__(
            self,
            facts: tuple[object, ...],
            *,
            scope_keys: dict[tuple[int, ...], tuple[int, ...]],
            generation_keys: frozenset[tuple[int, ...]] = frozenset(),
            ) -> None:
        if type(facts) is not tuple:
            raise TypeError("relation facts 必须是 tuple")
        if type(scope_keys) is not dict:
            raise TypeError("relation scope keys 必须是 dict")
        routes: dict[tuple[int, ...], RelationCapabilityRoute] = {}
        predicate_kinds = {
            authored_relation_identity(kind).stable_key(): kind
            for kind in REGISTERED_RELATION_KINDS
        }
        for fact in facts:
            predicate = fact.predicate.stable_key()
            relation_kind = predicate_kinds.get(predicate)
            if relation_kind is None:
                raise ValueError(
                    "active Core predicate 没有已登记能力路由")
            proposition = fact.proposition.stable_key()
            scope_key = scope_keys.get(proposition)
            if scope_key is None:
                raise ValueError(
                    "active Core relation route 缺少 H-00 scope")
            roles = tuple(sorted({
                item.role.stable_key() for item in fact.bindings
            }))
            route = RelationCapabilityRoute(
                relation_kind,
                predicate,
                proposition,
                roles,
                (fact.source_hash,),
                scope_key,
                int(proposition in generation_keys),
            )
            prior = routes.get(proposition)
            if prior is not None and prior != route:
                raise ValueError(
                    "同一 proposition 的 relation route 发生漂移")
            routes[proposition] = route
        if len(routes) != len(facts):
            raise ValueError("active Core relation route 数量不闭合")
        self._routes = tuple(sorted(
            routes.values(), key=lambda item: item.stable_key()))
        self._by_proposition = {
            item.proposition_key: item for item in self._routes
        }

    def for_proposition(
            self,
            proposition_key: tuple[int, ...],
            ) -> RelationCapabilityRoute | None:
        _key(proposition_key, label="route proposition_key")
        return self._by_proposition.get(proposition_key)

    def all(self) -> tuple[RelationCapabilityRoute, ...]:
        return self._routes

    def stable_key(self) -> tuple[int, ...]:
        return (
            ROUTE_VERSION,
            len(self._routes),
            *(item for route in self._routes
              for item in _pack(route.stable_key())),
        )


__all__ = [
    "REGISTERED_RELATION_KINDS",
    "ROUTE_BINDING_FIELD",
    "ROUTE_CONSUMER_FIELD",
    "ROUTE_EVIDENCE_FIELD",
    "ROUTE_GENERATION_FIELD",
    "ROUTE_NAMESPACE",
    "ROUTE_PRODUCER_FIELD",
    "ROUTE_VERSION",
    "RelationCapabilityRoute",
    "RelationCapabilityRouteRegistry",
]
