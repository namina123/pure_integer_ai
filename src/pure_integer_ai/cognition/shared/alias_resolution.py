"""R-01 strict alias、方向同指和目标语言 Representation 的 typed 消费。

关系、schema、Role、步骤指令和选择状态全部由调用方注入。本模块只消费 R-00
active typed fact，不读取 legacy PURE_ALIAS、MARK_LANG、surface 字符串或稳定排序 winner。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_REPRESENTATION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    ActiveRelationClosureFact,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(key), *key


def _identity(
        value: ObjectIdentity, *, label: str, kind: int | None = None,
        ) -> ObjectIdentity:
    """核验一等对象及可选宿主对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if kind is not None and value.object_kind != kind:
        raise ValueError(f"{label} 对象类型不匹配")
    return value


def _identity_tuple(
        values: tuple[ObjectIdentity, ...], *, label: str, kind: int,
        ) -> tuple[ObjectIdentity, ...]:
    """核验非空、无重复且同型的一等身份 tuple。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} 必须是非空 tuple")
    for value in values:
        _identity(value, label=label, kind=kind)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} 不得重复")
    return tuple(sorted(values, key=ObjectIdentity.stable_key))


def _fact_key(fact: ActiveRelationClosureFact) -> tuple[int, ...]:
    """返回 active relation fact 的命题、Evidence 和 H-04 完整键。"""
    if not isinstance(fact, ActiveRelationClosureFact):
        raise TypeError("active relation fact 类型错误")
    result = [
        *_packed(fact.proposition.proposition.stable_key()),
        *_packed(fact.hypothesis.stable_key()),
        len(fact.evidence_keys),
    ]
    for evidence in fact.evidence_keys:
        result.extend(_packed(evidence))
    result.extend(_packed(fact.decision_key))
    result.append(1 if fact.read_only_recovered else 0)
    return tuple(result)


class AliasRouteSearchExhausted(RuntimeError):
    """active route 搜索无法在调用方预算内证明候选集合完整。"""


@dataclass(frozen=True)
class AliasRouteSearchBudget:
    """限制 active fact、路径状态和完整 route 的搜索规模。"""

    max_facts: int
    max_states: int
    max_routes: int

    def __post_init__(self) -> None:
        assert_int(
            self.max_facts,
            self.max_states,
            self.max_routes,
            _where="AliasRouteSearchBudget",
        )
        if any(type(value) is not int or value <= 0 for value in (
                self.max_facts, self.max_states, self.max_routes)):
            raise ValueError("alias route 搜索预算必须全部为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回三项调用方预算。"""
        return self.max_facts, self.max_states, self.max_routes


@dataclass(frozen=True)
class AliasResolutionProtocol:
    """注入三类 relation/schema/Role、route 指令和选择状态。"""

    alias_relation: ObjectIdentity
    alias_schemas: tuple[ObjectIdentity, ...]
    alias_left_role: ObjectIdentity
    alias_right_role: ObjectIdentity
    alias_step: ObjectIdentity
    refers_relation: ObjectIdentity
    refers_schemas: tuple[ObjectIdentity, ...]
    refers_from_role: ObjectIdentity
    refers_to_role: ObjectIdentity
    refers_step: ObjectIdentity
    realizes_relation: ObjectIdentity
    realizes_schemas: tuple[ObjectIdentity, ...]
    realizes_bearer_role: ObjectIdentity
    realizes_representation_role: ObjectIdentity
    realizes_branch_role: ObjectIdentity
    realizes_step: ObjectIdentity
    selected_outcome: ObjectIdentity
    ambiguous_outcome: ObjectIdentity
    missing_outcome: ObjectIdentity

    def __post_init__(self) -> None:
        relations = (
            self.alias_relation,
            self.refers_relation,
            self.realizes_relation,
        )
        for relation in relations:
            _identity(relation, label="alias protocol relation", kind=OBJECT_CONCEPT)
        if len(set(relations)) != len(relations):
            raise ValueError("alias/refers/realizes relation 必须互不相同")
        for name in ("alias_schemas", "refers_schemas", "realizes_schemas"):
            object.__setattr__(self, name, _identity_tuple(
                getattr(self, name),
                label=name,
                kind=OBJECT_STRUCTURE_CONCEPT,
            ))
        roles = (
            self.alias_left_role,
            self.alias_right_role,
            self.refers_from_role,
            self.refers_to_role,
            self.realizes_bearer_role,
            self.realizes_representation_role,
            self.realizes_branch_role,
        )
        for role in roles:
            _identity(role, label="alias protocol Role", kind=OBJECT_ROLE)
        if self.alias_left_role == self.alias_right_role:
            raise ValueError("alias 左右 Role 必须不同")
        if self.refers_from_role == self.refers_to_role:
            raise ValueError("refers 起终 Role 必须不同")
        if len({
                self.realizes_bearer_role,
                self.realizes_representation_role,
                self.realizes_branch_role,
                }) != 3:
            raise ValueError("realizes 三个 Role 必须互不相同")
        instructions = self.step_instructions() + self.outcomes()
        for instruction in instructions:
            _identity(
                instruction,
                label="alias protocol instruction",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
        if len(set(instructions)) != len(instructions):
            raise ValueError("route 指令和 outcome 必须互不相同")

    def step_instructions(self) -> tuple[ObjectIdentity, ...]:
        """返回 alias、refers 和 realizes 三个最小步骤指令。"""
        return self.alias_step, self.refers_step, self.realizes_step

    def surface_prefix_steps(self) -> tuple[ObjectIdentity, ...]:
        """返回 surface route 可由调用方选择的 alias/refers 前缀步骤。"""
        return self.alias_step, self.refers_step

    def outcomes(self) -> tuple[ObjectIdentity, ...]:
        """返回 selected、ambiguous 和 missing 三个注入状态。"""
        return (
            self.selected_outcome,
            self.ambiguous_outcome,
            self.missing_outcome,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 relation/schema/Role/指令和 outcome 身份。"""
        result: list[int] = [*_packed(self.alias_relation.stable_key())]
        for schemas in (
                self.alias_schemas,
                self.refers_schemas,
                self.realizes_schemas):
            result.append(len(schemas))
            for schema in schemas:
                result.extend(_packed(schema.stable_key()))
        for identity in (
                self.alias_left_role,
                self.alias_right_role,
                self.alias_step,
                self.refers_relation,
                self.refers_from_role,
                self.refers_to_role,
                self.refers_step,
                self.realizes_relation,
                self.realizes_bearer_role,
                self.realizes_representation_role,
                self.realizes_branch_role,
                self.realizes_step,
                self.selected_outcome,
                self.ambiguous_outcome,
                self.missing_outcome):
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AliasRouteStep:
    """一条 active typed relation fact 在 route 中的有向采用。"""

    instruction: ObjectIdentity
    fact: ActiveRelationClosureFact
    source: ObjectIdentity
    target: ObjectIdentity

    def __post_init__(self) -> None:
        _identity(
            self.instruction,
            label="route step instruction",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        if not isinstance(self.fact, ActiveRelationClosureFact):
            raise TypeError("route step fact 类型错误")
        _identity(self.source, label="route step source")
        _identity(self.target, label="route step target")
        if self.source == self.target:
            raise ValueError("route step 不得自环")
        if not self.fact.evidence_keys or not self.fact.decision_key:
            raise ValueError("route step 必须保留 active Evidence/H-04 trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回指令、端点、Proposition 和 active 归因完整键。"""
        hypothesis = self.fact.hypothesis.stable_key()
        result = [
            *_packed(self.instruction.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.target.stable_key()),
            *_packed(self.fact.proposition.proposition.stable_key()),
            *_packed(hypothesis),
            len(self.fact.evidence_keys),
        ]
        for evidence in self.fact.evidence_keys:
            result.extend(_packed(evidence))
        result.extend(_packed(self.fact.decision_key))
        result.append(1 if self.fact.read_only_recovered else 0)
        return tuple(result)


@dataclass(frozen=True)
class AliasRoute:
    """从 query 起点到候选 referent 或 Representation 的连续无环 route。"""

    origin: ObjectIdentity
    steps: tuple[AliasRouteStep, ...]

    def __post_init__(self) -> None:
        _identity(self.origin, label="alias route origin")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("alias route 必须包含至少一个 step")
        if any(not isinstance(item, AliasRouteStep) for item in self.steps):
            raise TypeError("alias route steps 类型错误")
        current = self.origin
        nodes = [self.origin]
        facts: list[ObjectIdentity] = []
        for step in self.steps:
            if step.source != current:
                raise ValueError("alias route step 端点不连续")
            current = step.target
            nodes.append(current)
            facts.append(step.fact.proposition.proposition)
        if len(set(nodes)) != len(nodes):
            raise ValueError("alias route 不得含对象环")
        if len(set(facts)) != len(facts):
            raise ValueError("alias route 不得重复采用同一 Proposition")

    @property
    def target(self) -> ObjectIdentity:
        """返回 route 最后一个有向 step 的目标对象。"""
        return self.steps[-1].target

    def stable_key(self) -> tuple[int, ...]:
        """返回 origin 和全部 active fact step 的完整键。"""
        result = [*_packed(self.origin.stable_key()), len(self.steps)]
        for step in self.steps:
            result.extend(_packed(step.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ReferenceRouteDiscovery:
    """从全部相关 active fact 完整发现的方向同指 route 集。"""

    protocol: AliasResolutionProtocol
    origin: ObjectIdentity
    target_kinds: tuple[int, ...]
    budget: AliasRouteSearchBudget
    explored_states: int
    considered_facts: tuple[ActiveRelationClosureFact, ...]
    routes: tuple[AliasRoute, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, AliasResolutionProtocol):
            raise TypeError("reference discovery protocol 类型错误")
        _identity(self.origin, label="reference discovery origin")
        if not isinstance(self.target_kinds, tuple) or not self.target_kinds:
            raise ValueError("reference discovery target_kinds 必须是非空 tuple")
        assert_int(*self.target_kinds, _where="ReferenceRouteDiscovery.target_kinds")
        if (any(type(item) is not int or item <= 0 for item in self.target_kinds)
                or len(set(self.target_kinds)) != len(self.target_kinds)):
            raise ValueError("reference discovery target_kinds 必须是无重复正整数")
        object.__setattr__(self, "target_kinds", tuple(sorted(self.target_kinds)))
        _validate_discovery_common(
            self.origin,
            self.budget,
            self.explored_states,
            self.considered_facts,
            self.routes,
        )
        if any(route.target.object_kind not in self.target_kinds
               for route in self.routes):
            raise ValueError("reference discovery route 终点类型未被请求")
        AliasResolutionSelector(self.protocol).resolve_reference(self.query())

    def query(self) -> "ReferenceResolutionQuery":
        """返回 selector 可消费的完整方向同指 query。"""
        return ReferenceResolutionQuery(self.origin, self.routes)

    def stable_key(self) -> tuple[int, ...]:
        """返回预算、搜索使用量、相关 active fact 和完整 route。"""
        result = [
            *_packed(self.protocol.stable_key()),
            *_packed(self.origin.stable_key()),
            len(self.target_kinds),
            *self.target_kinds,
            *self.budget.stable_key(),
            self.explored_states,
            len(self.considered_facts),
        ]
        for fact in self.considered_facts:
            result.extend(_packed(_fact_key(fact)))
        result.append(len(self.routes))
        for route in self.routes:
            result.extend(_packed(route.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SurfaceRouteDiscovery:
    """从全部相关 active fact 完整发现的目标语言 Representation route 集。"""

    protocol: AliasResolutionProtocol
    origin: ObjectIdentity
    branch: ObjectIdentity
    budget: AliasRouteSearchBudget
    explored_states: int
    considered_facts: tuple[ActiveRelationClosureFact, ...]
    routes: tuple[AliasRoute, ...]
    allowed_prefix_steps: tuple[ObjectIdentity, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, AliasResolutionProtocol):
            raise TypeError("surface discovery protocol 类型错误")
        _identity(self.origin, label="surface discovery origin")
        _identity(
            self.branch,
            label="surface discovery branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        allowed = self.allowed_prefix_steps
        if allowed is None:
            allowed = self.protocol.surface_prefix_steps()
        if not isinstance(allowed, tuple):
            raise TypeError("surface discovery allowed_prefix_steps 必须是 tuple")
        if len(set(allowed)) != len(allowed):
            raise ValueError("surface discovery prefix step 不得重复")
        registered = set(self.protocol.surface_prefix_steps())
        for instruction in allowed:
            _identity(
                instruction,
                label="surface discovery prefix step",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
            if instruction not in registered:
                raise ValueError("surface discovery prefix step 未在 protocol 注册")
        object.__setattr__(self, "allowed_prefix_steps", tuple(sorted(
            allowed, key=ObjectIdentity.stable_key)))
        _validate_discovery_common(
            self.origin,
            self.budget,
            self.explored_states,
            self.considered_facts,
            self.routes,
        )
        if any(
                step.instruction not in set(self.allowed_prefix_steps)
                for route in self.routes
                for step in route.steps[:-1]):
            raise ValueError("surface discovery route 使用了未允许的前缀步骤")
        AliasResolutionSelector(self.protocol).select_surface(self.query())

    def query(self) -> "SurfaceRealizationQuery":
        """返回 selector 可消费的完整目标语言 surface query。"""
        return SurfaceRealizationQuery(self.origin, self.branch, self.routes)

    def stable_key(self) -> tuple[int, ...]:
        """返回预算、搜索使用量、相关 active fact 和完整 route。"""
        result = [
            *_packed(self.protocol.stable_key()),
            *_packed(self.origin.stable_key()),
            *_packed(self.branch.stable_key()),
            *self.budget.stable_key(),
            self.explored_states,
            len(self.allowed_prefix_steps),
            *(value for instruction in self.allowed_prefix_steps
              for value in _packed(instruction.stable_key())),
            len(self.considered_facts),
        ]
        for fact in self.considered_facts:
            result.extend(_packed(_fact_key(fact)))
        result.append(len(self.routes))
        for route in self.routes:
            result.extend(_packed(route.stable_key()))
        return tuple(result)


def _validate_discovery_common(
        origin: ObjectIdentity,
        budget: AliasRouteSearchBudget,
        explored_states: int,
        considered_facts: tuple[ActiveRelationClosureFact, ...],
        routes: tuple[AliasRoute, ...],
        ) -> None:
    """核验两类 discovery 共用的预算、事实覆盖和 route 唯一性。"""
    if not isinstance(budget, AliasRouteSearchBudget):
        raise TypeError("route discovery budget 类型错误")
    assert_int(explored_states, _where="route discovery explored_states")
    if (type(explored_states) is not int
            or explored_states <= 0
            or explored_states > budget.max_states):
        raise ValueError("route discovery explored_states 超出预算")
    if not isinstance(considered_facts, tuple) or any(
            not isinstance(item, ActiveRelationClosureFact)
            for item in considered_facts):
        raise TypeError("route discovery considered_facts 类型错误")
    if len(considered_facts) > budget.max_facts:
        raise ValueError("route discovery active fact 数超过预算")
    propositions = tuple(
        item.proposition.proposition for item in considered_facts)
    if len(set(propositions)) != len(propositions):
        raise ValueError("route discovery considered_facts 不得重复 Proposition")
    expected_facts = tuple(sorted(
        considered_facts,
        key=lambda item: item.proposition.proposition.stable_key(),
    ))
    if considered_facts != expected_facts:
        raise ValueError("route discovery considered_facts 必须规范排序")
    if not isinstance(routes, tuple) or any(
            not isinstance(item, AliasRoute) for item in routes):
        raise TypeError("route discovery routes 类型错误")
    if len(routes) > budget.max_routes:
        raise ValueError("route discovery route 数超过预算")
    if any(item.origin != origin for item in routes):
        raise ValueError("route discovery route 必须绑定 origin")
    route_keys = tuple(item.stable_key() for item in routes)
    if len(set(route_keys)) != len(route_keys):
        raise ValueError("route discovery 不得重复 route")
    if routes != tuple(sorted(routes, key=AliasRoute.stable_key)):
        raise ValueError("route discovery routes 必须规范排序")
    considered = set(propositions)
    if any(
            step.fact.proposition.proposition not in considered
            for route in routes for step in route.steps):
        raise ValueError("route discovery 使用了未列入 considered_facts 的事实")


@dataclass(frozen=True)
class ReferenceResolutionQuery:
    """对 occurrence/sense/atom 等起点保留全部同指候选 route。"""

    origin: ObjectIdentity
    routes: tuple[AliasRoute, ...]

    def __post_init__(self) -> None:
        _identity(self.origin, label="reference query origin")
        if not isinstance(self.routes, tuple) or any(
                not isinstance(item, AliasRoute) for item in self.routes):
            raise TypeError("reference query routes 类型错误")
        if any(item.origin != self.origin for item in self.routes):
            raise ValueError("reference route 必须绑定 query origin")


@dataclass(frozen=True)
class SurfaceRealizationQuery:
    """为语义目标和目标 LanguageBranch 提供全部 Representation route。"""

    origin: ObjectIdentity
    branch: ObjectIdentity
    routes: tuple[AliasRoute, ...]

    def __post_init__(self) -> None:
        _identity(self.origin, label="surface query origin")
        _identity(
            self.branch,
            label="surface query branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        if not isinstance(self.routes, tuple) or any(
                not isinstance(item, AliasRoute) for item in self.routes):
            raise TypeError("surface query routes 类型错误")
        if any(item.origin != self.origin for item in self.routes):
            raise ValueError("surface route 必须绑定 query origin")


@dataclass(frozen=True)
class AliasResolutionOption:
    """同一最终对象的全部独立 active Evidence route。"""

    value: ObjectIdentity
    routes: tuple[AliasRoute, ...]

    def __post_init__(self) -> None:
        _identity(self.value, label="resolution option value")
        if not isinstance(self.routes, tuple) or not self.routes:
            raise ValueError("resolution option 必须含 route")
        if any(not isinstance(item, AliasRoute) for item in self.routes):
            raise TypeError("resolution option routes 类型错误")
        if any(item.target != self.value for item in self.routes):
            raise ValueError("resolution option route 终点不一致")
        if len({item.stable_key() for item in self.routes}) != len(self.routes):
            raise ValueError("resolution option 不得重复 route")
        object.__setattr__(self, "routes", tuple(sorted(
            self.routes, key=AliasRoute.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回最终对象和全部 active route 完整键。"""
        result = [*_packed(self.value.stable_key()), len(self.routes)]
        for route in self.routes:
            result.extend(_packed(route.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AliasResolutionResult:
    """唯一、歧义或缺失选择，并保留所有 Representation/referent 选项。"""

    protocol: AliasResolutionProtocol
    outcome: ObjectIdentity
    origin: ObjectIdentity
    branch: ObjectIdentity | None
    options: tuple[AliasResolutionOption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, AliasResolutionProtocol):
            raise TypeError("alias resolution protocol 类型错误")
        _identity(
            self.outcome,
            label="alias resolution outcome",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        _identity(self.origin, label="alias resolution origin")
        if self.branch is not None:
            _identity(
                self.branch,
                label="alias resolution branch",
                kind=OBJECT_LANGUAGE_BRANCH,
            )
        if not isinstance(self.options, tuple) or any(
                not isinstance(item, AliasResolutionOption)
                for item in self.options):
            raise TypeError("alias resolution options 类型错误")
        values = tuple(item.value for item in self.options)
        if len(set(values)) != len(values):
            raise ValueError("alias resolution option value 不得重复")
        if any(
                route.origin != self.origin
                for option in self.options for route in option.routes):
            raise ValueError("alias resolution option route 与 origin 不一致")
        expected = (
            self.protocol.missing_outcome
            if not self.options
            else self.protocol.selected_outcome
            if len(self.options) == 1
            else self.protocol.ambiguous_outcome
        )
        if self.outcome != expected:
            raise ValueError("alias resolution outcome 与 option 数量不一致")
        object.__setattr__(self, "options", tuple(sorted(
            self.options, key=lambda item: item.value.stable_key())))

    @property
    def selected(self) -> AliasResolutionOption | None:
        """唯一选项存在时返回它，否则返回空。"""
        return self.options[0] if len(self.options) == 1 else None

    def stable_key(self) -> tuple[int, ...]:
        """返回 outcome、query 身份和全部候选 route。"""
        result = [
            *_packed(self.protocol.stable_key()),
            *_packed(self.outcome.stable_key()),
            *_packed(self.origin.stable_key()),
            0 if self.branch is None else 1,
        ]
        if self.branch is not None:
            result.extend(_packed(self.branch.stable_key()))
        result.append(len(self.options))
        for option in self.options:
            result.extend(_packed(option.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AliasResolutionProposal:
    """无写入的完整 route discovery 和唯一、歧义或缺失选择提案。"""

    result: AliasResolutionResult
    discovery: ReferenceRouteDiscovery | SurfaceRouteDiscovery

    def __post_init__(self) -> None:
        if not isinstance(self.result, AliasResolutionResult):
            raise TypeError("alias proposal result 类型错误")
        if not isinstance(
                self.discovery,
                (ReferenceRouteDiscovery, SurfaceRouteDiscovery)):
            raise TypeError("alias proposal discovery 类型错误")
        if (self.discovery.protocol != self.result.protocol
                or self.discovery.origin != self.result.origin):
            raise ValueError("alias proposal discovery 与 result query 不一致")
        if isinstance(self.discovery, ReferenceRouteDiscovery):
            if self.result.branch is not None:
                raise ValueError("reference proposal 不得绑定 surface branch")
        elif self.result.branch != self.discovery.branch:
            raise ValueError("surface proposal branch 与 result 不一致")
        routes = tuple(sorted(
            (
                route
                for option in self.result.options
                for route in option.routes
            ),
            key=AliasRoute.stable_key,
        ))
        if routes != self.discovery.routes:
            raise ValueError("alias proposal result 未守恒完整 discovery routes")

    def stable_key(self) -> tuple[int, ...]:
        """返回选择结果和完整 discovery trace。"""
        return (
            *_packed(self.result.stable_key()),
            *_packed(self.discovery.stable_key()),
        )


class AliasResolutionSelector:
    """验证 active relation route，并在多最终对象时显式保留歧义。"""

    def __init__(self, protocol: AliasResolutionProtocol) -> None:
        if not isinstance(protocol, AliasResolutionProtocol):
            raise TypeError("alias selector protocol 类型错误")
        self.protocol = protocol

    def resolve_reference(
            self, query: ReferenceResolutionQuery,
            ) -> AliasResolutionResult:
        """只允许 alias/refers route，按最终 referent 分组而不私选。"""
        if not isinstance(query, ReferenceResolutionQuery):
            raise TypeError("reference query 类型错误")
        valid: list[AliasRoute] = []
        for route in query.routes:
            for step in route.steps:
                kind, _ = self.validate_step(step)
                if kind == self.protocol.realizes_step:
                    raise ValueError("reference route 不得混入 realizes step")
            valid.append(route)
        return self._result(query.origin, None, tuple(valid))

    def select_surface(
            self, query: SurfaceRealizationQuery,
            ) -> AliasResolutionResult:
        """要求末步 realizes 到目标分支 Representation，多词形保持 ambiguous。"""
        if not isinstance(query, SurfaceRealizationQuery):
            raise TypeError("surface query 类型错误")
        valid: list[AliasRoute] = []
        for route in query.routes:
            for index, step in enumerate(route.steps):
                kind, branch = self.validate_step(step)
                final = index == len(route.steps) - 1
                if kind == self.protocol.realizes_step:
                    if not final:
                        raise ValueError("realizes step 必须是 surface route 末步")
                    if branch != query.branch:
                        raise ValueError("surface route realizes 分支与目标不一致")
                    if step.target.object_kind != OBJECT_REPRESENTATION:
                        raise ValueError("surface route 末端必须是 Representation")
                elif final:
                    raise ValueError("surface route 必须以 realizes step 结束")
            valid.append(route)
        return self._result(query.origin, query.branch, tuple(valid))

    def validate_step(
            self, step: AliasRouteStep,
            ) -> tuple[ObjectIdentity, ObjectIdentity | None]:
        """按注入指令核验 relation/schema/Role 和有向端点。"""
        if not isinstance(step, AliasRouteStep):
            raise TypeError("route step 类型错误")
        protocol = self.protocol
        relation = step.fact.proposition.predicate
        schema = step.fact.schema.schema
        if step.instruction == protocol.alias_step:
            if (relation != protocol.alias_relation
                    or schema not in protocol.alias_schemas):
                raise ValueError("alias step 使用了错误 relation/schema")
            left = self._binding(step.fact, protocol.alias_left_role)
            right = self._binding(step.fact, protocol.alias_right_role)
            if {step.source, step.target} != {left, right}:
                raise ValueError("alias step 端点与 typed RoleBinding 不一致")
            return protocol.alias_step, None
        if step.instruction == protocol.refers_step:
            if (relation != protocol.refers_relation
                    or schema not in protocol.refers_schemas):
                raise ValueError("refers step 使用了错误 relation/schema")
            source = self._binding(step.fact, protocol.refers_from_role)
            target = self._binding(step.fact, protocol.refers_to_role)
            if step.source != source or step.target != target:
                raise ValueError("refers step 不得反向或替换 typed 端点")
            return protocol.refers_step, None
        if step.instruction == protocol.realizes_step:
            if (relation != protocol.realizes_relation
                    or schema not in protocol.realizes_schemas):
                raise ValueError("realizes step 使用了错误 relation/schema")
            bearer = self._binding(
                step.fact, protocol.realizes_bearer_role)
            representation = self._binding(
                step.fact, protocol.realizes_representation_role)
            branch = self._binding(
                step.fact, protocol.realizes_branch_role)
            if step.source != bearer or step.target != representation:
                raise ValueError("realizes step 端点与 typed RoleBinding 不一致")
            if branch.object_kind != OBJECT_LANGUAGE_BRANCH:
                raise ValueError("realizes branch Role 必须绑定 LanguageBranch")
            if representation.object_kind != OBJECT_REPRESENTATION:
                raise ValueError("realizes representation Role 类型错误")
            return protocol.realizes_step, branch
        raise ValueError("route step instruction 未在 protocol 注册")

    def steps_for_fact(
            self, fact: ActiveRelationClosureFact,
            ) -> tuple[AliasRouteStep, ...]:
        """把一个 active fact 按注入 relation 分型为可遍历有向步骤。"""
        if not isinstance(fact, ActiveRelationClosureFact):
            raise TypeError("route fact 类型错误")
        protocol = self.protocol
        relation = fact.proposition.predicate
        if relation == protocol.alias_relation:
            left = self._binding(fact, protocol.alias_left_role)
            right = self._binding(fact, protocol.alias_right_role)
            steps = (
                AliasRouteStep(protocol.alias_step, fact, left, right),
                AliasRouteStep(protocol.alias_step, fact, right, left),
            )
        elif relation == protocol.refers_relation:
            steps = (AliasRouteStep(
                protocol.refers_step,
                fact,
                self._binding(fact, protocol.refers_from_role),
                self._binding(fact, protocol.refers_to_role),
            ),)
        elif relation == protocol.realizes_relation:
            steps = (AliasRouteStep(
                protocol.realizes_step,
                fact,
                self._binding(fact, protocol.realizes_bearer_role),
                self._binding(fact, protocol.realizes_representation_role),
            ),)
        else:
            raise ValueError("active fact relation 未在 alias protocol 注册")
        for step in steps:
            self.validate_step(step)
        return steps

    @staticmethod
    def _binding(
            fact: ActiveRelationClosureFact,
            role: ObjectIdentity,
            ) -> ObjectIdentity:
        """按完整 Role 身份读取唯一原子命题 filler。"""
        matches = tuple(
            item.filler for item in fact.proposition.bindings
            if item.role == role
        )
        if len(matches) != 1:
            raise ValueError("route fact 必须为协议 Role 提供唯一 filler")
        return matches[0]

    def _result(
            self,
            origin: ObjectIdentity,
            branch: ObjectIdentity | None,
            routes: tuple[AliasRoute, ...],
            ) -> AliasResolutionResult:
        """按最终对象聚合独立 route，并产生注入式三态 outcome。"""
        grouped: dict[ObjectIdentity, list[AliasRoute]] = {}
        for route in routes:
            grouped.setdefault(route.target, []).append(route)
        options = tuple(
            AliasResolutionOption(value, tuple(grouped[value]))
            for value in sorted(grouped, key=ObjectIdentity.stable_key)
        )
        if not options:
            outcome = self.protocol.missing_outcome
        elif len(options) == 1:
            outcome = self.protocol.selected_outcome
        else:
            outcome = self.protocol.ambiguous_outcome
        return AliasResolutionResult(
            self.protocol, outcome, origin, branch, options)


class ActiveAliasRouteFinder:
    """从 R-00 当前 active facts 完整发现 alias/refers/realizes route。"""

    def __init__(
            self,
            consumer: ActiveRelationClosureConsumer,
            selector: AliasResolutionSelector,
            ) -> None:
        if not isinstance(consumer, ActiveRelationClosureConsumer):
            raise TypeError("alias route finder consumer 类型错误")
        if not isinstance(selector, AliasResolutionSelector):
            raise TypeError("alias route finder selector 类型错误")
        self.consumer = consumer
        self.selector = selector

    def discover_reference(
            self,
            origin: ObjectIdentity,
            target_kinds: tuple[int, ...],
            budget: AliasRouteSearchBudget,
            ) -> ReferenceRouteDiscovery:
        """枚举至首个目标对象类型的全部 alias/refers 简单路径。"""
        _identity(origin, label="reference discovery origin")
        if not isinstance(target_kinds, tuple) or not target_kinds:
            raise ValueError("reference target_kinds 必须是非空 tuple")
        assert_int(*target_kinds, _where="ActiveAliasRouteFinder.target_kinds")
        if (any(type(item) is not int or item <= 0 for item in target_kinds)
                or len(set(target_kinds)) != len(target_kinds)):
            raise ValueError("reference target_kinds 必须是无重复正整数")
        self._budget(budget)
        routes, explored, facts = self._walk_reference(
            origin,
            frozenset(target_kinds),
            budget,
        )
        return ReferenceRouteDiscovery(
            self.selector.protocol,
            origin,
            tuple(sorted(target_kinds)),
            budget,
            explored,
            facts,
            routes,
        )

    def discover_surface(
            self,
            origin: ObjectIdentity,
            branch: ObjectIdentity,
            budget: AliasRouteSearchBudget,
            allowed_prefix_steps: tuple[ObjectIdentity, ...] | None = None,
            ) -> SurfaceRouteDiscovery:
        """按调用方允许的前缀步骤枚举全部 branch-scoped realizes 路径。"""
        _identity(origin, label="surface discovery origin")
        _identity(
            branch,
            label="surface discovery branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        self._budget(budget)
        protocol = self.selector.protocol
        allowed = (
            protocol.surface_prefix_steps()
            if allowed_prefix_steps is None else allowed_prefix_steps
        )
        if not isinstance(allowed, tuple):
            raise TypeError("surface allowed_prefix_steps 必须是 tuple")
        if len(set(allowed)) != len(allowed):
            raise ValueError("surface allowed_prefix_steps 不得重复")
        registered = set(protocol.surface_prefix_steps())
        for instruction in allowed:
            _identity(
                instruction,
                label="surface allowed prefix step",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
            if instruction not in registered:
                raise ValueError("surface prefix step 未在 alias protocol 注册")
        normalized = tuple(sorted(allowed, key=ObjectIdentity.stable_key))
        routes, explored, facts = self._walk_surface(
            origin, branch, budget, frozenset(normalized))
        return SurfaceRouteDiscovery(
            protocol,
            origin,
            branch,
            budget,
            explored,
            facts,
            routes,
            normalized,
        )

    @staticmethod
    def _budget(budget: AliasRouteSearchBudget) -> None:
        """要求调用方显式提供合法搜索预算。"""
        if not isinstance(budget, AliasRouteSearchBudget):
            raise TypeError("alias route finder budget 类型错误")

    def _steps_from(
            self,
            source: ObjectIdentity,
            *,
            include_realizes: bool,
            budget: AliasRouteSearchBudget,
            considered: dict[ObjectIdentity, ActiveRelationClosureFact],
            cache: dict[ObjectIdentity, tuple[AliasRouteStep, ...]],
            allowed_prefix_steps: frozenset[ObjectIdentity] | None = None,
            ) -> tuple[AliasRouteStep, ...]:
        """沿当前 filler 的图反向索引读取局部 active fact，并缓存只读邻接。"""
        cached = cache.get(source)
        if cached is not None:
            return cached
        protocol = self.selector.protocol
        lookups = [
            (
                protocol.alias_relation,
                protocol.alias_schemas,
                protocol.alias_left_role,
                protocol.alias_step,
            ),
            (
                protocol.alias_relation,
                protocol.alias_schemas,
                protocol.alias_right_role,
                protocol.alias_step,
            ),
            (
                protocol.refers_relation,
                protocol.refers_schemas,
                protocol.refers_from_role,
                protocol.refers_step,
            ),
        ]
        if allowed_prefix_steps is not None:
            lookups = [
                item for item in lookups if item[3] in allowed_prefix_steps]
        if include_realizes:
            lookups.append((
                protocol.realizes_relation,
                protocol.realizes_schemas,
                protocol.realizes_bearer_role,
                protocol.realizes_step,
            ))
        local: dict[ObjectIdentity, ActiveRelationClosureFact] = {}
        for relation, schemas, role, _ in lookups:
            for fact in self.consumer.lookup_role_filler(
                    relation, schemas, role, source):
                proposition = fact.proposition.proposition
                existing = local.get(proposition)
                if existing is not None and existing != fact:
                    raise ValueError("局部索引恢复出冲突 active fact")
                local[proposition] = fact
                known = considered.get(proposition)
                if known is not None and known != fact:
                    raise ValueError("同一 Proposition 的 active fact 前后不一致")
                considered[proposition] = fact
                if len(considered) > budget.max_facts:
                    raise AliasRouteSearchExhausted(
                        "可达 active alias fact 数超过调用方预算")
        steps = tuple(sorted((
            step
            for fact in local.values()
            for step in self.selector.steps_for_fact(fact)
            if step.source == source
        ), key=AliasRouteStep.stable_key))
        cache[source] = steps
        return steps

    def _walk_reference(
            self,
            origin: ObjectIdentity,
            target_kinds: frozenset[int],
            budget: AliasRouteSearchBudget,
            ) -> tuple[
                tuple[AliasRoute, ...], int,
                tuple[ActiveRelationClosureFact, ...],
            ]:
        """搜索方向同指路径；到达首个请求类型后形成终点且不继续越过。"""
        stack: list[tuple[
            ObjectIdentity, tuple[AliasRouteStep, ...], frozenset[ObjectIdentity]
        ]] = [(origin, (), frozenset({origin}))]
        routes: dict[tuple[int, ...], AliasRoute] = {}
        considered: dict[ObjectIdentity, ActiveRelationClosureFact] = {}
        cache: dict[ObjectIdentity, tuple[AliasRouteStep, ...]] = {}
        explored = 0
        while stack:
            if explored >= budget.max_states:
                raise AliasRouteSearchExhausted(
                    "reference route 状态数超过调用方预算")
            current, steps, visited = stack.pop()
            explored += 1
            pending = []
            for step in self._steps_from(
                    current,
                    include_realizes=False,
                    budget=budget,
                    considered=considered,
                    cache=cache):
                kind, _ = self.selector.validate_step(step)
                if kind == self.selector.protocol.realizes_step:
                    continue
                if step.target in visited:
                    continue
                next_steps = (*steps, step)
                if step.target.object_kind in target_kinds:
                    self._add_route(
                        routes, AliasRoute(origin, next_steps), budget)
                    continue
                pending.append((
                    step.target,
                    next_steps,
                    visited | frozenset({step.target}),
                ))
            stack.extend(reversed(pending))
        facts = tuple(
            considered[key]
            for key in sorted(considered, key=ObjectIdentity.stable_key)
        )
        return tuple(routes[key] for key in sorted(routes)), explored, facts

    def _walk_surface(
            self,
            origin: ObjectIdentity,
            branch: ObjectIdentity,
            budget: AliasRouteSearchBudget,
            allowed_prefix_steps: frozenset[ObjectIdentity],
            ) -> tuple[
                tuple[AliasRoute, ...], int,
                tuple[ActiveRelationClosureFact, ...],
            ]:
        """搜索 alias/refers 前缀和 branch-scoped realizes 末步的全部简单路径。"""
        stack: list[tuple[
            ObjectIdentity, tuple[AliasRouteStep, ...], frozenset[ObjectIdentity]
        ]] = [(origin, (), frozenset({origin}))]
        routes: dict[tuple[int, ...], AliasRoute] = {}
        considered: dict[ObjectIdentity, ActiveRelationClosureFact] = {}
        cache: dict[ObjectIdentity, tuple[AliasRouteStep, ...]] = {}
        explored = 0
        while stack:
            if explored >= budget.max_states:
                raise AliasRouteSearchExhausted(
                    "surface route 状态数超过调用方预算")
            current, steps, visited = stack.pop()
            explored += 1
            pending = []
            for step in self._steps_from(
                    current,
                    include_realizes=True,
                    budget=budget,
                    considered=considered,
                    cache=cache,
                    allowed_prefix_steps=allowed_prefix_steps):
                kind, step_branch = self.selector.validate_step(step)
                if kind == self.selector.protocol.realizes_step:
                    if step_branch == branch:
                        self._add_route(
                            routes, AliasRoute(origin, (*steps, step)), budget)
                    continue
                if kind not in allowed_prefix_steps:
                    raise ValueError("surface route finder 返回了未允许的前缀步骤")
                if step.target in visited:
                    continue
                pending.append((
                    step.target,
                    (*steps, step),
                    visited | frozenset({step.target}),
                ))
            stack.extend(reversed(pending))
        facts = tuple(
            considered[key]
            for key in sorted(considered, key=ObjectIdentity.stable_key)
        )
        return tuple(routes[key] for key in sorted(routes)), explored, facts

    @staticmethod
    def _add_route(
            routes: dict[tuple[int, ...], AliasRoute],
            route: AliasRoute,
            budget: AliasRouteSearchBudget,
            ) -> None:
        """去重完整 route；新增 route 将超预算时不返回不完整候选集。"""
        key = route.stable_key()
        if key in routes:
            return
        if len(routes) >= budget.max_routes:
            raise AliasRouteSearchExhausted(
                "alias route 数超过调用方预算")
        routes[key] = route


__all__ = [
    "ActiveAliasRouteFinder",
    "AliasResolutionOption",
    "AliasResolutionProposal",
    "AliasResolutionProtocol",
    "AliasResolutionResult",
    "AliasResolutionSelector",
    "AliasRoute",
    "AliasRouteSearchBudget",
    "AliasRouteSearchExhausted",
    "AliasRouteStep",
    "ReferenceRouteDiscovery",
    "ReferenceResolutionQuery",
    "SurfaceRouteDiscovery",
    "SurfaceRealizationQuery",
]
