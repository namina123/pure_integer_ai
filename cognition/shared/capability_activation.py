"""C-02 Capability 候选到 A-10 当前义务的显式 mapper route。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivationMapper,
    AttractorActivationProposal,
    AttractorDependency,
    AttractorScoreReason,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_CAPABILITY,
    require_memory_object_kind,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryActivationRequest
from pure_integer_ai.cognition.shared.memory_resolver import ResolvedCandidate
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningObligation
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """核验作用、理由和依赖角色均为一等最小指令。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


def _state_key(component: object, *, label: str) -> tuple[int, ...]:
    """读取 mapper 组件的非空严格整数状态键。"""
    method = getattr(component, "state_key", None)
    if not callable(method):
        raise TypeError(f"{label} 缺少 state_key")
    key = method()
    if not isinstance(key, tuple) or not key:
        raise ValueError(f"{label} state_key 必须是非空 tuple")
    assert_int(*key, _where=f"{label}.state_key")
    if any(type(item) is not int for item in key):
        raise ValueError(f"{label} state_key 必须使用严格整数")
    return key


@dataclass(frozen=True)
class CapabilityObligationProjection:
    """selector 对一个既有当前义务给出的整数方向调整。"""

    obligation: ReasoningObligation
    score_adjustment: int

    def __post_init__(self) -> None:
        """核验目标类型和调整值，不允许 selector 直接构造 activation。"""
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("Capability projection obligation 类型错误")
        assert_int(
            self.score_adjustment,
            _where="CapabilityObligationProjection.score_adjustment",
        )
        if type(self.score_adjustment) is not int:
            raise ValueError("Capability projection score 必须是严格整数")


class CapabilityObligationSelector(Protocol):
    """把已召回能力与调用方声明的当前义务显式关联。"""

    def select(
            self,
            request: MemoryActivationRequest,
            candidate: ResolvedCandidate,
            obligations: tuple[ReasoningObligation, ...],
            ) -> tuple[CapabilityObligationProjection, ...]:
        """只返回输入 obligations 的投影，不得生成新目标。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 selector 的版本化状态键。"""
        ...


class CapabilityActivationMapper:
    """把 Capability route 结果映射为带完整 Memory 依赖的 A-10 提案。"""

    memory_object_kind = MEMORY_OBJECT_CAPABILITY

    def __init__(
            self,
            selector: CapabilityObligationSelector,
            *,
            activation_kind: ObjectIdentity,
            score_reason: ObjectIdentity,
            capability_dependency_role: ObjectIdentity,
            ) -> None:
        """绑定注入 selector 以及三个图内最小指令身份。"""
        if not callable(getattr(selector, "select", None)):
            raise TypeError("Capability obligation selector 缺少 select")
        _state_key(selector, label="Capability obligation selector")
        for label, value in (
                ("activation_kind", activation_kind),
                ("score_reason", score_reason),
                ("capability_dependency_role", capability_dependency_role)):
            _require_instruction(value, label=label)
        self.selector = selector
        self.activation_kind = activation_kind
        self.score_reason = score_reason
        self.capability_dependency_role = capability_dependency_role

    def project(
            self,
            request: MemoryActivationRequest,
            candidate: ResolvedCandidate,
            obligations: tuple[ReasoningObligation, ...],
            ) -> tuple[AttractorActivationProposal, ...]:
        """只对真实 Capability 候选形成当前义务提案并保留依赖。"""
        if (candidate.memory_ref is None
                or candidate.memory_ref.object_kind
                != MEMORY_OBJECT_CAPABILITY
                or candidate.capability is None):
            return ()
        projections = self.selector.select(request, candidate, obligations)
        if not isinstance(projections, tuple) or any(
                not isinstance(item, CapabilityObligationProjection)
                for item in projections):
            raise TypeError("Capability obligation selector 返回类型错误")
        if len(set(item.obligation for item in projections)) != len(projections):
            raise ValueError("Capability obligation selector 不得重复目标")
        dependency = AttractorDependency(
            self.capability_dependency_role,
            candidate.memory_ref,
        )
        result = []
        for projection in projections:
            if projection.obligation not in obligations:
                raise ValueError("Capability obligation selector 伪造了当前目标")
            reason = AttractorScoreReason(
                self.score_reason,
                projection.score_adjustment,
                (dependency,),
            )
            result.append(AttractorActivationProposal(
                self.activation_kind,
                projection.obligation,
                projection.score_adjustment,
                (reason,),
                (dependency,),
            ))
        return tuple(result)

    def clone_for_context(self, ctx) -> "CapabilityActivationMapper":
        """为 V-06 克隆 selector，并复用不可变图内协议身份。"""
        clone = getattr(self.selector, "clone_for_context", None)
        selector = self.selector if clone is None else clone(ctx)
        result = CapabilityActivationMapper(
            selector,
            activation_kind=self.activation_kind,
            score_reason=self.score_reason,
            capability_dependency_role=self.capability_dependency_role,
        )
        if result.state_key() != self.state_key():
            raise ValueError("Capability activation mapper clone 改变了协议")
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回 route、图内指令和 selector 的版本化状态键。"""
        selector_key = _state_key(
            self.selector, label="Capability obligation selector")
        return (
            1,
            MEMORY_OBJECT_CAPABILITY,
            *self.activation_kind.stable_key(),
            *self.score_reason.stable_key(),
            *self.capability_dependency_role.stable_key(),
            len(selector_key),
            *selector_key,
        )


class AttractorMapperRouter:
    """保留默认 mapper，并按 Memory object kind 派发额外 typed route。"""

    def __init__(
            self,
            default_mapper: AttractorActivationMapper,
            routes: tuple[object, ...] = (),
            ) -> None:
        """绑定默认 mapper，并注册互异对象种类 route。"""
        if not callable(getattr(default_mapper, "project", None)):
            raise TypeError("默认 Attractor mapper 缺少 project")
        _state_key(default_mapper, label="默认 Attractor mapper")
        self.default_mapper = default_mapper
        self._routes: dict[int, object] = {}
        for route in routes:
            self.register_route(route)

    def register_route(self, route: object) -> None:
        """注册一个对象种类唯一的 mapper route。"""
        object_kind = getattr(route, "memory_object_kind", None)
        require_memory_object_kind(
            object_kind, where="Attractor mapper route object kind")
        if not callable(getattr(route, "project", None)):
            raise TypeError("Attractor mapper route 缺少 project")
        _state_key(route, label="Attractor mapper route")
        if object_kind in self._routes:
            raise ValueError("同一 Memory object kind 已注册 Attractor mapper route")
        self._routes[object_kind] = route

    def project(self, request, candidate, obligations):
        """Memory typed route 优先；无对应 route 时保持原 mapper 行为。"""
        object_kind = (
            None if candidate.memory_ref is None
            else candidate.memory_ref.object_kind)
        route = self._routes.get(object_kind)
        if route is None:
            return self.default_mapper.project(request, candidate, obligations)
        return route.project(request, candidate, obligations)

    def clone_for_context(self, ctx) -> "AttractorMapperRouter":
        """为 V-06 克隆默认 mapper 和所有 route。"""
        default_clone = getattr(self.default_mapper, "clone_for_context", None)
        if not callable(default_clone):
            raise TypeError("默认 Attractor mapper 缺少 clone_for_context")
        routes = []
        for object_kind in sorted(self._routes):
            route = self._routes[object_kind]
            clone = getattr(route, "clone_for_context", None)
            routes.append(route if clone is None else clone(ctx))
        result = AttractorMapperRouter(default_clone(ctx), tuple(routes))
        if result.state_key() != self.state_key():
            raise ValueError("Attractor mapper router clone 改变了协议")
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回默认 mapper 与所有 route 的稳定配置键。"""
        default_key = _state_key(
            self.default_mapper, label="默认 Attractor mapper")
        result = [1, len(default_key), *default_key, len(self._routes)]
        for object_kind in sorted(self._routes):
            route_key = _state_key(
                self._routes[object_kind], label="Attractor mapper route")
            result.extend((object_kind, len(route_key), *route_key))
        return tuple(result)


__all__ = [
    "AttractorMapperRouter",
    "CapabilityActivationMapper",
    "CapabilityObligationProjection",
    "CapabilityObligationSelector",
]
