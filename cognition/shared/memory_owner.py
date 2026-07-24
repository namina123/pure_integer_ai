"""Memory owner 选择、跨层优先级和管理授权类型。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


OWNER_SELECTION_EXACT = 1
OWNER_SELECTION_SUBTREE = 2
_OWNER_SELECTIONS = frozenset({
    OWNER_SELECTION_EXACT,
    OWNER_SELECTION_SUBTREE,
})


@dataclass(frozen=True)
class MemoryOwnerSelector:
    """管理 API 使用的 exact 或 owner 子树选择器。"""

    target: OwnerScope
    selection_kind: int

    def __post_init__(self) -> None:
        """核验目标 owner 和选择方式。"""
        if not isinstance(self.target, OwnerScope):
            raise TypeError("target 必须是 OwnerScope")
        assert_int(self.selection_kind, _where="MemoryOwnerSelector.kind")
        if self.selection_kind not in _OWNER_SELECTIONS:
            raise ValueError("owner selection_kind 未注册")
        if (self.target.visibility == VISIBILITY_SESSION
                and self.selection_kind == OWNER_SELECTION_SUBTREE):
            raise ValueError("session owner 没有可选择的子层")

    def stable_key(self) -> tuple[int, ...]:
        """返回目标 owner 和选择方式的稳定键。"""
        return self.selection_kind, *self.target.stable_key()

    def matches(self, owner: OwnerScope) -> bool:
        """判断一个对象 owner 是否落在显式管理范围。"""
        if not isinstance(owner, OwnerScope):
            raise TypeError("owner 必须是 OwnerScope")
        if self.selection_kind == OWNER_SELECTION_EXACT:
            return owner == self.target
        target = self.target
        if target.visibility == VISIBILITY_GLOBAL:
            return True
        if target.visibility == VISIBILITY_TENANT:
            return owner.tenant_id == target.tenant_id and owner.visibility in {
                VISIBILITY_TENANT,
                VISIBILITY_USER,
                VISIBILITY_SESSION,
            }
        if target.visibility == VISIBILITY_USER:
            return (
                owner.tenant_id == target.tenant_id
                and owner.user_id == target.user_id
                and owner.visibility in {
                    VISIBILITY_USER,
                    VISIBILITY_SESSION,
                }
            )
        raise ValueError("owner subtree target visibility 非法")


@runtime_checkable
class MemoryOwnerAuthorizer(Protocol):
    """跨 owner/session 管理操作的注入式授权协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回授权规则的稳定非空配置键。"""
        ...

    def authorize(
            self,
            actor: OwnerScope,
            selector: MemoryOwnerSelector,
            ) -> bool:
        """判断 actor 是否可管理显式 owner 范围。"""
        ...


@runtime_checkable
class MemoryReadAccess(Protocol):
    """层优先级只依赖的最小 owner 可读协议。"""

    def can_read(self, owner: OwnerScope) -> bool:
        """判断一个 owner 是否在普通读取范围。"""
        ...


@dataclass(frozen=True)
class MemoryManagementContext:
    """与普通读取分离的跨 owner/session 管理请求。"""

    actor: OwnerScope
    selector: MemoryOwnerSelector
    authorizer_state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 actor、目标选择和授权配置身份。"""
        if not isinstance(self.actor, OwnerScope):
            raise TypeError("actor 必须是 OwnerScope")
        if not isinstance(self.selector, MemoryOwnerSelector):
            raise TypeError("selector 必须是 MemoryOwnerSelector")
        if (not isinstance(self.authorizer_state_key, tuple)
                or not self.authorizer_state_key):
            raise ValueError("authorizer_state_key 必须是非空 tuple")
        assert_int(
            *self.authorizer_state_key,
            _where="MemoryManagementContext.authorizer_state_key",
        )
        if any(type(item) is not int for item in self.authorizer_state_key):
            raise ValueError("authorizer_state_key 必须使用严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回管理请求的完整稳定键。"""
        selector_key = self.selector.stable_key()
        return (
            *self.actor.stable_key(),
            len(selector_key),
            *selector_key,
            len(self.authorizer_state_key),
            *self.authorizer_state_key,
        )


@dataclass(frozen=True, order=True)
class MemoryLayerCandidate:
    """按开放 shadow key 参与 owner 层覆盖选择的候选描述。"""

    owner: OwnerScope
    shadow_key: tuple[int, ...]
    value_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 owner、shadow key 和完整值键。"""
        if not isinstance(self.owner, OwnerScope):
            raise TypeError("layer candidate owner 必须是 OwnerScope")
        for name, value in (
                ("shadow_key", self.shadow_key),
                ("value_key", self.value_key)):
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"{name} 必须是非空 tuple")
            assert_int(*value, _where=f"MemoryLayerCandidate.{name}")
            if any(type(item) is not int for item in value):
                raise ValueError(f"{name} 必须使用严格整数")


@dataclass(frozen=True)
class MemoryLayerSelection:
    """每个 shadow key 的最高可见层和完整被遮蔽审计轨迹。"""

    selected: tuple[MemoryLayerCandidate, ...]
    shadowed: tuple[MemoryLayerCandidate, ...]


def select_memory_layers(
        candidates: tuple[MemoryLayerCandidate, ...],
        *,
        access: MemoryReadAccess,
        ) -> MemoryLayerSelection:
    """按 global、tenant、user、session 层序选择同 shadow key 候选。"""
    if (not isinstance(candidates, tuple)
            or any(not isinstance(item, MemoryLayerCandidate)
                   for item in candidates)):
        raise TypeError("candidates 必须是 MemoryLayerCandidate tuple")
    if not isinstance(access, MemoryReadAccess):
        raise TypeError("access 必须实现 MemoryReadAccess")
    visible = tuple(item for item in candidates if access.can_read(item.owner))
    ranks = {
        VISIBILITY_GLOBAL: 0,
        VISIBILITY_TENANT: 1,
        VISIBILITY_USER: 2,
        VISIBILITY_SESSION: 3,
    }
    grouped: dict[tuple[int, ...], list[MemoryLayerCandidate]] = {}
    for item in visible:
        grouped.setdefault(item.shadow_key, []).append(item)
    selected = []
    shadowed = []
    for shadow_key in sorted(grouped):
        ordered = sorted(
            grouped[shadow_key],
            key=lambda item: (
                -ranks[item.owner.visibility],
                item.value_key,
                item.owner.stable_key(),
            ),
        )
        selected.append(ordered[0])
        shadowed.extend(ordered[1:])
    return MemoryLayerSelection(
        tuple(selected),
        tuple(sorted(shadowed)),
    )


__all__ = [
    "OWNER_SELECTION_EXACT",
    "OWNER_SELECTION_SUBTREE",
    "MemoryLayerCandidate",
    "MemoryLayerSelection",
    "MemoryManagementContext",
    "MemoryOwnerAuthorizer",
    "MemoryOwnerSelector",
    "MemoryReadAccess",
    "select_memory_layers",
]
