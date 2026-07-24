"""冷热放置建议协议；只描述策略结果，不执行物理迁移。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend_capability import (
    BackendCapabilityProfile,
)
from pure_integer_ai.storage.storage_role import StorageRoleDescriptor


PLACEMENT_KEEP = 1
PLACEMENT_MOVE = 2
PLACEMENT_PREFETCH = 3
PLACEMENT_RELEASE = 4


def _key(value: tuple[int, ...], *, label: str, empty: bool = False) -> tuple[int, ...]:
    """核验放置协议中的开放整数键。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是非空整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True, order=True)
class TemperatureTier:
    """一个注入式物理温层及其确定性顺序。"""

    tier_key: tuple[int, ...]
    order: int

    def __post_init__(self) -> None:
        """核验温层身份和非负物理顺序。"""
        _key(self.tier_key, label="temperature tier_key")
        assert_int(self.order, _where="temperature tier order")
        if type(self.order) is not int or self.order < 0:
            raise ValueError("temperature tier order 必须是非负严格整数")


@dataclass(frozen=True)
class TemperatureProfile:
    """由设备或后端注入的至少两级温层拓扑。"""

    profile_key: tuple[int, ...]
    tiers: tuple[TemperatureTier, ...]

    def __post_init__(self) -> None:
        """规范化温层顺序并拒绝重复身份或顺序。"""
        _key(self.profile_key, label="temperature profile_key")
        if not isinstance(self.tiers, tuple) or len(self.tiers) < 2:
            raise ValueError("temperature profile 至少需要两级温层")
        if any(not isinstance(item, TemperatureTier) for item in self.tiers):
            raise TypeError("temperature tiers 类型错误")
        normalized = tuple(sorted(self.tiers, key=lambda item: item.order))
        if len({item.tier_key for item in normalized}) != len(normalized):
            raise ValueError("temperature tier_key 不得重复")
        if len({item.order for item in normalized}) != len(normalized):
            raise ValueError("temperature tier order 不得重复")
        object.__setattr__(self, "tiers", normalized)

    def has(self, tier_key: tuple[int, ...]) -> bool:
        """判断温层是否属于当前注入 profile。"""
        return _key(tier_key, label="temperature lookup") in {
            item.tier_key for item in self.tiers
        }

    def order_of(self, tier_key: tuple[int, ...]) -> int:
        """读取温层顺序，未知温层拒绝继续。"""
        key = _key(tier_key, label="temperature order lookup")
        for item in self.tiers:
            if item.tier_key == key:
                return item.order
        raise KeyError(f"温层不属于当前 profile: {key}")

    def stable_key(self) -> tuple[int, ...]:
        """返回 profile、温层身份和物理顺序的完整稳定键。"""
        result = [len(self.profile_key), *self.profile_key, len(self.tiers)]
        for item in self.tiers:
            result.extend((len(item.tier_key), *item.tier_key, item.order))
        return tuple(result)


@dataclass(frozen=True)
class PlacementRequest:
    """一次对象放置决策所需的完整物理观察输入。"""

    object_key: tuple[int, ...]
    descriptor_key: tuple[int, ...]
    current_tier_key: tuple[int, ...] | None
    dirty: bool
    locality_score: int
    logical_seq: int
    size_bytes: int
    backend_profile: BackendCapabilityProfile
    temperature_profile: TemperatureProfile

    def __post_init__(self) -> None:
        """核验对象身份、逻辑序、大小、局部性和注入式物理 profile。"""
        _key(self.object_key, label="placement object_key")
        _key(self.descriptor_key, label="placement descriptor_key")
        if self.current_tier_key is not None:
            _key(self.current_tier_key, label="placement current_tier_key")
        if type(self.dirty) is not bool:
            raise TypeError("placement dirty 必须是 bool")
        for label, value in (
                ("locality_score", self.locality_score),
                ("logical_seq", self.logical_seq),
                ("size_bytes", self.size_bytes)):
            assert_int(value, _where=f"placement {label}")
            if type(value) is not int or value < 0:
                raise ValueError(f"placement {label} 必须是非负严格整数")
        if not isinstance(self.backend_profile, BackendCapabilityProfile):
            raise TypeError("placement backend_profile 类型错误")
        if not isinstance(self.temperature_profile, TemperatureProfile):
            raise TypeError("placement temperature_profile 类型错误")
        if (self.current_tier_key is not None
                and not self.temperature_profile.has(self.current_tier_key)):
            raise ValueError("当前温层不属于注入 profile")


@dataclass(frozen=True)
class PlacementAdvice:
    """策略产生的放置、预取或释放建议，不执行物理写入。"""

    action: int
    target_tier_key: tuple[int, ...] | None
    requires_flush: bool
    prefetch: bool
    reason_key: tuple[int, ...]
    policy_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验动作、目标、刷写要求和开放原因键。"""
        assert_int(self.action, _where="placement action")
        if type(self.action) is not int or self.action <= 0:
            raise ValueError("placement action 必须是正严格整数")
        if self.target_tier_key is not None:
            _key(self.target_tier_key, label="placement target_tier_key")
        if type(self.requires_flush) is not bool:
            raise TypeError("placement requires_flush 必须是 bool")
        if type(self.prefetch) is not bool:
            raise TypeError("placement prefetch 必须是 bool")
        _key(self.reason_key, label="placement reason_key")
        _key(self.policy_key, label="placement policy_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回动作、目标、刷写、原因和策略身份的稳定键。"""
        target = () if self.target_tier_key is None else self.target_tier_key
        return (
            self.action,
            len(target),
            *target,
            int(self.requires_flush),
            int(self.prefetch),
            len(self.reason_key),
            *self.reason_key,
            len(self.policy_key),
            *self.policy_key,
        )


@runtime_checkable
class PlacementPolicy(Protocol):
    """调用方注入的放置策略最小协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回策略版本、参数和组件组合的完整稳定身份。"""
        ...

    def advise(
            self,
            request: PlacementRequest,
            descriptor: StorageRoleDescriptor,
            ) -> PlacementAdvice:
        """根据完整物理观察输入返回建议，不执行存储操作。"""
        ...


class PlacementPolicyError(RuntimeError):
    """放置策略输出违反角色、温层或脏数据不变量。"""


class PlacementPlanner:
    """校验注入策略并形成可审计放置建议。"""

    def __init__(self, policy: PlacementPolicy) -> None:
        """绑定策略实例；策略不存在或不满足协议时拒绝装配。"""
        if not isinstance(policy, PlacementPolicy):
            raise TypeError("placement policy 未实现 state_key/advise 协议")
        self.policy = policy
        self.policy_key = _key(policy.state_key(), label="placement policy state_key")

    def advise(
            self,
            request: PlacementRequest,
            descriptor: StorageRoleDescriptor,
            ) -> PlacementAdvice:
        """执行一次纯校验放置规划，不修改图、事件、逻辑时钟或后端。"""
        if not isinstance(request, PlacementRequest):
            raise TypeError("placement request 类型错误")
        if not isinstance(descriptor, StorageRoleDescriptor):
            raise TypeError("placement descriptor 类型错误")
        if request.descriptor_key != descriptor.descriptor_key:
            raise PlacementPolicyError("placement descriptor 身份漂移")
        advice = self.policy.advise(request, descriptor)
        if not isinstance(advice, PlacementAdvice):
            raise PlacementPolicyError("placement policy 返回非法 advice")
        if advice.policy_key != self.policy_key:
            raise PlacementPolicyError("placement advice policy_key 漂移")
        self._validate_advice(request, descriptor, advice)
        return advice

    def _validate_advice(
            self,
            request: PlacementRequest,
            descriptor: StorageRoleDescriptor,
            advice: PlacementAdvice,
            ) -> None:
        """核验释放、迁移、预取和 dirty flush 的组合不变量。"""
        action = advice.action
        if action not in {
                PLACEMENT_KEEP,
                PLACEMENT_MOVE,
                PLACEMENT_PREFETCH,
                PLACEMENT_RELEASE,
                }:
            raise PlacementPolicyError("placement action 未注册")
        if advice.prefetch and action == PLACEMENT_RELEASE:
            raise PlacementPolicyError("release 不得同时要求 prefetch")
        if action == PLACEMENT_PREFETCH and not advice.prefetch:
            raise PlacementPolicyError("prefetch action 必须显式标记 prefetch")
        if action == PLACEMENT_RELEASE:
            if advice.target_tier_key is not None:
                raise PlacementPolicyError("release 不得携带 target tier")
            if request.dirty and not advice.requires_flush:
                raise PlacementPolicyError("dirty release 必须显式要求 flush")
            if not descriptor.can_discard():
                raise PlacementPolicyError("权威对象不得 release")
            return
        if action in {PLACEMENT_MOVE, PLACEMENT_PREFETCH}:
            if advice.target_tier_key is None:
                raise PlacementPolicyError("迁移或预取必须指定 target tier")
            if not request.temperature_profile.has(advice.target_tier_key):
                raise PlacementPolicyError("advice target tier 不属于 profile")
        if action == PLACEMENT_KEEP:
            if (advice.target_tier_key is not None
                    and advice.target_tier_key != request.current_tier_key):
                raise PlacementPolicyError("keep 的 target tier 必须保持当前温层")
        if (action == PLACEMENT_MOVE
                and request.dirty
                and advice.target_tier_key != request.current_tier_key
                and not advice.requires_flush):
            raise PlacementPolicyError("dirty move 必须显式要求 flush")
        if action == PLACEMENT_PREFETCH and advice.requires_flush:
            raise PlacementPolicyError("prefetch 不得要求 flush")


__all__ = [
    "PLACEMENT_KEEP",
    "PLACEMENT_MOVE",
    "PLACEMENT_PREFETCH",
    "PLACEMENT_RELEASE",
    "PlacementAdvice",
    "PlacementPlanner",
    "PlacementPolicy",
    "PlacementPolicyError",
    "PlacementRequest",
    "TemperatureProfile",
    "TemperatureTier",
]
