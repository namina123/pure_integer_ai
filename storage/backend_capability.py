"""后端物理能力、设备预算和显式协商协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


CAPABILITY_PERSISTENCE = 1
CAPABILITY_CONSISTENT_READ_VIEW = 2
CAPABILITY_ATOMIC_BATCH = 3
CAPABILITY_ATOMIC_MANIFEST_PUBLISH = 4
CAPABILITY_STABLE_ORDER_SCAN = 5
CAPABILITY_RANGE_SCAN = 6
CAPABILITY_BULK_READ = 7
CAPABILITY_BULK_WRITE = 8
CAPABILITY_CONCURRENT_READ = 9
CAPABILITY_CONCURRENT_WRITE = 10
CAPABILITY_DURABLE_COMMIT = 11
CAPABILITY_RECLAMATION = 12
CAPABILITY_COMPACTION = 13
CAPABILITY_LOCALITY_HINT = 14
CAPABILITY_SNAPSHOT_EXPORT = 15

CAPABILITY_MODE_UNSUPPORTED = 0
CAPABILITY_MODE_NATIVE = 1
CAPABILITY_MODE_FALLBACK = 2

_KNOWN_CAPABILITIES = tuple(range(
    CAPABILITY_PERSISTENCE,
    CAPABILITY_SNAPSHOT_EXPORT + 1,
))


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放配置键为非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True, order=True)
class BackendCapabilitySupport:
    """一个后端对某项开放物理能力的原生支持声明。"""

    capability: int
    mode: int
    detail_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验能力标识、原生支持状态和可选实现细节键。"""
        assert_int(self.capability, self.mode, _where="backend capability")
        if type(self.capability) is not int or self.capability <= 0:
            raise ValueError("backend capability 必须是正严格整数")
        if self.mode not in {
                CAPABILITY_MODE_UNSUPPORTED,
                CAPABILITY_MODE_NATIVE,
                }:
            raise ValueError("后端能力声明只能是 unsupported 或 native")
        if not isinstance(self.detail_key, tuple):
            raise TypeError("capability detail_key 必须是 tuple")
        if self.detail_key:
            _strict_key(self.detail_key, label="capability detail_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回能力、支持状态和实现细节的稳定键。"""
        return self.capability, self.mode, len(self.detail_key), *self.detail_key


@dataclass(frozen=True)
class BackendDeviceBudget:
    """由后端实例配置注入的工作集、批次和并发预算。"""

    working_bytes: int
    batch_bytes: int
    concurrent_readers: int
    concurrent_writers: int

    def __post_init__(self) -> None:
        """要求各预算为非负严格整数，零表示该实例未提供该资源。"""
        values = (
            self.working_bytes,
            self.batch_bytes,
            self.concurrent_readers,
            self.concurrent_writers,
        )
        assert_int(*values, _where="backend device budget")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("backend device budget 必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部实例资源预算。"""
        return (
            self.working_bytes,
            self.batch_bytes,
            self.concurrent_readers,
            self.concurrent_writers,
        )


@dataclass(frozen=True)
class BackendCapabilityProfile:
    """一个后端实例的开放能力集合和可选设备预算。

    profile_key 只用于稳定追踪配置身份；执行方案必须读取 capability 和预算，
    不得按该键推断产品、路径或领域行为。
    """

    profile_key: tuple[int, ...]
    capabilities: tuple[BackendCapabilitySupport, ...]
    device_budget: BackendDeviceBudget | None = None

    def __post_init__(self) -> None:
        """规范化能力顺序并拒绝重复、遗漏内建能力或非法预算。"""
        _strict_key(self.profile_key, label="backend profile_key")
        if (not isinstance(self.capabilities, tuple)
                or any(not isinstance(item, BackendCapabilitySupport)
                       for item in self.capabilities)):
            raise TypeError("backend capabilities 类型错误")
        normalized = tuple(sorted(
            self.capabilities,
            key=lambda item: item.capability,
        ))
        kinds = tuple(item.capability for item in normalized)
        if len(set(kinds)) != len(kinds):
            raise ValueError("backend capability 不得重复")
        if not set(_KNOWN_CAPABILITIES).issubset(kinds):
            raise ValueError("backend profile 必须显式声明全部内建能力")
        if self.device_budget is not None and not isinstance(
                self.device_budget, BackendDeviceBudget):
            raise TypeError("device_budget 必须是 BackendDeviceBudget 或 None")
        object.__setattr__(self, "capabilities", normalized)

    def mode(self, capability: int) -> int:
        """读取一项能力状态，开放未知能力未声明时按 unsupported 处理。"""
        assert_int(capability, _where="backend capability lookup")
        if type(capability) is not int or capability <= 0:
            raise ValueError("capability 必须是正严格整数")
        for item in self.capabilities:
            if item.capability == capability:
                return item.mode
        return CAPABILITY_MODE_UNSUPPORTED

    def stable_key(self) -> tuple[int, ...]:
        """返回 profile、全部能力和可选预算的完整稳定键。"""
        result = [1, len(self.profile_key), *self.profile_key]
        result.append(len(self.capabilities))
        for item in self.capabilities:
            key = item.stable_key()
            result.extend((len(key), *key))
        budget = (() if self.device_budget is None
                  else self.device_budget.stable_key())
        result.extend((len(budget), *budget))
        return tuple(result)


@runtime_checkable
class CapabilityAwareBackend(Protocol):
    """由具体后端或代理提供能力 profile 的最小附加协议。"""

    def storage_capabilities(self) -> BackendCapabilityProfile:
        """返回当前实例的物理能力和预算，不推断领域行为。"""
        ...


@dataclass(frozen=True, order=True)
class BackendCapabilityRequirement:
    """上层对一项能力的要求和可选显式 fallback 身份。"""

    capability: int
    fallback_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验能力标识和可选 fallback 配置键。"""
        assert_int(self.capability, _where="backend requirement")
        if type(self.capability) is not int or self.capability <= 0:
            raise ValueError("backend requirement 必须引用正整数能力")
        if not isinstance(self.fallback_key, tuple):
            raise TypeError("fallback_key 必须是 tuple")
        if self.fallback_key:
            _strict_key(self.fallback_key, label="fallback_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回能力要求和可选 fallback 身份的稳定键。"""
        return (
            self.capability,
            len(self.fallback_key),
            *self.fallback_key,
        )


@dataclass(frozen=True, order=True)
class NegotiatedBackendCapability:
    """一项要求最终采用 native 或显式 fallback 的协商结果。"""

    capability: int
    mode: int
    fallback_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """拒绝 unsupported 结果和没有身份的 fallback。"""
        assert_int(self.capability, self.mode, _where="negotiated capability")
        if type(self.capability) is not int or self.capability <= 0:
            raise ValueError("negotiated capability 必须引用正严格整数能力")
        if type(self.mode) is not int:
            raise ValueError("negotiated capability mode 必须是严格整数")
        if not isinstance(self.fallback_key, tuple):
            raise TypeError("negotiated fallback_key 必须是 tuple")
        if self.mode not in {
                CAPABILITY_MODE_NATIVE,
                CAPABILITY_MODE_FALLBACK,
                }:
            raise ValueError("协商结果只能是 native 或 fallback")
        if self.mode == CAPABILITY_MODE_NATIVE and self.fallback_key:
            raise ValueError("native 协商结果不得携带 fallback")
        if self.mode == CAPABILITY_MODE_FALLBACK:
            _strict_key(self.fallback_key, label="negotiated fallback")

    def stable_key(self) -> tuple[int, ...]:
        """返回能力、执行模式和 fallback 身份。"""
        return (
            self.capability,
            self.mode,
            len(self.fallback_key),
            *self.fallback_key,
        )


@dataclass(frozen=True)
class BackendNegotiationReport:
    """一次能力协商的确定性结果，不包含后端产品分支。"""

    profile_key: tuple[int, ...]
    capabilities: tuple[NegotiatedBackendCapability, ...]

    def __post_init__(self) -> None:
        """核验结果按能力唯一稳定排序。"""
        _strict_key(self.profile_key, label="negotiation profile_key")
        if (not isinstance(self.capabilities, tuple)
                or any(not isinstance(item, NegotiatedBackendCapability)
                       for item in self.capabilities)):
            raise TypeError("negotiated capabilities 类型错误")
        normalized = tuple(sorted(
            self.capabilities,
            key=lambda item: item.capability,
        ))
        if len({item.capability for item in normalized}) != len(normalized):
            raise ValueError("同一能力不得重复协商")
        object.__setattr__(self, "capabilities", normalized)

    def stable_key(self) -> tuple[int, ...]:
        """返回 profile 和全部协商执行模式。"""
        result = [1, len(self.profile_key), *self.profile_key]
        result.append(len(self.capabilities))
        for item in self.capabilities:
            key = item.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


class BackendCapabilityError(RuntimeError):
    """后端缺少能力且调用方没有声明同语义 fallback。"""


def capability_profile(backend: object) -> BackendCapabilityProfile:
    """从后端或代理读取显式 profile，缺少协议时 fail closed。"""
    if not isinstance(backend, CapabilityAwareBackend):
        raise BackendCapabilityError("后端没有声明 storage_capabilities")
    profile = backend.storage_capabilities()
    if not isinstance(profile, BackendCapabilityProfile):
        raise BackendCapabilityError("后端返回了非法 capability profile")
    return profile


def negotiate_backend_capabilities(
        backend: object,
        requirements: tuple[BackendCapabilityRequirement, ...],
        ) -> BackendNegotiationReport:
    """按实例 profile 选择 native 或显式 fallback，缺能力时拒绝降级。"""
    if (not isinstance(requirements, tuple)
            or any(not isinstance(item, BackendCapabilityRequirement)
                   for item in requirements)):
        raise TypeError("requirements 必须是 BackendCapabilityRequirement tuple")
    if len({item.capability for item in requirements}) != len(requirements):
        raise ValueError("同一能力不得重复要求")
    profile = capability_profile(backend)
    result = []
    for requirement in requirements:
        if profile.mode(requirement.capability) == CAPABILITY_MODE_NATIVE:
            result.append(NegotiatedBackendCapability(
                requirement.capability,
                CAPABILITY_MODE_NATIVE,
            ))
            continue
        if requirement.fallback_key:
            result.append(NegotiatedBackendCapability(
                requirement.capability,
                CAPABILITY_MODE_FALLBACK,
                requirement.fallback_key,
            ))
            continue
        raise BackendCapabilityError(
            f"后端缺少能力 {requirement.capability} 且没有显式 fallback")
    return BackendNegotiationReport(profile.profile_key, tuple(result))


def _profile(
        profile_key: tuple[int, ...],
        native: frozenset[int],
        budget: BackendDeviceBudget | None,
        ) -> BackendCapabilityProfile:
    """构造显式列出全部内建能力的标准后端 profile。"""
    return BackendCapabilityProfile(
        profile_key,
        tuple(BackendCapabilitySupport(
            capability,
            (CAPABILITY_MODE_NATIVE
             if capability in native
             else CAPABILITY_MODE_UNSUPPORTED),
        ) for capability in _KNOWN_CAPABILITIES),
        budget,
    )


def dict_backend_profile(
        budget: BackendDeviceBudget | None = None,
        ) -> BackendCapabilityProfile:
    """返回 DictBackend 的内存型物理能力声明。"""
    return _profile(
        (1, 1),
        frozenset({
            CAPABILITY_STABLE_ORDER_SCAN,
            CAPABILITY_RANGE_SCAN,
            CAPABILITY_BULK_READ,
            CAPABILITY_RECLAMATION,
            CAPABILITY_SNAPSHOT_EXPORT,
        }),
        budget,
    )


def sqlite_backend_profile(
        *,
        persistent: bool,
        budget: BackendDeviceBudget | None = None,
        ) -> BackendCapabilityProfile:
    """按 SQLite 实例是否持久化返回能力声明，不暴露路径给上层。"""
    if type(persistent) is not bool:
        raise TypeError("persistent 必须是 bool")
    native = {
        CAPABILITY_STABLE_ORDER_SCAN,
        CAPABILITY_RANGE_SCAN,
        CAPABILITY_BULK_READ,
        CAPABILITY_RECLAMATION,
        CAPABILITY_SNAPSHOT_EXPORT,
    }
    if persistent:
        native.update({
            CAPABILITY_PERSISTENCE,
            CAPABILITY_DURABLE_COMMIT,
        })
    return _profile((2, 2 if persistent else 1), frozenset(native), budget)


__all__ = [
    "BackendCapabilityError",
    "BackendCapabilityProfile",
    "BackendCapabilityRequirement",
    "BackendCapabilitySupport",
    "BackendDeviceBudget",
    "BackendNegotiationReport",
    "CapabilityAwareBackend",
    "NegotiatedBackendCapability",
    "CAPABILITY_ATOMIC_BATCH",
    "CAPABILITY_ATOMIC_MANIFEST_PUBLISH",
    "CAPABILITY_BULK_READ",
    "CAPABILITY_BULK_WRITE",
    "CAPABILITY_COMPACTION",
    "CAPABILITY_CONCURRENT_READ",
    "CAPABILITY_CONCURRENT_WRITE",
    "CAPABILITY_CONSISTENT_READ_VIEW",
    "CAPABILITY_DURABLE_COMMIT",
    "CAPABILITY_LOCALITY_HINT",
    "CAPABILITY_MODE_FALLBACK",
    "CAPABILITY_MODE_NATIVE",
    "CAPABILITY_MODE_UNSUPPORTED",
    "CAPABILITY_PERSISTENCE",
    "CAPABILITY_RANGE_SCAN",
    "CAPABILITY_RECLAMATION",
    "CAPABILITY_SNAPSHOT_EXPORT",
    "CAPABILITY_STABLE_ORDER_SCAN",
    "capability_profile",
    "dict_backend_profile",
    "negotiate_backend_capabilities",
    "sqlite_backend_profile",
]
