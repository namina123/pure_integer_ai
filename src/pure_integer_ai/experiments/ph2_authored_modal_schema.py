"""D-02D MODAL 的独立 resolver、scope 和预算 seed 合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_GENERATION,
    SCOPE_QUERY,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    AuthoredLogicCourseError,
    AuthoredLogicSeed,
)


RESOLVER_RESOLVED = "RESOLVED"
RESOLVER_MISSING = "MISSING"
RESOLVER_DENIED = "DENIED"
RESOLVER_BUDGET_UNDECIDED = "BUDGET_UNDECIDED"
RESOLVER_STATUSES = frozenset({
    RESOLVER_RESOLVED,
    RESOLVER_MISSING,
    RESOLVER_DENIED,
    RESOLVER_BUDGET_UNDECIDED,
})
SOURCE_MODE_SAME = "SAME_SOURCE"

_RESOLVER_FIELDS = frozenset({
    "evidence_ids",
    "resolution_refute",
    "resolution_support",
    "scope_kind",
    "scope_local_id",
    "source_mode",
    "status",
})
_SEED_FIELDS = frozenset({"logic", "max_resolver_calls", "resolver"})


class AuthoredModalCourseError(RuntimeError):
    """原创 modal seed 的 resolver、scope、source 或预算非法。"""


def _positive_int(value: Any, *, where: str) -> int:
    """要求 modal 身份与预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredModalCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 modal scope 坐标为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredModalCourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 modal resolution Evidence 位只能是严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredModalCourseError(f"{where} 必须是严格整数 0/1")
    return value


@dataclass(frozen=True)
class ModalResolverSeed:
    """一次受限 resolver 的状态、输出 scope、Evidence 和 source 约束。"""

    status: str
    source_mode: str
    scope_kind: int
    scope_local_id: int
    resolution_support: int
    resolution_refute: int
    evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in RESOLVER_STATUSES:
            raise AuthoredModalCourseError("modal resolver status 未注册")
        if self.source_mode != SOURCE_MODE_SAME:
            raise AuthoredModalCourseError("modal resolver 不得改变 source")
        _nonnegative_int(self.scope_kind, where="ModalResolverSeed.scope_kind")
        _nonnegative_int(
            self.scope_local_id, where="ModalResolverSeed.scope_local_id")
        _bit(
            self.resolution_support,
            where="ModalResolverSeed.resolution_support",
        )
        _bit(
            self.resolution_refute,
            where="ModalResolverSeed.resolution_refute",
        )
        if (not isinstance(self.evidence_ids, tuple)
                or any(type(item) is not int or item <= 0
                       for item in self.evidence_ids)
                or len(set(self.evidence_ids)) != len(self.evidence_ids)):
            raise AuthoredModalCourseError(
                "modal resolver evidence id 非法或重复")
        if self.status == RESOLVER_RESOLVED:
            if self.scope_kind not in {
                    SCOPE_DOCUMENT, SCOPE_QUERY, SCOPE_GENERATION}:
                raise AuthoredModalCourseError(
                    "resolved modal scope kind 未注册")
            if (self.scope_kind == SCOPE_DOCUMENT
                    and self.scope_local_id != 0):
                raise AuthoredModalCourseError(
                    "document modal scope 不得伪造 local id")
            if (self.scope_kind != SCOPE_DOCUMENT
                    and self.scope_local_id <= 0):
                raise AuthoredModalCourseError(
                    "nested modal scope local id 必须为正")
            if ((self.resolution_support or self.resolution_refute)
                    and not self.evidence_ids):
                raise AuthoredModalCourseError(
                    "非 unknown modal resolution 必须有 Evidence id")
        else:
            if (self.scope_kind != 0 or self.scope_local_id != 0
                    or self.resolution_support != 0
                    or self.resolution_refute != 0
                    or self.evidence_ids):
                raise AuthoredModalCourseError(
                    "未决 modal resolver 不得伪造 resolution")

    @classmethod
    def from_dict(cls, value: Any) -> "ModalResolverSeed":
        """从严格字段集合恢复 modal resolver seed。"""
        if not isinstance(value, dict) or set(value) != _RESOLVER_FIELDS:
            raise AuthoredModalCourseError("modal resolver 字段集合漂移")
        evidence_ids = value["evidence_ids"]
        if not isinstance(evidence_ids, list):
            raise AuthoredModalCourseError(
                "modal resolver evidence_ids 必须是列表")
        return cls(
            str(value["status"]),
            str(value["source_mode"]),
            value["scope_kind"],
            value["scope_local_id"],
            value["resolution_support"],
            value["resolution_refute"],
            tuple(evidence_ids),
        )


@dataclass(frozen=True)
class AuthoredModalSeed:
    """一条共享 unary logic seed 和独立 modal resolver 课程记录。"""

    logic: AuthoredLogicSeed
    max_resolver_calls: int
    resolver: ModalResolverSeed

    def __post_init__(self) -> None:
        if not isinstance(self.logic, AuthoredLogicSeed):
            raise AuthoredModalCourseError("modal logic seed 类型错误")
        _positive_int(
            self.max_resolver_calls,
            where="AuthoredModalSeed.max_resolver_calls",
        )
        if self.max_resolver_calls > self.logic.consumer_request.max_steps:
            raise AuthoredModalCourseError(
                "modal resolver calls 超过 step 预算")
        if not isinstance(self.resolver, ModalResolverSeed):
            raise AuthoredModalCourseError("modal resolver seed 类型错误")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredModalSeed":
        """从严格顶层字段恢复 modal seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredModalCourseError("modal seed 字段集合漂移")
        try:
            logic = AuthoredLogicSeed.from_dict(value["logic"])
        except (AuthoredLogicCourseError, KeyError, TypeError) as error:
            raise AuthoredModalCourseError("modal logic seed 非法") from error
        return cls(
            logic,
            value["max_resolver_calls"],
            ModalResolverSeed.from_dict(value["resolver"]),
        )


__all__ = [
    "AuthoredModalCourseError",
    "AuthoredModalSeed",
    "ModalResolverSeed",
    "RESOLVER_BUDGET_UNDECIDED",
    "RESOLVER_DENIED",
    "RESOLVER_MISSING",
    "RESOLVER_RESOLVED",
    "RESOLVER_STATUSES",
    "SOURCE_MODE_SAME",
]
