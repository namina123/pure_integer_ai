"""开放物理指标的逐维硬预算与 Pareto 变化协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.storage.integer_codec import (
    pack_key,
    strict_integer_tuple,
)


EDGE_BUDGET_MAXIMUM = 1
EDGE_BUDGET_MINIMUM = 2
_EDGE_BUDGET_DIRECTIONS = frozenset({
    EDGE_BUDGET_MAXIMUM,
    EDGE_BUDGET_MINIMUM,
})

PARETO_IMPROVED = 1
PARETO_UNCHANGED = 2
PARETO_REGRESSED = 3


@dataclass(frozen=True, order=True)
class EdgeMetricObservation:
    """一个由运行时或外部探针提供的非负整数物理观测。"""

    metric_key: tuple[int, ...]
    value: int

    def __post_init__(self) -> None:
        """核验开放指标身份和非负严格整数值。"""
        strict_integer_tuple(self.metric_key, label="edge metric key")
        if type(self.value) is not int or self.value < 0:
            raise ValueError("edge metric value 必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回指标身份和值的无歧义稳定键。"""
        result: list[int] = []
        pack_key(result, self.metric_key)
        result.append(self.value)
        return tuple(result)


@dataclass(frozen=True, order=True)
class EdgeBudgetLimit:
    """一个独立物理指标的上限或下限硬预算。"""

    metric_key: tuple[int, ...]
    direction: int
    hard_limit: int

    def __post_init__(self) -> None:
        """核验指标身份、方向和非负硬阈值。"""
        strict_integer_tuple(self.metric_key, label="edge budget metric_key")
        if self.direction not in _EDGE_BUDGET_DIRECTIONS:
            raise ValueError("edge budget direction 未注册")
        if type(self.hard_limit) is not int or self.hard_limit < 0:
            raise ValueError("edge budget hard_limit 必须是非负严格整数")

    def accepts(self, value: int) -> bool:
        """按当前方向判断观测是否单独满足硬预算。"""
        if type(value) is not int or value < 0:
            raise ValueError("edge budget observed value 必须是非负严格整数")
        if self.direction == EDGE_BUDGET_MAXIMUM:
            return value <= self.hard_limit
        return value >= self.hard_limit

    def stable_key(self) -> tuple[int, ...]:
        """返回指标、方向和阈值的完整稳定键。"""
        result: list[int] = []
        pack_key(result, self.metric_key)
        result.extend((self.direction, self.hard_limit))
        return tuple(result)


@dataclass(frozen=True, order=True)
class EdgeBudgetResult:
    """一个已注册物理指标的观测、阈值和独立通过状态。"""

    limit: EdgeBudgetLimit
    observed: int
    passed: bool

    def __post_init__(self) -> None:
        """核验结果类型并重新计算通过状态，拒绝伪造。"""
        if not isinstance(self.limit, EdgeBudgetLimit):
            raise TypeError("edge budget result limit 类型错误")
        if type(self.observed) is not int or self.observed < 0:
            raise ValueError("edge budget result observed 必须非负")
        if type(self.passed) is not bool:
            raise TypeError("edge budget result passed 必须是 bool")
        if self.passed != self.limit.accepts(self.observed):
            raise ValueError("edge budget result passed 与硬预算漂移")


@dataclass(frozen=True)
class EdgeBudgetReport:
    """一个 profile 的逐维结果、缺失项和未注册观测，不含综合分。"""

    profile_key: tuple[int, ...]
    results: tuple[EdgeBudgetResult, ...]
    missing_metric_keys: tuple[tuple[int, ...], ...]
    unregistered_metric_keys: tuple[tuple[int, ...], ...]

    @property
    def passed(self) -> bool:
        """仅当无缺失/未注册且每个独立硬预算均通过时返回真。"""
        return (
            not self.missing_metric_keys
            and not self.unregistered_metric_keys
            and all(item.passed for item in self.results)
        )


@dataclass(frozen=True)
class EdgeBudgetProfile:
    """绑定 device/backend 身份和开放逐维硬预算的冻结 profile。"""

    profile_key: tuple[int, ...]
    device_profile_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    limits: tuple[EdgeBudgetLimit, ...]

    def __post_init__(self) -> None:
        """核验 profile 身份和非空唯一预算维度。"""
        for label, value in (
                ("profile_key", self.profile_key),
                ("device_profile_key", self.device_profile_key),
                ("backend_profile_key", self.backend_profile_key)):
            strict_integer_tuple(value, label=f"edge budget {label}")
        if (not isinstance(self.limits, tuple)
                or not self.limits
                or any(not isinstance(item, EdgeBudgetLimit)
                       for item in self.limits)):
            raise TypeError("edge budget limits 必须是非空 tuple")
        normalized = tuple(sorted(self.limits))
        if len({item.metric_key for item in normalized}) != len(normalized):
            raise ValueError("edge budget metric_key 不得重复")
        object.__setattr__(self, "limits", normalized)

    def evaluate(
            self,
            observations: tuple[EdgeMetricObservation, ...],
            ) -> EdgeBudgetReport:
        """逐维评估全部观测，缺失和未注册指标都 fail closed。"""
        if (not isinstance(observations, tuple)
                or any(not isinstance(item, EdgeMetricObservation)
                       for item in observations)):
            raise TypeError("edge budget observations 必须是 tuple")
        observed = {item.metric_key: item for item in observations}
        if len(observed) != len(observations):
            raise ValueError("edge budget observations 不得重复指标")
        limits = {item.metric_key: item for item in self.limits}
        results = tuple(EdgeBudgetResult(
            limit,
            observed[key].value,
            limit.accepts(observed[key].value),
        ) for key, limit in sorted(limits.items()) if key in observed)
        return EdgeBudgetReport(
            self.profile_key,
            results,
            tuple(sorted(set(limits) - set(observed))),
            tuple(sorted(set(observed) - set(limits))),
        )


@dataclass(frozen=True, order=True)
class ParetoMetricChange:
    """同一预算维度从基线到当前观测的独立变化方向。"""

    metric_key: tuple[int, ...]
    change: int
    baseline: int
    current: int


@dataclass(frozen=True)
class ParetoBudgetReport:
    """逐维 Pareto 变化集合，不计算加权和或总分。"""

    profile_key: tuple[int, ...]
    changes: tuple[ParetoMetricChange, ...]

    @property
    def has_regression(self) -> bool:
        """返回是否至少一个独立维度发生退化。"""
        return any(item.change == PARETO_REGRESSED for item in self.changes)


def compare_pareto(
        profile: EdgeBudgetProfile,
        baseline: tuple[EdgeMetricObservation, ...],
        current: tuple[EdgeMetricObservation, ...],
        ) -> ParetoBudgetReport:
    """按每个预算方向比较两组完整观测，不允许跨维抵消。"""
    if not isinstance(profile, EdgeBudgetProfile):
        raise TypeError("pareto profile 类型错误")
    baseline_by_key = {item.metric_key: item.value for item in baseline}
    current_by_key = {item.metric_key: item.value for item in current}
    expected = {item.metric_key for item in profile.limits}
    if (set(baseline_by_key) != expected
            or set(current_by_key) != expected
            or len(baseline_by_key) != len(baseline)
            or len(current_by_key) != len(current)):
        raise ValueError("pareto 比较要求两组观测与 profile 维度完全一致")
    changes = []
    for limit in profile.limits:
        before = baseline_by_key[limit.metric_key]
        after = current_by_key[limit.metric_key]
        if before == after:
            change = PARETO_UNCHANGED
        elif ((limit.direction == EDGE_BUDGET_MAXIMUM and after < before)
              or (limit.direction == EDGE_BUDGET_MINIMUM and after > before)):
            change = PARETO_IMPROVED
        else:
            change = PARETO_REGRESSED
        changes.append(ParetoMetricChange(
            limit.metric_key, change, before, after))
    return ParetoBudgetReport(profile.profile_key, tuple(changes))


__all__ = [
    "EDGE_BUDGET_MAXIMUM",
    "EDGE_BUDGET_MINIMUM",
    "PARETO_IMPROVED",
    "PARETO_REGRESSED",
    "PARETO_UNCHANGED",
    "EdgeBudgetLimit",
    "EdgeBudgetProfile",
    "EdgeBudgetReport",
    "EdgeBudgetResult",
    "EdgeMetricObservation",
    "ParetoBudgetReport",
    "ParetoMetricChange",
    "compare_pareto",
]
