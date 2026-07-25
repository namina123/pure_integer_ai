"""F-01 设施总装的计数、完整性检查和最终报告协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)


@dataclass(frozen=True, order=True)
class FacilityCounter:
    """保存一个来源化设施计数及其实际采样数。"""

    metric_key: ProtocolKey
    value: int
    sample_count: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求计数非负、样本为正且 trace 使用严格整数。"""
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("F-01 counter metric_key 类型错误")
        assert_int(
            self.value,
            self.sample_count,
            *self.trace,
            _where="FacilityCounter",
        )
        if (type(self.value) is not int or self.value < 0
                or type(self.sample_count) is not int
                or self.sample_count <= 0):
            raise ValueError("F-01 counter 必须有非负值和正样本数")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise ValueError("F-01 counter trace 必须是非空严格整数 tuple")


@dataclass(frozen=True, order=True)
class FacilityIntegrityCheck:
    """保存一个恢复、隔离或确定性检查的前后规范身份。"""

    check_key: ProtocolKey
    passed: bool
    before: CanonicalIdentity
    after: CanonicalIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝无身份、非严格布尔或空检查 trace。"""
        if not isinstance(self.check_key, ProtocolKey):
            raise TypeError("F-01 integrity check_key 类型错误")
        if type(self.passed) is not bool:
            raise TypeError("F-01 integrity passed 必须是严格 bool")
        if (not isinstance(self.before, CanonicalIdentity)
                or not isinstance(self.after, CanonicalIdentity)):
            raise TypeError("F-01 integrity 前后状态必须是规范身份")
        assert_int(*self.trace, _where="FacilityIntegrityCheck.trace")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise ValueError("F-01 integrity trace 必须是非空严格整数 tuple")


@dataclass(frozen=True)
class FacilityExerciseMeasurement:
    """保存一次真实总装 exercise 的行为、计数和完整性证据。"""

    exercise_key: ProtocolKey
    query_key: tuple[int, ...]
    positive_behavior: int
    negative_behavior: int
    counters: tuple[FacilityCounter, ...]
    checks: tuple[FacilityIntegrityCheck, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求全部证据唯一稳定排序，且行为值来自正样本。"""
        if not isinstance(self.exercise_key, ProtocolKey):
            raise TypeError("F-01 exercise_key 类型错误")
        assert_int(
            *self.query_key,
            self.positive_behavior,
            self.negative_behavior,
            *self.trace,
            _where="FacilityExerciseMeasurement",
        )
        if (not isinstance(self.query_key, tuple) or not self.query_key
                or any(type(item) is not int for item in self.query_key)):
            raise ValueError("F-01 query_key 必须是非空严格整数 tuple")
        if (type(self.positive_behavior) is not int
                or type(self.negative_behavior) is not int):
            raise TypeError("F-01 behavior 必须是严格整数")
        if (not isinstance(self.counters, tuple) or not self.counters
                or any(not isinstance(item, FacilityCounter)
                       for item in self.counters)):
            raise TypeError("F-01 counters 必须是非空 typed tuple")
        if (not isinstance(self.checks, tuple) or not self.checks
                or any(not isinstance(item, FacilityIntegrityCheck)
                       for item in self.checks)):
            raise TypeError("F-01 checks 必须是非空 typed tuple")
        counter_keys = tuple(item.metric_key for item in self.counters)
        check_keys = tuple(item.check_key for item in self.checks)
        if counter_keys != tuple(sorted(set(counter_keys))):
            raise ValueError("F-01 counters 必须唯一稳定排序")
        if check_keys != tuple(sorted(set(check_keys))):
            raise ValueError("F-01 checks 必须唯一稳定排序")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise ValueError("F-01 measurement trace 必须非空")

    def counter(self, metric_key: ProtocolKey) -> FacilityCounter | None:
        """按完整协议键读取一个计数，不按位置或名称猜测。"""
        return next(
            (item for item in self.counters if item.metric_key == metric_key),
            None,
        )

    def check(self, check_key: ProtocolKey) -> FacilityIntegrityCheck | None:
        """按完整协议键读取一个完整性检查。"""
        return next(
            (item for item in self.checks if item.check_key == check_key),
            None,
        )


@dataclass(frozen=True, order=True)
class FacilityCounterRequirement:
    """声明一个维度所需计数的预注册下界。"""

    metric_key: ProtocolKey
    minimum_value: int

    def __post_init__(self) -> None:
        """要求下界为正严格整数，防止零计数冒充证据。"""
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("F-01 counter requirement 类型错误")
        assert_int(self.minimum_value, _where="FacilityCounterRequirement")
        if type(self.minimum_value) is not int or self.minimum_value <= 0:
            raise ValueError("F-01 counter requirement 必须为正")


@dataclass(frozen=True, order=True)
class FacilityDimensionRequirement:
    """冻结一个总装维度的行为差、计数和完整性要求。"""

    dimension_key: ProtocolKey
    minimum_behavior_improvement: int
    counters: tuple[FacilityCounterRequirement, ...]
    checks: tuple[ProtocolKey, ...]

    def __post_init__(self) -> None:
        """要求维度至少有一项真实计数和一项独立完整性检查。"""
        if not isinstance(self.dimension_key, ProtocolKey):
            raise TypeError("F-01 dimension key 类型错误")
        assert_int(
            self.minimum_behavior_improvement,
            _where="FacilityDimensionRequirement",
        )
        if (type(self.minimum_behavior_improvement) is not int
                or self.minimum_behavior_improvement <= 0):
            raise ValueError("F-01 行为改善下界必须为正")
        if (not isinstance(self.counters, tuple) or not self.counters
                or any(not isinstance(item, FacilityCounterRequirement)
                       for item in self.counters)):
            raise TypeError("F-01 dimension counters 非法")
        if (not isinstance(self.checks, tuple) or not self.checks
                or any(not isinstance(item, ProtocolKey)
                       for item in self.checks)):
            raise TypeError("F-01 dimension checks 非法")
        counter_keys = tuple(item.metric_key for item in self.counters)
        if counter_keys != tuple(sorted(set(counter_keys))):
            raise ValueError("F-01 dimension counter 不得重复或乱序")
        if self.checks != tuple(sorted(set(self.checks))):
            raise ValueError("F-01 dimension check 不得重复或乱序")


@dataclass(frozen=True)
class FacilityReadinessProtocol:
    """预注册 F-01 承重机制、exercise、禁用信号和逐维阈值。"""

    version: int
    exercise_key: ProtocolKey
    dimensions: tuple[FacilityDimensionRequirement, ...]
    required_mechanism_ids: tuple[str, ...]
    forbidden_counter_keys: tuple[ProtocolKey, ...]
    boundary_keys: tuple[ProtocolKey, ...]

    def __post_init__(self) -> None:
        """要求协议各集合完整、唯一并稳定排序。"""
        assert_int(self.version, _where="FacilityReadinessProtocol.version")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("F-01 protocol version 必须为正")
        if not isinstance(self.exercise_key, ProtocolKey):
            raise TypeError("F-01 protocol exercise_key 类型错误")
        if (not isinstance(self.dimensions, tuple) or not self.dimensions
                or any(not isinstance(item, FacilityDimensionRequirement)
                       for item in self.dimensions)):
            raise TypeError("F-01 protocol dimensions 非法")
        dimension_keys = tuple(item.dimension_key for item in self.dimensions)
        if dimension_keys != tuple(sorted(set(dimension_keys))):
            raise ValueError("F-01 protocol dimensions 必须唯一稳定排序")
        if (not isinstance(self.required_mechanism_ids, tuple)
                or not self.required_mechanism_ids
                or any(not isinstance(item, str) or not item
                       for item in self.required_mechanism_ids)
                or self.required_mechanism_ids
                != tuple(sorted(set(self.required_mechanism_ids)))):
            raise ValueError("F-01 required mechanisms 必须唯一稳定排序")
        for values, label in (
                (self.forbidden_counter_keys, "forbidden counter"),
                (self.boundary_keys, "boundary")):
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(item, ProtocolKey)
                           for item in values)
                    or values != tuple(sorted(set(values)))):
                raise ValueError(f"F-01 {label} keys 必须唯一稳定排序")

    def identity(self) -> CanonicalIdentity:
        """返回包含全部阈值、机制和边界的规范协议身份。"""
        return CanonicalIdentity.from_value(self)


@dataclass(frozen=True, order=True)
class FacilityMechanismCheck:
    """保存一个承重机制的四态、owner 和读写闭环审计。"""

    mechanism_id: str
    status: str
    owner: str
    writer_count: int
    reader_count: int
    passed: bool


@dataclass(frozen=True, order=True)
class FacilityDimensionResult:
    """保存一个总装维度的实际计数、检查和最终判定。"""

    requirement: FacilityDimensionRequirement
    observed_counters: tuple[FacilityCounter, ...]
    observed_checks: tuple[FacilityIntegrityCheck, ...]
    behavior_improvement: int
    passed: bool


@dataclass(frozen=True)
class FacilityExerciseResult:
    """保存总装 exercise、Core 前后身份和 adapter 状态身份。"""

    measurement: FacilityExerciseMeasurement
    core_before: CanonicalIdentity
    core_after: CanonicalIdentity
    adapter_state: CanonicalIdentity

    @property
    def core_unchanged(self) -> bool:
        """返回总装实际运行前后 Core 是否位级等价。"""
        return self.core_before == self.core_after


@dataclass(frozen=True)
class FacilityReadinessReport:
    """汇总 F-01 逐维、机制四态、Core 和诚实边界证据。"""

    protocol_identity: CanonicalIdentity
    exercise: FacilityExerciseResult
    mechanisms: tuple[FacilityMechanismCheck, ...]
    dimensions: tuple[FacilityDimensionResult, ...]
    forbidden_counters: tuple[FacilityCounter, ...]
    boundary_keys: tuple[ProtocolKey, ...]
    facility_complete: bool

    def __post_init__(self) -> None:
        """要求报告完整保留非空机制、维度、禁用信号和边界。"""
        if not isinstance(self.protocol_identity, CanonicalIdentity):
            raise TypeError("F-01 report protocol identity 类型错误")
        if not isinstance(self.exercise, FacilityExerciseResult):
            raise TypeError("F-01 report exercise 类型错误")
        if not self.mechanisms or not self.dimensions:
            raise ValueError("F-01 report 缺少机制或维度")
        if not self.forbidden_counters or not self.boundary_keys:
            raise ValueError("F-01 report 缺少禁用信号或诚实边界")
        if type(self.facility_complete) is not bool:
            raise TypeError("F-01 facility_complete 必须是严格 bool")


__all__ = [
    "FacilityCounter",
    "FacilityCounterRequirement",
    "FacilityDimensionRequirement",
    "FacilityDimensionResult",
    "FacilityExerciseMeasurement",
    "FacilityExerciseResult",
    "FacilityIntegrityCheck",
    "FacilityMechanismCheck",
    "FacilityReadinessProtocol",
    "FacilityReadinessReport",
]
