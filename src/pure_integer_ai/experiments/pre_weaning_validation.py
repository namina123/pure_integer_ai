"""断奶前消融、连续 held-out 窗口和资源停止协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationDataIdentity,
    EvaluationPlan,
    EvaluationProtocolError,
    ProbeObservation,
    ProtocolKey,
)


@dataclass(frozen=True)
class PreWeaningAblationCase:
    """绑定一个破坏项、评测输入、目标维度和无答案 evaluator。"""

    case_key: ProtocolKey
    intervention_key: ProtocolKey
    evaluator_key: ProtocolKey
    identity: EvaluationDataIdentity
    dimension: ProtocolKey

    def __post_init__(self) -> None:
        if any(not isinstance(value, ProtocolKey) for value in (
                self.case_key,
                self.intervention_key,
                self.evaluator_key,
                self.dimension)):
            raise TypeError("V-04 消融协议键类型错误")
        if not isinstance(self.identity, EvaluationDataIdentity):
            raise TypeError("V-04 消融 identity 类型错误")


@dataclass(frozen=True)
class PreWeaningProbeRoute:
    """把 V-00 probe kind 路由到一个注入式 evaluator。"""

    probe_kind: ProtocolKey
    evaluator_key: ProtocolKey

    def __post_init__(self) -> None:
        if not isinstance(self.probe_kind, ProtocolKey):
            raise TypeError("V-04 probe kind 类型错误")
        if not isinstance(self.evaluator_key, ProtocolKey):
            raise TypeError("V-04 evaluator key 类型错误")


@dataclass(frozen=True)
class ResourceBound:
    """声明一个资源指标独立适用的可选下界和上界。"""

    metric_key: ProtocolKey
    minimum_value: int | None = None
    maximum_value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("V-04 resource metric key 类型错误")
        values = tuple(
            value for value in (self.minimum_value, self.maximum_value)
            if value is not None
        )
        if values:
            assert_int(*values, _where="ResourceBound")
        if not values:
            raise ValueError("资源预算必须至少声明一个边界")
        if any(type(value) is not int for value in values):
            raise TypeError("资源预算边界必须是严格整数")
        if (self.minimum_value is not None
                and self.maximum_value is not None
                and self.minimum_value > self.maximum_value):
            raise ValueError("资源预算下界不得大于上界")

    def accepts(self, value: int) -> bool:
        """按本指标自己的上下界判断测量值，不允许跨指标抵消。"""
        assert_int(value, _where="ResourceBound.accepts")
        if type(value) is not int:
            raise TypeError("资源预算测量值必须是严格整数")
        return (
            (self.minimum_value is None or value >= self.minimum_value)
            and (self.maximum_value is None or value <= self.maximum_value)
        )


@dataclass(frozen=True)
class ResourceMeasurement:
    """保存一个有来源、非零采样的纯整数资源测量。"""

    metric_key: ProtocolKey
    source_key: ProtocolKey
    value: int
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("V-04 resource metric key 类型错误")
        if not isinstance(self.source_key, ProtocolKey):
            raise TypeError("V-04 resource source key 类型错误")
        assert_int(
            self.value,
            self.sample_count,
            _where="ResourceMeasurement",
        )
        if type(self.value) is not int or type(self.sample_count) is not int:
            raise TypeError("资源测量必须使用严格整数")
        if self.sample_count <= 0:
            raise ValueError("资源测量 sample_count 必须为正")


@dataclass(frozen=True)
class PreWeaningValidationProtocol:
    """冻结消融覆盖、停止维度、墙维度、资源预算和窗口间隔。"""

    version: int
    ablation_cases: tuple[PreWeaningAblationCase, ...]
    required_ablation_cases: tuple[ProtocolKey, ...]
    probe_routes: tuple[PreWeaningProbeRoute, ...]
    stopping_dimensions: tuple[ProtocolKey, ...]
    wall_dimensions: tuple[ProtocolKey, ...]
    resource_bounds: tuple[ResourceBound, ...]
    consecutive_windows: int
    checkpoint_step: int

    def __post_init__(self) -> None:
        assert_int(
            self.version,
            self.consecutive_windows,
            self.checkpoint_step,
            _where="PreWeaningValidationProtocol",
        )
        if any(type(value) is not int for value in (
                self.version,
                self.consecutive_windows,
                self.checkpoint_step)):
            raise TypeError("V-04 version/window/step 必须是严格整数")
        if self.version <= 0:
            raise ValueError("V-04 protocol version 必须为正")
        if self.consecutive_windows < 2:
            raise ValueError("V-04 停止至少需要两个连续窗口")
        if self.checkpoint_step <= 0:
            raise ValueError("V-04 checkpoint_step 必须为正")
        if not self.ablation_cases or not self.required_ablation_cases:
            raise ValueError("V-04 消融 case 和必需 case 均不能为空")
        if not self.probe_routes or not self.stopping_dimensions:
            raise ValueError("V-04 probe route 和停止维度均不能为空")
        if not self.resource_bounds:
            raise ValueError("V-04 资源预算不能为空")
        tuple_fields = (
            self.ablation_cases,
            self.required_ablation_cases,
            self.probe_routes,
            self.stopping_dimensions,
            self.wall_dimensions,
            self.resource_bounds,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise TypeError("V-04 protocol 集合字段必须是 tuple")
        if any(not isinstance(item, PreWeaningAblationCase)
               for item in self.ablation_cases):
            raise TypeError("V-04 ablation case 类型错误")
        if any(not isinstance(item, ProtocolKey)
               for item in self.required_ablation_cases):
            raise TypeError("V-04 required case key 类型错误")
        if any(not isinstance(item, PreWeaningProbeRoute)
               for item in self.probe_routes):
            raise TypeError("V-04 probe route 类型错误")
        if any(not isinstance(item, ProtocolKey)
               for item in (*self.stopping_dimensions,
                            *self.wall_dimensions)):
            raise TypeError("V-04 stopping/wall dimension 类型错误")
        if any(not isinstance(item, ResourceBound)
               for item in self.resource_bounds):
            raise TypeError("V-04 resource bound 类型错误")
        self._require_unique(
            self.required_ablation_cases,
            "required ablation case",
        )
        self._require_unique(self.stopping_dimensions, "停止维度")
        self._require_unique(self.wall_dimensions, "墙维度")
        case_keys = tuple(case.case_key for case in self.ablation_cases)
        self._require_unique(case_keys, "消融 case")
        if set(case_keys) != set(self.required_ablation_cases):
            raise ValueError("消融 case 必须精确覆盖预注册 required case")
        route_kinds = tuple(route.probe_kind for route in self.probe_routes)
        self._require_unique(route_kinds, "probe route")
        resource_keys = tuple(
            bound.metric_key for bound in self.resource_bounds)
        self._require_unique(resource_keys, "资源预算指标")
        if not set(self.wall_dimensions).issubset(self.stopping_dimensions):
            raise ValueError("墙维度必须属于停止维度")
        if any(
                case.dimension in set(self.wall_dimensions)
                for case in self.ablation_cases):
            raise ValueError("W1/W2 墙维度不得被健康消融臂伪造 PASS")

    @staticmethod
    def _require_unique(values: tuple[object, ...], label: str) -> None:
        """拒绝会让覆盖或汇总结果产生歧义的重复协议项。"""
        if len(set(values)) != len(values):
            raise ValueError(f"V-04 {label} 不得重复")

    def identity(self) -> CanonicalIdentity:
        """生成包含全部预注册阈值和路由的不可变协议身份。"""
        return CanonicalIdentity.from_value(self)

    def evaluator_key_for(self, probe_kind: ProtocolKey) -> ProtocolKey:
        """按显式 route 读取 evaluator，未知 probe kind 必须失败。"""
        for route in self.probe_routes:
            if route.probe_kind == probe_kind:
                return route.evaluator_key
        raise EvaluationProtocolError("V-04 probe kind 缺少 evaluator route")

    def validate_plan(self, plan: EvaluationPlan) -> None:
        """核验 V-00 计划能完整承载预注册消融和 held-out 停止维度。"""
        if set(self.stopping_dimensions) != set(
                plan.protocol.required_dimensions):
            raise EvaluationProtocolError(
                "V-04 停止维度必须精确覆盖 V-00 required dimensions")
        held_out = tuple(
            assignment for assignment in plan.assignments
            if assignment.split == plan.protocol.held_out_split
        )
        if not held_out:
            raise EvaluationProtocolError("V-04 缺少 held-out 记录")
        planned_dimensions = {
            dimension
            for assignment in held_out
            for dimension in assignment.dimensions
        }
        if planned_dimensions != set(self.stopping_dimensions):
            raise EvaluationProtocolError(
                "V-04 held-out 必须精确覆盖全部停止维度")
        probe_kinds = {assignment.probe_kind for assignment in held_out}
        if None in probe_kinds:
            raise EvaluationProtocolError("V-04 held-out 记录缺少 probe kind")
        routed = {route.probe_kind for route in self.probe_routes}
        if routed != probe_kinds:
            raise EvaluationProtocolError(
                "V-04 probe route 必须精确覆盖 held-out probe kind")
        adversarial_kinds: set[ProtocolKey] = set()
        for case in self.ablation_cases:
            assignment = plan.assignment_for(case.identity)
            if assignment.split == plan.protocol.training_split:
                raise EvaluationProtocolError("训练记录不得用作 V-04 消融")
            if assignment.probe_kind is None:
                raise EvaluationProtocolError("V-04 消融记录缺少 probe kind")
            if case.dimension not in assignment.dimensions:
                raise EvaluationProtocolError("V-04 消融维度未在计划中声明")
            if assignment.split == plan.protocol.adversarial_split:
                adversarial_kinds.add(assignment.probe_kind)
        if adversarial_kinds != set(
                plan.protocol.required_adversarial_kinds):
            raise EvaluationProtocolError(
                "V-04 消融必须覆盖全部 required adversarial kind")


@dataclass(frozen=True)
class PreWeaningDimensionResult:
    """保存一个 held-out 维度的 PASS、FAIL、NE 和实际采样数。"""

    dimension: ProtocolKey
    planned: int
    passed: int
    failed: int
    not_evaluated: int
    sample_count: int

    def __post_init__(self) -> None:
        assert_int(
            self.planned,
            self.passed,
            self.failed,
            self.not_evaluated,
            self.sample_count,
            _where="PreWeaningDimensionResult",
        )
        values = (
            self.planned,
            self.passed,
            self.failed,
            self.not_evaluated,
            self.sample_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("V-04 维度计数必须是非负严格整数")
        if self.passed + self.failed + self.not_evaluated != self.planned:
            raise ValueError("V-04 维度分项之和必须等于 planned")


@dataclass(frozen=True)
class AblationPairResult:
    """保存同一 case 在独立 ON/OFF clone 中的成对结果。"""

    case: PreWeaningAblationCase
    enabled: ProbeObservation
    disabled: ProbeObservation

    @property
    def complete(self) -> bool:
        """要求健康臂有非零 PASS，破坏臂有非零 FAIL。"""
        return (
            self.enabled.identity == self.case.identity
            and self.disabled.identity == self.case.identity
            and self.enabled.dimension == self.case.dimension
            and self.disabled.dimension == self.case.dimension
            and self.enabled.outcome.passed is True
            and self.enabled.outcome.sample_count > 0
            and self.disabled.outcome.passed is False
            and self.disabled.outcome.sample_count > 0
        )


@dataclass(frozen=True)
class PreWeaningHeldOutWindow:
    """保存一个真实 checkpoint 的 held-out、资源和运行身份快照。"""

    checkpoint: int
    plan_sha256: str
    protocol_sha256: str
    evaluator_state: CanonicalIdentity
    intervention_state: CanonicalIdentity
    state_reader_state: CanonicalIdentity
    host_state: CanonicalIdentity
    dimensions: tuple[PreWeaningDimensionResult, ...]
    resources: tuple[ResourceMeasurement, ...]

    def __post_init__(self) -> None:
        assert_int(self.checkpoint, _where="PreWeaningHeldOutWindow")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-04 checkpoint 不得为负")
        if not self.plan_sha256 or not self.protocol_sha256:
            raise ValueError("V-04 window 必须绑定 plan/protocol 身份")
        if not isinstance(self.dimensions, tuple) or any(
                not isinstance(item, PreWeaningDimensionResult)
                for item in self.dimensions):
            raise TypeError("V-04 window dimensions 类型错误")
        if not isinstance(self.resources, tuple) or any(
                not isinstance(item, ResourceMeasurement)
                for item in self.resources):
            raise TypeError("V-04 window resources 类型错误")
        dimensions = tuple(item.dimension for item in self.dimensions)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("V-04 window 维度结果不得重复")
        resources = tuple(item.metric_key for item in self.resources)
        if len(set(resources)) != len(resources):
            raise ValueError("V-04 window 资源测量不得重复")


@dataclass(frozen=True)
class PreWeaningValidationRequest:
    """请求测量一个 checkpoint，并只携带所需的连续历史尾窗。"""

    checkpoint: int
    resources: tuple[ResourceMeasurement, ...]
    previous_windows: tuple[PreWeaningHeldOutWindow, ...] = ()

    def __post_init__(self) -> None:
        assert_int(self.checkpoint, _where="PreWeaningValidationRequest")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-04 checkpoint 不得为负")
        if not self.resources:
            raise ValueError("V-04 当前窗口资源测量不能为空")
        if not isinstance(self.resources, tuple) or any(
                not isinstance(item, ResourceMeasurement)
                for item in self.resources):
            raise TypeError("V-04 request resources 类型错误")
        if not isinstance(self.previous_windows, tuple) or any(
                not isinstance(item, PreWeaningHeldOutWindow)
                for item in self.previous_windows):
            raise TypeError("V-04 request previous_windows 类型错误")


@dataclass(frozen=True)
class PreWeaningValidationReport:
    """汇总消融证据、连续窗口和不产生 readiness 的停止建议。"""

    protocol_version: int
    plan_sha256: str
    ablations: tuple[AblationPairResult, ...]
    windows: tuple[PreWeaningHeldOutWindow, ...]
    stop_allowed: bool

    @property
    def ablations_complete(self) -> bool:
        """返回全部预注册破坏是否都使对应指标失败。"""
        return bool(self.ablations) and all(
            pair.complete for pair in self.ablations)


__all__ = [
    "AblationPairResult",
    "PreWeaningAblationCase",
    "PreWeaningDimensionResult",
    "PreWeaningHeldOutWindow",
    "PreWeaningProbeRoute",
    "PreWeaningValidationProtocol",
    "PreWeaningValidationReport",
    "PreWeaningValidationRequest",
    "ResourceBound",
    "ResourceMeasurement",
]
