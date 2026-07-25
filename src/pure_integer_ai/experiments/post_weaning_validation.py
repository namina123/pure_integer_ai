"""断奶后 Memory 消融、独立轨道窗口和开放摄入停止协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationDataIdentity,
    EvaluationPlan,
    EvaluationProtocolError,
    ProbeObservation,
    ProbeOutcome,
    ProtocolKey,
)


@dataclass(frozen=True)
class PostWeaningResourceBound:
    """声明一条断奶后轨道上的独立资源上下界。"""

    metric_key: ProtocolKey
    minimum_value: int | None = None
    maximum_value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("V-05 resource metric key 类型错误")
        values = tuple(
            value for value in (self.minimum_value, self.maximum_value)
            if value is not None
        )
        if not values:
            raise ValueError("V-05 资源预算至少需要一个边界")
        assert_int(*values, _where="PostWeaningResourceBound")
        if any(type(value) is not int for value in values):
            raise TypeError("V-05 资源边界必须是严格整数")
        if (self.minimum_value is not None
                and self.maximum_value is not None
                and self.minimum_value > self.maximum_value):
            raise ValueError("V-05 资源下界不得大于上界")

    def accepts(self, value: int) -> bool:
        """按当前指标自身边界判断，不允许跨资源维度补偿。"""
        assert_int(value, _where="PostWeaningResourceBound.accepts")
        if type(value) is not int:
            raise TypeError("V-05 资源测量值必须是严格整数")
        return (
            (self.minimum_value is None or value >= self.minimum_value)
            and (self.maximum_value is None or value <= self.maximum_value)
        )


@dataclass(frozen=True)
class PostWeaningResourceMeasurement:
    """保存一项有来源且实际采样的断奶后资源测量。"""

    metric_key: ProtocolKey
    source_key: ProtocolKey
    value: int
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric_key, ProtocolKey):
            raise TypeError("V-05 resource metric key 类型错误")
        if not isinstance(self.source_key, ProtocolKey):
            raise TypeError("V-05 resource source key 类型错误")
        assert_int(
            self.value,
            self.sample_count,
            _where="PostWeaningResourceMeasurement",
        )
        if type(self.value) is not int or type(self.sample_count) is not int:
            raise TypeError("V-05 资源测量必须是严格整数")
        if self.sample_count <= 0:
            raise ValueError("V-05 资源测量 sample_count 必须为正")


@dataclass(frozen=True)
class PostWeaningHistoryTrial:
    """保存一个上一 episode 扰动下的行为、Use 和当前 query 身份。"""

    variant_key: ProtocolKey
    behavior_value: int
    memory_use_count: int
    current_query_key: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_int(
            self.behavior_value,
            self.memory_use_count,
            *self.current_query_key,
            _where="PostWeaningHistoryTrial",
        )
        if (type(self.behavior_value) is not int
                or type(self.memory_use_count) is not int
                or any(type(value) is not int
                       for value in self.current_query_key)):
            raise TypeError("V-05 history trial 必须使用严格整数")
        if self.memory_use_count < 0 or not self.current_query_key:
            raise ValueError("V-05 history trial 的 Use/当前 query 非法")


@dataclass(frozen=True)
class PostWeaningProbeMeasurement:
    """保留行为、Use、消费、来源、冲突和恢复的不可混计测量。"""

    outcome: ProbeOutcome
    memory_use_count: int
    attractor_consumption_count: int
    current_query_binding_checks: int
    source_trust_checks: int
    conflict_checks: int
    current_query_key: tuple[int, ...]
    work_memory_closed: bool
    query_resources_closed: bool
    rollback_before: CanonicalIdentity
    rollback_after: CanonicalIdentity
    recovery_before: CanonicalIdentity
    recovery_after: CanonicalIdentity
    history_trials: tuple[PostWeaningHistoryTrial, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ProbeOutcome):
            raise TypeError("V-05 measurement outcome 类型错误")
        values = (
            self.memory_use_count,
            self.attractor_consumption_count,
            self.current_query_binding_checks,
            self.source_trust_checks,
            self.conflict_checks,
        )
        assert_int(*values, *self.current_query_key,
                   _where="PostWeaningProbeMeasurement")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("V-05 measurement 计数必须是非负严格整数")
        if (not isinstance(self.current_query_key, tuple)
                or not self.current_query_key
                or any(type(value) is not int
                       for value in self.current_query_key)):
            raise TypeError("V-05 当前 query key 必须是非空严格整数 tuple")
        if (type(self.work_memory_closed) is not bool
                or type(self.query_resources_closed) is not bool):
            raise TypeError("V-05 资源关闭标记必须是 bool")
        identities = (
            self.rollback_before,
            self.rollback_after,
            self.recovery_before,
            self.recovery_after,
        )
        if any(not isinstance(item, CanonicalIdentity) for item in identities):
            raise TypeError("V-05 rollback/recovery 必须使用完整规范身份")
        if (not isinstance(self.history_trials, tuple)
                or len(self.history_trials) < 2
                or any(not isinstance(item, PostWeaningHistoryTrial)
                       for item in self.history_trials)):
            raise ValueError("V-05 至少需要两个上一 episode 扰动 trial")
        variants = tuple(item.variant_key for item in self.history_trials)
        if len(set(variants)) != len(variants):
            raise ValueError("V-05 history trial variant 不得重复")

    @property
    def rollback_verified(self) -> bool:
        """返回故障回滚前后完整状态是否一致。"""
        return self.rollback_before == self.rollback_after

    @property
    def recovery_verified(self) -> bool:
        """返回长期恢复前后完整状态是否一致。"""
        return self.recovery_before == self.recovery_after

    @property
    def history_independent(self) -> bool:
        """核验上一 episode 扰动不改变行为、Use 或当前 query 身份。"""
        baseline = self.history_trials[0]
        return all(
            (
                item.behavior_value,
                item.memory_use_count,
                item.current_query_key,
            ) == (
                baseline.behavior_value,
                baseline.memory_use_count,
                baseline.current_query_key,
            )
            for item in self.history_trials[1:]
        )


@dataclass(frozen=True)
class PostWeaningAblationCase:
    """绑定一条停止轨道上的 Memory 干预、输入、维度和行为增益。"""

    case_key: ProtocolKey
    track_key: ProtocolKey
    intervention_key: ProtocolKey
    evaluator_key: ProtocolKey
    identity: EvaluationDataIdentity
    dimension: ProtocolKey
    expected_query_key: tuple[int, ...]
    minimum_behavior_improvement: int

    def __post_init__(self) -> None:
        keys = (
            self.case_key,
            self.track_key,
            self.intervention_key,
            self.evaluator_key,
            self.dimension,
        )
        if any(not isinstance(item, ProtocolKey) for item in keys):
            raise TypeError("V-05 ablation case 协议键类型错误")
        if not isinstance(self.identity, EvaluationDataIdentity):
            raise TypeError("V-05 ablation identity 类型错误")
        if (not isinstance(self.expected_query_key, tuple)
                or not self.expected_query_key):
            raise ValueError("V-05 case 必须预注册非空 query 身份")
        assert_int(
            *self.expected_query_key,
            self.minimum_behavior_improvement,
            _where="PostWeaningAblationCase",
        )
        if (any(type(value) is not int for value in self.expected_query_key)
                or type(self.minimum_behavior_improvement) is not int
                or self.minimum_behavior_improvement <= 0):
            raise ValueError("V-05 query 身份和行为增益必须使用严格整数")


@dataclass(frozen=True)
class PostWeaningTrackRequirement:
    """冻结一条独立轨道的维度、资源、连续窗口和 checkpoint 间隔。"""

    track_key: ProtocolKey
    dimensions: tuple[ProtocolKey, ...]
    resource_bounds: tuple[PostWeaningResourceBound, ...]
    consecutive_windows: int
    checkpoint_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.track_key, ProtocolKey):
            raise TypeError("V-05 track key 类型错误")
        if (not isinstance(self.dimensions, tuple)
                or not self.dimensions
                or any(not isinstance(item, ProtocolKey)
                       for item in self.dimensions)
                or len(set(self.dimensions)) != len(self.dimensions)):
            raise ValueError("V-05 track dimensions 必须非空且不重复")
        if (not isinstance(self.resource_bounds, tuple)
                or not self.resource_bounds
                or any(not isinstance(item, PostWeaningResourceBound)
                       for item in self.resource_bounds)):
            raise ValueError("V-05 track resource bounds 非法")
        metric_keys = tuple(item.metric_key for item in self.resource_bounds)
        if len(set(metric_keys)) != len(metric_keys):
            raise ValueError("V-05 track resource metric 不得重复")
        assert_int(
            self.consecutive_windows,
            self.checkpoint_step,
            _where="PostWeaningTrackRequirement",
        )
        if (type(self.consecutive_windows) is not int
                or self.consecutive_windows < 2):
            raise ValueError("V-05 track 至少需要两个连续窗口")
        if type(self.checkpoint_step) is not int or self.checkpoint_step <= 0:
            raise ValueError("V-05 track checkpoint step 必须为正")


@dataclass(frozen=True)
class PostWeaningValidationProtocol:
    """冻结全部独立轨道、消融覆盖和预注册停止条件。"""

    version: int
    cases: tuple[PostWeaningAblationCase, ...]
    required_case_keys: tuple[ProtocolKey, ...]
    tracks: tuple[PostWeaningTrackRequirement, ...]
    required_track_keys: tuple[ProtocolKey, ...]

    def __post_init__(self) -> None:
        assert_int(self.version, _where="PostWeaningValidationProtocol")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("V-05 protocol version 必须为正严格整数")
        if (not isinstance(self.cases, tuple) or not self.cases
                or any(not isinstance(item, PostWeaningAblationCase)
                       for item in self.cases)):
            raise ValueError("V-05 cases 必须为非空 typed tuple")
        if (not isinstance(self.required_case_keys, tuple)
                or not self.required_case_keys
                or any(not isinstance(item, ProtocolKey)
                       for item in self.required_case_keys)):
            raise ValueError("V-05 required case keys 非法")
        if (not isinstance(self.tracks, tuple) or not self.tracks
                or any(not isinstance(item, PostWeaningTrackRequirement)
                       for item in self.tracks)):
            raise ValueError("V-05 tracks 必须为非空 typed tuple")
        if (not isinstance(self.required_track_keys, tuple)
                or not self.required_track_keys
                or any(not isinstance(item, ProtocolKey)
                       for item in self.required_track_keys)):
            raise ValueError("V-05 required track keys 非法")
        case_keys = tuple(item.case_key for item in self.cases)
        track_keys = tuple(item.track_key for item in self.tracks)
        for values, label in (
                (case_keys, "case"),
                (self.required_case_keys, "required case"),
                (track_keys, "track"),
                (self.required_track_keys, "required track")):
            if len(set(values)) != len(values):
                raise ValueError(f"V-05 {label} 不得重复")
        if set(case_keys) != set(self.required_case_keys):
            raise ValueError("V-05 cases 必须精确覆盖 required case")
        if set(track_keys) != set(self.required_track_keys):
            raise ValueError("V-05 tracks 必须精确覆盖 required track")
        track_map = {item.track_key: item for item in self.tracks}
        for track_key, track in track_map.items():
            case_dimensions = {
                item.dimension for item in self.cases
                if item.track_key == track_key
            }
            if case_dimensions != set(track.dimensions):
                raise ValueError("V-05 每条 track 必须精确覆盖预注册维度")
        if any(item.track_key not in track_map for item in self.cases):
            raise ValueError("V-05 case 引用了未注册 track")

    def identity(self) -> CanonicalIdentity:
        """生成包含全部轨道和阈值的不可变协议身份。"""
        return CanonicalIdentity.from_value(self)

    def track(self, track_key: ProtocolKey) -> PostWeaningTrackRequirement:
        """读取一条预注册停止轨道，未知键不得 fallback。"""
        for item in self.tracks:
            if item.track_key == track_key:
                return item
        raise EvaluationProtocolError("V-05 track key 未注册")

    def validate_plan(self, plan: EvaluationPlan) -> None:
        """核验所有 Memory 消融输入与 V-00 split、维度和对抗覆盖一致。"""
        all_dimensions = {
            dimension for track in self.tracks
            for dimension in track.dimensions
        }
        if all_dimensions != set(plan.protocol.required_dimensions):
            raise EvaluationProtocolError(
                "V-05 track 维度并集必须覆盖 V-00 required dimensions")
        adversarial_kinds: set[ProtocolKey] = set()
        for case in self.cases:
            assignment = plan.assignment_for(case.identity)
            if assignment.split not in {
                    plan.protocol.held_out_split,
                    plan.protocol.adversarial_split}:
                raise EvaluationProtocolError(
                    "V-05 case 只能使用 held-out/adversarial 输入")
            if case.dimension not in assignment.dimensions:
                raise EvaluationProtocolError("V-05 case 维度未在计划声明")
            if assignment.probe_kind is None:
                raise EvaluationProtocolError("V-05 case 缺少 probe kind")
            if assignment.split == plan.protocol.adversarial_split:
                adversarial_kinds.add(assignment.probe_kind)
        if adversarial_kinds != set(
                plan.protocol.required_adversarial_kinds):
            raise EvaluationProtocolError(
                "V-05 cases 必须覆盖全部 required adversarial kind")


@dataclass(frozen=True)
class PostWeaningProbeResult:
    """把 V-00 observation 与未压缩的 Memory 完整性测量绑定。"""

    observation: ProbeObservation
    measurement: PostWeaningProbeMeasurement


@dataclass(frozen=True)
class PostWeaningAblationPairResult:
    """保存同一 Memory case 的独立 ON/OFF 结果。"""

    case: PostWeaningAblationCase
    enabled: PostWeaningProbeResult
    disabled: PostWeaningProbeResult

    @staticmethod
    def _integrity_complete(measurement: PostWeaningProbeMeasurement) -> bool:
        """核验两臂都必须满足的隔离、回滚、恢复和历史独立性。"""
        return (
            measurement.work_memory_closed
            and measurement.query_resources_closed
            and measurement.rollback_verified
            and measurement.recovery_verified
            and measurement.history_independent
        )

    @property
    def complete(self) -> bool:
        """要求行为增益、真实 Use/消费及全部完整性证据同时成立。"""
        on = self.enabled.measurement
        off = self.disabled.measurement
        return (
            self.enabled.observation.identity == self.case.identity
            and self.disabled.observation.identity == self.case.identity
            and self.enabled.observation.dimension == self.case.dimension
            and self.disabled.observation.dimension == self.case.dimension
            and on.outcome.passed is True
            and on.outcome.sample_count > 0
            and off.outcome.passed is False
            and off.outcome.sample_count > 0
            and on.outcome.value - off.outcome.value
            >= self.case.minimum_behavior_improvement
            and on.memory_use_count > 0
            and off.memory_use_count == 0
            and on.attractor_consumption_count > 0
            and off.attractor_consumption_count == 0
            and on.current_query_binding_checks > 0
            and on.source_trust_checks > 0
            and on.conflict_checks > 0
            and on.current_query_key == self.case.expected_query_key
            and off.current_query_key == self.case.expected_query_key
            and all(
                trial.current_query_key == self.case.expected_query_key
                for measurement in (on, off)
                for trial in measurement.history_trials
            )
            and self._integrity_complete(on)
            and self._integrity_complete(off)
        )


@dataclass(frozen=True)
class PostWeaningDimensionResult:
    """保存一条轨道上每个维度的成对 case 通过计数。"""

    dimension: ProtocolKey
    planned: int
    passed: int
    failed: int

    def __post_init__(self) -> None:
        assert_int(
            self.planned,
            self.passed,
            self.failed,
            _where="PostWeaningDimensionResult",
        )
        if any(type(value) is not int or value < 0 for value in (
                self.planned, self.passed, self.failed)):
            raise ValueError("V-05 dimension 计数必须是非负严格整数")
        if self.passed + self.failed != self.planned:
            raise ValueError("V-05 dimension passed/failed 必须等于 planned")


@dataclass(frozen=True)
class PostWeaningTrackWindow:
    """保存单一轨道在一个真实 checkpoint 的行为和资源证据。"""

    track_key: ProtocolKey
    checkpoint: int
    plan_sha256: str
    protocol_sha256: str
    evaluator_state: CanonicalIdentity
    intervention_state: CanonicalIdentity
    state_reader_state: CanonicalIdentity
    host_state: CanonicalIdentity
    dimensions: tuple[PostWeaningDimensionResult, ...]
    resources: tuple[PostWeaningResourceMeasurement, ...]

    def __post_init__(self) -> None:
        assert_int(self.checkpoint, _where="PostWeaningTrackWindow")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-05 checkpoint 必须是非负严格整数")
        if not self.plan_sha256 or not self.protocol_sha256:
            raise ValueError("V-05 window 缺少 plan/protocol 身份")
        dimension_keys = tuple(item.dimension for item in self.dimensions)
        resource_keys = tuple(item.metric_key for item in self.resources)
        if len(set(dimension_keys)) != len(dimension_keys):
            raise ValueError("V-05 window dimension 不得重复")
        if len(set(resource_keys)) != len(resource_keys):
            raise ValueError("V-05 window resource 不得重复")


@dataclass(frozen=True)
class PostWeaningValidationRequest:
    """请求测量一条轨道的单一 checkpoint 和所需历史尾窗。"""

    track_key: ProtocolKey
    checkpoint: int
    resources: tuple[PostWeaningResourceMeasurement, ...]
    previous_windows: tuple[PostWeaningTrackWindow, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.track_key, ProtocolKey):
            raise TypeError("V-05 request track key 类型错误")
        assert_int(self.checkpoint, _where="PostWeaningValidationRequest")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-05 request checkpoint 必须非负")
        if (not isinstance(self.resources, tuple) or not self.resources
                or any(not isinstance(item, PostWeaningResourceMeasurement)
                       for item in self.resources)):
            raise ValueError("V-05 request resources 非法")
        if (not isinstance(self.previous_windows, tuple)
                or any(not isinstance(item, PostWeaningTrackWindow)
                       for item in self.previous_windows)):
            raise TypeError("V-05 request previous windows 类型错误")


@dataclass(frozen=True)
class PostWeaningValidationReport:
    """保存单一轨道的消融、连续窗口和非 readiness 停止建议。"""

    protocol_version: int
    track_key: ProtocolKey
    plan_sha256: str
    ablations: tuple[PostWeaningAblationPairResult, ...]
    windows: tuple[PostWeaningTrackWindow, ...]
    stop_allowed: bool

    @property
    def ablations_complete(self) -> bool:
        """返回当前轨道所有 Memory 破坏是否都使目标行为失败。"""
        return bool(self.ablations) and all(
            item.complete for item in self.ablations)


__all__ = [
    "PostWeaningAblationCase",
    "PostWeaningAblationPairResult",
    "PostWeaningDimensionResult",
    "PostWeaningHistoryTrial",
    "PostWeaningProbeMeasurement",
    "PostWeaningProbeResult",
    "PostWeaningResourceBound",
    "PostWeaningResourceMeasurement",
    "PostWeaningTrackRequirement",
    "PostWeaningTrackWindow",
    "PostWeaningValidationProtocol",
    "PostWeaningValidationReport",
    "PostWeaningValidationRequest",
]
