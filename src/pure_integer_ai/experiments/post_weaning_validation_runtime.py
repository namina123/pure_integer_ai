"""在 V-06 clone 中执行断奶后 Memory 消融和独立轨道停止测量。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationAssignment,
    EvaluationPlan,
    EvaluationProtocolError,
    ProtocolKey,
    collected_item_content_identity,
    evaluate_probe,
)
from pure_integer_ai.experiments.post_weaning_validation import (
    PostWeaningAblationCase,
    PostWeaningAblationPairResult,
    PostWeaningDimensionResult,
    PostWeaningProbeMeasurement,
    PostWeaningProbeResult,
    PostWeaningTrackWindow,
    PostWeaningValidationProtocol,
    PostWeaningValidationReport,
    PostWeaningValidationRequest,
)


@dataclass(frozen=True)
class PostWeaningProbeRequest:
    """向 evaluator 暴露无 expected/teacher 的单一 Memory 消融请求。"""

    checkpoint: int
    track_key: ProtocolKey
    case_key: ProtocolKey
    identity: Any
    split: ProtocolKey
    probe_kind: ProtocolKey
    dimension: ProtocolKey
    intervention_key: ProtocolKey
    memory_enabled: bool

    def __post_init__(self) -> None:
        assert_int(self.checkpoint, _where="PostWeaningProbeRequest")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-05 probe checkpoint 必须非负")
        if type(self.memory_enabled) is not bool:
            raise TypeError("V-05 memory_enabled 必须是 bool")


@dataclass(frozen=True)
class PostWeaningEvaluatorBinding:
    """给开放 evaluator key 绑定断奶后测量对象。"""

    evaluator_key: ProtocolKey
    evaluator: Any


@dataclass(frozen=True)
class PostWeaningInterventionBinding:
    """给开放 intervention key 绑定 clone 内 Memory 破坏对象。"""

    intervention_key: ProtocolKey
    intervention: Any


def _strict_state_key(owner: Any, label: str) -> tuple[int, ...]:
    """读取 owner 的非空稳定整数键并拒绝隐式对象身份。"""
    reader = getattr(owner, "state_key", None)
    if not callable(reader):
        raise TypeError(f"{label} 缺少 state_key()")
    key = reader()
    if not isinstance(key, tuple) or not key:
        raise TypeError(f"{label}.state_key() 必须返回非空 tuple")
    assert_int(*key, _where=f"{label}.state_key")
    if any(type(value) is not int for value in key):
        raise TypeError(f"{label}.state_key() 必须使用严格整数")
    return key


class PostWeaningEvaluatorRegistry:
    """保存断奶后 evaluator，并冻结全部行为阈值状态。"""

    def __init__(
            self,
            bindings: tuple[PostWeaningEvaluatorBinding, ...],
            ) -> None:
        """建立无重复 evaluator 路由并预验测量接口。"""
        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("V-05 evaluator bindings 不能为空")
        mapping = {}
        for binding in bindings:
            if not isinstance(binding, PostWeaningEvaluatorBinding):
                raise TypeError("V-05 evaluator binding 类型错误")
            if binding.evaluator_key in mapping:
                raise ValueError("V-05 evaluator key 不得重复")
            if not callable(getattr(binding.evaluator, "evaluate", None)):
                raise TypeError("V-05 evaluator 缺少 evaluate()")
            _strict_state_key(binding.evaluator, "V-05 evaluator")
            mapping[binding.evaluator_key] = binding.evaluator
        self._bindings = bindings
        self._mapping = mapping

    def get(self, key: ProtocolKey) -> Any:
        """读取 evaluator，未知键不允许 fallback。"""
        try:
            return self._mapping[key]
        except KeyError as exc:
            raise EvaluationProtocolError("V-05 evaluator 未注册") from exc

    def state_key(self) -> tuple[int, ...]:
        """按 evaluator key 排序组合完整稳定状态。"""
        out = [1, len(self._bindings)]
        for binding in sorted(
                self._bindings,
                key=lambda item: item.evaluator_key.stable_key()):
            key = binding.evaluator_key.stable_key()
            state = _strict_state_key(binding.evaluator, "V-05 evaluator")
            out.extend((len(key), *key, len(state), *state))
        return tuple(out)


class PostWeaningInterventionRegistry:
    """保存只修改当前 V-06 clone 的断奶后 Memory 干预。"""

    def __init__(
            self,
            bindings: tuple[PostWeaningInterventionBinding, ...],
            ) -> None:
        """建立无重复 intervention 路由并预验 apply/state_key。"""
        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("V-05 intervention bindings 不能为空")
        mapping = {}
        for binding in bindings:
            if not isinstance(binding, PostWeaningInterventionBinding):
                raise TypeError("V-05 intervention binding 类型错误")
            if binding.intervention_key in mapping:
                raise ValueError("V-05 intervention key 不得重复")
            if not callable(getattr(binding.intervention, "apply", None)):
                raise TypeError("V-05 intervention 缺少 apply()")
            _strict_state_key(binding.intervention, "V-05 intervention")
            mapping[binding.intervention_key] = binding.intervention
        self._bindings = bindings
        self._mapping = mapping

    def get(self, key: ProtocolKey) -> Any:
        """读取 intervention，未知键不允许 fallback。"""
        try:
            return self._mapping[key]
        except KeyError as exc:
            raise EvaluationProtocolError("V-05 intervention 未注册") from exc

    def state_key(self) -> tuple[int, ...]:
        """按 intervention key 排序组合完整稳定状态。"""
        out = [1, len(self._bindings)]
        for binding in sorted(
                self._bindings,
                key=lambda item: item.intervention_key.stable_key()):
            key = binding.intervention_key.stable_key()
            state = _strict_state_key(
                binding.intervention,
                "V-05 intervention",
            )
            out.extend((len(key), *key, len(state), *state))
        return tuple(out)


class PostWeaningValidationRuntime:
    """执行单一断奶后轨道 checkpoint 的 Memory 消融和停止判定。"""

    def __init__(
            self,
            protocol: PostWeaningValidationProtocol,
            evaluators: PostWeaningEvaluatorRegistry,
            interventions: PostWeaningInterventionRegistry,
            state_reader: Any,
            ) -> None:
        """绑定不可变协议、测量 owner、干预 owner 和宿主状态源。"""
        if not isinstance(protocol, PostWeaningValidationProtocol):
            raise TypeError("V-05 validation protocol 类型错误")
        if not isinstance(evaluators, PostWeaningEvaluatorRegistry):
            raise TypeError("V-05 evaluator registry 类型错误")
        if not isinstance(interventions, PostWeaningInterventionRegistry):
            raise TypeError("V-05 intervention registry 类型错误")
        if not callable(getattr(state_reader, "read", None)):
            raise TypeError("V-05 state reader 缺少 read()")
        _strict_state_key(state_reader, "V-05 state reader")
        self.protocol = protocol
        self.evaluators = evaluators
        self.interventions = interventions
        self.state_reader = state_reader

    def state_key(self) -> tuple[int, ...]:
        """返回 protocol 及三个注入 owner 的完整稳定状态。"""
        protocol_digest = bytes.fromhex(self.protocol.identity().sha256)
        evaluator = self.evaluators.state_key()
        intervention = self.interventions.state_key()
        reader = _strict_state_key(self.state_reader, "V-05 state reader")
        return (
            1,
            *protocol_digest,
            len(evaluator),
            *evaluator,
            len(intervention),
            *intervention,
            len(reader),
            *reader,
        )

    def _runtime_state(self, ctx: Any) -> tuple[Any, ...]:
        """读取宿主及注入 owner 状态，阻断评测写入和阈值后调。"""
        return (
            self.evaluators.state_key(),
            self.interventions.state_key(),
            _strict_state_key(self.state_reader, "V-05 state reader"),
            self.state_reader.read(ctx),
        )

    @staticmethod
    def _evaluation_item(ctx: Any, assignment: EvaluationAssignment) -> Any:
        """按完整来源和内容读取唯一 held-out/adversarial 输入。"""
        items = ctx.evaluation_corpora.get(assignment.split, ())
        matches = [
            item for item in items
            if item.source_ref is not None
            and item.source_ref.stable_key()
            == assignment.identity.source_ref.stable_key()
            and collected_item_content_identity(item)
            == assignment.identity.content
        ]
        if len(matches) != 1:
            raise EvaluationProtocolError(
                "V-05 split 中的完整数据身份缺失或不唯一")
        return matches[0]

    def _evaluate_arm(
            self,
            ctx: Any,
            case: PostWeaningAblationCase,
            checkpoint: int,
            *,
            enabled: bool,
            ) -> PostWeaningProbeResult:
        """在独立 clone 中执行单臂，并把 typed 测量交给 V-00 行为观察。"""
        plan = ctx.evaluation_plan
        assignment = plan.assignment_for(case.identity)
        if assignment.probe_kind is None:
            raise EvaluationProtocolError("V-05 case 缺少 probe kind")
        evaluator = self.evaluators.get(case.evaluator_key)
        intervention = self.interventions.get(case.intervention_key)
        item = self._evaluation_item(ctx, assignment)
        request = PostWeaningProbeRequest(
            checkpoint,
            case.track_key,
            case.case_key,
            case.identity,
            assignment.split,
            assignment.probe_kind,
            case.dimension,
            case.intervention_key,
            enabled,
        )
        measured: list[PostWeaningProbeMeasurement] = []

        def invoke():
            """建立无 teacher/expected 的 clone，并收集一次不可重复 typed 测量。"""
            with isolated_evaluation(
                    ctx,
                    label=(
                        f"v05-{self.protocol.version}-"
                        f"{checkpoint}-{case.case_key.components}-"
                        f"{int(enabled)}"
                    )) as eval_ctx:
                eval_ctx.teacher = None
                eval_ctx.evaluation_plan = EvaluationPlan(
                    plan.protocol,
                    tuple(
                        replace(row, expected_outcome=None)
                        for row in plan.assignments
                    ),
                )
                intervention.apply(eval_ctx, enabled=enabled)
                measurement = evaluator.evaluate(
                    eval_ctx,
                    copy.deepcopy(item),
                    request,
                )
                if not isinstance(measurement, PostWeaningProbeMeasurement):
                    raise TypeError(
                        "V-05 evaluator 必须返回 PostWeaningProbeMeasurement")
                measured.append(measurement)
                return measurement.outcome

        observation = evaluate_probe(
            plan,
            assignment,
            case.dimension,
            invoke,
            state_reader=lambda: self._runtime_state(ctx),
        )
        if len(measured) != 1:
            raise EvaluationProtocolError("V-05 单臂测量数量不为一")
        return PostWeaningProbeResult(observation, measured[0])

    def _run_case(
            self,
            ctx: Any,
            case: PostWeaningAblationCase,
            checkpoint: int,
            ) -> PostWeaningAblationPairResult:
        """分别运行 Memory ON 和破坏 OFF，禁止共享 clone 或可变状态。"""
        return PostWeaningAblationPairResult(
            case,
            self._evaluate_arm(ctx, case, checkpoint, enabled=True),
            self._evaluate_arm(ctx, case, checkpoint, enabled=False),
        )

    def _window_passes(
            self,
            window: PostWeaningTrackWindow,
            *,
            track: Any,
            plan_sha256: str,
            evaluator_state: CanonicalIdentity,
            intervention_state: CanonicalIdentity,
            state_reader_state: CanonicalIdentity,
            ) -> bool:
        """逐维和逐资源核验一条轨道窗口，不计算综合均值。"""
        if (
                window.track_key != track.track_key
                or window.plan_sha256 != plan_sha256
                or window.protocol_sha256 != self.protocol.identity().sha256
                or window.evaluator_state != evaluator_state
                or window.intervention_state != intervention_state
                or window.state_reader_state != state_reader_state):
            return False
        dimensions = {item.dimension: item for item in window.dimensions}
        if set(dimensions) != set(track.dimensions):
            return False
        if any(
                item.planned <= 0
                or item.passed != item.planned
                or item.failed != 0
                for item in dimensions.values()):
            return False
        resources = {item.metric_key: item for item in window.resources}
        bounds = {item.metric_key: item for item in track.resource_bounds}
        if set(resources) != set(bounds):
            return False
        return all(
            measurement.sample_count > 0
            and bounds[key].accepts(measurement.value)
            for key, measurement in resources.items()
        )

    def run(
            self,
            ctx: Any,
            request: PostWeaningValidationRequest,
            ) -> PostWeaningValidationReport:
        """测量单一轨道 checkpoint，满足独立连续窗口后才建议该轨道停止。"""
        if not isinstance(request, PostWeaningValidationRequest):
            raise TypeError("V-05 validation request 类型错误")
        if ctx.evaluation_plan is None or not ctx.evaluation_strictly_isolated:
            raise EvaluationProtocolError("V-05 要求严格 V-00 evaluation plan")
        plan = ctx.evaluation_plan
        self.protocol.validate_plan(plan)
        track = self.protocol.track(request.track_key)
        if len(request.previous_windows) > track.consecutive_windows - 1:
            raise EvaluationProtocolError("V-05 只接受当前轨道所需历史尾窗")
        cases = tuple(
            item for item in self.protocol.cases
            if item.track_key == request.track_key
        )
        plan_sha256 = plan.sha256()
        evaluator_state = CanonicalIdentity.from_value(
            self.evaluators.state_key())
        intervention_state = CanonicalIdentity.from_value(
            self.interventions.state_key())
        state_reader_state = CanonicalIdentity.from_value(
            _strict_state_key(self.state_reader, "V-05 state reader"))
        host_before = CanonicalIdentity.from_value(self.state_reader.read(ctx))
        ablations = tuple(
            self._run_case(ctx, case, request.checkpoint)
            for case in cases
        )
        host_after = CanonicalIdentity.from_value(self.state_reader.read(ctx))
        if host_after != host_before:
            raise EvaluationProtocolError("V-05 运行改变了宿主长期状态")
        dimensions = tuple(
            PostWeaningDimensionResult(
                dimension,
                sum(item.case.dimension == dimension for item in ablations),
                sum(item.case.dimension == dimension and item.complete
                    for item in ablations),
                sum(item.case.dimension == dimension and not item.complete
                    for item in ablations),
            )
            for dimension in track.dimensions
        )
        current = PostWeaningTrackWindow(
            request.track_key,
            request.checkpoint,
            plan_sha256,
            self.protocol.identity().sha256,
            evaluator_state,
            intervention_state,
            state_reader_state,
            host_before,
            dimensions,
            request.resources,
        )
        windows = (*request.previous_windows, current)
        for index in range(1, len(windows)):
            previous = windows[index - 1]
            current_window = windows[index]
            if current_window.track_key != previous.track_key:
                raise EvaluationProtocolError("V-05 历史混入其他停止轨道")
            if (current_window.checkpoint - previous.checkpoint
                    != track.checkpoint_step):
                raise EvaluationProtocolError("V-05 track checkpoint 不连续")
            if current_window.host_state == previous.host_state:
                raise EvaluationProtocolError(
                    "V-05 不得给同一宿主状态重复编号")
        windows_pass = (
            len(windows) == track.consecutive_windows
            and all(self._window_passes(
                item,
                track=track,
                plan_sha256=plan_sha256,
                evaluator_state=evaluator_state,
                intervention_state=intervention_state,
                state_reader_state=state_reader_state,
            ) for item in windows)
        )
        ablations_complete = bool(ablations) and all(
            item.complete for item in ablations)
        return PostWeaningValidationReport(
            self.protocol.version,
            request.track_key,
            plan_sha256,
            ablations,
            windows,
            ablations_complete and windows_pass,
        )


__all__ = [
    "PostWeaningEvaluatorBinding",
    "PostWeaningEvaluatorRegistry",
    "PostWeaningInterventionBinding",
    "PostWeaningInterventionRegistry",
    "PostWeaningProbeRequest",
    "PostWeaningValidationRuntime",
]
