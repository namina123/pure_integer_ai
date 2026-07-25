"""在 V-06 clone 中执行断奶前反向破坏和 held-out 停止测量。"""
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
    ProbeOutcome,
    ProtocolKey,
    collected_item_content_identity,
    evaluate_probe,
)
from pure_integer_ai.experiments.pre_weaning_validation import (
    AblationPairResult,
    PreWeaningAblationCase,
    PreWeaningDimensionResult,
    PreWeaningHeldOutWindow,
    PreWeaningValidationProtocol,
    PreWeaningValidationReport,
    PreWeaningValidationRequest,
)


@dataclass(frozen=True)
class PreWeaningProbeRequest:
    """向 evaluator 暴露无 expected、无 teacher 的最小探针请求。"""

    checkpoint: int
    identity: Any
    split: ProtocolKey
    probe_kind: ProtocolKey
    dimension: ProtocolKey
    case_key: ProtocolKey | None = None
    intervention_key: ProtocolKey | None = None
    intervention_enabled: bool | None = None

    def __post_init__(self) -> None:
        assert_int(self.checkpoint, _where="PreWeaningProbeRequest")
        if type(self.checkpoint) is not int or self.checkpoint < 0:
            raise ValueError("V-04 probe checkpoint 不得为负")
        if self.intervention_enabled is not None and not isinstance(
                self.intervention_enabled, bool):
            raise TypeError("V-04 intervention_enabled 必须是 bool 或 None")


@dataclass(frozen=True)
class PreWeaningEvaluatorBinding:
    """给一个开放 evaluator key 绑定无状态评测对象。"""

    evaluator_key: ProtocolKey
    evaluator: Any


@dataclass(frozen=True)
class PreWeaningInterventionBinding:
    """给一个开放 intervention key 绑定 clone 内破坏对象。"""

    intervention_key: ProtocolKey
    intervention: Any


def _strict_state_key(owner: Any, label: str) -> tuple[int, ...]:
    """读取对象的稳定整数状态键，并拒绝缺失、空键或非严格整数。"""
    reader = getattr(owner, "state_key", None)
    if not callable(reader):
        raise TypeError(f"{label} 缺少 state_key()")
    key = reader()
    if not isinstance(key, tuple) or not key:
        raise TypeError(f"{label}.state_key() 必须返回非空整数 tuple")
    assert_int(*key, _where=f"{label}.state_key")
    if any(type(value) is not int for value in key):
        raise TypeError(f"{label}.state_key() 必须返回严格整数")
    return key


class PreWeaningEvaluatorRegistry:
    """保存注入式 evaluator，并提供防阈值漂移的组合状态键。"""

    def __init__(
            self,
            bindings: tuple[PreWeaningEvaluatorBinding, ...],
            ) -> None:
        """建立不可重复的 evaluator 路由并预验每个状态协议。"""
        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("V-04 evaluator bindings 不能为空")
        mapping: dict[ProtocolKey, Any] = {}
        for binding in bindings:
            if not isinstance(binding, PreWeaningEvaluatorBinding):
                raise TypeError("V-04 evaluator binding 类型错误")
            if binding.evaluator_key in mapping:
                raise ValueError("V-04 evaluator key 不得重复")
            if not callable(getattr(binding.evaluator, "evaluate", None)):
                raise TypeError("V-04 evaluator 缺少 evaluate()")
            _strict_state_key(binding.evaluator, "V-04 evaluator")
            mapping[binding.evaluator_key] = binding.evaluator
        self._bindings = bindings
        self._mapping = mapping

    def get(self, evaluator_key: ProtocolKey) -> Any:
        """读取已注册 evaluator，未知键不允许 fallback。"""
        try:
            return self._mapping[evaluator_key]
        except KeyError as exc:
            raise EvaluationProtocolError(
                "V-04 evaluator key 未注册") from exc

    def state_key(self) -> tuple[int, ...]:
        """按 evaluator key 排序返回包含全部阈值状态的稳定整数键。"""
        out = [1, len(self._bindings)]
        for binding in sorted(
                self._bindings,
                key=lambda item: item.evaluator_key.stable_key()):
            evaluator_key = binding.evaluator_key.stable_key()
            state_key = _strict_state_key(
                binding.evaluator,
                "V-04 evaluator",
            )
            out.extend((len(evaluator_key), *evaluator_key))
            out.extend((len(state_key), *state_key))
        return tuple(out)


class PreWeaningInterventionRegistry:
    """保存只允许修改 V-06 clone 的注入式破坏实现。"""

    def __init__(
            self,
            bindings: tuple[PreWeaningInterventionBinding, ...],
            ) -> None:
        """建立不可重复的 intervention 路由并预验状态协议。"""
        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("V-04 intervention bindings 不能为空")
        mapping: dict[ProtocolKey, Any] = {}
        for binding in bindings:
            if not isinstance(binding, PreWeaningInterventionBinding):
                raise TypeError("V-04 intervention binding 类型错误")
            if binding.intervention_key in mapping:
                raise ValueError("V-04 intervention key 不得重复")
            if not callable(getattr(binding.intervention, "apply", None)):
                raise TypeError("V-04 intervention 缺少 apply()")
            _strict_state_key(binding.intervention, "V-04 intervention")
            mapping[binding.intervention_key] = binding.intervention
        self._bindings = bindings
        self._mapping = mapping

    def get(self, intervention_key: ProtocolKey) -> Any:
        """读取已注册破坏实现，未知键不允许 fallback。"""
        try:
            return self._mapping[intervention_key]
        except KeyError as exc:
            raise EvaluationProtocolError(
                "V-04 intervention key 未注册") from exc

    def state_key(self) -> tuple[int, ...]:
        """按 intervention key 排序返回完整稳定整数状态。"""
        out = [1, len(self._bindings)]
        for binding in sorted(
                self._bindings,
                key=lambda item: item.intervention_key.stable_key()):
            intervention_key = binding.intervention_key.stable_key()
            state_key = _strict_state_key(
                binding.intervention,
                "V-04 intervention",
            )
            out.extend((len(intervention_key), *intervention_key))
            out.extend((len(state_key), *state_key))
        return tuple(out)


class PreWeaningValidationRuntime:
    """执行一个真实 checkpoint 的全套消融、held-out 和停止判定。"""

    def __init__(
            self,
            protocol: PreWeaningValidationProtocol,
            evaluators: PreWeaningEvaluatorRegistry,
            interventions: PreWeaningInterventionRegistry,
            state_reader: Any,
            ) -> None:
        """绑定不可变协议、evaluator、破坏实现和宿主只读状态源。"""
        if not isinstance(protocol, PreWeaningValidationProtocol):
            raise TypeError("V-04 validation protocol 类型错误")
        if not isinstance(evaluators, PreWeaningEvaluatorRegistry):
            raise TypeError("V-04 evaluator registry 类型错误")
        if not isinstance(interventions, PreWeaningInterventionRegistry):
            raise TypeError("V-04 intervention registry 类型错误")
        if not callable(getattr(state_reader, "read", None)):
            raise TypeError("V-04 state reader 缺少 read()")
        _strict_state_key(state_reader, "V-04 state reader")
        self.protocol = protocol
        self.evaluators = evaluators
        self.interventions = interventions
        self.state_reader = state_reader

    def state_key(self) -> tuple[int, ...]:
        """返回运行协议及全部 owner 状态的稳定内容引用。"""
        protocol_digest = bytes.fromhex(self.protocol.identity().sha256)
        evaluator_key = self.evaluators.state_key()
        intervention_key = self.interventions.state_key()
        reader_key = _strict_state_key(
            self.state_reader,
            "V-04 state reader",
        )
        return (
            1,
            *protocol_digest,
            len(evaluator_key),
            *evaluator_key,
            len(intervention_key),
            *intervention_key,
            len(reader_key),
            *reader_key,
        )

    def _runtime_state(self, ctx: Any) -> tuple[Any, ...]:
        """读取宿主及三个注入 owner 的完整状态，供每次 probe 防漂移。"""
        return (
            self.evaluators.state_key(),
            self.interventions.state_key(),
            _strict_state_key(self.state_reader, "V-04 state reader"),
            self.state_reader.read(ctx),
        )

    @staticmethod
    def _evaluation_item(ctx: Any, assignment: EvaluationAssignment) -> Any:
        """按完整来源和内容从 held-out/adversarial split 取唯一输入。"""
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
                "V-04 split 中的完整数据身份缺失或不唯一")
        return matches[0]

    def _evaluate(
            self,
            ctx: Any,
            assignment: EvaluationAssignment,
            dimension: ProtocolKey,
            evaluator_key: ProtocolKey,
            request: PreWeaningProbeRequest,
            *,
            intervention: Any = None,
            enabled: bool | None = None,
            ) -> Any:
        """在独立 V-06 clone 中执行无 teacher 探针，并核验所有宿主状态。"""
        evaluator = self.evaluators.get(evaluator_key)
        item = self._evaluation_item(ctx, assignment)

        def invoke() -> ProbeOutcome:
            """建立单臂沙箱并只向 evaluator 提供最小无答案请求。"""
            with isolated_evaluation(
                    ctx,
                    label=(
                        f"v04-{self.protocol.version}-"
                        f"{request.checkpoint}-"
                        f"{assignment.identity.content.index}-"
                        f"{dimension.components}"
                    )) as eval_ctx:
                eval_ctx.teacher = None
                eval_ctx.evaluation_plan = EvaluationPlan(
                    ctx.evaluation_plan.protocol,
                    tuple(
                        replace(item, expected_outcome=None)
                        for item in ctx.evaluation_plan.assignments
                    ),
                )
                if intervention is not None:
                    intervention.apply(eval_ctx, enabled=enabled)
                outcome = evaluator.evaluate(
                    eval_ctx,
                    copy.deepcopy(item),
                    request,
                )
                if not isinstance(outcome, ProbeOutcome):
                    raise TypeError("V-04 evaluator 必须返回 ProbeOutcome")
                return outcome

        return evaluate_probe(
            ctx.evaluation_plan,
            assignment,
            dimension,
            invoke,
            state_reader=lambda: self._runtime_state(ctx),
        )

    def _run_ablation_case(
            self,
            ctx: Any,
            case: PreWeaningAblationCase,
            checkpoint: int,
            ) -> AblationPairResult:
        """分别克隆健康臂和破坏臂，禁止两个臂共享可变评测状态。"""
        assignment = ctx.evaluation_plan.assignment_for(case.identity)
        if assignment.probe_kind is None:
            raise EvaluationProtocolError("V-04 消融 assignment 缺 probe kind")
        intervention = self.interventions.get(case.intervention_key)

        def request(enabled: bool) -> PreWeaningProbeRequest:
            """构造不含 expected 或 teacher 的单臂请求。"""
            return PreWeaningProbeRequest(
                checkpoint=checkpoint,
                identity=case.identity,
                split=assignment.split,
                probe_kind=assignment.probe_kind,
                dimension=case.dimension,
                case_key=case.case_key,
                intervention_key=case.intervention_key,
                intervention_enabled=enabled,
            )

        enabled_result = self._evaluate(
            ctx,
            assignment,
            case.dimension,
            case.evaluator_key,
            request(True),
            intervention=intervention,
            enabled=True,
        )
        disabled_result = self._evaluate(
            ctx,
            assignment,
            case.dimension,
            case.evaluator_key,
            request(False),
            intervention=intervention,
            enabled=False,
        )
        return AblationPairResult(case, enabled_result, disabled_result)

    def _run_held_out(
            self,
            ctx: Any,
            checkpoint: int,
            ) -> tuple[PreWeaningDimensionResult, ...]:
        """逐项隔离执行 held-out，并按维度保留 PASS、FAIL、NE 原值。"""
        observations = []
        plan = ctx.evaluation_plan
        for assignment in plan.assignments:
            if assignment.split != plan.protocol.held_out_split:
                continue
            if assignment.probe_kind is None:
                raise EvaluationProtocolError("V-04 held-out 缺 probe kind")
            evaluator_key = self.protocol.evaluator_key_for(
                assignment.probe_kind)
            for dimension in assignment.dimensions:
                request = PreWeaningProbeRequest(
                    checkpoint=checkpoint,
                    identity=assignment.identity,
                    split=assignment.split,
                    probe_kind=assignment.probe_kind,
                    dimension=dimension,
                )
                observations.append(self._evaluate(
                    ctx,
                    assignment,
                    dimension,
                    evaluator_key,
                    request,
                ))
        results = []
        for dimension in self.protocol.stopping_dimensions:
            selected = tuple(
                observation for observation in observations
                if observation.dimension == dimension)
            results.append(PreWeaningDimensionResult(
                dimension=dimension,
                planned=len(selected),
                passed=sum(
                    observation.outcome.passed is True
                    for observation in selected),
                failed=sum(
                    observation.outcome.passed is False
                    for observation in selected),
                not_evaluated=sum(
                    observation.outcome.passed is None
                    for observation in selected),
                sample_count=sum(
                    observation.outcome.sample_count
                    for observation in selected),
            ))
        return tuple(results)

    def _window_passes(
            self,
            window: PreWeaningHeldOutWindow,
            *,
            plan_sha256: str,
            evaluator_state: CanonicalIdentity,
            intervention_state: CanonicalIdentity,
            state_reader_state: CanonicalIdentity,
            ) -> bool:
        """逐维、逐资源核验窗口，不计算综合均值或跨指标补偿。"""
        if (
                window.plan_sha256 != plan_sha256
                or window.protocol_sha256 != self.protocol.identity().sha256
                or window.evaluator_state != evaluator_state
                or window.intervention_state != intervention_state
                or window.state_reader_state != state_reader_state):
            return False
        dimensions = {item.dimension: item for item in window.dimensions}
        if set(dimensions) != set(self.protocol.stopping_dimensions):
            return False
        wall_dimensions = set(self.protocol.wall_dimensions)
        for dimension, result in dimensions.items():
            if result.planned <= 0:
                return False
            if dimension in wall_dimensions:
                if result.passed != 0:
                    return False
            elif not (
                    result.passed == result.planned
                    and result.failed == 0
                    and result.not_evaluated == 0
                    and result.sample_count > 0):
                return False
        resources = {item.metric_key: item for item in window.resources}
        bounds = {item.metric_key: item for item in self.protocol.resource_bounds}
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
            request: PreWeaningValidationRequest,
            ) -> PreWeaningValidationReport:
        """测量单一 checkpoint，并仅在连续历史、消融和资源全过时建议停止。"""
        if not isinstance(request, PreWeaningValidationRequest):
            raise TypeError("V-04 validation request 类型错误")
        if ctx.evaluation_plan is None or not ctx.evaluation_strictly_isolated:
            raise EvaluationProtocolError("V-04 要求严格 V-00 evaluation plan")
        plan = ctx.evaluation_plan
        self.protocol.validate_plan(plan)
        if len(request.previous_windows) > self.protocol.consecutive_windows - 1:
            raise EvaluationProtocolError("V-04 只接受停止所需的历史尾窗")
        plan_sha256 = plan.sha256()
        evaluator_state = CanonicalIdentity.from_value(
            self.evaluators.state_key())
        intervention_state = CanonicalIdentity.from_value(
            self.interventions.state_key())
        state_reader_state = CanonicalIdentity.from_value(
            _strict_state_key(self.state_reader, "V-04 state reader"))
        host_before = CanonicalIdentity.from_value(self.state_reader.read(ctx))
        ablations = tuple(
            self._run_ablation_case(ctx, case, request.checkpoint)
            for case in self.protocol.ablation_cases
        )
        dimensions = self._run_held_out(ctx, request.checkpoint)
        host_after = CanonicalIdentity.from_value(self.state_reader.read(ctx))
        if host_after != host_before:
            raise EvaluationProtocolError("V-04 运行改变了宿主只读状态")
        current = PreWeaningHeldOutWindow(
            checkpoint=request.checkpoint,
            plan_sha256=plan_sha256,
            protocol_sha256=self.protocol.identity().sha256,
            evaluator_state=evaluator_state,
            intervention_state=intervention_state,
            state_reader_state=state_reader_state,
            host_state=host_before,
            dimensions=dimensions,
            resources=request.resources,
        )
        windows = (*request.previous_windows, current)
        for index in range(1, len(windows)):
            if (windows[index].checkpoint - windows[index - 1].checkpoint
                    != self.protocol.checkpoint_step):
                raise EvaluationProtocolError(
                    "V-04 checkpoint 历史不连续或发生重复")
            if windows[index].host_state == windows[index - 1].host_state:
                raise EvaluationProtocolError(
                    "V-04 不得给同一宿主状态重复编号冒充连续窗口")
        windows_pass = (
            len(windows) == self.protocol.consecutive_windows
            and all(self._window_passes(
                window,
                plan_sha256=plan_sha256,
                evaluator_state=evaluator_state,
                intervention_state=intervention_state,
                state_reader_state=state_reader_state,
            ) for window in windows)
        )
        ablations_complete = bool(ablations) and all(
            pair.complete for pair in ablations)
        return PreWeaningValidationReport(
            protocol_version=self.protocol.version,
            plan_sha256=plan_sha256,
            ablations=ablations,
            windows=windows,
            stop_allowed=ablations_complete and windows_pass,
        )


__all__ = [
    "PreWeaningEvaluatorBinding",
    "PreWeaningEvaluatorRegistry",
    "PreWeaningInterventionBinding",
    "PreWeaningInterventionRegistry",
    "PreWeaningProbeRequest",
    "PreWeaningValidationRuntime",
]
