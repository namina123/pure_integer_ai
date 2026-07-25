"""F-01 在隔离 Core fixture 上执行真实总装并生成设施报告。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationPlan,
    ProtocolKey,
)
from pure_integer_ai.experiments.facility_readiness import (
    FacilityDimensionResult,
    FacilityExerciseMeasurement,
    FacilityExerciseResult,
    FacilityMechanismCheck,
    FacilityReadinessProtocol,
    FacilityReadinessReport,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_OPT_IN,
    STATUS_PRODUCTION,
    inventory_by_id,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
)


class FacilityReadinessError(RuntimeError):
    """F-01 协议、adapter 或证据状态不一致。"""


def _state_key(value: Any) -> tuple[int, ...]:
    """读取 adapter 稳定状态并拒绝布尔值或空键。"""
    reader = getattr(value, "state_key", None)
    if not callable(reader):
        raise TypeError("F-01 exercise 缺少 state_key")
    state = reader()
    if (not isinstance(state, tuple) or not state
            or any(type(item) is not int for item in state)):
        raise TypeError("F-01 exercise state_key 必须是非空严格整数 tuple")
    return state


@dataclass(frozen=True)
class FacilityExerciseBinding:
    """把预注册 exercise 键绑定到可准备、运行和自描述的 adapter。"""

    exercise_key: ProtocolKey
    exercise: Any

    def __post_init__(self) -> None:
        """要求 adapter 提供 prepare/run/state_key 三个明确边界。"""
        if not isinstance(self.exercise_key, ProtocolKey):
            raise TypeError("F-01 binding exercise_key 类型错误")
        for name in ("prepare", "run", "state_key"):
            if not callable(getattr(self.exercise, name, None)):
                raise TypeError(f"F-01 exercise 缺少 {name}")
        _state_key(self.exercise)


class FacilityReadinessRuntime:
    """在 V-06 clone 中执行一次完整 F-01 exercise 并独立裁决报告。"""

    def __init__(
            self,
            protocol: FacilityReadinessProtocol,
            binding: FacilityExerciseBinding,
            ) -> None:
        """绑定冻结协议和唯一总装 adapter，不接受隐式 fallback。"""
        if not isinstance(protocol, FacilityReadinessProtocol):
            raise TypeError("F-01 runtime protocol 类型错误")
        if not isinstance(binding, FacilityExerciseBinding):
            raise TypeError("F-01 runtime binding 类型错误")
        if binding.exercise_key != protocol.exercise_key:
            raise ValueError("F-01 binding exercise_key 与协议漂移")
        self.protocol = protocol
        self.binding = binding

    def state_key(self) -> tuple[int, ...]:
        """返回协议身份索引和 adapter 配置状态。"""
        return (
            1,
            self.protocol.version,
            self.protocol.identity().index,
            *_state_key(self.binding.exercise),
        )

    def _mechanism_checks(self) -> tuple[FacilityMechanismCheck, ...]:
        """从真实台账核验承重机制不是 dead/test-only 且有 owner/read/write。"""
        inventory = inventory_by_id()
        result = []
        for mechanism_id in self.protocol.required_mechanism_ids:
            record = inventory.get(mechanism_id)
            if record is None:
                result.append(FacilityMechanismCheck(
                    mechanism_id, "missing", "", 0, 0, False))
                continue
            passed = (
                record.status in {STATUS_PRODUCTION, STATUS_OPT_IN}
                and bool(record.owner)
                and bool(record.writers)
                and bool(record.readers)
            )
            result.append(FacilityMechanismCheck(
                record.mechanism_id,
                record.status,
                record.owner,
                len(record.writers),
                len(record.readers),
                passed,
            ))
        return tuple(result)

    @staticmethod
    def _scrub_evaluation_inputs(eval_ctx: Any) -> None:
        """清除 clone 中 teacher 和 expected，阻断 F-01 从评测答案直出。"""
        eval_ctx.teacher = None
        plan = eval_ctx.evaluation_plan
        if plan is not None:
            eval_ctx.evaluation_plan = EvaluationPlan(
                plan.protocol,
                tuple(
                    replace(item, expected_outcome=None)
                    for item in plan.assignments
                ),
            )

    def _exercise(self, ctx: Any) -> FacilityExerciseResult:
        """在独立 clone 先装 fixture、再冻结 Core 并执行真实总装。"""
        adapter = self.binding.exercise
        initial_state = _state_key(adapter)
        measured: list[FacilityExerciseResult] = []
        with isolated_evaluation(
                ctx,
                label=f"f01-{self.protocol.version}",
                ) as eval_ctx:
            self._scrub_evaluation_inputs(eval_ctx)
            adapter.prepare(eval_ctx)
            if _state_key(adapter) != initial_state:
                raise FacilityReadinessError("F-01 prepare 改变了 adapter 配置状态")
            core_reader = CoreCanonicalStateReader(eval_ctx)
            core_before = CanonicalIdentity.from_value(core_reader.read())
            measurement = adapter.run(eval_ctx)
            if not isinstance(measurement, FacilityExerciseMeasurement):
                raise TypeError("F-01 exercise 必须返回 FacilityExerciseMeasurement")
            if measurement.exercise_key != self.protocol.exercise_key:
                raise FacilityReadinessError("F-01 measurement exercise_key 漂移")
            core_after = CanonicalIdentity.from_value(core_reader.read())
            if _state_key(adapter) != initial_state:
                raise FacilityReadinessError("F-01 run 改变了 adapter 配置状态")
            if eval_ctx.teacher is not None:
                raise FacilityReadinessError("F-01 exercise 恢复了 teacher")
            if (eval_ctx.evaluation_plan is not None
                    and any(item.expected_outcome is not None
                            for item in eval_ctx.evaluation_plan.assignments)):
                raise FacilityReadinessError("F-01 exercise 恢复了 expected")
            measured.append(FacilityExerciseResult(
                measurement,
                core_before,
                core_after,
                CanonicalIdentity.from_value(initial_state),
            ))
        if len(measured) != 1:
            raise FacilityReadinessError("F-01 exercise 执行次数不为一")
        return measured[0]

    def run(self, ctx: Any) -> FacilityReadinessReport:
        """执行总装并逐维合取证据；任一承重失败都保持报告未完成。"""
        exercise = self._exercise(ctx)
        measurement = exercise.measurement
        forbidden = tuple(
            measurement.counter(key)
            for key in self.protocol.forbidden_counter_keys
        )
        if any(item is None for item in forbidden):
            raise FacilityReadinessError("F-01 measurement 缺少禁用信号计数")
        forbidden_counters = tuple(item for item in forbidden if item is not None)
        dimensions = []
        behavior_improvement = (
            measurement.positive_behavior - measurement.negative_behavior)
        for requirement in self.protocol.dimensions:
            counters = tuple(
                measurement.counter(item.metric_key)
                for item in requirement.counters
            )
            checks = tuple(
                measurement.check(item)
                for item in requirement.checks
            )
            if any(item is None for item in counters):
                raise FacilityReadinessError(
                    "F-01 measurement 缺少维度必需计数")
            if any(item is None for item in checks):
                raise FacilityReadinessError(
                    "F-01 measurement 缺少维度完整性检查")
            counters_complete = all(
                observed is not None
                and observed.value >= expected.minimum_value
                and observed.sample_count > 0
                for expected, observed in zip(requirement.counters, counters)
            )
            checks_complete = all(
                observed is not None
                and observed.passed
                and observed.before == observed.after
                for observed in checks
            )
            passed = (
                exercise.core_unchanged
                and behavior_improvement
                >= requirement.minimum_behavior_improvement
                and counters_complete
                and checks_complete
            )
            dimensions.append(FacilityDimensionResult(
                requirement,
                tuple(item for item in counters if item is not None),
                tuple(item for item in checks if item is not None),
                behavior_improvement,
                passed,
            ))
        mechanisms = self._mechanism_checks()
        facility_complete = (
            exercise.core_unchanged
            and all(item.passed for item in mechanisms)
            and all(item.passed for item in dimensions)
            and all(item.value == 0 for item in forbidden_counters)
        )
        return FacilityReadinessReport(
            self.protocol.identity(),
            exercise,
            mechanisms,
            tuple(dimensions),
            forbidden_counters,
            self.protocol.boundary_keys,
            facility_complete,
        )


__all__ = [
    "FacilityExerciseBinding",
    "FacilityReadinessError",
    "FacilityReadinessRuntime",
]
