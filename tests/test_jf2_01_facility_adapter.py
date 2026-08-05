"""JF2-01 生产 Facility adapter 的真实调用、合取和零写专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.evaluation_protocol import CanonicalIdentity
from pure_integer_ai.experiments.facility_readiness_adapter import (
    FACILITY_CHECK_KEYS,
    FACILITY_FORBIDDEN_KEYS,
    FACILITY_METRIC_KEYS,
    ProductionFacilityExercise,
    build_facility_readiness_context,
    run_production_facility_readiness,
)
from pure_integer_ai.experiments.facility_readiness_runtime import (
    FacilityReadinessRuntime,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_OPT_IN,
    STATUS_PRODUCTION,
)


@pytest.fixture(scope="module")
def production_run(tmp_path_factory):
    """只执行一次完整生产 F-01，并记录统一 runtime 的真实调用。"""
    calls = []
    original = FacilityReadinessRuntime.run

    def observed(runtime, ctx):
        """记录 runtime 身份后委托真实实现。"""
        calls.append(CanonicalIdentity.from_value(runtime.state_key()))
        return original(runtime, ctx)

    patch = pytest.MonkeyPatch()
    patch.setattr(FacilityReadinessRuntime, "run", observed)
    try:
        result = run_production_facility_readiness(
            run_dir=Path(tmp_path_factory.mktemp("jf2-01")) / "migration")
    finally:
        patch.undo()
    return result, tuple(calls)


def test_jf2_01_calls_facility_runtime_once_and_keeps_all_state_read_only(
        production_run,
        ):
    """生产 caller 必须真调 runtime 一次，并保持 Core、host 和受禁读取为零。"""
    result, calls = production_run
    report = result.report
    assert calls == (result.runtime_identity,)
    assert report.facility_complete is True
    assert report.exercise.core_unchanged is True
    assert result.host_before == result.host_after
    assert (
        result.teacher_reads,
        result.expected_reads,
        result.evaluator_label_reads,
    ) == (0, 0, 0)
    assert not hasattr(report, "mastered")
    assert not hasattr(report, "readiness")


def test_jf2_01_report_is_exact_conjunction_of_real_measurements(
        production_run,
        ):
    """逐维与最终完成态必须由实际计数、检查、机制和禁用信号合取。"""
    result, _ = production_run
    report = result.report
    measurement = report.exercise.measurement
    observed_counters = {
        item.metric_key: item for item in measurement.counters}
    observed_checks = {
        item.check_key: item for item in measurement.checks}

    assert set(FACILITY_METRIC_KEYS).issubset(observed_counters)
    assert set(FACILITY_FORBIDDEN_KEYS).issubset(observed_counters)
    assert set(FACILITY_CHECK_KEYS) == set(observed_checks)
    assert measurement.positive_behavior == 100
    assert measurement.negative_behavior == 0
    improvement = (
        measurement.positive_behavior - measurement.negative_behavior)

    for dimension in report.dimensions:
        requirement = dimension.requirement
        counters_complete = all(
            observed_counters[item.metric_key].value >= item.minimum_value
            and observed_counters[item.metric_key].sample_count > 0
            for item in requirement.counters
        )
        checks_complete = all(
            observed_checks[key].passed
            and observed_checks[key].before == observed_checks[key].after
            for key in requirement.checks
        )
        expected = (
            report.exercise.core_unchanged
            and improvement >= requirement.minimum_behavior_improvement
            and counters_complete
            and checks_complete
        )
        assert dimension.passed is expected

    mechanism_complete = all(
        item.status in {STATUS_PRODUCTION, STATUS_OPT_IN}
        and bool(item.owner)
        and item.writer_count > 0
        and item.reader_count > 0
        and item.passed
        for item in report.mechanisms
    )
    forbidden_complete = all(
        item.value == 0 for item in report.forbidden_counters)
    expected_complete = (
        report.exercise.core_unchanged
        and all(item.passed for item in report.dimensions)
        and mechanism_complete
        and forbidden_complete
    )
    assert report.facility_complete is expected_complete


def test_jf2_01_real_counter_and_integrity_inventory_is_complete(
        production_run,
        ):
    """十二项机制测量和十项恢复/隔离检查必须来自非空真实证据。"""
    result, _ = production_run
    measurement = result.report.exercise.measurement
    counters = {item.metric_key: item for item in measurement.counters}
    checks = {item.check_key: item for item in measurement.checks}
    assert tuple(counters[key].value for key in FACILITY_METRIC_KEYS) == (
        2,
        4,
        1,
        2,
        1,
        4,
        1,
        1,
        1,
        1,
        4,
        2,
    )
    assert all(counters[key].sample_count > 0 for key in FACILITY_METRIC_KEYS)
    assert tuple(counters[key].value for key in FACILITY_FORBIDDEN_KEYS) == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert all(
        checks[key].passed and checks[key].before == checks[key].after
        for key in FACILITY_CHECK_KEYS
    )


def test_jf2_01_adapter_state_key_does_not_include_temp_path(tmp_path):
    """不同迁移临时目录和宿主对象不得改变 adapter 配置身份。"""
    first_ctx = build_facility_readiness_context()
    second_ctx = build_facility_readiness_context()
    try:
        first = ProductionFacilityExercise(tmp_path / "a", first_ctx)
        second = ProductionFacilityExercise(tmp_path / "b", second_ctx)
        before = first.state_key()
        assert before == second.state_key()
        assert before == (2, 5290, 5, 12, 10, 7)
        assert first.state_key() == before
    finally:
        first_ctx.backend.close()
        second_ctx.backend.close()
