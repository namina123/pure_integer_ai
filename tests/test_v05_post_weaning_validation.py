"""V-05 断奶后 Memory 消融、独立轨道和停止判据测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationProtocolError,
    ProbeOutcome,
    ProtocolKey,
)
from pure_integer_ai.experiments.post_weaning_validation import (
    PostWeaningAblationCase,
    PostWeaningHistoryTrial,
    PostWeaningProbeMeasurement,
    PostWeaningResourceBound,
    PostWeaningResourceMeasurement,
    PostWeaningTrackRequirement,
    PostWeaningValidationProtocol,
    PostWeaningValidationRequest,
)
from pure_integer_ai.experiments.post_weaning_validation_runtime import (
    PostWeaningEvaluatorBinding,
    PostWeaningEvaluatorRegistry,
    PostWeaningInterventionBinding,
    PostWeaningInterventionRegistry,
    PostWeaningValidationRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend

from tests.test_v00_evaluation_protocol import _complete_plan


def _key(value: int) -> ProtocolKey:
    """构造测试使用的开放协议键。"""
    return ProtocolKey((value,))


class _StateReader:
    """读取宿主持久态和调用方推进的独立 checkpoint epoch。"""

    def __init__(self, epoch: int = 10):
        """设置首个可观察 checkpoint epoch。"""
        self.epoch = epoch

    def state_key(self) -> tuple[int, ...]:
        """返回 reader 协议身份，epoch 不属于阈值配置。"""
        return 1, 1400

    def read(self, ctx):
        """返回足以拒绝同一宿主状态换号的长期状态。"""
        return self.epoch, ctx.backend.snapshot(), ctx.work_memory.round_id


class _Intervention:
    """在 V-06 clone 内打开或关闭一个注入式 Memory 机制。"""

    def __init__(self, mode: int):
        """绑定不写入生产代码的测试机制编号。"""
        self.mode = mode

    def state_key(self) -> tuple[int, ...]:
        """返回机制破坏实现身份。"""
        return 1, self.mode

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """只在 clone 写入臂状态；一类破坏还故意只改 pr_vector。"""
        eval_ctx.v05_memory_enabled = enabled
        if not enabled and self.mode == 9:
            eval_ctx.work_memory.pr_vector[(1, self.mode)] = 999


class _Evaluator:
    """生成分账测量，并证明 request/context 均看不到 expected 或 teacher。"""

    def __init__(self, *, omit_use: bool = False,
                 rollback_drift: bool = False,
                 history_leak: bool = False,
                 query_drift: bool = False,
                 resource_leak: bool = False):
        """注入五类反 theater 故障，默认返回完整有效测量。"""
        self.omit_use = omit_use
        self.rollback_drift = rollback_drift
        self.history_leak = history_leak
        self.query_drift = query_drift
        self.resource_leak = resource_leak

    def state_key(self) -> tuple[int, ...]:
        """返回全部测量阈值和故障模式。"""
        return (
            1,
            int(self.omit_use),
            int(self.rollback_drift),
            int(self.history_leak),
            int(self.query_drift),
            int(self.resource_leak),
        )

    def evaluate(self, eval_ctx, item, request) -> PostWeaningProbeMeasurement:
        """按 clone 中真实臂状态形成行为和完整性分账，不使用行数判通过。"""
        assert eval_ctx.teacher is None
        assert all(
            assignment.expected_outcome is None
            for assignment in eval_ctx.evaluation_plan.assignments
        )
        assert not hasattr(request, "expected_outcome")
        enabled = bool(eval_ctx.v05_memory_enabled)
        assert enabled is request.memory_enabled
        query_key = (
            *item.source_ref.stable_key(),
            *request.dimension.stable_key(),
        )
        if self.query_drift:
            query_key = (*query_key, 999)
        behavior = 100 if enabled else 0
        use_count = 0 if self.omit_use else int(enabled)
        rollback_before = CanonicalIdentity.from_value((
            "rollback", request.case_key.stable_key()))
        rollback_after = (
            CanonicalIdentity.from_value(("rollback-drift", 1))
            if self.rollback_drift and enabled else rollback_before
        )
        recovery = CanonicalIdentity.from_value((
            "recovery", request.case_key.stable_key()))
        second_behavior = behavior + int(self.history_leak and enabled)
        trials = (
            PostWeaningHistoryTrial(
                _key(1410), behavior, use_count, query_key),
            PostWeaningHistoryTrial(
                _key(1411), second_behavior, use_count, query_key),
        )
        work = eval_ctx.work_memory
        closed = (
            work.active_query_scope is None
            and work.active_generation_scope is None
            and work.attractor_state is None
        )
        return PostWeaningProbeMeasurement(
            ProbeOutcome(enabled, value=behavior, sample_count=1),
            memory_use_count=use_count,
            attractor_consumption_count=int(enabled),
            current_query_binding_checks=int(enabled),
            source_trust_checks=int(enabled),
            conflict_checks=int(enabled),
            current_query_key=query_key,
            work_memory_closed=closed,
            query_resources_closed=not self.resource_leak,
            rollback_before=rollback_before,
            rollback_after=rollback_after,
            recovery_before=recovery,
            recovery_after=recovery,
            history_trials=trials,
        )


def _setup(*, evaluator: _Evaluator | None = None):
    """构造九类破坏、六维度和三条独立停止轨道。"""
    plan, items = _complete_plan()
    dimensions = plan.protocol.required_dimensions
    tracks = (_key(1420), _key(1421), _key(1422))
    evaluator_key = _key(1423)
    intervention_keys = tuple(
        _key(1430 + index)
        for index in range(len(plan.protocol.required_adversarial_kinds))
    )
    cases = []
    assignments = plan.assignments[3:3 + len(
        plan.protocol.required_adversarial_kinds)]
    for index, assignment in enumerate(assignments):
        track_index = min(index // 3, 2)
        dimension_pair = dimensions[track_index * 2:track_index * 2 + 2]
        dimension = dimension_pair[index % 2]
        expected_query_key = (
            *assignment.identity.source_ref.stable_key(),
            *dimension.stable_key(),
        )
        cases.append(PostWeaningAblationCase(
            _key(1440 + index),
            tracks[track_index],
            intervention_keys[index],
            evaluator_key,
            assignment.identity,
            dimension,
            expected_query_key,
            50,
        ))
    track_requirements = tuple(
        PostWeaningTrackRequirement(
            track,
            dimensions[index * 2:index * 2 + 2],
            (PostWeaningResourceBound(
                _key(1450 + index), maximum_value=64),),
            consecutive_windows=2,
            checkpoint_step=5,
        )
        for index, track in enumerate(tracks)
    )
    protocol = PostWeaningValidationProtocol(
        version=2,
        cases=tuple(cases),
        required_case_keys=tuple(item.case_key for item in cases),
        tracks=track_requirements,
        required_track_keys=tracks,
    )
    reader = _StateReader()
    runtime = PostWeaningValidationRuntime(
        protocol,
        PostWeaningEvaluatorRegistry((PostWeaningEvaluatorBinding(
            evaluator_key,
            evaluator or _Evaluator(),
        ),)),
        PostWeaningInterventionRegistry(tuple(
            PostWeaningInterventionBinding(
                intervention_key,
                _Intervention(index + 1),
            )
            for index, intervention_key in enumerate(intervention_keys)
        )),
        reader,
    )
    backend = DictBackend()
    ctx = make_train_context(backend)
    partition = plan.partition(items)
    ctx.evaluation_plan = plan
    ctx.evaluation_corpora = partition.as_dict()
    ctx.evaluation_strictly_isolated = True
    return plan, ctx, runtime, reader, tracks


def _resources(track_index: int, *, value: int = 32):
    """构造一条轨道的有来源资源测量。"""
    return (PostWeaningResourceMeasurement(
        _key(1450 + track_index),
        _key(1460 + track_index),
        value,
        2,
    ),)


def test_v05_three_tracks_stop_independently_after_distinct_windows():
    """阅读、交互、开放摄入分别积累窗口，不能相互替代停止证据。"""
    _plan, ctx, runtime, reader, tracks = _setup()
    try:
        first_reports = []
        for index, track in enumerate(tracks):
            report = runtime.run(ctx, PostWeaningValidationRequest(
                track,
                10,
                _resources(index),
            ))
            assert report.ablations_complete is True
            assert report.stop_allowed is False
            assert report.track_key == track
            assert not hasattr(report, "mastered")
            assert not hasattr(report, "readiness")
            first_reports.append(report)

        reader.epoch = 15
        for index, (track, first) in enumerate(zip(
                tracks, first_reports, strict=True)):
            report = runtime.run(ctx, PostWeaningValidationRequest(
                track,
                15,
                _resources(index),
                previous_windows=(first.windows[-1],),
            ))
            assert report.stop_allowed is True
            assert all(item.complete for item in report.ablations)
    finally:
        ctx.backend.close()


@pytest.mark.parametrize("evaluator", (
    _Evaluator(omit_use=True),
    _Evaluator(rollback_drift=True),
    _Evaluator(history_leak=True),
    _Evaluator(query_drift=True),
    _Evaluator(resource_leak=True),
))
def test_v05_rows_without_use_or_integrity_evidence_never_pass(evaluator):
    """缺 Use、状态完整性、当前 query 或 reader 关闭证据都阻断判停。"""
    _plan, ctx, runtime, reader, tracks = _setup(evaluator=evaluator)
    try:
        first = runtime.run(ctx, PostWeaningValidationRequest(
            tracks[0], 10, _resources(0)))
        reader.epoch = 15
        second = runtime.run(ctx, PostWeaningValidationRequest(
            tracks[0],
            15,
            _resources(0),
            previous_windows=(first.windows[-1],),
        ))
        assert second.ablations_complete is False
        assert second.stop_allowed is False
    finally:
        ctx.backend.close()


def test_v05_resource_overrun_cross_track_history_and_same_state_fail():
    """资源越界不能补偿，历史不能跨轨道或给同一宿主状态换号。"""
    _plan, ctx, runtime, reader, tracks = _setup()
    try:
        first = runtime.run(ctx, PostWeaningValidationRequest(
            tracks[0], 10, _resources(0, value=65)))
        with pytest.raises(EvaluationProtocolError, match="同一宿主状态"):
            runtime.run(ctx, PostWeaningValidationRequest(
                tracks[0],
                15,
                _resources(0),
                previous_windows=(first.windows[-1],),
            ))
        reader.epoch = 15
        second = runtime.run(ctx, PostWeaningValidationRequest(
            tracks[0],
            15,
            _resources(0),
            previous_windows=(first.windows[-1],),
        ))
        assert second.stop_allowed is False

        crossed = replace(first.windows[-1], track_key=tracks[1])
        reader.epoch = 20
        with pytest.raises(EvaluationProtocolError, match="其他停止轨道"):
            runtime.run(ctx, PostWeaningValidationRequest(
                tracks[0],
                15,
                _resources(0),
                previous_windows=(crossed,),
            ))
    finally:
        ctx.backend.close()


def test_v05_identical_inputs_produce_identical_track_reports():
    """同协议、同轨道和同 checkpoint 两次报告必须 bit-identical。"""
    reports = []
    for _index in range(2):
        _plan, ctx, runtime, _reader, tracks = _setup()
        try:
            reports.append(runtime.run(ctx, PostWeaningValidationRequest(
                tracks[2],
                10,
                _resources(2),
            )))
        finally:
            ctx.backend.close()
    assert reports[0] == reports[1]
