"""V-04 断奶前反向破坏、连续窗口和资源停止协议测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.evaluation_protocol import (
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationProtocolError,
    ProbeOutcome,
    ProtocolKey,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.pre_weaning_validation import (
    PreWeaningAblationCase,
    PreWeaningProbeRoute,
    PreWeaningValidationProtocol,
    PreWeaningValidationRequest,
    ResourceBound,
    ResourceMeasurement,
)
from pure_integer_ai.experiments.pre_weaning_validation_runtime import (
    PreWeaningEvaluatorBinding,
    PreWeaningEvaluatorRegistry,
    PreWeaningInterventionBinding,
    PreWeaningInterventionRegistry,
    PreWeaningValidationRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.cognition.shared.hypothesis import EPISTEMIC_SUPPORTED
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_CONSUMER_ACCEPTED,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.stages import STAGE1_SKELETON

from tests.test_r06_precedence_relation_runtime import (
    _Course as _PrecedenceCourse,
    _domain as _precedence_domain,
    _install as _install_precedence,
    _item as _precedence_item,
    _source as _precedence_source,
)
from tests.test_v00_evaluation_protocol import (
    _assignment,
    _complete_plan,
)


def _key(value: int) -> ProtocolKey:
    """构造测试使用的开放单分量协议键。"""
    return ProtocolKey((value,))


class _Evaluator:
    """用 clone 内干预标记模拟健康、破坏及墙维度三态。"""

    def __init__(self, wall_dimension: ProtocolKey, *, broken_off: bool = True):
        """冻结墙维度及破坏臂是否真实失败的测试规格。"""
        self.wall_dimension = wall_dimension
        self.broken_off = broken_off

    def state_key(self) -> tuple[int, ...]:
        """返回 evaluator 阈值和墙维度的稳定状态。"""
        return (
            1,
            int(self.broken_off),
            *self.wall_dimension.stable_key(),
        )

    def evaluate(self, eval_ctx, item, request) -> ProbeOutcome:
        """证明请求不含答案/teacher，并按 clone 内状态生成分维结果。"""
        assert eval_ctx.teacher is None
        assert all(
            assignment.expected_outcome is None
            for assignment in eval_ctx.evaluation_plan.assignments
        )
        assert item.source_ref == request.identity.source_ref
        assert not hasattr(request, "expected_outcome")
        assert not hasattr(request, "teacher")
        if request.intervention_enabled is not None:
            enabled = bool(getattr(eval_ctx, "v04_intervention_enabled"))
            passed = enabled or not self.broken_off
            return ProbeOutcome(passed, value=int(passed), sample_count=1)
        if request.dimension == self.wall_dimension:
            return ProbeOutcome(None, value=0, sample_count=0)
        return ProbeOutcome(True, value=1000, sample_count=1)


class _Intervention:
    """只在当前 V-06 clone 上写入健康或破坏状态。"""

    def __init__(self, key: ProtocolKey):
        """绑定一个不会在运行中变化的干预身份。"""
        self.key = key

    def state_key(self) -> tuple[int, ...]:
        """返回本破坏实现的稳定身份。"""
        return (1, *self.key.stable_key())

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """把臂状态写入 clone，绝不触碰宿主或模块全局 gate。"""
        eval_ctx.v04_intervention_enabled = enabled


class _StateReader:
    """读取宿主 backend 和调用方推进的真实 checkpoint epoch。"""

    def __init__(self, epoch: int = 0):
        """设置首个可观察训练 epoch。"""
        self.epoch = epoch

    def state_key(self) -> tuple[int, ...]:
        """返回 reader 协议身份，运行 epoch 不属于阈值配置。"""
        return (1, 904)

    def read(self, ctx):
        """返回足以识别同一宿主状态重标的完整只读快照。"""
        return self.epoch, ctx.backend.snapshot(), ctx.work_memory.round_id


class _InactiveLifecycle:
    """让 runtime 通过真实 projection 接口看到 inactive 约束。"""

    def __init__(self, lifecycle):
        """保存被破坏前的真实生命周期 facade。"""
        self._lifecycle = lifecycle
        self.protocol = lifecycle.protocol

    def project(self, constraint):
        """保留真实 constraint/history，但把当前投影破坏为 inactive。"""
        projection = self._lifecycle.project(constraint)
        return replace(projection, state=self.protocol.inactive_state)


class _PrecedenceLayerIntervention:
    """在 clone 内关闭 R-06 纵切的一个真实层级。"""

    def __init__(self, mode: int):
        """绑定测试 manifest 注入的层级编号。"""
        self.mode = mode

    def state_key(self) -> tuple[int, ...]:
        """返回破坏层级的稳定测试协议身份。"""
        return 1, self.mode

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """健康臂保持原设施，破坏臂只修改当前 V-06 clone。"""
        if enabled:
            return
        runtime = eval_ctx.precedence_relation_runtime
        if self.mode == 1:
            def disabled_map(_step):
                """模拟 structure mapper 未接线，不产 typed projection。"""
                return ()

            runtime.course.map_step = disabled_map
        elif self.mode == 2:
            eval_ctx.occurrence_order_writer = None
        elif self.mode == 3:
            runtime.engine = OrderHypothesisEngine(runtime.protocol.learning)
        elif self.mode == 4:
            runtime.lifecycle = _InactiveLifecycle(runtime.lifecycle)
        elif self.mode == 5:
            runtime.consumer = None
        else:
            raise AssertionError("测试层级未注册")


class _PrecedenceLayerEvaluator:
    """用真实 R-06 处理报告核验结构、事实、累计、晋升和消费。"""

    def __init__(self, dimension: ProtocolKey):
        """绑定本纵切唯一评测维度。"""
        self.dimension = dimension

    def state_key(self) -> tuple[int, ...]:
        """返回报告判据和维度的稳定身份。"""
        return 1, *self.dimension.stable_key()

    def evaluate(self, eval_ctx, item, request) -> ProbeOutcome:
        """观察新来源并要求累计状态及下游 parse/linearize 全部真实可见。"""
        assert request.dimension == self.dimension
        runtime = eval_ctx.precedence_relation_runtime
        try:
            eval_ctx.work_memory.end_session()
            from pure_integer_ai.experiments.round_runtime import (
                DefaultRoundRunner,
            )
            DefaultRoundRunner().run_round(
                eval_ctx,
                item,
                STAGE1_SKELETON,
                request.checkpoint + 1,
            )
            hypothesis = runtime.engine.hypothesis_for(
                runtime.course.pattern)
            cumulative = runtime.engine.ledger.snapshot(hypothesis)
            report = eval_ctx.precedence_relation_reports[-1]
            if len(report.observations) != 1:
                return ProbeOutcome(False, value=0, sample_count=1)
            observation = report.observations[0]
            passed = (
                cumulative.epistemic_status == EPISTEMIC_SUPPORTED
                and observation.parse is not None
                and observation.linearization is not None
                and observation.parse.status == ORDER_CONSUMER_ACCEPTED
                and observation.linearization.status == ORDER_CONSUMER_ACCEPTED
            )
            return ProbeOutcome(passed, value=int(passed), sample_count=1)
        except (
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError):
            return ProbeOutcome(False, value=0, sample_count=1)


def _precedence_validation_plan():
    """构造词面不同、来源独立且覆盖五种层级破坏的 V-00 计划。"""
    dimension = _key(1200)
    kinds = tuple(_key(1210 + index) for index in range(5))
    protocol = EvaluationProtocol(
        version=9,
        training_split=_key(1220),
        development_split=_key(1221),
        held_out_split=_key(1222),
        adversarial_split=_key(1223),
        external_split=_key(1224),
        statistical_evidence=_key(1225),
        external_evidence=_key(1226),
        required_dimensions=(dimension,),
        required_adversarial_kinds=kinds,
    )
    item_specs = (
        (("甲", "乙"), _precedence_source(1)),
        (("丙", "丁"), _precedence_source(2)),
        (("戊", "己"), _precedence_source(3)),
        (("庚", "辛"), _precedence_source(4)),
        (("壬", "癸"), _precedence_source(5)),
        (("春", "夏"), _precedence_source(6)),
        (("秋", "冬"), _precedence_source(7)),
        (("东", "西"), _precedence_source(8)),
        (("南", "北"), _precedence_source(9)),
    )
    items = tuple(
        _precedence_item(tokens, source)
        for tokens, source in item_specs
    )
    assignments = [
        _assignment(
            items[0],
            protocol=protocol,
            split=protocol.training_split,
            probe_kind=None,
            provenance=(1230, 0),
        ),
        _assignment(
            items[1],
            protocol=protocol,
            split=protocol.development_split,
            probe_kind=_key(1233),
            provenance=(1230, 1),
        ),
        _assignment(
            items[2],
            protocol=protocol,
            split=protocol.held_out_split,
            probe_kind=_key(1231),
            provenance=(1230, 2),
        ),
    ]
    assignments.extend(
        _assignment(
            item,
            protocol=protocol,
            split=protocol.adversarial_split,
            probe_kind=kind,
            provenance=(1230, index + 2),
        )
        for index, (item, kind) in enumerate(zip(items[3:8], kinds))
    )
    assignments.append(_assignment(
        items[8],
        protocol=protocol,
        split=protocol.external_split,
        probe_kind=_key(1232),
        provenance=(1230, 8),
    ))
    return EvaluationPlan(protocol, tuple(assignments)), items


def _setup(*, full_coverage: bool = True, broken_off: bool = True):
    """构造覆盖 V-00 required dimensions/adversarial kinds 的 V-04 runtime。"""
    plan, items = _complete_plan(full_coverage=full_coverage)
    partition = plan.partition(items)
    wall_dimensions = (
        (plan.protocol.required_dimensions[-1],)
        if len(plan.protocol.required_dimensions) > 1 else ()
    )
    wall = (
        wall_dimensions[0]
        if wall_dimensions else _key(7999)
    )
    ablation_dimensions = tuple(
        dimension for dimension in plan.protocol.required_dimensions
        if dimension not in wall_dimensions
    )
    evaluator_key = _key(700)
    intervention_keys = tuple(
        _key(800 + index)
        for index in range(len(plan.protocol.required_adversarial_kinds))
    )
    cases = tuple(
        PreWeaningAblationCase(
            case_key=_key(900 + index),
            intervention_key=intervention_keys[index],
            evaluator_key=evaluator_key,
            identity=assignment.identity,
            dimension=ablation_dimensions[
                index % len(ablation_dimensions)],
        )
        for index, assignment in enumerate(
            plan.assignments[3:3 + len(
                plan.protocol.required_adversarial_kinds)])
    )
    protocol = PreWeaningValidationProtocol(
        version=3,
        ablation_cases=cases,
        required_ablation_cases=tuple(case.case_key for case in cases),
        probe_routes=(PreWeaningProbeRoute(_key(301), evaluator_key),),
        stopping_dimensions=plan.protocol.required_dimensions,
        wall_dimensions=wall_dimensions,
        resource_bounds=(
            ResourceBound(_key(1000), maximum_value=64),
            ResourceBound(_key(1001), minimum_value=10),
        ),
        consecutive_windows=2,
        checkpoint_step=5,
    )
    evaluator = _Evaluator(wall, broken_off=broken_off)
    reader = _StateReader(epoch=10)
    runtime = PreWeaningValidationRuntime(
        protocol,
        PreWeaningEvaluatorRegistry((
            PreWeaningEvaluatorBinding(evaluator_key, evaluator),
        )),
        PreWeaningInterventionRegistry(tuple(
            PreWeaningInterventionBinding(key, _Intervention(key))
            for key in intervention_keys
        )),
        reader,
    )
    backend = DictBackend()
    ctx = make_train_context(backend)
    ctx.evaluation_plan = plan
    ctx.evaluation_corpora = partition.as_dict()
    ctx.evaluation_strictly_isolated = True
    return plan, items, ctx, runtime, reader


def _resources(*, first_value: int = 32):
    """构造两个互不抵消且都有实际采样的资源测量。"""
    return (
        ResourceMeasurement(_key(1000), _key(1100), first_value, 2),
        ResourceMeasurement(_key(1001), _key(1101), 12, 3),
    )


def test_v04_breaks_real_precedence_layers_in_independent_clones():
    """真实 R-06 纵切的结构映射、事实、累计、晋升和消费者逐层 OFF 均失败。"""
    plan, items = _precedence_validation_plan()
    backend = DictBackend()
    ctx = make_train_context(backend)
    try:
        _install_precedence(ctx, _PrecedenceCourse(_precedence_domain()))
        from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
        DefaultRoundRunner().run_round(
            ctx,
            items[0],
            STAGE1_SKELETON,
            1,
        )
        partition = plan.partition(list(items))
        ctx.evaluation_plan = plan
        ctx.evaluation_corpora = partition.as_dict()
        ctx.evaluation_strictly_isolated = True
        evaluator_key = _key(1240)
        case_keys = tuple(_key(1250 + index) for index in range(5))
        intervention_keys = tuple(
            _key(1260 + index) for index in range(5))
        cases = tuple(
            PreWeaningAblationCase(
                case_keys[index],
                intervention_keys[index],
                evaluator_key,
                assignment.identity,
                plan.protocol.required_dimensions[0],
            )
            for index, assignment in enumerate(plan.assignments[3:8])
        )
        validation = PreWeaningValidationProtocol(
            version=4,
            ablation_cases=cases,
            required_ablation_cases=case_keys,
            probe_routes=(PreWeaningProbeRoute(
                _key(1231), evaluator_key),),
            stopping_dimensions=plan.protocol.required_dimensions,
            wall_dimensions=(),
            resource_bounds=(ResourceBound(
                _key(1270), maximum_value=8),),
            consecutive_windows=2,
            checkpoint_step=1,
        )
        runtime = PreWeaningValidationRuntime(
            validation,
            PreWeaningEvaluatorRegistry((PreWeaningEvaluatorBinding(
                evaluator_key,
                _PrecedenceLayerEvaluator(
                    plan.protocol.required_dimensions[0]),
            ),)),
            PreWeaningInterventionRegistry(tuple(
                PreWeaningInterventionBinding(
                    intervention_key,
                    _PrecedenceLayerIntervention(index + 1),
                )
                for index, intervention_key in enumerate(intervention_keys)
            )),
            _StateReader(epoch=1),
        )
        host_before = backend.snapshot()

        report = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=1,
            resources=(ResourceMeasurement(
                _key(1270), _key(1271), 5, 1),),
        ))

        assert report.ablations_complete is True
        assert len(report.ablations) == 5
        assert all(pair.enabled.outcome.passed is True
                   for pair in report.ablations)
        assert all(pair.disabled.outcome.passed is False
                   for pair in report.ablations)
        assert report.stop_allowed is False
        assert backend.snapshot() == host_before
    finally:
        backend.close()


def test_v04_requires_real_ablation_and_two_distinct_consecutive_windows():
    """全部破坏 FAIL、逐维合取和两个不同宿主 checkpoint 才允许停止。"""
    plan, _items, ctx, runtime, reader = _setup()
    try:
        host_before = ctx.backend.snapshot()
        first = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=10,
            resources=_resources(),
        ))
        assert first.ablations_complete is True
        assert first.stop_allowed is False
        assert len(first.ablations) == len(
            plan.protocol.required_adversarial_kinds)
        assert all(pair.complete for pair in first.ablations)
        assert ctx.backend.snapshot() == host_before

        reader.epoch = 15
        second = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=15,
            resources=_resources(),
            previous_windows=(first.windows[-1],),
        ))
        assert second.ablations_complete is True
        assert second.stop_allowed is True
        assert not hasattr(second, "mastered")
        assert not hasattr(second, "readiness")
        assert ctx.backend.snapshot() == host_before
    finally:
        ctx.backend.close()


def test_v04_identical_inputs_produce_identical_reports():
    """同协议、同状态和同 checkpoint 两次运行必须得到 bit-identical 报告。"""
    reports = []
    for _index in range(2):
        _plan, _items, ctx, runtime, _reader = _setup(
            full_coverage=False)
        try:
            reports.append(runtime.run(
                ctx,
                PreWeaningValidationRequest(
                    checkpoint=10,
                    resources=_resources(),
                ),
            ))
        finally:
            ctx.backend.close()
    assert reports[0] == reports[1]


def test_v04_rejects_same_host_state_relabel_and_discontinuous_history():
    """同状态重标或跳过预注册 checkpoint 间隔均不得冒充连续窗口。"""
    _plan, _items, ctx, runtime, reader = _setup()
    try:
        first = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=10,
            resources=_resources(),
        ))
        with pytest.raises(EvaluationProtocolError, match="同一宿主状态"):
            runtime.run(ctx, PreWeaningValidationRequest(
                checkpoint=15,
                resources=_resources(),
                previous_windows=(first.windows[-1],),
            ))
        reader.epoch = 16
        with pytest.raises(EvaluationProtocolError, match="不连续"):
            runtime.run(ctx, PreWeaningValidationRequest(
                checkpoint=16,
                resources=_resources(),
                previous_windows=(first.windows[-1],),
            ))
    finally:
        ctx.backend.close()


def test_v04_broken_consumer_wall_pass_and_resource_overrun_block_stop():
    """破坏臂未失败、墙维度偷渡 PASS 或任一资源越界都独立阻断。"""
    _plan, _items, ctx, runtime, reader = _setup(broken_off=False)
    try:
        first = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=10,
            resources=_resources(),
        ))
        reader.epoch = 15
        second = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=15,
            resources=_resources(),
            previous_windows=(first.windows[-1],),
        ))
        assert second.ablations_complete is False
        assert second.stop_allowed is False
    finally:
        ctx.backend.close()

    _plan, _items, ctx, runtime, reader = _setup()
    try:
        first = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=10,
            resources=_resources(first_value=65),
        ))
        reader.epoch = 15
        second = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=15,
            resources=_resources(),
            previous_windows=(first.windows[-1],),
        ))
        assert second.stop_allowed is False

        wall = runtime.protocol.wall_dimensions[0]
        stolen = replace(
            second.windows[-1],
            dimensions=tuple(
                replace(item, passed=1, not_evaluated=0, sample_count=1)
                if item.dimension == wall else item
                for item in second.windows[-1].dimensions
            ),
        )
        reader.epoch = 20
        third = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=20,
            resources=_resources(),
            previous_windows=(stolen,),
        ))
        assert third.stop_allowed is False
    finally:
        ctx.backend.close()


def test_v04_owner_drift_and_incomplete_plan_fail_closed():
    """窗口间 evaluator 阈值漂移及 V-00 覆盖缺失都不能继续判停。"""
    _plan, _items, ctx, runtime, reader = _setup()
    try:
        first = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=10,
            resources=_resources(),
        ))
        runtime.evaluators._bindings[0].evaluator.broken_off = False
        reader.epoch = 15
        second = runtime.run(ctx, PreWeaningValidationRequest(
            checkpoint=15,
            resources=_resources(),
            previous_windows=(first.windows[-1],),
        ))
        assert second.stop_allowed is False
    finally:
        ctx.backend.close()

    _plan, _items, ctx, runtime, _reader = _setup(full_coverage=False)
    try:
        invalid = replace(
            runtime.protocol,
            stopping_dimensions=(_key(9999),),
            wall_dimensions=(),
        )
        runtime.protocol = invalid
        with pytest.raises(EvaluationProtocolError, match="required dimensions"):
            runtime.run(ctx, PreWeaningValidationRequest(
                checkpoint=10,
                resources=_resources(),
            ))
    finally:
        ctx.backend.close()


def test_v04_formal_train_opt_in_caller_and_default_off(tmp_path):
    """formal_train 只在成对 opt-in 时返回 V-04 报告，且不形成 readiness。"""
    plan, items, ctx, runtime, _reader = _setup(full_coverage=False)
    ctx.backend.close()
    backend = DictBackend()
    try:
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="v04-opt-in",
                rounds_per_stage=1,
                active_training_stages=(),
                persist_graph_dump=False,
                evaluation_plan=plan,
                pre_weaning_validation_runtime=runtime,
                pre_weaning_validation_request=PreWeaningValidationRequest(
                    checkpoint=10,
                    resources=_resources(),
                ),
            ),
            items,
            backend=backend,
        )
        assert result.pre_weaning_validation_report is not None
        assert result.pre_weaning_validation_report.stop_allowed is False
        assert result.weaning_ready is False
    finally:
        backend.close()

    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path),
                run_id="v04-incomplete",
                active_training_stages=(),
                persist_graph_dump=False,
                evaluation_plan=plan,
                pre_weaning_validation_runtime=runtime,
            ),
            items,
            backend=DictBackend(),
        )
