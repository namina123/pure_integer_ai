"""V-04 对共现、逻辑、WorkMemory 和 attractor 实际消费者的反向破坏测试。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.edge_types import EDGE_COOCCURS
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningBudget
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.cooccurs import build_cooccurs
from pure_integer_ai.experiments.evaluation_protocol import (
    ProbeOutcome,
    ProtocolKey,
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
from pure_integer_ai.storage.backend import DictBackend

from tests.test_a10_attractor_state import (
    _goals,
    _planner,
    _setup as _attractor_setup,
)
from tests.test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
)
from tests.test_r08_logic_closure import _production_fixture
from tests.test_v00_evaluation_protocol import _complete_plan


def _key(value: int) -> ProtocolKey:
    """构造测试使用的开放协议键。"""
    return ProtocolKey((value,))


class _StateReader:
    """读取实际宿主 backend 和 WorkMemory，供 V-04 双层零写核验。"""

    def state_key(self) -> tuple[int, ...]:
        """返回本测试状态读取协议的固定身份。"""
        return 1, 1300

    def read(self, ctx):
        """返回宿主持久态和全部活动生命周期边界。"""
        memory = ctx.work_memory
        return (
            ctx.backend.snapshot(),
            memory.active_session_scope,
            memory.active_document_scope,
            memory.active_episode_scope,
            memory.active_query_scope,
            memory.active_generation_scope,
        )


class _DropCooccurrenceStore:
    """保留 EdgeStore 读接口，但故意丢弃两类共现写入。"""

    def __init__(self, delegate):
        """绑定 clone 内真实 EdgeStore 作为只读委托。"""
        self._delegate = delegate

    def add(self, **_kwargs) -> None:
        """模拟共现 writer 未接线，不向 clone 写边。"""
        return None

    def add_cooccurs_dedup(self, **_kwargs) -> bool:
        """模拟 dedup writer 未接线，并报告没有新边。"""
        return False

    def __getattr__(self, name):
        """其余读取行为继续使用真实 clone EdgeStore。"""
        return getattr(self._delegate, name)


class _CooccurrenceIntervention:
    """关闭 clone 内的共现写入边界。"""

    def state_key(self) -> tuple[int, ...]:
        """返回共现破坏实现身份。"""
        return 1, 1310

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """健康臂保留真实 store，破坏臂安装丢写代理。"""
        if not enabled:
            eval_ctx.edge_store = _DropCooccurrenceStore(eval_ctx.edge_store)


class _CooccurrenceEvaluator:
    """要求实际共现 builder 在 clone 图中留下目标边。"""

    def state_key(self) -> tuple[int, ...]:
        """返回共现评测判据身份。"""
        return 1, 1311

    def evaluate(self, eval_ctx, item, _request) -> ProbeOutcome:
        """从评测输入物化三个概念，并核验实际 COOCCURS 边而非返回计数。"""
        symbols = tuple((item.raw_text or "")[:3])
        refs = [
            eval_ctx.concept_index.ensure(
                f"v04-cooccurs-{index}-{token}",
                space_id=eval_ctx.space_id,
            )
            for index, token in enumerate(symbols)
        ]
        before = len(eval_ctx.edge_store.query_type(EDGE_COOCCURS))
        build_cooccurs(
            eval_ctx.edge_store,
            refs,
            lang=item.lang,
            domain=item.domain,
            source=item.source,
            space_id=eval_ctx.space_id,
        )
        after = len(eval_ctx.edge_store.query_type(EDGE_COOCCURS))
        passed = len(refs) >= 2 and after > before
        return ProbeOutcome(passed, value=after - before, sample_count=1)


class _NoLogicExecutionCourse:
    """保留 R-08 课程身份和 owner，只撤掉当前 execution seed。"""

    def __init__(self, delegate):
        """绑定 clone 内真实课程 mapper。"""
        self._delegate = delegate

    def request(self, scope, *, read_only: bool):
        """复用真实请求的 scope/forming/recognition，仅移除 execution。"""
        request = self._delegate.request(scope, read_only=read_only)
        return replace(request, executions=())

    def clone_for_evaluation(self):
        """返回对独立真实课程 clone 的同种破坏包装。"""
        return _NoLogicExecutionCourse(
            self._delegate.clone_for_evaluation())

    def state_key(self) -> tuple[int, ...]:
        """返回破坏版本和底层课程的完整稳定状态。"""
        return 1, 1319, *self._delegate.state_key()


class _LogicIntervention:
    """关闭 clone 内 R-08 课程的 execution seed。"""

    def state_key(self) -> tuple[int, ...]:
        """返回逻辑破坏实现身份。"""
        return 1, 1320

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """破坏臂保留 owner 和图，只替换 clone 内课程请求 mapper。"""
        if not enabled:
            runtime = eval_ctx.logic_closure_runtime
            runtime.course = _NoLogicExecutionCourse(runtime.course)


class _LogicEvaluator:
    """要求 R-08 从已学图执行真实有限逻辑 bundle。"""

    def __init__(self, world):
        """绑定已由训练侧形成、评测侧只读采用的逻辑世界。"""
        self.world = world

    def state_key(self) -> tuple[int, ...]:
        """返回逻辑世界来源的稳定评测身份。"""
        return 1, *self.world.source.stable_key()

    def evaluate(self, eval_ctx, _item, _request) -> ProbeOutcome:
        """在 clone 内只读执行逻辑 runtime，并要求出现非空 execution。"""
        runtime = eval_ctx.logic_closure_runtime
        report = runtime.process(
            document_scope(self.world.source),
            read_only=True,
        )
        passed = len(report.executions) > 0
        return ProbeOutcome(
            passed,
            value=len(report.executions),
            sample_count=1,
        )


class _WorkMemoryIntervention:
    """故意移除 V-06 已建立的 session 边界。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 WorkMemory 破坏实现身份。"""
        return 1, 1330

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """破坏臂通过正式生命周期 API 关闭当前 session。"""
        if not enabled:
            eval_ctx.work_memory.end_session()


class _WorkMemoryEvaluator:
    """要求评测输入只能在活动 session 下进入并退出 document。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 WorkMemory 边界判据身份。"""
        return 1, 1331

    def evaluate(self, eval_ctx, item, _request) -> ProbeOutcome:
        """用真实 begin/end API 验证 document 边界，缺 session 时明确 FAIL。"""
        session = eval_ctx.work_memory.active_session_scope
        if session is None:
            return ProbeOutcome(False, value=0, sample_count=1)
        source = SourceRef(
            item.source_ref.source_kind,
            item.source_ref.source_id,
            item.source_ref.document_id,
            session.owner,
            session.versions,
        )
        try:
            eval_ctx.work_memory.begin_document(
                document_scope(source, parent=session))
        except RuntimeError:
            return ProbeOutcome(False, value=0, sample_count=1)
        eval_ctx.work_memory.end_document()
        return ProbeOutcome(True, value=1, sample_count=1)


class _AttractorIntervention:
    """保留 agenda 形成但关闭唯一 reasoning consumer，制造伪闭环。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 attractor consumer 破坏实现身份。"""
        return 1, 1340

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """破坏臂让 consume 调用无结果，agenda 本身仍可形成。"""
        if enabled:
            return

        def disabled_consume(*_args, **_kwargs):
            """模拟消费者未接线，不提交 processing trace。"""
            return None

        eval_ctx.attractor_runtime.consume_reasoning = disabled_consume


class _AttractorEvaluator:
    """要求 A-10 不只形成 agenda，还必须由 S-05 真消费并留下 trace。"""

    def __init__(self, source):
        """绑定已有 M-06/M-07/A-10 协议的 query 来源。"""
        self.source = source

    def state_key(self) -> tuple[int, ...]:
        """返回 query 来源的稳定评测身份。"""
        return 1, *self.source.stable_key()

    def evaluate(self, eval_ctx, _item, _request) -> ProbeOutcome:
        """编译当前 query、形成 agenda、执行 consumer，并拒绝只有 frontier 的伪闭环。"""
        eval_ctx.work_memory.end_session()
        scope = _open_query(eval_ctx, self.source)
        try:
            compilation = eval_ctx.memory_query_runtime.compile(
                _current(eval_ctx, self.source, scope),
                access=MemoryAccessContext(1, 2, 3),
            )
            goals = _goals(self.source, scope)
            state = eval_ctx.attractor_runtime.resolve_and_activate(
                compilation,
                goals,
            )
            _, _, consumer = _planner(self.source)
            result = eval_ctx.attractor_runtime.consume_reasoning(
                consumer,
                ReasoningBudget(1, 0, 0),
            )
            passed = (
                result is not None
                and len(state.processing_traces()) == 1
            )
            return ProbeOutcome(
                passed,
                value=len(state.processing_traces()),
                sample_count=1,
            )
        finally:
            _close_query(eval_ctx)


def _run_single_actual_case(ctx, evaluator, intervention) -> None:
    """把一个实际消费者放入标准 V-04 ON/OFF runner 并核验宿主零写。"""
    plan, items = _complete_plan(full_coverage=False)
    partition = plan.partition(items)
    ctx.evaluation_plan = plan
    ctx.evaluation_corpora = partition.as_dict()
    ctx.evaluation_strictly_isolated = True
    evaluator_key = _key(1350)
    intervention_key = _key(1351)
    case_key = _key(1352)
    case = PreWeaningAblationCase(
        case_key,
        intervention_key,
        evaluator_key,
        plan.assignments[3].identity,
        plan.protocol.required_dimensions[0],
    )
    protocol = PreWeaningValidationProtocol(
        version=5,
        ablation_cases=(case,),
        required_ablation_cases=(case_key,),
        probe_routes=(PreWeaningProbeRoute(_key(301), evaluator_key),),
        stopping_dimensions=plan.protocol.required_dimensions,
        wall_dimensions=(),
        resource_bounds=(ResourceBound(_key(1353), maximum_value=4),),
        consecutive_windows=2,
        checkpoint_step=1,
    )
    runtime = PreWeaningValidationRuntime(
        protocol,
        PreWeaningEvaluatorRegistry((
            PreWeaningEvaluatorBinding(evaluator_key, evaluator),
        )),
        PreWeaningInterventionRegistry((
            PreWeaningInterventionBinding(
                intervention_key,
                intervention,
            ),
        )),
        _StateReader(),
    )
    before = ctx.backend.snapshot()
    report = runtime.run(ctx, PreWeaningValidationRequest(
        checkpoint=1,
        resources=(ResourceMeasurement(
            _key(1353), _key(1354), 1, 1),),
    ))
    assert report.ablations_complete is True
    assert report.ablations[0].enabled.outcome.passed is True
    assert report.ablations[0].disabled.outcome.passed is False
    assert ctx.backend.snapshot() == before


def test_v04_actual_cooccurrence_writer_breaks_metric():
    """关闭实际共现 writer 后，计数返回不能掩盖图中没有 COOCCURS 边。"""
    ctx = make_train_context(DictBackend())
    try:
        _run_single_actual_case(
            ctx,
            _CooccurrenceEvaluator(),
            _CooccurrenceIntervention(),
        )
    finally:
        ctx.backend.close()


def test_v04_actual_logic_seed_owner_breaks_metric():
    """移除 R-08 owner 后，已有课程图不能凭 fixture 直接产出逻辑 PASS。"""
    ctx, world, runtime = _production_fixture()
    try:
        runtime.process(document_scope(world.source), read_only=False)
        _run_single_actual_case(
            ctx,
            _LogicEvaluator(world),
            _LogicIntervention(),
        )
    finally:
        ctx.backend.close()


def test_v04_actual_work_memory_boundary_breaks_metric():
    """关闭 session 后，document 输入必须由真实 WorkMemory 边界拒绝。"""
    ctx = make_train_context(DictBackend())
    try:
        _run_single_actual_case(
            ctx,
            _WorkMemoryEvaluator(),
            _WorkMemoryIntervention(),
        )
    finally:
        ctx.backend.close()


def test_v04_actual_attractor_without_consumer_is_pseudo_loop():
    """只有 AttractorState/frontier 而没有 processing trace 必须判为伪闭环。"""
    setup = _attractor_setup(prefer_matching_document=False)
    backend, ctx, source, _runtime, _strategy, _compilation, _goals_value = setup
    try:
        _close_query(ctx)
        _run_single_actual_case(
            ctx,
            _AttractorEvaluator(source),
            _AttractorIntervention(),
        )
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()
