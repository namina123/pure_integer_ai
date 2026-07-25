"""F-01 设施总装协议、隔离裁决和反 theater 基础测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.post_weaning import (
    PostWeaningIntakeRequest,
)
from pure_integer_ai.cognition.shared.source_trust import (
    SOURCE_ADMISSION_ACCEPTED,
    SourceTrustAssessment,
)
from pure_integer_ai.experiments import facility_readiness_runtime as runtime_module
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.experiments.facility_readiness import (
    FacilityCounter,
    FacilityCounterRequirement,
    FacilityDimensionRequirement,
    FacilityExerciseMeasurement,
    FacilityIntegrityCheck,
    FacilityReadinessProtocol,
)
from pure_integer_ai.experiments.facility_readiness_runtime import (
    FacilityExerciseBinding,
    FacilityReadinessError,
    FacilityReadinessRuntime,
)
from pure_integer_ai.experiments.memory_generation_outcome_runtime import (
    MemoryQuestionOutcomeCommitter,
)
from pure_integer_ai.experiments.memory_generation_runtime import (
    MemoryAwareQuestionDialogueRuntime,
    MemoryQuestionSelectionCommitter,
    ResolvedMemoryQuestionExecutor,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_TEST_ONLY,
    inventory_by_id,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
    PostWeaningDryRunRuntime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.node_store import NODE_CONCEPT
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_from_payload,
    dump_run,
    load_run_package,
)

from tests.test_a08_memory_reparse import (
    _LICENSE as _A08_LICENSE,
    _NewParser as _A08NewParser,
    _RejectParser as _A08RejectParser,
    _TEXT as _A08_TEXT,
    _world as _a08_world,
)
from tests.test_a10_attractor_state import _goals
from tests.test_c02_capability_memory import (
    _execution_setup as _capability_execution_setup,
)
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import _fixture as _question_fixture
from tests.test_g05_memory_generation_evidence import (
    _EmptyQuestionExecutor,
    _UseBeforePostcheckMapper,
    _complete_source,
    _outcome_protocol,
)
from tests.test_k03_training_shard_merge import _published_bytes
from tests.test_k04_memory_hot_set import (
    _batch_config,
    _core_refs,
    _install_resolver,
    _publish_projection,
    _query_source,
    _seed_memory,
)
from tests.test_m06_memory_query import _current, _open_query
from tests.test_v00_evaluation_protocol import _complete_plan
from tests.test_pw00_post_weaning_runtime import (
    _Parser as _PostWeaningParser,
    _install_post_weaning_consumers,
    _instruction,
    _post_weaning_manifest,
    _question_dialogue,
    _restore_runtime,
    _source as _post_weaning_source,
)
from tests.test_v05_actual_ablation import (
    _close_outer_lifecycle,
    _install_memory_ablation,
    _observation,
    _run_question_once,
)


def _key(value: int) -> ProtocolKey:
    """构造 F-01 测试使用的开放协议键。"""
    return ProtocolKey((value,))


_EXERCISE = _key(5100)
_DIMENSION = _key(5101)
_METRIC = _key(5102)
_CHECK = _key(5103)
_FORBIDDEN = tuple(_key(value) for value in range(5110, 5114))
_BOUNDARIES = tuple(_key(value) for value in range(5120, 5123))
_ACCESS = MemoryAccessContext(1, 2, 3)


def _protocol(
        *,
        mechanisms: tuple[str, ...] = ("memory.source_trust_admission",),
        ) -> FacilityReadinessProtocol:
    """构造一维行为、计数、完整性和四类禁用信号协议。"""
    return FacilityReadinessProtocol(
        version=1,
        exercise_key=_EXERCISE,
        dimensions=(FacilityDimensionRequirement(
            _DIMENSION,
            minimum_behavior_improvement=5,
            counters=(FacilityCounterRequirement(_METRIC, 1),),
            checks=(_CHECK,),
        ),),
        required_mechanism_ids=mechanisms,
        forbidden_counter_keys=_FORBIDDEN,
        boundary_keys=_BOUNDARIES,
    )


class _Exercise:
    """产生可注入缺证据、污染、状态漂移或 Core 改写的测量。"""

    def __init__(
            self,
            *,
            omit_metric: bool = False,
            omit_check: bool = False,
            forbidden_value: int = 0,
            mutate_core: bool = False,
            drift_prepare: bool = False,
            drift_run: bool = False,
            ) -> None:
        """冻结全部故障模式，使 state_key 能覆盖 adapter 配置。"""
        self.omit_metric = omit_metric
        self.omit_check = omit_check
        self.forbidden_value = forbidden_value
        self.mutate_core = mutate_core
        self.drift_prepare = drift_prepare
        self.drift_run = drift_run
        self.epoch = 0

    def state_key(self) -> tuple[int, ...]:
        """返回全部配置及运行中不得变化的 epoch。"""
        return (
            1,
            int(self.omit_metric),
            int(self.omit_check),
            self.forbidden_value,
            int(self.mutate_core),
            int(self.drift_prepare),
            int(self.drift_run),
            self.epoch,
        )

    def prepare(self, eval_ctx) -> None:
        """核验评测输入已清洗，并可模拟准备阶段配置漂移。"""
        assert eval_ctx.teacher is None
        assert all(
            item.expected_outcome is None
            for item in eval_ctx.evaluation_plan.assignments
        )
        if self.drift_prepare:
            self.epoch += 1

    def run(self, eval_ctx) -> FacilityExerciseMeasurement:
        """生成真实 typed measurement，并可模拟 Core 或 adapter 违规。"""
        if self.mutate_core:
            local_id = eval_ctx.core_space.new_local_id()
            eval_ctx.core_space.nodes.put(
                eval_ctx.core_space.space_id,
                local_id,
                node_type=NODE_CONCEPT,
            )
        counters = []
        if not self.omit_metric:
            counters.append(FacilityCounter(_METRIC, 3, 1, (1, 3)))
        counters.extend(
            FacilityCounter(key, self.forbidden_value, 1, (2, index))
            for index, key in enumerate(_FORBIDDEN, start=1)
        )
        check_key = _key(5199) if self.omit_check else _CHECK
        checks = (FacilityIntegrityCheck(
            check_key,
            True,
            CanonicalIdentity.from_value(("stable", 1)),
            CanonicalIdentity.from_value(("stable", 1)),
            (3, 1),
        ),)
        measurement = FacilityExerciseMeasurement(
            _EXERCISE,
            (4, 1),
            10,
            0,
            tuple(sorted(counters)),
            checks,
            (5, 1),
        )
        if self.drift_run:
            self.epoch += 1
        return measurement


class _BadStateExercise(_Exercise):
    """返回含 bool 的伪整数状态键。"""

    def state_key(self) -> tuple[int, ...]:
        """故意违反严格整数 adapter 状态协议。"""
        return 1, True


def _context():
    """构造带完整 expected 的宿主，供 F-01 证明 clone 内清洗。"""
    plan, items = _complete_plan()
    ctx = make_train_context(DictBackend())
    ctx.evaluation_plan = plan
    ctx.evaluation_corpora = plan.partition(items).as_dict()
    ctx.evaluation_strictly_isolated = True
    return ctx


def _runtime(
        exercise: _Exercise,
        *,
        protocol: FacilityReadinessProtocol | None = None,
        ) -> FacilityReadinessRuntime:
    """把测试 adapter 绑定到冻结协议。"""
    selected = protocol or _protocol()
    return FacilityReadinessRuntime(
        selected,
        FacilityExerciseBinding(selected.exercise_key, exercise),
    )


def test_f01_rejects_bad_or_drifting_adapter_state_keys():
    """非法状态键及 prepare/run 配置漂移都必须 fail closed。"""
    with pytest.raises(TypeError, match="严格整数"):
        FacilityExerciseBinding(_EXERCISE, _BadStateExercise())

    for exercise in (
            _Exercise(drift_prepare=True),
            _Exercise(drift_run=True),
            ):
        ctx = _context()
        try:
            with pytest.raises(FacilityReadinessError, match="adapter 配置状态"):
                _runtime(exercise).run(ctx)
        finally:
            ctx.backend.close()


@pytest.mark.parametrize(
    ("exercise", "message"),
    (
        (_Exercise(omit_metric=True), "缺少维度必需计数"),
        (_Exercise(omit_check=True), "缺少维度完整性检查"),
    ),
)
def test_f01_rejects_missing_required_measurement_evidence(exercise, message):
    """缺失预注册计数或完整性检查时不得生成可消费报告。"""
    ctx = _context()
    try:
        with pytest.raises(FacilityReadinessError, match=message):
            _runtime(exercise).run(ctx)
    finally:
        ctx.backend.close()


def test_f01_core_change_and_forbidden_signal_never_complete():
    """Core 改写或任一评测污染信号非零都阻断设施完成。"""
    for exercise in (
            _Exercise(mutate_core=True),
            _Exercise(forbidden_value=1),
            ):
        ctx = _context()
        try:
            report = _runtime(exercise).run(ctx)
            assert report.facility_complete is False
        finally:
            ctx.backend.close()


def test_f01_rejects_test_only_mechanism_but_accepts_production_candidate(
        monkeypatch,
        ):
    """四态只拒绝 dead/test-only，不误拒真实 production 机制。"""
    records = inventory_by_id()
    source_record = records["memory.source_trust_admission"]
    monkeypatch.setattr(
        runtime_module,
        "inventory_by_id",
        lambda: {
            source_record.mechanism_id: replace(
                source_record,
                status=STATUS_TEST_ONLY,
            ),
        },
    )
    ctx = _context()
    try:
        report = _runtime(_Exercise()).run(ctx)
        assert report.facility_complete is False
        assert report.mechanisms[0].passed is False
    finally:
        ctx.backend.close()

    monkeypatch.setattr(runtime_module, "inventory_by_id", inventory_by_id)
    ctx = _context()
    try:
        report = _runtime(
            _Exercise(),
            protocol=_protocol(mechanisms=("generation.slot_dispatch",)),
        ).run(ctx)
        assert report.facility_complete is True
        assert report.mechanisms[0].passed is True
    finally:
        ctx.backend.close()


def test_f01_scrubs_expected_and_preserves_host_state():
    """exercise 只能看清洗后的 clone，宿主 backend 和计划保持不变。"""
    ctx = _context()
    before_backend = ctx.backend.snapshot()
    before_plan = ctx.evaluation_plan
    try:
        report = _runtime(_Exercise()).run(ctx)
        assert report.facility_complete is True
        assert ctx.backend.snapshot() == before_backend
        assert ctx.evaluation_plan == before_plan
        assert any(
            item.expected_outcome is not None
            for item in ctx.evaluation_plan.assignments
        )
    finally:
        ctx.backend.close()


def test_f01_identical_inputs_produce_identical_reports():
    """相同协议、fixture 和 adapter 必须生成 bit-identical 报告。"""
    reports = []
    for _ordinal in range(2):
        ctx = _context()
        try:
            reports.append(_runtime(_Exercise()).run(ctx))
        finally:
            ctx.backend.close()
    assert reports[0] == reports[1]


def test_f01_protocol_requires_real_behavior_difference():
    """零改善阈值不能把 ON/OFF 无差异包装成总装证据。"""
    with pytest.raises(ValueError, match="必须为正"):
        FacilityDimensionRequirement(
            _DIMENSION,
            minimum_behavior_improvement=0,
            counters=(FacilityCounterRequirement(_METRIC, 1),),
            checks=(_CHECK,),
        )


_ACTUAL_EXERCISE = _key(5200)
_ACTUAL_DIMENSIONS = tuple(_key(value) for value in range(5210, 5215))
_ACTUAL_METRICS = tuple(_key(value) for value in range(5230, 5242))
_ACTUAL_CHECKS = tuple(_key(value) for value in range(5250, 5260))
_ACTUAL_FORBIDDEN = tuple(_key(value) for value in range(5270, 5274))
_ACTUAL_BOUNDARIES = tuple(_key(value) for value in range(5280, 5285))


def _actual_protocol() -> FacilityReadinessProtocol:
    """冻结 F-01 五个承重维度、真实机制和诚实边界。"""
    requirements = (
        FacilityDimensionRequirement(
            _ACTUAL_DIMENSIONS[0],
            50,
            tuple(sorted((
                FacilityCounterRequirement(_ACTUAL_METRICS[0], 2),
                FacilityCounterRequirement(_ACTUAL_METRICS[1], 1),
                FacilityCounterRequirement(_ACTUAL_METRICS[2], 1),
                FacilityCounterRequirement(_ACTUAL_METRICS[3], 1),
            ))),
            tuple(sorted((_ACTUAL_CHECKS[0], _ACTUAL_CHECKS[1]))),
        ),
        FacilityDimensionRequirement(
            _ACTUAL_DIMENSIONS[1],
            50,
            tuple(sorted((
                FacilityCounterRequirement(_ACTUAL_METRICS[4], 1),
                FacilityCounterRequirement(_ACTUAL_METRICS[5], 4),
                FacilityCounterRequirement(_ACTUAL_METRICS[6], 1),
            ))),
            tuple(sorted((_ACTUAL_CHECKS[2], _ACTUAL_CHECKS[3]))),
        ),
        FacilityDimensionRequirement(
            _ACTUAL_DIMENSIONS[2],
            50,
            tuple(sorted((
                FacilityCounterRequirement(_ACTUAL_METRICS[7], 1),
                FacilityCounterRequirement(_ACTUAL_METRICS[8], 1),
            ))),
            (_ACTUAL_CHECKS[4],),
        ),
        FacilityDimensionRequirement(
            _ACTUAL_DIMENSIONS[3],
            50,
            tuple(sorted((
                FacilityCounterRequirement(_ACTUAL_METRICS[9], 1),
                FacilityCounterRequirement(_ACTUAL_METRICS[10], 4),
            ))),
            tuple(sorted((
                _ACTUAL_CHECKS[5],
                _ACTUAL_CHECKS[6],
                _ACTUAL_CHECKS[7],
                _ACTUAL_CHECKS[8],
            ))),
        ),
        FacilityDimensionRequirement(
            _ACTUAL_DIMENSIONS[4],
            50,
            (FacilityCounterRequirement(_ACTUAL_METRICS[11], 2),),
            (_ACTUAL_CHECKS[9],),
        ),
    )
    mechanisms = tuple(sorted((
        "capability.verified_memory_reuse",
        "evaluation.post_weaning_memory_ablation_stop",
        "evaluation.pre_weaning_ablation_stop",
        "memory.batch_recovery_protocol",
        "memory.generation_use_outcome_bridge",
        "memory.parser_revision_rebuild",
        "memory.query_attractor_agenda",
        "memory.query_hot_set_runtime",
        "memory.source_trust_admission",
        "question.typed_answer_generation_runtime",
        "runtime.facility_readiness_assembly",
        "runtime.post_weaning_dry_run",
        "training.sharded_barrier_protocol",
    )))
    return FacilityReadinessProtocol(
        version=1,
        exercise_key=_ACTUAL_EXERCISE,
        dimensions=tuple(sorted(requirements)),
        required_mechanism_ids=mechanisms,
        forbidden_counter_keys=_ACTUAL_FORBIDDEN,
        boundary_keys=_ACTUAL_BOUNDARIES,
    )


@dataclass(frozen=True)
class _MemoryPathEvidence:
    """保存同一 clone 内 A-05 到 G-05 主纵切的可汇总证据。"""

    positive_behavior: int
    negative_behavior: int
    admissions: int
    source_clusters: int
    candidates: int
    conflicts: int
    uses: int
    outcomes: int
    query_key: tuple[int, ...]
    query_before: CanonicalIdentity
    query_after: CanonicalIdentity
    resources_closed: bool
    result_identity: CanonicalIdentity
    observation_ref: object
    source: object


def _refresh_projection(ctx):
    """重建脏 aggregate 并发布与当前 Memory 状态一致的 K-04 投影。"""
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
    projection = _publish_projection(
        ctx,
        ctx.memory_resolver_runtime.resolver,
    )
    ctx.memory_hot_set_runtime.replace_projection(projection)
    return projection


def _event_count(ctx, event_kind: int) -> int:
    """跨阅读和交互 Memory 空间统计一种真实事件。"""
    return sum(
        len(event_log.query(access=_ACCESS, event_kind=event_kind))
        for event_log in (
            ctx.memory_read_events,
            ctx.memory_interact_events,
        )
    )


def _prepare_actual_fixture(eval_ctx) -> None:
    """在冻结 Core 前安装主纵切所需的真实 M/K/A/PW-00 owner。"""
    from pure_integer_ai.cognition.shared.memory_batch import (
        install_memory_batch_runtimes,
    )

    eval_ctx.work_memory.end_session()
    install_memory_batch_runtimes(eval_ctx, _batch_config())
    _seed_memory(eval_ctx)
    source = _query_source(document_id=1)
    core_refs = _core_refs(eval_ctx)
    _, resolver_runtime = _install_resolver(
        eval_ctx,
        source,
        core_refs[1],
    )
    from tests.test_m08_memory_use import _append_observation

    observation = _append_observation(eval_ctx, source, core_refs)
    projection = _publish_projection(eval_ctx, resolver_runtime.resolver)
    _install_post_weaning_consumers(eval_ctx, source, projection)
    routes, _ = _post_weaning_manifest(eval_ctx, source)
    warm_fixture, _ = _question_dialogue(eval_ctx, source, observation)
    warm_fixture.close()
    _close_outer_lifecycle(eval_ctx)
    routes, manifest = _post_weaning_manifest(eval_ctx, source)
    eval_ctx.f01_source = source
    eval_ctx.f01_observation = observation
    eval_ctx.f01_projection = projection
    eval_ctx.f01_routes = routes
    eval_ctx.f01_manifest = manifest


def _run_main_memory_path(ctx) -> _MemoryPathEvidence:
    """执行 A-05 双文档准入、Memory OFF/ON、Use 与逐维 outcome。"""
    source = ctx.f01_source
    routes = ctx.f01_routes
    intake_runtime = PostWeaningDryRunRuntime(ctx, ctx.f01_manifest)
    first_source = _post_weaning_source(601)
    second_source = replace(first_source, document_id=602)
    for ordinal, admitted_source in enumerate(
            (first_source, second_source), start=1):
        intake_runtime.run_intake(PostWeaningIntakeRequest(
            routes.reading,
            admitted_source,
            f"F-01 来源 {ordinal}",
            f"license-f01-{ordinal}",
            52600 + ordinal,
            parser=_PostWeaningParser(admitted_source, 40 + ordinal),
            trace=(52601, ordinal),
        ))
    admission_records = tuple(
        ctx.source_trust_records.find(item.stable_key())
        for item in (first_source, second_source)
    )
    assessments = tuple(
        SourceTrustAssessment.from_stable_key(item.assessment_key)
        for item in admission_records
        if item is not None
    )
    admissions = sum(
        item.decision == SOURCE_ADMISSION_ACCEPTED
        for item in assessments
    )
    clusters = len({item.source_cluster_key for item in assessments})
    _close_outer_lifecycle(ctx)
    _refresh_projection(ctx)

    scope = _open_query(ctx, source)
    current = _current(ctx, source, scope)
    compilation = ctx.memory_query_runtime.compile(current, access=_ACCESS)
    resolution = ctx.memory_resolver_runtime.resolve(compilation)
    candidates = tuple(
        item
        for candidate_set in resolution.sets
        for item in candidate_set.candidates
    )
    conflicts = sum(
        item.aggregate is not None
        and item.aggregate.support_count > 0
        and item.aggregate.contradict_count > 0
        for item in candidates
    )
    repository = SourceRecordRepository(ctx.backend)
    traces = {
        trace.source.stable_key(): trace
        for item in candidates
        for trace in item.memory_source_traces
    }
    for ordinal, trace in enumerate(
            (traces[key] for key in sorted(traces)), start=1):
        _complete_source(repository, trace, ordinal)
    goals = _goals(source, scope)
    ctx.work_memory.end_query()
    target = goals[1].proposition
    off_fixture = _question_fixture(
        executor_factory=lambda route: _EmptyQuestionExecutor(
            _instruction(source, 20140)),
        world=(source, current.scope, target),
    )
    on_fixture = None
    try:
        off_dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            off_fixture.runtime,
            trace_prefix=(20146, 1),
            source_records=repository,
        )
        off_run = off_dialogue.run(off_fixture.request)
        executor = ResolvedMemoryQuestionExecutor(
            ctx,
            current,
            _ACCESS,
            goals,
            executed_reason=_instruction(source, 20140),
            binding_reason=_instruction(source, 20141),
            trace_prefix=(20142, 1),
            source_records=repository,
        )
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 20143),
            input_observation_ref=ctx.f01_observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 20144)),
            trace_prefix=(20145, 1),
        )
        mapper, postchecker, _, _, _ = _postcheck_owners()
        ordered_mapper = _UseBeforePostcheckMapper(ctx, mapper)
        outcome_committer = MemoryQuestionOutcomeCommitter(
            ctx.memory_use_runtime,
            _outcome_protocol(source, postchecker),
            trace_prefix=(52618, 1),
        )
        on_fixture = _question_fixture(
            executor_factory=lambda route: executor,
            world=(source, current.scope, target),
            selection_committer=committer,
            postcheck_mapper=ordered_mapper,
            postchecker=postchecker,
            outcome_committer=outcome_committer,
        )
        on_dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            on_fixture.runtime,
            trace_prefix=(20146, 1),
            source_records=repository,
        )
        _, question_manifest = _post_weaning_manifest(ctx, source)
        operation = PostWeaningDryRunRuntime(
            ctx, question_manifest).run_question(
                on_dialogue, on_fixture.request)
        question = operation.result.question
        rendered_key = question.generation.rendered.stable_key()
        result_identity = CanonicalIdentity.from_value((
            question.status,
            question.query.request.target,
            rendered_key,
        ))
        resources_closed = (
            operation.report.query_closed
            and ctx.memory_hot_set_runtime.query_resources_closed()
            and ctx.work_memory.active_query_scope is None
            and ctx.work_memory.attractor_state is None
        )
        _close_outer_lifecycle(ctx)
        return _MemoryPathEvidence(
            positive_behavior=100 if question.complete else 0,
            negative_behavior=(
                100
                if off_run.question.selection.selected_candidate_keys
                else 0
            ),
            admissions=admissions,
            source_clusters=clusters,
            candidates=len(candidates),
            conflicts=conflicts,
            uses=_event_count(ctx, MEMORY_EVENT_USE),
            outcomes=_event_count(ctx, MEMORY_EVENT_USE_OUTCOME),
            query_key=off_fixture.request.stable_key(),
            query_before=CanonicalIdentity.from_value(
                off_fixture.request.stable_key()),
            query_after=CanonicalIdentity.from_value(
                on_fixture.request.stable_key()),
            resources_closed=(resources_closed
                              and ctx.work_memory.active_session_scope is None),
            result_identity=result_identity,
            observation_ref=ctx.f01_observation.event.object_ref,
            source=source,
        )
    finally:
        off_fixture.close()
        if on_fixture is not None:
            on_fixture.close()


def _run_clone_history_check(ctx, evidence: _MemoryPathEvidence):
    """在两个 V-06 clone 扰动上一 episode，并核验宿主零写与行为稳定。"""
    _refresh_projection(ctx)
    host_before = CanonicalIdentity.from_value(
        ctx.backend.recovery_state_snapshot())
    measurements = []
    for variant in (1, 2):
        with isolated_evaluation(
                ctx,
                label=f"f01-history-{variant}",
                ) as clone:
            clone.work_memory.end_session()
            _install_memory_ablation(clone, enabled=True)
            clone.work_memory.pr_vector[(52620,)] = variant
            measurements.append(_run_question_once(
                clone,
                evidence.source,
                _observation(clone, evidence.observation_ref),
                enabled=True,
            ))
    host_after = CanonicalIdentity.from_value(
        ctx.backend.recovery_state_snapshot())
    return measurements[0], measurements[1], host_before, host_after


def _run_rollback_check(ctx, evidence: _MemoryPathEvidence):
    """在真实 question 已写 Use 后注入异常，并比较完整恢复状态。"""
    _refresh_projection(ctx)
    with isolated_evaluation(ctx, label="f01-rollback") as clone:
        clone.work_memory.end_session()
        fixture, dialogue = _question_dialogue(
            clone,
            evidence.source,
            _observation(clone, evidence.observation_ref),
        )
        try:
            _, manifest = _post_weaning_manifest(clone, evidence.source)
            runtime = PostWeaningDryRunRuntime(clone, manifest)
            before = CanonicalIdentity.from_value(
                clone.backend.recovery_state_snapshot())
            original = dialogue.run

            def fail_after_question(request):
                """先完成真实问答，再模拟 F-01 调用边界故障。"""
                original(request)
                raise RuntimeError("F-01 rollback injection")

            dialogue.run = fail_after_question
            failed = False
            try:
                runtime.run_question(dialogue, fixture.request)
            except RuntimeError as exc:
                if str(exc) != "F-01 rollback injection":
                    raise
                failed = True
            after = CanonicalIdentity.from_value(
                clone.backend.recovery_state_snapshot())
            return failed, before, after
        finally:
            fixture.close()


def _run_cross_backend_migration(
        ctx,
        evidence: _MemoryPathEvidence,
        run_dir: Path,
        ):
    """把 Dict fresh 状态打包到 SQLite，并重跑同一 PW-00 question。"""
    from pure_integer_ai.cognition.shared.memory_batch import (
        install_memory_batch_runtimes,
    )

    projection = _refresh_projection(ctx)
    spaces = [
        row["space_id"]
        for row in ctx.backend.select("space", order_by="space_id")
    ]
    cursor = CursorState(
        base_run_id="f01-fixture",
        run_id="f01-migrate",
        completed={1},
        non_skippable={2},
    )
    dumped = dump_run(
        ctx.backend,
        str(run_dir),
        "f01-migrate",
        spaces=spaces,
        tables=None,
        require_all_spaces=True,
        versions=evidence.source.versions,
        cursor_state=cursor,
    )
    target_backend = SQLiteBackend(":memory:")
    fixture = None
    try:
        schema_ctx = make_train_context(target_backend, companion=True)
        install_memory_batch_runtimes(schema_ctx, _batch_config())
        loaded = load_run_package(
            target_backend,
            str(run_dir),
            "f01-migrate",
            expected_versions=evidence.source.versions,
            expected_dependencies=(),
            expected_publish_epoch=1,
        )
        restored_cursor = cursor_state_from_payload(
            loaded.cursor_payload,
            fallback_run_id="f01-migrate",
        )
        target, source, _, _, runtime = _restore_runtime(
            target_backend,
            projection.stable_key(),
        )
        observations = target.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_OBSERVATION,
            object_ref=evidence.observation_ref,
        )
        fixture, dialogue = _question_dialogue(
            target, source, observations[0])
        resumed = runtime.run_question(dialogue, fixture.request)
        question = resumed.result.question
        restored_identity = CanonicalIdentity.from_value((
            question.status,
            question.query.request.target,
            question.generation.rendered.stable_key(),
        ))
        passed = (
            dumped == spaces
            and bool(loaded.loaded_tables)
            and restored_cursor == cursor
            and len(observations) == 1
            and resumed.report.core_unchanged
            and resumed.report.query_closed
        )
        return passed, evidence.result_identity, restored_identity
    finally:
        if fixture is not None:
            fixture.close()
        target_backend.close()


class _ActualFacilityExercise:
    """把 PH1 主纵切、纠错、能力、恢复和 worker 证据汇成一次测量。"""

    def __init__(self, run_dir: Path) -> None:
        """绑定只供跨后端迁移包使用的测试目录。"""
        self.run_dir = run_dir

    def state_key(self) -> tuple[int, ...]:
        """返回不含临时路径噪声的固定 F-01 adapter 协议身份。"""
        return 1, 5290, 5, 12, 10, 4

    def prepare(self, eval_ctx) -> None:
        """在 Core 冻结前物化全部主纵切 Core fixture 和 runtime owner。"""
        _prepare_actual_fixture(eval_ctx)

    def run(self, eval_ctx) -> FacilityExerciseMeasurement:
        """执行全部真实 exercise，并把通过与失败信号来源化计数。"""
        pollution = int(
            eval_ctx.teacher is not None
            or any(
                item.expected_outcome is not None
                for item in eval_ctx.evaluation_plan.assignments
            )
        )
        path = _run_main_memory_path(eval_ctx)
        history_a, history_b, clone_before, clone_after = (
            _run_clone_history_check(eval_ctx, path))
        rollback_failed, rollback_before, rollback_after = (
            _run_rollback_check(eval_ctx, path))
        migrated, migrate_before, migrate_after = _run_cross_backend_migration(
            eval_ctx,
            path,
            self.run_dir,
        )

        a08 = _a08_world(DictBackend)
        try:
            a08_core_reader = CoreCanonicalStateReader(a08.ctx)
            a08_core_before = CanonicalIdentity.from_value(
                a08_core_reader.read())
            reparse = a08.runtime.apply(
                a08.request,
                raw_text=_A08_TEXT,
                license_id=_A08_LICENSE,
                batch_id=102,
                parser=_A08NewParser(a08.new_source),
            )
            event_count = len(a08.ctx.memory_read_events.query(access=_ACCESS))
            replay = a08.runtime.apply(
                a08.request,
                raw_text=_A08_TEXT,
                license_id=_A08_LICENSE,
                batch_id=102,
                parser=_A08RejectParser(),
            )
            replay_event_count = len(
                a08.ctx.memory_read_events.query(access=_ACCESS))
            a08_core_after = CanonicalIdentity.from_value(
                a08_core_reader.read())
        finally:
            a08.backend.close()

        capability = _capability_execution_setup(failing=False)
        (
            capability_backend,
            capability_ctx,
            _,
            _,
            execution,
            request,
            observation_ref,
            used_at,
            failed_at,
            _,
        ) = capability
        try:
            capability_core_reader = CoreCanonicalStateReader(capability_ctx)
            capability_core_before = CanonicalIdentity.from_value(
                capability_core_reader.read())
            capability_result = execution.execute_frontier(
                request,
                input_observation_ref=observation_ref,
                used_at=used_at,
                failed_at=failed_at,
            )
            capability_core_after = CanonicalIdentity.from_value(
                capability_core_reader.read())
        finally:
            if capability_ctx.work_memory.active_query_scope is not None:
                capability_ctx.work_memory.end_query()
            capability_backend.close()

        worker_one = _published_bytes(1)
        worker_two = _published_bytes(2)
        worker_four = _published_bytes(4)
        worker_before = CanonicalIdentity.from_value(worker_one)
        worker_after = CanonicalIdentity.from_value(worker_four)

        counters = tuple(sorted((
            FacilityCounter(
                _ACTUAL_METRICS[0], path.admissions, 2, (5291, 1)),
            FacilityCounter(
                _ACTUAL_METRICS[1], path.candidates, 1, (5291, 2)),
            FacilityCounter(
                _ACTUAL_METRICS[2], path.uses, 1, (5291, 3)),
            FacilityCounter(
                _ACTUAL_METRICS[3], path.outcomes, 1, (5291, 4)),
            FacilityCounter(
                _ACTUAL_METRICS[4], path.conflicts, 1, (5291, 5)),
            FacilityCounter(
                _ACTUAL_METRICS[5], len(reparse.new_hypothesis_refs),
                1, (5291, 6)),
            FacilityCounter(
                _ACTUAL_METRICS[6], len(reparse.preserved_use_refs),
                1, (5291, 7)),
            FacilityCounter(
                _ACTUAL_METRICS[7], int(
                    capability_result.binding_run.succeeded),
                1, (5291, 8)),
            FacilityCounter(
                _ACTUAL_METRICS[8], int(capability_result.use is not None),
                1, (5291, 9)),
            FacilityCounter(
                _ACTUAL_METRICS[9], int(rollback_failed),
                1, (5291, 10)),
            FacilityCounter(
                _ACTUAL_METRICS[10], 4, 4, (5291, 11)),
            FacilityCounter(
                _ACTUAL_METRICS[11],
                int(worker_one == worker_two)
                + int(worker_two == worker_four),
                2,
                (5291, 12),
            ),
            FacilityCounter(
                _ACTUAL_FORBIDDEN[0], 0, 1, (5292, 1)),
            FacilityCounter(
                _ACTUAL_FORBIDDEN[1], pollution, 1, (5292, 2)),
            FacilityCounter(
                _ACTUAL_FORBIDDEN[2],
                int(path.source_clusters != 1),
                2,
                (5292, 3),
            ),
            FacilityCounter(
                _ACTUAL_FORBIDDEN[3],
                int(history_a != history_b),
                2,
                (5292, 4),
            ),
        )))
        checks = tuple(sorted((
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[0],
                path.query_before == path.query_after,
                path.query_before,
                path.query_after,
                (5293, 1),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[1],
                path.resources_closed,
                CanonicalIdentity.from_value((1,)),
                CanonicalIdentity.from_value((1,)),
                (5293, 2),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[2],
                a08_core_before == a08_core_after,
                a08_core_before,
                a08_core_after,
                (5293, 3),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[3],
                replay.replayed and event_count == replay_event_count,
                CanonicalIdentity.from_value(event_count),
                CanonicalIdentity.from_value(replay_event_count),
                (5293, 4),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[4],
                capability_core_before == capability_core_after,
                capability_core_before,
                capability_core_after,
                (5293, 5),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[5],
                rollback_failed and rollback_before == rollback_after,
                rollback_before,
                rollback_after,
                (5293, 6),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[6],
                clone_before == clone_after,
                clone_before,
                clone_after,
                (5293, 7),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[7],
                migrated and migrate_before == migrate_after,
                migrate_before,
                migrate_after,
                (5293, 8),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[8],
                history_a == history_b,
                CanonicalIdentity.from_value(history_a),
                CanonicalIdentity.from_value(history_b),
                (5293, 9),
            ),
            FacilityIntegrityCheck(
                _ACTUAL_CHECKS[9],
                worker_one == worker_two == worker_four,
                worker_before,
                worker_after,
                (5293, 10),
            ),
        )))
        return FacilityExerciseMeasurement(
            _ACTUAL_EXERCISE,
            path.query_key,
            path.positive_behavior,
            path.negative_behavior,
            counters,
            checks,
            (5294, 1),
        )


def _actual_context():
    """构造只带 V-00 计划和 Companion 的 F-01 宿主上下文。"""
    plan, items = _complete_plan()
    ctx = make_train_context(DictBackend(), companion=True)
    ctx.evaluation_plan = plan
    ctx.evaluation_corpora = plan.partition(items).as_dict()
    ctx.evaluation_strictly_isolated = True
    return ctx


def test_f01_actual_assembly_closes_all_ph1_facility_dimensions(tmp_path):
    """真实 F-01 总装应逐维通过，同时保持宿主 Core/Memory 零写。"""
    ctx = _actual_context()
    before = ctx.backend.recovery_state_snapshot()
    protocol = _actual_protocol()
    runtime = FacilityReadinessRuntime(
        protocol,
        FacilityExerciseBinding(
            protocol.exercise_key,
            _ActualFacilityExercise(tmp_path / "f01-migrate"),
        ),
    )
    try:
        report = runtime.run(ctx)
        assert report.facility_complete is True
        assert report.exercise.core_unchanged is True
        assert all(item.passed for item in report.dimensions)
        assert all(item.passed for item in report.mechanisms)
        assert all(item.value == 0 for item in report.forbidden_counters)
        assert ctx.backend.recovery_state_snapshot() == before
        assert not hasattr(report, "mastered")
        assert not hasattr(report, "readiness")
    finally:
        ctx.backend.close()
