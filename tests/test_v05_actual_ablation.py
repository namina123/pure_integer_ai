"""V-05 对真实 PW-00、Memory Use、冲突和恢复链的反向消融。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE,
)
from pure_integer_ai.cognition.shared.memory_resolver import MemoryResolution
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProbeOutcome,
    ProtocolKey,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
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
from pure_integer_ai.storage.source_record import (
    SOURCE_RECORD_TABLE,
    SourceRecordRepository,
)

from tests.test_m06_memory_query import _close_query, _current, _open_query
from tests.test_m08_memory_use import _events
from tests.test_pw00_post_weaning_runtime import (
    _ACCESS,
    _batch_config,
    _build_runtime,
    _post_weaning_manifest,
    _question_dialogue,
    _restore_runtime,
    _query_source,
    _instruction,
)
from tests.test_v00_evaluation_protocol import _complete_plan


def _key(value: int) -> ProtocolKey:
    """构造测试使用的开放协议键。"""
    return ProtocolKey((value,))


def _identity(value: object) -> CanonicalIdentity:
    """把完整恢复状态转换为 V-05 可比较身份。"""
    return CanonicalIdentity.from_value(value)


def _close_outer_lifecycle(ctx) -> None:
    """关闭 dialogue 保留的 episode/document/session 外层生命周期。"""
    work = ctx.work_memory
    if work.active_generation_scope is not None:
        work.end_generation()
    if work.active_query_scope is not None:
        work.end_query()
    if work.active_episode_scope is not None:
        work.end_episode()
    if work.active_document_scope is not None:
        work.end_document()
    if work.active_session_scope is not None:
        work.end_session()


def _observation(ctx, observation_ref):
    """从恢复后的真实 Memory event 中读取唯一输入 Observation。"""
    matches = ctx.memory_interact_events.query(
        access=_ACCESS,
        event_kind=MEMORY_EVENT_OBSERVATION,
        object_ref=observation_ref,
    )
    if len(matches) != 1:
        raise AssertionError("V-05 fixture 缺少唯一 Observation")
    return matches[0]


def _install_memory_ablation(ctx, *, enabled: bool) -> None:
    """在当前 clone 的 K-04 resolve 边界安装可延迟启用的 Memory OFF。"""
    ctx.v05_memory_ablation_active = False
    if enabled:
        return
    runtime = ctx.memory_hot_set_runtime
    original = runtime.resolve

    def resolve(compilation):
        """破坏激活时保留真实读取但移除全部 Memory 候选。"""
        result = original(compilation)
        if not ctx.v05_memory_ablation_active:
            return result
        return MemoryResolution(
            result.compilation,
            tuple(
                replace(candidate_set, candidates=())
                for candidate_set in result.sets
            ),
        )

    runtime.resolve = resolve


class _ConditionalDropMapper:
    """保留 A-10 mapper 身份，但在破坏臂拒绝形成 activation。"""

    def __init__(self, ctx, delegate) -> None:
        """绑定当前 clone 和原始 mapper。"""
        self.ctx = ctx
        self.delegate = delegate

    def project(self, request, candidate, obligations):
        """健康臂委托真实 mapper，破坏臂返回空 agenda。"""
        if self.ctx.v05_component_break_active:
            return ()
        return self.delegate.project(request, candidate, obligations)

    def state_key(self) -> tuple[int, ...]:
        """返回包装协议和原 mapper 的完整身份。"""
        delegate = self.delegate.state_key()
        return 1, 20560, len(delegate), *delegate


class _ComponentIntervention:
    """对 query、overlay、A-10、Use 和来源边界实施实际破坏。"""

    _KINDS = {
        "compiler": 1,
        "overlay": 2,
        "attractor": 3,
        "agenda": 4,
        "use": 5,
        "source": 6,
        "current-query": 7,
    }

    def __init__(self, kind: str) -> None:
        """绑定一种互斥的实际破坏，不把测试开关放入生产模块。"""
        if kind not in self._KINDS:
            raise ValueError("未知 V-05 实际破坏")
        self.kind = kind

    def state_key(self) -> tuple[int, ...]:
        """返回实际破坏种类的稳定身份。"""
        return 1, 20561, self._KINDS[self.kind]

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """在当前 V-06 clone 安装延迟破坏，fixture 准备期保持健康。"""
        eval_ctx.v05_component_break_active = False
        if self.kind in {"compiler", "current-query"}:
            self._install_compiler(eval_ctx)
        elif self.kind == "overlay":
            self._install_overlay(eval_ctx)
        elif self.kind == "attractor":
            eval_ctx.attractor_runtime.mapper = _ConditionalDropMapper(
                eval_ctx,
                eval_ctx.attractor_runtime.mapper,
            )
        elif self.kind == "agenda":
            self._install_agenda(eval_ctx)
        elif self.kind == "use":
            self._install_use(eval_ctx)
        elif self.kind == "source":
            self._install_source_trust(eval_ctx)
        if enabled:
            eval_ctx.v05_component_break_active = False

    def _install_compiler(self, ctx) -> None:
        """关闭 request 形成，或记录被打乱的完整当前 query 后拒绝。"""
        runtime = ctx.memory_query_runtime
        original = runtime.compile

        def compile(current, *, access):
            """根据注入种类形成空 compilation 或拒绝漂移 query。"""
            if not ctx.v05_component_break_active:
                return original(current, access=access)
            if self.kind == "current-query":
                drifted = replace(
                    current,
                    task=_instruction(current.source, 20562),
                )
                ctx.v05_last_query_key = current.stable_key()
                ctx.v05_last_query_binding_checks = 0
                ctx.v05_drifted_query_key = drifted.stable_key()
                raise ValueError("V-05 current query 被打乱")
            return replace(
                original(current, access=access),
                requests=(),
            )

        runtime.compile = compile

    @staticmethod
    def _install_overlay(ctx) -> None:
        """保留 M-07 实际读取和 request，但移除 overlay 仲裁结果。"""
        runtime = ctx.memory_resolver_runtime
        original = runtime.resolve

        def resolve(compilation):
            """破坏臂返回同 request 的空候选集合。"""
            result = original(compilation)
            if not ctx.v05_component_break_active:
                return result
            return MemoryResolution(
                result.compilation,
                tuple(
                    replace(candidate_set, candidates=())
                    for candidate_set in result.sets
                ),
            )

        runtime.resolve = resolve

    @staticmethod
    def _install_agenda(ctx) -> None:
        """保留真实 frontier，关闭 selection 后的 consumed 提交。"""
        runtime = ctx.attractor_runtime
        original = runtime.resolve_and_activate

        def resolve_and_activate(compilation, obligations):
            """破坏臂让 frontier 存在但 consumption 明确失败。"""
            state = original(compilation, obligations)
            if ctx.v05_component_break_active:
                def blocked(_decision):
                    """拒绝把任何 frontier 项伪装成 consumed。"""
                    raise RuntimeError("V-05 agenda consumer disabled")

                state.commit_consumption = blocked
            return state

        runtime.resolve_and_activate = resolve_and_activate

    @staticmethod
    def _install_use(ctx) -> None:
        """关闭 M-08 的实际 Use writer，使 processing 不能伪装采用。"""
        runtime = ctx.memory_use_runtime
        original = runtime.record_selection_use

        def record_selection_use(*args, **kwargs):
            """破坏臂拒绝写 Use，健康臂调用真实 M-08。"""
            if ctx.v05_component_break_active:
                raise RuntimeError("V-05 Memory Use disabled")
            return original(*args, **kwargs)

        runtime.record_selection_use = record_selection_use

    @staticmethod
    def _install_source_trust(ctx) -> None:
        """只在破坏臂隐藏 SourceRecord，使来源信任检查 fail closed。"""
        ctx.v05_source_trust_break = True
        backend = ctx.backend
        original = backend.select

        def select(table, *args, **kwargs):
            """对 SourceRecord 查询返回空，其余真实 backend 读取不变。"""
            if (ctx.v05_component_break_active
                    and table == SOURCE_RECORD_TABLE):
                return []
            return original(table, *args, **kwargs)

        backend.select = select


def _audit_current_resolution(ctx, source, scope):
    """读取当前 query 身份和实际冲突 aggregate，并关闭临时 reader。"""
    ctx.work_memory.begin_query(scope)
    try:
        current = _current(ctx, source, scope)
        compilation = ctx.memory_query_runtime.compile(
            current,
            access=_ACCESS,
        )
        ctx.v05_last_query_key = compilation.current.stable_key()
        ctx.v05_last_query_binding_checks = 1
        resolution = ctx.memory_resolver_runtime.resolve(compilation)
        conflicts = sum(
            candidate.aggregate is not None
            and candidate.aggregate.support_count > 0
            and candidate.aggregate.contradict_count > 0
            for candidate_set in resolution.sets
            for candidate in candidate_set.candidates
        )
        ctx.v05_last_conflict_checks = conflicts
        return compilation.current.stable_key(), conflicts
    finally:
        ctx.work_memory.end_query()


def _run_question_once(ctx, source, observation, *, enabled: bool):
    """在同一恢复状态执行一次真实 J-G 问答并返回分账测量。"""
    ctx.v05_memory_ablation_active = False
    if hasattr(ctx, "v05_component_break_active"):
        ctx.v05_component_break_active = False
    fixture, dialogue = _question_dialogue(ctx, source, observation)
    try:
        ctx.v05_memory_ablation_active = not enabled
        if hasattr(ctx, "v05_component_break_active"):
            ctx.v05_component_break_active = not enabled
        if (not enabled
                and getattr(ctx, "v05_source_trust_break", False)):
            dialogue.source_records.clear_runtime_caches()
        query_key, conflicts = _audit_current_resolution(
            ctx,
            source,
            fixture.request.response_scope,
        )
        _, manifest = _post_weaning_manifest(ctx, source)
        runtime = PostWeaningDryRunRuntime(ctx, manifest)
        uses_before = len(_events(ctx, MEMORY_EVENT_USE))
        operation = runtime.run_question(dialogue, fixture.request)
        uses_after = len(_events(ctx, MEMORY_EVENT_USE))
        commits = operation.result.question.selection_commit.commits
        sources = operation.result.sources
        repository = SourceRecordRepository(ctx.backend)
        trusted = sum(
            repository.read(item.source_record_hash).metadata_complete
            for item in sources
        )
        behavior = 100 if commits and sources else 0
        return (
            behavior,
            uses_after - uses_before,
            len(commits),
            trusted,
            conflicts,
            query_key,
            getattr(ctx, "v05_last_query_binding_checks", 1),
        )
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)


def _run_rollback_trial(ctx, source, observation, *, enabled: bool):
    """在真实 dialogue 返回后注入故障，并核验 PW-00 撤销全部写入。"""
    ctx.v05_memory_ablation_active = False
    if hasattr(ctx, "v05_component_break_active"):
        ctx.v05_component_break_active = False
    fixture, dialogue = _question_dialogue(ctx, source, observation)
    try:
        ctx.v05_memory_ablation_active = not enabled
        if hasattr(ctx, "v05_component_break_active"):
            ctx.v05_component_break_active = not enabled
        _, manifest = _post_weaning_manifest(ctx, source)
        runtime = PostWeaningDryRunRuntime(ctx, manifest)
        before = _identity(ctx.backend.recovery_state_snapshot())
        original = dialogue.run

        def fail_after_question(request):
            """先完成真实问答和可能的 Use，再模拟调用边界故障。"""
            original(request)
            raise RuntimeError("v05 rollback trial")

        dialogue.run = fail_after_question
        with pytest.raises(RuntimeError, match="rollback trial"):
            runtime.run_question(dialogue, fixture.request)
        after = _identity(ctx.backend.recovery_state_snapshot())
        return before, after
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)


def _run_restored_history(
        recovery_state,
        projection_key,
        observation_ref,
        *,
        enabled: bool,
        variant: int,
        ):
    """从同一长期快照恢复，扰动上一 episode 状态后执行当前问题。"""
    backend = DictBackend()
    try:
        from pure_integer_ai.cognition.shared.memory_batch import (
            install_memory_batch_runtimes,
        )
        schema_ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(schema_ctx, _batch_config())
        backend.restore_recovery_state(recovery_state)
        ctx, source, _, _, _ = _restore_runtime(backend, projection_key)
        _install_memory_ablation(ctx, enabled=enabled)
        ctx.work_memory.pr_vector[(20500,)] = variant
        measured = _run_question_once(
            ctx,
            source,
            _observation(ctx, observation_ref),
            enabled=enabled,
        )
        return measured
    finally:
        backend.close()


class _StateReader:
    """读取 V-05 宿主的完整后端和 WorkMemory 生命周期。"""

    def state_key(self) -> tuple[int, ...]:
        """返回状态读取协议身份。"""
        return 1, 20510

    def read(self, ctx):
        """返回可发现宿主写入或生命周期泄漏的完整状态。"""
        work = ctx.work_memory
        return (
            ctx.backend.recovery_state_snapshot(),
            work.active_session_scope,
            work.active_document_scope,
            work.active_episode_scope,
            work.active_query_scope,
            work.active_generation_scope,
            work.attractor_state,
            tuple(sorted(work.pr_vector.items())),
        )


class _MemoryIntervention:
    """关闭 V-06 clone 内 K-04 返回的真实 Memory 候选。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 Memory 消融实现身份。"""
        return 1, 20520

    def apply(self, eval_ctx, *, enabled: bool) -> None:
        """安装只作用于当前 clone 的延迟 Memory 消融。"""
        _install_memory_ablation(eval_ctx, enabled=enabled)


class _ActualEvaluator:
    """执行真实 PW-00 问答、回滚、恢复和历史扰动测量。"""

    def __init__(self, observation_ref, projection_key) -> None:
        """绑定恢复后仍可查验的 Observation 与 K-04 manifest。"""
        self.observation_ref = observation_ref
        self.projection_key = projection_key

    def state_key(self) -> tuple[int, ...]:
        """返回实际 evaluator 的输入和恢复协议身份。"""
        observation = self.observation_ref.stable_key()
        return (
            1,
            20530,
            len(observation),
            *observation,
            len(self.projection_key),
            *self.projection_key,
        )

    def evaluate(self, eval_ctx, _item, request):
        """从实际事件、processing、来源和恢复状态形成 V-05 测量。"""
        if eval_ctx.work_memory.active_session_scope is not None:
            eval_ctx.work_memory.end_session()
        source = _query_source(document_id=1)
        observation = _observation(eval_ctx, self.observation_ref)
        rollback_before, rollback_after = _run_rollback_trial(
            eval_ctx,
            source,
            observation,
            enabled=request.memory_enabled,
        )
        recovery_state = eval_ctx.backend.recovery_state_snapshot()
        recovery_before = _identity(recovery_state)
        restored = DictBackend()
        try:
            from pure_integer_ai.cognition.shared.memory_batch import (
                install_memory_batch_runtimes,
            )
            schema_ctx = make_train_context(restored, companion=True)
            install_memory_batch_runtimes(schema_ctx, _batch_config())
            restored.restore_recovery_state(recovery_state)
            recovery_after = _identity(restored.recovery_state_snapshot())
        finally:
            restored.close()
        trials = []
        for ordinal, variant in enumerate((11, 29), start=1):
            measured = _run_restored_history(
                recovery_state,
                self.projection_key,
                self.observation_ref,
                enabled=request.memory_enabled,
                variant=variant,
            )
            trials.append(PostWeaningHistoryTrial(
                _key(20540 + ordinal),
                measured[0],
                measured[1],
                measured[5],
            ))
        measured = _run_question_once(
            eval_ctx,
            source,
            observation,
            enabled=request.memory_enabled,
        )
        work = eval_ctx.work_memory
        return PostWeaningProbeMeasurement(
            ProbeOutcome(
                measured[0] > 0,
                value=measured[0],
                sample_count=1,
            ),
            memory_use_count=measured[1],
            attractor_consumption_count=measured[2],
            current_query_binding_checks=measured[6],
            source_trust_checks=measured[3],
            conflict_checks=measured[4],
            current_query_key=measured[5],
            work_memory_closed=(
                work.active_query_scope is None
                and work.active_generation_scope is None
                and work.attractor_state is None
            ),
            query_resources_closed=(
                eval_ctx.memory_hot_set_runtime.query_resources_closed()
            ),
            rollback_before=rollback_before,
            rollback_after=rollback_after,
            recovery_before=recovery_before,
            recovery_after=recovery_after,
            history_trials=tuple(trials),
        )


class _ActualBreakEvaluator:
    """对一个实际破坏点执行主问答并按完整证据判定行为。"""

    def __init__(self, observation_ref, expected_query_key) -> None:
        """绑定真实 Observation 和预注册 query 身份。"""
        self.observation_ref = observation_ref
        self.expected_query_key = expected_query_key

    def state_key(self) -> tuple[int, ...]:
        """返回实际破坏 evaluator 的完整输入身份。"""
        observation = self.observation_ref.stable_key()
        return (
            1,
            20570,
            len(observation),
            *observation,
            len(self.expected_query_key),
            *self.expected_query_key,
        )

    def evaluate(self, eval_ctx, _item, request):
        """执行实际组件链，异常由 PW-00 回滚后记为明确行为失败。"""
        if eval_ctx.work_memory.active_session_scope is not None:
            eval_ctx.work_memory.end_session()
        source = _query_source(document_id=1)
        observation = _observation(eval_ctx, self.observation_ref)
        uses_before = len(_events(eval_ctx, MEMORY_EVENT_USE))
        try:
            measured = _run_question_once(
                eval_ctx,
                source,
                observation,
                enabled=request.memory_enabled,
            )
        except (RuntimeError, ValueError):
            measured = (
                0,
                len(_events(eval_ctx, MEMORY_EVENT_USE)) - uses_before,
                0,
                0,
                getattr(eval_ctx, "v05_last_conflict_checks", 0),
                getattr(
                    eval_ctx,
                    "v05_last_query_key",
                    self.expected_query_key,
                ),
                getattr(eval_ctx, "v05_last_query_binding_checks", 0),
            )
        passed = (
            measured[0] > 0
            and measured[1] > 0
            and measured[2] > 0
            and measured[3] > 0
            and measured[4] > 0
            and measured[5] == self.expected_query_key
            and measured[6] > 0
        )
        score = measured[0] if passed else 0
        state = _identity(eval_ctx.backend.recovery_state_snapshot())
        trials = tuple(
            PostWeaningHistoryTrial(
                _key(20571 + index),
                score,
                measured[1],
                measured[5],
            )
            for index in range(2)
        )
        work = eval_ctx.work_memory
        return PostWeaningProbeMeasurement(
            ProbeOutcome(passed, value=score, sample_count=1),
            memory_use_count=measured[1],
            attractor_consumption_count=measured[2],
            current_query_binding_checks=measured[6],
            source_trust_checks=measured[3],
            conflict_checks=measured[4],
            current_query_key=measured[5],
            work_memory_closed=(
                work.active_query_scope is None
                and work.active_generation_scope is None
                and work.attractor_state is None
            ),
            query_resources_closed=(
                eval_ctx.memory_hot_set_runtime.query_resources_closed()
            ),
            rollback_before=state,
            rollback_after=state,
            recovery_before=state,
            recovery_after=state,
            history_trials=trials,
        )


def _run_actual_component_case(kind: str):
    """把一种实际组件破坏接入标准 V-05 ON/OFF runner。"""
    backend, ctx, _, _, _, observation, _projection = _build_runtime()
    try:
        source = _query_source(document_id=1)
        scope = _open_query(ctx, source)
        expected_query_key = _current(ctx, source, scope).stable_key()
        _close_query(ctx)
        plan, items = _complete_plan(full_coverage=False)
        assignment = plan.assignments[3]
        dimension = plan.protocol.required_dimensions[0]
        track_key = _key(20580)
        case_key = _key(20581)
        evaluator_key = _key(20582)
        intervention_key = _key(20583)
        metric_key = _key(20584)
        protocol = PostWeaningValidationProtocol(
            version=4,
            cases=(PostWeaningAblationCase(
                case_key,
                track_key,
                intervention_key,
                evaluator_key,
                assignment.identity,
                dimension,
                expected_query_key,
                50,
            ),),
            required_case_keys=(case_key,),
            tracks=(PostWeaningTrackRequirement(
                track_key,
                (dimension,),
                (PostWeaningResourceBound(
                    metric_key,
                    maximum_value=64,
                ),),
                consecutive_windows=2,
                checkpoint_step=1,
            ),),
            required_track_keys=(track_key,),
        )
        runtime = PostWeaningValidationRuntime(
            protocol,
            PostWeaningEvaluatorRegistry((PostWeaningEvaluatorBinding(
                evaluator_key,
                _ActualBreakEvaluator(
                    observation.event.object_ref,
                    expected_query_key,
                ),
            ),)),
            PostWeaningInterventionRegistry((
                PostWeaningInterventionBinding(
                    intervention_key,
                    _ComponentIntervention(kind),
                ),
            )),
            _StateReader(),
        )
        partition = plan.partition(items)
        ctx.evaluation_plan = plan
        ctx.evaluation_corpora = partition.as_dict()
        ctx.evaluation_strictly_isolated = True
        before = backend.recovery_state_snapshot()
        report = runtime.run(ctx, PostWeaningValidationRequest(
            track_key,
            checkpoint=1,
            resources=(PostWeaningResourceMeasurement(
                metric_key,
                _key(20585),
                value=8,
                sample_count=1,
            ),),
        ))
        assert backend.recovery_state_snapshot() == before
        return report.ablations[0]
    finally:
        backend.close()


def test_v05_actual_memory_on_off_requires_use_conflict_and_recovery():
    """真实 Memory OFF 必须失去回答、Use 和 consumption，ON 才完整通过。"""
    backend, ctx, _, _, _, observation, projection = _build_runtime()
    try:
        source = _query_source(document_id=1)
        scope = _open_query(ctx, source)
        expected_query_key = _current(ctx, source, scope).stable_key()
        _close_query(ctx)
        plan, items = _complete_plan(full_coverage=False)
        assignment = plan.assignments[3]
        dimension = plan.protocol.required_dimensions[0]
        track_key = _key(20550)
        case_key = _key(20551)
        evaluator_key = _key(20552)
        intervention_key = _key(20553)
        metric_key = _key(20554)
        case = PostWeaningAblationCase(
            case_key,
            track_key,
            intervention_key,
            evaluator_key,
            assignment.identity,
            dimension,
            expected_query_key,
            50,
        )
        protocol = PostWeaningValidationProtocol(
            version=3,
            cases=(case,),
            required_case_keys=(case_key,),
            tracks=(PostWeaningTrackRequirement(
                track_key,
                (dimension,),
                (PostWeaningResourceBound(
                    metric_key,
                    maximum_value=64,
                ),),
                consecutive_windows=2,
                checkpoint_step=1,
            ),),
            required_track_keys=(track_key,),
        )
        runtime = PostWeaningValidationRuntime(
            protocol,
            PostWeaningEvaluatorRegistry((PostWeaningEvaluatorBinding(
                evaluator_key,
                _ActualEvaluator(
                    observation.event.object_ref,
                    projection.stable_key(),
                ),
            ),)),
            PostWeaningInterventionRegistry((
                PostWeaningInterventionBinding(
                    intervention_key,
                    _MemoryIntervention(),
                ),
            )),
            _StateReader(),
        )
        partition = plan.partition(items)
        ctx.evaluation_plan = plan
        ctx.evaluation_corpora = partition.as_dict()
        ctx.evaluation_strictly_isolated = True
        before = backend.recovery_state_snapshot()

        report = runtime.run(ctx, PostWeaningValidationRequest(
            track_key,
            checkpoint=1,
            resources=(PostWeaningResourceMeasurement(
                metric_key,
                _key(20555),
                value=8,
                sample_count=1,
            ),),
        ))

        pair = report.ablations[0]
        assert pair.complete
        assert pair.enabled.measurement.memory_use_count > 0
        assert pair.enabled.measurement.attractor_consumption_count > 0
        assert pair.enabled.measurement.current_query_binding_checks > 0
        assert pair.enabled.measurement.source_trust_checks > 0
        assert pair.enabled.measurement.conflict_checks > 0
        assert pair.disabled.measurement.memory_use_count == 0
        assert pair.disabled.measurement.attractor_consumption_count == 0
        assert report.ablations_complete
        assert not report.stop_allowed
        assert backend.recovery_state_snapshot() == before
    finally:
        backend.close()


@pytest.mark.parametrize("kind", (
    "compiler",
    "overlay",
    "attractor",
    "agenda",
    "use",
    "source",
    "current-query",
))
def test_v05_actual_component_breaks_fail_the_target_metric(kind):
    """关闭任一实际承重组件后不得保留同一完整行为 PASS。"""
    pair = _run_actual_component_case(kind)

    assert pair.enabled.observation.outcome.passed is True
    assert pair.disabled.observation.outcome.passed is False
    assert pair.enabled.measurement.memory_use_count > 0
    assert pair.enabled.measurement.attractor_consumption_count > 0
    assert pair.enabled.measurement.current_query_binding_checks > 0
    assert pair.enabled.measurement.source_trust_checks > 0
    assert pair.enabled.measurement.conflict_checks > 0
    assert pair.complete
