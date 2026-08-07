"""PW-00 独立 dry-run runtime 的启动、分型入口和 Core 只读对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.post_weaning import (
    POST_WEANING_OPERATION_COMMITTED,
    PostWeaningFacilityCheck,
    PostWeaningFacilityProbe,
    PostWeaningIntakeRequest,
    PostWeaningResourceBudget,
    PostWeaningRouteProtocol,
)
from pure_integer_ai.cognition.shared.source_trust import (
    SOURCE_ADMISSION_ACCEPTED,
    SourceTrustAssessment,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.attractor_runtime import install_attractor_runtime
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    install_memory_hot_set_runtime,
)
from pure_integer_ai.experiments.memory_generation_runtime import (
    MemoryAwareQuestionDialogueRuntime,
    MemoryQuestionSelectionCommitter,
    ResolvedMemoryQuestionExecutor,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
    PostWeaningRuntimeError,
    PostWeaningStartupError,
    build_post_weaning_dry_run_manifest,
)
from pure_integer_ai.experiments.source_trust_runtime import (
    install_source_admission_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_repository import (
    SEGMENT_OBJECT_PART_TABLE,
)
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_from_payload,
    dump_run,
    load_run_package,
)
from tests.test_a10_attractor_state import (
    _attractor_protocol,
    _GoalMapper,
    _goals,
    _SupersedeChanged,
)
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import _fixture as _question_fixture
from tests.test_g05_memory_generation_evidence import _complete_source
from tests.test_k04_memory_hot_set import (
    _batch_config,
    _core_refs,
    _install_resolver,
    _prefetch,
    _publish_projection,
    _query_source,
    _seed_memory,
)
from tests.test_m06_memory_query import _current, _open_query
from tests.test_m08_memory_use import _append_observation
from pure_integer_ai.cognition.shared.attractor_state import AttractorBudget
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.write_guard import RuntimeWriteGuardError


_ACCESS = MemoryAccessContext(1, 2, 3)


def _source(
        source_id: int,
        *,
        interaction: bool = False,
        ) -> SourceRef:
    """构造阅读或 session 交互所需的来源身份。"""
    owner = (
        OwnerScope(1, 2, 3, VISIBILITY_SESSION)
        if interaction else OwnerScope()
    )
    return SourceRef(
        71,
        source_id,
        source_id,
        owner,
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(1),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


def _instruction(source: SourceRef, value: int) -> ObjectIdentity:
    """构造测试注入的一等最小指令。"""
    return ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION,
        (value,),
        source.owner,
        source.versions,
    )


class _Parser:
    """把来源切片转换为一个 Memory 候选，不写 Core。"""

    def __init__(self, source: SourceRef, candidate: int) -> None:
        """绑定来源与候选身份。"""
        self.source = source
        self.candidate = candidate

    def parse(self, source_slice):
        """核验当前来源后返回来源化 Observation/Hypothesis 草案。"""
        if source_slice.source != self.source:
            raise ValueError("parser 收到其他来源")
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (1000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(_instruction(
            self.source, 2000 + self.candidate))
        return ObservationIntakeDraft(
            (3000 + self.candidate,),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (4000 + self.candidate,),
                (5000 + self.candidate,),
                (6000 + self.candidate,),
                (7000 + self.candidate,),
                1,
                signal_ref=signal,
            ),),
        )


class _FixtureSourceTrustPolicy:
    """只供 PW-00 受控夹具使用的显式宽准入 policy。"""

    def __init__(self, refs) -> None:
        """绑定来源 kind、许可、信任、时效和理由图身份。"""
        self.refs = refs

    def state_key(self):
        """返回全部图身份组成的固定 policy 状态。"""
        result = [1, len(self.refs)]
        for ref in self.refs:
            key = ref.stable_key()
            result.extend((len(key), *key))
        return tuple(result)

    def assess(self, request):
        """按 SourceRef 来源簇接受受控请求，不解释原文或许可词面。"""
        return SourceTrustAssessment(
            request.stable_key(),
            self.state_key(),
            SOURCE_ADMISSION_ACCEPTED,
            (request.source.source_kind, request.source.source_id),
            (
                request.source.versions.corpus.value,
                request.source.versions.parser.value,
            ),
            self.refs[0],
            self.refs[1],
            self.refs[2],
            (self.refs[4],),
            (),
            (self.refs[3].local_id,),
        )

    def clone_for_context(self, ctx):
        """核验克隆图仍可回读相同 refs，并复用只读 policy。"""
        for ref in self.refs:
            ctx.graph_ontology.identity_of(ref)
        return self


def _install_source_trust(ctx, routes):
    """为 PW-00 fixture 物化 policy 概念并安装 A-05 runtime。"""
    if ctx.source_trust_runtime is not None:
        return ctx.source_trust_runtime
    from pure_integer_ai.cognition.shared.graph_ontology import (
        relation_concept_identity,
    )
    refs = tuple(
        ctx.graph_ontology.materialize(relation_concept_identity((value,)))
        for value in range(20201, 20206)
    )
    return install_source_admission_runtime(
        ctx,
        _FixtureSourceTrustPolicy(refs),
        reading_route=routes.reading,
        interaction_route=routes.interaction,
        record_only_routes=(routes.external_define,),
    )


def _post_weaning_protocol(source: SourceRef):
    """构造可跨 fresh/restart 重建的入口协议和逐维设施探针。"""
    routes = PostWeaningRouteProtocol(*tuple(
        _instruction(source, value)
        for value in range(20101, 20105)
    ))
    checks = tuple(sorted((
        PostWeaningFacilityCheck(
            _instruction(source, 20110),
            True,
            (20111, 1),
        ),
        PostWeaningFacilityCheck(
            _instruction(source, 20112),
            True,
            (20113, 1),
        ),
    ), key=lambda item: item.requirement.stable_key()))
    return routes, PostWeaningFacilityProbe(checks, (20114, 1))


def _post_weaning_manifest(ctx, source: SourceRef):
    """从当前 K/M/A owner 和固定 fixture 身份形成可重建 dry-run manifest。"""
    routes, probe = _post_weaning_protocol(source)
    _install_source_trust(ctx, routes)
    manifest = build_post_weaning_dry_run_manifest(
        ctx,
        runtime_owner=_instruction(source, 20115),
        fixture_artifact_key=(20116, 1),
        routes=routes,
        probe=probe,
        budget=PostWeaningResourceBudget(8, 64, 1, 1),
        trace=(20117, 1),
    )
    return routes, manifest


def _install_post_weaning_consumers(ctx, query_source, projection):
    """在已安装 resolver 的上下文上重装 K-04、A-10 和 M-08 消费链。"""
    install_memory_hot_set_runtime(
        ctx,
        projection,
        QueryHotSetPolicy(
            SegmentBudget(4, 4_000_000),
            SegmentBudget(1, 1_000_000),
            _prefetch(False),
            8,
        ),
    )
    install_attractor_runtime(
        ctx,
        _attractor_protocol(query_source),
        AttractorBudget(2, 4, 4),
        _GoalMapper(prefer_matching_document=False),
        _SupersedeChanged(),
    )
    install_memory_use_runtime(ctx)


def _build_runtime(backend=None):
    """在调用方后端装配 PW-00 的真实 M/K/A owner、投影和 dry-run manifest。"""
    backend = backend or DictBackend()
    ctx = make_train_context(backend, companion=True)
    from pure_integer_ai.cognition.shared.memory_batch import (
        install_memory_batch_runtimes,
    )
    install_memory_batch_runtimes(ctx, _batch_config())
    _seed_memory(ctx)
    query_source = _query_source(document_id=1)
    core_refs = _core_refs(ctx)
    _, resolver_runtime = _install_resolver(
        ctx,
        query_source,
        core_refs[1],
    )
    observation = _append_observation(ctx, query_source, core_refs)
    projection = _publish_projection(ctx, resolver_runtime.resolver)
    _install_post_weaning_consumers(ctx, query_source, projection)
    routes, manifest = _post_weaning_manifest(ctx, query_source)
    return (
        backend,
        ctx,
        routes,
        manifest,
        PostWeaningDryRunRuntime(ctx, manifest),
        observation,
        projection,
    )


def _restore_runtime(backend, projection_key):
    """从已加载后端和 K-04 manifest 重装同一 PW-00 消费链。"""
    from pure_integer_ai.cognition.shared.memory_batch import (
        install_memory_batch_runtimes,
    )
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _batch_config())
    source = _query_source(document_id=1)
    _, resolver = _install_resolver(ctx, source, _core_refs(ctx)[1])
    projection = MemoryCandidateProjectionManifest.from_stable_key(
        projection_key)
    projection.validate_store(ctx.tiered_segment_store)
    _install_post_weaning_consumers(ctx, source, projection)
    routes, manifest = _post_weaning_manifest(ctx, source)
    return ctx, source, routes, manifest, PostWeaningDryRunRuntime(
        ctx, manifest)


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00_fresh_routes_intakes_without_core_change(backend_type):
    """双后端 fresh 都应分流三类摄入，并逐次证明 Core bit-identical。"""
    backend, ctx, routes, _, runtime, _, _ = _build_runtime(backend_type())
    try:
        reading = _source(101)
        first = runtime.run_intake(PostWeaningIntakeRequest(
            routes.reading,
            reading,
            "阅读来源",
            "license-read",
            301,
            parser=_Parser(reading, 1),
            trace=(20120, 1),
        ))
        interaction = _source(102, interaction=True)
        second = runtime.run_intake(PostWeaningIntakeRequest(
            routes.interaction,
            interaction,
            "交互来源",
            "license-interaction",
            302,
            parser=_Parser(interaction, 2),
            trace=(20120, 2),
        ))
        defined = _source(103)
        third = runtime.run_intake(PostWeaningIntakeRequest(
            routes.external_define,
            defined,
            "外部定义",
            "license-define",
            303,
            trace=(20120, 3),
        ))

        reports = runtime.reports()
        assert (first.report, second.report, third.report) == reports
        assert all(
            item.status == POST_WEANING_OPERATION_COMMITTED
            and item.core_unchanged
            and item.query_closed
            for item in reports
        )
        assert first.result.observation_ref is not None
        assert second.result.observation_ref is not None
        assert third.result.source_key == defined.stable_key()
        assert ctx.memory_read_events.query(access=_ACCESS)
        session_access = MemoryAccessContext(1, 2, 3)
        assert ctx.memory_interact_events.query(access=session_access)
        assert ctx.weaning_phase == 0
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00_rejects_failed_probe_and_formal_start(backend_type):
    """未通过设施维度和缺少 PW-00A verifier 都不得启动正式或 dry-run 状态。"""
    backend, ctx, routes, manifest, _, _, _ = _build_runtime(backend_type())
    try:
        failed_check = PostWeaningFacilityCheck(
            manifest.probe.checks[0].requirement,
            False,
            (20130, 1),
        )
        failed_probe = PostWeaningFacilityProbe(
            tuple(sorted(
                (failed_check, manifest.probe.checks[1]),
                key=lambda item: item.requirement.stable_key(),
            )),
            (20131, 1),
        )
        failed_manifest = build_post_weaning_dry_run_manifest(
            ctx,
            runtime_owner=manifest.runtime_owner,
            fixture_artifact_key=(20132, 1),
            routes=routes,
            probe=failed_probe,
            budget=manifest.budget,
            trace=(20133, 1),
        )

        with pytest.raises(PostWeaningStartupError, match="未通过"):
            PostWeaningDryRunRuntime(ctx, failed_manifest)
        with pytest.raises(PostWeaningStartupError, match="PW-00A"):
            PostWeaningDryRunRuntime.start_formal(ctx, manifest)
        assert ctx.weaning_phase == 0
    finally:
        backend.close()


def test_pw00_rejects_budget_below_bound_m10_unit_limit():
    """PW-00 启动预算若不能覆盖 M-10 单元上限，必须在任何写入前拒绝。"""
    backend, ctx, routes, manifest, _, _, _ = _build_runtime()
    try:
        under_budgeted = build_post_weaning_dry_run_manifest(
            ctx,
            runtime_owner=manifest.runtime_owner,
            fixture_artifact_key=(20134, 1),
            routes=routes,
            probe=manifest.probe,
            budget=PostWeaningResourceBudget(8, 63, 1, 1),
            trace=(20135, 1),
        )
        before = backend.recovery_state_snapshot()

        with pytest.raises(PostWeaningStartupError, match="M-10"):
            PostWeaningDryRunRuntime(ctx, under_budgeted)

        assert backend.recovery_state_snapshot() == before
    finally:
        backend.close()


def _question_dialogue(ctx, source, observation):
    """在同一 session/document/episode 下装配走 K-04 的完整 J-G question caller。"""
    scope = _open_query(ctx, source)
    current = _current(ctx, source, scope)
    compilation = ctx.memory_query_runtime.compile(current, access=_ACCESS)
    resolution = ctx.memory_resolver_runtime.resolve(compilation)
    repository = SourceRecordRepository(ctx.backend)
    traces = {
        trace.source.stable_key(): trace
        for candidate_set in resolution.sets
        for candidate in candidate_set.candidates
        for trace in candidate.memory_source_traces
    }
    for ordinal, trace in enumerate(
            (traces[key] for key in sorted(traces)), start=1):
        _complete_source(repository, trace, ordinal)
    goals = _goals(source, scope)
    ctx.work_memory.end_query()
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
        input_observation_ref=observation.event.object_ref,
        influence_kind=MemoryLinkedRef.object(_instruction(source, 20144)),
        trace_prefix=(20145, 1),
    )
    mapper, postchecker, _, _, _ = _postcheck_owners()
    fixture = _question_fixture(
        executor_factory=lambda route: executor,
        world=(source, scope, goals[1].proposition),
        selection_committer=committer,
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    dialogue = MemoryAwareQuestionDialogueRuntime(
        ctx,
        fixture.runtime,
        trace_prefix=(20146, 1),
        source_records=repository,
    )
    return fixture, dialogue


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00_question_uses_jg_caller_and_reports_closed_cold_pages(
        backend_type):
    """PW-00 question 必须走 J-G、写真实 Use，并在报告前关闭 K-04 reader。"""
    backend, ctx, routes, _, runtime, observation, _ = _build_runtime(
        backend_type())
    fixture = None
    try:
        source = _query_source(document_id=1)
        fixture, dialogue = _question_dialogue(ctx, source, observation)

        run = runtime.run_question(dialogue, fixture.request)

        assert run.result.question.complete
        assert run.result.question.selection_commit.commits
        assert run.report.route_kind == routes.question
        assert run.report.core_unchanged
        assert run.report.query_closed
        assert run.report.physical_metrics_key
        assert ctx.memory_hot_set_runtime.query_resources_closed()
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.attractor_state is None
        assert max(
            len(item.input_key)
            for item in run.result.question.generation.plan.layers
        ) <= 256
        assert len(run.result.stable_key()) <= 4096
    finally:
        if fixture is not None:
            fixture.close()
        backend.close()


def test_pw00_sqlite_question_skips_full_segment_payload_snapshot(
        monkeypatch):
    """question 回滚快照不得全量读取大型 K-02 part 表。"""
    backend, ctx, _, _, runtime, observation, _ = _build_runtime(
        SQLiteBackend())
    fixture = None
    try:
        source = _query_source(document_id=1)
        fixture, dialogue = _question_dialogue(ctx, source, observation)
        original_select = backend._do_select

        def guarded_select(
                table, where, where_gt, order_by, descending, limit):
            if table == SEGMENT_OBJECT_PART_TABLE and where is None:
                raise AssertionError("question 执行了 K-02 part 全表快照")
            return original_select(
                table, where, where_gt, order_by, descending, limit)

        monkeypatch.setattr(backend, "_do_select", guarded_select)
        run = runtime.run_question(dialogue, fixture.request)

        assert run.result.question.complete
        assert run.report.query_closed
    finally:
        if fixture is not None:
            fixture.close()
        backend.close()


def test_pw00_question_blocks_segment_write_and_rolls_back_use(monkeypatch):
    """被排除的 K-02 表若尝试写入，必须在首写前失败并撤销 Use。"""
    backend, ctx, _, _, runtime, observation, _ = _build_runtime(
        SQLiteBackend())
    fixture = None
    try:
        source = _query_source(document_id=1)
        fixture, dialogue = _question_dialogue(ctx, source, observation)
        before = backend.recovery_state_snapshot()
        original = dialogue.run

        def write_segment_after_use(request):
            result = original(request)
            assert result.question.selection_commit.commits
            backend.insert(SEGMENT_OBJECT_PART_TABLE, {})
            return result

        monkeypatch.setattr(dialogue, "run", write_segment_after_use)
        with pytest.raises(RuntimeWriteGuardError, match="只读调用链"):
            runtime.run_question(dialogue, fixture.request)

        assert backend.recovery_state_snapshot() == before
        assert runtime.reports()[-1].query_closed
    finally:
        if fixture is not None:
            fixture.close()
        backend.close()


def test_runtime_content_fingerprint_is_fixed_and_domain_separated():
    """运行期内容引用须固定长度，并对内容和使用域的扰动分别敏感。"""
    baseline = integer_tuple_fingerprint(
        (1, -2, 3), domain="pw00.fingerprint.baseline.v1")

    assert len(baseline) == 34
    assert baseline == integer_tuple_fingerprint(
        (1, -2, 3), domain="pw00.fingerprint.baseline.v1")
    assert baseline != integer_tuple_fingerprint(
        (1, -2, 4), domain="pw00.fingerprint.baseline.v1")
    assert baseline != integer_tuple_fingerprint(
        (1, -2, 3), domain="pw00.fingerprint.other.v1")
    with pytest.raises(TypeError, match="严格整数"):
        integer_tuple_fingerprint(
            (1, True), domain="pw00.fingerprint.baseline.v1")


def test_pw00_sqlite_restart_restores_projection_manifest_and_question(
        tmp_path):
    """SQLite 重启须恢复新 K-04 generation、PW-00 owner 和同一 J-G/Use 行为。"""
    path = str(tmp_path / "pw00_restart.sqlite3")
    first_backend = SQLiteBackend(path)
    first_fixture = None
    try:
        (
            _, first, _, _, first_runtime, observation, first_projection,
        ) = _build_runtime(first_backend)
        source = _query_source(document_id=1)
        first_fixture, first_dialogue = _question_dialogue(
            first, source, observation)
        first_run = first_runtime.run_question(
            first_dialogue, first_fixture.request)
        assert (first.memory_interact_events.projection_state_key()
                != first_projection.source_state_key)

        first.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        current_projection = _publish_projection(
            first, first.memory_resolver_runtime.resolver)
        first.memory_hot_set_runtime.replace_projection(current_projection)
        _, current_manifest = _post_weaning_manifest(first, source)
        projection_key = current_projection.stable_key()
        manifest_key = current_manifest.stable_key()
        core_key = current_manifest.core_state_key
        expected_request_key = first_fixture.request.stable_key()
        expected_status = first_run.result.question.status
        expected_target = first_run.result.question.query.request.target
        expected_rendered = first_run.result.question.generation.rendered.stable_key()
        observation_ref = observation.event.object_ref
        first_backend.commit()
    finally:
        if first_fixture is not None:
            first_fixture.close()
        first_backend.close()

    second_backend = SQLiteBackend(path)
    second_fixture = None
    try:
        (
            second, source, routes, restored_manifest, runtime,
        ) = _restore_runtime(second_backend, projection_key)
        assert restored_manifest.stable_key() == manifest_key
        assert restored_manifest.core_state_key == core_key

        restored_observations = second.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_OBSERVATION,
            object_ref=observation_ref,
        )
        assert len(restored_observations) == 1
        second_fixture, second_dialogue = _question_dialogue(
            second, source, restored_observations[0])
        second_run = runtime.run_question(
            second_dialogue, second_fixture.request)

        assert second_fixture.request.stable_key() == expected_request_key
        assert second_run.result.question.status == expected_status
        assert second_run.result.question.query.request.target == expected_target
        assert (second_run.result.question.generation.rendered.stable_key()
                == expected_rendered)
        assert second_run.report.route_kind == routes.question
        assert second_run.report.core_unchanged
        assert second_run.report.query_closed
        assert second_run.report.physical_metrics_key
        assert second_run.result.question.selection_commit.commits
    finally:
        if second_fixture is not None:
            second_fixture.close()
        second_backend.close()


def test_pw00_v03_package_resume_restores_runtime_and_new_use(tmp_path):
    """V-03 全量包须在同 profile 新后端恢复 PW-00 manifest、K-04 查询和新 Use。"""
    source_backend = SQLiteBackend(":memory:")
    source_fixture = None
    try:
        (
            _, source_ctx, _, _, source_runtime, observation, old_projection,
        ) = _build_runtime(source_backend)
        source = _query_source(document_id=1)
        source_fixture, dialogue = _question_dialogue(
            source_ctx, source, observation)
        first_run = source_runtime.run_question(
            dialogue, source_fixture.request)
        assert (source_ctx.memory_interact_events.projection_state_key()
                != old_projection.source_state_key)

        source_ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        projection = _publish_projection(
            source_ctx, source_ctx.memory_resolver_runtime.resolver)
        source_ctx.memory_hot_set_runtime.replace_projection(projection)
        _, manifest = _post_weaning_manifest(source_ctx, source)
        projection_key = projection.stable_key()
        manifest_key = manifest.stable_key()
        core_key = manifest.core_state_key
        observation_ref = observation.event.object_ref
        expected_status = first_run.result.question.status
        expected_target = first_run.result.question.query.request.target
        expected_rendered = (
            first_run.result.question.generation.rendered.stable_key())

        run_dir = str(tmp_path / "pw00-v03-runs")
        spaces = [
            row["space_id"] for row in source_backend.select(
                "space", order_by="space_id")]
        cursor = CursorState(
            base_run_id="pw00-fixture",
            run_id="pw00-resume",
            completed={1},
            non_skippable={2},
        )
        assert dump_run(
            source_backend,
            run_dir,
            "pw00-resume",
            spaces=spaces,
            tables=None,
            require_all_spaces=True,
            versions=source.versions,
            cursor_state=cursor,
        ) == spaces
    finally:
        if source_fixture is not None:
            source_fixture.close()
        source_backend.close()

    target_backend = SQLiteBackend(":memory:")
    target_fixture = None
    try:
        from pure_integer_ai.cognition.shared.memory_batch import (
            install_memory_batch_runtimes,
        )
        schema_ctx = make_train_context(target_backend, companion=True)
        install_memory_batch_runtimes(schema_ctx, _batch_config())
        loaded = load_run_package(
            target_backend,
            run_dir,
            "pw00-resume",
            expected_versions=source.versions,
            expected_dependencies=(),
            expected_publish_epoch=1,
        )
        assert loaded.loaded_tables
        assert cursor_state_from_payload(
            loaded.cursor_payload,
            fallback_run_id="pw00-resume",
        ) == cursor

        (
            target, source, routes, restored_manifest, runtime,
        ) = _restore_runtime(target_backend, projection_key)
        assert restored_manifest.stable_key() == manifest_key
        assert restored_manifest.core_state_key == core_key
        restored_observations = target.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_OBSERVATION,
            object_ref=observation_ref,
        )
        assert len(restored_observations) == 1
        target_fixture, dialogue = _question_dialogue(
            target, source, restored_observations[0])

        resumed = runtime.run_question(dialogue, target_fixture.request)

        assert resumed.result.question.status == expected_status
        assert resumed.result.question.query.request.target == expected_target
        assert (resumed.result.question.generation.rendered.stable_key()
                == expected_rendered)
        assert resumed.result.question.selection_commit.commits
        assert resumed.report.route_kind == routes.question
        assert resumed.report.core_unchanged
        assert resumed.report.query_closed
        assert resumed.report.physical_metrics_key
    finally:
        if target_fixture is not None:
            target_fixture.close()
        target_backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00_external_define_failure_rolls_back_companion_and_watermark(
        monkeypatch, backend_type):
    """Companion 已分配后 SourceRecord 失败时应恢复物理行和下一个 assoc 水位。"""
    backend, ctx, routes, _, runtime, _, _ = _build_runtime(backend_type())
    try:
        source_intake = ctx.memory_read_intake.source_intake
        repository = source_intake.repository
        companion = source_intake.companion
        before_rows = tuple(companion.all_items())
        before_sources = repository.source_count()

        def fail_put(*args, **kwargs):
            """模拟 Companion 写入完成后的 SourceRecord 持久化故障。"""
            del args, kwargs
            raise RuntimeError("source persistence failed")

        monkeypatch.setattr(repository, "put_complete", fail_put)
        failed_source = _source(104)
        with pytest.raises(RuntimeError, match="source persistence failed"):
            runtime.run_intake(PostWeaningIntakeRequest(
                routes.external_define,
                failed_source,
                "失败定义",
                "license-failed-define",
                304,
                trace=(20150, 1),
            ))

        restored_companion = ctx.memory_read_intake.source_intake.companion
        assert tuple(restored_companion.all_items()) == before_rows
        assert repository.source_count() == before_sources
        assert runtime.reports()[-1].status != POST_WEANING_OPERATION_COMMITTED
        assert runtime.reports()[-1].core_unchanged
        monkeypatch.undo()
        assoc_id = restored_companion.put_text("恢复后定义", meta=71)
        expected = 1 if not before_rows else before_rows[-1]["assoc_id"] + 1
        assert assoc_id == expected
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_pw00_intake_postcommit_validation_failure_restores_full_state(
        monkeypatch, backend_type):
    """M-10 已提交后若 PW-00 最终校验失败，来源和 Memory 仍须完整撤销。"""
    backend, _, routes, _, runtime, _, _ = _build_runtime(backend_type())
    try:
        before = backend.recovery_state_snapshot()
        original = runtime._validate_committed

        def fail_after_validation(report):
            """先证明提交合法，再模拟返回前的最终校验故障。"""
            original(report)
            raise PostWeaningRuntimeError("postcommit validation failed")

        monkeypatch.setattr(runtime, "_validate_committed", fail_after_validation)
        source = _source(105)
        with pytest.raises(PostWeaningRuntimeError, match="postcommit"):
            runtime.run_intake(PostWeaningIntakeRequest(
                routes.reading,
                source,
                "提交后失败来源",
                "license-postcommit-failure",
                305,
                parser=_Parser(source, 5),
                trace=(20151, 1),
            ))

        assert backend.recovery_state_snapshot() == before
        assert runtime.reports()[-1].status != POST_WEANING_OPERATION_COMMITTED
        assert runtime.reports()[-1].before == runtime.reports()[-1].after
    finally:
        backend.close()


def test_pw00_question_failure_after_use_restores_full_state(monkeypatch):
    """J-G 已写真实 Use 后若调用失败，PW-00 必须撤销 Use 并关闭查询资源。"""
    backend, ctx, _, _, runtime, observation, _ = _build_runtime()
    fixture = None
    try:
        source = _query_source(document_id=1)
        fixture, dialogue = _question_dialogue(ctx, source, observation)
        before = backend.recovery_state_snapshot()
        original = dialogue.run

        def fail_after_use(request):
            """先完成真实 J-G 和 Use 提交，再模拟调用边界故障。"""
            result = original(request)
            assert result.question.selection_commit.commits
            raise RuntimeError("question failed after use")

        monkeypatch.setattr(dialogue, "run", fail_after_use)
        with pytest.raises(RuntimeError, match="after use"):
            runtime.run_question(dialogue, fixture.request)

        assert backend.recovery_state_snapshot() == before
        assert runtime.reports()[-1].status != POST_WEANING_OPERATION_COMMITTED
        assert runtime.reports()[-1].before == runtime.reports()[-1].after
        assert runtime.reports()[-1].query_closed
        assert ctx.memory_hot_set_runtime.query_resources_closed()
        assert ctx.work_memory.active_query_scope is None
        assert ctx.work_memory.attractor_state is None
    finally:
        if fixture is not None:
            fixture.close()
        backend.close()
