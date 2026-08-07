"""PW-01 双 Memory 读后问答、回滚、隔离和重启专项。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.memory_event import MEMORY_EVENT_USE
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _ACCESS,
    _close_outer_lifecycle,
    _observation,
    _post_weaning_manifest,
    _refresh_projection,
    _restore_runtime,
    prepare_facility_context,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
)
from pure_integer_ai.experiments.pw01_controlled_reading import (
    PW01ControlledReadingParser,
    PW01_HYPOTHESIS_KIND,
    build_pw01_question_dialogue,
    install_pw01_controlled_query,
    pw01_source,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend


def _question(ctx, source, observation):
    """执行一次目标 3 完整问答，并保证 fixture 和外层 scope 关闭。"""
    _close_outer_lifecycle(ctx)
    _refresh_projection(ctx)
    fixture, dialogue = build_pw01_question_dialogue(
        ctx, source, observation)
    try:
        _, manifest = _post_weaning_manifest(ctx, source)
        operation = PostWeaningDryRunRuntime(
            ctx, manifest).run_question(dialogue, fixture.request)
        return operation
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)


def _intake(
        ctx,
        source,
        *,
        stance: int,
        batch_id: int,
        lineage_id: int,
        supersedes_source=None,
        ):
    """经正式 reading route 摄入一个受控来源并返回操作结果。"""
    routes, manifest = _post_weaning_manifest(ctx, ctx.f01_source)
    request = PostWeaningIntakeRequest(
        routes.reading,
        source,
        f"PW-01 controlled source {source.document_id}",
        "CC0-1.0",
        batch_id,
        parser=PW01ControlledReadingParser(source, stance, lineage_id),
        supersedes_source=supersedes_source,
        trace=(20260807, 2, lineage_id),
    )
    return PostWeaningDryRunRuntime(ctx, manifest).run_intake(request)


def _uses(ctx):
    """返回当前 owner 可见的交互 Use 事件。"""
    return ctx.memory_interact_events.query(
        access=_ACCESS, event_kind=MEMORY_EVENT_USE)


def test_pw01_reading_causes_held_out_answer_and_survives_restart(tmp_path):
    """只允许新增阅读来源成为 held-out 改善的充分且必要原因。"""
    database = tmp_path / "pw01.sqlite3"
    backend = SQLiteBackend(str(database))
    support_batch = 2026080711
    refute_batch = 2026080712
    projection_key = None
    observation_ref = None
    try:
        ctx = make_train_context(backend, companion=True)
        prepare_facility_context(ctx)
        install_pw01_controlled_query(ctx)
        source = ctx.f01_source
        observation = ctx.f01_observation
        observation_ref = observation.event.object_ref

        before = _question(ctx, source, observation)
        assert not before.result.question.complete
        assert _uses(ctx) == ()

        learned_source = pw01_source(parser_version=1)
        learned = _intake(
            ctx,
            learned_source,
            stance=EVIDENCE_SUPPORT,
            batch_id=support_batch,
            lineage_id=1,
        )
        assert learned.report.core_unchanged
        ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)

        same_user_other_session = MemoryAccessContext(1, 2, 4)
        other_user = MemoryAccessContext(1, 9, 4)
        assert len(ctx.memory_read_aggregates.query(
            access=same_user_other_session,
            hypothesis_kind=PW01_HYPOTHESIS_KIND,
            source=learned_source,
        )) == 1
        assert ctx.memory_read_aggregates.query(
            access=other_user,
            hypothesis_kind=PW01_HYPOTHESIS_KIND,
            source=learned_source,
        ) == ()
        assert ctx.memory_interact_aggregates.query(
            access=same_user_other_session,
            hypothesis_kind=PW01_HYPOTHESIS_KIND,
        ) == ()

        with isolated_evaluation(ctx, label="pw01-exact-source-ablation") as clone:
            clone.work_memory.end_session()
            clone.memory_batch_coordinator.rollback_batch(support_batch)
            clone.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
            ablated = _question(
                clone,
                source,
                _observation(clone, observation_ref),
            )
            assert not ablated.result.question.complete

        bad_source = pw01_source(parser_version=2)
        _intake(
            ctx,
            bad_source,
            stance=EVIDENCE_REFUTE,
            batch_id=refute_batch,
            lineage_id=2,
            supersedes_source=learned_source,
        )
        ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
        with isolated_evaluation(ctx, label="pw01-bad-revision") as clone:
            clone.work_memory.end_session()
            conflicted = _question(
                clone,
                source,
                _observation(clone, observation_ref),
            )
            assert not conflicted.result.question.complete
        ctx.memory_batch_coordinator.rollback_batch(refute_batch)
        ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)

        after = _question(ctx, source, observation)
        assert after.result.question.complete
        assert {item.trace.source for item in after.result.sources} == {
            learned_source}
        use = _uses(ctx)[-1].event.payload
        assert use.memory_ref.memory_space == (
            ctx.memory_read_events.memory_space_identity)
        assert use.episode_ref.memory_space == (
            ctx.memory_interact_events.memory_space_identity)
        ctx.memory_read_aggregates.require_clean(access=_ACCESS)
        ctx.memory_interact_aggregates.require_clean(access=_ACCESS)
        projection_key = _refresh_projection(ctx).stable_key()
    finally:
        backend.close()

    restored_backend = SQLiteBackend(str(database))
    try:
        restored, source, _ = _restore_runtime(
            restored_backend, projection_key)
        install_pw01_controlled_query(restored)
        restored.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
        observation = _observation(restored, observation_ref)
        resumed = _question(restored, source, observation)
        assert resumed.result.question.complete
        assert {item.trace.source for item in resumed.result.sources} == {
            pw01_source(parser_version=1)}
        assert resumed.report.core_unchanged
        assert resumed.report.query_closed
    finally:
        restored_backend.close()
