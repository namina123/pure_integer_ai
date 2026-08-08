"""PW-01 双 Memory 完整回答性能 runner 专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

import pure_integer_ai.experiments.pw01_dual_memory_performance as performance

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.memory_event import (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_LIFECYCLE,
    LifecycleTransitionPayload,
    MemoryEvent,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_LIFECYCLE,
    LogicalClockIdentity,
)
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.experiments.pw01_dual_memory_performance import (
    _measure_answer,
    _performance_policy,
    _prepare_fresh,
    _publish_synthetic_query_index,
    _publish_synthetic_read_projection,
    _restore,
    run_pw01_dual_memory_scale_curve,
)
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _COLD,
    _DEPENDENCIES,
    _CurrentContextScorer,
    _post_weaning_manifest,
    _refresh_projection,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryProjectionPublication,
    MemoryQueryIndexProjectionManifest,
    memory_hot_set_runtimes,
)
from pure_integer_ai.experiments.memory_query_index_maintenance import (
    MemoryQueryIndexMaintainer,
    MemoryQueryIndexMaintenancePolicy,
    MemoryQueryIndexRebuildRequired,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
)
from pure_integer_ai.experiments.pw01_controlled_reading import (
    PW01ControlledReadingParser,
    build_pw01_question_dialogue,
    pw01_source,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_HYPOTHESIS_EVENT_TABLE,
)


def _selection_snapshot(ctx):
    """返回按 Memory 空间排序的逻辑计数和逐序位候选身份。"""
    return tuple(
        (
            item.resolver.aggregates.event_log.memory_space_identity.stable_key(),
            item.considered_count(),
            item.selected_candidate_keys(),
        )
        for item in memory_hot_set_runtimes(ctx)
    )


def test_pw01_dual_memory_scale_curve_is_complete_and_restartable(tmp_path):
    """小规模合同须完成回答、扫描精确总记录数并在重启后保持。"""
    report = run_pw01_dual_memory_scale_curve(
        tmp_path / "pw01-performance.sqlite3",
        scales=(64,),
    )
    assert report.scales == (64,)
    assert len(report.points) == 1
    point = report.points[0]
    assert point.total_projection_records == 64
    assert point.read_projection_records == 61
    assert point.interact_projection_records == 3
    for measurement in (point.fresh, point.restart):
        assert measurement.answer_complete == 1
        assert measurement.source_exact == 1
        assert measurement.considered_candidates >= 64
        assert measurement.partition_rows_scanned == 64
        assert measurement.page_in_records == 64
        assert measurement.page_faults >= 2
        assert measurement.elapsed_ns > 0
        assert measurement.rss_peak_bytes > 0
        assert measurement.database_bytes > 0


def test_pw01_exact_query_index_preserves_logical_count_with_bounded_scan(
        tmp_path):
    """精确索引须保持完整 considered，同时把物理扫描降为有界前缀。"""
    report = run_pw01_dual_memory_scale_curve(
        tmp_path / "pw01-indexed-performance.sqlite3",
        scales=(64,),
        use_query_index=True,
    )
    point = report.points[0]
    for measurement in (point.fresh, point.restart):
        assert measurement.answer_complete == 1
        assert measurement.source_exact == 1
        assert measurement.considered_candidates == 68
        assert measurement.partition_rows_scanned < 64
        assert measurement.page_in_records < 64


def test_pw01_exact_query_index_matches_full_scan_candidate_by_candidate(
        tmp_path):
    """同一冻结投影的索引路径须逐 Memory、逐 request、逐候选等价。"""
    database = tmp_path / "pw01-index-equivalence.sqlite3"
    cloned_backend = None
    ctx, source, observation = _prepare_fresh(
        database, use_query_index=False)
    try:
        primary = _refresh_projection(ctx)
        read_count = 64 - primary.record_count
        read_projection = _publish_synthetic_read_projection(ctx, read_count)
        ctx.memory_read_hot_set_runtime.replace_projection(read_projection)

        full = _measure_answer(ctx, database, source, observation)
        full_selection = _selection_snapshot(ctx)
        assert all(item[1] is not None and item[2] is not None
                   for item in full_selection)

        _refresh_projection(ctx)
        query_index = _publish_synthetic_query_index(ctx, read_count)
        ctx.memory_read_hot_set_runtime.replace_policy(
            _performance_policy(indexed=True))
        ctx.memory_read_hot_set_runtime.replace_query_index(query_index)
        indexed = _measure_answer(ctx, database, source, observation)
        indexed_selection = _selection_snapshot(ctx)

        assert indexed_selection == full_selection
        assert indexed.considered_candidates == full.considered_candidates
        assert indexed.partition_rows_scanned < full.partition_rows_scanned

        cloned_backend = clone_backend(ctx.backend)
        cloned = clone_train_context(
            ctx, cloned_backend, label="pw01-index-clone")
        cloned_index = (
            cloned.memory_read_hot_set_runtime.query_index_projection())
        assert cloned_index == query_index
        _measure_answer(cloned, database, source, observation)
        assert _selection_snapshot(cloned) == indexed_selection
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        ctx.backend.close()


def test_current_context_index_fallback_prefix_covers_duplicate_exact_values():
    """fallback 读取 2K，足以跳过最多 K 个 exact 重复身份。"""
    scorer = _CurrentContextScorer()

    class _Request:
        """仅提供 planner 形成查询组所需字段。"""

        budget = 7

        class _Source:
            document_id = 901

        source = _Source()

    groups = scorer.query_groups(_Request())
    assert tuple(item.limit for item in groups) == (7, 14)


def test_query_index_segments_never_cross_planner_group_boundary(
        tmp_path, monkeypatch):
    """段预算命中组边界时 exact/fallback canonical range 不得重叠。"""
    monkeypatch.setattr(
        performance, "_PERFORMANCE_INDEX_SEGMENT_OBJECT_LIMIT", 64)
    database = tmp_path / "pw01-index-group-boundary.sqlite3"
    ctx, _, _ = _prepare_fresh(database, use_query_index=True)
    try:
        primary = _refresh_projection(ctx)
        candidate_count = 64 - primary.record_count
        read_projection = _publish_synthetic_read_projection(
            ctx, candidate_count)
        ctx.memory_read_hot_set_runtime.replace_projection(read_projection)
        query_index = _publish_synthetic_query_index(ctx, candidate_count)

        assert query_index.storage.record_count == candidate_count * 2
        assert len(query_index.storage.segments) == 2
        query_index.validate_store(ctx.tiered_segment_store)
    finally:
        ctx.backend.close()


def test_pw01_query_index_builder_generates_each_candidate_once(
        tmp_path, monkeypatch):
    """双入口索引发布不得为 exact/fallback 重复构造候选。"""
    database = tmp_path / "pw01-index-single-pass.sqlite3"
    ctx, _, _ = _prepare_fresh(database, use_query_index=True)
    synthetic_calls = 0
    payload_calls = 0
    planner_calls = 0
    original_synthetic = performance._synthetic_bundle
    original_payload = performance.encode_memory_candidate_payload
    try:
        primary = _refresh_projection(ctx)
        candidate_count = 64 - primary.record_count

        def synthetic(bundle, ordinal):
            nonlocal synthetic_calls
            synthetic_calls += 1
            return original_synthetic(bundle, ordinal)

        def payload(bundle):
            nonlocal payload_calls
            payload_calls += 1
            return original_payload(bundle)

        planner = next(
            item.score_provider
            for item in ctx.memory_resolver_runtime.resolvers
            if item.aggregates is ctx.memory_read_aggregates)
        original_entries = type(planner).index_entries

        def entries(self, bundle):
            nonlocal planner_calls
            planner_calls += 1
            return original_entries(self, bundle)

        monkeypatch.setattr(performance, "_synthetic_bundle", synthetic)
        monkeypatch.setattr(
            performance, "encode_memory_candidate_payload", payload)
        monkeypatch.setattr(type(planner), "index_entries", entries)

        query_index = _publish_synthetic_query_index(ctx, candidate_count)
        assert query_index.candidate_count == candidate_count
        assert synthetic_calls == candidate_count - 1
        assert payload_calls == candidate_count
        assert planner_calls == candidate_count
    finally:
        ctx.backend.close()


def test_pw01_real_initial_query_index_builds_bounded_runs_and_restarts(
        tmp_path):
    """真实 aggregate 初建索引须有界 spill、压实并与全扫逐候选等价。"""
    database = tmp_path / "pw01-real-initial-index.sqlite3"
    ctx, source, observation = _prepare_fresh(
        database, use_query_index=True)
    restored = None

    def answer_snapshot(current, current_source, current_observation):
        """运行生产问答并返回完整性和逐 resolver 候选快照。"""
        fixture, dialogue = build_pw01_question_dialogue(
            current, current_source, current_observation)
        try:
            _, current_manifest = _post_weaning_manifest(
                current, current_source)
            operation = PostWeaningDryRunRuntime(
                current, current_manifest).run_question(
                    dialogue, fixture.request)
            return operation.result.question.complete, _selection_snapshot(current)
        finally:
            fixture.close()
            performance._close_outer_lifecycle(current)

    try:
        routes, manifest = _post_weaning_manifest(ctx, ctx.f01_source)
        for ordinal in (94, 95):
            added_source = pw01_source(
                parser_version=2,
                document_id=900 + ordinal,
            )
            request = PostWeaningIntakeRequest(
                routes.reading,
                added_source,
                f"PW-01 real initial source {ordinal}",
                "CC0-1.0",
                2026080800 + ordinal,
                parser=PW01ControlledReadingParser(
                    added_source,
                    EVIDENCE_REFUTE,
                    ordinal,
                ),
                trace=(20260808, ordinal, 1),
            )
            operation = PostWeaningDryRunRuntime(
                ctx, manifest).run_intake(request)
            assert operation.report.core_unchanged
        read_projection = _refresh_projection(ctx)
        read_projection = ctx.memory_read_hot_set_runtime.projection
        ctx.memory_read_hot_set_runtime.replace_policy(
            _performance_policy(indexed=False))
        full_complete, full_selection = answer_snapshot(
            ctx, source, observation)
        assert full_complete
        _refresh_projection(ctx)
        read_projection = ctx.memory_read_hot_set_runtime.projection

        read_resolver = next(
            item for item in ctx.memory_resolver_runtime.resolvers
            if item.aggregates is ctx.memory_read_aggregates
        )
        initial = MemoryQueryIndexMaintainer(
            read_resolver, ctx.tiered_segment_store).build_initial(
                access=read_projection.access,
                hypothesis_kinds=read_projection.hypothesis_kinds,
                publication=MemoryProjectionPublication(
                    (20260808, 99, 30),
                    _COLD,
                    performance._PERFORMANCE_VERSION_KEY,
                    _DEPENDENCIES,
                    SegmentBudget(64, 2_000_000),
                    32,
                ),
                policy=MemoryQueryIndexMaintenancePolicy(
                    16, 16, compaction_page_limit=1),
            )
        assert initial.candidate_count >= 3
        assert len(initial.query_runs()) == 1
        assert len(initial.query_runs()[0].changes) == initial.candidate_count
        assert MemoryQueryIndexProjectionManifest.from_stable_key(
            initial.stable_key()) == initial
        ctx.memory_read_hot_set_runtime.replace_policy(
            _performance_policy(indexed=True))
        ctx.memory_read_hot_set_runtime.replace_query_index(initial)
        indexed_complete, indexed_selection = answer_snapshot(
            ctx, source, observation)
        assert indexed_complete
        assert indexed_selection == full_selection

        primary = _refresh_projection(ctx)
        ctx.backend.commit()
        observation_ref = observation.event.object_ref
        primary_key = primary.stable_key()
        read_key = read_projection.stable_key()
        index_key = initial.stable_key()
        ctx.backend.close()
        restored, source, observation = _restore(
            database,
            primary_key,
            read_key,
            index_key,
            observation_ref,
        )
        restarted_complete, restarted_selection = answer_snapshot(
            restored, source, observation)
        assert restarted_complete
        assert restarted_selection == full_selection
    finally:
        if restored is not None:
            restored.backend.close()
        else:
            ctx.backend.close()


def test_pw01_query_index_incrementally_tracks_new_reading_and_restarts(
        tmp_path, monkeypatch):
    """新 reading batch 只发布变化 run，旧基础段可跨真实重启复用。"""
    database = tmp_path / "pw01-index-incremental.sqlite3"
    ctx, source, observation = _prepare_fresh(
        database, use_query_index=True)
    restored = None
    try:
        primary = _refresh_projection(ctx)
        read_count = 64 - primary.record_count
        read_projection = _publish_synthetic_read_projection(ctx, read_count)
        ctx.memory_read_hot_set_runtime.replace_projection(read_projection)
        baseline = _publish_synthetic_query_index(ctx, read_count)
        ctx.memory_read_hot_set_runtime.replace_query_index(baseline)

        routes, manifest = _post_weaning_manifest(ctx, ctx.f01_source)
        hidden_source = replace(
            pw01_source(parser_version=2, document_id=903),
            owner=OwnerScope(1, 9, 0, VISIBILITY_USER),
        )
        hidden_request = PostWeaningIntakeRequest(
            routes.reading,
            hidden_source,
            "PW-01 ACL hidden incremental source",
            "CC0-1.0",
            2026080893,
            parser=PW01ControlledReadingParser(
                hidden_source,
                EVIDENCE_REFUTE,
                93,
            ),
            trace=(20260808, 93, 1),
        )
        hidden_operation = PostWeaningDryRunRuntime(
            ctx, manifest).run_intake(hidden_request)
        assert hidden_operation.report.core_unchanged

        added_source = pw01_source(parser_version=2, document_id=902)
        request = PostWeaningIntakeRequest(
            routes.reading,
            added_source,
            "PW-01 incremental source",
            "CC0-1.0",
            2026080892,
            parser=PW01ControlledReadingParser(
                added_source,
                EVIDENCE_REFUTE,
                92,
            ),
            trace=(20260808, 92, 1),
        )
        operation = PostWeaningDryRunRuntime(
            ctx, manifest).run_intake(request)
        assert operation.report.core_unchanged
        ctx.memory_read_aggregates.rebuild_dirty(
            access=read_projection.access)

        maintainer = MemoryQueryIndexMaintainer(
            next(
                item for item in ctx.memory_resolver_runtime.resolvers
                if item.aggregates is ctx.memory_read_aggregates
            ),
            ctx.tiered_segment_store,
        )
        original_select = type(ctx.backend).select

        def two_changed_hypotheses(
                backend, table, where=None, where_gt=None, order_by=None, *,
                descending=False, limit=None):
            """只替换 tail 反向索引页，其他读取继续委托真实后端。"""
            if table == MEMORY_HYPOTHESIS_EVENT_TABLE:
                fence = baseline.storage.source_fence
                return [
                    {"event_seq": fence + 1, "hypothesis_hash": 101},
                    {"event_seq": fence + 2, "hypothesis_hash": 102},
                ]
            return original_select(
                backend,
                table,
                where,
                where_gt,
                order_by,
                descending=descending,
                limit=limit,
            )

        with monkeypatch.context() as bounded_patch:
            bounded_patch.setattr(
                type(ctx.backend), "select", two_changed_hypotheses)
            with pytest.raises(
                    MemoryQueryIndexRebuildRequired,
                    match="变化候选数"):
                maintainer._changed_hypothesis_hashes(
                    baseline.storage.source_fence,
                    MemoryQueryIndexMaintenancePolicy(16, 1),
                )

        before_failure_keys = tuple(
            item.segment_key
            for item in ctx.tiered_segment_store.current_manifest().entries
        )
        original_publish = ctx.tiered_segment_store.publish_segment

        def fail_after_publish(*args, **kwargs):
            """模拟 segment 已发布后 caller 失败，要求精确清理本代。"""
            original_publish(*args, **kwargs)
            raise RuntimeError("query index maintenance injected failure")

        monkeypatch.setattr(
            ctx.tiered_segment_store, "publish_segment", fail_after_publish)
        with pytest.raises(RuntimeError, match="injected failure"):
            maintainer.maintain(
                baseline,
                publication=MemoryProjectionPublication(
                    (20260808, 98, 1),
                    _COLD,
                    performance._PERFORMANCE_VERSION_KEY,
                    _DEPENDENCIES,
                    SegmentBudget(64, 2_000_000),
                    32,
                ),
                policy=MemoryQueryIndexMaintenancePolicy(16, 16),
            )
        monkeypatch.setattr(
            ctx.tiered_segment_store, "publish_segment", original_publish)
        after_failure_keys = tuple(
            item.segment_key
            for item in ctx.tiered_segment_store.current_manifest().entries
        )
        assert after_failure_keys == before_failure_keys

        updated = maintainer.maintain(
            baseline,
            publication=MemoryProjectionPublication(
                (20260808, 99, 1),
                _COLD,
                performance._PERFORMANCE_VERSION_KEY,
                _DEPENDENCIES,
                SegmentBudget(64, 2_000_000),
                32,
            ),
            policy=MemoryQueryIndexMaintenancePolicy(16, 16),
        )
        assert updated.candidate_count == baseline.candidate_count + 1
        assert len(updated.query_runs()) == 2
        assert updated.query_runs()[-1].storage.record_count == 2
        assert len(updated.query_runs()[-1].changes) == 1
        assert all(
            item.owner_key != hidden_source.owner.stable_key()
            for item in updated.partitions
        )
        assert MemoryQueryIndexProjectionManifest.from_stable_key(
            updated.stable_key()) == updated
        ctx.memory_read_hot_set_runtime.replace_query_index(updated)
        _refresh_projection(ctx)

        fresh = _measure_answer(ctx, database, source, observation)
        assert fresh.considered_candidates == 69
        assert fresh.partition_rows_scanned < 64
        primary = _refresh_projection(ctx)
        ctx.backend.commit()
        observation_ref = observation.event.object_ref
        primary_key = primary.stable_key()
        read_key = read_projection.stable_key()
        updated_key = updated.stable_key()
        ctx.backend.close()

        restored, source, observation = _restore(
            database,
            primary_key,
            read_key,
            updated_key,
            observation_ref,
        )
        restarted = _measure_answer(
            restored, database, source, observation)
        assert restarted.considered_candidates == 69
        assert restarted.partition_rows_scanned == fresh.partition_rows_scanned
        assert restarted.answer_complete == 1
        assert restarted.source_exact == 1

        read_resolver = next(
            item for item in restored.memory_resolver_runtime.resolvers
            if item.aggregates is restored.memory_read_aggregates
        )
        added_records = restored.memory_read_aggregates.query(
            access=read_projection.access,
            source=added_source,
        )
        assert len(added_records) == 1
        added_bundle = read_resolver.load_bundle(
            added_records[0], access=read_projection.access)
        evidence_refs = tuple(
            item.event.object_ref
            for item in restored.memory_read_aggregates.events(
                added_bundle.hypothesis_ref,
                access=read_projection.access,
            )
            if item.event.event_kind == MEMORY_EVENT_EVIDENCE
        )
        assert len(evidence_refs) == 1
        changed_at = restored.memory_read_events.scoped_identities.resume_clock(
            LogicalClockIdentity(
                added_bundle.hypothesis.scope,
                CLOCK_MEMORY_LIFECYCLE,
            )
        ).advance()
        restored.memory_read_events.append(MemoryEvent(
            MEMORY_EVENT_LIFECYCLE,
            added_bundle.hypothesis_ref,
            added_bundle.hypothesis.scope,
            LifecycleTransitionPayload(
                added_bundle.hypothesis_ref,
                LIFECYCLE_ACTIVE,
                LIFECYCLE_ARCHIVED,
                evidence_refs,
                None,
                changed_at,
            ),
        ))
        restored.memory_read_aggregates.rebuild(
            added_bundle.hypothesis_ref,
            access=read_projection.access,
        )
        archive_maintainer = MemoryQueryIndexMaintainer(
            read_resolver, restored.tiered_segment_store)
        old_segment_keys = {
            segment.segment_key
            for run in updated.query_runs()
            for segment in run.storage.segments
        }
        archived = archive_maintainer.maintain(
                updated,
                publication=MemoryProjectionPublication(
                    (20260808, 98, 2),
                    _COLD,
                    performance._PERFORMANCE_VERSION_KEY,
                    _DEPENDENCIES,
                    SegmentBudget(64, 2_000_000),
                    32,
                ),
                policy=MemoryQueryIndexMaintenancePolicy(16, 16, 2),
            )
        assert archived.candidate_count == baseline.candidate_count
        assert len(archived.query_runs()) == 1
        assert len(archived.query_runs()[0].changes) == (
            archived.candidate_count)
        assert archived.query_runs()[0].storage.record_count == (
            sum(item.index_record_count for item in archived.partitions))
        assert MemoryQueryIndexProjectionManifest.from_stable_key(
            archived.stable_key()) == archived
        current_segment_keys = {
            item.segment_key
            for item in restored.tiered_segment_store.current_manifest().entries
        }
        assert old_segment_keys.isdisjoint(current_segment_keys)
        restored.memory_read_hot_set_runtime.replace_query_index(archived)
        after_archive = _measure_answer(
            restored, database, source, observation)
        assert after_archive.considered_candidates == 68
        assert after_archive.partition_rows_scanned < 64

        receipts = restored.memory_batch_visibility.receipts
        assert receipts.has_group_intent(2026080892)
        receipts.rollback_group(2026080892)
        restored_maintainer = MemoryQueryIndexMaintainer(
            next(
                item for item in restored.memory_resolver_runtime.resolvers
                if item.aggregates is restored.memory_read_aggregates
            ),
            restored.tiered_segment_store,
        )
        with pytest.raises(
                MemoryQueryIndexRebuildRequired,
                match="严格追加"):
            restored_maintainer.maintain(
                archived,
                publication=MemoryProjectionPublication(
                    (20260808, 99, 3),
                    _COLD,
                    performance._PERFORMANCE_VERSION_KEY,
                    _DEPENDENCIES,
                    SegmentBudget(64, 2_000_000),
                    32,
                ),
                policy=MemoryQueryIndexMaintenancePolicy(16, 16),
            )
    finally:
        if restored is not None:
            restored.backend.close()
        else:
            ctx.backend.close()


def test_pw01_query_index_allows_last_partition_to_become_inactive(tmp_path):
    """最后一个 Memory 候选归档后保留失效 run，而不是伪造非空分区。"""
    database = tmp_path / "pw01-index-all-inactive.sqlite3"
    ctx, source, observation = _prepare_fresh(
        database, use_query_index=True)
    try:
        _refresh_projection(ctx)
        read_projection = _publish_synthetic_read_projection(ctx, 1)
        ctx.memory_read_hot_set_runtime.replace_projection(read_projection)
        baseline = _publish_synthetic_query_index(ctx, 1)
        ctx.memory_read_hot_set_runtime.replace_query_index(baseline)
        read_resolver = next(
            item for item in ctx.memory_resolver_runtime.resolvers
            if item.aggregates is ctx.memory_read_aggregates
        )
        records = ctx.memory_read_aggregates.query(
            access=read_projection.access,
            source=pw01_source(parser_version=1),
        )
        assert len(records) == 1
        bundle = read_resolver.load_bundle(
            records[0], access=read_projection.access)
        evidence_refs = tuple(
            item.event.object_ref
            for item in ctx.memory_read_aggregates.events(
                bundle.hypothesis_ref,
                access=read_projection.access,
            )
            if item.event.event_kind == MEMORY_EVENT_EVIDENCE
        )
        changed_at = ctx.memory_read_events.scoped_identities.resume_clock(
            LogicalClockIdentity(
                bundle.hypothesis.scope,
                CLOCK_MEMORY_LIFECYCLE,
            )
        ).advance()
        ctx.memory_read_events.append(MemoryEvent(
            MEMORY_EVENT_LIFECYCLE,
            bundle.hypothesis_ref,
            bundle.hypothesis.scope,
            LifecycleTransitionPayload(
                bundle.hypothesis_ref,
                LIFECYCLE_ACTIVE,
                LIFECYCLE_ARCHIVED,
                evidence_refs,
                None,
                changed_at,
            ),
        ))
        ctx.memory_read_aggregates.rebuild(
            bundle.hypothesis_ref,
            access=read_projection.access,
        )

        archived = MemoryQueryIndexMaintainer(
            read_resolver, ctx.tiered_segment_store).maintain(
                baseline,
                publication=MemoryProjectionPublication(
                    (20260808, 99, 10),
                    _COLD,
                    performance._PERFORMANCE_VERSION_KEY,
                    _DEPENDENCIES,
                    SegmentBudget(64, 2_000_000),
                    32,
                ),
                policy=MemoryQueryIndexMaintenancePolicy(16, 16),
            )

        assert archived.candidate_count == 0
        assert archived.partitions == ()
        assert len(archived.query_runs()) == 2
        assert archived.query_runs()[-1].storage.record_count == 0
        assert archived.query_runs()[-1].changes[0].active == 0
        assert MemoryQueryIndexProjectionManifest.from_stable_key(
            archived.stable_key()) == archived
        ctx.memory_read_hot_set_runtime.replace_query_index(archived)
        _refresh_projection(ctx)
        fixture, dialogue = build_pw01_question_dialogue(
            ctx, source, observation)
        try:
            _, manifest = _post_weaning_manifest(ctx, source)
            operation = PostWeaningDryRunRuntime(
                ctx, manifest).run_question(dialogue, fixture.request)
            assert not operation.result.question.complete
            selected = (
                ctx.memory_read_hot_set_runtime.selected_candidate_keys())
            assert selected is not None
            assert all(
                item[2] != RESOLUTION_ORIGIN_MEMORY for item in selected)
        finally:
            fixture.close()
            performance._close_outer_lifecycle(ctx)
    finally:
        ctx.backend.close()
