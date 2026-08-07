"""PW-01 双 Memory 完整回答性能 runner 专项。"""
from __future__ import annotations

import pure_integer_ai.experiments.pw01_dual_memory_performance as performance

from pure_integer_ai.experiments.pw01_dual_memory_performance import (
    _measure_answer,
    _performance_policy,
    _prepare_fresh,
    _publish_synthetic_query_index,
    _publish_synthetic_read_projection,
    run_pw01_dual_memory_scale_curve,
)
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _CurrentContextScorer,
    _refresh_projection,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    memory_hot_set_runtimes,
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
