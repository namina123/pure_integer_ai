"""DLG-04 单中心 Memory demand read 的真实 M-06/M-07 专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

import pure_integer_ai.experiments.facility_readiness_scenarios as facility
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_memory_demand_runtime import (
    ConversationMemoryDemandConsumer,
    ConversationMemoryDemandError,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    EXPANSION_CHANNELS,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    MemoryProjectionPublication,
    MemoryProjectionError,
    MemoryQueryIndexProjectionManifest,
    install_memory_hot_set_runtime,
)
from pure_integer_ai.experiments.memory_query_index_maintenance import (
    MemoryQueryIndexMaintainer,
    MemoryQueryIndexMaintenancePolicy,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
from pure_integer_ai.storage.sealed_segment import SegmentBudget

from test_d02_md01_memory_dynamics_contract import _profile
from test_d02_md03_directional_center_adapter import _adapter
from test_m03_memory_event import _core_refs
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source as _query_source,
)
from test_m07_memory_resolver import (
    _IndexFilterProvider as _FilteredIndexProvider,
    _install_resolver,
    _protocol,
    _seed_memory,
)


def _profile_for_understanding():
    """把 MD-03 UNDERSTANDING expansion key 绑定到完整 channel profile。"""
    profile = _profile(30100)
    return replace(
        profile,
        profile_key=type(profile.profile_key)((30100, 3, 3)),
        obligation_kind_keys=(type(profile.profile_key)((30100, 3, 1)),),
    )


def _fixture(backend=None, *, seed: bool = True):
    """建立或恢复真实 Memory 候选、M-06/M-07 runtime 与 typed query。"""
    if backend is None:
        backend = DictBackend()
    ctx = make_train_context(backend)
    if seed:
        _seed_memory(ctx)
    source = _query_source(document_id=1)
    query_runtime, resolver_runtime = _install_resolver(
        ctx, _protocol(source), _core_refs(ctx)[1])
    scope = _open_query(ctx, source)
    current = _current(ctx, source, scope)
    center = _adapter().from_understanding(
        current, current.occurrences[0], strength="CONDITIONAL")
    consumer = ConversationMemoryDemandConsumer(
        ctx, query_runtime, resolver_runtime)
    return backend, ctx, current, center, consumer


def _exact_index_fixture(backend=None):
    """建立真实 K-04 sealed projection 与 exact query-index consumer。"""
    if backend is None:
        backend = DictBackend()
    ctx = make_train_context(backend, companion=True)
    facility.install_memory_batch_runtimes(ctx, facility._batch_config())
    facility._seed_memory(ctx)
    source = facility._query_source(document_id=1)
    core_ref = facility._core_refs(ctx)[1]
    query_runtime, resolver_runtime = facility._install_resolver(
        ctx, source, core_ref)
    hot_set = facility._install_additional_memory_hot_set(
        ctx,
        resolver_runtime.resolver,
        namespace=1,
        hypothesis_kinds=facility._KINDS,
    )
    index = MemoryQueryIndexMaintainer(
        resolver_runtime.resolver,
        ctx.tiered_segment_store,
    ).build_initial(
        access=facility._ACCESS,
        hypothesis_kinds=hot_set.projection.hypothesis_kinds,
        publication=MemoryProjectionPublication(
            (20260819, 1),
            facility._COLD,
            (20260819, 1),
            facility._DEPENDENCIES,
            SegmentBudget(8, 1_000_000),
            1,
        ),
        policy=MemoryQueryIndexMaintenancePolicy(
            8, 100, 8, 4),
    )
    return backend, ctx, source, query_runtime, resolver_runtime, hot_set, index


def _sealed_page_profile():
    """为真实小投影保留明确且可审计的 page-in 硬上限。"""
    profile = _profile_for_understanding()
    return replace(
        profile,
        channel_budgets=tuple(
            replace(item, max_page_reads=64, max_cold_bytes=4_096)
            if item.channel_key == "L4_SEALED_PAGE" else item
            for item in profile.channel_budgets
        ),
    )


def test_dlg04_real_memory_read_preserves_context_boundary_and_unknown_proof():
    """真实 M-06/M-07 read 返回候选、预算和零写回滚证据。"""
    backend, ctx, current, center, consumer = _fixture()
    try:
        profile = _profile_for_understanding()
        context_read = start_conversation_context((9901, 1)).read(0)
        before = backend.snapshot()
        result = consumer.read(
            current,
            center,
            profile,
            access=MemoryAccessContext(1, 2, 3),
            context_read=context_read,
        )
        receipt = result.receipt
        assert receipt.status == "HIT"
        assert receipt.channel_key == "L3_MEMORY_OVERLAY"
        assert receipt.source == current.source
        assert receipt.scope == current.scope
        assert receipt.context_revision == 0
        assert receipt.context_read_digest == context_read.digest
        assert receipt.selected_candidate_keys
        assert receipt.zero_unrelated_scan_proven == 0
        assert receipt.page_read_count == 0
        assert receipt.rollback_before_key == receipt.rollback_after_key
        assert backend.snapshot() == before
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_dlg04_fails_closed_on_profile_mismatch_and_scan_budget():
    """中心/profile 漂移和当前 channel 超预算均不得返回 partial read。"""
    backend, ctx, current, center, consumer = _fixture()
    try:
        profile = _profile_for_understanding()
        context_read = start_conversation_context((9901, 2)).read(0)
        wrong = replace(profile, profile_key=type(profile.profile_key)((9901, 9)))
        with pytest.raises(ConversationMemoryDemandError, match="expansion key"):
            consumer.read(
                current, center, wrong,
                access=MemoryAccessContext(1, 2, 3),
                context_read=context_read,
            )
        constrained = replace(
            profile,
            channel_budgets=tuple(
                replace(item, max_scanned_objects=1)
                if item.channel_key == "L3_MEMORY_OVERLAY" else item
                for item in profile.channel_budgets
            ),
        )
        with pytest.raises(ConversationMemoryDemandError, match="scanned objects"):
            consumer.read(
                current, center, constrained,
                access=MemoryAccessContext(1, 2, 3),
                context_read=context_read,
            )
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_dlg04_exact_index_page_read_matches_overlay_and_proves_zero_scan():
    """真实 K-04 exact index page-in 与无索引候选等价且只读。"""
    (
        backend,
        ctx,
        source,
        query_runtime,
        resolver_runtime,
        hot_set,
        query_index,
    ) = _exact_index_fixture()
    consumer = ConversationMemoryDemandConsumer(
        ctx, query_runtime, resolver_runtime)
    profile = _sealed_page_profile()

    def run_once(indexed: bool):
        """在独立 query scope 中执行一次 demand read。"""
        hot_set.replace_query_index(query_index if indexed else None)
        scope = facility._open_query(ctx, source)
        current = facility._current(ctx, source, scope)
        center = _adapter().from_understanding(
            current, current.occurrences[0], strength="CONDITIONAL")
        context_read = start_conversation_context(
            (9901, 4 if indexed else 3)).read(0)
        before = backend.snapshot()
        source_state = resolver_runtime.resolver.aggregates.event_log.projection_state_key()
        try:
            result = consumer.read(
                current,
                center,
                profile,
                access=MemoryAccessContext(1, 2, 3),
                context_read=context_read,
            )
            assert backend.snapshot() == before
            assert resolver_runtime.resolver.aggregates.event_log.projection_state_key() == source_state
            return result
        finally:
            facility._close_outer_lifecycle(ctx)

    cloned_backend = None
    try:
        baseline = run_once(False)
        indexed = run_once(True)
        assert baseline.receipt.channel_key == "L4_SEALED_PAGE"
        assert indexed.receipt.channel_key == "L4_SEALED_PAGE"
        assert baseline.receipt.zero_unrelated_scan_proven == 0
        assert indexed.receipt.zero_unrelated_scan_proven == 1
        assert indexed.receipt.selected_candidate_keys == baseline.receipt.selected_candidate_keys
        assert indexed.receipt.considered_count == baseline.receipt.considered_count
        assert indexed.receipt.page_read_count > 0
        assert indexed.receipt.page_read_count <= 64
        assert indexed.receipt.cold_read_bytes <= 4_096
        assert indexed.receipt.rollback_before_key == indexed.receipt.rollback_after_key

        # 安装 index 不足以证明所有 request 都走 exact path；额外 filter
        # 会让 K-04 回退到普通投影扫描，receipt 必须如实回到 0。
        original_filter_provider = resolver_runtime.resolver.index_filter_provider
        resolver_runtime.resolver.index_filter_provider = (
            _FilteredIndexProvider(source))
        try:
            filtered = run_once(True)
            assert filtered.receipt.zero_unrelated_scan_proven == 0
        finally:
            resolver_runtime.resolver.index_filter_provider = (
                original_filter_provider)

        cloned_backend = clone_backend(ctx.backend)
        cloned = clone_train_context(
            ctx, cloned_backend, label="dlg04-exact-index-clone")
        assert cloned.memory_hot_set_runtime.query_index_projection() == query_index
        clone_consumer = ConversationMemoryDemandConsumer(
            cloned,
            cloned.memory_query_runtime,
            cloned.memory_resolver_runtime,
        )
        try:
            clone_scope = facility._open_query(cloned, source)
            clone_current = facility._current(cloned, source, clone_scope)
            clone_center = _adapter().from_understanding(
                clone_current,
                clone_current.occurrences[0],
                strength="CONDITIONAL",
            )
            clone_result = clone_consumer.read(
                clone_current,
                clone_center,
                profile,
                access=MemoryAccessContext(1, 2, 3),
                context_read=start_conversation_context((9901, 5)).read(0),
            )
            assert clone_result.receipt.zero_unrelated_scan_proven == 1
            assert clone_result.receipt.selected_candidate_keys == (
                indexed.receipt.selected_candidate_keys)
            assert clone_result.receipt.considered_count == (
                indexed.receipt.considered_count)
        finally:
            facility._close_outer_lifecycle(cloned)
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def test_dlg04_exact_index_fails_closed_after_memory_source_state_advance():
    """索引 source state 过期时，consumer 不得读取旧 sealed page。"""
    (
        backend,
        ctx,
        source,
        query_runtime,
        resolver_runtime,
        hot_set,
        query_index,
    ) = _exact_index_fixture()
    consumer = ConversationMemoryDemandConsumer(
        ctx, query_runtime, resolver_runtime)
    profile = _sealed_page_profile()
    hot_set.replace_query_index(query_index)
    try:
        aggregate = ctx.memory_interact_aggregates.query(
            access=facility._ACCESS,
            hypothesis_kind=(7201,),
        )[0]
        bundle = resolver_runtime.resolver.load_bundle(
            aggregate,
            access=facility._ACCESS,
        )
        state_before = resolver_runtime.resolver.aggregates.event_log.projection_state_key()
        ledger = HypothesisLedger(
            MemoryHypothesisEventSink(ctx.memory_interact_events),
        )
        ledger.register(bundle.hypothesis)
        ledger.append_evidence(EvidenceRecord(
            99,
            bundle.hypothesis,
            EVIDENCE_SUPPORT,
            (8799,),
            bundle.sources[0],
            99,
        ))
        ctx.memory_interact_aggregates.rebuild_dirty(access=facility._ACCESS)
        state_after = resolver_runtime.resolver.aggregates.event_log.projection_state_key()
        assert state_after != state_before
        scope = facility._open_query(ctx, source)
        current = facility._current(ctx, source, scope)
        center = _adapter().from_understanding(
            current,
            current.occurrences[0],
            strength="CONDITIONAL",
        )
        with pytest.raises(MemoryProjectionError, match="失效"):
            consumer.read(
                current,
                center,
                profile,
                access=MemoryAccessContext(1, 2, 3),
                context_read=start_conversation_context((9901, 6)).read(0),
            )
    finally:
        facility._close_outer_lifecycle(ctx)
        backend.close()


def test_dlg04_sqlite_fresh_resume_preserves_exact_index_consumer(tmp_path):
    """SQLite fresh/resume 恢复投影、索引和 consumer 的只读证明。"""
    database = tmp_path / "dlg04-fresh-resume.sqlite3"
    backend = SQLiteBackend(str(database))
    restored_backend = None
    try:
        (
            backend,
            ctx,
            source,
            _query_runtime,
            _resolver_runtime,
            hot_set,
            query_index,
        ) = _exact_index_fixture(backend)
        projection_key = hot_set.projection.stable_key()
        query_index_key = query_index.stable_key()
        # 先落首回合 typed graph object，fresh/resume 只能恢复已提交身份。
        scope = facility._open_query(ctx, source)
        facility._current(ctx, source, scope)
        facility._close_outer_lifecycle(ctx)
        backend.commit()
        backend.close()
        backend = None

        restored_backend = SQLiteBackend(str(database))
        restored_ctx = make_train_context(restored_backend, companion=True)
        facility.install_memory_batch_runtimes(
            restored_ctx, facility._batch_config())
        restored_source = facility._query_source(document_id=1)
        restored_query_runtime, restored_resolver_runtime = (
            facility._install_resolver(
                restored_ctx,
                restored_source,
                facility._core_refs(restored_ctx)[1],
            )
        )
        projection = MemoryCandidateProjectionManifest.from_stable_key(
            projection_key)
        projection.validate_store(restored_ctx.tiered_segment_store)
        restored_hot_set = install_memory_hot_set_runtime(
            restored_ctx,
            projection,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                facility._NoPrefetch(),
                8,
            ),
            resolver=restored_resolver_runtime.resolver,
        )
        restored_index = MemoryQueryIndexProjectionManifest.from_stable_key(
            query_index_key)
        restored_index.validate_store(restored_ctx.tiered_segment_store)
        restored_hot_set.replace_query_index(restored_index)

        scope = facility._open_query(restored_ctx, restored_source)
        current = facility._current(restored_ctx, restored_source, scope)
        center = _adapter().from_understanding(
            current,
            current.occurrences[0],
            strength="CONDITIONAL",
        )
        before = restored_backend.snapshot()
        result = ConversationMemoryDemandConsumer(
            restored_ctx,
            restored_query_runtime,
            restored_resolver_runtime,
        ).read(
            current,
            center,
            _sealed_page_profile(),
            access=MemoryAccessContext(1, 2, 3),
            context_read=start_conversation_context((9901, 7)).read(0),
        )
        assert restored_backend.snapshot() == before
        assert result.receipt.status == "HIT"
        assert result.receipt.zero_unrelated_scan_proven == 1
        assert result.receipt.page_read_count > 0
        assert result.receipt.rollback_before_key == result.receipt.rollback_after_key
    finally:
        if restored_backend is not None:
            if 'restored_ctx' in locals():
                facility._close_outer_lifecycle(restored_ctx)
            restored_backend.close()
        if backend is not None:
            backend.close()
