from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_CONFLICTED,
    MEMORY_EVIDENCE_CORROBORATED,
    MEMORY_EVIDENCE_PROVISIONAL,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EvidencePayload,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_USE,
    MemoryLinkedRef,
    UsePayload,
    MemoryEvent,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED,
    CLOCK_MEMORY_USED,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend

from test_m03_memory_event import (
    _append_episode,
    _append_observation,
    _core_refs,
    _scopes,
    _source,
    _timestamp,
    _hypothesis,
)


def _hypothesis_ref(ctx, hypothesis):
    """从事件声明中恢复指定 H-00 候选的完整 Memory 引用。"""
    access = MemoryAccessContext(1, 2, 3)
    entries = ctx.memory_interact_events.query(
        access=access, event_kind=MEMORY_EVENT_HYPOTHESIS)
    for entry in entries:
        if entry.event.payload.hypothesis == hypothesis:
            return entry.event.object_ref
    raise AssertionError("测试候选没有 Hypothesis 声明")


class _TwoSourcePolicy:
    """测试用按来源数注入的 corroboration 门。"""

    def is_corroborated(
            self,
            hypothesis,
            *,
            support_count,
            contradict_count,
            support_source_count,
            contradict_source_count,
            ):
        """仅在两个独立支持来源且无反对来源时升格。"""
        del hypothesis, support_count, contradict_count
        return support_source_count >= 2 and contradict_source_count == 0


def test_episode_source_is_recovered_and_corroboration_is_injected():
    """Evidence 通过 Episode 回溯 Observation 来源，证据门不写死在聚合器中。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source_a = _source(source_id=10)
        source_b = _source(source_id=11)
        refs = _core_refs(ctx)
        observation_a = _append_observation(ctx, source_a, refs)
        episode_a = _append_episode(ctx, source_a, observation_a, refs)
        observation_b = _append_observation(ctx, source_b, refs)
        episode_b = _append_episode(ctx, source_b, observation_b, refs)
        hypothesis = _hypothesis(source_a, 1)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(hypothesis)
        hypothesis_ref = _hypothesis_ref(ctx, hypothesis)

        for evidence_id, episode, seq in (
                (401, episode_a, 2), (402, episode_b, 3)):
            payload = EvidencePayload(
                hypothesis_ref,
                EVIDENCE_SUPPORT,
                None,
                (evidence_id,),
                None,
                episode.event.object_ref,
                (evidence_id + 1000,),
                None,
                _timestamp(episode.event.scope, CLOCK_MEMORY_OBSERVED, seq),
            )
            evidence_ref = memory_object_ref(
                ctx.memory_interact_events.memory_space_identity,
                4,
                payload.stable_key(),
                owner=source_a.owner,
                versions=source_a.versions,
            )
            ctx.memory_interact_events.append(MemoryEvent(
                MEMORY_EVENT_EVIDENCE,
                evidence_ref,
                episode.event.scope,
                payload,
            ))

        access = MemoryAccessContext(1, 2, 3)
        ctx.memory_interact_aggregates.policy = _TwoSourcePolicy()
        ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        aggregate = ctx.memory_interact_aggregates.read(
            hypothesis_ref, access=access)
        assert aggregate is not None
        assert aggregate.independent_source_count == 2
        assert aggregate.evidence_state == MEMORY_EVIDENCE_CORROBORATED
        assert len(ctx.memory_interact_aggregates.sources(
            hypothesis_ref, access=access)) == 2
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_aggregate_deduplicates_sources_and_preserves_conflict(backend_type):
    """同源重复只增加原始事件计数，正负来源并存且不直接排除候选。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        source_a = _source(source_id=10)
        source_b = _source(source_id=11)
        old = _hypothesis(source_a, 1)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(old)
        ledger.append_evidence(EvidenceRecord(
            101, old, EVIDENCE_SUPPORT, (501,), source_a, 1, (601,)))
        ledger.append_evidence(EvidenceRecord(
            102, old, EVIDENCE_SUPPORT, (502,), source_a, 2, (602,)))
        ledger.append_evidence(EvidenceRecord(
            103, old, EVIDENCE_REFUTE, (503,), source_b, 3, (603,)))

        access = MemoryAccessContext(1, 2, 3)
        report = ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        assert report.scanned_event_count == 0
        aggregate = ctx.memory_interact_aggregates.read(
            _hypothesis_ref(ctx, old), access=access)
        assert aggregate is not None
        assert aggregate.support_count == 2
        assert aggregate.contradict_count == 1
        assert aggregate.independent_source_count == 2
        assert aggregate.support_source_count == 1
        assert aggregate.contradict_source_count == 1
        assert aggregate.evidence_state == MEMORY_EVIDENCE_CONFLICTED
        assert ctx.memory_interact_aggregates.query(
            access=access, source=source_a) == (aggregate,)
        assert ctx.memory_interact_aggregates.query(
            access=access, source=source_b) == (aggregate,)
        assert ctx.memory_interact_aggregates.query(
            access=access, hypothesis_kind=old.hypothesis_kind,
            context=old.scope.stable_key()) == (aggregate,)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_rebuild_all_after_derived_tables_are_deleted_and_use_is_explicit(
        backend_type):
    """删除所有派生表后，仅凭 event 重建，并且未写 Use 不计使用失败。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        source = _source()
        refs = _core_refs(ctx)
        observation = _append_observation(ctx, source, refs)
        episode = _append_episode(ctx, source, observation, refs)
        hypothesis = _hypothesis(source, 1)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(hypothesis)
        ledger.append_evidence(EvidenceRecord(
            201, hypothesis, EVIDENCE_SUPPORT, (701,), source, 1, (801,)))
        access = MemoryAccessContext(1, 2, 3)
        ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        ref = _hypothesis_ref(ctx, hypothesis)
        before = ctx.memory_interact_aggregates.read(ref, access=access)
        assert before is not None
        assert before.use_count == 0

        use_payload = UsePayload(
            ref,
            episode.event.object_ref,
            MemoryLinkedRef.core(refs[0]),
            None,
            _timestamp(_scopes(source)[2], CLOCK_MEMORY_USED, 4),
        )
        use_ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_USE,
            use_payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_USE, use_ref, _scopes(source)[2], use_payload))
        ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        used = ctx.memory_interact_aggregates.read(ref, access=access)
        assert used is not None and used.use_count == 1

        ctx.memory_interact_aggregates.store.clear_all()
        assert ctx.memory_interact_aggregates.read(ref, access=access) is None
        report = ctx.memory_interact_aggregates.rebuild_all(access=access)
        assert report.scanned_event_count > 0
        restored = ctx.memory_interact_aggregates.read(ref, access=access)
        assert restored == used
    finally:
        backend.close()


def test_dirty_queue_processes_only_changed_hypothesis():
    """两个候选同时存在时，新增证据只处理对应 dirty key。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        first = _hypothesis(source, 1)
        second = _hypothesis(source, 2)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(first)
        ledger.register(second)
        access = MemoryAccessContext(1, 2, 3)
        initial = ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        assert initial.processed_hypothesis_count == 2
        ledger.append_evidence(EvidenceRecord(
            301, first, EVIDENCE_SUPPORT, (901,), source, 1, (902,)))
        report = ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        assert report.processed_hypothesis_count == 1
        first_record = ctx.memory_interact_aggregates.read(
            _hypothesis_ref(ctx, first), access=access)
        second_record = ctx.memory_interact_aggregates.read(
            _hypothesis_ref(ctx, second), access=access)
        assert first_record is not None and first_record.support_count == 1
        assert second_record is not None and second_record.support_count == 0
        assert second_record.evidence_state == MEMORY_EVIDENCE_PROVISIONAL
    finally:
        backend.close()


def test_aggregate_acl_does_not_process_or_reveal_private_dirty_key():
    """错误 owner 只能看到空结果，不能通过 dirty rebuild 触碰私有聚合。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        hypothesis = _hypothesis(source, 1)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(hypothesis)
        private_ref = _hypothesis_ref(ctx, hypothesis)
        wrong_access = MemoryAccessContext(1, 2, 4)
        assert ctx.memory_interact_aggregates.rebuild_dirty(
            access=wrong_access).processed_hypothesis_count == 0
        assert ctx.memory_interact_aggregates.read(
            private_ref, access=wrong_access) is None
        assert ctx.memory_interact_aggregates.query(access=wrong_access) == ()
        assert ctx.memory_interact_aggregates.read(
            private_ref, access=MemoryAccessContext(1, 2, 3)) is None
    finally:
        backend.close()
