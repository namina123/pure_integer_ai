"""Evidence 独立记录身份、完整声明权威和三个生产入口的承重验证。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT, EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import OBJECT_CONTEXT_SCOPE, ObjectIdentity
from pure_integer_ai.cognition.shared.memory_event import (
    EvidencePayload, MEMORY_EVENT_EVIDENCE, MEMORY_OBJECT_EVIDENCE,
    MemoryEvent, MemoryLinkedRef, MemoryObjectRef, memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import MemoryHypothesisEventSink
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_record_identity import evidence_record_key
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft, ObservationIntakeDraft,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CATEGORY_TIME, MEMORY_GENERIC_CANDIDATE_VERSION, frame_key,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph, _VERSIONS,
)
from pure_integer_ai.experiments.trained_memory_candidate_index import (
    CANDIDATE_CURSOR_TABLE, CANDIDATE_INDEX_TABLE,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_CHUNK_WIDTH, MEMORY_EVENT_PART_TABLE, MemoryEventIntegrityError,
    MemoryEventRecordStore,
)
from test_m03_memory_event import (
    CLOCK_MEMORY_CREATED, CLOCK_MEMORY_OBSERVED, _backend, _hypothesis,
    _source, _timestamp,
)


_ACCESS = MemoryAccessContext(1, 2, 3)
_SCOPE_VECTOR = (1, 1, 11, 1, 2, 3, 4, 1, 2, 3, 4,
                 11, 1, 10, 11, 1, 2, 3, 4, 1, 2, 3, 4, 0)
_SOURCE_VECTOR = (1, 10, 11, 1, 2, 3, 4, 1, 2, 3, 4)
_TIMESTAMP_VECTOR = (1, 7, 27, 1, 6, 24, *_SCOPE_VECTOR)


def test_evidence_record_identity_language_neutral_vectors_and_h00_boundary():
    """固定整数向量定义原生时钟及 ledger/id，不把 candidate 或证据内容当身份。"""
    observed = LogicalTimestamp.from_stable_key(_TIMESTAMP_VECTOR)
    assert evidence_record_key(observed) == (91574, 2, 3, 1, 30, *_TIMESTAMP_VECTOR)
    hypothesis = _hypothesis(_source(), 1)
    record = EvidenceRecord(101, hypothesis, EVIDENCE_SUPPORT, (9,), _source(), 6)
    expected = (91574, 2, 3, 2, 1, 41, 24, *_SCOPE_VECTOR,
                11, *_SOURCE_VECTOR, 101)
    assert evidence_record_key(observed, record.stable_key()) == expected
    changed = replace(record, hypothesis=replace(hypothesis, candidate_key=(999,)),
                      payload=(7, 8), reason_key=(10,))
    assert evidence_record_key(observed, changed.stable_key()) == expected
    other = replace(record, hypothesis=replace(hypothesis, hypothesis_kind=(42,)))
    assert evidence_record_key(observed, other.stable_key()) != expected
    assert evidence_record_key(observed, replace(record, evidence_id=102).stable_key()) != expected


@pytest.mark.parametrize("mutation", ["bool", "list", "truncated", "trailing", "clock", "sequence"])
def test_evidence_record_identity_rejects_invalid_compatibility_or_clock(mutation):
    """非法兼容帧和非 observed 时钟不能定义新证据身份。"""
    hypothesis = _hypothesis(_source(), 1)
    record = EvidenceRecord(101, hypothesis, EVIDENCE_SUPPORT, (9,), _source(), 6)
    observed = _timestamp(hypothesis.scope, CLOCK_MEMORY_OBSERVED, 7)
    key = record.stable_key()
    if mutation == "bool":
        key = (True, *key[1:])
    elif mutation == "list":
        key = list(key)
    elif mutation == "truncated":
        key = key[:-1]
    elif mutation == "trailing":
        key = (*key, 0)
    elif mutation == "clock":
        observed = _timestamp(hypothesis.scope, CLOCK_MEMORY_CREATED, 7)
    else:
        observed = _timestamp(hypothesis.scope, CLOCK_MEMORY_OBSERVED, 8)
    with pytest.raises((TypeError, ValueError)):
        evidence_record_key(observed, key)


def _declared_hypothesis(ctx):
    """通过真实 H-00 sink 声明候选并返回完整引用及领域记录。"""
    hypothesis = _hypothesis(_source(), 1)
    sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
    sink.append_hypothesis(hypothesis)
    return hypothesis, sink, sink._hypothesis_ref(hypothesis)


def _evidence(ctx, hypothesis, ref, *, seq, previous=None, legacy=False, compatibility=()):
    """构造并经事件日志追加完整证据，可显式保留旧 V1 身份。"""
    payload = EvidencePayload(
        ref, EVIDENCE_SUPPORT, None, (91,), _source(), None,
        (23025, seq), previous,
        _timestamp(hypothesis.scope, CLOCK_MEMORY_OBSERVED, seq), compatibility)
    events = ctx.memory_interact_events
    object_ref = (memory_object_ref(
        events.memory_space_identity, MEMORY_OBJECT_EVIDENCE, payload.stable_key(),
        owner=ref.owner, versions=ref.versions) if legacy else events.declaration_ref(
            MEMORY_EVENT_EVIDENCE, payload, scope=hypothesis.scope))
    return events.append(MemoryEvent(
        MEMORY_EVENT_EVIDENCE, object_ref, hypothesis.scope, payload))


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_evidence_mixed_identity_supersedes_idempotence_and_conflict(kind):
    """旧前驱、新记录及完整声明可混读，同记录合法内容漂移仍须拒绝。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        hypothesis, _, ref = _declared_hypothesis(ctx)
        old = _evidence(ctx, hypothesis, ref, seq=2, legacy=True)
        new = _evidence(ctx, hypothesis, ref, seq=3, previous=old.event.object_ref)
        last = _evidence(ctx, hypothesis, ref, seq=4, previous=new.event.object_ref)
        assert old.event.object_ref.object_key[0] == 1
        assert new.event.object_ref.object_key[:4] == (91574, 2, 3, 1)
        assert len(last.event.object_ref.stable_key()) == len(new.event.object_ref.stable_key())
        for item in (old, new, last):
            assert ctx.memory_interact_events.append(item.event) == item
            assert MemoryEvent.from_stable_key(item.event.stable_key()) == item.event
        changed = replace(new.event.payload, detail=(777,))
        assert ctx.memory_interact_events.declaration_ref(
            MEMORY_EVENT_EVIDENCE, changed, scope=hypothesis.scope) == new.event.object_ref
        with pytest.raises(MemoryEventIntegrityError, match="不同声明"):
            ctx.memory_interact_events.append(replace(new.event, payload=changed))
        assert len(ctx.memory_interact_events.query(
            access=_ACCESS, event_kind=MEMORY_EVENT_EVIDENCE)) == 3
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_h00_old_record_is_reused_and_equal_timestamp_distinct_ids_survive(kind):
    """重开 sink 复用旧兼容引用，同时间的两个 evidence_id 恢复为不同记录。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        hypothesis, sink, ref = _declared_hypothesis(ctx)
        first = EvidenceRecord(101, hypothesis, EVIDENCE_SUPPORT, (91,), _source(), 1,
                               (23025, 2))
        old = _evidence(ctx, hypothesis, ref, seq=2, legacy=True,
                        compatibility=first.stable_key())
        sink.append_evidence(first)
        second = replace(first, evidence_id=102, supersedes_evidence_id=101)
        sink.append_evidence(second)
        third = replace(first, evidence_id=103, payload=(999,))
        sink.append_evidence(third)
        reopened = MemoryHypothesisEventSink(ctx.memory_interact_events)
        reopened.append_evidence(first)
        reopened.append_evidence(second)
        ledger = reopened.load_ledger(access=_ACCESS, hypotheses=(hypothesis,))
        assert ledger.evidence_history(hypothesis) == (first, second, third)
        rows = ctx.memory_interact_events.query(access=_ACCESS, event_kind=MEMORY_EVENT_EVIDENCE)
        assert len(rows) == 3 and old in rows
        assert len({item.event.object_ref for item in rows}) == 3
        assert rows[1].event.object_ref.object_key[:4] == (91574, 2, 3, 2)
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["marker", "detail", "compatibility"])
def test_evidence_compact_reference_does_not_hide_corrupt_complete_declaration(corruption):
    """记录身份不含内容时仍核完整声明 hash，禁止引用掩盖载荷或 marker 损坏。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        hypothesis, sink, _ = _declared_hypothesis(ctx)
        record = EvidenceRecord(101, hypothesis, EVIDENCE_SUPPORT, (91,), _source(), 1,
                                (23025001,))
        sink.append_evidence(record)
        entry, = ctx.memory_interact_events.query(access=_ACCESS, event_kind=MEMORY_EVENT_EVIDENCE)
        values = MemoryEventRecordStore(backend).read_payload(entry.event_hash)
        position = 2 if corruption == "marker" else values.index(23025001)
        if corruption == "compatibility":
            position = len(values) - 1
        chunk_index, offset = divmod(position, MEMORY_EVENT_CHUNK_WIDTH)
        snapshot = backend.snapshot()
        chunk = next(row for row in snapshot[MEMORY_EVENT_PART_TABLE]
                     if row["event_hash"] == entry.event_hash and row["chunk_index"] == chunk_index)
        chunk[f"part_{offset:02d}"] += 1
        backend.load_snapshot(snapshot)
        with pytest.raises((MemoryEventIntegrityError, ValueError)):
            ctx.memory_interact_events._require_declared_object(entry.event.object_ref)
    finally:
        backend.close()


def test_real_intake_and_evidence_append_reopen_after_derived_index_removed(tmp_path):
    """真实摄入和追加链在删除本测试派生索引后恢复全部记录与显式替代边。"""
    path = tmp_path / "evidence.sqlite3"
    expected = []
    with TrainedDialogueMemoryGraph(path, session_id=23025) as memory:
        draft = ObservationIntakeDraft(
            (23025, 0), MemoryLinkedRef.object(ObjectIdentity(
                OBJECT_CONTEXT_SCOPE, (23025,), memory.owner, _VERSIONS)),
            hypotheses=(HypothesisIntakeDraft(
                (23025, 1), (23025, 2),
                frame_key(MEMORY_GENERIC_CANDIDATE_VERSION, (MEMORY_CATEGORY_TIME,), (23025, 3)),
                (23025, 4), 1,
                reason_key=(23025, 5), detail=tuple(range(23025000, 23025064))),))
        appended = memory.append("证据记录", speaker_kind=2, observation_draft=draft)
        result = memory.intake.result_for_source(appended.source)
        hypothesis = result.hypothesis_refs[0]
        previous = result.evidence_refs[0]
        for index in range(1, 4):
            record = memory.append_evidence(
                hypothesis.stable_key(), source=appended.source, stance=1 + index % 3,
                reason_key=(23025, 5), detail=tuple(range(23025000 + index * 64,
                                                       23025064 + index * 64)),
                supersedes_key=previous.stable_key())
            previous = MemoryObjectRef.from_stable_key(record.evidence_key)
        expected = memory.intake.event_log.query(
            access=memory.candidate_index.access, event_kind=MEMORY_EVENT_EVIDENCE)
        assert len(expected) == 4
        assert len({len(item.event.object_ref.stable_key()) for item in expected}) == 1
        for table in (CANDIDATE_INDEX_TABLE, CANDIDATE_CURSOR_TABLE):
            memory.backend.delete(table, memory.candidate_index.selector)
        memory.backend.commit()
    with TrainedDialogueMemoryGraph(path, session_id=23025) as memory:
        actual = memory.intake.event_log.query(
            access=memory.candidate_index.access, event_kind=MEMORY_EVENT_EVIDENCE)
        assert actual == expected
        assert memory.intake.result_for_source(appended.source).evidence_refs == result.evidence_refs
        assert memory.backend.count(CANDIDATE_INDEX_TABLE, memory.candidate_index.selector) > 0
