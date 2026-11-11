"""V2 记录本体、完整成员声明与 V1 混合恢复的承重验证。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE, MEMORY_OBJECT_USE, MemoryEvent, MemoryLinkedRef,
    ObservationPayload, UsePayload, memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_record_identity import (
    MEMORY_RECORD_IDENTITY, observation_reference_source,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph, _VERSIONS
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, OBJECT_CONTEXT_SCOPE
from pure_integer_ai.cognition.understanding.memory_intake import ObservationIntakeDraft
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_PART_TABLE, MEMORY_EVENT_CHUNK_WIDTH,
    MemoryEventRecordStore, MemoryEventIntegrityError,
)
from test_m03_memory_event import (
    _append_observation, _backend, _core_refs, _scopes, _source, _timestamp,
    CLOCK_MEMORY_CREATED, CLOCK_MEMORY_OBSERVED, CLOCK_MEMORY_USED,
)


_ACCESS = MemoryAccessContext(1, 2, 3)


def _record_event(ctx, event_kind, payload, scope):
    """通过真实 facade 建立新记录声明，再执行完整 append。"""
    ref = ctx.memory_interact_events.declaration_ref(event_kind, payload, scope=scope)
    return ctx.memory_interact_events.append(MemoryEvent(event_kind, ref, scope, payload))


def _mixed_events(ctx):
    """在同一空间创建旧 O、新 O、新 Episode 和真实正规化 Use。"""
    source = _source()
    refs = _core_refs(ctx)
    old = _append_observation(ctx, source, refs)
    session, _, scope = _scopes(source)
    payload = replace(old.event.payload, observed_at=_timestamp(scope, CLOCK_MEMORY_OBSERVED, 2),
                      relation_occurrences=(MemoryLinkedRef.memory(old.event.object_ref),))
    new = _record_event(ctx, MEMORY_EVENT_OBSERVATION, payload, scope)
    episode_payload = EpisodePayload(
        old.event.object_ref, None, (MemoryLinkedRef.memory(new.event.object_ref),), None,
        MemoryLinkedRef.memory(new.event.object_ref), (old.event.object_ref, new.event.object_ref),
        (), None, 3, session, _timestamp(scope, CLOCK_MEMORY_CREATED, 3))
    episode = _record_event(ctx, MEMORY_EVENT_EPISODE, episode_payload, scope)
    use_payload = UsePayload(
        new.event.object_ref, episode.event.object_ref, MemoryLinkedRef.core(refs[0]), None,
        _timestamp(scope, CLOCK_MEMORY_USED, 4))
    use_ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity, MEMORY_OBJECT_USE,
        use_payload.identity_key(), owner=source.owner, versions=source.versions)
    use = ctx.memory_interact_events.append(MemoryEvent(MEMORY_EVENT_USE, use_ref, scope, use_payload))
    return old, new, episode, use


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_record_identity_mixed_payloads_and_same_identity_content_change_are_checked(kind):
    """新键不携成员；完整事件仍可恢复，同身份改成员必须拒绝重新声明。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        old, new, episode, use = _mixed_events(ctx)
        assert old.event.object_ref.object_key[0] == 1
        assert new.event.object_ref.object_key[:3] == (MEMORY_RECORD_IDENTITY, 2, 1)
        assert episode.event.object_ref.object_key[:3] == (MEMORY_RECORD_IDENTITY, 2, 2)
        assert observation_reference_source(old.event.object_ref) == observation_reference_source(new.event.object_ref)
        records = MemoryEventRecordStore(backend)
        assert records.episode_output_source_key(records.read(episode.event_hash)) == _source().stable_key()
        for item in (old, new, episode, use):
            assert MemoryEvent.from_stable_key(item.event.stable_key()) == item.event
            assert ctx.memory_interact_events.read(item.event_hash, access=_ACCESS) == item
        changed = replace(new.event.payload, relation_occurrences=())
        assert ctx.memory_interact_events.declaration_ref(
            MEMORY_EVENT_OBSERVATION, changed, scope=new.event.scope) == new.event.object_ref
        with pytest.raises(MemoryEventIntegrityError, match="不同声明"):
            ctx.memory_interact_events.append(MemoryEvent(
                MEMORY_EVENT_OBSERVATION, new.event.object_ref, new.event.scope, changed))
    finally:
        backend.close()


def test_record_identity_cold_reopen_keeps_full_membership_and_use(tmp_path):
    """新进程式 owner 重建不依赖旧对象或解码缓存，完整成员/Use 一致。"""
    path = tmp_path / "records.sqlite3"
    backend = SQLiteBackend(str(path))
    try:
        ctx = make_train_context(backend)
        expected = _mixed_events(ctx)
        backend.commit()
    finally:
        backend.close()
    restored = SQLiteBackend(str(path))
    try:
        ctx = make_train_context(restored)
        for item in expected:
            assert ctx.memory_interact_events.read(item.event_hash, access=_ACCESS) == item
    finally:
        restored.close()


@pytest.mark.parametrize("position,value", [(0, 91539), (1, 4), (2, 3), (3, 2)])
def test_record_identity_corrupt_storage_marker_is_rejected(position, value):
    """正规化声明的未知 marker、种类、身份或载荷版本必须拒绝。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, new, _, use = _mixed_events(ctx)
        snapshot = backend.snapshot()
        first = next(row for row in snapshot[MEMORY_EVENT_PART_TABLE]
                     if row["event_hash"] == new.event_hash and row["chunk_index"] == 0)
        first[f"part_{position:02d}"] = value
        backend.load_snapshot(snapshot)
        with pytest.raises((MemoryEventIntegrityError, ValueError)):
            ctx.memory_interact_events.read(use.event_hash, access=_ACCESS)
    finally:
        backend.close()


def test_record_reference_checks_full_declaration_hash_after_nonidentity_payload_change():
    """来源/时钟身份未变时，合法整数成员漂移也必须由完整声明 hash 拒绝。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, new, _, use = _mixed_events(ctx)
        values = MemoryEventRecordStore(backend).read_payload(new.event_hash)
        # V2 storage 三整数头之后仍是 V1 payload；改 context 的完整 Core 身份。
        context_size_position = 3 + 1 + 1 + len(new.event.payload.source.stable_key())
        value_position = context_size_position + values[context_size_position]
        chunk_index, offset = divmod(value_position, MEMORY_EVENT_CHUNK_WIDTH)
        snapshot = backend.snapshot()
        chunk = next(row for row in snapshot[MEMORY_EVENT_PART_TABLE]
                     if row["event_hash"] == new.event_hash and row["chunk_index"] == chunk_index)
        chunk[f"part_{offset:02d}"] += 1
        backend.load_snapshot(snapshot)
        with pytest.raises((MemoryEventIntegrityError, ValueError)):
            ctx.memory_interact_events.read(use.event_hash, access=_ACCESS)
    finally:
        backend.close()


def test_real_source_intake_record_references_do_not_embed_previous_observation_payload(tmp_path):
    """实际同 session 摄入保留十二轮显式边，记录引用尺寸不随历史载荷递归增长。"""
    path = tmp_path / "memory.sqlite3"
    payload_sizes, ref_sizes = [], []
    last = None
    with TrainedDialogueMemoryGraph(path, session_id=23024) as memory:
        initial = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE, (23024,), memory.owner, _VERSIONS))
        for turn in range(12):
            draft = ObservationIntakeDraft(
                (23024, turn), context=initial,
                relation_occurrences=() if last is None else (MemoryLinkedRef.memory(last),))
            appended = memory.append(f"记录{turn}", speaker_kind=2, observation_draft=draft)
            result = memory.intake.result_for_source(appended.source)
            last = result.observation_ref
            declarations = memory.intake.event_log.query(
                access=memory.candidate_index.access, object_ref=last)
            assert len(declarations) == 1
            payload_sizes.append(len(declarations[0].event.payload.stable_key()))
            ref_sizes.append(len(last.stable_key()))
        assert len(set(ref_sizes)) == 1
        assert len(set(payload_sizes[1:])) == 1
    with TrainedDialogueMemoryGraph(path, session_id=23024) as memory:
        restored = memory.intake.event_log.query(access=memory.candidate_index.access,
                                               event_kind=MEMORY_EVENT_OBSERVATION)
        assert len(restored) == 12 and last in {item.event.object_ref for item in restored}
