"""训练后对话的 interaction Memory 图跨进程专项。"""
import pytest

from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    DIALOGUE_MEMORY_POSTING_TABLE,
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload,
    MEMORY_EVENT_EPISODE,
    MEMORY_OBJECT_EPISODE,
    MemoryEvent,
    MemoryLinkedRef,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    LogicalClockIdentity,
    session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    ObservationIntakeDraft,
)


def test_fresh_memory_owner_supports_bounded_case_commit_and_rollback(tmp_path):
    """新 owner 先提交 schema，再允许真实 Memory append 受 case 事务管理。"""
    database = tmp_path / "bounded-session.sqlite3"
    failed = TrainedDialogueMemoryGraph(database, session_id=23)
    backend = failed.backend
    initial_events = backend.count("memory_event")
    assert initial_events == 0
    assert not backend._conn.in_transaction
    with pytest.raises(RuntimeError, match="rollback probe"):
        with backend.bounded_case(max_database_bytes=64 * 1024**2):
            failed.append("rollback probe", speaker_kind=1)
            raise RuntimeError("rollback probe")
    assert not backend._conn.in_transaction
    assert backend.count("memory_event") == initial_events
    failed.abandon()

    with TrainedDialogueMemoryGraph(database, session_id=23) as restored:
        with restored.backend.bounded_case(max_database_bytes=64 * 1024**2) as case:
            restored.append("committed probe", speaker_kind=1)
            case.commit()
        assert not restored.backend._conn.in_transaction
        assert restored.backend.count("memory_event") > initial_events


def test_dialogue_surface_enters_memory_graph_and_recalls_after_reopen(tmp_path):
    """用户表层必须先成为 Memory Observation，重开后才能由 posting 召回。"""
    database = tmp_path / "session.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=7) as memory:
        appended = memory.append("我叫小明，住在北京。", speaker_kind=1)
        assert appended.source_hash > 0
        assert appended.posting_count > 0
        assert memory.backend.count("memory_event") >= 2
        assert memory.backend.count(DIALOGUE_MEMORY_POSTING_TABLE) == (
            appended.posting_count)

    with TrainedDialogueMemoryGraph(database, session_id=7) as restored:
        recent = restored.recent_turns()
        assert tuple((item.turn_seq, item.speaker_kind, item.surface)
                     for item in recent) == (
            (1, 1, "我叫小明，住在北京。"),)
        recalled = restored.recall(
            "还记得我叫小明吗？", minimum_similarity_permille=250)
        assert recalled is not None
        assert recalled.surface == "我叫小明，住在北京。"
        assert recalled.source_hash == appended.source_hash
        assert recalled.posting_reads > 0
        assert restored.intake.require_current_manifest(
            recalled.source).source == recalled.source


def test_dialogue_memory_postings_are_isolated_by_full_owner(tmp_path):
    """同一文件中的不同 tenant/user/session 不得互相召回表层。"""
    database = tmp_path / "owners.sqlite3"
    with TrainedDialogueMemoryGraph(
            database, tenant_id=2, user_id=3, session_id=5) as first:
        first.append("跨进程保留的独立会话内容。", speaker_kind=1)

    for owner in ((7, 3, 5), (2, 11, 5), (2, 3, 13)):
        with TrainedDialogueMemoryGraph(
                database,
                tenant_id=owner[0],
                user_id=owner[1],
                session_id=owner[2],
                ) as isolated:
            assert isolated.recall(
                "独立会话内容还在吗？",
                minimum_similarity_permille=200,
            ) is None

    with TrainedDialogueMemoryGraph(
            database, tenant_id=2, user_id=3, session_id=5) as restored:
        assert restored.recall(
            "独立会话内容还在吗？",
            minimum_similarity_permille=200,
        ) is not None


def test_cross_turn_context_link_is_bounded_and_recoverable(tmp_path):
    """多轮输出承接只写紧凑 ContextScope，不递归复制 Observation 键。"""
    database = tmp_path / "bounded-context.sqlite3"
    input_lengths = []
    with TrainedDialogueMemoryGraph(database, session_id=11) as memory:
        for turn in range(1, 5):
            current = memory.append(f"input {turn}", speaker_kind=1)
            current_ref = memory.intake.result_for_source(current.source).observation_ref
            assert current_ref is not None
            input_lengths.append(len(current_ref.stable_key()))
            output = memory.append(f"output {turn}", speaker_kind=2)
            output_ref = memory.intake.result_for_source(output.source).observation_ref
            assert output_ref is not None
            session = session_scope(
                memory.session_id, owner=memory.owner,
                versions=current.source.versions)
            created = memory.intake.event_log.scoped_identities.resume_clock(
                LogicalClockIdentity(session, CLOCK_MEMORY_CREATED)).advance()
            episode_payload = EpisodePayload(
                current_ref, None, (), None, MemoryLinkedRef.memory(output_ref),
                (), (), None, output.turn_seq, session, created)
            episode_ref = memory_object_ref(
                memory.intake.event_log.memory_space_identity,
                MEMORY_OBJECT_EPISODE, episode_payload.stable_key(),
                owner=memory.owner, versions=current.source.versions)
            memory.intake.event_log.append(MemoryEvent(
                MEMORY_EVENT_EPISODE, episode_ref, session, episode_payload))
            if turn == 1:
                first_input, first_output = current_ref, output_ref

        assert input_lengths[-1] <= input_lengths[0] + 32
        assert memory._preceding_response_sources() == (output.source,)
        assert tuple(item.document_id for item in memory._preceding_response_sources(
            history_limit=2)) == (6, 8)
        assert memory.response_episodes_for(output_ref) == (episode_payload,)
        assert memory.response_episodes_for(output_ref) == (episode_payload,)
        active = memory.active_structures(observation_refs=(current_ref,))
        assert [item.turn_seq for item in active.observations] == list(range(1, 9))
        delayed = EpisodePayload(
            first_input, None, (), None, MemoryLinkedRef.memory(first_output),
            (), (), None, 2, session,
            memory.intake.event_log.scoped_identities.resume_clock(
                LogicalClockIdentity(session, CLOCK_MEMORY_CREATED)).advance())
        delayed_ref = memory_object_ref(
            memory.intake.event_log.memory_space_identity, MEMORY_OBJECT_EPISODE,
            delayed.stable_key(), owner=memory.owner, versions=current.source.versions)
        memory.intake.event_log.append(MemoryEvent(
            MEMORY_EVENT_EPISODE, delayed_ref, session, delayed))
        assert memory._preceding_response_sources() == (output.source,)
    with TrainedDialogueMemoryGraph(database, session_id=11) as restored:
        assert restored.response_episodes_for(output_ref) == (episode_payload,)


def test_response_episode_lookup_keeps_duplicate_outputs_visible(tmp_path):
    """增量索引不能把两个指向同一输出的 Episode 合并成唯一交付。"""
    database = tmp_path / "duplicate-response-episode.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=19) as memory:
        given = memory.append("input", speaker_kind=1)
        output = memory.append("output", speaker_kind=2)
        input_ref = memory.intake.result_for_source(given.source).observation_ref
        output_ref = memory.intake.result_for_source(output.source).observation_ref
        assert input_ref is not None and output_ref is not None
        session = session_scope(memory.session_id, owner=memory.owner,
                                versions=given.source.versions)
        clock = memory.intake.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(session, CLOCK_MEMORY_CREATED))
        for ordinal in (1, 2):
            payload = EpisodePayload(
                input_ref, None, (), None, MemoryLinkedRef.memory(output_ref),
                (), (), None, ordinal, session, clock.advance())
            ref = memory_object_ref(
                memory.intake.event_log.memory_space_identity, MEMORY_OBJECT_EPISODE,
                payload.stable_key(), owner=memory.owner, versions=given.source.versions)
            memory.intake.event_log.append(MemoryEvent(
                MEMORY_EVENT_EPISODE, ref, session, payload))
            assert len(memory.response_episodes_for(output_ref)) == ordinal


def test_observation_anchor_does_not_enumerate_unrelated_history(tmp_path):
    """A bounded anchored restore must materialize only the requested turn."""
    database = tmp_path / "anchored-restore.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=12) as memory:
        observations = []
        for turn in range(1, 7):
            appended = memory.append(f"isolated input {turn}", speaker_kind=1)
            observation = memory.intake.result_for_source(
                appended.source).observation_ref
            assert observation is not None
            observations.append(observation)

        active = memory.active_structures(
            observation_refs=(observations[-1],), context_depth_limit=0)
        assert [item.turn_seq for item in active.observations] == [6]


def test_anchored_restore_reuses_manifest_only_within_one_call(tmp_path, monkeypatch):
    """Evidence rows from one source share validation, while later calls recheck it."""
    database = tmp_path / "manifest-cache.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=31) as memory:
        appended = memory.append("我叫小明，住在北京。", speaker_kind=1)
        observation = memory.intake.result_for_source(
            appended.source).observation_ref
        assert observation is not None
        original = memory.intake.require_current_manifest
        calls: dict[tuple[int, ...], int] = {}

        def counted(source):
            key = source.stable_key()
            calls[key] = calls.get(key, 0) + 1
            return original(source)

        monkeypatch.setattr(memory.intake, "require_current_manifest", counted)
        snapshot = memory.active_structures(
            observation_refs=(observation,), context_depth_limit=0)
        assert len(snapshot.observations) == 1
        assert len(snapshot.hypotheses) == len(appended.hypothesis_keys)
        assert len(snapshot.evidence) == len(appended.evidence_keys)
        assert calls == {appended.source.stable_key(): 1}

        assert memory.active_structures(
            observation_refs=(observation,), context_depth_limit=0) == snapshot
        assert calls[appended.source.stable_key()] == 2
