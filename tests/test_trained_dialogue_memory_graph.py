"""训练后对话的 interaction Memory 图跨进程专项。"""
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

        assert input_lengths[-1] <= input_lengths[0] + 32
        active = memory.active_structures(observation_refs=(current_ref,))
        assert [item.turn_seq for item in active.observations] == list(range(1, 9))
