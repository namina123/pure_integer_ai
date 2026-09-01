"""训练后对话的 interaction Memory 图跨进程专项。"""
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    DIALOGUE_MEMORY_POSTING_TABLE,
    TrainedDialogueMemoryGraph,
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
