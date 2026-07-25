"""A-09 WorkMemory 生命周期、边界清理和异常收口对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    generation_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.work_memory import (
    WorkMemory,
    WorkMemoryScopeError,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT


def _scope(document_id: int):
    """构造测试所需的稳定文档 scope。"""
    versions = VersionBundle(
        CorpusVersion(1),
        ParserVersion(2),
        PrimitiveVersion(3),
        CurriculumVersion(4),
    )
    source = SourceRef(
        SOURCE_BARE_TEXT,
        10,
        document_id,
        GLOBAL_OWNER_SCOPE,
        versions,
    )
    return document_scope(source)


def _session():
    """构造不依赖墙钟的测试 session scope。"""
    return session_scope(
        1,
        versions=VersionBundle(
            CorpusVersion(1),
            ParserVersion(2),
            PrimitiveVersion(3),
            CurriculumVersion(4),
        ),
    )


def test_lifecycle_requires_scope_and_parent_order():
    """没有 scope 或父边界时必须拒绝打开生命周期。"""
    work_memory = WorkMemory()
    with pytest.raises(WorkMemoryScopeError, match="ScopeIdentity"):
        work_memory.begin_session(None)
    with pytest.raises(WorkMemoryScopeError, match="活动 session"):
        work_memory.begin_document(_scope(1))

    work_memory.begin_session(_session())
    document = _scope(1)
    with pytest.raises(WorkMemoryScopeError, match="父生命周期"):
        work_memory.begin_episode(episode_scope(1, parent=document))


def test_document_reset_does_not_delete_session_bridge():
    """无关文档不能继承临时状态，但稳定键桥接仍由 session 持有。"""
    work_memory = WorkMemory()
    work_memory.begin_session(_session())
    document_a = _scope(1)
    document_b = _scope(2)
    work_memory.begin_document(document_a)
    work_memory.begin_episode(episode_scope(1, parent=document_a), round_id=7)
    work_memory.produced_refs[:] = [(1, 11)]
    work_memory.prior_topic_refs[:] = [(1, 12)]
    work_memory.dangling_units.add((1, 13))
    work_memory.lang_skeleton_by_item[(101, 0)] = (1, 99)
    work_memory.begin_segment(0)
    work_memory.next_occurrence_ordinal()
    work_memory.end_segment([(1, 14)])
    work_memory.end_episode()
    work_memory.end_document()

    work_memory.begin_document(document_b)
    work_memory.begin_episode(episode_scope(2, parent=document_b), round_id=8)
    assert work_memory.produced_refs == []
    assert work_memory.prior_topic_refs == []
    assert work_memory.dangling_units == set()
    assert work_memory.prior_segments() == []
    assert work_memory.lang_skeleton_by_item[(101, 0)] == (1, 99)


def test_replay_is_explicitly_carried_only_within_one_document():
    """回放只能由 episode 结束时转移到同文档的下一 episode。"""
    work_memory = WorkMemory()
    work_memory.begin_session(_session())
    document = _scope(3)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode_scope(1, parent=document))
    work_memory.replay_candidates[:] = [(1, 20)]
    work_memory.exclude_refs.add((1, 21))
    work_memory.end_episode()

    work_memory.begin_episode(episode_scope(2, parent=document))
    assert work_memory.replay_candidates == [(1, 20)]
    assert work_memory.exclude_refs == {(1, 21)}
    work_memory.end_episode()
    work_memory.end_document()


def test_occurrence_ordinal_is_monotone_and_does_not_merge_repeated_ref():
    """同一概念 ref 的多次 occurrence 仍各自获得序号。"""
    work_memory = WorkMemory()
    work_memory.begin_session(_session())
    document = _scope(4)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode_scope(1, parent=document))
    work_memory.begin_segment(0)
    first = work_memory.next_occurrence_ordinal()
    second = work_memory.next_occurrence_ordinal()
    work_memory.end_segment([(1, 30), (1, 30)])
    work_memory.begin_segment(1)
    third = work_memory.next_occurrence_ordinal()
    work_memory.end_segment([(1, 30)])
    assert (first, second, third) == (1, 2, 3)
    assert work_memory.prior_segments()[0][1] == ((1, 30),)


def test_query_and_generation_close_clear_only_their_owned_state():
    """query 向量和 generation 槽位关闭后不可进入下一 query。"""
    work_memory = WorkMemory()
    work_memory.begin_session(_session())
    document = _scope(5)
    episode = episode_scope(1, parent=document)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    query = query_scope(1, parent=episode)
    work_memory.begin_query(query)
    work_memory.pr_vector[(1, 1)] = 7
    generation = generation_scope(1, parent=query)
    work_memory.begin_generation(generation)
    work_memory.current_slot_idx = 4
    work_memory.current_cue_sig = ((1, 2),)
    work_memory.end_generation()
    assert work_memory.current_slot_idx == 0
    assert work_memory.current_cue_sig == ()
    work_memory.end_query()
    assert work_memory.pr_vector == {}
    work_memory.end_episode()
    work_memory.end_document()


def test_abort_episode_clears_nested_state_and_drops_replay():
    """异常中止必须清理 segment/query/generation 及待继承回放。"""
    work_memory = WorkMemory()
    work_memory.begin_session(_session())
    document = _scope(6)
    episode = episode_scope(1, parent=document)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    work_memory.replay_candidates[:] = [(1, 40)]
    work_memory.begin_segment(0)
    work_memory.abort_episode()
    assert work_memory.active_episode_scope is None
    assert work_memory.active_query_scope is None
    assert work_memory.active_generation_scope is None
    assert work_memory.active_segment_index is None
    assert work_memory.replay_candidates == []
    assert work_memory.exclude_refs == set()
    assert work_memory.prior_segments() == []
    work_memory.end_document()
    work_memory.end_session()
