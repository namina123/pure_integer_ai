"""A-02 来源化内容、生命周期、容量及 typed adapter 对抗测试。"""
from __future__ import annotations

import copy

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_EPISODE,
    SCOPE_QUERY,
    SCOPE_SESSION,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity,
    event_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentError,
    WorkMemoryContentItem,
    WorkMemoryContentProtocol,
    WorkMemoryOccurrenceAnchor,
    WorkMemoryRoleDefinition,
)
from pure_integer_ai.cognition.shared.work_memory_discourse import (
    WorkMemoryDiscourseRoles,
    project_generation_plans_to_work_memory,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_reference_memory import (
    project_occurrence_reference_to_work_memory,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from tests.test_a01_occurrence_reference import _claim, _request
from tests.test_f00_reference_question_runtime import _reference_fixture
from tests.test_g02_generation_structure_plan import (
    _discourse,
    _propositions,
    _request as _generation_request,
    _selection,
)


_BASE = 19800


def _source(document_id: int) -> SourceRef:
    """构造 owner/version 相同但文档身份不同的测试来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 1,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _index():
    """建立可物化并回读 occurrence 图身份的独立内存上下文。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((_BASE + 2, 1), (_BASE + 2, 2)),
    )
    return backend, ctx, index


def _record(index, source: SourceRef, raw_text: str, position: int):
    """在指定来源建立一个可回源的独立 occurrence。"""
    scope = document_scope(source)
    return index.record(
        source=source,
        raw_text=raw_text,
        scope=scope,
        start=position,
        end=position + 1,
        ordinal=0,
        segment_index=position,
        local_index=0,
        document_index=position,
    )


def _anchor(index, record) -> WorkMemoryOccurrenceAnchor:
    """从真实 OccurrenceIndex 读模型构造完整图锚。"""
    return WorkMemoryOccurrenceAnchor(
        record.occurrence,
        index.ontology.identity_of(record.occurrence),
        record.source,
        record.scope,
    )


def _role(source: SourceRef, ordinal: int):
    """构造由测试协议注入、未写死具体语义的 Role。"""
    return role_identity(
        (_BASE + 3, ordinal),
        owner=source.owner,
        versions=source.versions,
    )


def _definition(role, kinds, lifespan, maximum):
    """简化测试中的开放 Role 容量定义。"""
    return WorkMemoryRoleDefinition(role, tuple(kinds), lifespan, maximum)


def _start(
        work_memory: WorkMemory,
        source: SourceRef,
        *,
        session=None,
        episode_id: int = 1,
        ):
    """按 A-09 顺序打开 session、document 和 episode。"""
    current_session = session or session_scope(
        _BASE + 4,
        owner=source.owner,
        versions=source.versions,
    )
    if work_memory.active_session_scope is None:
        work_memory.begin_session(current_session)
    document = document_scope(source)
    episode = episode_scope(episode_id, parent=document)
    work_memory.begin_document(document)
    work_memory.begin_episode(episode)
    return current_session, document, episode


def _item(
        work_memory: WorkMemory,
        role,
        value,
        anchor,
        seq: int,
        *,
        supersedes=(),
        ) -> WorkMemoryContentItem:
    """使用 Role 当前精确 lifespan 建立一个来源化内容项。"""
    return WorkMemoryContentItem(
        role,
        value,
        anchor,
        work_memory.require_content_store().scope_for_role(role),
        seq,
        (_BASE + 5, seq),
        tuple(supersedes),
    )


def test_a02_keeps_occurrences_and_lifespans_separate_across_documents():
    """同值多 occurrence 不合并，各 lifespan 清理且 session 项保留原来源。"""
    backend, _ctx, index = _index()
    source_a = _source(1)
    source_b = _source(2)
    entity_role = _role(source_a, 1)
    event_role = _role(source_a, 2)
    goal_role = _role(source_a, 3)
    work_memory = WorkMemory()
    work_memory.configure_content(WorkMemoryContentProtocol((
        _definition(entity_role, (OBJECT_ENTITY,), SCOPE_DOCUMENT, 2),
        _definition(event_role, (OBJECT_EVENT,), SCOPE_EPISODE, 1),
        _definition(goal_role, (OBJECT_PROPOSITION,), SCOPE_SESSION, 1),
    ), 16))
    try:
        first = _record(index, source_a, "甲甲", 0)
        second = _record(index, source_a, "甲甲", 1)
        session, document_a, _episode_a = _start(work_memory, source_a)
        same_entity = entity_identity(source_a, (_BASE + 6, 1))
        first_item = work_memory.put_content(_item(
            work_memory, entity_role, same_entity, _anchor(index, first), 1))
        second_item = work_memory.put_content(_item(
            work_memory, entity_role, same_entity, _anchor(index, second), 2))
        work_memory.put_content(_item(
            work_memory,
            event_role,
            event_identity(source_a, (_BASE + 6, 2)),
            _anchor(index, first),
            3,
        ))
        old_goal = work_memory.put_content(_item(
            work_memory,
            goal_role,
            proposition_identity(source_a, (_BASE + 6, 3)),
            _anchor(index, first),
            4,
        ))

        assert first_item.content_ref() != second_item.content_ref()
        assert len(work_memory.active_content(role=entity_role)) == 2
        work_memory.end_episode()
        assert work_memory.active_content(role=event_role) == ()
        assert len(work_memory.active_content(role=entity_role)) == 2
        work_memory.end_document()
        assert work_memory.active_content(role=entity_role) == ()
        assert work_memory.active_content(role=goal_role) == (old_goal,)

        third = _record(index, source_b, "乙", 0)
        _start(work_memory, source_b, session=session, episode_id=2)
        new_goal = work_memory.put_content(_item(
            work_memory,
            goal_role,
            proposition_identity(source_b, (_BASE + 6, 4)),
            _anchor(index, third),
            5,
            supersedes=(old_goal.content_ref(),),
        ))
        assert work_memory.active_content(role=goal_role) == (new_goal,)
        history = work_memory.content_history(role=goal_role)
        assert tuple(item.source for item in history) == (source_a, source_b)
        work_memory.end_episode()
        work_memory.end_document()
        assert work_memory.content_history(role=goal_role) == history
        work_memory.end_session()
        assert work_memory.content_history() == ()
        assert document_a.source == source_a
    finally:
        backend.close()


def test_a02_capacity_fails_closed_and_supersede_is_explicit():
    """active/history 超限均在首写前失败，exact replay 不增行也不 FIFO 私删。"""
    backend, _ctx, index = _index()
    source = _source(3)
    role = _role(source, 4)
    with pytest.raises(ValueError, match="Role 身份版本"):
        _definition(
            ObjectIdentity(OBJECT_ROLE, (_BASE + 7, 0)),
            (OBJECT_ENTITY,),
            SCOPE_DOCUMENT,
            1,
        )
    work_memory = WorkMemory()
    work_memory.configure_content(WorkMemoryContentProtocol((
        _definition(role, (OBJECT_ENTITY,), SCOPE_DOCUMENT, 1),
    ), 2))
    try:
        first = _record(index, source, "甲乙", 0)
        second = _record(index, source, "甲乙", 1)
        _start(work_memory, source)
        with pytest.raises(ValueError, match="语义 value 与 occurrence 来源"):
            _item(
                work_memory,
                role,
                entity_identity(_source(99), (_BASE + 7, 9)),
                _anchor(index, first),
                1,
            )
        initial = work_memory.put_content(_item(
            work_memory,
            role,
            entity_identity(source, (_BASE + 7, 1)),
            _anchor(index, first),
            1,
        ))
        before = work_memory.content_state_key()
        with pytest.raises(WorkMemoryContentError, match="active 容量"):
            work_memory.put_content(_item(
                work_memory,
                role,
                entity_identity(source, (_BASE + 7, 2)),
                _anchor(index, second),
                2,
            ))
        assert work_memory.content_state_key() == before

        replacement = work_memory.put_content(_item(
            work_memory,
            role,
            entity_identity(source, (_BASE + 7, 2)),
            _anchor(index, second),
            2,
            supersedes=(initial.content_ref(),),
        ))
        state = work_memory.content_state_key()
        assert work_memory.put_content(replacement) == replacement
        assert work_memory.content_state_key() == state
        with pytest.raises(WorkMemoryContentError, match="history 容量"):
            work_memory.put_content(_item(
                work_memory,
                role,
                entity_identity(source, (_BASE + 7, 3)),
                _anchor(index, first),
                3,
                supersedes=(replacement.content_ref(),),
            ))
        assert work_memory.content_state_key() == state
        assert work_memory.active_content(role=role) == (replacement,)
    finally:
        work_memory.abort_episode()
        work_memory.end_document()
        work_memory.end_session()
        backend.close()


def test_a02_abort_and_clone_do_not_leak_nested_or_host_state():
    """clone 独立清理 query/episode 内容，abort 不删除 session 项且不影响宿主。"""
    backend, _ctx, index = _index()
    source = _source(4)
    goal_role = _role(source, 5)
    event_role = _role(source, 6)
    question_role = _role(source, 7)
    work_memory = WorkMemory()
    work_memory.configure_content(WorkMemoryContentProtocol((
        _definition(goal_role, (OBJECT_PROPOSITION,), SCOPE_SESSION, 1),
        _definition(event_role, (OBJECT_EVENT,), SCOPE_EPISODE, 1),
        _definition(question_role, (OBJECT_PROPOSITION,), SCOPE_QUERY, 1),
    ), 8))
    try:
        record = _record(index, source, "甲", 0)
        _session_value, _document, episode = _start(work_memory, source)
        query = query_scope(1, parent=episode)
        work_memory.begin_query(query)
        anchor = _anchor(index, record)
        work_memory.put_content(_item(
            work_memory,
            goal_role,
            proposition_identity(source, (_BASE + 8, 1)),
            anchor,
            1,
        ))
        work_memory.put_content(_item(
            work_memory,
            event_role,
            event_identity(source, (_BASE + 8, 2)),
            anchor,
            2,
        ))
        work_memory.put_content(_item(
            work_memory,
            question_role,
            proposition_identity(source, (_BASE + 8, 3)),
            anchor,
            3,
        ))
        host_state = work_memory.content_state_key()
        cloned = copy.deepcopy(work_memory)

        cloned.end_query()
        cloned.abort_episode()

        assert cloned.active_content(role=question_role) == ()
        assert cloned.active_content(role=event_role) == ()
        assert len(cloned.active_content(role=goal_role)) == 1
        assert work_memory.content_state_key() == host_state
        assert len(work_memory.active_content(role=question_role)) == 1
    finally:
        work_memory.end_query()
        work_memory.abort_episode()
        work_memory.end_document()
        work_memory.end_session()
        backend.close()


def test_a01_adapter_writes_only_singleton_adopted_occurrence():
    """A-01 多 adopted 零写，定向 Evidence 形成 singleton 后才写 typed antecedent。"""
    world = _reference_fixture(ambiguous=True)
    target = WorkMemory()
    source = world.current.source
    role = _role(source, 8)
    target.configure_content(WorkMemoryContentProtocol((
        _definition(role, (OBJECT_OCCURRENCE,), SCOPE_EPISODE, 1),
    ), 4))
    target.begin_session(session_scope(
        _BASE + 9,
        owner=source.owner,
        versions=source.versions,
        source=source,
    ))
    target.begin_document(world.current.occurrence_scope)
    target.begin_episode(world.current.runtime_scope)
    try:
        ambiguous = world.reference_runtime.resolve(
            _request(world, timestamp_seq=10))
        assert project_occurrence_reference_to_work_memory(
            target,
            world.ctx.occurrence_index,
            ambiguous,
            role=role,
            logical_seq=10,
            trace=(_BASE + 10, 1),
        ) is None
        assert target.active_content(role=role) == ()

        first, second = world.candidate_occurrences
        reference = world.current.occurrences[0]
        resolved = world.reference_runtime.resolve(_request(
            world,
            timestamp_seq=11,
            evidence=(
                _claim(
                    world,
                    first,
                    dimension=1,
                    stance=EVIDENCE_SUPPORT,
                    timestamp_seq=11,
                    reason=1,
                    visible_occurrences=(first, reference),
                ),
                _claim(
                    world,
                    second,
                    dimension=1,
                    stance=EVIDENCE_REFUTE,
                    timestamp_seq=11,
                    reason=2,
                    visible_occurrences=(second, reference),
                ),
            ),
        ))
        item = project_occurrence_reference_to_work_memory(
            target,
            world.ctx.occurrence_index,
            resolved,
            role=role,
            logical_seq=11,
            trace=(_BASE + 10, 2),
        )
        assert item is not None
        assert item.value == resolved.winner.antecedent_identity
        state = target.content_state_key()
        replay = project_occurrence_reference_to_work_memory(
            target,
            world.ctx.occurrence_index,
            resolved,
            role=role,
            logical_seq=11,
            trace=(_BASE + 10, 2),
        )
        assert replay == item
        assert target.content_state_key() == state
    finally:
        target.end_episode()
        target.end_document()
        target.end_session()
        world.close()


def test_g02_adapter_projects_typed_context_questions_and_propositions():
    """G-02 三类 typed 输出按来源 anchor 原子写入并服从各自 lifespan。"""
    request, _unresolved = _generation_request(with_unresolved=True, count=1)
    selection, _selector, _content = _selection(request)
    discourse = _discourse(selection)
    propositions = _propositions(selection)
    source = request.goal.source
    backend, _ctx, index = _index()
    context_role = _role(source, 9)
    question_role = _role(source, 10)
    center_role = _role(source, 11)
    roles = WorkMemoryDiscourseRoles(
        context_role, question_role, center_role)
    work_memory = WorkMemory()
    work_memory.configure_content(WorkMemoryContentProtocol((
        _definition(context_role, (OBJECT_CONCEPT,), SCOPE_DOCUMENT, 4),
        _definition(question_role, (OBJECT_PROPOSITION,), SCOPE_QUERY, 4),
        _definition(center_role, (OBJECT_PROPOSITION,), SCOPE_EPISODE, 4),
    ), 16))
    try:
        record = _record(index, source, "问", 0)
        _session_value, _document, episode = _start(work_memory, source)
        work_memory.begin_query(query_scope(2, parent=episode))
        projection = project_generation_plans_to_work_memory(
            work_memory,
            discourse,
            propositions,
            roles=roles,
            anchors=(_anchor(index, record),),
            logical_seq_start=20,
            trace=(_BASE + 11, 1),
        )

        assert len(projection.items) == 3
        assert len(work_memory.active_content(role=context_role)) == 1
        assert len(work_memory.active_content(role=question_role)) == 1
        assert len(work_memory.active_content(role=center_role)) == 1
        state = work_memory.content_state_key()
        replay = project_generation_plans_to_work_memory(
            work_memory,
            discourse,
            propositions,
            roles=roles,
            anchors=(_anchor(index, record),),
            logical_seq_start=20,
            trace=(_BASE + 11, 1),
        )
        assert replay == projection
        assert work_memory.content_state_key() == state

        work_memory.end_query()
        assert work_memory.active_content(role=question_role) == ()
        assert len(work_memory.active_content(role=center_role)) == 1
        work_memory.end_episode()
        assert work_memory.active_content(role=center_role) == ()
        assert len(work_memory.active_content(role=context_role)) == 1
    finally:
        if work_memory.active_query_scope is not None:
            work_memory.end_query()
        if work_memory.active_episode_scope is not None:
            work_memory.abort_episode()
        if work_memory.active_document_scope is not None:
            work_memory.end_document()
        if work_memory.active_session_scope is not None:
            work_memory.end_session()
        backend.close()
