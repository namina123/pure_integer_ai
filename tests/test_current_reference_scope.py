"""当前指向须来自持久Observation，完整历史窗口不能成为同级当前作用域。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle,
)
from pure_integer_ai.cognition.shared.memory_event import (
    LogicalTimestamp, MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
    MEMORY_OBJECT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED, LogicalClockIdentity, document_scope,
)
from pure_integer_ai.experiments.current_reference_scope import (
    CURRENT_REFERENCE_SCOPE, read_current_reference_scope,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


def _fixture():
    """保留两个真实格式来源边，另以显式边声明当前指向，不以深度挑选。"""
    source = SourceRef(41, 7302, 5, GLOBAL_OWNER_SCOPE, VersionBundle())
    older, active = replace(source, document_id=1), replace(source, document_id=3)
    observation = MemoryObjectRef(SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1),
        GLOBAL_OWNER_SCOPE, VersionBundle(), MEMORY_OBJECT_OBSERVATION, (1,))
    links = tuple(MemoryLinkedRef.object(context_scope_identity(item, (91543,))) for item in (older, active))
    focus = MemoryLinkedRef.object(context_scope_identity(active, (CURRENT_REFERENCE_SCOPE,)))
    payload = ObservationPayload(source, MemoryLinkedRef.object(context_scope_identity(source, (1,))),
        (), (), None, (), (*links, focus),
        LogicalTimestamp(LogicalClockIdentity(document_scope(source), CLOCK_MEMORY_CREATED), 5))
    contexts = tuple(SimpleNamespace(source_ref=(7, *item.stable_key()),
        proof_ref=context_scope_identity(item, (2,)), context_depth=1,
        stable_key=lambda index=index: (91532,index)) for index,item in enumerate((older,active)))
    memory = SimpleNamespace(candidate_index=SimpleNamespace(access=object()),
        intake=SimpleNamespace(event_log=SimpleNamespace(query=lambda **_kw:
            (SimpleNamespace(event=SimpleNamespace(payload=payload)),))))
    return memory, observation, contexts, payload, active


def test_current_scope_keeps_background_candidates_and_uses_only_explicit_members():
    """历史与当前上下文同深度仍按来源边区分，输入与历史成员保持。"""
    memory, observation, contexts, payload, _active = _fixture()
    before = payload.stable_key()
    scope = read_current_reference_scope(memory, observation, contexts)
    assert not scope.permits(contexts[0])
    assert scope.permits(contexts[1])
    assert len(contexts) == 2 and payload.stable_key() == before
    assert scope.stable_key()
    assert read_current_reference_scope(memory, observation, ()).context_keys == ()


@pytest.mark.parametrize('fault', ('legacy', 'missing_history', 'future', 'duplicate'))
def test_scope_missing_or_drift_cannot_authorize_a_historical_object(fault):
    """旧版本保持原行为，损坏的新来源边不能静默回退到背景对象。"""
    memory, observation, contexts, payload, active = _fixture()
    links = payload.relation_occurrences
    if fault == 'legacy':
        updated = replace(payload, relation_occurrences=links[:-1])
    elif fault == 'missing_history':
        updated = replace(payload, relation_occurrences=(links[-1],))
    elif fault == 'future':
        future = replace(active, document_id=8)
        updated = replace(payload, relation_occurrences=(
            MemoryLinkedRef.object(context_scope_identity(future, (91543,))),
            MemoryLinkedRef.object(context_scope_identity(future, (CURRENT_REFERENCE_SCOPE,)))))
    else:
        updated = replace(payload, relation_occurrences=(*links,
            MemoryLinkedRef.object(context_scope_identity(replace(active,document_id=1), (CURRENT_REFERENCE_SCOPE,)))))
    memory.intake.event_log.query = lambda **_kw: (SimpleNamespace(event=SimpleNamespace(payload=updated)),)
    if fault == 'legacy':
        assert read_current_reference_scope(memory, observation, contexts) is None
    else:
        with pytest.raises(ValueError):
            read_current_reference_scope(memory, observation, contexts)


def test_memory_context_window_persists_a_separate_current_source_link():
    """生产写入器保留完整窗口，并独立追加由Episode来源序转移的指向。"""
    _memory, _observation, _contexts, payload, active = _fixture()
    older = replace(active, document_id=1)
    reader = SimpleNamespace(_preceding_response_sources=lambda **_kw:(active,older))
    links = TrainedDialogueMemoryGraph._preceding_response_context_links(reader,history_limit=8)
    assert len(links) == 3
    assert set(payload.relation_occurrences) == set(links)
