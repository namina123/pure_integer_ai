"""来源局部投影不能无限跨轮重开；断言、完整竞争与显式依赖不得丢失。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, OBJECT_CONTEXT_SCOPE, concept_identity
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef, MemoryEvent, EpisodePayload, MEMORY_EVENT_EPISODE
from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
from pure_integer_ai.cognition.shared.generation_observed_surface import DeliveredGraphSurface
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,session_scope,LogicalClockIdentity,CLOCK_MEMORY_CREATED,
)
from pure_integer_ai.cognition.understanding.memory_intake import HypothesisIntakeDraft, ObservationIntakeDraft
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.current_reference_scope import read_current_reference_scope
from pure_integer_ai.experiments.memory_reference_read_scope import (
    retain_reference_role_dependencies, verify_reference_read_scope,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_graph_query_bridge import _root_response_context_inputs
from pure_integer_ai.experiments.trained_response_delivery import _delivery_proof_identity
from tests.test_memory_provenance_inputs import _context


PROJECTION = frame_key(91502, (1,), (4,1,1), (72,1), (73,1), (74,1), (75,1))
GRAPH = frame_key(91502, (11,), (72,2))
CLOSED = frame_key(91502, (5,), (73,1), (74,1), (76,1), (77,1), (2,2,2,1,0,2,0))


def _append(memory, ordinal, candidates, *, links=()):
    """以真实intake保存完整多分支来源，测试不凭数量或交付状态赋权。"""
    context = MemoryLinkedRef.object(ObjectIdentity(OBJECT_CONTEXT_SCOPE,
        (91501,memory.session_id,1),memory.owner,memory.candidate_index.source.versions))
    result = memory.append('来源'+str(ordinal),speaker_kind=1,observation_draft=ObservationIntakeDraft(
        (91,ordinal),context,relation_occurrences=links,hypotheses=tuple(
            HypothesisIntakeDraft((92,ordinal,n),(93,1),key,(94,n),3,reason_key=(95,n))
            for n,key in enumerate(candidates,1))))
    return memory.intake.result_for_source(result.source).observation_ref


def _delivered_context(memory, inp, number):
    """真实保存有图对象的通用交付，避免无对象回应按原规则延续到空焦点。"""
    source=observation_reference_source(inp)
    events=memory.intake.event_log
    target=concept_identity((73,number),owner=memory.owner,versions=source.versions)
    act=concept_identity((74,number),owner=memory.owner,versions=source.versions)
    surface=DeliveredGraphSurface(target,source,document_scope(source),(71,),0,1,(65,),(72,),((73,),))
    slots=frame_key(2,frame_key(1,(1,),surface.stable_key()))
    proof_key=frame_key(91530,(3,),source.stable_key(),inp.stable_key(),act.stable_key(),frame_key(1),slots)
    digest=events.scoped_identities.registry.register(91530,proof_key)
    proof=MemoryLinkedRef.object(_delivery_proof_identity(source,digest))
    candidate=frame_key(91502,(9,),proof.value().stable_key(),act.stable_key())
    draft=ObservationIntakeDraft((75,number),MemoryLinkedRef.object(ObjectIdentity(
        OBJECT_CONTEXT_SCOPE,(91501,memory.session_id,2),memory.owner,source.versions)),
        relation_occurrences=(proof,MemoryLinkedRef.object(target)),hypotheses=(
            HypothesisIntakeDraft((76,number),(77,1),candidate,(77,2),3,reason_key=(77,3)),))
    result=memory.append('x',speaker_kind=2,observation_draft=draft)
    output=memory.intake.result_for_source(result.source).observation_ref
    scope=document_scope(result.source)
    created=events.scoped_identities.resume_clock(LogicalClockIdentity(scope,CLOCK_MEMORY_CREATED)).advance()
    episode=EpisodePayload(inp,MemoryLinkedRef.object(act),(),proof,MemoryLinkedRef.memory(output),(),(),None,
        result.turn_seq,session_scope(memory.session_id,owner=memory.owner,versions=source.versions),created)
    ref=events.declaration_ref(MEMORY_EVENT_EPISODE,episode,scope=scope)
    events.append(MemoryEvent(MEMORY_EVENT_EPISODE,ref,scope,episode))
    return replace(_context(memory,inp,output),graph_surfaces=((1,surface),))


@pytest.fixture
def history(tmp_path):
    """两个可竞争交付、两个未交付来源、当前输入共同处在原候选账本。"""
    with TrainedDialogueMemoryGraph(tmp_path/'scope.sqlite3',session_id=88) as memory:
        first = _append(memory,1,(PROJECTION,))
        c1 = _delivered_context(memory,first,1)
        second = _append(memory,2,(PROJECTION,))
        c2 = _delivered_context(memory,second,2)
        assertion = _append(memory,3,(PROJECTION,CLOSED))
        mention = _append(memory,4,(PROJECTION,))
        graph = _append(memory,5,(PROJECTION,GRAPH))
        current = _append(memory,6,(PROJECTION,),links=memory._preceding_response_context_links())
        contexts = (c1,c2)
        scope = read_current_reference_scope(memory,current,contexts)
        active = memory.active_structures(candidate_keys=(PROJECTION,),observation_refs=(current,),
            context_depth_limit=1,recall_delivered_outputs=True)
        active = _root_response_context_inputs(memory,active,contexts)
        yield SimpleNamespace(memory=memory,current=current,contexts=contexts,scope=scope,
            active=active,mention=mention,assertion=assertion,first=first,second=second,graph=graph)


def _selected(h, contexts=None):
    """经生产使用的同一准入边界，不修改底层声明。"""
    return retain_reference_role_dependencies(h.memory,h.active,
        h.contexts if contexts is None else contexts,h.current,h.scope)


def _keys(active, observation):
    """恢复完整来源下的实际H引用。"""
    return {item.hypothesis_key for item in active.hypotheses
            if item.observation_key==observation.stable_key()}


def test_full_candidates_and_assertion_sources_survive_while_local_mentions_wait(history):
    """无交付的真实闭合断言来源保护全部投影；旧普通提及不获指代读取权。"""
    h=history
    before=h.memory.backend.count('memory_event')
    selected=_selected(h)
    assert selected.observations==h.active.observations
    assert set(h.active.match_routes)<set(selected.match_routes)
    assert _keys(selected,h.assertion)==_keys(h.active,h.assertion)
    assert _keys(selected,h.current)==_keys(h.active,h.current)
    assert not _keys(selected,h.mention)
    assert not _keys(selected,h.first) and not _keys(selected,h.second)
    for context in h.contexts:
        assert context.hypothesis.stable_key() in {item.hypothesis_key for item in selected.hypotheses}
    assert selected.evidence==tuple(item for item in h.active.evidence
        if item.hypothesis_key in {entry.hypothesis_key for entry in selected.hypotheses})
    assert h.memory.backend.count('memory_event')==before
    again=_selected(h)
    assert again==selected


def test_explicit_context_hypothesis_is_retained_even_outside_current_focus(history):
    """保留全部候选context的明示H，不能按最终winner缩小竞争分母。"""
    h=history
    dependency=next(item for item in h.active.hypotheses
                    if item.observation_key==h.first.stable_key()).hypothesis_key
    from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
    original=h.contexts[0]
    context=SimpleNamespace(stable_key=original.stable_key,
        memory_refs=lambda:(*original.memory_refs(),MemoryObjectRef.from_stable_key(dependency)))
    selected=_selected(h,(context,h.contexts[1]))
    assert dependency in {item.hypothesis_key for item in selected.hypotheses}


@pytest.mark.parametrize('fault',('none','empty','other_observation','missing_context'))
def test_missing_scope_preserves_original_and_damaged_scope_rejects(history,fault):
    """不在当前合法指代作用域内时不能使用新准入规则。"""
    h=history
    scope={'none':None,'empty':replace(h.scope,context_keys=()),
           'other_observation':replace(h.scope,observation=h.mention),
           'missing_context':replace(h.scope,context_keys=((999,),))}[fault]
    if fault in ('none','empty'):
        assert retain_reference_role_dependencies(h.memory,h.active,h.contexts,h.current,scope) is h.active
    else:
        with pytest.raises(ValueError):
            retain_reference_role_dependencies(h.memory,h.active,h.contexts,h.current,scope)


def test_explicit_graph_input_without_delivery_protects_source(history):
    """显式图对象不因没有Episode而丢失其余来源内分支。"""
    h=history
    selected=_selected(h)
    assert _keys(selected,h.graph)==_keys(h.active,h.graph)


@pytest.mark.parametrize('fault',('valid','omit_match','deferred_consumed','missing_consumption','changed_scope'))
def test_cold_read_scope_checks_full_members_and_consumption(history,monkeypatch,fault):
    """冷审重建原物理来源；改变候选分母、分割或实际消费都拒绝。"""
    h=history
    selected=_selected(h)
    routes=list(selected.match_routes)
    consumed={item.hypothesis_key for item in selected.hypotheses}
    proof_index=next(i for i,r in enumerate(routes) if r[0]==6)
    fields=list(unpack_record(routes[proof_index],6))
    if fault=='omit_match':
        original=unpack_record(fields[3],1)
        deferred=unpack_record(fields[5],3)
        removed=deferred[0]
        fields[3]=frame_key(1,*(k for k in original if k!=removed))
        fields[5]=frame_key(3,*(k for k in deferred if k!=removed))
    elif fault=='deferred_consumed':
        consumed.add(unpack_record(fields[5],3)[0])
    elif fault=='missing_consumption':
        consumed.pop()
    elif fault=='changed_scope':
        fields[2]=(123,)
    routes[proof_index]=frame_key(6,*fields)
    trace=SimpleNamespace(memory_routes=lambda:tuple(routes),hops=lambda:tuple(
        SimpleNamespace(space=2,predicate_key=(3,),proposition_key=k) for k in sorted(consumed)))
    def declaration(ref):
        """仅从真实SQLite读取唯一声明，测试不合成H或来源。"""
        row,=tuple(item for item in h.memory.intake.event_log.query(
            access=h.memory.candidate_index.access,object_ref=ref) if item.event.is_declaration)
        return row
    monkeypatch.setattr('pure_integer_ai.cognition.understanding.query_reference_context.extended_reference_context_required',lambda *_args:True)
    def verify():
        """除受训输入操作前置外，执行完整现役冷审。"""
        return verify_reference_read_scope(h.memory,trace,h.contexts,h.current,
            input_structure=object(),question_frames=(),declaration=declaration)
    if fault=='valid':
        assert verify()['status']=='MATCHED_PHYSICAL_REFERENCE_ROLE_READ_SCOPE'
    else:
        with pytest.raises(ValueError):
            verify()
