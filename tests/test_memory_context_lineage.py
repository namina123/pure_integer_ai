"""物理 Episode 的输入可作语境，完整采用依赖不能递归激活无关历史。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_CONTEXT_SCOPE, ObjectIdentity, concept_identity
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION, EpisodePayload, MemoryEvent, MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED, LogicalClockIdentity, LogicalTimestamp, document_scope, session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import HypothesisIntakeDraft, ObservationIntakeDraft
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.memory_context_lineage import delivered_context_links
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_response_delivery import (
    DELIVERY_PROOF_IDENTITY, _delivery_proof_identity, read_delivery_proof,
)


@pytest.fixture
def delivery(tmp_path):
    """保存两次输入、一次输出和唯一 Episode，包含明确的旧采用依赖。"""
    with TrainedDialogueMemoryGraph(tmp_path/'context.sqlite3',session_id=71) as memory:
        prior = memory.append_graph_input(((72,1),))
        current = memory.append_graph_input(((72,2),))
        old_ref = memory.intake.result_for_source(prior.source).observation_ref
        input_ref = memory.intake.result_for_source(current.source).observation_ref
        proof = MemoryLinkedRef.object(concept_identity((91530,71),owner=memory.owner,versions=current.source.versions))
        candidate = frame_key(91503,(6,),(71,2))
        links = (proof,MemoryLinkedRef.memory(input_ref),MemoryLinkedRef.memory(old_ref))
        draft = ObservationIntakeDraft((71,1),MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,(91501,71,2),memory.owner,current.source.versions)),relation_occurrences=links,
            hypotheses=(HypothesisIntakeDraft((71,2),(71,3),candidate,(71,4),3,reason_key=(71,5)),))
        output = memory.append('x',speaker_kind=2,observation_draft=draft)
        output_ref = memory.intake.result_for_source(output.source).observation_ref
        scope = document_scope(output.source)
        episode = EpisodePayload(input_ref,None,tuple(MemoryLinkedRef.memory(r) for r in (input_ref,old_ref)),proof,
            MemoryLinkedRef.memory(output_ref),(input_ref,old_ref),(),None,output.turn_seq,
            session_scope(71,owner=memory.owner,versions=output.source.versions),
            LogicalTimestamp(LogicalClockIdentity(scope,CLOCK_MEMORY_CREATED),1))
        events = memory.intake.event_log
        ref = events.declaration_ref(MEMORY_EVENT_EPISODE,episode,scope=scope)
        events.append(MemoryEvent(MEMORY_EVENT_EPISODE,ref,scope,episode))
        rows = events.query(access=memory.candidate_index.access,event_kind=MEMORY_EVENT_OBSERVATION,object_ref=output_ref)
        payload = rows[0].event.payload
        yield memory,output_ref,payload,episode,old_ref,input_ref


def test_real_closure_uses_episode_input_without_replaying_proof_dependencies(delivery):
    """调用真实恢复器及物理 Episode 索引，原 output/used_refs 完整保留。"""
    memory,output,payload,episode,old,current = delivery
    before = payload.stable_key()
    episodes = memory.response_episodes_for(output)
    assert episodes == (episode,)
    links = delivered_context_links(output,payload,episodes)
    assert MemoryLinkedRef.memory(current) in links and MemoryLinkedRef.memory(old) not in links
    rows,_,_,_ = memory._observation_closure((),(output,),memory.candidate_index.access,context_depth_limit=1)
    assert {r.event.object_ref for r in rows} == {output,current}
    assert payload.stable_key() == before and old in episode.used_memory_refs
    assert MemoryLinkedRef.memory(old) in payload.relation_occurrences


def test_ordinary_observation_keeps_its_original_memory_edges(delivery):
    """无交付 Episode 的普通观察不会被静默改写或失去自己的上下文边。"""
    _,output,payload,_,_,_ = delivery
    assert delivered_context_links(output,payload,()) == payload.relation_occurrences


@pytest.mark.parametrize('damage',('duplicate','wrong_output','missing_proof','missing_input','foreign_input'))
def test_invalid_episode_cannot_authorize_context_edges(delivery,damage):
    """缺成员、输出替换、竞争 Episode 和跨来源输入继续失败关闭。"""
    _,output,payload,episode,old,current = delivery
    episodes = (episode,)
    if damage=='duplicate':
        episodes = (episode,episode)
    elif damage=='wrong_output':
        episodes = (replace(episode,output_ref=MemoryLinkedRef.memory(old)),)
    elif damage=='missing_proof':
        payload = replace(payload,relation_occurrences=tuple(l for l in payload.relation_occurrences if l!=episode.selected_path_ref))
    elif damage=='missing_input':
        payload = replace(payload,relation_occurrences=tuple(l for l in payload.relation_occurrences if l!=MemoryLinkedRef.memory(current)))
    else:
        episodes = (replace(episode,input_observation_ref=output),)
    with pytest.raises(ValueError):
        delivered_context_links(output,payload,episodes)


@pytest.fixture
def generic_delivery(tmp_path):
    """按原有bounded-generic格式保存真实proof、缺直接输入边的输出及物理Episode。"""
    with TrainedDialogueMemoryGraph(tmp_path/'generic.sqlite3', session_id=72) as memory:
        prior = memory.append_graph_input(((73,1),))
        current = memory.append_graph_input(((73,2),))
        old = memory.intake.result_for_source(prior.source).observation_ref
        inp = memory.intake.result_for_source(current.source).observation_ref
        act = concept_identity((73,3), owner=memory.owner, versions=current.source.versions)
        fields = ((3,),current.source.stable_key(),inp.stable_key(),act.stable_key(),
                  frame_key(1,old.stable_key()),frame_key(2))
        key = frame_key(DELIVERY_PROOF_IDENTITY,*fields)
        events = memory.intake.event_log
        proof_hash = events.scoped_identities.registry.register(DELIVERY_PROOF_IDENTITY,key)
        proof = MemoryLinkedRef.object(_delivery_proof_identity(current.source,proof_hash))
        candidate = frame_key(91503,(6,),(73,2))
        draft = ObservationIntakeDraft((73,4),MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,(91501,72,2),memory.owner,current.source.versions)),
            relation_occurrences=(proof,MemoryLinkedRef.memory(old)),
            hypotheses=(HypothesisIntakeDraft((73,5),(73,6),candidate,(73,7),3,reason_key=(73,8)),))
        output = memory.append('x',speaker_kind=2,observation_draft=draft)
        out = memory.intake.result_for_source(output.source).observation_ref
        scope = document_scope(output.source)
        episode = EpisodePayload(inp,MemoryLinkedRef.object(act),(MemoryLinkedRef.memory(old),),proof,
            MemoryLinkedRef.memory(out),(old,),(),None,output.turn_seq,
            session_scope(72,owner=memory.owner,versions=output.source.versions),
            LogicalTimestamp(LogicalClockIdentity(scope,CLOCK_MEMORY_CREATED),1))
        ref = events.declaration_ref(MEMORY_EVENT_EPISODE,episode,scope=scope)
        events.append(MemoryEvent(MEMORY_EVENT_EPISODE,ref,scope,episode))
        row, = events.query(access=memory.candidate_index.access,event_kind=MEMORY_EVENT_OBSERVATION,object_ref=out)
        yield memory,out,row.event.payload,episode,old,inp,fields


def test_original_generic_proof_restores_its_input_as_derived_context_only(generic_delivery):
    """真实恢复器消费原proof中的输入，原输出关系及采用集合逐字节保持。"""
    memory,out,payload,episode,old,inp,_ = generic_delivery
    before = payload.stable_key(),episode.stable_key()
    links = delivered_context_links(out,payload,(episode,),proof_reader=lambda ref: read_delivery_proof(memory,ref))
    assert MemoryLinkedRef.memory(inp) in links and MemoryLinkedRef.memory(old) not in links
    assert MemoryLinkedRef.memory(inp) not in payload.relation_occurrences
    rows,_,_,_ = memory._observation_closure((),(out,),memory.candidate_index.access,context_depth_limit=1)
    assert {r.event.object_ref for r in rows} == {out,inp}
    assert (payload.stable_key(),episode.stable_key()) == before


@pytest.mark.parametrize('damage',('input','source','act','used','kind','reader'))
def test_generic_without_direct_input_requires_exact_original_proof(generic_delivery,damage):
    """输入、来源、回应行为、采用集合或格式不符，不能以Episode自报授权语境。"""
    _,out,payload,episode,old,_,fields = generic_delivery
    damaged = list(fields)
    if damage=='input':
        damaged[2] = old.stable_key()
    elif damage=='source':
        from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
        damaged[1] = observation_reference_source(old).stable_key()
    elif damage=='act':
        damaged[3] = concept_identity((73,99),owner=out.owner,versions=out.versions).stable_key()
    elif damage=='used':
        damaged[4] = frame_key(1)
    elif damage=='kind':
        damaged[0] = (4,)
    reader = None if damage=='reader' else lambda _ref: frame_key(DELIVERY_PROOF_IDENTITY,*damaged)
    with pytest.raises(ValueError):
        delivered_context_links(out,payload,(episode,),proof_reader=reader)
