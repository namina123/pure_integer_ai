"""真实SQLite索引/闭包和独立冷恢复必须只采用结构命中输入的实际输出。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, OBJECT_CONTEXT_SCOPE, concept_identity
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, MemoryEvent, MemoryLinkedRef, MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope, session_scope, LogicalClockIdentity, CLOCK_MEMORY_CREATED,
)
from pure_integer_ai.cognition.understanding.memory_intake import ObservationIntakeDraft, HypothesisIntakeDraft
from pure_integer_ai.cognition.understanding.query_input_structure import QueryInputStructure
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key, memory_input_candidate_keys
from pure_integer_ai.experiments.memory_delivered_recall import verify_delivered_recall
from pure_integer_ai.experiments.saved_memory_recall_anchors import read_saved_recall_observation_anchors
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_response_delivery import _delivery_proof_identity, read_delivery_proof
from pure_integer_ai.experiments.trained_memory_candidate_index import episode_input_route


def _output(memory, given, number, *, used=()):
    """写入原bounded-generic proof及真实Episode，允许包含无关审计依赖。"""
    inp = memory.intake.result_for_source(given.source).observation_ref
    events = memory.intake.event_log
    act = concept_identity((73, number), owner=memory.owner, versions=given.source.versions)
    proof_key = frame_key(91530, (3,), given.source.stable_key(), inp.stable_key(), act.stable_key(),
                          frame_key(1, *(ref.stable_key() for ref in used)), frame_key(2))
    digest = events.scoped_identities.registry.register(91530, proof_key)
    proof = MemoryLinkedRef.object(_delivery_proof_identity(given.source, digest))
    candidate = frame_key(91503, (9,), proof.value().stable_key(), act.stable_key())
    draft = ObservationIntakeDraft((74, number), MemoryLinkedRef.object(ObjectIdentity(
        OBJECT_CONTEXT_SCOPE, (91501, memory.session_id, 2), memory.owner, given.source.versions)),
        relation_occurrences=(proof, *(MemoryLinkedRef.memory(ref) for ref in used)),
        hypotheses=(HypothesisIntakeDraft((75, number), (75, 1), candidate, (75, 2), 3, reason_key=(75, 3)),))
    result = memory.append('x', speaker_kind=2, observation_draft=draft)
    out = memory.intake.result_for_source(result.source).observation_ref
    scope = document_scope(result.source)
    created = events.scoped_identities.resume_clock(LogicalClockIdentity(scope, CLOCK_MEMORY_CREATED)).advance()
    payload = EpisodePayload(inp, MemoryLinkedRef.object(act), (), proof, MemoryLinkedRef.memory(out), used,
        (), None, result.turn_seq, session_scope(memory.session_id, owner=memory.owner, versions=result.source.versions), created)
    ref = events.declaration_ref(MEMORY_EVENT_EPISODE, payload, scope=scope)
    events.append(MemoryEvent(MEMORY_EVENT_EPISODE, ref, scope, payload))
    return out, ref


@pytest.fixture
def history(tmp_path):
    """两个相关输入、两个无关输入和未来输入交错，当前原窗口只含最新输出。"""
    with TrainedDialogueMemoryGraph(tmp_path/'history.sqlite3', session_id=77) as memory:
        noise = memory.append_graph_input(((72, 2),))
        noise_ref = memory.intake.result_for_source(noise.source).observation_ref
        noise_out, _ = _output(memory, noise, 1)
        first = memory.append_graph_input(((72, 1),))
        first_out, first_episode = _output(memory, first, 2, used=(noise_ref,))
        later_noise = memory.append_graph_input(((72, 2),))
        later_noise_out, _ = _output(memory, later_noise, 3)
        second = memory.append_graph_input(((72, 1),))
        second_out, _ = _output(memory, second, 4)
        current = memory.append_graph_input(((72, 1),))
        current_ref = memory.intake.result_for_source(current.source).observation_ref
        structure = QueryInputStructure((1,), (), (1,), (), (), (), (), (), (), graph_object_keys=((72, 1),))
        candidates = memory_input_candidate_keys(structure, session_id=77)
        def declarations(ref):
            """只回读原唯一声明，冷恢复不访问候选索引。"""
            return tuple(row for row in memory.intake.event_log.query(access=memory.candidate_index.access, object_ref=ref)
                         if row.event.is_declaration)
        yield SimpleNamespace(memory=memory, first=first, first_out=first_out, first_episode=first_episode,
            second_out=second_out, current=current, current_ref=current_ref, candidates=candidates,
            noise_outputs=(noise_out,later_noise_out), declarations=declarations)


def _active(h, enabled=True):
    """实际生产Memory API，原预算边界和当前锚点保持。"""
    return h.memory.active_structures(candidate_keys=h.candidates, observation_refs=(h.current_ref,),
        context_depth_limit=1, recall_delivered_outputs=enabled)


def test_real_index_and_closure_recover_all_actual_outputs_without_old_prior_turns(history):
    """新输入反向Episode索引生效；旧前序/审计依赖不成为新输出锚点。"""
    h = history
    base, selected = _active(h, False), _active(h)
    baseline = {row.observation_key for row in base.observations}
    actual = {row.observation_key for row in selected.observations}
    assert h.first_out.stable_key() not in baseline and h.second_out.stable_key() in baseline
    assert {h.first_out.stable_key(),h.second_out.stable_key()} <= actual
    assert not actual.intersection(ref.stable_key() for ref in h.noise_outputs)
    assert all(row.context_depth==0 for row in selected.observations
               if row.observation_key in {h.first_out.stable_key(),h.second_out.stable_key()})
    matches = tuple(h.memory.candidate_index.matching_events((episode_input_route(
        h.memory.intake.result_for_source(h.first.source).observation_ref.stable_key()),)))
    assert len(matches)==1 and matches[0].event.object_ref==h.first_episode


def test_independent_cold_recovery_matches_all_derived_output_anchors(history):
    """冷审独立读取原Episode与proof，不借待审context自报来源。"""
    h = history
    selected = _active(h)
    trace = SimpleNamespace(memory_routes=lambda: selected.match_routes)
    rows = read_saved_recall_observation_anchors(h.memory, trace, input_observation=h.current_ref,
        input_candidate_keys=h.candidates, declaration_reader=h.declarations)
    refs = {row.event.object_ref for row in rows}
    assert {h.first_out,h.second_out} <= refs
    routes = list(selected.match_routes)
    routes.remove(next(route for route in routes if route[0]==5))
    with pytest.raises(QueryMemoryOheEvidenceError, match='omitted'):
        read_saved_recall_observation_anchors(h.memory, SimpleNamespace(memory_routes=lambda: routes),
            input_observation=h.current_ref,input_candidate_keys=h.candidates,declaration_reader=h.declarations)


@pytest.mark.parametrize('damage',('proof_input','proof_action','episode_output','source_order'))
def test_changed_proof_episode_or_current_source_cannot_authorize_recall(history,damage):
    """保留原声明，只向验证器提交损坏成员，缺任一完整绑定必须关闭。"""
    h = history
    inp = h.declarations(h.memory.intake.result_for_source(h.first.source).observation_ref)[0]
    ep = h.declarations(h.first_episode)[0]
    out = h.declarations(h.first_out)[0]
    source = h.current.source
    reader = lambda ref: read_delivery_proof(h.memory,ref)
    if damage.startswith('proof_'):
        fields=list(unpack_record(reader(ep.event.payload.selected_path_ref.value()),91530))
        fields[2 if damage=='proof_input' else 3]=h.current_ref.stable_key()
        reader=lambda _ref:frame_key(91530,*fields)
    elif damage=='episode_output':
        ep=replace(ep,event=replace(ep.event,payload=replace(ep.event.payload,output_ref=MemoryLinkedRef.memory(h.second_out))))
    else:
        source=h.first.source
    with pytest.raises(ValueError):
        verify_delivered_recall(inp,ep,out,source,proof_reader=reader)


def test_new_recall_requires_one_current_bounded_content_query(history):
    """不能把新模式用于完整历史枚举或无当前输入的普通查询。"""
    h=history
    with pytest.raises(ValueError,match='bounded content'):
        h.memory.active_structures(candidate_keys=h.candidates,context_depth_limit=None,recall_delivered_outputs=True)
