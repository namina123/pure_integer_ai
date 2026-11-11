"""真实SQLite历史内容命中不应自动激活其当年的前序话轮语境。"""
from pure_integer_ai.cognition.shared.identity import OBJECT_CONTEXT_SCOPE, ObjectIdentity, concept_identity
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, MemoryEvent, MemoryLinkedRef, MEMORY_EVENT_EPISODE,
    MEMORY_OBJECT_EPISODE, memory_object_ref,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    session_scope, LogicalClockIdentity, CLOCK_MEMORY_CREATED,
)
from pure_integer_ai.cognition.understanding.query_input_structure import QueryInputStructure
from pure_integer_ai.cognition.understanding.memory_intake import HypothesisIntakeDraft, ObservationIntakeDraft
from pure_integer_ai.cognition.understanding.query_memory_candidates import memory_input_candidate_keys, frame_key
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph


def _episode(memory, given, ordinal):
    """输出明确保留实际输入、选择proof和采用成员，不能用残缺Episode测试联合资格。"""
    input_ref = memory.intake.result_for_source(given.source).observation_ref
    proof = MemoryLinkedRef.object(concept_identity((91530, 31, ordinal),
        owner=memory.owner,versions=given.source.versions))
    draft = ObservationIntakeDraft((71,ordinal),MemoryLinkedRef.object(ObjectIdentity(
        OBJECT_CONTEXT_SCOPE,(91501,memory.session_id,2),memory.owner,given.source.versions)),
        relation_occurrences=(proof,MemoryLinkedRef.memory(input_ref)),hypotheses=(HypothesisIntakeDraft(
            (71,ordinal),(71,3),frame_key(91503,(6,),(71,ordinal)),(71,4),3,reason_key=(71,5)),))
    output = memory.append('独立回应'+str(ordinal),speaker_kind=2,observation_draft=draft)
    output_ref = memory.intake.result_for_source(output.source).observation_ref
    session = session_scope(memory.session_id, owner=memory.owner, versions=given.source.versions)
    created = memory.intake.event_log.scoped_identities.resume_clock(
        LogicalClockIdentity(session, CLOCK_MEMORY_CREATED)).advance()
    payload = EpisodePayload(input_ref, None, (MemoryLinkedRef.memory(input_ref),), proof,
        MemoryLinkedRef.memory(output_ref),(input_ref,),(),None,output.turn_seq,session,created)
    ref = memory_object_ref(memory.intake.event_log.memory_space_identity, MEMORY_OBJECT_EPISODE,
                            payload.stable_key(), owner=memory.owner, versions=given.source.versions)
    memory.intake.event_log.append(MemoryEvent(MEMORY_EVENT_EPISODE, ref, session, payload))
    return input_ref, output_ref


def _given(memory):
    """同一图节点的全部历史输入都必须被内容索引保留。"""
    refs, outputs = [], []
    for number in (1, 2):
        given = memory.append_graph_input(((72, 1),))
        input_ref, output_ref = _episode(memory, given, number)
        refs.append(input_ref)
        outputs.append(output_ref)
    current = memory.append_graph_input(((72, 1),))
    refs.append(memory.intake.result_for_source(current.source).observation_ref)
    structure = QueryInputStructure((1,), (), (1,), (), (), (), (), (), (), graph_object_keys=((72, 1),))
    return tuple(refs), tuple(outputs), memory_input_candidate_keys(structure, session_id=memory.session_id)


def test_content_match_keeps_all_inputs_without_reactivating_their_prior_responses(tmp_path):
    """保留全部直接命中H/E，只引入当前输入的完整前序回应。"""
    with TrainedDialogueMemoryGraph(tmp_path/'current.sqlite3', session_id=31) as memory:
        inputs, outputs, candidates = _given(memory)
        active = memory.active_structures(candidate_keys=candidates,
            observation_refs=(inputs[-1],), context_depth_limit=1)
        keys = {row.observation_key for row in active.observations}
        assert keys == {ref.stable_key() for ref in (*inputs, outputs[-1])}
        for ref in inputs:
            hypothesis = [row for row in active.hypotheses if row.observation_key == ref.stable_key()]
            assert hypothesis
            assert all(any(entry.hypothesis_key == row.hypothesis_key for entry in active.evidence)
                       for row in hypothesis)


def test_each_explicit_current_anchor_retains_its_complete_prior_response(tmp_path):
    """明确加入两个输入锚点时，两份原语境都必须保留。"""
    with TrainedDialogueMemoryGraph(tmp_path/'multiple.sqlite3', session_id=31) as memory:
        inputs, outputs, candidates = _given(memory)
        active = memory.active_structures(candidate_keys=candidates,
            observation_refs=inputs[1:], context_depth_limit=1)
        assert {row.observation_key for row in active.observations} == {
            ref.stable_key() for ref in (*inputs, *outputs)}


def test_unbounded_diagnostic_preserves_complete_historical_context(tmp_path):
    """全图诊断保持旧合同，不能为了生产预算删除历史关系。"""
    with TrainedDialogueMemoryGraph(tmp_path/'all.sqlite3', session_id=31) as memory:
        inputs, outputs, candidates = _given(memory)
        active = memory.active_structures(candidate_keys=candidates,
            observation_refs=(inputs[-1],), context_depth_limit=None)
        assert {row.observation_key for row in active.observations} == {
            ref.stable_key() for ref in (*inputs, *outputs)}


def test_zero_depth_keeps_content_roots_and_no_implicit_prior_turn(tmp_path):
    """零深度不恢复前序回应，仍保留全部内容召回输入。"""
    with TrainedDialogueMemoryGraph(tmp_path/'zero.sqlite3', session_id=31) as memory:
        inputs, _outputs, candidates = _given(memory)
        active = memory.active_structures(candidate_keys=candidates,
            observation_refs=(inputs[-1],), context_depth_limit=0)
        assert {row.observation_key for row in active.observations} == {
            ref.stable_key() for ref in inputs}
