"""无对象交付的焦点须沿真实输入/来源链延续，新端点及证据漂移关闭。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_PROPOSITION, ObjectIdentity, OwnerScope, SourceRef,
    VersionBundle, VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, LogicalTimestamp, MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
    MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED, LogicalClockIdentity, document_scope, session_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame, pack_record, unpack_record
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember, TrainedInputStructure,
)
from pure_integer_ai.experiments.current_reference_scope import (
    CURRENT_REFERENCE_SCOPE, read_current_reference_scope,
)
from pure_integer_ai.experiments import reference_scope_continuation as module
from pure_integer_ai.experiments.trained_response_context import DeliveredGenericResponseContext
from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_PROPERTY, ROLE_PROPERTY_ATTRIBUTE, ROLE_PROPERTY_SUBJECT,
)
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


def _fixture(monkeypatch):
    """构造完整分型输入、无对象交付和显式历史边，事实对象无需伪造答案。"""
    owner = OwnerScope(tenant_id=1, user_id=1, session_id=71, visibility=VISIBILITY_SESSION)
    source = SourceRef(41, 7302, 5, owner, VersionBundle())
    origin, prior, output = (replace(source, document_id=index) for index in (1, 2, 3))
    space = SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1)

    def ref(kind, key):
        """本专项使用完整分型Memory引用，owner和版本与生产合同相同。"""
        return MemoryObjectRef(space, owner, source.versions, kind, (key,))

    def payload(value, target):
        """每个输入只声明自己的当前来源及同次有界历史边。"""
        return ObservationPayload(value, MemoryLinkedRef.object(context_scope_identity(value, (1,))),
            (), (), None, (), tuple(MemoryLinkedRef.object(context_scope_identity(target, (namespace,)))
                for namespace in (91543, CURRENT_REFERENCE_SCOPE)),
            LogicalTimestamp(LogicalClockIdentity(document_scope(value), CLOCK_MEMORY_CREATED), value.document_id))

    current_ref, input_ref, output_ref = (ref(MEMORY_OBJECT_OBSERVATION, key) for key in (5, 2, 3))
    proof = context_scope_identity(output, (2,))
    act = context_scope_identity(output, (3,))
    context = DeliveredGenericResponseContext(output_ref, ref(MEMORY_OBJECT_HYPOTHESIS, 3),
        (7, *output.stable_key()), (1,), proof, input_ref, act, ((1,),), 1)
    factual = SimpleNamespace(source_ref=(7, *origin.stable_key()),
        proof_ref=context_scope_identity(origin, (2,)), stable_key=lambda: (91532, 91))
    rows = {current_ref: payload(source, output), input_ref: payload(prior, origin)}
    episode = EpisodePayload(input_ref, MemoryLinkedRef.object(act), (), MemoryLinkedRef.object(proof),
        MemoryLinkedRef.memory(output_ref), (), (), None, output.document_id,
        session_scope(owner.session_id, owner=owner, versions=source.versions),
        LogicalTimestamp(LogicalClockIdentity(document_scope(output), CLOCK_MEMORY_CREATED), 1))
    memory = SimpleNamespace(candidate_index=SimpleNamespace(access=object()),
        intake=SimpleNamespace(event_log=SimpleNamespace(query=lambda **kw: tuple(
            SimpleNamespace(event=SimpleNamespace(payload=rows[key])) for key in rows if key == kw['object_ref']))),
        response_episodes_for=lambda _ref: (episode,))
    proof_key = pack_record(DELIVERY_PROOF_IDENTITY, (3,), prior.stable_key(), input_ref.stable_key(),
                            act.stable_key(), pack_record(1), pack_record(2))
    monkeypatch.setattr(module, 'read_delivery_proof', lambda *_args: proof_key)
    return memory, current_ref, (context, factual), rows, episode, proof_key


def test_objectless_delivery_preserves_only_its_explicit_previous_scope(monkeypatch):
    """真实读者越过零事实交付，保留完整过渡及所有原历史上下文。"""
    memory, current, contexts, rows, _episode, _proof = _fixture(monkeypatch)
    original = {ref: value.stable_key() for ref, value in rows.items()}
    scope = read_current_reference_scope(memory, current, contexts)
    assert not scope.permits(contexts[0]) and scope.permits(contexts[1])
    assert len(scope.continuation_keys) == 1
    assert unpack_record(scope.stable_key(), CURRENT_REFERENCE_SCOPE)[0] == (2,)
    assert unpack_record(scope.continuation_keys[0], 91659)[0] == contexts[0].stable_key()
    assert original == {ref: value.stable_key() for ref, value in rows.items()}
    assert read_current_reference_scope(memory, current, contexts).stable_key() == scope.stable_key()


@pytest.mark.parametrize('fault', ('new_entity', 'missing_link', 'missing_context'))
def test_continuation_never_borrows_an_unlinked_or_replaced_focus(monkeypatch, fault):
    """输入新端点、旧无边或未召回来源均无法以背景事实补焦点。"""
    memory, current, contexts, rows, _episode, _proof = _fixture(monkeypatch)
    prior = contexts[0].input_observation
    if fault == 'new_entity':
        rows[prior] = replace(rows[prior], relation_occurrences=(*rows[prior].relation_occurrences,
            MemoryLinkedRef.object(ObjectIdentity(OBJECT_ENTITY, (99,), prior.owner, prior.versions))))
    elif fault == 'missing_link':
        rows[prior] = replace(rows[prior], relation_occurrences=())
    else:
        contexts = contexts[:1]
    scope = read_current_reference_scope(memory, current, contexts)
    assert scope.context_keys == ()


@pytest.mark.parametrize('fault', ('episode_missing', 'episode_proof', 'episode_input',
                                  'episode_intent', 'proof_slots', 'proof_input', 'source_order'))
def test_continuation_rejects_drift_in_the_original_delivered_chain(monkeypatch, fault):
    """每一类来源/交付损坏均拒绝，不能据零claim数量推定焦点延续。"""
    memory, current, contexts, rows, episode, proof = _fixture(monkeypatch)
    if fault == 'episode_missing':
        memory.response_episodes_for = lambda _ref: ()
    elif fault.startswith('episode_'):
        field, value = {
            'episode_proof': ('selected_path_ref', None),
            'episode_input': ('input_observation_ref', current),
            'episode_intent': ('intent_ref', None),
        }[fault]
        memory.response_episodes_for = lambda _ref: (replace(episode, **{field: value}),)
    elif fault.startswith('proof_'):
        fields = list(unpack_record(proof, DELIVERY_PROOF_IDENTITY))
        fields[5 if fault == 'proof_slots' else 2] = pack_record(2, (99,))
        monkeypatch.setattr(module, 'read_delivery_proof', lambda *_args:
                            pack_record(DELIVERY_PROOF_IDENTITY, *fields))
    else:
        prior = contexts[0].input_observation
        rows[prior] = replace(rows[prior], source=replace(rows[prior].source, document_id=4))
    with pytest.raises(ValueError):
        read_current_reference_scope(memory, current, contexts)


def test_continuation_keeps_competing_contexts_without_selecting_one(monkeypatch):
    """同一指向下两项背景均完整保留，不按稳定键或深度选winner。"""
    memory, current, contexts, _rows, _episode, _proof = _fixture(monkeypatch)
    alternative = SimpleNamespace(source_ref=contexts[1].source_ref,
        proof_ref=contexts[1].proof_ref, stable_key=lambda: (91532, 92))
    scope = read_current_reference_scope(memory, current, (*contexts, alternative))
    assert scope.permits(contexts[1]) and scope.permits(alternative)
    assert len(scope.context_keys) == 2 and len(scope.continuation_keys) == 1


@pytest.mark.parametrize('hops,expected', ((8, 1), (9, 0)))
def test_continuation_chain_has_a_strict_eight_transition_budget(monkeypatch, hops, expected):
    """每步使用完整typed交付，八步可恢复，九步保持空且不借用终点。"""
    memory, current, contexts, rows, original_episode, _proof = _fixture(monkeypatch)
    template, factual = contexts
    payload_template = rows[template.input_observation]
    episodes, proofs, chain = {}, {}, []
    previous = SourceRef.from_stable_key(factual.source_ref[1:])
    for index in range(1, hops + 1):
        source = replace(payload_template.source, document_id=2 * index)
        output = replace(source, document_id=2 * index + 1)
        input_ref = MemoryObjectRef(current.memory_space, current.owner, current.versions,
                                    MEMORY_OBJECT_OBSERVATION, (100 + 2 * index,))
        output_ref = replace(input_ref, object_key=(101 + 2 * index,))
        proof = context_scope_identity(output, (2,))
        context = replace(template, observation=output_ref, input_observation=input_ref,
            proof_ref=proof, source_ref=(7, *output.stable_key()))
        chain.append(context)
        rows[input_ref] = replace(payload_template, source=source,
            relation_occurrences=tuple(MemoryLinkedRef.object(context_scope_identity(previous, (namespace,)))
                for namespace in (91543, CURRENT_REFERENCE_SCOPE)))
        episodes[output_ref] = (replace(original_episode, input_observation_ref=input_ref,
            output_ref=MemoryLinkedRef.memory(output_ref), selected_path_ref=MemoryLinkedRef.object(proof),
            round_seq=output.document_id),)
        proofs[proof] = pack_record(DELIVERY_PROOF_IDENTITY, (3,), source.stable_key(),
            input_ref.stable_key(), context.response_act.stable_key(), pack_record(1), pack_record(2))
        previous = output
    rows[current] = replace(rows[current], source=replace(rows[current].source, document_id=2 * hops + 3),
        relation_occurrences=tuple(MemoryLinkedRef.object(context_scope_identity(previous, (namespace,)))
            for namespace in (91543, CURRENT_REFERENCE_SCOPE)))
    memory.response_episodes_for = lambda ref: episodes[ref]
    monkeypatch.setattr(module, 'read_delivery_proof', lambda _memory, proof: proofs[proof])
    scope = read_current_reference_scope(memory, current, (*chain, factual))
    assert len(scope.context_keys) == expected and len(scope.continuation_keys) == 8


@pytest.mark.parametrize('fault', ('none', 'new_concept_subject', 'missing_parent', 'untrained_predicate'))
def test_trained_predicate_and_property_attribute_are_not_new_subjects(monkeypatch, fault):
    """中性谓词/属性限定必须逐来源证明，真正概念主体和缺帧仍阻断。"""
    memory, current, contexts, rows, _episode, _proof = _fixture(monkeypatch)
    prior = contexts[0].input_observation
    source = rows[prior].source.stable_key()
    predicate = authored_relation_identity(RELATION_PROPERTY)
    attribute = ObjectIdentity(OBJECT_CONCEPT, (99,))
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (99,))
    role = authored_relation_role_identity(
        ROLE_PROPERTY_SUBJECT if fault == 'new_concept_subject' else ROLE_PROPERTY_ATTRIBUTE).stable_key()
    cue = TrainedInputMember((80,), predicate.stable_key(), OBJECT_CONCEPT, PROJECTION_PREDICATE,
        proposition.stable_key(), predicate.stable_key(), source, 0, 1, 0)
    filler = TrainedInputMember((65,), attribute.stable_key(), OBJECT_CONCEPT, PROJECTION_FILLER,
        proposition.stable_key(), predicate.stable_key(), source, 1, 2, 1, role)
    structure = TrainedInputStructure(proposition.stable_key(), predicate.stable_key(), source, (1,), (cue, filler))
    frame = OpenRoleFrame(proposition.stable_key(), predicate.stable_key(), source, (role,), ((), ()), (0, 2))
    memory.input_projector = SimpleNamespace(structures=(structure,),
        open_role_frames=() if fault == 'untrained_predicate' else (frame,), envelope_role_frames=())
    linked = (predicate, attribute) if fault == 'missing_parent' else (predicate, attribute, proposition)
    rows[prior] = replace(rows[prior], relation_occurrences=(*rows[prior].relation_occurrences,
        *(MemoryLinkedRef.object(value) for value in linked)))
    scope = read_current_reference_scope(memory, current, contexts)
    assert bool(scope.context_keys) is (fault == 'none')
    if fault == 'none':
        witnesses = unpack_record(unpack_record(scope.continuation_keys[0], 91659)[3], 1)
        assert frame.stable_key() in witnesses
        assert module.pack_neutral_member(structure, filler) in witnesses
