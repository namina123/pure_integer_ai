"""对称来源反序须有完整整数见证，不能清除其它真实冲突。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OwnerScope, SourceRef, VersionBundle, VISIBILITY_SESSION, concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import proposition_identity
from pure_integer_ai.cognition.shared.query_state import BindingEntry
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_CLOSED,
    STRUCTURE_ORDER_CONFLICT, TrainedInputMember, TrainedInputStructure,
    TrainedInputStructureProjector,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_SIMILAR, RELATION_CAUSES, ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
    ROLE_CAUSE, ROLE_EFFECT,
)
from pure_integer_ai.experiments.symmetric_input_order import (
    symmetric_input_order_witnesses, unresolved_input_conflict_bindings,
    verify_saved_symmetric_input_order,
)


def _fixture(*, symmetric=True, predicate_middle=False):
    """由真实投影器产生相同物理端点的正序/反序独立来源。"""
    predicate = authored_relation_identity(RELATION_SIMILAR if symmetric else RELATION_CAUSES)
    roles = tuple(authored_relation_role_identity(kind) for kind in (
        (ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT) if symmetric else (ROLE_CAUSE, ROLE_EFFECT)))
    endpoints = (concept_identity((801,)), concept_identity((802,)))
    trained, facts = [], {}
    for index, pair in enumerate((endpoints, endpoints[::-1]), 101):
        source = SourceRef(206, index, 0, OwnerScope(), VersionBundle())
        prop = proposition_identity(source, (1, 1))
        key, pred, ref = prop.stable_key(), predicate.stable_key(), (index,)
        rows = [(11 if endpoint == endpoints[0] else 22, endpoint, PROJECTION_FILLER,
                 role.stable_key()) for endpoint, role in zip(pair, roles)]
        rows.insert(1 if predicate_middle else 2, (33, predicate, PROJECTION_PREDICATE, ()))
        members = tuple(TrainedInputMember((value,), identity.stable_key(), identity.object_kind,
            kind, key, pred, ref, ordinal, ordinal + 1, ordinal, role)
            for ordinal, (value, identity, kind, role) in enumerate(rows))
        trained.append(TrainedInputStructure(key, pred, ref, (1,), members))
        facts[key] = SimpleNamespace(proposition=prop, predicate=predicate, source_hash=index,
            bindings=tuple(SimpleNamespace(role=role, filler=endpoint) for role, endpoint in zip(roles, pair)))
    structure = TrainedInputStructureProjector(tuple(trained)).project(
        (11, 33, 22) if predicate_middle else (11, 22, 33))
    assert sorted(item.state for item in structure.relation_candidates) == [STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT]
    return structure, facts


def _state(witness):
    """保留全部原证据和冲突，另加UNKNOWN等价见证。"""
    return SimpleNamespace(query_key=witness.query_key, bindings=witness.original_bindings(),
        evidence=(*witness.original_evidence(), witness.query_evidence()))


def _trace(state):
    """仅编码已存在的整数记录，用于冷消费者的漂移反例。"""
    keys = tuple(item.stable_key() for item in (*state.bindings, *state.evidence))
    return SimpleNamespace(trace={'state_key': {'bindings': list(range(1, len(state.bindings) + 1)),
        'evidence': list(range(len(state.bindings) + 1, len(keys) + 1))}}, resolve_ref=lambda ref: keys[ref - 1])


@pytest.mark.parametrize('predicate_middle', (False, True))
def test_reversed_symmetric_source_has_explicit_witness_without_mutating_projection(predicate_middle):
    structure, facts = _fixture(predicate_middle=predicate_middle)
    original = structure.stable_key()
    witness, = symmetric_input_order_witnesses(structure, facts)
    state = _state(witness)
    assert witness.relation.state == STRUCTURE_ORDER_CONFLICT
    assert witness.conflict_binding().conflict_kept == 1
    assert all(item.polarity == 3 for item in state.evidence)
    assert not unresolved_input_conflict_bindings(state, (witness,))
    assert verify_saved_symmetric_input_order(_trace(state), structure, facts) == 1
    assert structure.stable_key() == original


@pytest.mark.parametrize('drift', ('witness_missing', 'witness_truth', 'witness_scope',
    'projection_missing', 'projection_truth', 'projection_payload', 'query', 'extra_conflict',
    'support_binding_missing', 'filler_binding_missing', 'support_binding_conflict'))
def test_real_consumers_keep_every_unexplained_binding_blocking(drift):
    structure, facts = _fixture()
    witness, = symmetric_input_order_witnesses(structure, facts)
    state = _state(witness)
    if drift == 'witness_missing': state.evidence = state.evidence[:-1]
    elif drift == 'witness_truth': state.evidence = (*state.evidence[:-1], replace(state.evidence[-1], polarity=1))
    elif drift == 'witness_scope': state.evidence = (*state.evidence[:-1], replace(state.evidence[-1], scope_key=(999,)))
    elif drift == 'projection_missing': state.evidence = state.evidence[1:]
    elif drift == 'projection_truth': state.evidence = (replace(state.evidence[0], polarity=1), *state.evidence[1:])
    elif drift == 'projection_payload': state.evidence = (replace(state.evidence[0], payload_key=(999,)), *state.evidence[1:])
    elif drift == 'query': state.query_key = (999,)
    elif drift == 'support_binding_missing': state.bindings = (state.bindings[0], *state.bindings[2:])
    elif drift == 'filler_binding_missing': state.bindings = state.bindings[:-1]
    elif drift == 'support_binding_conflict': state.bindings = (state.bindings[0], replace(state.bindings[1], conflict_kept=1), *state.bindings[2:])
    else: state.bindings += (BindingEntry((1,), (2,), conflict_kept=1),)
    assert unresolved_input_conflict_bindings(state, (witness,))
    if drift not in ('query', 'extra_conflict'):
        with pytest.raises(ValueError): verify_saved_symmetric_input_order(_trace(state), structure, facts)


@pytest.mark.parametrize('drift', ('non_symmetric', 'missing_support', 'source', 'endpoint',
    'predicate', 'carrier', 'structure', 'member_role', 'member_identity', 'member_ordinal',
    'span', 'overlap', 'owner', 'extra_member'))
def test_witness_never_explains_other_semantic_source_or_topology_changes(drift):
    structure, facts = _fixture(symmetric=drift != 'non_symmetric')
    relation = next(item for item in structure.relation_candidates if item.state == STRUCTURE_ORDER_CONFLICT)
    support = next(item for item in structure.relation_candidates if item.state == STRUCTURE_CLOSED)
    fact = facts[relation.proposition_key]
    if drift == 'non_symmetric':
        pass
    elif drift == 'missing_support':
        structure = replace(structure, relation_candidates=(relation,))
    elif drift == 'source': fact.source_hash += 100
    elif drift == 'endpoint': fact.bindings = (replace_binding(fact.bindings[0], filler=concept_identity((999,))), fact.bindings[1])
    elif drift == 'predicate': fact.predicate = authored_relation_identity(RELATION_CAUSES)
    elif drift in ('carrier', 'structure'):
        changes = {'carrier_key': (999,)} if drift == 'carrier' else {'structure_key': (999,)}
        structure = replace(structure, relation_candidates=tuple(sorted(
            (replace(item, **changes) if item == relation else item for item in structure.relation_candidates), key=lambda item: item.stable_key())))
    else:
        target = next(item for item in structure.semantic_candidates if item.proposition_key == relation.proposition_key and item.projection_kind == PROJECTION_FILLER)
        edits = {'member_role': {'role_key': (999,)}, 'member_identity': {'candidate_key': concept_identity((999,)).stable_key()},
            'member_ordinal': {'member_ordinal': 7}, 'span': {'span_ref': (1, 999, 0, 1, 2)},
            'overlap': {'span_ref': next(item.span_ref for item in structure.semantic_candidates
                if item.proposition_key == relation.proposition_key and item.projection_kind == PROJECTION_PREDICATE)},
            'owner': {'candidate_key': concept_identity((801,), owner=OwnerScope(9, 8, 7, VISIBILITY_SESSION)).stable_key()},
            'extra_member': {'member_ordinal': 8}}
        changed = replace(target, **edits[drift])
        members = ((*structure.semantic_candidates, changed) if drift == 'extra_member' else
            tuple(changed if item == target else item for item in structure.semantic_candidates))
        members = tuple(sorted(members, key=lambda item: item.stable_key()))
        if drift == 'span':
            with pytest.raises(ValueError): replace(structure, semantic_candidates=members)
            fields = {name: getattr(structure, name) for name in structure.__dataclass_fields__}
            structure = SimpleNamespace(**(fields | {'semantic_candidates': members}))
        else:
            structure = replace(structure, semantic_candidates=members)
    assert symmetric_input_order_witnesses(structure, facts) == ()


def replace_binding(binding, **changes):
    """替换测试Core记录字段，不改变生产类型或合法性门。"""
    return SimpleNamespace(**(vars(binding) | changes))


@pytest.mark.parametrize('field', ('operation_keys', 'relation_question_bindings',
    'operation_request_bindings', 'assertion_scope_bindings', 'discourse_compositions',
    'discourse_gaps', 'pending_relation_composition_bindings'))
def test_other_input_scopes_never_borrow_plain_assertion_equivalence(field):
    structure, facts = _fixture()
    fields = {name: getattr(structure, name) for name in structure.__dataclass_fields__}
    fields[field] = ((999,),)
    assert symmetric_input_order_witnesses(SimpleNamespace(**fields), facts) == ()
