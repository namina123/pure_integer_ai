"""Independent evidence and actual graph-edit boundaries of operation execution."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding, CandidateVerification, EvidenceCandidateDefinition,
    EvidenceCandidateEngine, EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE, OBJECT_ENTITY, ObjectIdentity, SourceRef, VersionBundle,
    concept_identity, occurrence_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, UNKNOWN, CLARIFY, OperationExecution, OperationGraphFact, execute_integer_operation,
    operation_context_identity, requested_operation_context,
)
from pure_integer_ai.cognition.shared.integer_operation_program import (
    EQUAL, UNEQUAL, ORDER_DIRECTED_CHAIN, IntegerOperationProgram, OperationAtom,
    OperationPathQuery, OperationProgramBudget, OperationProgramWitness, OperationProjection,
)
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import QueryState
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition, AtomicRoleBinding, context_scope_identity,
    entity_identity, proposition_identity, role_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import RelationSchema, RelationSlotSchema
from pure_integer_ai.experiments.operation_program_query import read_operation_core_context
from pure_integer_ai.experiments.operation_core_context import request_context_proposition_keys
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, write_preflight_json
from pure_integer_ai.experiments.trained_operation_program_graph import (
    PART_TABLE, materialize_operation_program_graph, read_operation_program_graph,
)
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence, current_query_evidence
from pure_integer_ai.storage.assertion_identity import IntegerIdentityRegistry, register_assertion_identity_tables
from pure_integer_ai.storage.backend import DictBackend


def _source(number):
    return SourceRef(92800, number, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


LEFT, RIGHT = role_identity((92801, 1)), role_identity((92801, 2))


def _schema(number):
    return RelationSchema(structure_concept_identity((92802, number)),
        concept_identity((92803, number)), tuple(RelationSlotSchema(role,
            frozenset({OBJECT_ENTITY}), 1, 1) for role in (LEFT, RIGHT)))


def _fact(number, relation, first, second, *, stance=1, lifecycle=None):
    source, observation = _source(number), _source(number + 1000)
    definition = AtomicPropositionDefinition(proposition_identity(source, (number,)), relation,
        occurrence_identity(source, start=0, end=2, ordinal=1), context_scope_identity(source, (1,)),
        (AtomicRoleBinding(LEFT, first), AtomicRoleBinding(RIGHT, second)))
    candidate = EvidenceCandidateDefinition(definition.proposition, (92804, number),
        (CandidateBinding(concept_identity((92805, 1)), relation),
         CandidateBinding(concept_identity((92805, 2)), first),
         CandidateBinding(concept_identity((92805, 3)), second)), (source,))
    aggregate = _source(number + 2000)
    engine = EvidenceCandidateEngine(EvidenceCandidateProtocol((92806, 1), (92806, 2),
        aggregate, document_scope(aggregate), 1))
    hypothesis = engine.register(candidate)
    prediction = engine.predict(hypothesis, observation=observation, scope=document_scope(observation),
        event_key=(92807, number), visible_inputs=(first, second), predicted=definition.proposition)
    engine.reveal(prediction, CandidateVerification(stance, (92808, 1), observation,
        concept_identity((92808, 2)), (1,), (number,)), timestamp_seq=2)
    engine.resolve(hypothesis, timestamp_seq=3)
    snapshot = engine.ledger.snapshot(hypothesis)
    if lifecycle is not None:
        snapshot = replace(snapshot, lifecycle=lifecycle)
    backend = DictBackend()
    register_assertion_identity_tables(backend)
    entries = current_query_evidence(core_query_evidence(definition.proposition,
        engine.ledger.evidence_history(hypothesis), IntegerIdentityRegistry(backend)))
    return OperationGraphFact(definition, snapshot, candidate,
        tuple(sorted(entry.stable_key() for entry in entries)))


def _atom(schema, variables):
    return OperationAtom(schema, tuple(sorted(zip((LEFT, RIGHT), variables), key=lambda row: row[0].stable_key())))


def _witness(program, facts):
    digest = hashlib.sha256(bytes(str(program.stable_key()), "ascii")).digest()
    source = SourceRef(92221, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
                       0, GLOBAL_OWNER_SCOPE, VersionBundle())
    return OperationProgramWitness(program, source, tuple(digest), (92810, 1), (92810, 2),
        tuple(sorted({source for fact in facts for source in fact.candidate.forming_sources},
                     key=SourceRef.stable_key)))


@pytest.fixture(scope="module")
def context():
    source = _source(100)
    a, b, c, d = tuple(entity_identity(source, (number,)) for number in range(1, 5))
    first, second = _schema(1), _schema(2)
    facts = (_fact(1, first.relation, a, b), _fact(2, second.relation, b, c),
             _fact(3, second.relation, c, d))
    program = IntegerOperationProgram((92811, 1), (1, 2), (_atom(first, (1, 2)), _atom(second, (2, 3))),
        (), (OperationProjection(1, (LEFT, RIGHT)),))
    return a, b, c, d, first, second, facts, program


def test_exact_evidence_join_roundtrip_and_independent_verification(context, monkeypatch):
    a, b, _, _, _, _, facts, program = context
    witness = _witness(program, facts)
    result = execute_integer_operation(witness, (92812, 1), ((1, a), (2, b)), facts)
    assert result.status == READY
    assert result.outputs == ((facts[0].proposition_key, (LEFT.stable_key(), RIGHT.stable_key())),)
    assert len(result.fact_proofs) == 2
    assert OperationExecution.from_stable_key(result.stable_key()) == result
    assert OperationProgramWitness.from_stable_key(witness.stable_key()) == witness
    with pytest.raises(ValueError):
        OperationExecution.from_stable_key(result.stable_key()[:-1])
    with pytest.raises(ValueError):
        IntegerOperationProgram.from_stable_key(program.stable_key() + (1,))
    monkeypatch.setattr("pure_integer_ai.cognition.shared.integer_operation_execution.execute_integer_operation",
                        lambda *args, **kwargs: pytest.fail("independent verifier invoked producer"))
    assert verify_integer_operation(result, facts, (witness,))


def test_request_context_excludes_large_unrelated_core_without_losing_matches(context):
    a, b, _, _, first, _, facts, program = context
    noise = tuple(_fact(10000 + index, first.relation,
        entity_identity(_source(10000 + index), (1,)),
        entity_identity(_source(10000 + index), (2,))) for index in range(320))
    whole_core = (*facts, *noise)
    witness = _witness(program, whole_core)
    variables = ((1, a), (2, b))
    budget = OperationProgramBudget(max_context_values=32768)
    with pytest.raises(ValueError, match="context integer budget"):
        operation_context_identity(whole_core, budget)
    assert requested_operation_context(witness, variables, whole_core, budget) == facts[:2]
    result = execute_integer_operation(witness, (92812, 13), variables, whole_core, budget=budget)
    assert result.status == READY and len(result.fact_proofs) == 2
    assert verify_integer_operation(result, whole_core, (witness,), budget=budget)


def test_verifier_accepts_exact_legacy_whole_core_identity_but_detects_its_drift(context):
    a, b, _, _, first, _, facts, program = context
    noise = _fact(10000, first.relation,
        entity_identity(_source(10000), (1,)), entity_identity(_source(10000), (2,)))
    whole_core = (*facts, noise)
    witness = _witness(program, whole_core)
    variables = ((1, a), (2, b))
    execution = execute_integer_operation(witness, (92812, 131), variables, whole_core)
    legacy = replace(execution, context_identity=operation_context_identity(whole_core, OperationProgramBudget()))

    assert verify_integer_operation(legacy, whole_core, (witness,))
    changed_noise = _fact(10001, first.relation,
        entity_identity(_source(10001), (1,)), entity_identity(_source(10001), (2,)))
    with pytest.raises(ValueError, match="context drifted"):
        verify_integer_operation(legacy, (*facts, changed_noise), (witness,))


def test_requested_context_keeps_rewrite_evidence_and_detects_relevant_drift(context):
    a, b, c, d, first, _, _, _ = context
    facts = (_fact(6001, first.relation, a, b), _fact(6002, first.relation, a, c),
             _fact(6003, first.relation, a, d))
    projections = tuple(OperationProjection(index, (LEFT, RIGHT)) for index in (1, 2, 3))
    summary = IntegerOperationProgram((92811, 10), (1, 2, 3, 4),
        tuple(_atom(first, (1, index)) for index in (2, 3, 4)), (), projections)
    rewrite = replace(summary, operation_key=(92811, 11), projections=tuple(reversed(projections)))
    variables = ((1, a), (2, b), (3, c), (4, d))
    before = execute_integer_operation(_witness(summary, facts), (92812, 15), variables, facts)
    after = execute_integer_operation(_witness(rewrite, facts), (92812, 16), variables, facts)
    assert before.status == after.status == READY
    assert before.context_identity == after.context_identity
    assert before.fact_proofs == after.fact_proofs
    assert after.outputs == tuple(reversed(before.outputs))
    assert verify_integer_operation(after, facts, (after.witness,))
    drifted = (facts[0], _fact(6002, first.relation, a, c, stance=2), facts[2])
    with pytest.raises(ValueError, match="context drifted"):
        verify_integer_operation(after, drifted, (after.witness,))


def test_request_context_respects_repeated_variables(context):
    a, b, _, _, first, _, _, _ = context
    facts = (_fact(6101, first.relation, a, a), _fact(6102, first.relation, a, b))
    program = IntegerOperationProgram((92811, 12), (1,), (_atom(first, (1, 1)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, facts)
    assert requested_operation_context(witness, ((1, a),), facts) == facts[:1]
    result = execute_integer_operation(witness, (92812, 17), ((1, a),), facts)
    assert result.status == READY and result.matches == ((facts[0].proposition_key,),)
    assert verify_integer_operation(result, facts, (witness,))


@pytest.mark.parametrize('change', ('unrelated_refute', 'relevant_refute', 'unknown', 'unauthorized', 'partial'))
def test_single_anchor_context_propagates_all_structural_variables(context, change):
    """仅连接结构排除无关反证；相关反证、无证、来源不足和部分分支原样保留。"""
    a, b, c, d, first, second, _, _ = context
    third = _schema(3)
    x, y = (entity_identity(_source(31000), (index,)) for index in (1, 2))
    facts = (_fact(31001, first.relation, a, b), _fact(31002, second.relation, a, c),
             _fact(31003, third.relation, a, d))
    if change == 'unknown':
        facts = (facts[0], _fact(31002, second.relation, a, c, stance=3), facts[2])
    if change == 'partial':
        facts = facts[:2]
    noise = (_fact(31004, second.relation, x, y), _fact(31005, second.relation, x, y, stance=2))
    if change == 'relevant_refute':
        facts = (*facts, _fact(31006, second.relation, a, c, stance=2))
    program = IntegerOperationProgram((92811, 31000), (2,),
        (_atom(first, (1, 2)), _atom(second, (1, 3)), _atom(third, (1, 4))), (),
        tuple(OperationProjection(index, (LEFT, RIGHT)) for index in (1, 2, 3)))
    witness = _witness(program, (*facts, *noise))
    if change == 'unauthorized':
        witness = replace(witness, authorized_forming_sources=facts[0].candidate.forming_sources)
    selected = requested_operation_context(witness, ((2, b),), (*facts, *noise))
    assert {fact.proposition_key for fact in selected} == {fact.proposition_key for fact in facts}
    execution = execute_integer_operation(witness, (92812, 31000), ((2, b),), (*facts, *noise))
    expected = READY if change == 'unrelated_refute' else CLARIFY if change == 'relevant_refute' else UNKNOWN
    assert execution.status == expected
    assert len(execution.outputs) == (3 if expected == READY else 0)
    assert verify_integer_operation(execution, (*facts, *noise), (witness,))
    assert execute_integer_operation(witness, (92812, 31000), ((2, b),),
        tuple(reversed((*facts, *noise)))).stable_key() == execution.stable_key()


def test_context_environments_preserve_correlated_variables_and_partial_branches(context):
    """不同前缀不能按单变量集合拼接，未完成分支的原命题仍被保留。"""
    a, b, c, d, first, second, _, _ = context
    third = _schema(3)
    e = entity_identity(_source(32000), (1,))
    prefix = (_fact(32001, first.relation, a, b), _fact(32002, first.relation, a, c),
              _fact(32003, second.relation, b, d), _fact(32004, second.relation, c, e))
    connected = _fact(32005, third.relation, b, d)
    disconnected = (_fact(32006, third.relation, b, e), _fact(32007, third.relation, b, e, stance=2))
    facts = (*prefix, connected, *disconnected)
    program = IntegerOperationProgram((92811, 32000), (1,),
        (_atom(first, (1, 2)), _atom(second, (2, 3)), _atom(third, (2, 3))), (),
        (OperationProjection(3, (LEFT, RIGHT)),))
    witness = _witness(program, facts)
    selected = requested_operation_context(witness, ((1, a),), facts)
    assert {fact.proposition_key for fact in selected} == {fact.proposition_key for fact in (*prefix, connected)}
    result = execute_integer_operation(witness, (92812, 32000), ((1, a),), facts)
    assert result.status == READY and len(result.matches) == 1
    assert verify_integer_operation(result, facts, (witness,))
    for budget in (OperationProgramBudget(max_matches=1), OperationProgramBudget(max_match_attempts=1)):
        with pytest.raises(ValueError, match='budget exceeded; no partial result'):
            requested_operation_context(witness, ((1, a),), facts, budget)


def test_independent_verifier_joins_large_candidate_groups_within_attempt_budget(context):
    a, b, _, d, first, second, _, _ = context
    third, fourth = _schema(3), _schema(4)
    c0, z0 = entity_identity(_source(15000), (1,)), entity_identity(_source(15001), (1,))
    facts = [_fact(5000, first.relation, a, b), _fact(5001, second.relation, b, c0)]
    facts.append(_fact(5002, third.relation, c0, z0))
    for index in range(1, 80):
        left = entity_identity(_source(15100 + index), (1,))
        right = entity_identity(_source(15200 + index), (1,))
        facts.append(_fact(5100 + index, second.relation, left, right))
        next_left = entity_identity(_source(15300 + index), (1,))
        next_right = entity_identity(_source(15400 + index), (1,))
        facts.append(_fact(5200 + index, third.relation, next_left, next_right))
    facts.append(_fact(5300, fourth.relation, z0, d))
    facts = tuple(facts)
    program = IntegerOperationProgram((92811, 9), (1, 4),
        (_atom(first, (1, 2)), _atom(second, (2, 3)), _atom(third, (3, 5)),
         _atom(fourth, (5, 4))), (), (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, facts)
    budget = OperationProgramBudget(max_match_attempts=200)
    result = execute_integer_operation(witness, (92812, 14), ((1, a), (4, d)), facts, budget=budget)
    assert result.status == READY and len(result.matches) == 1
    assert verify_integer_operation(result, facts, (witness,), budget=budget)


def test_projection_changes_real_claim_and_label_alone_does_not(context):
    a, b, _, _, _, _, facts, program = context
    original = _witness(program, facts)
    baseline = execute_integer_operation(original, (92812, 1), ((1, a), (2, b)), facts)
    changed = _witness(replace(program, projections=(OperationProjection(2, (LEFT, RIGHT)),)), facts)
    swapped = execute_integer_operation(changed, (92812, 1), ((1, a), (2, b)), facts)
    assert swapped.context_identity == baseline.context_identity
    assert swapped.matches == baseline.matches
    assert swapped.outputs != baseline.outputs
    renamed = _witness(replace(program, operation_key=(92811, 2)), facts)
    label_only = execute_integer_operation(renamed, (92812, 1), ((1, a), (2, b)), facts)
    assert label_only.outputs == baseline.outputs
    with pytest.raises(ValueError, match="permission"):
        verify_integer_operation(swapped, facts, (original,))


@pytest.mark.parametrize("kind", [EQUAL, UNEQUAL])
def test_data_supplied_guard_drives_selection(context, kind):
    a, b, _, _, first, _, facts, program = context
    guarded = replace(program, atoms=(_atom(first, (1, 2)),), guards=((kind, 1, 2),))
    witness = _witness(guarded, facts)
    result = execute_integer_operation(witness, (92812, 1), ((1, a), (2, b)), facts)
    assert result.status == (UNKNOWN if kind == EQUAL else READY)
    assert verify_integer_operation(result, facts, (witness,))


def test_same_display_is_not_a_full_object_identity(context):
    a, b, _, _, _, _, facts, program = context
    witness = _witness(program, facts)
    foreign = entity_identity(_source(101), (1,))
    result = execute_integer_operation(witness, (92812, 1), ((1, foreign), (2, b)), facts)
    assert result.status == UNKNOWN and result.outputs == ()
    assert foreign != a


def test_directed_order_is_driven_by_edges_and_rejects_ties_and_cycles(context):
    a, b, c, d, first, _, _, _ = context
    facts = (_fact(41, first.relation, a, b), _fact(42, first.relation, b, c),
             _fact(43, first.relation, c, d))
    program = IntegerOperationProgram((92811, 3), (1,), (_atom(first, (1, 2)), _atom(first, (2, 3)),
        _atom(first, (3, 4))), (), (OperationProjection(3, (LEFT, RIGHT)),
        OperationProjection(1, (LEFT, RIGHT)), OperationProjection(2, (LEFT, RIGHT))),
        order_kind=ORDER_DIRECTED_CHAIN, order_roles=(LEFT, RIGHT))
    witness = _witness(program, facts)
    result = execute_integer_operation(witness, (92812, 1), ((1, a),), facts)
    assert result.status == READY
    assert tuple(key for key, _ in result.outputs) == tuple(fact.proposition_key for fact in facts)
    assert verify_integer_operation(result, facts, (witness,))
    cycle = _fact(43, first.relation, c, a)
    closed_facts = (*facts[:2], cycle)
    cycle_witness = _witness(program, closed_facts)
    closed = execute_integer_operation(cycle_witness, (92812, 1), ((1, a),), closed_facts)
    assert closed.status == CLARIFY and not closed.outputs
    assert verify_integer_operation(closed, closed_facts, (cycle_witness,))
    fork = (_fact(42, first.relation, b, c), _fact(43, first.relation, b, d))
    # Two branches can be matched, but there is no authorized total edge order.
    branching = replace(program, atoms=(_atom(first, (1, 2)), _atom(first, (2, 3)),
        _atom(first, (2, 4))), guards=((UNEQUAL, 3, 4),), maximum_matches=2)
    branch_facts = (facts[0], *fork)
    branch_witness = _witness(branching, branch_facts)
    branch = execute_integer_operation(branch_witness, (92812, 1), ((1, a),), branch_facts)
    assert branch.status == CLARIFY and not branch.outputs
    assert verify_integer_operation(branch, branch_facts, (branch_witness,))


def test_bounded_relation_path_roundtrip_unique_unknown_ambiguous_and_budget(context, tmp_path):
    a, b, c, d, first, _, _, _ = context
    program = IntegerOperationProgram((92811, 4), (1, 2), (_atom(first, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),), 2, 4, ORDER_DIRECTED_CHAIN, (),
        path_query=OperationPathQuery(1, 2, ((1, LEFT, RIGHT),)))
    assert IntegerOperationProgram.from_stable_key(program.stable_key()) == program
    facts = (_fact(41, first.relation, a, b), _fact(42, first.relation, b, c),
             _fact(43, first.relation, c, d))
    witness = _witness(program, facts)
    assert OperationProgramWitness.from_stable_key(witness.stable_key()) == witness
    graph = tmp_path / "path-programs.sqlite3"
    materialize_operation_program_graph((witness,), graph)
    assert read_operation_program_graph(graph) == (witness,)
    result = execute_integer_operation(witness, (92812, 4), ((1, a), (2, d)), facts)
    assert result.status == READY
    assert result.matches == ((facts[0].proposition_key, facts[1].proposition_key, facts[2].proposition_key),)
    assert tuple(key for key, _ in result.outputs) == result.matches[0]
    assert verify_integer_operation(result, facts, (witness,))

    missing = entity_identity(_source(102), (1,))
    absent = execute_integer_operation(witness, (92812, 5), ((1, a), (2, missing)), facts)
    assert absent.status == UNKNOWN and absent.outputs == ()
    assert verify_integer_operation(absent, facts, (witness,))

    alternatives = (_fact(44, first.relation, a, b), _fact(45, first.relation, b, d),
                    _fact(46, first.relation, a, c), _fact(47, first.relation, c, d))
    ambiguous_witness = _witness(program, alternatives)
    dead, further = entity_identity(_source(103), (1,)), entity_identity(_source(104), (1,))
    with_distractors = (*alternatives, _fact(48, first.relation, a, dead),
                        _fact(49, first.relation, dead, further))
    assert set(requested_operation_context(ambiguous_witness, ((1, a), (2, d)), with_distractors)) == set(alternatives)
    ambiguous = execute_integer_operation(ambiguous_witness, (92812, 6), ((1, a), (2, d)), with_distractors)
    assert ambiguous.status == CLARIFY and ambiguous.outputs == ()
    assert len(ambiguous.matches) == 2
    assert verify_integer_operation(ambiguous, with_distractors, (ambiguous_witness,))
    with pytest.raises(ValueError, match="budget"):
        execute_integer_operation(witness, (92812, 7), ((1, a), (2, d)), facts,
                                  budget=OperationProgramBudget(max_match_attempts=1))


def test_bounded_path_rejects_cycles_and_unlicensed_or_conflicting_edges(context):
    a, b, c, _, first, _, _, _ = context
    program = IntegerOperationProgram((92811, 5), (1, 2), (_atom(first, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),), 2, 4, ORDER_DIRECTED_CHAIN, (),
        path_query=OperationPathQuery(1, 2, ((1, LEFT, RIGHT),)))
    cycle = (_fact(51, first.relation, a, b), _fact(52, first.relation, b, a),
             _fact(53, first.relation, a, c))
    witness = _witness(program, cycle)
    result = execute_integer_operation(witness, (92812, 8), ((1, a), (2, c)), cycle)
    assert result.status == UNKNOWN and result.outputs == ()
    assert verify_integer_operation(result, cycle, (witness,))

    chain = (_fact(54, first.relation, a, b), _fact(55, first.relation, b, c))
    permitted = _witness(program, chain)
    limited = replace(permitted, authorized_forming_sources=chain[0].candidate.forming_sources)
    denied = execute_integer_operation(limited, (92812, 9), ((1, a), (2, c)), chain)
    assert denied.status == UNKNOWN and denied.outputs == ()
    assert verify_integer_operation(denied, chain, (limited,))

    conflict = (*chain, _fact(56, first.relation, b, c, stance=2))
    conflicted = _witness(program, conflict)
    assert set(requested_operation_context(conflicted, ((1, a), (2, c)), conflict)) == set(conflict)
    blocked = execute_integer_operation(conflicted, (92812, 10), ((1, a), (2, c)), conflict)
    assert blocked.status == CLARIFY and blocked.outputs == ()
    assert verify_integer_operation(blocked, conflict, (conflicted,))


def test_bounded_path_can_join_distinct_licensed_relation_templates(context):
    a, _, _, d, first, second, facts, _program = context
    program = IntegerOperationProgram((92811, 6), (1, 2),
        (_atom(first, (1, 2)), _atom(second, (1, 2))), (),
        (OperationProjection(1, (LEFT, RIGHT)), OperationProjection(2, (LEFT, RIGHT))),
        2, 4, ORDER_DIRECTED_CHAIN, (),
        path_query=OperationPathQuery(1, 2, ((1, LEFT, RIGHT), (2, LEFT, RIGHT))))
    assert IntegerOperationProgram.from_stable_key(program.stable_key()) == program
    witness = _witness(program, facts)
    result = execute_integer_operation(witness, (92812, 11), ((1, a), (2, d)), facts)
    assert result.status == READY
    assert result.matches == ((facts[0].proposition_key, facts[1].proposition_key, facts[2].proposition_key),)
    assert tuple(key for key, _ in result.outputs) == result.matches[0]
    assert verify_integer_operation(result, facts, (witness,))


@pytest.mark.parametrize("stance", [2, 3])
def test_unknown_refuted_and_unauthorized_premises_are_not_projected(context, stance):
    a, b, c, _, _, second, facts, program = context
    changed = (facts[0], _fact(2, second.relation, b, c, stance=stance), facts[2])
    witness = _witness(program, changed)
    result = execute_integer_operation(witness, (92812, 1), ((1, a), (2, b)), changed)
    assert result.status == UNKNOWN and result.outputs == ()
    assert verify_integer_operation(result, changed, (witness,))
    permitted = replace(_witness(program, facts), authorized_forming_sources=(facts[0].candidate.forming_sources[0],))
    assert execute_integer_operation(permitted, (92812, 1), ((1, a), (2, b)), facts).status == UNKNOWN


def test_conflicting_same_content_from_another_source_blocks_positive(context):
    a, b, c, _, _, second, facts, program = context
    conflict = (*facts, _fact(90, second.relation, b, c, stance=2))
    witness = _witness(program, conflict)
    result = execute_integer_operation(witness, (92812, 1), ((1, a), (2, b)), conflict)
    assert result.status == CLARIFY and not result.outputs
    assert len(result.fact_proofs) == 2
    assert verify_integer_operation(result, conflict, (witness,))


def test_complete_match_ambiguity_and_omitted_match_proof_are_rejected(context):
    a, b, _, d, _, second, facts, program = context
    ambiguous = (*facts, _fact(4, second.relation, b, d))
    witness = _witness(program, ambiguous)
    result = execute_integer_operation(witness, (92812, 1), ((1, a), (2, b)), ambiguous)
    assert result.status == CLARIFY and not result.outputs
    assert verify_integer_operation(result, ambiguous, (witness,))
    with pytest.raises(ValueError, match="complete graph matches"):
        verify_integer_operation(replace(result, matches=result.matches[:1]), ambiguous, (witness,))
    complete = execute_integer_operation(_witness(program, facts), (92812, 1), ((1, a), (2, b)), facts)
    with pytest.raises(ValueError, match="full fact/Evidence"):
        verify_integer_operation(replace(complete, fact_proofs=complete.fact_proofs[:1]), facts, (complete.witness,))


def test_full_current_evidence_missing_or_forged_fails_at_context_boundary(context):
    *_, facts, _ = context
    fact = facts[0]
    with pytest.raises(ValueError, match="exact current"):
        replace(fact, evidence_keys=fact.evidence_keys[:-1])
    with pytest.raises(ValueError, match="forged"):
        key = fact.evidence_keys[0]
        replace(fact, evidence_keys=tuple(sorted((key[:1] + (2 if key[1] == 1 else 1,) + key[2:],
                                                *fact.evidence_keys[1:]))))


@pytest.mark.parametrize("budget", [
    OperationProgramBudget(max_atoms=1), OperationProgramBudget(max_facts=1),
    OperationProgramBudget(max_match_attempts=1), OperationProgramBudget(max_key_values=1),
    OperationProgramBudget(max_proof_values=1), OperationProgramBudget(max_context_values=1),
])
def test_exceeded_budget_returns_no_partial_result(context, budget):
    a, b, _, _, _, _, facts, program = context
    with pytest.raises(ValueError, match="budget"):
        execute_integer_operation(_witness(program, facts), (92812, 1), ((1, a), (2, b)), facts, budget=budget)


def test_graph_cold_restore_and_actual_instruction_removal(context, tmp_path):
    a, b, _, _, _, _, facts, program = context
    witness = _witness(program, facts)
    baseline, removed = tmp_path / "baseline.sqlite3", tmp_path / "removed.sqlite3"
    materialize_operation_program_graph((witness,), baseline)
    materialize_operation_program_graph((), removed)
    before = file_sha256(baseline), file_sha256(removed)
    restored = read_operation_program_graph(baseline)
    assert restored == (witness,)
    assert read_operation_program_graph(removed) == ()
    result = execute_integer_operation(restored[0], (92812, 1), ((1, a), (2, b)), facts)
    with pytest.raises(ValueError, match="permission"):
        verify_integer_operation(result, facts, read_operation_program_graph(removed))
    assert before == (file_sha256(baseline), file_sha256(removed))
    with sqlite3.connect(baseline) as connection:
        connection.execute(f"DELETE FROM {PART_TABLE} WHERE part_ordinal=2")
    with pytest.raises(ValueError, match="incomplete"):
        read_operation_program_graph(baseline)


def test_real_readonly_core_physical_train_program_and_same_query_evidence(tmp_path):
    """Reuse immutable teacher/Core artifacts without rerunning the old question."""
    source_path = os.environ.get("PIA_SAVED_COMPOSITION_SOURCE")
    if source_path is None:
        pytest.skip("requires an explicitly declared immutable original Core source")
    from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
    from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
    from pure_integer_ai.experiments.operation_program_query import (
        OPERATION_EXECUTION_ROLE, consume_operation_program,
    )
    from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
    from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
    from pure_integer_ai.experiments.trained_relation_composition_graph import read_relation_composition_graph
    from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
    from pure_integer_ai.experiments.trained_relation_question_graph import read_relation_question_graph
    source = Path(source_path).resolve(strict=True)
    proof_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if proof_root:
        tmp_path = Path(proof_root) / "development_operation_program_artifacts"
        tmp_path.mkdir(exist_ok=False)
    hashes = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in source.rglob("*") if path.is_file()}
    manifest = tuple((source / "teachers").rglob("manifest.json"))[0]
    teachers = read_independent_relation_fact_source(source / "data/ph2/facts.jsonl", manifest.parent)
    rule = read_relation_composition_graph(source / "rules.sqlite3")[0].rule
    frames, _ = read_relation_question_graph(source / "questions.sqlite3")
    with TrainedRelationGraphRuntime(source / "core.sqlite3") as core:
        facts = read_operation_core_context(core)
        schemas = {schema.relation: schema for schema in read_relation_composition_graph(source / "rules.sqlite3")[0].schemas}
        program = IntegerOperationProgram(frames[0].operation_key, (1, 2), (
            OperationAtom(schemas[rule.first_relation], tuple(sorted(((rule.first_input_role, 1),
                (rule.first_join_role, 2)), key=lambda row: row[0].stable_key()))),
            OperationAtom(schemas[rule.second_relation], tuple(sorted(((rule.second_join_role, 2),
                (rule.second_output_role, 3)), key=lambda row: row[0].stable_key())))),
            (), (OperationProjection(1, tuple(ObjectIdentity.from_stable_key(role)
                                             for role in frames[0].roles)),))
    physical = tmp_path / "operation.train.json"
    physical.write_bytes(canonical_json_bytes({"program_key": list(program.stable_key()),
        "license_id": "CC0-1.0", "owner": "teacher", "split": "train"}) + b"\n")
    witness = compile_train_operation_program(physical, teachers)
    graph = tmp_path / "operation.sqlite3"
    materialize_operation_program_graph((witness,), graph)
    syntax = json.loads((source / "question.train.json").read_bytes())["question"]
    with TrainedGraphQueryBridge(source / "core.sqlite3", memory_database=tmp_path / "memory.sqlite3",
            relation_question_graph_database=source / "questions.sqlite3",
            operation_program_graph_database=graph) as bridge:
        prepared = bridge.prepare_input_structure(syntax)
        state = QueryState((92812, 8))
        assert consume_operation_program(bridge.core_runtime, (witness,), prepared, state) == (state, None)
        entries = tuple(entry for fact in facts for entry in core_query_evidence(fact.definition.proposition,
            bridge.core_runtime.evidence_history(fact.definition.proposition),
            bridge.core_runtime.context.scoped_identity_store.registry))
        supplied = state.with_(evidence=entries)
        current, execution = consume_operation_program(bridge.core_runtime, (witness,), prepared, supplied)
        assert execution is not None and execution.status == READY
        assert any(row.role_key == OPERATION_EXECUTION_ROLE for row in current.bindings)
        assert current.read_count == len(request_context_proposition_keys(witness, prepared)) + len(facts)
        assert consume_operation_program(bridge.core_runtime, (), prepared, supplied) == (supplied, None)
        if proof_root:
            write_preflight_json(tmp_path, "checkpoint.json", {
                "stage": "PHYSICAL_PROGRAM_CORE_QUERY_EVIDENCE_GATE_NOT_NATURAL_PRODUCTION",
                "operation_program_sha256": file_sha256(graph), "physical_source_sha256": file_sha256(physical),
                "context_identity": list(execution.context_identity), "execution": list(execution.stable_key()),
                "actual_query_reexecution_count": 0, "generation_reexecution_count": 0,
                "language_operation_behavior": "UNKNOWN_NOT_MEASURED", "free_dialogue_complete": 0,
            })
    assert hashes == {path.relative_to(source).as_posix(): file_sha256(path)
                      for path in source.rglob("*") if path.is_file()}
