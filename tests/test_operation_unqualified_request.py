"""完整对象可用于无许可 UNKNOWN，事实消费与真实来源核验仍严格关闭。"""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from test_integer_operation_program import _atom, _fact, _schema, _source, _witness, LEFT, RIGHT
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, UNKNOWN, CLARIFY, execute_integer_operation,
)
from pure_integer_ai.cognition.shared.integer_operation_program import IntegerOperationProgram, OperationProjection
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import QueryState
from pure_integer_ai.cognition.shared.semantic_object import entity_identity
from pure_integer_ai.cognition.understanding.query_input_structure import InputSemanticCandidate
from pure_integer_ai.cognition.understanding.query_operation_requests import (
    FRAME, OperationRequestFrame, match_operation_request_frames,
)
from pure_integer_ai.experiments.operation_program_query import (
    _request_variables, _unknown_request_variables, consume_operation_program,
    operation_execution_input_matches,
)


@pytest.fixture
def reader_case():
    """使用真实整数 frame、reader、事实与独立执行器，资源 owner 仅提供词形视图。"""
    schema = _schema(81)
    a, b = (entity_identity(_source(8100), (i,)) for i in (1, 2))
    fact = _fact(8101, schema.relation, a, b)
    other = _fact(8102, schema.relation, a, b)
    program = IntegerOperationProgram((92811, 81), (1, 2), (_atom(schema, (1, 2)),), (),
        (OperationProjection(1, (LEFT, RIGHT)),))
    witness = _witness(program, (other,))
    digest = hashlib.sha256(b'integer-reader-unknown-fixture').digest()
    source = SourceRef(FRAME, (int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)) or 1,
        0, witness.source.owner, witness.source.versions)
    frame = OperationRequestFrame(program.operation_key, source, tuple(digest),
        witness.source_identity, witness.license_identity,
        ((1, schema.relation.stable_key(), LEFT.stable_key(), (a.object_kind,)),
         (2, schema.relation.stable_key(), RIGHT.stable_key(), (b.object_kind,))),
        ((101,), (102,), (103,)))
    spans = tuple(SimpleNamespace(ref_key=(8103, i), start=i * 2 - 1, end=i * 2) for i in (1, 2))
    members = tuple(InputSemanticCandidate(target.stable_key(), target.object_kind, 1,
        span.ref_key, fact.proposition_key, schema.relation.stable_key(), role.stable_key(), (8104,), i)
        for i, (target, role, span) in enumerate(zip((a, b), (LEFT, RIGHT), spans)))
    structure = SimpleNamespace(tokens=(1,), token_values=(101, 201, 102, 202, 103),
        source_ref=(92812, 81), spans=spans, semantic_candidates=members,
        relation_question_bindings=(), pending_relation_composition_bindings=(),
        operation_keys=(program.operation_key,), operation_request_bindings=())
    structure.operation_request_bindings = match_operation_request_frames((frame,), structure)
    assert len(structure.operation_request_bindings) == 1
    surface = SimpleNamespace(proposition=fact.definition.proposition, source_hash=8104)
    core = SimpleNamespace(active_surface_facts=lambda: (surface,))
    return witness, structure, (fact,), core, ((1, a), (2, b)), other


def test_identity_only_unknown_keeps_original_permission_and_full_fact_context(reader_case):
    witness, structure, facts, core, variables, _other = reader_case
    before = witness.stable_key(), tuple(f.stable_key() for f in facts)
    assert _request_variables(witness, structure, facts, core=core) is None
    assert _unknown_request_variables(witness, structure, facts, core=core) == variables
    execution = execute_integer_operation(witness, structure.source_ref, variables, facts)
    assert execution.status == UNKNOWN and not execution.matches and not execution.fact_proofs
    assert verify_integer_operation(execution, facts, (witness,))
    assert operation_execution_input_matches(execution, structure, facts, core=core)
    assert (witness.stable_key(), tuple(f.stable_key() for f in facts)) == before
    assert _request_variables(witness, structure, facts, core=core) is None


@pytest.mark.parametrize('change', ('ready', 'clarify', 'matches', 'proofs', 'foreign_variable'))
def test_identity_only_cannot_authorize_any_selection_or_foreign_variable(reader_case, change):
    witness, structure, facts, core, variables, _other = reader_case
    execution = execute_integer_operation(witness, structure.source_ref, variables, facts)
    if change in ('ready', 'clarify'):
        execution = replace(execution, status=READY if change == 'ready' else CLARIFY)
    elif change == 'matches':
        execution = replace(execution, matches=((facts[0].proposition_key,),))
    elif change == 'proofs':
        execution = replace(execution, fact_proofs=(facts[0].stable_key(),))
    else:
        execution = replace(execution, variables=((1, variables[1][1]), (2, variables[0][1])))
    assert not operation_execution_input_matches(execution, structure, facts, core=core)


@pytest.mark.parametrize('change', ('source', 'fact', 'core', 'prefix', 'operation'))
def test_identity_only_still_requires_exact_physical_source_complete_input_and_current_program(reader_case, change):
    witness, structure, facts, core, _variables, _other = reader_case
    if change == 'source':
        structure.semantic_candidates = tuple(replace(member, source_ref=(999,))
            for member in structure.semantic_candidates)
    elif change == 'fact':
        facts = ()
    elif change == 'core':
        core = None
    elif change == 'prefix':
        structure.token_values = (999, *structure.token_values)
    else:
        structure.operation_keys = ((999,),)
    assert _unknown_request_variables(witness, structure, facts, core=core) is None


def test_actual_consumer_commits_empty_unknown_and_cold_selection_matches(reader_case, monkeypatch):
    witness, structure, facts, core, _variables, _other = reader_case
    monkeypatch.setattr('pure_integer_ai.experiments.operation_program_query.read_operation_core_context',
        lambda *_args, **_kwargs: facts)
    state = QueryState(structure.source_ref)
    updated, execution = consume_operation_program(core, (witness,), structure, state)
    assert execution.status == UNKNOWN and not execution.outputs
    assert updated.read_count == state.read_count + 2
    assert updated.node_count == state.node_count + 1
    assert len(updated.bindings) == len(updated.evidence) == 1
    assert verify_integer_operation(execution, facts, (witness,))


def test_actual_consumer_refuses_ready_when_only_reader_source_is_unlicensed(reader_case, monkeypatch):
    witness, structure, facts, core, variables, other = reader_case
    calls = 0
    def read(*_args, **_kwargs):
        """输入仍为无许可来源，真实程序上下文另有已许可正证据。"""
        nonlocal calls
        calls += 1
        return facts if calls == 1 else (*facts, other)
    monkeypatch.setattr('pure_integer_ai.experiments.operation_program_query.read_operation_core_context', read)
    assert execute_integer_operation(witness, structure.source_ref, variables, (*facts, other)).status == READY
    state = QueryState(structure.source_ref)
    updated, execution = consume_operation_program(core, (witness,), structure, state)
    assert execution is None and updated == state


def test_authorized_reader_keeps_original_ready_selection(reader_case):
    witness, structure, facts, core, variables, _other = reader_case
    witness = replace(witness, authorized_forming_sources=facts[0].candidate.forming_sources)
    execution = execute_integer_operation(witness, structure.source_ref, variables, facts)
    assert execution.status == READY
    assert _request_variables(witness, structure, facts, core=core) == variables
    assert operation_execution_input_matches(execution, structure, facts, core=core)
