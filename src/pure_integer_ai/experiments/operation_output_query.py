"""Consume licensed output obligations in the current production QueryState."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.integer_operation_output_verifier import prove_operation_output
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramBudget
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.experiments.operation_program_query import (
    OPERATION_EXECUTION_ROLE, operation_evidence_matches, read_operation_core_context,
)
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase


OPERATION_OUTPUT_ROLE = (92410, 1)


@observed_relation_phase("core")
def consume_operation_output(core, witnesses, programs, execution, state, *, budget=OperationProgramBudget(),
                             memory_hypotheses=(), current_observation_key=()):
    """A missing or unsatisfied obligation supplies no generation permission."""
    if not execution_proof_consumed(execution, state):
        return state, None
    selected = tuple(item for item in witnesses if item.contract.program_witness_key == execution.witness.stable_key())
    if len(selected) != 1:
        return state, None
    facts = read_operation_core_context(core, budget,
        witness=execution.witness, variables=execution.variables)
    if (state.read_count + len(facts) > state.budget.max_reads
            or state.node_count + 1 > state.budget.max_nodes
            or state.edge_count + 2 > state.budget.max_edges
            or not operation_evidence_matches(execution, facts, state.evidence)):
        return state, None
    try:
        current_request = (() if not selected[0].contract.current_request_required else
            consumed_current_request_key(execution, state, memory_hypotheses, current_observation_key))
        if selected[0].contract.current_request_required and not current_request:
            return state, None
        proof = prove_operation_output(selected[0], execution, facts, programs, budget=budget,
                                       current_request_key=current_request)
    except ValueError:
        return state, None
    binding = BindingEntry(OPERATION_OUTPUT_ROLE, proof.stable_key(), space=1, scope_key=state.query_key)
    permission = output_permission_evidence(proof, state.query_key)
    return state.with_(
        bindings=tuple(sorted(set((*state.bindings, binding)), key=lambda item: item.stable_key())),
        evidence=tuple(sorted(set((*state.evidence, permission)), key=lambda item: item.stable_key())),
        read_count=state.read_count + len(facts), node_count=state.node_count + 1,
        edge_count=state.edge_count + 2), proof


def output_permission_evidence(proof, query_key):
    return EvidenceEntry(proof.witness.source.stable_key(), proof.witness.contract.program_witness_key,
        space=1, polarity=3, evidence_key=(92410, *query_key), scope_key=query_key, payload_key=proof.stable_key())


def output_proof_consumed(proof, execution, state):
    if proof is None or execution is None or proof.execution_key != execution.stable_key():
        return False
    return (execution_proof_consumed(execution, state)
        and BindingEntry(OPERATION_OUTPUT_ROLE, proof.stable_key(), space=1,
            scope_key=state.query_key) in state.bindings
        and output_permission_evidence(proof, state.query_key) in state.evidence)


def execution_proof_consumed(execution, state):
    if execution is None or execution.status != READY or execution.query_key != state.query_key:
        return False
    permission = EvidenceEntry(execution.witness.source.stable_key(), execution.witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *state.query_key), scope_key=state.query_key,
        payload_key=execution.stable_key())
    return (BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), space=1,
            scope_key=state.query_key) in state.bindings and permission in state.evidence)


def consumed_current_request_key(execution, state, hypotheses, current_observation_key):
    """Use the current complete O/H/E and its Companion mirror, never an old answer."""
    from dataclasses import replace
    from pure_integer_ai.cognition.shared.memory_event import MEMORY_OBJECT_EVIDENCE, MEMORY_OBJECT_OBSERVATION
    from pure_integer_ai.cognition.understanding.query_memory_candidates import (
        MEMORY_CATEGORY_OPERATION_REQUEST, MEMORY_TOPIC_PROOF_VERSION, _parts,
        frame_key, memory_operation_request_binding,
    )

    selected = []
    for hypothesis in hypotheses:
        request = memory_operation_request_binding(hypothesis.candidate_key)
        if (request is None or request.source_ref != execution.query_key
                or request.operation_key != execution.witness.program.operation_key
                or not current_observation_key or hypothesis.observation_key != current_observation_key):
            continue
        if (BindingEntry(hypothesis.hypothesis_kind, hypothesis.candidate_key, space=2,
                scope_key=hypothesis.scope_key) not in state.bindings
                or BindingEntry((MEMORY_OBJECT_OBSERVATION,), hypothesis.observation_key, space=2,
                    scope_key=hypothesis.scope_key) not in state.bindings):
            continue
        evidence = tuple(item for item in state.evidence
            if item.space == 2 and item.hypothesis_key == hypothesis.hypothesis_key)
        if len(evidence) != 1:
            continue
        entry, = evidence
        from pure_integer_ai.cognition.shared.memory_event import EvidencePayload
        try:
            payload = EvidencePayload.from_stable_key(entry.payload_key)
            detail = _parts(payload.detail, 91507)
        except ValueError:
            continue
        expected = frame_key(MEMORY_TOPIC_PROOF_VERSION, request.source_ref, request.stable_key())
        if (entry.polarity != 3 or entry.trust != 1 or entry.supersedes_key
                or entry.evidence_key not in hypothesis.evidence_keys
                or len(hypothesis.source_ref) < 2 or type(hypothesis.source_ref[0]) is not int
                or hypothesis.source_ref[0] <= 0
                or entry.source_ref != hypothesis.source_ref or entry.scope_key != hypothesis.scope_key
                or payload.source is None or payload.source.stable_key() != hypothesis.source_ref[1:]
                or payload.hypothesis_ref.stable_key() != hypothesis.hypothesis_key
                or payload.stance != 3 or payload.supersedes_ref is not None
                or len(detail) != 3 or detail[0][-1:] != (MEMORY_CATEGORY_OPERATION_REQUEST,)
                or detail[1:] != (hypothesis.candidate_key, expected)
                or replace(entry, space=3) not in state.evidence
                or BindingEntry((MEMORY_OBJECT_EVIDENCE,), entry.evidence_key, space=2,
                    scope_key=entry.scope_key) not in state.bindings):
            continue
        selected.append(request.stable_key())
    return selected[0] if len(selected) == 1 else ()
