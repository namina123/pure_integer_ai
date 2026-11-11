"""Cold diagnostics of saved production gates; no query, executor or proof producer."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.integer_operation_execution import OperationExecution
from pure_integer_ai.cognition.shared.integer_operation_verifier import independent_operation_matches
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef, MEMORY_OBJECT_EVIDENCE
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.understanding.query_memory_candidates import memory_operation_request_binding
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.operation_output_query import consumed_current_request_key
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE, _request_variables
from pure_integer_ai.experiments.operation_request_call_audit import verify_operation_request_memory
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_graph_query_bridge import MemoryHypothesisAnchor
from pure_integer_ai.experiments.trained_query_evidence import current_query_evidence
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


def saved_operation_state(value):
    trace = GraphQueryTraceEvidence(value)
    bindings, evidence = [], []
    for ref in value["state_key"]["bindings"]:
        key = trace.resolve_ref(ref)
        role, filler, scope = _unpack(key, 1, 3, tail=1)
        bindings.append(BindingEntry(role, filler, key[0], scope, key[-1]))
    for ref in value["state_key"]["evidence"]:
        key = trace.resolve_ref(ref)
        fields = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
    return SimpleNamespace(query_key=trace.resolve_ref(value["query_ref"]),
                           bindings=tuple(bindings), evidence=tuple(evidence))


def read_saved_request_hypotheses(database, value, report, prepared):
    """Use the actual persisted hash-prefixed source format of runtime anchors."""
    source = SourceRef.from_stable_key(tuple(report["input_source"]))
    access = MemoryAccessContext(source.owner.tenant_id, source.owner.user_id, source.owner.session_id)
    backend = SQLiteBackend(str(database), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        declarations = lambda ref: tuple(row for row in intake.event_log.query(access=access, object_ref=ref)
                                         if row.event.is_declaration)
        verify_operation_request_memory(prepared, value, report, declarations)
        source_hash = intake.event_log.scoped_identities.registry.find(IDENTITY_SOURCE_RECORD, source.stable_key())
        if source_hash is None:
            raise ValueError("saved request has no durable source hash")
        evidence = {}
        for ref in report["evidence_refs"]:
            key = tuple(ref)
            rows = declarations(MemoryObjectRef.from_stable_key(key))
            if len(rows) != 1:
                raise ValueError("saved request Evidence declaration is not unique")
            evidence.setdefault(rows[0].event.payload.hypothesis_ref.stable_key(), []).append(key)
        hypotheses = []
        for ref in report["hypothesis_refs"]:
            key = tuple(ref)
            rows = declarations(MemoryObjectRef.from_stable_key(key))
            if len(rows) != 1:
                raise ValueError("saved request Hypothesis declaration is not unique")
            hypothesis = rows[0].event.payload.hypothesis
            if memory_operation_request_binding(hypothesis.candidate_key) is None:
                continue
            if hypothesis.observation != source:
                raise ValueError("saved request Hypothesis belongs to another source")
            hypotheses.append(MemoryHypothesisAnchor(tuple(report["observation_ref"]), key,
                hypothesis.hypothesis_kind, hypothesis.candidate_key, hypothesis.competition_key,
                (source_hash, *source.stable_key()), hypothesis.scope.stable_key(), tuple(sorted(evidence[key]))))
        return tuple(hypotheses)
    finally:
        backend.close()


def inspect_saved_operation_gate(value, report, prepared, witness, facts, *, core, memory_database):
    """Report reference readiness and missing premises without recreating output."""
    state = saved_operation_state(value)
    trace = GraphQueryTraceEvidence(value)
    executions = tuple(OperationExecution.from_stable_key(item.filler_key) for item in state.bindings
                       if item.role_key == OPERATION_EXECUTION_ROLE)
    if len(executions) > 1:
        raise ValueError("saved query has competing execution proofs")
    variables = _request_variables(witness, prepared, facts, core=core)
    matches = () if variables is None else independent_operation_matches(witness, variables, facts)
    used = {key for match in matches for key in match}
    current = current_query_evidence(state.evidence)
    premises = []
    for fact in facts:
        if fact.proposition_key not in used:
            continue
        present = {entry.stable_key() for entry in current if entry.space == 1
            and entry.hypothesis_key == fact.proposition_key
            and (entry.evidence_key[:1] == (1,) or entry.polarity != 3)}
        expected = set(fact.evidence_keys)
        premises.append({"claim_values": len(fact.proposition_key),
            "missing_evidence_count": len(expected - present), "unexpected_evidence_count": len(present - expected)})
    hypotheses = read_saved_request_hypotheses(memory_database, value, report, prepared)
    request = (() if not executions else consumed_current_request_key(
        executions[0], state, hypotheses, tuple(report["observation_ref"])))
    bound = {item.filler_key for item in state.bindings if item.space == 2 and item.role_key == (MEMORY_OBJECT_EVIDENCE,)}
    role_keys = {trace.resolve_ref(ref) for ref in value["input_role_occurrence_refs"]}
    initial = [item for item in state.evidence if item.space == 2 and item.evidence_key[:1] == (1,)]
    structural = [item for item in state.evidence if item.space == 2
        and item.evidence_key[:1] != (1,) and item.evidence_key not in role_keys
        and item.evidence_key[:2] != (91572, 1)]
    prefixes = Counter(tuple(item.evidence_key[:3]) for item in structural)
    return {"complete_current_variables_present": int(variables is not None),
        "independent_complete_match_count": len(matches), "independent_premise_count": len(premises),
        "premise_evidence_differences": premises, "original_execution_count": len(executions),
        "current_request_gate_readiness": int(bool(request)),
        "current_request_hypothesis_count": len(hypotheses),
        "unconsumed_current_intake_evidence_count": len(set(map(tuple, report["evidence_refs"])) - bound),
        "binding_conflict_count": sum(item.conflict_kept for item in state.bindings),
        "initial_ohe_evidence_count": len(initial), "structural_memory_evidence_count": len(structural),
        "structural_memory_prefix_counts": [{"prefix": list(key), "count": count} for key, count in sorted(prefixes.items())],
        "reference_candidate_set_count": len(value["reference_candidate_set_refs"]),
        "reference_resolution_count": len(value["reference_resolution_refs"]),
        "independent_match_scope": "COLD_DIAGNOSTIC_ONLY_NOT_ORIGINAL_EXECUTION_OR_DELIVERY",
        "query_reexecution_count": 0, "executor_reexecution_count": 0, "generation_reexecution_count": 0,
        "proof_producer_reexecution_count": 0, "source_write_count": 0, "free_dialogue_complete": 0}
