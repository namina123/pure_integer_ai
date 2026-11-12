"""核验原 QueryState 中被消费的语义义务；不查询图、不重新生成答案。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.integer_operation_behavior import (
    OperationOutputBehaviorEditReceipt, OperationSemanticEditReceipt, semantic_selection_obligation,
)
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, UNKNOWN, OperationExecution, operation_context_identity,
)
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramBudget
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, TERMINATION_ANSWER_CLOSED, TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING, TERMINATION_DIALOGUE_UNKNOWN, TERMINATION_NO_FRONTIER,
)
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.operation_program_query import (
    OPERATION_EXECUTION_ROLE, operation_execution_input_matches,
    operation_evidence_matches, operation_plan_trace_matches,
)
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack


FAIL_CLOSED = frozenset({TERMINATION_CLARIFY_CONFLICT, TERMINATION_CLARIFY_MISSING_BINDING,
                       TERMINATION_DIALOGUE_UNKNOWN, TERMINATION_NO_FRONTIER})


def read_saved_operation_output_proofs(value, trace=None):
    """按空间、当前scope和完整payload恢复权限，局部role不能代替对象类型。"""
    from pure_integer_ai.cognition.shared.integer_operation_output import OperationOutputProof
    from pure_integer_ai.cognition.understanding.query_operation_request_scope import (
        PENDING_REQUEST_SCOPE, PendingOperationRequestScope,
    )
    from pure_integer_ai.experiments.operation_output_query import OPERATION_OUTPUT_ROLE
    trace = GraphQueryTraceEvidence(value) if trace is None else trace
    query = trace.resolve_ref(value['query_ref'])
    proofs, pending, seen = [], 0, set()
    for ref in value['state_key']['bindings']:
        key = trace.resolve_ref(ref)
        role, filler, scope = _unpack(key, 1, 3, tail=1)
        if role != OPERATION_OUTPUT_ROLE:
            continue
        if key in seen or scope != query:
            raise ValueError('output or pending request binding is duplicated or has a foreign scope')
        seen.add(key)
        if key[0] == 3 and filler[:1] == (PENDING_REQUEST_SCOPE,):
            current = PendingOperationRequestScope.from_stable_key(filler)
            if (current.source_ref != query
                    or key != BindingEntry(role, filler, space=3, scope_key=query).stable_key()):
                raise ValueError('pending request binding changes its complete source/space')
            pending += 1
            continue
        if key != BindingEntry(role, filler, space=1, scope_key=query).stable_key():
            raise ValueError('output obligation binding has a foreign space/conflict')
        proofs.append(OperationOutputProof.from_stable_key(filler))
    if pending and proofs:
        raise ValueError('unresolved current request cannot also consume factual output permission')
    return tuple(proofs)


def read_consumed_operation_output_proof(value, facts, programs, witnesses, execution, *, prepared=None,
                                         report=None, declaration_reader=None):
    """Verify a consumed proof without treating it as delivered output."""
    from pure_integer_ai.cognition.shared.integer_operation_output_verifier import verify_operation_output
    from pure_integer_ai.experiments.operation_output_query import output_permission_evidence

    trace = GraphQueryTraceEvidence(value)
    query = trace.resolve_ref(value["query_ref"])
    bindings = tuple(trace.resolve_ref(ref) for ref in value["state_key"]["bindings"])
    selected = read_saved_operation_output_proofs(value, trace)
    if len(selected) != 1:
        raise ValueError("output trace lacks one exact consumed obligation proof")
    proof, = selected
    verify_operation_output(proof, facts, programs, witnesses)
    if proof.current_request_key:
        from pure_integer_ai.experiments.operation_request_call_audit import verify_operation_request_memory
        if (prepared is None or report is None or declaration_reader is None
                or len(prepared.operation_request_bindings) != 1
                or prepared.operation_request_bindings[0].stable_key() != proof.current_request_key):
            raise ValueError("output reference requires independently declared current request O/H/E")
        verify_operation_request_memory(prepared, value, report, declaration_reader)
    permission_keys = {trace.resolve_ref(ref) for ref in value["state_key"]["evidence"]}
    evidence = []
    for key in permission_keys:
        fields = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
    execution_permission = EvidenceEntry(execution.witness.source.stable_key(), execution.witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *query), scope_key=query, payload_key=execution.stable_key())
    if (query != execution.query_key or proof.execution_key != execution.stable_key()
            or BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), space=1,
                            scope_key=query).stable_key() not in bindings
            or output_permission_evidence(proof, query).stable_key() not in permission_keys
            or execution_permission not in evidence
            or not operation_evidence_matches(execution, facts, tuple(evidence))):
        raise ValueError("output proof lacks its original current selection/permission/Evidence")
    return proof


def read_consumed_operation_output(value, facts, programs, witnesses, execution, *, prepared=None,
                                   report=None, declaration_reader=None):
    """Admit delivery only when the original proof also closed Plan/generation."""
    proof = read_consumed_operation_output_proof(value, facts, programs, witnesses, execution,
        prepared=prepared, report=report, declaration_reader=declaration_reader)
    trace = GraphQueryTraceEvidence(value)
    generated = _decode_input(value.get("generation"), trace)
    if (value["termination"] != TERMINATION_ANSWER_CLOSED
            or value["state_key"]["scalars"][6:9] != [0, 1, 0]
            or value["state_key"]["scalars"][15] != 1
            or not operation_plan_trace_matches(execution, value.get("response_plan"), trace=trace)
            or not value.get("response_surface")
            or not isinstance(generated, dict)
            or generated.get("operation_execution") != list(execution.stable_key())
            or generated.get("operation_truth_added") != 0
            or tuple(generated.get("surface_values", ())) != tuple(map(ord, value["response_surface"]))
            or generated.get("operation_output_proof") != list(proof.stable_key())):
        raise ValueError("output obligation differs from the original consumed execution/Plan/generation")
    _verify_plan_fillers(value["response_plan"], trace, facts, execution)
    return proof


def _verify_plan_fillers(plan, trace, facts, execution):
    decoded = _decode_input(plan, trace)
    by_key = {fact.proposition_key: fact for fact in facts}
    slots = decoded["slot_sequence"]
    ranges = decoded["claim_slot_ranges"] or [
        {"claim": list(execution.outputs[0][0]), "start": 0, "end": len(slots)}]
    for row in ranges:
        fact = by_key[tuple(row["claim"])]
        fillers = {binding.role.stable_key(): binding.filler.stable_key() for binding in fact.definition.bindings}
        if any(tuple(slot["filler"]) != fillers[tuple(slot["role"])]
               for slot in slots[row["start"]:row["end"]]):
            raise ValueError("semantic Plan filler differs from current typed fact")


def read_consumed_operation_selection(value, facts, witness, prepared, *, core=None):
    """完整重验原选择证据；READY 选择不等于已经生成或交付。"""
    trace = GraphQueryTraceEvidence(value)
    query = trace.resolve_ref(value["query_ref"])
    state = value["state_key"]
    if (query != trace.resolve_ref(state["query_ref"]) or query != prepared.source_ref
            or _decode_input(value["input_structure"], trace) != prepared.integer_trace()
            or tuple(map(tuple, value["operation_keys"])) != (witness.program.operation_key,)
            or tuple(key for key, _mode in trace.operation_contracts()) != (witness.program.operation_key,)):
        raise ValueError("semantic trace input/query/operation binding drifted")
    selected = []
    for ref in state["bindings"]:
        key = trace.resolve_ref(ref)
        role, filler, scope = _unpack(key, 1, 3, tail=1)
        if role == OPERATION_EXECUTION_ROLE:
            execution = OperationExecution.from_stable_key(filler)
            if key != BindingEntry(role, filler, space=1, scope_key=query).stable_key() or scope != query:
                raise ValueError("semantic execution binding has a foreign scope/space/conflict")
            selected.append(execution)
    if len(selected) != 1:
        raise ValueError("semantic trace lacks one exact consumed execution proof")
    execution, = selected
    verify_integer_operation(execution, facts, (witness,))
    if (execution.query_key != query
            or not operation_execution_input_matches(execution, prepared, facts, core=core)):
        raise ValueError("semantic execution did not consume complete current input variables")
    evidence = []
    for ref in state["evidence"]:
        key = trace.resolve_ref(ref)
        fields = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
    expected = EvidenceEntry(witness.source.stable_key(), witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *query), scope_key=query, payload_key=execution.stable_key())
    if expected not in evidence or not operation_evidence_matches(execution, facts, tuple(evidence)):
        raise ValueError("semantic QueryState lacks exact permission/current premise Evidence")
    return execution


def read_consumed_operation_execution(value, facts, witness, prepared, *, core=None):
    """验证原选择与完整交付/拒答；READY 仍必须有闭合 Plan 和生成。"""
    execution = read_consumed_operation_selection(value, facts, witness, prepared, core=core)
    trace = GraphQueryTraceEvidence(value)
    state = value["state_key"]
    generated, plan = value.get("generation"), value.get("response_plan")
    if execution.status == READY:
        if (value["termination"] != TERMINATION_ANSWER_CLOSED or not generated
                or state["scalars"][6:9] != [0, 1, 0] or state["scalars"][15] != 1
                or not operation_plan_trace_matches(execution, plan, trace=trace)
                or _decode_input(generated, trace)["operation_execution"] != list(execution.stable_key())
                or _decode_input(generated, trace)["operation_truth_added"] != 0
                or tuple(_decode_input(generated, trace)["surface_values"]) != tuple(map(ord, value["response_surface"]))):
            raise ValueError("semantic ready execution lacks exact Plan/generation/closure")
        _verify_plan_fillers(plan, trace, facts, execution)
    elif (execution.status == UNKNOWN and generated
            and 'unknown_operation_execution' in _decode_input(generated, trace)):
        from pure_integer_ai.experiments.operation_unknown_response import verify_unknown_operation_delivery
        verify_unknown_operation_delivery(execution, value, trace)
    elif (value["termination"] not in FAIL_CLOSED or plan or generated or value.get("response_surface")
            or state["scalars"][7] or state["scalars"][15]):
        raise ValueError("semantic failed obligation retained factual Plan/generation/closure")
    return execution


def audit_consumed_semantic_pair(edit, baseline, counterfactual, facts, prepared, *, core=None):
    """真实被消费义务须改变 matches/proof/输出或拒答；标签和空闲编辑无接纳资格。"""
    if not isinstance(edit, OperationSemanticEditReceipt):
        raise TypeError("semantic pair requires a full verified physical edit receipt")
    if (operation_context_identity(facts, OperationProgramBudget()) != edit.context_identity
            or prepared.stable_key() != edit.input_structure_key):
        raise ValueError("semantic pair independent Core/input identity drifted")
    before = read_consumed_operation_execution(baseline, facts, edit.before, prepared, core=core)
    after = read_consumed_operation_execution(counterfactual, facts, edit.after, prepared, core=core)
    if before.status != READY:
        raise ValueError("semantic baseline obligation did not produce a current answer")
    if before.variables != after.variables or before.context_identity != after.context_identity:
        raise ValueError("semantic pair changed actual variables or Core context")
    changes = tuple(name for name, left, right in (
        ("complete_matches", before.matches, after.matches),
        ("consumed_fact_proofs", before.fact_proofs, after.fact_proofs),
        ("operation_status", before.status, after.status),
        ("ordered_outputs", before.outputs, after.outputs),
        ("termination", baseline["termination"], counterfactual["termination"]),
        ("response_surface", baseline["response_surface"], counterfactual["response_surface"]),
    ) if left != right)
    if (not {"complete_matches", "ordered_outputs", "operation_status"}.intersection(changes)
            or not {"ordered_outputs", "termination"}.intersection(changes)
            or "response_surface" not in changes):
        raise ValueError("semantic graph edit did not change actual consumed selection and delivery/refusal")
    return {"status": "PASS_DEVELOPMENT_CONSUMED_GRAPH_SEMANTIC_COUNTERFACTUAL_ONLY",
        "edit_receipt_key": list(edit.stable_key()), "changed_fields": list(changes),
        "baseline_obligation_key": list(semantic_selection_obligation(edit.before.program)),
        "counterfactual_obligation_key": list(semantic_selection_obligation(edit.after.program)),
        "baseline_operation_status": before.status, "counterfactual_operation_status": after.status,
        "baseline_output_count": len(before.outputs), "counterfactual_output_count": len(after.outputs),
        "counterfactual_fail_closed": int(counterfactual["termination"] in FAIL_CLOSED),
        "verified_semantic_pair_count": 1, "qualified_semantic_operation_families": 0,
        "operation_family_qualification": "UNKNOWN_REQUIRES_EIGHT_DISTINCT_FROZEN_BEHAVIOR_OBLIGATIONS",
        "held_out_novelty": "UNKNOWN_NOT_QUALIFIED_BY_A_SEMANTIC_PAIR", "free_dialogue_complete": 0}


def audit_consumed_output_behavior_pair(edit, baseline, counterfactual, facts,
                                        before_prepared, after_prepared,
                                        before_programs, after_programs, *, core=None):
    """Verify a real output-scope refusal or a content-preserving graph rewrite."""
    if not isinstance(edit, OperationOutputBehaviorEditReceipt):
        raise TypeError("output behavior pair requires a complete physical output edit receipt")
    if (operation_context_identity(facts, OperationProgramBudget()) != edit.context_identity
            or before_prepared.stable_key() != edit.before_input_structure_key
            or after_prepared.stable_key() != edit.after_input_structure_key):
        raise ValueError("output behavior pair input or independent Core identity drifted")
    before_execution = read_consumed_operation_selection(
        baseline, facts, edit.before, before_prepared, core=core)
    after_execution = read_consumed_operation_selection(
        counterfactual, facts, edit.after, after_prepared, core=core)
    if (before_execution.query_key != after_execution.query_key
            or before_execution.variables != after_execution.variables
            or before_execution.context_identity != after_execution.context_identity):
        raise ValueError("output behavior pair changed the current query, variables, or Core context")

    def output_role_count(value):
        """统计当前完整空间1权限，不把空间3缺槽证明当作权限。"""
        return len(read_saved_operation_output_proofs(value))

    before_count, after_count = output_role_count(baseline), output_role_count(counterfactual)
    if before_count != 1:
        raise ValueError("baseline output behavior lacks one consumed output proof")
    before_proof = read_consumed_operation_output(baseline, facts, before_programs,
        (edit.before_output,), before_execution, prepared=before_prepared)
    after_proof = None
    if after_count == 1 and edit.after_output is None:
        raise ValueError("counterfactual consumed output permission absent from its physical graph")
    if after_count == 1:
        after_proof = read_consumed_operation_output(counterfactual, facts, after_programs,
            (edit.after_output,), after_execution, prepared=after_prepared)
    elif after_count != 0:
        raise ValueError("counterfactual output behavior has competing output proofs")

    projection_before = frozenset((row.atom, row.roles) for row in edit.before.program.projections)
    projection_after = frozenset((row.atom, row.roles) for row in edit.after.program.projections)
    if projection_before != projection_after:
        state = counterfactual["state_key"]
        if (edit.after_output is not None or after_count != 0
                or before_execution.status != READY or after_execution.status != READY or after_proof is not None
                or counterfactual["termination"] not in FAIL_CLOSED
                or counterfactual.get("response_plan") or counterfactual.get("generation")
                or counterfactual.get("response_surface") or state["scalars"][7]
                or state["scalars"][15]
                or len(after_execution.outputs) >= len(before_execution.outputs)
                or not set(key for key, _roles in after_execution.outputs) <
                    set(key for key, _roles in before_execution.outputs)
                or before_execution.matches != after_execution.matches
                or before_execution.fact_proofs != after_execution.fact_proofs
                or baseline.get("response_surface") == counterfactual.get("response_surface")):
            raise ValueError("scope reduction did not alter the delivered set and fail closed")
        shape = "CURRENT_SCOPE_REDUCTION_REFUSED"
    else:
        before_roles = {key: frozenset(roles) for key, roles in before_execution.outputs}
        after_roles = {key: frozenset(roles) for key, roles in after_execution.outputs}
        if (before_execution.status != READY or after_execution.status != READY or after_proof is None
                or after_proof.preserved_execution_key != before_execution.stable_key()
                or before_execution.matches != after_execution.matches
                or before_execution.fact_proofs != after_execution.fact_proofs
                or before_roles != after_roles or before_execution.outputs == after_execution.outputs
                or baseline["termination"] != TERMINATION_ANSWER_CLOSED
                or counterfactual["termination"] != TERMINATION_ANSWER_CLOSED
                or baseline.get("response_surface") == counterfactual.get("response_surface")):
            raise ValueError("rewrite changed claim evidence, failed preservation, or did not change realization")
        shape = "SAME_CLAIMS_PRESERVED_GRAPH_REWRITE"
    return {"status": "PASS_DEVELOPMENT_OUTPUT_BEHAVIOR_COUNTERFACTUAL_ONLY",
        "shape": shape,
        "before_program_key": list(edit.before.stable_key()),
        "after_program_key": list(edit.after.stable_key()),
        "before_output_key": list(edit.before_output.stable_key()),
        "after_output_key": list(edit.after_output.stable_key()),
        "before_operation_status": before_execution.status,
        "after_operation_status": after_execution.status,
        "before_output_count": len(before_execution.outputs),
        "after_output_count": len(after_execution.outputs),
        "before_output_proof_present": 1, "after_output_proof_present": int(after_proof is not None),
        "baseline_termination": baseline["termination"],
        "counterfactual_termination": counterfactual["termination"],
        "response_surface_changed": int(baseline.get("response_surface") != counterfactual.get("response_surface")),
        "verified_output_behavior_pair_count": 1,
        "qualified_semantic_operation_families": 0,
        "operation_family_qualification": "UNKNOWN_REQUIRES_FROZEN_HELD_OUT_BEHAVIOR_EVALUATION",
        "held_out_novelty": "UNKNOWN_NOT_QUALIFIED_BY_A_SEMANTIC_PAIR",
        "free_dialogue_complete": 0}
