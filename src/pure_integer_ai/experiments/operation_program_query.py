"""Current Core/QueryState boundary for source-qualified operation programs."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, execute_integer_operation, operation_context_conflicted,
)
from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramBudget
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_CLARIFY_CONFLICT, BindingEntry, EvidenceEntry,
)
from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_FILLER
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase
from pure_integer_ai.experiments.operation_core_context import (
    read_operation_core_context, request_context_proposition_keys,
)
from pure_integer_ai.experiments.trained_query_evidence import current_query_evidence


OPERATION_EXECUTION_ROLE = (92230, 1)


def _request_variables(witness, structure, facts, *, core=None):
    """Bind full parsed question slots, not an operation label supplied by a runner."""
    program = witness.program
    if structure.operation_keys != (program.operation_key,):
        return None
    if structure.operation_request_bindings:
        from pure_integer_ai.experiments.operation_request_consumer import request_variables_from_readers
        return request_variables_from_readers(witness, structure, facts, core)
    questions = tuple(binding for binding in structure.relation_question_bindings
                      if binding.operation_key == program.operation_key)
    if len(questions) != 1:
        return None
    question = questions[0]
    relations = tuple(item for item in structure.relation_candidates
                      if item.stable_key() == question.relation_key)
    if len(relations) != 1:
        return None
    first = program.atoms[0]
    if first.schema.relation.stable_key() != relations[0].predicate_key:
        return None
    source = next((fact for fact in facts if fact.proposition_key == relations[0].proposition_key), None)
    if source is None:
        return None
    first.schema.validate_definition(source.definition)
    fillers = {binding.role: binding.filler for binding in source.definition.bindings}
    variables = {}
    for role, variable in first.role_variables:
        if variable not in program.external_variables:
            continue
        candidates = {item.candidate_key for item in structure.semantic_candidates
            if item.projection_kind == PROJECTION_FILLER
            and item.proposition_key == source.proposition_key and item.role_key == role.stable_key()
            and item.predicate_key == relations[0].predicate_key and item.source_ref == relations[0].source_ref
            and item.span_ref in question.slot_span_refs
            and item.candidate_key == fillers[role].stable_key()}
        if len(candidates) != 1:
            return None
        if variable in variables and variables[variable] != fillers[role]:
            return None
        variables[variable] = fillers[role]
    if set(variables) != set(program.external_variables):
        return None
    return tuple(sorted(variables.items()))


@observed_relation_phase("core")
def consume_operation_program(core, witnesses, structure, state, budget=OperationProgramBudget()):
    """No matched fact can be consumed without its exact Evidence in this query."""
    choices = tuple(witness for witness in witnesses if witness.program.operation_key in structure.operation_keys)
    if len(choices) != 1:
        return state, None
    inputs = read_operation_core_context(core, budget,
        proposition_keys=request_context_proposition_keys(choices[0], structure))
    variables = _request_variables(choices[0], structure, inputs, core=core)
    if variables is None:
        return state, None
    facts = read_operation_core_context(core, budget, witness=choices[0], variables=variables)
    reads = len(inputs) + len(facts)
    if (state.read_count + reads > state.budget.max_reads
            or state.node_count + 1 > state.budget.max_nodes
            or state.edge_count + 2 > state.budget.max_edges):
        return state, None
    execution = execute_integer_operation(choices[0], state.query_key, variables, facts, budget=budget)
    verify_integer_operation(execution, facts, witnesses, budget=budget)
    if not operation_evidence_matches(execution, facts, state.evidence):
        return state, None
    proof = execution.stable_key()
    binding = BindingEntry(OPERATION_EXECUTION_ROLE, proof, space=1, scope_key=state.query_key)
    evidence = EvidenceEntry(execution.witness.source.stable_key(), execution.witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *state.query_key), scope_key=state.query_key,
        payload_key=proof)
    state = state.with_(bindings=tuple(sorted(set((*state.bindings, binding)), key=lambda row: row.stable_key())),
        evidence=tuple(sorted(set((*state.evidence, evidence)), key=lambda row: row.stable_key())),
        read_count=state.read_count + reads, node_count=state.node_count + 1,
        edge_count=state.edge_count + 2)
    if operation_context_conflicted(facts):
        state = state.with_(termination=TERMINATION_CLARIFY_CONFLICT,
            evidence_closed=0, generation_ready=0, conflict_open=1)
    return state, execution


def operation_evidence_matches(execution, facts, evidence):
    """Every used premise needs its exact complete current Evidence in this query."""
    current = current_query_evidence(evidence)
    used = {key for match in execution.matches for key in match}
    return all(fact.evidence_keys == tuple(sorted(entry.stable_key() for entry in current
        if entry.space == 1 and entry.hypothesis_key == fact.proposition_key
        and (entry.evidence_key[:1] == (1,) or entry.polarity != 3)))
        for fact in facts if fact.proposition_key in used)


def _projection_matches(execution, claims, ranges, slot_roles):
    """核对 claim 集合及按输出 occurrence 连续覆盖的完整角色范围。"""
    if execution is None or execution.status != READY:
        return False
    selected = tuple(key for key, _ in execution.outputs)
    if not selected or len(claims) != len(selected) or set(claims) != set(selected):
        return False
    if not ranges and len(selected) == 1:
        ranges = ((selected[0], 0, len(slot_roles)),)
    if len(ranges) != len(selected) or {key for key, _start, _end in ranges} != set(selected):
        return False
    if any(type(start) is not int or type(end) is not int or start < 0
           or end <= start or end > len(slot_roles) for _key, start, end in ranges):
        return False
    ordered = tuple(sorted(ranges, key=lambda row: row[1]))
    offset = 0
    for (key, roles), (claim, start, end) in zip(execution.outputs, ordered):
        if key != claim or start != offset or slot_roles[start:end] != roles:
            return False
        offset = end
    return offset == len(slot_roles)


def operation_plan_matches(execution, plan):
    """验证真实 Plan；身份排序的 claim_refs 不定义输出顺序。"""
    if plan is None:
        return False
    return _projection_matches(execution,
        tuple(claim.stable_key() for claim in plan.claim_refs), plan.claim_slot_ranges,
        tuple(slot.role.stable_key() for slot in plan.slot_sequence))


def operation_plan_trace_matches(execution, plan, *, trace=None):
    """采集与冷审计复用相同合同，拒绝缺槽、范围重叠或输出次序漂移。"""
    def key(value):
        if isinstance(value, dict) and set(value) == {"integer_key_ref"} and trace is not None:
            return trace.resolve_ref(value["integer_key_ref"])
        if not isinstance(value, (tuple, list)) or any(type(item) is not int for item in value):
            raise ValueError("operation plan trace key must contain integers")
        return tuple(value)

    try:
        return _projection_matches(execution,
            tuple(key(claim) for claim in plan["claim_refs"]),
            tuple((key(row["claim"]), row["start"], row["end"])
                  for row in plan["claim_slot_ranges"]),
            tuple(key(row["role"]) for row in plan["slot_sequence"]))
    except (KeyError, TypeError, ValueError):
        return False
