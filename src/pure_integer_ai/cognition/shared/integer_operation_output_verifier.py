"""Output proof producer and independent admission over complete current facts."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, OperationExecution, execute_integer_operation,
)
from pure_integer_ai.cognition.shared.integer_operation_output import (
    OperationOutputProof, OperationOutputWitness, PATH_EDGE,
)
from pure_integer_ai.cognition.shared.integer_operation_program import (
    OperationProgramBudget, OperationProgramWitness, SEQUENCE,
)
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record


def validate_output_permission(witness, programs):
    if not isinstance(witness, OperationOutputWitness) or type(programs) is not tuple:
        raise TypeError("output permission requires complete physical witnesses")
    current = OperationProgramWitness.from_stable_key(witness.contract.program_witness_key)
    if current not in programs:
        raise ValueError("output obligation has no exact current program permission")
    if witness.contract.preserved_witness_key and OperationProgramWitness.from_stable_key(
            witness.contract.preserved_witness_key) not in programs:
        raise ValueError("output preservation has no independently authorized reference program")
    return current


def _obligation_members(contract, execution, facts):
    """Use actual fillers, not variable IDs or proposition ordering, for paths."""
    by_key = {fact.proposition_key: fact for fact in facts}
    paths, comparisons, scope = set(), set(), set()
    output_keys = {key for key, _ in execution.outputs}
    for match in execution.matches:
        program = execution.witness.program
        if program.path_query is not None:
            if (len(execution.matches) != 1 or len(match) != len(set(match))
                    or not program.minimum_matches <= len(match) <= program.maximum_matches
                    or set(match) != output_keys):
                raise ValueError("output lacks one complete bounded path selection")
            path_query = program.path_query
            endpoints = dict(execution.variables)
            expected_start = endpoints[path_query.start_variable]
            expected_end = endpoints[path_query.end_variable]
            for path in contract.paths:
                templates = {program.atoms[coordinate - 1].schema.relation: (left, right)
                    for coordinate, left, right in path.edges}
                previous, members, nodes = None, [], set()
                for key in match:
                    edge = templates.get(by_key[key].definition.predicate)
                    if edge is None:
                        raise ValueError("output path contains an unlicensed relation edge")
                    left, right = edge
                    fillers = {binding.role: binding.filler for binding in by_key[key].definition.bindings}
                    start, end = fillers[left], fillers[right]
                    if key not in output_keys or (previous is not None and previous != start):
                        raise ValueError("output lacks the actual directed premise path")
                    if previous is None and start != expected_start:
                        raise ValueError("output path starts outside the current request")
                    if start == end or end in nodes:
                        raise ValueError("output path is cyclic; identities cannot supply direction")
                    nodes.update((start, end))
                    previous = end
                    members.append(pack_record(PATH_EDGE, key, left.stable_key(), right.stable_key(),
                                               start.stable_key(), end.stable_key()))
                if previous != expected_end:
                    raise ValueError("output path ends outside the current request")
                paths.add(pack_record(SEQUENCE, path.stable_key(), *members))
            if contract.scope_variables:
                scope.update(match)
            continue
        environment = {}
        for atom, key in zip(execution.witness.program.atoms, match, strict=True):
            fillers = {binding.role: binding.filler for binding in by_key[key].definition.bindings}
            for role, variable in atom.role_variables:
                environment[variable] = fillers[role]
        for path in contract.paths:
            previous, members, nodes = None, [], []
            for coordinate, left, right in path.edges:
                key = match[coordinate - 1]
                fillers = {binding.role: binding.filler for binding in by_key[key].definition.bindings}
                start, end = fillers[left], fillers[right]
                if key not in output_keys or (previous is not None and previous != start):
                    raise ValueError("output lacks the actual directed premise path")
                if start == end or end in nodes:
                    raise ValueError("output path is cyclic; identities cannot supply direction")
                if not nodes:
                    nodes.append(start)
                nodes.append(end)
                previous = end
                members.append(pack_record(PATH_EDGE, key, left.stable_key(), right.stable_key(),
                                           start.stable_key(), end.stable_key()))
            if path.rule_key:
                from pure_integer_ai.cognition.shared.relation_composition import RelationCompositionWitness
                rule = RelationCompositionWitness.from_stable_key(path.rule_key).rule
                key = match[path.result_atom - 1]
                fillers = {binding.role: binding.filler for binding in by_key[key].definition.bindings}
                if (key not in output_keys or fillers[rule.result_input_role] != nodes[0]
                        or fillers[rule.result_output_role] != nodes[-1]):
                    raise ValueError("output explanation has no actual supported result for its path")
                members.append(pack_record(PATH_EDGE, path.rule_key, key,
                    fillers[rule.result_input_role].stable_key(), fillers[rule.result_output_role].stable_key()))
            paths.add(pack_record(SEQUENCE, path.stable_key(), *members))
        for index in contract.comparison_guards:
            kind, left, right = execution.witness.program.guards[index - 1]
            comparisons.add(pack_record(PATH_EDGE, (index, kind),
                environment[left].stable_key(), environment[right].stable_key()))
        if contract.scope_variables:
            scope.update(match)
    if contract.scope_variables and scope != output_keys:
        raise ValueError("output scope selection omitted or added an evidence-bearing claim")
    return tuple(sorted(paths)), tuple(sorted(comparisons)), tuple(sorted(scope))


def verify_operation_output(proof, facts, programs, authorized, *, budget=OperationProgramBudget()):
    """Independently verify both selections; never invoke an executor or generator."""
    if not isinstance(proof, OperationOutputProof) or type(authorized) is not tuple or proof.witness not in authorized:
        raise ValueError("output verification lacks exact current obligation permission")
    if len(proof.stable_key()) > budget.max_proof_values:
        raise ValueError("output proof integer budget exceeded")
    current = validate_output_permission(proof.witness, programs)
    execution = OperationExecution.from_stable_key(proof.execution_key)
    verify_integer_operation(execution, facts, programs, budget=budget)
    if execution.witness != current or execution.status != READY:
        raise ValueError("output obligation requires the exact ready current execution")
    if proof.current_request_key:
        from pure_integer_ai.cognition.shared.identity import ObjectIdentity
        from pure_integer_ai.cognition.understanding.query_operation_requests import InputOperationRequestBinding, OperationRequestFrame
        request = InputOperationRequestBinding.from_stable_key(proof.current_request_key)
        frame = OperationRequestFrame.from_stable_key(request.frame_key)
        variables = tuple(sorted((variable, ObjectIdentity.from_stable_key(target))
                                  for variable, target, _span, _projections in request.readings))
        if (request.source_ref != execution.query_key or variables != execution.variables
                or request.operation_key != current.program.operation_key
                or frame.source_identity != current.source_identity or frame.license_identity != current.license_identity):
            raise ValueError("output request O/H/E differs from the complete current input")
    expected = _obligation_members(proof.witness.contract, execution, facts)
    if (proof.path_instances, proof.comparison_instances, proof.scope_claims) != expected:
        raise ValueError("output proof lacks exact complete path/comparison/scope coverage")
    reference = proof.witness.contract.preserved_witness_key
    if bool(reference) != bool(proof.preserved_execution_key):
        raise ValueError("output preservation proof is missing or unsolicited")
    if reference:
        prior = OperationExecution.from_stable_key(proof.preserved_execution_key)
        verify_integer_operation(prior, facts, programs, budget=budget)
        if (prior.witness.stable_key() != reference or prior.status != READY
                or prior.query_key != execution.query_key or prior.variables != execution.variables
                or prior.context_identity != execution.context_identity or prior.matches != execution.matches
                or prior.fact_proofs != execution.fact_proofs
                or {key: frozenset(roles) for key, roles in prior.outputs}
                != {key: frozenset(roles) for key, roles in execution.outputs}):
            raise ValueError("output preservation changed current claims/roles/sources/Evidence")
        if prior.outputs == execution.outputs:
            raise ValueError("output preservation changed only a literal or connector label")
    return True


def prove_operation_output(witness, execution, facts, programs, *, budget=OperationProgramBudget(),
                           current_request_key=()):
    current = validate_output_permission(witness, programs)
    verify_integer_operation(execution, facts, programs, budget=budget)
    if execution.witness != current or execution.status != READY:
        raise ValueError("output obligation has no ready current execution")
    reference = witness.contract.preserved_witness_key
    preserved = (() if not reference else execute_integer_operation(
        OperationProgramWitness.from_stable_key(reference), execution.query_key, execution.variables,
        facts, budget=budget).stable_key())
    proof = OperationOutputProof(witness, execution.stable_key(),
        *_obligation_members(witness.contract, execution, facts), preserved, current_request_key)
    verify_operation_output(proof, facts, programs, (witness,), budget=budget)
    return proof
