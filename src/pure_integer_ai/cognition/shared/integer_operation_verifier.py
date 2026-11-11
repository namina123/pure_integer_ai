"""Independent complete-match verifier; it never invokes the program executor."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import EPISTEMIC_SUPPORTED, LIFECYCLE_ACTIVE
from pure_integer_ai.cognition.shared.integer_operation_execution import (
    READY, UNKNOWN, CLARIFY, OperationExecution, operation_context_identity, request_environment,
    operation_conflict_facts, requested_operation_context,
)
from pure_integer_ai.cognition.shared.integer_operation_program import (
    EQUAL, ORDER_DECLARED, OperationProgramBudget, OperationProgramWitness,
)


def independent_operation_matches(witness, variables, facts, *, budget=OperationProgramBudget()):
    """Cold-read complete matches; do not produce an execution/output proof."""
    if not isinstance(witness, OperationProgramWitness) or not isinstance(budget, OperationProgramBudget):
        raise TypeError("independent matches require a licensed witness and budget")
    program = witness.program
    if len(program.atoms) > budget.max_atoms or len(witness.stable_key()) > budget.max_key_values:
        raise ValueError("operation verifier integer budget exceeded")
    facts = requested_operation_context(witness, variables, facts, budget)
    external = request_environment(program, variables)
    refuted = {(fact.definition.predicate, fact.definition.canonical_bindings())
               for fact in facts if fact.snapshot.refute_evidence_ids}
    selected = tuple(fact for fact in facts
        if fact.snapshot.lifecycle == LIFECYCLE_ACTIVE
        and fact.snapshot.epistemic_status == EPISTEMIC_SUPPORTED
        and not fact.snapshot.refute_evidence_ids
        and set(fact.candidate.forming_sources) <= set(witness.authorized_forming_sources)
        and (fact.definition.predicate, fact.definition.canonical_bindings()) not in refuted)
    if program.path_query is not None:
        path = program.path_query
        templates = {program.atoms[index - 1].schema.relation:
            (program.atoms[index - 1].schema, left, right)
            for index, left, right in path.edge_templates}
        adjacency = {}
        for fact in selected:
            template = templates.get(fact.definition.predicate)
            if template is None:
                continue
            schema, left, right = template
            schema.validate_definition(fact.definition)
            fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
            adjacency.setdefault(fillers[left], []).append((fact.proposition_key, fillers[right]))
        for edges in adjacency.values():
            edges.sort(key=lambda row: (row[0], row[1].stable_key()))
        start, target = external[path.start_variable], external[path.end_variable]
        paths, attempts = set(), 0
        pending = [(start, (), frozenset((start,)))]
        while pending:
            current, selected_path, visited = pending.pop()
            if len(selected_path) >= program.maximum_matches:
                continue
            for fact_key, following in adjacency.get(current, ()):
                attempts += 1
                if attempts > budget.max_match_attempts:
                    raise ValueError("operation independent path budget exceeded")
                if following in visited:
                    continue
                next_path = (*selected_path, fact_key)
                if following == target and len(next_path) >= program.minimum_matches:
                    paths.add(next_path)
                    if len(paths) > budget.max_matches:
                        raise ValueError("operation independent path-count budget exceeded")
                if len(next_path) < program.maximum_matches and following != target:
                    pending.append((following, next_path, visited | {following}))
        return tuple(sorted(paths))
    groups = []
    for atom in program.atoms:
        group = []
        for fact in selected:
            if fact.definition.predicate != atom.schema.relation:
                continue
            atom.schema.validate_definition(fact.definition)
            fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
            if all(variable not in external or external[variable] == fillers[role]
                   for role, variable in atom.role_variables):
                group.append(fact)
        groups.append(tuple(group))
    groups = tuple(groups)
    rows, attempts = [((), external)], 0
    for atom, group in zip(program.atoms, groups, strict=True):
        updated = []
        for selected_keys, bindings in rows:
            for fact in group:
                attempts += 1
                if attempts > budget.max_match_attempts:
                    raise ValueError("operation independent complete-match budget exceeded; no partial result")
                atom.schema.validate_definition(fact.definition)
                fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
                next_bindings = dict(bindings)
                consistent = True
                for role, variable in atom.role_variables:
                    filler = fillers[role]
                    if variable in next_bindings and next_bindings[variable] != filler:
                        consistent = False
                        break
                    next_bindings[variable] = filler
                if consistent:
                    updated.append(((*selected_keys, fact.proposition_key), next_bindings))
                    if len(updated) > budget.max_matches:
                        raise ValueError("operation independent intermediate match-count budget exceeded; no partial result")
        rows = updated
    matches = set()
    for selected_keys, bindings in rows:
        if any((bindings[left] == bindings[right]) != (kind == EQUAL)
               for kind, left, right in program.guards):
            continue
        matches.add(selected_keys)
        if len(matches) > budget.max_matches:
            raise ValueError("operation independent match-count budget exceeded; no partial result")
    expected_matches = tuple(sorted(matches))
    return expected_matches


def verify_integer_operation(execution, facts, authorized, *, budget=OperationProgramBudget()):
    """Check complete coverage independently of the producer and directed order."""
    if not isinstance(execution, OperationExecution) or not isinstance(budget, OperationProgramBudget):
        raise TypeError("operation verification requires an integer execution and budget")
    witness, program = execution.witness, execution.witness.program
    if type(authorized) is not tuple or witness not in authorized:
        raise ValueError("operation verification lacks exact current graph permission")
    if len(execution.stable_key()) > budget.max_proof_values:
        raise ValueError("operation verifier integer budget exceeded")
    source_facts = facts
    facts = requested_operation_context(witness, execution.variables, source_facts, budget)
    context_matches = operation_context_identity(facts, budget) == execution.context_identity
    if not context_matches:
        try:
            context_matches = operation_context_identity(source_facts, budget) == execution.context_identity
        except ValueError:
            context_matches = False
    if not context_matches:
        raise ValueError("operation verification semantic context drifted")
    expected_matches = independent_operation_matches(witness, execution.variables, facts, budget=budget)
    if execution.matches != expected_matches:
        raise ValueError("operation verification lacks exact complete graph matches")
    by_key = {fact.proposition_key: fact for fact in facts}
    used = {key for match in expected_matches for key in match}
    supporting, refuting = set(), set()
    for fact in facts:
        content = fact.definition.predicate, fact.definition.canonical_bindings()
        if fact.snapshot.support_evidence_ids:
            supporting.add(content)
        if fact.snapshot.refute_evidence_ids:
            refuting.add(content)
    conflicted = bool(supporting & refuting)
    if conflicted:
        used.update(fact.proposition_key for fact in operation_conflict_facts(facts))
    if execution.fact_proofs != tuple(sorted(by_key[key].stable_key() for key in used)):
        raise ValueError("operation verification lacks exact full fact/Evidence proofs")
    expected = []
    if program.path_query is not None:
        if not expected_matches:
            status = UNKNOWN
        elif len(expected_matches) > 1 or len(set(expected_matches[0])) != len(expected_matches[0]):
            status = CLARIFY
        else:
            atoms = {atom.schema.relation: index for index, atom in enumerate(program.atoms, 1)}
            projections = {projection.atom: projection for projection in program.projections}
            for key in expected_matches[0]:
                atom = atoms[by_key[key].definition.predicate]
                roles = tuple(role.stable_key() for role in projections[atom].roles)
                expected.append((key, roles))
            status = READY
    else:
        for match in expected_matches:
            for projection in program.projections:
                row = (match[projection.atom - 1], tuple(role.stable_key() for role in projection.roles))
                if any(key == row[0] and roles != row[1] for key, roles in expected):
                    raise ValueError("operation verification found competing per-claim role orders")
                if row not in expected:
                    expected.append(row)
        if len(expected_matches) < program.minimum_matches:
            expected, status = [], UNKNOWN
        elif len(expected_matches) > program.maximum_matches:
            expected, status = [], CLARIFY
        elif program.order_kind == ORDER_DECLARED:
            status = READY
        else:
            left, right = program.order_roles
            keys = tuple(key for key, _ in expected)
            bindings = tuple({binding.role: binding.filler for binding in by_key[key].definition.bindings}
                             for key in keys)
            reach = {(i, j) for i in range(len(keys)) for j in range(len(keys))
                     if bindings[i][right] == bindings[j][left]}
            for middle in range(len(keys)):
                for i in range(len(keys)):
                    for j in range(len(keys)):
                        if (i, middle) in reach and (middle, j) in reach:
                            reach.add((i, j))
            total = (not any((i, i) in reach for i in range(len(keys))) and all(
                ((i, j) in reach) != ((j, i) in reach)
                for i in range(len(keys)) for j in range(i + 1, len(keys))))
            if total:
                ranks = [sum((j, i) in reach for j in range(len(keys))) for i in range(len(keys))]
                expected = [expected[i] for i in sorted(range(len(keys)), key=lambda i: ranks[i])]
                status = READY
            else:
                expected, status = [], CLARIFY
    if conflicted:
        expected, status = [], CLARIFY
    if execution.status != status or execution.outputs != tuple(expected):
        raise ValueError("operation verification output selection or directed order drifted")
    return True
