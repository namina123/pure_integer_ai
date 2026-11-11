"""Bounded joins and evidence-preserving projections of current integer graphs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.evidence_candidate import EvidenceCandidateDefinition
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED, LIFECYCLE_ACTIVE, EvidenceRecord, HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import _snapshot_key
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_context_join import (
    bind_operation_atom, join_operation_context,
)
from pure_integer_ai.cognition.shared.integer_operation_program import (
    EQUAL, ORDER_DECLARED, OperationProgramBudget, OperationProgramWitness,
    SEQUENCE, integer_key,
)
from pure_integer_ai.cognition.shared.relation_composition import _entry_fields
from pure_integer_ai.cognition.shared.semantic_object import AtomicPropositionDefinition
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


FACT = 92210
EXECUTION = 92211
REQUEST_VARIABLE = 92212
OUTPUT = 92213
READY = 1
UNKNOWN = 2
CLARIFY = 3


@dataclass(frozen=True, slots=True)
class OperationGraphFact:
    definition: AtomicPropositionDefinition
    snapshot: HypothesisSnapshot
    candidate: EvidenceCandidateDefinition
    evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if (not isinstance(self.definition, AtomicPropositionDefinition)
                or not isinstance(self.snapshot, HypothesisSnapshot)
                or not isinstance(self.candidate, EvidenceCandidateDefinition)):
            raise TypeError("operation context requires complete H-05 declarations")
        hypothesis = self.snapshot.hypothesis
        if (self.candidate.candidate != self.definition.proposition
                or self.candidate.stable_key() != hypothesis.candidate_key
                or self.candidate.competition_key != hypothesis.competition_key
                or not self.candidate.forming_sources):
            raise ValueError("operation fact and H-05 declaration differ")
        if (type(self.evidence_keys) is not tuple
                or self.evidence_keys != tuple(sorted(set(self.evidence_keys)))):
            raise ValueError("operation fact needs canonical complete current Evidence")
        records, references, supersedes = [], set(), set()
        for key in self.evidence_keys:
            source, claim, reference, scope, payload, previous = _entry_fields(key)
            record = EvidenceRecord.from_stable_key(payload)
            expected_ref = (1, len(hypothesis.stable_key()), *hypothesis.stable_key(), record.evidence_id)
            expected_previous = ((1, len(hypothesis.stable_key()), *hypothesis.stable_key(),
                                  record.supersedes_evidence_id) if record.supersedes_evidence_id else ())
            if (record.hypothesis != hypothesis or claim != self.definition.proposition.stable_key()
                    or key[1] != record.stance or len(source) < 2 or source[0] <= 0
                    or source[1:] != record.source.stable_key() or reference != expected_ref
                    or scope != hypothesis.scope.stable_key() or previous != expected_previous):
                raise ValueError("operation fact has a forged full Evidence projection")
            records.append(record)
            references.add(reference)
            if previous:
                supersedes.add(previous)
        if references.intersection(supersedes):
            raise ValueError("operation context contains superseded Evidence")
        if len({record.evidence_id for record in records}) != len(records):
            raise ValueError("operation Evidence identifiers are duplicated")
        for stance, expected in ((1, self.snapshot.support_evidence_ids),
                                 (2, self.snapshot.refute_evidence_ids),
                                 (3, self.snapshot.unknown_evidence_ids)):
            if tuple(sorted(record.evidence_id for record in records if record.stance == stance)) != tuple(sorted(expected)):
                raise ValueError("operation context lacks the exact current H-00 Evidence set")

    @property
    def proposition_key(self):
        return self.definition.proposition.stable_key()

    def stable_key(self):
        definition = self.definition
        return pack_record(FACT, definition.proposition.stable_key(), definition.predicate.stable_key(),
            definition.source_anchor.stable_key(), definition.context.stable_key(),
            pack_record(SEQUENCE, *(pack_record(FACT, binding.role.stable_key(),
                binding.filler.stable_key(), (binding.ordinal,)) for binding in definition.canonical_bindings())),
            self.candidate.stable_key(), _snapshot_key(self.snapshot),
            pack_record(SEQUENCE, *self.evidence_keys))


def operation_context_identity(facts, budget):
    if (type(facts) is not tuple or any(not isinstance(fact, OperationGraphFact) for fact in facts)
            or len(facts) > budget.max_facts
            or len({fact.proposition_key for fact in facts}) != len(facts)):
        raise ValueError("operation context fact identity or budget is invalid")
    digest, count = hashlib.sha256(), 0
    for fact in sorted(facts, key=lambda item: item.proposition_key):
        key = fact.stable_key()
        count += len(key)
        if (len(key) > budget.max_key_values or count > budget.max_context_values
                or any(len(entry) > budget.max_key_values for entry in fact.evidence_keys)):
            raise ValueError("operation context integer budget exceeded")
        digest.update(len(key).to_bytes(8, "big"))
        for value in integer_key(key):
            digest.update(value.to_bytes(8, "big"))
    return tuple(digest.digest())


def eligible_operation_facts(witness, facts):
    """Refutations of identical typed content block projection across sources."""
    contradicted = {(fact.definition.predicate, fact.definition.canonical_bindings())
                    for fact in facts if fact.snapshot.refute_evidence_ids}
    return tuple(fact for fact in facts
        if fact.snapshot.lifecycle == LIFECYCLE_ACTIVE
        and fact.snapshot.epistemic_status == EPISTEMIC_SUPPORTED
        and not fact.snapshot.refute_evidence_ids
        and set(fact.candidate.forming_sources) <= set(witness.authorized_forming_sources)
        and (fact.definition.predicate, fact.definition.canonical_bindings()) not in contradicted)


def operation_conflict_facts(facts):
    """Return every current declaration proving a typed support/refute tie."""
    supporting, refuting = set(), set()
    for fact in facts:
        content = fact.definition.predicate, fact.definition.canonical_bindings()
        if fact.snapshot.support_evidence_ids:
            supporting.add(content)
        if fact.snapshot.refute_evidence_ids:
            refuting.add(content)
    conflicted = supporting & refuting
    return tuple(fact for fact in facts
        if (fact.definition.predicate, fact.definition.canonical_bindings()) in conflicted)


def operation_context_conflicted(facts):
    """Return whether current typed content has both support and refutation."""
    return bool(operation_conflict_facts(facts))


def request_environment(program, variables):
    if (type(variables) is not tuple
            or any(type(row) is not tuple or len(row) != 2 for row in variables)
            or tuple(variable for variable, _ in variables) != program.external_variables
            or any(not isinstance(filler, ObjectIdentity) for _, filler in variables)):
        raise ValueError("operation request must bind exactly the declared full variables")
    return dict(variables)


def requested_operation_context(witness, variables, facts, budget=OperationProgramBudget()):
    """按完整当前连接环境收窄前提；保留相关反驳、全部候选路径及完整证据。"""
    if (not isinstance(witness, OperationProgramWitness) or type(facts) is not tuple
            or any(not isinstance(fact, OperationGraphFact) for fact in facts)
            or len(facts) > budget.max_facts
            or len({fact.proposition_key for fact in facts}) != len(facts)):
        raise ValueError("operation context fact identity or budget is invalid")
    program = witness.program
    external = request_environment(program, variables)
    if program.path_query is None:
        index = {}
        for fact in facts:
            for binding in fact.definition.bindings:
                key = fact.definition.predicate, binding.role, binding.filler
                index.setdefault(key, []).append(fact)
        selected = join_operation_context(program, external,
            lambda predicate, role, filler: index.get((predicate, role, filler), ()), budget)
    else:
        path = program.path_query
        templates = {program.atoms[index - 1].schema.relation:
            (program.atoms[index - 1].schema, left, right)
            for index, left, right in path.edge_templates}
        adjacency = {}
        for fact in facts:
            template = templates.get(fact.definition.predicate)
            if template is None:
                continue
            schema, left, right = template
            schema.validate_definition(fact.definition)
            fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
            adjacency.setdefault(fillers[left], []).append((fact, fillers[right]))
        start, target = external[path.start_variable], external[path.end_variable]
        pending, keys, attempts = [(start, (), frozenset((start,)))], set(), 0
        while pending:
            current, selected_path, visited = pending.pop()
            if len(selected_path) >= program.maximum_matches:
                continue
            for fact, following in adjacency.get(current, ()):
                attempts += 1
                if attempts > budget.max_match_attempts:
                    raise ValueError("operation context path budget exceeded; no partial result")
                if following in visited:
                    continue
                next_path = (*selected_path, fact.proposition_key)
                if following == target:
                    if len(next_path) >= program.minimum_matches:
                        keys.update(next_path)
                    continue
                if len(next_path) < program.maximum_matches:
                    pending.append((following, next_path, visited | {following}))
        selected = tuple(fact for fact in facts if fact.proposition_key in keys)
    selected = tuple(sorted(selected, key=lambda fact: fact.proposition_key))
    operation_context_identity(selected, budget)
    return selected


def operation_guards_hold(program, environment):
    return all((environment[left] == environment[right]) == (kind == EQUAL)
               for kind, left, right in program.guards)


def projected_operation_outputs(program, matches, by_key):
    """A nonunique directed order is CLARIFY; IDs never break ordering ties."""
    if program.path_query is not None:
        if len(matches) != 1:
            return (), CLARIFY
        match, = matches
        if len(set(match)) != len(match):
            return (), CLARIFY
        atoms = {atom.schema.relation: index for index, atom in enumerate(program.atoms, 1)}
        projections = {projection.atom: projection for projection in program.projections}
        outputs = []
        for key in match:
            atom = atoms.get(by_key[key].definition.predicate)
            if atom is None:
                return (), CLARIFY
            roles = tuple(role.stable_key() for role in projections[atom].roles)
            outputs.append((key, roles))
        return tuple(outputs), READY
    outputs = []
    for match in matches:
        for projection in program.projections:
            row = (match[projection.atom - 1], tuple(role.stable_key() for role in projection.roles))
            if any(key == row[0] and roles != row[1] for key, roles in outputs):
                raise ValueError("operation has competing per-claim role orders")
            if row not in outputs:
                outputs.append(row)
    if program.order_kind == ORDER_DECLARED:
        return tuple(outputs), READY
    # The data explicitly authorizes edge-to-edge chaining by these two roles.
    left, right = program.order_roles
    fillers = {key: {binding.role: binding.filler for binding in by_key[key].definition.bindings}
               for key, _ in outputs}
    if any(filler[left] == filler[right] for filler in fillers.values()):
        return (), CLARIFY
    edges = {(first, second) for first in fillers for second in fillers
             if first != second and fillers[first][right] == fillers[second][left]}
    pending, ordered = {key for key, _ in outputs}, []
    while pending:
        starts = tuple(key for key in pending if not any(end == key and start in pending for start, end in edges))
        if len(starts) != 1:
            return (), CLARIFY
        key = starts[0]
        ordered.append(next(row for row in outputs if row[0] == key))
        pending.remove(key)
    return tuple(ordered), READY


@dataclass(frozen=True, slots=True)
class OperationExecution:
    witness: OperationProgramWitness
    query_key: tuple[int, ...]
    variables: tuple[tuple[int, ObjectIdentity], ...]
    context_identity: tuple[int, ...]
    matches: tuple[tuple[tuple[int, ...], ...], ...]
    # Each used context fact is retained exactly once, including full Evidence.
    fact_proofs: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]
    status: int

    def __post_init__(self):
        if not isinstance(self.witness, OperationProgramWitness):
            raise TypeError("operation execution requires a complete witness")
        integer_key(self.query_key)
        request_environment(self.witness.program, self.variables)
        if (type(self.context_identity) is not tuple or len(self.context_identity) != 32
                or any(type(item) is not int or not 0 <= item <= 255 for item in self.context_identity)):
            raise ValueError("operation execution context digest is incomplete")
        program = self.witness.program
        if (type(self.matches) is not tuple or self.matches != tuple(sorted(set(self.matches)))
                or any(type(row) is not tuple or not row for row in self.matches)
                or (program.path_query is None and any(len(row) != len(program.atoms)
                    for row in self.matches))
                or (program.path_query is not None and any(not program.minimum_matches <= len(row)
                    <= program.maximum_matches for row in self.matches))):
            raise ValueError("operation execution matches are not canonical")
        for row in self.matches:
            for key in row:
                integer_key(key)
        if type(self.fact_proofs) is not tuple or self.fact_proofs != tuple(sorted(set(self.fact_proofs))):
            raise ValueError("operation execution full fact proofs are not canonical")
        for key in self.fact_proofs:
            integer_key(key)
        if type(self.outputs) is not tuple or len({row[0] for row in self.outputs}) != len(self.outputs):
            raise ValueError("operation execution output coordinates are not unique")
        for key, roles in self.outputs:
            integer_key(key)
            if type(roles) is not tuple or not roles:
                raise ValueError("operation execution output role order is missing")
            for role in roles:
                integer_key(role)
        if type(self.status) is not int or self.status not in {READY, UNKNOWN, CLARIFY}:
            raise ValueError("operation execution outcome is invalid")
        if self.status != READY and self.outputs:
            raise ValueError("unqualified operation cannot project factual outputs")

    def stable_key(self):
        return pack_record(EXECUTION, self.witness.stable_key(), self.query_key,
            pack_record(SEQUENCE, *(pack_record(REQUEST_VARIABLE, (variable,), filler.stable_key())
                for variable, filler in self.variables)), self.context_identity,
            pack_record(SEQUENCE, *(pack_record(SEQUENCE, *match) for match in self.matches)),
            pack_record(SEQUENCE, *self.fact_proofs),
            pack_record(SEQUENCE, *(pack_record(OUTPUT, key, pack_record(SEQUENCE, *roles))
                                   for key, roles in self.outputs)), (self.status,))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(key, EXECUTION)
        if len(fields) != 8 or len(fields[7]) != 1:
            raise ValueError("operation execution is incomplete")
        variables, outputs = [], []
        for row in unpack_record(fields[2], SEQUENCE):
            pair = unpack_record(row, REQUEST_VARIABLE)
            if len(pair) != 2 or len(pair[0]) != 1:
                raise ValueError("operation request variable is incomplete")
            variables.append((pair[0][0], ObjectIdentity.from_stable_key(pair[1])))
        for row in unpack_record(fields[6], SEQUENCE):
            pair = unpack_record(row, OUTPUT)
            if len(pair) != 2:
                raise ValueError("operation output coordinate is incomplete")
            outputs.append((pair[0], unpack_record(pair[1], SEQUENCE)))
        result = cls(OperationProgramWitness.from_stable_key(fields[0]), fields[1], tuple(variables),
            fields[3], tuple(unpack_record(row, SEQUENCE) for row in unpack_record(fields[4], SEQUENCE)),
            unpack_record(fields[5], SEQUENCE), tuple(outputs), fields[7][0])
        if result.stable_key() != key:
            raise ValueError("operation execution is not canonical")
        return result


def _operation_path_matches(witness, variables, facts, budget):
    """枚举当前端点之间受许可且有界的简单路径。"""
    program = witness.program
    path = program.path_query
    endpoints = request_environment(program, variables)
    start, target = endpoints[path.start_variable], endpoints[path.end_variable]
    templates = {program.atoms[index - 1].schema.relation:
        (program.atoms[index - 1].schema, left, right)
        for index, left, right in path.edge_templates}
    adjacency = {}
    for fact in eligible_operation_facts(witness, facts):
        template = templates.get(fact.definition.predicate)
        if template is None:
            continue
        schema, left, right = template
        schema.validate_definition(fact.definition)
        fillers = {binding.role: binding.filler for binding in fact.definition.bindings}
        adjacency.setdefault(fillers[left], []).append((fact.proposition_key, fillers[right]))
    for edges in adjacency.values():
        edges.sort(key=lambda row: (row[0], row[1].stable_key()))

    paths, attempts = set(), 0
    pending = [(start, (), frozenset((start,)))]
    while pending:
        current, selected, visited = pending.pop()
        if len(selected) >= program.maximum_matches:
            continue
        for fact_key, following in adjacency.get(current, ()):
            attempts += 1
            if attempts > budget.max_match_attempts:
                raise ValueError("operation path traversal budget exceeded; no partial result")
            if following in visited:
                continue
            next_path = (*selected, fact_key)
            if (following == target and len(next_path) >= program.minimum_matches):
                paths.add(next_path)
                if len(paths) > budget.max_matches:
                    raise ValueError("operation path match budget exceeded; no partial result")
            if len(next_path) < program.maximum_matches and following != target:
                pending.append((following, next_path, visited | {following}))
    return tuple(sorted(paths))


def execute_integer_operation(witness, query_key, variables, facts, *, budget=OperationProgramBudget()):
    """Index by the declared relation; bound each join and retain all complete matches."""
    if not isinstance(witness, OperationProgramWitness) or not isinstance(budget, OperationProgramBudget):
        raise TypeError("operation execution requires typed permission and budgets")
    program = witness.program
    if len(program.atoms) > budget.max_atoms or len(witness.stable_key()) > budget.max_key_values:
        raise ValueError("operation program integer budget exceeded")
    facts = requested_operation_context(witness, variables, facts, budget)
    context = operation_context_identity(facts, budget)
    conflicted = operation_context_conflicted(facts)
    initial = request_environment(program, variables)
    by_relation = {}
    for fact in eligible_operation_facts(witness, facts):
        by_relation.setdefault(fact.definition.predicate, []).append(fact)
    if program.path_query is not None:
        matches = _operation_path_matches(witness, variables, facts, budget)
    else:
        rows, attempts = [((), initial)], 0
        for atom in program.atoms:
            updated = []
            for selected, environment in rows:
                for fact in by_relation.get(atom.schema.relation, ()):
                    attempts += 1
                    if attempts > budget.max_match_attempts:
                        raise ValueError("operation join budget exceeded; no partial result")
                    bound = bind_operation_atom(atom, fact, environment)
                    if bound is not None:
                        updated.append(((*selected, fact.proposition_key), bound))
                        if len(updated) > budget.max_matches:
                            raise ValueError("operation intermediate match budget exceeded; no partial result")
            rows = updated
        matches = tuple(sorted({selected for selected, environment in rows
                                if operation_guards_hold(program, environment)}))
    used = {key for match in matches for key in match}
    if conflicted:
        used.update(fact.proposition_key for fact in operation_conflict_facts(facts))
    by_key = {fact.proposition_key: fact for fact in facts}
    proofs = tuple(sorted(by_key[key].stable_key() for key in used))
    if conflicted:
        outputs, status = (), CLARIFY
    elif program.path_query is not None:
        if not matches:
            outputs, status = (), UNKNOWN
        elif len(matches) > 1:
            outputs, status = (), CLARIFY
        else:
            outputs, status = projected_operation_outputs(program, matches, by_key)
    elif len(matches) < program.minimum_matches:
        outputs, status = (), UNKNOWN
    elif len(matches) > program.maximum_matches:
        outputs, status = (), CLARIFY
    else:
        outputs, status = projected_operation_outputs(program, matches, by_key)
    result = OperationExecution(witness, query_key, variables, context, matches, proofs, outputs, status)
    if len(result.stable_key()) > budget.max_proof_values:
        raise ValueError("operation execution proof budget exceeded; no partial result")
    return result
