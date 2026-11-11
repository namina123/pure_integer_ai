"""Source-qualified output obligations over current integer graph selections.

These records describe coverage, directed paths, comparisons and preservation.
They contain no operation family labels, language tokens or expected answers.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.integer_operation_behavior import semantic_selection_obligation
from pure_integer_ai.cognition.shared.integer_operation_program import (
    OperationProgramWitness, SEQUENCE, integer_key,
)
from pure_integer_ai.cognition.shared.relation_composition import RelationCompositionWitness, schema_key
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


OUTPUT_CONTRACT = 92400
OUTPUT_PATH = 92401
OUTPUT_WITNESS = 92402
OUTPUT_PROOF = 92403
PATH_EDGE = 92404


def _canonical_coordinates(value, *, empty=False):
    integer_key(value, allow_empty=empty)
    if value != tuple(sorted(set(value))) or any(item <= 0 for item in value):
        raise ValueError("output coordinates must be canonical positive integers")


@dataclass(frozen=True, slots=True)
class OperationOutputPath:
    # Ordered atom coordinate and its actual directed source/target roles.
    edges: tuple[tuple[int, ObjectIdentity, ObjectIdentity], ...]
    rule_key: tuple[int, ...] = ()
    result_atom: int = 0

    def __post_init__(self):
        if (type(self.edges) is not tuple or not self.edges or len(self.edges) > 8
                or any(type(row) is not tuple or len(row) != 3 for row in self.edges)):
            raise ValueError("output path needs complete bounded directed edges")
        for atom, left, right in self.edges:
            if (type(atom) is not int or atom <= 0 or not isinstance(left, ObjectIdentity)
                    or not isinstance(right, ObjectIdentity) or left == right):
                raise ValueError("output path has invalid directed coordinates")
        if len({row[0] for row in self.edges}) != len(self.edges):
            raise ValueError("output path repeats a premise coordinate")
        integer_key(self.rule_key, allow_empty=True)
        if type(self.result_atom) is not int or self.result_atom < 0 or bool(self.rule_key) != bool(self.result_atom):
            raise ValueError("output explanation needs both its licensed rule and result coordinate")
        if self.rule_key:
            RelationCompositionWitness.from_stable_key(self.rule_key)

    def stable_key(self):
        return pack_record(OUTPUT_PATH, pack_record(SEQUENCE, *(pack_record(PATH_EDGE,
            (atom,), left.stable_key(), right.stable_key()) for atom, left, right in self.edges)),
            self.rule_key, (self.result_atom,))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(integer_key(key), OUTPUT_PATH)
        if len(fields) != 3 or len(fields[2]) != 1:
            raise ValueError("output path is incomplete")
        rows = []
        for edge in unpack_record(fields[0], SEQUENCE):
            values = unpack_record(edge, PATH_EDGE)
            if len(values) != 3 or len(values[0]) != 1:
                raise ValueError("output path edge is incomplete")
            rows.append((values[0][0], *(ObjectIdentity.from_stable_key(value) for value in values[1:])))
        result = cls(tuple(rows), fields[1], fields[2][0])
        if result.stable_key() != key:
            raise ValueError("output path is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationOutputContract:
    program_witness_key: tuple[int, ...]
    projected_atoms: tuple[int, ...]
    paths: tuple[OperationOutputPath, ...] = ()
    comparison_guards: tuple[int, ...] = ()
    scope_variables: tuple[int, ...] = ()
    preserved_witness_key: tuple[int, ...] = ()
    current_request_required: int = 0

    def __post_init__(self):
        witness = OperationProgramWitness.from_stable_key(integer_key(self.program_witness_key))
        program = witness.program
        if type(self.current_request_required) is not int or self.current_request_required not in {0, 1}:
            raise ValueError("current request O/H/E requirement must be an integer flag")
        _canonical_coordinates(self.projected_atoms)
        _canonical_coordinates(self.comparison_guards, empty=True)
        _canonical_coordinates(self.scope_variables, empty=True)
        if self.projected_atoms != tuple(sorted(item.atom for item in program.projections)):
            raise ValueError("output contract must cover exactly the actual projected atoms")
        if (type(self.paths) is not tuple or len(self.paths) > 8
                or any(not isinstance(path, OperationOutputPath) for path in self.paths)
                or self.paths != tuple(sorted(set(self.paths), key=OperationOutputPath.stable_key))):
            raise ValueError("output paths must be complete and canonical")
        if program.path_query is not None:
            if len(self.paths) != 1:
                raise ValueError("path program output must declare one licensed edge template set")
            path, = self.paths
            if (path.rule_key or path.result_atom
                    or path.edges != program.path_query.edge_templates):
                raise ValueError("path output templates differ from the licensed path schemas and roles")
        for path in self.paths:
            prior_variable = None
            for coordinate, left, right in path.edges:
                if coordinate not in self.projected_atoms:
                    raise ValueError("output path omits a necessary premise from delivery")
                variables = dict(program.atoms[coordinate - 1].role_variables)
                if left not in variables or right not in variables:
                    raise ValueError("output path roles differ from the actual schema")
                if program.path_query is not None:
                    if (variables[left] != program.path_query.start_variable
                            or variables[right] != program.path_query.end_variable):
                        raise ValueError("output path edge differs from its licensed endpoint variables")
                elif prior_variable is not None and variables[left] != prior_variable:
                    raise ValueError("output path does not declare an actual directed variable join")
                prior_variable = variables[right]
            if path.rule_key:
                rule_witness = RelationCompositionWitness.from_stable_key(path.rule_key)
                rule = rule_witness.rule
                expected = ((rule.first_relation, rule.first_input_role, rule.first_join_role),
                            (rule.second_relation, rule.second_join_role, rule.second_output_role))
                actual = tuple((program.atoms[index - 1].schema.relation, left, right)
                               for index, left, right in path.edges)
                if (actual != expected or rule_witness.source_identity != witness.source_identity
                        or rule_witness.license_identity != witness.license_identity
                        or not set(witness.authorized_forming_sources) <= set(rule_witness.authorized_forming_sources)
                        or any(schema_key(program.atoms[index - 1].schema) not in {
                            schema_key(schema) for schema in rule_witness.schemas} for index, _, _ in path.edges)):
                    raise ValueError("output path rule/schema/source permission differs")
                if path.result_atom not in self.projected_atoms or path.result_atom in {row[0] for row in path.edges}:
                    raise ValueError("output explanation must deliver a separate supported result")
                result = program.atoms[path.result_atom - 1]
                roles = dict(result.role_variables)
                first_index, left, _ = path.edges[0]
                last_index, _, right = path.edges[-1]
                if (result.schema.relation != rule.result_relation
                        or schema_key(result.schema) not in {schema_key(schema) for schema in rule_witness.schemas}
                        or roles.get(rule.result_input_role) != dict(program.atoms[first_index - 1].role_variables)[left]
                        or roles.get(rule.result_output_role) != dict(program.atoms[last_index - 1].role_variables)[right]):
                    raise ValueError("output explanation result differs from its actual rule and endpoint variables")
        if any(index > len(program.guards) for index in self.comparison_guards):
            raise ValueError("output comparison does not select a declared guard")
        delivered_variables = {variable for index in self.projected_atoms
                               for _, variable in program.atoms[index - 1].role_variables}
        if any(not set(program.guards[index - 1][1:]) <= delivered_variables
               for index in self.comparison_guards):
            raise ValueError("output comparison must deliver both evidence-bearing operands")
        if self.scope_variables and (self.projected_atoms != tuple(range(1, len(program.atoms) + 1))
                or not set(self.scope_variables) <= set(program.external_variables)):
            raise ValueError("scoped output must preserve every premise and current request scope")
        integer_key(self.preserved_witness_key, allow_empty=True)
        if self.preserved_witness_key:
            prior = OperationProgramWitness.from_stable_key(self.preserved_witness_key)
            if (semantic_selection_obligation(prior.program) != semantic_selection_obligation(program)
                    or prior.source_identity != witness.source_identity
                    or prior.license_identity != witness.license_identity
                    or prior.authorized_forming_sources != witness.authorized_forming_sources
                    or prior.program.realization_key == program.realization_key):
                raise ValueError("preservation requires the same semantic obligation and another graph organization")

    def stable_key(self):
        return pack_record(OUTPUT_CONTRACT, self.program_witness_key, self.projected_atoms,
            pack_record(SEQUENCE, *(path.stable_key() for path in self.paths)),
            self.comparison_guards, self.scope_variables, self.preserved_witness_key,
            (self.current_request_required,))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(integer_key(key), OUTPUT_CONTRACT)
        if len(fields) != 7 or len(fields[6]) != 1:
            raise ValueError("output contract is incomplete")
        result = cls(fields[0], fields[1], tuple(OperationOutputPath.from_stable_key(path)
            for path in unpack_record(fields[2], SEQUENCE)), *fields[3:6], fields[6][0])
        if result.stable_key() != key:
            raise ValueError("output contract is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationOutputWitness:
    contract: OperationOutputContract
    source: SourceRef
    source_sha256: tuple[int, ...]

    def __post_init__(self):
        digest = self.source_sha256
        if (not isinstance(self.contract, OperationOutputContract) or not isinstance(self.source, SourceRef)
                or type(digest) is not tuple or len(digest) != 32
                or any(type(value) is not int or not 0 <= value <= 255 for value in digest)
                or self.source.source_kind != OUTPUT_WITNESS
                or self.source.source_id != ((int.from_bytes(bytes(digest[:8]), "big") & ((1 << 63) - 1)) or 1)):
            raise ValueError("output witness lacks its full physical source digest")

    def stable_key(self):
        return pack_record(OUTPUT_WITNESS, self.contract.stable_key(), self.source.stable_key(), self.source_sha256)

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(integer_key(key), OUTPUT_WITNESS)
        if len(fields) != 3:
            raise ValueError("output witness is incomplete")
        result = cls(OperationOutputContract.from_stable_key(fields[0]), SourceRef.from_stable_key(fields[1]), fields[2])
        if result.stable_key() != key:
            raise ValueError("output witness is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationOutputProof:
    witness: OperationOutputWitness
    execution_key: tuple[int, ...]
    path_instances: tuple[tuple[int, ...], ...]
    comparison_instances: tuple[tuple[int, ...], ...]
    scope_claims: tuple[tuple[int, ...], ...]
    preserved_execution_key: tuple[int, ...] = ()
    current_request_key: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.witness, OperationOutputWitness):
            raise TypeError("output proof requires a complete physical witness")
        integer_key(self.execution_key)
        integer_key(self.preserved_execution_key, allow_empty=True)
        integer_key(self.current_request_key, allow_empty=True)
        if bool(self.current_request_key) != bool(self.witness.contract.current_request_required):
            raise ValueError("output proof has missing or unsolicited current request O/H/E")
        for rows in (self.path_instances, self.comparison_instances, self.scope_claims):
            if type(rows) is not tuple or rows != tuple(sorted(set(rows))):
                raise ValueError("output proof members must be complete canonical integer records")
            for row in rows:
                integer_key(row)

    def stable_key(self):
        return pack_record(OUTPUT_PROOF, self.witness.stable_key(), self.execution_key,
            *(pack_record(SEQUENCE, *rows) for rows in (
                self.path_instances, self.comparison_instances, self.scope_claims)), self.preserved_execution_key,
            self.current_request_key)

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(integer_key(key), OUTPUT_PROOF)
        if len(fields) != 7:
            raise ValueError("output proof is incomplete")
        result = cls(OperationOutputWitness.from_stable_key(fields[0]), fields[1],
            *(unpack_record(value, SEQUENCE) for value in fields[2:5]), *fields[5:])
        if result.stable_key() != key:
            raise ValueError("output proof is not canonical")
        return result
