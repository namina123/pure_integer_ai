"""Source-qualified conjunctive graph programs, without language or relation rules.

Programs declare exact schemas, variable joins, equality guards, projections and
an optional directed ordering obligation. Their identifiers carry no behavior.
This protocol supplies no truth, source text, answer surface or language tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.relation_composition import schema_from_key, schema_key
from pure_integer_ai.cognition.shared.typed_relation import RelationSchema
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


PROGRAM = 92200
ATOM = 92201
SEQUENCE = 92202
ROLE_VARIABLE = 92203
GUARD = 92204
PROJECTION = 92205
WITNESS = 92206
PROGRAM_PATH = 92207
OPERATION_PATH_QUERY = 92208
OPERATION_PATH_EDGE = 92209
EQUAL = 1
UNEQUAL = 2
ORDER_DECLARED = 1
ORDER_DIRECTED_CHAIN = 2


def integer_key(value, *, allow_empty=False):
    if (type(value) is not tuple or (not value and not allow_empty)
            or any(type(item) is not int or item < 0 or item >= 1 << 63 for item in value)):
        raise ValueError("operation requires a canonical nonnegative integer key")
    return value


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError("operation coordinates require positive integers")
    return value


@dataclass(frozen=True, slots=True)
class OperationAtom:
    schema: RelationSchema
    # Every schema role is named once; repeated variables express actual joins.
    role_variables: tuple[tuple[ObjectIdentity, int], ...]

    def __post_init__(self):
        if not isinstance(self.schema, RelationSchema) or type(self.role_variables) is not tuple:
            raise TypeError("operation atom requires a typed schema and integer variables")
        if any(type(row) is not tuple or len(row) != 2 for row in self.role_variables):
            raise ValueError("operation role-variable mapping is malformed")
        roles = tuple(slot.role for slot in self.schema.slots)
        if (len(self.role_variables) != len(roles)
                or {role for role, _ in self.role_variables} != set(roles)):
            raise ValueError("operation atom must bind every schema role exactly once")
        if any(slot.min_count != 1 or slot.max_count != 1 for slot in self.schema.slots):
            raise ValueError("operation V1 requires explicitly single-valued roles")
        for _, variable in self.role_variables:
            positive(variable)
        if self.role_variables != tuple(sorted(self.role_variables, key=lambda row: row[0].stable_key())):
            raise ValueError("operation role variables are not canonical")

    def stable_key(self):
        return pack_record(ATOM, schema_key(self.schema), pack_record(SEQUENCE, *(
            pack_record(ROLE_VARIABLE, role.stable_key(), (variable,))
            for role, variable in self.role_variables)))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(key, ATOM)
        if len(fields) != 2:
            raise ValueError("operation atom is incomplete")
        rows = []
        for row in unpack_record(fields[1], SEQUENCE):
            pair = unpack_record(row, ROLE_VARIABLE)
            if len(pair) != 2 or len(pair[1]) != 1:
                raise ValueError("operation variable coordinate is incomplete")
            rows.append((ObjectIdentity.from_stable_key(pair[0]), pair[1][0]))
        result = cls(schema_from_key(fields[0]), tuple(rows))
        if result.stable_key() != key:
            raise ValueError("operation atom is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationProjection:
    atom: int
    roles: tuple[ObjectIdentity, ...]

    def __post_init__(self):
        positive(self.atom)
        if (type(self.roles) is not tuple or not self.roles
                or any(not isinstance(role, ObjectIdentity) for role in self.roles)
                or len(set(self.roles)) != len(self.roles)):
            raise ValueError("operation projection requires explicit unique role order")

    def stable_key(self):
        return pack_record(PROJECTION, (self.atom,),
                           pack_record(SEQUENCE, *(role.stable_key() for role in self.roles)))


@dataclass(frozen=True, slots=True)
class OperationPathQuery:
    """在有限许可关系模板集合上查询有界有向简单路径。"""

    start_variable: int
    end_variable: int
    edge_templates: tuple[tuple[int, ObjectIdentity, ObjectIdentity], ...]

    def __post_init__(self):
        positive(self.start_variable)
        positive(self.end_variable)
        if (self.start_variable == self.end_variable or type(self.edge_templates) is not tuple
                or not self.edge_templates or len(self.edge_templates) > 8
                or any(type(row) is not tuple or len(row) != 3 for row in self.edge_templates)):
            raise ValueError("operation path requires distinct endpoints and bounded edge templates")
        for atom, left, right in self.edge_templates:
            if (type(atom) is not int or atom <= 0 or not isinstance(left, ObjectIdentity)
                    or not isinstance(right, ObjectIdentity) or left == right):
                raise ValueError("operation path edge template is malformed")
        if (len({atom for atom, _, _ in self.edge_templates}) != len(self.edge_templates)
                or self.edge_templates != tuple(sorted(self.edge_templates, key=lambda row: row[0]))):
            raise ValueError("operation path edge templates must be unique and canonical")

    def stable_key(self):
        edges = pack_record(SEQUENCE, *(pack_record(OPERATION_PATH_EDGE, (atom,), left.stable_key(),
            right.stable_key()) for atom, left, right in self.edge_templates))
        return pack_record(OPERATION_PATH_QUERY, (self.start_variable,), (self.end_variable,), edges)

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(key, OPERATION_PATH_QUERY)
        if len(fields) != 3 or len(fields[0]) != 1 or len(fields[1]) != 1:
            raise ValueError("operation path query is incomplete")
        templates = []
        for edge in unpack_record(fields[2], SEQUENCE):
            row = unpack_record(edge, OPERATION_PATH_EDGE)
            if len(row) != 3 or len(row[0]) != 1:
                raise ValueError("operation path edge template is incomplete")
            templates.append((row[0][0], ObjectIdentity.from_stable_key(row[1]),
                              ObjectIdentity.from_stable_key(row[2])))
        result = cls(fields[0][0], fields[1][0], tuple(templates))
        if result.stable_key() != key:
            raise ValueError("operation path query is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class IntegerOperationProgram:
    operation_key: tuple[int, ...]
    external_variables: tuple[int, ...]
    atoms: tuple[OperationAtom, ...]
    guards: tuple[tuple[int, int, int], ...]
    projections: tuple[OperationProjection, ...]
    minimum_matches: int = 1
    maximum_matches: int = 1
    order_kind: int = ORDER_DECLARED
    order_roles: tuple[ObjectIdentity, ...] = ()
    # 普通 program 引用单一 connector；path program 可引用按 ID 排序的有界许可集合。
    realization_key: tuple[int, ...] = ()
    path_query: OperationPathQuery | None = None

    def __post_init__(self):
        integer_key(self.operation_key)
        integer_key(self.realization_key, allow_empty=True)
        if (type(self.external_variables) is not tuple or not self.external_variables
                or self.external_variables != tuple(sorted(set(self.external_variables)))):
            raise ValueError("operation external variables must be canonical")
        for variable in self.external_variables:
            positive(variable)
        if (type(self.atoms) is not tuple or not self.atoms
                or any(not isinstance(atom, OperationAtom) for atom in self.atoms)):
            raise ValueError("operation requires explicit typed graph atoms")
        known = set(self.external_variables)
        schemas = {}
        for atom in self.atoms:
            variables = {variable for _, variable in atom.role_variables}
            if not known.intersection(variables):
                raise ValueError("operation atom is disconnected from the request")
            known.update(variables)
            prior = schemas.setdefault(atom.schema.relation, schema_key(atom.schema))
            if prior != schema_key(atom.schema):
                raise ValueError("operation gives one relation conflicting schemas")
        if not set(self.external_variables) <= {
                variable for atom in self.atoms for _, variable in atom.role_variables}:
            raise ValueError("operation ignores an external request variable")
        if type(self.guards) is not tuple:
            raise ValueError("operation guards must be an integer tuple")
        for row in self.guards:
            if (type(row) is not tuple or len(row) != 3
                    or any(type(value) is not int for value in row)
                    or row[0] not in {EQUAL, UNEQUAL} or not set(row[1:]) <= known):
                raise ValueError("operation guard has an unknown variable or primitive")
        if (type(self.projections) is not tuple or not self.projections
                or any(not isinstance(item, OperationProjection) for item in self.projections)
                or len({item.atom for item in self.projections}) != len(self.projections)):
            raise ValueError("operation requires unique graph projections")
        for item in self.projections:
            if item.atom > len(self.atoms) or set(item.roles) != {
                    slot.role for slot in self.atoms[item.atom - 1].schema.slots}:
                raise ValueError("operation projection must preserve every fact role")
        positive(self.minimum_matches)
        positive(self.maximum_matches)
        if self.maximum_matches < self.minimum_matches:
            raise ValueError("operation match bounds are reversed")
        if type(self.order_kind) is not int or self.order_kind not in {ORDER_DECLARED, ORDER_DIRECTED_CHAIN}:
            raise ValueError("operation ordering primitive is unrecognized")
        if type(self.order_roles) is not tuple:
            raise ValueError("operation ordering roles must be a tuple")
        if self.path_query is None:
            if self.order_kind == ORDER_DECLARED:
                if self.order_roles or self.maximum_matches != 1:
                    raise ValueError("declared order requires a unique complete match")
            elif (len(self.order_roles) != 2 or len(set(self.order_roles)) != 2
                  or any(not set(self.order_roles) <= set(item.roles) for item in self.projections)):
                raise ValueError("directed ordering needs two actual projected roles")
        elif self.order_kind != ORDER_DIRECTED_CHAIN or self.order_roles:
            raise ValueError("path order is supplied by its licensed edge templates")
        if self.path_query is not None:
            path = self.path_query
            if not isinstance(path, OperationPathQuery):
                raise TypeError("operation path query has the wrong record type")
            if (self.realization_key and (len(self.realization_key) < 2
                    or len(self.realization_key) > 65
                    or any(value <= 0 for value in self.realization_key)
                    or self.realization_key[1:] != tuple(sorted(set(self.realization_key[1:]))))):
                raise ValueError("path realization permission must list unique ordered graph variants")
            templates = {atom: (left, right) for atom, left, right in path.edge_templates}
            projection_atoms = {item.atom for item in self.projections}
            relations = tuple(atom.schema.relation for atom in self.atoms)
            if (self.external_variables != tuple(sorted((path.start_variable, path.end_variable)))
                    or len(self.atoms) != len(templates) or set(templates) != set(range(1, len(self.atoms) + 1))
                    or len(set(relations)) != len(relations)
                    or any(len(atom.schema.slots) != 2 for atom in self.atoms)
                    or any(dict(atom.role_variables).get(templates[index][0]) != path.start_variable
                           or dict(atom.role_variables).get(templates[index][1]) != path.end_variable
                           for index, atom in enumerate(self.atoms, 1))
                    or projection_atoms != set(range(1, len(self.atoms) + 1))
                    or self.guards or self.minimum_matches < 2 or self.maximum_matches > 64):
                raise ValueError("operation path differs from its licensed binary schemas, projections, or bounds")

    def stable_key(self):
        base = pack_record(PROGRAM, self.operation_key, self.external_variables,
            pack_record(SEQUENCE, *(atom.stable_key() for atom in self.atoms)),
            pack_record(SEQUENCE, *(pack_record(GUARD, row) for row in self.guards)),
            pack_record(SEQUENCE, *(item.stable_key() for item in self.projections)),
            (self.minimum_matches, self.maximum_matches, self.order_kind),
            pack_record(SEQUENCE, *(role.stable_key() for role in self.order_roles)),
            self.realization_key)
        return (base if self.path_query is None else
                pack_record(PROGRAM_PATH, base, self.path_query.stable_key()))

    @classmethod
    def from_stable_key(cls, key):
        if key and key[0] == PROGRAM_PATH:
            wrapper = unpack_record(key, PROGRAM_PATH)
            if len(wrapper) != 2:
                raise ValueError("operation path program wrapper is incomplete")
            path = OperationPathQuery.from_stable_key(wrapper[1])
            result = cls._from_program_fields(unpack_record(wrapper[0], PROGRAM), path)
            if result.stable_key() != key:
                raise ValueError("operation path program is not canonical")
            return result
        fields = unpack_record(key, PROGRAM)
        result = cls._from_program_fields(fields)
        if result.stable_key() != key:
            raise ValueError("operation program is not canonical")
        return result

    @classmethod
    def _from_program_fields(cls, fields, path_query=None):
        if len(fields) != 8 or len(fields[5]) != 3:
            raise ValueError("operation program is incomplete")
        guards = []
        for key_guard in unpack_record(fields[3], SEQUENCE):
            row = unpack_record(key_guard, GUARD)
            if len(row) != 1:
                raise ValueError("operation guard is incomplete")
            guards.append(row[0])
        projections = []
        for key_projection in unpack_record(fields[4], SEQUENCE):
            row = unpack_record(key_projection, PROJECTION)
            if len(row) != 2 or len(row[0]) != 1:
                raise ValueError("operation projection is incomplete")
            projections.append(OperationProjection(row[0][0], tuple(
                ObjectIdentity.from_stable_key(role) for role in unpack_record(row[1], SEQUENCE))))
        return cls(fields[0], fields[1], tuple(OperationAtom.from_stable_key(atom)
            for atom in unpack_record(fields[2], SEQUENCE)), tuple(guards), tuple(projections),
            *fields[5], tuple(ObjectIdentity.from_stable_key(role)
                for role in unpack_record(fields[6], SEQUENCE)), fields[7], path_query)


@dataclass(frozen=True, slots=True)
class OperationProgramWitness:
    program: IntegerOperationProgram
    source: SourceRef
    source_sha256: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    authorized_forming_sources: tuple[SourceRef, ...]

    def __post_init__(self):
        if not isinstance(self.program, IntegerOperationProgram) or not isinstance(self.source, SourceRef):
            raise TypeError("operation witness requires an integer program and physical source")
        if (type(self.source_sha256) is not tuple or len(self.source_sha256) != 32
                or any(type(value) is not int or not 0 <= value <= 255 for value in self.source_sha256)
                or self.source.source_id != ((int.from_bytes(bytes(self.source_sha256[:8]), "big")
                                             & ((1 << 63) - 1)) or 1)):
            raise ValueError("operation witness physical digest does not match its source")
        integer_key(self.source_identity)
        integer_key(self.license_identity)
        if (type(self.authorized_forming_sources) is not tuple or not self.authorized_forming_sources
                or any(not isinstance(source, SourceRef) for source in self.authorized_forming_sources)
                or self.authorized_forming_sources != tuple(sorted(set(self.authorized_forming_sources),
                                                                   key=SourceRef.stable_key))):
            raise ValueError("operation needs canonical complete forming-source permission")

    def stable_key(self):
        return pack_record(WITNESS, self.program.stable_key(), self.source.stable_key(),
            self.source_sha256, self.source_identity, self.license_identity,
            pack_record(SEQUENCE, *(source.stable_key() for source in self.authorized_forming_sources)))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(key, WITNESS)
        if len(fields) != 6:
            raise ValueError("operation witness is incomplete")
        result = cls(IntegerOperationProgram.from_stable_key(fields[0]),
            SourceRef.from_stable_key(fields[1]), fields[2], fields[3], fields[4],
            tuple(SourceRef.from_stable_key(source) for source in unpack_record(fields[5], SEQUENCE)))
        if result.stable_key() != key:
            raise ValueError("operation witness is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationProgramBudget:
    max_atoms: int = 8
    max_facts: int = 512
    max_match_attempts: int = 4096
    max_matches: int = 64
    max_key_values: int = 65536
    max_proof_values: int = 262144
    max_context_values: int = 1048576

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            positive(getattr(self, field))
