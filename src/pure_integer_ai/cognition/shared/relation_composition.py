"""Source-qualified composition instructions and unresolved derivation proofs.

The primitive implements an explicitly supplied two-premise join. It assigns
no law to a relation name and gives derived candidates no factual support.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord, EVIDENCE_SUPPORT, HypothesisKey
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.query_state import SPACE_CORE
from pure_integer_ai.cognition.shared.semantic_graph import AtomicRoleBinding
from pure_integer_ai.cognition.shared.typed_relation import (
    CompositionRule, DerivedRelationCandidate, RelationSchema,
    RelationPremise, RelationSlotSchema, SameKindConstraint,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    pack_record, unpack_record,
)


COMPOSITION_WITNESS = 92114
COMPOSITION_SCHEMA = 92101
COMPOSITION_SLOT = 92102
COMPOSITION_CONSTRAINT = 92103
COMPOSITION_SCHEMA_LIST = 92104
COMPOSITION_SOURCE_LIST = 92105
COMPOSITION_RULE = 92106
COMPOSITION_PENDING_PROOF_V1 = 92107
COMPOSITION_PENDING_PROOF = 92115
COMPOSITION_PREMISE_EVIDENCE = 92109


def _key(value: tuple[int, ...]) -> tuple[int, ...]:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError("composition requires a complete nonnegative integer key")
    return value


def schema_key(schema: RelationSchema) -> tuple[int, ...]:
    slots = tuple(pack_record(
        COMPOSITION_SLOT, slot.role.stable_key(),
        tuple(sorted(slot.allowed_object_kinds)),
        (slot.min_count, 0 if slot.max_count is None else 1,
         0 if slot.max_count is None else slot.max_count),
    ) for slot in sorted(schema.slots, key=lambda item: item.role.stable_key()))
    constraints = tuple(pack_record(
        COMPOSITION_CONSTRAINT, item.constraint.stable_key(),
        pack_record(COMPOSITION_SOURCE_LIST,
                    *(role.stable_key() for role in item.roles)),
    ) for item in sorted(schema.same_kind_constraints,
                         key=lambda item: item.constraint.stable_key()))
    return pack_record(
        COMPOSITION_SCHEMA, schema.schema.stable_key(), schema.relation.stable_key(),
        pack_record(COMPOSITION_SCHEMA_LIST, *slots),
        pack_record(COMPOSITION_SCHEMA_LIST, *constraints))


def schema_from_key(key: tuple[int, ...]) -> RelationSchema:
    fields = unpack_record(key, COMPOSITION_SCHEMA)
    if len(fields) != 4:
        raise ValueError("composition schema field count mismatch")
    slots = []
    for slot_key in unpack_record(fields[2], COMPOSITION_SCHEMA_LIST):
        parts = unpack_record(slot_key, COMPOSITION_SLOT)
        if (len(parts) != 3 or len(parts[2]) != 3 or parts[2][1] not in {0, 1}
                or (parts[2][1] == 0 and parts[2][2] != 0)):
            raise ValueError("composition slot cardinality is incomplete")
        slots.append(RelationSlotSchema(
            ObjectIdentity.from_stable_key(parts[0]), frozenset(parts[1]),
            parts[2][0], parts[2][2] if parts[2][1] else None))
    constraints = []
    for constraint_key in unpack_record(fields[3], COMPOSITION_SCHEMA_LIST):
        parts = unpack_record(constraint_key, COMPOSITION_CONSTRAINT)
        if len(parts) != 2:
            raise ValueError("composition constraint field count mismatch")
        constraints.append(SameKindConstraint(
            ObjectIdentity.from_stable_key(parts[0]), tuple(
                ObjectIdentity.from_stable_key(role)
                for role in unpack_record(parts[1], COMPOSITION_SOURCE_LIST))))
    result = RelationSchema(
        ObjectIdentity.from_stable_key(fields[0]),
        ObjectIdentity.from_stable_key(fields[1]), tuple(slots), tuple(constraints))
    if schema_key(result) != key:
        raise ValueError("composition schema key is not canonical")
    return result


def rule_key(rule: CompositionRule) -> tuple[int, ...]:
    return pack_record(COMPOSITION_RULE, *(getattr(rule, field).stable_key()
                                          for field in rule.__dataclass_fields__))


def _entry_fields(key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Decode the six complete fields in an EvidenceEntry stable key."""
    if (type(key) is not tuple or len(key) < 9
            or any(type(item) is not int or item < 0 for item in key)
            or key[0] != SPACE_CORE or key[1] not in {1, 2, 3}
            or key[2] != 1):
        raise ValueError("composition evidence must be exact current Core Evidence")
    cursor, fields = 3, []
    for _ in range(6):
        if cursor >= len(key):
            raise ValueError("composition EvidenceEntry is truncated")
        size = key[cursor]
        cursor += 1
        end = cursor + size
        if end > len(key):
            raise ValueError("composition EvidenceEntry field is truncated")
        fields.append(key[cursor:end])
        cursor = end
    if cursor != len(key):
        raise ValueError("composition EvidenceEntry has trailing data")
    return tuple(fields)


@dataclass(frozen=True, slots=True)
class RelationCompositionWitness:
    """A licensed train instruction plus exact schemas and allowed sources."""

    rule: CompositionRule
    schemas: tuple[RelationSchema, ...]
    source: SourceRef
    source_sha256: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    authorized_forming_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rule, CompositionRule) or not isinstance(self.source, SourceRef):
            raise TypeError("composition witness needs a typed rule and source")
        if (type(self.source_sha256) is not tuple or len(self.source_sha256) != 32
                or any(type(item) is not int or not 0 <= item <= 255
                       for item in self.source_sha256)
                or self.source.source_id != ((int.from_bytes(bytes(self.source_sha256[:8]), "big")
                                             & ((1 << 63) - 1)) or 1)):
            raise ValueError("composition physical instruction digest differs")
        _key(self.source_identity)
        _key(self.license_identity)
        if (type(self.schemas) is not tuple or not self.schemas
                or any(not isinstance(item, RelationSchema) for item in self.schemas)):
            raise TypeError("composition witness requires complete schemas")
        if (type(self.authorized_forming_sources) is not tuple
                or not self.authorized_forming_sources
                or any(not isinstance(item, SourceRef)
                       for item in self.authorized_forming_sources)):
            raise TypeError("composition witness needs actual permitted forming sources")
        object.__setattr__(self, "schemas", tuple(sorted(
            self.schemas, key=lambda item: item.schema.stable_key())))
        object.__setattr__(self, "authorized_forming_sources", tuple(sorted(
            self.authorized_forming_sources, key=SourceRef.stable_key)))
        if (len({schema.schema for schema in self.schemas}) != len(self.schemas)
                or len({schema.relation for schema in self.schemas}) != len(self.schemas)
                or len(set(self.authorized_forming_sources))
                != len(self.authorized_forming_sources)):
            raise ValueError("composition witness has duplicate schema/source identities")
        by_relation = {schema.relation: schema for schema in self.schemas}
        if set(by_relation) != {
                self.rule.first_relation, self.rule.second_relation, self.rule.result_relation}:
            raise ValueError("composition witness schemas do not cover the exact instruction")
        for relation, left, right in (
                (self.rule.first_relation, self.rule.first_input_role, self.rule.first_join_role),
                (self.rule.second_relation, self.rule.second_join_role, self.rule.second_output_role),
                (self.rule.result_relation, self.rule.result_input_role, self.rule.result_output_role)):
            schema = by_relation[relation]
            if (len(schema.slots) != 2 or {slot.role for slot in schema.slots} != {left, right}
                    or any((slot.min_count, slot.max_count) != (1, 1) for slot in schema.slots)):
                raise ValueError("composition instruction requires exact single-valued binary schemas")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            COMPOSITION_WITNESS, rule_key(self.rule),
            pack_record(COMPOSITION_SCHEMA_LIST, *(schema_key(item) for item in self.schemas)),
            self.source.stable_key(), self.source_sha256,
            self.source_identity, self.license_identity,
            pack_record(COMPOSITION_SOURCE_LIST,
                        *(item.stable_key() for item in self.authorized_forming_sources)))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> RelationCompositionWitness:
        fields = unpack_record(key, COMPOSITION_WITNESS)
        if len(fields) != 7:
            raise ValueError("composition witness field count mismatch")
        rule_parts = unpack_record(fields[0], COMPOSITION_RULE)
        if len(rule_parts) != 10:
            raise ValueError("composition instruction is incomplete")
        result = cls(
            CompositionRule(*(ObjectIdentity.from_stable_key(item) for item in rule_parts)),
            tuple(schema_from_key(item) for item in unpack_record(fields[1], COMPOSITION_SCHEMA_LIST)),
            SourceRef.from_stable_key(fields[2]), fields[3], fields[4], fields[5], tuple(
                SourceRef.from_stable_key(item)
                for item in unpack_record(fields[6], COMPOSITION_SOURCE_LIST)))
        if result.stable_key() != key:
            raise ValueError("composition witness is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PendingRelationComposition:
    """An algebra candidate and its rule authority; still has no factual Evidence."""

    witness: RelationCompositionWitness
    candidate: DerivedRelationCandidate
    premise_evidence_keys: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.witness, RelationCompositionWitness)
                or not isinstance(self.candidate, DerivedRelationCandidate)
                or self.candidate.rule != self.witness.rule.rule
                or self.candidate.relation != self.witness.rule.result_relation):
            raise ValueError("pending composition does not belong to its witness")
        if (type(self.premise_evidence_keys) is not tuple
                or len(self.premise_evidence_keys) != len(self.candidate.premises)
                or any(type(group) is not tuple or not group
                       or len(set(group)) != len(group)
                       or any(type(key) is not tuple or not key
                              or any(type(value) is not int or value < 0 for value in key)
                              for key in group)
                       for group in self.premise_evidence_keys)):
            raise ValueError("pending composition requires full current premise Evidence keys")
        for premise, group in zip(self.candidate.premises,
                                  self.premise_evidence_keys, strict=True):
            if group != tuple(sorted(group)):
                raise ValueError("pending premise Evidence keys must be canonical")
            records = []
            expected_hypothesis = premise.proposition.stable_key()
            for entry_key in group:
                source_ref, hypothesis, evidence_ref, scope, payload, supersedes = (
                    _entry_fields(entry_key))
                record = EvidenceRecord.from_stable_key(payload)
                reference = (1, len(record.hypothesis.stable_key()),
                             *record.hypothesis.stable_key(), record.evidence_id)
                prior = (() if not record.supersedes_evidence_id else (
                    1, len(record.hypothesis.stable_key()),
                    *record.hypothesis.stable_key(), record.supersedes_evidence_id))
                if (record.hypothesis != premise.hypothesis
                        or record.stance != entry_key[1]
                        or not source_ref or source_ref[0] <= 0
                        or source_ref[1:] != record.source.stable_key()
                        or hypothesis != expected_hypothesis
                        or evidence_ref != reference
                        or scope != premise.hypothesis.scope.stable_key()
                        or supersedes != prior):
                    raise ValueError("pending premise Evidence identity/provenance mismatch")
                if record.stance == EVIDENCE_SUPPORT:
                    records.append(record.evidence_id)
            if tuple(sorted(records)) != tuple(sorted(premise.support_evidence_ids)):
                raise ValueError("pending premise Evidence IDs differ from Core support snapshot")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(COMPOSITION_PENDING_PROOF,
                           self.witness.stable_key(), self.candidate.stable_key(),
                           pack_record(COMPOSITION_PREMISE_EVIDENCE, *(
                               pack_record(COMPOSITION_PREMISE_EVIDENCE, *group)
                               for group in self.premise_evidence_keys)))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> PendingRelationComposition:
        fields = unpack_record(key, COMPOSITION_PENDING_PROOF)
        if len(fields) != 3:
            raise ValueError("pending composition proof field count mismatch")
        result = cls(
            RelationCompositionWitness.from_stable_key(fields[0]),
            _candidate_from_key(fields[1]), tuple(
                unpack_record(group, COMPOSITION_PREMISE_EVIDENCE)
                for group in unpack_record(fields[2], COMPOSITION_PREMISE_EVIDENCE)))
        if result.stable_key() != key:
            raise ValueError("pending composition proof is not canonical")
        return result


def _candidate_from_key(key: tuple[int, ...]) -> DerivedRelationCandidate:
    """Restore the existing complete DerivedRelationCandidate encoding."""
    _key(key)
    cursor = 0

    def scalar() -> int:
        nonlocal cursor
        if cursor >= len(key):
            raise ValueError("composition candidate is truncated")
        value = key[cursor]
        cursor += 1
        return value

    def part() -> tuple[int, ...]:
        nonlocal cursor
        size = scalar()
        end = cursor + size
        if size <= 0 or end > len(key):
            raise ValueError("composition candidate field is truncated")
        value = key[cursor:end]
        cursor = end
        return value

    relation, rule = ObjectIdentity.from_stable_key(part()), ObjectIdentity.from_stable_key(part())
    schema_count = scalar()
    if not 1 <= schema_count <= 3:
        raise ValueError("pending composition schema count differs")
    schemas = tuple(ObjectIdentity.from_stable_key(part()) for _ in range(schema_count))
    if scalar() != 2:
        raise ValueError("pending composition requires two result bindings")
    bindings = []
    for _ in range(2):
        binding = part()
        role_size = binding[0]
        offset = role_size + 1
        if role_size <= 0 or offset >= len(binding):
            raise ValueError("composition result role is truncated")
        filler_size = binding[offset]
        if filler_size <= 0 or offset + filler_size + 2 != len(binding):
            raise ValueError("composition result filler is truncated")
        bindings.append(AtomicRoleBinding(
            ObjectIdentity.from_stable_key(binding[1:offset]),
            ObjectIdentity.from_stable_key(binding[offset + 1:-1]), binding[-1]))
    if scalar() != 2:
        raise ValueError("pending composition requires exactly two premises")
    premises = []
    for _ in range(2):
        premise = part()
        proposition_size = premise[0]
        offset = proposition_size + 1
        if proposition_size <= 0 or offset >= len(premise):
            raise ValueError("composition premise proposition is truncated")
        hypothesis_size = premise[offset]
        end = offset + hypothesis_size + 1
        if hypothesis_size <= 0 or end >= len(premise) or end + premise[end] + 1 != len(premise):
            raise ValueError("composition premise hypothesis/support is truncated")
        premises.append(RelationPremise(
            ObjectIdentity.from_stable_key(premise[1:offset]),
            HypothesisKey.from_stable_key(premise[offset + 1:end]), premise[end + 1:]))
    if cursor != len(key):
        raise ValueError("composition candidate has trailing data")
    return DerivedRelationCandidate(relation, tuple(bindings), rule, schemas, tuple(premises))
