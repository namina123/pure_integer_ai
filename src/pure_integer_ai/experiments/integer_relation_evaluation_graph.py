"""Label-free integer graph compiler and cold course for evaluation splits."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicRoleBinding,
    semantic_source,
)
from pure_integer_ai.experiments.integer_graph_novelty import (
    IntegerGraphNoveltyCase,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    parse_w06_relation_graph_structure,
)
from pure_integer_ai.experiments.ph2_w06_registry import W06_AUTHORED_SOURCE
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
)


INTEGER_RELATION_EVALUATION_GRAPH_FORMAT = (
    "PURE_INTEGER_RELATION_EVALUATION_GRAPH_V1")
EVALUATION_SPLITS = ("held_out", "negative", "conflict")
_SPLIT_CODES = {name: index for index, name in enumerate(EVALUATION_SPLITS, 2)}
_TOP_LEVEL_FIELDS = frozenset({
    "format", "split", "split_code", "owner_identity", "source_identity",
    "license_identity", "case_count", "record_count", "cases", "records",
    "rejection_count", "rejections", "free_dialogue_complete",
})
_CASE_FIELDS = frozenset({
    "case_key", "input_values", "dialogue_chain_keys", "observation_keys",
})
_RECORD_FIELDS = frozenset({
    "source_ref_key", "dataset_key", "artifact_key", "observation_key",
    "semantic_key", "proposition_key", "predicate_key", "context_key",
    "source_anchor_key", "schema_key", "relation_key", "role_bindings",
    "slots", "relation_constraints", "endpoints", "surface_values",
    "anchor_span", "relation_kind", "directionality", "source_cluster_key",
    "dedup_cluster_key", "content_group_key", "template_group_key",
    "shape_group_key",
})
_ROLE_BINDING_FIELDS = frozenset({
    "binding_key", "filler_key", "ordinal", "role_key",
})
_SLOT_FIELDS = frozenset({
    "allowed_object_kinds", "max_count", "min_count", "role_key",
})
_CONSTRAINT_FIELDS = frozenset({"constraint_key", "role_keys"})
_ENDPOINT_FIELDS = frozenset({
    "endpoint_key", "object_kind", "ordinal", "start", "end",
})
_REJECTION_FIELDS = frozenset({
    "source_ref_key", "dataset_key", "artifact_key", "observation_key",
    "semantic_key", "proposition_key", "predicate_key", "context_key",
    "source_anchor_key", "schema_key", "relation_key", "violations",
    "endpoints", "surface_values", "anchor_span", "relation_kind",
    "directionality", "source_cluster_key", "dedup_cluster_key",
    "content_group_key", "template_group_key", "shape_group_key",
})
_VIOLATION_FIELDS = frozenset({
    "allowed_object_kinds", "declared_object_kinds", "filler_key", "role_key",
})


class IntegerRelationEvaluationGraphError(ValueError):
    """Evaluation graph has labels, incomplete identity, or invalid topology."""


def _key(value: Any, *, where: str, positive: bool = False) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(item) is not int or item < 0
                   or (positive and item <= 0) for item in value)):
        raise IntegerRelationEvaluationGraphError(
            f"{where} must be a nonempty strict integer key")
    return tuple(value)


def _unicode_values(value: Any, *, where: str) -> tuple[int, ...]:
    result = _key(value, where=where)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise IntegerRelationEvaluationGraphError(
            f"{where} must contain Unicode scalar values")
    return result


def _identity(value: Any, *, where: str) -> ObjectIdentity:
    try:
        return ObjectIdentity.from_stable_key(_key(value, where=where))
    except (TypeError, ValueError) as error:
        raise IntegerRelationEvaluationGraphError(
            f"{where} object identity is invalid") from error


def _integer_tree(value: Any, *, where: str) -> None:
    if value is None or type(value) is int:
        return
    if isinstance(value, (str, bool, float)):
        raise IntegerRelationEvaluationGraphError(
            f"{where} must not contain {type(value).__name__}")
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _integer_tree(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise IntegerRelationEvaluationGraphError(
                    f"{where} object key is invalid")
            _integer_tree(item, where=f"{where}.{key}")
        return
    raise IntegerRelationEvaluationGraphError(
        f"{where} contains unsupported type {type(value).__name__}")


def _objects(value: Any, *, fields: frozenset[str], where: str,
             ) -> tuple[CanonicalJsonObject, ...]:
    if not isinstance(value, (tuple, list)):
        raise IntegerRelationEvaluationGraphError(f"{where} must be a list")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, CanonicalJsonObject):
            raw = item.to_value()
        else:
            raw = item
        if not isinstance(raw, dict) or set(raw) != fields:
            raise IntegerRelationEvaluationGraphError(
                f"{where}[{index}] fields drifted")
        _integer_tree(raw, where=f"{where}[{index}]")
        result.append(CanonicalJsonObject.from_value(raw))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class IntegerRelationEvaluationGraphRecord:
    """One typed proposal's topology, with all teacher/evaluator labels omitted."""

    source_ref_key: tuple[int, ...]
    dataset_key: tuple[int, ...]
    artifact_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    semantic_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    context_key: tuple[int, ...]
    source_anchor_key: tuple[int, ...]
    schema_key: tuple[int, ...]
    relation_key: tuple[int, ...]
    role_bindings: tuple[CanonicalJsonObject, ...]
    slots: tuple[CanonicalJsonObject, ...]
    relation_constraints: tuple[CanonicalJsonObject, ...]
    endpoints: tuple[CanonicalJsonObject, ...]
    surface_values: tuple[int, ...]
    anchor_span: tuple[int, int, int]
    relation_kind: int
    directionality: int
    source_cluster_key: tuple[int, ...]
    dedup_cluster_key: tuple[int, ...]
    content_group_key: tuple[int, ...]
    template_group_key: tuple[int, ...]
    shape_group_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
                "source_ref_key", "dataset_key", "artifact_key",
                "observation_key", "source_cluster_key", "dedup_cluster_key",
                "content_group_key", "template_group_key", "shape_group_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), where=name, positive=True))
        for name in (
                "semantic_key", "proposition_key", "predicate_key",
                "context_key", "source_anchor_key", "schema_key", "relation_key"):
            object.__setattr__(self, name, _key(getattr(self, name), where=name))
        object.__setattr__(self, "surface_values", _unicode_values(
            self.surface_values, where="surface_values"))
        span = _key(self.anchor_span, where="anchor_span")
        if (len(span) != 3 or span[1] <= span[0]
                or span[1] > len(self.surface_values)):
            raise IntegerRelationEvaluationGraphError("anchor_span is invalid")
        object.__setattr__(self, "anchor_span", span)
        object.__setattr__(self, "role_bindings", _objects(
            self.role_bindings, fields=_ROLE_BINDING_FIELDS,
            where="role_bindings"))
        object.__setattr__(self, "slots", _objects(
            self.slots, fields=_SLOT_FIELDS, where="slots"))
        object.__setattr__(self, "relation_constraints", _objects(
            self.relation_constraints, fields=_CONSTRAINT_FIELDS,
            where="relation_constraints"))
        object.__setattr__(self, "endpoints", _objects(
            self.endpoints, fields=_ENDPOINT_FIELDS, where="endpoints"))

        proposition = _identity(self.proposition_key, where="proposition_key")
        predicate = _identity(self.predicate_key, where="predicate_key")
        context = _identity(self.context_key, where="context_key")
        anchor = _identity(self.source_anchor_key, where="source_anchor_key")
        schema = _identity(self.schema_key, where="schema_key")
        relation = _identity(self.relation_key, where="relation_key")
        if (proposition.object_kind != OBJECT_PROPOSITION
                or predicate.object_kind != OBJECT_CONCEPT
                or context.object_kind != OBJECT_CONTEXT_SCOPE
                or anchor.object_kind != OBJECT_OCCURRENCE
                or schema.object_kind != OBJECT_STRUCTURE_CONCEPT
                or relation.object_kind != OBJECT_CONCEPT or relation != predicate):
            raise IntegerRelationEvaluationGraphError(
                "proposition, schema, relation, or scope identity drifted")
        try:
            if (semantic_source(proposition) != semantic_source(context)
                    or semantic_source(proposition)
                    != _occurrence_source(anchor)):
                raise IntegerRelationEvaluationGraphError(
                    "proposition, context, and anchor source drifted")
        except (TypeError, ValueError) as error:
            raise IntegerRelationEvaluationGraphError(
                "semantic source identity is invalid") from error
        profile = next((item for item in W06_RELATION_PROFILES.values()
                        if item.relation_kind == self.relation_kind), None)
        if (profile is None or type(self.directionality) is not int
                or self.directionality != profile.directionality
                or relation != authored_relation_identity(self.relation_kind)):
            raise IntegerRelationEvaluationGraphError(
                "relation kind or directionality is not registered")
        _validate_structure(self, proposition, profile)

    def stable_key(self) -> tuple[int, ...]:
        return self.observation_key

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ref_key": list(self.source_ref_key),
            "dataset_key": list(self.dataset_key),
            "artifact_key": list(self.artifact_key),
            "observation_key": list(self.observation_key),
            "semantic_key": list(self.semantic_key),
            "proposition_key": list(self.proposition_key),
            "predicate_key": list(self.predicate_key),
            "context_key": list(self.context_key),
            "source_anchor_key": list(self.source_anchor_key),
            "schema_key": list(self.schema_key),
            "relation_key": list(self.relation_key),
            "role_bindings": [item.to_value() for item in self.role_bindings],
            "slots": [item.to_value() for item in self.slots],
            "relation_constraints": [
                item.to_value() for item in self.relation_constraints],
            "endpoints": [item.to_value() for item in self.endpoints],
            "surface_values": list(self.surface_values),
            "anchor_span": list(self.anchor_span),
            "relation_kind": self.relation_kind,
            "directionality": self.directionality,
            "source_cluster_key": list(self.source_cluster_key),
            "dedup_cluster_key": list(self.dedup_cluster_key),
            "content_group_key": list(self.content_group_key),
            "template_group_key": list(self.template_group_key),
            "shape_group_key": list(self.shape_group_key),
        }

    @classmethod
    def from_dict(cls, value: object) -> "IntegerRelationEvaluationGraphRecord":
        if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
            raise IntegerRelationEvaluationGraphError(
                "evaluation graph record fields drifted or contain labels")
        try:
            return cls(
                *(tuple(value[name]) for name in (
                    "source_ref_key", "dataset_key", "artifact_key",
                    "observation_key", "semantic_key", "proposition_key",
                    "predicate_key", "context_key", "source_anchor_key",
                    "schema_key", "relation_key")),
                tuple(value["role_bindings"]), tuple(value["slots"]),
                tuple(value["relation_constraints"]), tuple(value["endpoints"]),
                tuple(value["surface_values"]), tuple(value["anchor_span"]),
                value["relation_kind"], value["directionality"],
                *(tuple(value[name]) for name in (
                    "source_cluster_key", "dedup_cluster_key",
                    "content_group_key", "template_group_key", "shape_group_key")),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IntegerRelationEvaluationGraphError):
                raise
            raise IntegerRelationEvaluationGraphError(
                "evaluation graph record is invalid") from error


@dataclass(frozen=True, slots=True)
class IntegerRelationEvaluationGraphRejection:
    """A schema-rejected proposal, retained without an expected outcome label."""

    source_ref_key: tuple[int, ...]
    dataset_key: tuple[int, ...]
    artifact_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    semantic_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    context_key: tuple[int, ...]
    source_anchor_key: tuple[int, ...]
    schema_key: tuple[int, ...]
    relation_key: tuple[int, ...]
    violations: tuple[CanonicalJsonObject, ...]
    endpoints: tuple[CanonicalJsonObject, ...]
    surface_values: tuple[int, ...]
    anchor_span: tuple[int, int, int]
    relation_kind: int
    directionality: int
    source_cluster_key: tuple[int, ...]
    dedup_cluster_key: tuple[int, ...]
    content_group_key: tuple[int, ...]
    template_group_key: tuple[int, ...]
    shape_group_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
                "source_ref_key", "dataset_key", "artifact_key",
                "observation_key", "source_cluster_key", "dedup_cluster_key",
                "content_group_key", "template_group_key", "shape_group_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), where=name, positive=True))
        for name in (
                "semantic_key", "proposition_key", "predicate_key",
                "context_key", "source_anchor_key", "schema_key", "relation_key"):
            object.__setattr__(self, name, _key(getattr(self, name), where=name))
        object.__setattr__(self, "violations", _objects(
            self.violations, fields=_VIOLATION_FIELDS, where="violations"))
        object.__setattr__(self, "endpoints", _objects(
            self.endpoints, fields=_ENDPOINT_FIELDS, where="endpoints"))
        object.__setattr__(self, "surface_values", _unicode_values(
            self.surface_values, where="surface_values"))
        span = _key(self.anchor_span, where="anchor_span")
        if (len(span) != 3 or span[1] <= span[0]
                or span[1] > len(self.surface_values)):
            raise IntegerRelationEvaluationGraphError(
                "rejection anchor_span is invalid")
        object.__setattr__(self, "anchor_span", span)
        profile = next((item for item in W06_RELATION_PROFILES.values()
                        if item.relation_kind == self.relation_kind), None)
        if (profile is None or type(self.directionality) is not int
                or profile.directionality != self.directionality):
            raise IntegerRelationEvaluationGraphError(
                "rejection relation kind or directionality is invalid")
        if not self.violations:
            raise IntegerRelationEvaluationGraphError(
                "rejection must retain at least one structural violation")
        proposition = _identity(self.proposition_key, where="proposition_key")
        predicate = _identity(self.predicate_key, where="predicate_key")
        context = _identity(self.context_key, where="context_key")
        anchor = _identity(self.source_anchor_key, where="source_anchor_key")
        schema = _identity(self.schema_key, where="schema_key")
        relation = _identity(self.relation_key, where="relation_key")
        if (proposition.object_kind != OBJECT_PROPOSITION
                or predicate.object_kind != OBJECT_CONCEPT
                or context.object_kind != OBJECT_CONTEXT_SCOPE
                or anchor.object_kind != OBJECT_OCCURRENCE
                or schema.object_kind != OBJECT_STRUCTURE_CONCEPT
                or relation != predicate
                or relation != authored_relation_identity(self.relation_kind)):
            raise IntegerRelationEvaluationGraphError(
                "rejection proposition or schema identity drifted")
        try:
            if (semantic_source(proposition) != semantic_source(context)
                    or semantic_source(proposition) != _occurrence_source(anchor)):
                raise IntegerRelationEvaluationGraphError(
                    "rejection semantic source identities drifted")
        except (TypeError, ValueError) as error:
            raise IntegerRelationEvaluationGraphError(
                "rejection semantic source identity is invalid") from error
        roles = {
            authored_relation_role_identity(kind).stable_key(): allowed
            for kind, allowed in profile.role_object_kinds
        }
        endpoint_by_key = {}
        for item in self.endpoints:
            value = item.to_value()
            endpoint = _identity(value["endpoint_key"], where="endpoint_key")
            if (endpoint.object_kind != value["object_kind"]
                    or type(value["ordinal"]) is not int or value["ordinal"] < 0
                    or type(value["start"]) is not int or value["start"] < 0
                    or type(value["end"]) is not int
                    or value["end"] <= value["start"]
                    or value["end"] > len(self.surface_values)
                    or endpoint.stable_key() in endpoint_by_key):
                raise IntegerRelationEvaluationGraphError(
                    "rejection endpoint is invalid")
            endpoint_by_key[endpoint.stable_key()] = endpoint
        if len(endpoint_by_key) != len(profile.role_object_kinds):
            raise IntegerRelationEvaluationGraphError(
                "rejection endpoint count is not registered")
        seen_roles = set()
        for item in self.violations:
            value = item.to_value()
            role = tuple(value["role_key"])
            filler = _identity(value["filler_key"], where="violation.filler_key")
            allowed = value["allowed_object_kinds"]
            declared = value["declared_object_kinds"]
            if (role not in roles or role in seen_roles
                    or not isinstance(allowed, list) or not isinstance(declared, list)
                    or not allowed or not declared
                    or any(type(kind) is not int or kind <= 0
                           for kind in (*allowed, *declared))
                    or len(set(allowed)) != len(allowed)
                    or len(set(declared)) != len(declared)
                    or set(allowed) != set(roles[role])
                    or filler.stable_key() not in endpoint_by_key
                    or filler.object_kind in roles[role]
                    and set(declared).issubset(roles[role])):
                raise IntegerRelationEvaluationGraphError(
                    "rejection violation does not prove a registered type mismatch")
            seen_roles.add(role)

    def stable_key(self) -> tuple[int, ...]:
        return self.observation_key

    def to_dict(self) -> dict[str, object]:
        return {
            "source_ref_key": list(self.source_ref_key),
            "dataset_key": list(self.dataset_key),
            "artifact_key": list(self.artifact_key),
            "observation_key": list(self.observation_key),
            "semantic_key": list(self.semantic_key),
            "proposition_key": list(self.proposition_key),
            "predicate_key": list(self.predicate_key),
            "context_key": list(self.context_key),
            "source_anchor_key": list(self.source_anchor_key),
            "schema_key": list(self.schema_key),
            "relation_key": list(self.relation_key),
            "violations": [item.to_value() for item in self.violations],
            "endpoints": [item.to_value() for item in self.endpoints],
            "surface_values": list(self.surface_values),
            "anchor_span": list(self.anchor_span),
            "relation_kind": self.relation_kind,
            "directionality": self.directionality,
            "source_cluster_key": list(self.source_cluster_key),
            "dedup_cluster_key": list(self.dedup_cluster_key),
            "content_group_key": list(self.content_group_key),
            "template_group_key": list(self.template_group_key),
            "shape_group_key": list(self.shape_group_key),
        }

    @classmethod
    def from_dict(cls, value: object) -> "IntegerRelationEvaluationGraphRejection":
        if not isinstance(value, dict) or set(value) != _REJECTION_FIELDS:
            raise IntegerRelationEvaluationGraphError(
                "evaluation rejection fields drifted or contain labels")
        try:
            return cls(
                *(tuple(value[name]) for name in (
                    "source_ref_key", "dataset_key", "artifact_key",
                    "observation_key", "semantic_key", "proposition_key",
                    "predicate_key", "context_key", "source_anchor_key",
                    "schema_key", "relation_key")),
                tuple(value["violations"]), tuple(value["endpoints"]),
                tuple(value["surface_values"]), tuple(value["anchor_span"]),
                value["relation_kind"], value["directionality"],
                *(tuple(value[name]) for name in (
                    "source_cluster_key", "dedup_cluster_key",
                    "content_group_key", "template_group_key", "shape_group_key")),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IntegerRelationEvaluationGraphError):
                raise
            raise IntegerRelationEvaluationGraphError(
                "evaluation rejection is invalid") from error


def _occurrence_source(identity: ObjectIdentity):
    if identity.object_kind != OBJECT_OCCURRENCE or len(identity.components) < 4:
        raise IntegerRelationEvaluationGraphError(
            "source anchor occurrence identity is invalid")
    from pure_integer_ai.cognition.shared.identity import SourceRef
    return SourceRef.from_stable_key(identity.components[:-3])


def _validate_structure(
        record: IntegerRelationEvaluationGraphRecord,
        proposition: ObjectIdentity,
        profile,
        ) -> None:
    role_keys = tuple(
        authored_relation_role_identity(kind).stable_key()
        for kind, _allowed in profile.role_object_kinds)
    bindings = {
        tuple(value["role_key"]): value
        for item in record.role_bindings for value in (item.to_value(),)
    }
    slots = {
        tuple(value["role_key"]): value
        for item in record.slots for value in (item.to_value(),)
    }
    if (len(bindings) != len(record.role_bindings)
            or len(slots) != len(record.slots)
            or set(bindings) != set(role_keys) or set(slots) != set(role_keys)
            or len(record.endpoints) != len(role_keys)):
        raise IntegerRelationEvaluationGraphError(
            "relation roles, slots, and endpoints are not closed")
    endpoints = {}
    for item in record.endpoints:
        value = item.to_value()
        endpoint = _identity(value["endpoint_key"], where="endpoint_key")
        if (endpoint.object_kind != value["object_kind"]
                or type(value["ordinal"]) is not int or value["ordinal"] < 0
                or type(value["start"]) is not int or value["start"] < 0
                or type(value["end"]) is not int
                or value["end"] <= value["start"]
                or value["end"] > len(record.surface_values)
                or endpoint.stable_key() in endpoints):
            raise IntegerRelationEvaluationGraphError("endpoint is invalid")
        endpoints[endpoint.stable_key()] = endpoint
    if set(endpoints) != {
            tuple(binding["filler_key"]) for binding in bindings.values()}:
        raise IntegerRelationEvaluationGraphError(
            "endpoint identities do not match relation fillers")
    for role_kind, allowed_kinds in profile.role_object_kinds:
        role = authored_relation_role_identity(role_kind)
        key = role.stable_key()
        binding = bindings[key]
        filler = _identity(binding["filler_key"], where="filler_key")
        binding_identity = _identity(binding["binding_key"], where="binding_key")
        slot = slots[key]
        declared = slot["allowed_object_kinds"]
        if (type(binding["ordinal"]) is not int or binding["ordinal"] < 0
                or binding_identity.object_kind != OBJECT_ROLE_BINDING
                or AtomicRoleBinding(role, filler, binding["ordinal"])
                .identity_for(proposition) != binding_identity
                or filler.stable_key() not in endpoints
                or not isinstance(declared, list) or not declared
                or any(type(item) is not int or item not in allowed_kinds
                       for item in declared)
                or len(set(declared)) != len(declared)
                or filler.object_kind not in allowed_kinds
                or type(slot["min_count"]) is not int or slot["min_count"] != 1
                or type(slot["max_count"]) is not int or slot["max_count"] != 1):
            raise IntegerRelationEvaluationGraphError(
                "role binding or slot violates its registered relation profile")
    role_set = set(role_keys)
    constraint_keys = set()
    for item in record.relation_constraints:
        value = item.to_value()
        identity = _identity(value["constraint_key"], where="constraint_key")
        roles = value["role_keys"]
        if (identity.object_kind != OBJECT_STRUCTURE_CONCEPT
                or identity.stable_key() in constraint_keys
                or not isinstance(roles, list) or len(roles) < 2
                or len(set(tuple(role) for role in roles)) != len(roles)
                or any(tuple(role) not in role_set for role in roles)):
            raise IntegerRelationEvaluationGraphError(
                "relation constraint is invalid")
        constraint_keys.add(identity.stable_key())


@dataclass(frozen=True, slots=True)
class EvaluationGraphAssignment:
    """Explicit evaluator-owned case assignment; contains no expected label."""

    split: str
    case_key: tuple[int, ...]
    input_values: tuple[int, ...]
    dialogue_chain_keys: tuple[tuple[int, ...], ...]
    observation_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.split not in EVALUATION_SPLITS:
            raise IntegerRelationEvaluationGraphError(
                "evaluation assignment split is invalid")
        object.__setattr__(self, "case_key", _key(
            self.case_key, where="case_key"))
        object.__setattr__(self, "input_values", _unicode_values(
            self.input_values, where="input_values"))
        for name in ("dialogue_chain_keys", "observation_keys"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not value:
                raise IntegerRelationEvaluationGraphError(f"{name} is empty")
            keys = tuple(_key(
                item, where=name[:-1], positive=name == "observation_keys")
                for item in value)
            if len(set(keys)) != len(keys):
                raise IntegerRelationEvaluationGraphError(f"{name} is duplicated")
            object.__setattr__(self, name, keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_key": list(self.case_key),
            "input_values": list(self.input_values),
            "dialogue_chain_keys": [list(item) for item in self.dialogue_chain_keys],
            "observation_keys": [list(item) for item in self.observation_keys],
        }

    @classmethod
    def from_dict(cls, split: str, value: object) -> "EvaluationGraphAssignment":
        if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
            raise IntegerRelationEvaluationGraphError(
                "evaluation case fields drifted")
        try:
            return cls(
                split, tuple(value["case_key"]), tuple(value["input_values"]),
                tuple(tuple(item) for item in value["dialogue_chain_keys"]),
                tuple(tuple(item) for item in value["observation_keys"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IntegerRelationEvaluationGraphError):
                raise
            raise IntegerRelationEvaluationGraphError(
                "evaluation case is invalid") from error


@dataclass(frozen=True, slots=True)
class IntegerRelationEvaluationGraphCase:
    assignment: EvaluationGraphAssignment
    records: tuple[IntegerRelationEvaluationGraphRecord, ...]
    rejections: tuple[IntegerRelationEvaluationGraphRejection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, EvaluationGraphAssignment):
            raise TypeError("assignment type is invalid")
        records = tuple(self.records)
        rejections = tuple(self.rejections)
        if (any(not isinstance(item, IntegerRelationEvaluationGraphRecord)
                for item in records)
                or any(not isinstance(item, IntegerRelationEvaluationGraphRejection)
                       for item in rejections)
                or tuple(sorted(records, key=lambda item: item.observation_key)) != records
                or tuple(sorted(rejections, key=lambda item: item.observation_key))
                != rejections
                or tuple(sorted(item.observation_key for item in
                                (*records, *rejections)))
                != tuple(sorted(self.assignment.observation_keys))):
            raise IntegerRelationEvaluationGraphError(
                "evaluation case records do not match the frozen assignment")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejections", rejections)

    def to_novelty_case(self) -> IntegerGraphNoveltyCase:
        if not self.records:
            raise IntegerRelationEvaluationGraphError(
                "schema-rejected case has no topology for novelty admission")
        return IntegerGraphNoveltyCase(
            self.assignment.case_key,
            self.assignment.input_values,
            self.records,
            self.assignment.dialogue_chain_keys,
        )


@dataclass(frozen=True, slots=True)
class IntegerRelationEvaluationGraphSplitCourse:
    split: str
    owner_identity: tuple[int, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    cases: tuple[IntegerRelationEvaluationGraphCase, ...]

    def __post_init__(self) -> None:
        if self.split not in EVALUATION_SPLITS:
            raise IntegerRelationEvaluationGraphError("split is invalid")
        for name in ("owner_identity", "source_identity", "license_identity"):
            object.__setattr__(self, name, _key(
                getattr(self, name), where=name, positive=True))
        cases = tuple(self.cases)
        if (not cases or any(not isinstance(
                    item, IntegerRelationEvaluationGraphCase) for item in cases)
                or any(item.assignment.split != self.split for item in cases)
                or tuple(sorted(cases, key=lambda item: item.assignment.case_key))
                != cases):
            raise IntegerRelationEvaluationGraphError("split cases are invalid")
        if len({item.assignment.case_key for item in cases}) != len(cases):
            raise IntegerRelationEvaluationGraphError("case key is duplicated")
        observations = [record.observation_key for case in cases
                        for record in (*case.records, *case.rejections)]
        if len(set(observations)) != len(observations):
            raise IntegerRelationEvaluationGraphError(
                "an observation belongs to multiple evaluation cases")
        object.__setattr__(self, "cases", cases)

    @property
    def records(self) -> tuple[IntegerRelationEvaluationGraphRecord, ...]:
        return tuple(record for case in self.cases for record in case.records)

    @property
    def rejections(self) -> tuple[IntegerRelationEvaluationGraphRejection, ...]:
        return tuple(sorted((record for case in self.cases
                             for record in case.rejections),
                            key=lambda item: item.observation_key))

    def novelty_cases(self) -> tuple[IntegerGraphNoveltyCase, ...]:
        return tuple(case.to_novelty_case() for case in self.cases
                     if case.records)


def write_integer_relation_evaluation_graph_course(
        path: str | Path,
        course: IntegerRelationEvaluationGraphSplitCourse,
        ) -> str:
    if not isinstance(course, IntegerRelationEvaluationGraphSplitCourse):
        raise TypeError("course type is invalid")
    payload = {
        "format": INTEGER_RELATION_EVALUATION_GRAPH_FORMAT,
        "split": course.split,
        "split_code": _SPLIT_CODES[course.split],
        "owner_identity": list(course.owner_identity),
        "source_identity": list(course.source_identity),
        "license_identity": list(course.license_identity),
        "case_count": len(course.cases),
        "record_count": len(course.records),
        "cases": [item.assignment.to_dict() for item in course.cases],
        "records": [item.to_dict() for item in sorted(
            course.records, key=lambda record: record.observation_key)],
        "rejection_count": len(course.rejections),
        "rejections": [item.to_dict() for item in course.rejections],
        "free_dialogue_complete": 0,
    }
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")) + "\n").encode("ascii")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def read_integer_relation_evaluation_graph_course(
        path: str | Path,
        ) -> IntegerRelationEvaluationGraphSplitCourse:
    raw = Path(path).resolve().read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise IntegerRelationEvaluationGraphError(
            "evaluation course must have one trailing newline")
    try:
        payload = parse_canonical_json_bytes(raw[:-1], require_object=True)
    except ValueError as error:
        raise IntegerRelationEvaluationGraphError(
            "evaluation course JSON is not canonical") from error
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise IntegerRelationEvaluationGraphError(
            "evaluation course fields drifted")
    split = payload.get("split")
    if (payload.get("format") != INTEGER_RELATION_EVALUATION_GRAPH_FORMAT
            or split not in EVALUATION_SPLITS
            or type(payload.get("split_code")) is not int
            or payload["split_code"] != _SPLIT_CODES[split]
            or type(payload.get("free_dialogue_complete")) is not int
            or payload["free_dialogue_complete"] != 0):
        raise IntegerRelationEvaluationGraphError(
            "evaluation course identity or free-dialogue gate drifted")
    try:
        assignments_raw = payload["cases"]
        records_raw = payload["records"]
        rejections_raw = payload["rejections"]
        if (not isinstance(assignments_raw, list)
                or not isinstance(records_raw, list)
                or not isinstance(rejections_raw, list)):
            raise IntegerRelationEvaluationGraphError(
                "evaluation cases, records, and rejections must be lists")
        assignments = tuple(
            EvaluationGraphAssignment.from_dict(split, item)
            for item in assignments_raw)
        if tuple(sorted(assignments, key=lambda item: item.case_key)) != assignments:
            raise IntegerRelationEvaluationGraphError(
                "evaluation cases are not canonically sorted")
        records = tuple(
            IntegerRelationEvaluationGraphRecord.from_dict(item)
            for item in records_raw)
        rejections = tuple(
            IntegerRelationEvaluationGraphRejection.from_dict(item)
            for item in rejections_raw)
        if (tuple(sorted(records, key=lambda item: item.observation_key)) != records
                or tuple(sorted(rejections,
                                key=lambda item: item.observation_key)) != rejections):
            raise IntegerRelationEvaluationGraphError(
                "evaluation records are not canonically sorted")
        by_key = {item.observation_key: item for item in records}
        reject_by_key = {item.observation_key: item for item in rejections}
        cases = []
        for assignment in assignments:
            record_keys = tuple(key for key in assignment.observation_keys
                                if key in by_key)
            rejection_keys = tuple(key for key in assignment.observation_keys
                                   if key in reject_by_key)
            cases.append(IntegerRelationEvaluationGraphCase(
                assignment,
                tuple(sorted((by_key[key] for key in record_keys),
                             key=lambda item: item.observation_key)),
                tuple(sorted((reject_by_key[key] for key in rejection_keys),
                             key=lambda item: item.observation_key)),
            ))
        course = IntegerRelationEvaluationGraphSplitCourse(
            split,
            _key(payload["owner_identity"], where="owner_identity", positive=True),
            _key(payload["source_identity"], where="source_identity", positive=True),
            _key(payload["license_identity"], where="license_identity", positive=True),
            tuple(cases),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        if isinstance(error, IntegerRelationEvaluationGraphError):
            raise
        raise IntegerRelationEvaluationGraphError(
            "evaluation course records are invalid") from error
    if (type(payload.get("case_count")) is not int
            or payload["case_count"] != len(course.cases)
            or type(payload.get("record_count")) is not int
            or payload["record_count"] != len(course.records)
            or type(payload.get("rejection_count")) is not int
            or payload["rejection_count"] != len(course.rejections)
            or set(by_key) | set(reject_by_key) != {
                key for case in assignments for key in case.observation_keys}):
        raise IntegerRelationEvaluationGraphError(
            "evaluation course counts or record ownership drifted")
    return course


def _compile_record(
        observation: ObservationRecord,
        source: SourceRefRecord,
        candidate,
        ) -> IntegerRelationEvaluationGraphRecord:
    value = observation.typed_payload.to_value()
    family = W06_RELATION_PROFILES[candidate.relation_family]
    return IntegerRelationEvaluationGraphRecord(
        tuple(source.stable_key.components),
        tuple(observation.dataset_key.components),
        tuple(observation.artifact_key.components),
        tuple(observation.stable_key.components),
        tuple(value["semantic_key"]),
        tuple(value["candidate_definition"]["proposition_key"]),
        tuple(value["candidate_definition"]["predicate_key"]),
        tuple(value["candidate_definition"]["context_key"]),
        tuple(value["candidate_definition"]["source_anchor_key"]),
        tuple(value["relation_schema"]["schema_key"]),
        tuple(value["relation_schema"]["relation_key"]),
        tuple(value["candidate_definition"]["role_bindings"]),
        tuple(value["relation_schema"]["slots"]),
        tuple(value["relation_schema"]["constraints"]),
        tuple({key: endpoint[key] for key in _ENDPOINT_FIELDS}
              for endpoint in value["endpoints"]),
        tuple(map(ord, candidate.surface)),
        tuple(value["source_anchor_span"]),
        family.relation_kind,
        candidate.directionality,
        tuple(source.source_cluster_key.components),
        tuple(observation.dedup_cluster_key.components),
        tuple(observation.content_group_key.components),
        tuple(observation.template_group_key.components),
        tuple(observation.shape_group_key.components),
    )


def _compile_rejection(
        observation: ObservationRecord,
        source: SourceRefRecord,
        rejection,
        ) -> IntegerRelationEvaluationGraphRejection:
    value = observation.typed_payload.to_value()
    profile = W06_RELATION_PROFILES[rejection.relation_family]
    violations = tuple({
        "role_key": item.role.stable_key(),
        "filler_key": item.filler.stable_key(),
        "allowed_object_kinds": item.allowed_object_kinds,
        "declared_object_kinds": item.declared_object_kinds,
    } for item in rejection.violations)
    endpoints = tuple({
        "endpoint_key": item.identity.stable_key(),
        "object_kind": item.object_kind,
        "ordinal": item.ordinal,
        "start": item.start,
        "end": item.end,
    } for item in rejection.endpoints)
    return IntegerRelationEvaluationGraphRejection(
        tuple(source.stable_key.components),
        tuple(observation.dataset_key.components),
        tuple(observation.artifact_key.components),
        tuple(observation.stable_key.components),
        tuple(value["semantic_key"]),
        rejection.proposition.proposition.stable_key(),
        rejection.schema.relation.stable_key(),
        tuple(value["candidate_definition"]["context_key"]),
        tuple(value["candidate_definition"]["source_anchor_key"]),
        rejection.schema.schema.stable_key(),
        rejection.schema.relation.stable_key(),
        violations,
        endpoints,
        tuple(map(ord, rejection.surface)),
        tuple(value["source_anchor_span"]),
        profile.relation_kind,
        rejection.directionality,
        tuple(source.source_cluster_key.components),
        tuple(observation.dedup_cluster_key.components),
        tuple(observation.content_group_key.components),
        tuple(observation.template_group_key.components),
        tuple(observation.shape_group_key.components),
    )


def compile_w06_evaluation_graph_splits(
        observations: Sequence[ObservationRecord],
        source_refs: Sequence[SourceRefRecord],
        assignments: Mapping[str, Sequence[EvaluationGraphAssignment]],
        *,
        owner_identity: tuple[int, ...],
        source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        ) -> dict[str, IntegerRelationEvaluationGraphSplitCourse]:
    """Compile typed observations without accepting or opening any labels.

    The public structural parser performs the source, schema, endpoint, role,
    and span checks shared with train. It does not consult label-bearing
    Observation fields. Only the graph and frozen assignments are persisted.
    """
    if not isinstance(observations, (tuple, list)) or not observations:
        raise IntegerRelationEvaluationGraphError("observations are empty")
    if not isinstance(source_refs, (tuple, list)) or not source_refs:
        raise IntegerRelationEvaluationGraphError("source refs are empty")
    if not isinstance(assignments, Mapping) or set(assignments) != set(EVALUATION_SPLITS):
        raise IntegerRelationEvaluationGraphError(
            "held_out, negative, and conflict assignments are required")
    observation_map = {}
    for item in observations:
        if (not isinstance(item, ObservationRecord)
                or item.w_stage != "W-06"
                or item.payload_kind != "TypedRelationQuery"
                or item.split != "held_out"):
            raise IntegerRelationEvaluationGraphError(
                "only W-06 held_out typed observations may compile here")
        if item.stable_key in observation_map:
            raise IntegerRelationEvaluationGraphError(
                "observation identity is duplicated")
        observation_map[item.stable_key] = item
    source_map = {}
    for item in source_refs:
        if not isinstance(item, SourceRefRecord) or item.stable_key in source_map:
            raise IntegerRelationEvaluationGraphError(
                "source ref type or identity is invalid")
        if (item.source_key != W06_AUTHORED_SOURCE
                or item.license_id != "CC0-1.0"
                or item.redistribution_policy != "PUBLIC"):
            raise IntegerRelationEvaluationGraphError(
                "evaluation compiler requires the frozen public CC0 source")
        source_map[item.stable_key] = item
    if any(item.source_ref_key not in source_map
           or source_map[item.source_ref_key].license_id != item.license_partition
           for item in observations):
        raise IntegerRelationEvaluationGraphError(
            "observation source/license provenance is not closed")

    graph_records = {}
    graph_rejections = {}
    for key in sorted(observation_map):
        observation = observation_map[key]
        source = source_map[observation.source_ref_key]
        try:
            candidate = parse_w06_relation_graph_structure(observation, source)
        except (TypeError, ValueError, KeyError) as error:
            raise IntegerRelationEvaluationGraphError(
                f"observation graph structure is not admissible: {key.components}") from error
        if candidate.violations:
            graph_rejections[key.components] = _compile_rejection(
                observation, source, candidate)
        else:
            graph_records[key.components] = _compile_record(
                observation, source, candidate)

    used = set()
    result = {}
    for split in EVALUATION_SPLITS:
        raw_assignments = assignments[split]
        if not isinstance(raw_assignments, (tuple, list)) or not raw_assignments:
            raise IntegerRelationEvaluationGraphError(f"{split} has no assignments")
        cases = []
        for assignment in raw_assignments:
            if (not isinstance(assignment, EvaluationGraphAssignment)
                    or assignment.split != split):
                raise IntegerRelationEvaluationGraphError(
                    f"{split} assignment identity drifted")
            if any(key not in graph_records for key in assignment.observation_keys):
                missing = [key for key in assignment.observation_keys
                           if key not in graph_records and key not in graph_rejections]
            else:
                missing = []
            if missing:
                raise IntegerRelationEvaluationGraphError(
                    "case references a missing held_out observation")
            if used.intersection(assignment.observation_keys):
                raise IntegerRelationEvaluationGraphError(
                    "observation is assigned to multiple evaluation cases")
            used.update(assignment.observation_keys)
            records = tuple(sorted(
                (graph_records[key] for key in assignment.observation_keys
                 if key in graph_records),
                key=lambda item: item.observation_key))
            rejections = tuple(sorted(
                (graph_rejections[key] for key in assignment.observation_keys
                 if key in graph_rejections),
                key=lambda item: item.observation_key))
            cases.append(IntegerRelationEvaluationGraphCase(
                assignment, records, rejections))
        result[split] = IntegerRelationEvaluationGraphSplitCourse(
            split,
            owner_identity,
            source_identity,
            license_identity,
            tuple(sorted(cases, key=lambda item: item.assignment.case_key)),
        )
    if used != set(graph_records) | set(graph_rejections):
        raise IntegerRelationEvaluationGraphError(
            "every held_out observation must have exactly one explicit case assignment")
    return result


__all__ = [
    "EVALUATION_SPLITS",
    "EvaluationGraphAssignment",
    "INTEGER_RELATION_EVALUATION_GRAPH_FORMAT",
    "IntegerRelationEvaluationGraphCase",
    "IntegerRelationEvaluationGraphError",
    "IntegerRelationEvaluationGraphRecord",
    "IntegerRelationEvaluationGraphRejection",
    "IntegerRelationEvaluationGraphSplitCourse",
    "compile_w06_evaluation_graph_splits",
    "read_integer_relation_evaluation_graph_course",
    "write_integer_relation_evaluation_graph_course",
]
