"""Strict CC0 seed contract for explicit W-03 -> W-04 bridge Evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.operator_primitives import (
    OP_ADD,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_MUL,
    OP_SUB,
)
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES,
    REL_EQUAL,
    REL_MEMBER,
    REL_MEREOLOGY,
    REL_PRECEDES,
    REL_PROPERTY,
    REL_SIMILAR,
    REL_SUBSET,
)
from pure_integer_ai.cognition.shared.symbol_types import (
    TYPE_ATTR_MARKER,
    TYPE_CAUSES,
    TYPE_CMP,
    TYPE_COPULA,
    TYPE_NEGATION,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
    DatasetContractError,
    parse_canonical_json_bytes,
)


BRIDGE_SOURCE_KEY = "AUTHORED_CC0_SEMANTIC_PRIMITIVE_BRIDGE_V1"
BRIDGE_LICENSE_ID = "CC0-1.0"
BRIDGE_PACK_NAME = (
    "AUTHORED_CC0_SEMANTIC_PRIMITIVE_BRIDGE_V1--CC0-1.0--bridge-v1")
BRIDGE_STAGES = ("W-03", "W-04")
BRIDGE_SUBSTAGES = ("SENSE_CONCEPT", "PRIMITIVE_SURFACE_MAPPING")
_ALLOWED_PRIMITIVES = {
    "relation": frozenset({
        REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES,
        REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR,
    }),
    "operator": frozenset({OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE}),
    "symbol_type": frozenset({
        TYPE_NEGATION, TYPE_COPULA, TYPE_CMP, TYPE_CAUSES, TYPE_ATTR_MARKER,
    }),
}
_REQUIRED_ROLES = frozenset({"support", "refute", "conflict", "supersede"})
_FIELDS = frozenset({
    "bridge_id",
    "candidate_sense",
    "context",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "primitive_expected_payload",
    "primitive_expected_state",
    "primitive_kind",
    "primitive_registry",
    "sample_role",
    "sense_expected_payload",
    "sense_expected_state",
    "split",
    "supersedes_bridge_id",
    "surface",
    "template_family",
})


# object-model: exception
class AuthoredSemanticPrimitiveBridgeError(RuntimeError):
    """A bridge row, link, owner split, or primitive coordinate is invalid."""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise AuthoredSemanticPrimitiveBridgeError(
            f"{where} must be canonical text")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise AuthoredSemanticPrimitiveBridgeError(
            f"{where} must be a positive strict integer")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SemanticPrimitiveBridgeSeed:
    """One authored occurrence with separate sense and primitive outcomes."""

    bridge_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    surface: str
    context: str
    candidate_sense: str
    primitive_registry: str
    primitive_kind: int
    sense_expected_state: str
    sense_expected_payload: CanonicalJsonObject
    primitive_expected_state: str
    primitive_expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_bridge_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name in (
                "bridge_id", "family", "template_family", "surface", "context",
                "candidate_sense", "primitive_registry", "perturbation_kind"):
            _text(getattr(self, name), where=f"bridge seed {name}")
        _text(
            self.supersedes_bridge_id,
            where="bridge seed supersedes_bridge_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge label_owner must be teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge label_owner/split drifted")
        if self.sample_role not in _REQUIRED_ROLES:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge sample role is unavailable")
        if (self.sample_role == "supersede") != bool(self.supersedes_bridge_id):
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge supersede role/link drifted")
        if (self.sense_expected_state not in EXPECTED_STATES
                or self.primitive_expected_state not in EXPECTED_STATES):
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge expected state is invalid")
        allowed = _ALLOWED_PRIMITIVES.get(self.primitive_registry)
        if allowed is None or self.primitive_kind not in allowed:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge primitive coordinate is not frozen")
        _positive_int(self.logical_order, where="bridge logical_order")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticPrimitiveBridgeSeed":
        if set(value) != _FIELDS:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge seed field inventory drifted")
        if value["license_id"] != BRIDGE_LICENSE_ID:
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge seed must be CC0-1.0")
        return cls(
            str(value["bridge_id"]),
            str(value["family"]),
            str(value["template_family"]),
            str(value["label_owner"]),
            str(value["split"]),
            str(value["sample_role"]),
            str(value["surface"]),
            str(value["context"]),
            str(value["candidate_sense"]),
            str(value["primitive_registry"]),
            value["primitive_kind"],
            str(value["sense_expected_state"]),
            CanonicalJsonObject.from_value(value["sense_expected_payload"]),
            str(value["primitive_expected_state"]),
            CanonicalJsonObject.from_value(value["primitive_expected_payload"]),
            str(value["perturbation_kind"]),
            str(value["supersedes_bridge_id"]),
            value["logical_order"],
        )


def read_authored_semantic_primitive_bridge_seeds(
        path: str | Path,
        ) -> tuple[SemanticPrimitiveBridgeSeed, ...]:
    """Read canonical JSONL and close owner, family, role, and supersede links."""
    sample = Path(path)
    try:
        payload = sample.read_bytes()
    except OSError as error:
        raise AuthoredSemanticPrimitiveBridgeError(
            "bridge sample cannot be read") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredSemanticPrimitiveBridgeError(
            "bridge sample must be nonempty and newline terminated")
    seeds = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredSemanticPrimitiveBridgeError(
                f"bridge sample line {line_number} is empty or unterminated")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredSemanticPrimitiveBridgeError(
                f"bridge sample line {line_number} is not canonical JSON") from error
        assert isinstance(value, dict)
        seeds.append(SemanticPrimitiveBridgeSeed.from_dict(value))
    if len({item.bridge_id for item in seeds}) != len(seeds):
        raise AuthoredSemanticPrimitiveBridgeError("bridge_id values are not unique")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredSemanticPrimitiveBridgeError(
            "bridge logical_order must increase strictly")
    index = {item.bridge_id: item for item in seeds}
    for item in seeds:
        if not item.supersedes_bridge_id:
            continue
        target = index.get(item.supersedes_bridge_id)
        if (target is None or target.logical_order >= item.logical_order
                or target.family != item.family or target.split != item.split):
            raise AuthoredSemanticPrimitiveBridgeError(
                "bridge supersede target is missing, later, or cross-family")
    teacher_families = {item.family for item in seeds if item.label_owner == "teacher"}
    evaluator_families = {
        item.family for item in seeds if item.label_owner == "evaluator"}
    teacher_templates = {
        item.template_family for item in seeds if item.label_owner == "teacher"}
    evaluator_templates = {
        item.template_family for item in seeds if item.label_owner == "evaluator"}
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates
            or {item.sample_role for item in seeds} != _REQUIRED_ROLES):
        raise AuthoredSemanticPrimitiveBridgeError(
            "bridge owner families/templates/roles are not isolated")
    return tuple(seeds)


__all__ = [
    "BRIDGE_LICENSE_ID",
    "BRIDGE_PACK_NAME",
    "BRIDGE_SOURCE_KEY",
    "BRIDGE_STAGES",
    "BRIDGE_SUBSTAGES",
    "AuthoredSemanticPrimitiveBridgeError",
    "SemanticPrimitiveBridgeSeed",
    "read_authored_semantic_primitive_bridge_seeds",
]
