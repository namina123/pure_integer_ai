"""Immutable contracts for explicit public W-03 -> W-04 -> W-05 chains."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge_contract import (
    W03W04PublicBridgeResult,
)
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge_contract import (
    W04W05PublicBridgeResult,
)


W03_W04_W05_VERTICAL_STATUSES = {"BRIDGED", "UNKNOWN", "CLARIFY"}


# object-model: exception
class W03W04W05VerticalError(ValueError):
    """A three-stage query, result, or exact observation chain drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05VerticalError(f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05VerticalError(f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03W04W05VerticalError(f"{where} is not SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalQuery:
    """One exact sense, primitive, and proposition query without an answer."""

    surface: str
    context_text: str
    proposition_surface: str
    language: str = "zh"
    allow_generation: int = 1

    def __post_init__(self) -> None:
        for name in ("surface", "context_text", "proposition_surface", "language"):
            _text(getattr(self, name), where=f"vertical query {name}")
        if self.allow_generation not in {0, 1}:
            raise W03W04W05VerticalError(
                "vertical allow_generation must be zero or one")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "context_text": self.context_text,
            "language": self.language,
            "proposition_surface": self.proposition_surface,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalLink:
    """The one source-bound chain authorized by both prerequisite edges."""

    source_ref_key: tuple[int, ...]
    source_commitment: str
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    primitive_registry: str
    primitive_kind: int
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    predicate_occurrence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
                "source_ref_key", "w03_observation_key",
                "w04_observation_key", "w05_observation_key", "sense_key",
                "concept_key", "proposition_key", "predicate_key",
                "predicate_occurrence_key"):
            _strict_key(getattr(self, name), where=f"vertical link {name}")
        _sha256(self.source_commitment, where="vertical source commitment")
        _text(self.primitive_registry, where="vertical primitive registry")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W03W04W05VerticalError("vertical primitive kind drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": list(self.concept_key),
            "predicate_key": list(self.predicate_key),
            "predicate_occurrence_key": list(self.predicate_occurrence_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "proposition_key": list(self.proposition_key),
            "sense_key": list(self.sense_key),
            "source_commitment": self.source_commitment,
            "source_ref_key": list(self.source_ref_key),
            "w03_observation_key": list(self.w03_observation_key),
            "w04_observation_key": list(self.w04_observation_key),
            "w05_observation_key": list(self.w05_observation_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalResult:
    """Two adjacent bridge results and their optional exact shared link."""

    query: W03W04W05VerticalQuery
    status: str
    w03_w04: W03W04PublicBridgeResult
    w04_w05: W04W05PublicBridgeResult
    link: W03W04W05VerticalLink | None
    overlay_validation_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W03W04W05VerticalQuery)
                or self.status not in W03_W04_W05_VERTICAL_STATUSES
                or not isinstance(self.w03_w04, W03W04PublicBridgeResult)
                or not isinstance(self.w04_w05, W04W05PublicBridgeResult)):
            raise W03W04W05VerticalError("vertical result projection drifted")
        if self.status == "BRIDGED":
            if (not isinstance(self.link, W03W04W05VerticalLink)
                    or self.w03_w04.status != "BRIDGED"
                    or self.w04_w05.status != "BRIDGED"):
                raise W03W04W05VerticalError(
                    "BRIDGED vertical result lacks both adjacent links")
        elif self.link is not None:
            raise W03W04W05VerticalError(
                "non-BRIDGED vertical result cannot publish a link")
        _sha256(
            self.overlay_validation_sha256,
            where="vertical overlay validation",
        )
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05VerticalError("vertical boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "link": None if self.link is None else self.link.to_dict(),
            "overlay_validation_sha256": self.overlay_validation_sha256,
            "query": self.query.to_dict(),
            "status": self.status,
            "w03_started": self.w03_started,
            "w03_w04": self.w03_w04.to_dict(),
            "w04_started": self.w04_started,
            "w04_w05": self.w04_w05.to_dict(),
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W03_W04_W05_VERTICAL_STATUSES",
    "W03W04W05VerticalError",
    "W03W04W05VerticalLink",
    "W03W04W05VerticalQuery",
    "W03W04W05VerticalResult",
]
