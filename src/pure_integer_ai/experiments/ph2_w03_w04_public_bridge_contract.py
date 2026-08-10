"""Immutable contracts for explicit public W-03 -> W-04 bridge queries."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_query import (
    W03V2PublicQueryResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQueryResult,
)


W03_W04_PUBLIC_BRIDGE_STATUSES = {"BRIDGED", "UNKNOWN", "CLARIFY"}


# object-model: exception
class W03W04PublicBridgeError(ValueError):
    """A bridge request, stage result, or explicit prerequisite link drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04PublicBridgeError(f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04PublicBridgeError(f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04PublicBridgeQuery:
    """Exact surface occurrence to bridge without expected output fields."""

    surface: str
    context_text: str
    language: str = "zh"
    allow_generation: int = 1

    def __post_init__(self) -> None:
        for name in ("surface", "context_text", "language"):
            _text(getattr(self, name), where=f"bridge query {name}")
        if self.allow_generation not in {0, 1}:
            raise W03W04PublicBridgeError(
                "bridge allow_generation must be zero or one")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "context_text": self.context_text,
            "language": self.language,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04PublicBridgeLink:
    """Explicit observation prerequisite joining Sense and primitive records."""

    source_ref_key: tuple[int, ...]
    source_commitment: str
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    primitive_registry: str
    primitive_kind: int

    def __post_init__(self) -> None:
        for name in (
                "source_ref_key", "w03_observation_key", "w04_observation_key",
                "sense_key", "concept_key"):
            _strict_key(getattr(self, name), where=f"bridge link {name}")
        _text(self.primitive_registry, where="bridge primitive registry")
        if (not isinstance(self.source_commitment, str)
                or len(self.source_commitment) != 64
                or type(self.primitive_kind) is not int
                or self.primitive_kind <= 0):
            raise W03W04PublicBridgeError("bridge link identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": list(self.concept_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "sense_key": list(self.sense_key),
            "source_commitment": self.source_commitment,
            "source_ref_key": list(self.source_ref_key),
            "w03_observation_key": list(self.w03_observation_key),
            "w04_observation_key": list(self.w04_observation_key),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04PublicBridgeResult:
    """Both stage results plus an optional prerequisite-authorized typed link."""

    query: W03W04PublicBridgeQuery
    status: str
    w03_result: W03V2PublicQueryResult
    w04_result: W04V2PublicQueryResult
    link: W03W04PublicBridgeLink | None
    w03_source_binding_sha256: str
    w04_source_binding_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W03W04PublicBridgeQuery)
                or self.status not in W03_W04_PUBLIC_BRIDGE_STATUSES
                or not isinstance(self.w03_result, W03V2PublicQueryResult)
                or not isinstance(self.w04_result, W04V2PublicQueryResult)):
            raise W03W04PublicBridgeError("bridge result projection drifted")
        if self.status == "BRIDGED":
            if (not isinstance(self.link, W03W04PublicBridgeLink)
                    or self.w03_result.status != "UNIQUE"
                    or self.w04_result.status != "UNIQUE"):
                raise W03W04PublicBridgeError(
                    "BRIDGED result lacks two unique stages and link")
        elif self.link is not None:
            raise W03W04PublicBridgeError(
                "non-BRIDGED result cannot publish a link")
        if (not isinstance(self.w03_source_binding_sha256, str)
                or len(self.w03_source_binding_sha256) != 64
                or not isinstance(self.w04_source_binding_sha256, str)
                or len(self.w04_source_binding_sha256) != 64
                or (self.experimental, self.formal_mastery_claim,
                    self.w03_started, self.w04_started) != (1, 0, 0, 0)):
            raise W03W04PublicBridgeError("bridge result boundary drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "link": None if self.link is None else self.link.to_dict(),
            "query": self.query.to_dict(),
            "status": self.status,
            "w03_result": self.w03_result.to_dict(),
            "w03_source_binding_sha256": self.w03_source_binding_sha256,
            "w03_started": self.w03_started,
            "w04_result": self.w04_result.to_dict(),
            "w04_source_binding_sha256": self.w04_source_binding_sha256,
            "w04_started": self.w04_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W03_W04_PUBLIC_BRIDGE_STATUSES",
    "W03W04PublicBridgeError",
    "W03W04PublicBridgeLink",
    "W03W04PublicBridgeQuery",
    "W03W04PublicBridgeResult",
]
