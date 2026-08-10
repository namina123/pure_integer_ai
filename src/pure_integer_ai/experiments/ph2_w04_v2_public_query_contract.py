"""Immutable external contracts for source-bound public W-04 queries."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)


W04_V2_PUBLIC_QUERY_GENERATION_NOT_RUN = "NOT_RUN"
W04_V2_PUBLIC_QUERY_REASONING_NOT_RUN = "NOT_RUN"
W04_V2_PUBLIC_QUERY_STATUSES = {"UNIQUE", "MULTI", "UNKNOWN"}


# object-model: exception
class W04V2PublicQueryError(ValueError):
    """A public W-04 query or its source-bound projection drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W04V2PublicQueryError(f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W04V2PublicQueryError(f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicQuery:
    """Exact external surface/context query without an expected answer."""

    surface: str
    context_text: str | None = None
    allow_generation: int = 1

    def __post_init__(self) -> None:
        _text(self.surface, where="query surface")
        if self.context_text is not None:
            _text(self.context_text, where="query context")
        if self.allow_generation not in {0, 1}:
            raise W04V2PublicQueryError(
                "query allow_generation must be zero or one")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "context_text": self.context_text,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicCandidateProjection:
    """One learned primitive candidate and its redistributable SourceRef."""

    surface: str
    context_text: str
    primitive_registry: str
    primitive_kind: int
    candidate_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    source_key: str
    source_commitment: str
    license_id: str
    active: int
    superseded: int

    def __post_init__(self) -> None:
        for name in ("surface", "context_text", "primitive_registry",
                     "source_key", "license_id"):
            _text(getattr(self, name), where=f"candidate {name}")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W04V2PublicQueryError("candidate primitive kind drifted")
        for name in ("candidate_key", "source_ref_key"):
            _strict_key(getattr(self, name), where=f"candidate {name}")
        if (not isinstance(self.source_commitment, str)
                or len(self.source_commitment) != 64
                or self.active not in {0, 1}
                or self.superseded not in {0, 1}
                or self.active + self.superseded > 1):
            raise W04V2PublicQueryError("candidate projection drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "candidate_key": list(self.candidate_key),
            "context_text": self.context_text,
            "license_id": self.license_id,
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "source_commitment": self.source_commitment,
            "source_key": self.source_key,
            "source_ref_key": list(self.source_ref_key),
            "superseded": self.superseded,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicGenerationProjection:
    """One surface authorized by an active primitive/context projection."""

    surface: str
    primitive_registry: str
    primitive_kind: int
    candidate_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    source_commitment: str

    def __post_init__(self) -> None:
        _text(self.surface, where="generation surface")
        _text(self.primitive_registry, where="generation primitive registry")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W04V2PublicQueryError("generation primitive kind drifted")
        for name in ("candidate_key", "source_ref_key"):
            _strict_key(getattr(self, name), where=f"generation {name}")
        if (not isinstance(self.source_commitment, str)
                or len(self.source_commitment) != 64):
            raise W04V2PublicQueryError(
                "generation source commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_key": list(self.candidate_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "source_commitment": self.source_commitment,
            "source_ref_key": list(self.source_ref_key),
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W04V2PublicQueryResult:
    """Safe external primitive result without labels or prewritten prose."""

    query: W04V2PublicQuery
    status: str
    candidates: tuple[W04V2PublicCandidateProjection, ...]
    selected_primitive_registry: str | None
    selected_primitive_kind: int | None
    clarify_required: int
    reasoning_status: str
    generation_status: str
    generation_options: tuple[W04V2PublicGenerationProjection, ...]
    source_binding_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w04_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.query, W04V2PublicQuery)
                or self.status not in W04_V2_PUBLIC_QUERY_STATUSES
                or not isinstance(self.candidates, tuple)
                or any(not isinstance(item, W04V2PublicCandidateProjection)
                       for item in self.candidates)
                or not isinstance(self.generation_options, tuple)
                or any(not isinstance(item, W04V2PublicGenerationProjection)
                       for item in self.generation_options)):
            raise W04V2PublicQueryError("query result projection drifted")
        selected = (
            self.selected_primitive_registry,
            self.selected_primitive_kind,
        )
        if self.status == "UNIQUE":
            _text(selected[0], where="selected primitive registry")
            if type(selected[1]) is not int or selected[1] <= 0:
                raise W04V2PublicQueryError("UNIQUE query lacks primitive")
        elif selected != (None, None):
            raise W04V2PublicQueryError(
                "non-UNIQUE query selected a primitive")
        if self.clarify_required != int(self.status == "MULTI"):
            raise W04V2PublicQueryError("query clarify flag drifted")
        if (not isinstance(self.reasoning_status, str)
                or not self.reasoning_status
                or not isinstance(self.generation_status, str)
                or not self.generation_status
                or not isinstance(self.source_binding_sha256, str)
                or len(self.source_binding_sha256) != 64
                or (self.experimental, self.formal_mastery_claim,
                    self.w04_started) != (1, 0, 0)):
            raise W04V2PublicQueryError("query result boundary drifted")
        if (self.generation_status == W04_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
                and self.generation_options):
            raise W04V2PublicQueryError(
                "NOT_RUN generation cannot publish options")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "clarify_required": self.clarify_required,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "generation_options": [
                item.to_dict() for item in self.generation_options],
            "generation_status": self.generation_status,
            "query": self.query.to_dict(),
            "reasoning_status": self.reasoning_status,
            "selected_primitive_kind": self.selected_primitive_kind,
            "selected_primitive_registry": self.selected_primitive_registry,
            "source_binding_sha256": self.source_binding_sha256,
            "status": self.status,
            "w04_started": self.w04_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W04_V2_PUBLIC_QUERY_GENERATION_NOT_RUN",
    "W04_V2_PUBLIC_QUERY_REASONING_NOT_RUN",
    "W04_V2_PUBLIC_QUERY_STATUSES",
    "W04V2PublicCandidateProjection",
    "W04V2PublicGenerationProjection",
    "W04V2PublicQuery",
    "W04V2PublicQueryError",
    "W04V2PublicQueryResult",
]
