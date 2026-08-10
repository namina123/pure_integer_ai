"""FT12 learned predicate-alias routes and raw-question result contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RAW_QUESTION_STATUSES,
    RawQuestionAnswerResult,
    RawQuestionConstruction,
    RawQuestionRequest,
)


PREDICATE_ALIAS_RESOLUTION_STATUSES = {
    "AMBIGUOUS", "MISSING", "SELECTED"}


# object-model: exception
class W03W04W05RawQuestionAliasError(ValueError):
    """A learned lexical route or alias-aware answer escaped its evidence."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05RawQuestionAliasError(
            f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05RawQuestionAliasError(
            f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03W04W05RawQuestionAliasError(f"{where} is not SHA-256")
    return value


def _predicate_segment(
        construction: RawQuestionConstruction,
        ):
    values = tuple(
        item for item in construction.segments if item.kind == "PREDICATE")
    if len(values) != 1:
        raise W03W04W05RawQuestionAliasError(
            "question construction predicate inventory drifted")
    return values[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedPredicateAliasRoute:
    """One public W03/W04 replacement chain bound to one Proposition occurrence."""

    alias_surface: str
    predicate_surface: str
    sense_label: str
    primitive_registry: str
    primitive_kind: int
    alias_context: str
    predicate_context: str
    alias_source_ref_key: tuple[int, ...]
    predicate_source_ref_key: tuple[int, ...]
    alias_source_commitment: str
    predicate_source_commitment: str
    alias_w03_observation_key: tuple[int, ...]
    alias_w04_observation_key: tuple[int, ...]
    predicate_w03_observation_key: tuple[int, ...]
    predicate_w04_observation_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    proposition_key: tuple[int, ...]
    predicate_occurrence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
                "alias_surface", "predicate_surface", "sense_label",
                "primitive_registry", "alias_context", "predicate_context"):
            _text(getattr(self, name), where=f"predicate alias route {name}")
        if self.alias_surface == self.predicate_surface:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias route did not replace a lexical surface")
        if (self.alias_context.count(self.alias_surface) != 1
                or self.predicate_context.count(self.predicate_surface) != 1
                or self.alias_context.replace(
                    self.alias_surface, self.predicate_surface, 1)
                != self.predicate_context):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias contexts are not one exact replacement")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias primitive kind drifted")
        for name in (
                "alias_source_ref_key", "predicate_source_ref_key",
                "alias_w03_observation_key", "alias_w04_observation_key",
                "predicate_w03_observation_key",
                "predicate_w04_observation_key", "proposition_key",
                "predicate_occurrence_key"):
            _strict_key(getattr(self, name), where=f"predicate alias {name}")
        if self.alias_source_ref_key == self.predicate_source_ref_key:
            raise W03W04W05RawQuestionAliasError(
                "replacement route must retain both source records")
        for name in (
                "alias_source_commitment", "predicate_source_commitment"):
            _sha256(getattr(self, name), where=f"predicate alias {name}")
        if (not isinstance(self.evidence_keys, tuple)
                or len(self.evidence_keys) != 4):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias route must retain four public evidence keys")
        for key in self.evidence_keys:
            _strict_key(key, where="predicate alias evidence")
        if (self.evidence_keys != tuple(sorted(self.evidence_keys))
                or len(set(self.evidence_keys)) != len(self.evidence_keys)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias evidence keys are not canonical")

    def semantic_key(self) -> tuple[str, str, int, str]:
        """Return the sense, primitive, and current predicate target."""
        return (
            self.sense_label,
            self.primitive_registry,
            self.primitive_kind,
            self.predicate_surface,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_context": self.alias_context,
            "alias_source_commitment": self.alias_source_commitment,
            "alias_source_ref_key": list(self.alias_source_ref_key),
            "alias_surface": self.alias_surface,
            "alias_w03_observation_key": list(
                self.alias_w03_observation_key),
            "alias_w04_observation_key": list(
                self.alias_w04_observation_key),
            "evidence_keys": [list(item) for item in self.evidence_keys],
            "predicate_context": self.predicate_context,
            "predicate_occurrence_key": list(self.predicate_occurrence_key),
            "predicate_source_commitment": self.predicate_source_commitment,
            "predicate_source_ref_key": list(self.predicate_source_ref_key),
            "predicate_surface": self.predicate_surface,
            "predicate_w03_observation_key": list(
                self.predicate_w03_observation_key),
            "predicate_w04_observation_key": list(
                self.predicate_w04_observation_key),
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "proposition_key": list(self.proposition_key),
            "sense_label": self.sense_label,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedPredicateAliasOption:
    """Independent public routes that agree on one semantic predicate target."""

    alias_surface: str
    predicate_surface: str
    sense_label: str
    primitive_registry: str
    primitive_kind: int
    routes: tuple[LearnedPredicateAliasRoute, ...]

    def __post_init__(self) -> None:
        for name in (
                "alias_surface", "predicate_surface", "sense_label",
                "primitive_registry"):
            _text(getattr(self, name), where=f"predicate alias option {name}")
        if type(self.primitive_kind) is not int or self.primitive_kind <= 0:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias option primitive kind drifted")
        if (not isinstance(self.routes, tuple) or len(self.routes) < 2
                or any(not isinstance(item, LearnedPredicateAliasRoute)
                       for item in self.routes)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias option needs two independent routes")
        if self.routes != tuple(sorted(
                self.routes, key=LearnedPredicateAliasRoute.sha256)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias option routes are not canonical")
        if len({item.sha256() for item in self.routes}) != len(self.routes):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias option routes are duplicated")
        expected = (
            self.sense_label,
            self.primitive_registry,
            self.primitive_kind,
            self.predicate_surface,
        )
        if (any(item.alias_surface != self.alias_surface
                or item.semantic_key() != expected for item in self.routes)
                or len({item.proposition_key for item in self.routes}) < 2
                or len({item.predicate_source_ref_key
                        for item in self.routes}) < 2):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias option lacks independent agreement")

    def semantic_key(self) -> tuple[str, str, int, str]:
        return (
            self.sense_label,
            self.primitive_registry,
            self.primitive_kind,
            self.predicate_surface,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_surface": self.alias_surface,
            "predicate_surface": self.predicate_surface,
            "primitive_kind": self.primitive_kind,
            "primitive_registry": self.primitive_registry,
            "routes": [item.to_dict() for item in self.routes],
            "sense_label": self.sense_label,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedPredicateAliasResolution:
    """Missing, unique semantic option, or non-equivalent option ambiguity."""

    alias_surface: str
    status: str
    options: tuple[LearnedPredicateAliasOption, ...]

    def __post_init__(self) -> None:
        _text(self.alias_surface, where="predicate alias resolution surface")
        if (self.status not in PREDICATE_ALIAS_RESOLUTION_STATUSES
                or not isinstance(self.options, tuple)
                or any(not isinstance(item, LearnedPredicateAliasOption)
                       for item in self.options)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias resolution projection drifted")
        if self.options != tuple(sorted(
                self.options, key=LearnedPredicateAliasOption.sha256)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias options are not canonical")
        if (any(item.alias_surface != self.alias_surface
                for item in self.options)
                or len({item.semantic_key() for item in self.options})
                != len(self.options)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias options escaped their query")
        expected_status = (
            "MISSING" if not self.options
            else "SELECTED" if len(self.options) == 1
            else "AMBIGUOUS"
        )
        if self.status != expected_status:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias status does not match option count")

    @property
    def selected(self) -> LearnedPredicateAliasOption | None:
        return self.options[0] if self.status == "SELECTED" else None

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_surface": self.alias_surface,
            "options": [item.to_dict() for item in self.options],
            "status": self.status,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedPredicateAliasBridge:
    """Public W03/W04 evidence routes available to the FT12 read-only query."""

    overlay_validation_sha256: str
    raw_question_bundle_sha256: str
    w03_source_binding_sha256: str
    w04_source_binding_sha256: str
    routes: tuple[LearnedPredicateAliasRoute, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
                "overlay_validation_sha256", "raw_question_bundle_sha256",
                "w03_source_binding_sha256", "w04_source_binding_sha256",
                "identity_sha256"):
            _sha256(getattr(self, name), where=f"predicate alias bridge {name}")
        if (not isinstance(self.routes, tuple) or len(self.routes) < 2
                or any(not isinstance(item, LearnedPredicateAliasRoute)
                       for item in self.routes)
                or self.routes != tuple(sorted(
                    self.routes, key=LearnedPredicateAliasRoute.sha256))
                or len({item.sha256() for item in self.routes})
                != len(self.routes)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias bridge routes are not canonical")
        grouped: dict[tuple[str, tuple[str, str, int, str]], list] = {}
        for route in self.routes:
            grouped.setdefault(
                (route.alias_surface, route.semantic_key()), []).append(route)
        for (alias_surface, semantic), routes in grouped.items():
            LearnedPredicateAliasOption(
                alias_surface,
                semantic[3],
                semantic[0],
                semantic[1],
                semantic[2],
                tuple(routes),
            )
        if self.identity_sha256 != self.sha256():
            raise W03W04W05RawQuestionAliasError(
                "predicate alias bridge identity drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "overlay_validation_sha256": self.overlay_validation_sha256,
            "raw_question_bundle_sha256": self.raw_question_bundle_sha256,
            "routes": [item.to_dict() for item in self.routes],
            "w03_source_binding_sha256": self.w03_source_binding_sha256,
            "w04_source_binding_sha256": self.w04_source_binding_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionPredicateAliasMatch:
    """One structural question match and all learned lexical route outcomes."""

    construction: RawQuestionConstruction
    alias_surface: str
    resolution: LearnedPredicateAliasResolution
    aligned_route_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.construction, RawQuestionConstruction)
                or not isinstance(
                    self.resolution, LearnedPredicateAliasResolution)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias match projection drifted")
        _text(self.alias_surface, where="predicate alias match surface")
        if self.resolution.alias_surface != self.alias_surface:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias match resolution escaped the surface")
        for item in self.aligned_route_sha256s:
            _sha256(item, where="aligned predicate alias route")
        if (self.aligned_route_sha256s
                != tuple(sorted(self.aligned_route_sha256s))
                or len(set(self.aligned_route_sha256s))
                != len(self.aligned_route_sha256s)):
            raise W03W04W05RawQuestionAliasError(
                "aligned predicate alias routes are not canonical")
        vertical = self.construction.vertical_result
        if vertical.link is None:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias match lacks a vertical link")
        predicate = _predicate_segment(self.construction)
        expected = tuple(sorted(
            route.sha256()
            for option in self.resolution.options
            for route in option.routes
            if (option.predicate_surface == predicate.surface
                and (option.primitive_registry, option.primitive_kind)
                == (self.construction.pattern.primitive_registry,
                    self.construction.pattern.primitive_kind)
                and route.proposition_key == vertical.link.proposition_key
                and route.predicate_occurrence_key
                == vertical.link.predicate_occurrence_key)
        ))
        if self.aligned_route_sha256s != expected:
            raise W03W04W05RawQuestionAliasError(
                "predicate alias match did not align the Proposition occurrence")

    @property
    def selected(self) -> bool:
        return (
            self.resolution.status == "SELECTED"
            and bool(self.aligned_route_sha256s)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_surface": self.alias_surface,
            "aligned_route_sha256s": list(self.aligned_route_sha256s),
            "construction": self.construction.to_dict(),
            "resolution": self.resolution.to_dict(),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionPredicateAliasAnswerResult:
    """FT11 exact result plus an optional FT12 learned predicate rewrite."""

    request: RawQuestionRequest
    status: str
    answer_surface: str | None
    exact_result: RawQuestionAnswerResult
    matches: tuple[RawQuestionPredicateAliasMatch, ...]
    selected_match: RawQuestionPredicateAliasMatch | None
    normalized_result: RawQuestionAnswerResult | None
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.request, RawQuestionRequest)
                or self.status not in RAW_QUESTION_STATUSES
                or not isinstance(self.exact_result, RawQuestionAnswerResult)
                or self.exact_result.request != self.request
                or not isinstance(self.matches, tuple)
                or any(not isinstance(item, RawQuestionPredicateAliasMatch)
                       for item in self.matches)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias answer projection drifted")
        if (self.matches != tuple(sorted(
                self.matches, key=RawQuestionPredicateAliasMatch.sha256))
                or len({item.sha256() for item in self.matches})
                != len(self.matches)):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias matches are not canonical")
        if self.exact_result.status != "UNKNOWN":
            if (self.matches or self.selected_match is not None
                    or self.normalized_result is not None
                    or self.status != self.exact_result.status
                    or self.answer_surface != self.exact_result.answer_surface):
                raise W03W04W05RawQuestionAliasError(
                    "exact raw question result was not preserved")
        elif self.selected_match is None:
            if (self.normalized_result is not None
                    or self.answer_surface is not None
                    or self.status not in {"CLARIFY", "UNKNOWN"}):
                raise W03W04W05RawQuestionAliasError(
                    "unselected predicate alias published an answer")
        else:
            if (self.selected_match not in self.matches
                    or not self.selected_match.selected
                    or not isinstance(
                        self.normalized_result, RawQuestionAnswerResult)):
                raise W03W04W05RawQuestionAliasError(
                    "selected predicate alias lacks normalized FT11 execution")
            construction = self.selected_match.construction
            normalized = self.normalized_result
            if (normalized.request.question_surface
                    != construction.question_surface
                    or normalized.request.source_record_key
                    != self.request.source_record_key
                    or self.status != normalized.status
                    or self.answer_surface != normalized.answer_surface):
                raise W03W04W05RawQuestionAliasError(
                    "predicate alias answer escaped normalized FT11")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05RawQuestionAliasError(
                "predicate alias boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "exact_result": self.exact_result.to_dict(),
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "matches": [item.to_dict() for item in self.matches],
            "normalized_result": (
                None if self.normalized_result is None
                else self.normalized_result.to_dict()
            ),
            "request": self.request.to_dict(),
            "selected_match": (
                None if self.selected_match is None
                else self.selected_match.to_dict()
            ),
            "status": self.status,
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "LearnedPredicateAliasBridge",
    "LearnedPredicateAliasOption",
    "LearnedPredicateAliasResolution",
    "LearnedPredicateAliasRoute",
    "PREDICATE_ALIAS_RESOLUTION_STATUSES",
    "RawQuestionPredicateAliasAnswerResult",
    "RawQuestionPredicateAliasMatch",
    "W03W04W05RawQuestionAliasError",
]
