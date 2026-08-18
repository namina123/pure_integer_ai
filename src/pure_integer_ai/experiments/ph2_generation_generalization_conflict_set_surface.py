"""Typed multi-sentence surface generation/parser for ``CONFLICT_SET``.

This is a public dry-run layer over the pure contract.  It does not replace
the existing GG03 runner and intentionally carries no evaluator label or
private surface set.  The later connector integration can consume the same
sentence/attribution contract.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    CONFLICT_SET_FAIL,
    CONFLICT_SET_NE,
    CONFLICT_SET_PASS,
    ConflictSetPlan,
    ConflictSetSemanticProjection,
    evaluate_conflict_set_projection,
)


class ConflictSetSurfaceError(ValueError):
    """Typed surface generation or parsing violated the public contract."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConflictSetSurfaceError(f"{where} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ConflictSetGeneratedSentence:
    """One generated sentence with explicit claim/source/state attribution."""

    ordinal: int
    claim_id: str
    scope_id: int
    source_ids: tuple[str, ...]
    support: int
    refute: int
    surface: str
    units: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ConflictSetSurfaceError("sentence ordinal must be positive")
        _text(self.claim_id, where="sentence.claim_id")
        if type(self.scope_id) is not int or self.scope_id <= 0:
            raise ConflictSetSurfaceError("sentence scope must be positive")
        if (not isinstance(self.source_ids, tuple)
                or not self.source_ids
                or self.source_ids != tuple(sorted(set(self.source_ids)))):
            raise ConflictSetSurfaceError(
                "sentence source ids must be canonical and non-empty")
        for source_id in self.source_ids:
            _text(source_id, where="sentence.source_id")
        if (type(self.support) is not int or self.support not in {0, 1}
                or type(self.refute) is not int or self.refute not in {0, 1}):
            raise ConflictSetSurfaceError(
                "sentence support/refute must be bits")
        _text(self.surface, where="sentence.surface")
        if (not isinstance(self.units, tuple) or not self.units
                or any(type(item) is not int or item < 0 for item in self.units)):
            raise ConflictSetSurfaceError("sentence units must be non-empty ints")
        if self.units != tuple(ord(item) for item in self.surface):
            raise ConflictSetSurfaceError("sentence units/surface drift")


@dataclass(frozen=True, slots=True)
class ConflictSetSurfaceParseResult:
    """Parser output and semantic classification against one expected plan."""

    status: str
    projection: ConflictSetSemanticProjection | None
    sentence_count: int

    def __post_init__(self) -> None:
        if self.status not in {CONFLICT_SET_PASS, CONFLICT_SET_FAIL,
                               CONFLICT_SET_NE}:
            raise ConflictSetSurfaceError("parse status is not registered")
        if type(self.sentence_count) is not int or self.sentence_count < 0:
            raise ConflictSetSurfaceError("sentence count must be non-negative")
        if self.status == CONFLICT_SET_NE and self.projection is not None:
            raise ConflictSetSurfaceError("NE parse must not expose projection")
        if self.status != CONFLICT_SET_NE and self.projection is None:
            raise ConflictSetSurfaceError(
                "PASS/FAIL parse must expose projection")


def generate_conflict_set_sentences(
        plan: ConflictSetPlan,
        surfaces: tuple[str, ...],
        ) -> tuple[ConflictSetGeneratedSentence, ...]:
    """Generate one deterministic sentence per ordered conflict claim."""
    if not isinstance(plan, ConflictSetPlan):
        raise TypeError("plan must be ConflictSetPlan")
    if (not isinstance(surfaces, tuple)
            or len(surfaces) != len(plan.claim_ids)):
        raise ConflictSetSurfaceError(
            "surface count must equal ordered conflict claim count")
    return tuple(
        ConflictSetGeneratedSentence(
            index,
            claim.claim_id,
            plan.scope_id,
            claim.source_ids,
            1,
            1,
            _text(surface, where="surface"),
            tuple(ord(item) for item in surface),
        )
        for index, (claim, surface)
        in enumerate(zip(plan.claims, surfaces, strict=True), start=1)
    )


def parse_conflict_set_sentences(
        plan: ConflictSetPlan,
        sentences: tuple[ConflictSetGeneratedSentence, ...],
        ) -> ConflictSetSurfaceParseResult:
    """Recover claim/source meaning from typed sentence metadata and units."""
    if not isinstance(plan, ConflictSetPlan):
        raise TypeError("plan must be ConflictSetPlan")
    if not isinstance(sentences, tuple):
        raise TypeError("sentences must be a tuple")
    count = len(sentences)
    if not sentences or any(not isinstance(item, ConflictSetGeneratedSentence)
                            for item in sentences):
        return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, count)
    if tuple(item.ordinal for item in sentences) != tuple(
            range(1, count + 1)):
        return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, count)
    if any(item.scope_id != plan.scope_id for item in sentences):
        return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, count)
    if count != len(plan.claim_ids):
        return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, count)
    claim_ids = tuple(item.claim_id for item in sentences)
    source_ids = tuple(sorted({
        source_id for item in sentences for source_id in item.source_ids
    }))
    claim_states = tuple(
        (item.claim_id, item.support, item.refute) for item in sentences)
    claim_source_ids = tuple(
        (item.claim_id, item.source_ids) for item in sentences)
    actual = ConflictSetSemanticProjection(
        "CONFLICT_SET", "CONFLICT_SET", plan.scope_id, claim_ids,
        claim_states, claim_source_ids, source_ids)
    expected = plan.projection
    status = evaluate_conflict_set_projection(expected, actual)
    return ConflictSetSurfaceParseResult(status, actual, count)


def classify_conflict_set_surface(
        plan: ConflictSetPlan,
        sentences: tuple[ConflictSetGeneratedSentence, ...] | None,
        ) -> ConflictSetSurfaceParseResult:
    """Treat unavailable actual generation as explicit capability NE."""
    if sentences is None:
        return ConflictSetSurfaceParseResult(CONFLICT_SET_NE, None, 0)
    return parse_conflict_set_sentences(plan, sentences)


__all__ = [
    "ConflictSetGeneratedSentence",
    "ConflictSetSurfaceError",
    "ConflictSetSurfaceParseResult",
    "classify_conflict_set_surface",
    "generate_conflict_set_sentences",
    "parse_conflict_set_sentences",
]
