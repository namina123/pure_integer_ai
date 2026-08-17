"""Pure typed contract for a future multi-proposition GG03 conflict set.

This module is intentionally disconnected from the current ``CONFLICT``
runner.  It defines only the semantic input/projection boundary; generation,
parser, private labels, and formal-family publication are later stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_RESPONSE_ACT = "CONFLICT_SET"


class ConflictSetContractError(ValueError):
    """A conflict-set value violates the public typed contract."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConflictSetContractError(f"{where} must be non-empty text")
    return value


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConflictSetContractError(f"{where} must be a positive integer")
    return value


def _canonical_ids(values: Iterable[str], *, where: str) -> tuple[str, ...]:
    result = tuple(values)
    if (not result or any(not isinstance(item, str) or not item
                          for item in result)
            or len(set(result)) != len(result)):
        raise ConflictSetContractError(f"{where} must contain unique ids")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class ConflictSetEvidence:
    """One source observation for one conflict-set claim."""

    evidence_id: str
    claim_id: str
    source_id: str
    scope_id: int
    support: int
    refute: int

    def __post_init__(self) -> None:
        _text(self.evidence_id, where="evidence_id")
        _text(self.claim_id, where="claim_id")
        _text(self.source_id, where="source_id")
        _positive(self.scope_id, where="scope_id")
        if self.support not in {0, 1} or self.refute not in {0, 1}:
            raise ConflictSetContractError("support/refute must be bits")
        if not self.support and not self.refute:
            raise ConflictSetContractError(
                "each evidence row must carry support or refute")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "refute": self.refute,
            "scope_id": self.scope_id,
            "source_id": self.source_id,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class ConflictSetClaim:
    """Canonical source/stance closure for one conflict claim."""

    claim_id: str
    source_ids: tuple[str, ...]
    support_source_ids: tuple[str, ...]
    refute_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.claim_id, where="claim_id")
        source_ids = _canonical_ids(self.source_ids, where="source_ids")
        support = _canonical_ids(
            self.support_source_ids, where="support_source_ids")
        refute = _canonical_ids(
            self.refute_source_ids, where="refute_source_ids")
        if self.source_ids != source_ids:
            raise ConflictSetContractError(
                "source_ids must already be canonical")
        if self.support_source_ids != support or self.refute_source_ids != refute:
            raise ConflictSetContractError(
                "support/refute source ids must already be canonical")
        if len(source_ids) < 2 or not support or not refute:
            raise ConflictSetContractError(
                "each claim needs at least two sources and both stances")
        if not set(support) | set(refute) == set(source_ids):
            raise ConflictSetContractError(
                "stance source ids must cover every source")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "refute_source_ids": list(self.refute_source_ids),
            "source_ids": list(self.source_ids),
            "support_source_ids": list(self.support_source_ids),
        }


@dataclass(frozen=True, slots=True)
class ConflictSetSemanticProjection:
    """The claim/source meaning expected from a future actual generation."""

    carrier_kind: str
    response_act: str
    scope_id: int
    claim_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.carrier_kind != _RESPONSE_ACT:
            raise ConflictSetContractError("carrier_kind must be CONFLICT_SET")
        if self.response_act != _RESPONSE_ACT:
            raise ConflictSetContractError("response_act must be CONFLICT_SET")
        _positive(self.scope_id, where="projection.scope_id")
        if not self.claim_ids or len(set(self.claim_ids)) != len(self.claim_ids):
            raise ConflictSetContractError("projection claim_ids must be unique")
        for item in self.claim_ids:
            _text(item, where="projection.claim_id")
        canonical = _canonical_ids(
            self.cited_source_ids, where="projection.cited_source_ids")
        if self.cited_source_ids != canonical:
            raise ConflictSetContractError(
                "projection cited_source_ids must be canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "carrier_kind": self.carrier_kind,
            "cited_source_ids": list(self.cited_source_ids),
            "claim_ids": list(self.claim_ids),
            "response_act": self.response_act,
            "scope_id": self.scope_id,
        }


@dataclass(frozen=True, slots=True)
class ConflictSetPlan:
    """Validated multi-proposition conflict set and its expected projection."""

    scope_id: int
    claim_ids: tuple[str, ...]
    claims: tuple[ConflictSetClaim, ...]
    evidence: tuple[ConflictSetEvidence, ...]

    def __post_init__(self) -> None:
        _positive(self.scope_id, where="plan.scope_id")
        if len(self.claim_ids) < 2 or len(set(self.claim_ids)) != len(
                self.claim_ids):
            raise ConflictSetContractError(
                "a conflict set must contain at least two claims")
        for item in self.claim_ids:
            _text(item, where="plan.claim_id")
        if (len(self.claims) != len(self.claim_ids)
                or tuple(item.claim_id for item in self.claims)
                != self.claim_ids):
            raise ConflictSetContractError(
                "claims must exactly follow the declared claim order")
        if not self.evidence:
            raise ConflictSetContractError("conflict set evidence is required")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ConflictSetContractError("evidence ids must be unique")
        if any(item.scope_id != self.scope_id for item in self.evidence):
            raise ConflictSetContractError("evidence scope drift")
        evidence_claim_ids = {item.claim_id for item in self.evidence}
        if evidence_claim_ids != set(self.claim_ids):
            raise ConflictSetContractError(
                "evidence must cover every declared claim")
        for claim in self.claims:
            rows = tuple(
                item for item in self.evidence if item.claim_id == claim.claim_id)
            source_ids = {item.source_id for item in rows}
            support = {item.source_id for item in rows if item.support}
            refute = {item.source_id for item in rows if item.refute}
            if (source_ids != set(claim.source_ids)
                    or support != set(claim.support_source_ids)
                    or refute != set(claim.refute_source_ids)):
                raise ConflictSetContractError(
                    "claim source/stance closure does not match evidence")

    @property
    def response_act(self) -> str:
        return _RESPONSE_ACT

    @property
    def projection(self) -> ConflictSetSemanticProjection:
        return ConflictSetSemanticProjection(
            _RESPONSE_ACT,
            _RESPONSE_ACT,
            self.scope_id,
            self.claim_ids,
            tuple(sorted({
                item.source_id for item in self.evidence})),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_ids": list(self.claim_ids),
            "claims": [item.to_dict() for item in self.claims],
            "evidence": [item.to_dict() for item in self.evidence],
            "response_act": self.response_act,
            "scope_id": self.scope_id,
        }


def build_conflict_set_plan(
        *,
        scope_id: int,
        claim_ids: tuple[str, ...],
        evidence: tuple[ConflictSetEvidence, ...],
        ) -> ConflictSetPlan:
    """Build canonical claim closure from label-free Evidence rows."""
    _positive(scope_id, where="scope_id")
    if not isinstance(claim_ids, tuple) or not isinstance(evidence, tuple):
        raise TypeError("claim_ids and evidence must be tuples")
    if len(claim_ids) < 2 or len(set(claim_ids)) != len(claim_ids):
        raise ConflictSetContractError(
            "claim_ids must contain at least two unique claims")
    for claim_id in claim_ids:
        _text(claim_id, where="claim_id")
    if any(not isinstance(item, ConflictSetEvidence) for item in evidence):
        raise TypeError("evidence contains an invalid row")
    claims = []
    for claim_id in claim_ids:
        rows = tuple(item for item in evidence if item.claim_id == claim_id)
        source_ids = {item.source_id for item in rows}
        support = {item.source_id for item in rows if item.support}
        refute = {item.source_id for item in rows if item.refute}
        claims.append(ConflictSetClaim(
            claim_id,
            tuple(sorted(source_ids)),
            tuple(sorted(support)),
            tuple(sorted(refute)),
        ))
    return ConflictSetPlan(scope_id, claim_ids, tuple(claims), evidence)


__all__ = [
    "ConflictSetClaim",
    "ConflictSetContractError",
    "ConflictSetEvidence",
    "ConflictSetPlan",
    "ConflictSetSemanticProjection",
    "build_conflict_set_plan",
]
