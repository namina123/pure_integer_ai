"""Pure contract tests for the future GG03 CONFLICT_SET response act."""
import pytest

from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetContractError,
    ConflictSetEvidence,
    build_conflict_set_plan,
)


def _evidence() -> tuple[ConflictSetEvidence, ...]:
    return (
        ConflictSetEvidence("e1", "claim-a", "source-b", 901, 1, 0),
        ConflictSetEvidence("e2", "claim-a", "source-a", 901, 0, 1),
        ConflictSetEvidence("e3", "claim-b", "source-c", 901, 1, 1),
        ConflictSetEvidence("e4", "claim-b", "source-d", 901, 0, 1),
    )


def test_conflict_set_plan_closes_all_claims_sources_and_projection() -> None:
    plan = build_conflict_set_plan(
        scope_id=901,
        claim_ids=("claim-b", "claim-a"),
        evidence=_evidence(),
    )
    assert plan.response_act == "CONFLICT_SET"
    assert tuple(item.claim_id for item in plan.claims) == (
        "claim-b", "claim-a")
    assert plan.projection.to_dict() == {
        "carrier_kind": "CONFLICT_SET",
        "cited_source_ids": ["source-a", "source-b", "source-c", "source-d"],
        "claim_ids": ["claim-b", "claim-a"],
        "response_act": "CONFLICT_SET",
        "scope_id": 901,
    }


def test_conflict_set_rejects_partial_claim_and_single_source() -> None:
    with pytest.raises(ConflictSetContractError):
        build_conflict_set_plan(
            scope_id=901,
            claim_ids=("claim-a", "claim-b"),
            evidence=(
                *_evidence()[:3],
                ConflictSetEvidence("e4", "claim-b", "source-c", 901, 1, 0),
            ),
        )
    with pytest.raises(ConflictSetContractError):
        build_conflict_set_plan(
            scope_id=901,
            claim_ids=("claim-a", "claim-b"),
            evidence=(
                *_evidence()[:2],
                ConflictSetEvidence("e3", "claim-b", "source-c", 901, 1, 0),
                ConflictSetEvidence("e4", "claim-b", "source-c", 901, 0, 1),
            ),
        )


def test_conflict_set_rejects_scope_drift_and_uncovered_claim() -> None:
    with pytest.raises(ConflictSetContractError):
        build_conflict_set_plan(
            scope_id=901,
            claim_ids=("claim-a", "claim-b"),
            evidence=tuple(
                _evidence()[:3]
            ) + (ConflictSetEvidence("e4", "claim-b", "source-d", 902, 0, 1),),
        )
    with pytest.raises(ConflictSetContractError):
        build_conflict_set_plan(
            scope_id=901,
            claim_ids=("claim-a", "claim-b", "claim-c"),
            evidence=_evidence(),
        )
