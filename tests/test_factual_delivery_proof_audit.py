from __future__ import annotations

from copy import deepcopy

import pytest

from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.factual_delivery_proof_audit import (
    FactualDeliveryProofEvidenceError,
    verify_factual_delivery_proof,
)


def _fixture() -> tuple[dict, tuple[int, ...]]:
    plan = {
        "response_act": [33], "claim_refs": [[7]],
        "claim_evidence_refs": [{"claim": [7], "evidence": [[90]]}],
        "evidence_refs": [[90]],
        "slot_sequence": [{
            "role": [1], "filler": [7], "required": 1,
            "source_hash": 12, "token_digest": [77],
            "allowed_node_kinds": [4],
        }],
    }
    captured = {
        "status": "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS",
        "delivery_proof_ref": [999], "input_source": [104],
        "input_observation": [103], "expected_claim": [7],
        "trace": {"response_plan": plan},
        "delivery_summary": {
            "claim_count": 1, "evidence_ref_count": 1,
            "adopted_slots": 1,
        },
    }
    ordered = frame_key(5, frame_key(1, (7,)))
    claims = frame_key(1, (7,))
    audit = frame_key(5, frame_key(1, (7,), frame_key(1, (90,))))
    slots = frame_key(2, frame_key(1, (1,), (7,), (1, 12), (77,), (4,)))
    evidence = frame_key(3, (555,))
    factual = frame_key(4, (7,), ordered, claims, audit, slots, evidence)
    proof = frame_key(91530, (4,), (104,), (103,), (33,), factual)
    return captured, proof


def _verify(captured: dict, proof: tuple[int, ...]) -> dict:
    return verify_factual_delivery_proof(
        captured,
        proof_reader=lambda ref: proof if ref == (999,) else (),
        evidence_reader=lambda ref: (90,) if ref == (555,) else (),
    )


def test_factual_proof_members_match_exact_captured_plan() -> None:
    captured, proof = _fixture()
    result = _verify(captured, proof)
    assert result["status"] == "MATCHED_DURABLE_PROOF_CONTENT_TO_CAPTURED_PLAN"
    assert result["claim_count"] == result["slot_count"] == 1
    assert result["ohe_exact_join"].startswith("UNKNOWN")


@pytest.mark.parametrize("path,value,match", [
    (("input_source",), [105], "input source differs"),
    (("input_observation",), [106], "input Observation differs"),
    (("expected_claim",), [8], "target claim differs"),
    (("trace", "response_plan", "response_act"), [34], "ResponseAct differs"),
    (("trace", "response_plan", "evidence_refs"), [[91]], "proof evidence differs"),
    (("trace", "response_plan", "slot_sequence", 0, "source_hash"), 13,
     "proof slot differs"),
])
def test_proof_rejects_changed_source_observation_claim_act_evidence_or_slot(
        path: tuple, value: object, match: str) -> None:
    captured, proof = _fixture()
    current = deepcopy(captured)
    node = current
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    with pytest.raises(FactualDeliveryProofEvidenceError, match=match):
        _verify(current, proof)


def test_missing_durable_proof_is_rejected() -> None:
    captured, _ = _fixture()
    with pytest.raises(FactualDeliveryProofEvidenceError, match="durable delivery proof"):
        _verify(captured, ())


def _multi_fixture():
    captured, _ = _fixture()
    plan = captured["trace"]["response_plan"]
    plan["claim_refs"] = [[7], [8]]
    plan["claim_evidence_refs"].append({"claim": [8], "evidence": [[90]]})
    captured["expected_claim"] = [8]
    captured["delivery_summary"]["claim_count"] = 2
    captured["trace"].update({
        "schema_version": 4, "integer_key_table": [[7], [8], [100]],
        "query_ref": 3, "input_structure": {}, "frontier_trace": {"steps": []},
        "state_key": {"query_ref": 3, "scalars": [0] * 11 + [1] + [0] * 5},
    })
    ordered = frame_key(5, frame_key(1, (7,)), frame_key(2, (8,)))
    claims = frame_key(1, (7,), (8,))
    audit = frame_key(5, *(frame_key(1, (claim,), frame_key(1, (90,))) for claim in (7, 8)))
    slots = frame_key(2, frame_key(1, (1,), (7,), (1, 12), (77,), (4,)))
    factual = frame_key(4, (7,), ordered, claims, audit, slots, frame_key(3, (555,)))
    return captured, frame_key(91530, (4,), (104,), (103,), (33,), factual)


def test_multi_claim_target_is_distinct_from_saved_primary_and_both_are_verified():
    captured, proof = _multi_fixture()
    result = _verify(captured, proof)
    assert result["claim_count"] == 2 and result["primary_claim"] == [7]


@pytest.mark.parametrize("changed,match", [
    ("primary", "primary differs"), ("target", "outside the plan"),
    ("missing_state", "lacks saved QueryState"),
])
def test_multi_claim_primary_or_target_cannot_be_forged(changed, match):
    captured, proof = _multi_fixture()
    if changed == "primary":
        captured["trace"]["state_key"]["scalars"][11] = 2
    elif changed == "target":
        captured["expected_claim"] = [9]
    else:
        del captured["trace"]["state_key"]
    with pytest.raises(FactualDeliveryProofEvidenceError, match=match):
        _verify(captured, proof)
