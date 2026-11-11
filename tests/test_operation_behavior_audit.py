from __future__ import annotations

import pytest

from pure_integer_ai.experiments.operation_behavior_audit import (
    OperationBehaviorEvidenceError,
    audit_operation_counterfactual,
)


def _trace(*, operation_keys: list[list[int]], termination: int = 1,
           plan: dict | None = None, surface: str = "X") -> dict:
    return {
        "schema_version": 4,
        "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
        "integer_key_table": [[1], [2], [3], [4]],
        "context_key": [900],
        "operation_keys": operation_keys,
        "operation_contracts": [[key, [91672, 3]] for key in operation_keys],
        "termination": termination,
        "hops": [
            {"space": 2, "proposition_ref": 1, "predicate_ref": 2,
             "source_hash": 10},
            {"space": 3, "proposition_ref": 3, "predicate_ref": 4,
             "source_hash": 11},
        ],
        "memory_route_refs": [1],
        "expanded_edge_refs": [],
        "response_plan": plan,
        "generation": ({"surface_values": [88]} if plan else None),
        "response_surface": surface,
    }


def _plan(act: list[int] = [33]) -> dict:
    return {
        "response_act": act,
        "stable_key": [44],
        "claim_refs": [[7]],
        "claim_evidence_refs": [{"claim": [7], "evidence": [[90]]}],
        "claim_slot_ranges": [],
        "slot_sequence": [{"role": [1], "filler": [7], "required": 1}],
    }


def test_removed_operation_that_changes_to_fail_closed_passes() -> None:
    baseline = _trace(operation_keys=[[22]], plan=_plan())
    removed = _trace(operation_keys=[], termination=4, plan=None, surface="")
    result = audit_operation_counterfactual(
        baseline, removed, context_key=(900,), mode="REMOVED")
    assert result.status == "OBSERVED_PAIR_DIFFERENCE_CAUSAL_PROVENANCE_UNKNOWN"
    assert result.fail_closed is True
    assert "claims" in result.changed_fields
    assert result.operation_family_qualification.startswith("UNKNOWN")


def test_swapping_key_without_behavior_change_is_blocked() -> None:
    baseline = _trace(operation_keys=[[22]], plan=_plan())
    swapped = _trace(operation_keys=[[23]], plan=_plan())
    result = audit_operation_counterfactual(
        baseline, swapped, context_key=(900,), mode="SWAPPED",
        swapped_operation_key=(23,))
    assert result.status == "BLOCKED_OPERATION_BEHAVIOR_NO_DIFFERENCE"
    assert result.changed_fields == ()


def test_swapping_operation_with_only_response_act_change_stays_blocked() -> None:
    baseline = _trace(operation_keys=[[22]], plan=_plan())
    swapped = _trace(operation_keys=[[23]], plan=_plan([34]))
    result = audit_operation_counterfactual(
        baseline, swapped, context_key=(900,), mode="SWAPPED",
        swapped_operation_key=(23,))
    assert result.status == "BLOCKED_OPERATION_BEHAVIOR_NO_DIFFERENCE"
    assert "response_act" in result.changed_fields


def test_swapping_operation_with_claim_selection_change_is_observed_only() -> None:
    baseline = _trace(operation_keys=[[22]], plan=_plan())
    swapped_plan = _plan()
    swapped_plan["claim_refs"] = [[8]]
    swapped_plan["claim_evidence_refs"] = [{"claim": [8], "evidence": [[90]]}]
    swapped = _trace(operation_keys=[[23]], plan=swapped_plan)
    result = audit_operation_counterfactual(
        baseline, swapped, context_key=(900,), mode="SWAPPED",
        swapped_operation_key=(23,))
    assert result.status == "OBSERVED_PAIR_DIFFERENCE_CAUSAL_PROVENANCE_UNKNOWN"
    assert "claims" in result.changed_fields


@pytest.mark.parametrize("edit,match", [
    (lambda b, c: b.__setitem__("context_key", [901]), "share the frozen context"),
        (lambda b, c: c["operation_keys"].append([23]), "operation/contract consumption"),
    (lambda b, c: c.__setitem__("operation_contracts", [[[23], [1, 1]]]),
     "operation/contract consumption"),
    (lambda b, c: b.__setitem__("termination", 4), "did not close"),
])
def test_counterfactual_contract_rejects_unqualified_pairs(edit, match: str) -> None:
    baseline = _trace(operation_keys=[[22]], plan=_plan())
    counterfactual = _trace(operation_keys=[], termination=4, plan=None, surface="")
    edit(baseline, counterfactual)
    with pytest.raises(OperationBehaviorEvidenceError, match=match):
        audit_operation_counterfactual(
            baseline, counterfactual, context_key=(900,), mode="REMOVED")
