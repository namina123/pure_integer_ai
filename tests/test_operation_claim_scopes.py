"""Shared roles are separate claim occurrences; integer IDs do not order output."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.experiments.generation_organization import build_multi_response_plan
from pure_integer_ai.experiments.operation_program_query import operation_plan_matches, operation_plan_trace_matches
from test_stage_c_response_plan import _multi_claim_evidence


def _shared():
    evidences, generation = _multi_claim_evidence()
    second = replace(evidences[1], bindings=(evidences[0].bindings[0], evidences[1].bindings[1]))
    evidences = (evidences[0], second)
    contract = tuple((item.proposition_key, tuple(binding.role.stable_key() for binding in item.bindings))
                     for item in evidences)
    return evidences, replace(generation, surface="甲包含一；甲包含二"), contract


def test_shared_slot_needs_exact_graph_projection_and_both_output_occurrences():
    evidences, generation, contract = _shared()
    assert build_multi_response_plan(evidences, generation) is None
    plan = build_multi_response_plan(evidences, generation, claim_slot_contract=contract)
    assert plan is not None and len(plan.slot_sequence) == 4
    assert plan.slot_sequence[0].stable_key() == plan.slot_sequence[2].stable_key()
    assert plan.claim_slot_ranges == ((contract[0][0], 0, 2), (contract[1][0], 2, 4))
    with pytest.raises(ValueError, match="同一 claim"):
        replace(plan, claim_slot_ranges=())
    assert build_multi_response_plan(evidences, replace(generation, surface="甲包含一；包含二"),
                                     claim_slot_contract=contract) is None
    assert build_multi_response_plan(evidences, generation, claim_slot_contract=tuple(reversed(contract))) is None


def test_actual_claim_ranges_determine_operation_order_even_when_ids_sort_differently():
    evidences, generation, contract = _shared()
    reverse = tuple(reversed(evidences))
    selected = tuple(reversed(contract))
    plan = build_multi_response_plan(reverse, replace(generation, surface="甲包含二；甲包含一"),
                                     claim_slot_contract=selected)
    assert plan is not None and tuple(claim.stable_key() for claim in plan.claim_refs) != tuple(key for key, _ in selected)
    assert operation_plan_matches(SimpleNamespace(status=READY, outputs=selected), plan)
    assert not operation_plan_matches(SimpleNamespace(status=READY, outputs=contract), plan)
    assert operation_plan_trace_matches(SimpleNamespace(status=READY, outputs=selected), plan.to_dict())
    assert not operation_plan_trace_matches(SimpleNamespace(status=READY, outputs=contract), plan.to_dict())
    with pytest.raises(ValueError, match="同一 claim"):
        replace(plan, claim_slot_ranges=())


@pytest.mark.parametrize("change", ["missing_role", "overlap", "extra_slot", "missing_range"])
def test_saved_plan_projection_rejects_incomplete_or_overlapping_occurrences(change):
    """即使 claim 集合相同，也必须保留每个 occurrence 的完整连续角色。"""
    evidences, generation, contract = _shared()
    plan = build_multi_response_plan(evidences, generation, claim_slot_contract=contract).to_dict()
    if change == "missing_role":
        plan["slot_sequence"].pop()
    elif change == "overlap":
        plan["claim_slot_ranges"][1]["start"] = 1
    elif change == "extra_slot":
        plan["slot_sequence"].append(plan["slot_sequence"][-1])
    else:
        plan["claim_slot_ranges"].pop()
    assert not operation_plan_trace_matches(SimpleNamespace(status=READY, outputs=contract), plan)
