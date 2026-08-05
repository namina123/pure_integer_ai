"""W09-05 resource stop、规模证伪与 worker 规范专项。"""
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w09_authority import W09_RESOURCE_BUDGET
from pure_integer_ai.experiments.ph2_w09_contract import open_w09_frozen_contract
from pure_integer_ai.experiments.ph2_w09_resource import (
    W09ResourceError,
    W09ResourceUsage,
    W09_STOP_POLICIES,
    open_w09_resource_controller,
)


ROOT = Path(__file__).parents[1]


def _counts(**updates):
    values = {key: 0 for key in W09_RESOURCE_BUDGET}
    values.update(updates)
    return tuple(sorted(values.items()))


def test_all_stop_states_are_distinct_and_budget_is_manifest_bound():
    context = open_w09_frozen_contract(ROOT)
    controller = open_w09_resource_controller(ROOT, context)
    zero = W09ResourceUsage.zero()
    request = _counts(max_records=1, max_logic_operations=4)

    resolved = controller.evaluate(zero, request)
    assert resolved.decision.stop_state == "RESOLVED"
    assert resolved.decision.publication_allowed == 1
    assert dict(resolved.decision.resource_audit.limits) == W09_RESOURCE_BUDGET

    assert controller.evaluate(
        zero,
        request,
        clarify_required=1,
    ).decision.stop_state == "CLARIFY"
    assert controller.evaluate(
        zero,
        request,
        candidate_count=0,
    ).decision.stop_state == "UNKNOWN"
    assert controller.evaluate(
        zero,
        request,
        access_blocked=1,
    ).decision.stop_state == "ACCESS_BLOCKED"
    assert controller.evaluate(
        zero,
        request,
        grounding_blocked=1,
    ).decision.stop_state == "GROUNDING_BLOCKED"

    full = W09ResourceUsage(_counts(max_records=W09_RESOURCE_BUDGET["max_records"]))
    exhausted = controller.evaluate(full, _counts(max_records=1))
    assert exhausted.decision.stop_state == "BUDGET_EXHAUSTED"
    assert exhausted.overflow_keys == ("max_records",)
    assert exhausted.decision.publication_allowed == 0


def test_invalid_counters_and_bypassed_policy_fail_closed():
    controller = open_w09_resource_controller(ROOT)
    with pytest.raises(W09ResourceError):
        W09ResourceUsage(_counts(max_records=-1))
    with pytest.raises(W09ResourceError):
        W09ResourceUsage(tuple())
    with pytest.raises(W09ResourceError):
        controller.evaluate(
            W09ResourceUsage.zero(),
            _counts(max_segments=W09_RESOURCE_BUDGET["max_segments"] + 1),
        )
    with pytest.raises(W09ResourceError):
        controller.evaluate(
            W09ResourceUsage.zero(),
            _counts(),
            policy_key="FULL_LIBRARY_SCAN",
        )
    with pytest.raises(W09ResourceError):
        controller.evaluate(
            W09ResourceUsage.zero(),
            _counts(),
            access_blocked=True,
        )


def test_scale_workers_policies_and_ablation_are_canonical():
    controller = open_w09_resource_controller(ROOT)
    probes = tuple(controller.run_scale_probe(factor) for factor in (1, 10, 100))
    assert tuple(item.irrelevant_records for item in probes) == (8, 80, 800)
    assert len({item.actual_records for item in probes}) == 1
    assert len({item.actual_segments for item in probes}) == 1
    assert len({item.actual_logic_operations for item in probes}) == 1
    assert len({item.canonical_result_key for item in probes}) == 1
    assert all(item.irrelevant_scan_count == 0 for item in probes)

    workers = tuple(
        controller.run_workers(worker, W09ResourceUsage.zero(), _counts(max_records=1))
        for worker in (1, 2, 4)
    )
    assert len({item.canonical_result_key for item in workers}) == 1
    assert len({item.stop.stable_key() for item in workers}) == 1
    with pytest.raises(W09ResourceError):
        controller.run_workers(3, W09ResourceUsage.zero(), _counts())

    policies = tuple(
        controller.evaluate(
            W09ResourceUsage.zero(),
            _counts(max_records=1),
            policy_key=policy,
        )
        for policy in W09_STOP_POLICIES
    )
    assert tuple(item.policy_key for item in policies) == W09_STOP_POLICIES
    assert all(item.decision.stop_state == "RESOLVED" for item in policies)

    ablation = controller.ablate_controller()
    assert ablation.target_dimension_key == "W-09-RESOURCE_STOP"
    assert ablation.target_status == "FAIL"
    assert ablation.unrelated_dimension_failure_count == 0
