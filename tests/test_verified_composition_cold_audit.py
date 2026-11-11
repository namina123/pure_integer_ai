"""Use the completed original call to verify proof drift without rerunning it."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.relation_composition_verifier import VerifiedRelationComposition
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.verified_composition_call_audit import (
    inspect_saved_verified_composition, verify_saved_composition_trace,
)


@pytest.fixture(scope="module")
def source():
    root = os.environ.get("PIA_SAVED_COMPOSITION_SOURCE")
    if root is None:
        pytest.skip("this cold audit requires an explicitly supplied completed source")
    return Path(root).resolve(strict=True)


def test_completed_original_call_restores_core_ohe_proof_plan_delivery(source):
    result = inspect_saved_verified_composition(source)
    assert result["status"] == "PASS_DEVELOPMENT_QUERY_LOCAL_DERIVATION_CORE_OHE_PLAN_DELIVERY"
    assert result["query_count"] == result["derived_support_count"] == 1
    assert result["source_write_count"] == result["query_reexecution_count"] == 0
    assert result["generation_reexecution_count"] == result["core_claims_written"] == 0
    assert result["held_out_novelty"] == "UNKNOWN_NO_FOUR_WAY_COURSE"


def test_original_derived_trace_fails_when_one_exact_premise_is_omitted(source):
    case = source / "cases/001"
    value = json.loads((case / "trace.int.json").read_bytes())
    report = json.loads((case / "call_receipt.json").read_bytes())
    trace = GraphQueryTraceEvidence(value)
    verified = VerifiedRelationComposition.from_stable_key(trace.resolve_ref(
        value["generation"]["verified_composition"]["integer_key_ref"]))
    prepared = SimpleNamespace(integer_trace=lambda: _decode_input(value["input_structure"], trace),
        token_values=tuple(report["input_values"]), operation_keys=tuple(
            tuple(key) for key in report["consumed_operation_keys"]))
    changed = deepcopy(value)
    expected = verified.pending.premise_evidence_keys[0][0]
    removed = next(ref for ref in changed["state_key"]["evidence"]
                   if trace.resolve_ref(ref) == expected)
    changed["state_key"]["evidence"].remove(removed)
    with TrainedRelationGraphRuntime(source / "core.sqlite3") as core:
        with pytest.raises(ValueError, match="exact Core premise"):
            verify_saved_composition_trace(changed, report, prepared, verified, core)
        changed = deepcopy(value)
        # A missing proof binding cannot be replaced by retaining derived support.
        marker = next(ref for ref in changed["state_key"]["bindings"]
                      if trace.resolve_ref(ref)[:4] == (1, 2, 92183, 1))
        changed["state_key"]["bindings"].remove(marker)
        with pytest.raises(ValueError, match="verification binding"):
            verify_saved_composition_trace(changed, report, prepared, verified, core)
