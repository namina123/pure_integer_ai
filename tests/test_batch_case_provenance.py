"""The next runner must capture a claim-to-delivery chain in the same call."""
from __future__ import annotations

from copy import deepcopy

import pytest

from pure_integer_ai.experiments.batch_case_provenance import (
    CaseProvenanceError,
    capture_factual_case_provenance,
)


def _call() -> tuple[dict, dict]:
    table = [[100], [65], [88], [7], [50], [60], [90]]
    query = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V4",
        "schema_version": 4,
        "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
        "integer_key_table": table,
        "query_ref": 1,
        "input_structure": {
            "source_ref": {"integer_key_ref": 1},
            "token_values": {"integer_key_ref": 2},
        },
        "state_key": {
            "query_ref": 1,
            "active_spaces": [1, 2, 3],
            "evidence": [7],
            "scalars": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        },
        "active_spaces": [1, 2, 3],
        "termination": 1,
        "hops": [
            {"space": 2, "proposition_ref": 4, "predicate_ref": 5,
             "source_hash": 17},
            {"space": 3, "proposition_ref": 0, "predicate_ref": 6,
             "source_hash": 18},
        ],
        "memory_route_refs": [5],
        "expanded_edge_refs": [],
        "operation_keys": [[22]],
        "operation_contracts": [[[22], [91672, 3]]],
        "response_plan": {
            "response_act": [33], "stable_key": [44],
            "claim_refs": [[7]],
            "evidence_refs": [[90]],
            "claim_evidence_refs": [{"claim": [7], "evidence": [[90]]}],
        },
        "response_surface": "X",
        "generation": {"surface_values": {"integer_key_ref": 3}},
        "frontier_trace": {"format": "INTEGER_FRONTIER_TRACE_V1", "steps": []},
    }
    delivery = {
        "proof_ref": [101], "output_source": [102],
        "episode_event_hash": 19, "use_event_hashes": [20],
        "claim_count": 1, "evidence_ref_count": 1, "delivered": 1,
    }
    return query, delivery


def _capture(query: dict, delivery: dict) -> dict:
    return capture_factual_case_provenance(
        query_result=query, query_values=(65,), expected_claim=(7,),
        expected_operation=(22,), input_observation=(103,), input_source=(104,),
        delivery_summary=delivery)


def test_captures_exact_claim_and_compact_trace_without_boundary_text() -> None:
    query, delivery = _call()
    result = _capture(query, delivery)
    assert result["status"] == "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS"
    assert result["delivery_plan_exact_join"] == "UNKNOWN_requires_durable_proof_read_audit"
    assert result["delivery_summary"] == delivery
    assert result["trace_counts"]["memory_route_count"] == 1
    assert result["direct_core_hop_qualification"] == "UNKNOWN_TYPED_PROJECTION_ONLY"
    assert result["trace"]["generation"]["surface_values"] == {"integer_key_ref": 3}
    assert "response_surface" not in result["trace"]
    assert "format" not in result["trace"]["frontier_trace"]


@pytest.mark.parametrize("edit,match", [
    (lambda q, d: q["response_plan"]["claim_refs"].__setitem__(0, [8]), "target claim"),
    (lambda q, d: q["input_structure"]["token_values"].__setitem__("integer_key_ref", 4),
     "input tokens"),
    (lambda q, d: q.__setitem__("operation_keys", [[23]]), "operation"),
    (lambda q, d: q["generation"]["surface_values"].__setitem__("integer_key_ref", 2),
     "generation differs"),
    (lambda q, d: d.__setitem__("delivered", 0), "delivery"),
])
def test_rejects_missing_claim_input_operation_generation_or_delivery(edit, match: str) -> None:
    query, delivery = _call()
    edit(query, delivery)
    with pytest.raises(CaseProvenanceError, match=match):
        _capture(query, delivery)


def test_unexpected_trace_string_cannot_be_silently_dropped() -> None:
    query, delivery = _call()
    query["unexplained"] = "unverified text"
    with pytest.raises(CaseProvenanceError, match="unexpected non-integer"):
        _capture(query, delivery)


def test_original_call_is_not_mutated() -> None:
    query, delivery = _call()
    original = deepcopy(query)
    _capture(query, delivery)
    assert query == original


def test_plan_evidence_must_include_the_original_query_state() -> None:
    query, delivery = _call()
    query["state_key"]["evidence"] = [5]
    with pytest.raises(CaseProvenanceError, match="current QueryState evidence"):
        _capture(query, delivery)


def test_boolean_delivery_counts_cannot_be_integer_evidence() -> None:
    query, delivery = _call()
    delivery["claim_count"] = True
    with pytest.raises(CaseProvenanceError, match="delivery"):
        _capture(query, delivery)
