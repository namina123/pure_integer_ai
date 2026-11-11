from __future__ import annotations

from copy import deepcopy
import json

import pytest

from pure_integer_ai.experiments.batch_case_provenance import (
    capture_factual_case_provenance,
)
from pure_integer_ai.experiments.batch_case_provenance_contract import (
    BatchCaseProvenanceContractError,
    build_batch_case_provenance_record,
    validate_batch_case_provenance_records,
)
from pure_integer_ai.experiments.batch_operation_schedule import (
    BatchOperationAssignment,
)


def _captured(claim: int = 7,
              query_values: tuple[int, ...] = (65, 65306, 88)) -> dict:
    table = [[100], list(query_values), [88], [claim], [50], [60], [90]]
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
            "query_ref": 1, "active_spaces": [1, 2, 3], "evidence": [7],
            "scalars": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        },
        "active_spaces": [1, 2, 3], "termination": 1,
        "hops": [
            {"space": 2, "proposition_ref": 4, "predicate_ref": 5, "source_hash": 17},
            {"space": 3, "proposition_ref": 0, "predicate_ref": 6, "source_hash": 18},
        ],
        "memory_route_refs": [5], "expanded_edge_refs": [],
        "operation_keys": [[22]], "operation_contracts": [[[22], [91672, 3]]],
        "response_plan": {
            "response_act": [33], "stable_key": [44], "claim_refs": [[claim]],
            "evidence_refs": [[90]],
            "claim_evidence_refs": [{"claim": [claim], "evidence": [[90]]}],
        },
        "response_surface": "X",
        "generation": {"surface_values": {"integer_key_ref": 3}},
        "frontier_trace": {"format": "INTEGER_FRONTIER_TRACE_V1", "steps": []},
    }
    delivery = {
        "proof_ref": [101], "output_source": [102], "episode_event_hash": 19,
        "use_event_hashes": [20], "claim_count": 1,
        "evidence_ref_count": 1, "delivered": 1,
    }
    return capture_factual_case_provenance(
        query_result=query, query_values=query_values, expected_claim=(claim,),
        expected_operation=(22,), input_observation=(103,), input_source=(104,),
        delivery_summary=delivery)


def _assignment(case: int = 1) -> BatchOperationAssignment:
    return BatchOperationAssignment(
        case=case, proposition_key=(6 + case,), operation_key=(22,),
        operation_cue=(65,))


def _record(assignment: BatchOperationAssignment | None = None) -> dict:
    assignment = assignment or _assignment()
    return build_batch_case_provenance_record(
        identity="run-a", assignment=assignment,
        provenance=_captured(assignment.proposition_key[0]))


def test_builder_binds_assignment_and_preserves_unknown_boundaries() -> None:
    assignment = _assignment()
    record = _record(assignment)
    assert record["proposition_key"] == [7]
    assert record["operation_key"] == [22]
    audit = validate_batch_case_provenance_records(
        (assignment,), (json.loads(json.dumps(record)),), identity="run-a")
    assert audit["status"] == "PASS_BATCH_PROVENANCE_RECORD_CONTRACT"
    assert audit["full_chain_qualification"].startswith("UNKNOWN")
    assert audit["held_out_novelty"].startswith("UNKNOWN")


@pytest.mark.parametrize("field", [
    "proposition_key", "operation_key", "operation_cue", "provenance",
])
def test_record_rejects_assignment_drift(field: str) -> None:
    assignment = _assignment()
    record = _record(assignment)
    changed = deepcopy(record)
    if field == "provenance":
        changed[field]["expected_operation"] = [23]
    else:
        changed[field] = [999]
    with pytest.raises(BatchCaseProvenanceContractError):
        validate_batch_case_provenance_records((assignment,), (changed,), identity="run-a")


def test_batch_contract_rejects_duplicate_or_missing_cases() -> None:
    first = _assignment(1)
    second = _assignment(2)
    with pytest.raises(BatchCaseProvenanceContractError, match="duplicate"):
        validate_batch_case_provenance_records(
            (first, second), (_record(first), _record(first)), identity="run-a")
    with pytest.raises(BatchCaseProvenanceContractError, match="do not match"):
        validate_batch_case_provenance_records(
            (first, second), (_record(first),), identity="run-a")


def test_record_is_a_snapshot_of_the_same_supplied_capture() -> None:
    provenance = _captured()
    original = deepcopy(provenance)
    record = build_batch_case_provenance_record(
        identity="run-a", assignment=_assignment(), provenance=provenance)
    assert provenance == original
    provenance["trace"]["response_plan"]["claim_refs"][0][0] = 999
    assert record["provenance"] == original
    record["provenance"]["delivery_summary"]["proof_ref"][0] = 777
    assert provenance["delivery_summary"]["proof_ref"] == [101]


def test_rejects_a_consistent_record_from_another_run() -> None:
    with pytest.raises(BatchCaseProvenanceContractError, match="identity drift"):
        validate_batch_case_provenance_records(
            (_assignment(),), (_record(),), identity="run-b")


def test_builder_cannot_attach_a_different_factual_claim_to_the_case() -> None:
    with pytest.raises(BatchCaseProvenanceContractError, match="claim differs"):
        build_batch_case_provenance_record(
            identity="run-a", assignment=_assignment(), provenance=_captured(claim=8))


def test_reader_rejects_replacing_a_saved_record_with_another_valid_capture() -> None:
    record = _record()
    record["provenance"] = _captured(claim=8)
    with pytest.raises(BatchCaseProvenanceContractError, match="claim differs"):
        validate_batch_case_provenance_records(
            (_assignment(),), (record,), identity="run-a")


def test_natural_input_does_not_need_literal_assigned_operation_cue() -> None:
    provenance = _captured(query_values=(88, 90, 91))
    record = build_batch_case_provenance_record(
        identity="run-a", assignment=_assignment(), provenance=provenance)
    assert record["provenance"]["query_values"] == [88, 90, 91]


def test_saved_query_values_must_match_original_trace() -> None:
    provenance = _captured()
    provenance["query_values"] = [88, 65, 65306]
    with pytest.raises(BatchCaseProvenanceContractError, match="input tokens"):
        build_batch_case_provenance_record(
            identity="run-a", assignment=_assignment(), provenance=provenance)


@pytest.mark.parametrize("field,new_value", [
    ("trace", {}),
    ("trace_counts", {"memory_route_count": 100}),
    ("delivery_proof_ref", [999]),
    ("plan_claims", [[999]]),
    ("output_values", [89]),
    ("output_values", [0x110000]),
    ("query_values", [66, 65306, 88]),
    ("delivery_plan_exact_join", "PASS"),
    ("free_dialogue_complete", True),
])
def test_rejects_stale_summaries_missing_trace_or_unsupported_qualification(
        field: str, new_value: object) -> None:
    record = _record()
    record["provenance"][field] = new_value
    with pytest.raises(BatchCaseProvenanceContractError):
        validate_batch_case_provenance_records(
            (_assignment(),), (record,), identity="run-a")


@pytest.mark.parametrize("field,new_value", [
    ("response_plan", {}),
    ("generation", {}),
    ("memory_route_refs", []),
    ("operation_keys", [[23]]),
    ("input_structure", {"token_values": {"integer_key_ref": 3}}),
    ("hops", []),
    ("unexplained", "source text must not enter an integer trace"),
])
def test_revalidates_original_trace_instead_of_trusting_outer_counts(
        field: str, new_value: object) -> None:
    record = _record()
    record["provenance"]["trace"][field] = new_value
    with pytest.raises(BatchCaseProvenanceContractError):
        validate_batch_case_provenance_records(
            (_assignment(),), (record,), identity="run-a")


def test_boolean_count_is_not_integer_evidence() -> None:
    record = _record()
    record["provenance"]["trace_counts"]["memory_route_count"] = True
    with pytest.raises(BatchCaseProvenanceContractError):
        validate_batch_case_provenance_records(
            (_assignment(),), (record,), identity="run-a")


def test_audits_a_complete_batch_but_rejects_reused_assignments() -> None:
    first, second = _assignment(), _assignment(2)
    audit = validate_batch_case_provenance_records(
        (first, second), (_record(first), _record(second)), identity="run-a")
    assert audit["covered_case_count"] == 2
    assert audit["scope"] == "SAVED_TRACE_AND_FROZEN_ASSIGNMENT_CONSISTENCY_ONLY"
    for assignments in ((second,), (first, first),
                        (first, BatchOperationAssignment(2, (7,), (22,), (65,)))):
        with pytest.raises(BatchCaseProvenanceContractError):
            validate_batch_case_provenance_records(
                assignments, (_record(first),), identity="run-a")
