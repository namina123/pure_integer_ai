"""Bind a saved factual trace to its caller-frozen batch assignment.

Records are prospective runner artifacts. This module checks their consistency
without querying a graph or producing an answer. Durable O/H/E and delivery
joins, novelty and semantic operation qualification remain UNKNOWN.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from pure_integer_ai.experiments.batch_case_provenance import (
    CaseProvenanceError,
    capture_factual_case_provenance,
)
from pure_integer_ai.experiments.batch_operation_schedule import BatchOperationAssignment
from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceContractError,
    GraphQueryTraceEvidence,
)


class BatchCaseProvenanceContractError(ValueError):
    """A saved case record is incomplete or differs from its assignment."""


_FORMAT = "PURE_INTEGER_BATCH_CASE_PROVENANCE_RECORD_V1"


def _key(value: object, field: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise BatchCaseProvenanceContractError(f"{field} requires an integer key")
    return tuple(value)


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BatchCaseProvenanceContractError("caller-frozen identity is required")
    return value


def _assignments(value: object) -> tuple[BatchOperationAssignment, ...]:
    if (not isinstance(value, Sequence) or not value
            or any(not isinstance(item, BatchOperationAssignment) for item in value)):
        raise BatchCaseProvenanceContractError("caller-frozen assignments are required")
    items = tuple(value)
    if tuple(item.case for item in items) != tuple(range(1, len(items) + 1)):
        raise BatchCaseProvenanceContractError("assignments are not a contiguous case prefix")
    if len({item.proposition_key for item in items}) != len(items):
        raise BatchCaseProvenanceContractError("assignments repeat a proposition")
    return items


def _validate_capture(value: object, assignment: BatchOperationAssignment) -> None:
    if not isinstance(value, Mapping):
        raise BatchCaseProvenanceContractError("captured provenance must be an object")
    if _key(value.get("expected_claim"), "expected_claim") != assignment.proposition_key:
        raise BatchCaseProvenanceContractError("captured claim differs from assignment")
    if _key(value.get("expected_operation"), "expected_operation") != assignment.operation_key:
        raise BatchCaseProvenanceContractError("captured operation differs from assignment")
    query = _key(value.get("query_values"), "query_values")
    try:
        saved_trace = value.get("trace")
        GraphQueryTraceEvidence.from_integer_artifact(saved_trace)
        if saved_trace.get("operation_keys") != [list(assignment.operation_key)]:
            raise BatchCaseProvenanceContractError("trace consumed a different operation")
        # Only encoding/display headers stripped by capture are reconstructed.
        # All query/state/plan/generation evidence must already be in the file.
        # Decoding the recorded output is an audit-boundary check, never a
        # runtime answer source. The original trace is neither queried nor edited.
        query_result = dict(saved_trace)
        query_result["integer_key_encoding"] = "INTEGER_KEY_TABLE_V1"
        query_result["response_surface"] = "".join(
            chr(item) for item in _key(value.get("output_values"), "output_values"))
        checked = capture_factual_case_provenance(
            query_result=query_result,
            query_values=query,
            expected_claim=assignment.proposition_key,
            expected_operation=assignment.operation_key,
            input_observation=_key(value.get("input_observation"), "input_observation"),
            input_source=_key(value.get("input_source"), "input_source"),
            delivery_summary=value.get("delivery_summary"),
        )
        # Mapping equality treats True as 1; canonical JSON comparison does not.
        if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(
                checked, sort_keys=True, allow_nan=False):
            raise BatchCaseProvenanceContractError("saved capture differs from trace evidence")
    except BatchCaseProvenanceContractError:
        raise
    except (CaseProvenanceError, GraphQueryTraceContractError, TypeError, ValueError) as error:
        raise BatchCaseProvenanceContractError(str(error)) from error


def build_batch_case_provenance_record(
        *, identity: str, assignment: BatchOperationAssignment,
        provenance: Mapping[str, Any]) -> dict[str, object]:
    """Build an independent snapshot for the prospective worker artifact."""
    _identity(identity)
    if not isinstance(assignment, BatchOperationAssignment):
        raise BatchCaseProvenanceContractError("caller-frozen assignment is required")
    snapshot = deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else provenance
    _validate_capture(snapshot, assignment)
    return {
        "format": _FORMAT,
        "identity": identity,
        **assignment.to_value(),
        "provenance": snapshot,
        "free_dialogue_complete": 0,
    }


def validate_batch_case_provenance_records(
        assignments: Sequence[BatchOperationAssignment],
        records: Sequence[Mapping[str, Any]], *, identity: str) -> dict[str, object]:
    """Revalidate saved traces against caller-frozen cases and run identity."""
    _identity(identity)
    assignments = _assignments(assignments)
    if (not isinstance(records, Sequence)
            or any(not isinstance(item, Mapping) for item in records)):
        raise BatchCaseProvenanceContractError("provenance records must be objects")
    by_case = {}
    for record in records:
        case = record.get("case")
        if type(case) is not int or case <= 0:
            raise BatchCaseProvenanceContractError("record case must be a positive integer")
        if case in by_case:
            raise BatchCaseProvenanceContractError("duplicate provenance case")
        by_case[case] = record
    if tuple(sorted(by_case)) != tuple(item.case for item in assignments):
        raise BatchCaseProvenanceContractError("record cases do not match assignments")
    for assignment in assignments:
        record = by_case[assignment.case]
        if record.get("format") != _FORMAT or record.get("identity") != identity:
            raise BatchCaseProvenanceContractError("record format or frozen identity drift")
        if type(record.get("free_dialogue_complete")) is not int \
                or record["free_dialogue_complete"] != 0:
            raise BatchCaseProvenanceContractError("free_dialogue_complete must remain zero")
        expected_fields = assignment.to_value()
        if set(record) != {*expected_fields, "format", "identity", "provenance",
                           "free_dialogue_complete"}:
            raise BatchCaseProvenanceContractError("record envelope fields drift")
        for field, expected in expected_fields.items():
            if field != "case" and _key(record.get(field), field) != tuple(expected):
                raise BatchCaseProvenanceContractError(f"record {field} differs from assignment")
        _validate_capture(record.get("provenance"), assignment)
    return {
        "format": "PURE_INTEGER_BATCH_PROVENANCE_RECORD_AUDIT_V1",
        "status": "PASS_BATCH_PROVENANCE_RECORD_CONTRACT",
        "scope": "SAVED_TRACE_AND_FROZEN_ASSIGNMENT_CONSISTENCY_ONLY",
        "identity": identity,
        "covered_case_count": len(assignments),
        "full_chain_qualification": "UNKNOWN_requires_durable_proof_read_audit",
        "held_out_novelty": "UNKNOWN_requires_split_audit",
        "operation_behavior": "UNKNOWN_requires_counterfactual_audit",
        "free_dialogue_complete": 0,
    }


__all__ = [
    "BatchCaseProvenanceContractError",
    "build_batch_case_provenance_record",
    "validate_batch_case_provenance_records",
]
