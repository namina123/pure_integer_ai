"""Reference-table corruption and missing fields must not become zero evidence."""
from __future__ import annotations

from copy import deepcopy

import pytest

from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceContractError,
    GraphQueryTraceEvidence,
)


def _expanded() -> dict:
    return {
        "schema_version": 3,
        "hops": [
            {"space": 1, "proposition": [7, 1], "predicate": [4, 2], "source_hash": 17},
            {"space": 2, "proposition": [], "predicate": [4, 2], "source_hash": 18},
            {"space": 3, "proposition": [], "predicate": [], "source_hash": 0},
        ],
        "memory_routes": [[1, 99], [2, 88]],
        "expanded_edges": [[6, 77]],
        "operation_contracts": [[[1, 1], [2, 1]], [[1, 2], [2, 1]]],
    }


def _compact() -> dict:
    return {
        "schema_version": 4,
        "integer_key_encoding": "INTEGER_KEY_TABLE_V1",
        "integer_key_table": [[7, 1], [4, 2], [1, 99], [2, 88], [6, 77]],
        "hops": [
            {"space": 1, "proposition_ref": 1, "predicate_ref": 2, "source_hash": 17},
            {"space": 2, "proposition_ref": 0, "predicate_ref": 2, "source_hash": 18},
            {"space": 3, "proposition_ref": 0, "predicate_ref": 0, "source_hash": 0},
        ],
        "memory_route_refs": [3, 4],
        "expanded_edge_refs": [5],
        "operation_contracts": [[[1, 1], [2, 1]], [[1, 2], [2, 1]]],
    }


def test_compact_and_expanded_traces_preserve_the_same_evidence() -> None:
    expanded = GraphQueryTraceEvidence(_expanded())
    compact = GraphQueryTraceEvidence(_compact())
    assert compact.memory_routes() == expanded.memory_routes() == ((1, 99), (2, 88))
    assert compact.hops() == expanded.hops()
    assert compact.expanded_edges() == expanded.expanded_edges()
    assert compact.counts() == expanded.counts() == {
        "core_hop_count": 1,
        "memory_hop_count": 1,
        "companion_hop_count": 1,
        "all_space_hop_count": 3,
        "memory_route_count": 2,
        "operation_contract_count": 2,
        "operation_mode_count": 1,
    }


@pytest.mark.parametrize("factory,field", [
    (_expanded, "memory_routes"), (_compact, "memory_route_refs"),
])
def test_missing_routes_are_unknown_not_zero(factory, field: str) -> None:
    trace = factory()
    del trace[field]
    with pytest.raises(GraphQueryTraceContractError, match="missing trace evidence field"):
        GraphQueryTraceEvidence(trace).memory_routes()


def test_explicit_empty_routes_are_an_observed_zero() -> None:
    trace = _compact()
    trace["memory_route_refs"] = []
    assert GraphQueryTraceEvidence(trace).counts()["memory_route_count"] == 0


@pytest.mark.parametrize("bad_ref", [True, -1, 6, "3", None, 0])
def test_invalid_or_empty_route_reference_is_rejected(bad_ref: object) -> None:
    trace = _compact()
    trace["memory_route_refs"] = [bad_ref]
    with pytest.raises(GraphQueryTraceContractError):
        GraphQueryTraceEvidence(trace).memory_routes()


def test_different_expanded_and_compact_evidence_is_rejected() -> None:
    trace = _compact()
    trace["memory_routes"] = [[1, 99]]
    with pytest.raises(GraphQueryTraceContractError, match="evidence drift"):
        GraphQueryTraceEvidence(trace).memory_routes()


def test_duplicate_key_table_and_boolean_schema_are_rejected() -> None:
    trace = _compact()
    trace["integer_key_table"].append(deepcopy(trace["integer_key_table"][0]))
    with pytest.raises(GraphQueryTraceContractError, match="duplicated"):
        GraphQueryTraceEvidence(trace)
    trace = _expanded()
    trace["schema_version"] = True
    with pytest.raises(GraphQueryTraceContractError, match="unsupported"):
        GraphQueryTraceEvidence(trace)


def test_contract_keys_do_not_count_as_distinct_operation_modes() -> None:
    evidence = GraphQueryTraceEvidence(_compact())
    assert evidence.counts()["operation_contract_count"] == 2
    assert evidence.counts()["operation_mode_count"] == 1


def test_missing_or_duplicated_operation_contracts_are_rejected() -> None:
    trace = _compact()
    del trace["operation_contracts"]
    with pytest.raises(GraphQueryTraceContractError, match="missing trace evidence field"):
        GraphQueryTraceEvidence(trace).operation_contracts()
    trace = _compact()
    trace["operation_contracts"].append(deepcopy(trace["operation_contracts"][0]))
    with pytest.raises(GraphQueryTraceContractError, match="duplicated"):
        GraphQueryTraceEvidence(trace).operation_contracts()


def _integer_artifact() -> dict:
    trace = _compact()
    del trace["integer_key_encoding"]
    trace.update({
        "query_ref": 3,
        "state_key": {"query_ref": 3, "scalars": [0] * 17},
        "input_structure": {},
        "frontier_trace": {"steps": []},
    })
    return trace


def test_integer_projection_requires_explicit_artifact_entry() -> None:
    trace = _integer_artifact()
    with pytest.raises(GraphQueryTraceContractError, match="encoding is missing"):
        GraphQueryTraceEvidence(trace)
    assert GraphQueryTraceEvidence.from_integer_artifact(trace).counts() == (
        GraphQueryTraceEvidence(_compact()).counts())
    assert "integer_key_encoding" not in trace


@pytest.mark.parametrize("field,value", [
    ("format", "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V4"),
    ("unexpected", True), ("unexpected", 1.5),
    ("unexpected", {"nested": "text"}), ("integer_key_encoding", None),
    ("query_ref", 0), ("query_ref", 6),
    ("state_key", {"query_ref": 4, "scalars": [0] * 17}),
    ("state_key", {"query_ref": 3, "scalars": [0] * 16}),
    ("frontier_trace", {}), ("input_structure", None),
])
def test_integer_artifact_rejects_text_and_invalid_envelope(field, value) -> None:
    trace = _integer_artifact()
    trace[field] = value
    with pytest.raises(GraphQueryTraceContractError):
        GraphQueryTraceEvidence.from_integer_artifact(trace)


def test_integer_artifact_does_not_accept_missing_route_evidence() -> None:
    trace = _integer_artifact()
    del trace["memory_route_refs"]
    with pytest.raises(GraphQueryTraceContractError, match="missing trace evidence field"):
        GraphQueryTraceEvidence.from_integer_artifact(trace).counts()
