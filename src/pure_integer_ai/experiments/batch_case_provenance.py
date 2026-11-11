"""Capture strict per-case evidence from the original QueryBridge call.

Use while the result and delivery receipt are still in memory. A later audit
must never run the query again to fill missing provenance. This module neither
queries a graph nor writes an artifact; the caller owns atomic persistence.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceEvidence,
    GraphQueryTraceContractError,
)


class CaseProvenanceError(ValueError):
    """The current call cannot support a factual case qualification."""


_STRING_FIELDS = frozenset({
    ("format",), ("integer_key_encoding",), ("termination_reason",),
    ("response_surface",), ("frontier_trace", "format"),
    ("state_key", "encoding"),
})


def _key(value: object, label: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise CaseProvenanceError(f"{label} requires a nonempty integer key")
    return tuple(value)


def _keys(value: object, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise CaseProvenanceError(f"{label} requires integer keys")
    return tuple(_key(item, label) for item in value)


def _integer_projection(value: object, path: tuple[str, ...] = ()) -> Any:
    """Strip only known boundary strings; unexpected text fails closed."""
    if type(value) is int:
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CaseProvenanceError("trace contains a non-string object key")
            child = (*path, key)
            if child == ("failure_reason",) and item is not None:
                if not isinstance(item, str):
                    raise CaseProvenanceError("trace failure_reason header is invalid")
                continue
            if child in _STRING_FIELDS:
                if not isinstance(item, str):
                    raise CaseProvenanceError(f"trace header {child} is invalid")
                continue
            result[key] = _integer_projection(item, child)
        return result
    if isinstance(value, (list, tuple)):
        # 每项仍严格核验整数类型；复合项与其它类型保留原递归和拒绝路径。
        return [item if type(item) is int else _integer_projection(item, path)
                for item in value]
    raise CaseProvenanceError(f"unexpected non-integer trace field: {'.'.join(path)}")


def _compact_key_field(trace: GraphQueryTraceEvidence, value: object,
                       label: str) -> tuple[int, ...]:
    if (not isinstance(value, Mapping)
            or set(value) != {"integer_key_ref"}):
        raise CaseProvenanceError(f"{label} is not a compact integer key")
    try:
        return trace.resolve_ref(value["integer_key_ref"], label)
    except GraphQueryTraceContractError as error:
        raise CaseProvenanceError(str(error)) from error


def capture_factual_case_provenance(
        *, query_result: Mapping[str, Any], query_values: tuple[int, ...],
        expected_claim: tuple[int, ...], expected_operation: tuple[int, ...],
        input_observation: tuple[int, ...], input_source: tuple[int, ...],
        delivery_summary: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Validate and freeze exact call evidence, with explicit unknown limits.

    This captures the requested claim in the selected plan and the delivery
    references reported in the same call. A summary cannot establish that the
    durable proof belongs to that plan; that join needs a separate read audit.
    """
    if not isinstance(query_result, Mapping) or not isinstance(delivery_summary, Mapping):
        raise CaseProvenanceError("query and delivery evidence must be mappings")
    query_key = _key(query_values, "query")
    claim_key = _key(expected_claim, "expected claim")
    operation_key = _key(expected_operation, "expected operation")
    observation_key = _key(input_observation, "input Observation")
    input_source_key = _key(input_source, "input source")
    if query_result.get("schema_version") != 4:
        raise CaseProvenanceError("production case requires compact QueryBridge trace v4")
    try:
        trace = GraphQueryTraceEvidence(query_result)
        counts = trace.counts()
        trace_query = trace.resolve_ref(query_result.get("query_ref"), "query_ref")
    except GraphQueryTraceContractError as error:
        raise CaseProvenanceError(str(error)) from error
    structure = query_result.get("input_structure")
    if not isinstance(structure, Mapping):
        raise CaseProvenanceError("input structure trace is missing")
    structure_source = _compact_key_field(trace, structure.get("source_ref"), "input source")
    structure_tokens = _compact_key_field(trace, structure.get("token_values"), "input tokens")
    if trace_query != structure_source or structure_tokens != query_key:
        raise CaseProvenanceError("QueryState source/input tokens differ from delivered input")
    state = query_result.get("state_key")
    if not isinstance(state, Mapping) or state.get("query_ref") != query_result.get("query_ref"):
        raise CaseProvenanceError("QueryState/query reference drift")
    scalars = state.get("scalars")
    if (not isinstance(scalars, list) or len(scalars) != 17
            or any(type(item) is not int for item in scalars)):
        raise CaseProvenanceError("QueryState scalar contract missing")
    if (query_result.get("termination") != 1 or scalars[-1] != 1
            or scalars[6] != 0 or scalars[7] != 1 or scalars[8] != 0
            or scalars[15] != 1):
        raise CaseProvenanceError("query is not an evidence-closed factual answer")
    active_spaces = query_result.get("active_spaces")
    if active_spaces != [1, 2, 3] or state.get("active_spaces") != [1, 2, 3]:
        raise CaseProvenanceError("Core/Memory/Companion do not share active state")
    if (counts["memory_route_count"] < 1 or counts["memory_hop_count"] < 1
            or counts["companion_hop_count"] < 1):
        raise CaseProvenanceError("Memory or Companion consumption is unproven")
    # A typed Core input projection may close a claim with zero direct Core
    # hops. Preserve that distinction; do not turn active_spaces into a hop.
    inferred = _keys(query_result.get("operation_keys"), "operation keys")
    contracts = trace.operation_contracts()
    if (operation_key not in inferred or len(set(inferred)) != len(inferred)
            or set(inferred) != {key for key, _ in contracts}):
        raise CaseProvenanceError("requested operation was not consumed")
    plan = query_result.get("response_plan")
    if not isinstance(plan, Mapping):
        raise CaseProvenanceError("ResponsePlan is missing")
    claims = _keys(plan.get("claim_refs"), "plan claims")
    if claim_key not in claims:
        raise CaseProvenanceError("target claim is absent from ResponsePlan")
    if len(set(claims)) != len(claims):
        raise CaseProvenanceError("plan claims are duplicated")
    evidence_rows = plan.get("claim_evidence_refs")
    if not isinstance(evidence_rows, list):
        raise CaseProvenanceError("claim evidence mapping is missing")
    by_claim = {}
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            raise CaseProvenanceError("claim evidence row is invalid")
        key = _key(row.get("claim"), "evidence claim")
        refs = _keys(row.get("evidence"), "claim evidence")
        if key in by_claim or not refs:
            raise CaseProvenanceError("claim evidence is duplicated or empty")
        by_claim[key] = refs
    if set(by_claim) != set(claims):
        raise CaseProvenanceError("evidence does not cover every plan claim")
    plan_evidence = set(_keys(plan.get("evidence_refs"), "plan evidence"))
    if any(ref not in plan_evidence for refs in by_claim.values() for ref in refs):
        raise CaseProvenanceError("claim evidence is outside the selected plan")
    state_evidence = state.get("evidence")
    if not isinstance(state_evidence, list):
        raise CaseProvenanceError("QueryState evidence references are missing")
    try:
        query_evidence = {trace.resolve_ref(ref, "state evidence") for ref in state_evidence}
    except GraphQueryTraceContractError as error:
        raise CaseProvenanceError(str(error)) from error
    if not query_evidence or () in query_evidence or not query_evidence.issubset(plan_evidence):
        raise CaseProvenanceError("selected plan does not carry the current QueryState evidence")
    _key(plan.get("response_act"), "response act")
    _key(plan.get("stable_key"), "ResponsePlan")
    output = query_result.get("response_surface")
    if not isinstance(output, str) or not output:
        raise CaseProvenanceError("response surface is missing")
    output_values = [ord(value) for value in output]
    generation = query_result.get("generation")
    if (not isinstance(generation, Mapping)
            or _compact_key_field(trace, generation.get("surface_values"),
                                  "generated surface") != tuple(output_values)):
        raise CaseProvenanceError("graph generation differs from ResponsePlan output")
    if (type(delivery_summary.get("delivered")) is not int or delivery_summary["delivered"] != 1
            or type(delivery_summary.get("claim_count")) is not int
            or delivery_summary["claim_count"] != len(claims)
            or type(delivery_summary.get("evidence_ref_count")) is not int
            or delivery_summary["evidence_ref_count"] != len(plan_evidence)):
        raise CaseProvenanceError("delivery does not adopt the factual plan")
    proof = _key(delivery_summary.get("proof_ref"), "delivery proof")
    source = _key(delivery_summary.get("output_source"), "output Observation source")
    if type(delivery_summary.get("episode_event_hash")) is not int \
            or delivery_summary["episode_event_hash"] <= 0:
        raise CaseProvenanceError("delivery Episode is missing")
    if not isinstance(delivery_summary.get("use_event_hashes"), list):
        raise CaseProvenanceError("delivery Use evidence is missing")
    projection = _integer_projection(query_result)
    return {
        "format": "PURE_INTEGER_BATCH_CASE_PROVENANCE_V1",
        "status": "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS",
        "query_values": list(query_key),
        "expected_claim": list(claim_key),
        "expected_operation": list(operation_key),
        "input_observation": list(observation_key),
        "input_source": list(input_source_key),
        "plan_claims": [list(key) for key in claims],
        "delivery_proof_ref": list(proof),
        "delivery_output_source": list(source),
        "delivery_episode_event_hash": delivery_summary["episode_event_hash"],
        "delivery_summary": _integer_projection(delivery_summary),
        "delivery_plan_exact_join": "UNKNOWN_requires_durable_proof_read_audit",
        "output_values": output_values,
        "trace": projection,
        "trace_counts": counts,
        "direct_core_hop_qualification": (
            "OBSERVED" if counts["core_hop_count"] else "UNKNOWN_TYPED_PROJECTION_ONLY"),
        "ohe_to_plan_exact_join": "UNKNOWN_requires_durable_Memory_ref_audit",
        "held_out_novelty": "UNKNOWN_requires_split_audit",
        "operation_behavior": "UNKNOWN_requires_counterfactual_audit",
        "free_dialogue_complete": 0,
    }
