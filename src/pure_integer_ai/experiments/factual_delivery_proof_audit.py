"""Join one persisted factual delivery proof to its captured ResponsePlan.

Callers provide readers bound to the completed run's immutable Memory. The
proof reader must resolve the supplied receipt proof ref with
``read_delivery_proof``; evidence readers must use ``read_factual_evidence_ref``.
This module performs no query, replay, Memory write or course access.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pure_integer_ai.cognition.understanding.query_memory_candidates import _parts
from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence


class FactualDeliveryProofEvidenceError(ValueError):
    """The durable proof is missing, malformed or differs from the captured call."""


def _key(value: object, label: str) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise FactualDeliveryProofEvidenceError(f"{label} requires a full integer key")
    return tuple(value)


def _parts_checked(key: tuple[int, ...], tag: int, label: str) -> tuple[tuple[int, ...], ...]:
    try:
        return _parts(key, tag)
    except ValueError as error:
        raise FactualDeliveryProofEvidenceError(f"invalid {label}") from error


def verify_factual_delivery_proof(
        provenance: Mapping[str, Any], *,
        proof_reader: Callable[[tuple[int, ...]], tuple[int, ...]],
        evidence_reader: Callable[[tuple[int, ...]], tuple[int, ...]],
        ) -> dict[str, Any]:
    """Compare exact proof members; leave O/H/E/event adoption to another audit."""
    if not isinstance(provenance, Mapping):
        raise FactualDeliveryProofEvidenceError("captured provenance is missing")
    if provenance.get("status") != "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS":
        raise FactualDeliveryProofEvidenceError("provenance was not captured in the original call")
    proof_ref = _key(provenance.get("delivery_proof_ref"), "delivery proof ref")
    proof_key = _key(proof_reader(proof_ref), "durable delivery proof")
    if len(proof_key) > 1_000_000:
        raise FactualDeliveryProofEvidenceError("delivery proof exceeds audit integer budget")
    outer = _parts_checked(proof_key, DELIVERY_PROOF_IDENTITY, "delivery proof")
    if len(outer) != 5 or outer[0] != (4,):
        raise FactualDeliveryProofEvidenceError("durable proof is not factual kind 4")
    if outer[1] != _key(provenance.get("input_source"), "input source"):
        raise FactualDeliveryProofEvidenceError("proof input source differs")
    if outer[2] != _key(provenance.get("input_observation"), "input Observation"):
        raise FactualDeliveryProofEvidenceError("proof input Observation differs")
    trace = provenance.get("trace")
    if not isinstance(trace, Mapping) or not isinstance(trace.get("response_plan"), Mapping):
        raise FactualDeliveryProofEvidenceError("captured ResponsePlan is missing")
    plan = trace["response_plan"]
    if outer[3] != _key(plan.get("response_act"), "ResponseAct"):
        raise FactualDeliveryProofEvidenceError("proof ResponseAct differs")
    factual = _parts_checked(outer[4], 4, "factual proof body")
    if len(factual) != 6:
        raise FactualDeliveryProofEvidenceError("factual proof lacks ordered claim/audit records")
    claims = plan.get("claim_refs")
    if not isinstance(claims, list) or not claims:
        raise FactualDeliveryProofEvidenceError("captured plan claims are missing")
    plan_claims = tuple(_key(item, "plan claim") for item in claims)
    target_claim = _key(provenance.get("expected_claim"), "expected claim")
    if len(plan_claims) == 1:
        if factual[0] != target_claim:
            raise FactualDeliveryProofEvidenceError("proof target claim differs")
    else:
        if target_claim not in plan_claims:
            raise FactualDeliveryProofEvidenceError("proof target claim is outside the plan")
        try:
            state = trace["state_key"]
            scalars = state["scalars"]
            if (type(scalars) is not list or len(scalars) != 17
                    or any(type(item) is not int for item in scalars)):
                raise ValueError("invalid QueryState scalars")
            primary = GraphQueryTraceEvidence.from_integer_artifact(trace).resolve_ref(
                scalars[11], "best claim")
        except (KeyError, TypeError, ValueError) as error:
            raise FactualDeliveryProofEvidenceError("multi-claim primary lacks saved QueryState proof") from error
        if factual[0] != primary or primary not in plan_claims:
            raise FactualDeliveryProofEvidenceError("proof primary differs from current QueryState")
    if tuple(_parts_checked(factual[2], 1, "proof claim set")) != plan_claims:
        raise FactualDeliveryProofEvidenceError("proof claim set differs from ResponsePlan")
    ordered = _parts_checked(factual[1], 5, "ordered proof claims")
    ordered_claims = []
    for ordinal, record in enumerate(ordered, 1):
        values = _parts_checked(record, ordinal, "ordered claim")
        if len(values) != 1:
            raise FactualDeliveryProofEvidenceError("ordered claim record is incomplete")
        ordered_claims.append(values[0])
    if (len(ordered_claims) != len(plan_claims)
            or set(ordered_claims) != set(plan_claims)
            or len(set(ordered_claims)) != len(ordered_claims)):
        raise FactualDeliveryProofEvidenceError("proof ordered claim set differs")
    claim_audit = _parts_checked(factual[3], 5, "per-claim audit")
    audit_map = {}
    for record in claim_audit:
        fields = _parts_checked(record, 1, "per-claim audit row")
        if len(fields) != 2 or fields[0] in audit_map:
            raise FactualDeliveryProofEvidenceError("per-claim audit is duplicated or incomplete")
        audit_map[fields[0]] = _parts_checked(fields[1], 1, "per-claim evidence")
    plan_audit = plan.get("claim_evidence_refs")
    if not isinstance(plan_audit, list):
        raise FactualDeliveryProofEvidenceError("plan claim evidence is missing")
    expected_audit = {}
    for row in plan_audit:
        if not isinstance(row, Mapping) or not isinstance(row.get("evidence"), list):
            raise FactualDeliveryProofEvidenceError("plan claim evidence is invalid")
        claim = _key(row.get("claim"), "plan evidence claim")
        if claim in expected_audit:
            raise FactualDeliveryProofEvidenceError("plan claim evidence is duplicated")
        expected_audit[claim] = tuple(_key(ref, "plan claim evidence")
                                      for ref in row["evidence"])
    if audit_map != expected_audit or set(audit_map) != set(plan_claims):
        raise FactualDeliveryProofEvidenceError("proof claim evidence differs from ResponsePlan")
    slots = _parts_checked(factual[4], 2, "proof slots")
    plan_slots = plan.get("slot_sequence")
    if not isinstance(plan_slots, list) or len(slots) != len(plan_slots):
        raise FactualDeliveryProofEvidenceError("proof slot count differs")
    slot_hypotheses = []
    for ordinal, (slot_record, planned) in enumerate(zip(slots, plan_slots), 1):
        if not isinstance(planned, Mapping):
            raise FactualDeliveryProofEvidenceError("captured plan slot is invalid")
        fields = _parts_checked(slot_record, ordinal, "proof slot")
        if len(fields) not in {5, 6}:
            raise FactualDeliveryProofEvidenceError("proof slot lacks required fields")
        expected = (
            _key(planned.get("role"), "plan slot role"),
            _key(planned.get("filler"), "plan slot filler"),
            (planned.get("required"), planned.get("source_hash")),
            _key(planned.get("token_digest"), "plan slot token digest"),
            _key(planned.get("allowed_node_kinds"), "plan slot allowed kinds"),
        )
        if fields[:5] != expected:
            raise FactualDeliveryProofEvidenceError("proof slot differs from ResponsePlan")
        if len(fields) == 6:
            slot_hypotheses.append(fields[5])
    locators = _parts_checked(factual[5], 3, "proof evidence locators")
    plan_refs = plan.get("evidence_refs")
    if not isinstance(plan_refs, list) or len(locators) != len(plan_refs):
        raise FactualDeliveryProofEvidenceError("proof evidence count differs")
    recovered = tuple(_key(evidence_reader(locator), "durable evidence")
                      for locator in locators)
    if recovered != tuple(_key(item, "plan evidence") for item in plan_refs):
        raise FactualDeliveryProofEvidenceError("proof evidence differs from ResponsePlan")
    summary = provenance.get("delivery_summary")
    if (not isinstance(summary, Mapping)
            or summary.get("claim_count") != len(plan_claims)
            or summary.get("evidence_ref_count") != len(locators)
            or summary.get("adopted_slots") != len(slots)):
        raise FactualDeliveryProofEvidenceError("delivery summary/proof count drift")
    return {
        "format": "PURE_INTEGER_FACTUAL_DELIVERY_PROOF_AUDIT_V1",
        "status": "MATCHED_DURABLE_PROOF_CONTENT_TO_CAPTURED_PLAN",
        "proof_ref": list(proof_ref),
        "claim_count": len(plan_claims), "slot_count": len(slots),
        "primary_claim": list(factual[0]),
        "evidence_locator_count": len(locators),
        "slot_hypothesis_count": len(slot_hypotheses),
        "episode_output_join": "UNKNOWN_requires_output_Observation_Episode_audit",
        "ohe_exact_join": "UNKNOWN_requires_Memory_declaration_and_QueryState_audit",
        "free_dialogue_complete": 0,
    }
