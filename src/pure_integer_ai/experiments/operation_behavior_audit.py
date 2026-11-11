"""Counterfactual operation behavior admission for integer QueryBridge traces.

Operation identifiers, modes, cues, and metadata do not establish distinct
behavior. This read-only contract compares already saved paired traces. A
caller-supplied context key is not independent proof of graph identity or of
which relation was actually edited. It therefore reports a paired observation,
never a fully qualified operation-family PASS. No graph is replayed here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_DIALOGUE_UNKNOWN,
    TERMINATION_NO_FRONTIER,
)
from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceContractError,
    GraphQueryTraceEvidence,
)


class OperationBehaviorEvidenceError(ValueError):
    """The paired traces cannot prove an operation behavior distinction."""


_FAIL_CLOSED = frozenset({
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_DIALOGUE_UNKNOWN,
    TERMINATION_NO_FRONTIER,
})


def _key(value: object, field: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise OperationBehaviorEvidenceError(f"{field} requires a nonempty integer key")
    return tuple(value)


def _operation_keys(trace: Mapping[str, Any], field: str) -> tuple[tuple[int, ...], ...]:
    value = trace.get(field)
    if not isinstance(value, list):
        raise OperationBehaviorEvidenceError(f"{field} is missing")
    try:
        result = tuple(_key(item, field) for item in value)
    except OperationBehaviorEvidenceError:
        raise
    if len(set(result)) != len(result):
        raise OperationBehaviorEvidenceError(f"{field} contains duplicates")
    return result


def _plan_projection(trace: Mapping[str, Any], evidence: GraphQueryTraceEvidence) -> tuple:
    plan = trace.get("response_plan")
    if plan is None:
        return ()
    if not isinstance(plan, Mapping):
        raise OperationBehaviorEvidenceError("response_plan is invalid")
    claims = plan.get("claim_refs")
    if not isinstance(claims, list):
        raise OperationBehaviorEvidenceError("response_plan claims are missing")
    claim_keys = tuple(_key(item, "response_plan.claim_refs") for item in claims)
    response_act = _key(plan.get("response_act"), "response_plan.response_act")
    _key(plan.get("stable_key"), "response_plan.stable_key")
    claim_ranges = plan.get("claim_slot_ranges", [])
    if not isinstance(claim_ranges, list):
        raise OperationBehaviorEvidenceError("response_plan claim ranges are invalid")
    ranges = []
    for row in claim_ranges:
        if not isinstance(row, Mapping):
            raise OperationBehaviorEvidenceError("response_plan claim range is invalid")
        ranges.append((_key(row.get("claim"), "claim range"), row.get("start"), row.get("end")))
    if any(type(start) is not int or type(end) is not int or start < 0 or end < start
           for _, start, end in ranges):
        raise OperationBehaviorEvidenceError("response_plan claim range bounds are invalid")
    generation = trace.get("generation")
    generated_surface = None
    if generation is not None:
        if not isinstance(generation, Mapping):
            raise OperationBehaviorEvidenceError("generation is invalid")
        surface = generation.get("surface_values")
        if isinstance(surface, Mapping):
            if set(surface) != {"integer_key_ref"} or type(surface["integer_key_ref"]) is not int:
                raise OperationBehaviorEvidenceError("generation surface ref is invalid")
            try:
                generated_surface = evidence.resolve_ref(
                    surface["integer_key_ref"], "generated surface")
            except GraphQueryTraceContractError as error:
                raise OperationBehaviorEvidenceError(str(error)) from error
        elif isinstance(surface, list) and all(type(item) is int for item in surface):
            generated_surface = tuple(surface)
        else:
            raise OperationBehaviorEvidenceError("generation surface evidence is missing")
    response_surface = trace.get("response_surface")
    if not isinstance(response_surface, str) or not response_surface:
        raise OperationBehaviorEvidenceError("response surface boundary is missing")
    slots = plan.get("slot_sequence")
    if not isinstance(slots, list):
        raise OperationBehaviorEvidenceError("response_plan slot sequence is missing")
    slot_organization = []
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise OperationBehaviorEvidenceError("response_plan slot is invalid")
        slot_organization.append((
            _key(slot.get("role"), "slot role"),
            _key(slot.get("filler"), "slot filler"),
            slot.get("required"),
        ))
    if any(type(required) is not int or required not in {0, 1}
           for _, _, required in slot_organization):
        raise OperationBehaviorEvidenceError("response_plan required-slot bit is invalid")
    return (claim_keys, response_act, (tuple(ranges), tuple(slot_organization)), generated_surface,
            tuple(map(ord, response_surface)))


@dataclass(frozen=True, slots=True)
class OperationBehaviorAudit:
    status: Literal[
        "OBSERVED_PAIR_DIFFERENCE_CAUSAL_PROVENANCE_UNKNOWN",
        "BLOCKED_OPERATION_BEHAVIOR_NO_DIFFERENCE",
    ]
    mode: Literal["REMOVED", "SWAPPED"]
    baseline_operation_keys: tuple[tuple[int, ...], ...]
    counterfactual_operation_keys: tuple[tuple[int, ...], ...]
    changed_fields: tuple[str, ...]
    baseline_termination: int
    counterfactual_termination: int
    fail_closed: bool
    context_qualification: str = "UNKNOWN_requires_independent_semantic_graph_identity"
    mutation_qualification: str = "UNKNOWN_requires_actual_operation_relation_edit_receipt"
    operation_family_qualification: str = "UNKNOWN_pair_observation_is_not_family_training"
    free_dialogue_complete: int = 0


def audit_operation_counterfactual(
        baseline: Mapping[str, Any], counterfactual: Mapping[str, Any], *,
        context_key: tuple[int, ...], mode: Literal["REMOVED", "SWAPPED"],
        swapped_operation_key: tuple[int, ...] | None = None,
        ) -> OperationBehaviorAudit:
    """Compare two same-context traces and refuse metadata-only distinctions."""
    if mode not in {"REMOVED", "SWAPPED"}:
        raise OperationBehaviorEvidenceError("counterfactual mode must be REMOVED or SWAPPED")
    context = _key(context_key, "context_key")
    if not isinstance(baseline, Mapping) or not isinstance(counterfactual, Mapping):
        raise OperationBehaviorEvidenceError("paired traces must be mappings")
    if baseline.get("context_key") != list(context) or counterfactual.get("context_key") != list(context):
        raise OperationBehaviorEvidenceError("paired traces do not share the frozen context")
    try:
        base_trace = GraphQueryTraceEvidence(baseline)
        counter_trace = GraphQueryTraceEvidence(counterfactual)
        base_counts = base_trace.counts()
        counter_counts = counter_trace.counts()
    except GraphQueryTraceContractError as error:
        raise OperationBehaviorEvidenceError(str(error)) from error
    base_keys = _operation_keys(baseline, "operation_keys")
    counter_keys = _operation_keys(counterfactual, "operation_keys")
    if tuple(key for key, _ in base_trace.operation_contracts()) != base_keys:
        raise OperationBehaviorEvidenceError("baseline operation/contract consumption drift")
    if tuple(key for key, _ in counter_trace.operation_contracts()) != counter_keys:
        raise OperationBehaviorEvidenceError("counterfactual operation/contract consumption drift")
    if mode == "REMOVED":
        if counter_keys:
            raise OperationBehaviorEvidenceError("removed counterfactual still consumes an operation")
    else:
        replacement = _key(swapped_operation_key, "swapped_operation_key")
        if counter_keys != (replacement,):
            raise OperationBehaviorEvidenceError("swapped counterfactual operation identity drift")
        if replacement in base_keys:
            raise OperationBehaviorEvidenceError("swapped operation is not distinct")
    if len(base_keys) != 1:
        raise OperationBehaviorEvidenceError("baseline must contain exactly one operation")
    if baseline.get("termination") != TERMINATION_ANSWER_CLOSED:
        raise OperationBehaviorEvidenceError("baseline operation did not close an answer")
    base_plan = _plan_projection(baseline, base_trace)
    if not base_plan:
        raise OperationBehaviorEvidenceError("baseline ResponsePlan is missing")
    counter_plan = _plan_projection(counterfactual, counter_trace)
    changed = []
    for label, left, right in (
            ("termination", baseline.get("termination"), counterfactual.get("termination")),
            ("claims", base_plan[:1], counter_plan[:1]),
            ("response_act", base_plan[1:2], counter_plan[1:2]),
            ("response_plan", base_plan[2:3], counter_plan[2:3]),
            ("generation_surface", base_plan[3:4], counter_plan[3:4]),
            ("response_surface", base_plan[4:5], counter_plan[4:5]),
            ("hop_counts", base_counts, counter_counts)):
        if left != right:
            changed.append(label)
    termination = counterfactual.get("termination")
    if type(termination) is not int:
        raise OperationBehaviorEvidenceError("counterfactual termination is missing")
    fail_closed = termination in _FAIL_CLOSED
    if (termination == TERMINATION_ANSWER_CLOSED and not counter_plan):
        raise OperationBehaviorEvidenceError("counterfactual closed without ResponsePlan")
    if fail_closed and counter_plan and counter_plan[0]:
        raise OperationBehaviorEvidenceError("fail-closed counterfactual still emitted factual claims")
    status = ("OBSERVED_PAIR_DIFFERENCE_CAUSAL_PROVENANCE_UNKNOWN"
              if changed and ("claims" in changed or "response_plan" in changed or fail_closed)
              else "BLOCKED_OPERATION_BEHAVIOR_NO_DIFFERENCE")
    return OperationBehaviorAudit(
        status=status, mode=mode,
        baseline_operation_keys=base_keys,
        counterfactual_operation_keys=counter_keys,
        changed_fields=tuple(changed),
        baseline_termination=baseline.get("termination"),
        counterfactual_termination=termination,
        fail_closed=fail_closed,
    )
