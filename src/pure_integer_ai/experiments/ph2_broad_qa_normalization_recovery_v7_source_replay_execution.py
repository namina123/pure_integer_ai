"""执行 recovery-v7 exact source identity segment transaction。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout_for_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_execution import (
    _input_layout,
    _overlap_count,
    _rebuild_with_proposals,
    _segment_spans,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_program import (
    source_commitment_identity_sha256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


def _sha256(payload: bytes) -> str:
    """返回规范执行结果 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _route_candidates(
        *,
        program: dict[str, object],
        observation: dict[str, object],
        source_identity: str,
        text: str,
        position: int,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """返回 exact source identity 下 indexed 或 reference routes。"""
    family = str(observation["source_family"])
    policy = str(observation["source_policy_scope"])
    if indexed:
        return program["buckets"].get(
            (family, policy, source_identity, text[position]), ())
    return tuple(
        route for route in program["reference_routes_by_family"].get(
            family, ())
        if route["source_family"] == family
        and route["source_policy_scope"] == policy
        and route["source_commitment_sha256"] == source_identity
        and str(route["input_text"])[0] == text[position])


def execute_source_replay_segment_transaction(
        *,
        observation: dict[str, object],
        plan: dict[str, object] | None,
        program: dict[str, object],
        indexed: bool,
        ) -> dict[str, object]:
    """在 structure segments 内执行 exact seen-source replay 原子事务。"""
    if type(indexed) is not bool:
        raise BroadQaExternalDataError("v7 source replay interpreter 非法")
    text = str(observation.get("input_text", ""))
    tokens = tuple(observation.get("structure_tokens", ()))
    if plan is not None and plan.get("representation_eligible") != 1:
        payload = {
            "decision": "UNKNOWN_PLAN_INELIGIBLE",
            "input_text": text,
            "output_text": text,
            "partial_commit_count": 0,
            "proposal_count": 0,
            "structure_token_mismatch_count": 0,
        }
        return {**payload, "result_sha256": _sha256(
            canonical_json_bytes(payload))}
    layout = _input_layout(observation, plan)
    spans = _segment_spans(layout)
    source_identity = source_commitment_identity_sha256(observation)
    raw_proposals = []
    counters = Counter()
    for segment_ordinal, (segment_start, segment_end) in enumerate(spans):
        for start in range(segment_start, segment_end):
            for route in _route_candidates(
                    program=program,
                    observation=observation,
                    source_identity=source_identity,
                    text=text,
                    position=start,
                    indexed=indexed):
                phrase = str(route["input_text"])
                end = start + len(phrase)
                if (route["fragment_kind"] == "WHOLE_INPUT"
                        or end > segment_end or text[start:end] != phrase):
                    continue
                counters["route_match_count"] += 1
                veto_key = (
                    str(observation["source_family"]),
                    str(observation["source_policy_scope"]),
                    source_identity,
                    phrase,
                )
                if veto_key in program["identity_veto_keys"]:
                    counters["identity_veto_count"] += 1
                    continue
                raw_proposals.append({
                    "input_end": end,
                    "input_start": start,
                    "output_text": str(route["output_text"]),
                    "route_id": str(route["route_id"]),
                    "segment_ordinal": segment_ordinal,
                })
    grouped = defaultdict(list)
    for proposal in raw_proposals:
        grouped[(
            int(proposal["input_start"]),
            int(proposal["input_end"]),
            str(proposal["output_text"]),
        )].append(proposal)
    proposals = tuple(sorted(({
        **values[0],
        "route_ids": sorted(str(item["route_id"]) for item in values),
    } for values in grouped.values()), key=lambda item: (
        int(item["input_start"]), int(item["input_end"]),
        str(item["output_text"]))))
    overlap_count = _overlap_count(proposals)
    decision = "COMMIT"
    if not proposals:
        decision = "UNKNOWN_NO_SOURCE_IDENTITY_ROUTE"
    elif overlap_count:
        decision = "UNKNOWN_OVERLAPPING_OR_CONFLICTING_ROUTE"
    output = text if decision != "COMMIT" else _rebuild_with_proposals(
        text, proposals)
    structure_mismatch = 0
    if decision == "COMMIT":
        try:
            output_layout = localization_structure_layout_for_tokens(
                output, tokens)
            structure_mismatch = int(
                output_layout["raw_tokens"] != layout["raw_tokens"])
        except BroadQaExternalDataError:
            structure_mismatch = 1
        if structure_mismatch or output == text:
            output = text
            decision = (
                "UNKNOWN_STRUCTURE_TOKEN_MISMATCH" if structure_mismatch
                else "UNKNOWN_IDENTITY_ROUTE")
    payload = {
        "conflict_resolved_by_exact_source_identity_count": (
            len(proposals) if decision == "COMMIT" else 0),
        "decision": decision,
        "identity_veto_count": counters["identity_veto_count"],
        "input_text": text,
        "output_text": output,
        "overlap_count": overlap_count,
        "partial_commit_count": 0,
        "proposal_count": len(proposals),
        "route_match_count": counters["route_match_count"],
        "source_commitment_sha256": source_identity,
        "structure_token_mismatch_count": structure_mismatch,
    }
    return {**payload, "result_sha256": _sha256(
        canonical_json_bytes(payload))}


__all__ = ["execute_source_replay_segment_transaction"]
