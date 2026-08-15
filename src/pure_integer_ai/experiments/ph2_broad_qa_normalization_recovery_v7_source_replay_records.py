"""派生 recovery-v7 source identity replay full-pack 与 LOSO 记录。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
    derive_normalization_recovery_v5_groups,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_execution import (
    execute_source_replay_segment_transaction,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_program import (
    SOURCE_REPLAY_TARGET_SCOPE,
    derive_source_replay_program,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


SOURCE_REPLAY_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_LOSO_V1")


def _sha256(payload: bytes) -> str:
    """返回规范记录或 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 source replay LOSO text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 id。"""
    return _sha256(canonical_json_bytes(identity))


def _outcome(observation: dict[str, object], output: str) -> str:
    """按 held-out TRAIN label 评分，label 不参与 route 选择。"""
    if output == observation["output_text"]:
        return "EXACT"
    if output == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _indexes(
        observations: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
        ]:
    """核验 observations/plans identity 与来源闭合。"""
    observation_by_id = {}
    for observation in observations:
        observation_id = observation.get("observation_id") \
            if isinstance(observation, dict) else None
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or observation_id in observation_by_id):
            raise BroadQaExternalDataError(
                "v7 source replay observation identity 漂移")
        observation_by_id[observation_id] = observation
    plan_by_id = {}
    for plan in plans:
        observation_id = plan.get("observation_id") \
            if isinstance(plan, dict) else None
        owner = observation_by_id.get(str(observation_id))
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or observation_id in plan_by_id or owner is None
                or plan.get("source_family") != owner.get("source_family")):
            raise BroadQaExternalDataError(
                "v7 source replay plan/observation identity 漂移")
        plan_by_id[observation_id] = plan
    if not observation_by_id or not plan_by_id:
        raise BroadQaExternalDataError(
            "v7 source replay observation/plan 为空")
    return observation_by_id, plan_by_id


def _family_record(
        *,
        held_out: str,
        training_observations: tuple[dict[str, object], ...],
        training_fragments: tuple[dict[str, object], ...],
        held_out_observations: tuple[dict[str, object], ...],
        family_plans: tuple[dict[str, object], ...],
        observation_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], Counter]:
    """重派生三来源 conflict routes 并执行 held-out family。"""
    training_groups = derive_normalization_recovery_v5_groups(
        training_fragments)
    program, _representations, route_summary = derive_source_replay_program(
        observations=training_observations,
        fragments=training_fragments,
        groups=training_groups,
    )
    outcomes = Counter()
    decisions = Counter()
    counters = Counter()
    result_rows = []
    for plan in family_plans:
        observation = observation_by_id[str(plan["observation_id"])]
        indexed_result = execute_source_replay_segment_transaction(
            observation=observation,
            plan=plan,
            program=program,
            indexed=True,
        )
        reference_result = execute_source_replay_segment_transaction(
            observation=observation,
            plan=plan,
            program=program,
            indexed=False,
        )
        equal = int(indexed_result == reference_result)
        counters["indexed_reference_mismatch_count"] += int(not equal)
        actual = str(indexed_result["output_text"])
        outcome = _outcome(observation, actual)
        outcomes[outcome] += 1
        decisions[str(indexed_result["decision"])] += 1
        counters["route_match_count"] += int(
            indexed_result.get("route_match_count", 0))
        counters["structure_token_execution_mismatch_count"] += int(
            indexed_result["structure_token_mismatch_count"])
        counters["partial_commit_count"] += int(
            indexed_result["partial_commit_count"])
        result_rows.append({
            "actual_output_sha256": _text_sha256(actual),
            "decision": indexed_result["decision"],
            "held_out_observation_id": observation["observation_id"],
            "indexed_reference_equal": equal,
            "outcome": outcome,
            "result_sha256": indexed_result["result_sha256"],
        })
    for observation in held_out_observations:
        if (observation.get("identity_preservation") != 1
                or not observation.get("structure_tokens")):
            continue
        counters["identity_probe_count"] += 1
        left = execute_source_replay_segment_transaction(
            observation=observation,
            plan=None,
            program=program,
            indexed=True,
        )
        right = execute_source_replay_segment_transaction(
            observation=observation,
            plan=None,
            program=program,
            indexed=False,
        )
        counters["indexed_reference_mismatch_count"] += int(left != right)
        counters["identity_false_change_count"] += int(
            left["output_text"] != observation["input_text"])
        counters["route_match_count"] += int(left.get("route_match_count", 0))
        counters["structure_token_execution_mismatch_count"] += int(
            left["structure_token_mismatch_count"])
        counters["partial_commit_count"] += int(left["partial_commit_count"])
    held_out_route_count = sum(
        route["source_family"] == held_out
        for route in program["reference_routes"])
    result_rows.sort(key=lambda item: str(item["held_out_observation_id"]))
    facility_pass = int(
        counters["indexed_reference_mismatch_count"] == 0
        and counters["structure_token_execution_mismatch_count"] == 0
        and counters["partial_commit_count"] == 0
        and route_summary["seen_source_self_replay_counts"]["WRONG"] == 0)
    capability_status = (
        "PASS" if facility_pass and outcomes["WRONG"] == 0
        and outcomes["EXACT"] > 0
        and counters["identity_false_change_count"] == 0
        else "NE" if facility_pass and outcomes["WRONG"] == 0
        and counters["identity_false_change_count"] == 0
        else "FAIL")
    identity = {
        "held_out_source_family": held_out,
        "target_policy_scope": SOURCE_REPLAY_TARGET_SCOPE,
        "training_source_families": sorted({
            str(item["source_family"]) for item in training_observations}),
    }
    record = {
        **identity,
        "capability_status": capability_status,
        "decision_counts": {
            key: decisions[key] for key in sorted(decisions)},
        "format_version": 1,
        "held_out_family_compiled_route_count": held_out_route_count,
        "identity_false_change_count": counters[
            "identity_false_change_count"],
        "identity_probe_count": counters["identity_probe_count"],
        "indexed_reference_mismatch_count": counters[
            "indexed_reference_mismatch_count"],
        "loso_id": _record_id(identity),
        "outcome_counts": {
            key: outcomes[key] for key in ("EXACT", "UNKNOWN", "WRONG")},
        "partial_commit_count": counters["partial_commit_count"],
        "record_kind": SOURCE_REPLAY_LOSO_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "route_match_count": counters["route_match_count"],
        "route_summary": route_summary,
        "structure_token_execution_mismatch_count": counters[
            "structure_token_execution_mismatch_count"],
        "variable_plan_count": len(family_plans),
    }
    counters["exact_count"] = outcomes["EXACT"]
    counters["unknown_count"] = outcomes["UNKNOWN"]
    counters["wrong_count"] = outcomes["WRONG"]
    counters["variable_plan_count"] = len(family_plans)
    counters["held_out_family_compiled_route_count"] = held_out_route_count
    counters["seen_source_self_replay_exact_count"] = route_summary[
        "seen_source_self_replay_counts"]["EXACT"]
    counters["seen_source_self_replay_unknown_count"] = route_summary[
        "seen_source_self_replay_counts"]["UNKNOWN"]
    counters["seen_source_self_replay_wrong_count"] = route_summary[
        "seen_source_self_replay_counts"]["WRONG"]
    return record, counters


def derive_source_replay_audit_records(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生 full-pack representation 与四方向 source-family LOSO。"""
    observation_by_id, _plan_by_id = _indexes(observations, plans)
    _full_program, representations, full_summary = derive_source_replay_program(
        observations=observations,
        fragments=fragments,
        groups=groups,
    )
    loso_records = []
    aggregate = Counter()
    for held_out in V5_SOURCE_FAMILIES:
        training_observations = tuple(
            item for item in observations
            if item["source_family"] != held_out)
        held_out_observations = tuple(
            item for item in observations
            if item["source_family"] == held_out)
        training_ids = {str(item["observation_id"])
                        for item in training_observations}
        training_fragments = tuple(
            item for item in fragments
            if str(item["observation_id"]) in training_ids)
        family_plans = tuple(
            plan for plan in plans if plan["source_family"] == held_out)
        if (len({str(item["source_family"])
                 for item in training_observations}) != 3
                or not training_fragments or not held_out_observations):
            raise BroadQaExternalDataError(
                "v7 source replay LOSO source partition 漂移")
        record, counters = _family_record(
            held_out=held_out,
            training_observations=training_observations,
            training_fragments=training_fragments,
            held_out_observations=held_out_observations,
            family_plans=family_plans,
            observation_by_id=observation_by_id,
        )
        loso_records.append(record)
        aggregate.update(counters)
    if aggregate["variable_plan_count"] != len(plans):
        raise BroadQaExternalDataError(
            "v7 source replay LOSO plan denominator 未闭合")
    facility_pass = int(
        aggregate["indexed_reference_mismatch_count"] == 0
        and aggregate["structure_token_execution_mismatch_count"] == 0
        and aggregate["partial_commit_count"] == 0
        and aggregate["seen_source_self_replay_wrong_count"] == 0)
    capability_outcome = (
        "PASS_NONZERO_UNSEEN_FAMILY_EXACT" if facility_pass
        and aggregate["wrong_count"] == 0
        and aggregate["exact_count"] > 0
        and aggregate["identity_false_change_count"] == 0
        else "NE_ZERO_UNSEEN_FAMILY_EXACT" if facility_pass
        and aggregate["wrong_count"] == 0
        and aggregate["identity_false_change_count"] == 0
        else "FAIL_HARD_GATE")
    loso_records.sort(key=lambda item: str(item["held_out_source_family"]))
    return representations, tuple(loso_records), {
        "capability_outcome": capability_outcome,
        "exact_count": aggregate["exact_count"],
        "facility_outcome": "PASS" if facility_pass else "FAIL",
        "full_pack": full_summary,
        "held_out_family_compiled_route_count": aggregate[
            "held_out_family_compiled_route_count"],
        "identity_false_change_count": aggregate[
            "identity_false_change_count"],
        "identity_probe_count": aggregate["identity_probe_count"],
        "indexed_reference_mismatch_count": aggregate[
            "indexed_reference_mismatch_count"],
        "loso_family_count": len(V5_SOURCE_FAMILIES),
        "partial_commit_count": aggregate["partial_commit_count"],
        "route_match_count": aggregate["route_match_count"],
        "seen_source_self_replay_counts": {
            "EXACT": aggregate["seen_source_self_replay_exact_count"],
            "UNKNOWN": aggregate["seen_source_self_replay_unknown_count"],
            "WRONG": aggregate["seen_source_self_replay_wrong_count"],
        },
        "structure_token_execution_mismatch_count": aggregate[
            "structure_token_execution_mismatch_count"],
        "unknown_count": aggregate["unknown_count"],
        "variable_plan_count": aggregate["variable_plan_count"],
        "wrong_count": aggregate["wrong_count"],
    }


__all__ = [
    "SOURCE_REPLAY_LOSO_KIND",
    "derive_source_replay_audit_records",
]
