"""派生 recovery-v7 context-scoped local TRAIN-only LOSO 记录。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    derive_normalization_recovery_v5_loso_execution,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_execution import (
    CONTEXT_LOCAL_TARGET_SCOPE,
    execute_context_scoped_local_transfer,
    prepare_context_local_execution,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


CONTEXT_LOCAL_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_LOSO_V1")


def _sha256(payload: bytes) -> str:
    """返回规范记录摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 context local LOSO text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 id。"""
    return _sha256(canonical_json_bytes(identity))


def _outcome(observation: dict[str, object], output: str) -> str:
    """按 held-out TRAIN label 评分，label 不参与执行选择。"""
    if output == observation["output_text"]:
        return "EXACT"
    if output == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _validated_plan_index(
        plans: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """核验 variable plan identity 与 TRAIN observations 闭合。"""
    plan_by_id = {}
    for plan in plans:
        observation_id = plan.get("observation_id") \
            if isinstance(plan, dict) else None
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or observation_id in plan_by_id):
            raise BroadQaExternalDataError(
                "v7 context local variable plan identity 漂移")
        plan_by_id[observation_id] = plan
    observation_ids = {str(item["observation_id"]) for item in observations}
    if (len(observation_ids) != len(observations)
            or any(key not in observation_ids for key in plan_by_id)):
        raise BroadQaExternalDataError(
            "v7 context local plan/observation 未闭合")
    return plan_by_id


def _family_loso_record(
        *,
        held_out: str,
        material: dict[str, object],
        family_plans: tuple[dict[str, object], ...],
        observation_by_id: dict[str, dict[str, object]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            dict[str, object],
            Counter,
        ]:
    """执行一个已重学 LOSO 方向并形成无 surface 摘要。"""
    context, representations, representation_summary = (
        prepare_context_local_execution(
            program=material["program"],
            outputs=material["outputs"],
            training_groups=material["training_groups"],
            held_out_source_family=held_out,
        ))
    outcomes = Counter()
    decisions = Counter()
    result_rows = []
    counters = Counter()
    for plan in family_plans:
        observation = observation_by_id[str(plan["observation_id"])]
        indexed_result = execute_context_scoped_local_transfer(
            observation=observation, plan=plan,
            context=context, indexed=True)
        reference_result = execute_context_scoped_local_transfer(
            observation=observation, plan=plan,
            context=context, indexed=False)
        equal = int(indexed_result == reference_result)
        counters["indexed_reference_mismatch_count"] += int(not equal)
        actual = str(indexed_result["output_text"])
        outcome = _outcome(observation, actual)
        outcomes[outcome] += 1
        decisions[str(indexed_result["decision"])] += 1
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
    for observation in material["held_out_observations"]:
        if (observation.get("identity_preservation") != 1
                or not observation.get("structure_tokens")):
            continue
        counters["identity_probe_count"] += 1
        left = execute_context_scoped_local_transfer(
            observation=observation, plan=None,
            context=context, indexed=True)
        right = execute_context_scoped_local_transfer(
            observation=observation, plan=None,
            context=context, indexed=False)
        counters["indexed_reference_mismatch_count"] += int(left != right)
        counters["identity_false_change_count"] += int(
            left["output_text"] != observation["input_text"])
        counters["structure_token_execution_mismatch_count"] += int(
            left["structure_token_mismatch_count"])
        counters["partial_commit_count"] += int(left["partial_commit_count"])
    result_rows.sort(key=lambda item: str(item["held_out_observation_id"]))
    identity = {
        "held_out_source_family": held_out,
        "subset_protocol_manifest_sha256": material[
            "subset_protocol_manifest_sha256"],
        "target_policy_scope": CONTEXT_LOCAL_TARGET_SCOPE,
    }
    facility_pass = int(
        counters["indexed_reference_mismatch_count"] == 0
        and counters["structure_token_execution_mismatch_count"] == 0
        and counters["partial_commit_count"] == 0)
    capability_status = (
        "PASS" if facility_pass and outcomes["WRONG"] == 0
        and outcomes["EXACT"] > 0
        and counters["identity_false_change_count"] == 0
        else "NE" if facility_pass and outcomes["WRONG"] == 0
        and counters["identity_false_change_count"] == 0
        else "FAIL")
    record = {
        **identity,
        "capability_status": capability_status,
        "decision_counts": {
            key: decisions[key] for key in sorted(decisions)},
        "format_version": 1,
        "identity_false_change_count": counters[
            "identity_false_change_count"],
        "identity_probe_count": counters["identity_probe_count"],
        "indexed_reference_mismatch_count": counters[
            "indexed_reference_mismatch_count"],
        "loso_id": _record_id(identity),
        "outcome_counts": {
            key: outcomes[key] for key in ("EXACT", "UNKNOWN", "WRONG")},
        "partial_commit_count": counters["partial_commit_count"],
        "record_kind": CONTEXT_LOCAL_LOSO_KIND,
        "representation_summary": representation_summary,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "structure_token_execution_mismatch_count": counters[
            "structure_token_execution_mismatch_count"],
        "variable_plan_count": len(family_plans),
    }
    counters["exact_count"] = outcomes["EXACT"]
    counters["unknown_count"] = outcomes["UNKNOWN"]
    counters["wrong_count"] = outcomes["WRONG"]
    counters["variable_plan_count"] = len(family_plans)
    return representations, record, counters


def derive_context_scoped_local_loso(
        *,
        protocol_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        plans: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """四方向各重学一次并执行 variable structure TRAIN-only LOSO。"""
    _validated_plan_index(plans, observations)
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    rule_records = []
    loso_records = []
    aggregate = Counter()
    for held_out in V5_SOURCE_FAMILIES:
        material = derive_normalization_recovery_v5_loso_execution(
            protocol_manifest_sha256=protocol_manifest_sha256,
            observations=observations,
            fragments=fragments,
            held_out_source_family=held_out,
            include_reference=False,
        )
        representations, record, counters = _family_loso_record(
            held_out=held_out,
            material=material,
            family_plans=tuple(
                plan for plan in plans if plan["source_family"] == held_out),
            observation_by_id=observation_by_id,
        )
        rule_records.extend(representations)
        loso_records.append(record)
        aggregate.update(counters)
    if aggregate["variable_plan_count"] != len(plans):
        raise BroadQaExternalDataError(
            "v7 context local LOSO plan denominator 未闭合")
    facility_pass = int(
        aggregate["indexed_reference_mismatch_count"] == 0
        and aggregate["structure_token_execution_mismatch_count"] == 0
        and aggregate["partial_commit_count"] == 0)
    capability_outcome = (
        "PASS_NONZERO_VARIABLE_EXACT" if facility_pass
        and aggregate["wrong_count"] == 0
        and aggregate["exact_count"] > 0
        and aggregate["identity_false_change_count"] == 0
        else "NE_ZERO_VARIABLE_EXACT" if facility_pass
        and aggregate["wrong_count"] == 0
        and aggregate["identity_false_change_count"] == 0
        else "FAIL_HARD_GATE")
    rule_records.sort(key=lambda item: (
        str(item["held_out_source_family"]),
        str(item["representation_id"])))
    loso_records.sort(key=lambda item: str(item["held_out_source_family"]))
    return tuple(rule_records), tuple(loso_records), {
        "capability_outcome": capability_outcome,
        "exact_count": aggregate["exact_count"],
        "facility_outcome": "PASS" if facility_pass else "FAIL",
        "identity_false_change_count": aggregate[
            "identity_false_change_count"],
        "identity_probe_count": aggregate["identity_probe_count"],
        "indexed_reference_mismatch_count": aggregate[
            "indexed_reference_mismatch_count"],
        "loso_family_count": len(V5_SOURCE_FAMILIES),
        "partial_commit_count": aggregate["partial_commit_count"],
        "structure_token_execution_mismatch_count": aggregate[
            "structure_token_execution_mismatch_count"],
        "unknown_count": aggregate["unknown_count"],
        "variable_plan_count": aggregate["variable_plan_count"],
        "wrong_count": aggregate["wrong_count"],
    }


__all__ = [
    "CONTEXT_LOCAL_LOSO_KIND",
    "derive_context_scoped_local_loso",
]
