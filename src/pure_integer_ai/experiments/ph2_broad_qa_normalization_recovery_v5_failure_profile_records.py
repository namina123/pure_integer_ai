"""派生 recovery-v5 TRAIN-only LOSO 错误归因记录。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit import (
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    AUDIT_BUCKETS,
    LOSO_OUTCOMES,
    derive_normalization_recovery_v5_loso_execution,
    normalization_recovery_v5_result_bucket,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V5_FAILURE_CASE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_FAILURE_CASE_V1")
NORMALIZATION_RECOVERY_V5_RULE_IMPACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_RULE_IMPACT_V1")
NORMALIZATION_RECOVERY_V5_FAILURE_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_FAILURE_FAMILY_SUMMARY_V1")

_LENGTH_BUCKETS = (
    "L01",
    "L02",
    "L03_04",
    "L05_08",
    "L09_16",
    "L17_PLUS",
)


def _sha256(payload: bytes) -> str:
    """返回规范 profile identity 摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 record id。"""
    return _sha256(canonical_json_bytes(identity))


def _length_bucket(value: int) -> str:
    """把正 scalar 长度归入固定诊断桶。"""
    if type(value) is not int or value <= 0:
        raise BroadQaExternalDataError("v5 failure profile scalar 长度非法")
    if value == 1:
        return "L01"
    if value == 2:
        return "L02"
    if value <= 4:
        return "L03_04"
    if value <= 8:
        return "L05_08"
    if value <= 16:
        return "L09_16"
    return "L17_PLUS"


def _audit_summary_contract(
        *,
        protocol_manifest_sha256: str,
        audit_manifest_sha256: str,
        audit_manifest: dict[str, object],
        ) -> dict[str, object]:
    """核验 sealed audit 仅允许 facility PASS/capability FAIL 输入。"""
    protocol_sha = _sha_value(
        protocol_manifest_sha256, label="v5 profile protocol manifest")
    _sha_value(audit_manifest_sha256, label="v5 profile audit manifest")
    if not isinstance(audit_manifest, dict):
        raise BroadQaExternalDataError(
            "v5 failure profile audit contract 漂移")
    summary = audit_manifest.get("summary")
    if (audit_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND
            or audit_manifest.get("status")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS
            or audit_manifest.get("protocol_manifest_sha256") != protocol_sha
            or audit_manifest.get("formal_run_count") != 0
            or audit_manifest.get("production_enabled") != 0
            or audit_manifest.get("mastery_claimed") != 0
            or audit_manifest.get("evaluation_payload_read_count") != 0
            or not isinstance(summary, dict)
            or summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_FAIL"
            or summary.get("facility_failure_count") != 0
            or summary.get("capability_gate_pass") != 0
            or summary.get("loso_family_count") != len(V5_SOURCE_FAMILIES)
            or summary.get("source_leak_count") != 0
            or summary.get("subset_defeater_mismatch_count") != 0):
        raise BroadQaExternalDataError(
            "v5 failure profile audit contract 漂移")
    return summary


def _outcome(
        observation: dict[str, object],
        result: dict[str, object],
        ) -> str:
    """按 audit 合同重算 held-out EXACT/UNKNOWN/WRONG。"""
    actual = str(result["output_text"])
    if actual == observation["output_text"]:
        return "EXACT"
    if actual == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _selected_steps(
        *,
        result: dict[str, object],
        target_rule_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """核验并保留一次 WRONG 中实际选择的 target rule 步骤。"""
    values = []
    steps = result.get("steps")
    if not isinstance(steps, list):
        raise BroadQaExternalDataError("v5 failure profile runtime steps 漂移")
    for ordinal, step in enumerate(steps):
        if not isinstance(step, dict):
            raise BroadQaExternalDataError(
                "v5 failure profile runtime step 非字典")
        rule_id = str(step.get("rule_id", ""))
        if not rule_id:
            continue
        rule = target_rule_by_id.get(rule_id)
        if (rule is None
                or rule.get("candidate_scope_kind") != "TARGET_CROSS_FAMILY"
                or step.get("candidate_scope_kind") != "TARGET_CROSS_FAMILY"
                or step.get("rule_class") != rule["rule_class"]
                or step.get("fragment_kind") != rule["fragment_kind"]
                or step.get("source_execution_family") != ""):
            raise BroadQaExternalDataError(
                "v5 failure profile selected rule scope 漂移")
        blocked = step.get("blocked_defeater_ids")
        if (not isinstance(blocked, list)
                or any(not isinstance(item, str) for item in blocked)):
            raise BroadQaExternalDataError(
                "v5 failure profile blocked defeater 漂移")
        values.append({
            "blocked_defeater_ids": sorted(set(blocked)),
            "fragment_kind": rule["fragment_kind"],
            "input_end": step["input_end"],
            "input_start": step["input_start"],
            "mode": step["mode"],
            "rule_class": rule["rule_class"],
            "rule_id": rule_id,
            "step_ordinal": ordinal,
        })
    return tuple(values)


def _wrong_case(
        *,
        audit_manifest_sha256: str,
        held_out_source_family: str,
        observation: dict[str, object],
        result: dict[str, object],
        target_rule_by_id: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """物化一条 held-out WRONG 的规则选择与结构诊断。"""
    selected = _selected_steps(
        result=result, target_rule_by_id=target_rule_by_id)
    if not selected:
        raise BroadQaExternalDataError(
            "v5 failure profile WRONG 无 selected target rule")
    selected_rule_ids = sorted({str(item["rule_id"]) for item in selected})
    blocked_ids = set()
    for step in result["steps"]:
        blocked = step.get("blocked_defeater_ids")
        if (not isinstance(blocked, list)
                or any(not isinstance(item, str) for item in blocked)):
            raise BroadQaExternalDataError(
                "v5 failure profile blocked defeater 漂移")
        blocked_ids.update(blocked)
    identity = {
        "audit_manifest_sha256": audit_manifest_sha256,
        "held_out_observation_id": observation["observation_id"],
        "held_out_source_family": held_out_source_family,
    }
    actual = str(result["output_text"])
    expected = str(observation["output_text"])
    input_text = str(observation["input_text"])
    identity_false_change = int(
        observation["identity_preservation"] == 1
        and actual != input_text)
    return {
        **identity,
        "actual_output_scalar_length": len(actual),
        "actual_output_sha256": _sha256(actual.encode("utf-8")),
        "blocked_defeater_ids": sorted(blocked_ids),
        "bucket": normalization_recovery_v5_result_bucket(
            observation, result),
        "case_id": _record_id(identity),
        "expected_output_scalar_length": len(expected),
        "expected_output_sha256": _sha256(expected.encode("utf-8")),
        "format_version": 1,
        "identity_false_change": identity_false_change,
        "input_scalar_length": len(input_text),
        "input_sha256": _sha256(input_text.encode("utf-8")),
        "multi_rule_wrong": int(len(selected_rule_ids) > 1),
        "outcome": "WRONG",
        "record_kind": NORMALIZATION_RECOVERY_V5_FAILURE_CASE_KIND,
        "result_sha256": result["result_sha256"],
        "selected_rule_ids": selected_rule_ids,
        "selected_rule_occurrence_count": len(selected),
        "selected_steps": list(selected),
        "structure_token_count": len(observation["structure_tokens"]),
        "wrong_kind": (
            "IDENTITY_FALSE_CHANGE"
            if identity_false_change else "NONIDENTITY_WRONG"),
    }


def _priority_contract(rule: dict[str, object]) -> str:
    """返回 rule 在 runtime 中使用的固定优先级合同。"""
    if rule["fragment_kind"] == "WHOLE_INPUT":
        return "WHOLE_INPUT_EXACT_PRECEDES_LONGEST_LOCAL_MATCH"
    return "LONGEST_MATCH_THEN_RULE_CLASS_THEN_RULE_ID"


def _rule_impact_records(
        *,
        audit_manifest_sha256: str,
        wrong_cases: tuple[dict[str, object], ...],
        rules_by_family: dict[str, dict[str, dict[str, object]]],
        ) -> tuple[tuple[dict[str, object], ...],
                   dict[tuple[str, str], set[str]]]:
    """按 held-out family+rule 聚合 WRONG 观察和选择次数。"""
    aggregates = {}
    observation_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in wrong_cases:
        family = str(case["held_out_source_family"])
        case_id = str(case["case_id"])
        selected_steps = case["selected_steps"]
        unique_ids = set(str(value) for value in case["selected_rule_ids"])
        occurrence_counts = Counter(
            str(item["rule_id"]) for item in selected_steps)
        ordinal_counts: dict[str, Counter] = defaultdict(Counter)
        for step in selected_steps:
            ordinal_counts[str(step["rule_id"])][
                str(step["step_ordinal"])] += 1
        for rule_id in sorted(unique_ids):
            key = family, rule_id
            rule = rules_by_family[family].get(rule_id)
            if rule is None:
                raise BroadQaExternalDataError(
                    "v5 failure profile impact rule 缺失")
            value = aggregates.setdefault(key, {
                "blocked_defeater_case_count": 0,
                "exclusive_wrong_observation_count": 0,
                "identity_false_change_count": 0,
                "selection_occurrence_count": 0,
                "selection_ordinal_counts": Counter(),
                "structured_wrong_observation_count": 0,
                "wrong_observation_count": 0,
            })
            value["wrong_observation_count"] += 1
            value["selection_occurrence_count"] += occurrence_counts[rule_id]
            value["exclusive_wrong_observation_count"] += int(
                len(unique_ids) == 1)
            value["identity_false_change_count"] += int(
                case["identity_false_change"])
            value["structured_wrong_observation_count"] += int(
                case["structure_token_count"] > 0)
            value["blocked_defeater_case_count"] += int(
                bool(case["blocked_defeater_ids"]))
            value["selection_ordinal_counts"].update(
                ordinal_counts[rule_id])
            observation_sets[key].add(case_id)
    records = []
    for family, rule_id in sorted(aggregates):
        rule = rules_by_family[family][rule_id]
        value = aggregates[(family, rule_id)]
        identity = {
            "audit_manifest_sha256": audit_manifest_sha256,
            "held_out_source_family": family,
            "rule_id": rule_id,
        }
        case_ids = sorted(observation_sets[(family, rule_id)])
        records.append({
            **identity,
            "authority_basis": rule["authority_basis"],
            "blocked_defeater_case_count": value[
                "blocked_defeater_case_count"],
            "candidate_scope_kind": rule["candidate_scope_kind"],
            "defeater_count": len(rule["defeater_ids"]),
            "exclusive_wrong_observation_count": value[
                "exclusive_wrong_observation_count"],
            "format_version": 1,
            "fragment_kind": rule["fragment_kind"],
            "identity_false_change_count": value[
                "identity_false_change_count"],
            "impact_id": _record_id(identity),
            "input_length_bucket": _length_bucket(len(rule["input_text"])),
            "input_scalar_length": len(rule["input_text"]),
            "negative_evidence_count": len(rule["negative_evidence_ids"]),
            "observed_distinct_source_family_count": rule[
                "observed_distinct_source_family_count"],
            "output_length_bucket": _length_bucket(len(rule["output_text"]))
            if rule["output_text"] else "L00",
            "output_scalar_length": len(rule["output_text"]),
            "positive_evidence_count": len(rule["positive_evidence_ids"]),
            "priority_contract": _priority_contract(rule),
            "record_kind": NORMALIZATION_RECOVERY_V5_RULE_IMPACT_KIND,
            "required_distinct_source_family_count": rule[
                "required_distinct_source_family_count"],
            "rule_class": rule["rule_class"],
            "selection_occurrence_count": value[
                "selection_occurrence_count"],
            "selection_ordinal_counts": {
                key: value["selection_ordinal_counts"][key]
                for key in sorted(value["selection_ordinal_counts"],
                                  key=int)},
            "source_families": rule["source_families"],
            "structure_match_required": rule["application_scope"][
                "structure_match_required"],
            "structure_token_variant_count": len(
                rule["structure_token_variants"]),
            "structured_wrong_observation_count": value[
                "structured_wrong_observation_count"],
            "wrong_case_ids_sha256": _sha256(
                canonical_json_bytes(case_ids)),
            "wrong_observation_count": value["wrong_observation_count"],
        })
    return tuple(records), observation_sets


def _counter_payload(counter: Counter, keys: tuple[str, ...]) -> dict[str, int]:
    """按冻结 key roster 输出含零项的计数。"""
    return {key: counter[key] for key in keys}


def _coverage_counts(
        impacts: tuple[dict[str, object], ...],
        observation_sets: dict[tuple[str, str], set[str]],
        *,
        family: str | None,
        ) -> dict[str, int]:
    """按 rule WRONG 数排序计算 top-N 唯一 case 覆盖。"""
    selected = tuple(item for item in impacts
                     if family is None
                     or item["held_out_source_family"] == family)
    ordered = sorted(
        selected,
        key=lambda item: (
            -int(item["wrong_observation_count"]),
            str(item["held_out_source_family"]),
            str(item["rule_id"]),
        ),
    )
    result = {}
    for limit in (1, 10, 50, 100):
        case_ids = set()
        for item in ordered[:limit]:
            key = (str(item["held_out_source_family"]), str(item["rule_id"]))
            case_ids.update(observation_sets[key])
        result[f"top_{limit}_rule_unique_wrong_case_count"] = len(case_ids)
    return result


def _family_summaries(
        *,
        audit_manifest_sha256: str,
        wrong_cases: tuple[dict[str, object], ...],
        impacts: tuple[dict[str, object], ...],
        observation_sets: dict[tuple[str, str], set[str]],
        outcomes_by_family: dict[str, Counter],
        buckets_by_family: dict[str, dict[str, Counter]],
        ) -> tuple[dict[str, object], ...]:
    """形成每个 held-out family 的维度分账和集中度摘要。"""
    values = []
    for family in V5_SOURCE_FAMILIES:
        cases = tuple(item for item in wrong_cases
                      if item["held_out_source_family"] == family)
        family_impacts = tuple(item for item in impacts
                               if item["held_out_source_family"] == family)
        class_case_ids: dict[str, set[str]] = defaultdict(set)
        authority_case_ids: dict[str, set[str]] = defaultdict(set)
        support_count_case_ids: dict[str, set[str]] = defaultdict(set)
        input_length_case_ids: dict[str, set[str]] = defaultdict(set)
        for impact in family_impacts:
            key = family, str(impact["rule_id"])
            case_ids = observation_sets[key]
            class_case_ids[str(impact["rule_class"])].update(case_ids)
            authority_case_ids[str(impact["authority_basis"])].update(case_ids)
            support_count_case_ids[str(
                impact["observed_distinct_source_family_count"])].update(
                    case_ids)
            input_length_case_ids[str(impact["input_length_bucket"])].update(
                case_ids)
        identity = {
            "audit_manifest_sha256": audit_manifest_sha256,
            "held_out_source_family": family,
        }
        values.append({
            **identity,
            "authority_basis_wrong_case_counts": {
                key: len(authority_case_ids[key])
                for key in sorted(authority_case_ids)},
            "bucket_outcome_counts": {
                bucket: _counter_payload(
                    buckets_by_family[family][bucket], LOSO_OUTCOMES)
                for bucket in AUDIT_BUCKETS},
            "exclusive_rule_wrong_case_count": sum(
                int(item["multi_rule_wrong"] == 0) for item in cases),
            "family_summary_id": _record_id(identity),
            "format_version": 1,
            "identity_false_change_count": sum(
                int(item["identity_false_change"]) for item in cases),
            "input_length_wrong_case_counts": {
                key: len(input_length_case_ids[key])
                for key in _LENGTH_BUCKETS
                if input_length_case_ids[key]},
            "multi_rule_wrong_case_count": sum(
                int(item["multi_rule_wrong"]) for item in cases),
            "outcome_counts": _counter_payload(
                outcomes_by_family[family], LOSO_OUTCOMES),
            "record_kind": NORMALIZATION_RECOVERY_V5_FAILURE_FAMILY_KIND,
            "rule_class_wrong_case_counts": {
                key: len(class_case_ids[key]) for key in sorted(class_case_ids)},
            "selected_rule_count": len(family_impacts),
            "selected_rule_occurrence_count": sum(
                int(item["selected_rule_occurrence_count"]) for item in cases),
            "structured_wrong_case_count": sum(
                int(item["structure_token_count"] > 0) for item in cases),
            "support_family_count_wrong_case_counts": {
                key: len(support_count_case_ids[key])
                for key in sorted(support_count_case_ids, key=int)},
            "top_rule_coverage": _coverage_counts(
                impacts, observation_sets, family=family),
            "wrong_case_count": len(cases),
        })
    return tuple(values)


def derive_normalization_recovery_v5_failure_profile(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        audit_manifest_sha256: str,
        audit_manifest: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], dict[str, object]]:
    """重学四个 LOSO 子集并派生 WRONG case/rule 影响画像。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v5 profile protocol manifest")
    audit_sha = _sha_value(
        audit_manifest_sha256, label="v5 profile audit manifest")
    audit_summary = _audit_summary_contract(
        protocol_manifest_sha256=protocol_sha,
        audit_manifest_sha256=audit_sha,
        audit_manifest=audit_manifest,
    )
    if not observations or not fragments:
        raise BroadQaExternalDataError("v5 failure profile TRAIN material 为空")
    wrong_cases = []
    rules_by_family = {}
    outcomes_by_family = {family: Counter() for family in V5_SOURCE_FAMILIES}
    buckets_by_family = {
        family: {bucket: Counter() for bucket in AUDIT_BUCKETS}
        for family in V5_SOURCE_FAMILIES}
    for family in V5_SOURCE_FAMILIES:
        material = derive_normalization_recovery_v5_loso_execution(
            protocol_manifest_sha256=protocol_sha,
            observations=observations,
            fragments=fragments,
            held_out_source_family=family,
            include_reference=False,
        )
        outputs = material["outputs"]
        target_rules = outputs["target-phrase-rules.jsonl"]
        target_rule_by_id = {
            str(item["rule_id"]): item for item in target_rules}
        if len(target_rule_by_id) != len(target_rules):
            raise BroadQaExternalDataError(
                "v5 failure profile target rule identity 重复")
        rules_by_family[family] = target_rule_by_id
        for observation, result in zip(
                material["held_out_observations"],
                material["indexed_results"]):
            outcome = _outcome(observation, result)
            bucket = normalization_recovery_v5_result_bucket(
                observation, result)
            if bucket not in AUDIT_BUCKETS:
                raise BroadQaExternalDataError(
                    "v5 failure profile bucket 漂移")
            outcomes_by_family[family][outcome] += 1
            buckets_by_family[family][bucket][outcome] += 1
            if outcome == "WRONG":
                wrong_cases.append(_wrong_case(
                    audit_manifest_sha256=audit_sha,
                    held_out_source_family=family,
                    observation=observation,
                    result=result,
                    target_rule_by_id=target_rule_by_id,
                ))
    actual_loso = {
        f"{family}:{outcome}": outcomes_by_family[family][outcome]
        for family in V5_SOURCE_FAMILIES for outcome in LOSO_OUTCOMES}
    actual_buckets = {
        f"{bucket}:{outcome}": sum(
            buckets_by_family[family][bucket][outcome]
            for family in V5_SOURCE_FAMILIES)
        for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES}
    if (actual_loso != audit_summary.get("loso_counts")
            or actual_buckets != audit_summary.get("bucket_outcome_counts")):
        raise BroadQaExternalDataError(
            "v5 failure profile 与 sealed audit 分母漂移")
    frozen_cases = tuple(sorted(
        wrong_cases, key=lambda item: str(item["case_id"])))
    impacts, observation_sets = _rule_impact_records(
        audit_manifest_sha256=audit_sha,
        wrong_cases=frozen_cases,
        rules_by_family=rules_by_family,
    )
    family_summaries = _family_summaries(
        audit_manifest_sha256=audit_sha,
        wrong_cases=frozen_cases,
        impacts=impacts,
        observation_sets=observation_sets,
        outcomes_by_family=outcomes_by_family,
        buckets_by_family=buckets_by_family,
    )
    class_case_ids: dict[str, set[str]] = defaultdict(set)
    authority_case_ids: dict[str, set[str]] = defaultdict(set)
    support_count_case_ids: dict[str, set[str]] = defaultdict(set)
    for impact in impacts:
        key = (str(impact["held_out_source_family"]), str(impact["rule_id"]))
        case_ids = observation_sets[key]
        class_case_ids[str(impact["rule_class"])].update(case_ids)
        authority_case_ids[str(impact["authority_basis"])].update(case_ids)
        support_count_case_ids[str(
            impact["observed_distinct_source_family_count"])].update(case_ids)
    summary = {
        "audit_manifest_sha256": audit_sha,
        "authority_basis_wrong_case_counts": {
            key: len(authority_case_ids[key])
            for key in sorted(authority_case_ids)},
        "family_summary_count": len(family_summaries),
        "identity_false_change_count": sum(
            int(item["identity_false_change"]) for item in frozen_cases),
        "multi_rule_wrong_case_count": sum(
            int(item["multi_rule_wrong"]) for item in frozen_cases),
        "protocol_manifest_sha256": protocol_sha,
        "rule_class_wrong_case_counts": {
            key: len(class_case_ids[key]) for key in sorted(class_case_ids)},
        "rule_impact_count": len(impacts),
        "selected_rule_occurrence_count": sum(
            int(item["selected_rule_occurrence_count"])
            for item in frozen_cases),
        "support_family_count_wrong_case_counts": {
            key: len(support_count_case_ids[key])
            for key in sorted(support_count_case_ids, key=int)},
        "top_rule_coverage": _coverage_counts(
            impacts, observation_sets, family=None),
        "wrong_case_count": len(frozen_cases),
        "wrong_case_ids_sha256": _sha256(canonical_json_bytes(
            [item["case_id"] for item in frozen_cases])),
    }
    if (summary["wrong_case_count"]
            != sum(actual_loso[f"{family}:WRONG"]
                   for family in V5_SOURCE_FAMILIES)
            or summary["identity_false_change_count"]
            != actual_buckets["IDENTITY:WRONG"]):
        raise BroadQaExternalDataError(
            "v5 failure profile summary 未闭合")
    return frozen_cases, impacts, family_summaries, summary


__all__ = [
    "NORMALIZATION_RECOVERY_V5_FAILURE_CASE_KIND",
    "NORMALIZATION_RECOVERY_V5_FAILURE_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V5_RULE_IMPACT_KIND",
    "derive_normalization_recovery_v5_failure_profile",
]
