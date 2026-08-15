"""执行 recovery-v6 Qt fixed-denominator 八维纯 evaluator。"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V5_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate import (
    execute_normalization_recovery_v6_candidate_batch,
    reference_normalization_recovery_v6_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_label_materialization import (
    NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V6_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_EVALUATION_REPORT_V1")
NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER = (
    "CONTEXT_CONDITIONED_TRANSFER",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "END_TO_END_COVERAGE",
    "IDENTITY_PRESERVATION",
    "LOCAL_MAPPING_TRANSFER",
    "RUNTIME_PRODUCTION_BEHAVIOR",
    "SOURCE_POLICY_CONFLICT",
    "VARIABLE_LENGTH_WHOLE_INPUT_TRANSFER",
)


def _sha256(payload: bytes) -> str:
    """返回 report、runtime 或维度 identity。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def _dimension(key: str, metrics: dict[str, int], outcome: str) -> dict[str, object]:
    """构造固定顺序、整数指标的一个 formal dimension。"""
    if key not in NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER:
        raise BroadQaExternalDataError("v6 evaluator dimension key 漂移")
    if outcome not in {"FAIL", "NE", "PASS"}:
        raise BroadQaExternalDataError("v6 evaluator outcome 非法")
    values = {name: metrics[name] for name in sorted(metrics)}
    payload = {
        "dimension_key": key,
        "metrics": values,
        "outcome": outcome,
    }
    return {**payload, "dimension_sha256": _sha256(
        canonical_json_bytes(payload))}


def _validate_records(
        records: tuple[dict[str, object], ...],
        *,
        commitment: dict[str, object],
        materialization: dict[str, object],
        ) -> None:
    """核验完整分母、互斥 mapping 桶、结构与来源 identity。"""
    denominator = commitment.get("denominator")
    aggregates = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    required = {
        "context_conditioned", "equal_length", "evaluation_id",
        "expected_output", "identity_preservation", "input_text",
        "local_mapping", "record_kind", "single_han_difference",
        "source_conflict", "source_identity", "source_identity_sha256",
        "source_pack_manifest_sha256", "structure_tokens", "variable_length",
    }
    if (not isinstance(records, tuple) or not records
            or not isinstance(aggregates, dict)
            or len(records) != denominator.get("record_count")
            or len(records) != materialization.get("label_materialization_count")
            or materialization.get("evaluation_record_roster_sha256")
            != _sha256(canonical_json_bytes(records))
            or len({item.get("evaluation_id") for item in records})
            != len(records)):
        raise BroadQaExternalDataError("v6 evaluator denominator/roster 漂移")
    for record in records:
        if (not isinstance(record, dict) or not required.issubset(record)
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND
                or not isinstance(record.get("input_text"), str)
                or not record["input_text"]
                or not isinstance(record.get("expected_output"), str)
                or not isinstance(record.get("structure_tokens"), list)
                or any(not isinstance(item, str)
                       for item in record["structure_tokens"])
                or any(type(record.get(name)) is not int
                       or record[name] not in {0, 1}
                       for name in (
                           "context_conditioned", "equal_length",
                           "identity_preservation", "local_mapping",
                           "single_han_difference", "source_conflict",
                           "variable_length"))
                or record["equal_length"] + record["variable_length"] != 1
                or record["local_mapping"] + record["context_conditioned"]
                != record["single_han_difference"]
                or record["source_pack_manifest_sha256"]
                != materialization.get("qt_source_manifest_sha256")):
            raise BroadQaExternalDataError("v6 evaluator record schema 漂移")
    checks = {
        "identity_count": sum(item["identity_preservation"] for item in records),
        "nonidentity_count": sum(1 - item["identity_preservation"]
                                 for item in records),
        "equal_length_count": sum(item["equal_length"] for item in records),
        "variable_length_count": sum(item["variable_length"] for item in records),
        "single_han_difference_count": sum(
            item["single_han_difference"] for item in records),
    }
    if any(checks[name] != aggregates.get(name) for name in checks):
        raise BroadQaExternalDataError("v6 evaluator aggregate denominator 漂移")


def _run_candidate(
        candidate: dict[str, object],
        records: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], int]:
    """执行 indexed 两次与 independent reference，并封装异常计数。"""
    texts = tuple(str(item["input_text"]) for item in records)
    structures = tuple(tuple(item["structure_tokens"]) for item in records)
    try:
        first = execute_normalization_recovery_v6_candidate_batch(
            candidate, texts, policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
            structure_tokens=structures)
        second = execute_normalization_recovery_v6_candidate_batch(
            candidate, texts, policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
            structure_tokens=structures)
        reference = reference_normalization_recovery_v6_candidate_batch(
            candidate, texts, policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
            structure_tokens=structures)
    except (BroadQaExternalDataError, TypeError, ValueError):
        return (), (), (), 1
    return first, second, reference, 0


def _transfer_dimension(
        key: str,
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        *,
        selector: str,
        ) -> dict[str, object]:
    """判定 local、context 或 variable 子分母的 exact whole transfer。"""
    pairs = tuple((record, result) for record, result in zip(records, results)
                  if record[selector] == 1)
    metrics = {
        "applicable_count": sum(item[1].get("applicable") == 1 for item in pairs),
        "exact_output_count": sum(
            result.get("output_text") == record["expected_output"]
            for record, result in pairs),
        "inventory_count": len(pairs),
        "scope_mismatch_count": sum(
            result.get("scope_mismatch") != 0 for _record, result in pairs),
        "wrong_output_count": sum(
            result.get("output_text") != record["expected_output"]
            for record, result in pairs),
    }
    if key == "VARIABLE_LENGTH_WHOLE_INPUT_TRANSFER":
        metrics["length_mismatch_count"] = sum(
            len(str(result.get("output_text")))
            != len(str(record["expected_output"]))
            for record, result in pairs)
    if key == "LOCAL_MAPPING_TRANSFER":
        metrics["unscoped_rule_count"] = sum(
            any(step.get("mode") not in {"IDENTITY", "WHOLE_INPUT_EXACT"}
                for step in result.get("steps", ()))
            for _record, result in pairs)
        metrics["false_accept_count"] = sum(
            record["identity_preservation"] == 1
            and result.get("output_text") != record["input_text"]
            for record, result in pairs)
        metrics["false_reject_count"] = sum(
            record["identity_preservation"] == 0
            and result.get("output_text") == record["input_text"]
            for record, result in pairs)
    outcome = "NE" if not pairs or len(results) != len(records) else "PASS"
    if pairs and len(results) == len(records) and (
            metrics["applicable_count"] != len(pairs)
            or metrics["exact_output_count"] != len(pairs)
            or metrics["wrong_output_count"] != 0
            or metrics["scope_mismatch_count"] != 0
            or metrics.get("length_mismatch_count", 0) != 0
            or metrics.get("unscoped_rule_count", 0) != 0
            or metrics.get("false_accept_count", 0) != 0
            or metrics.get("false_reject_count", 0) != 0):
        outcome = "FAIL"
    return _dimension(key, metrics, outcome)


def _coverage_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """判定完整 fixed denominator 的 end-to-end exact coverage。"""
    pairs = tuple(zip(records, results))
    wrong_positions = 0
    for record, result in pairs:
        source = str(record["input_text"])
        expected = str(record["expected_output"])
        actual = str(result.get("output_text"))
        if len(source) == len(expected) == len(actual):
            expected_positions = tuple(
                index for index, (left, right) in enumerate(zip(source, expected))
                if left != right)
            actual_positions = tuple(
                index for index, (left, right) in enumerate(zip(source, actual))
                if left != right)
            wrong_positions += int(expected_positions != actual_positions)
        elif actual != expected:
            wrong_positions += 1
    metrics = {
        "applicable_count": sum(result.get("applicable") == 1
                                for _record, result in pairs),
        "exact_output_count": sum(
            result.get("output_text") == record["expected_output"]
            for record, result in pairs),
        "false_accept_count": sum(
            record["identity_preservation"] == 1
            and result.get("output_text") != record["input_text"]
            for record, result in pairs),
        "false_reject_count": sum(
            record["identity_preservation"] == 0
            and result.get("output_text") == record["input_text"]
            for record, result in pairs),
        "inventory_count": len(records),
        "length_mismatch_count": sum(
            len(str(result.get("output_text")))
            != len(str(record["expected_output"]))
            for record, result in pairs),
        "wrong_changed_position_count": wrong_positions,
        "wrong_output_count": sum(
            result.get("output_text") != record["expected_output"]
            for record, result in pairs),
    }
    outcome = "NE" if len(results) != len(records) else "PASS"
    if len(results) == len(records) and (
            metrics["applicable_count"] != len(records)
            or metrics["exact_output_count"] != len(records)
            or any(metrics[name] != 0 for name in (
                "false_accept_count", "false_reject_count",
                "length_mismatch_count", "wrong_changed_position_count",
                "wrong_output_count"))):
        outcome = "FAIL"
    return _dimension("END_TO_END_COVERAGE", metrics, outcome)


def _identity_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """判定 identity hard conjunct，不允许任何 false change。"""
    pairs = tuple((record, result) for record, result in zip(records, results)
                  if record["identity_preservation"] == 1)
    metrics = {
        "applicable_count": sum(result.get("applicable") == 1
                                for _record, result in pairs),
        "exact_output_count": sum(
            result.get("output_text") == record["expected_output"]
            for record, result in pairs),
        "false_change_count": sum(
            result.get("output_text") != record["input_text"]
            for record, result in pairs),
        "inventory_count": len(pairs),
    }
    outcome = "NE" if not pairs or len(results) != len(records) else "PASS"
    if pairs and len(results) == len(records) and (
            metrics["applicable_count"] != len(pairs)
            or metrics["exact_output_count"] != len(pairs)
            or metrics["false_change_count"] != 0):
        outcome = "FAIL"
    return _dimension("IDENTITY_PRESERVATION", metrics, outcome)


def _defeater_matches(
        record: dict[str, object], text: str, start: int, end: int,
        ) -> bool:
    """独立执行 frozen exact-adjacent negative-context predicate。"""
    left = str(record["left_context"])
    right = str(record["right_context"])
    left_match = (start == 0 if record["left_boundary"] == 1
                  else start >= len(left) and text[start - len(left):start] == left)
    right_match = (end == len(text) if record["right_boundary"] == 1
                   else text[end:end + len(right)] == right)
    return left_match and right_match


def _defeater_dimension(candidate: dict[str, object]) -> dict[str, object]:
    """独立构造每个 v6 defeater 的命中上下文并执行 predicate。"""
    phrase = candidate.get("phrase_program")
    buckets = phrase.get("target_buckets") if isinstance(phrase, dict) else None
    records = phrase.get("defeaters") if isinstance(phrase, dict) else None
    if not isinstance(buckets, list) or not isinstance(records, list):
        return _dimension("DEFEATER_REPRESENTATION_EXECUTABILITY", {
            "declared_defeater_count": 0,
            "executable_defeater_count": 0,
            "identity_only_defeater_count": 0,
            "malformed_defeater_count": 1,
        }, "NE")
    rules = {str(rule["rule_id"]): rule for bucket in buckets
             for rule in bucket.get("rules", ())}
    executable = 0
    malformed = 0
    identity_only = 0
    for record in records:
        rule = rules.get(str(record.get("rule_id")))
        if rule is None:
            malformed += 1
            continue
        left = "" if record.get("left_boundary") == 1 else str(
            record.get("left_context"))
        right = "" if record.get("right_boundary") == 1 else str(
            record.get("right_context"))
        text = left + str(rule["input_text"]) + right
        try:
            executable += int(_defeater_matches(
                record, text, len(left), len(left) + len(str(rule["input_text"]))))
        except (KeyError, TypeError, ValueError):
            malformed += 1
        identity_only += int(
            record.get("left_boundary") == 0
            and record.get("right_boundary") == 0
            and record.get("left_context") == ""
            and record.get("right_context") == "")
    metrics = {
        "declared_defeater_count": len(records),
        "executable_defeater_count": executable,
        "identity_only_defeater_count": identity_only,
        "malformed_defeater_count": malformed,
    }
    outcome = "NE" if not records else "PASS"
    if records and (executable != len(records)
                    or malformed != 0 or identity_only != 0):
        outcome = "FAIL"
    return _dimension("DEFEATER_REPRESENTATION_EXECUTABILITY", metrics, outcome)


def _source_dimension(candidate: dict[str, object]) -> dict[str, object]:
    """核对 declared conflict block，并如实报告 source replay 缺失。"""
    phrase = candidate.get("phrase_program")
    conflicts = phrase.get("conflict_vetoes") if isinstance(phrase, dict) else None
    if not isinstance(conflicts, list):
        return _dimension("SOURCE_POLICY_CONFLICT", {
            "declared_conflict_count": 0,
            "identity_conflict_overlap_block_count": 0,
            "observed_conflict_block_count": 0,
            "policy_specific_replay_count": 0,
            "unscoped_conflict_execution_count": 0,
        }, "NE")
    texts = tuple(str(item["input_text"]) for item in conflicts)
    try:
        results = (() if not texts else
                   execute_normalization_recovery_v6_candidate_batch(
                       candidate, texts,
                       policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE))
    except (BroadQaExternalDataError, TypeError, ValueError):
        results = ()
    blocked = sum(
        result.get("output_text") == text
        and bool({"CONFLICT_VETO", "IDENTITY_VETO"}.intersection(
            result.get("decision_reasons", ())))
        for text, result in zip(texts, results))
    overlap = sum(
        result.get("output_text") == text
        and "IDENTITY_VETO" in result.get("decision_reasons", ())
        for text, result in zip(texts, results))
    metrics = {
        "declared_conflict_count": len(conflicts),
        "identity_conflict_overlap_block_count": overlap,
        "observed_conflict_block_count": blocked,
        "policy_specific_replay_count": 0,
        "unscoped_conflict_execution_count": sum(
            result.get("output_text") != text
            for text, result in zip(texts, results)),
    }
    if len(results) != len(texts) or blocked != len(conflicts):
        outcome = "FAIL"
    else:
        outcome = "NE"
    return _dimension("SOURCE_POLICY_CONFLICT", metrics, outcome)


def _runtime_dimension(
        first: tuple[dict[str, object], ...],
        second: tuple[dict[str, object], ...],
        reference: tuple[dict[str, object], ...],
        *, expected_count: int, exception_count: int,
        ) -> tuple[dict[str, object], str]:
    """核对双运行、indexed/reference、禁用态和完整 target scope。"""
    triples = tuple(zip(first, second, reference))
    mismatch = sum(left != right or left != linear
                   for left, right, linear in triples)
    metrics = {
        "exception_count": exception_count,
        "executed_twice_count": len(triples),
        "indexed_reference_mismatch_count": mismatch,
        "inventory_count": expected_count,
        "mastery_claimed_count": sum(
            item.get("mastery_claimed") != 0
            for triple in triples for item in triple),
        "production_enabled_count": sum(
            item.get("production_enabled") != 0
            for triple in triples for item in triple),
        "target_policy_scope_count": sum(
            item.get("requested_policy_scope") == RECOVERY_V6_TARGET_POLICY_SCOPE
            for item in first),
    }
    outcome = "PASS"
    if not triples:
        outcome = "NE"
    elif (len(triples) != expected_count or exception_count != 0
          or mismatch != 0 or metrics["mastery_claimed_count"] != 0
          or metrics["production_enabled_count"] != 0
          or metrics["target_policy_scope_count"] != expected_count):
        outcome = "FAIL"
    digest = hashlib.sha256()
    for item in first:
        digest.update(canonical_json_bytes({
            "input_text": item["input_text"],
            "result_sha256": item["result_sha256"],
        }))
    return _dimension("RUNTIME_PRODUCTION_BEHAVIOR", metrics, outcome), (
        digest.hexdigest())


def evaluate_normalization_recovery_v6_candidate(
        *,
        commitment: dict[str, object],
        candidate_manifest: dict[str, object],
        candidate: dict[str, object],
        materialization: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        family_freeze_manifest_sha256: str,
        ) -> dict[str, object]:
    """执行八维 formal evaluator，不写 publication 或修改 candidate。"""
    if (not isinstance(commitment, dict)
            or not _strict_equal(commitment.get("dimensions"),
                                 NORMALIZATION_RECOVERY_V5_DIMENSIONS)
            or candidate_manifest.get("candidate_program_sha256")
            != candidate.get("candidate_program_sha256")
            or candidate.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or materialization.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")):
        raise BroadQaExternalDataError("v6 evaluator commitment/candidate 漂移")
    _validate_records(
        evaluation_records, commitment=commitment,
        materialization=materialization)
    first, second, reference, exceptions = _run_candidate(
        candidate, evaluation_records)
    dimensions = [
        _transfer_dimension(
            "CONTEXT_CONDITIONED_TRANSFER", evaluation_records, first,
            selector="context_conditioned"),
        _defeater_dimension(candidate),
        _coverage_dimension(evaluation_records, first),
        _identity_dimension(evaluation_records, first),
        _transfer_dimension(
            "LOCAL_MAPPING_TRANSFER", evaluation_records, first,
            selector="local_mapping"),
    ]
    runtime, runtime_sha = _runtime_dimension(
        first, second, reference,
        expected_count=len(evaluation_records), exception_count=exceptions)
    dimensions.extend((
        runtime,
        _source_dimension(candidate),
        _transfer_dimension(
            "VARIABLE_LENGTH_WHOLE_INPUT_TRANSFER",
            evaluation_records, first, selector="variable_length"),
    ))
    if tuple(item["dimension_key"] for item in dimensions) != (
            NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER):
        raise BroadQaExternalDataError("v6 evaluator dimension 顺序漂移")
    outcomes = tuple(str(item["outcome"]) for item in dimensions)
    overall = "FAIL" if "FAIL" in outcomes else "NE" if "NE" in outcomes else "PASS"
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_EVALUATION_REPORT_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "dimensions": dimensions,
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_run_count": 1,
        "family_freeze_manifest_sha256": family_freeze_manifest_sha256,
        "format_version": 1,
        "label_materialization_count": materialization[
            "label_materialization_count"],
        "mastery_claimed": 0,
        "overall_outcome": overall,
        "production_enabled": 0,
        "qt_source_payload_read_count": materialization[
            "qt_source_payload_read_count"],
        "runtime_result_sha256": runtime_sha,
        "teacher_api_llm_call_count": 0,
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    return {**report, "evaluation_report_sha256": _sha256(
        canonical_json_bytes(report))}


__all__ = [
    "NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_REPORT_KIND",
    "evaluate_normalization_recovery_v6_candidate",
]
