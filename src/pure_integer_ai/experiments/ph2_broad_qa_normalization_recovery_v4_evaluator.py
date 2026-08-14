"""按标签盲 commitment 纯评估 recovery-v4 composite candidate。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_candidate import (
    NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
    execute_normalization_recovery_v4_candidate_batch,
    reference_normalization_recovery_v4_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_phrase_runtime import (
    normalization_recovery_v4_defeater_matches,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V4_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_EVALUATION_REPORT_V1")
NORMALIZATION_RECOVERY_V4_DIMENSION_ORDER = (
    "LOCAL_MAPPING_TRANSFER",
    "END_TO_END_COVERAGE",
    "SOURCE_POLICY_CONFLICT",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "INDEPENDENT_CONTEXT_TRANSFER",
    "RUNTIME_PRODUCTION_BEHAVIOR",
)


def _sha256(payload: bytes) -> str:
    """返回运行结果流或正式报告 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _dimension(key: str, metrics: dict[str, int], outcome: str) -> dict[str, object]:
    """构造一个规范 bearing dimension 结果。"""
    if (key not in NORMALIZATION_RECOVERY_V4_DIMENSION_ORDER
            or outcome not in {"FAIL", "NE", "PASS"}
            or not metrics
            or any(not isinstance(name, str) or not name
                   or type(value) is not int or value < 0
                   for name, value in metrics.items())):
        raise BroadQaExternalDataError("recovery v4 dimension result 非法")
    return {
        "dimension_key": key,
        "metrics": dict(sorted(metrics.items())),
        "outcome": outcome,
    }


def _phrase_rules(
        program: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...],
                   dict[str, tuple[dict[str, object], ...]]]:
    """提取内嵌 phrase program 的 target/source rules。"""
    phrase = program["phrase_program"]
    target = tuple(rule for bucket in phrase["target_buckets"]
                   for rule in bucket["rules"])
    source = {}
    for item in phrase["source_programs"]:
        source[str(item["source_family"])] = tuple(
            rule for bucket in item["buckets"] for rule in bucket["rules"])
    return target, source


def _validate_records(
        records: tuple[dict[str, object], ...],
        *,
        commitment: dict[str, object],
        source_pack_manifest_sha256: str,
        ) -> None:
    """核验 formal reserve 完整分母、来源、label 与排序。"""
    denominator = commitment.get("denominator")
    if (not isinstance(records, tuple) or not records
            or not isinstance(denominator, dict)
            or len(records) != denominator.get("record_count")):
        raise BroadQaExternalDataError("recovery v4 formal denominator 漂移")
    identities = []
    for record in records:
        if (not isinstance(record, dict)
                or record.get("split") != "RESERVE"
                or record.get("format_version") != 2
                or record.get("source_pack_manifest_sha256")
                != source_pack_manifest_sha256
                or not isinstance(record.get("evaluation_id"), str)
                or len(record["evaluation_id"]) != 64
                or not isinstance(record.get("input_text"), str)
                or not record["input_text"]
                or not isinstance(record.get("expected_output"), str)
                or record.get("input_scalar_count")
                != len(record["input_text"])
                or record.get("output_scalar_count")
                != len(record["expected_output"])
                or not isinstance(record.get("family_keys"), list)
                or not record["family_keys"]):
            raise BroadQaExternalDataError(
                "recovery v4 formal record schema/label 漂移")
        identities.append(record["evaluation_id"])
    if identities != sorted(set(identities)):
        raise BroadQaExternalDataError(
            "recovery v4 formal record identity/顺序漂移")


def _run_target(
        program: dict[str, object],
        records: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            int,
        ]:
    """对完整 reserve 各执行两次 indexed 与一次 reference。"""
    texts = tuple(str(item["input_text"]) for item in records)
    try:
        first = execute_normalization_recovery_v4_candidate_batch(
            program, texts,
            policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
            regional_scope="ZH_CN",
        )
        second = execute_normalization_recovery_v4_candidate_batch(
            program, texts,
            policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
            regional_scope="ZH_CN",
        )
        reference = reference_normalization_recovery_v4_candidate_batch(
            program, texts,
            policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
            regional_scope="ZH_CN",
        )
    except (BroadQaExternalDataError, TypeError, ValueError):
        return (), (), (), len(records)
    return first, second, reference, 0


def _applicable(result: dict[str, object]) -> bool:
    """要求 Firefox transfer scope 对完整分母均成立。"""
    return (result.get("applicable") == 1
            and result.get("projection_used") == 1
            and result.get("scope_mismatch") == 0
            and result.get("requested_policy_scope")
            == NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE
            and result.get("regional_scope") == "ZH_CN")


def _local_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """核对局部映射分母、全输出 exact 与 scope hard conjunct。"""
    pairs = [(record, result) for record, result in zip(records, results)
             if "LOCAL_MAPPING_TRANSFER" in record["family_keys"]]
    metrics = {
        "applicable_mapping_count": sum(_applicable(result)
                                        for _record, result in pairs),
        "exact_output_count": sum(
            result["output_text"] == record["expected_output"]
            for record, result in pairs),
        "false_accept_count": sum(
            result["output_text"] not in {
                record["input_text"], record["expected_output"]}
            for record, result in pairs),
        "false_reject_count": sum(
            result["output_text"] == record["input_text"]
            and record["input_text"] != record["expected_output"]
            for record, result in pairs),
        "inventory_count": len(pairs),
        "scope_mismatch_count": sum(result["scope_mismatch"] != 0
                                    for _record, result in pairs),
        "unscoped_rule_count": sum(
            result["unscoped_conflict_blocked"] != 0
            for _record, result in pairs),
    }
    outcome = "NE" if not pairs else "PASS"
    if pairs and (metrics["applicable_mapping_count"] != len(pairs)
                  or metrics["exact_output_count"] != len(pairs)
                  or any(metrics[name] != 0 for name in (
                      "false_accept_count", "false_reject_count",
                      "scope_mismatch_count", "unscoped_rule_count"))):
        outcome = "FAIL"
    return _dimension("LOCAL_MAPPING_TRANSFER", metrics, outcome)


def _coverage_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """核对 phrase/identity 全分母 exact、长度与错误改写。"""
    pairs = [(record, result) for record, result in zip(records, results)
             if "END_TO_END_COVERAGE" in record["family_keys"]]
    buckets = Counter()
    wrong_positions = 0
    for record, result in pairs:
        length = len(record["input_text"])
        bucket = ("L01" if length == 1 else "L02_04" if length <= 4
                  else "L05_08" if length <= 8 else "L09_16"
                  if length <= 16 else "L17_160")
        buckets[bucket] += 1
        output = str(result["output_text"])
        expected = str(record["expected_output"])
        wrong_positions += sum(left != right
                               for left, right in zip(output, expected))
        wrong_positions += abs(len(output) - len(expected))
    metrics = {
        "applicable_count": sum(_applicable(result)
                                for _record, result in pairs),
        "equal_length_inventory_count": sum(
            len(record["input_text"]) == len(record["expected_output"])
            for record, _result in pairs),
        "exact_output_count": sum(
            result["output_text"] == record["expected_output"]
            for record, result in pairs),
        "false_accept_count": sum(
            record["identity_preservation"] == 1
            and result["output_text"] != record["input_text"]
            for record, result in pairs),
        "false_reject_count": sum(
            record["identity_preservation"] == 0
            and result["output_text"] == record["input_text"]
            for record, result in pairs),
        "identity_false_accept_count": sum(
            record["identity_preservation"] == 1
            and result["output_text"] != record["expected_output"]
            for record, result in pairs),
        "inventory_count": len(pairs),
        "length_mismatch_count": sum(
            len(str(result["output_text"]))
            != len(str(record["expected_output"]))
            for record, result in pairs),
        "variable_length_inventory_count": sum(
            len(record["input_text"]) != len(record["expected_output"])
            for record, _result in pairs),
        "wrong_changed_position_count": wrong_positions,
        **{f"phrase_bucket_{key}": value
           for key, value in sorted(buckets.items())},
    }
    outcome = "NE" if not pairs else "PASS"
    if pairs and (metrics["applicable_count"] != len(pairs)
                  or metrics["exact_output_count"] != len(pairs)
                  or any(metrics[name] != 0 for name in (
                      "false_accept_count", "false_reject_count",
                      "identity_false_accept_count", "length_mismatch_count",
                      "wrong_changed_position_count"))):
        outcome = "FAIL"
    return _dimension("END_TO_END_COVERAGE", metrics, outcome)


def _source_dimension(program: dict[str, object]) -> dict[str, object]:
    """核对 source rule 只在正确 family/policy 内执行并阻断无 scope 冲突。"""
    _target, source_by_family = _phrase_rules(program)
    policy_by_family = {
        str(item["source_family"]): str(item["source_policy_scope"])
        for item in program["transfer_profile"]["source_policy_to_family"]
    }
    metrics = {
        "declared_conflict_count": len(program["conflicts"]),
        "policy_rule_count": 0,
        "policy_rule_output_mismatch_count": 0,
        "source_rule_target_leak_count": 0,
        "unscoped_conflict_execution_count": 0,
        "unscoped_conflict_blocked_count": 0,
    }
    try:
        for family, rules in sorted(source_by_family.items()):
            policy = policy_by_family[family]
            texts = tuple(str(item["input_text"]) for item in rules)
            scoped = execute_normalization_recovery_v4_candidate_batch(
                program, texts, policy_scope=policy)
            target = execute_normalization_recovery_v4_candidate_batch(
                program, texts,
                policy_scope=(
                    NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE),
                regional_scope="ZH_CN",
            )
            metrics["policy_rule_count"] += len(rules)
            for rule, scoped_result, target_result in zip(rules, scoped, target):
                scoped_ids = {str(step["rule_id"])
                              for step in scoped_result["steps"]
                              if step["rule_id"]}
                target_source_ids = {
                    str(step["rule_id"]) for step in target_result["steps"]
                    if step["candidate_scope_kind"] == "SOURCE_ONLY"}
                if (str(rule["rule_id"]) in scoped_ids
                        and scoped_result["output_text"] != rule["output_text"]):
                    metrics["policy_rule_output_mismatch_count"] += 1
                metrics["source_rule_target_leak_count"] += int(
                    bool(target_source_ids))
        conflict_texts = tuple(str(item["input_text"])
                               for item in program["conflicts"])
        conflict_results = execute_normalization_recovery_v4_candidate_batch(
            program, conflict_texts, policy_scope="")
        metrics["unscoped_conflict_blocked_count"] = sum(
            item["unscoped_conflict_blocked"] == 1
            for item in conflict_results)
        metrics["unscoped_conflict_execution_count"] = sum(
            item["output_text"] != item["input_text"]
            for item in conflict_results)
    except (BroadQaExternalDataError, TypeError, ValueError):
        return _dimension("SOURCE_POLICY_CONFLICT", metrics, "NE")
    outcome = "PASS"
    if (metrics["policy_rule_count"] == 0
            or metrics["declared_conflict_count"] == 0):
        outcome = "NE"
    elif (metrics["policy_rule_output_mismatch_count"] != 0
          or metrics["source_rule_target_leak_count"] != 0
          or metrics["unscoped_conflict_execution_count"] != 0
          or metrics["unscoped_conflict_blocked_count"]
          != metrics["declared_conflict_count"]):
        outcome = "FAIL"
    return _dimension("SOURCE_POLICY_CONFLICT", metrics, outcome)


def _defeater_dimension(program: dict[str, object]) -> dict[str, object]:
    """独立构造每个 defeater 的命中上下文并执行 predicate。"""
    target, source_by_family = _phrase_rules(program)
    rule_by_id = {str(item["rule_id"]): item for item in target}
    for rules in source_by_family.values():
        rule_by_id.update({str(item["rule_id"]): item for item in rules})
    defeaters = tuple(program["phrase_program"]["defeaters"])
    executable = 0
    malformed = 0
    identity_only = 0
    for record in defeaters:
        rule = rule_by_id.get(str(record.get("rule_id")))
        if rule is None:
            malformed += 1
            continue
        left = "" if record["left_boundary"] == 1 else record["left_context"]
        right = "" if record["right_boundary"] == 1 else record["right_context"]
        text = str(left) + str(rule["input_text"]) + str(right)
        start = len(str(left))
        end = start + len(str(rule["input_text"]))
        try:
            executable += int(normalization_recovery_v4_defeater_matches(
                record, text, start, end))
        except (BroadQaExternalDataError, TypeError, ValueError):
            malformed += 1
        identity_only += int(
            record["left_boundary"] == 0
            and record["right_boundary"] == 0
            and record["left_context"] == ""
            and record["right_context"] == "")
    metrics = {
        "declared_defeater_count": len(defeaters),
        "executable_defeater_count": executable,
        "identity_only_defeater_count": identity_only,
        "malformed_defeater_count": malformed,
    }
    outcome = "NE" if not defeaters else "PASS"
    if defeaters and (executable != len(defeaters)
                      or malformed != 0 or identity_only != 0):
        outcome = "FAIL"
    return _dimension("DEFEATER_REPRESENTATION_EXECUTABILITY", metrics, outcome)


def _context_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """核对 context-sensitive reserve 的完整 exact transfer。"""
    pairs = [(record, result) for record, result in zip(records, results)
             if record["context_sensitive"] == 1]
    metrics = {
        "applicable_count": sum(_applicable(result)
                                for _record, result in pairs),
        "exact_output_count": sum(
            result["output_text"] == record["expected_output"]
            for record, result in pairs),
        "inventory_count": len(pairs),
        "wrong_output_count": sum(
            result["output_text"] != record["expected_output"]
            for record, result in pairs),
    }
    outcome = "NE" if not pairs else "PASS"
    if pairs and (metrics["applicable_count"] != len(pairs)
                  or metrics["wrong_output_count"] != 0):
        outcome = "FAIL"
    return _dimension("INDEPENDENT_CONTEXT_TRANSFER", metrics, outcome)


def _runtime_dimension(
        first: tuple[dict[str, object], ...],
        second: tuple[dict[str, object], ...],
        reference: tuple[dict[str, object], ...],
        *,
        expected_count: int,
        exception_count: int,
        ) -> tuple[dict[str, object], str]:
    """核对双运行、indexed/reference、禁用态与完整执行。"""
    triples = tuple(zip(first, second, reference))
    mismatch = sum(
        left.get("result_sha256") != right.get("result_sha256")
        or left.get("result_sha256") != linear.get("result_sha256")
        for left, right, linear in triples)
    production = sum(
        item.get("production_enabled") != 0
        for triple in triples for item in triple)
    target_scope = sum(
        item.get("requested_policy_scope")
        == NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE
        for item in first)
    metrics = {
        "exception_count": exception_count,
        "executed_twice_count": len(triples),
        "indexed_reference_mismatch_count": mismatch,
        "inventory_count": expected_count,
        "production_enabled_count": production,
        "target_policy_scope_count": target_scope,
    }
    outcome = "PASS"
    if not triples:
        outcome = "NE"
    elif (len(triples) != expected_count or exception_count != 0
          or mismatch != 0 or production != 0
          or target_scope != expected_count):
        outcome = "FAIL"
    digest = hashlib.sha256()
    for item in first:
        digest.update(canonical_json_bytes({
            "input_text": item["input_text"],
            "result_sha256": item["result_sha256"],
        }))
    return _dimension("RUNTIME_PRODUCTION_BEHAVIOR", metrics, outcome), (
        digest.hexdigest())


def evaluate_normalization_recovery_v4_candidate(
        *,
        commitment: dict[str, object],
        candidate_manifest: dict[str, object],
        program: dict[str, object],
        materialization: dict[str, object],
        reserve_records: tuple[dict[str, object], ...],
        family_freeze_manifest_sha256: str,
        ) -> dict[str, object]:
    """执行六维 formal evaluator，不写 publication 或修改 candidate。"""
    if (not isinstance(commitment, dict)
            or commitment.get("manifest_sha256")
            != program.get("evaluation_commitment_manifest_sha256")
            or candidate_manifest.get("candidate_program_sha256")
            != program.get("candidate_program_sha256")
            or materialization.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")):
        raise BroadQaExternalDataError(
            "recovery v4 evaluator commitment/candidate 漂移")
    _validate_records(
        reserve_records,
        commitment=commitment,
        source_pack_manifest_sha256=str(
            materialization["firefox_source_pack_manifest_sha256"]),
    )
    first, second, reference, exceptions = _run_target(
        program, reserve_records)
    dimensions = [
        _local_dimension(reserve_records, first),
        _coverage_dimension(reserve_records, first),
        _source_dimension(program),
        _defeater_dimension(program),
        _context_dimension(reserve_records, first),
    ]
    runtime, runtime_sha = _runtime_dimension(
        first, second, reference,
        expected_count=len(reserve_records),
        exception_count=exceptions,
    )
    dimensions.append(runtime)
    if tuple(item["dimension_key"] for item in dimensions) != (
            NORMALIZATION_RECOVERY_V4_DIMENSION_ORDER):
        raise BroadQaExternalDataError("recovery v4 dimension 顺序漂移")
    outcomes = tuple(str(item["outcome"]) for item in dimensions)
    overall = "FAIL" if "FAIL" in outcomes else "NE" if "NE" in outcomes else "PASS"
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_EVALUATION_REPORT_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": program["candidate_program_sha256"],
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
        "reserve_identity_sha256": materialization[
            "reserve_identity_sha256"],
        "reserve_payload_read_count": materialization[
            "reserve_payload_read_count"],
        "runtime_result_sha256": runtime_sha,
        "teacher_api_llm_call_count": 0,
        "transfer_profile_sha256": program["transfer_profile_sha256"],
    }
    return {**report, "evaluation_report_sha256": _sha256(
        canonical_json_bytes(report))}


__all__ = [
    "NORMALIZATION_RECOVERY_V4_DIMENSION_ORDER",
    "NORMALIZATION_RECOVERY_V4_EVALUATION_REPORT_KIND",
    "evaluate_normalization_recovery_v4_candidate",
]
