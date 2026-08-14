"""严格回读 recovery candidate TRAIN-only profile artifact。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_clone import (
    compile_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_profile import (
    NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_KIND,
    NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_STATUS,
    derive_normalization_recovery_training_queries,
    normalization_recovery_candidate_code_files,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_rule_pack import (
    read_normalization_recovery_rule_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha256(payload: bytes) -> str:
    """返回 profile 或 query roster 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _validate_executor_metrics(
        value: object,
        *,
        query_count: int,
        label: str,
        ) -> dict[str, object]:
    """核验一套 executor 的性能、正确性与稳定结果字段。"""
    expected = {
        "cpu_ns", "failure_count", "input_bytes", "mismatch_count",
        "output_bytes", "p50_ns", "p95_ns", "query_count",
        "result_sha256", "wall_ns",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise BroadQaExternalDataError(f"{label} metrics schema 漂移")
    positive = (
        "input_bytes", "output_bytes", "p50_ns", "p95_ns", "query_count",
        "wall_ns",
    )
    zero = ("failure_count", "mismatch_count")
    if (any(type(value[key]) is not int or value[key] <= 0
            for key in positive)
            or type(value["cpu_ns"]) is not int or value["cpu_ns"] < 0
            or any(type(value[key]) is not int or value[key] != 0
                   for key in zero)
            or value["query_count"] != query_count
            or value["p50_ns"] > value["p95_ns"]):
        raise BroadQaExternalDataError(f"{label} metrics value 漂移")
    _sha_value(value["result_sha256"], label=f"{label} result SHA")
    return value


def read_normalization_recovery_candidate_profile(
        profile_dir: str | Path,
        *,
        expected_profile_sha256: str,
        evaluation_protocol_dir: str | Path,
        expected_evaluation_manifest_sha256: str,
        training_protocol_dir: str | Path,
        expected_training_manifest_sha256: str,
        rule_pack_dir: str | Path,
        expected_rule_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """重编译 candidate 与 query identity，不重跑或覆盖性能 workload。"""
    root = Path(profile_dir).resolve()
    expected_sha = _sha_value(
        expected_profile_sha256, label="recovery candidate profile SHA")
    try:
        encoded = (root / "profile.json").read_bytes()
        report = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "recovery candidate profile 不可读") from error
    if (_sha256(encoded) != expected_sha or not isinstance(report, dict)
            or canonical_json_line(report) != encoded):
        raise BroadQaExternalDataError(
            "recovery candidate profile identity/encoding 漂移")
    expected_fields = {
        "artifact_kind", "candidate_compile_wall_ns",
        "candidate_program_sha256", "candidate_rule_pack_read_count",
        "code_files", "evaluation_manifest_read_count",
        "evaluation_payload_read_count", "evaluation_protocol_manifest_sha256",
        "format_version", "mastery_claimed", "production_enabled", "profile",
        "reserve_payload_read_count", "rule_pack_manifest_sha256", "status",
        "teacher_api_llm_call_count", "training_protocol_manifest_sha256",
        "transfer_profile_sha256",
    }
    if set(report) != expected_fields:
        raise BroadQaExternalDataError(
            "recovery candidate profile schema 漂移")
    if (report["artifact_kind"]
            != NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_KIND
            or report["status"]
            != NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_STATUS
            or report["format_version"] != 1
            or type(report["candidate_compile_wall_ns"]) is not int
            or report["candidate_compile_wall_ns"] <= 0
            or any(report[key] != 0 for key in (
                "evaluation_payload_read_count", "mastery_claimed",
                "production_enabled", "reserve_payload_read_count",
                "teacher_api_llm_call_count"))
            or report["candidate_rule_pack_read_count"] != 1
            or report["evaluation_manifest_read_count"] != 1):
        raise BroadQaExternalDataError(
            "recovery candidate profile 固定边界漂移")

    evaluation = read_normalization_recovery_evaluation_manifest_only(
        evaluation_protocol_dir,
        expected_manifest_sha256=expected_evaluation_manifest_sha256,
    )
    pack, outputs = read_normalization_recovery_rule_pack(
        rule_pack_dir,
        protocol_dir=training_protocol_dir,
        expected_protocol_manifest_sha256=expected_training_manifest_sha256,
        expected_pack_manifest_sha256=expected_rule_pack_manifest_sha256,
    )
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    if (report["evaluation_protocol_manifest_sha256"]
            != evaluation["manifest_sha256"]
            or report["training_protocol_manifest_sha256"]
            != expected_training_manifest_sha256
            or report["rule_pack_manifest_sha256"] != pack["manifest_sha256"]
            or report["candidate_program_sha256"] != program.sha256()
            or report["transfer_profile_sha256"]
            != program.transfer_profile.sha256()
            or report["code_files"]
            != normalization_recovery_candidate_code_files()):
        raise BroadQaExternalDataError(
            "recovery candidate profile identity/code 漂移")

    queries = derive_normalization_recovery_training_queries(program)
    query_payload = b"".join(canonical_json_line(item) for item in queries)
    profile = report["profile"]
    expected_profile_fields = {
        "indexed", "indexed_reference_result_bytes_equal",
        "peak_working_set_bytes", "query_count", "query_kind_counts",
        "query_roster_bytes", "query_roster_sha256", "reference",
    }
    if (not isinstance(profile, dict) or set(profile) != expected_profile_fields
            or type(profile["query_count"]) is not int
            or profile["query_count"] != len(queries)
            or profile["query_kind_counts"] != dict(sorted(Counter(
                str(item["kind"]) for item in queries).items()))
            or profile["query_roster_bytes"] != len(query_payload)
            or profile["query_roster_sha256"] != _sha256(query_payload)
            or type(profile["peak_working_set_bytes"]) is not int
            or profile["peak_working_set_bytes"] <= 0
            or profile["indexed_reference_result_bytes_equal"] != 1):
        raise BroadQaExternalDataError(
            "recovery candidate profile query/performance 边界漂移")
    indexed = _validate_executor_metrics(
        profile["indexed"], query_count=len(queries), label="indexed")
    reference = _validate_executor_metrics(
        profile["reference"], query_count=len(queries), label="reference")
    if indexed["result_sha256"] != reference["result_sha256"]:
        raise BroadQaExternalDataError(
            "recovery candidate profile 双解释器结果漂移")
    return {**report, "profile_sha256": expected_sha}


__all__ = ["read_normalization_recovery_candidate_profile"]
