"""严格回读 recovery-v10 precision TRAIN-only runtime 性能门。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack import (
    V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
    V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256,
    V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT,
    V10_PRECISION_RUNTIME_QUERY_COUNT,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate import (
    NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_KIND,
    NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_STATUS,
    V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT,
    V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE,
    V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE,
    normalization_recovery_v10_precision_runtime_gate_code_files,
    read_normalization_recovery_v10_precision_runtime_inputs,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha256(payload: bytes) -> str:
    """返回 runtime gate 文件的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 precision runtime gate reader {label} 非法")
    return value


def _run_metrics(
        value: object, *, ordinal: int, executor_kind: str,
        ) -> dict[str, object]:
    """核验一轮 indexed/reference 性能与提交边界 aggregate。"""
    expected = {
        "behavior_counts", "committed_output_count", "cpu_ns",
        "exception_count", "executor_kind", "input_bytes", "max_ns",
        "non_source_commit_count", "ordinal", "output_bytes", "p50_ns",
        "p95_ns", "partial_commit_count", "production_enabled_count",
        "queries_per_second", "query_count", "result_sha256",
        "route_counts", "structure_mismatch_count", "unknown_output_count",
        "wall_ns",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise BroadQaExternalDataError(
            "v10 precision runtime gate run schema 漂移")
    positive = (
        "input_bytes", "max_ns", "output_bytes", "p50_ns", "p95_ns",
        "queries_per_second", "query_count", "wall_ns",
    )
    zero = (
        "exception_count", "non_source_commit_count", "partial_commit_count",
        "production_enabled_count", "structure_mismatch_count",
        "unknown_output_count",
    )
    routes = value["route_counts"]
    behaviors = value["behavior_counts"]
    if (value["ordinal"] != ordinal
            or value["executor_kind"] != executor_kind
            or any(type(value[name]) is not int or value[name] <= 0
                   for name in positive)
            or type(value["cpu_ns"]) is not int or value["cpu_ns"] < 0
            or any(type(value[name]) is not int or value[name] != 0
                   for name in zero)
            or value["query_count"] != V10_PRECISION_RUNTIME_QUERY_COUNT
            or value["committed_output_count"]
            != V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
            or value["p50_ns"] > value["p95_ns"]
            or value["p95_ns"] > value["max_ns"]
            or value["queries_per_second"]
            != V10_PRECISION_RUNTIME_QUERY_COUNT * 1_000_000_000 // value[
                "wall_ns"]
            or behaviors != {
                "EXACT": V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT,
                "UNKNOWN": (V10_PRECISION_RUNTIME_QUERY_COUNT
                            - V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT),
            }
            or not isinstance(routes, dict)
            or any(not isinstance(key, str) or type(count) is not int
                   or count < 0 for key, count in routes.items())
            or sum(routes.values()) != V10_PRECISION_RUNTIME_QUERY_COUNT):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate run value 漂移")
    _sha_value(value["result_sha256"], label="run result SHA")
    return value


def _validate_profile(value: object) -> dict[str, object]:
    """核验三次执行、固定资源预算与全部零失败门。"""
    if (not isinstance(value, dict)
            or set(value) != {"aggregate", "budget", "gate_outcome", "runs"}
            or value["gate_outcome"] != "PASS"
            or not isinstance(value["runs"], list)
            or len(value["runs"])
            != V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate profile schema 漂移")
    order = ("INDEXED", "INDEXED", "REFERENCE")
    runs = tuple(_run_metrics(
        item, ordinal=index + 1, executor_kind=order[index])
        for index, item in enumerate(value["runs"]))
    budget = value["budget"]
    aggregate = value["aggregate"]
    if (budget != {
            "execution_order": list(order),
            "peak_working_set_bytes_max_exclusive": (
                V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE),
            "query_count": V10_PRECISION_RUNTIME_QUERY_COUNT,
            "total_wall_ns_max_exclusive": (
                V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE),
            }
            or not isinstance(aggregate, dict)
            or set(aggregate) != {
                "committed_output_count", "exception_count",
                "indexed_reference_mismatch_count",
                "indexed_repeat_mismatch_count", "non_source_commit_count",
                "partial_commit_count", "peak_working_set_bytes",
                "production_enabled_count", "query_count", "result_sha256",
                "structure_mismatch_count", "total_cpu_ns", "total_wall_ns",
                "unknown_output_count",
            }):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate budget/aggregate 漂移")
    zero = (
        "exception_count", "indexed_reference_mismatch_count",
        "indexed_repeat_mismatch_count", "non_source_commit_count",
        "partial_commit_count", "production_enabled_count",
        "structure_mismatch_count", "unknown_output_count",
    )
    if (any(type(aggregate[name]) is not int or aggregate[name] != 0
            for name in zero)
            or aggregate["committed_output_count"]
            != (V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
                * V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT)
            or aggregate["query_count"] != V10_PRECISION_RUNTIME_QUERY_COUNT
            or type(aggregate["peak_working_set_bytes"]) is not int
            or aggregate["peak_working_set_bytes"] <= 0
            or aggregate["peak_working_set_bytes"]
            >= V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE
            or type(aggregate["total_cpu_ns"]) is not int
            or aggregate["total_cpu_ns"] < 0
            or type(aggregate["total_wall_ns"]) is not int
            or aggregate["total_wall_ns"] <= 0
            or aggregate["total_wall_ns"]
            >= V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE
            or aggregate["total_wall_ns"]
            < sum(int(item["wall_ns"]) for item in runs)
            or aggregate["result_sha256"] != runs[0]["result_sha256"]
            or len({item["result_sha256"] for item in runs}) != 1):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate hard gate 漂移")
    _sha_value(aggregate["result_sha256"], label="aggregate result SHA")
    return value


def read_normalization_recovery_v10_precision_runtime_gate(
        source_dir: str | Path, *,
        candidate_pack_dir: str | Path,
        expected_candidate_pack_manifest_sha256: str,
        expected_runtime_gate_sha256: str,
        ) -> dict[str, object]:
    """重读固定 pack 与 code identity，不重跑非确定性性能负载。"""
    root = Path(source_dir).resolve()
    expected_pack_sha = _sha_value(
        expected_candidate_pack_manifest_sha256,
        label="candidate pack manifest SHA")
    expected_gate_sha = _sha_value(
        expected_runtime_gate_sha256, label="runtime gate SHA")
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "runtime-gate.json").read_bytes()
        report = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 precision runtime gate artifact 不可读") from error
    if (physical != {"runtime-gate.json"}
            or _sha256(encoded) != expected_gate_sha
            or not isinstance(report, dict)
            or canonical_json_line(report) != encoded):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate artifact identity 漂移")
    expected_fields = {
        "artifact_kind", "candidate_pack_manifest_sha256",
        "candidate_program_read_count", "candidate_program_sha256",
        "candidate_rule_counts", "code_files", "formal_guard_write_count",
        "formal_or_evaluation_payload_read_count", "format_version",
        "individual_candidate_output_publication_count", "mastery_claimed",
        "production_enabled", "profile", "query_roster_bytes",
        "query_roster_sha256", "runtime_shape_read_count",
        "runtime_shapes_sha256", "status", "teacher_api_llm_call_count",
        "v2_feasibility_manifest_sha256",
    }
    if set(report) != expected_fields:
        raise BroadQaExternalDataError(
            "v10 precision runtime gate report schema 漂移")
    manifest, _candidate, shapes, metadata = (
        read_normalization_recovery_v10_precision_runtime_inputs(
            candidate_pack_dir=candidate_pack_dir,
            expected_candidate_pack_manifest_sha256=expected_pack_sha,
        ))
    if (report["artifact_kind"]
            != NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_KIND
            or report["status"]
            != NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_STATUS
            or report["format_version"] != 1
            or report["candidate_pack_manifest_sha256"] != expected_pack_sha
            or report["candidate_program_sha256"]
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or report["candidate_rule_counts"] != manifest["rule_counts"]
            or report["candidate_program_read_count"] != 1
            or report["v2_feasibility_manifest_sha256"]
            != V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256
            or report["runtime_shape_read_count"] != len(shapes)
            or report["runtime_shapes_sha256"]
            != metadata["runtime_shapes_sha256"]
            or report["query_roster_bytes"]
            != metadata["query_roster_bytes"]
            or report["query_roster_sha256"]
            != metadata["query_roster_sha256"]
            or report["code_files"]
            != normalization_recovery_v10_precision_runtime_gate_code_files()
            or any(report[name] != 0 for name in (
                "formal_guard_write_count",
                "formal_or_evaluation_payload_read_count",
                "individual_candidate_output_publication_count",
                "mastery_claimed", "production_enabled",
                "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate lineage/state 漂移")
    _sha_value(report["query_roster_sha256"], label="query roster SHA")
    _sha_value(report["runtime_shapes_sha256"], label="runtime shapes SHA")
    _validate_profile(report["profile"])
    return {**report, "runtime_gate_sha256": expected_gate_sha}


__all__ = ["read_normalization_recovery_v10_precision_runtime_gate"]
