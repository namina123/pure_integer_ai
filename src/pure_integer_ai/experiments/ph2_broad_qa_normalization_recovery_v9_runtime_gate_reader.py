"""严格回读 recovery-v9 GIMP 标签盲 runtime 性能门。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    V8_CANDIDATE_RULE_COUNTS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate import (
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND,
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS,
    V8_BATCH_FIX_GIT_COMMIT,
    V8_CANDIDATE_PACK_MANIFEST_SHA256,
    V8_CANDIDATE_PROGRAM_SHA256,
    V9_RUNTIME_GATE_QUERY_COUNT,
    V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE,
    V9_RUNTIME_SHAPES_SHA256,
    V9_SOURCE_PACK_MANIFEST_SHA256,
    normalization_recovery_v9_runtime_gate_code_files,
    read_normalization_recovery_v9_runtime_probe_inputs,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha256(payload: bytes) -> str:
    """返回runtime gate或query roster的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v9 runtime gate {label} 非法")
    return value


def _run_metrics(value: object, *, ordinal: int,
                 executor_kind: str) -> dict[str, object]:
    """核验一轮indexed/reference性能与安全aggregate。"""
    expected = {
        "cpu_ns", "exception_count", "executor_kind", "input_bytes",
        "max_ns", "ordinal", "output_bytes", "p50_ns", "p95_ns",
        "partial_commit_count", "production_enabled_count", "query_count",
        "queries_per_second", "result_sha256", "route_counts",
        "structure_mismatch_count", "wall_ns",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise BroadQaExternalDataError("v9 runtime gate run schema漂移")
    positive = (
        "input_bytes", "max_ns", "output_bytes", "p50_ns", "p95_ns",
        "query_count", "queries_per_second", "wall_ns",
    )
    zero = (
        "exception_count", "partial_commit_count",
        "production_enabled_count", "structure_mismatch_count",
    )
    routes = value["route_counts"]
    if (value["ordinal"] != ordinal
            or value["executor_kind"] != executor_kind
            or any(type(value[name]) is not int or value[name] <= 0
                   for name in positive)
            or type(value["cpu_ns"]) is not int or value["cpu_ns"] < 0
            or any(type(value[name]) is not int or value[name] != 0
                   for name in zero)
            or value["query_count"] != V9_RUNTIME_GATE_QUERY_COUNT
            or value["p50_ns"] > value["p95_ns"]
            or value["p95_ns"] > value["max_ns"]
            or value["queries_per_second"]
            != V9_RUNTIME_GATE_QUERY_COUNT * 1_000_000_000 // value["wall_ns"]
            or not isinstance(routes, dict)
            or any(not isinstance(key, str) or type(count) is not int
                   or count < 0 for key, count in routes.items())
            or sum(routes.values()) != V9_RUNTIME_GATE_QUERY_COUNT):
        raise BroadQaExternalDataError("v9 runtime gate run value漂移")
    _sha_value(value["result_sha256"], label="result SHA")
    return value


def _validate_profile(value: object) -> dict[str, object]:
    """核验三次执行、硬预算与全部零失败门。"""
    if (not isinstance(value, dict)
            or set(value) != {"aggregate", "budget", "gate_outcome", "runs"}
            or value["gate_outcome"] != "PASS"
            or not isinstance(value["runs"], list)
            or len(value["runs"]) != 3):
        raise BroadQaExternalDataError("v9 runtime gate profile schema漂移")
    runs = tuple(_run_metrics(item, ordinal=index + 1,
                              executor_kind=("INDEXED", "INDEXED", "REFERENCE")[index])
                 for index, item in enumerate(value["runs"]))
    budget = value["budget"]
    aggregate = value["aggregate"]
    if (budget != {
            "execution_order": ["INDEXED", "INDEXED", "REFERENCE"],
            "query_count": V9_RUNTIME_GATE_QUERY_COUNT,
            "total_wall_ns_max_exclusive": (
                V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE),
            } or not isinstance(aggregate, dict)
            or set(aggregate) != {
                "exception_count", "indexed_reference_mismatch_count",
                "indexed_repeat_mismatch_count", "partial_commit_count",
                "peak_working_set_bytes", "production_enabled_count",
                "query_count", "result_sha256", "structure_mismatch_count",
                "total_cpu_ns", "total_wall_ns",
            }):
        raise BroadQaExternalDataError("v9 runtime gate budget/aggregate漂移")
    zero = (
        "exception_count", "indexed_reference_mismatch_count",
        "indexed_repeat_mismatch_count", "partial_commit_count",
        "production_enabled_count", "structure_mismatch_count",
    )
    if (any(type(aggregate[name]) is not int or aggregate[name] != 0
            for name in zero)
            or aggregate["query_count"] != V9_RUNTIME_GATE_QUERY_COUNT
            or type(aggregate["peak_working_set_bytes"]) is not int
            or aggregate["peak_working_set_bytes"] <= 0
            or type(aggregate["total_cpu_ns"]) is not int
            or aggregate["total_cpu_ns"] < 0
            or type(aggregate["total_wall_ns"]) is not int
            or aggregate["total_wall_ns"] <= 0
            or aggregate["total_wall_ns"]
            >= V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE
            or aggregate["total_wall_ns"]
            < sum(int(item["wall_ns"]) for item in runs)
            or aggregate["result_sha256"] != runs[0]["result_sha256"]
            or len({item["result_sha256"] for item in runs}) != 1):
        raise BroadQaExternalDataError("v9 runtime gate hard gate漂移")
    _sha_value(aggregate["result_sha256"], label="aggregate result SHA")
    return value


def read_normalization_recovery_v9_runtime_gate(
        source_dir: str | Path, *,
        source_pack_dir: str | Path,
        candidate_pack_dir: str | Path,
        expected_runtime_gate_sha256: str,
        ) -> dict[str, object]:
    """重读固定输入与code identity，不重跑非确定性性能workload。"""
    root = Path(source_dir).resolve()
    expected_sha = _sha_value(
        expected_runtime_gate_sha256, label="artifact SHA")
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "runtime-gate.json").read_bytes()
        report = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v9 runtime gate artifact不可读") from error
    if (physical != {"runtime-gate.json"}
            or _sha256(encoded) != expected_sha
            or not isinstance(report, dict)
            or canonical_json_line(report) != encoded):
        raise BroadQaExternalDataError("v9 runtime gate artifact identity漂移")
    expected_fields = {
        "artifact_kind", "batch_fix_git_commit",
        "candidate_pack_manifest_sha256", "candidate_program_read_count",
        "candidate_program_sha256", "candidate_rule_counts", "code_files",
        "formal_guard_write_count",
        "formal_label_read_count", "format_version",
        "individual_candidate_output_publication_count", "mastery_claimed",
        "production_enabled", "profile", "query_roster_bytes",
        "query_roster_sha256", "source_pair_identity_read_count",
        "source_pack_manifest_sha256", "source_raw_archive_read_count",
        "source_runtime_shape_read_count", "source_runtime_shapes_sha256",
        "source_translation_surface_read_count", "status",
        "teacher_api_llm_call_count",
    }
    if set(report) != expected_fields:
        raise BroadQaExternalDataError("v9 runtime gate report schema漂移")
    _source, shapes, _candidate_manifest, material = (
        read_normalization_recovery_v9_runtime_probe_inputs(
            source_pack_dir=source_pack_dir,
            candidate_pack_dir=candidate_pack_dir))
    if (report["artifact_kind"] != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND
            or report["status"] != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS
            or report["format_version"] != 1
            or report["batch_fix_git_commit"] != V8_BATCH_FIX_GIT_COMMIT
            or report["source_pack_manifest_sha256"]
            != V9_SOURCE_PACK_MANIFEST_SHA256
            or report["source_runtime_shapes_sha256"]
            != V9_RUNTIME_SHAPES_SHA256
            or report["candidate_pack_manifest_sha256"]
            != V8_CANDIDATE_PACK_MANIFEST_SHA256
            or report["candidate_program_sha256"]
            != V8_CANDIDATE_PROGRAM_SHA256
            or report["candidate_rule_counts"] != V8_CANDIDATE_RULE_COUNTS
            or report["candidate_program_read_count"] != 1
            or report["source_runtime_shape_read_count"] != len(shapes)
            or report["query_roster_bytes"]
            != material["metadata"]["query_roster_bytes"]
            or report["query_roster_sha256"]
            != material["metadata"]["query_roster_sha256"]
            or report["code_files"]
            != normalization_recovery_v9_runtime_gate_code_files()
            or any(report[name] != 0 for name in (
                "formal_guard_write_count", "formal_label_read_count",
                "individual_candidate_output_publication_count",
                "mastery_claimed", "production_enabled",
                "source_pair_identity_read_count", "source_raw_archive_read_count",
                "source_translation_surface_read_count",
                "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError("v9 runtime gate lineage/state漂移")
    _sha_value(report["query_roster_sha256"], label="query roster SHA")
    _validate_profile(report["profile"])
    return {**report, "runtime_gate_sha256": expected_sha}


__all__ = ["read_normalization_recovery_v9_runtime_gate"]
