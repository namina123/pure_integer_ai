"""构建和发布 append-only 的 PERF-P3 SQLite 试验 receipt。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from scripts.performance_baseline_contract import (
    file_identity,
    read_head,
    require_external_state_root,
    sha256_bytes,
)
from scripts.performance_p3_sqlite_trial_worker import (
    CONTRACT as TRIAL_CONTRACT,
    READINESS_TRANSITION,
    SCHEMA_VERSION as TRIAL_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_P3_SQLITE_TRIAL_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-P3-SQLITE-TRIAL-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/performance_p3_sqlite_trial_receipt_v1.json"
STATUS = "PERFORMANCE_P3_SQLITE_TRIAL_EVIDENCED"
TRIAL_COMMIT = "4845fa44debb5cb71a2f95bd9c051efb2010938f"
TRIAL_ROOT_NAME = ".perf-p3-sqlite-trial-20260807-r1"
SCALES = (32, 128, 512)
REPETITIONS = 5
PRIOR_RECEIPT_PATH = "data/ph2/manifests/performance_phase_profile_receipt_v1.json"
PRIOR_RECEIPT_SHA256 = (
    "cbf94342625485861a38d46bc4146bddfe6668f03c81506ecc9e184590e52eea"
)
SOURCE_PATHS = (
    "scripts/performance_p3_sqlite_trial_worker.py",
    "src/pure_integer_ai/experiments/ph2_dataset_core.py",
    "src/pure_integer_ai/storage/backend.py",
    "src/pure_integer_ai/storage/integer_codec.py",
    "src/pure_integer_ai/storage/segment_repository.py",
)
EXPECTED_PHASES = (
    "query_read_cold",
    "query_read_warm",
    "segment_decode",
    "sqlite_commit",
    "sqlite_index_registration",
    "sqlite_table_registration",
)
EXPECTED_METRICS = (
    "cold_query_duration_ns",
    "database_file_count",
    "disk_bytes",
    "exception_path_verified",
    "publish_duration_ns",
    "query_count",
    "reopen_duration_ns",
    "repository_init_duration_ns",
    "repository_reopen_duration_ns",
    "rollback_duration_ns",
    "schema_duration_ns",
    "schema_index_count",
    "schema_table_count",
    "visible_object_count",
    "warm_query_duration_ns",
)
EXPECTED_STABLE = {
    32: {
        "schema_digest": "2306bad6caa66603bedc15248f952adc92b58db017d3fd9d5247c60112470db1",
        "publish_digest": "cd79d056fe0963a1e81bd7326c9124f2973300802d7c022f056eeae9dfef0808",
        "cold_query_digest": "e496673b670ccccce0524847f943619f86bd0833022ff7c156dde2da587845fb",
        "warm_query_digest": "e496673b670ccccce0524847f943619f86bd0833022ff7c156dde2da587845fb",
        "rollback_digest": "cbe29348b02a2aea71ed7dfa42909836470f59dba0e518f2c5ce0c552b817cc6",
        "final_visible_digest": "cbe29348b02a2aea71ed7dfa42909836470f59dba0e518f2c5ce0c552b817cc6",
    },
    128: {
        "schema_digest": "2306bad6caa66603bedc15248f952adc92b58db017d3fd9d5247c60112470db1",
        "publish_digest": "90fc16479aff3e7e4f229ed93d43010415c59c2f5558569b5b84e6280260aadf",
        "cold_query_digest": "d61ce0a322bc5735ff27ed5dc7946da6814078674cc01745bc979b4e4fc19fd8",
        "warm_query_digest": "d61ce0a322bc5735ff27ed5dc7946da6814078674cc01745bc979b4e4fc19fd8",
        "rollback_digest": "3939b37cee5d2d0a283bd5e99bd59aa2da6c6325113e49c54b138e97098c8d57",
        "final_visible_digest": "3939b37cee5d2d0a283bd5e99bd59aa2da6c6325113e49c54b138e97098c8d57",
    },
    512: {
        "schema_digest": "2306bad6caa66603bedc15248f952adc92b58db017d3fd9d5247c60112470db1",
        "publish_digest": "ff20f92f374952889c7564a6a934882ca6b4901c3f8a602b88b8ac86a515fc15",
        "cold_query_digest": "2b096019c41461522b1f028821f799c2b8622c5e0cc1c2035cd1b197788fd066",
        "warm_query_digest": "2b096019c41461522b1f028821f799c2b8622c5e0cc1c2035cd1b197788fd066",
        "rollback_digest": "6f5c68c8e631c0c6f708b8122fe14f34ad4943834000d4abba942c1f15c64280",
        "final_visible_digest": "6f5c68c8e631c0c6f708b8122fe14f34ad4943834000d4abba942c1f15c64280",
    },
}


def _canonical_bytes(value: object) -> bytes:
    """把对象编码为带单个末尾换行的规范 JSON 字节。"""
    return canonical_json_bytes(value) + b"\n"


def _safe_relative(relative: str) -> PurePosixPath:
    """核验 Git 外证据使用不可逃逸的 POSIX 相对路径。"""
    path = PurePosixPath(relative)
    if (not relative or "\\" in relative or path.is_absolute()
            or ".." in path.parts or ":" in path.parts[0]):
        raise ValueError("P3 artifact relative path is invalid")
    return path


def _identity(root: Path, relative: str) -> dict[str, object]:
    """读取相对文件并返回路径、尺寸和 SHA-256 身份。"""
    payload = root.joinpath(*_safe_relative(relative).parts).read_bytes()
    return {
        "relative_path": relative,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _require_integer_tree(value: object) -> None:
    """递归拒绝报告中的布尔、浮点和非文本映射键。"""
    if isinstance(value, (bool, float)):
        raise ValueError("P3 report contains bool or float")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("P3 report key is not text")
            _require_integer_tree(item)
    elif isinstance(value, list):
        for item in value:
            _require_integer_tree(item)


def _read_report(root: Path, relative: str) -> tuple[dict[str, Any], dict[str, object]]:
    """严格回读一份规范报告并同时返回其文件身份。"""
    payload = root.joinpath(*_safe_relative(relative).parts).read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError(f"P3 report newline is not canonical: {relative}")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError(f"P3 report bytes are not canonical: {relative}")
    return value, {
        "relative_path": relative,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _validate_report(report: dict[str, Any], scale: int) -> None:
    """核验单次试验报告的固定合同、规模和稳定结果。"""
    if set(report) != {
            "contract", "duration_ns", "metrics", "phase_semantics", "phases",
            "readiness_transition", "scale", "scenario", "schema_version", "stable",
            }:
        raise ValueError("P3 report fields drifted")
    if (report["contract"] != TRIAL_CONTRACT
            or report["schema_version"] != TRIAL_SCHEMA_VERSION
            or report["scenario"] != "storage_sqlite_schema_init_read_trial"
            or report["scale"] != scale
            or report["phase_semantics"] != "OVERLAPPING_CALL_AGGREGATES"
            or report["readiness_transition"] != READINESS_TRANSITION
            or type(report["duration_ns"]) is not int
            or report["duration_ns"] <= 0):
        raise ValueError("P3 report fixed identity drifted")
    phases = report["phases"]
    if not isinstance(phases, dict) or tuple(phases) != EXPECTED_PHASES:
        raise ValueError("P3 phase set drifted")
    expected_counts = {
        "query_read_cold": 3,
        "query_read_warm": 3,
        "segment_decode": 2 * scale + 6,
        "sqlite_commit": 2 * scale + 1,
        "sqlite_index_registration": 8,
        "sqlite_table_registration": 4,
    }
    for name, phase in phases.items():
        if (not isinstance(phase, dict)
                or set(phase) != {"call_count", "duration_ns"}
                or phase["call_count"] != expected_counts[name]
                or type(phase["duration_ns"]) is not int
                or phase["duration_ns"] <= 0):
            raise ValueError(f"P3 phase drifted: {name}")
    metrics = report["metrics"]
    if not isinstance(metrics, dict) or tuple(metrics) != EXPECTED_METRICS:
        raise ValueError("P3 metric set drifted")
    if (metrics["database_file_count"] != 1
            or metrics["exception_path_verified"] != 1
            or metrics["query_count"] != 3
            or metrics["schema_index_count"] != 8
            or metrics["schema_table_count"] != 4
            or metrics["visible_object_count"] != scale - 1):
        raise ValueError("P3 fixed metric drifted")
    if any(type(value) is not int or value < 0 for value in metrics.values()):
        raise ValueError("P3 metric is not a nonnegative integer")
    stable = report["stable"]
    if (not isinstance(stable, dict)
            or stable.get("object_count_after_rollback") != scale - 1
            or stable.get("rollback_restart_equivalent") != 1):
        raise ValueError("P3 stable state drifted")
    observed = {key: stable.get(key) for key in EXPECTED_STABLE[scale]}
    if observed != EXPECTED_STABLE[scale]:
        raise ValueError("P3 stable digest drifted")
    if set(stable) != {*EXPECTED_STABLE[scale], "object_count_after_rollback",
                       "rollback_restart_equivalent"}:
        raise ValueError("P3 stable fields drifted")
    _require_integer_tree(report)


def _median(values: list[int]) -> int:
    """返回固定五个严格整数样本的中位数。"""
    if len(values) != REPETITIONS or any(type(value) is not int for value in values):
        raise ValueError("P3 median input drifted")
    return sorted(values)[REPETITIONS // 2]


def _summary(scale: int, reports: list[dict[str, Any]]) -> dict[str, object]:
    """汇总一个规模下五次报告的阶段与指标中位数。"""
    for key in EXPECTED_STABLE[scale]:
        if {report["stable"][key] for report in reports} != {
                EXPECTED_STABLE[scale][key]}:
            raise ValueError(f"P3 stable digest differs across attempts: {key}")
    metric_medians = {
        key: _median([report["metrics"][key] for report in reports])
        for key in EXPECTED_METRICS
    }
    phase_medians = {
        name: {
            "call_count": reports[0]["phases"][name]["call_count"],
            "median_duration_ns": _median([
                report["phases"][name]["duration_ns"] for report in reports
            ]),
        }
        for name in EXPECTED_PHASES
    }
    return {
        "scale": scale,
        "report_count": REPETITIONS,
        "median_duration_ns": _median([report["duration_ns"] for report in reports]),
        "metric_medians": metric_medians,
        "phase_medians": phase_medians,
        "stable": {
            **EXPECTED_STABLE[scale],
            "object_count_after_rollback": scale - 1,
            "rollback_restart_equivalent": 1,
        },
    }


def _collect(
        repository_root: Path,
        artifact_root: Path,
        *,
        require_trial_head: bool,
        ) -> dict[str, object]:
    """严格收集固定目录中的报告、stderr、数据库和规模汇总。"""
    require_external_state_root(repository_root, artifact_root)
    if artifact_root.name != TRIAL_ROOT_NAME:
        raise ValueError("P3 artifact root name drifted")
    if require_trial_head and read_head(repository_root) != TRIAL_COMMIT:
        raise ValueError("P3 trial HEAD drifted")
    reports_by_scale = {scale: [] for scale in SCALES}
    report_bindings = []
    stderr_bindings = []
    database_bindings = []
    entries = sorted(
        (f"r{repetition:02d}-s{scale}", repetition, scale)
        for repetition in range(1, REPETITIONS + 1)
        for scale in SCALES
    )
    for slug, repetition, scale in entries:
        report, binding = _read_report(
            artifact_root, f"reports/{slug}.json")
        _validate_report(report, scale)
        reports_by_scale[scale].append(report)
        report_bindings.append({
            **binding,
            "repetition": repetition,
            "scale": scale,
        })
        stderr = _identity(artifact_root, f"stderr/{slug}.log")
        if stderr["size_bytes"] != 0 or stderr["sha256"] != hashlib.sha256(b"").hexdigest():
            raise ValueError(f"P3 stderr is not empty: {slug}")
        stderr_bindings.append(stderr)
        database_bindings.append(
            _identity(artifact_root, f"databases/{slug}.sqlite3"))
    return {
        "external_report_bindings": report_bindings,
        "external_stderr_bindings": stderr_bindings,
        "external_database_bindings": database_bindings,
        "scale_summaries": [
            _summary(scale, reports_by_scale[scale]) for scale in SCALES
        ],
    }


def build_performance_p3_sqlite_trial_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> dict[str, Any]:
    """从固定外部试验原件构建 v1 receipt，不写入仓库。"""
    root = Path(repository_root).resolve()
    external = Path(artifact_root).resolve()
    collected = _collect(root, external, require_trial_head=True)
    prior = file_identity(root, PRIOR_RECEIPT_PATH)
    if prior["sha256"] != PRIOR_RECEIPT_SHA256:
        raise ValueError("prior performance receipt identity drifted")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "format_version": FORMAT_VERSION,
        "phase_semantics": "OVERLAPPING_CALL_AGGREGATES",
        "prior_performance_receipt": {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PROFILE_EVIDENCED",
        },
        "readiness_transition": dict(READINESS_TRANSITION),
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": [file_identity(root, path) for path in SOURCE_PATHS],
        "status": STATUS,
        "trial_commit": TRIAL_COMMIT,
        "trial_contract": TRIAL_CONTRACT,
        "trial_repetitions": REPETITIONS,
        "trial_root_name": TRIAL_ROOT_NAME,
        "trial_scales": list(SCALES),
        **collected,
        "verification": {
            "canonical_report_count": 15,
            "empty_stderr_count": 15,
            "exception_path_pass_count": 15,
            "float_value_count": 0,
            "sqlite_database_count": 15,
        },
    }


def _validate_receipt(value: dict[str, Any]) -> None:
    """核验 v1 receipt 的字段、身份、计数和整数树约束。"""
    required = {
        "artifact_kind", "artifact_version", "external_database_bindings",
        "external_report_bindings", "external_stderr_bindings", "format_version",
        "phase_semantics", "prior_performance_receipt", "readiness_transition",
        "receipt_relative_path", "receipt_self_excluded", "scale_summaries",
        "source_bindings", "status", "trial_commit", "trial_contract",
        "trial_repetitions", "trial_root_name", "trial_scales", "verification",
    }
    if set(value) != required:
        raise ValueError("P3 receipt fields drifted")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["phase_semantics"] != "OVERLAPPING_CALL_AGGREGATES"
            or value["readiness_transition"] != READINESS_TRANSITION
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["status"] != STATUS
            or value["trial_commit"] != TRIAL_COMMIT
            or value["trial_contract"] != TRIAL_CONTRACT
            or value["trial_repetitions"] != REPETITIONS
            or value["trial_root_name"] != TRIAL_ROOT_NAME
            or value["trial_scales"] != list(SCALES)):
        raise ValueError("P3 receipt fixed identity drifted")
    if value["prior_performance_receipt"] != {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PROFILE_EVIDENCED",
            }:
        raise ValueError("P3 prior receipt declaration drifted")
    if [item.get("path") for item in value["source_bindings"]] != list(SOURCE_PATHS):
        raise ValueError("P3 source bindings drifted")
    expected_count = REPETITIONS * len(SCALES)
    for field, allow_empty in (
            ("external_report_bindings", False),
            ("external_stderr_bindings", True),
            ("external_database_bindings", False)):
        bindings = value[field]
        if not isinstance(bindings, list) or len(bindings) != expected_count:
            raise ValueError(f"P3 binding count drifted: {field}")
        paths = [item.get("relative_path") for item in bindings]
        if paths != sorted(paths) or len(set(paths)) != expected_count:
            raise ValueError(f"P3 binding paths drifted: {field}")
        for item in bindings:
            _safe_relative(item["relative_path"])
            if (type(item.get("size_bytes")) is not int
                    or item["size_bytes"] < (0 if allow_empty else 1)
                    or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64):
                raise ValueError(f"P3 binding identity drifted: {field}")
    if any(item["size_bytes"] != 0 for item in value["external_stderr_bindings"]):
        raise ValueError("P3 stderr binding is not empty")
    summaries = value["scale_summaries"]
    if not isinstance(summaries, list) or [item.get("scale") for item in summaries] != list(SCALES):
        raise ValueError("P3 scale summaries drifted")
    for summary in summaries:
        scale = summary["scale"]
        if (summary.get("report_count") != REPETITIONS
                or summary.get("stable", {}).get("publish_digest")
                != EXPECTED_STABLE[scale]["publish_digest"]):
            raise ValueError("P3 scale summary identity drifted")
    if value["verification"] != {
            "canonical_report_count": 15,
            "empty_stderr_count": 15,
            "exception_path_pass_count": 15,
            "float_value_count": 0,
            "sqlite_database_count": 15,
            }:
        raise ValueError("P3 receipt verification drifted")
    _require_integer_tree(value)


def read_performance_p3_sqlite_trial_receipt(
        repository_root: str | Path,
        artifact_root: str | Path | None = None,
        path: str | Path = RECEIPT_PATH,
        *,
        verify_external: bool = True,
        verify_current_sources: bool = True,
        ) -> dict[str, Any]:
    """规范回读 v1 receipt，并按选项核验外部原件和当前源码。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("P3 receipt newline is not canonical")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError("P3 receipt bytes are not canonical")
    _validate_receipt(value)
    if verify_current_sources:
        for binding in value["source_bindings"]:
            if file_identity(root, binding["path"]) != binding:
                raise ValueError(f"P3 source identity drifted: {binding['path']}")
        if file_identity(root, PRIOR_RECEIPT_PATH)["sha256"] != PRIOR_RECEIPT_SHA256:
            raise ValueError("P3 prior receipt current identity drifted")
    if verify_external:
        if artifact_root is None:
            raise ValueError("strict P3 receipt read requires artifact_root")
        collected = _collect(root, Path(artifact_root).resolve(), require_trial_head=False)
        for field in (
                "external_report_bindings", "external_stderr_bindings",
                "external_database_bindings", "scale_summaries"):
            if value[field] != collected[field]:
                raise ValueError(f"P3 external evidence drifted: {field}")
    return value


def publish_performance_p3_sqlite_trial_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """独占发布 v1 receipt，目标已存在时在重放前拒绝。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("P3 receipt already published; overwrite forbidden")
    value = build_performance_p3_sqlite_trial_receipt(root, artifact_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(_canonical_bytes(value))
    except FileExistsError as error:
        raise ValueError("P3 receipt already published; overwrite forbidden") from error
    restored = read_performance_p3_sqlite_trial_receipt(
        root, artifact_root, destination)
    if restored != value:
        raise ValueError("P3 receipt publish readback drifted")
    return restored


def main(argv: list[str] | None = None) -> int:
    """解析命令行并构建或独占发布 PERF-P3 v1 receipt。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--target", type=Path, default=Path(RECEIPT_PATH))
    arguments = parser.parse_args(argv)
    try:
        if arguments.publish:
            value = publish_performance_p3_sqlite_trial_receipt(
                REPOSITORY_ROOT, arguments.artifact_root, target=arguments.target)
        else:
            value = build_performance_p3_sqlite_trial_receipt(
                REPOSITORY_ROOT, arguments.artifact_root)
    except (OSError, TypeError, ValueError) as error:
        print(f"performance_p3_sqlite_trial_receipt: ERROR: {error}")
        return 1
    print("performance_p3_sqlite_trial_receipt: ready "
          f"(sha256={hashlib.sha256(_canonical_bytes(value)).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "RECEIPT_PATH", "STATUS",
    "TRIAL_COMMIT", "TRIAL_ROOT_NAME", "build_performance_p3_sqlite_trial_receipt",
    "publish_performance_p3_sqlite_trial_receipt",
    "read_performance_p3_sqlite_trial_receipt",
]
