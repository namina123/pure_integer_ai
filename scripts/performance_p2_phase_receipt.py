"""发布 PERF-P2 阶段 profile 的 append-only、非 readiness receipt。"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from scripts.performance_p2_phase_worker import (
    CONTRACT as PROFILE_CONTRACT,
    READINESS_TRANSITION,
    SCHEMA_VERSION as PROFILE_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_PHASE_PROFILE_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-P2-PHASE-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/performance_phase_profile_receipt_v1.json"
STATUS = "PERFORMANCE_PROFILE_EVIDENCED"
PROFILE_COMMIT = "db5619b1ea7db0fe499a3ae1243f83c80b2f7c2a"
PROFILE_ROOT_NAME = ".perf-p2-phase-s128-r3"
SCALE = 128
REPETITIONS = 5
PRIOR_RECEIPT_PATH = "data/ph2/manifests/performance_baseline_receipt_v1.json"
PRIOR_RECEIPT_SHA256 = (
    "5c9f3d866ba5bb922e5c2f77a3da3c8728618b521fde7f6a308450eec050eaa1"
)
SCENARIOS = (
    "long_memory_projection",
    "storage_sqlite",
)
EXPECTED_STABLE_DIGESTS = {
    "long_memory_projection": (
        "6d485577d310cbb8ecd360be61b983d7f86c0fa2a80badbb779f2f241a8aab10"
    ),
    "storage_sqlite": (
        "cc8372398d270b1a6fcb1bd93a36714bded1f7aeafa9439ed980e254ee227c4b"
    ),
}
EXPECTED_PHASES = {
    "long_memory_projection": (
        "memory_dependencies",
        "memory_manifest_construction",
        "memory_restored_key_encode",
        "memory_runtime_import",
        "memory_segment_construction",
        "memory_stable_key_encode",
        "memory_stable_key_restore",
    ),
    "storage_sqlite": (
        "publish_delta",
        "publish_segment",
        "query_total",
        "repository_init",
        "seal_delta",
        "sqlite_backend_init",
        "sqlite_close",
        "sqlite_commit",
        "sqlite_create_table",
        "sqlite_delete",
        "sqlite_ensure_index",
        "sqlite_insert",
        "sqlite_select",
        "store_init",
    ),
}
SOURCE_PATHS = (
    "scripts/performance_baseline_worker.py",
    "scripts/performance_p2_phase_worker.py",
    "src/pure_integer_ai/experiments/memory_hot_set_runtime.py",
    "src/pure_integer_ai/storage/backend.py",
    "src/pure_integer_ai/storage/segment_repository.py",
    "src/pure_integer_ai/storage/tiered_segment_store.py",
)


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _digest(value: object, *, length: int = 64) -> None:
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("profile receipt digest 无效")


def _exact(value: object, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} 字段漂移")
    return value


def _safe_relative(relative: str) -> PurePosixPath:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("profile artifact relative path 无效")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("profile artifact relative path 越界")
    return path


def _artifact_identity(root: Path, relative: str) -> dict[str, object]:
    path = root.joinpath(*_safe_relative(relative).parts)
    payload = path.read_bytes()
    return {
        "relative_path": relative,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _read_report(root: Path, relative: str) -> tuple[dict[str, Any], dict[str, object]]:
    path = root.joinpath(*_safe_relative(relative).parts)
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError(f"phase report newline 非 canonical: {relative}")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError(f"phase report bytes 非 canonical: {relative}")
    return value, {
        "relative_path": relative,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _require_integer_tree(value: object) -> None:
    if isinstance(value, float) or isinstance(value, bool):
        raise ValueError("phase report 不得包含 float")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("phase report object key 必须是字符串")
        for item in value.values():
            _require_integer_tree(item)
    elif isinstance(value, list):
        for item in value:
            _require_integer_tree(item)


def _median(values: list[int], *, where: str) -> int:
    if (len(values) != REPETITIONS
            or any(type(value) is not int or value < 0 for value in values)):
        raise ValueError(f"{where} 中位数输入无效")
    return sorted(values)[REPETITIONS // 2]


def _validate_report(value: dict[str, Any], scenario: str) -> None:
    raw = _exact(value, {
        "baseline_import_duration_ns", "baseline_report", "contract",
        "phase_semantics", "phases", "profile_duration_ns",
        "readiness_transition", "scale", "scenario", "schema_version",
    }, where="phase report")
    if (raw["contract"] != PROFILE_CONTRACT
            or raw["schema_version"] != PROFILE_SCHEMA_VERSION
            or raw["scenario"] != scenario
            or raw["scale"] != SCALE
            or raw["phase_semantics"] != "OVERLAPPING_CALL_AGGREGATES"
            or raw["readiness_transition"] != READINESS_TRANSITION):
        raise ValueError("phase report 固定身份漂移")
    for field in ("baseline_import_duration_ns", "profile_duration_ns"):
        if type(raw[field]) is not int or raw[field] <= 0:
            raise ValueError(f"phase report {field} 无效")
    report = raw["baseline_report"]
    if (not isinstance(report, dict)
            or report.get("scenario") != scenario
            or report.get("scale") != SCALE
            or report.get("stable_digest") != EXPECTED_STABLE_DIGESTS[scenario]):
        raise ValueError("baseline report identity 漂移")
    phases = raw["phases"]
    if (not isinstance(phases, dict)
            or tuple(phases) != EXPECTED_PHASES[scenario]):
        raise ValueError("phase 名称或顺序漂移")
    for name, phase in phases.items():
        entry = _exact(
            phase, {"call_count", "duration_ns"}, where=f"phase {name}")
        if (type(entry["call_count"]) is not int
                or entry["call_count"] < 1
                or type(entry["duration_ns"]) is not int
                or entry["duration_ns"] <= 0):
            raise ValueError(f"phase {name} 指标无效")
    _require_integer_tree(raw)


def _scenario_summary(
        scenario: str,
        reports: list[dict[str, Any]],
        ) -> dict[str, object]:
    phases = {}
    for name in EXPECTED_PHASES[scenario]:
        counts = {report["phases"][name]["call_count"] for report in reports}
        if len(counts) != 1:
            raise ValueError(f"phase call_count 跨 attempt 漂移: {name}")
        phases[name] = {
            "call_count": counts.pop(),
            "median_duration_ns": _median([
                report["phases"][name]["duration_ns"] for report in reports
            ], where=name),
        }
    baseline_reports = [report["baseline_report"] for report in reports]
    metric_keys = tuple(sorted(baseline_reports[0]["metrics"]))
    if any(tuple(sorted(report["metrics"])) != metric_keys
           for report in baseline_reports):
        raise ValueError("baseline metrics 跨 attempt 字段漂移")
    metric_medians: dict[str, object] = {}
    for key in metric_keys:
        values = [report["metrics"][key] for report in baseline_reports]
        if all(type(value) is int for value in values):
            metric_medians[key] = _median(values, where=f"metric {key}")
        elif len(set(values)) == 1 and isinstance(values[0], str):
            metric_medians[key] = values[0]
        else:
            raise ValueError(f"baseline metric 类型或值漂移: {key}")
    stable_digests = {report["stable_digest"] for report in baseline_reports}
    if stable_digests != {EXPECTED_STABLE_DIGESTS[scenario]}:
        raise ValueError("stable digest 跨 attempt 漂移")
    return {
        "scenario": scenario,
        "stable_digest": EXPECTED_STABLE_DIGESTS[scenario],
        "report_count": REPETITIONS,
        "median_baseline_import_duration_ns": _median([
            report["baseline_import_duration_ns"] for report in reports
        ], where="baseline import"),
        "median_profile_duration_ns": _median([
            report["profile_duration_ns"] for report in reports
        ], where="profile duration"),
        "median_workload_duration_ns": _median([
            report["duration_ns"] for report in baseline_reports
        ], where="workload duration"),
        "median_peak_rss_bytes": _median([
            report["peak_rss_bytes"] for report in baseline_reports
        ], where="peak rss"),
        "median_tracemalloc_peak_bytes": _median([
            report["tracemalloc_peak_bytes"] for report in baseline_reports
        ], where="tracemalloc peak"),
        "baseline_metric_medians": metric_medians,
        "phase_medians": phases,
    }


def _collect(
        repository_root: Path,
        artifact_root: Path,
        *, require_profile_head: bool,
        ) -> dict[str, object]:
    require_external_state_root(repository_root, artifact_root)
    if artifact_root.name != PROFILE_ROOT_NAME:
        raise ValueError("P2 profile artifact root name 漂移")
    if require_profile_head and read_head(repository_root) != PROFILE_COMMIT:
        raise ValueError("P2 profile HEAD 与当前仓库不一致")
    reports_by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    report_bindings = []
    stderr_bindings = []
    database_bindings = []
    for scenario, slug in (
            ("long_memory_projection", "memory"),
            ("storage_sqlite", "sqlite")):
        for attempt in range(1, REPETITIONS + 1):
            relative = f"{slug}-attempt-{attempt:03d}.json"
            report, identity = _read_report(artifact_root, relative)
            _validate_report(report, scenario)
            reports_by_scenario[scenario].append(report)
            report_bindings.append({
                **identity,
                "attempt": attempt,
                "scenario": scenario,
            })
            stderr_relative = f"{slug}-attempt-{attempt:03d}.stderr.log"
            stderr_identity = _artifact_identity(
                artifact_root, stderr_relative)
            if (stderr_identity["size_bytes"] != 0
                    or stderr_identity["sha256"] != hashlib.sha256(b"").hexdigest()):
                raise ValueError(f"phase stderr 非空: {stderr_relative}")
            stderr_bindings.append(stderr_identity)
            if scenario == "storage_sqlite":
                database_bindings.append(_artifact_identity(
                    artifact_root,
                    f"sqlite-attempt-{attempt:03d}.sqlite3",
                ))
    return {
        "external_report_bindings": report_bindings,
        "external_stderr_bindings": stderr_bindings,
        "external_database_bindings": database_bindings,
        "scenario_summaries": [
            _scenario_summary(scenario, reports_by_scenario[scenario])
            for scenario in SCENARIOS
        ],
    }


def build_performance_phase_profile_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    external = Path(artifact_root).resolve()
    collected = _collect(root, external, require_profile_head=True)
    prior = file_identity(root, PRIOR_RECEIPT_PATH)
    if prior["sha256"] != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 performance baseline receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "format_version": FORMAT_VERSION,
        "profile_commit": PROFILE_COMMIT,
        "profile_contract": PROFILE_CONTRACT,
        "profile_root_name": PROFILE_ROOT_NAME,
        "profile_scale": SCALE,
        "profile_repetitions": REPETITIONS,
        "phase_semantics": "OVERLAPPING_CALL_AGGREGATES",
        "prior_performance_receipt": {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_BASELINE_EVIDENCED",
        },
        "readiness_transition": dict(READINESS_TRANSITION),
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": [
            {
                "path": identity["path"],
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
            for path in SOURCE_PATHS
            for identity in (file_identity(root, path),)
        ],
        **collected,
        "status": STATUS,
        "verification": {
            "canonical_report_count": 10,
            "empty_stderr_count": 10,
            "float_value_count": 0,
            "sqlite_database_count": 5,
        },
    }


def _validate_binding_list(
        values: object,
        *, count: int,
        allow_empty: bool = False,
        ) -> None:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("profile artifact binding 数量漂移")
    previous = ""
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("profile artifact binding 必须为 object")
        path = value.get("relative_path")
        if not isinstance(path, str) or path <= previous:
            raise ValueError("profile artifact binding 路径未严格排序")
        _safe_relative(path)
        size = value.get("size_bytes")
        if type(size) is not int or size < 0 or (not allow_empty and size < 1):
            raise ValueError("profile artifact binding size 无效")
        _digest(value.get("sha256"))
        previous = path


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "external_database_bindings",
        "external_report_bindings", "external_stderr_bindings", "format_version",
        "phase_semantics", "prior_performance_receipt", "profile_commit",
        "profile_contract", "profile_repetitions", "profile_root_name",
        "profile_scale", "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "scenario_summaries", "source_bindings",
        "status", "verification",
    }, where="performance phase profile receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["format_version"] != FORMAT_VERSION
            or raw["profile_commit"] != PROFILE_COMMIT
            or raw["profile_contract"] != PROFILE_CONTRACT
            or raw["profile_root_name"] != PROFILE_ROOT_NAME
            or raw["profile_scale"] != SCALE
            or raw["profile_repetitions"] != REPETITIONS
            or raw["phase_semantics"] != "OVERLAPPING_CALL_AGGREGATES"
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != STATUS):
        raise ValueError("performance phase profile receipt 固定身份漂移")
    _digest(raw["profile_commit"], length=40)
    if raw["prior_performance_receipt"] != {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_BASELINE_EVIDENCED",
    }:
        raise ValueError("前序 performance receipt 声明漂移")
    if raw["readiness_transition"] != READINESS_TRANSITION:
        raise ValueError("performance profile 不得转移 readiness")
    source_bindings = raw["source_bindings"]
    if (not isinstance(source_bindings, list)
            or len(source_bindings) != len(SOURCE_PATHS)):
        raise ValueError("performance profile source binding 数量漂移")
    previous = ""
    for binding in source_bindings:
        entry = _exact(
            binding, {"path", "size_bytes", "sha256"},
            where="profile source binding")
        path = entry["path"]
        if not isinstance(path, str) or path <= previous:
            raise ValueError("performance profile source path 未严格排序")
        _safe_relative(path)
        if (type(entry["size_bytes"]) is not int
                or entry["size_bytes"] < 1):
            raise ValueError("performance profile source size 无效")
        _digest(entry["sha256"])
        previous = path
    if tuple(item["path"] for item in source_bindings) != SOURCE_PATHS:
        raise ValueError("performance profile source path 漂移")
    _validate_binding_list(raw["external_report_bindings"], count=10)
    _validate_binding_list(
        raw["external_stderr_bindings"], count=10, allow_empty=True)
    if any(item["size_bytes"] != 0 for item in raw["external_stderr_bindings"]):
        raise ValueError("performance profile stderr 必须为空")
    _validate_binding_list(raw["external_database_bindings"], count=5)
    summaries = raw["scenario_summaries"]
    if (not isinstance(summaries, list)
            or tuple(item.get("scenario") for item in summaries) != SCENARIOS):
        raise ValueError("performance profile scenario summary 漂移")
    for summary in summaries:
        if (summary.get("report_count") != REPETITIONS
                or summary.get("stable_digest")
                != EXPECTED_STABLE_DIGESTS[summary["scenario"]]):
            raise ValueError("performance profile summary identity 漂移")
        _require_integer_tree(summary)
    if raw["verification"] != {
            "canonical_report_count": 10,
            "empty_stderr_count": 10,
            "float_value_count": 0,
            "sqlite_database_count": 5,
    }:
        raise ValueError("performance profile verification 漂移")


def read_performance_phase_profile_receipt(
        repository_root: str | Path,
        artifact_root: str | Path | None = None,
        path: str | Path = RECEIPT_PATH,
        *, verify_external: bool = True,
        verify_current_sources: bool = True,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("performance phase profile receipt newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError("performance phase profile receipt 非 canonical")
    _validate(value)
    if verify_current_sources:
        for binding in value["source_bindings"]:
            if file_identity(root, binding["path"]) != binding:
                raise ValueError(f"profile source identity 漂移: {binding['path']}")
        if file_identity(root, PRIOR_RECEIPT_PATH)["sha256"] != PRIOR_RECEIPT_SHA256:
            raise ValueError("前序 performance receipt 当前 identity 漂移")
    if not verify_external:
        return value
    if artifact_root is None:
        raise ValueError("严格回读 performance profile 需要 artifact_root")
    collected = _collect(
        root, Path(artifact_root).resolve(), require_profile_head=False)
    for field in (
            "external_report_bindings", "external_stderr_bindings",
            "external_database_bindings", "scenario_summaries"):
        if value[field] != collected[field]:
            raise ValueError(f"performance profile external evidence 漂移: {field}")
    return value


def publish_performance_phase_profile_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("performance phase profile receipt 已发布，禁止覆盖")
    value = build_performance_phase_profile_receipt(root, artifact_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(_canonical_bytes(value))
    except FileExistsError as error:
        raise ValueError("performance phase profile receipt 已发布，禁止覆盖") from error
    restored = read_performance_phase_profile_receipt(
        root, artifact_root, destination)
    if restored != value:
        raise ValueError("performance phase profile receipt 发布回读不一致")
    return restored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布 PERF-P2 phase receipt。")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--target", type=Path, default=Path(RECEIPT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.publish:
            value = publish_performance_phase_profile_receipt(
                REPOSITORY_ROOT,
                arguments.artifact_root,
                target=arguments.target,
            )
        else:
            value = build_performance_phase_profile_receipt(
                REPOSITORY_ROOT, arguments.artifact_root)
    except (OSError, TypeError, ValueError) as error:
        print(f"performance_p2_phase_receipt: ERROR: {error}")
        return 1
    print(
        "performance_p2_phase_receipt: ready "
        f"(sha256={hashlib.sha256(_canonical_bytes(value)).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "PROFILE_COMMIT", "RECEIPT_PATH",
    "STATUS", "build_performance_phase_profile_receipt",
    "publish_performance_phase_profile_receipt",
    "read_performance_phase_profile_receipt",
]
