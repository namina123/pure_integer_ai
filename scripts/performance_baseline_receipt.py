"""发布 PERF-P0 的 append-only、非 readiness 性能基线 receipt。"""
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
    CONTRACT as BASELINE_CONTRACT,
    PerformanceBaselineError,
    read_head,
    read_state,
    require_external_state_root,
    sha256_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_BASELINE_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-BASELINE-P0-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/performance_baseline_receipt_v1.json"
STATUS = "PERFORMANCE_BASELINE_EVIDENCED"
SCENARIOS = (
    "long_input_hierarchy",
    "long_session_checkpoint",
    "long_memory_projection",
    "storage_dict",
    "storage_sqlite",
)


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _digest(value: object, *, length: int = 64) -> None:
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("receipt digest 无效")


def _exact(value: object, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} 字段漂移")
    return value


def _resolve_external(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("external artifact path 无效")
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts
            or ":" in path.parts[0]):
        raise ValueError("external artifact path 无效")
    return root.joinpath(*path.parts)


def _artifact_identity(root: Path, relative: str) -> dict[str, object]:
    path = _resolve_external(root, relative)
    payload = path.read_bytes()
    return {
        "relative_path": relative.replace("\\", "/"),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _state_artifacts(
        state_root: Path,
        state: dict[str, Any],
        ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    logs: list[dict[str, object]] = []
    databases: list[dict[str, object]] = []
    for name in SCENARIOS:
        result = state["results"].get(name)
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise ValueError(f"P0 场景未 PASS: {name}")
        attempts = result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"P0 场景没有 attempt: {name}")
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "PASS":
                raise ValueError(f"P0 场景存在非 PASS attempt: {name}")
            number = attempt.get("attempt")
            log_digest = attempt.get("log_sha256")
            if type(number) is not int or number < 1:
                raise ValueError(f"P0 attempt 编号无效: {name}")
            _digest(log_digest)
            log_relative = (
                f"logs/{name.replace('_', '-')}-attempt-{number:03d}.log"
            )
            identity = _artifact_identity(state_root, log_relative)
            if identity["sha256"] != log_digest:
                raise ValueError(f"P0 log 摘要不匹配: {log_relative}")
            logs.append(identity)
            if name == "storage_sqlite":
                database_relative = (
                    f"databases/storage-sqlite-attempt-{number:03d}.sqlite3"
                )
                databases.append(
                    _artifact_identity(state_root, database_relative))
    logs.sort(key=lambda item: item["relative_path"])
    databases.sort(key=lambda item: item["relative_path"])
    return logs, databases


def build_performance_baseline_receipt(
        repository_root: str | Path,
        checkpoint_root: str | Path,
        ) -> dict[str, Any]:
    """从 Git 外 PASS checkpoint 构造不可覆盖 receipt。"""
    root = Path(repository_root).resolve()
    state_root = Path(checkpoint_root).resolve()
    require_external_state_root(root, state_root)
    state = read_state(state_root)
    manifest = state.get("manifest")
    if (state.get("aggregate_status") != "PASS"
            or not isinstance(manifest, dict)
            or manifest.get("contract") != BASELINE_CONTRACT
            or tuple(state.get("selected_scenarios", ())) != SCENARIOS):
        raise ValueError("P0 checkpoint 不是五场景 PASS")
    head = manifest.get("head")
    if head != read_head(root):
        raise ValueError("P0 checkpoint HEAD 与当前仓库不一致")
    transition = manifest.get("readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("P0 checkpoint readiness transition 漂移")
    logs, databases = _state_artifacts(state_root, state)
    state_payload = (state_root / "state.json").read_bytes()
    manifest_payload = _canonical_bytes(manifest)
    scenario_results = []
    for name in SCENARIOS:
        result = deepcopy(state["results"][name])
        scenario_results.append({
            "attempt_count": len(result["attempts"]),
            "name": name,
            "result": result,
        })
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "baseline_contract": BASELINE_CONTRACT,
        "baseline_head": head,
        "checkpoint_root_name": state_root.name,
        "checkpoint_state": {
            "relative_path": "state.json",
            "size_bytes": len(state_payload),
            "sha256": sha256_bytes(state_payload),
        },
        "format_version": FORMAT_VERSION,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "readiness_transition": deepcopy(transition),
        "scenario_results": scenario_results,
        "external_logs": logs,
        "external_databases": databases,
        "status": STATUS,
        "verification": {
            "all_scenarios_pass": 1,
            "canonical_checkpoint": 1,
            "log_digests_verified": 1,
            "sqlite_artifact_verified": 1,
            "retry_count": 0,
            "scale": manifest["scale"],
            "repetitions": manifest["repetitions"],
        },
    }


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "baseline_contract",
        "baseline_head", "checkpoint_root_name", "checkpoint_state",
        "external_databases", "external_logs", "format_version",
        "manifest_sha256", "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "scenario_results", "status", "verification",
    }, where="performance baseline receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["baseline_contract"] != BASELINE_CONTRACT
            or raw["format_version"] != FORMAT_VERSION
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != STATUS):
        raise ValueError("performance baseline receipt 固定身份漂移")
    _digest(raw["baseline_head"], length=40)
    _digest(raw["manifest_sha256"])
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("performance baseline 不得转移 readiness")
    checkpoint = _exact(raw["checkpoint_state"], {
        "relative_path", "size_bytes", "sha256",
    }, where="checkpoint_state")
    if (checkpoint["relative_path"] != "state.json"
            or type(checkpoint["size_bytes"]) is not int
            or checkpoint["size_bytes"] < 1):
        raise ValueError("checkpoint_state identity 无效")
    _digest(checkpoint["sha256"])
    checkpoint_root_name = raw["checkpoint_root_name"]
    if (not isinstance(checkpoint_root_name, str)
            or not checkpoint_root_name
            or PurePosixPath(checkpoint_root_name).name != checkpoint_root_name
            or "\\" in checkpoint_root_name
            or ":" in checkpoint_root_name):
        raise ValueError("checkpoint_root_name 无效")
    scenario_results = raw["scenario_results"]
    if not isinstance(scenario_results, list):
        raise ValueError("scenario_results 必须为 list")
    if any(not isinstance(item, dict) for item in scenario_results):
        raise ValueError("scenario_result 必须为 object")
    if tuple(item.get("name") for item in scenario_results) != SCENARIOS:
        raise ValueError("scenario_results 顺序或数量漂移")
    for item in scenario_results:
        entry = _exact(item, {"attempt_count", "name", "result"}, where="scenario_result")
        if type(entry["attempt_count"]) is not int or entry["attempt_count"] < 1:
            raise ValueError("scenario_result attempt_count 无效")
        result = entry["result"]
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise ValueError("scenario_result 必须 PASS")
        attempts = result.get("attempts")
        if (not isinstance(attempts, list)
                or entry["attempt_count"] != len(attempts)
                or any(not isinstance(attempt, dict)
                       or attempt.get("status") != "PASS"
                       for attempt in attempts)):
            raise ValueError("scenario_result attempts 必须全部 PASS")
    expected_logs = sum(item["attempt_count"] for item in scenario_results)
    expected_databases = next(
        item["attempt_count"] for item in scenario_results
        if item["name"] == "storage_sqlite"
    )
    if (len(raw["external_logs"]) != expected_logs
            or len(raw["external_databases"]) != expected_databases):
        raise ValueError("external artifact 数量与 attempt 不一致")
    for field in ("external_logs", "external_databases"):
        artifacts = raw[field]
        if not isinstance(artifacts, list):
            raise ValueError(f"{field} 必须为 list")
        previous = ""
        for artifact in artifacts:
            entry = _exact(artifact, {"relative_path", "size_bytes", "sha256"}, where=field)
            path = entry["relative_path"]
            if not isinstance(path, str) or path <= previous:
                raise ValueError(f"{field} 路径未严格排序或越界")
            _resolve_external(Path.cwd(), path)
            if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 1:
                raise ValueError(f"{field} size 无效")
            _digest(entry["sha256"])
            previous = path
    verification = _exact(raw["verification"], {
        "all_scenarios_pass", "canonical_checkpoint", "log_digests_verified",
        "repetitions", "retry_count", "scale", "sqlite_artifact_verified",
    }, where="verification")
    if any(verification[key] != 1 for key in (
            "all_scenarios_pass", "canonical_checkpoint", "log_digests_verified",
            "sqlite_artifact_verified")) or verification["retry_count"] != 0:
        raise ValueError("verification 固定证据不满足")
    if (type(verification["scale"]) is not int
            or verification["scale"] < 1
            or type(verification["repetitions"]) is not int
            or verification["repetitions"] < 1):
        raise ValueError("verification 规模必须为整数")


def read_performance_baseline_receipt(
        repository_root: str | Path,
        checkpoint_root: str | Path | None = None,
        path: str | Path = RECEIPT_PATH,
        *, verify_external: bool = True,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("performance baseline receipt newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError("performance baseline receipt canonical bytes 漂移")
    _validate(value)
    if not verify_external:
        return value
    if checkpoint_root is None:
        raise ValueError("严格回读 performance baseline receipt 需要 checkpoint_root")
    state_root = Path(checkpoint_root).resolve()
    require_external_state_root(root, state_root)
    state_payload = (state_root / "state.json").read_bytes()
    if value["checkpoint_state"] != {
            "relative_path": "state.json",
            "size_bytes": len(state_payload),
            "sha256": sha256_bytes(state_payload),
    }:
        raise ValueError("P0 checkpoint identity 漂移")
    state = read_state(state_root)
    manifest = state["manifest"]
    if value["checkpoint_root_name"] != state_root.name:
        raise ValueError("P0 checkpoint root name 漂移")
    if value["baseline_head"] != manifest["head"]:
        raise ValueError("P0 checkpoint HEAD 声明漂移")
    if value["manifest_sha256"] != sha256_bytes(_canonical_bytes(manifest)):
        raise ValueError("P0 manifest identity 漂移")
    expected_results = [
        {
            "attempt_count": len(state["results"][name]["attempts"]),
            "name": name,
            "result": state["results"][name],
        }
        for name in SCENARIOS
    ]
    if value["scenario_results"] != expected_results:
        raise ValueError("P0 scenario results 漂移")
    for field in ("external_logs", "external_databases"):
        for artifact in value[field]:
            current = _artifact_identity(state_root, artifact["relative_path"])
            if current != artifact:
                raise ValueError(f"P0 external artifact identity 漂移: {artifact['relative_path']}")
    return value


def publish_performance_baseline_receipt(
        repository_root: str | Path,
        checkpoint_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("performance baseline receipt 已发布，禁止覆盖")
    value = build_performance_baseline_receipt(root, checkpoint_root)
    payload = _canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("performance baseline receipt 已发布，禁止覆盖") from error
    restored = read_performance_baseline_receipt(root, checkpoint_root, destination)
    if restored != value:
        raise ValueError("performance baseline receipt 发布回读不一致")
    return restored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布 PERF-P0 baseline receipt。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--target", type=Path, default=Path(RECEIPT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    try:
        if args.publish:
            value = publish_performance_baseline_receipt(
                REPOSITORY_ROOT, args.checkpoint, target=args.target)
        else:
            value = build_performance_baseline_receipt(
                REPOSITORY_ROOT, args.checkpoint)
    except (OSError, UnicodeError, ValueError, PerformanceBaselineError) as error:
        print(f"performance_baseline_receipt: ERROR: {error}")
        return 1
    payload = _canonical_bytes(value)
    print(
        "performance_baseline_receipt: ready "
        f"(sha256={hashlib.sha256(payload).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "FORMAT_VERSION", "RECEIPT_PATH",
    "STATUS", "build_performance_baseline_receipt",
    "publish_performance_baseline_receipt", "read_performance_baseline_receipt",
]
