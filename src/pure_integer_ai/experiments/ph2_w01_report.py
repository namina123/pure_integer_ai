"""W-01 规范 report/cursor/execution bundle 的 immutable 发布与回读。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_w01_contract import (
    W01FrozenContext,
    W01RunRequest,
)
from pure_integer_ai.experiments.ph2_w01_faults import W01FaultPoint
from pure_integer_ai.experiments.ph2_w01_shards import W01ShardResult
from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes


W01_RUN_REPORT_NAME = "w01_protocol_report.json"
W01_RUN_CURSOR_NAME = "w01_cursor.json"
W01_RUN_EXECUTION_NAME = "execution_identity.json"
W01_RUN_RESOURCE_NAME = "resource_report.json"
W01_RUN_MANIFEST_NAME = "run.manifest.json"
W01_RUN_SEAL_NAME = "run.manifest.seal"
W01_RUN_ARTIFACT_VERSION = "PH2-W01-stage0-run-v1"


class W01ReportError(RuntimeError):
    """W-01 bundle 非规范、缺文件、seal/identity 漂移或覆盖冲突。"""


def _sha256(payload: bytes) -> str:
    """返回文件和逻辑投影统一使用的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_object(path: Path) -> dict[str, Any]:
    """读取单换行 canonical JSON object。"""
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W01ReportError(f"W-01 JSON 损坏: {path.name}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W01ReportError(f"W-01 JSON 非规范: {path.name}")
    return value


def _write_bytes(path: Path, payload: bytes) -> None:
    """完整刷写一个 staging 文件，不借 close 代替 flush/fsync。"""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _file_identity(path: Path) -> dict[str, Any]:
    """返回 bundle 内文件的名称、大小和 SHA-256。"""
    payload = path.read_bytes()
    return {
        "name": path.name,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def build_w01_report(
        context: W01FrozenContext,
        request: W01RunRequest,
        shards: W01ShardResult,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """构造 worker 数无关 report/cursor/execution 与独立资源报告。"""
    execution_identity = {
        "backend_profile_key": list(request.backend_profile_key),
        "base_fence_key": list(request.base_fence_key),
        "base_run_id": request.base_run_id,
        "d03_context_key": list(request.d03_context_key),
        "execution_identity_key": list(request.execution_identity_key()),
        "owner_key": request.owner_key,
        "parent_run_id": request.parent_run_id,
        "run_id": request.run_id,
        "runner_key": request.runner_key,
        "stage_key": request.stage_key,
    }
    execution_state = context.verified_zero_learning_state(
        protocol_execution_runs=1)
    semantic = {
        "artifact_digest": shards.artifact_digest,
        "d03_context_key": list(context.stable_key()),
        "execution_identity_key": list(request.execution_identity_key()),
        "stage_key": context.stage_key,
        "status": "W01_PROTOCOL_VERIFIED",
    }
    logical_state_digest = _sha256(canonical_json_bytes(semantic))
    cursor = {
        "artifact_digest": shards.artifact_digest,
        "base_run_id": request.base_run_id,
        "completed_stage_keys": [context.stage_key],
        "cursor_version": context.cursor_version,
        "d03_context_key": list(context.stable_key()),
        "execution_identity_key": list(request.execution_identity_key()),
        "format_version": 1,
        "next_stage_key": context.next_stage_key,
        "parent_run_id": request.parent_run_id,
        "run_id": request.run_id,
        "stage_key": context.stage_key,
        "status": "W01_PROTOCOL_VERIFIED",
        "w02_started": 0,
    }
    report = {
        "artifact_kind": "PH2_W01_STAGE0_PROTOCOL_REPORT",
        "artifact_version": W01_RUN_ARTIFACT_VERSION,
        "d03_identity": {
            "content_commit_sha1": context.d03_content_commit_sha1,
            "context_key": list(context.stable_key()),
            "global_manifest_path": context.global_manifest_path,
            "global_manifest_sha256": context.global_manifest_sha256,
            "receipt_sha256": context.d03_receipt_sha256,
            "release_key": context.d03_release_key,
            "stage_manifest_path": context.stage_manifest_path,
            "stage_manifest_sha256": context.stage_manifest_sha256,
        },
        "execution_identity": execution_identity,
        "execution_state": execution_state,
        "fault_contract": {
            "injectable_points": list(W01FaultPoint.injectable_points()),
            "sqlite_process_restart": 1,
            "sqlite_process_restart_key": W01FaultPoint.SQLITE_PROCESS_RESTART,
        },
        "format_version": 1,
        "honest_boundary": {
            "language_capability_mastered": 0,
            "next_unique_stage": context.next_stage_key,
            "protocol_only": 1,
            "teacher_or_llm_used": 0,
            "w02_started": 0,
        },
        "logical_state_digest": logical_state_digest,
        "metrics": {
            "canonical_persistence_bytes": shards.canonical_artifact_bytes,
            "logical_operations": shards.merged_records,
            "payload_bytes": context.payload_bytes,
            "payload_gets": context.payload_reads,
            "records": len(context.protocol_inputs),
            "segments": 1,
            "startup_protocol_files": context.startup_protocol_file_count,
        },
        "shard_merge": {
            "allowed_worker_counts": list(context.allowed_worker_counts),
            "artifact_digest": shards.artifact_digest,
            "barrier_result_key": list(shards.barrier_result_key),
            "logical_shard_count": shards.logical_shards,
            "merge_barrier_key": context.merge_barrier_key,
            "merge_publication_count": shards.merge_publication_count,
            "merged_records": shards.merged_records,
            "receipt_key": list(shards.receipt_key),
            "worker_count_is_scheduling_only": 1,
        },
        "stage": {
            "next_stage_key": context.next_stage_key,
            "ordinal": context.stage_ordinal,
            "prerequisite_stage_keys": [],
            "stage_key": context.stage_key,
            "status": "W01_PROTOCOL_VERIFIED",
        },
        "status": "W01_PROTOCOL_VERIFIED",
        "version_keys": {key: value for key, value in context.version_keys},
        "visibility": {
            "candidate_allowed_path_count": len(context.candidate_allowed_paths),
            "evaluator_visible_count": context.evaluator_visible_count,
            "future_pack_count": context.future_pack_count,
            "held_out_visible_count": context.held_out_visible_count,
            "payload_bytes": context.payload_bytes,
            "payload_reads": context.payload_reads,
            "train_pack_count": len(context.train_pack_keys),
        },
    }
    resource = {
        "artifact_version": W01_RUN_ARTIFACT_VERSION,
        "run_id": request.run_id,
        **dict(shards.resource_report),
    }
    return report, cursor, execution_identity, resource


def run_directory(run_root: str | Path, run_id: int) -> Path:
    """返回正整数 run 的稳定单层目录。"""
    if type(run_id) is not int or run_id <= 0:
        raise W01ReportError("W-01 run id 必须为正严格整数")
    root = Path(run_root).resolve()
    return root / f"run_{run_id:020d}"


def publish_w01_run(
        run_root: str | Path,
        run_id: int,
        report: dict[str, Any],
        cursor: dict[str, Any],
        execution_identity: dict[str, Any],
        resource_report: dict[str, Any],
        *,
        merge_publication_count: int,
        ) -> Path:
    """在同盘 staging 写齐四文件，最后原子 rename 为唯一 adopted run。"""
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = run_directory(root, run_id)
    if final.is_dir():
        return read_w01_run(final).run_manifest_path
    staging = root / f".{final.name}.staging"
    if not staging.is_relative_to(root) or not final.is_relative_to(root):
        raise W01ReportError("W-01 run 发布路径越界")
    if staging.exists():
        if not staging.is_dir():
            raise W01ReportError("W-01 staging 被非目录对象占用")
        shutil.rmtree(staging)
    staging.mkdir()
    payloads = {
        W01_RUN_REPORT_NAME: canonical_json_bytes(report),
        W01_RUN_CURSOR_NAME: canonical_json_bytes(cursor),
        W01_RUN_EXECUTION_NAME: canonical_json_bytes(execution_identity),
        W01_RUN_RESOURCE_NAME: canonical_json_bytes(resource_report),
    }
    for name, payload in payloads.items():
        _write_bytes(staging / name, payload)
    inventory = tuple(
        _file_identity(staging / name) for name in sorted(payloads))
    manifest = {
        "adopted_manifest_count": 1,
        "artifact_kind": "PH2_W01_STAGE0_RUN_MANIFEST",
        "artifact_version": W01_RUN_ARTIFACT_VERSION,
        "execution_identity_key": execution_identity["execution_identity_key"],
        "file_inventory": list(inventory),
        "format_version": 1,
        "merge_publication_count": merge_publication_count,
        "run_id": run_id,
        "status": "W01_PROTOCOL_VERIFIED",
        "transaction_event_count": 4,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _write_bytes(staging / W01_RUN_MANIFEST_NAME, manifest_bytes)
    _write_bytes(
        staging / W01_RUN_SEAL_NAME,
        (_sha256(manifest_bytes) + "\n").encode("ascii"),
    )
    try:
        os.replace(staging, final)
    except FileExistsError:
        if staging.is_dir():
            shutil.rmtree(staging)
    return read_w01_run(final).run_manifest_path


@dataclass(frozen=True)
class W01RunOutcome:
    """规范回读后的逻辑摘要、报告、cursor 和资源观测。"""

    report: dict[str, Any]
    cursor: dict[str, Any]
    execution_identity: dict[str, Any]
    resource_report: dict[str, Any]
    logical_state_digest: str
    report_digest: str
    cursor_digest: str
    artifact_digest: str
    run_manifest_path: Path
    adopted_manifest_count: int
    merge_publication_count: int
    transaction_event_count: int


def read_w01_run(run_dir: str | Path) -> W01RunOutcome:
    """先核 seal/manifest/inventory，再恢复 report/cursor 和独立资源报告。"""
    root = Path(run_dir).resolve()
    manifest_path = root / W01_RUN_MANIFEST_NAME
    seal_path = root / W01_RUN_SEAL_NAME
    if not manifest_path.is_file() or not seal_path.is_file():
        raise W01ReportError("W-01 run manifest 或 seal 缺失")
    manifest_bytes = manifest_path.read_bytes()
    try:
        expected_seal = seal_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise W01ReportError("W-01 run seal 非法") from exc
    if _sha256(manifest_bytes) != expected_seal:
        raise W01ReportError("W-01 run manifest seal 不匹配")
    manifest = _read_canonical_object(manifest_path)
    if (manifest.get("artifact_kind") != "PH2_W01_STAGE0_RUN_MANIFEST"
            or manifest.get("artifact_version") != W01_RUN_ARTIFACT_VERSION
            or manifest.get("status") != "W01_PROTOCOL_VERIFIED"):
        raise W01ReportError("W-01 run manifest identity/status 非法")
    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise W01ReportError("W-01 run inventory 必须恰好四项")
    expected_names = {
        W01_RUN_REPORT_NAME,
        W01_RUN_CURSOR_NAME,
        W01_RUN_EXECUTION_NAME,
        W01_RUN_RESOURCE_NAME,
    }
    if {item.get("name") for item in inventory if isinstance(item, dict)} != expected_names:
        raise W01ReportError("W-01 run inventory 文件集合漂移")
    for identity in inventory:
        if not isinstance(identity, dict):
            raise W01ReportError("W-01 run inventory identity 非法")
        path = root / str(identity.get("name"))
        actual = _file_identity(path)
        if actual != identity:
            raise W01ReportError("W-01 run inventory 文件身份漂移")
    report = _read_canonical_object(root / W01_RUN_REPORT_NAME)
    cursor = _read_canonical_object(root / W01_RUN_CURSOR_NAME)
    execution = _read_canonical_object(root / W01_RUN_EXECUTION_NAME)
    resource = _read_canonical_object(root / W01_RUN_RESOURCE_NAME)
    if (report.get("status") != "W01_PROTOCOL_VERIFIED"
            or cursor.get("status") != "W01_PROTOCOL_VERIFIED"):
        raise W01ReportError("W-01 report/cursor 未协议验证")
    if (execution.get("execution_identity_key")
            != manifest.get("execution_identity_key")):
        raise W01ReportError("W-01 execution identity 与 manifest 漂移")
    report_payload = (root / W01_RUN_REPORT_NAME).read_bytes()
    cursor_payload = (root / W01_RUN_CURSOR_NAME).read_bytes()
    return W01RunOutcome(
        report=report,
        cursor=cursor,
        execution_identity=execution,
        resource_report=resource,
        logical_state_digest=str(report["logical_state_digest"]),
        report_digest=_sha256(report_payload),
        cursor_digest=_sha256(cursor_payload),
        artifact_digest=str(report["shard_merge"]["artifact_digest"]),
        run_manifest_path=manifest_path,
        adopted_manifest_count=int(manifest["adopted_manifest_count"]),
        merge_publication_count=int(manifest["merge_publication_count"]),
        transaction_event_count=int(manifest["transaction_event_count"]),
    )


__all__ = [
    "W01ReportError",
    "W01RunOutcome",
    "build_w01_report",
    "publish_w01_run",
    "read_w01_run",
    "run_directory",
]
