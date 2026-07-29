"""独立正式中文 PH2 W-01 阶段 0 orchestrator 薄入口。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_w01_contract import (
    W01_OWNER_KEY,
    W01_RUNNER_KEY,
    W01RunRequest,
    open_w01_frozen_context,
    validate_w01_request,
)
from pure_integer_ai.experiments.ph2_w01_faults import (
    W01FaultPoint,
    W01InjectedFault,
    hit_w01_fault,
)
from pure_integer_ai.experiments.ph2_w01_report import (
    W01RunOutcome,
    build_w01_report,
    publish_w01_run,
    read_w01_run,
    run_directory,
)
from pure_integer_ai.experiments.ph2_w01_shards import run_w01_protocol_shards
from pure_integer_ai.experiments.ph2_w01_transaction import W01TransactionStore
from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes
from pure_integer_ai.storage.backend import SQLiteBackend


@dataclass(frozen=True)
class W01RuntimeConfig:
    """W-01 orchestrator 的仓库、持久介质、run 和恢复配置。"""

    repository_root: str | Path
    global_manifest_path: str
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int
    parent_run_id: int
    base_run_id: int
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    fault_point: str | None = None
    dependency_root: str | Path | None = None


def _digest(value: dict[str, object]) -> str:
    """返回事务 payload 使用的规范 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request_payload(request: W01RunRequest) -> dict[str, object]:
    """形成不含物理 worker/mode 的 staged 事务身份。"""
    return {
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


def run_language_stage0(config: W01RuntimeConfig) -> W01RunOutcome:
    """执行或恢复 W-01 协议事务；不调用 formal_train、teacher 或 evaluator。"""
    if not isinstance(config, W01RuntimeConfig):
        raise TypeError("config 必须是 W01RuntimeConfig")
    if (config.fault_point is not None
            and config.fault_point not in W01FaultPoint.injectable_points()):
        raise ValueError("未知 W-01 fault point")
    repository_root = Path(config.repository_root).resolve()
    run_root = Path(config.run_root).resolve()
    sqlite_path = Path(config.sqlite_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    context = open_w01_frozen_context(
        repository_root,
        config.global_manifest_path,
        dependency_root=config.dependency_root,
    )
    backend = SQLiteBackend(str(sqlite_path))
    request = W01RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=context.stage_key,
        owner_key=W01_OWNER_KEY,
        runner_key=W01_RUNNER_KEY,
        d03_context_key=context.stable_key(),
        backend_profile_key=backend.storage_capabilities().stable_key(),
        base_fence_key=config.base_fence_key,
        worker_count=config.worker_count,
        mode=config.mode,
        requested_payload_paths=(),
    )
    validate_w01_request(context, request)
    transaction = W01TransactionStore(
        backend,
        run_id=request.run_id,
        execution_identity_key=request.execution_identity_key(),
    )
    try:
        existing_events = transaction.events()
        existing_final = run_directory(run_root, request.run_id).is_dir()
        if request.mode == "fresh" and (existing_events or existing_final):
            raise RuntimeError("fresh mode 要求不存在既有 W-01 transaction/run")
        if request.mode == "restart" and (
                not existing_events or existing_final):
            raise RuntimeError("restart mode 只允许恢复未 adopted 的既有 transaction")
        if request.mode == "resume" and (
                not existing_events or not existing_final):
            raise RuntimeError("resume mode 只允许重放已 adopted 的 W-01 run")
        begin_payload = _request_payload(request)
        transaction.begin(begin_payload)
        final_dir = run_directory(run_root, request.run_id)
        if final_dir.is_dir():
            outcome = read_w01_run(final_dir)
            if (outcome.execution_identity.get("execution_identity_key")
                    != list(request.execution_identity_key())):
                raise RuntimeError("已发布 W-01 run execution identity 漂移")
            events = transaction.events()
            if len(events) not in {3, 4}:
                raise RuntimeError("adopted W-01 manifest 缺少已提交事务")
            preview = events[1].payload
            committed = events[2].payload
            if (preview.get("artifact_digest") != outcome.artifact_digest
                    or preview.get("logical_state_digest")
                    != outcome.logical_state_digest
                    or committed != {
                        "cursor_digest": outcome.cursor_digest,
                        "logical_state_digest": outcome.logical_state_digest,
                        "report_digest": outcome.report_digest,
                    }):
                raise RuntimeError("adopted W-01 bundle 与事务 preview/commit 漂移")
            transaction.published({
                "manifest_name": outcome.run_manifest_path.name,
                "manifest_sha256": hashlib.sha256(
                    outcome.run_manifest_path.read_bytes()).hexdigest(),
            })
            return outcome

        shards = run_w01_protocol_shards(
            context,
            request,
            sqlite_path,
            fault_point=config.fault_point,
        )
        report, cursor, execution, resource = build_w01_report(
            context, request, shards)
        preview_payload = {
            **shards.preview_payload(),
            "logical_state_digest": report["logical_state_digest"],
        }
        transaction.preview(preview_payload)
        hit_w01_fault(
            config.fault_point,
            W01FaultPoint.AFTER_MERGE_BEFORE_COMMIT,
        )
        report_digest = _digest(report)
        cursor_digest = _digest(cursor)
        transaction.commit({
            "cursor_digest": cursor_digest,
            "logical_state_digest": report["logical_state_digest"],
            "report_digest": report_digest,
        })
        hit_w01_fault(
            config.fault_point,
            W01FaultPoint.AFTER_COMMIT_BEFORE_CURSOR,
        )
        manifest_path = publish_w01_run(
            run_root,
            request.run_id,
            report,
            cursor,
            execution,
            resource,
            merge_publication_count=shards.merge_publication_count,
        )
        hit_w01_fault(
            config.fault_point,
            W01FaultPoint.AFTER_MANIFEST_PUBLISH,
        )
        transaction.published({
            "manifest_name": manifest_path.name,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
        })
        if transaction.event_count() != 4:
            raise RuntimeError("W-01 transaction 未闭合四个显式事件")
        return read_w01_run(manifest_path.parent)
    finally:
        transaction.close()


__all__ = [
    "W01FaultPoint",
    "W01InjectedFault",
    "W01RunOutcome",
    "W01RuntimeConfig",
    "read_w01_run",
    "run_language_stage0",
]
