"""PH2 W-03 独立 candidate host、事务、恢复与 dump 编排。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_artifacts import (
    W03ArtifactStore,
    restore_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_context import open_w03_frozen_context
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationObservation,
    formal_w03_publication_baseline,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_OWNER_KEY,
    W03_RUNNER_KEY,
    W03PayloadAudit,
    W03RunRequest,
    validate_w03_request,
)
from pure_integer_ai.experiments.ph2_w03_faults import (
    W03FaultPoint,
    W03InjectedFault,
    hit_w03_fault,
)
from pure_integer_ai.experiments.ph2_w03_firewall import W03PayloadFirewall
from pure_integer_ai.experiments.ph2_w03_learning import (
    W03LearningResult,
    run_w03_learning,
)
from pure_integer_ai.experiments.ph2_w03_shards import (
    W03ShardResult,
    run_w03_training_shards,
)
from pure_integer_ai.experiments.ph2_w03_transaction import (
    W03_TRANSACTION_EVENT_TABLE,
    W03TransactionStore,
)
from pure_integer_ai.storage import paths
from pure_integer_ai.storage.backend import SQLiteBackend, StorageBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.recovery_package import inspect_recovery_package
from pure_integer_ai.storage.recovery_protocol import RecoveryDependency
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_payload,
    dump_run,
    load_run_package,
)


_PUBLISH_EPOCH = 1
_W02_DEPENDENCY_DESCRIPTOR = (20260730, 102, 3)


@dataclass(frozen=True)
class W03RuntimeConfig:
    """W-03 candidate host 的冻结依赖与物理调度。"""

    repository_root: str | Path
    global_manifest_path: str
    w02_artifacts_root: str | Path
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int
    parent_run_id: int
    base_run_id: int
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    current_remote_commit_sha1: str
    fault_point: str | None = None
    dependency_root: str | Path | None = None


@dataclass(frozen=True)
class W03RunOutcome:
    """一次 W03-04 执行或 fresh dump readback 的可比较证据。"""

    logical_state_digest: str
    candidate_history_digest: str
    projection_digest: str
    generation_digest: str
    cursor_digest: str
    artifact_digest: str
    dump_manifest_sha256: str
    retention_digest: str
    artifact_counts: tuple[tuple[str, int], ...]
    execution_state: dict[str, int]
    resource_report: dict[str, int]
    resource_budget: dict[str, int]
    transaction_event_count: int
    merge_publication_count: int
    adopted_manifest_count: int
    new_learning_write_count: int
    w02_host_write_count: int
    w02_retention_passed: bool
    sqlite_path: str
    owned_tables: tuple[str, ...]
    dump_readback: bool = False


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_physical_roots(config: W03RuntimeConfig) -> None:
    """W-03 SQLite/run root 与 W-02 artifact 树必须双向无包含。"""
    if not isinstance(config, W03RuntimeConfig):
        raise TypeError("config 必须是 W03RuntimeConfig")
    w02 = Path(config.w02_artifacts_root).resolve()
    targets = (
        Path(config.sqlite_path).resolve(),
        Path(config.run_root).resolve(),
    )
    for target in targets:
        if (target == w02
                or target.is_relative_to(w02)
                or w02.is_relative_to(target)):
            raise RuntimeError("W-03 与 W-02 必须使用物理隔离 root")


def _context_for_backend(config: W03RuntimeConfig, backend: StorageBackend):
    """以实际持久 backend profile 重开 context，拒绝 profile 偷换。"""
    continuity = verify_formal_w02_continuity(
        Path(config.repository_root).resolve(),
        Path(config.w02_artifacts_root).resolve(),
    )
    return open_w03_frozen_context(
        Path(config.repository_root).resolve(),
        config.global_manifest_path,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        w02_continuity=continuity,
        publication_baseline=formal_w03_publication_baseline(),
        backend_profile_key=backend.storage_capabilities().stable_key(),
        dependency_root=config.dependency_root,
    )


def _request(config: W03RuntimeConfig, context, backend: StorageBackend):
    return validate_w03_request(context, W03RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=context.stage_key,
        owner_key=W03_OWNER_KEY,
        runner_key=W03_RUNNER_KEY,
        publication_baseline_key=context.publication_baseline.stable_key(),
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        w02_continuity_key=context.w02_continuity.stable_key(),
        d03_context_key=context.stable_key(),
        backend_profile_key=backend.storage_capabilities().stable_key(),
        base_fence_key=config.base_fence_key,
        worker_count=config.worker_count,
        mode=config.mode,
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    ))


def _request_payload(request: W03RunRequest) -> dict[str, object]:
    """排除 mode/worker 的 staged execution identity。"""
    return {
        "backend_profile_key": list(request.backend_profile_key),
        "base_fence_key": list(request.base_fence_key),
        "base_run_id": request.base_run_id,
        "candidate_payload_paths": list(request.candidate_payload_paths),
        "current_remote_commit_sha1": request.current_remote_commit_sha1,
        "d03_context_key": list(request.d03_context_key),
        "execution_identity_key": list(request.execution_identity_key()),
        "owner_key": request.owner_key,
        "parent_run_id": request.parent_run_id,
        "publication_baseline_key": list(request.publication_baseline_key),
        "resource_budget": dict(request.resource_budget),
        "run_id": request.run_id,
        "runner_key": request.runner_key,
        "stage_key": request.stage_key,
        "teacher_evidence_paths": list(request.teacher_evidence_paths),
        "w02_continuity_key": list(request.w02_continuity_key),
    }


def _publication_observation(context) -> W03PublicationObservation:
    """复用 W03-00C 已冻结四项成功证据，不在每个故障注入点重查网络。"""
    baseline = context.publication_baseline
    return W03PublicationObservation(
        local_head_sha1=baseline.head_sha1,
        tracking_head_sha1=baseline.head_sha1,
        remote_head_sha1=baseline.head_sha1,
        ci_run_id=baseline.ci_run_id,
        ci_head_sha1=baseline.head_sha1,
        ci_status="completed",
        ci_conclusion="success",
        ci_jobs=baseline.ci_jobs,
    )


def _cursor(context, request: W03RunRequest) -> CursorState:
    return CursorState(
        base_run_id=str(request.base_run_id),
        run_id=str(request.run_id),
        completed={2, context.stage_ordinal},
        non_skippable=set(),
    )


def _w02_dependency(context) -> tuple[RecoveryDependency, ...]:
    host = dict(context.w02_continuity.host_digests)
    try:
        checksum = tuple(bytes.fromhex(host["manifest"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("W-02 run-3 manifest dependency identity 非法") from exc
    return (RecoveryDependency(
        _W02_DEPENDENCY_DESCRIPTOR,
        context.w02_continuity.base_fence_key(),
        checksum,
    ),)


def _manifest_path(config: W03RuntimeConfig) -> Path:
    return Path(paths.run_manifest_path(
        str(Path(config.run_root).resolve()), str(config.run_id)))


def _package(context, config: W03RuntimeConfig):
    return inspect_recovery_package(
        str(Path(config.run_root).resolve()),
        str(config.run_id),
        expected_version_key=context.stable_key(),
        expected_dependencies=_w02_dependency(context),
        expected_publish_epoch=_PUBLISH_EPOCH,
    )


def _state_projection(backend: StorageBackend) -> tuple:
    schema = backend.schema_snapshot()
    return tuple(
        (table, tuple(tuple(sorted(row.items()))
                      for row in backend.select(table)))
        for table in sorted(schema)
        if table != W03_TRANSACTION_EVENT_TABLE
    )


def _logical_digest(backend: StorageBackend) -> str:
    return _digest(_state_projection(backend))


def _resource_report(
        backend: StorageBackend,
        context,
        audit: W03PayloadAudit,
        shards: W03ShardResult,
        package,
        ) -> dict[str, int]:
    snapshot = backend.snapshot()
    actual = {
        "checkpoint_count": 1,
        "logic_operations": sum(len(rows) for rows in snapshot.values()),
        "payload_bytes": audit.payload_bytes,
        "payload_gets": audit.payload_gets,
        "recompute_objects": len(snapshot.get(GRAPH_OBJECT_TABLE, ())),
        "records": sum(len(rows) for rows in snapshot.values()),
        "segments": len(package.manifest.segments),
        "workers": shards.resource_report["requested_workers"],
    }
    for suffix, value in actual.items():
        budget_key = f"max_{suffix}"
        if value > context.resource_budget[budget_key]:
            raise RuntimeError(f"W-03 resource budget exceeded: {budget_key}")
    return {
        **shards.resource_report,
        **{f"actual_{key}": value for key, value in actual.items()},
        "teacher_evidence_reads": audit.teacher_evidence_reads,
        "teacher_calls": audit.teacher_calls,
    }


def _outcome(
        *,
        backend: StorageBackend,
        learning: W03LearningResult,
        audit: W03PayloadAudit,
        shards: W03ShardResult,
        context,
        config: W03RuntimeConfig,
        transaction: W03TransactionStore,
        dump_readback: bool,
        ) -> W03RunOutcome:
    package = _package(context, config)
    if package.cursor_payload is None:
        raise RuntimeError("W-03 recovery package 缺 cursor")
    expected_cursor = cursor_state_payload(_cursor(
        context, _request(config, context, backend)))
    if package.cursor_payload != expected_cursor:
        raise RuntimeError("W-03 dump cursor identity 漂移")
    cursor_digest = _digest(package.cursor_payload)
    logical = _logical_digest(backend)
    events = transaction.events()
    if len(events) < 3:
        raise RuntimeError("W-03 dump 缺 committed transaction")
    committed = events[2].payload
    expected_commit = {
        "artifact_counts": [list(item) for item in learning.artifact_counts],
        "candidate_history_digest": learning.candidate_history_digest,
        "cursor_digest": cursor_digest,
        "generation_digest": learning.generation_digest,
        "logical_state_digest": logical,
        "projection_digest": learning.projection_digest,
        "retention_digest": learning.retention_digest,
    }
    if committed != expected_commit:
        raise RuntimeError("W-03 committed host/cursor 摘要漂移")
    schema = backend.schema_snapshot()
    w02_tables = tuple(name for name in schema if name.startswith("ph2_w02_"))
    if w02_tables:
        raise RuntimeError("W-03 candidate host 混入 W-02 owned table")
    owned = tuple(sorted(
        name for name in schema if backend.count(name) > 0))
    retention = learning.artifact_store.payloads("W02_RETENTION")
    retention_passed = (
        len(retention) == 1
        and retention[0].get("continuity_key")
        == list(context.w02_continuity.stable_key())
        and retention[0].get("fail_count") == 0
        and retention[0].get("ne_count") == 0
    )
    if not retention_passed:
        raise RuntimeError("W-02 retention artifact 未闭合")
    manifest_path = _manifest_path(config)
    return W03RunOutcome(
        logical,
        learning.candidate_history_digest,
        learning.projection_digest,
        learning.generation_digest,
        cursor_digest,
        shards.artifact_digest,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        learning.retention_digest,
        learning.artifact_counts,
        dict(context.execution_state),
        _resource_report(backend, context, audit, shards, package),
        dict(context.resource_budget),
        len(events),
        shards.merge_publication_count,
        1,
        learning.new_learning_write_count,
        0,
        True,
        str(Path(config.sqlite_path).resolve()),
        owned,
        dump_readback,
    )


def run_language_stage2(config: W03RuntimeConfig) -> W03RunOutcome:
    """执行或恢复 W03-04 candidate host；不读取 evaluator/private payload。"""
    _validate_physical_roots(config)
    if (config.fault_point is not None
            and config.fault_point not in W03FaultPoint.injectable_points()):
        raise ValueError("未知 W-03 fault point")
    sqlite_path = Path(config.sqlite_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteBackend(str(sqlite_path))
    transaction = None
    try:
        context = _context_for_backend(config, backend)
        request = _request(config, context, backend)
        transaction = W03TransactionStore(
            backend,
            run_id=request.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        existing = transaction.events()
        final_exists = _manifest_path(config).is_file()
        if request.mode == "fresh" and (existing or final_exists):
            raise RuntimeError("fresh mode 要求不存在既有 W-03 transaction/run")
        if request.mode == "restart" and (not existing or final_exists):
            raise RuntimeError("restart mode 只允许恢复未 published 的 W-03 run")
        if request.mode == "resume" and (not existing or not final_exists):
            raise RuntimeError("resume mode 只允许重放已 published 的 W-03 run")
        committed = len(existing) >= 3
        transaction.begin(_request_payload(request))

        audit = W03PayloadAudit()
        if committed:
            payload = restore_training_payload(W03ArtifactStore(backend))
        else:
            payload = W03PayloadFirewall.open(
                config.repository_root,
                context,
                request,
                publication_observation=_publication_observation(context),
                dependency_root=config.dependency_root,
                audit=audit,
            ).read_training_payload()
        shards = run_w03_training_shards(
            context,
            request,
            payload,
            sqlite_path,
            fault_point=config.fault_point,
        )
        transaction.preview(shards.preview_payload())

        rollback_state = backend.recovery_state_snapshot()
        learning = run_w03_learning(
            backend,
            shards.payload,
            context,
            restore=committed,
        )
        cursor = _cursor(context, request)
        cursor_digest = _digest(cursor_state_payload(cursor))
        logical = _logical_digest(backend)
        commit_payload = {
            "artifact_counts": [list(item) for item in learning.artifact_counts],
            "candidate_history_digest": learning.candidate_history_digest,
            "cursor_digest": cursor_digest,
            "generation_digest": learning.generation_digest,
            "logical_state_digest": logical,
            "projection_digest": learning.projection_digest,
            "retention_digest": learning.retention_digest,
        }
        hit_w03_fault(
            config.fault_point,
            W03FaultPoint.AFTER_MERGE_BEFORE_COMMIT,
        )
        transaction.commit(commit_payload, rollback_state=rollback_state)
        hit_w03_fault(
            config.fault_point,
            W03FaultPoint.AFTER_COMMIT_BEFORE_CURSOR,
        )

        manifest_path = _manifest_path(config)
        if manifest_path.is_file():
            package = _package(context, config)
            if package.cursor_payload != cursor_state_payload(cursor):
                raise RuntimeError("已发布 W-03 cursor 与 committed host 漂移")
        else:
            dump_run(
                backend,
                str(Path(config.run_root).resolve()),
                str(request.run_id),
                spaces=None,
                tables=None,
                require_all_spaces=True,
                versions=context.stable_key(),
                dependencies=_w02_dependency(context),
                publish_epoch=_PUBLISH_EPOCH,
                cursor_state=cursor,
            )
        hit_w03_fault(
            config.fault_point,
            W03FaultPoint.AFTER_MANIFEST_PUBLISH,
        )
        transaction.published({
            "manifest_name": manifest_path.name,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
        })
        if len(transaction.events()) != 4:
            raise RuntimeError("W-03 transaction 未闭合四个显式事件")
        return _outcome(
            backend=backend,
            learning=learning,
            audit=audit,
            shards=shards,
            context=context,
            config=config,
            transaction=transaction,
            dump_readback=False,
        )
    finally:
        if transaction is not None:
            transaction.close()
        else:
            backend.close()


def load_w03_candidate_dump(
        config: W03RuntimeConfig,
        *,
        target_sqlite_path: str | Path,
        ) -> W03RunOutcome:
    """把 commit-bound package 加载到 fresh SQLite 并恢复全部 W-03 状态。"""
    _validate_physical_roots(config)
    target = Path(target_sqlite_path).resolve()
    source = Path(config.sqlite_path).resolve()
    w02 = Path(config.w02_artifacts_root).resolve()
    if target == source:
        raise RuntimeError("W-03 dump readback target 必须独立于 candidate host")
    if target == w02 or target.is_relative_to(w02) or w02.is_relative_to(target):
        raise RuntimeError("W-03 dump readback target 与 W-02 root 未隔离")
    if target.exists() and target.stat().st_size:
        raise RuntimeError("W-03 dump readback target 必须是 fresh SQLite")
    target.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteBackend(str(target))
    transaction = None
    try:
        context = _context_for_backend(config, backend)
        request = _request(config, context, backend)
        transaction = W03TransactionStore(
            backend,
            run_id=request.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        make_train_context(backend)
        store = W03ArtifactStore(backend)
        blank = {table: [] for table in backend.snapshot()}
        backend.restore_recovery_state({
            "tables": blank,
            "id_pool": {},
            "isa_edge_gen": {},
        })
        backend.commit()
        loaded = load_run_package(
            backend,
            str(Path(config.run_root).resolve()),
            str(config.run_id),
            expected_versions=context.stable_key(),
            expected_dependencies=_w02_dependency(context),
            expected_publish_epoch=_PUBLISH_EPOCH,
        )
        expected_cursor = cursor_state_payload(_cursor(context, request))
        if loaded.cursor_payload != expected_cursor:
            raise RuntimeError("W-03 fresh readback cursor identity 漂移")
        events = transaction.events()
        if len(events) != 3:
            raise RuntimeError("W-03 dump 必须绑定 commit 前三事件")
        payload = restore_training_payload(store)
        learning = run_w03_learning(
            backend, payload, context, restore=True)
        preview = events[1].payload
        shards = W03ShardResult(
            payload,
            str(preview["artifact_digest"]),
            tuple(preview["barrier_result_key"]),
            tuple(preview["receipt_key"]),
            int(preview["logical_shards"]),
            int(preview["merged_records"]),
            int(preview["merge_publication_count"]),
            0,
            {
                "canonical_segment_bytes": 0,
                "in_flight_shard_limit": config.worker_count,
                "logical_shards": int(preview["logical_shards"]),
                "merged_records": int(preview["merged_records"]),
                "produced_shards": 0,
                "raw_records": int(preview["merged_records"]),
                "requested_workers": config.worker_count,
                "restored_shards": int(preview["logical_shards"]),
                "sealed_cold_bytes": 0,
                "worker_byte_limit": context.resource_budget["max_payload_bytes"],
                "worker_object_limit": context.resource_budget["max_records"],
            },
        )
        return _outcome(
            backend=backend,
            learning=learning,
            audit=W03PayloadAudit(),
            shards=shards,
            context=context,
            config=config,
            transaction=transaction,
            dump_readback=True,
        )
    finally:
        if transaction is not None:
            transaction.close()
        else:
            backend.close()


__all__ = [
    "W03FaultPoint",
    "W03InjectedFault",
    "W03RunOutcome",
    "W03RuntimeConfig",
    "load_w03_candidate_dump",
    "run_language_stage2",
]
