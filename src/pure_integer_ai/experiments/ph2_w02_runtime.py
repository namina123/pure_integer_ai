"""正式中文 PH2 W-02 的稳定 shard、学习事务与 V-03 dump orchestrator。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w02_contract import (
    W02_OWNER_KEY,
    W02_RUNNER_KEY,
    W02PayloadAudit,
    W02PayloadFirewall,
    W02RunRequest,
    open_w02_frozen_context,
    validate_w02_request,
)
from pure_integer_ai.experiments.ph2_w02_faults import (
    W02FaultPoint,
    W02InjectedFault,
    hit_w02_fault,
)
from pure_integer_ai.experiments.ph2_w02_learning import (
    OUTCOME_SUCCESS,
    W02LearningReport,
    W02MorphologyTarget,
    open_w02_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w02_shards import (
    W02ShardResult,
    run_w02_training_shards,
)
from pure_integer_ai.experiments.ph2_w02_transaction import (
    W02_TRANSACTION_EVENT_TABLE,
    W02TransactionStore,
)
from pure_integer_ai.experiments.ph2_w02_use import W02AttributionReport
from pure_integer_ai.storage import paths
from pure_integer_ai.storage.backend import SQLiteBackend, StorageBackend
from pure_integer_ai.storage.recovery_package import inspect_recovery_package
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_payload,
    dump_run,
    load_run_package,
)


_PUBLISH_EPOCH = 1
_UNDERSTANDING_PROBE = "研究生命起源"
_GENERATION_PROBE = W02MorphologyTarget(
    construction_key="suffix-hua-construction-v1",
    stem_surface="纸",
)


@dataclass(frozen=True)
class W02RuntimeConfig:
    """W-02 candidate host 的冻结依赖、持久介质和恢复调度。"""

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
    current_remote_commit_sha1: str
    fault_point: str | None = None
    dependency_root: str | Path | None = None


@dataclass(frozen=True)
class W02RunOutcome:
    """一次 W-02 执行或 dump readback 的可比较真实结果。"""

    logical_state_digest: str
    core_digest: str
    memory_digest: str
    use_digest: str
    cursor_digest: str
    artifact_digest: str
    dump_manifest_sha256: str
    execution_state: dict[str, int]
    resource_report: dict[str, int]
    learning_report: W02LearningReport
    attribution_report: W02AttributionReport
    adopted_manifest_count: int
    transaction_event_count: int
    merge_publication_count: int
    dump_readback: bool = False


def _digest(value: Any) -> str:
    """返回无墙钟、无宿主路径的 canonical SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request_payload(request: W02RunRequest) -> dict[str, object]:
    """形成不含 worker/mode 的 staged 事务身份。"""
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
        "run_id": request.run_id,
        "runner_key": request.runner_key,
        "stage_key": request.stage_key,
        "teacher_evidence_paths": list(request.teacher_evidence_paths),
        "w01_receipt_sha256": request.w01_receipt_sha256,
    }


def _request(config: W02RuntimeConfig, context, backend: StorageBackend) -> (
        W02RunRequest):
    """从冻结 context 构造唯一 payload 白名单请求。"""
    request = W02RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=context.stage_key,
        owner_key=W02_OWNER_KEY,
        runner_key=W02_RUNNER_KEY,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        d03_context_key=context.stable_key(),
        w01_receipt_sha256=context.w01_receipt_sha256,
        backend_profile_key=backend.storage_capabilities().stable_key(),
        base_fence_key=config.base_fence_key,
        worker_count=config.worker_count,
        mode=config.mode,
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    return validate_w02_request(context, request)


def _cursor(context, request: W02RunRequest) -> CursorState:
    """把 W-02 完成位封进 V-03 标准 cursor，不触碰 W-03。"""
    return CursorState(
        base_run_id=str(request.base_run_id),
        run_id=str(request.run_id),
        completed={context.stage_ordinal},
        non_skippable=set(),
    )


def _state_projection(backend: StorageBackend, predicate) -> tuple:
    """按稳定表名和持久行序投影 backend 逻辑状态。"""
    schema = backend.schema_snapshot()
    return tuple(
        (table, tuple(tuple(sorted(row.items()))
                      for row in backend.select(table, where=None)))
        for table in sorted(schema)
        if predicate(table, schema[table])
    )


def _state_digests(backend: StorageBackend, learning) -> tuple[str, str, str, str]:
    """分别摘要 Core、Memory、Use，并从中形成排除事务账的 host 摘要。"""
    # state_key() 负责解析 typed Candidate/envelope；物理摘要使用 dump 同源行投影。
    learning.state_key()
    core_state = _state_projection(
        backend, lambda _table, meta: bool(meta["core"]))
    memory_state = _state_projection(
        backend, lambda table, _meta: table.startswith("memory_"))
    host_state = _state_projection(
        backend, lambda table, _meta: table != W02_TRANSACTION_EVENT_TABLE)
    use_state = learning.use_outcomes.state_key()
    core_digest = _digest(core_state)
    memory_digest = _digest(memory_state)
    use_digest = _digest(use_state)
    logical_digest = _digest(host_state)
    return logical_digest, core_digest, memory_digest, use_digest


def _run_consumer_probes(learning) -> None:
    """同事务运行理解和 typed morphology 生成的实际采用归因。"""
    understanding = learning.understand(_UNDERSTANDING_PROBE)
    if len(understanding.active_boundary_candidates) < 2:
        raise RuntimeError("W-02 train probe 未保留多边界 Candidate")
    learning.record_understanding_outcome(
        _UNDERSTANDING_PROBE,
        understanding.active_boundary_candidates[0],
        outcome_kind=OUTCOME_SUCCESS,
        commit=False,
    )
    generated = learning.generate(_GENERATION_PROBE)
    if "纸化" not in generated.surfaces:
        raise RuntimeError("W-02 typed morphology probe 未生成新内容 surface")
    learning.record_generation_outcome(
        _GENERATION_PROBE,
        "纸化",
        outcome_kind=OUTCOME_SUCCESS,
        commit=False,
    )


def _execution_state(audit: W02PayloadAudit) -> dict[str, int]:
    """返回正式 W-02 的诚实启动/读取状态，不越界声称掌握或 W-03。"""
    if audit.teacher_calls != 0:
        raise RuntimeError("W-02 正式运行禁止 teacher call")
    return {
        "W02_STARTED": 1,
        "formal_training_runs": 1,
        "teacher_evidence_reads": audit.teacher_evidence_reads,
        "teacher_calls": 0,
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "mastered_claims": 0,
        "readiness_claims": 0,
        "W03_STARTED": 0,
    }


def _resource_report(
        backend: StorageBackend,
        context,
        audit: W02PayloadAudit,
        shards: W02ShardResult,
        manifest,
        learning_report: W02LearningReport,
        attribution_report: W02AttributionReport,
        ) -> dict[str, int]:
    """核 D-03 八项预算，并返回物理调度与 owner 写的诚实计数。"""
    snapshot = backend.snapshot()
    actual = {
        "max_checkpoint_count": 1,
        "max_logic_operations": sum(len(rows) for rows in snapshot.values()),
        "max_payload_bytes": audit.payload_bytes,
        "max_payload_gets": audit.payload_gets,
        "max_recompute_objects": len(snapshot.get("graph_object", ())),
        "max_records": sum(len(rows) for rows in snapshot.values()),
        "max_segments": len(manifest.segments),
        "max_workers": shards.resource_report["requested_workers"],
    }
    for key, value in actual.items():
        if value > context.resource_budget[key]:
            raise RuntimeError(f"W-02 resource budget exceeded: {key}")
    return {
        **shards.resource_report,
        "actual_checkpoint_count": actual["max_checkpoint_count"],
        "actual_logic_operations": actual["max_logic_operations"],
        "actual_payload_bytes": actual["max_payload_bytes"],
        "actual_payload_gets": actual["max_payload_gets"],
        "actual_recompute_objects": actual["max_recompute_objects"],
        "actual_records": actual["max_records"],
        "actual_segments": actual["max_segments"],
        "core_learning_writes": learning_report.core_learning_writes,
        "memory_learning_writes": learning_report.memory_learning_writes,
        "teacher_evidence_reads": audit.teacher_evidence_reads,
        "teacher_calls": audit.teacher_calls,
        "use_learning_writes": (
            attribution_report.outcome_count
            + attribution_report.assessment_count
            + sum(count for _direction, count
                  in attribution_report.use_count_by_direction)),
        "word_form_writes": learning_report.word_form_writes,
    }


def _manifest_path(config: W02RuntimeConfig) -> Path:
    """返回本 run 的 V-03 唯一可见性指针。"""
    return Path(paths.run_manifest_path(
        str(Path(config.run_root).resolve()), str(config.run_id)))


def _package(context, config: W02RuntimeConfig):
    """只读并完整核验已发布 recovery package。"""
    return inspect_recovery_package(
        str(Path(config.run_root).resolve()),
        str(config.run_id),
        expected_version_key=context.stable_key(),
        expected_publish_epoch=_PUBLISH_EPOCH,
    )


def _outcome(
        *,
        backend: StorageBackend,
        learning,
        learning_report: W02LearningReport,
        audit: W02PayloadAudit,
        shards: W02ShardResult,
        context,
        config: W02RuntimeConfig,
        transaction: W02TransactionStore,
        dump_readback: bool,
        ) -> W02RunOutcome:
    """从持久 host、事务和已封存 package 回读结果。"""
    package = _package(context, config)
    cursor_payload = package.cursor_payload
    if cursor_payload is None:
        raise RuntimeError("W-02 recovery package 缺 cursor")
    cursor_digest = _digest(cursor_payload)
    expected_cursor = cursor_state_payload(_cursor(context, _request(
        config, context, backend)))
    if cursor_payload != expected_cursor:
        raise RuntimeError("W-02 dump cursor identity 漂移")
    logical, core, memory, use = _state_digests(backend, learning)
    events = transaction.events()
    if len(events) < 3:
        raise RuntimeError("W-02 dump 缺少 committed transaction")
    committed = events[2].payload
    if committed != {
            "core_digest": core,
            "cursor_digest": cursor_digest,
            "logical_state_digest": logical,
            "memory_digest": memory,
            "use_digest": use,
            }:
        raise RuntimeError("W-02 committed host/cursor 摘要漂移")
    attribution = learning.attribution_report()
    manifest_path = _manifest_path(config)
    resource = _resource_report(
        backend, context, audit, shards, package.manifest,
        learning_report, attribution)
    return W02RunOutcome(
        logical,
        core,
        memory,
        use,
        cursor_digest,
        shards.artifact_digest,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        _execution_state(audit),
        resource,
        learning_report,
        attribution,
        1,
        len(events),
        shards.merge_publication_count,
        dump_readback,
    )


def _open_context(config: W02RuntimeConfig):
    """在 payload 前回读并冻结 D-03/W-01/current remote identity。"""
    return open_w02_frozen_context(
        Path(config.repository_root).resolve(),
        config.global_manifest_path,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        dependency_root=config.dependency_root,
    )


def run_language_stage1(config: W02RuntimeConfig) -> W02RunOutcome:
    """执行或恢复 W-02；candidate host 永不读取 evaluator 私有 payload。"""
    if not isinstance(config, W02RuntimeConfig):
        raise TypeError("config 必须是 W02RuntimeConfig")
    if (config.fault_point is not None
            and config.fault_point not in W02FaultPoint.injectable_points()):
        raise ValueError("未知 W-02 fault point")
    context = _open_context(config)
    sqlite_path = Path(config.sqlite_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteBackend(str(sqlite_path))
    transaction = None
    try:
        request = _request(config, context, backend)
        transaction = W02TransactionStore(
            backend,
            run_id=request.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        existing_events = transaction.events()
        final_exists = _manifest_path(config).is_file()
        if request.mode == "fresh" and (existing_events or final_exists):
            raise RuntimeError("fresh mode 要求不存在既有 W-02 transaction/run")
        if request.mode == "restart" and (
                not existing_events or final_exists):
            raise RuntimeError("restart mode 只允许恢复未 published 的 W-02 transaction")
        if request.mode == "resume" and (
                not existing_events or not final_exists):
            raise RuntimeError("resume mode 只允许重放已 published 的 W-02 run")
        transaction.begin(_request_payload(request))

        audit = W02PayloadAudit()
        firewall = W02PayloadFirewall.open(
            config.repository_root,
            context,
            request,
            dependency_root=config.dependency_root,
            audit=audit,
        )
        payload = firewall.read_training_payload()
        shards = run_w02_training_shards(
            context,
            request,
            payload,
            sqlite_path,
            fault_point=config.fault_point,
        )
        transaction.preview(shards.preview_payload())

        rollback_state = backend.recovery_state_snapshot()
        learning = open_w02_learning_runtime(backend, mode=request.mode)
        learning_report = learning.consume(shards.payload, commit=False)
        _run_consumer_probes(learning)
        logical, core, memory, use = _state_digests(backend, learning)
        cursor = _cursor(context, request)
        cursor_digest = _digest(cursor_state_payload(cursor))
        hit_w02_fault(
            config.fault_point,
            W02FaultPoint.AFTER_MERGE_BEFORE_COMMIT,
        )
        transaction.commit({
            "core_digest": core,
            "cursor_digest": cursor_digest,
            "logical_state_digest": logical,
            "memory_digest": memory,
            "use_digest": use,
        }, rollback_state=rollback_state)
        hit_w02_fault(
            config.fault_point,
            W02FaultPoint.AFTER_COMMIT_BEFORE_CURSOR,
        )

        manifest_path = _manifest_path(config)
        if manifest_path.is_file():
            package = _package(context, config)
            if package.cursor_payload != cursor_state_payload(cursor):
                raise RuntimeError("已发布 W-02 cursor 与 committed host 漂移")
        else:
            dump_run(
                backend,
                str(Path(config.run_root).resolve()),
                str(request.run_id),
                spaces=None,
                tables=None,
                require_all_spaces=True,
                versions=context.stable_key(),
                publish_epoch=_PUBLISH_EPOCH,
                cursor_state=cursor,
            )
        hit_w02_fault(
            config.fault_point,
            W02FaultPoint.AFTER_MANIFEST_PUBLISH,
        )
        transaction.published({
            "manifest_name": manifest_path.name,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
        })
        if len(transaction.events()) != 4:
            raise RuntimeError("W-02 transaction 未闭合四个显式事件")
        return _outcome(
            backend=backend,
            learning=learning,
            learning_report=learning_report,
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


def load_w02_candidate_dump(
        config: W02RuntimeConfig,
        *,
        target_sqlite_path: str | Path,
        ) -> W02RunOutcome:
    """把权威 W-02 package 加载到全新 SQLite，并从 Core 历史恢复 candidate。"""
    if not isinstance(config, W02RuntimeConfig):
        raise TypeError("config 必须是 W02RuntimeConfig")
    context = _open_context(config)
    target = Path(target_sqlite_path).resolve()
    source = Path(config.sqlite_path).resolve()
    if target == source:
        raise RuntimeError("W-02 dump readback target 必须独立于 candidate host")
    if target.exists() and target.stat().st_size:
        raise RuntimeError("W-02 dump readback target 必须是 fresh SQLite")
    target.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteBackend(str(target))
    transaction = None
    try:
        request = _request(config, context, backend)
        transaction = W02TransactionStore(
            backend,
            run_id=request.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        # 装配一次只为注册完整 schema；恢复包加载前清除装配产生的基础对象。
        open_w02_learning_runtime(backend, mode="restart")
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
            expected_publish_epoch=_PUBLISH_EPOCH,
        )
        expected_cursor = cursor_state_payload(_cursor(context, request))
        if loaded.cursor_payload != expected_cursor:
            raise RuntimeError("W-02 fresh readback cursor identity 漂移")
        events = transaction.events()
        if len(events) != 3:
            raise RuntimeError("W-02 dump 必须绑定 commit 前三事件")

        audit = W02PayloadAudit()
        payload = W02PayloadFirewall.open(
            config.repository_root,
            context,
            request,
            dependency_root=config.dependency_root,
            audit=audit,
        ).read_training_payload()
        learning = open_w02_learning_runtime(backend, mode="resume")
        learning_report = learning.consume(payload, commit=False)
        _run_consumer_probes(learning)
        preview = events[1].payload
        shards = W02ShardResult(
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
            learning_report=learning_report,
            audit=audit,
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
    "W02FaultPoint",
    "W02InjectedFault",
    "W02RunOutcome",
    "W02RuntimeConfig",
    "load_w02_candidate_dump",
    "run_language_stage1",
]
