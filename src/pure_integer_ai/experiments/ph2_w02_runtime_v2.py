"""W-02 v2 正式 runtime：以当前 Observation Evidence 执行形态生成归因。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_morphology_course import PAYLOAD_KIND
from pure_integer_ai.experiments.ph2_w02_contract import (
    W02PayloadAudit,
    W02PayloadFirewall,
    W02TrainingPayload,
)
from pure_integer_ai.experiments.ph2_w02_faults import (
    W02FaultPoint,
    W02InjectedFault,
    hit_w02_fault,
)
from pure_integer_ai.experiments.ph2_w02_learning import OUTCOME_SUCCESS
from pure_integer_ai.experiments.ph2_w02_learning_v2 import (
    W02LearningRuntimeV2,
    W02MorphologyTargetV2,
    build_w02_morphology_target_v2,
    morphology_unit_evidence_v2,
    open_w02_learning_runtime_v2,
)
from pure_integer_ai.experiments.ph2_w02_runtime import (
    W02RunOutcome,
    W02RuntimeConfig,
    _PUBLISH_EPOCH,
    _cursor,
    _digest,
    _manifest_path,
    _open_context,
    _outcome,
    _package,
    _request,
    _request_payload,
    _state_digests,
)
from pure_integer_ai.experiments.ph2_w02_shards import (
    W02ShardResult,
    run_w02_training_shards,
)
from pure_integer_ai.experiments.ph2_w02_transaction import W02TransactionStore
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.training.cursor import (
    cursor_state_payload,
    dump_run,
    load_run_package,
)


W02_FORMAL_RUNTIME_VERSION = "PH2-W02-formal-runtime-v2"
_UNDERSTANDING_PROBE = "研究生命起源"
_PROBE_SOURCE_CANDIDATE_ID = "teacher-unknown-candidate-v1"
_PROBE_CONSTRUCTION_KEY = "suffix-hua-construction-v1"


def train_generation_probe_target_v2(
        payload: W02TrainingPayload,
        ) -> W02MorphologyTargetV2:
    """从唯一 public train Observation 提取未入词形表的来源化 probe stem。"""
    if not isinstance(payload, W02TrainingPayload):
        raise TypeError("W-02 v2 probe payload 类型错误")
    matches = []
    for observation in payload.observations:
        if observation.payload_kind != PAYLOAD_KIND:
            continue
        value = observation.typed_payload.to_value()
        if value["candidate_id"] == _PROBE_SOURCE_CANDIDATE_ID:
            matches.append((observation, value))
    if len(matches) != 1:
        raise RuntimeError("W-02 v2 probe Observation 不唯一或不存在")
    observation, value = matches[0]
    stems = tuple(
        item for item in value["analysis_units"]
        if item["unit_kind"] == "STEM")
    if len(stems) != 1:
        raise RuntimeError("W-02 v2 probe Observation 缺唯一 STEM")
    stem = morphology_unit_evidence_v2(observation, stems[0]["unit_id"])
    return build_w02_morphology_target_v2(_PROBE_CONSTRUCTION_KEY, stem)


def _run_consumer_probes_v2(
        learning: W02LearningRuntimeV2,
        payload: W02TrainingPayload,
        ) -> None:
    """同事务运行理解和 evidence-bound 生成，并只归因实际采用 Candidate。"""
    understanding = learning.understand(_UNDERSTANDING_PROBE)
    if len(understanding.active_boundary_candidates) < 2:
        raise RuntimeError("W-02 v2 train probe 未保留多边界 Candidate")
    learning.record_understanding_outcome(
        _UNDERSTANDING_PROBE,
        understanding.active_boundary_candidates[0],
        outcome_kind=OUTCOME_SUCCESS,
        commit=False,
    )
    target = train_generation_probe_target_v2(payload)
    if learning.word_forms.lookup(
            target.stem_surface, branch=learning.branch) is not None:
        raise RuntimeError("W-02 v2 generation probe stem 已落入历史词形表")
    generated = learning.generate(target)
    if len(generated.surfaces) != 1:
        raise RuntimeError("W-02 v2 typed morphology probe 未形成唯一新 surface")
    observed_surfaces = {
        observation.typed_payload.to_value()["observed_surface"]["text"]
        for observation in payload.observations
        if observation.payload_kind == PAYLOAD_KIND
    }
    chosen = generated.surfaces[0]
    if chosen in observed_surfaces:
        raise RuntimeError("W-02 v2 generation probe 退化为训练整词回放")
    learning.record_generation_outcome(
        target,
        chosen,
        outcome_kind=OUTCOME_SUCCESS,
        commit=False,
    )


def run_language_stage1_v2(config: W02RuntimeConfig) -> W02RunOutcome:
    """执行或恢复 W-02 v2 candidate host，永不读取 evaluator 私有 payload。"""
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
            raise RuntimeError("fresh mode 要求不存在既有 W-02 v2 transaction/run")
        if request.mode == "restart" and (
                not existing_events or final_exists):
            raise RuntimeError(
                "restart mode 只允许恢复未 published 的 W-02 v2 transaction")
        if request.mode == "resume" and (
                not existing_events or not final_exists):
            raise RuntimeError(
                "resume mode 只允许重放已 published 的 W-02 v2 run")
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
        learning = open_w02_learning_runtime_v2(backend, mode=request.mode)
        learning_report = learning.consume(shards.payload, commit=False)
        _run_consumer_probes_v2(learning, shards.payload)
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
                raise RuntimeError("已发布 W-02 v2 cursor 与 committed host 漂移")
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
            raise RuntimeError("W-02 v2 transaction 未闭合四个显式事件")
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


def load_w02_candidate_dump_v2(
        config: W02RuntimeConfig,
        *,
        target_sqlite_path: str | Path,
        ) -> W02RunOutcome:
    """把 v2 权威 package 加载到全新 SQLite，并恢复 evidence-bound consumer。"""
    if not isinstance(config, W02RuntimeConfig):
        raise TypeError("config 必须是 W02RuntimeConfig")
    context = _open_context(config)
    target = Path(target_sqlite_path).resolve()
    source = Path(config.sqlite_path).resolve()
    if target == source:
        raise RuntimeError("W-02 v2 dump readback target 必须独立于 candidate host")
    if target.exists() and target.stat().st_size:
        raise RuntimeError("W-02 v2 dump readback target 必须是 fresh SQLite")
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
        # 只注册完整 schema；恢复包加载前移除装配产生的基础对象。
        open_w02_learning_runtime_v2(backend, mode="restart")
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
            raise RuntimeError("W-02 v2 fresh readback cursor identity 漂移")
        events = transaction.events()
        if len(events) != 3:
            raise RuntimeError("W-02 v2 dump 必须绑定 commit 前三事件")

        audit = W02PayloadAudit()
        payload = W02PayloadFirewall.open(
            config.repository_root,
            context,
            request,
            dependency_root=config.dependency_root,
            audit=audit,
        ).read_training_payload()
        learning = open_w02_learning_runtime_v2(backend, mode="resume")
        learning_report = learning.consume(payload, commit=False)
        _run_consumer_probes_v2(learning, payload)
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
    "W02_FORMAL_RUNTIME_VERSION",
    "W02FaultPoint",
    "W02InjectedFault",
    "W02RunOutcome",
    "W02RuntimeConfig",
    "load_w02_candidate_dump_v2",
    "run_language_stage1_v2",
    "train_generation_probe_target_v2",
]
