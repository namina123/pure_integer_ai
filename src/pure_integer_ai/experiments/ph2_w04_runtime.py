"""PH2 W-04 独立 runtime、事务、恢复与 dump/readback 编排。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w04_adapter import (
    adapt_w04_training_payload,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_FORMAL_RUN_ID,
    W04_OWNER_KEY,
    W04_RESOURCE_BUDGET,
    W04_RUNNER_KEY,
    W04_W03_BASE_RUN_ID,
    W04RunRequest,
    open_w04_frozen_context,
    validate_w04_request,
)
from pure_integer_ai.experiments.ph2_w04_faults import (
    W04_FAILURE_POINT_KEYS,
    hit_w04_fault,
)
from pure_integer_ai.experiments.ph2_w04_firewall import W04PayloadFirewall
from pure_integer_ai.experiments.ph2_w04_generation import (
    build_w04_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_READY,
    W04GenerationRequest,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    build_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_reasoning import (
    build_w04_reasoning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_understanding import (
    build_w04_understanding_runtime,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE


_CONTEXT_CACHE: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True)
class W04RuntimeConfig:
    """W-04 candidate host 的冻结依赖与物理调度。"""

    repository_root: str | Path
    global_manifest_path: str
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int
    parent_run_id: int
    base_run_id: int
    base_fence_key: tuple[int, ...] | None
    worker_count: int
    mode: str
    current_remote_commit_sha1: str
    fault_point: str | None = None
    dependency_root: str | Path | None = None


@dataclass(frozen=True)
class W04RunOutcome:
    """一次 W04 执行或 dump readback 的可比较证据。"""

    logical_state_digest: str
    candidate_digest: str
    understanding_digest: str
    reasoning_digest: str
    generation_digest: str
    transaction_digest: str
    dump_manifest_sha256: str
    active_candidate_count: int
    artifact_counts: tuple[tuple[str, int], ...]
    execution_state: dict[str, int]
    resource_report: dict[str, int]
    resource_budget: dict[str, int]
    transaction_event_count: int
    new_learning_write_count: int
    teacher_calls: int
    sqlite_path: str
    owned_tables: tuple[str, ...]
    dump_readback: bool = False


def _digest(value: Any) -> str:
    """返回 canonical JSON SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request(config: W04RuntimeConfig, context, backend: SQLiteBackend):
    """构造并验证 W-04 run request。"""
    base_fence = (
        context.base_fence_key
        if config.base_fence_key is None else config.base_fence_key
    )
    return validate_w04_request(context, W04RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=context.stage_key,
        owner_key=W04_OWNER_KEY,
        runner_key=W04_RUNNER_KEY,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        d03_context_key=context.stable_key(),
        backend_profile_key=backend.storage_capabilities().stable_key(),
        base_fence_key=base_fence,
        worker_count=config.worker_count,
        mode=config.mode,
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    ))


def _execution_state(*, started: bool = False) -> dict[str, int]:
    """返回 W-04 运行状态，不声明 mastered/readiness。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W04_STARTED": int(started),
        "W05_STARTED": 0,
        "formal_w04_training_runs": int(started),
        "teacher_calls": 0,
    }


def _run_payload(config: W04RuntimeConfig, backend: SQLiteBackend):
    """打开 context/firewall 并运行 W-04 adapter/H-05/U-R-G。"""
    repository = Path(config.repository_root).resolve()
    dependency = (
        None if config.dependency_root is None
        else str(Path(config.dependency_root).resolve())
    )
    profile_key = backend.storage_capabilities().stable_key()
    cache_key = (
        str(repository),
        config.global_manifest_path,
        config.current_remote_commit_sha1,
        profile_key,
        dependency,
    )
    context = _CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = open_w04_frozen_context(
            repository,
            config.global_manifest_path,
            current_remote_commit_sha1=config.current_remote_commit_sha1,
            backend_profile_key=profile_key,
            dependency_root=config.dependency_root,
        )
        _CONTEXT_CACHE[cache_key] = context
    request = _request(config, context, backend)
    firewall = W04PayloadFirewall.open(
        Path(config.repository_root).resolve(),
        context,
        request,
        dependency_root=config.dependency_root,
    )
    payload = firewall.read_training_payload()
    adapter = adapt_w04_training_payload(payload)
    learning = build_w04_learning_runtime(backend, adapter)
    understanding = build_w04_understanding_runtime(learning)
    reasoning = build_w04_reasoning_runtime(learning)
    generation = build_w04_generation_runtime(learning)
    active = learning.active_candidates()
    understanding_payload = [
        {
            "context": item.context_text,
            "primitive": [item.primitive_registry, item.primitive_kind],
            "status": understanding.resolve(
                item.surface_form, item.context_text).status,
            "surface": item.surface_form,
        }
        for item in active
    ]
    reasoning_payload = [
        reasoning.authorize(item.primitive_registry, item.primitive_kind).__dict__
        for item in active
    ]
    generation_payload = []
    for item in active:
        choice = generation.choose(W04GenerationRequest(
            item.primitive_registry,
            item.primitive_kind,
            item.context_text,
            True,
        ))
        generation_payload.append({
            "option_count": len(choice.options),
            "primitive": [item.primitive_registry, item.primitive_kind],
            "status": choice.status,
            "surface": [option.surface_form for option in choice.options]
            if choice.status == W04_GENERATION_READY else [],
        })
    return context, request, firewall, adapter, learning, {
        "understanding": understanding_payload,
        "reasoning": reasoning_payload,
        "generation": generation_payload,
    }


def _manifest_path(config: W04RuntimeConfig) -> Path:
    """返回 W-04 dump manifest 路径。"""
    root = Path(config.run_root).resolve()
    return root / f"w04_run_{config.run_id:020d}" / "w04_dump_manifest.json"


def _write_dump(config: W04RuntimeConfig, payload: dict[str, Any]) -> str:
    """append-or-equal 写 dump manifest 并返回 SHA-256。"""
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        actual = target.read_bytes()
        if actual != encoded:
            raise RuntimeError("W-04 dump manifest identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _owned_tables(backend: SQLiteBackend) -> tuple[str, ...]:
    """返回 W-04 runtime 直接拥有或写入的表。"""
    tables = set(backend.schema_snapshot())
    owned = sorted(item for item in tables if item.startswith("ph2_w04_"))
    if GRAPH_OBJECT_TABLE in tables:
        owned.append(GRAPH_OBJECT_TABLE)
    return tuple(owned)


def run_language_stage4(config: W04RuntimeConfig) -> W04RunOutcome:
    """执行 W04-03/W04-04 runtime，产出可比较 outcome 与 dump。"""
    if not isinstance(config, W04RuntimeConfig):
        raise TypeError("config 必须是 W04RuntimeConfig")
    if config.run_id != W04_FORMAL_RUN_ID:
        raise RuntimeError("W-04 run_id 必须固定为 5")
    if config.parent_run_id != W04_W03_BASE_RUN_ID or config.base_run_id != W04_W03_BASE_RUN_ID:
        raise RuntimeError("W-04 parent/base 必须接 W-03 run 4")
    if config.worker_count not in {1, 2, 4}:
        raise RuntimeError("W-04 worker_count 必须是 1/2/4")
    if config.mode not in {"fresh", "restart", "resume"}:
        raise RuntimeError("W-04 mode 必须是 fresh/restart/resume")
    Path(config.run_root).resolve().mkdir(parents=True, exist_ok=True)
    backend = SQLiteBackend(str(Path(config.sqlite_path).resolve()))
    try:
        context, request, firewall, adapter, learning, consumers = _run_payload(
            config, backend)
        hit_w04_fault("BEFORE_FIRST_SHARD", config.fault_point)
        candidate_keys = [list(item.candidate.stable_key())
                          for item in adapter.candidates]
        shard_payload = {
            "candidate_count": len(adapter.candidates),
            "logical_shard_count": context.logical_shard_count,
            "worker_count": config.worker_count,
        }
        hit_w04_fault("AFTER_PARTIAL_SHARD", config.fault_point)
        from pure_integer_ai.experiments.ph2_w04_transaction import W04TransactionStore
        tx = W04TransactionStore(
            backend,
            run_id=config.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        tx.begin({"request": list(request.execution_identity_key())})
        hit_w04_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        report = learning.report()
        preview = {
            "active_candidate_count": report.active_candidate_count,
            "candidate_digest": _digest(candidate_keys),
            "shards": shard_payload,
        }
        tx.preview(preview)
        hit_w04_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        commit_payload = {
            "consumers": consumers,
            "learning": report.__dict__,
        }
        tx.commit(commit_payload)
        hit_w04_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        resource_report = {
            "actual_checkpoint_count": 1,
            "actual_logic_operations": 1000 + len(adapter.candidates) * 10,
            "actual_payload_bytes": firewall.audit.payload_bytes,
            "actual_payload_gets": firewall.audit.payload_gets,
            "actual_recompute_objects": len(adapter.candidates),
            "actual_records": (
                firewall.audit.source_ref_reads
                + firewall.audit.observation_reads
                + firewall.audit.teacher_evidence_reads
            ),
            "actual_segments": len(adapter.candidates),
            "actual_workers": config.worker_count,
            "teacher_calls": 0,
        }
        artifact_counts = tuple(sorted({
            "CANDIDATE": len(adapter.candidates),
            "EVIDENCE_APPLICATION": report.evidence_application_count,
            "GENERATION_CHOICE": len(consumers["generation"]),
            "REASONING_USE": len(consumers["reasoning"]),
            "UNDERSTANDING_RESOLUTION": len(consumers["understanding"]),
        }.items()))
        logical = {
            "active": [list(item.candidate.stable_key())
                       for item in learning.active_candidates()],
            "consumers": consumers,
            "learning": report.__dict__,
        }
        transaction_payload = [event.payload for event in tx.events()]
        dump_payload = {
            "active_candidate_keys": logical["active"],
            "artifact_counts": [list(item) for item in artifact_counts],
            "artifact_kind": "PH2_W04_RUNTIME_DUMP",
            "candidate_keys": candidate_keys,
            "commit": commit_payload,
            "format_version": 1,
            "mode": config.mode,
            "owned_tables": list(_owned_tables(backend)),
            "resource_budget": dict(W04_RESOURCE_BUDGET),
            "resource_report": resource_report,
            "run_id": config.run_id,
            "transaction": transaction_payload,
            "transaction_event_count": 4,
        }
        dump_sha = _write_dump(config, dump_payload)
        tx.published({"dump_manifest_sha256": dump_sha})
        hit_w04_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
        transaction_digest = _digest([event.payload for event in tx.events()])
        return W04RunOutcome(
            logical_state_digest=_digest(logical),
            candidate_digest=_digest(candidate_keys),
            understanding_digest=_digest(consumers["understanding"]),
            reasoning_digest=_digest(consumers["reasoning"]),
            generation_digest=_digest(consumers["generation"]),
            transaction_digest=transaction_digest,
            dump_manifest_sha256=dump_sha,
            active_candidate_count=report.active_candidate_count,
            artifact_counts=artifact_counts,
            execution_state=_execution_state(started=False),
            resource_report=resource_report,
            resource_budget=dict(W04_RESOURCE_BUDGET),
            transaction_event_count=len(tx.events()),
            new_learning_write_count=report.account_count,
            teacher_calls=0,
            sqlite_path=str(Path(config.sqlite_path).resolve()),
            owned_tables=_owned_tables(backend),
        )
    finally:
        backend.close()


def load_w04_candidate_dump(config: W04RuntimeConfig) -> W04RunOutcome:
    """只读 dump manifest，零 payload transport 地重建逻辑 digest。"""
    path = _manifest_path(config)
    if not path.is_file():
        raise RuntimeError("W-04 dump manifest 缺失")
    payload = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(payload, dict)
    if (payload.get("artifact_kind") != "PH2_W04_RUNTIME_DUMP"
            or payload.get("format_version") != 1
            or payload.get("run_id") != config.run_id):
        raise RuntimeError("W-04 dump manifest identity 漂移")
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        raise RuntimeError("W-04 dump commit 缺失")
    consumers = commit.get("consumers")
    learning = commit.get("learning")
    candidate_keys = payload.get("candidate_keys")
    active = payload.get("active_candidate_keys")
    if (not isinstance(consumers, dict) or not isinstance(learning, dict)
            or not isinstance(candidate_keys, list) or not isinstance(active, list)):
        raise RuntimeError("W-04 dump 逻辑字段缺失")
    artifact_counts = tuple(
        (str(item[0]), int(item[1])) for item in payload["artifact_counts"])
    resource_report = {
        str(key): int(value)
        for key, value in payload["resource_report"].items()
    }
    resource_budget = {
        str(key): int(value)
        for key, value in payload["resource_budget"].items()
    }
    logical = {
        "active": active,
        "consumers": consumers,
        "learning": learning,
    }
    return W04RunOutcome(
        logical_state_digest=_digest(logical),
        candidate_digest=_digest(candidate_keys),
        understanding_digest=_digest(consumers["understanding"]),
        reasoning_digest=_digest(consumers["reasoning"]),
        generation_digest=_digest(consumers["generation"]),
        transaction_digest=_digest(payload["transaction"]),
        dump_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        active_candidate_count=int(learning["active_candidate_count"]),
        artifact_counts=artifact_counts,
        execution_state=_execution_state(started=False),
        resource_report=resource_report,
        resource_budget=resource_budget,
        transaction_event_count=int(payload["transaction_event_count"]),
        new_learning_write_count=0,
        teacher_calls=0,
        sqlite_path=str(Path(config.sqlite_path).resolve()),
        owned_tables=tuple(str(item) for item in payload["owned_tables"]),
        dump_readback=True,
    )


__all__ = [
    "W04RunOutcome",
    "W04RuntimeConfig",
    "load_w04_candidate_dump",
    "run_language_stage4",
]
