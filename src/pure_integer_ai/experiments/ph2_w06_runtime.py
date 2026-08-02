"""W06-04 public transaction、恢复、资源、载体与 retention runtime。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import concept_identity
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w06_adapter import (
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_carrier_scope import (
    build_w06_carrier_scope_report,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_ALLOWED_MODES,
    W06_ALLOWED_WORKER_COUNTS,
    W06_FORMAL_RUN_ID,
    W06_OPEN_GENERATION_STATE,
    W06_OWNER_KEY,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_W05_BASE_RUN_ID,
    W06RunRequest,
    open_w06_frozen_context,
    validate_w06_request,
)
from pure_integer_ai.experiments.ph2_w06_faults import hit_w06_fault
from pure_integer_ai.experiments.ph2_w06_firewall import W06PayloadFirewall
from pure_integer_ai.experiments.ph2_w06_learning import (
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.experiments.ph2_w06_transaction import (
    W06_TRANSACTION_EVENT_TABLE,
    W06TransactionStore,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE


_CONTEXT_CACHE: dict[tuple[object, ...], object] = {}
_NAMESPACE = 50650
_RETENTION_IDENTITIES = {
    "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json": (
        "141a6c2341671d4d92d9974a355b8081fd12dff17315f5d1f60913a45c31c8f1"
    ),
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json": (
        "ef64636ab287eacbacae4040f59da74bb4105374cba31d756e1ddefaf86043f6"
    ),
    "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json": (
        "153db3d7f3c0fca04642f4198df16e3c1adb0f5c78e4d6c7c59d35122989727b"
    ),
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json": (
        "64c2fff496e766df880d2db1b184e2b8a009abd3b37b1a1b1331900458ccff78"
    ),
}


@dataclass(frozen=True)
class W06RuntimeConfig:
    """W06-04 public host 的冻结依赖、Git 外 root 与物理调度。"""

    repository_root: str | Path
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


@dataclass(frozen=True)
class W06RunOutcome:
    """一次 W06-04 执行或 dump readback 的可比较公开证据。"""

    logical_state_digest: str
    candidate_digest: str
    relation_digest: str
    source_evidence_digest: str
    active_projection_digest: str
    carrier_scope_digest: str
    transaction_digest: str
    dump_manifest_sha256: str
    candidate_count: int
    active_candidate_count: int
    relation_summary_digests: tuple[tuple[str, str], ...]
    retention_sha256: tuple[tuple[str, str], ...]
    artifact_counts: tuple[tuple[str, int], ...]
    execution_state: dict[str, int]
    open_generation_state: str
    resource_report: dict[str, int]
    resource_budget: dict[str, int]
    transaction_event_count: int
    new_learning_write_count: int
    payload_gets_this_call: int
    payload_bytes_this_call: int
    teacher_calls: int
    sqlite_path: str
    owned_tables: tuple[str, ...]
    learning_attempt_count: int
    dump_readback: bool = False


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _execution_state() -> dict[str, int]:
    """W06-04 仍是 public bounded gate，不声明正式 candidate run。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_STARTED": 0,
        "W07_STARTED": 0,
        "formal_w06_training_runs": 0,
        "teacher_calls": 0,
    }


def _open_context(config: W06RuntimeConfig, backend: SQLiteBackend):
    repository = Path(config.repository_root).resolve()
    profile_key = backend.storage_capabilities().stable_key()
    cache_key = (
        str(repository),
        config.current_remote_commit_sha1,
        profile_key,
    )
    context = _CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = open_w06_frozen_context(
            repository,
            current_remote_commit_sha1=config.current_remote_commit_sha1,
            backend_profile_key=profile_key,
        )
        _CONTEXT_CACHE[cache_key] = context
    return context


def _request(config: W06RuntimeConfig, context) -> W06RunRequest:
    base_fence = (
        context.base_fence_key
        if config.base_fence_key is None else config.base_fence_key
    )
    return validate_w06_request(context, W06RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=W06_STAGE_KEY,
        owner_key=W06_OWNER_KEY,
        runner_key=W06_RUNNER_KEY,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        source_overlay_sha256=context.source_overlay_sha256,
        context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=base_fence,
        worker_count=config.worker_count,
        mode=config.mode,
        resource_budget=tuple(sorted(W06_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    ))


def _run_directory(config: W06RuntimeConfig) -> Path:
    return Path(config.run_root).resolve() / f"w06_run_{config.run_id:020d}"


def _manifest_path(config: W06RuntimeConfig) -> Path:
    return _run_directory(config) / "w06_dump_manifest.json"


def _attempt_path(config: W06RuntimeConfig) -> tuple[Path, int]:
    root = _run_directory(config) / "learning_attempts"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("learning_attempt_*.sqlite"))
    ordinal = len(existing) + 1
    target = root / f"learning_attempt_{ordinal:06d}.sqlite"
    if target.exists():
        raise RuntimeError("W-06 learning attempt identity 冲突")
    return target, ordinal


def _write_dump(config: W06RuntimeConfig, payload: dict[str, Any]) -> str:
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError("W-06 dump manifest identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _retention(repository: Path) -> tuple[tuple[str, str], ...]:
    result = []
    for relative, expected in sorted(_RETENTION_IDENTITIES.items()):
        target = (repository / relative).resolve()
        if not target.is_relative_to(repository) or not target.is_file():
            raise RuntimeError("W-06 retention receipt 缺失")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"W-06 retention identity 漂移：{relative}")
        result.append((relative, actual))
    return tuple(result)


def _direction_artifacts(learning):
    """以同一 active typed relation 形成三向 exact Core Use。"""
    active = learning.active_candidates()
    if not active or learning.closure is None:
        raise RuntimeError("W-06 public runtime 缺 active relation")
    candidate = active[0]
    artifacts = {}
    for ordinal, direction in enumerate(
            ("GENERATION", "REASONING", "UNDERSTANDING"), start=1):
        context = RelationUseContext(
            candidate.source_ref,
            document_scope(candidate.source_ref),
            concept_identity((_NAMESPACE, 10, ordinal)),
            concept_identity((_NAMESPACE, 20, ordinal)),
        )
        uses = learning.closure.consume_many(((
            candidate.proposition.proposition,
            (_NAMESPACE, 30, ordinal),
            context,
        ),))
        if len(uses) != 1:
            raise RuntimeError("W-06 relation exact Use 数量漂移")
        definition_key = uses[0].to_definition().stable_key()
        current = learning.snapshot_for(candidate.proposition.proposition)
        if current.active_fact is None:
            raise RuntimeError("W-06 relation Use 后 active fact 丢失")
        outcome_key = digest_value({
            "active": list(current.active_fact.decision_key),
            "direction": direction,
            "use": list(definition_key),
        })
        artifacts[direction] = (definition_key, outcome_key, "SUPPORT")
    return candidate, artifacts


def _candidate_value(candidate, snapshot) -> dict[str, Any]:
    return {
        "active": int(snapshot.active_fact is not None),
        "bindings": [
            [list(item.role.stable_key()), list(item.filler.stable_key())]
            for item in candidate.proposition.canonical_bindings()
        ],
        "directionality": candidate.directionality,
        "endpoints": [list(item.identity.stable_key())
                      for item in candidate.endpoints],
        "epistemic_status": snapshot.snapshot.epistemic_status,
        "evidence": [list(item.stable_key()) for item in snapshot.evidence],
        "lifecycle": snapshot.snapshot.lifecycle,
        "proposition": list(candidate.proposition.proposition.stable_key()),
        "relation_family": candidate.relation_family,
        "schema": list(candidate.schema.schema.stable_key()),
        "source": list(candidate.source_ref.stable_key()),
        "substage": candidate.substage_key,
    }


def _relation_summaries(adapter, learning):
    summaries = {}
    all_values = []
    active_values = []
    evidence_values = []
    for substage in W06_RELATION_SUBSTAGE_ORDER:
        values = []
        for candidate in adapter.candidates_for_substage(substage):
            snapshot = learning.snapshot_for(candidate.proposition.proposition)
            value = _candidate_value(candidate, snapshot)
            values.append(value)
            all_values.append(value)
            evidence_values.append({
                "evidence": value["evidence"],
                "proposition": value["proposition"],
                "source": value["source"],
            })
            if snapshot.active_fact is not None:
                active_values.append(value)
        summaries[substage] = {
            "active_count": sum(item["active"] for item in values),
            "candidate_count": len(values),
            "digest": _digest(values),
        }
    return (
        summaries,
        _digest(all_values),
        _digest(evidence_values),
        _digest(active_values),
    )


def _shards(adapter, logical_shard_count: int) -> list[dict[str, Any]]:
    buckets: list[list[list[int]]] = [[] for _ in range(logical_shard_count)]
    for candidate in adapter.candidates:
        key = list(candidate.proposition.proposition.stable_key())
        ordinal = int.from_bytes(
            hashlib.sha256(canonical_json_bytes(key)).digest()[:8], "big"
        ) % logical_shard_count
        buckets[ordinal].append(key)
    return [
        {
            "candidate_keys": sorted(bucket),
            "shard_ordinal": ordinal,
        }
        for ordinal, bucket in enumerate(buckets)
    ]


def _owned_tables(
        coordinator: SQLiteBackend,
        learning_backend: SQLiteBackend,
        ) -> tuple[str, ...]:
    values = set()
    coordinator_tables = set(coordinator.schema_snapshot())
    learning_tables = set(learning_backend.schema_snapshot())
    if W06_TRANSACTION_EVENT_TABLE in coordinator_tables:
        values.add(W06_TRANSACTION_EVENT_TABLE)
    if GRAPH_OBJECT_TABLE in learning_tables:
        values.add(GRAPH_OBJECT_TABLE)
    values.update(
        item for item in learning_tables if item.startswith("ph2_w06_"))
    return tuple(sorted(values))


def _resource_report(config, context, firewall, report, carrier_scope):
    budget = dict(context.resource_budget)
    value = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": (
            4_000
            + report.candidate_count * 100
            + report.evidence_account_count * 50
            + len(carrier_scope.records) * 200
        ),
        "actual_payload_bytes": firewall.audit.payload_bytes,
        "actual_payload_gets": firewall.audit.payload_gets,
        "actual_recompute_objects": (
            report.candidate_count
            + report.evidence_account_count
            + len(carrier_scope.records)
        ),
        "actual_records": (
            firewall.audit.source_ref_reads
            + firewall.audit.observation_reads
            + firewall.audit.teacher_evidence_reads
        ),
        "actual_segments": report.candidate_count + len(carrier_scope.records),
        "actual_workers": config.worker_count,
        "teacher_calls": 0,
    }
    pairs = (
        ("actual_checkpoint_count", "max_checkpoint_count"),
        ("actual_logic_operations", "max_logic_operations"),
        ("actual_payload_bytes", "max_payload_bytes"),
        ("actual_payload_gets", "max_payload_gets"),
        ("actual_recompute_objects", "max_recompute_objects"),
        ("actual_records", "max_records"),
        ("actual_segments", "max_segments"),
        ("actual_workers", "max_workers"),
    )
    for actual, maximum in pairs:
        if value[actual] > budget[maximum]:
            raise RuntimeError(f"W-06 resource budget exceeded: {actual}")
    return value


def _run_payload(
        config: W06RuntimeConfig,
        coordinator: SQLiteBackend,
        learning_backend: SQLiteBackend,
        ):
    context = _open_context(config, coordinator)
    request = _request(config, context)
    firewall = W06PayloadFirewall.open(
        Path(config.repository_root).resolve(), context, request)
    payload = firewall.read_training_payload()
    adapter = adapt_w06_training_payload(payload)
    learning = build_w06_learning_runtime(learning_backend, adapter)
    candidate, directions = _direction_artifacts(learning)
    carrier_scope = build_w06_carrier_scope_report(
        config.repository_root,
        learning_backend,
        learning,
        candidate,
        directions,
    )
    return (
        context, request, firewall, adapter, learning, directions,
        carrier_scope,
    )


def run_language_stage6_public(config: W06RuntimeConfig) -> W06RunOutcome:
    """执行 W06-04 public runtime，按五事件提交并产出 canonical dump。"""
    if not isinstance(config, W06RuntimeConfig):
        raise TypeError("config 必须是 W06RuntimeConfig")
    if config.run_id != W06_FORMAL_RUN_ID:
        raise RuntimeError("W-06 run_id 必须固定为 7")
    if (config.parent_run_id != W06_W05_BASE_RUN_ID
            or config.base_run_id != W06_W05_BASE_RUN_ID):
        raise RuntimeError("W-06 parent/base 必须接 W-05 run 6")
    if config.worker_count not in W06_ALLOWED_WORKER_COUNTS:
        raise RuntimeError("W-06 worker_count 必须是 1/2/4")
    if config.mode not in W06_ALLOWED_MODES:
        raise RuntimeError("W-06 mode 必须是 fresh/restart/resume")
    _run_directory(config).mkdir(parents=True, exist_ok=True)
    coordinator = SQLiteBackend(str(Path(config.sqlite_path).resolve()))
    learning_path, attempt_ordinal = _attempt_path(config)
    learning_backend = SQLiteBackend(str(learning_path))
    try:
        (
            context, request, firewall, adapter, learning, directions,
            carrier_scope,
        ) = _run_payload(config, coordinator, learning_backend)
        hit_w06_fault("BEFORE_FIRST_SHARD", config.fault_point)
        shards = _shards(adapter, context.logical_shard_count)
        hit_w06_fault("AFTER_PARTIAL_SHARD", config.fault_point)
        tx = W06TransactionStore(
            coordinator,
            run_id=config.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        tx.begin({
            "base_fence_key": list(request.base_fence_key),
            "request": list(request.execution_identity_key()),
        })
        hit_w06_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        report = learning.report()
        summaries, candidate_digest, evidence_digest, active_digest = (
            _relation_summaries(adapter, learning))
        preview = {
            "candidate_digest": candidate_digest,
            "logical_shard_count": context.logical_shard_count,
            "merge_barrier_key": context.merge_barrier_key,
            "shards": shards,
        }
        tx.preview(preview)
        hit_w06_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        retention = _retention(Path(config.repository_root).resolve())
        commit_payload = {
            "carrier_scope": carrier_scope.to_dict(),
            "direction_artifacts": {
                key: [list(value[0]), list(value[1]), value[2]]
                for key, value in sorted(directions.items())
            },
            "learning": report.__dict__,
            "relation_summaries": summaries,
            "retention": dict(retention),
        }
        commit = tx.commit(commit_payload)
        hit_w06_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        tx.cursor({
            "commit_sha256": commit.payload_sha256,
            "completed_shards": list(range(context.logical_shard_count)),
            "cursor_version": context.cursor_version,
        })
        resource = _resource_report(
            config, context, firewall, report, carrier_scope)
        artifact_counts = tuple(sorted({
            "ACTIVE_RELATION": report.active_candidate_count,
            "CANDIDATE": report.candidate_count,
            "CARRIER_PROJECTION": len(carrier_scope.records),
            "EVIDENCE_ACCOUNT": report.evidence_account_count,
            "EVIDENCE_APPLICATION": report.evidence_application_count,
            "LOGICAL_SHARD": context.logical_shard_count,
            "RELATION_FAMILY": report.relation_family_count,
            "RELATION_SCOPE_CELL": 27,
            "RELATION_USE": 3,
            "SCHEMA_REJECTION": report.schema_rejection_count,
            "SUBSTAGE": len(W06_RELATION_SUBSTAGE_ORDER),
        }.items()))
        logical = {
            "active_projection_digest": active_digest,
            "candidate_digest": candidate_digest,
            "carrier_scope": carrier_scope.to_dict(),
            "direction_artifacts": commit_payload["direction_artifacts"],
            "learning": report.__dict__,
            "relation_summaries": summaries,
            "source_evidence_digest": evidence_digest,
        }
        transaction_events = tx.events()
        if len(transaction_events) not in {4, 5}:
            raise RuntimeError("W-06 manifest 前事务必须停在 cursor 或已发布")
        transaction_payload = [event.payload for event in transaction_events[:4]]
        owned_tables = _owned_tables(coordinator, learning_backend)
        dump_payload = {
            "active_candidate_count": report.active_candidate_count,
            "artifact_counts": [list(item) for item in artifact_counts],
            "artifact_kind": "PH2_W06_PUBLIC_RUNTIME_DUMP",
            "base_fence_key": list(request.base_fence_key),
            "candidate_count": report.candidate_count,
            "commit": commit_payload,
            "digests": {
                "active_projection": active_digest,
                "candidate": candidate_digest,
                "relation": _digest(summaries),
                "source_evidence": evidence_digest,
            },
            "execution_state": _execution_state(),
            "format_version": 1,
            "open_generation_state": W06_OPEN_GENERATION_STATE,
            "owned_tables": list(owned_tables),
            "parent_run_id": config.parent_run_id,
            "resource_budget": dict(W06_RESOURCE_BUDGET),
            "resource_report": resource,
            "run_id": config.run_id,
            "transaction": transaction_payload,
            "transaction_event_count": 5,
        }
        dump_sha = _write_dump(config, dump_payload)
        tx.published({"dump_manifest_sha256": dump_sha})
        hit_w06_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
        return W06RunOutcome(
            _digest(logical),
            candidate_digest,
            _digest(summaries),
            evidence_digest,
            active_digest,
            carrier_scope.digest,
            _digest([event.payload for event in tx.events()]),
            dump_sha,
            report.candidate_count,
            report.active_candidate_count,
            tuple((key, value["digest"])
                  for key, value in sorted(summaries.items())),
            retention,
            artifact_counts,
            _execution_state(),
            W06_OPEN_GENERATION_STATE,
            resource,
            dict(W06_RESOURCE_BUDGET),
            len(tx.events()),
            report.candidate_count + report.evidence_account_count + 12,
            firewall.audit.payload_gets,
            firewall.audit.payload_bytes,
            0,
            str(Path(config.sqlite_path).resolve()),
            owned_tables,
            attempt_ordinal,
        )
    finally:
        learning_backend.close()
        coordinator.close()


def load_w06_public_dump(config: W06RuntimeConfig) -> W06RunOutcome:
    """只读 canonical dump，零 payload transport、零 learning write 地回读。"""
    path = _manifest_path(config)
    if not path.is_file():
        raise RuntimeError("W-06 dump manifest 缺失")
    payload = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(payload, dict)
    if (payload.get("artifact_kind") != "PH2_W06_PUBLIC_RUNTIME_DUMP"
            or payload.get("format_version") != 1
            or payload.get("run_id") != config.run_id
            or payload.get("parent_run_id") != config.parent_run_id
            or payload.get("open_generation_state")
            != W06_OPEN_GENERATION_STATE):
        raise RuntimeError("W-06 dump manifest identity 漂移")
    commit = payload.get("commit")
    digests = payload.get("digests")
    if not isinstance(commit, dict) or not isinstance(digests, dict):
        raise RuntimeError("W-06 dump commit/digests 缺失")
    summaries = commit["relation_summaries"]
    logical = {
        "active_projection_digest": digests["active_projection"],
        "candidate_digest": digests["candidate"],
        "carrier_scope": commit["carrier_scope"],
        "direction_artifacts": commit["direction_artifacts"],
        "learning": commit["learning"],
        "relation_summaries": summaries,
        "source_evidence_digest": digests["source_evidence"],
    }
    dump_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    transaction = [
        *payload["transaction"], {"dump_manifest_sha256": dump_sha}]
    attempts = _run_directory(config) / "learning_attempts"
    return W06RunOutcome(
        _digest(logical),
        digests["candidate"],
        digests["relation"],
        digests["source_evidence"],
        digests["active_projection"],
        _digest(commit["carrier_scope"]),
        _digest(transaction),
        dump_sha,
        int(payload["candidate_count"]),
        int(payload["active_candidate_count"]),
        tuple((key, value["digest"])
              for key, value in sorted(summaries.items())),
        tuple(sorted((str(key), str(value))
                     for key, value in commit["retention"].items())),
        tuple((str(item[0]), int(item[1]))
              for item in payload["artifact_counts"]),
        {str(key): int(value)
         for key, value in payload["execution_state"].items()},
        W06_OPEN_GENERATION_STATE,
        {str(key): int(value)
         for key, value in payload["resource_report"].items()},
        {str(key): int(value)
         for key, value in payload["resource_budget"].items()},
        int(payload["transaction_event_count"]),
        0,
        0,
        0,
        0,
        str(Path(config.sqlite_path).resolve()),
        tuple(str(item) for item in payload["owned_tables"]),
        len(tuple(attempts.glob("learning_attempt_*.sqlite"))),
        True,
    )


__all__ = [
    "W06RunOutcome",
    "W06RuntimeConfig",
    "load_w06_public_dump",
    "run_language_stage6_public",
]
