"""PH2 W-05 独立 runtime、事务、恢复、九载体 scope 与 dump/readback。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05_IDENTITY_VERSIONS,
    adapt_w05_training_payload,
)
from pure_integer_ai.experiments.ph2_w05_carrier_scope import (
    build_w05_carrier_scope_report,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_OPEN_GENERATION_STATE,
    W05_OWNER_KEY,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_W04_BASE_RUN_ID,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
    validate_w05_request,
)
from pure_integer_ai.experiments.ph2_w05_faults import hit_w05_fault
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.experiments.ph2_w05_generation import (
    build_w05_generation_runtime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_HARD_CASES,
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_READY,
    W05GenerationCaseResult,
    run_w05_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_reasoning import (
    build_w05_reasoning_runtime,
    reasoning_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_transaction import (
    W05_TRANSACTION_EVENT_TABLE,
    W05TransactionStore,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    W05_UNDERSTANDING_UNIQUE,
    build_w05_understanding_runtime,
    understanding_request_for_candidate,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE


_CONTEXT_CACHE: dict[tuple[object, ...], object] = {}
_NAMESPACE = 50550


@dataclass(frozen=True)
class W05RuntimeConfig:
    """W-05 candidate host 的冻结依赖、独立 root 与物理调度。"""

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
class W05RunOutcome:
    """一次 W-05 执行或 dump readback 的可比较公开证据。"""

    logical_state_digest: str
    candidate_digest: str
    understanding_digest: str
    reasoning_digest: str
    generation_digest: str
    carrier_scope_digest: str
    transaction_digest: str
    dump_manifest_sha256: str
    active_candidate_count: int
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
    """返回 canonical JSON SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _execution_state(*, started: bool = False) -> dict[str, int]:
    """返回 W-05 状态；W05-04 public runtime 不声明正式运行。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W05_STARTED": int(started),
        "W06_STARTED": 0,
        "formal_w05_training_runs": int(started),
        "teacher_calls": 0,
    }


def _request(config: W05RuntimeConfig, context, backend: SQLiteBackend):
    """构造并验证不含 private/evaluator 字段的 W-05 run request。"""
    base_fence = (
        context.base_fence_key
        if config.base_fence_key is None else config.base_fence_key
    )
    return validate_w05_request(context, W05RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=context.stage_key,
        owner_key=W05_OWNER_KEY,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
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


def _open_context(config: W05RuntimeConfig, backend: SQLiteBackend):
    """按冻结公开身份缓存只读 context，不缓存或读取训练 payload。"""
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
        context = open_w05_frozen_context(
            repository,
            config.global_manifest_path,
            current_remote_commit_sha1=config.current_remote_commit_sha1,
            backend_profile_key=profile_key,
            dependency_root=config.dependency_root,
        )
        _CONTEXT_CACHE[cache_key] = context
    return context


def _attempt_path(config: W05RuntimeConfig) -> tuple[Path, int]:
    """为每次 fresh/restart/resume 建 append-only learning attempt 文件。"""
    root = _run_directory(config) / "learning_attempts"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("learning_attempt_*.sqlite"))
    ordinal = len(existing) + 1
    target = root / f"learning_attempt_{ordinal:06d}.sqlite"
    if target.exists():
        raise RuntimeError("W-05 learning attempt identity 冲突")
    return target, ordinal


def _consumer_artifacts(adapter, learning):
    """运行 exact U/R/G Use/outcome，并返回结构化持久化证据。"""
    supported = adapter.candidates_for_perturbation("NONE")
    if len(supported) != 1:
        raise RuntimeError("W-05 public runtime 缺唯一 NONE Proposition")
    candidate = supported[0]

    understanding = build_w05_understanding_runtime(learning)
    resolution = understanding.resolve(understanding_request_for_candidate(
        candidate,
        request_key=LosslessIntegerKey((_NAMESPACE, 10, 1)),
    ))
    if resolution.status != W05_UNDERSTANDING_UNIQUE:
        raise RuntimeError("W-05 Understanding 未形成唯一 Proposition")
    understanding_use = understanding.adopt(resolution, candidate)
    understanding_outcome = understanding.verify_use(understanding_use)

    reasoning = build_w05_reasoning_runtime(learning)
    reasoning_use = reasoning.authorize(reasoning_request_for_candidate(
        candidate,
        request_key=LosslessIntegerKey((_NAMESPACE, 20, 1)),
    ))
    reasoning_outcome = reasoning.verify_use(reasoning_use)

    branch = language_branch_identity(
        (_NAMESPACE, 30, 1), versions=W05_IDENTITY_VERSIONS)
    uncertainty = concept_identity(
        (_NAMESPACE, 30, 2), versions=W05_IDENTITY_VERSIONS)
    constraints = GenerationExpressionConstraints(
        branch, (), (), 0, 0, 0, 128)
    generation = build_w05_generation_runtime(learning)
    choice = generation.choose(generation_request_for_candidate(
        candidate,
        request_key=LosslessIntegerKey((_NAMESPACE, 30, 3)),
        uncertainty=uncertainty,
        constraints=constraints,
    ))
    if choice.status != W05_GENERATION_READY or not choice.options:
        raise RuntimeError("W-05 Generation 未形成合法 construction options")
    selected = tuple(item.stable_key() for item in choice.options)
    uses = generation.adopt(choice, selected)
    adopted = tuple(
        item for item in uses
        if item.decision.action == W05_GENERATION_ADOPTED
    )
    if len(adopted) != len(choice.options):
        raise RuntimeError("W-05 Generation 未原子采用全部公开合法 option")
    independent = build_w05_understanding_runtime(learning)
    generation_outcomes = tuple(
        generation.verify_use(item, understanding=independent)
        for item in adopted
    )
    case_values = (
        choice.status == W05_GENERATION_READY,
        all(item.occurrence_preserved for item in generation_outcomes),
        all(item.role_preserved for item in generation_outcomes),
        all(item.scope_preserved for item in generation_outcomes),
        all(item.understanding_status == W05_UNDERSTANDING_UNIQUE
            for item in generation_outcomes),
        all(item.verdict == W05_GENERATION_OUTCOME_SUPPORT
            for item in generation_outcomes),
    )
    cases = tuple(
        W05GenerationCaseResult(
            name,
            passed,
            LosslessIntegerKey((_NAMESPACE, 40, ordinal)),
        )
        for ordinal, (name, passed) in enumerate(
            zip(W05_GENERATION_HARD_CASES, case_values, strict=True), start=1)
    )
    hard = run_w05_generation_hard_conjunct(
        cases, protocol=generation.protocol)
    if hard.status != "PASS":
        raise RuntimeError("W-05 public Generation hard conjunct 未通过")

    candidate_structure = {
        "context_key": list(candidate.proposition_definition.context.stable_key()),
        "occurrence_keys": [
            list(item.stable_key()) for item in candidate.occurrence_order],
        "proposition_key": list(candidate.candidate.stable_key()),
        "role_binding_keys": [
            list(item.stable_key()) for item in sorted(
                candidate.role_binding_identities(), key=lambda value: value.stable_key())],
        "source_key": list(candidate.source_ref.stable_key()),
    }
    generation_use_key = digest_value(
        [list(item.stable_key()) for item in adopted])
    generation_outcome_key = digest_value(
        [list(item.stable_key()) for item in generation_outcomes])
    direction_artifacts = {
        "UNDERSTANDING": (
            understanding_use.stable_key(),
            understanding_outcome.stable_key(),
            understanding_outcome.verdict,
        ),
        "REASONING": (
            reasoning_use.stable_key(),
            reasoning_outcome.stable_key(),
            reasoning_outcome.verdict,
        ),
        "GENERATION": (
            generation_use_key,
            generation_outcome_key,
            hard.status,
        ),
    }
    payload = {
        "candidate_structure": candidate_structure,
        "generation": {
            "choice_key": list(choice.stable_key()),
            "decision_keys": [
                list(item.decision.stable_key()) for item in uses],
            "hard_cases": [
                {
                    "case_name": item.case_name,
                    "evidence_key": list(item.evidence_key.components),
                    "passed": int(item.passed),
                }
                for item in hard.cases
            ],
            "hard_status": hard.status,
            "option_keys": [list(item.stable_key()) for item in choice.options],
            "outcome_keys": [
                list(item.stable_key()) for item in generation_outcomes],
            "status": choice.status,
            "use_keys": [list(item.stable_key()) for item in uses],
        },
        "reasoning": {
            "evidence_keys": [list(item) for item in reasoning_use.evidence_keys],
            "outcome_key": list(reasoning_outcome.stable_key()),
            "status": reasoning_use.status,
            "use_key": list(reasoning_use.stable_key()),
            "verdict": reasoning_outcome.verdict,
        },
        "understanding": {
            "evidence_keys": [list(item) for item in resolution.evidence_keys],
            "outcome_key": list(understanding_outcome.stable_key()),
            "resolution_key": list(resolution.stable_key()),
            "status": resolution.status,
            "use_key": list(understanding_use.stable_key()),
            "verdict": understanding_outcome.verdict,
        },
    }
    return candidate, payload, direction_artifacts, {
        "understanding_use_count": len(understanding.uses),
        "understanding_outcome_count": len(understanding.outcomes),
        "reasoning_use_count": len(reasoning.uses),
        "reasoning_outcome_count": len(reasoning.outcomes),
        "generation_choice_count": len(generation.choices),
        "generation_decision_count": len(generation.decisions),
        "generation_use_count": len(generation.uses),
        "generation_outcome_count": len(generation.outcomes),
    }


def _run_payload(
        config: W05RuntimeConfig,
        coordinator: SQLiteBackend,
        learning_backend: SQLiteBackend,
        ):
    """打开 firewall 一次并运行 adapter/H-05/U-R-G/九载体 projection。"""
    context = _open_context(config, coordinator)
    request = _request(config, context, coordinator)
    firewall = W05PayloadFirewall.open(
        Path(config.repository_root).resolve(),
        context,
        request,
        dependency_root=config.dependency_root,
    )
    payload = firewall.read_training_payload()
    adapter = adapt_w05_training_payload(payload)
    learning = build_w05_learning_runtime(learning_backend, adapter)
    candidate, consumers, direction_artifacts, consumer_counts = (
        _consumer_artifacts(adapter, learning)
    )
    carrier_scope = build_w05_carrier_scope_report(
        config.repository_root,
        learning_backend,
        learning,
        candidate,
        direction_artifacts,
        dependency_root=config.dependency_root,
    )
    return (
        context, request, firewall, adapter, learning, consumers,
        consumer_counts, carrier_scope,
    )


def _run_directory(config: W05RuntimeConfig) -> Path:
    """返回 W-05 独立 run 目录。"""
    return Path(config.run_root).resolve() / f"w05_run_{config.run_id:020d}"


def _manifest_path(config: W05RuntimeConfig) -> Path:
    """返回 manifest-last 的 W-05 dump 路径。"""
    return _run_directory(config) / "w05_dump_manifest.json"


def _write_dump(config: W05RuntimeConfig, payload: dict[str, Any]) -> str:
    """append-or-equal 写 dump manifest 并返回 SHA-256。"""
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        actual = target.read_bytes()
        if actual != encoded:
            raise RuntimeError("W-05 dump manifest identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _owned_tables(
        coordinator: SQLiteBackend,
        learning_backend: SQLiteBackend,
        ) -> tuple[str, ...]:
    """返回 W-05 coordinator 与共享 graph 的实际写入表。"""
    values = set()
    coordinator_tables = set(coordinator.schema_snapshot())
    learning_tables = set(learning_backend.schema_snapshot())
    if W05_TRANSACTION_EVENT_TABLE in coordinator_tables:
        values.add(W05_TRANSACTION_EVENT_TABLE)
    if GRAPH_OBJECT_TABLE in learning_tables:
        values.add(GRAPH_OBJECT_TABLE)
    values.update(
        item for item in learning_tables if item.startswith("ph2_w05_"))
    return tuple(sorted(values))


def _shards(adapter, logical_shard_count: int) -> list[dict[str, Any]]:
    """按 Proposition identity 将全部候选确定性分入 16 logical shards。"""
    buckets: list[list[list[int]]] = [[] for _ in range(logical_shard_count)]
    for candidate in adapter.candidates:
        key = list(candidate.candidate.stable_key())
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


def _resource_report(config, context, firewall, report, carrier_scope):
    """按现场 transport、结构对象和 worker 生成有界整数资源报告。"""
    value = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": (
            2_000
            + report.candidate_count * 100
            + report.account_count * 50
            + len(carrier_scope.records) * 200
        ),
        "actual_payload_bytes": firewall.audit.payload_bytes,
        "actual_payload_gets": firewall.audit.payload_gets,
        "actual_recompute_objects": (
            report.candidate_count
            + report.occurrence_count
            + report.role_binding_count
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
        if value[actual] > context.resource_budget[maximum]:
            raise RuntimeError(f"W-05 resource budget exceeded: {actual}")
    return value


def run_language_stage5(config: W05RuntimeConfig) -> W05RunOutcome:
    """执行 W05-04 runtime，按五事件提交并产出 canonical dump。"""
    if not isinstance(config, W05RuntimeConfig):
        raise TypeError("config 必须是 W05RuntimeConfig")
    if config.run_id != W05_FORMAL_RUN_ID:
        raise RuntimeError("W-05 run_id 必须固定为 6")
    if (config.parent_run_id != W05_W04_BASE_RUN_ID
            or config.base_run_id != W05_W04_BASE_RUN_ID):
        raise RuntimeError("W-05 parent/base 必须接 W-04 run 5")
    if config.worker_count not in {1, 2, 4}:
        raise RuntimeError("W-05 worker_count 必须是 1/2/4")
    if config.mode not in {"fresh", "restart", "resume"}:
        raise RuntimeError("W-05 mode 必须是 fresh/restart/resume")
    _run_directory(config).mkdir(parents=True, exist_ok=True)
    coordinator = SQLiteBackend(str(Path(config.sqlite_path).resolve()))
    learning_path, attempt_ordinal = _attempt_path(config)
    learning_backend = SQLiteBackend(str(learning_path))
    try:
        (
            context, request, firewall, adapter, learning, consumers,
            consumer_counts, carrier_scope,
        ) = _run_payload(config, coordinator, learning_backend)
        hit_w05_fault("BEFORE_FIRST_SHARD", config.fault_point)
        shards = _shards(adapter, context.logical_shard_count)
        hit_w05_fault("AFTER_PARTIAL_SHARD", config.fault_point)
        tx = W05TransactionStore(
            coordinator,
            run_id=config.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        tx.begin({
            "base_fence_key": list(request.base_fence_key),
            "request": list(request.execution_identity_key()),
        })
        hit_w05_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        report = learning.report()
        candidate_keys = [
            list(item.candidate.stable_key()) for item in adapter.candidates]
        preview = {
            "candidate_digest": _digest(candidate_keys),
            "logical_shard_count": context.logical_shard_count,
            "merge_barrier_key": context.merge_barrier_key,
            "shards": shards,
        }
        tx.preview(preview)
        hit_w05_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        retention = {
            "d03_global_sha256": context.d03_global_manifest_identity.sha256,
            "lc16_directional_sha256": context.lc16_directional_identity.sha256,
            "lc16_mapper_sha256": context.lc16_mapper_identity.sha256,
            "lc16_overlay_sha256": context.lc16_overlay_identity.sha256,
            "lc16_projection_sha256": context.lc16_projection_identity.sha256,
            "w04_receipt_sha256": context.w04_receipt_identity.sha256,
        }
        commit_payload = {
            "carrier_scope": carrier_scope.to_dict(),
            "consumers": consumers,
            "learning": report.__dict__,
            "retention": retention,
        }
        commit = tx.commit(commit_payload)
        hit_w05_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        tx.cursor({
            "commit_sha256": commit.payload_sha256,
            "completed_shards": list(range(context.logical_shard_count)),
            "cursor_version": context.cursor_version,
        })
        resource_report = _resource_report(
            config, context, firewall, report, carrier_scope)
        artifact_counts = tuple(sorted({
            "CANDIDATE": report.candidate_count,
            "CARRIER_PROJECTION": len(carrier_scope.records),
            "EVIDENCE_ACCOUNT": report.account_count,
            "EVIDENCE_APPLICATION": report.evidence_application_count,
            "GENERATION_CHOICE": consumer_counts["generation_choice_count"],
            "GENERATION_DECISION": consumer_counts["generation_decision_count"],
            "GENERATION_OUTCOME": consumer_counts["generation_outcome_count"],
            "GENERATION_USE": consumer_counts["generation_use_count"],
            "LOGICAL_SHARD": context.logical_shard_count,
            "OCCURRENCE": report.occurrence_count,
            "REASONING_OUTCOME": consumer_counts["reasoning_outcome_count"],
            "REASONING_USE": consumer_counts["reasoning_use_count"],
            "ROLE_BINDING": report.role_binding_count,
            "ROLE_PROPOSITION_SCOPE_CELL": 27,
            "UNDERSTANDING_OUTCOME": consumer_counts[
                "understanding_outcome_count"],
            "UNDERSTANDING_USE": consumer_counts["understanding_use_count"],
        }.items()))
        logical = {
            "active": [
                list(item.candidate.stable_key())
                for item in learning.active_candidates()
            ],
            "carrier_scope": carrier_scope.to_dict(),
            "consumers": consumers,
            "learning": report.__dict__,
        }
        transaction_events = tx.events()
        if len(transaction_events) not in {4, 5}:
            raise RuntimeError("W-05 manifest 前事务必须停在 cursor 或精确恢复 published")
        transaction_payload = [event.payload for event in transaction_events[:4]]
        owned_tables = _owned_tables(coordinator, learning_backend)
        dump_payload = {
            "active_candidate_keys": logical["active"],
            "artifact_counts": [list(item) for item in artifact_counts],
            "artifact_kind": "PH2_W05_RUNTIME_DUMP",
            "base_fence_key": list(request.base_fence_key),
            "candidate_keys": candidate_keys,
            "commit": commit_payload,
            "execution_state": _execution_state(started=False),
            "format_version": 1,
            "open_generation_state": W05_OPEN_GENERATION_STATE,
            "owned_tables": list(owned_tables),
            "parent_run_id": config.parent_run_id,
            "resource_budget": dict(W05_RESOURCE_BUDGET),
            "resource_report": resource_report,
            "run_id": config.run_id,
            "transaction": transaction_payload,
            "transaction_event_count": 5,
        }
        dump_sha = _write_dump(config, dump_payload)
        tx.published({"dump_manifest_sha256": dump_sha})
        hit_w05_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
        return W05RunOutcome(
            logical_state_digest=_digest(logical),
            candidate_digest=_digest(candidate_keys),
            understanding_digest=_digest(consumers["understanding"]),
            reasoning_digest=_digest(consumers["reasoning"]),
            generation_digest=_digest(consumers["generation"]),
            carrier_scope_digest=carrier_scope.digest,
            transaction_digest=_digest([event.payload for event in tx.events()]),
            dump_manifest_sha256=dump_sha,
            active_candidate_count=report.active_candidate_count,
            artifact_counts=artifact_counts,
            execution_state=_execution_state(started=False),
            open_generation_state=W05_OPEN_GENERATION_STATE,
            resource_report=resource_report,
            resource_budget=dict(W05_RESOURCE_BUDGET),
            transaction_event_count=len(tx.events()),
            new_learning_write_count=(
                report.candidate_count + report.account_count
                + len(carrier_scope.records)
            ),
            payload_gets_this_call=firewall.audit.payload_gets,
            payload_bytes_this_call=firewall.audit.payload_bytes,
            teacher_calls=0,
            sqlite_path=str(Path(config.sqlite_path).resolve()),
            owned_tables=owned_tables,
            learning_attempt_count=attempt_ordinal,
        )
    finally:
        learning_backend.close()
        coordinator.close()


def load_w05_candidate_dump(config: W05RuntimeConfig) -> W05RunOutcome:
    """只读 canonical dump，零 payload transport、零 learning write 地回读。"""
    path = _manifest_path(config)
    if not path.is_file():
        raise RuntimeError("W-05 dump manifest 缺失")
    payload = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(payload, dict)
    if (payload.get("artifact_kind") != "PH2_W05_RUNTIME_DUMP"
            or payload.get("format_version") != 1
            or payload.get("run_id") != config.run_id
            or payload.get("parent_run_id") != config.parent_run_id
            or payload.get("open_generation_state") != W05_OPEN_GENERATION_STATE):
        raise RuntimeError("W-05 dump manifest identity 漂移")
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        raise RuntimeError("W-05 dump commit 缺失")
    consumers = commit.get("consumers")
    learning = commit.get("learning")
    carrier_scope = commit.get("carrier_scope")
    candidate_keys = payload.get("candidate_keys")
    active = payload.get("active_candidate_keys")
    if (not isinstance(consumers, dict) or not isinstance(learning, dict)
            or not isinstance(carrier_scope, dict)
            or not isinstance(candidate_keys, list)
            or not isinstance(active, list)):
        raise RuntimeError("W-05 dump 逻辑字段缺失")
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
        "carrier_scope": carrier_scope,
        "consumers": consumers,
        "learning": learning,
    }
    dump_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    transaction = [*payload["transaction"], {"dump_manifest_sha256": dump_sha}]
    attempts = _run_directory(config) / "learning_attempts"
    attempt_count = len(tuple(attempts.glob("learning_attempt_*.sqlite")))
    return W05RunOutcome(
        logical_state_digest=_digest(logical),
        candidate_digest=_digest(candidate_keys),
        understanding_digest=_digest(consumers["understanding"]),
        reasoning_digest=_digest(consumers["reasoning"]),
        generation_digest=_digest(consumers["generation"]),
        carrier_scope_digest=_digest(carrier_scope),
        transaction_digest=_digest(transaction),
        dump_manifest_sha256=dump_sha,
        active_candidate_count=int(learning["active_candidate_count"]),
        artifact_counts=artifact_counts,
        execution_state={
            str(key): int(value)
            for key, value in payload["execution_state"].items()
        },
        open_generation_state=W05_OPEN_GENERATION_STATE,
        resource_report=resource_report,
        resource_budget=resource_budget,
        transaction_event_count=int(payload["transaction_event_count"]),
        new_learning_write_count=0,
        payload_gets_this_call=0,
        payload_bytes_this_call=0,
        teacher_calls=0,
        sqlite_path=str(Path(config.sqlite_path).resolve()),
        owned_tables=tuple(str(item) for item in payload["owned_tables"]),
        learning_attempt_count=attempt_count,
        dump_readback=True,
    )


__all__ = [
    "W05RunOutcome",
    "W05RuntimeConfig",
    "load_w05_candidate_dump",
    "run_language_stage5",
]
