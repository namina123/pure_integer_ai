"""W07-04 public transaction、恢复、资源、载体与 retention runtime。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_adapter import (
    W07LogicProposal,
    adapt_w07_training_payload,
)
from pure_integer_ai.experiments.ph2_w07_carrier_scope import (
    W07LogicDirectionArtifact,
    W07LogicProjectionTarget,
    build_w07_carrier_scope_report,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_ALLOWED_MODES,
    W07_ALLOWED_WORKER_COUNTS,
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_OPEN_GENERATION_STATE,
    W07_OWNER_KEY,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
    open_w07_frozen_context,
    validate_w07_request,
)
from pure_integer_ai.experiments.ph2_w05_contract import W05_LC16_DIRECTIONS
from pure_integer_ai.experiments.ph2_w07_faults import hit_w07_fault
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.experiments.ph2_w07_learning import (
    build_w07_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w07_logic_consumer import (
    W07LogicReasoningRuntime,
    W07LogicUnderstandingRuntime,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicConsumerProtocol,
    W07LogicRequest,
)
from pure_integer_ai.experiments.ph2_w07_logic_generation import (
    W07LogicGenerationRuntime,
    generation_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    W07LogicView,
    role_tree_key,
    structure_tree_key,
    w07_logic_language_branch,
)
from pure_integer_ai.experiments.ph2_w07_transaction import (
    W07_TRANSACTION_EVENT_TABLE,
    W07TransactionStore,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE


_CONTEXT_CACHE: dict[tuple[object, ...], object] = {}
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
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json": (
        "aaf35a8346446e80d71f057ae391d9a734a864ced317fa06f2ea01f99efbc0e7"
    ),
}


@dataclass(frozen=True)
class W07RuntimeConfig:
    """W07-04 public host 的冻结依赖、Git 外 root 与物理调度。"""

    repository_root: str | Path
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int
    parent_run_id: int
    base_run_id: int
    base_fence_key: tuple[int, ...] | None
    worker_count: int
    mode: str
    baseline_commit_sha1: str = W07_BASELINE_COMMIT_SHA1
    fault_point: str | None = None


@dataclass(frozen=True)
class W07RunOutcome:
    """一次 W07-04 执行或 dump readback 的可比较公开证据。"""

    logical_state_digest: str
    candidate_digest: str
    logic_digest: str
    source_evidence_digest: str
    active_projection_digest: str
    carrier_scope_digest: str
    transaction_digest: str
    dump_manifest_sha256: str
    candidate_count: int
    active_candidate_count: int
    logic_summary_digests: tuple[tuple[str, str], ...]
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
    """W07-04 仍是 public bounded gate，不声明正式 candidate run。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_RUNTIME_EVIDENCED": 1,
        "W07_STARTED": 0,
        "W08_STARTED": 0,
        "formal_w07_training_runs": 0,
        "teacher_calls": 0,
    }


def _open_context(config: W07RuntimeConfig, backend: SQLiteBackend):
    repository = Path(config.repository_root).resolve()
    profile_key = backend.storage_capabilities().stable_key()
    cache_key = (str(repository), config.baseline_commit_sha1, profile_key)
    context = _CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = open_w07_frozen_context(
            repository,
            baseline_commit_sha1=config.baseline_commit_sha1,
            backend_profile_key=profile_key,
        )
        _CONTEXT_CACHE[cache_key] = context
    return context


def _request(config: W07RuntimeConfig, context) -> W07RunRequest:
    base_fence = (
        context.base_fence_key
        if config.base_fence_key is None else config.base_fence_key)
    return validate_w07_request(context, W07RunRequest(
        run_id=config.run_id,
        parent_run_id=config.parent_run_id,
        base_run_id=config.base_run_id,
        stage_key=W07_STAGE_KEY,
        owner_key=W07_OWNER_KEY,
        runner_key=W07_RUNNER_KEY,
        baseline_commit_sha1=config.baseline_commit_sha1,
        context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=base_fence,
        worker_count=config.worker_count,
        mode=config.mode,
        resource_budget=tuple(sorted(W07_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    ))


def _run_directory(config: W07RuntimeConfig) -> Path:
    return Path(config.run_root).resolve() / f"w07_run_{config.run_id:020d}"


def _manifest_path(config: W07RuntimeConfig) -> Path:
    return _run_directory(config) / "w07_dump_manifest.json"


def _attempt_path(config: W07RuntimeConfig) -> tuple[Path, int]:
    root = _run_directory(config) / "learning_attempts"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("learning_attempt_*.sqlite"))
    ordinal = len(existing) + 1
    target = root / f"learning_attempt_{ordinal:06d}.sqlite"
    if target.exists():
        raise RuntimeError("W-07 learning attempt identity 冲突")
    return target, ordinal


def _write_dump(config: W07RuntimeConfig, payload: dict[str, Any]) -> str:
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError("W-07 dump manifest identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _retention(repository: Path) -> tuple[tuple[str, str], ...]:
    result = []
    for relative, expected in sorted(_RETENTION_IDENTITIES.items()):
        target = (repository / relative).resolve()
        if not target.is_relative_to(repository) or not target.is_file():
            raise RuntimeError("W-07 retention receipt 缺失")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"W-07 retention identity 漂移：{relative}")
        result.append((relative, actual))
    return tuple(result)


def _proposal_for(view: W07LogicView, substage: str) -> W07LogicProposal:
    values = view.executable_proposals(substage)
    preferred = tuple(
        item for item in values
        if item.observation.perturbation_kind == "NONE")
    candidates = preferred or values
    if not candidates:
        raise RuntimeError(f"W-07 {substage} 缺 executable proposal")
    return candidates[0]


def _independent_evidence_source(learning, proposal: W07LogicProposal):
    structures = {item.definition.structure for item in proposal.specs}
    sources = tuple(sorted({
        evidence.source
        for spec in learning.registered_specs()
        if spec.definition.structure in structures
        for adoption in (learning.adoption(spec),)
        if adoption is not None
        for evidence in adoption.evidence
        if evidence.source != proposal.source_binding.source_ref
    }, key=lambda item: item.stable_key()))
    if not sources:
        raise RuntimeError("W-07 logic projection 缺独立 adoption Evidence source")
    return sources[0]


def _logic_targets(adapter, learning) -> tuple[W07LogicProjectionTarget, ...]:
    """逐维执行真实 U/R/G，并冻结每个 exact Use/outcome。"""
    protocol = W07LogicConsumerProtocol(W07_SUBSTAGE_ORDER)
    view = W07LogicView(learning, adapter, protocol)
    understanding = W07LogicUnderstandingRuntime(view)
    reasoning = W07LogicReasoningRuntime(view)
    generation = W07LogicGenerationRuntime(view)
    targets = []
    for ordinal, substage in enumerate(W07_SUBSTAGE_ORDER, start=1):
        proposal = _proposal_for(view, substage)
        logic_request_key = LosslessIntegerKey((70741, ordinal, 1))
        u_request = W07LogicRequest(
            logic_request_key,
            substage,
            proposal.bound_root.template,
            proposal.source_binding.source_ref,
            proposal.request_scope,
        )
        u_use = understanding.adopt(understanding.resolve(u_request))
        u_outcome = understanding.verify(u_use)
        r_request = W07LogicRequest(
            logic_request_key,
            substage,
            proposal.bound_root.template,
            proposal.source_binding.source_ref,
            proposal.request_scope,
        )
        r_use = reasoning.adopt(reasoning.resolve(r_request))
        r_outcome = reasoning.verify(r_use)
        branch = w07_logic_language_branch(proposal)
        constraints = GenerationExpressionConstraints(
            branch,
            tuple(item.definition.structure for item in proposal.specs),
            (branch,),
            0,
            0,
            0,
            256,
        )
        choice = generation.choose(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((70741, ordinal, 3)),
            logic_request_key=logic_request_key,
            constraints=constraints,
        ))
        if choice.status != "READY" or len(choice.options) != 1:
            raise RuntimeError(f"W-07 {substage} generation 未 READY")
        g_use = generation.adopt(choice, choice.options[0].stable_key())
        g_outcome = generation.verify(g_use)
        if {u_outcome.verdict, r_outcome.verdict, g_outcome.verdict} != {
                "SUPPORT"}:
            raise RuntimeError(f"W-07 {substage} U/R/G postcheck 未闭合")
        execution = u_use.resolution.execution
        assert execution is not None
        artifacts = {
            "UNDERSTANDING": W07LogicDirectionArtifact(
                "UNDERSTANDING",
                u_use.use_key.components,
                u_outcome.outcome_key.components,
                u_outcome.verdict,
            ),
            "REASONING": W07LogicDirectionArtifact(
                "REASONING",
                r_use.use_key.components,
                r_outcome.outcome_key.components,
                r_outcome.verdict,
            ),
            "GENERATION": W07LogicDirectionArtifact(
                "GENERATION",
                g_use.ref.use_key.components,
                g_outcome.ref.outcome_key.components,
                g_outcome.verdict,
            ),
        }
        targets.append(W07LogicProjectionTarget(
            substage,
            proposal.bound_root.template,
            structure_tree_key(proposal.bound_root),
            role_tree_key(
                proposal.bound_root, include_bound_provenance=True),
            proposal.source_binding.source_ref,
            execution.evaluation.scope,
            _independent_evidence_source(learning, proposal),
            execution.evaluation.state.stable_key(),
            tuple(artifacts[item] for item in W05_LC16_DIRECTIONS),
        ))
    return tuple(targets)


def _logic_summaries(adapter, learning):
    summaries = {}
    all_values = []
    active_values = []
    source_values = []
    for substage in W07_SUBSTAGE_ORDER:
        values = []
        for proposal in adapter.proposals:
            if proposal.observation.substage != substage:
                continue
            for family, spec in zip(
                    proposal.operator_families, proposal.specs, strict=True):
                adoption = learning.adoption(spec)
                value = {
                    "active": int(adoption is not None),
                    "candidate": list(spec.candidate.stable_key()),
                    "definition": list(spec.definition.stable_key()),
                    "family": family,
                    "source": list(proposal.source_binding.source_ref.stable_key()),
                    "substage": substage,
                }
                values.append(value)
                all_values.append(value)
                if adoption is not None:
                    active_values.append(value)
                    source_values.append({
                        "candidate": value["candidate"],
                        "evidence": [list(item.stable_key())
                                     for item in adoption.evidence],
                    })
        summaries[substage] = {
            "active_count": sum(item["active"] for item in values),
            "candidate_count": len(values),
            "digest": _digest(values),
        }
    return (
        summaries,
        _digest(all_values),
        _digest(source_values),
        _digest(active_values),
    )


def _shards(adapter, logical_shard_count: int) -> list[dict[str, Any]]:
    buckets: list[list[list[int]]] = [[] for _ in range(logical_shard_count)]
    for spec in adapter.specs:
        key = list(spec.candidate.stable_key())
        ordinal = int.from_bytes(
            hashlib.sha256(canonical_json_bytes(key)).digest()[:8], "big"
        ) % logical_shard_count
        buckets[ordinal].append(key)
    return [{
        "candidate_keys": sorted(bucket),
        "shard_ordinal": ordinal,
    } for ordinal, bucket in enumerate(buckets)]


def _owned_tables(
        coordinator: SQLiteBackend,
        learning_backend,
        ) -> tuple[str, ...]:
    values = set()
    coordinator_tables = set(coordinator.schema_snapshot())
    learning_tables = set(learning_backend.schema_snapshot())
    if W07_TRANSACTION_EVENT_TABLE in coordinator_tables:
        values.add(W07_TRANSACTION_EVENT_TABLE)
    if GRAPH_OBJECT_TABLE in learning_tables:
        values.add(GRAPH_OBJECT_TABLE)
    values.update(
        item for item in learning_tables if item.startswith("ph2_w07_"))
    return tuple(sorted(values))


def _resource_report(config, context, firewall, report, carrier_scope):
    budget = dict(context.resource_budget)
    projection_count = len(carrier_scope.records)
    cell_count = sum(len(item.logic_cells) for item in carrier_scope.records)
    value = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": (
            8_000
            + report.candidate_count * 160
            + report.operator_evidence_account_count * 80
            + projection_count * 200
            + cell_count * 40
        ),
        "actual_payload_bytes": firewall.audit.payload_bytes,
        "actual_payload_gets": firewall.audit.payload_gets,
        "actual_recompute_objects": (
            report.candidate_count
            + report.operator_evidence_account_count
            + projection_count
            + cell_count
        ),
        "actual_records": (
            firewall.audit.source_ref_reads
            + firewall.audit.observation_reads
            + firewall.audit.teacher_evidence_reads
        ),
        "actual_segments": report.candidate_count + projection_count,
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
            raise RuntimeError(f"W-07 resource budget exceeded: {actual}")
    return value


def _run_payload(config, coordinator, learning_backend):
    context = _open_context(config, coordinator)
    request = _request(config, context)
    firewall = W07PayloadFirewall.open(
        Path(config.repository_root).resolve(), context, request)
    payload = firewall.read_training_payload()
    adapter = adapt_w07_training_payload(payload)
    learning = build_w07_learning_runtime(learning_backend, adapter)
    targets = _logic_targets(adapter, learning)
    carrier_backend = DictBackend()
    try:
        carrier_scope = build_w07_carrier_scope_report(
            config.repository_root, carrier_backend, targets)
    finally:
        carrier_backend.close()
    return context, request, firewall, adapter, learning, targets, carrier_scope


def run_language_stage7_public(config: W07RuntimeConfig) -> W07RunOutcome:
    """执行 W07-04 public runtime，按五事件提交并产出 canonical dump。"""
    if not isinstance(config, W07RuntimeConfig):
        raise TypeError("config 必须是 W07RuntimeConfig")
    if config.run_id != W07_FORMAL_RUN_ID:
        raise RuntimeError("W-07 run_id 必须固定为 8")
    if (config.parent_run_id != W07_W06_BASE_RUN_ID
            or config.base_run_id != W07_W06_BASE_RUN_ID):
        raise RuntimeError("W-07 parent/base 必须接 W-06 run 7")
    if config.worker_count not in W07_ALLOWED_WORKER_COUNTS:
        raise RuntimeError("W-07 worker_count 必须是 1/2/4")
    if config.mode not in W07_ALLOWED_MODES:
        raise RuntimeError("W-07 mode 必须是 fresh/restart/resume")
    _run_directory(config).mkdir(parents=True, exist_ok=True)
    coordinator = SQLiteBackend(str(Path(config.sqlite_path).resolve()))
    learning_path, attempt_ordinal = _attempt_path(config)
    attempt_backend = SQLiteBackend(str(learning_path))
    attempt_backend.close()
    learning_backend = DictBackend()
    try:
        context = _open_context(config, coordinator)
        request = _request(config, context)
        tx = W07TransactionStore(
            coordinator,
            run_id=config.run_id,
            execution_identity_key=request.execution_identity_key(),
        )
        if config.mode == "fresh" and tx.events():
            raise RuntimeError("fresh mode 要求不存在既有 W-07 transaction")
        (
            context, request, firewall, adapter, learning, targets,
            carrier_scope,
        ) = _run_payload(config, coordinator, learning_backend)
        hit_w07_fault("BEFORE_FIRST_SHARD", config.fault_point)
        shards = _shards(adapter, context.logical_shard_count)
        hit_w07_fault("AFTER_PARTIAL_SHARD", config.fault_point)
        tx.begin({
            "base_fence_key": list(request.base_fence_key),
            "request": list(request.execution_identity_key()),
        })
        hit_w07_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        report = learning.report()
        summaries, candidate_digest, evidence_digest, active_digest = (
            _logic_summaries(adapter, learning))
        preview = {
            "candidate_digest": candidate_digest,
            "logical_shard_count": context.logical_shard_count,
            "merge_barrier_key": context.merge_barrier_key,
            "shards": shards,
        }
        tx.preview(preview)
        hit_w07_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        retention = _retention(Path(config.repository_root).resolve())
        target_payload = [item.to_dict() for item in targets]
        commit_payload = {
            "carrier_scope": carrier_scope.to_dict(),
            "learning": report.__dict__,
            "logic_summaries": summaries,
            "logic_targets": target_payload,
            "retention": dict(retention),
        }
        commit = tx.commit(commit_payload)
        hit_w07_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        tx.cursor({
            "commit_sha256": commit.payload_sha256,
            "completed_shards": list(range(context.logical_shard_count)),
            "cursor_version": context.cursor_version,
        })
        resource = _resource_report(
            config, context, firewall, report, carrier_scope)
        projection_count = len(carrier_scope.records)
        cell_count = sum(
            len(item.logic_cells) for item in carrier_scope.records)
        profile_count = len({
            spec.definition.structure.stable_key()
            for spec in learning.active_specs()})
        artifact_counts = tuple(sorted({
            "ACTIVE_OPERATOR": report.active_operator_count,
            "CANDIDATE": report.candidate_count,
            "CARRIER_PROJECTION": projection_count,
            "EVIDENCE_ACCOUNT": report.operator_evidence_account_count,
            "EVIDENCE_APPLICATION": report.evidence_application_count,
            "LOGICAL_SHARD": context.logical_shard_count,
            "LOGIC_SCOPE_CELL": cell_count,
            "LOGIC_USE": len(W07_SUBSTAGE_ORDER) * 3,
            "OPERATOR_PROFILE": profile_count,
            "SCHEMA_REJECTION": report.schema_rejection_count,
            "SUBSTAGE": len(W07_SUBSTAGE_ORDER),
        }.items()))
        logical = {
            "active_projection_digest": active_digest,
            "candidate_digest": candidate_digest,
            "carrier_scope": carrier_scope.to_dict(),
            "learning": report.__dict__,
            "logic_summaries": summaries,
            "logic_targets": target_payload,
            "source_evidence_digest": evidence_digest,
        }
        transaction_events = tx.events()
        if len(transaction_events) not in {4, 5}:
            raise RuntimeError("W-07 manifest 前事务必须停在 cursor 或已发布")
        transaction_payload = [
            event.payload for event in transaction_events[:4]]
        owned_tables = _owned_tables(coordinator, learning_backend)
        dump_payload = {
            "active_candidate_count": report.active_operator_count,
            "artifact_counts": [list(item) for item in artifact_counts],
            "artifact_kind": "PH2_W07_PUBLIC_RUNTIME_DUMP",
            "base_fence_key": list(request.base_fence_key),
            "candidate_count": report.candidate_count,
            "commit": commit_payload,
            "digests": {
                "active_projection": active_digest,
                "candidate": candidate_digest,
                "logic": _digest(summaries),
                "source_evidence": evidence_digest,
            },
            "execution_state": _execution_state(),
            "format_version": 1,
            "open_generation_state": W07_OPEN_GENERATION_STATE,
            "owned_tables": list(owned_tables),
            "parent_run_id": config.parent_run_id,
            "resource_budget": dict(W07_RESOURCE_BUDGET),
            "resource_report": resource,
            "run_id": config.run_id,
            "transaction": transaction_payload,
            "transaction_event_count": 5,
        }
        dump_sha = _write_dump(config, dump_payload)
        tx.published({"dump_manifest_sha256": dump_sha})
        hit_w07_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
        return W07RunOutcome(
            _digest(logical),
            candidate_digest,
            _digest(summaries),
            evidence_digest,
            active_digest,
            carrier_scope.digest,
            _digest([event.payload for event in tx.events()]),
            dump_sha,
            report.candidate_count,
            report.active_operator_count,
            tuple((key, value["digest"])
                  for key, value in sorted(summaries.items())),
            retention,
            artifact_counts,
            _execution_state(),
            W07_OPEN_GENERATION_STATE,
            resource,
            dict(W07_RESOURCE_BUDGET),
            len(tx.events()),
            report.candidate_count
            + report.operator_evidence_account_count
            + projection_count,
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


def load_w07_public_dump(config: W07RuntimeConfig) -> W07RunOutcome:
    """只读 canonical dump，零 payload transport、零 learning write 地回读。"""
    path = _manifest_path(config)
    if not path.is_file():
        raise RuntimeError("W-07 dump manifest 缺失")
    payload = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(payload, dict)
    if (payload.get("artifact_kind") != "PH2_W07_PUBLIC_RUNTIME_DUMP"
            or payload.get("format_version") != 1
            or payload.get("run_id") != config.run_id
            or payload.get("parent_run_id") != config.parent_run_id
            or payload.get("open_generation_state")
            != W07_OPEN_GENERATION_STATE):
        raise RuntimeError("W-07 dump manifest identity 漂移")
    commit = payload.get("commit")
    digests = payload.get("digests")
    if not isinstance(commit, dict) or not isinstance(digests, dict):
        raise RuntimeError("W-07 dump commit/digests 缺失")
    summaries = commit["logic_summaries"]
    logical = {
        "active_projection_digest": digests["active_projection"],
        "candidate_digest": digests["candidate"],
        "carrier_scope": commit["carrier_scope"],
        "learning": commit["learning"],
        "logic_summaries": summaries,
        "logic_targets": commit["logic_targets"],
        "source_evidence_digest": digests["source_evidence"],
    }
    dump_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    transaction = [*payload["transaction"], {"dump_manifest_sha256": dump_sha}]
    attempts = _run_directory(config) / "learning_attempts"
    return W07RunOutcome(
        _digest(logical),
        digests["candidate"],
        digests["logic"],
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
        W07_OPEN_GENERATION_STATE,
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
    "W07RunOutcome",
    "W07RuntimeConfig",
    "load_w07_public_dump",
    "run_language_stage7_public",
]
