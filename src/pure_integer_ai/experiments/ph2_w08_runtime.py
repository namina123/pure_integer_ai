"""W08-07 独立事务、恢复、资源、retention 与 canonical dump runtime。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_DIMENSION_KEYS,
    read_w08_authority,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_ALLOWED_MODES,
    W08_ALLOWED_WORKER_COUNTS,
    W08_CONSUMER_KEYS,
    W08_FAILURE_POINT_KEYS,
    W08_OWNER_KEY,
    W08_ZERO_EXECUTION_STATE,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_discourse_training import (
    audit_w08_discourse_training,
)
from pure_integer_ai.experiments.ph2_w08_faults import hit_w08_fault
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_long_context_training import (
    compile_w08_long_context_training,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08CandidateInferenceState,
)
from pure_integer_ai.experiments.ph2_w08_inference_training import (
    compile_w08_candidate_inference_state,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_training import (
    compile_w08_p3ia_training,
)
from pure_integer_ai.experiments.ph2_w08_recompute_training import (
    audit_w08_recompute_training,
)
from pure_integer_ai.experiments.ph2_w08_registry import audit_w08_registry_payload
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_FORMAL_RUN_ID,
    W08_OPEN_GENERATION_PREFORMAL_STATE,
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
    W08_RUNTIME_OWNED_TABLES,
    W08_W07_BASE_RUN_ID,
    W08HardConjunctEvidence,
    W08LogicalShard,
    W08RunOutcome,
    W08RuntimeConfig,
    W08RuntimeError,
    W08RuntimeResourceReceipt,
    W08RuntimeUse,
    W08TrainArtifact,
    build_semantic_state_key,
)
from pure_integer_ai.experiments.ph2_w08_transaction import W08TransactionStore
from pure_integer_ai.experiments.ph2_w08_variation import learn_w08_variation
from pure_integer_ai.storage.backend import SQLiteBackend


W08_STAGE6_VALIDATION_PATH = (
    "data/ph2/manifests/d03_v1/w08_06_validation_v1.json"
)
W08_STAGE6_VALIDATION_SHA256 = (
    "3059a368651eb9ae06378031bdaef376bed10f085e7810f0fa01116010a4e866"
)
W08_PUBLIC_DUMP_NAME = "w08_dump_manifest.json"


def _digest(value: Any) -> tuple[int, ...]:
    return digest_value(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W08RuntimeError(f"W08 runtime parent 缺失：{path.name}") from error
    return digest.hexdigest()


def _validate_config(config: W08RuntimeConfig) -> None:
    if not isinstance(config, W08RuntimeConfig):
        raise TypeError("config 必须是 W08RuntimeConfig")
    if config.run_id != W08_FORMAL_RUN_ID:
        raise W08RuntimeError("W-08 run_id 必须固定为 9")
    if (
        config.parent_run_id != W08_W07_BASE_RUN_ID
        or config.base_run_id != W08_W07_BASE_RUN_ID
    ):
        raise W08RuntimeError("W-08 parent/base 必须接 W-07 run 8")
    if config.worker_count not in W08_ALLOWED_WORKER_COUNTS:
        raise W08RuntimeError("W-08 worker_count 必须是 1/2/4")
    if config.mode not in W08_ALLOWED_MODES:
        raise W08RuntimeError("W-08 mode 必须是 fresh/restart/resume")
    if config.fault_point is not None and config.fault_point not in W08_FAILURE_POINT_KEYS:
        raise W08RuntimeError("W-08 fault point 未注册")
    repository = Path(config.repository_root).resolve()
    run_root = Path(config.run_root).resolve()
    sqlite_path = Path(config.sqlite_path).resolve()
    if run_root == repository:
        raise W08RuntimeError("W-08 run root 不得是公开 Git 根")
    try:
        sqlite_path.relative_to(run_root)
    except ValueError as error:
        raise W08RuntimeError("W-08 coordinator 必须位于 run root 内") from error


def _run_directory(config: W08RuntimeConfig) -> Path:
    return Path(config.run_root).resolve() / f"w08_run_{config.run_id:020d}"


def _manifest_path(config: W08RuntimeConfig) -> Path:
    return _run_directory(config) / W08_PUBLIC_DUMP_NAME


def _read_stage6_validation(repository: Path) -> tuple[dict[str, Any], str]:
    path = repository / W08_STAGE6_VALIDATION_PATH
    actual_sha = _sha256(path)
    if actual_sha != W08_STAGE6_VALIDATION_SHA256:
        raise W08RuntimeError("W08-06 validation identity 漂移")
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W08RuntimeError("W08-06 validation JSON 损坏") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != encoded:
        raise W08RuntimeError("W08-06 validation 不是 frozen canonical line")
    result = value.get("result")
    restricted = value.get("restricted_evidence")
    state = value.get("state")
    if (
        value.get("artifact_kind") != "PH2_W08_06_VALIDATION"
        or value.get("artifact_version") != "PH2-W08-06-VALIDATION-V1"
        or not isinstance(result, dict)
        or result.get("status") != "PASS"
        or result.get("passed_tests") != result.get("collected_tests")
        or any(result.get(name) != 0 for name in ("collection_errors", "failed_tests", "skipped_tests"))
        or not isinstance(restricted, dict)
        or any(
            restricted.get(name) != 0
            for name in (
                "blocked_network_calls",
                "formal_guard_writes",
                "future_pack_reads",
                "future_source_reads",
                "future_test_reads",
                "host_learning_writes",
                "memory_learning_writes",
                "private_evaluator_reads",
                "public_materialization_writes",
                "teacher_calls",
            )
        )
        or not isinstance(state, dict)
        or state.get("W08_STARTED") != 0
        or state.get("formal_w08_training_runs") != 0
        or state.get("OPEN_GENERATION") != W08_OPEN_GENERATION_PREFORMAL_STATE
    ):
        raise W08RuntimeError("W08-06 validation 未形成安全 public bounded PASS")
    return value, actual_sha


def _retention(repository: Path) -> tuple[tuple[str, str], ...]:
    authority = read_w08_authority(repository)
    identities = authority.get("retention_identities")
    if not isinstance(identities, list) or not identities:
        raise W08RuntimeError("W08 retention inventory 缺失")
    result = []
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise W08RuntimeError("W08 retention identity schema 漂移")
        relative = identity["relative_path"]
        expected = identity["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise W08RuntimeError("W08 retention identity type 漂移")
        path = (repository / relative).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise W08RuntimeError("W08 retention path 越界") from error
        if (
            not path.is_file()
            or path.stat().st_size != identity["size_bytes"]
            or _sha256(path) != expected
        ):
            raise W08RuntimeError(f"W08 retention identity 漂移：{relative}")
        result.append((relative, expected))
    return tuple(result)


def _evidence_for(payload, observation_keys: set[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    values = tuple(
        sorted(
            tuple(item.stable_key.components)
            for item in payload.teacher_evidence
            if tuple(item.observation_key.components) in observation_keys
        )
    )
    if not values:
        raise W08RuntimeError("W08 train artifact 没有 exact Evidence")
    return values


def _training_artifacts(payload) -> tuple[W08TrainArtifact, ...]:
    registry = audit_w08_registry_payload(payload)
    variation = learn_w08_variation(payload)
    discourse = audit_w08_discourse_training(payload)
    recompute = audit_w08_recompute_training(payload)
    long_context = compile_w08_long_context_training(payload)
    p3ia = compile_w08_p3ia_training(payload)
    all_observations = {
        tuple(item.stable_key.components) for item in payload.observations
    }
    discourse_observations = {
        tuple(item.stable_key.components)
        for item in payload.observations
        if item.payload_kind != "RAW_SOURCE_OBSERVATION_V1"
    }
    revision_observations = {
        tuple(item.stable_key.components)
        for item in payload.observations
        if item.payload_kind == "DiscourseRevisionQuery"
    }
    long_observations = {item.observation_key for item in long_context.material}
    p3ia_observations = {item.observation_key for item in p3ia.cases}
    artifacts = (
        W08TrainArtifact(
            W08_DIMENSION_KEYS[0],
            "W08_VARIATION_TRAIN_ARTIFACT",
            _digest(
                {
                    "allowed_typed_operations": list(variation.allowed_typed_operations),
                    "evidence_bindings": [list(item) for item in variation.evidence_bindings],
                    "payload_kinds": list(variation.payload_kinds),
                    "schema_fingerprints": [
                        [kind, list(key)] for kind, key in variation.schema_fingerprints
                    ],
                    "source_parser_receipt_count": variation.source_parser_receipt_count,
                }
            ),
            _evidence_for(payload, all_observations),
            registry.observation_count,
        ),
        W08TrainArtifact(
            W08_DIMENSION_KEYS[1],
            "W08_DISCOURSE_TRAIN_ARTIFACT",
            _digest(asdict(discourse)),
            _evidence_for(payload, discourse_observations),
            discourse.observation_count,
        ),
        W08TrainArtifact(
            W08_DIMENSION_KEYS[2],
            "W08_LOCAL_RECOMPUTE_TRAIN_ARTIFACT",
            _digest(asdict(recompute)),
            _evidence_for(payload, revision_observations),
            recompute.discourse_revision_count,
        ),
        W08TrainArtifact(
            W08_DIMENSION_KEYS[3],
            "W08_LONG_CONTEXT_TRAIN_ARTIFACT",
            _digest(
                {
                    "audit": asdict(long_context.audit),
                    "material_key": list(long_context.material_key),
                }
            ),
            _evidence_for(payload, long_observations),
            long_context.audit.material_item_count,
        ),
        W08TrainArtifact(
            W08_DIMENSION_KEYS[4],
            "W08_P3IA_TRAIN_ARTIFACT",
            p3ia.family_identity,
            _evidence_for(payload, p3ia_observations),
            p3ia.audit.case_count,
        ),
    )
    return artifacts


def _uses(
    request_key: tuple[int, ...],
    artifacts: tuple[W08TrainArtifact, ...],
) -> tuple[W08RuntimeUse, ...]:
    result = []
    for artifact in artifacts:
        for consumer in W08_CONSUMER_KEYS:
            base = {
                "artifact": list(artifact.artifact_key),
                "consumer": consumer,
                "dimension": artifact.dimension_key,
                "request": list(request_key),
            }
            result.append(
                W08RuntimeUse(
                    artifact.dimension_key,
                    consumer,
                    request_key,
                    artifact.artifact_key,
                    artifact.evidence_keys,
                    _digest({**base, "kind": "directional-choice"}),
                    _digest({**base, "kind": "use"}),
                    "RESOLVED",
                    _digest({**base, "kind": "outcome", "state": "RESOLVED"}),
                )
            )
    return tuple(result)


def _hard_conjuncts(
    validation: dict[str, Any],
    validation_sha: str,
) -> tuple[W08HardConjunctEvidence, ...]:
    implementation = validation.get("implementation_files")
    if not isinstance(implementation, list) or not implementation:
        raise W08RuntimeError("W08-06 implementation inventory 缺失")
    return tuple(
        W08HardConjunctEvidence(
            key,
            "PUBLIC_BOUNDED_PASS",
            validation_sha,
            _digest(
                {
                    "conjunct": key,
                    "implementation": implementation,
                    "validation_sha256": validation_sha,
                }
            ),
        )
        for key in W08_RUNTIME_HARD_CONJUNCT_KEYS
    )


def _logical_shards(
    payload,
    *,
    fault_point: str | None = None,
) -> tuple[W08LogicalShard, ...]:
    records = tuple(
        sorted(
            {
                *(tuple(item.stable_key.components) for item in payload.observations),
                *(tuple(item.stable_key.components) for item in payload.teacher_evidence),
            }
        )
    )
    buckets: list[list[tuple[int, ...]]] = [[] for _ in range(16)]
    halfway = max(1, len(records) // 2)
    for ordinal, record in enumerate(records, start=1):
        shard = _digest({"record": list(record), "shard_count": 16})[0] % 16
        buckets[shard].append(record)
        if ordinal == halfway:
            hit_w08_fault("AFTER_PARTIAL_SHARD", fault_point)
    return tuple(
        W08LogicalShard(
            index,
            tuple(sorted(values)),
            _digest(
                {
                    "records": [list(item) for item in sorted(values)],
                    "shard_index": index,
                }
            ),
        )
        for index, values in enumerate(buckets)
    )


def _resource_report(config, payload, firewall, artifacts, uses, shards) -> W08RuntimeResourceReceipt:
    long_context = compile_w08_long_context_training(payload)
    recompute = audit_w08_recompute_training(payload)
    return W08RuntimeResourceReceipt(
        len(payload.source_refs)
        + len(payload.observations)
        + len(payload.teacher_evidence)
        + len(artifacts)
        + len(uses)
        + len(shards),
        firewall.audit.payload_bytes,
        firewall.audit.payload_gets,
        sum(item.record_count for item in artifacts) + len(uses) * 3 + len(shards),
        recompute.affected_occurrence_count + recompute.recompute_query_count,
        long_context.audit.material_item_count,
        1,
        config.worker_count,
    )


def _write_dump(config: W08RuntimeConfig, payload: dict[str, Any]) -> str:
    target = _manifest_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if target.exists():
        if target.read_bytes() != encoded:
            raise W08RuntimeError("W08 dump manifest 已存在但 identity 漂移")
    else:
        target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _commit_digests(commit: dict[str, Any]) -> dict[str, list[int]]:
    artifacts = tuple(W08TrainArtifact.from_dict(item) for item in commit["artifacts"])
    uses = tuple(W08RuntimeUse.from_dict(item) for item in commit["uses"])
    hard = tuple(W08HardConjunctEvidence.from_dict(item) for item in commit["hard_conjuncts"])
    retention = tuple(tuple(item) for item in commit["retention_sha256"])
    inference_state = W08CandidateInferenceState.from_dict(
        commit["candidate_inference_state"]
    )
    if commit.get("candidate_inference_state_sha256") != inference_state.sha256():
        raise W08RuntimeError("W08 dump inference state SHA 漂移")
    return {
        "artifacts": list(_digest([item.to_dict() for item in artifacts])),
        "hard_conjuncts": list(_digest([item.to_dict() for item in hard])),
        "inference_state": list(inference_state.state_key),
        "retention": list(_digest([list(item) for item in retention])),
        "semantic_state": list(build_semantic_state_key(
            artifacts,
            uses,
            hard,
            retention,
            inference_state.state_key,
        )),
        "uses": list(_digest([item.to_dict() for item in uses])),
    }


def _dump_payload(
    config: W08RuntimeConfig,
    request,
    tx: W08TransactionStore,
    commit: dict[str, Any],
) -> dict[str, Any]:
    events = tx.events()
    if len(events) not in {4, 5}:
        raise W08RuntimeError("W08 manifest 前事务必须停在 cursor 或 published")
    return {
        "artifact_kind": "PH2_W08_PUBLIC_RUNTIME_DUMP",
        "base_fence_key": list(request.base_fence_key),
        "base_run_id": config.base_run_id,
        "commit": commit,
        "digests": _commit_digests(commit),
        "execution_state": dict(W08_ZERO_EXECUTION_STATE),
        "format_version": 1,
        "open_generation_state": W08_OPEN_GENERATION_PREFORMAL_STATE,
        "owned_tables": list(W08_RUNTIME_OWNED_TABLES),
        "parent_run_id": config.parent_run_id,
        "run_id": config.run_id,
        "transaction": [item.payload for item in events[:4]],
        "transaction_event_count": 5,
    }


def _parse_dump(config: W08RuntimeConfig) -> tuple[dict[str, Any], str]:
    path = _manifest_path(config)
    if not path.is_file():
        raise W08RuntimeError("W08 dump manifest 缺失")
    try:
        payload = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    except DatasetContractError as error:
        raise W08RuntimeError("W08 dump identity/state 漂移") from error
    assert isinstance(payload, dict)
    if (
        payload.get("artifact_kind") != "PH2_W08_PUBLIC_RUNTIME_DUMP"
        or payload.get("format_version") != 1
        or payload.get("run_id") != config.run_id
        or payload.get("parent_run_id") != config.parent_run_id
        or payload.get("base_run_id") != config.base_run_id
        or payload.get("execution_state") != W08_ZERO_EXECUTION_STATE
        or payload.get("open_generation_state") != W08_OPEN_GENERATION_PREFORMAL_STATE
        or tuple(payload.get("owned_tables", ())) != W08_RUNTIME_OWNED_TABLES
        or payload.get("transaction_event_count") != 5
    ):
        raise W08RuntimeError("W08 dump identity/state 漂移")
    commit = payload.get("commit")
    digests = payload.get("digests")
    if not isinstance(commit, dict) or digests != _commit_digests(commit):
        raise W08RuntimeError("W08 dump commit digest 漂移")
    return payload, _sha256(path)


def _outcome_from_dump(
    config: W08RuntimeConfig,
    request,
    *,
    payload_gets_this_call: int,
    payload_bytes_this_call: int,
    dump_readback: bool,
) -> W08RunOutcome:
    payload, dump_sha = _parse_dump(config)
    commit = payload["commit"]
    artifacts = tuple(W08TrainArtifact.from_dict(item) for item in commit["artifacts"])
    uses = tuple(W08RuntimeUse.from_dict(item) for item in commit["uses"])
    hard = tuple(W08HardConjunctEvidence.from_dict(item) for item in commit["hard_conjuncts"])
    retention = tuple((str(item[0]), str(item[1])) for item in commit["retention_sha256"])
    resource = W08RuntimeResourceReceipt.from_dict(commit["resource_report"])
    inference_state = W08CandidateInferenceState.from_dict(
        commit["candidate_inference_state"]
    )
    published = {"dump_manifest_sha256": dump_sha}
    transaction = [*payload["transaction"], published]
    digests = payload["digests"]
    audit = commit["payload_audit"]
    return W08RunOutcome(
        tuple(digests["semantic_state"]),
        tuple(digests["artifacts"]),
        tuple(digests["uses"]),
        tuple(digests["hard_conjuncts"]),
        tuple(digests["retention"]),
        _digest(transaction),
        request.scheduling_key(),
        dump_sha,
        inference_state.state_key,
        inference_state.sha256(),
        inference_state.interface_version,
        len(inference_state.rules),
        artifacts,
        uses,
        hard,
        retention,
        resource,
        W08_RUNTIME_OWNED_TABLES,
        tuple(sorted(W08_ZERO_EXECUTION_STATE.items())),
        W08_OPEN_GENERATION_PREFORMAL_STATE,
        5,
        len(artifacts),
        payload_gets_this_call,
        payload_bytes_this_call,
        int(audit["teacher_calls"]),
        int(audit["evaluator_label_reads"]),
        int(audit["future_payload_reads"]),
        int(audit["learning_writes"]),
        int(audit["memory_learning_writes"]),
        dump_readback,
    )


def _finish_committed(
    config: W08RuntimeConfig,
    request,
    tx: W08TransactionStore,
    *,
    payload_gets_this_call: int,
    payload_bytes_this_call: int,
) -> W08RunOutcome:
    events = tx.events()
    if len(events) < 3:
        raise W08RuntimeError("W08 commit 尚未形成")
    commit_event = events[2]
    if len(events) == 3:
        tx.cursor(
            {
                "commit_sha256": commit_event.payload_sha256,
                "completed_shards": list(range(16)),
                "cursor_version": "PH2-D03-CURSOR-V1",
            }
        )
    dump_payload = _dump_payload(config, request, tx, commit_event.payload)
    dump_sha = _write_dump(config, dump_payload)
    tx.published({"dump_manifest_sha256": dump_sha})
    hit_w08_fault("AFTER_MANIFEST_PUBLISH", config.fault_point)
    return _outcome_from_dump(
        config,
        request,
        payload_gets_this_call=payload_gets_this_call,
        payload_bytes_this_call=payload_bytes_this_call,
        dump_readback=False,
    )


def run_language_stage8_public(config: W08RuntimeConfig) -> W08RunOutcome:
    """运行 W08-07 public transaction，不消费 formal guard 或 evaluator label。"""
    _validate_config(config)
    repository = Path(config.repository_root).resolve()
    _run_directory(config).mkdir(parents=True, exist_ok=True)
    Path(config.sqlite_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    coordinator = SQLiteBackend(str(Path(config.sqlite_path).resolve()))
    try:
        context = open_w08_frozen_contract(repository)
        request = make_w08_request(
            context,
            worker_count=config.worker_count,
            mode=config.mode,
        )
        tx = W08TransactionStore(
            coordinator,
            run_id=config.run_id,
            owner_key=W08_OWNER_KEY,
            execution_identity_key=request.execution_identity_key(),
        )
        events = tx.events()
        if config.mode == "fresh" and events:
            raise W08RuntimeError("fresh mode 要求不存在既有 W08 transaction")
        if len(events) >= 3:
            return _finish_committed(
                config,
                request,
                tx,
                payload_gets_this_call=0,
                payload_bytes_this_call=0,
            )

        firewall = W08PayloadFirewall.open(repository, context, request)
        payload = firewall.read_training_payload()
        artifacts = _training_artifacts(payload)
        inference_state = compile_w08_candidate_inference_state(payload)
        uses = _uses(request.execution_identity_key(), artifacts)
        validation, validation_sha = _read_stage6_validation(repository)
        hard = _hard_conjuncts(validation, validation_sha)
        retention = _retention(repository)
        hit_w08_fault("BEFORE_FIRST_SHARD", config.fault_point)
        shards = _logical_shards(payload, fault_point=config.fault_point)
        tx.begin(
            {
                "base_fence_key": list(request.base_fence_key),
                "owner_key": context.owner_key,
                "request_key": list(request.execution_identity_key()),
            }
        )
        hit_w08_fault("BEFORE_MERGE_PREVIEW", config.fault_point)
        preview = {
            "logical_shard_count": 16,
            "merge_barrier_key": context.merge_barrier_key,
            "shards": [item.to_dict() for item in shards],
        }
        tx.preview(preview)
        hit_w08_fault("AFTER_MERGE_BEFORE_COMMIT", config.fault_point)
        resource = _resource_report(config, payload, firewall, artifacts, uses, shards)
        commit_payload = {
            "artifacts": [item.to_dict() for item in artifacts],
            "candidate_inference_state": inference_state.to_dict(),
            "candidate_inference_state_sha256": inference_state.sha256(),
            "hard_conjuncts": [item.to_dict() for item in hard],
            "payload_audit": {
                "evaluator_label_reads": firewall.audit.evaluator_label_reads,
                "future_payload_reads": firewall.audit.future_payload_reads,
                "learning_writes": firewall.audit.learning_writes,
                "memory_learning_writes": firewall.audit.memory_learning_writes,
                "payload_bytes": firewall.audit.payload_bytes,
                "payload_gets": firewall.audit.payload_gets,
                "teacher_calls": firewall.audit.teacher_calls,
            },
            "resource_report": resource.to_dict(),
            "retention_sha256": [list(item) for item in retention],
            "stage6_validation_sha256": validation_sha,
            "uses": [item.to_dict() for item in uses],
        }
        tx.commit(commit_payload)
        hit_w08_fault("AFTER_COMMIT_BEFORE_CURSOR", config.fault_point)
        return _finish_committed(
            config,
            request,
            tx,
            payload_gets_this_call=firewall.audit.payload_gets,
            payload_bytes_this_call=firewall.audit.payload_bytes,
        )
    finally:
        coordinator.close()


def load_w08_public_dump(config: W08RuntimeConfig) -> W08RunOutcome:
    """只读 canonical dump，不二次读取 train payload 或写学习状态。"""
    _validate_config(config)
    context = open_w08_frozen_contract(Path(config.repository_root).resolve())
    request = make_w08_request(
        context,
        worker_count=config.worker_count,
        mode=config.mode,
    )
    return _outcome_from_dump(
        config,
        request,
        payload_gets_this_call=0,
        payload_bytes_this_call=0,
        dump_readback=True,
    )


def load_w08_candidate_inference_state(
    config: W08RuntimeConfig,
) -> W08CandidateInferenceState:
    """从 canonical dump 只读恢复可执行 state，不读取 train transport。"""
    _validate_config(config)
    payload, _ = _parse_dump(config)
    state = W08CandidateInferenceState.from_dict(
        payload["commit"]["candidate_inference_state"]
    )
    if payload["commit"].get("candidate_inference_state_sha256") != state.sha256():
        raise W08RuntimeError("W08 Candidate inference state readback 漂移")
    return state


__all__ = [
    "W08_PUBLIC_DUMP_NAME",
    "W08_STAGE6_VALIDATION_PATH",
    "W08_STAGE6_VALIDATION_SHA256",
    "load_w08_candidate_inference_state",
    "load_w08_public_dump",
    "run_language_stage8_public",
]
