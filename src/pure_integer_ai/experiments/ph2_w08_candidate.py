"""W08-08 唯一 Candidate executor、host freeze 与终态封存。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_CONTRACT_FREEZE_NAME,
    W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
    W08_EXPECTED_COMMITMENTS,
    W08_EXPECTED_COUNTS,
    W08_EXPECTED_EVIDENCE_COUNTS,
    W08_EXPECTED_RECORD_COUNTS,
    _strict_sha256,
    _validate_contract,
    _validate_external_root,
    _verify_inventory,
    consume_w08_candidate_first_run_guard,
    verify_w08_candidate_contract_freeze,
    w08_candidate_contract_key,
)
from pure_integer_ai.experiments.ph2_w08_runtime import (
    load_w08_public_dump,
    run_language_stage8_public,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INPUT_KIND,
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
    W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_FORMAL_EXECUTION_STATE,
    W08_OPEN_GENERATION_PREFORMAL_STATE,
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
    W08_RUNTIME_OWNED_TABLES,
    W08RunOutcome,
    W08RuntimeConfig,
)


W08_CANDIDATE_HOST_FREEZE_KIND = "PH2_W08_CANDIDATE_HOST_FREEZE"
W08_CANDIDATE_TERMINAL_SEAL_KIND = "PH2_W08_CANDIDATE_TERMINAL_SEAL"
W08_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W08_CANDIDATE_TERMINAL_SEAL_NAME = "candidate_terminal_seal.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def formalize_w08_candidate_outcome(outcome: W08RunOutcome) -> W08RunOutcome:
    """把验证过的 preformal dump 投影到 guard 已消费的正式状态。"""
    if not isinstance(outcome, W08RunOutcome):
        raise TypeError("W08 Candidate outcome 类型非法")
    if dict(outcome.execution_state).get("W08_STARTED") != 0:
        raise RuntimeError("W08 Candidate preformal execution state 漂移")
    return replace(
        outcome,
        execution_state=tuple(sorted(W08_FORMAL_EXECUTION_STATE.items())),
    )


def _outcome_evidence(outcome: W08RunOutcome) -> dict[str, object]:
    return {
        "artifact_commitment_key": list(outcome.artifact_commitment_key),
        "artifact_evidence_counts": [
            len(item.evidence_keys) for item in outcome.artifacts
        ],
        "artifact_record_counts": [item.record_count for item in outcome.artifacts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "future_payload_reads": outcome.future_payload_reads,
        "hard_conjunct_commitment_key": list(
            outcome.hard_conjunct_commitment_key
        ),
        "host_learning_writes": outcome.host_learning_writes,
        "private_inference_interface": _inference_interface(outcome),
        "memory_learning_writes": outcome.memory_learning_writes,
        "open_generation_state": outcome.open_generation_state,
        "owned_tables": list(outcome.owned_tables),
        "payload_bytes_this_call": outcome.payload_bytes_this_call,
        "payload_gets_this_call": outcome.payload_gets_this_call,
        "resource_report": outcome.resource_report.to_dict(),
        "retention_commitment_key": list(outcome.retention_commitment_key),
        "retention_sha256": [list(item) for item in outcome.retention_sha256],
        "semantic_state_key": list(outcome.semantic_state_key),
        "teacher_calls": outcome.teacher_calls,
        "transaction_event_count": outcome.transaction_event_count,
        "use_commitment_key": list(outcome.use_commitment_key),
    }


def _inference_interface(outcome: W08RunOutcome) -> dict[str, object]:
    return {
        "component_keys": list(W08_DIMENSION_KEYS),
        "evaluator_label_inputs": 0,
        "executable": 1,
        "input_kind": W08_CANDIDATE_INFERENCE_INPUT_KIND,
        "output_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
        "per_case_invocation_required": 1,
        "rule_count": outcome.inference_rule_count,
        "state_commitment": outcome.inference_state_sha256,
        "state_key": list(outcome.inference_state_key),
        "version": outcome.inference_interface_version,
    }


def _validate_formal_outcome(
    outcome: W08RunOutcome,
    readback: W08RunOutcome,
) -> None:
    expected_commitments = {
        "artifacts": outcome.artifact_commitment_key,
        "hard_conjuncts": outcome.hard_conjunct_commitment_key,
        "inference_state": outcome.inference_state_key,
        "retention": outcome.retention_commitment_key,
        "semantic_state": outcome.semantic_state_key,
        "uses": outcome.use_commitment_key,
    }
    if (
        dict(outcome.execution_state) != W08_FORMAL_EXECUTION_STATE
        or dict(readback.execution_state) != W08_FORMAL_EXECUTION_STATE
        or outcome.dump_readback
        or not readback.dump_readback
        or outcome.canonical_key() != readback.canonical_key()
        or outcome.dump_manifest_sha256 != readback.dump_manifest_sha256
        or outcome.inference_state_key != readback.inference_state_key
        or outcome.inference_state_sha256 != readback.inference_state_sha256
        or outcome.inference_interface_version
        != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
        or readback.inference_interface_version
        != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
        or outcome.inference_rule_count
        != W08_EXPECTED_COUNTS["inference_rule_count"]
        or readback.inference_rule_count != outcome.inference_rule_count
        or tuple(item.dimension_key for item in outcome.artifacts)
        != W08_DIMENSION_KEYS
        or tuple(item.record_count for item in outcome.artifacts)
        != W08_EXPECTED_RECORD_COUNTS
        or tuple(len(item.evidence_keys) for item in outcome.artifacts)
        != W08_EXPECTED_EVIDENCE_COUNTS
        or len(outcome.uses) != W08_EXPECTED_COUNTS["use_count"]
        or any(item.outcome_state != "RESOLVED" for item in outcome.uses)
        or tuple(item.conjunct_key for item in outcome.hard_conjuncts)
        != W08_RUNTIME_HARD_CONJUNCT_KEYS
        or any(item.state != "PUBLIC_BOUNDED_PASS" for item in outcome.hard_conjuncts)
        or outcome.transaction_event_count
        != W08_EXPECTED_COUNTS["transaction_event_count"]
        or outcome.compiled_artifact_count
        != W08_EXPECTED_COUNTS["compiled_artifact_count"]
        or len(outcome.retention_sha256)
        != W08_EXPECTED_COUNTS["retention_count"]
        or outcome.owned_tables != W08_RUNTIME_OWNED_TABLES
        or outcome.payload_gets_this_call <= 0
        or outcome.payload_bytes_this_call <= 0
        or readback.payload_gets_this_call != 0
        or readback.payload_bytes_this_call != 0
        or expected_commitments != W08_EXPECTED_COMMITMENTS
    ):
        raise RuntimeError("W08 Candidate formal host/dump readback 未闭合")
    forbidden_counts = (
        outcome.teacher_calls,
        outcome.evaluator_label_reads,
        outcome.future_payload_reads,
        outcome.host_learning_writes,
        outcome.memory_learning_writes,
        readback.teacher_calls,
        readback.evaluator_label_reads,
        readback.future_payload_reads,
        readback.host_learning_writes,
        readback.memory_learning_writes,
    )
    if any(forbidden_counts):
        raise RuntimeError("W08 Candidate 越过 teacher/private/future/learning 边界")
    if (
        outcome.open_generation_state != W08_OPEN_GENERATION_PREFORMAL_STATE
        or readback.open_generation_state != W08_OPEN_GENERATION_PREFORMAL_STATE
    ):
        raise RuntimeError("W08 Candidate 提前改变 OPEN_GENERATION")


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    excluded = {
        W08_CANDIDATE_HOST_FREEZE_NAME,
        W08_CANDIDATE_TERMINAL_SEAL_NAME,
    }
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        payload = path.read_bytes()
        result.append({
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    if not result:
        raise RuntimeError("W08 Candidate root 没有可封存 artifact")
    return result


def _publish_exclusive(path: Path, value: dict[str, object], label: str) -> str:
    payload = canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"W08 Candidate {label} 不可覆盖") from error
    return _sha256_bytes(payload)


def publish_w08_candidate_host_freeze(
    repository_root: str | Path,
    artifact_root: str | Path,
    *,
    config: W08RuntimeConfig,
    contract: dict[str, object],
    candidate_contract_sha256: str,
    candidate_guard_sha256: str,
    outcome: W08RunOutcome,
    dump_readback: W08RunOutcome,
) -> tuple[Path, str]:
    """验证正式 host、零 transport readback、资源与 owner 后排他封存。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    expected_contract = _strict_sha256(
        candidate_contract_sha256, label="Candidate contract"
    )
    actual_contract = verify_w08_candidate_contract_freeze(
        root / W08_CANDIDATE_CONTRACT_FREEZE_NAME,
        value,
    )
    if actual_contract != expected_contract:
        raise RuntimeError("W08 Candidate contract SHA 漂移")
    guard = root / W08_CANDIDATE_FIRST_RUN_GUARD_NAME
    expected_guard = _strict_sha256(candidate_guard_sha256, label="Candidate guard")
    if (
        not guard.is_file()
        or guard.is_symlink()
        or _sha256_bytes(guard.read_bytes()) != expected_guard
    ):
        raise RuntimeError("W08 Candidate first-run guard 漂移")
    request = value["candidate_request"]
    if (
        not isinstance(request, dict)
        or Path(config.repository_root).resolve() != repository
        or config.run_id != request["run_id"]
        or config.parent_run_id != request["parent_run_id"]
        or config.base_run_id != request["base_run_id"]
        or config.worker_count != request["worker_count"]
        or config.mode != request["mode"]
        or config.fault_point is not None
    ):
        raise RuntimeError("W08 Candidate formal config 与 freeze 漂移")
    _validate_formal_outcome(outcome, dump_readback)
    payload = {
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W08_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_key": list(w08_candidate_contract_key(value)),
        "candidate_contract_sha256": expected_contract,
        "candidate_first_run_guard_sha256": expected_guard,
        "candidate_sealed": 1,
        "dump_readback_evidence": _outcome_evidence(dump_readback),
        "execution_state": dict(W08_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": _outcome_evidence(outcome),
        "private_inference_interface": _inference_interface(outcome),
        "open_generation_state": W08_OPEN_GENERATION_PREFORMAL_STATE,
        "owner_write_counts": {
            "candidate_artifact_writes": len(outcome.artifacts),
            "candidate_inference_state_writes": 1,
            "companion_writes": 0,
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "host_learning_writes": 0,
            "memory_learning_writes": 0,
            "readback_payload_gets": dump_readback.payload_gets_this_call,
            "teacher_calls": 0,
        },
        "public_head_commit_sha1": value["public_head_commit_sha1"],
        "terminal_state": "PASS",
    }
    target = root / W08_CANDIDATE_HOST_FREEZE_NAME
    return target, _publish_exclusive(target, payload, "host freeze")


def _publish_terminal_seal(
    root: Path,
    *,
    candidate_contract_sha256: str,
    candidate_guard_sha256: str,
    terminal_state: str,
    host_freeze_sha256: str | None = None,
    inference_state_sha256: str | None = None,
    inference_state_key: tuple[int, ...] | None = None,
    failure_phase: str | None = None,
) -> tuple[Path, str]:
    if terminal_state not in {"PASS", "FAIL", "NE", "PROCESS_EXCEPTION"}:
        raise RuntimeError("W08 Candidate terminal state 非法")
    value: dict[str, object] = {
        "artifact_kind": W08_CANDIDATE_TERMINAL_SEAL_KIND,
        "candidate_contract_sha256": _strict_sha256(
            candidate_contract_sha256, label="Candidate contract"
        ),
        "candidate_first_run_guard_sha256": _strict_sha256(
            candidate_guard_sha256, label="Candidate guard"
        ),
        "candidate_sealed": 1,
        "formal_run_count": 1,
        "format_version": 1,
        "terminal_state": terminal_state,
    }
    if host_freeze_sha256 is not None:
        value["candidate_host_freeze_sha256"] = _strict_sha256(
            host_freeze_sha256, label="Candidate host freeze"
        )
    if inference_state_sha256 is not None:
        value["candidate_inference_state_sha256"] = _strict_sha256(
            inference_state_sha256, label="Candidate inference state"
        )
    if inference_state_key is not None:
        if not inference_state_key or any(type(item) is not int for item in inference_state_key):
            raise RuntimeError("W08 Candidate inference state key 非法")
        value["candidate_inference_state_key"] = list(inference_state_key)
    if failure_phase is not None:
        value["failure_phase"] = failure_phase
    target = root / W08_CANDIDATE_TERMINAL_SEAL_NAME
    return target, _publish_exclusive(target, value, "terminal seal")


def execute_w08_candidate_once(
    repository_root: str | Path,
    artifact_root: str | Path,
    *,
    config: W08RuntimeConfig,
    contract: dict[str, object],
    candidate_contract_sha256: str,
) -> tuple[
    W08RunOutcome,
    W08RunOutcome,
    Path,
    str,
    Path,
    str,
    Path,
    str,
]:
    """消费唯一 guard，执行 train-only Candidate 并永久封存终态。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    verify_w08_candidate_contract_freeze(
        root / W08_CANDIDATE_CONTRACT_FREEZE_NAME,
        value,
    )
    run_root = Path(config.run_root).resolve()
    sqlite_path = Path(config.sqlite_path).resolve()
    if (
        not run_root.is_relative_to(root)
        or not sqlite_path.is_relative_to(root)
        or not sqlite_path.is_relative_to(run_root)
        or config.worker_count != W08_CANDIDATE_FORMAL_WORKER_COUNT
        or config.mode != W08_CANDIDATE_FORMAL_MODE
        or config.fault_point is not None
    ):
        raise RuntimeError("W08 Candidate formal host/run 必须位于专属 root")
    guard_path, guard_sha = consume_w08_candidate_first_run_guard(
        root,
        candidate_contract_sha256=candidate_contract_sha256,
    )
    try:
        raw = run_language_stage8_public(config)
        raw_readback = load_w08_public_dump(config)
        outcome = formalize_w08_candidate_outcome(raw)
        readback = formalize_w08_candidate_outcome(raw_readback)
        host_path, host_sha = publish_w08_candidate_host_freeze(
            repository,
            root,
            config=config,
            contract=value,
            candidate_contract_sha256=candidate_contract_sha256,
            candidate_guard_sha256=guard_sha,
            outcome=outcome,
            dump_readback=readback,
        )
        seal_path, seal_sha = _publish_terminal_seal(
            root,
            candidate_contract_sha256=candidate_contract_sha256,
            candidate_guard_sha256=guard_sha,
            terminal_state="PASS",
            host_freeze_sha256=host_sha,
            inference_state_sha256=outcome.inference_state_sha256,
            inference_state_key=outcome.inference_state_key,
        )
    except Exception:
        seal_path = root / W08_CANDIDATE_TERMINAL_SEAL_NAME
        if not seal_path.exists():
            _publish_terminal_seal(
                root,
                candidate_contract_sha256=candidate_contract_sha256,
                candidate_guard_sha256=guard_sha,
                terminal_state="PROCESS_EXCEPTION",
                failure_phase="FORMAL_EXECUTION_OR_SEAL",
            )
        raise
    return (
        outcome,
        readback,
        host_path,
        host_sha,
        guard_path,
        guard_sha,
        seal_path,
        seal_sha,
    )


__all__ = [
    "W08_CANDIDATE_HOST_FREEZE_KIND",
    "W08_CANDIDATE_HOST_FREEZE_NAME",
    "W08_CANDIDATE_TERMINAL_SEAL_KIND",
    "W08_CANDIDATE_TERMINAL_SEAL_NAME",
    "execute_w08_candidate_once",
    "formalize_w08_candidate_outcome",
    "publish_w08_candidate_host_freeze",
]
