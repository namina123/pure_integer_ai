"""PH2 W-04 candidate 合同、唯一正式运行守卫与 host 封存。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_ABLATION_KEYS,
    W04_AGGREGATION_POLICY,
    W04_ALLOWED_MODES,
    W04_ALLOWED_WORKER_COUNTS,
    W04_EVALUATION_ORDER,
    W04_FORMAL_RUN_ID,
    W04_GENERATION_HARD_CONJUNCT,
    W04_OPEN_GENERATION_STATE,
    W04_RESOURCE_BUDGET,
    W04_RUNNER_KEY,
    W04_TRAIN_PACK_KEYS,
    W04_W03_BASE_RUN_ID,
    W04_ZERO_EXECUTION_STATE,
    W04RunRequest,
    digest_value,
    open_w04_frozen_context,
    safe_relative_path,
)
from pure_integer_ai.experiments.ph2_w04_runtime import (
    W04RunOutcome,
    W04RuntimeConfig,
    load_w04_candidate_dump,
    run_language_stage4,
)


W04_CANDIDATE_CONTRACT_KIND = "PH2_W04_CANDIDATE_CONTRACT_FREEZE"
W04_CANDIDATE_HOST_FREEZE_KIND = "PH2_W04_CANDIDATE_HOST_FREEZE"
W04_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W04_CANDIDATE_FIRST_RUN_GUARD"
W04_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W04_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W04_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W04_CANDIDATE_FORMAL_WORKER_COUNT = 4
W04_CANDIDATE_FORMAL_MODE = "fresh"
W04_FORMAL_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 1,
    "W05_STARTED": 0,
    "formal_w04_training_runs": 1,
    "teacher_calls": 0,
}

W04_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w04_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w04_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w04_contract.py",
    "src/pure_integer_ai/experiments/ph2_w04_faults.py",
    "src/pure_integer_ai/experiments/ph2_w04_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w04_generation.py",
    "src/pure_integer_ai/experiments/ph2_w04_generation_contract.py",
    "src/pure_integer_ai/experiments/ph2_w04_learning.py",
    "src/pure_integer_ai/experiments/ph2_w04_payload.py",
    "src/pure_integer_ai/experiments/ph2_w04_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w04_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w04_transaction.py",
    "src/pure_integer_ai/experiments/ph2_w04_understanding.py",
)
W04_CANDIDATE_TEST_PATHS = (
    "tests/test_w04_stage2_adapter_learning.py",
    "tests/test_w04_stage2_candidate.py",
    "tests/test_w04_stage2_contract.py",
    "tests/test_w04_stage2_runtime.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise RuntimeError(f"W-04 {label} 不是规范 SHA-256")
    return value


def _inventory(repository: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
    result = []
    for relative in paths:
        normalized = safe_relative_path(relative, label="W-04 inventory path")
        target = (repository / Path(*PurePosixPath(normalized).parts)).resolve()
        if (not target.is_file() or target.is_symlink()
                or not target.is_relative_to(repository)):
            raise RuntimeError("W-04 candidate inventory 缺失、逃逸或为链接")
        payload = target.read_bytes()
        result.append({
            "path": normalized,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    return result


def _validate_external_root(repository: Path, artifact_root: Path) -> None:
    if (artifact_root == repository
            or artifact_root.is_relative_to(repository)
            or repository.is_relative_to(artifact_root)):
        raise RuntimeError("W-04 candidate root 必须与 Git root 物理隔离")


def _identity(value: Any) -> dict[str, object]:
    result = value.to_dict()
    if not isinstance(result, dict):
        raise RuntimeError("W-04 identity 不能规范序列化")
    return result


def build_w04_candidate_contract(
        repository_root: str | Path,
        *,
        global_manifest_path: str,
        backend_profile_key: tuple[int, ...],
        current_remote_commit_sha1: str,
        dependency_root: str | Path | None = None,
        ) -> dict[str, object]:
    """在 payload 读取前冻结 W-04 candidate 的全部合同。"""
    repository = Path(repository_root).resolve()
    context = open_w04_frozen_context(
        repository,
        global_manifest_path,
        current_remote_commit_sha1=current_remote_commit_sha1,
        backend_profile_key=backend_profile_key,
        dependency_root=dependency_root,
    )
    request = W04RunRequest(
        run_id=context.run_id,
        parent_run_id=context.parent_run_id,
        base_run_id=context.base_run_id,
        stage_key=context.stage_key,
        owner_key=context.owner_key,
        runner_key=W04_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=W04_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W04_CANDIDATE_FORMAL_MODE,
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    return {
        "aggregation_policy": W04_AGGREGATION_POLICY,
        "artifact_kind": W04_CANDIDATE_CONTRACT_KIND,
        "candidate_request": {
            "base_fence_key": list(request.base_fence_key),
            "base_run_id": request.base_run_id,
            "candidate_payload_count": len(request.candidate_payload_paths),
            "candidate_payload_paths": list(request.candidate_payload_paths),
            "context_key": list(context.stable_key()),
            "mode": request.mode,
            "owner_key": request.owner_key,
            "parent_run_id": request.parent_run_id,
            "run_id": request.run_id,
            "teacher_evidence_count": len(request.teacher_evidence_paths),
            "teacher_evidence_paths": list(request.teacher_evidence_paths),
            "worker_count": request.worker_count,
        },
        "code_inventory": _inventory(repository, W04_CANDIDATE_CODE_PATHS),
        "d03_w04_binding": {
            "d03_global_manifest_identity": _identity(
                context.d03_global_manifest_identity),
            "d03_receipt_identity": _identity(context.d03_receipt_identity),
            "pack_bindings": [item.to_dict() for item in context.pack_bindings],
            "pre_w04_gate_sha256": context.pre_w04_gate_sha256,
            "stage_manifest_identity": _identity(context.stage_manifest_identity),
            "train_pack_keys": list(context.train_pack_keys),
            "version_keys": [list(item) for item in context.version_keys],
        },
        "evaluation_contract": {
            "ablation_order": list(W04_ABLATION_KEYS),
            "aggregation_policy": context.aggregation_policy,
            "dimension_keys": list(context.dimension_keys),
            "evaluation_order": list(W04_EVALUATION_ORDER),
            "generation_hard_conjunct": W04_GENERATION_HARD_CONJUNCT,
            "thresholds": [item.to_dict() for item in context.d03_thresholds],
        },
        "execution_state": dict(W04_ZERO_EXECUTION_STATE),
        "formal_w04_training_runs": 0,
        "format_version": 1,
        "open_generation_state": W04_OPEN_GENERATION_STATE,
        "payload_audit": {
            "learning_writes": 0,
            "payload_bytes": 0,
            "payload_gets": 0,
            "teacher_calls": 0,
        },
        "recovery_protocol": {
            "failure_points": list(context.failure_point_keys),
            "logical_shard_count": context.logical_shard_count,
            "modes": list(W04_ALLOWED_MODES),
            "worker_counts": list(W04_ALLOWED_WORKER_COUNTS),
        },
        "remote_commit_sha1": context.current_remote_commit_sha1,
        "resource_budget": dict(W04_RESOURCE_BUDGET),
        "self_excluded": 1,
        "test_inventory": _inventory(repository, W04_CANDIDATE_TEST_PATHS),
        "visibility_counts": {
            "candidate_payloads": len(context.candidate_payload_bindings),
            "evaluator_visible": len(context.evaluator_visible_bindings),
            "teacher_evidence": len(context.teacher_evidence_bindings),
            "train_pack_count": len(W04_TRAIN_PACK_KEYS),
        },
    }


def _validate_inventory(value: object, expected: tuple[str, ...], label: str) -> None:
    if (not isinstance(value, list)
            or tuple(item.get("path") for item in value
                     if isinstance(item, dict)) != expected
            or len(value) != len(expected)):
        raise RuntimeError(f"W-04 {label} inventory 路径漂移")
    for item in value:
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] <= 0):
            raise RuntimeError(f"W-04 {label} inventory 字段非法")
        _strict_sha256(item["sha256"], label=f"{label} inventory")


def _validate_contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("W-04 candidate 合同类型非法")
    if (value.get("artifact_kind") != W04_CANDIDATE_CONTRACT_KIND
            or value.get("format_version") != 1
            or value.get("self_excluded") != 1
            or value.get("execution_state") != W04_ZERO_EXECUTION_STATE
            or value.get("formal_w04_training_runs") != 0
            or value.get("open_generation_state") != W04_OPEN_GENERATION_STATE):
        raise RuntimeError("W-04 candidate 合同状态漂移")
    _validate_inventory(value.get("code_inventory"),
                        W04_CANDIDATE_CODE_PATHS, "code")
    _validate_inventory(value.get("test_inventory"),
                        W04_CANDIDATE_TEST_PATHS, "test")
    request = value.get("candidate_request")
    if (not isinstance(request, dict)
            or request.get("run_id") != W04_FORMAL_RUN_ID
            or request.get("parent_run_id") != W04_W03_BASE_RUN_ID
            or request.get("base_run_id") != W04_W03_BASE_RUN_ID
            or request.get("worker_count") != W04_CANDIDATE_FORMAL_WORKER_COUNT
            or request.get("mode") != W04_CANDIDATE_FORMAL_MODE):
        raise RuntimeError("W-04 candidate request 合同漂移")
    recovery = value.get("recovery_protocol")
    if (not isinstance(recovery, dict)
            or recovery.get("logical_shard_count") != 16
            or tuple(recovery.get("worker_counts", ()))
            != W04_ALLOWED_WORKER_COUNTS
            or tuple(recovery.get("modes", ())) != W04_ALLOWED_MODES
            or len(recovery.get("failure_points", ())) != 6):
        raise RuntimeError("W-04 recovery 合同漂移")
    evaluation = value.get("evaluation_contract")
    if (not isinstance(evaluation, dict)
            or evaluation.get("evaluation_order") != list(W04_EVALUATION_ORDER)
            or evaluation.get("aggregation_policy") != W04_AGGREGATION_POLICY
            or evaluation.get("generation_hard_conjunct")
            != W04_GENERATION_HARD_CONJUNCT):
        raise RuntimeError("W-04 evaluation 合同漂移")
    if value.get("resource_budget") != W04_RESOURCE_BUDGET:
        raise RuntimeError("W-04 resource 合同漂移")
    return value


def w04_candidate_contract_key(value: dict[str, object]) -> tuple[int, ...]:
    _validate_contract(value)
    return digest_value(value)


def _verify_inventory(repository: Path, contract: dict[str, object]) -> None:
    if (contract["code_inventory"] != _inventory(
            repository, W04_CANDIDATE_CODE_PATHS)
            or contract["test_inventory"] != _inventory(
                repository, W04_CANDIDATE_TEST_PATHS)):
        raise RuntimeError("W-04 candidate code/test identity 漂移")


def publish_w04_candidate_contract_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        contract: dict[str, object],
        ) -> tuple[Path, str]:
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    root.mkdir(parents=True, exist_ok=True)
    target = root / W04_CANDIDATE_CONTRACT_FREEZE_NAME
    encoded = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise RuntimeError("W-04 candidate contract freeze 不可覆盖") from exc
    return target, _sha256_bytes(encoded)


def verify_w04_candidate_contract_freeze(
        freeze_path: str | Path,
        contract: dict[str, object],
        ) -> str:
    value = _validate_contract(contract)
    path = Path(freeze_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W-04 candidate contract freeze 缺失")
    encoded = path.read_bytes()
    expected = canonical_json_bytes(value)
    if encoded != expected:
        raise RuntimeError("W-04 candidate contract identity 漂移")
    return _sha256_bytes(encoded)


def consume_w04_candidate_first_run_guard(
        artifact_root: str | Path,
        *,
        candidate_contract_sha256: str,
        ) -> tuple[Path, str]:
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(candidate_contract_sha256,
                              label="candidate contract")
    freeze = root / W04_CANDIDATE_CONTRACT_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or _sha256_bytes(freeze.read_bytes()) != expected):
        raise RuntimeError("W-04 candidate contract SHA 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": W04_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": expected,
        "execution_state_after_start": dict(W04_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "run_id": W04_FORMAL_RUN_ID,
    })
    target = root / W04_CANDIDATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-04 candidate first-run 已消费，不可重跑") from exc
    return target, _sha256_bytes(payload)


def formalize_w04_candidate_outcome(outcome: W04RunOutcome) -> W04RunOutcome:
    if not isinstance(outcome, W04RunOutcome):
        raise TypeError("W-04 outcome 类型非法")
    if outcome.execution_state != W04_ZERO_EXECUTION_STATE:
        raise RuntimeError("W-04 pre-formal execution state 漂移")
    return replace(outcome, execution_state=dict(W04_FORMAL_EXECUTION_STATE))


def _outcome_evidence(outcome: W04RunOutcome) -> dict[str, object]:
    return {
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "host_digests": {
            "candidate": outcome.candidate_digest,
            "generation": outcome.generation_digest,
            "logical": outcome.logical_state_digest,
            "reasoning": outcome.reasoning_digest,
            "transaction": outcome.transaction_digest,
            "understanding": outcome.understanding_digest,
        },
        "new_learning_write_count": outcome.new_learning_write_count,
        "owned_tables": list(outcome.owned_tables),
        "resource_report": dict(sorted(outcome.resource_report.items())),
        "teacher_calls": outcome.teacher_calls,
        "transaction_event_count": outcome.transaction_event_count,
    }


def _logical_outcome_key(outcome: W04RunOutcome) -> tuple[int, ...]:
    return digest_value({
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "candidate": outcome.candidate_digest,
        "generation": outcome.generation_digest,
        "logical": outcome.logical_state_digest,
        "reasoning": outcome.reasoning_digest,
        "understanding": outcome.understanding_digest,
    })


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W04_CANDIDATE_HOST_FREEZE_NAME:
            continue
        payload = path.read_bytes()
        result.append({
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    if not result:
        raise RuntimeError("W-04 candidate root 没有可封存 artifact")
    return result


def publish_w04_candidate_host_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W04RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        outcome: W04RunOutcome,
        dump_readback: W04RunOutcome,
        ) -> tuple[Path, str]:
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    expected_contract = _strict_sha256(candidate_contract_sha256,
                                       label="candidate contract")
    actual_contract = verify_w04_candidate_contract_freeze(
        root / W04_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    if actual_contract != expected_contract:
        raise RuntimeError("W-04 candidate contract SHA 漂移")
    guard = root / W04_CANDIDATE_FIRST_RUN_GUARD_NAME
    if not guard.is_file() or guard.is_symlink():
        raise RuntimeError("W-04 first-run guard 缺失")
    request = value["candidate_request"]
    if (not isinstance(request, dict)
            or config.run_id != request["run_id"]
            or config.parent_run_id != request["parent_run_id"]
            or config.base_run_id != request["base_run_id"]
            or tuple(config.base_fence_key or ())
            != tuple(request["base_fence_key"])
            or config.worker_count != request["worker_count"]
            or config.mode != request["mode"]
            or config.current_remote_commit_sha1 != value["remote_commit_sha1"]):
        raise RuntimeError("W-04 formal config 与 candidate 合同漂移")
    if (outcome.execution_state != W04_FORMAL_EXECUTION_STATE
            or dump_readback.execution_state != W04_FORMAL_EXECUTION_STATE
            or outcome.dump_readback
            or not dump_readback.dump_readback
            or outcome.active_candidate_count != 1
            or dump_readback.active_candidate_count != 1
            or outcome.transaction_event_count != 4
            or dump_readback.transaction_event_count != 4
            or outcome.new_learning_write_count <= 0
            or dump_readback.new_learning_write_count != 0
            or outcome.teacher_calls != 0
            or dump_readback.teacher_calls != 0
            or _logical_outcome_key(outcome) != _logical_outcome_key(dump_readback)):
        raise RuntimeError("W-04 formal host/dump readback 未闭合")
    for key, actual in outcome.resource_report.items():
        if key.startswith("actual_"):
            budget_key = "max_" + key.removeprefix("actual_")
            if actual > W04_RESOURCE_BUDGET.get(budget_key, actual):
                raise RuntimeError("W-04 formal resource 超预算")
    evidence = _outcome_evidence(outcome)
    readback_evidence = _outcome_evidence(dump_readback)
    artifact_counts = dict(outcome.artifact_counts)
    payload = canonical_json_bytes({
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W04_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_key": list(w04_candidate_contract_key(value)),
        "candidate_contract_sha256": expected_contract,
        "code_inventory": value["code_inventory"],
        "dump_readback_evidence": readback_evidence,
        "execution_state": dict(W04_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": evidence,
        "open_generation_state": W04_OPEN_GENERATION_STATE,
        "owner_write_counts": {
            "artifact_writes": sum(artifact_counts.values()),
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": dump_readback.new_learning_write_count,
            "teacher_calls": 0,
        },
        "remote_commit_sha1": value["remote_commit_sha1"],
        "request": value["candidate_request"],
        "self_excluded": 1,
        "test_inventory": value["test_inventory"],
    })
    target = root / W04_CANDIDATE_HOST_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-04 candidate host freeze 不可覆盖") from exc
    return target, _sha256_bytes(payload)


def execute_w04_candidate_once(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W04RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        dump_readback_sqlite_path: str | Path,
        ) -> tuple[W04RunOutcome, W04RunOutcome, Path, str, Path, str]:
    """消费唯一 guard，运行正式 candidate 一次并封存 host。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    verify_w04_candidate_contract_freeze(
        root / W04_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    sqlite_path = Path(config.sqlite_path).resolve()
    run_root = Path(config.run_root).resolve()
    readback_path = Path(dump_readback_sqlite_path).resolve()
    if (not sqlite_path.is_relative_to(root)
            or not run_root.is_relative_to(root)
            or not readback_path.is_relative_to(root)
            or readback_path == sqlite_path):
        raise RuntimeError("W-04 formal host/run/readback 必须位于 candidate root")
    guard_path, guard_sha = consume_w04_candidate_first_run_guard(
        root, candidate_contract_sha256=candidate_contract_sha256)
    raw = run_language_stage4(config)
    raw_readback = load_w04_candidate_dump(
        replace(config, sqlite_path=readback_path, fault_point=None))
    outcome = formalize_w04_candidate_outcome(raw)
    readback = formalize_w04_candidate_outcome(raw_readback)
    freeze_path, freeze_sha = publish_w04_candidate_host_freeze(
        repository,
        root,
        config=config,
        contract=value,
        candidate_contract_sha256=candidate_contract_sha256,
        outcome=outcome,
        dump_readback=readback,
    )
    return outcome, readback, freeze_path, freeze_sha, guard_path, guard_sha


__all__ = [
    "W04_CANDIDATE_CODE_PATHS",
    "W04_CANDIDATE_CONTRACT_FREEZE_NAME",
    "W04_CANDIDATE_CONTRACT_KIND",
    "W04_CANDIDATE_FIRST_RUN_GUARD_NAME",
    "W04_CANDIDATE_FIRST_RUN_GUARD_KIND",
    "W04_CANDIDATE_FORMAL_MODE",
    "W04_CANDIDATE_FORMAL_WORKER_COUNT",
    "W04_CANDIDATE_HOST_FREEZE_NAME",
    "W04_CANDIDATE_HOST_FREEZE_KIND",
    "W04_CANDIDATE_TEST_PATHS",
    "W04_FORMAL_EXECUTION_STATE",
    "build_w04_candidate_contract",
    "consume_w04_candidate_first_run_guard",
    "execute_w04_candidate_once",
    "formalize_w04_candidate_outcome",
    "publish_w04_candidate_contract_freeze",
    "publish_w04_candidate_host_freeze",
    "verify_w04_candidate_contract_freeze",
    "w04_candidate_contract_key",
]
