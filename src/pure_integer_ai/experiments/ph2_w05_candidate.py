"""PH2 W-05 candidate 合同、唯一正式运行守卫与 host 封存。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_ABLATION_KEYS,
    W05_AGGREGATION_POLICY,
    W05_ALLOWED_MODES,
    W05_ALLOWED_WORKER_COUNTS,
    W05_EVALUATION_ORDER,
    W05_FORMAL_RUN_ID,
    W05_GENERATION_HARD_CONJUNCT,
    W05_OPEN_GENERATION_STATE,
    W05_PRIVATE_ABLATION_KEYS,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_TRAIN_PACK_KEYS,
    W05_W04_BASE_RUN_ID,
    W05_ZERO_EXECUTION_STATE,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
    safe_relative_path,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_HARD_CASES,
)
from pure_integer_ai.experiments.ph2_w05_runtime import (
    W05RunOutcome,
    W05RuntimeConfig,
    load_w05_candidate_dump,
    run_language_stage5,
)


W05_CANDIDATE_CONTRACT_KIND = "PH2_W05_CANDIDATE_CONTRACT_FREEZE"
W05_CANDIDATE_HOST_FREEZE_KIND = "PH2_W05_CANDIDATE_HOST_FREEZE"
W05_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W05_CANDIDATE_FIRST_RUN_GUARD"
W05_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W05_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W05_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W05_CANDIDATE_FORMAL_WORKER_COUNT = 4
W05_CANDIDATE_FORMAL_MODE = "fresh"
W05_FORMAL_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W05_STARTED": 1,
    "W06_STARTED": 0,
    "formal_w05_training_runs": 1,
    "teacher_calls": 0,
}

W05_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w05_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w05_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w05_carrier_scope.py",
    "src/pure_integer_ai/experiments/ph2_w05_contract.py",
    "src/pure_integer_ai/experiments/ph2_w05_faults.py",
    "src/pure_integer_ai/experiments/ph2_w05_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w05_generation.py",
    "src/pure_integer_ai/experiments/ph2_w05_generation_contract.py",
    "src/pure_integer_ai/experiments/ph2_w05_learning.py",
    "src/pure_integer_ai/experiments/ph2_w05_payload.py",
    "src/pure_integer_ai/experiments/ph2_w05_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w05_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w05_transaction.py",
    "src/pure_integer_ai/experiments/ph2_w05_understanding.py",
)
W05_CANDIDATE_TEST_PATHS = (
    "tests/test_w05_stage2_adapter_learning.py",
    "tests/test_w05_stage2_candidate.py",
    "tests/test_w05_stage2_consumers.py",
    "tests/test_w05_stage2_contract.py",
    "tests/test_w05_stage2_runtime.py",
)


def _sha256_bytes(value: bytes) -> str:
    """返回一段字节的 SHA-256。"""
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    """校验并返回规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise RuntimeError(f"W-05 {label} 不是规范 SHA-256")
    return value


def _inventory(repository: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
    """冻结精确路径、大小和文件 SHA，不跟随 symlink 或逃逸路径。"""
    result = []
    for relative in paths:
        normalized = safe_relative_path(relative, label="W-05 inventory path")
        target = (repository / Path(*PurePosixPath(normalized).parts)).resolve()
        if (not target.is_file() or target.is_symlink()
                or not target.is_relative_to(repository)):
            raise RuntimeError("W-05 candidate inventory 缺失、逃逸或为链接")
        payload = target.read_bytes()
        result.append({
            "path": normalized,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    return result


def _validate_external_root(repository: Path, artifact_root: Path) -> None:
    """拒绝把 candidate artifact 放进 Git root 或父子目录。"""
    if (artifact_root == repository
            or artifact_root.is_relative_to(repository)
            or repository.is_relative_to(artifact_root)):
        raise RuntimeError("W-05 candidate root 必须与 Git root 物理隔离")


def _identity(value: Any) -> dict[str, object]:
    """把公开 identity 转成规范 object。"""
    result = value.to_dict()
    if not isinstance(result, dict):
        raise RuntimeError("W-05 identity 不能规范序列化")
    return result


def build_w05_candidate_contract(
        repository_root: str | Path,
        *,
        global_manifest_path: str,
        backend_profile_key: tuple[int, ...],
        current_remote_commit_sha1: str,
        dependency_root: str | Path | None = None,
        ) -> dict[str, object]:
    """在 payload 读取和 first-run guard 前冻结 W-05 candidate 全合同。"""
    repository = Path(repository_root).resolve()
    context = open_w05_frozen_context(
        repository,
        global_manifest_path,
        current_remote_commit_sha1=current_remote_commit_sha1,
        backend_profile_key=backend_profile_key,
        dependency_root=dependency_root,
    )
    request = W05RunRequest(
        run_id=context.run_id,
        parent_run_id=context.parent_run_id,
        base_run_id=context.base_run_id,
        stage_key=context.stage_key,
        owner_key=context.owner_key,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=W05_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W05_CANDIDATE_FORMAL_MODE,
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    return {
        "aggregation_policy": W05_AGGREGATION_POLICY,
        "artifact_kind": W05_CANDIDATE_CONTRACT_KIND,
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
            "w04_receipt_key": list(request.w04_receipt_key),
            "worker_count": request.worker_count,
        },
        "code_inventory": _inventory(repository, W05_CANDIDATE_CODE_PATHS),
        "d03_w05_binding": {
            "d03_global_manifest_identity": _identity(
                context.d03_global_manifest_identity),
            "d03_receipt_identity": _identity(context.d03_receipt_identity),
            "lc16_directional_identity": _identity(
                context.lc16_directional_identity),
            "lc16_mapper_identity": _identity(context.lc16_mapper_identity),
            "lc16_overlay_identity": _identity(context.lc16_overlay_identity),
            "lc16_projection_identity": _identity(
                context.lc16_projection_identity),
            "pack_bindings": [item.to_dict() for item in context.pack_bindings],
            "pre_w04_gate_sha256": context.pre_w04_gate_sha256,
            "stage_manifest_identity": _identity(context.stage_manifest_identity),
            "train_pack_keys": list(context.train_pack_keys),
            "version_keys": [list(item) for item in context.version_keys],
            "w04_receipt_identity": _identity(context.w04_receipt_identity),
        },
        "evaluation_contract": {
            "aggregation_policy": context.aggregation_policy,
            "d03_ablation_order": list(W05_ABLATION_KEYS),
            "dimension_keys": list(context.dimension_keys),
            "evaluation_order": list(W05_EVALUATION_ORDER),
            "formal_ablation_order": list(W05_PRIVATE_ABLATION_KEYS),
            "generation_case_order": list(W05_GENERATION_HARD_CASES),
            "generation_hard_conjunct": W05_GENERATION_HARD_CONJUNCT,
            "thresholds": [item.to_dict() for item in context.d03_thresholds],
        },
        "execution_state": dict(W05_ZERO_EXECUTION_STATE),
        "formal_w05_training_runs": 0,
        "format_version": 1,
        "open_generation_state": W05_OPEN_GENERATION_STATE,
        "payload_audit": {
            "learning_writes": 0,
            "payload_bytes": 0,
            "payload_gets": 0,
            "teacher_calls": 0,
        },
        "recovery_protocol": {
            "failure_points": list(context.failure_point_keys),
            "logical_shard_count": context.logical_shard_count,
            "modes": list(W05_ALLOWED_MODES),
            "worker_counts": list(W05_ALLOWED_WORKER_COUNTS),
        },
        "remote_commit_sha1": context.current_remote_commit_sha1,
        "resource_budget": dict(W05_RESOURCE_BUDGET),
        "self_excluded": 1,
        "test_inventory": _inventory(repository, W05_CANDIDATE_TEST_PATHS),
        "visibility_counts": {
            "candidate_payloads": len(context.candidate_payload_bindings),
            "evaluator_visible": len(context.evaluator_visible_bindings),
            "teacher_evidence": len(context.teacher_evidence_bindings),
            "train_pack_count": len(W05_TRAIN_PACK_KEYS),
        },
    }


def _validate_inventory(value: object, expected: tuple[str, ...], label: str) -> None:
    """校验 inventory 路径顺序、字段、大小和 SHA。"""
    if (not isinstance(value, list)
            or tuple(item.get("path") for item in value
                     if isinstance(item, dict)) != expected
            or len(value) != len(expected)):
        raise RuntimeError(f"W-05 {label} inventory 路径漂移")
    for item in value:
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] <= 0):
            raise RuntimeError(f"W-05 {label} inventory 字段非法")
        _strict_sha256(item["sha256"], label=f"{label} inventory")


def _validate_contract(value: object) -> dict[str, object]:
    """校验 zero-state candidate freeze 的全部承重字段。"""
    if not isinstance(value, dict):
        raise RuntimeError("W-05 candidate 合同类型非法")
    if (value.get("artifact_kind") != W05_CANDIDATE_CONTRACT_KIND
            or value.get("format_version") != 1
            or value.get("self_excluded") != 1
            or value.get("execution_state") != W05_ZERO_EXECUTION_STATE
            or value.get("formal_w05_training_runs") != 0
            or value.get("open_generation_state") != W05_OPEN_GENERATION_STATE):
        raise RuntimeError("W-05 candidate 合同状态漂移")
    _validate_inventory(value.get("code_inventory"),
                        W05_CANDIDATE_CODE_PATHS, "code")
    _validate_inventory(value.get("test_inventory"),
                        W05_CANDIDATE_TEST_PATHS, "test")
    request = value.get("candidate_request")
    if (not isinstance(request, dict)
            or request.get("run_id") != W05_FORMAL_RUN_ID
            or request.get("parent_run_id") != W05_W04_BASE_RUN_ID
            or request.get("base_run_id") != W05_W04_BASE_RUN_ID
            or request.get("worker_count") != W05_CANDIDATE_FORMAL_WORKER_COUNT
            or request.get("mode") != W05_CANDIDATE_FORMAL_MODE):
        raise RuntimeError("W-05 candidate request 合同漂移")
    recovery = value.get("recovery_protocol")
    if (not isinstance(recovery, dict)
            or recovery.get("logical_shard_count") != 16
            or tuple(recovery.get("worker_counts", ()))
            != W05_ALLOWED_WORKER_COUNTS
            or tuple(recovery.get("modes", ())) != W05_ALLOWED_MODES
            or len(recovery.get("failure_points", ())) != 6):
        raise RuntimeError("W-05 recovery 合同漂移")
    evaluation = value.get("evaluation_contract")
    if (not isinstance(evaluation, dict)
            or evaluation.get("evaluation_order") != list(W05_EVALUATION_ORDER)
            or evaluation.get("aggregation_policy") != W05_AGGREGATION_POLICY
            or evaluation.get("d03_ablation_order") != list(W05_ABLATION_KEYS)
            or evaluation.get("formal_ablation_order")
            != list(W05_PRIVATE_ABLATION_KEYS)
            or evaluation.get("generation_case_order")
            != list(W05_GENERATION_HARD_CASES)
            or evaluation.get("generation_hard_conjunct")
            != W05_GENERATION_HARD_CONJUNCT):
        raise RuntimeError("W-05 evaluation 合同漂移")
    if value.get("resource_budget") != W05_RESOURCE_BUDGET:
        raise RuntimeError("W-05 resource 合同漂移")
    return value


def w05_candidate_contract_key(value: dict[str, object]) -> tuple[int, ...]:
    """返回已验证 candidate contract 的稳定键。"""
    _validate_contract(value)
    return digest_value(value)


def _verify_inventory(repository: Path, contract: dict[str, object]) -> None:
    """在 freeze/guard/host 发布前重算 code/test inventory。"""
    if (contract["code_inventory"] != _inventory(
            repository, W05_CANDIDATE_CODE_PATHS)
            or contract["test_inventory"] != _inventory(
                repository, W05_CANDIDATE_TEST_PATHS)):
        raise RuntimeError("W-05 candidate code/test identity 漂移")


def publish_w05_candidate_contract_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        contract: dict[str, object],
        ) -> tuple[Path, str]:
    """在 Git 外 root 排他写入 candidate contract freeze。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    root.mkdir(parents=True, exist_ok=True)
    target = root / W05_CANDIDATE_CONTRACT_FREEZE_NAME
    encoded = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise RuntimeError("W-05 candidate contract freeze 不可覆盖") from exc
    return target, _sha256_bytes(encoded)


def verify_w05_candidate_contract_freeze(
        freeze_path: str | Path,
        contract: dict[str, object],
        ) -> str:
    """逐字节回验 candidate freeze 与内存合同一致。"""
    value = _validate_contract(contract)
    path = Path(freeze_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W-05 candidate contract freeze 缺失")
    encoded = path.read_bytes()
    expected = canonical_json_bytes(value)
    if encoded != expected:
        raise RuntimeError("W-05 candidate contract identity 漂移")
    return _sha256_bytes(encoded)


def consume_w05_candidate_first_run_guard(
        artifact_root: str | Path,
        *,
        candidate_contract_sha256: str,
        ) -> tuple[Path, str]:
    """排他创建 first-run guard，并把正式运行计数从 0 推到 1。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(candidate_contract_sha256,
                              label="candidate contract")
    freeze = root / W05_CANDIDATE_CONTRACT_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or _sha256_bytes(freeze.read_bytes()) != expected):
        raise RuntimeError("W-05 candidate contract SHA 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": W05_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": expected,
        "execution_state_after_start": dict(W05_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "run_id": W05_FORMAL_RUN_ID,
    })
    target = root / W05_CANDIDATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-05 candidate first-run 已消费，不可重跑") from exc
    return target, _sha256_bytes(payload)


def formalize_w05_candidate_outcome(outcome: W05RunOutcome) -> W05RunOutcome:
    """只在 guard 消费后把 public runtime outcome 标为唯一 formal run。"""
    if not isinstance(outcome, W05RunOutcome):
        raise TypeError("W-05 outcome 类型非法")
    if outcome.execution_state != W05_ZERO_EXECUTION_STATE:
        raise RuntimeError("W-05 pre-formal execution state 漂移")
    return replace(outcome, execution_state=dict(W05_FORMAL_EXECUTION_STATE))


def _outcome_evidence(outcome: W05RunOutcome) -> dict[str, object]:
    """返回不含训练 surface/label 的 host 或 readback 安全证据。"""
    return {
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "host_digests": {
            "candidate": outcome.candidate_digest,
            "carrier_scope": outcome.carrier_scope_digest,
            "generation": outcome.generation_digest,
            "logical": outcome.logical_state_digest,
            "reasoning": outcome.reasoning_digest,
            "transaction": outcome.transaction_digest,
            "understanding": outcome.understanding_digest,
        },
        "learning_attempt_count": outcome.learning_attempt_count,
        "new_learning_write_count": outcome.new_learning_write_count,
        "owned_tables": list(outcome.owned_tables),
        "payload_bytes_this_call": outcome.payload_bytes_this_call,
        "payload_gets_this_call": outcome.payload_gets_this_call,
        "resource_report": dict(sorted(outcome.resource_report.items())),
        "teacher_calls": outcome.teacher_calls,
        "transaction_event_count": outcome.transaction_event_count,
    }


def _logical_outcome_key(outcome: W05RunOutcome) -> tuple[int, ...]:
    """返回排除 readback 物理路径的完整逻辑 outcome key。"""
    return digest_value({
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "candidate": outcome.candidate_digest,
        "carrier_scope": outcome.carrier_scope_digest,
        "generation": outcome.generation_digest,
        "logical": outcome.logical_state_digest,
        "reasoning": outcome.reasoning_digest,
        "transaction": outcome.transaction_digest,
        "understanding": outcome.understanding_digest,
    })


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    """冻结 candidate root 中 host freeze 自身以外的全部物理文件。"""
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W05_CANDIDATE_HOST_FREEZE_NAME:
            continue
        payload = path.read_bytes()
        result.append({
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    if not result:
        raise RuntimeError("W-05 candidate root 没有可封存 artifact")
    return result


def publish_w05_candidate_host_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W05RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        outcome: W05RunOutcome,
        dump_readback: W05RunOutcome,
        ) -> tuple[Path, str]:
    """闭合 formal host、零 transport readback、资源和 inventory 后封存。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    expected_contract = _strict_sha256(candidate_contract_sha256,
                                       label="candidate contract")
    actual_contract = verify_w05_candidate_contract_freeze(
        root / W05_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    if actual_contract != expected_contract:
        raise RuntimeError("W-05 candidate contract SHA 漂移")
    guard = root / W05_CANDIDATE_FIRST_RUN_GUARD_NAME
    if not guard.is_file() or guard.is_symlink():
        raise RuntimeError("W-05 first-run guard 缺失")
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
        raise RuntimeError("W-05 formal config 与 candidate 合同漂移")
    if (outcome.execution_state != W05_FORMAL_EXECUTION_STATE
            or dump_readback.execution_state != W05_FORMAL_EXECUTION_STATE
            or outcome.dump_readback
            or not dump_readback.dump_readback
            or outcome.active_candidate_count != 2
            or dump_readback.active_candidate_count != 2
            or outcome.transaction_event_count != 5
            or dump_readback.transaction_event_count != 5
            or outcome.new_learning_write_count <= 0
            or dump_readback.new_learning_write_count != 0
            or outcome.payload_gets_this_call <= 0
            or outcome.payload_bytes_this_call <= 0
            or dump_readback.payload_gets_this_call != 0
            or dump_readback.payload_bytes_this_call != 0
            or outcome.teacher_calls != 0
            or dump_readback.teacher_calls != 0
            or outcome.open_generation_state != W05_OPEN_GENERATION_STATE
            or dump_readback.open_generation_state != W05_OPEN_GENERATION_STATE
            or outcome.dump_manifest_sha256 != dump_readback.dump_manifest_sha256
            or _logical_outcome_key(outcome) != _logical_outcome_key(dump_readback)):
        raise RuntimeError("W-05 formal host/dump readback 未闭合")
    for key, actual in outcome.resource_report.items():
        if key.startswith("actual_"):
            budget_key = "max_" + key.removeprefix("actual_")
            if actual > W05_RESOURCE_BUDGET.get(budget_key, actual):
                raise RuntimeError("W-05 formal resource 超预算")
    artifact_counts = dict(outcome.artifact_counts)
    payload = canonical_json_bytes({
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W05_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_key": list(w05_candidate_contract_key(value)),
        "candidate_contract_sha256": expected_contract,
        "code_inventory": value["code_inventory"],
        "dump_readback_evidence": _outcome_evidence(dump_readback),
        "execution_state": dict(W05_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": _outcome_evidence(outcome),
        "open_generation_state": W05_OPEN_GENERATION_STATE,
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
    target = root / W05_CANDIDATE_HOST_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-05 candidate host freeze 不可覆盖") from exc
    return target, _sha256_bytes(payload)


def execute_w05_candidate_once(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W05RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        dump_readback_sqlite_path: str | Path,
        ) -> tuple[W05RunOutcome, W05RunOutcome, Path, str, Path, str]:
    """消费唯一 guard，执行一次 candidate 并封存 host/readback。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    verify_w05_candidate_contract_freeze(
        root / W05_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    sqlite_path = Path(config.sqlite_path).resolve()
    run_root = Path(config.run_root).resolve()
    readback_path = Path(dump_readback_sqlite_path).resolve()
    if (not sqlite_path.is_relative_to(root)
            or not run_root.is_relative_to(root)
            or not readback_path.is_relative_to(root)
            or readback_path == sqlite_path):
        raise RuntimeError("W-05 formal host/run/readback 必须位于 candidate root")
    guard_path, guard_sha = consume_w05_candidate_first_run_guard(
        root, candidate_contract_sha256=candidate_contract_sha256)
    raw = run_language_stage5(config)
    raw_readback = load_w05_candidate_dump(
        replace(config, sqlite_path=readback_path, fault_point=None))
    outcome = formalize_w05_candidate_outcome(raw)
    readback = formalize_w05_candidate_outcome(raw_readback)
    freeze_path, freeze_sha = publish_w05_candidate_host_freeze(
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
    "W05_CANDIDATE_CODE_PATHS",
    "W05_CANDIDATE_CONTRACT_FREEZE_NAME",
    "W05_CANDIDATE_CONTRACT_KIND",
    "W05_CANDIDATE_FIRST_RUN_GUARD_NAME",
    "W05_CANDIDATE_FIRST_RUN_GUARD_KIND",
    "W05_CANDIDATE_FORMAL_MODE",
    "W05_CANDIDATE_FORMAL_WORKER_COUNT",
    "W05_CANDIDATE_HOST_FREEZE_NAME",
    "W05_CANDIDATE_HOST_FREEZE_KIND",
    "W05_CANDIDATE_TEST_PATHS",
    "W05_FORMAL_EXECUTION_STATE",
    "build_w05_candidate_contract",
    "consume_w05_candidate_first_run_guard",
    "execute_w05_candidate_once",
    "formalize_w05_candidate_outcome",
    "publish_w05_candidate_contract_freeze",
    "publish_w05_candidate_host_freeze",
    "verify_w05_candidate_contract_freeze",
    "w05_candidate_contract_key",
]
