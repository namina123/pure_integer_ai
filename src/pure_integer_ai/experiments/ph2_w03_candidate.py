"""PH2 W-03 candidate 的运行前冻结、唯一运行守卫与 host 封存。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_contract_core import sha1_text
from pure_integer_ai.experiments.ph2_w03_context import open_w03_frozen_context
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationObservation,
    formal_w03_publication_baseline,
    validate_w03_publication_observation,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_AGGREGATION_POLICY,
    W03_ALLOWED_MODES,
    W03_ALLOWED_WORKER_COUNTS,
    W03_EVALUATION_ORDER,
    W03_FORMAL_RUN_ID,
    W03_RESOURCE_BUDGET,
    W03_ZERO_EXECUTION_STATE,
    digest_value,
    safe_relative_path,
    strict_key,
)
from pure_integer_ai.experiments.ph2_w03_runtime import (
    W03RunOutcome,
    W03RuntimeConfig,
    load_w03_candidate_dump,
    run_language_stage2,
)


W03_CANDIDATE_CONTRACT_KIND = "PH2_W03_CANDIDATE_CONTRACT_FREEZE"
W03_CANDIDATE_HOST_FREEZE_KIND = "PH2_W03_CANDIDATE_HOST_FREEZE"
W03_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W03_CANDIDATE_FIRST_RUN_GUARD"
W03_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W03_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W03_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W03_CANDIDATE_FORMAL_WORKER_COUNT = 4
W03_CANDIDATE_FORMAL_MODE = "fresh"
W03_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_d03_release_reader.py",
    "src/pure_integer_ai/experiments/ph2_wikidata_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w03_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w03_adapter_extractors.py",
    "src/pure_integer_ai/experiments/ph2_w03_artifacts.py",
    "src/pure_integer_ai/experiments/ph2_w03_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w03_context.py",
    "src/pure_integer_ai/experiments/ph2_w03_continuity.py",
    "src/pure_integer_ai/experiments/ph2_w03_contract.py",
    "src/pure_integer_ai/experiments/ph2_w03_faults.py",
    "src/pure_integer_ai/experiments/ph2_w03_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w03_generation.py",
    "src/pure_integer_ai/experiments/ph2_w03_generation_contract.py",
    "src/pure_integer_ai/experiments/ph2_w03_learning.py",
    "src/pure_integer_ai/experiments/ph2_w03_payload.py",
    "src/pure_integer_ai/experiments/ph2_w03_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w03_shards.py",
    "src/pure_integer_ai/experiments/ph2_w03_transaction.py",
    "src/pure_integer_ai/experiments/ph2_w03_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w03_understanding_contract.py",
)
W03_CANDIDATE_TEST_PATHS = (
    "tests/test_w03_stage2_visibility.py",
    "tests/test_w03_stage2_contract.py",
    "tests/test_w03_stage2_adapter.py",
    "tests/test_w03_stage2_understanding.py",
    "tests/test_w03_stage2_generation.py",
    "tests/test_w03_stage2_runtime.py",
    "tests/test_w03_stage2_candidate.py",
)
W03_FORMAL_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W03_STARTED": 1,
    "W04_STARTED": 0,
    "formal_w03_training_runs": 1,
    "teacher_calls": 0,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    """验证小写 SHA-256 文本并返回原值。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"{label} 不是规范 SHA-256")
    return value


def _inventory(
        repository_root: Path,
        paths: tuple[str, ...],
        ) -> list[dict[str, object]]:
    """形成逐文件 size/SHA inventory，拒绝逃逸、链接和缺失文件。"""
    result = []
    for relative in paths:
        normalized = safe_relative_path(relative, label="candidate inventory path")
        target = (repository_root / Path(*PurePosixPath(normalized).parts)).resolve()
        if (not target.is_relative_to(repository_root)
                or not target.is_file() or target.is_symlink()):
            raise RuntimeError("W-03 candidate inventory 文件缺失、逃逸或为链接")
        payload = target.read_bytes()
        result.append({
            "path": normalized,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    return result


def _identity_payload(identity: Any) -> dict[str, object]:
    """把正式文件身份转成可冻结的规范对象。"""
    value = identity.to_dict()
    if not isinstance(value, dict):
        raise RuntimeError("W-03 identity 不能规范序列化")
    return value


def _publication_payload(observation: W03PublicationObservation) -> dict[str, object]:
    """保存实际观测到的 local/tracking/remote 与四项成功 CI。"""
    return {
        "ci_conclusion": observation.ci_conclusion,
        "ci_head_sha1": observation.ci_head_sha1,
        "ci_jobs": [item.to_dict() for item in observation.ci_jobs],
        "ci_run_id": observation.ci_run_id,
        "ci_status": observation.ci_status,
        "local_head_sha1": observation.local_head_sha1,
        "remote_head_sha1": observation.remote_head_sha1,
        "tracking_head_sha1": observation.tracking_head_sha1,
    }


def _w02_payload(continuity: Any) -> dict[str, object]:
    """冻结 W-02 receipt、四个公开件、10/11 inventory 与五维 retention。"""
    return {
        "aggregate_identity": _identity_payload(continuity.aggregate_identity),
        "candidate_attestation_identity": _identity_payload(
            continuity.candidate_attestation_identity),
        "candidate_freeze_identity": _identity_payload(
            continuity.candidate_freeze_identity),
        "candidate_run_id": continuity.candidate_run_id,
        "capability_code_count": len(continuity.capability_code_identities),
        "capability_code_identities": [
            _identity_payload(item) for item in continuity.capability_code_identities],
        "dimension_pass_counts": list(continuity.dimension_pass_counts),
        "dimension_statuses": list(continuity.dimension_statuses),
        "execution_state": dict(continuity.execution_state),
        "fail_count": continuity.fail_count,
        "formal_training_runs": continuity.formal_training_runs,
        "historical_ci_jobs": [
            item.to_dict() for item in continuity.historical_ci_jobs],
        "historical_ci_run_id": continuity.historical_ci_run_id,
        "historical_publication_head_sha1": (
            continuity.historical_publication_head_sha1),
        "host_artifact_count": len(continuity.host_artifact_identities),
        "host_artifact_identities": [
            _identity_payload(item) for item in continuity.host_artifact_identities],
        "host_digests": dict(continuity.host_digests),
        "ne_count": continuity.ne_count,
        "receipt_identity": _identity_payload(continuity.receipt_identity),
        "stable_key": list(continuity.stable_key()),
    }


def build_w03_candidate_contract(
        repository_root: str | Path,
        w02_artifacts_root: str | Path,
        *,
        global_manifest_path: str,
        backend_profile_key: tuple[int, ...],
        current_remote_commit_sha1: str,
        publication_observation: W03PublicationObservation,
        dependency_root: str | Path | None = None,
        ) -> dict[str, object]:
    """零 payload 地绑定 W03-05 正式运行前的全部不可变合同。"""
    repository = Path(repository_root).resolve()
    w02_root = Path(w02_artifacts_root).resolve()
    if not repository.is_dir() or not w02_root.is_dir():
        raise RuntimeError("W-03 candidate repository/W-02 root 不存在")
    remote = sha1_text(
        current_remote_commit_sha1, where="W-03 candidate remote commit")
    strict_key(backend_profile_key, label="candidate backend profile")
    baseline = formal_w03_publication_baseline()
    observation = validate_w03_publication_observation(
        baseline, publication_observation)
    if remote != baseline.head_sha1:
        raise RuntimeError("W-03 candidate remote commit 与 publication baseline 漂移")
    continuity = verify_formal_w02_continuity(repository, w02_root)
    context = open_w03_frozen_context(
        repository,
        global_manifest_path,
        current_remote_commit_sha1=remote,
        w02_continuity=continuity,
        publication_baseline=baseline,
        backend_profile_key=backend_profile_key,
        dependency_root=dependency_root,
    )
    threshold = {
        "fail_allowed": 0,
        "ne_policy": "BLOCK",
        "required_pass_denominator": 1,
        "required_pass_numerator": 1,
    }
    return {
        "aggregation_policy": W03_AGGREGATION_POLICY,
        "artifact_kind": W03_CANDIDATE_CONTRACT_KIND,
        "candidate_request": {
            "base_fence_key": list(context.base_fence_key),
            "base_run_id": context.base_run_id,
            "candidate_payload_count": len(context.candidate_payload_bindings),
            "candidate_payload_paths": [
                item.relative_path for item in context.candidate_payload_bindings],
            "context_key": list(context.stable_key()),
            "mode": W03_CANDIDATE_FORMAL_MODE,
            "owner_key": context.owner_key,
            "parent_run_id": context.parent_run_id,
            "run_id": context.run_id,
            "teacher_evidence_count": len(context.teacher_evidence_bindings),
            "teacher_evidence_paths": [
                item.relative_path for item in context.teacher_evidence_bindings],
            "worker_count": W03_CANDIDATE_FORMAL_WORKER_COUNT,
        },
        "code_inventory": _inventory(repository, W03_CANDIDATE_CODE_PATHS),
        "d03_w03_binding": {
            "global_manifest_identity": _identity_payload(
                context.d03_global_manifest_identity),
            "pack_bindings": [item.to_dict() for item in context.pack_bindings],
            "post_publication_receipt_identity": _identity_payload(
                context.d03_receipt_identity),
            "stage_manifest_identity": _identity_payload(
                context.stage_manifest_identity),
            "train_pack_count": len(context.pack_bindings),
            "version_keys": [list(item) for item in context.version_keys],
        },
        "evaluation_contract": {
            "ablation_order": list(context.ablation_keys),
            "aggregation_policy": context.aggregation_policy,
            "d03_ablation_order": list(context.d03_ablation_keys),
            "d03_thresholds": [item.to_dict() for item in context.d03_thresholds],
            "dimension_key_map": [list(item) for item in context.dimension_key_map],
            "evaluation_order": list(W03_EVALUATION_ORDER),
            "generation_hard_conjunct": context.generation_hard_conjunct,
            "threshold": threshold,
        },
        "execution_state": dict(W03_ZERO_EXECUTION_STATE),
        "formal_w03_training_runs": 0,
        "format_version": 1,
        "payload_audit": {
            "learning_writes": 0,
            "payload_bytes": 0,
            "payload_gets": 0,
            "teacher_calls": 0,
        },
        "publication_binding": _publication_payload(observation),
        "recovery_protocol": {
            "failure_points": list(context.failure_point_keys),
            "logical_shard_count": context.logical_shard_count,
            "modes": list(W03_ALLOWED_MODES),
            "worker_counts": list(W03_ALLOWED_WORKER_COUNTS),
        },
        "remote_commit_sha1": remote,
        "resource_budget": dict(W03_RESOURCE_BUDGET),
        "self_excluded": 1,
        "test_inventory": _inventory(repository, W03_CANDIDATE_TEST_PATHS),
        "visibility_counts": {
            "candidate": len(context.candidate_payload_bindings),
            "evaluator": len(context.evaluator_visible_bindings),
            "teacher": (
                len(context.candidate_payload_bindings)
                + len(context.teacher_evidence_bindings)
            ),
        },
        "w02_binding": _w02_payload(continuity),
    }


def w03_candidate_contract_key(value: dict[str, object]) -> tuple[int, ...]:
    """把完整候选合同摘要为稳定整数键。"""
    _validate_contract(value)
    return digest_value(value)


def _validate_inventory(
        inventory: object,
        expected_paths: tuple[str, ...],
        *,
        label: str,
        ) -> None:
    """核验冻结 inventory 的路径顺序、size 和 SHA 字段。"""
    if (not isinstance(inventory, list)
            or tuple(item.get("path") for item in inventory
                     if isinstance(item, dict)) != expected_paths
            or len(inventory) != len(expected_paths)):
        raise RuntimeError(f"W-03 {label} inventory 路径漂移")
    for item in inventory:
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] <= 0):
            raise RuntimeError(f"W-03 {label} inventory 行非法")
        _strict_sha256(item["sha256"], label=f"{label} inventory")


def _validate_contract(value: object) -> dict[str, object]:
    """在写 freeze 前核验合同的承重状态和精确 inventory。"""
    if not isinstance(value, dict):
        raise RuntimeError("W-03 candidate 合同类型非法")
    if (value.get("artifact_kind") != W03_CANDIDATE_CONTRACT_KIND
            or value.get("format_version") != 1
            or value.get("self_excluded") != 1
            or value.get("formal_w03_training_runs") != 0
            or value.get("execution_state") != W03_ZERO_EXECUTION_STATE):
        raise RuntimeError("W-03 candidate 合同状态漂移")
    _validate_inventory(
        value.get("code_inventory"), W03_CANDIDATE_CODE_PATHS, label="code")
    _validate_inventory(
        value.get("test_inventory"), W03_CANDIDATE_TEST_PATHS, label="test")
    request = value.get("candidate_request")
    recovery = value.get("recovery_protocol")
    evaluation = value.get("evaluation_contract")
    if (not isinstance(request, dict)
            or request.get("run_id") != W03_FORMAL_RUN_ID
            or request.get("parent_run_id") != 3
            or request.get("base_run_id") != 3
            or request.get("worker_count") != W03_CANDIDATE_FORMAL_WORKER_COUNT
            or request.get("mode") != W03_CANDIDATE_FORMAL_MODE
            or request.get("candidate_payload_count") != 12
            or request.get("teacher_evidence_count") != 6):
        raise RuntimeError("W-03 candidate request 合同漂移")
    if (not isinstance(recovery, dict)
            or recovery.get("logical_shard_count") != 16
            or recovery.get("worker_counts") != list(W03_ALLOWED_WORKER_COUNTS)
            or recovery.get("modes") != list(W03_ALLOWED_MODES)
            or len(recovery.get("failure_points", ())) != 6):
        raise RuntimeError("W-03 candidate recovery 合同漂移")
    if (not isinstance(evaluation, dict)
            or evaluation.get("evaluation_order") != list(W03_EVALUATION_ORDER)
            or evaluation.get("aggregation_policy") != W03_AGGREGATION_POLICY
            or evaluation.get("threshold") != {
                "fail_allowed": 0,
                "ne_policy": "BLOCK",
                "required_pass_denominator": 1,
                "required_pass_numerator": 1,
            }):
        raise RuntimeError("W-03 candidate evaluation 合同漂移")
    return value


def _verify_repository_inventory(
        repository: Path,
        contract: dict[str, object],
        ) -> None:
    """在 guard 前和 host freeze 前重新回验全部能力代码与测试字节。"""
    if (contract["code_inventory"] != _inventory(
            repository, W03_CANDIDATE_CODE_PATHS)
            or contract["test_inventory"] != _inventory(
                repository, W03_CANDIDATE_TEST_PATHS)):
        raise RuntimeError("W-03 candidate capability identity 漂移")


def _validate_external_root(
        repository_root: Path,
        w02_artifacts_root: Path,
        artifact_root: Path,
        ) -> None:
    """要求 W-03 candidate root 与 Git/W-02 双向无包含。"""
    for forbidden in (repository_root, w02_artifacts_root):
        if (artifact_root == forbidden
                or artifact_root.is_relative_to(forbidden)
                or forbidden.is_relative_to(artifact_root)):
            raise RuntimeError("W-03 candidate 必须位于 Git 外并与 W-02 物理隔离")


def publish_w03_candidate_contract_freeze(
        repository_root: str | Path,
        w02_artifacts_root: str | Path,
        artifact_root: str | Path,
        contract: dict[str, object],
        ) -> tuple[Path, str]:
    """在 Git 外新 root 以排他写发布 run-count=0 的候选合同。"""
    repository = Path(repository_root).resolve()
    w02_root = Path(w02_artifacts_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, w02_root, root)
    value = _validate_contract(contract)
    _verify_repository_inventory(repository, value)
    root.mkdir(parents=True, exist_ok=True)
    target = root / W03_CANDIDATE_CONTRACT_FREEZE_NAME
    payload = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-03 candidate contract freeze 不可覆盖") from exc
    return target, _sha256_bytes(payload)


def verify_w03_candidate_contract_freeze(
        freeze_path: str | Path,
        contract: dict[str, object],
        ) -> str:
    """逐字节回验已发布合同，拒绝非规范 JSON 或任一字段漂移。"""
    value = _validate_contract(contract)
    path = Path(freeze_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W-03 candidate contract freeze identity 缺失")
    actual = path.read_bytes()
    expected = canonical_json_bytes(value)
    if actual != expected:
        raise RuntimeError("W-03 candidate contract freeze identity 漂移")
    return _sha256_bytes(actual)


def consume_w03_candidate_first_run_guard(
        artifact_root: str | Path,
        *,
        candidate_contract_sha256: str,
        ) -> tuple[Path, str]:
    """在正式 payload 前排他消费唯一 run 4，失败后也禁止同 candidate 重跑。"""
    root = Path(artifact_root).resolve()
    expected_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract SHA-256")
    freeze = root / W03_CANDIDATE_CONTRACT_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or _sha256_bytes(freeze.read_bytes()) != expected_sha):
        raise RuntimeError("W-03 candidate contract SHA-256 identity 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": W03_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": expected_sha,
        "execution_state_after_start": dict(W03_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "run_id": W03_FORMAL_RUN_ID,
    })
    target = root / W03_CANDIDATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-03 candidate first-run 已经消费，不可重跑") from exc
    return target, _sha256_bytes(payload)


def formalize_w03_candidate_outcome(outcome: W03RunOutcome) -> W03RunOutcome:
    """仅在 first-run 已消费后把正式运行状态附到完整 runtime outcome。"""
    if not isinstance(outcome, W03RunOutcome):
        raise TypeError("W-03 candidate outcome 类型非法")
    if outcome.execution_state != W03_ZERO_EXECUTION_STATE:
        raise RuntimeError("W-03 runtime pre-formal execution state 漂移")
    return replace(outcome, execution_state=dict(W03_FORMAL_EXECUTION_STATE))


def _outcome_evidence(outcome: W03RunOutcome) -> dict[str, object]:
    """形成 host/dump 共用的完整逻辑、资源和 owner 证据。"""
    return {
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "host_digests": {
            "artifact": outcome.artifact_digest,
            "candidate_history": outcome.candidate_history_digest,
            "cursor": outcome.cursor_digest,
            "generation": outcome.generation_digest,
            "logical": outcome.logical_state_digest,
            "projection": outcome.projection_digest,
            "retention": outcome.retention_digest,
        },
        "new_learning_write_count": outcome.new_learning_write_count,
        "owned_tables": list(outcome.owned_tables),
        "publication_counts": {
            "adopted_manifest_count": outcome.adopted_manifest_count,
            "merge_publication_count": outcome.merge_publication_count,
            "transaction_event_count": outcome.transaction_event_count,
        },
        "resource_actual": {
            key.removeprefix("actual_"): value
            for key, value in sorted(outcome.resource_report.items())
            if key.startswith("actual_")
        },
        "resource_report": dict(sorted(outcome.resource_report.items())),
        "w02_host_write_count": outcome.w02_host_write_count,
        "w02_retention_passed": int(outcome.w02_retention_passed),
    }


def _logical_outcome_key(outcome: W03RunOutcome) -> tuple[int, ...]:
    """排除 readback 物理路径，比较正式 host 与 fresh dump 的规范状态。"""
    evidence = _outcome_evidence(outcome)
    return digest_value({
        "artifact_counts": evidence["artifact_counts"],
        "execution_state": evidence["execution_state"],
        "host_digests": evidence["host_digests"],
        "w02_host_write_count": evidence["w02_host_write_count"],
        "w02_retention_passed": evidence["w02_retention_passed"],
    })


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    """封存 candidate root 中除 host freeze 自身外的全部文件。"""
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W03_CANDIDATE_HOST_FREEZE_NAME:
            continue
        payload = path.read_bytes()
        result.append({
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    if not result:
        raise RuntimeError("W-03 candidate host 没有可冻结 artifact")
    return result


def publish_w03_candidate_host_freeze(
        repository_root: str | Path,
        w02_artifacts_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W03RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        outcome: W03RunOutcome,
        dump_readback: W03RunOutcome,
        ) -> tuple[Path, str]:
    """正式 run 4 后封存 host、artifact、资源、owner 写和 fresh dump 证据。"""
    repository = Path(repository_root).resolve()
    w02_root = Path(w02_artifacts_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, w02_root, root)
    value = _validate_contract(contract)
    _verify_repository_inventory(repository, value)
    freeze_path = root / W03_CANDIDATE_CONTRACT_FREEZE_NAME
    actual_contract_sha = verify_w03_candidate_contract_freeze(
        freeze_path, value)
    expected_contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract SHA-256")
    if actual_contract_sha != expected_contract_sha:
        raise RuntimeError("W-03 candidate contract identity 漂移")
    guard = root / W03_CANDIDATE_FIRST_RUN_GUARD_NAME
    if not guard.is_file() or guard.is_symlink():
        raise RuntimeError("W-03 candidate first-run guard 缺失")
    if not isinstance(config, W03RuntimeConfig):
        raise TypeError("W-03 candidate config 类型非法")
    request = value["candidate_request"]
    if (config.run_id != request["run_id"]
            or config.parent_run_id != request["parent_run_id"]
            or config.base_run_id != request["base_run_id"]
            or list(config.base_fence_key) != request["base_fence_key"]
            or config.worker_count != request["worker_count"]
            or config.mode != request["mode"]
            or config.current_remote_commit_sha1 != value["remote_commit_sha1"]):
        raise RuntimeError("W-03 formal config 与 candidate request 漂移")
    if (outcome.execution_state != W03_FORMAL_EXECUTION_STATE
            or dump_readback.execution_state != W03_FORMAL_EXECUTION_STATE
            or outcome.dump_readback
            or not dump_readback.dump_readback
            or outcome.new_learning_write_count <= 0
            or dump_readback.new_learning_write_count != 0
            or outcome.w02_host_write_count != 0
            or dump_readback.w02_host_write_count != 0
            or not outcome.w02_retention_passed
            or not dump_readback.w02_retention_passed
            or outcome.resource_report.get("teacher_calls") != 0):
        raise RuntimeError("W-03 formal host/dump 状态、零写或 retention 未闭合")
    if _logical_outcome_key(outcome) != _logical_outcome_key(dump_readback):
        raise RuntimeError("W-03 formal host 与 fresh dump readback 漂移")
    artifact_counts = dict(outcome.artifact_counts)
    expected_counts = {
        "EVIDENCE_ACCOUNT": 64,
        "GENERATION_CHOICE": 2,
        "GENERATION_DECISION": 3,
        "GENERATION_OUTCOME": 4,
        "GENERATION_USE": 3,
        "PROJECTION": 59,
        "TRAIN_ENVELOPE": 163,
        "W02_RETENTION": 1,
    }
    if artifact_counts != expected_counts:
        raise RuntimeError("W-03 formal artifact count 漂移")
    payload = canonical_json_bytes({
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W03_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_key": list(w03_candidate_contract_key(value)),
        "candidate_contract_sha256": expected_contract_sha,
        "code_inventory": value["code_inventory"],
        "dump_readback_evidence": _outcome_evidence(dump_readback),
        "execution_state": dict(W03_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": _outcome_evidence(outcome),
        "owner_write_counts": {
            "artifact_writes": sum(artifact_counts.values()),
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": dump_readback.new_learning_write_count,
            "teacher_calls": outcome.resource_report["teacher_calls"],
            "w02_host_writes": outcome.w02_host_write_count,
        },
        "remote_commit_sha1": value["remote_commit_sha1"],
        "request": value["candidate_request"],
        "self_excluded": 1,
        "test_inventory": value["test_inventory"],
        "w02_binding": value["w02_binding"],
    })
    target = root / W03_CANDIDATE_HOST_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-03 candidate host freeze 不可覆盖") from exc
    return target, _sha256_bytes(payload)


def execute_w03_candidate_once(
        repository_root: str | Path,
        w02_artifacts_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W03RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        dump_readback_sqlite_path: str | Path,
        ) -> tuple[W03RunOutcome, W03RunOutcome, Path, str, Path, str]:
    """消费唯一 guard，执行一次 run 4，fresh 回读并发布不可覆盖 host freeze。"""
    repository = Path(repository_root).resolve()
    w02_root = Path(w02_artifacts_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, w02_root, root)
    value = _validate_contract(contract)
    _verify_repository_inventory(repository, value)
    verify_w03_candidate_contract_freeze(
        root / W03_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    if not isinstance(config, W03RuntimeConfig):
        raise TypeError("W-03 candidate config 类型非法")
    sqlite_path = Path(config.sqlite_path).resolve()
    run_root = Path(config.run_root).resolve()
    readback_path = Path(dump_readback_sqlite_path).resolve()
    if (not sqlite_path.is_relative_to(root)
            or not run_root.is_relative_to(root)
            or not readback_path.is_relative_to(root)
            or readback_path == sqlite_path):
        raise RuntimeError("W-03 formal host/run/readback 必须位于独立 candidate root")
    guard_path, guard_sha = consume_w03_candidate_first_run_guard(
        root,
        candidate_contract_sha256=candidate_contract_sha256,
    )
    raw_outcome = run_language_stage2(config)
    raw_readback = load_w03_candidate_dump(
        replace(config, mode="resume"),
        target_sqlite_path=readback_path,
    )
    outcome = formalize_w03_candidate_outcome(raw_outcome)
    readback = formalize_w03_candidate_outcome(raw_readback)
    freeze_path, freeze_sha = publish_w03_candidate_host_freeze(
        repository,
        w02_root,
        root,
        config=config,
        contract=value,
        candidate_contract_sha256=candidate_contract_sha256,
        outcome=outcome,
        dump_readback=readback,
    )
    return (
        outcome, readback, freeze_path, freeze_sha, guard_path, guard_sha)


__all__ = [
    "W03_CANDIDATE_CODE_PATHS",
    "W03_CANDIDATE_CONTRACT_FREEZE_NAME",
    "W03_CANDIDATE_FIRST_RUN_GUARD_NAME",
    "W03_CANDIDATE_HOST_FREEZE_NAME",
    "W03_CANDIDATE_TEST_PATHS",
    "W03_FORMAL_EXECUTION_STATE",
    "build_w03_candidate_contract",
    "consume_w03_candidate_first_run_guard",
    "execute_w03_candidate_once",
    "formalize_w03_candidate_outcome",
    "publish_w03_candidate_contract_freeze",
    "publish_w03_candidate_host_freeze",
    "verify_w03_candidate_contract_freeze",
    "w03_candidate_contract_key",
]
