"""PH2 W-06 candidate 合同、唯一正式运行守卫与 host 封存。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_ABLATION_KEYS,
    W06_AGGREGATION_POLICY,
    W06_ALLOWED_MODES,
    W06_ALLOWED_WORKER_COUNTS,
    W06_DIMENSION_KEYS,
    W06_EVALUATION_ORDER,
    W06_FORMAL_RUN_ID,
    W06_OPEN_GENERATION_STATE,
    W06_PRIVATE_ABLATION_KEYS,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_W05_BASE_RUN_ID,
    W06_ZERO_EXECUTION_STATE,
    W06RunRequest,
    open_w06_frozen_context,
    validate_w06_request,
)
from pure_integer_ai.experiments.ph2_w06_runtime import (
    W06RunOutcome,
    W06RuntimeConfig,
    load_w06_public_dump,
    run_language_stage6_public,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_PROFILES,
    W06_RELATION_SUBSTAGE_ORDER,
)


W06_CANDIDATE_CONTRACT_KIND = "PH2_W06_CANDIDATE_CONTRACT_FREEZE"
W06_CANDIDATE_HOST_FREEZE_KIND = "PH2_W06_CANDIDATE_HOST_FREEZE"
W06_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W06_CANDIDATE_FIRST_RUN_GUARD"
W06_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W06_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W06_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W06_CANDIDATE_FORMAL_WORKER_COUNT = 4
W06_CANDIDATE_FORMAL_MODE = "fresh"
W06_FORMAL_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W06_STARTED": 1,
    "W07_STARTED": 0,
    "formal_w06_training_runs": 1,
    "teacher_calls": 0,
}

W06_EXPECTED_COUNTS = {
    "active_candidate_count": 17,
    "candidate_count": 50,
    "carrier_count": 9,
    "evidence_account_count": 64,
    "logical_shard_count": 16,
    "relation_family_count": 14,
    "relation_scope_cell_count": 27,
    "schema_rejection_count": 1,
    "substage_count": 7,
    "transaction_event_count": 5,
}
W06_EXPECTED_DIGESTS = {
    "active_projection": (
        "53dbc65fb63819a593ea15da4ba41d7bb227fc680db6bb7fded94babc1be1804"
    ),
    "candidate": (
        "7841573b0c24113bb5e4e3204f8aff38142fabfa4dbc0ddc698b2c1e7ed951e2"
    ),
    "carrier_scope": (
        "6255fa146963502ad36cecff43a8689d9a11b94cad6623d50722b336ede16883"
    ),
    "logical": (
        "f020608dc1a8edc92c5fe42e982bbe28ca8d3400734ecba7ba59f6d5c345306c"
    ),
    "relation": (
        "aae08dbf5de3e7a5f419c84077591fdee5d09ca0720b40c15561f8ca2d53667b"
    ),
    "source_evidence": (
        "107cb862cde2a4561be6d5b8e6feec6bad8933351f68870bc1eb86b45a3e5828"
    ),
}
W06_CASE_FAMILIES = (
    ("PURE_ALIAS_REFERS", 5, 2),
    ("SUBSET_MEMBER", 5, 2),
    ("PROPERTY", 7, 2),
    ("MEREOLOGY", 7, 2),
    ("SIMILAR_ANTONYM", 7, 3),
    ("PRECEDES", 9, 3),
    ("CAUSES", 10, 3),
)

W06_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w06_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w06_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w06_carrier_scope.py",
    "src/pure_integer_ai/experiments/ph2_w06_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_faults.py",
    "src/pure_integer_ai/experiments/ph2_w06_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w06_learning.py",
    "src/pure_integer_ai/experiments/ph2_w06_payload.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r01_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02_endpoint_projection.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02_query.py",
    "src/pure_integer_ai/experiments/ph2_w06_r02_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03_reasoning.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r03_understanding.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04_endpoint_projection.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04_query.py",
    "src/pure_integer_ai/experiments/ph2_w06_r04_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r05.py",
    "src/pure_integer_ai/experiments/ph2_w06_r05_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r05_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r05_query.py",
    "src/pure_integer_ai/experiments/ph2_w06_r05_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06_endpoint_projection.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06_query.py",
    "src/pure_integer_ai/experiments/ph2_w06_r06_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07_contract.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07_endpoint_projection.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07_generation.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07_query.py",
    "src/pure_integer_ai/experiments/ph2_w06_r07_shared.py",
    "src/pure_integer_ai/experiments/ph2_w06_registry.py",
    "src/pure_integer_ai/experiments/ph2_w06_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w06_source_semantic.py",
    "src/pure_integer_ai/experiments/ph2_w06_source_semantic_overlay.py",
    "src/pure_integer_ai/experiments/ph2_w06_transaction.py",
)
W06_CANDIDATE_TEST_PATHS = (
    "tests/test_w06_r01.py",
    "tests/test_w06_r02.py",
    "tests/test_w06_r03.py",
    "tests/test_w06_r04.py",
    "tests/test_w06_r05.py",
    "tests/test_w06_r06.py",
    "tests/test_w06_r07.py",
    "tests/test_w06_source_semantic.py",
    "tests/test_w06_source_semantic_overlay.py",
    "tests/test_w06_stage2_adapter.py",
    "tests/test_w06_stage2_candidate.py",
    "tests/test_w06_stage2_contract.py",
    "tests/test_w06_stage2_learning.py",
    "tests/test_w06_stage2_registry.py",
    "tests/test_w06_stage2_runtime.py",
)
W06_CANDIDATE_PUBLIC_ARTIFACT_PATHS = (
    "data/ph2/manifests/d03_v1/stages/w06_stage_manifest_v1.json",
    "data/ph2/manifests/w06_r02_endpoint_projection_v1.json",
    "data/ph2/manifests/w06_r04_endpoint_projection_v1.json",
    "data/ph2/manifests/w06_r06_endpoint_projection_v1.json",
    "data/ph2/manifests/w06_r07_endpoint_projection_v1.json",
    "data/ph2/manifests/w06_source_semantic_overlay_v1.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise RuntimeError(f"W-06 {label} 不是规范 SHA-256")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    """要求路径是无逃逸的规范 POSIX 相对路径。"""
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} 不是非空路径")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or ":" in value or path.as_posix() != value or "//" in value):
        raise RuntimeError(f"{label} 不是规范安全路径")
    return value


def _inventory(repository: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
    result = []
    for relative in paths:
        normalized = _safe_relative_path(relative, label="W-06 inventory path")
        target = (repository / Path(*PurePosixPath(normalized).parts)).resolve()
        if (not target.is_file() or target.is_symlink()
                or not target.is_relative_to(repository)):
            raise RuntimeError("W-06 candidate inventory 缺失、逃逸或为链接")
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
        raise RuntimeError("W-06 candidate root 必须与 Git root 物理隔离")


def _relation_profiles() -> list[dict[str, object]]:
    order = {key: index for index, key in enumerate(W06_RELATION_SUBSTAGE_ORDER)}
    result = []
    for family, profile in sorted(
            W06_RELATION_PROFILES.items(),
            key=lambda item: (order[item[1].substage_key], item[0])):
        result.append({
            "closure_policy": profile.closure_policy,
            "directionality": profile.directionality,
            "family": family,
            "relation_kind": profile.relation_kind,
            "roles": [
                {
                    "allowed_object_kinds": sorted(allowed),
                    "role_kind": role,
                }
                for role, allowed in profile.role_object_kinds
            ],
            "substage": profile.substage_key,
        })
    if len(result) != 14:
        raise RuntimeError("W-06 relation profile 数量漂移")
    return result


def build_w06_candidate_contract(
        repository_root: str | Path,
        *,
        backend_profile_key: tuple[int, ...],
        current_remote_commit_sha1: str,
        ) -> dict[str, object]:
    """在 payload 读取和 first-run guard 前冻结 W-06 candidate 全合同。"""
    repository = Path(repository_root).resolve()
    context = open_w06_frozen_context(
        repository,
        current_remote_commit_sha1=current_remote_commit_sha1,
        backend_profile_key=backend_profile_key,
    )
    request = validate_w06_request(context, W06RunRequest(
        run_id=context.run_id,
        parent_run_id=context.parent_run_id,
        base_run_id=context.base_run_id,
        stage_key=W06_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W06_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        source_overlay_sha256=context.source_overlay_sha256,
        context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=W06_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W06_CANDIDATE_FORMAL_MODE,
        resource_budget=tuple(sorted(W06_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    ))
    return {
        "aggregation_policy": W06_AGGREGATION_POLICY,
        "artifact_kind": W06_CANDIDATE_CONTRACT_KIND,
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
            "source_overlay_sha256": request.source_overlay_sha256,
            "teacher_evidence_count": len(request.teacher_evidence_paths),
            "teacher_evidence_paths": list(request.teacher_evidence_paths),
            "worker_count": request.worker_count,
        },
        "code_inventory": _inventory(repository, W06_CANDIDATE_CODE_PATHS),
        "evaluation_contract": {
            "ablation_order": list(W06_ABLATION_KEYS),
            "aggregation_policy": W06_AGGREGATION_POLICY,
            "dimension_order": list(W06_DIMENSION_KEYS),
            "evaluation_order": list(W06_EVALUATION_ORDER),
            "formal_ablation_order": list(W06_PRIVATE_ABLATION_KEYS),
            "generation_hard_conjunct": W06_GENERATION_HARD_CONJUNCT,
        },
        "execution_state": dict(W06_ZERO_EXECUTION_STATE),
        "expected_counts": dict(W06_EXPECTED_COUNTS),
        "expected_digests": dict(W06_EXPECTED_DIGESTS),
        "formal_w06_training_runs": 0,
        "format_version": 1,
        "open_generation_state": W06_OPEN_GENERATION_STATE,
        "payload_audit": {
            "learning_writes": 0,
            "payload_bytes": 0,
            "payload_gets": 0,
            "teacher_calls": 0,
        },
        "public_artifact_inventory": _inventory(
            repository, W06_CANDIDATE_PUBLIC_ARTIFACT_PATHS),
        "recovery_protocol": {
            "failure_points": list(context.failure_point_keys),
            "logical_shard_count": context.logical_shard_count,
            "modes": list(W06_ALLOWED_MODES),
            "worker_counts": list(W06_ALLOWED_WORKER_COUNTS),
        },
        "relation_contract": {
            "case_families": [list(item) for item in W06_CASE_FAMILIES],
            "profiles": _relation_profiles(),
            "substage_order": list(W06_RELATION_SUBSTAGE_ORDER),
        },
        "remote_commit_sha1": context.current_remote_commit_sha1,
        "resource_budget": dict(W06_RESOURCE_BUDGET),
        "self_excluded": 1,
        "source_contract": {
            "effective_train_pack_keys": list(context.effective_train_pack_keys),
            "pack_bindings": [item.to_dict() for item in context.pack_bindings],
            "parent_sha256": [list(item) for item in context.parent_sha256],
            "stage_train_pack_keys": list(context.stage_train_pack_keys),
        },
        "test_inventory": _inventory(repository, W06_CANDIDATE_TEST_PATHS),
        "visibility_counts": {
            "candidate_payloads": len(context.candidate_payload_bindings),
            "evaluator_visible": len(context.evaluator_visible_bindings),
            "teacher_evidence": len(context.teacher_evidence_bindings),
            "train_pack_count": len(context.effective_train_pack_keys),
        },
    }


def _validate_inventory(value: object, expected: tuple[str, ...], label: str) -> None:
    if (not isinstance(value, list)
            or tuple(item.get("path") for item in value
                     if isinstance(item, dict)) != expected
            or len(value) != len(expected)):
        raise RuntimeError(f"W-06 {label} inventory 路径漂移")
    for item in value:
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] <= 0):
            raise RuntimeError(f"W-06 {label} inventory 字段非法")
        _strict_sha256(item["sha256"], label=f"{label} inventory")


def _validate_contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("W-06 candidate 合同类型非法")
    if (value.get("artifact_kind") != W06_CANDIDATE_CONTRACT_KIND
            or value.get("format_version") != 1
            or value.get("self_excluded") != 1
            or value.get("execution_state") != W06_ZERO_EXECUTION_STATE
            or value.get("formal_w06_training_runs") != 0
            or value.get("open_generation_state") != W06_OPEN_GENERATION_STATE):
        raise RuntimeError("W-06 candidate 合同状态漂移")
    _validate_inventory(value.get("code_inventory"),
                        W06_CANDIDATE_CODE_PATHS, "code")
    _validate_inventory(value.get("test_inventory"),
                        W06_CANDIDATE_TEST_PATHS, "test")
    _validate_inventory(value.get("public_artifact_inventory"),
                        W06_CANDIDATE_PUBLIC_ARTIFACT_PATHS, "artifact")
    request = value.get("candidate_request")
    if (not isinstance(request, dict)
            or request.get("run_id") != W06_FORMAL_RUN_ID
            or request.get("parent_run_id") != W06_W05_BASE_RUN_ID
            or request.get("base_run_id") != W06_W05_BASE_RUN_ID
            or request.get("worker_count") != W06_CANDIDATE_FORMAL_WORKER_COUNT
            or request.get("mode") != W06_CANDIDATE_FORMAL_MODE):
        raise RuntimeError("W-06 candidate request 合同漂移")
    recovery = value.get("recovery_protocol")
    if (not isinstance(recovery, dict)
            or recovery.get("logical_shard_count") != 16
            or tuple(recovery.get("worker_counts", ()))
            != W06_ALLOWED_WORKER_COUNTS
            or tuple(recovery.get("modes", ())) != W06_ALLOWED_MODES
            or len(recovery.get("failure_points", ())) != 6):
        raise RuntimeError("W-06 recovery 合同漂移")
    evaluation = value.get("evaluation_contract")
    if (not isinstance(evaluation, dict)
            or tuple(evaluation.get("dimension_order", ()))
            != W06_DIMENSION_KEYS
            or tuple(evaluation.get("ablation_order", ())) != W06_ABLATION_KEYS
            or tuple(evaluation.get("formal_ablation_order", ()))
            != W06_PRIVATE_ABLATION_KEYS
            or tuple(evaluation.get("evaluation_order", ()))
            != W06_EVALUATION_ORDER
            or evaluation.get("generation_hard_conjunct")
            != W06_GENERATION_HARD_CONJUNCT
            or evaluation.get("aggregation_policy") != W06_AGGREGATION_POLICY):
        raise RuntimeError("W-06 evaluation 合同漂移")
    relation = value.get("relation_contract")
    if (not isinstance(relation, dict)
            or tuple(relation.get("substage_order", ()))
            != W06_RELATION_SUBSTAGE_ORDER
            or tuple(tuple(item) for item in relation.get("case_families", ()))
            != W06_CASE_FAMILIES
            or relation.get("profiles") != _relation_profiles()):
        raise RuntimeError("W-06 relation/case/profile 合同漂移")
    if (value.get("expected_counts") != W06_EXPECTED_COUNTS
            or value.get("expected_digests") != W06_EXPECTED_DIGESTS
            or value.get("resource_budget") != W06_RESOURCE_BUDGET):
        raise RuntimeError("W-06 expected/resource 合同漂移")
    return value


def w06_candidate_contract_key(value: dict[str, object]) -> tuple[int, ...]:
    _validate_contract(value)
    return digest_value(value)


def _verify_inventory(repository: Path, contract: dict[str, object]) -> None:
    if (contract["code_inventory"] != _inventory(
            repository, W06_CANDIDATE_CODE_PATHS)
            or contract["test_inventory"] != _inventory(
                repository, W06_CANDIDATE_TEST_PATHS)
            or contract["public_artifact_inventory"] != _inventory(
                repository, W06_CANDIDATE_PUBLIC_ARTIFACT_PATHS)):
        raise RuntimeError("W-06 candidate code/test/artifact identity 漂移")


def publish_w06_candidate_contract_freeze(
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
    target = root / W06_CANDIDATE_CONTRACT_FREEZE_NAME
    encoded = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise RuntimeError("W-06 candidate contract freeze 不可覆盖") from error
    return target, _sha256_bytes(encoded)


def verify_w06_candidate_contract_freeze(
        freeze_path: str | Path,
        contract: dict[str, object],
        ) -> str:
    value = _validate_contract(contract)
    path = Path(freeze_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W-06 candidate contract freeze 缺失")
    encoded = path.read_bytes()
    if encoded != canonical_json_bytes(value):
        raise RuntimeError("W-06 candidate contract identity 漂移")
    return _sha256_bytes(encoded)


def consume_w06_candidate_first_run_guard(
        artifact_root: str | Path,
        *,
        candidate_contract_sha256: str,
        ) -> tuple[Path, str]:
    """排他创建 first-run guard，并把正式运行计数从 0 推到 1。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(candidate_contract_sha256,
                              label="candidate contract")
    freeze = root / W06_CANDIDATE_CONTRACT_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or _sha256_bytes(freeze.read_bytes()) != expected):
        raise RuntimeError("W-06 candidate contract SHA 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": W06_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": expected,
        "execution_state_after_start": dict(W06_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "run_id": W06_FORMAL_RUN_ID,
    })
    target = root / W06_CANDIDATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError("W-06 candidate first-run 已消费，不可重跑") from error
    return target, _sha256_bytes(payload)


def formalize_w06_candidate_outcome(outcome: W06RunOutcome) -> W06RunOutcome:
    if not isinstance(outcome, W06RunOutcome):
        raise TypeError("W-06 outcome 类型非法")
    if outcome.execution_state != W06_ZERO_EXECUTION_STATE:
        raise RuntimeError("W-06 pre-formal execution state 漂移")
    return replace(outcome, execution_state=dict(W06_FORMAL_EXECUTION_STATE))


def _outcome_evidence(outcome: W06RunOutcome) -> dict[str, object]:
    return {
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "host_digests": {
            "active_projection": outcome.active_projection_digest,
            "candidate": outcome.candidate_digest,
            "carrier_scope": outcome.carrier_scope_digest,
            "logical": outcome.logical_state_digest,
            "relation": outcome.relation_digest,
            "source_evidence": outcome.source_evidence_digest,
            "transaction": outcome.transaction_digest,
        },
        "learning_attempt_count": outcome.learning_attempt_count,
        "new_learning_write_count": outcome.new_learning_write_count,
        "owned_tables": list(outcome.owned_tables),
        "payload_bytes_this_call": outcome.payload_bytes_this_call,
        "payload_gets_this_call": outcome.payload_gets_this_call,
        "resource_report": dict(sorted(outcome.resource_report.items())),
        "retention_sha256": [list(item) for item in outcome.retention_sha256],
        "teacher_calls": outcome.teacher_calls,
        "transaction_event_count": outcome.transaction_event_count,
    }


def _logical_outcome_key(outcome: W06RunOutcome) -> tuple[int, ...]:
    return digest_value({
        "active_projection": outcome.active_projection_digest,
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "candidate": outcome.candidate_digest,
        "carrier_scope": outcome.carrier_scope_digest,
        "logical": outcome.logical_state_digest,
        "relation": outcome.relation_digest,
        "relation_summaries": [list(item)
                               for item in outcome.relation_summary_digests],
        "retention": [list(item) for item in outcome.retention_sha256],
        "source_evidence": outcome.source_evidence_digest,
        "transaction": outcome.transaction_digest,
    })


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W06_CANDIDATE_HOST_FREEZE_NAME:
            continue
        payload = path.read_bytes()
        result.append({
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    if not result:
        raise RuntimeError("W-06 candidate root 没有可封存 artifact")
    return result


def _validate_formal_outcome(
        outcome: W06RunOutcome,
        readback: W06RunOutcome,
        ) -> None:
    counts = dict(outcome.artifact_counts)
    if (outcome.execution_state != W06_FORMAL_EXECUTION_STATE
            or readback.execution_state != W06_FORMAL_EXECUTION_STATE
            or outcome.dump_readback or not readback.dump_readback
            or outcome.candidate_count != W06_EXPECTED_COUNTS["candidate_count"]
            or outcome.active_candidate_count
            != W06_EXPECTED_COUNTS["active_candidate_count"]
            or counts.get("EVIDENCE_ACCOUNT")
            != W06_EXPECTED_COUNTS["evidence_account_count"]
            or counts.get("RELATION_FAMILY")
            != W06_EXPECTED_COUNTS["relation_family_count"]
            or counts.get("SCHEMA_REJECTION")
            != W06_EXPECTED_COUNTS["schema_rejection_count"]
            or counts.get("CARRIER_PROJECTION")
            != W06_EXPECTED_COUNTS["carrier_count"]
            or counts.get("RELATION_SCOPE_CELL")
            != W06_EXPECTED_COUNTS["relation_scope_cell_count"]
            or counts.get("LOGICAL_SHARD")
            != W06_EXPECTED_COUNTS["logical_shard_count"]
            or outcome.transaction_event_count != 5
            or readback.transaction_event_count != 5
            or outcome.new_learning_write_count <= 0
            or readback.new_learning_write_count != 0
            or outcome.payload_gets_this_call <= 0
            or outcome.payload_bytes_this_call <= 0
            or readback.payload_gets_this_call != 0
            or readback.payload_bytes_this_call != 0
            or outcome.teacher_calls != 0 or readback.teacher_calls != 0
            or outcome.open_generation_state != W06_OPEN_GENERATION_STATE
            or readback.open_generation_state != W06_OPEN_GENERATION_STATE
            or outcome.dump_manifest_sha256 != readback.dump_manifest_sha256
            or _logical_outcome_key(outcome) != _logical_outcome_key(readback)
            or {
                "active_projection": outcome.active_projection_digest,
                "candidate": outcome.candidate_digest,
                "carrier_scope": outcome.carrier_scope_digest,
                "logical": outcome.logical_state_digest,
                "relation": outcome.relation_digest,
                "source_evidence": outcome.source_evidence_digest,
            } != W06_EXPECTED_DIGESTS):
        raise RuntimeError("W-06 formal host/dump readback 未闭合")
    if len(outcome.retention_sha256) != 4:
        raise RuntimeError("W-06 formal retention 不完整")
    if any(item.startswith(("ph2_w02_", "ph2_w03_", "ph2_w04_", "ph2_w05_"))
           for item in outcome.owned_tables):
        raise RuntimeError("W-06 formal host 注册历史 owner table")
    for key, actual in outcome.resource_report.items():
        if key.startswith("actual_"):
            budget_key = "max_" + key.removeprefix("actual_")
            if actual > W06_RESOURCE_BUDGET.get(budget_key, actual):
                raise RuntimeError("W-06 formal resource 超预算")


def publish_w06_candidate_host_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W06RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        outcome: W06RunOutcome,
        dump_readback: W06RunOutcome,
        ) -> tuple[Path, str]:
    """闭合 formal host、零 transport readback、资源和 inventory 后封存。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    expected_contract = _strict_sha256(candidate_contract_sha256,
                                       label="candidate contract")
    actual_contract = verify_w06_candidate_contract_freeze(
        root / W06_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    if actual_contract != expected_contract:
        raise RuntimeError("W-06 candidate contract SHA 漂移")
    guard = root / W06_CANDIDATE_FIRST_RUN_GUARD_NAME
    if not guard.is_file() or guard.is_symlink():
        raise RuntimeError("W-06 first-run guard 缺失")
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
        raise RuntimeError("W-06 formal config 与 candidate 合同漂移")
    _validate_formal_outcome(outcome, dump_readback)
    artifact_counts = dict(outcome.artifact_counts)
    payload = canonical_json_bytes({
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W06_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_key": list(w06_candidate_contract_key(value)),
        "candidate_contract_sha256": expected_contract,
        "code_inventory": value["code_inventory"],
        "dump_readback_evidence": _outcome_evidence(dump_readback),
        "execution_state": dict(W06_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": _outcome_evidence(outcome),
        "open_generation_state": W06_OPEN_GENERATION_STATE,
        "owner_write_counts": {
            "artifact_writes": sum(artifact_counts.values()),
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": dump_readback.new_learning_write_count,
            "teacher_calls": 0,
        },
        "public_artifact_inventory": value["public_artifact_inventory"],
        "remote_commit_sha1": value["remote_commit_sha1"],
        "request": value["candidate_request"],
        "self_excluded": 1,
        "test_inventory": value["test_inventory"],
    })
    target = root / W06_CANDIDATE_HOST_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError("W-06 candidate host freeze 不可覆盖") from error
    return target, _sha256_bytes(payload)


def execute_w06_candidate_once(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W06RuntimeConfig,
        contract: dict[str, object],
        candidate_contract_sha256: str,
        dump_readback_sqlite_path: str | Path,
        ) -> tuple[W06RunOutcome, W06RunOutcome, Path, str, Path, str]:
    """消费唯一 guard，执行一次 candidate 并封存 host/readback。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    verify_w06_candidate_contract_freeze(
        root / W06_CANDIDATE_CONTRACT_FREEZE_NAME, value)
    sqlite_path = Path(config.sqlite_path).resolve()
    run_root = Path(config.run_root).resolve()
    readback_path = Path(dump_readback_sqlite_path).resolve()
    if (not sqlite_path.is_relative_to(root)
            or not run_root.is_relative_to(root)
            or not readback_path.is_relative_to(root)
            or readback_path == sqlite_path):
        raise RuntimeError("W-06 formal host/run/readback 必须位于 candidate root")
    guard_path, guard_sha = consume_w06_candidate_first_run_guard(
        root, candidate_contract_sha256=candidate_contract_sha256)
    raw = run_language_stage6_public(config)
    raw_readback = load_w06_public_dump(
        replace(config, sqlite_path=readback_path, fault_point=None))
    outcome = formalize_w06_candidate_outcome(raw)
    readback = formalize_w06_candidate_outcome(raw_readback)
    freeze_path, freeze_sha = publish_w06_candidate_host_freeze(
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
    "W06_CANDIDATE_CODE_PATHS",
    "W06_CANDIDATE_CONTRACT_FREEZE_NAME",
    "W06_CANDIDATE_CONTRACT_KIND",
    "W06_CANDIDATE_FIRST_RUN_GUARD_NAME",
    "W06_CANDIDATE_FIRST_RUN_GUARD_KIND",
    "W06_CANDIDATE_FORMAL_MODE",
    "W06_CANDIDATE_FORMAL_WORKER_COUNT",
    "W06_CANDIDATE_HOST_FREEZE_NAME",
    "W06_CANDIDATE_HOST_FREEZE_KIND",
    "W06_CANDIDATE_PUBLIC_ARTIFACT_PATHS",
    "W06_CANDIDATE_TEST_PATHS",
    "W06_CASE_FAMILIES",
    "W06_EXPECTED_COUNTS",
    "W06_EXPECTED_DIGESTS",
    "W06_FORMAL_EXECUTION_STATE",
    "build_w06_candidate_contract",
    "consume_w06_candidate_first_run_guard",
    "execute_w06_candidate_once",
    "formalize_w06_candidate_outcome",
    "publish_w06_candidate_contract_freeze",
    "publish_w06_candidate_host_freeze",
    "verify_w06_candidate_contract_freeze",
    "w06_candidate_contract_key",
]
