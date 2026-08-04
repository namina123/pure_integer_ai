"""W08-08 Candidate 的公开冻结合同与唯一 first-run guard。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_AUTHORITY_RELATIVE_PATH,
    W08_DIMENSION_KEYS,
    W08_PARENT_PATHS,
    W08_RETENTION_PATHS,
    read_w08_authority,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_ALLOWED_MODES,
    W08_ALLOWED_WORKER_COUNTS,
    W08_CARRIER_KEYS,
    W08_CONSUMER_KEYS,
    W08_FAILURE_POINT_KEYS,
    W08_LEARNING_PACK_KEYS,
    W08_RESOURCE_BUDGET,
    W08_ZERO_EXECUTION_STATE,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_lc16 import (
    W08_LC16_CELL_STATES,
    W08_LC16_EVALUATOR_KEY,
    W08_LC16_SCOPE_KEY,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INPUT_KIND,
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
    W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
    W08_INFERENCE_SHORTCUT_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08_OPEN_GENERATION_COVERAGE_KEYS,
    W08_OPEN_GENERATION_LAYER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_registry import (
    W08_DIMENSION_REGISTRY,
    W08_PACK_REGISTRY,
)
from pure_integer_ai.experiments.ph2_w08_runtime import (
    W08_STAGE6_VALIDATION_PATH,
    W08_STAGE6_VALIDATION_SHA256,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_FORMAL_EXECUTION_STATE,
    W08_FORMAL_RUN_ID,
    W08_OPEN_GENERATION_PREFORMAL_STATE,
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
    W08_W07_BASE_RUN_ID,
)


W08_CANDIDATE_CONTRACT_KIND = "PH2_W08_CANDIDATE_CONTRACT_FREEZE"
W08_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W08_CANDIDATE_FIRST_RUN_GUARD"
W08_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W08_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W08_CANDIDATE_FORMAL_WORKER_COUNT = 4
W08_CANDIDATE_FORMAL_MODE = "fresh"
W08_CANDIDATE_CLUSTER_AXES = (
    "SOURCE",
    "SCHEMA",
    "CASE",
    "TEMPLATE",
    "CONTENT",
    "STRUCTURE",
    "COMBINATION",
)

W08_EXPECTED_COUNTS = {
    "compiled_artifact_count": 5,
    "hard_conjunct_count": 2,
    "inference_rule_count": 60,
    "logical_shard_count": 16,
    "retention_count": 6,
    "transaction_event_count": 5,
    "use_count": 15,
}
W08_EXPECTED_RECORD_COUNTS = (63, 63, 9, 59, 1)
W08_EXPECTED_EVIDENCE_COUNTS = (63, 59, 9, 59, 1)
W08_EXPECTED_COMMITMENTS = {
    "artifacts": (
        228, 132, 244, 155, 191, 253, 9, 46, 170, 164, 4, 143, 249, 87,
        112, 199, 182, 117, 36, 189, 71, 70, 49, 231, 94, 38, 107, 242,
        146, 178, 208, 178,
    ),
    "hard_conjuncts": (
        68, 121, 19, 25, 110, 76, 12, 239, 118, 242, 67, 166, 33, 143,
        52, 242, 86, 68, 2, 110, 33, 148, 35, 31, 183, 72, 230, 203,
        223, 199, 225, 255,
    ),
    "inference_state": (
        193, 124, 211, 52, 26, 32, 18, 26, 20, 94, 142, 153, 140,
        167, 205, 190, 49, 191, 239, 226, 222, 180, 184, 89, 219, 106,
        193, 114, 81, 15, 168, 51,
    ),
    "retention": (
        253, 237, 112, 184, 25, 152, 93, 141, 148, 37, 208, 187, 167,
        192, 220, 149, 169, 56, 200, 93, 160, 227, 6, 157, 128, 203, 14,
        230, 173, 115, 55, 178,
    ),
    "semantic_state": (
        78, 136, 85, 252, 212, 62, 233, 180, 78, 38, 210, 177, 171,
        144, 104, 83, 142, 97, 171, 243, 135, 165, 197, 36, 201, 236,
        79, 219, 83, 160, 106, 154,
    ),
    "uses": (
        143, 60, 157, 166, 27, 125, 142, 242, 4, 171, 254, 88, 31, 227,
        34, 201, 8, 207, 181, 104, 173, 241, 52, 219, 23, 222, 172, 100,
        168, 214, 80, 252,
    ),
}

W08_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w08_authority.py",
    "src/pure_integer_ai/experiments/ph2_w08_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w08_candidate_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_discourse.py",
    "src/pure_integer_ai/experiments/ph2_w08_discourse_adapters.py",
    "src/pure_integer_ai/experiments/ph2_w08_discourse_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_discourse_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_faults.py",
    "src/pure_integer_ai/experiments/ph2_w08_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w08_inference.py",
    "src/pure_integer_ai/experiments/ph2_w08_inference_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_inference_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_lc16.py",
    "src/pure_integer_ai/experiments/ph2_w08_long_context.py",
    "src/pure_integer_ai/experiments/ph2_w08_long_context_adapters.py",
    "src/pure_integer_ai/experiments/ph2_w08_long_context_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_long_context_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_open_generation.py",
    "src/pure_integer_ai/experiments/ph2_w08_open_generation_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_payload.py",
    "src/pure_integer_ai/experiments/ph2_w08_recompute.py",
    "src/pure_integer_ai/experiments/ph2_w08_recompute_adapters.py",
    "src/pure_integer_ai/experiments/ph2_w08_recompute_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_recompute_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_registry.py",
    "src/pure_integer_ai/experiments/ph2_w08_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w08_runtime_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_stage6.py",
    "src/pure_integer_ai/experiments/ph2_w08_transaction.py",
    "src/pure_integer_ai/experiments/ph2_w08_variation.py",
    "scripts/run_w08_08_candidate.py",
)
W08_CANDIDATE_TEST_PATHS = (
    "tests/test_w08_00_authority.py",
    "tests/test_w08_01_contract.py",
    "tests/test_w08_01_registry.py",
    "tests/test_w08_02_variation.py",
    "tests/test_w08_03_discourse.py",
    "tests/test_w08_04_recompute.py",
    "tests/test_w08_05_long_context.py",
    "tests/test_w08_06_stage6.py",
    "tests/test_w08_07_runtime.py",
    "tests/test_w08_08_candidate.py",
    "tests/test_w08_09_ne_recovery.py",
)
W08_CANDIDATE_PUBLIC_ARTIFACT_PATHS = tuple(dict.fromkeys((
    W08_AUTHORITY_RELATIVE_PATH,
    W08_STAGE6_VALIDATION_PATH,
    *W08_PARENT_PATHS,
)))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"W08 {label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"W08 {label} 不是规范 Git SHA-1")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"W08 {label} 不是非空路径")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or path.as_posix() != value
        or "//" in value
    ):
        raise RuntimeError(f"W08 {label} 不是规范安全路径")
    return value


def _git_sha(repository: Path, revision: str) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--verify", revision],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"W08 无法解析 Git revision: {revision}") from error
    return _strict_sha1(value, label=revision)


def _inventory(repository: Path, paths: tuple[str, ...]) -> list[dict[str, object]]:
    result = []
    for relative in paths:
        normalized = _safe_relative_path(relative, label="inventory path")
        target = (repository / Path(*PurePosixPath(normalized).parts)).resolve()
        if (
            not target.is_file()
            or target.is_symlink()
            or not target.is_relative_to(repository)
        ):
            raise RuntimeError("W08 Candidate inventory 缺失、逃逸或为链接")
        payload = target.read_bytes()
        result.append({
            "path": normalized,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        })
    return result


def _dimension_registry() -> list[dict[str, object]]:
    result = []
    for subtask, schema in W08_DIMENSION_REGISTRY.items():
        result.append({
            "ablation_key": schema.ablation_key,
            "allowed_write_owners": list(schema.allowed_write_owners),
            "consumer_keys": list(schema.consumer_keys),
            "dimension_key": schema.dimension_key,
            "outcome_states": list(schema.outcome_states),
            "request_kind": schema.request_kind,
            "resource_fields": list(schema.resource_fields),
            "result_kind": schema.result_kind,
            "subtask_key": subtask,
            "trace_fields": list(schema.trace_fields),
            "use_fields": list(schema.use_fields),
        })
    return result


def _pack_registry() -> list[dict[str, object]]:
    return [
        {
            "earliest_stage": values[2],
            "pack_key": pack_key,
            "payload_kind": values[0],
            "substage": values[1],
        }
        for pack_key, values in W08_PACK_REGISTRY.items()
    ]


def _binding_inventory(context) -> list[dict[str, object]]:
    result = []
    for binding in (*context.candidate_bindings, *context.teacher_bindings):
        result.append(binding.to_dict())
    return result


def _identity_commitment(bindings: tuple[object, ...]) -> tuple[int, ...]:
    return digest_value([binding.to_dict() for binding in bindings])


def _validate_external_root(repository: Path, artifact_root: Path) -> None:
    if (
        artifact_root == repository
        or artifact_root.is_relative_to(repository)
        or repository.is_relative_to(artifact_root)
    ):
        raise RuntimeError("W08 Candidate root 必须与 Git root 物理隔离")


def build_w08_candidate_contract(
    repository_root: str | Path,
    *,
    current_public_head_commit_sha1: str | None = None,
) -> dict[str, object]:
    """在 train payload transport 和 first-run guard 前构造完整冻结合同。"""
    repository = Path(repository_root).resolve()
    current_head = _git_sha(repository, "HEAD")
    supplied_head = _strict_sha1(
        current_public_head_commit_sha1 or current_head,
        label="public HEAD",
    )
    if supplied_head != current_head:
        raise RuntimeError("W08 Candidate public HEAD 参数漂移")
    context = open_w08_frozen_contract(repository)
    authority = read_w08_authority(repository)
    request = make_w08_request(
        context,
        worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W08_CANDIDATE_FORMAL_MODE,
    )
    if W08_STAGE6_VALIDATION_SHA256 != _sha256_bytes(
        (repository / W08_STAGE6_VALIDATION_PATH).read_bytes()
    ):
        raise RuntimeError("W08-06 validation identity 漂移")
    train_bindings = (*context.candidate_bindings, *context.teacher_bindings)
    return {
        "artifact_kind": W08_CANDIDATE_CONTRACT_KIND,
        "candidate_request": {
            "base_fence_key": list(request.base_fence_key),
            "base_run_id": W08_W07_BASE_RUN_ID,
            "candidate_payload_count": len(request.candidate_payload_paths),
            "candidate_payload_paths": list(request.candidate_payload_paths),
            "contract_key": list(request.contract_key),
            "mode": request.mode,
            "owner_key": request.owner_key,
            "parent_run_id": W08_W07_BASE_RUN_ID,
            "run_id": W08_FORMAL_RUN_ID,
            "teacher_evidence_count": len(request.teacher_evidence_paths),
            "teacher_evidence_paths": list(request.teacher_evidence_paths),
            "worker_count": request.worker_count,
        },
        "cluster_contract": {
            "axis_keys": list(W08_CANDIDATE_CLUSTER_AXES),
            "case_policy": "MECHANICAL_FROM_EXACT_TRAIN_OWNER",
            "combination_policy": "UNSEEN_CONTENT_AND_SURFACE_HARD_CONJUNCT",
            "content_policy": "TYPED_CLAIM_CONTENT_WITH_EXACT_EVIDENCE",
            "schema_registry": _dimension_registry(),
            "source_pack_registry": _pack_registry(),
            "structure_policy": "ORDER_REFERENCE_INFORMATION_REVISION",
            "template_policy": "COMPLETE_TEMPLATE_REPLAY_FORBIDDEN",
            "train_inventory_commitment_key": list(
                _identity_commitment(tuple(train_bindings))
            ),
        },
        "code_inventory": _inventory(repository, W08_CANDIDATE_CODE_PATHS),
        "evaluation_contract": {
            "ablation_order": list(W08_ABLATION_KEYS),
            "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
            "candidate_inference_interface": {
                "component_keys": list(W08_DIMENSION_KEYS),
                "evaluator_label_inputs": 0,
                "executable": 1,
                "input_kind": W08_CANDIDATE_INFERENCE_INPUT_KIND,
                "output_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
                "per_case_invocation_required": 1,
                "shortcut_account_keys": list(W08_INFERENCE_SHORTCUT_KEYS),
                "state_policy": "TRAIN_ONLY_TYPED_RULES",
                "version": W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
            },
            "dimension_order": list(W08_DIMENSION_KEYS),
            "lc16": {
                "carrier_keys": list(W08_CARRIER_KEYS),
                "cell_states": list(W08_LC16_CELL_STATES),
                "directions": list(W08_CONSUMER_KEYS),
                "evaluator_key": W08_LC16_EVALUATOR_KEY,
                "scope_key": W08_LC16_SCOPE_KEY,
            },
            "open_generation": {
                "coverage_keys": list(W08_OPEN_GENERATION_COVERAGE_KEYS),
                "layer_keys": list(W08_OPEN_GENERATION_LAYER_KEYS),
                "public_state": W08_OPEN_GENERATION_PREFORMAL_STATE,
            },
            "runtime_hard_conjuncts": list(W08_RUNTIME_HARD_CONJUNCT_KEYS),
        },
        "execution_state": dict(W08_ZERO_EXECUTION_STATE),
        "expected_commitments": {
            key: list(value) for key, value in W08_EXPECTED_COMMITMENTS.items()
        },
        "expected_counts": dict(W08_EXPECTED_COUNTS),
        "expected_evidence_counts": list(W08_EXPECTED_EVIDENCE_COUNTS),
        "expected_record_counts": list(W08_EXPECTED_RECORD_COUNTS),
        "formal_w08_training_runs": 0,
        "format_version": 1,
        "future_firewall": {
            "forbidden_inventory": list(context.future_forbidden_paths),
            "future_pack_keys": list(context.future_pack_keys),
            "future_payload_reads": 0,
            "future_source_reads": 0,
        },
        "open_generation_state": W08_OPEN_GENERATION_PREFORMAL_STATE,
        "payload_audit": {
            "companion_writes": 0,
            "evaluator_label_reads": 0,
            "future_payload_reads": 0,
            "held_out_reads": 0,
            "host_learning_writes": 0,
            "memory_learning_writes": 0,
            "payload_bytes": 0,
            "payload_gets": 0,
            "teacher_calls": 0,
        },
        "public_artifact_inventory": _inventory(
            repository, W08_CANDIDATE_PUBLIC_ARTIFACT_PATHS
        ),
        "public_head_commit_sha1": supplied_head,
        "recovery_protocol": {
            "failure_points": list(W08_FAILURE_POINT_KEYS),
            "logical_shard_count": context.logical_shard_count,
            "modes": list(W08_ALLOWED_MODES),
            "worker_counts": list(W08_ALLOWED_WORKER_COUNTS),
        },
        "resource_budget": dict(W08_RESOURCE_BUDGET),
        "retention_inventory": _inventory(repository, W08_RETENTION_PATHS),
        "self_excluded": 1,
        "source_contract": {
            "authority_baseline_head_commit_sha1": authority[
                "baseline_public_head_commit_sha1"
            ],
            "authority_sha256": context.authority_sha256,
            "learning_pack_keys": list(context.learning_pack_keys),
            "train_bindings": _binding_inventory(context),
            "train_pack_keys": list(context.stage_train_pack_keys),
        },
        "test_inventory": _inventory(repository, W08_CANDIDATE_TEST_PATHS),
        "visibility_commitments": {
            "evaluator_binding_count": len(context.evaluator_bindings),
            "evaluator_inventory_commitment_key": list(
                _identity_commitment(context.evaluator_bindings)
            ),
            "forbidden_binding_count": len(context.forbidden_bindings),
            "forbidden_inventory_commitment_key": list(
                _identity_commitment(context.forbidden_bindings)
            ),
            "train_binding_count": len(train_bindings),
        },
    }


def _validate_inventory(value: object, expected: tuple[str, ...], label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != len(expected)
        or tuple(
            item.get("path") for item in value if isinstance(item, dict)
        ) != expected
    ):
        raise RuntimeError(f"W08 {label} inventory 路径漂移")
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size_bytes"}
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] <= 0
        ):
            raise RuntimeError(f"W08 {label} inventory 字段非法")
        _strict_sha256(item["sha256"], label=f"{label} inventory")


def _validate_contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("W08 Candidate 合同类型非法")
    if (
        value.get("artifact_kind") != W08_CANDIDATE_CONTRACT_KIND
        or value.get("format_version") != 1
        or value.get("self_excluded") != 1
        or value.get("execution_state") != W08_ZERO_EXECUTION_STATE
        or value.get("formal_w08_training_runs") != 0
        or value.get("open_generation_state")
        != W08_OPEN_GENERATION_PREFORMAL_STATE
    ):
        raise RuntimeError("W08 Candidate 合同状态漂移")
    _strict_sha1(value.get("public_head_commit_sha1"), label="public HEAD")
    _validate_inventory(value.get("code_inventory"), W08_CANDIDATE_CODE_PATHS, "code")
    _validate_inventory(value.get("test_inventory"), W08_CANDIDATE_TEST_PATHS, "test")
    _validate_inventory(
        value.get("public_artifact_inventory"),
        W08_CANDIDATE_PUBLIC_ARTIFACT_PATHS,
        "public artifact",
    )
    _validate_inventory(value.get("retention_inventory"), W08_RETENTION_PATHS, "retention")
    request = value.get("candidate_request")
    if (
        not isinstance(request, dict)
        or request.get("run_id") != W08_FORMAL_RUN_ID
        or request.get("parent_run_id") != W08_W07_BASE_RUN_ID
        or request.get("base_run_id") != W08_W07_BASE_RUN_ID
        or request.get("worker_count") != W08_CANDIDATE_FORMAL_WORKER_COUNT
        or request.get("mode") != W08_CANDIDATE_FORMAL_MODE
    ):
        raise RuntimeError("W08 Candidate request 合同漂移")
    recovery = value.get("recovery_protocol")
    if (
        not isinstance(recovery, dict)
        or recovery.get("logical_shard_count") != 16
        or tuple(recovery.get("worker_counts", ())) != W08_ALLOWED_WORKER_COUNTS
        or tuple(recovery.get("modes", ())) != W08_ALLOWED_MODES
        or tuple(recovery.get("failure_points", ())) != W08_FAILURE_POINT_KEYS
    ):
        raise RuntimeError("W08 Candidate recovery 合同漂移")
    evaluation = value.get("evaluation_contract")
    inference = (
        evaluation.get("candidate_inference_interface")
        if isinstance(evaluation, dict)
        else None
    )
    open_generation = (
        evaluation.get("open_generation") if isinstance(evaluation, dict) else None
    )
    lc16 = evaluation.get("lc16") if isinstance(evaluation, dict) else None
    if (
        not isinstance(evaluation, dict)
        or tuple(evaluation.get("dimension_order", ())) != W08_DIMENSION_KEYS
        or tuple(evaluation.get("ablation_order", ())) != W08_ABLATION_KEYS
        or tuple(evaluation.get("runtime_hard_conjuncts", ()))
        != W08_RUNTIME_HARD_CONJUNCT_KEYS
        or evaluation.get("aggregation_policy")
        != "ALL_BEARING_DIMENSIONS_MUST_PASS"
        or inference != {
            "component_keys": list(W08_DIMENSION_KEYS),
            "evaluator_label_inputs": 0,
            "executable": 1,
            "input_kind": W08_CANDIDATE_INFERENCE_INPUT_KIND,
            "output_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
            "per_case_invocation_required": 1,
            "shortcut_account_keys": list(W08_INFERENCE_SHORTCUT_KEYS),
            "state_policy": "TRAIN_ONLY_TYPED_RULES",
            "version": W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
        }
        or not isinstance(open_generation, dict)
        or tuple(open_generation.get("layer_keys", ()))
        != W08_OPEN_GENERATION_LAYER_KEYS
        or tuple(open_generation.get("coverage_keys", ()))
        != W08_OPEN_GENERATION_COVERAGE_KEYS
        or open_generation.get("public_state")
        != W08_OPEN_GENERATION_PREFORMAL_STATE
        or not isinstance(lc16, dict)
        or tuple(lc16.get("carrier_keys", ())) != W08_CARRIER_KEYS
        or tuple(lc16.get("directions", ())) != W08_CONSUMER_KEYS
        or tuple(lc16.get("cell_states", ())) != W08_LC16_CELL_STATES
        or lc16.get("scope_key") != W08_LC16_SCOPE_KEY
        or lc16.get("evaluator_key") != W08_LC16_EVALUATOR_KEY
    ):
        raise RuntimeError("W08 Candidate evaluation 合同漂移")
    cluster = value.get("cluster_contract")
    if (
        not isinstance(cluster, dict)
        or tuple(cluster.get("axis_keys", ())) != W08_CANDIDATE_CLUSTER_AXES
        or cluster.get("schema_registry") != _dimension_registry()
        or cluster.get("source_pack_registry") != _pack_registry()
    ):
        raise RuntimeError("W08 Candidate source/schema/cluster 合同漂移")
    future = value.get("future_firewall")
    source = value.get("source_contract")
    audit = value.get("payload_audit")
    if (
        not isinstance(future, dict)
        or any(future.get(key) != 0 for key in ("future_payload_reads", "future_source_reads"))
        or not isinstance(future.get("forbidden_inventory"), list)
        or not future["forbidden_inventory"]
        or not isinstance(source, dict)
        or tuple(source.get("learning_pack_keys", ())) != W08_LEARNING_PACK_KEYS
        or not isinstance(source.get("train_bindings"), list)
        or not source["train_bindings"]
        or not isinstance(audit, dict)
        or any(audit.values())
    ):
        raise RuntimeError("W08 Candidate source/visibility/audit firewall 漂移")
    if (
        value.get("expected_counts") != W08_EXPECTED_COUNTS
        or tuple(value.get("expected_record_counts", ())) != W08_EXPECTED_RECORD_COUNTS
        or tuple(value.get("expected_evidence_counts", ()))
        != W08_EXPECTED_EVIDENCE_COUNTS
        or value.get("expected_commitments")
        != {key: list(item) for key, item in W08_EXPECTED_COMMITMENTS.items()}
        or value.get("resource_budget") != W08_RESOURCE_BUDGET
    ):
        raise RuntimeError("W08 Candidate expected/resource 合同漂移")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if any(token in encoded for token in ('"label"', '"expected"', '"surface"')):
        raise RuntimeError("W08 Candidate freeze 泄露 private/surface 字段")
    return value


def w08_candidate_contract_key(value: dict[str, object]) -> tuple[int, ...]:
    return digest_value(_validate_contract(value))


def _verify_inventory(repository: Path, contract: dict[str, object]) -> None:
    if (
        contract["public_head_commit_sha1"] != _git_sha(repository, "HEAD")
        or contract["code_inventory"] != _inventory(repository, W08_CANDIDATE_CODE_PATHS)
        or contract["test_inventory"] != _inventory(repository, W08_CANDIDATE_TEST_PATHS)
        or contract["public_artifact_inventory"]
        != _inventory(repository, W08_CANDIDATE_PUBLIC_ARTIFACT_PATHS)
        or contract["retention_inventory"] != _inventory(repository, W08_RETENTION_PATHS)
    ):
        raise RuntimeError("W08 Candidate public identity 漂移")


def publish_w08_candidate_contract_freeze(
    repository_root: str | Path,
    artifact_root: str | Path,
    contract: dict[str, object],
) -> tuple[Path, str]:
    """在全新 Git 外 root 排他发布 Candidate contract freeze。"""
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    _validate_external_root(repository, root)
    value = _validate_contract(contract)
    _verify_inventory(repository, value)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise RuntimeError("W08 Candidate root 必须全新且不可复用") from error
    target = root / W08_CANDIDATE_CONTRACT_FREEZE_NAME
    encoded = canonical_json_bytes(value)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise RuntimeError("W08 Candidate contract freeze 不可覆盖") from error
    return target, _sha256_bytes(encoded)


def verify_w08_candidate_contract_freeze(
    freeze_path: str | Path,
    contract: dict[str, object],
) -> str:
    value = _validate_contract(contract)
    path = Path(freeze_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W08 Candidate contract freeze 缺失")
    encoded = path.read_bytes()
    if encoded != canonical_json_bytes(value):
        raise RuntimeError("W08 Candidate contract identity 漂移")
    return _sha256_bytes(encoded)


def consume_w08_candidate_first_run_guard(
    artifact_root: str | Path,
    *,
    candidate_contract_sha256: str,
) -> tuple[Path, str]:
    """排他创建 first-run guard，并且只在此刻把正式计数推进到一。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(candidate_contract_sha256, label="Candidate contract")
    freeze = root / W08_CANDIDATE_CONTRACT_FREEZE_NAME
    if (
        not freeze.is_file()
        or freeze.is_symlink()
        or _sha256_bytes(freeze.read_bytes()) != expected
    ):
        raise RuntimeError("W08 Candidate contract SHA 漂移")
    payload = canonical_json_bytes({
        "artifact_kind": W08_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": expected,
        "execution_state_after_start": dict(W08_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "open_generation_state_after_start": W08_OPEN_GENERATION_PREFORMAL_STATE,
        "run_id": W08_FORMAL_RUN_ID,
    })
    target = root / W08_CANDIDATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError("W08 Candidate first-run 已消费，不可重跑") from error
    return target, _sha256_bytes(payload)


__all__ = [
    "W08_CANDIDATE_CLUSTER_AXES",
    "W08_CANDIDATE_CODE_PATHS",
    "W08_CANDIDATE_CONTRACT_FREEZE_NAME",
    "W08_CANDIDATE_CONTRACT_KIND",
    "W08_CANDIDATE_FIRST_RUN_GUARD_KIND",
    "W08_CANDIDATE_FIRST_RUN_GUARD_NAME",
    "W08_CANDIDATE_FORMAL_MODE",
    "W08_CANDIDATE_FORMAL_WORKER_COUNT",
    "W08_CANDIDATE_PUBLIC_ARTIFACT_PATHS",
    "W08_CANDIDATE_TEST_PATHS",
    "W08_EXPECTED_COMMITMENTS",
    "W08_EXPECTED_COUNTS",
    "W08_EXPECTED_EVIDENCE_COUNTS",
    "W08_EXPECTED_RECORD_COUNTS",
    "build_w08_candidate_contract",
    "consume_w08_candidate_first_run_guard",
    "publish_w08_candidate_contract_freeze",
    "verify_w08_candidate_contract_freeze",
    "w08_candidate_contract_key",
]
