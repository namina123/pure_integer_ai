"""W07-05 candidate 合同冻结、唯一 guard、正式 run 与 host freeze。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_OPEN_GENERATION_STATE,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
    W07_RESOURCE_BUDGET,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_runtime import (
    W07RunOutcome,
    W07RuntimeConfig,
    load_w07_public_dump,
    run_language_stage7_public,
)
from pure_integer_ai.storage.backend import SQLiteBackend


W07_CANDIDATE_CONTRACT_KIND = "PH2_W07_CANDIDATE_CONTRACT_FREEZE"
W07_CANDIDATE_HOST_FREEZE_KIND = "PH2_W07_CANDIDATE_HOST_FREEZE"
W07_CANDIDATE_FIRST_RUN_GUARD_KIND = "PH2_W07_CANDIDATE_FIRST_RUN_GUARD"
W07_CANDIDATE_CONTRACT_FREEZE_NAME = "candidate_contract_freeze.json"
W07_CANDIDATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W07_CANDIDATE_HOST_FREEZE_NAME = "candidate_host_freeze.json"
W07_CANDIDATE_FORMAL_WORKER_COUNT = 4
W07_CANDIDATE_FORMAL_MODE = "fresh"
W07_EXPECTED_LOGICAL_STATE_DIGEST = (
    "91aefde4c31be00f5143ca691809988c7e2afa7e180b886cd60ce61176288df3")
W07_FORMAL_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W06_RUNTIME_EVIDENCED": 1,
    "W07_STARTED": 1,
    "W08_STARTED": 0,
    "formal_w07_training_runs": 1,
    "teacher_calls": 0,
}
W07_EXPECTED_COUNTS = {
    "active_operator_count": 36,
    "candidate_count": 71,
    "carrier_projection_count": 9,
    "evidence_account_count": 94,
    "evidence_application_count": 63,
    "logical_shard_count": 16,
    "logic_scope_cell_count": 189,
    "logic_use_count": 21,
    "operator_profile_count": 7,
    "schema_rejection_count": 3,
    "substage_count": 7,
}
W07_CASE_FAMILIES = tuple(
    (dimension, ablation, substage)
    for dimension, ablation, substage in zip(
        W07_PUBLIC_DIMENSION_KEYS[:-1],
        W07_PUBLIC_ABLATION_KEYS[:-1],
        (
            "AND_OR", "CONDITION", "EXISTS", "FORALL", "MODAL",
            "NESTED_SCOPE", "NOT",
        ),
        strict=True,
    )
) + ((
    W07_PUBLIC_DIMENSION_KEYS[-1],
    W07_PUBLIC_ABLATION_KEYS[-1],
    "GENERATION",
),)
W07_CANDIDATE_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w07_adapter.py",
    "src/pure_integer_ai/experiments/ph2_w07_candidate.py",
    "src/pure_integer_ai/experiments/ph2_w07_carrier_scope.py",
    "src/pure_integer_ai/experiments/ph2_w07_contract.py",
    "src/pure_integer_ai/experiments/ph2_w07_faults.py",
    "src/pure_integer_ai/experiments/ph2_w07_firewall.py",
    "src/pure_integer_ai/experiments/ph2_w07_l01.py",
    "src/pure_integer_ai/experiments/ph2_w07_l02.py",
    "src/pure_integer_ai/experiments/ph2_w07_l03.py",
    "src/pure_integer_ai/experiments/ph2_w07_l04.py",
    "src/pure_integer_ai/experiments/ph2_w07_l05.py",
    "src/pure_integer_ai/experiments/ph2_w07_l06.py",
    "src/pure_integer_ai/experiments/ph2_w07_l07.py",
    "src/pure_integer_ai/experiments/ph2_w07_learning.py",
    "src/pure_integer_ai/experiments/ph2_w07_logic_consumer.py",
    "src/pure_integer_ai/experiments/ph2_w07_logic_contract.py",
    "src/pure_integer_ai/experiments/ph2_w07_logic_generation.py",
    "src/pure_integer_ai/experiments/ph2_w07_logic_shared.py",
    "src/pure_integer_ai/experiments/ph2_w07_payload.py",
    "src/pure_integer_ai/experiments/ph2_w07_registry.py",
    "src/pure_integer_ai/experiments/ph2_w07_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w07_transaction.py",
)
W07_CANDIDATE_TEST_PATHS = tuple(
    [f"tests/test_w07_l0{index}.py" for index in range(1, 8)]
    + [
        "tests/test_w07_stage2_adapter.py",
        "tests/test_w07_stage2_candidate.py",
        "tests/test_w07_stage2_contract.py",
        "tests/test_w07_stage2_learning.py",
        "tests/test_w07_stage2_registry.py",
        "tests/test_w07_stage2_runtime.py",
    ]
)
W07_CANDIDATE_PUBLIC_ARTIFACT_PATHS = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json",
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json",
    "data/ph2/manifests/d03_v1/stages/w07_stage_manifest_v1.json",
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json",
    "data/ph2/manifests/d03_lc16_successor_overlay_v1.json",
    "data/ph2/manifests/lc16_carrier_directional_runtime_v1.json",
    "data/ph2/manifests/lc13_directional_consumer_manifest_v2.json",
    "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
    "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
)


class W07CandidateError(RuntimeError):
    """candidate freeze、guard、runtime 或 host evidence 未闭合。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W07CandidateError(f"{label} SHA-256 非法")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(item not in "0123456789abcdef" for item in value)):
        raise W07CandidateError(f"{label} SHA-1 非法")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise W07CandidateError(f"{label} path 非法")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise W07CandidateError(f"{label} path 非安全相对路径")
    return value


def _inventory(repository: Path, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for relative in paths:
        safe = _safe_relative_path(relative, label="inventory")
        target = (repository / safe).resolve()
        if not target.is_relative_to(repository) or not target.is_file():
            raise W07CandidateError(f"candidate inventory 缺失：{safe}")
        payload = target.read_bytes()
        result.append({
            "bytes": len(payload),
            "relative_path": safe,
            "sha256": _sha256_bytes(payload),
        })
    return result


def _validate_external_root(repository: Path, artifact_root: Path) -> None:
    if artifact_root == repository or artifact_root.is_relative_to(repository):
        raise W07CandidateError("candidate root 必须位于公开 Git 外")
    if repository.is_relative_to(artifact_root):
        raise W07CandidateError("candidate root 不得包含公开 Git")


def _zero_state() -> dict[str, Any]:
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "OPEN_GENERATION": W07_OPEN_GENERATION_STATE,
        "W07_STARTED": 0,
        "W08_STARTED": 0,
        "formal_w07_training_runs": 0,
        "teacher_calls": 0,
    }


def build_w07_candidate_contract(
        repository_root: str | Path,
        *,
        backend_profile_key: tuple[int, ...],
        public_head_commit_sha1: str,
        ) -> dict[str, Any]:
    """冻结当前 public code/tests/parents/request 与八维评测零状态。"""
    repository = Path(repository_root).resolve()
    head = _strict_sha1(public_head_commit_sha1, label="public HEAD")
    context = open_w07_frozen_context(
        repository,
        baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
        backend_profile_key=backend_profile_key,
    )
    return {
        "artifact_kind": W07_CANDIDATE_CONTRACT_KIND,
        "candidate_request": {
            "base_fence_key": list(context.base_fence_key),
            "base_run_id": W07_W06_BASE_RUN_ID,
            "logical_shard_count": context.logical_shard_count,
            "mode": W07_CANDIDATE_FORMAL_MODE,
            "parent_run_id": W07_W06_BASE_RUN_ID,
            "resource_budget": dict(context.resource_budget),
            "run_id": W07_FORMAL_RUN_ID,
            "worker_count": W07_CANDIDATE_FORMAL_WORKER_COUNT,
        },
        "code_inventory": _inventory(repository, W07_CANDIDATE_CODE_PATHS),
        "evaluation_contract": {
            "ablation_order": list(W07_PUBLIC_ABLATION_KEYS),
            "aggregation_policy": context.aggregation_policy,
            "case_families": [list(item) for item in W07_CASE_FAMILIES],
            "dimension_order": list(W07_PUBLIC_DIMENSION_KEYS),
            "failure_points": list(context.failure_point_keys),
            "substage_order": list(W07_SUBSTAGE_ORDER),
        },
        "execution_state": _zero_state(),
        "expected_counts": dict(W07_EXPECTED_COUNTS),
        "expected_logical_state_digest": W07_EXPECTED_LOGICAL_STATE_DIGEST,
        "formal_w07_training_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "open_generation_state": W07_OPEN_GENERATION_STATE,
        "pack_bindings": [item.to_dict() for item in context.pack_bindings],
        "parent_sha256": [list(item) for item in context.parent_sha256],
        "public_artifact_inventory": _inventory(
            repository, W07_CANDIDATE_PUBLIC_ARTIFACT_PATHS),
        "public_head_commit_sha1": head,
        "self_excluded": 1,
        "stage_key": "W-07",
        "test_inventory": _inventory(repository, W07_CANDIDATE_TEST_PATHS),
        "teacher_calls": 0,
    }


def _validate_inventory(
        value: object,
        expected: tuple[str, ...],
        label: str,
        ) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise W07CandidateError(f"{label} inventory 数量漂移")
    paths = tuple(item.get("relative_path") for item in value
                  if isinstance(item, dict))
    if paths != expected:
        raise W07CandidateError(f"{label} inventory 顺序漂移")
    for item in value:
        if (not isinstance(item, dict)
                or set(item) != {"bytes", "relative_path", "sha256"}
                or type(item["bytes"]) is not int or item["bytes"] <= 0):
            raise W07CandidateError(f"{label} inventory 字段漂移")
        _safe_relative_path(item["relative_path"], label=label)
        _strict_sha256(item["sha256"], label=label)


def _validate_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise W07CandidateError("candidate contract 根非法")
    required = {
        "artifact_kind", "candidate_request", "code_inventory",
        "evaluation_contract", "execution_state", "expected_counts",
        "expected_logical_state_digest", "formal_w07_training_runs",
        "format_version", "guard_consumed", "open_generation_state",
        "pack_bindings", "parent_sha256", "public_artifact_inventory",
        "public_head_commit_sha1", "self_excluded", "stage_key",
        "test_inventory", "teacher_calls",
    }
    if set(value) != required:
        raise W07CandidateError("candidate contract 字段漂移")
    if (value["artifact_kind"] != W07_CANDIDATE_CONTRACT_KIND
            or value["format_version"] != 1
            or value["stage_key"] != "W-07"
            or value["self_excluded"] != 1
            or value["execution_state"] != _zero_state()
            or value["formal_w07_training_runs"] != 0
            or value["guard_consumed"] != 0
            or value["teacher_calls"] != 0
            or value["open_generation_state"] != W07_OPEN_GENERATION_STATE
            or value["expected_counts"] != W07_EXPECTED_COUNTS
            or value["expected_logical_state_digest"]
            != W07_EXPECTED_LOGICAL_STATE_DIGEST):
        raise W07CandidateError("candidate contract 零状态或期望漂移")
    _strict_sha1(value["public_head_commit_sha1"], label="public HEAD")
    _validate_inventory(
        value["code_inventory"], W07_CANDIDATE_CODE_PATHS, "code")
    _validate_inventory(
        value["test_inventory"], W07_CANDIDATE_TEST_PATHS, "test")
    _validate_inventory(
        value["public_artifact_inventory"],
        W07_CANDIDATE_PUBLIC_ARTIFACT_PATHS,
        "public artifact",
    )
    evaluation = value.get("evaluation_contract")
    request = value.get("candidate_request")
    if (not isinstance(evaluation, dict)
            or tuple(evaluation.get("dimension_order", ()))
            != W07_PUBLIC_DIMENSION_KEYS
            or tuple(evaluation.get("ablation_order", ()))
            != W07_PUBLIC_ABLATION_KEYS
            or tuple(tuple(item) for item in evaluation.get(
                "case_families", ())) != W07_CASE_FAMILIES
            or tuple(evaluation.get("substage_order", ()))
            != W07_SUBSTAGE_ORDER):
        raise W07CandidateError("candidate evaluation freeze 漂移")
    if (not isinstance(request, dict)
            or request.get("run_id") != W07_FORMAL_RUN_ID
            or request.get("parent_run_id") != W07_W06_BASE_RUN_ID
            or request.get("base_run_id") != W07_W06_BASE_RUN_ID
            or request.get("worker_count") != W07_CANDIDATE_FORMAL_WORKER_COUNT
            or request.get("mode") != W07_CANDIDATE_FORMAL_MODE
            or request.get("logical_shard_count") != 16
            or request.get("resource_budget") != W07_RESOURCE_BUDGET
            or not request.get("base_fence_key")):
        raise W07CandidateError("candidate request freeze 漂移")
    if (not isinstance(value.get("pack_bindings"), list)
            or len(value["pack_bindings"]) != 7
            or not isinstance(value.get("parent_sha256"), list)
            or len(value["parent_sha256"]) != 9):
        raise W07CandidateError("candidate parent/pack freeze 漂移")
    return value


def _verify_inventory(repository: Path, contract: dict[str, Any]) -> None:
    for field, paths in (
            ("code_inventory", W07_CANDIDATE_CODE_PATHS),
            ("test_inventory", W07_CANDIDATE_TEST_PATHS),
            ("public_artifact_inventory", W07_CANDIDATE_PUBLIC_ARTIFACT_PATHS),
            ):
        expected = _inventory(repository, paths)
        if contract[field] != expected:
            raise W07CandidateError(f"{field} identity 漂移")


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> str:
    if path.exists():
        raise W07CandidateError(f"{label} 不可覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W07CandidateError(f"{label} 不可覆盖") from error
    return _sha256_bytes(payload)


def publish_w07_candidate_contract_freeze(
        repository_root: str | Path,
        candidate_root: str | Path,
        contract: dict[str, Any],
        ) -> tuple[Path, str]:
    repository = Path(repository_root).resolve()
    root = Path(candidate_root).resolve()
    _validate_external_root(repository, root)
    frozen = _validate_contract(contract)
    _verify_inventory(repository, frozen)
    path = root / W07_CANDIDATE_CONTRACT_FREEZE_NAME
    digest = _write_exclusive(
        path, canonical_json_bytes(frozen), label="candidate contract freeze")
    return path, digest


def verify_w07_candidate_contract_freeze(
        repository_root: str | Path,
        candidate_root: str | Path,
        *,
        candidate_contract_sha256: str,
        ) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    root = Path(candidate_root).resolve()
    _validate_external_root(repository, root)
    expected = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    path = root / W07_CANDIDATE_CONTRACT_FREEZE_NAME
    if not path.is_file() or _sha256_bytes(path.read_bytes()) != expected:
        raise W07CandidateError("candidate contract freeze identity 漂移")
    value = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(value, dict)
    contract = _validate_contract(value)
    _verify_inventory(repository, contract)
    return contract


def consume_w07_candidate_first_run_guard(
        candidate_root: str | Path,
        *,
        candidate_contract_sha256: str,
        public_head_commit_sha1: str,
        ) -> tuple[Path, str]:
    root = Path(candidate_root).resolve()
    contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    head = _strict_sha1(public_head_commit_sha1, label="public HEAD")
    payload = canonical_json_bytes({
        "artifact_kind": W07_CANDIDATE_FIRST_RUN_GUARD_KIND,
        "candidate_contract_sha256": contract_sha,
        "execution_state_after": dict(W07_FORMAL_EXECUTION_STATE),
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
        "public_head_commit_sha1": head,
        "run_id": W07_FORMAL_RUN_ID,
        "self_excluded": 1,
    })
    path = root / W07_CANDIDATE_FIRST_RUN_GUARD_NAME
    digest = _write_exclusive(path, payload, label="candidate guard 不可重跑")
    return path, digest


def formalize_w07_candidate_outcome(outcome: W07RunOutcome) -> W07RunOutcome:
    if not isinstance(outcome, W07RunOutcome):
        raise TypeError("formalize 需要 W07RunOutcome")
    return replace(outcome, execution_state=dict(W07_FORMAL_EXECUTION_STATE))


def _outcome_evidence(outcome: W07RunOutcome) -> dict[str, Any]:
    return {
        "artifact_counts": [list(item) for item in outcome.artifact_counts],
        "dump_manifest_sha256": outcome.dump_manifest_sha256,
        "dump_readback": int(outcome.dump_readback),
        "execution_state": dict(outcome.execution_state),
        "host_digests": {
            "active_projection": outcome.active_projection_digest,
            "candidate": outcome.candidate_digest,
            "carrier_scope": outcome.carrier_scope_digest,
            "logic": outcome.logic_digest,
            "logical": outcome.logical_state_digest,
            "source_evidence": outcome.source_evidence_digest,
            "transaction": outcome.transaction_digest,
        },
        "learning_attempt_count": outcome.learning_attempt_count,
        "new_learning_write_count": outcome.new_learning_write_count,
        "owned_tables": list(outcome.owned_tables),
        "payload_bytes_this_call": outcome.payload_bytes_this_call,
        "payload_gets_this_call": outcome.payload_gets_this_call,
        "resource_report": dict(outcome.resource_report),
        "retention_sha256": [list(item) for item in outcome.retention_sha256],
        "teacher_calls": outcome.teacher_calls,
        "transaction_event_count": outcome.transaction_event_count,
    }


def _validate_formal_outcome(
        outcome: W07RunOutcome,
        readback: W07RunOutcome,
        ) -> None:
    expected_artifacts = {
        "ACTIVE_OPERATOR": 36,
        "CANDIDATE": 71,
        "CARRIER_PROJECTION": 9,
        "EVIDENCE_ACCOUNT": 94,
        "EVIDENCE_APPLICATION": 63,
        "LOGICAL_SHARD": 16,
        "LOGIC_SCOPE_CELL": 189,
        "LOGIC_USE": 21,
        "OPERATOR_PROFILE": 7,
        "SCHEMA_REJECTION": 3,
        "SUBSTAGE": 7,
    }
    if (outcome.logical_state_digest != W07_EXPECTED_LOGICAL_STATE_DIGEST
            or readback.logical_state_digest != outcome.logical_state_digest
            or outcome.candidate_count != W07_EXPECTED_COUNTS["candidate_count"]
            or outcome.active_candidate_count
            != W07_EXPECTED_COUNTS["active_operator_count"]
            or dict(outcome.artifact_counts) != expected_artifacts
            or readback.artifact_counts != outcome.artifact_counts
            or outcome.transaction_event_count != 5
            or readback.transaction_event_count != 5
            or outcome.teacher_calls != 0 or readback.teacher_calls != 0
            or outcome.payload_gets_this_call <= 0
            or outcome.payload_bytes_this_call <= 0
            or outcome.new_learning_write_count <= 0
            or not readback.dump_readback
            or readback.payload_gets_this_call != 0
            or readback.payload_bytes_this_call != 0
            or readback.new_learning_write_count != 0):
        raise W07CandidateError("candidate formal outcome 未闭合")


def publish_w07_candidate_host_freeze(
        candidate_root: str | Path,
        *,
        candidate_contract_sha256: str,
        candidate_first_run_guard_sha256: str,
        public_head_commit_sha1: str,
        outcome: W07RunOutcome,
        readback: W07RunOutcome,
        ) -> tuple[Path, str]:
    _validate_formal_outcome(outcome, readback)
    formal_outcome = formalize_w07_candidate_outcome(outcome)
    formal_readback = formalize_w07_candidate_outcome(readback)
    payload = canonical_json_bytes({
        "artifact_kind": W07_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_sha256": _strict_sha256(
            candidate_contract_sha256, label="candidate contract"),
        "candidate_first_run_guard_sha256": _strict_sha256(
            candidate_first_run_guard_sha256, label="candidate guard"),
        "dump_readback_evidence": _outcome_evidence(formal_readback),
        "execution_state": dict(W07_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "format_version": 1,
        "host_evidence": _outcome_evidence(formal_outcome),
        "open_generation_state": W07_OPEN_GENERATION_STATE,
        "owner_write_counts": {
            "artifact_writes": outcome.new_learning_write_count,
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": 0,
            "teacher_calls": 0,
        },
        "public_head_commit_sha1": _strict_sha1(
            public_head_commit_sha1, label="public HEAD"),
        "self_excluded": 1,
    })
    path = Path(candidate_root).resolve() / W07_CANDIDATE_HOST_FREEZE_NAME
    digest = _write_exclusive(path, payload, label="candidate host freeze")
    return path, digest


def execute_w07_candidate_once(
        repository_root: str | Path,
        candidate_root: str | Path,
        *,
        config: W07RuntimeConfig,
        contract: dict[str, Any],
        candidate_contract_sha256: str,
        ) -> tuple[W07RunOutcome, W07RunOutcome, Path, str, Path, str]:
    """消费唯一 guard，运行一次并封存 host/readback；异常也不恢复 guard。"""
    repository = Path(repository_root).resolve()
    root = Path(candidate_root).resolve()
    frozen = verify_w07_candidate_contract_freeze(
        repository,
        root,
        candidate_contract_sha256=candidate_contract_sha256,
    )
    if frozen != _validate_contract(contract):
        raise W07CandidateError("内存 contract 与 freeze 漂移")
    request = frozen["candidate_request"]
    expected_run_root = (root / "run").resolve()
    expected_sqlite = (root / "coordinator.sqlite").resolve()
    if (Path(config.repository_root).resolve() != repository
            or Path(config.run_root).resolve() != expected_run_root
            or Path(config.sqlite_path).resolve() != expected_sqlite
            or config.run_id != request["run_id"]
            or config.parent_run_id != request["parent_run_id"]
            or config.base_run_id != request["base_run_id"]
            or tuple(config.base_fence_key or ())
            != tuple(request["base_fence_key"])
            or config.worker_count != request["worker_count"]
            or config.mode != request["mode"]
            or config.fault_point is not None):
        raise W07CandidateError("candidate formal config 与 freeze 漂移")
    guard_path, guard_sha = consume_w07_candidate_first_run_guard(
        root,
        candidate_contract_sha256=candidate_contract_sha256,
        public_head_commit_sha1=frozen["public_head_commit_sha1"],
    )
    outcome = run_language_stage7_public(config)
    readback = load_w07_public_dump(config)
    _validate_formal_outcome(outcome, readback)
    host_path, host_sha = publish_w07_candidate_host_freeze(
        root,
        candidate_contract_sha256=candidate_contract_sha256,
        candidate_first_run_guard_sha256=guard_sha,
        public_head_commit_sha1=frozen["public_head_commit_sha1"],
        outcome=outcome,
        readback=readback,
    )
    return outcome, readback, host_path, host_sha, guard_path, guard_sha


__all__ = [name for name in globals() if name.startswith("W07_")] + [
    "W07CandidateError",
    "build_w07_candidate_contract",
    "consume_w07_candidate_first_run_guard",
    "execute_w07_candidate_once",
    "formalize_w07_candidate_outcome",
    "publish_w07_candidate_contract_freeze",
    "publish_w07_candidate_host_freeze",
    "verify_w07_candidate_contract_freeze",
]
