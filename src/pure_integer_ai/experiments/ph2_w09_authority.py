"""W-09 authority 基线构建、规范回读和 append-only 发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any


W09_AUTHORITY_KIND = "PH2_W09_AUTHORITY_BASELINE"
W09_AUTHORITY_VERSION = "PH2-W09-AUTHORITY-BASELINE-V1"
W09_AUTHORITY_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w09_authority_baseline_v1.json"
)
W09_STAGE_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/stages/w09_stage_manifest_v1.json"
)
W09_GLOBAL_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
W09_INVALIDATION_GRAPH_PATH = (
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json"
)
W09_LC16_OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
W09_W08_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w08_runtime_evidence_receipt_v1.json"
)

W09_DIMENSION_KEYS = (
    "W-09-DIMENSIONAL_PASS",
    "W-09-RESOURCE_STOP",
    "W-09-ROLLBACK",
    "W-09-TEACHER_ZERO_WINDOW",
    "W-09-V06_CLONE",
)
W09_ABLATION_KEYS = tuple(f"{key}-ABLATION" for key in W09_DIMENSION_KEYS) + (
    "W-09-W1_PHYSICAL_GROUNDING-ABLATION",
    "W-09-W2_DEFINITIVE_TRUTH-ABLATION",
)
W09_SUBTASK_ORDER = (
    "AUTHORITY",
    "CONTRACT_REGISTRY_FIREWALL",
    "TYPED_WEANING",
    "CUMULATIVE_RUNTIME",
    "DIMENSIONAL_J_LC",
    "RESOURCE_STOP",
    "ROLLBACK",
    "V06_CLONE",
    "TRANSACTION",
    "CANDIDATE",
    "PRIVATE_EVALUATOR",
    "RECEIPTS_STOP",
)
W09_CARRIER_KEYS = (
    "DOCUMENT_CONTAINER",
    "HTML",
    "MARKDOWN",
    "MATH_NOTATION",
    "PLAIN_TEXT",
    "REFERENCE_LINK_EMBED",
    "SOURCE_CODE",
    "TABLE_GRID",
    "TRANSCRIBED_OCR_ASR",
)
W09_CONSUMER_KEYS = ("UNDERSTANDING", "REASONING", "GENERATION")
W09_STOP_STATES = (
    "RESOLVED",
    "CLARIFY",
    "UNKNOWN",
    "ACCESS_BLOCKED",
    "GROUNDING_BLOCKED",
    "BUDGET_EXHAUSTED",
)
W09_RESOURCE_BUDGET = {
    "max_records": 900000,
    "max_segments": 36864,
    "max_payload_gets": 589824,
    "max_payload_bytes": 603979776,
    "max_logic_operations": 9000000,
    "max_recompute_objects": 900000,
    "max_workers": 4,
    "max_checkpoint_count": 2304,
}
W09_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W09_FAILURE_POINT_KEYS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)
W09_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W08_RUNTIME_EVIDENCED": 1,
    "W09_STARTED": 0,
    "W09_RUNTIME_EVIDENCED": 0,
    "W09_BLOCKED_FAILED": 0,
    "formal_w09_training_runs": 0,
    "teacher_calls": 0,
    "memory_learning_writes": 0,
    "use_learning_writes": 0,
    "assessment_updates": 0,
    "companion_writes": 0,
    "readiness_claims": 0,
}
_ROOT_FIELDS = {
    "ablation_keys",
    "aggregation_policy",
    "artifact_kind",
    "artifact_version",
    "baseline_public_head_commit_sha1",
    "carrier_keys",
    "consumer_keys",
    "dimension_keys",
    "execution_state",
    "format_version",
    "historical_exposure",
    "lc16_scope",
    "parent_identities",
    "resource_budget",
    "stage_inventory",
    "stop_states",
    "subtask_order",
    "transaction",
}


class W09AuthorityError(RuntimeError):
    """W-09 authority 父身份、可见性或状态发生漂移。"""


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W09AuthorityError("W-09 authority path is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise W09AuthorityError("W-09 authority path escapes repository")
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W09AuthorityError(f"missing W-09 authority parent: {path.name}") from error
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W09AuthorityError(f"invalid W-09 authority JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise W09AuthorityError("W-09 authority JSON must be an object")
    return value


def _identity(repository: Path, relative_path: str) -> dict[str, Any]:
    relative = _safe_relative(relative_path)
    path = repository / relative
    try:
        size = path.stat().st_size
    except OSError as error:
        raise W09AuthorityError(f"missing W-09 authority parent: {relative}") from error
    if size <= 0:
        raise W09AuthorityError(f"empty W-09 authority parent: {relative}")
    return {"relative_path": relative, "sha256": _sha256(path), "size_bytes": size}


def _git_sha(repository: Path, revision: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", revision],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise W09AuthorityError(f"cannot resolve Git revision: {revision}") from error
    if len(result) != 40 or any(char not in "0123456789abcdef" for char in result):
        raise W09AuthorityError("invalid public Git revision")
    return result


def _baseline_public_head(repository: Path) -> str:
    """冻结首次现场 HEAD；authority 后续只允许在该提交上追加。"""
    destination = repository / W09_AUTHORITY_RELATIVE_PATH
    if destination.is_file():
        existing = _canonical_object(destination)
        baseline = existing.get("baseline_public_head_commit_sha1")
        if not isinstance(baseline, str) or _git_sha(repository, baseline) != baseline:
            raise W09AuthorityError("W-09 baseline commit is missing")
        current = _git_sha(repository, "HEAD")
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", baseline, current],
                cwd=repository,
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise W09AuthorityError("cannot verify W-09 baseline ancestry") from error
        if result.returncode != 0:
            raise W09AuthorityError("current HEAD is not descended from W-09 baseline")
        return baseline
    local = _git_sha(repository, "HEAD")
    tracking = _git_sha(repository, "origin/master")
    if local != tracking:
        raise W09AuthorityError("local and origin/master drifted before W-09")
    return local


def build_w09_authority(repository_root: str | Path) -> dict[str, Any]:
    """从现场父 artifact 重算 W-09 authority，不读取任何 payload。"""
    repository = Path(repository_root).resolve()
    stage = _canonical_object(repository / W09_STAGE_MANIFEST_PATH)
    global_manifest = _canonical_object(repository / W09_GLOBAL_MANIFEST_PATH)
    invalidation = _canonical_object(repository / W09_INVALIDATION_GRAPH_PATH)
    overlay = _canonical_object(repository / W09_LC16_OVERLAY_PATH)
    w08 = _canonical_object(repository / W09_W08_RECEIPT_PATH)
    stage_identity = stage.get("stage_identity", {})
    evaluation = stage.get("evaluation_binding", {})
    visibility = stage.get("data_visibility", {})
    thresholds = evaluation.get("thresholds", [])
    bearing = tuple(
        item.get("dimension_key") for item in thresholds if item.get("bearing") == 1
    )
    if (
        stage_identity.get("stage_key") != "W-09"
        or stage_identity.get("ordinal") != 9
        or stage_identity.get("prerequisite_stage_keys") != ["W-08"]
        or stage_identity.get("substage_keys") != []
        or bearing != W09_DIMENSION_KEYS
        or tuple(evaluation.get("ablation_keys", ())) != W09_ABLATION_KEYS
        or evaluation.get("aggregation_policy") != "ALL_BEARING_DIMENSIONS_MUST_PASS"
        or evaluation.get("continuous_window_count") != 3
        or any(item.get("ne_policy") != "BLOCK" for item in thresholds if item.get("bearing") == 1)
    ):
        raise W09AuthorityError("W-09 bearing contract drifted")
    if (
        tuple(stage.get("recovery_binding", {}).get("allowed_worker_counts", ()))
        != W09_ALLOWED_WORKER_COUNTS
        or stage.get("recovery_binding", {}).get("logical_shard_count") != 16
        or stage.get("recovery_binding", {}).get("failure_point_keys") != list(W09_FAILURE_POINT_KEYS)
        or stage.get("recovery_binding", {}).get("fresh_resume_equivalent") != 1
        or stage.get("resource_budget") != W09_RESOURCE_BUDGET
    ):
        raise W09AuthorityError("W-09 recovery/resource contract drifted")
    train_pack_keys = visibility.get("train_pack_keys", [])
    dev_pack_keys = visibility.get("dev_pack_keys", [])
    held_out_pack_keys = visibility.get("held_out_pack_keys", [])
    evaluator_pack_keys = visibility.get("evaluator_pack_keys", [])
    future_pack_keys = visibility.get("future_pack_keys")
    pack_bindings = global_manifest.get("pack_bindings", [])
    registered_pack_keys = [
        item.get("pack_key") for item in pack_bindings if isinstance(item, dict)
    ] if isinstance(pack_bindings, list) else []
    registered_train_pack_keys = [
        item.get("pack_key")
        for item in pack_bindings
        if isinstance(item, dict) and item.get("train_observation_paths")
    ] if isinstance(pack_bindings, list) else []
    registered_dev_pack_keys = [
        item.get("pack_key")
        for item in pack_bindings
        if isinstance(item, dict) and item.get("dev_observation_paths")
    ] if isinstance(pack_bindings, list) else []
    if (
        not isinstance(train_pack_keys, list)
        or not isinstance(dev_pack_keys, list)
        or not isinstance(held_out_pack_keys, list)
        or not isinstance(evaluator_pack_keys, list)
        or not isinstance(future_pack_keys, list)
        or len(train_pack_keys) != 34
        or len(dev_pack_keys) != 1
        or len(held_out_pack_keys) != 37
        or len(evaluator_pack_keys) != 37
        or future_pack_keys != []
        or len(set(train_pack_keys)) != len(train_pack_keys)
        or len(set(dev_pack_keys)) != len(dev_pack_keys)
        or len(set(held_out_pack_keys)) != len(held_out_pack_keys)
        or len(set(evaluator_pack_keys)) != len(evaluator_pack_keys)
        or train_pack_keys != registered_train_pack_keys
        or dev_pack_keys != registered_dev_pack_keys
        or held_out_pack_keys != registered_pack_keys
        or evaluator_pack_keys != registered_pack_keys
        or visibility.get("candidate_allowed_splits") != ["train"]
        or visibility.get("candidate_forbidden_splits") != ["dev", "held_out", "adversarial", "wall"]
    ):
        raise W09AuthorityError("W-09 visibility inventory drifted")
    if (
        w08.get("status") != "RUNTIME_EVIDENCED"
        or w08.get("execution_state", {}).get("W08_RUNTIME_EVIDENCED") != 1
        or w08.get("execution_state", {}).get("W09_STARTED") != 0
        or w08.get("exposure_audit", {}).get("public_sample_heldout_label_exposure") != 1
        or w08.get("exposure_audit", {}).get("prior_fixed_family_eligible_for_pass") != 0
    ):
        raise W09AuthorityError("W08 receipt does not release W-09")
    if not any(
        item.get("consumer_stage") == "W-09" and item.get("prerequisite_stage") == "W-08"
        for item in invalidation.get("stage_edges", [])
    ):
        raise W09AuthorityError("W-09 invalidation edge is missing")
    carrier_courses = overlay.get("carrier_courses")
    overlay_carrier_keys = (
        [item.get("carrier_key") for item in carrier_courses]
        if isinstance(carrier_courses, list)
        else []
    )
    if (
        not overlay.get("scope_records")
        or overlay_carrier_keys != list(W09_CARRIER_KEYS)
        or len(overlay.get("scope_records", [])) != 8
    ):
        raise W09AuthorityError("LC-16 overlay is missing carrier scope")
    parent_paths = (
        W09_STAGE_MANIFEST_PATH,
        W09_GLOBAL_MANIFEST_PATH,
        W09_INVALIDATION_GRAPH_PATH,
        W09_LC16_OVERLAY_PATH,
        W09_W08_RECEIPT_PATH,
    )
    parents = [_identity(repository, path) for path in parent_paths]
    return {
        "ablation_keys": list(W09_ABLATION_KEYS),
        "aggregation_policy": "ALL_BEARING_DIMENSIONS_MUST_PASS",
        "artifact_kind": W09_AUTHORITY_KIND,
        "artifact_version": W09_AUTHORITY_VERSION,
        "baseline_public_head_commit_sha1": _baseline_public_head(repository),
        "carrier_keys": list(W09_CARRIER_KEYS),
        "consumer_keys": list(W09_CONSUMER_KEYS),
        "dimension_keys": list(W09_DIMENSION_KEYS),
        "execution_state": dict(W09_ZERO_EXECUTION_STATE),
        "format_version": 1,
        "historical_exposure": {
            "public_sample_heldout_label_exposure": 1,
            "prior_fixed_family_eligible_for_pass": 0,
            "current_rotation_exposure_audit_clean": 1,
        },
        "lc16_scope": {
            "carrier_count": 9,
            "direction_count": 3,
            "scope_count": 8,
            "cell_count": 216,
            "retention_continual_learning_cells": 27,
        },
        "parent_identities": parents,
        "resource_budget": dict(W09_RESOURCE_BUDGET),
        "stage_inventory": {
            "train_pack_count": 34,
            "dev_pack_count": 1,
            "held_out_pack_count": 37,
            "evaluator_pack_count": 37,
            "future_pack_count": 0,
            "train_pack_keys": list(visibility.get("train_pack_keys", [])),
            "dev_pack_keys": list(visibility.get("dev_pack_keys", [])),
            "held_out_pack_keys": list(visibility.get("held_out_pack_keys", [])),
            "evaluator_pack_keys": list(visibility.get("evaluator_pack_keys", [])),
            "future_pack_keys": [],
        },
        "stop_states": list(W09_STOP_STATES),
        "subtask_order": list(W09_SUBTASK_ORDER),
        "transaction": {
            "allowed_worker_counts": list(W09_ALLOWED_WORKER_COUNTS),
            "logical_shard_count": 16,
            "failure_point_keys": list(W09_FAILURE_POINT_KEYS),
            "base_fence_required": 1,
            "fresh_resume_equivalent": 1,
            "run_id_policy": "NEW_POSITIVE_INTEGER_REQUIRED",
        },
    }


def canonical_w09_authority_bytes(value: dict[str, Any]) -> bytes:
    """返回 W-09 authority 的规范 JSON 字节。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_w09_authority(value: object) -> dict[str, Any]:
    """验证 authority 字段、状态和父身份格式。"""
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise W09AuthorityError("W-09 authority fields drifted")
    if value.get("artifact_kind") != W09_AUTHORITY_KIND or value.get("artifact_version") != W09_AUTHORITY_VERSION:
        raise W09AuthorityError("W-09 authority version drifted")
    if tuple(value.get("dimension_keys", ())) != W09_DIMENSION_KEYS:
        raise W09AuthorityError("W-09 dimensions drifted")
    if tuple(value.get("ablation_keys", ())) != W09_ABLATION_KEYS:
        raise W09AuthorityError("W-09 ablations drifted")
    if value.get("execution_state") != W09_ZERO_EXECUTION_STATE:
        raise W09AuthorityError("W-09 zero execution state drifted")
    if value.get("stage_inventory", {}).get("future_pack_count") != 0:
        raise W09AuthorityError("W-09 future inventory is not empty")
    if value.get("transaction", {}).get("allowed_worker_counts") != list(W09_ALLOWED_WORKER_COUNTS):
        raise W09AuthorityError("W-09 worker contract drifted")
    for item in value.get("parent_identities", []):
        if not isinstance(item, dict) or len(item.get("sha256", "")) != 64 or type(item.get("size_bytes")) is not int:
            raise W09AuthorityError("W-09 parent identity is invalid")
        _safe_relative(item.get("relative_path"))
    return value


def publish_w09_authority(repository_root: str | Path) -> str:
    """排他 append-only 发布 W-09 authority。"""
    repository = Path(repository_root).resolve()
    value = validate_w09_authority(build_w09_authority(repository))
    payload = canonical_w09_authority_bytes(value)
    destination = repository / W09_AUTHORITY_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W09AuthorityError("W-09 authority is append-only and already exists") from error
    return hashlib.sha256(payload).hexdigest()


def read_w09_authority(repository_root: str | Path) -> dict[str, Any]:
    """规范回读 authority 并与现场父身份逐字比较。"""
    repository = Path(repository_root).resolve()
    path = repository / W09_AUTHORITY_RELATIVE_PATH
    value = _canonical_object(path)
    validate_w09_authority(value)
    if canonical_w09_authority_bytes(value) != path.read_bytes():
        raise W09AuthorityError("W-09 authority is not canonical JSON")
    if value != build_w09_authority(repository):
        raise W09AuthorityError("W-09 authority no longer matches live parents")
    return value


__all__ = [
    "W09_ABLATION_KEYS",
    "W09_ALLOWED_WORKER_COUNTS",
    "W09_AUTHORITY_RELATIVE_PATH",
    "W09AuthorityError",
    "W09_DIMENSION_KEYS",
    "W09_FAILURE_POINT_KEYS",
    "W09_RESOURCE_BUDGET",
    "W09_STOP_STATES",
    "build_w09_authority",
    "canonical_w09_authority_bytes",
    "publish_w09_authority",
    "read_w09_authority",
    "validate_w09_authority",
]
