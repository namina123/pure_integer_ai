"""冻结 recovery-v4 candidate、live code 与标签盲 formal family。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    read_normalization_recovery_v3_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_candidate_pack import (
    read_normalization_recovery_v4_candidate_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_FREEZE_V1")
NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_STATUS = (
    "FROZEN_CANDIDATE_CODE_AND_DENOMINATOR_LABELS_UNREAD")

NORMALIZATION_RECOVERY_V4_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_learning_checkpoint.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_learning_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_learner_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_evaluation_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_training_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_learning_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_learner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_candidate_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_candidate_compile.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_candidate_execution.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_godot_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_vscode_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_vscode_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_training_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_phrase_learning.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_learning_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_learning_evidence.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_learning_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_learner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_phrase_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_training_audit_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_training_audit.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_candidate.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_candidate_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_reserve_materialization.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_evaluator.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_evaluation_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_evaluation_runner.py",
    "src/pure_integer_ai/experiments/train_execution.py",
)


def _sha256(payload: bytes) -> str:
    """返回 code、candidate 或 family commitment SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def require_normalization_recovery_v4_k_root(value: str | Path) -> Path:
    """要求 formal family 工作根是显式存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "recovery v4 evaluation run root 必须是 K 盘目录")
    return root


def normalization_recovery_v4_path_within(
        root: Path,
        value: str | Path,
        *,
        label: str,
        ) -> Path:
    """解析 formal 输入输出并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结 candidate、materializer、evaluator、runner 与直接依赖字节。"""
    values = []
    for relative in NORMALIZATION_RECOVERY_V4_EVALUATION_CODE_FILES:
        path = (repository / Path(*relative.split("/"))).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise BroadQaExternalDataError(
                f"recovery v4 code file 缺失: {relative}")
        payload = path.read_bytes()
        values.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    return values, _sha256(canonical_json_bytes(values))


def _candidate_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择 candidate strict reader 的物理路径与外部 identity。"""
    names = (
        "prior_evaluation_protocol_dir",
        "expected_prior_evaluation_manifest_sha256",
        "base_training_protocol_dir",
        "expected_base_training_manifest_sha256",
        "base_rule_pack_dir",
        "expected_base_rule_pack_manifest_sha256",
        "v4_training_protocol_dir",
        "expected_v4_training_manifest_sha256",
        "v4_rule_pack_dir",
        "expected_v4_rule_pack_manifest_sha256",
        "v4_training_audit_dir",
        "expected_v4_training_audit_manifest_sha256",
        "evaluation_commitment_dir",
        "expected_evaluation_commitment_manifest_sha256",
    )
    missing = [name for name in names if name not in arguments]
    if missing:
        raise BroadQaExternalDataError(
            f"recovery v4 family 缺少 candidate 参数: {missing[0]}")
    return {name: arguments[name] for name in names}


def _validate_paths(
        root: Path,
        arguments: dict[str, object],
        ) -> dict[str, object]:
    """核验 family 的全部目录和 SHA identity。"""
    values = dict(arguments)
    for name, value in list(values.items()):
        if name.endswith("_dir"):
            path = normalization_recovery_v4_path_within(
                root, value, label=name)
            if not path.is_dir():
                raise BroadQaExternalDataError(
                    f"recovery v4 family 输入目录不存在: {name}")
            values[name] = path
        elif name.startswith("expected_") and name.endswith("_sha256"):
            values[name] = _sha_value(value, label=name)
    return values


def build_normalization_recovery_v4_evaluation_family_freeze(
        *,
        repository_root: str | Path,
        candidate_dir: str | Path,
        expected_candidate_manifest_sha256: str,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object],
                   dict[str, object]]:
    """重算 candidate/code/commitment，构造零 label read family freeze。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise BroadQaExternalDataError("recovery v4 repository root 非法")
    candidate_manifest, program, _profile = (
        read_normalization_recovery_v4_candidate_pack(
            candidate_dir,
            expected_candidate_manifest_sha256=(
                expected_candidate_manifest_sha256),
            **_candidate_arguments(arguments),
        ))
    commitment = read_normalization_recovery_v3_evaluation_commitment(
        arguments["evaluation_commitment_dir"],
        prior_evaluation_protocol_dir=arguments[
            "prior_evaluation_protocol_dir"],
        expected_manifest_sha256=arguments[
            "expected_evaluation_commitment_manifest_sha256"],
    )
    if (program["evaluation_commitment_manifest_sha256"]
            != commitment["manifest_sha256"]):
        raise BroadQaExternalDataError(
            "recovery v4 family candidate/commitment 漂移")
    code_files, code_sha = _code_identity(repository)
    candidate_freeze = {
        "base_character_rule_count": len(program["base_character_rules"]),
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": program["candidate_program_sha256"],
        "conflict_count": len(program["conflicts"]),
        "phrase_program_sha256": program["phrase_program"]["program_sha256"],
        "transfer_profile_sha256": program["transfer_profile_sha256"],
        "v4_protocol_manifest_sha256": program[
            "v4_protocol_manifest_sha256"],
        "v4_rule_pack_manifest_sha256": program[
            "v4_rule_pack_manifest_sha256"],
        "v4_training_audit_manifest_sha256": program[
            "v4_training_audit_manifest_sha256"],
    }
    core = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_KIND,
        "candidate_freeze": candidate_freeze,
        "candidate_freeze_sha256": _sha256(
            canonical_json_bytes(candidate_freeze)),
        "code_files": code_files,
        "code_freeze_sha256": code_sha,
        "denominator": commitment["denominator"],
        "dimensions": commitment["dimensions"],
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_run_count": 0,
        "formal_contract": commitment["formal_contract"],
        "format_version": 1,
        "label_materialization_contract": {
            "allowed_after_unique_guard": 1,
            "allowed_before_unique_guard": 0,
            "candidate_result_based_reselection_allowed": 0,
            "entire_prior_reserve_required": 1,
            "source_pack_sha256": commitment["source_exclusion"][
                "excluded_source_pack_manifest_sha256"],
        },
        "mastery_claimed": 0,
        "prior_reserve_identity": commitment["prior_reserve_identity"],
        "production_enabled": 0,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    freeze = {**core, "family_commitment_sha256": _sha256(
        canonical_json_bytes(core))}
    return freeze, candidate_manifest, program, commitment


def publish_normalization_recovery_v4_evaluation_family_freeze(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 family freeze，且不读取 reserve identity/payload。"""
    root = require_normalization_recovery_v4_k_root(run_root)
    checked = _validate_paths(root, arguments)
    target = normalization_recovery_v4_path_within(
        root, target_dir, label="recovery v4 family target")
    if target.exists():
        raise BroadQaExternalDataError("recovery v4 family target 已存在")
    freeze, _candidate, _program, _commitment = (
        build_normalization_recovery_v4_evaluation_family_freeze(**checked))
    target.mkdir()
    path = target / "family-freeze.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(freeze))
    return {**freeze, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v4_evaluation_family_freeze(
        family_dir: str | Path,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object],
                   dict[str, object]]:
    """严格回读 family 并重算 live candidate、code 与 commitment。"""
    root = Path(family_dir).resolve()
    try:
        encoded = (root / "family-freeze.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("recovery v4 family freeze 不可读") from error
    expected, candidate, program, commitment = (
        build_normalization_recovery_v4_evaluation_family_freeze(**arguments))
    if (not isinstance(stored, dict) or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "recovery v4 family freeze 与 live identity 漂移")
    return ({**stored, "manifest_sha256": _sha256(encoded)},
            candidate, program, commitment)


__all__ = [
    "NORMALIZATION_RECOVERY_V4_EVALUATION_CODE_FILES",
    "NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V4_EVALUATION_FAMILY_STATUS",
    "build_normalization_recovery_v4_evaluation_family_freeze",
    "normalization_recovery_v4_path_within",
    "publish_normalization_recovery_v4_evaluation_family_freeze",
    "read_normalization_recovery_v4_evaluation_family_freeze",
    "require_normalization_recovery_v4_k_root",
]
