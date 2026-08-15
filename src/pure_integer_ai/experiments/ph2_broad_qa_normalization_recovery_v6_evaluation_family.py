"""冻结 recovery-v6 candidate v2、live code 与 Qt formal family。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    read_normalization_recovery_v5_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate_clone import (
    clone_normalization_recovery_v6_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate_pack import (
    read_normalization_recovery_v6_candidate_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256 = (
    "f85a62dc022dce2950986f3bde56b09797fdfc65ffb24a5e1925fcfea4954ba2")
NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_FREEZE_V1")
NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_STATUS = (
    "FROZEN_V2_CANDIDATE_CODE_AND_QT_DENOMINATOR_LABELS_UNREAD")

NORMALIZATION_RECOVERY_V6_EVALUATION_DATA_ARGUMENTS = (
    "candidate_dir",
    "protocol_dir",
    "predecessor_pack_dir",
    "pack_dir",
    "audit_dir",
    "qt_source_pack_dir",
    "evaluation_commitment_dir",
)
NORMALIZATION_RECOVERY_V6_EVALUATION_IDENTITY_ARGUMENTS = (
    "expected_candidate_manifest_sha256",
    "expected_protocol_manifest_sha256",
    "expected_predecessor_pack_manifest_sha256",
    "expected_pack_manifest_sha256",
    "expected_audit_manifest_sha256",
    "expected_qt_source_manifest_sha256",
    "expected_evaluation_commitment_manifest_sha256",
)

NORMALIZATION_RECOVERY_V6_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_formal_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_index.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_joint_eval.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_learner_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_contrastive_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_phrase_learning.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_evaluation_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_godot_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_vscode_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_vscode_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v4_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_libreoffice_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_libreoffice_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_qt_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_qt_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_failure_profile.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_failure_profile_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_learning_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_learning_evidence.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_learning_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_training_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_training_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_learner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_phrase_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_training_audit_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_training_audit.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_successor_simulation_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_successor_simulation.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_learning_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_learning_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_phrase_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_training_audit_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_training_audit.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_candidate.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_candidate_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_candidate_clone.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_label_materialization.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_evaluator.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_evaluation_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v6_evaluation_runner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_query.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_question_slots.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_selection.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_alignment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_dossier.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_decision.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_feasibility.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_learning_checkpoint.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_learning_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_review.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_training.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_training_census.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_inference_training_dossier.py",
    "src/pure_integer_ai/experiments/ph2_dataset_core.py",
    "src/pure_integer_ai/experiments/ph2_dataset_manifest.py",
    "src/pure_integer_ai/experiments/ph2_dataset_owner_records.py",
    "src/pure_integer_ai/experiments/ph2_dataset_records.py",
    "src/pure_integer_ai/experiments/ph2_mediawiki_multistream_adapter.py",
    "src/pure_integer_ai/experiments/ph2_mediawiki_snapshot.py",
    "src/pure_integer_ai/experiments/v02_run_store.py",
)


def _sha256(payload: bytes) -> str:
    """返回 code、candidate 或 family identity。"""
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
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def require_normalization_recovery_v6_k_root(value: str | Path) -> Path:
    """要求显式 formal 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("recovery v6 formal run root 必须是 K 盘目录")
    return root


def normalization_recovery_v6_path_within(
        root: Path,
        value: str | Path,
        *,
        label: str,
        ) -> Path:
    """解析 formal 路径并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_normalization_recovery_v6_evaluation_paths(
        root: Path,
        arguments: dict[str, object],
        ) -> dict[str, Path]:
    """核验全部 family 输入目录、SHA 与物理隔离。"""
    paths = {}
    for name in NORMALIZATION_RECOVERY_V6_EVALUATION_DATA_ARGUMENTS:
        if name not in arguments:
            raise BroadQaExternalDataError(f"v6 family 缺少路径参数: {name}")
        path = normalization_recovery_v6_path_within(
            root, arguments[name], label=name)
        if not path.is_dir():
            raise BroadQaExternalDataError(f"v6 family 输入目录不存在: {name}")
        paths[name] = path
    if any(_overlap(left, right)
           for index, left in enumerate(paths.values())
           for right in tuple(paths.values())[index + 1:]):
        raise BroadQaExternalDataError("v6 family artifact 根混淆")
    for name in NORMALIZATION_RECOVERY_V6_EVALUATION_IDENTITY_ARGUMENTS:
        if name not in arguments:
            raise BroadQaExternalDataError(f"v6 family 缺少 identity: {name}")
        _sha_value(arguments[name], label=name)
    return paths


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结 candidate、Qt parser、evaluator、runner 与直接依赖字节。"""
    values = []
    for relative in NORMALIZATION_RECOVERY_V6_EVALUATION_CODE_FILES:
        path = (repository / Path(*relative.split("/"))).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise BroadQaExternalDataError(f"v6 evaluation code file 缺失: {relative}")
        payload = path.read_bytes()
        values.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    return values, _sha256(canonical_json_bytes(values))


def _candidate_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择 candidate strict reader 所需冻结参数。"""
    names = tuple(
        name for name in (
            *NORMALIZATION_RECOVERY_V6_EVALUATION_DATA_ARGUMENTS,
            *NORMALIZATION_RECOVERY_V6_EVALUATION_IDENTITY_ARGUMENTS,
        ) if name not in {"candidate_dir", "expected_candidate_manifest_sha256"})
    return {name: arguments[name] for name in names}


def build_normalization_recovery_v6_evaluation_family_freeze(
        *,
        repository_root: str | Path,
        candidate_dir: str | Path,
        expected_candidate_manifest_sha256: str,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object],
                   dict[str, object]]:
    """重算 v2 candidate/code/commitment，构造零 Qt label read family。"""
    candidate_sha = _sha_value(
        expected_candidate_manifest_sha256, label="v6 family candidate")
    if candidate_sha != NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256:
        raise BroadQaExternalDataError("v6 family 只接受正式 candidate v2")
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise BroadQaExternalDataError("v6 family repository root 非法")
    candidate_manifest, candidate, candidate_preflight = (
        read_normalization_recovery_v6_candidate_pack(
            candidate_dir,
            expected_candidate_manifest_sha256=candidate_sha,
            **_candidate_arguments(arguments),
        ))
    commitment = read_normalization_recovery_v5_evaluation_commitment(
        arguments["evaluation_commitment_dir"],
        qt_source_pack_dir=arguments["qt_source_pack_dir"],
        expected_qt_source_manifest_sha256=arguments[
            "expected_qt_source_manifest_sha256"],
        expected_manifest_sha256=arguments[
            "expected_evaluation_commitment_manifest_sha256"],
    )
    clone, clone_identity = clone_normalization_recovery_v6_candidate(
        candidate, candidate_manifest_sha256=candidate_sha)
    if (candidate_manifest.get("candidate_program_sha256")
            != clone.get("candidate_program_sha256")
            or clone.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or candidate_preflight.get("result_rows_sha256")
            != clone_identity["preflight_result_rows_sha256"]):
        raise BroadQaExternalDataError("v6 family candidate/clone/commitment 漂移")
    code_files, code_sha = _code_identity(repository)
    candidate_freeze = {
        **clone_identity,
        "candidate_pack_status": candidate_manifest["status"],
        "phrase_program_sha256": clone["phrase_program_sha256"],
        "preflight_case_count": candidate_preflight["case_count"],
        "preflight_failure_count": candidate_preflight["failure_count"],
        "v6_rule_pack_manifest_sha256": clone[
            "v6_rule_pack_manifest_sha256"],
        "v6_training_audit_manifest_sha256": clone[
            "v6_training_audit_manifest_sha256"],
    }
    core = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_KIND,
        "candidate_freeze": candidate_freeze,
        "candidate_freeze_sha256": _sha256(canonical_json_bytes(
            candidate_freeze)),
        "code_files": code_files,
        "code_freeze_sha256": code_sha,
        "denominator": commitment["denominator"],
        "dimensions": commitment["dimensions"],
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_or_reserve_payload_read_count": 0,
        "evaluation_run_count": 0,
        "formal_contract": commitment["formal_contract"],
        "format_version": 1,
        "label_materialization_contract": {
            "allowed_after_unique_guard": 1,
            "allowed_before_unique_guard": 0,
            "candidate_result_based_reselection_allowed": 0,
            "entire_qt_inventory_required": 1,
            "qt_source_pack_manifest_sha256": commitment[
                "source_exclusion"]["excluded_source_pack_manifest_sha256"],
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "qt_source_manifest_only_read_count": 2,
        "qt_source_non_manifest_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    freeze = {**core, "family_commitment_sha256": _sha256(
        canonical_json_bytes(core))}
    return freeze, candidate_manifest, clone, commitment


def publish_normalization_recovery_v6_evaluation_family_freeze(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 v6 family freeze，且不读取 Qt non-manifest。"""
    root = require_normalization_recovery_v6_k_root(run_root)
    paths = validate_normalization_recovery_v6_evaluation_paths(root, arguments)
    target = normalization_recovery_v6_path_within(
        root, target_dir, label="v6 family target")
    if target.exists() or any(_overlap(target, path) for path in paths.values()):
        raise BroadQaExternalDataError("v6 family target 已存在或 artifact 混淆")
    freeze, _candidate, _clone, _commitment = (
        build_normalization_recovery_v6_evaluation_family_freeze(**arguments))
    target.mkdir(parents=True)
    path = target / "family-freeze.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(freeze))
    return {**freeze, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v6_evaluation_family_freeze(
        family_dir: str | Path,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object],
                   dict[str, object]]:
    """严格回读 family 并重算 live candidate、clone、code 与 commitment。"""
    root = Path(family_dir).resolve()
    try:
        encoded = (root / "family-freeze.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 family freeze 不可读") from error
    expected, candidate, clone, commitment = (
        build_normalization_recovery_v6_evaluation_family_freeze(**arguments))
    if (not isinstance(stored, dict) or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError("v6 family freeze 与 live identity 漂移")
    return ({**stored, "manifest_sha256": _sha256(encoded)},
            candidate, clone, commitment)


def main(argv: list[str] | None = None) -> int:
    """不可覆盖发布或严格回读 v6 evaluation family。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("publish", "read"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--target-dir", required=True)
    for name in NORMALIZATION_RECOVERY_V6_EVALUATION_DATA_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in NORMALIZATION_RECOVERY_V6_EVALUATION_IDENTITY_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    run_root = values.pop("run_root")
    target = values.pop("target_dir")
    root = require_normalization_recovery_v6_k_root(run_root)
    validate_normalization_recovery_v6_evaluation_paths(root, values)
    target = normalization_recovery_v6_path_within(
        root, target, label="v6 family target")
    if command == "publish":
        report = publish_normalization_recovery_v6_evaluation_family_freeze(
            run_root=root, target_dir=target, **values)
    else:
        report = read_normalization_recovery_v6_evaluation_family_freeze(
            target, **values)[0]
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_CODE_FILES",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_DATA_ARGUMENTS",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_FAMILY_STATUS",
    "NORMALIZATION_RECOVERY_V6_EVALUATION_IDENTITY_ARGUMENTS",
    "build_normalization_recovery_v6_evaluation_family_freeze",
    "normalization_recovery_v6_path_within",
    "publish_normalization_recovery_v6_evaluation_family_freeze",
    "read_normalization_recovery_v6_evaluation_family_freeze",
    "require_normalization_recovery_v6_k_root",
    "validate_normalization_recovery_v6_evaluation_paths",
]
