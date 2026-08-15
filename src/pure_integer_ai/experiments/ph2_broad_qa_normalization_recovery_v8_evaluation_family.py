"""冻结 recovery-v8 candidate、live code 与 VLC formal family。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate_pack import (
    read_normalization_recovery_v8_candidate_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
    read_normalization_recovery_v8_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes, canonical_json_line


NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_V1")
NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_STATUS = (
    "CANDIDATE_CODE_DENOMINATOR_FROZEN_ZERO_VLC_LABEL_READS")

NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS = (
    "protocol_dir",
    "pack_dir",
    "audit_dir",
    "v7_commitment_dir",
    "evaluation_commitment_dir",
    "candidate_dir",
    "vlc_source_pack_dir",
)
NORMALIZATION_RECOVERY_V8_EVALUATION_IDENTITY_ARGUMENTS = (
    "expected_protocol_manifest_sha256",
    "expected_pack_manifest_sha256",
    "expected_audit_manifest_sha256",
    "expected_v7_commitment_manifest_sha256",
    "expected_evaluation_commitment_manifest_sha256",
    "expected_candidate_manifest_sha256",
    "expected_vlc_source_manifest_sha256",
)

NORMALIZATION_RECOVERY_V8_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_learner_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_vlc_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_vlc_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_candidate.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_candidate_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_evaluation_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_evaluation_runner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_evaluator.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_label_materialization.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_learner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_learning_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_learning_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_rule_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_training_audit.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_training_audit_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_training_protocol.py",
)


def _sha256(payload: bytes) -> str:
    """返回 code、family 或 manifest identity。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v8 family {label} 非法")
    return value


def require_normalization_recovery_v8_k_root(value: str | Path) -> Path:
    """要求显式 formal 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 family run root 必须在 K 盘")
    return root


def normalization_recovery_v8_path_within(
        root: Path, value: str | Path, *, label: str,
        ) -> Path:
    """解析路径并拒绝逃出唯一 K 盘工作根。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v8 family {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或互为祖先。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _git_output(repository: Path, *arguments: str) -> str:
    """执行只读 Git 命令并返回严格 UTF-8 文本。"""
    try:
        result = subprocess.run(
            ("git", *arguments), cwd=repository,
            check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise BroadQaExternalDataError("v8 family Git identity 不可读") from error
    return result.stdout.strip()


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str, str]:
    """绑定 clean tracked Git HEAD 与全部承重代码文件。"""
    if not repository.is_dir() or not (repository / ".git").exists():
        raise BroadQaExternalDataError("v8 family repository root 非法")
    if _git_output(repository, "status", "--porcelain", "--untracked-files=all"):
        raise BroadQaExternalDataError("v8 family repository 必须 clean")
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(head) != 40:
        raise BroadQaExternalDataError("v8 family Git HEAD 漂移")
    records = []
    for relative in NORMALIZATION_RECOVERY_V8_EVALUATION_CODE_FILES:
        if _git_output(repository, "ls-files", "--error-unmatch", relative) != relative:
            raise BroadQaExternalDataError("v8 family bearing code 未跟踪")
        path = repository / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError("v8 family bearing code 不可读") from error
        records.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    return records, _sha256(canonical_json_bytes(records)), head


def _source_manifest_only(
        directory: Path, *, expected_sha256: str,
        ) -> dict[str, object]:
    """只读 VLC manifest，绝不打开 archive 或 identity roster。"""
    expected = _sha_value(expected_sha256, label="VLC source SHA")
    try:
        encoded = (directory / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 family VLC manifest 不可读") from error
    if (_sha256(encoded) != expected or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("evaluation_state", {}).get(
                "formal_label_jsonl_materialized") != 0
            or stored.get("evaluation_state", {}).get(
                "formal_evaluation_run_count") != 0):
        raise BroadQaExternalDataError("v8 family VLC manifest 漂移")
    return {**stored, "manifest_sha256": expected}


def _candidate_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择 candidate strict reader 所需参数。"""
    names = (
        "protocol_dir", "expected_protocol_manifest_sha256",
        "pack_dir", "expected_pack_manifest_sha256",
        "audit_dir", "expected_audit_manifest_sha256",
        "v7_commitment_dir", "expected_v7_commitment_manifest_sha256",
        "evaluation_commitment_dir",
        "expected_evaluation_commitment_manifest_sha256",
        "expected_candidate_manifest_sha256",
    )
    return {name: arguments[name] for name in names}


def build_normalization_recovery_v8_evaluation_family_freeze(
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, object]]:
    """重算 candidate/code/commitment 并构造零 VLC label read family。"""
    repository = Path(arguments["repository_root"]).resolve()
    candidate_manifest, candidate, preflight = (
        read_normalization_recovery_v8_candidate_pack(
            arguments["candidate_dir"], **_candidate_arguments(arguments)))
    commitment = read_normalization_recovery_v8_evaluation_commitment(
        arguments["evaluation_commitment_dir"],
        v7_commitment_dir=arguments["v7_commitment_dir"],
        expected_v7_commitment_manifest_sha256=arguments[
            "expected_v7_commitment_manifest_sha256"],
        expected_manifest_sha256=arguments[
            "expected_evaluation_commitment_manifest_sha256"])
    source = _source_manifest_only(
        Path(arguments["vlc_source_pack_dir"]).resolve(),
        expected_sha256=str(arguments["expected_vlc_source_manifest_sha256"]))
    code_files, code_sha, head = _code_identity(repository)
    if (candidate.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or candidate_manifest.get("candidate_program_sha256")
            != candidate.get("candidate_program_sha256")
            or candidate_manifest.get("preflight_failure_count") != 0
            or preflight.get("failure_count") != 0
            or commitment.get("source_exclusion", {}).get(
                "excluded_source_pack_manifest_sha256")
            != source.get("manifest_sha256")):
        raise BroadQaExternalDataError("v8 family candidate/source lineage 漂移")
    core = {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "code_files": code_files,
        "code_identity_sha256": code_sha,
        "denominator": commitment["denominator"],
        "dimensions": commitment["dimensions"],
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_or_reserve_payload_read_count": 0,
        "evaluation_run_count": 0,
        "format_version": 1,
        "formal_contract": commitment["formal_contract"],
        "git_head": head,
        "individual_label_read_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
        "vlc_source_manifest_read_count": 1,
        "vlc_source_manifest_sha256": source["manifest_sha256"],
        "vlc_source_non_manifest_read_count": 0,
    }
    freeze = {**core, "family_commitment_sha256": _sha256(
        canonical_json_bytes(core))}
    return freeze, candidate_manifest, candidate, commitment


def publish_normalization_recovery_v8_evaluation_family_freeze(
        *, run_root: str | Path, target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 v8 family freeze，且不读取 VLC payload。"""
    root = require_normalization_recovery_v8_k_root(run_root)
    inputs = tuple(normalization_recovery_v8_path_within(
        root, arguments[name], label=name)
        for name in NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS)
    target = normalization_recovery_v8_path_within(
        root, target_dir, label="target_dir")
    if (target.exists() or any(not path.is_dir() for path in inputs)
            or any(_overlap(left, right)
                   for index, left in enumerate((*inputs, target))
                   for right in (*inputs, target)[index + 1:])):
        raise BroadQaExternalDataError("v8 family artifact path 非法")
    freeze, _manifest, _candidate, _commitment = (
        build_normalization_recovery_v8_evaluation_family_freeze(**arguments))
    target.mkdir()
    path = target / "family-freeze.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(freeze))
    return {**freeze, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_evaluation_family_freeze(
        family_dir: str | Path, **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, object]]:
    """严格回读 family，并重算 live candidate、commitment 与 code。"""
    root = Path(family_dir).resolve()
    expected, candidate_manifest, candidate, commitment = (
        build_normalization_recovery_v8_evaluation_family_freeze(**arguments))
    try:
        encoded = (root / "family-freeze.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 family freeze 不可读") from error
    if (not isinstance(stored, dict) or canonical_json_line(stored) != encoded
            or not strict_json_equal(stored, expected)):
        raise BroadQaExternalDataError("v8 family freeze 与 live identity 漂移")
    return ({**stored, "manifest_sha256": _sha256(encoded)},
            candidate_manifest, candidate, commitment)


__all__ = [
    "NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS",
    "NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V8_EVALUATION_FAMILY_STATUS",
    "NORMALIZATION_RECOVERY_V8_EVALUATION_IDENTITY_ARGUMENTS",
    "build_normalization_recovery_v8_evaluation_family_freeze",
    "normalization_recovery_v8_path_within",
    "publish_normalization_recovery_v8_evaluation_family_freeze",
    "read_normalization_recovery_v8_evaluation_family_freeze",
    "require_normalization_recovery_v8_k_root",
]
