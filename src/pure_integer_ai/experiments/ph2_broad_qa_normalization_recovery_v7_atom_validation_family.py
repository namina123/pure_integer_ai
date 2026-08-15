"""冻结 Audacity atom-validation 的 public code、来源与一次性协议。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit import (
    NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment_v2 import (
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    AUDACITY_SOURCE_FAMILY,
    AUDACITY_SOURCE_POLICY_SCOPE,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "AUDACITY_ATOM_VALIDATION_FAMILY_FREEZE_V1")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_STATUS = (
    "AUDACITY_ATOM_VALIDATION_CODE_AND_FAMILY_FROZEN_"
    "ZH_CN_LABEL_UNREAD_NOT_RUN")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "AUDACITY_ATOM_VALIDATION_FAMILY_FREEZE_V2")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_STATUS = (
    "AUDACITY_ATOM_VALIDATION_CODE_FAMILY_AND_UNIQUE_PUBLICATION_FROZEN_"
    "ZH_CN_LABEL_UNREAD_NOT_RUN")

AUDACITY_ATOM_VALIDATION_SOURCE_MANIFEST_SHA256 = (
    "64cfb20d34aa4bb4597e84fd325dfcd8b86659602010a07bc9b99e94882e86d4")
AUDACITY_ATOM_IDENTIFIABILITY_MANIFEST_SHA256 = (
    "4dbe00153a8859f1f38c26ca652368facf9536b9f3b2cea6eb7356ee7626b343")
AUDACITY_ATOM_VALIDATION_COMMITMENT_V2_MANIFEST_SHA256 = (
    "528d0d85debd7bb1f991fc735c71d2497ac896ece91e8420c9bc280ca866fef7")
AUDACITY_ATOM_VALIDATION_FAMILY_V1_MANIFEST_SHA256 = (
    "aac1ad13a868c2948e5588157b8eec0b30670ae35e4ca42e7c58fbbb8e45d4eb")
AUDACITY_ATOM_VALIDATION_PUBLICATION_RELATIVE_PATH = (
    "normalization-recovery-v7-audacity-atom-validation-formal-v1")

AUDACITY_ATOM_VALIDATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_identifiability_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment_v2.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v7_atom_validation_runner.py",
)


def _sha256(payload: bytes) -> str:
    """返回 code、manifest 或 family identity。"""
    return hashlib.sha256(payload).hexdigest()


def _read_manifest_only(
        directory: str | Path,
        *,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """只读规范 manifest，不打开同目录其他文件。"""
    try:
        encoded = (Path(directory).resolve() / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"Audacity atom-validation family {label} 不可读") from error
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or _sha256(encoded) != expected_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            f"Audacity atom-validation family {label} identity 漂移")
    return {**stored, "manifest_sha256": expected_sha256}


def _validate_inputs(
        source: dict[str, object],
        atom: dict[str, object],
        commitment: dict[str, object],
        ) -> None:
    """核对三份 predecessor 的固定状态、分母与交叉 SHA。"""
    source_state = source.get("validation_state")
    source_summary = source.get("parser_summary")
    atom_summary = atom.get("summary")
    atom_scoring = atom_summary.get("identifiability", {}).get(
        "scoring", {}) if isinstance(atom_summary, dict) else {}
    commitment_inputs = commitment.get("inputs")
    denominator = commitment.get("denominator")
    gates = commitment.get("gates")
    if (source.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND
            or source.get("status")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS
            or source.get("source_family") != AUDACITY_SOURCE_FAMILY
            or source.get("source_policy_scope")
            != AUDACITY_SOURCE_POLICY_SCOPE
            or not isinstance(source_state, dict)
            or any(source_state.get(key) != 0 for key in (
                "candidate_or_runtime_read_count",
                "formal_label_jsonl_materialized", "validation_run_count"))
            or not isinstance(source_summary, dict)
            or source_summary.get("plain_pair_count") != 4404
            or atom.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND
            or atom.get("status")
            != "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME"
            or atom.get("candidate_family_formal_run_count") != 0
            or atom_scoring.get("outcome_counts")
            != {"EXACT": 2, "UNKNOWN": 12, "WRONG": 0}
            or commitment.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND
            or commitment.get("status")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS
            or not isinstance(commitment_inputs, dict)
            or commitment_inputs.get(
                "audacity_source_pack_manifest_sha256")
            != source["manifest_sha256"]
            or commitment_inputs.get(
                "atom_identifiability_manifest_sha256")
            != atom["manifest_sha256"]
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != 4404
            or denominator.get("source_family") != AUDACITY_SOURCE_FAMILY
            or not isinstance(gates, dict)
            or gates.get("authorized_changed_exact_output_count_min") != 1
            or gates.get("wrong_output_count_max") != 0):
        raise BroadQaExternalDataError(
            "Audacity atom-validation family predecessor boundary 漂移")


def _validate_family_v1(
        value: dict[str, object],
        *,
        source_sha256: str,
        atom_sha256: str,
        commitment_sha256: str,
        ) -> None:
    """核对未运行 v1，并只把它作为 superseded lineage。"""
    reads = value.get("validation_reads")
    inputs = value.get("inputs")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_STATUS
            or value.get("format_version") != 1
            or not isinstance(reads, dict)
            or reads.get("validation_run_count") != 0
            or reads.get("zh_cn_label_read_count") != 0
            or reads.get("audacity_identity_raw_or_translation_read_count")
            != 0
            or not isinstance(inputs, dict)
            or inputs.get("audacity_source_pack_manifest_sha256")
            != source_sha256
            or inputs.get("atom_identifiability_manifest_sha256")
            != atom_sha256
            or inputs.get("commitment_v2_manifest_sha256")
            != commitment_sha256):
        raise BroadQaExternalDataError(
            "Audacity atom-validation family v1 lineage 漂移")


def _git_text(repository: Path, *arguments: str) -> str:
    """执行只读 Git identity 命令。"""
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise BroadQaExternalDataError(
            "Audacity atom-validation public Git identity 不可读") from error
    return completed.stdout.strip()


def _repository_identity(repository: Path) -> dict[str, object]:
    """要求 tracked worktree clean 且 HEAD 已推送到 origin/master。"""
    head = _git_text(repository, "rev-parse", "HEAD")
    origin = _git_text(repository, "rev-parse", "origin/master")
    dirty = _git_text(
        repository, "status", "--porcelain", "--untracked-files=no")
    if (len(head) != 40 or head != origin or dirty):
        raise BroadQaExternalDataError(
            "Audacity atom-validation public Git 未冻结或未推送")
    return {
        "head_commit_sha1": head,
        "origin_master_commit_sha1": origin,
        "remote_origin_url": _git_text(
            repository, "config", "--get", "remote.origin.url"),
        "tracked_worktree_clean": 1,
    }


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结全部 proposal/authorization/scoring/runner bearing code。"""
    values = []
    for relative_path in AUDACITY_ATOM_VALIDATION_CODE_FILES:
        path = (repository / Path(*relative_path.split("/"))).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise BroadQaExternalDataError(
                "Audacity atom-validation bearing code 缺失")
        payload = path.read_bytes()
        values.append({
            "bytes": len(payload),
            "relative_path": relative_path,
            "sha256": _sha256(payload),
        })
    return values, _sha256(canonical_json_bytes(values))


def build_audacity_atom_validation_family_freeze(
        *,
        repository_root: str | Path,
        source_pack_dir: str | Path,
        expected_source_manifest_sha256: str,
        atom_audit_dir: str | Path,
        expected_atom_manifest_sha256: str,
        commitment_v2_dir: str | Path,
        expected_commitment_v2_manifest_sha256: str,
        family_v1_dir: str | Path,
        expected_family_v1_manifest_sha256: str,
        ) -> dict[str, object]:
    """从三份 manifest 与 live pushed code 构造零 label read family。"""
    if (expected_source_manifest_sha256
            != AUDACITY_ATOM_VALIDATION_SOURCE_MANIFEST_SHA256
            or expected_atom_manifest_sha256
            != AUDACITY_ATOM_IDENTIFIABILITY_MANIFEST_SHA256
            or expected_commitment_v2_manifest_sha256
            != AUDACITY_ATOM_VALIDATION_COMMITMENT_V2_MANIFEST_SHA256
            or expected_family_v1_manifest_sha256
            != AUDACITY_ATOM_VALIDATION_FAMILY_V1_MANIFEST_SHA256):
        raise BroadQaExternalDataError(
            "Audacity atom-validation family 非正式 predecessor SHA")
    source = _read_manifest_only(
        source_pack_dir, expected_sha256=expected_source_manifest_sha256,
        label="source manifest")
    atom = _read_manifest_only(
        atom_audit_dir, expected_sha256=expected_atom_manifest_sha256,
        label="atom manifest")
    commitment = _read_manifest_only(
        commitment_v2_dir,
        expected_sha256=expected_commitment_v2_manifest_sha256,
        label="commitment v2 manifest")
    _validate_inputs(source, atom, commitment)
    family_v1 = _read_manifest_only(
        family_v1_dir,
        expected_sha256=expected_family_v1_manifest_sha256,
        label="superseded family v1 manifest")
    _validate_family_v1(
        family_v1,
        source_sha256=source["manifest_sha256"],
        atom_sha256=atom["manifest_sha256"],
        commitment_sha256=commitment["manifest_sha256"])
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise BroadQaExternalDataError(
            "Audacity atom-validation repository root 非法")
    public_git = _repository_identity(repository)
    code_files, code_sha = _code_identity(repository)
    core = {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_KIND,
        "authorization_protocol": {
            "changed_segment_requires_stable_multi_scalar_lexical_atom": 1,
            "held_output_or_held_derived_plan_allowed": 0,
            "indexed_and_independent_reference_must_match": 1,
            "marked_unimorph_morphology_outcome": "DEFER",
            "minimum_independent_train_family_consensus": 2,
            "official_source_layout_preservation_required": 1,
            "opencc_only_for_remaining_orthographic_atoms": 1,
            "unchanged_segment_must_remain_unchanged": 1,
            "uncovered_changed_segment_scalar_requires_stable_copy": 1,
            "zh_cn_label_read_before_authorization_commitment_allowed": 0,
        },
        "code_files": code_files,
        "code_freeze_sha256": code_sha,
        "denominator": commitment["denominator"],
        "format_version": 2,
        "gates": commitment["gates"],
        "inputs": {
            "atom_identifiability_manifest_sha256": atom["manifest_sha256"],
            "audacity_source_pack_manifest_sha256": source[
                "manifest_sha256"],
            "commitment_v2_manifest_sha256": commitment[
                "manifest_sha256"],
            "superseded_family_v1_manifest_sha256": family_v1[
                "manifest_sha256"],
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "public_git": public_git,
        "publication_contract": {
            "alternate_publication_path_allowed": 0,
            "relative_path": AUDACITY_ATOM_VALIDATION_PUBLICATION_RELATIVE_PATH,
            "run_ordinal": 1,
            "v1_per_target_guard_superseded": 1,
        },
        "runtime_program_published": 0,
        "scoring_protocol": {
            "all_denominator_rows_receive_one_of": [
                "EXACT", "UNKNOWN", "WRONG"],
            "authorization_freeze_required_before_zh_cn_open": 1,
            "identity_only_exact_satisfies_transfer_pass": 0,
            "overall_precedence": "FAIL_DOMINATES_NE_DOMINATES_PASS",
        },
        "status": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
        "validation_runner_frozen": 1,
        "validation_reads": {
            "atom_manifest_read_count": 1,
            "audacity_identity_raw_or_translation_read_count": 0,
            "commitment_v2_manifest_read_count": 1,
            "source_pack_manifest_read_count": 1,
            "superseded_family_v1_manifest_read_count": 1,
            "validation_run_count": 0,
            "zh_cn_label_read_count": 0,
        },
    }
    return {**core, "family_commitment_sha256": _sha256(
        canonical_json_bytes(core))}


def _require_k_root(value: str | Path) -> Path:
    """要求 family 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "Audacity atom-validation family root 必须是 K 盘目录")
    return root


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或包含。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def publish_audacity_atom_validation_family_freeze(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 family，且只读取三份 predecessor manifest。"""
    root = _require_k_root(run_root)
    inputs = tuple(Path(arguments[name]).resolve() for name in (
        "source_pack_dir", "atom_audit_dir", "commitment_v2_dir",
        "family_v1_dir"))
    target = Path(target_dir).resolve()
    if (any(not path.is_dir() or not path.is_relative_to(root)
            for path in inputs)
            or not target.is_relative_to(root) or target.exists()
            or any(_overlap(target, path) for path in inputs)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation family path 非法")
    freeze = build_audacity_atom_validation_family_freeze(**arguments)
    target.mkdir()
    encoded = canonical_json_line(freeze)
    with (target / "manifest.json").open("xb") as handle:
        handle.write(encoded)
    return {**freeze, "manifest_sha256": _sha256(encoded)}


def read_audacity_atom_validation_family_freeze(
        family_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        **arguments: object,
        ) -> dict[str, object]:
    """严格回读 family，并重算 pushed Git 与 bearing code identity。"""
    stored = _read_manifest_only(
        family_dir, expected_sha256=expected_manifest_sha256,
        label="family manifest")
    expected = build_audacity_atom_validation_family_freeze(**arguments)
    comparable = {key: value for key, value in stored.items()
                  if key != "manifest_sha256"}
    if not strict_json_equal(comparable, expected):
        raise BroadQaExternalDataError(
            "Audacity atom-validation family fields/code 漂移")
    return stored


__all__ = [
    "AUDACITY_ATOM_IDENTIFIABILITY_MANIFEST_SHA256",
    "AUDACITY_ATOM_VALIDATION_CODE_FILES",
    "AUDACITY_ATOM_VALIDATION_COMMITMENT_V2_MANIFEST_SHA256",
    "AUDACITY_ATOM_VALIDATION_FAMILY_V1_MANIFEST_SHA256",
    "AUDACITY_ATOM_VALIDATION_PUBLICATION_RELATIVE_PATH",
    "AUDACITY_ATOM_VALIDATION_SOURCE_MANIFEST_SHA256",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_STATUS",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_KIND",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_STATUS",
    "build_audacity_atom_validation_family_freeze",
    "publish_audacity_atom_validation_family_freeze",
    "read_audacity_atom_validation_family_freeze",
]
