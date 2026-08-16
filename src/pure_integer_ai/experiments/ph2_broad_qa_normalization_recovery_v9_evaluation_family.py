"""冻结 recovery-v9 candidate、live code与GIMP formal family。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_candidate_pack import (
    read_normalization_recovery_v9_candidate_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
    read_normalization_recovery_v9_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate import (
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND,
    NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS,
    V9_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_V1")
NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_STATUS = (
    "CANDIDATE_CODE_DENOMINATOR_FROZEN_ZERO_GIMP_LABEL_READS")

NORMALIZATION_RECOVERY_V9_EVALUATION_DATA_ARGUMENTS = (
    "base_candidate_dir",
    "evaluation_commitment_dir",
    "candidate_dir",
    "gimp_source_pack_dir",
    "runtime_gate_dir",
)
NORMALIZATION_RECOVERY_V9_EVALUATION_IDENTITY_ARGUMENTS = (
    "expected_evaluation_commitment_manifest_sha256",
    "expected_candidate_manifest_sha256",
    "expected_gimp_source_manifest_sha256",
    "expected_runtime_gate_sha256",
)

NORMALIZATION_RECOVERY_V9_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_candidate.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v8_evaluator.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_candidate_pack.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_evaluation_commitment.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_evaluation_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_evaluation_runner.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_evaluator.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_gettext_source_records.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_label_materialization.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_runtime_gate.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_runtime_gate_reader.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_normalization_recovery_v9_source_pack.py",
)


def _sha256(payload: bytes) -> str:
    """返回code、family或manifest identity。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v9 family {label}非法")
    return value


def require_normalization_recovery_v9_k_root(value: str | Path) -> Path:
    """要求显式formal工作根位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 family run root必须在K盘")
    return root


def normalization_recovery_v9_path_within(
        root: Path, value: str | Path, *, label: str,
        ) -> Path:
    """解析路径并拒绝逃出唯一K盘工作根。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v9 family {label}越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _git_output(repository: Path, *arguments: str) -> str:
    """执行只读Git命令并返回严格UTF-8文本。"""
    try:
        result = subprocess.run(
            ("git", *arguments), cwd=repository,
            check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise BroadQaExternalDataError("v9 family Git identity不可读") from error
    return result.stdout.strip()


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str, str]:
    """绑定clean tracked Git HEAD与全部承重代码文件。"""
    if not repository.is_dir() or not (repository / ".git").exists():
        raise BroadQaExternalDataError("v9 family repository root非法")
    if _git_output(repository, "status", "--porcelain", "--untracked-files=all"):
        raise BroadQaExternalDataError("v9 family repository必须clean")
    head = _git_output(repository, "rev-parse", "HEAD")
    if len(head) != 40:
        raise BroadQaExternalDataError("v9 family Git HEAD漂移")
    records = []
    for relative in NORMALIZATION_RECOVERY_V9_EVALUATION_CODE_FILES:
        if _git_output(
                repository, "ls-files", "--error-unmatch", relative) != relative:
            raise BroadQaExternalDataError("v9 family bearing code未跟踪")
        try:
            payload = (repository / relative).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "v9 family bearing code不可读") from error
        records.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    return records, _sha256(canonical_json_bytes(records)), head


def _canonical_manifest(
        path: Path, *, expected_sha256: str, label: str,
        ) -> dict[str, object]:
    """只读一份规范JSON，绝不打开同目录其他payload。"""
    expected = _sha_value(expected_sha256, label=label)
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"v9 family {label}不可读") from error
    if (_sha256(encoded) != expected or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(f"v9 family {label} identity漂移")
    return {**stored, "manifest_sha256": expected}


def _source_manifest_only(
        directory: Path, *, expected_sha256: str,
        ) -> dict[str, object]:
    """只读GIMP source manifest，不打开archive或identity roster。"""
    stored = _canonical_manifest(
        directory / "manifest.json", expected_sha256=expected_sha256,
        label="GIMP source manifest")
    if (stored.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"
            or stored.get("status")
            != "GIMP_RAW_AND_LABEL_FREE_IDENTITY_FROZEN_NOT_FORMAL"
            or stored.get("production_enabled") != 0
            or stored.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v9 family GIMP source state漂移")
    return stored


def _runtime_gate_only(
        directory: Path, *, expected_sha256: str,
        ) -> dict[str, object]:
    """只读runtime gate aggregate，不重跑candidate。"""
    stored = _canonical_manifest(
        directory / "runtime-gate.json", expected_sha256=expected_sha256,
        label="runtime gate")
    profile = stored.get("profile")
    if (stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND
            or stored.get("status")
            != NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS
            or not isinstance(profile, dict)
            or profile.get("gate_outcome") != "PASS"
            or stored.get("formal_guard_write_count") != 0
            or stored.get("formal_label_read_count") != 0):
        raise BroadQaExternalDataError("v9 family runtime gate state漂移")
    return stored


def _candidate_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择candidate strict reader所需参数。"""
    names = (
        "base_candidate_dir", "evaluation_commitment_dir",
        "gimp_source_pack_dir", "runtime_gate_dir",
        "expected_candidate_manifest_sha256",
    )
    values = {name: arguments[name] for name in names}
    values["source_pack_dir"] = values.pop("gimp_source_pack_dir")
    values["expected_manifest_sha256"] = values.pop(
        "expected_candidate_manifest_sha256")
    return values


def build_normalization_recovery_v9_evaluation_family_freeze(
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, object]]:
    """重算candidate/code/commitment并构造零GIMP label read family。"""
    repository = Path(arguments["repository_root"]).resolve()
    candidate_manifest, candidate, preflight = (
        read_normalization_recovery_v9_candidate_pack(
            arguments["candidate_dir"], **_candidate_arguments(arguments)))
    commitment = read_normalization_recovery_v9_evaluation_commitment(
        arguments["evaluation_commitment_dir"],
        source_pack_dir=arguments["gimp_source_pack_dir"],
        runtime_gate_dir=arguments["runtime_gate_dir"],
        expected_manifest_sha256=arguments[
            "expected_evaluation_commitment_manifest_sha256"])
    source = _source_manifest_only(
        Path(arguments["gimp_source_pack_dir"]).resolve(),
        expected_sha256=str(arguments["expected_gimp_source_manifest_sha256"]))
    gate = _runtime_gate_only(
        Path(arguments["runtime_gate_dir"]).resolve(),
        expected_sha256=str(arguments["expected_runtime_gate_sha256"]))
    code_files, code_sha, head = _code_identity(repository)
    if (candidate.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or candidate_manifest.get("candidate_program_sha256")
            != candidate.get("candidate_program_sha256")
            or candidate_manifest.get("preflight_failure_count") != 0
            or preflight.get("failure_count") != 0
            or commitment.get("denominator", {}).get(
                "source_pack_manifest_sha256") != source.get("manifest_sha256")
            or commitment.get("runtime_gate_sha256")
            != gate.get("manifest_sha256")):
        raise BroadQaExternalDataError("v9 family lineage漂移")
    core = {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "code_files": code_files,
        "code_identity_sha256": code_sha,
        "denominator": commitment["denominator"],
        "dimension_order": commitment["dimension_order"],
        "dimensions": commitment["dimensions"],
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_or_reserve_payload_read_count": 0,
        "evaluation_run_count": 0,
        "format_version": 1,
        "formal_contract": commitment["formal_contract"],
        "gimp_source_manifest_read_count": 1,
        "gimp_source_manifest_sha256": source["manifest_sha256"],
        "gimp_source_non_manifest_read_count": 0,
        "git_head": head,
        "individual_label_read_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_gate_manifest_read_count": 1,
        "runtime_gate_sha256": gate["manifest_sha256"],
        "status": NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    freeze = {**core, "family_commitment_sha256": _sha256(
        canonical_json_bytes(core))}
    return freeze, candidate_manifest, candidate, commitment


def publish_normalization_recovery_v9_evaluation_family_freeze(
        *, run_root: str | Path, target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布v9 family freeze，且不读取GIMP payload。"""
    root = require_normalization_recovery_v9_k_root(run_root)
    inputs = tuple(normalization_recovery_v9_path_within(
        root, arguments[name], label=name)
        for name in NORMALIZATION_RECOVERY_V9_EVALUATION_DATA_ARGUMENTS)
    target = normalization_recovery_v9_path_within(
        root, target_dir, label="target_dir")
    paths = (*inputs, target)
    if (target.exists() or any(not path.is_dir() for path in inputs)
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v9 family artifact path非法")
    freeze, _manifest, _candidate, _commitment = (
        build_normalization_recovery_v9_evaluation_family_freeze(**arguments))
    target.mkdir()
    path = target / "family-freeze.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(freeze))
    return {**freeze, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v9_evaluation_family_freeze(
        family_dir: str | Path, **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, object]]:
    """严格回读family并重算live candidate、commitment与code。"""
    root = Path(family_dir).resolve()
    expected, candidate_manifest, candidate, commitment = (
        build_normalization_recovery_v9_evaluation_family_freeze(**arguments))
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "family-freeze.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v9 family freeze不可读") from error
    if (physical != {"family-freeze.json"}
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not strict_json_equal(stored, expected)):
        raise BroadQaExternalDataError("v9 family freeze与live identity漂移")
    return ({**stored, "manifest_sha256": _sha256(encoded)},
            candidate_manifest, candidate, commitment)


__all__ = [
    "NORMALIZATION_RECOVERY_V9_EVALUATION_DATA_ARGUMENTS",
    "NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V9_EVALUATION_FAMILY_STATUS",
    "NORMALIZATION_RECOVERY_V9_EVALUATION_IDENTITY_ARGUMENTS",
    "build_normalization_recovery_v9_evaluation_family_freeze",
    "normalization_recovery_v9_path_within",
    "publish_normalization_recovery_v9_evaluation_family_freeze",
    "read_normalization_recovery_v9_evaluation_family_freeze",
    "require_normalization_recovery_v9_k_root",
]
