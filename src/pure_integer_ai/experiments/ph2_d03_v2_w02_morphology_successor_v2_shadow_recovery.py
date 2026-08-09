"""W-02 successor V2 shadow 外部中断后的 append-only recovery authority。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _hash_value,
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_audit import (
    W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS,
    W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES,
    W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH,
    publish_w02_morphology_successor_v2_shadow_audit_report,
    read_w02_morphology_successor_v2_shadow_audit_freeze,
)


W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-SHADOW-RECOVERY-FREEZE-V1")
W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery_freeze_v1.json")
W02_MORPH_V2_SHADOW_RECOVERY_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery.py",
)
W02_MORPH_V2_SHADOW_RECOVERY_FAMILY_NAME = (
    ".d03-w02-morph-successor-v2-shadow-recovery-formal-20260809-b")
W02_MORPH_V2_SHADOW_RECOVERY_AVAILABLE = "run-000001.available.json"
W02_MORPH_V2_SHADOW_RECOVERY_INTENT = "run-000001.intent.json"
W02_MORPH_V2_SHADOW_RECOVERY_CONSUMED = "run-000001.consumed.json"
W02_MORPH_V2_SHADOW_RECOVERY_REPORT = "run-000001.report.json"
W02_MORPH_V2_SHADOW_RECOVERY_FAILURE = "run-000001.failure.json"
W02_MORPH_V2_SHADOW_ABORTED_FAILURE_SEAL_SHA256 = (
    "927dcbb184b7ae0b844c61c11c1cc1b45947eac6bbd7cc62f582ffd73f103929")
W02_MORPH_V2_SHADOW_ABORTED_ERROR_EVIDENCE_SHA256 = (
    "fca8dd4bd6dcd0f456559f5ab3806ad5e41927a136d794d9744bd151a6202ecc")


# object-model: exception
class W02MorphologySuccessorV2ShadowRecoveryError(RuntimeError):
    """Recovery freeze、guard、运行状态或公开投影不闭合。"""


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(repository) or not target.is_file()):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_MORPH_V2_SHADOW_RECOVERY_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({"repository_file": relative, "sha256": digest,
                     "size_bytes": size})
    return rows, _hash_value(rows)


def build_w02_morphology_successor_v2_shadow_recovery_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """只授权 A 外部中断后的一个 crash-evidenced recovery B。"""
    repository = Path(repository_root).resolve()
    parent = read_w02_morphology_successor_v2_shadow_audit_freeze(repository)
    parent_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_SHADOW_FREEZE_PATH)
    parent_size, parent_sha = _sha256_file(parent_path)
    code_rows, code_sha = _code_rows(repository)
    if (parent.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT_FREEZE_COMPLETE"
            or parent.get("formal_shadow_audit_runs") != 0
            or parent.get("formal_private_evaluation_runs") != 0
            or parent.get("private_payload_reads") != 0
            or parent.get("label_reads") != 0
            or parent.get("expected_counts")
            != W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery parent freeze 漂移")
    return {
        "aborted_error_evidence_sha256":
            W02_MORPH_V2_SHADOW_ABORTED_ERROR_EVIDENCE_SHA256,
        "aborted_failure_seal_sha256":
            W02_MORPH_V2_SHADOW_ABORTED_FAILURE_SEAL_SHA256,
        "aborted_formal_family_key": "SHADOW_FORMAL_20260809_A",
        "aborted_formal_shadow_attempts": 1,
        "aborted_formal_shadow_passes": 0,
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_FREEZE"),
        "artifact_version": W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_VERSION,
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "expected_counts": dict(W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS),
        "expected_gates": [
            {"denominator": denominator, "failed": 0, "gate_key": name,
             "ne": 0, "numerator": denominator, "status": "PASS"}
            for name, denominator in W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_GATES
        ],
        "formal_private_evaluation_runs": 0,
        "formal_shadow_recovery_runs": 0,
        "label_reads": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_SHADOW_RECOVERY_GUARD_PUBLICATION",
        "parent_shadow_code_freeze_sha256": parent["code_freeze_sha256"],
        "parent_shadow_freeze_file_sha256": parent_sha,
        "parent_shadow_freeze_size_bytes": parent_size,
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "recovery_authority": "EXTERNAL_PROCESS_INTERRUPTION_BEFORE_RESULT",
        "recovery_family_key": "SHADOW_RECOVERY_FORMAL_20260809_B",
        "recovery_run_id": 1,
        "release_key": "PH2-D03-V2",
        "stage_key": "W-02",
        "status": "W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_FREEZE_COMPLETE",
        "teacher_calls": 0,
    }


def publish_w02_morphology_successor_v2_shadow_recovery_freeze(
        repository_root: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v2_shadow_recovery_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_v2_shadow_recovery_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH)
    value = read_canonical_object(target)
    if value != build_w02_morphology_successor_v2_shadow_recovery_freeze(
            repository):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery freeze 与 live identity 漂移")
    return value


def _family_root(value: str | Path, *, require_exists: bool) -> Path:
    root = Path(value).resolve()
    if (root.name != W02_MORPH_V2_SHADOW_RECOVERY_FAMILY_NAME
            or root.is_symlink()
            or (require_exists and not root.is_dir())):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery family root 非法")
    return root


def _guard_value(repository: Path) -> dict[str, object]:
    freeze = read_w02_morphology_successor_v2_shadow_recovery_freeze(repository)
    freeze_path = _repository_file(
        repository, W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH)
    return {
        "aborted_failure_seal_sha256":
            freeze["aborted_failure_seal_sha256"],
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_GUARD"),
        "available": 1,
        "consumed": 0,
        "formal_shadow_recovery_runs": 0,
        "recovery_code_freeze_sha256": freeze["code_freeze_sha256"],
        "recovery_freeze_file_sha256": _sha256_file(freeze_path)[1],
        "recovery_run_id": 1,
        "stage_key": "W-02",
        "status": "AVAILABLE_FOR_SINGLE_RECOVERY_RUN",
    }


def publish_w02_morphology_successor_v2_shadow_recovery_guard(
        repository_root: str | Path,
        family_root: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=False)
    if root.exists():
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery family root 已存在")
    root.mkdir(parents=False, exist_ok=False)
    target = root / W02_MORPH_V2_SHADOW_RECOVERY_AVAILABLE
    write_immutable_json(_guard_value(repository), target)
    return target


def _intent_value(guard_sha256: str) -> dict[str, object]:
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_INTENT"),
        "formal_shadow_recovery_runs": 1,
        "guard_sha256": guard_sha256,
        "recovery_run_id": 1,
        "stage_key": "W-02",
        "status": "RUN_INTENT_COMMITTED",
    }


def _consumed_value(intent_sha256: str) -> dict[str, object]:
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_CONSUMED"),
        "consumed": 1,
        "formal_shadow_recovery_runs": 1,
        "intent_sha256": intent_sha256,
        "recovery_run_id": 1,
        "stage_key": "W-02",
        "status": "RECOVERY_GUARD_CONSUMED",
    }


def consume_w02_morphology_successor_v2_shadow_recovery_guard(
        repository_root: str | Path,
        family_root: str | Path,
        ) -> dict[str, str]:
    """在任何全量输入读取前不可覆盖提交 intent 与 consumed。"""
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=True)
    guard_path = root / W02_MORPH_V2_SHADOW_RECOVERY_AVAILABLE
    intent_path = root / W02_MORPH_V2_SHADOW_RECOVERY_INTENT
    consumed_path = root / W02_MORPH_V2_SHADOW_RECOVERY_CONSUMED
    blockers = (
        intent_path, consumed_path,
        root / W02_MORPH_V2_SHADOW_RECOVERY_REPORT,
        root / W02_MORPH_V2_SHADOW_RECOVERY_FAILURE,
    )
    if (not guard_path.is_file() or guard_path.is_symlink()
            or any(path.exists() for path in blockers)
            or read_canonical_object(guard_path) != _guard_value(repository)):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery guard 已消费或漂移")
    guard_sha = _sha256_file(guard_path)[1]
    write_immutable_json(_intent_value(guard_sha), intent_path)
    intent_sha = _sha256_file(intent_path)[1]
    write_immutable_json(_consumed_value(intent_sha), consumed_path)
    return {
        "recovery_consumed_sha256": _sha256_file(consumed_path)[1],
        "recovery_guard_sha256": guard_sha,
        "recovery_intent_sha256": intent_sha,
    }


def read_w02_morphology_successor_v2_shadow_recovery_state(
        repository_root: str | Path,
        family_root: str | Path,
        ) -> dict[str, str]:
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=True)
    guard_path = root / W02_MORPH_V2_SHADOW_RECOVERY_AVAILABLE
    intent_path = root / W02_MORPH_V2_SHADOW_RECOVERY_INTENT
    consumed_path = root / W02_MORPH_V2_SHADOW_RECOVERY_CONSUMED
    if read_canonical_object(guard_path) != _guard_value(repository):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery available guard 漂移")
    guard_sha = _sha256_file(guard_path)[1]
    if read_canonical_object(intent_path) != _intent_value(guard_sha):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery intent 漂移")
    intent_sha = _sha256_file(intent_path)[1]
    if read_canonical_object(consumed_path) != _consumed_value(intent_sha):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery consumed guard 漂移")
    return {
        "recovery_consumed_sha256": _sha256_file(consumed_path)[1],
        "recovery_guard_sha256": guard_sha,
        "recovery_intent_sha256": intent_sha,
    }


def recovery_report_fields(
        repository_root: str | Path,
        family_root: str | Path,
        *,
        passed: bool,
        ) -> dict[str, object]:
    if type(passed) is not bool:
        raise TypeError("shadow recovery passed 必须为 bool")
    repository = Path(repository_root).resolve()
    freeze = read_w02_morphology_successor_v2_shadow_recovery_freeze(repository)
    freeze_path = _repository_file(
        repository, W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH)
    state = read_w02_morphology_successor_v2_shadow_recovery_state(
        repository, family_root)
    return {
        "aborted_parent_failure_seal_sha256":
            freeze["aborted_failure_seal_sha256"],
        "formal_shadow_audit_attempts": 2,
        "formal_shadow_audit_passes": int(passed),
        "formal_shadow_recovery_runs": 1,
        "interrupted_parent_formal_shadow_runs": 1,
        "recovery_code_freeze_sha256": freeze["code_freeze_sha256"],
        "recovery_freeze_file_sha256": _sha256_file(freeze_path)[1],
        **state,
    }


def publish_w02_morphology_successor_v2_shadow_recovery_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    report_path = Path(external_report).resolve()
    family_root = _family_root(report_path.parent, require_exists=True)
    if report_path.name != W02_MORPH_V2_SHADOW_RECOVERY_REPORT:
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery report path 非法")
    value = read_canonical_object(report_path)
    validate_v2_safe_report(value)
    expected = recovery_report_fields(repository, family_root, passed=True)
    if (value.get("status") != "PASS"
            or value.get("formal_shadow_audit_runs") != 1
            or any(value.get(key) != expected_value
                   for key, expected_value in expected.items())):
        raise W02MorphologySuccessorV2ShadowRecoveryError(
            "shadow recovery report identity 漂移")
    return publish_w02_morphology_successor_v2_shadow_audit_report(
        repository, report_path)


def write_w02_morphology_successor_v2_shadow_recovery_failure(
        repository_root: str | Path,
        family_root: str | Path,
        error: BaseException,
        ) -> Path:
    repository = Path(repository_root).resolve()
    root = _family_root(family_root, require_exists=True)
    state = read_w02_morphology_successor_v2_shadow_recovery_state(
        repository, root)
    target = root / W02_MORPH_V2_SHADOW_RECOVERY_FAILURE
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_FAILURE"),
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_recovery_runs": 1,
        "private_payload_reads": 0,
        "recovery_consumed_sha256": state["recovery_consumed_sha256"],
        "recovery_run_id": 1,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_PRIVATE_REGISTRATION",
    }, target)
    return target


__all__ = [
    "W02_MORPH_V2_SHADOW_RECOVERY_CODE_PATHS",
    "W02_MORPH_V2_SHADOW_RECOVERY_FAMILY_NAME",
    "W02_MORPH_V2_SHADOW_RECOVERY_FREEZE_PATH",
    "W02MorphologySuccessorV2ShadowRecoveryError",
    "build_w02_morphology_successor_v2_shadow_recovery_freeze",
    "consume_w02_morphology_successor_v2_shadow_recovery_guard",
    "publish_w02_morphology_successor_v2_shadow_recovery_freeze",
    "publish_w02_morphology_successor_v2_shadow_recovery_guard",
    "publish_w02_morphology_successor_v2_shadow_recovery_report",
    "read_w02_morphology_successor_v2_shadow_recovery_freeze",
    "read_w02_morphology_successor_v2_shadow_recovery_state",
    "recovery_report_fields",
    "write_w02_morphology_successor_v2_shadow_recovery_failure",
]
