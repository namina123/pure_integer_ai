"""W-02 morphology successor blind private family 的 metadata-only 冻结。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_BOUNDARY_PATH,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    build_v2_private_family_registration,
    read_v2_evaluator_boundary_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02_CANDIDATE_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _hash_value,
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_DEV_REPORT_PATH,
    read_w02_morphology_successor_dev_calibration_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_private_evaluator import (
    W02_PRIVATE_EVALUATOR_VERSION,
    W02_PRIVATE_LAYOUT_PATHS,
    W02_PRIVATE_SUPPORT_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02_MORPH_SUCCESSOR_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_shadow_audit import (
    W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_SHADOW_REPORT_PATH,
    read_w02_morphology_successor_shadow_audit_freeze,
)


W02_PRIVATE_FAMILY_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-PRIVATE-FAMILY-V1")
W02_PRIVATE_FAMILY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_private_family_freeze_v1.json")
W02_PRIVATE_FAMILY_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_private_evaluator.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_private_family.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_private_evaluation.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_private_evaluation.py",
)
W02_PRIVATE_FAMILY_DOCUMENT = "private-family-freeze.json"
W02_PRIVATE_REGISTRATION_DOCUMENT = "private-family-registration.json"
W02_PRIVATE_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_PRIVATE_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_PRIVATE_RUN_INTENT = "run-guard/run-intent.json"
W02_PRIVATE_EXPOSURE_LEDGER = "exposure-ledger"


# object-model: exception
class W02MorphologySuccessorPrivateFamilyError(RuntimeError):
    """Private family、公开依赖、代码或一次性 guard 发生漂移。"""


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorPrivateFamilyError(
            f"{where} 不是小写 SHA-256")
    return value


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or target.is_symlink() or not target.is_relative_to(repository)
            or not target.is_file()):
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private repository file 非法")
    return target


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_PRIVATE_FAMILY_CODE_PATHS:
        size, digest = _sha256_file(_repository_file(repository, relative))
        rows.append({
            "repository_file": relative,
            "sha256": digest,
            "size_bytes": size,
        })
    return rows, _hash_value(rows)


def _public_identity(repository: Path, relative: str) -> dict[str, object]:
    size, digest = _sha256_file(_repository_file(repository, relative))
    return {"repository_file": relative, "sha256": digest, "size_bytes": size}


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    boundary = read_v2_evaluator_boundary_contract(repository)
    dev_freeze = read_w02_morphology_successor_dev_calibration_freeze(repository)
    shadow_freeze = read_w02_morphology_successor_shadow_audit_freeze(repository)
    paths = {
        "boundary": V2_EVALUATOR_BOUNDARY_PATH,
        "candidate_receipt": W02_CANDIDATE_RECEIPT_PATH,
        "overlay_receipt": W02_MORPH_SUCCESSOR_RECEIPT_PATH,
        "dev_freeze": W02_MORPH_SUCCESSOR_DEV_FREEZE_PATH,
        "dev_report": W02_MORPH_SUCCESSOR_DEV_REPORT_PATH,
        "shadow_freeze": W02_MORPH_SUCCESSOR_SHADOW_FREEZE_PATH,
        "shadow_report": W02_MORPH_SUCCESSOR_SHADOW_REPORT_PATH,
    }
    identities = {key: _public_identity(repository, value)
                  for key, value in paths.items()}
    candidate_receipt = read_canonical_object(
        _repository_file(repository, paths["candidate_receipt"]))
    overlay_receipt = read_canonical_object(
        _repository_file(repository, paths["overlay_receipt"]))
    dev_report = read_canonical_object(
        _repository_file(repository, paths["dev_report"]))
    shadow_report = read_canonical_object(
        _repository_file(repository, paths["shadow_report"]))
    if (candidate_receipt.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or candidate_receipt.get("formal_training_runs") != 1
            or overlay_receipt.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN"
            or overlay_receipt.get("formal_successor_transform_runs") != 1
            or dev_report.get("status") != "PASS"
            or dev_report.get("formal_dev_calibration_runs") != 1
            or shadow_report.get("status") != "PASS"
            or shadow_report.get("formal_shadow_audit_runs") != 1
            or shadow_report.get("label_reads") != 0
            or shadow_report.get("shadow_freeze_file_sha256")
            != identities["shadow_freeze"]["sha256"]
            or shadow_report.get("dev_pass_report_file_sha256")
            != identities["dev_report"]["sha256"]
            or dev_report.get("dev_freeze_file_sha256")
            != identities["dev_freeze"]["sha256"]
            or overlay_receipt.get("parent_candidate_manifest_sha256")
            != candidate_receipt.get("candidate_artifact_manifest_sha256")
            or any(value != 0 for value in (
                candidate_receipt.get("formal_private_evaluation_runs"),
                candidate_receipt.get("private_payload_reads"),
                overlay_receipt.get("formal_private_evaluation_runs"),
                overlay_receipt.get("private_payload_reads"),
                dev_report.get("formal_private_evaluation_runs"),
                dev_report.get("private_payload_reads"),
                shadow_report.get("formal_private_evaluation_runs"),
                shadow_report.get("private_payload_reads"),
            ))):
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private parent evidence 未闭合")
    if (dev_freeze.get("code_freeze_sha256")
            != dev_report.get("code_freeze_sha256")
            or shadow_freeze.get("code_freeze_sha256")
            != shadow_report.get("code_freeze_sha256")):
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private dev/shadow code freeze 漂移")
    private_files = tuple(
        item for item in parent.files
        if item.root_key == "PRIVATE_EVALUATOR_ROOT")
    if (tuple(item.layout_key for item in private_files)
            != tuple(W02_PRIVATE_LAYOUT_PATHS)
            or _hash_value([item.to_dict() for item in private_files])
            != parent.private_payload_commitment):
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private public payload commitment 漂移")
    candidate_binding_value = {
        "candidate_artifact_manifest_sha256":
            candidate_receipt["candidate_artifact_manifest_sha256"],
        "candidate_receipt_file_sha256":
            identities["candidate_receipt"]["sha256"],
        "candidate_semantic_sha256": candidate_receipt["candidate_semantic_sha256"],
        "compile_freeze_sha256": parent.sha256(),
        "dev_pass_report_file_sha256": identities["dev_report"]["sha256"],
        "overlay_artifact_manifest_sha256":
            overlay_receipt["overlay_artifact_manifest_sha256"],
        "overlay_receipt_file_sha256": identities["overlay_receipt"]["sha256"],
        "overlay_semantic_sha256": overlay_receipt["overlay_semantic_sha256"],
        "shadow_pass_report_file_sha256":
            identities["shadow_report"]["sha256"],
    }
    return {
        "boundary_contract_sha256": boundary.sha256(),
        "candidate_binding_sha256": _hash_value(candidate_binding_value),
        "candidate_binding_value": candidate_binding_value,
        "identities": identities,
        "parent": parent,
        "private_files": private_files,
    }


def w02_private_guard_value(
        *, family_commitment: str, candidate_binding_sha256: str,
        code_freeze_sha256: str) -> dict[str, object]:
    """构造正式 private run 的规范一次性 guard。"""
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_FIRST_RUN_GUARD"),
        "candidate_binding_sha256": _sha256(
            candidate_binding_sha256, where="private guard candidate binding"),
        "code_freeze_sha256": _sha256(
            code_freeze_sha256, where="private guard code freeze"),
        "family_commitment": _sha256(
            family_commitment, where="private guard family"),
        "formal_private_evaluation_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "guard_version": W02_PRIVATE_FAMILY_VERSION,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "run_id_policy": "EXACTLY_ONE",
        "stage_key": "W-02",
        "status": "AVAILABLE",
    }


def _guard_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


def build_w02_morphology_successor_private_family_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """仅从公开 metadata 构造 blind family、代码和 guard 冻结。"""
    repository = Path(repository_root).resolve()
    dependency = _dependency_state(repository)
    parent = dependency["parent"]
    code_rows, code_sha = _code_rows(repository)
    registration = build_v2_private_family_registration(
        "W-02",
        payload_commitment=parent.private_payload_commitment,
        case_commitment=parent.private_case_commitment,
        label_commitment=parent.private_label_commitment,
        cluster_commitment=parent.private_cluster_commitment,
        candidate_freeze_sha256=dependency["candidate_binding_sha256"],
        code_freeze_sha256=code_sha,
        resource_budget=parent.resource_budget,
    )
    guard = w02_private_guard_value(
        family_commitment=registration.family_commitment,
        candidate_binding_sha256=dependency["candidate_binding_sha256"],
        code_freeze_sha256=code_sha)
    private_counts = {
        item.layout_key: item.record_count for item in dependency["private_files"]
    }
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_FAMILY_FREEZE"),
        "artifact_version": W02_PRIVATE_FAMILY_VERSION,
        "boundary_contract_sha256": dependency["boundary_contract_sha256"],
        "candidate_binding": dependency["candidate_binding_value"],
        "candidate_binding_sha256": dependency["candidate_binding_sha256"],
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "evaluator_version": W02_PRIVATE_EVALUATOR_VERSION,
        "first_run_guard_sha256": _guard_sha(guard),
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 1,
        "hard_conjunct_keys": list(registration.policy.hard_conjunct_keys),
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "ne_policy": "BLOCK",
        "next_action": "W02_FORMAL_BLIND_PRIVATE_EVALUATION",
        "parent_evidence_files": [
            dependency["identities"][key]
            for key in (
                "boundary", "candidate_receipt", "overlay_receipt",
                "dev_freeze", "dev_report", "shadow_freeze", "shadow_report")
        ],
        "private_family_registered": 1,
        "private_file_counts": private_counts,
        "private_input_files": [
            item.to_dict() for item in dependency["private_files"]],
        "private_payload_reads": 0,
        "registration": registration.to_dict(),
        "release_key": "PH2-D03-V2",
        "resource_budget": parent.resource_budget.to_dict(),
        "stage_key": "W-02",
        "status": "W02_BLIND_PRIVATE_FAMILY_REGISTERED_AND_FROZEN",
        "support_hard_conjunct_keys": list(W02_PRIVATE_SUPPORT_KEYS),
        "teacher_calls": 0,
        "threshold_reduction": 0,
        "zero_call_window_count": 3,
        "zero_write_required": 1,
    }


def publish_w02_morphology_successor_private_family_freeze(
        repository_root: str | Path) -> Path:
    """不可覆盖发布 safe public family freeze。"""
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_private_family_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_PRIVATE_FAMILY_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private family freeze 发布字节漂移")
    return target


def read_w02_morphology_successor_private_family_freeze(
        repository_root: str | Path) -> dict[str, object]:
    """严格回读 family freeze 并重算 live code 与全部公开父证据。"""
    repository = Path(repository_root).resolve()
    target = _repository_file(repository, W02_PRIVATE_FAMILY_FREEZE_PATH)
    value = read_canonical_object(target)
    expected = build_w02_morphology_successor_private_family_freeze(repository)
    if value != expected or target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private family freeze 与 live identity 漂移")
    return value


def w02_private_registration_from_freeze(
        value: dict[str, object]) -> V2PrivateFamilyRegistration:
    """从公开 freeze 严格重建 generic evaluator registration。"""
    registration = value.get("registration")
    budget = value.get("resource_budget")
    if not isinstance(registration, dict) or not isinstance(budget, dict):
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private registration 字段非法")
    rebuilt = build_v2_private_family_registration(
        "W-02",
        payload_commitment=str(registration.get("payload_commitment", "")),
        case_commitment=str(registration.get("case_commitment", "")),
        label_commitment=str(registration.get("label_commitment", "")),
        cluster_commitment=str(registration.get("cluster_commitment", "")),
        candidate_freeze_sha256=str(
            registration.get("candidate_freeze_sha256", "")),
        code_freeze_sha256=str(registration.get("code_freeze_sha256", "")),
        resource_budget=V2EvaluatorResourceBudget.from_dict(budget),
    )
    if rebuilt.to_dict() != registration:
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private registration canonical 字节漂移")
    return rebuilt


def publish_w02_morphology_successor_private_family_root(
        repository_root: str | Path,
        family_root: str | Path,
        ) -> str:
    """在全新 Git 外 root 发布 registration、freeze、ledger 与 available guard。"""
    repository = Path(repository_root).resolve()
    freeze = read_w02_morphology_successor_private_family_freeze(repository)
    registration = w02_private_registration_from_freeze(freeze)
    root = Path(family_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(
        registration.to_dict(), root / W02_PRIVATE_REGISTRATION_DOCUMENT)
    write_immutable_json(freeze, root / W02_PRIVATE_FAMILY_DOCUMENT)
    (root / W02_PRIVATE_EXPOSURE_LEDGER).mkdir(exist_ok=False)
    guard = w02_private_guard_value(
        family_commitment=registration.family_commitment,
        candidate_binding_sha256=str(freeze["candidate_binding_sha256"]),
        code_freeze_sha256=registration.code_freeze_sha256)
    target = root / Path(*PurePosixPath(W02_PRIVATE_GUARD_AVAILABLE).parts)
    write_immutable_json(guard, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != freeze["first_run_guard_sha256"]:
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private available guard SHA 漂移")
    return digest


def consume_w02_morphology_successor_private_guard(
        family_root: str | Path,
        *, expected_guard_sha256: str, run_identity_sha256: str,
        ) -> None:
    """在首次 private transport 读取前原子消费唯一 guard。"""
    _sha256(expected_guard_sha256, where="private expected guard")
    _sha256(run_identity_sha256, where="private run identity")
    root = Path(family_root).resolve()
    available = root / Path(*PurePosixPath(W02_PRIVATE_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(W02_PRIVATE_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_PRIVATE_RUN_INTENT).parts)
    if consumed.exists() or intent.exists() or not available.is_file():
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private guard 不可用或已消费")
    if _sha256_file(available)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private guard 字节漂移")
    os.replace(available, consumed)
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_RUN_INTENT"),
        "formal_private_evaluation_runs": 1,
        "guard_sha256": expected_guard_sha256,
        "private_payload_reads_before_guard": 0,
        "run_id": 1,
        "run_identity_sha256": run_identity_sha256,
        "stage_key": "W-02",
        "status": "GUARD_CONSUMED_BEFORE_PRIVATE_READ",
    }, intent)


def verify_w02_morphology_successor_private_consumed_guard(
        family_root: str | Path,
        *, expected_guard_sha256: str, run_identity_sha256: str,
        ) -> None:
    """只读验证 consumed guard 与 run intent。"""
    root = Path(family_root).resolve()
    available = root / Path(*PurePosixPath(W02_PRIVATE_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(W02_PRIVATE_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_PRIVATE_RUN_INTENT).parts)
    if available.exists() or not consumed.is_file() or not intent.is_file():
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private consumed guard 状态不闭合")
    if _sha256_file(consumed)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private consumed guard SHA 漂移")
    if read_canonical_object(intent) != {
            "artifact_kind": (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_RUN_INTENT"),
            "formal_private_evaluation_runs": 1,
            "guard_sha256": expected_guard_sha256,
            "private_payload_reads_before_guard": 0,
            "run_id": 1,
            "run_identity_sha256": run_identity_sha256,
            "stage_key": "W-02",
            "status": "GUARD_CONSUMED_BEFORE_PRIVATE_READ",
            }:
        raise W02MorphologySuccessorPrivateFamilyError(
            "W-02 private run intent 漂移")


__all__ = [
    "W02_PRIVATE_EXPOSURE_LEDGER", "W02_PRIVATE_FAMILY_CODE_PATHS",
    "W02_PRIVATE_FAMILY_DOCUMENT", "W02_PRIVATE_FAMILY_FREEZE_PATH",
    "W02_PRIVATE_FAMILY_VERSION", "W02_PRIVATE_GUARD_AVAILABLE",
    "W02_PRIVATE_GUARD_CONSUMED", "W02_PRIVATE_REGISTRATION_DOCUMENT",
    "W02_PRIVATE_RUN_INTENT", "W02MorphologySuccessorPrivateFamilyError",
    "build_w02_morphology_successor_private_family_freeze",
    "consume_w02_morphology_successor_private_guard",
    "publish_w02_morphology_successor_private_family_freeze",
    "publish_w02_morphology_successor_private_family_root",
    "read_w02_morphology_successor_private_family_freeze",
    "verify_w02_morphology_successor_private_consumed_guard",
    "w02_private_guard_value", "w02_private_registration_from_freeze",
]
