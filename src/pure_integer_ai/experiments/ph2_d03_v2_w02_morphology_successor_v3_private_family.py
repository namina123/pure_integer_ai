"""Metadata-only freeze for the successor V3 blind private family."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH,
    read_blind_private_source_extension_v3_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_BOUNDARY_PATH,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    build_v2_private_family_registration,
    read_v2_evaluator_boundary_contract,
    validate_v2_safe_report,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02_MORPH_SUCCESSOR_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_publication import (
    W02_MORPH_V2_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_dev_probe import (
    W02_MORPH_V3_DEV_FREEZE_PATH,
    read_w02_morphology_successor_v3_dev_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION,
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    read_w02_morphology_successor_v3_private_owner_receipt,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_shadow_probe import (
    W02_MORPH_V3_DEV_REPORT_PATH,
    W02_MORPH_V3_SHADOW_FREEZE_PATH,
    W02_MORPH_V3_SHADOW_REPORT_PATH,
    read_w02_morphology_successor_v3_shadow_freeze,
)


W02_MORPH_V3_PRIVATE_FAMILY_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-FAMILY-V1"
)
W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_family_freeze_v1.json"
)
W02_MORPH_V3_PRIVATE_FAMILY_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_io.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_evaluator.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_family.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_publication.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v3_private_evaluation.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v3_private_evaluation.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v3_private_family.py",
)
W02_MORPH_V3_PRIVATE_FORMAL_FAMILY_NAME = (
    ".d03-w02-morph-successor-v3-private-formal-r2-20260809-a"
)
W02_MORPH_V3_PRIVATE_FAMILY_DOCUMENT = "private-family-freeze.json"
W02_MORPH_V3_PRIVATE_REGISTRATION_DOCUMENT = "private-family-registration.json"
W02_MORPH_V3_PRIVATE_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_MORPH_V3_PRIVATE_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_MORPH_V3_PRIVATE_RUN_INTENT = "run-guard/run-intent.json"
W02_MORPH_V3_PRIVATE_EXPOSURE_LEDGER = "exposure-ledger"


# object-model: exception
class W02MorphologySuccessorV3PrivateFamilyError(RuntimeError):
    """The V3 public freeze, dependency chain, guard, or receipt drifted."""


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            f"{where} is not lowercase SHA-256")
    return value


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or target.is_symlink() or not target.is_relative_to(repository)
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family repository file is invalid")
    return target


def _public_identity(repository: Path, relative: str) -> dict[str, object]:
    size, digest = _sha256_file(_repository_file(repository, relative))
    return {"repository_file": relative, "sha256": digest, "size_bytes": size}


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = [
        _public_identity(repository, relative)
        for relative in W02_MORPH_V3_PRIVATE_FAMILY_CODE_PATHS
    ]
    return rows, _hash_value(rows)


def _owner_budget(value: object) -> V2EvaluatorResourceBudget:
    if (not isinstance(value, dict) or set(value) != {"limits", "usage"}
            or not isinstance(value.get("limits"), dict)
            or set(value["limits"]) != {
                "max_logic_operations", "max_payload_bytes",
                "max_payload_gets", "max_records",
            }):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 owner resource budget fields drifted")
    limits = value["limits"]
    return V2EvaluatorResourceBudget(
        512,
        limits["max_logic_operations"],
        limits["max_payload_bytes"],
        limits["max_payload_gets"],
        limits["max_records"],
        4,
    )


def _family_root(value: str | Path, *, require_exists: bool) -> Path:
    root = Path(value).resolve()
    if (root.name != W02_MORPH_V3_PRIVATE_FORMAL_FAMILY_NAME
            or root.is_symlink()
            or (require_exists and not root.is_dir())):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private formal family root is invalid")
    return root


def _dependency_state(repository: Path) -> dict[str, Any]:
    compile_freeze = read_w02_compile_freeze(repository)
    boundary = read_v2_evaluator_boundary_contract(repository)
    dev_freeze = read_w02_morphology_successor_v3_dev_freeze(repository)
    shadow_freeze = read_w02_morphology_successor_v3_shadow_freeze(repository)
    source_extension = read_blind_private_source_extension_v3_manifest(repository)
    owner, files = read_w02_morphology_successor_v3_private_owner_receipt(
        repository)
    paths = {
        "boundary": V2_EVALUATOR_BOUNDARY_PATH,
        "candidate_receipt": W02_CANDIDATE_RECEIPT_PATH,
        "v1_receipt": W02_MORPH_SUCCESSOR_RECEIPT_PATH,
        "v2_receipt": W02_MORPH_V2_RECEIPT_PATH,
        "v3_dev_freeze": W02_MORPH_V3_DEV_FREEZE_PATH,
        "v3_dev_report": W02_MORPH_V3_DEV_REPORT_PATH,
        "v3_shadow_freeze": W02_MORPH_V3_SHADOW_FREEZE_PATH,
        "v3_shadow_report": W02_MORPH_V3_SHADOW_REPORT_PATH,
        "source_extension_v3": BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH,
        "owner_receipt": W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH,
    }
    identities = {
        key: _public_identity(repository, relative)
        for key, relative in paths.items()
    }
    candidate = read_canonical_object(
        _repository_file(repository, paths["candidate_receipt"]))
    v1 = read_canonical_object(
        _repository_file(repository, paths["v1_receipt"]))
    v2 = read_canonical_object(
        _repository_file(repository, paths["v2_receipt"]))
    dev = read_canonical_object(
        _repository_file(repository, paths["v3_dev_report"]))
    shadow = read_canonical_object(
        _repository_file(repository, paths["v3_shadow_report"]))
    validate_v2_safe_report(dev)
    validate_v2_safe_report(shadow)
    if (candidate.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or candidate.get("formal_training_runs") != 1
            or v1.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN"
            or v1.get("formal_successor_transform_runs") != 1
            or v1.get("parent_candidate_manifest_sha256")
            != candidate.get("candidate_artifact_manifest_sha256")
            or v1.get("parent_candidate_semantic_sha256")
            != candidate.get("candidate_semantic_sha256")
            or v2.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V2_ARTIFACT_FROZEN"
            or v2.get("formal_successor_v2_transform_runs") != 1
            or v2.get("parent_candidate_manifest_sha256")
            != candidate.get("candidate_artifact_manifest_sha256")
            or v2.get("parent_candidate_semantic_sha256")
            != candidate.get("candidate_semantic_sha256")
            or v2.get("parent_v1_overlay_manifest_sha256")
            != v1.get("overlay_artifact_manifest_sha256")
            or v2.get("parent_v1_overlay_semantic_sha256")
            != v1.get("overlay_semantic_sha256")
            or dev.get("status") != "PASS"
            or dev.get("run_scope") != "FORMAL"
            or dev.get("run_id") != 1
            or dev.get("formal_dev_calibration_runs") != 1
            or dev.get("formal_private_evaluation_runs") != 0
            or dev.get("private_payload_reads") != 0
            or dev.get("teacher_calls") != 0
            or dev.get("code_freeze_sha256")
            != dev_freeze.get("code_freeze_sha256")
            or dev.get("freeze_file_sha256")
            != identities["v3_dev_freeze"]["sha256"]
            or dev.get("parent_v1_semantic_sha256")
            != v1.get("overlay_semantic_sha256")
            or dev.get("parent_v2_semantic_sha256")
            != v2.get("semantic_sha256")
            or shadow.get("status") != "PASS"
            or shadow.get("run_scope") != "FORMAL"
            or shadow.get("run_id") != 1
            or shadow.get("formal_shadow_audit_runs") != 1
            or shadow.get("formal_private_evaluation_runs") != 0
            or shadow.get("private_payload_reads") != 0
            or shadow.get("label_reads") != 0
            or shadow.get("teacher_calls") != 0
            or shadow.get("code_freeze_sha256")
            != shadow_freeze.get("code_freeze_sha256")
            or shadow.get("freeze_file_sha256")
            != identities["v3_shadow_freeze"]["sha256"]
            or shadow.get("parent_v1_semantic_sha256")
            != v1.get("overlay_semantic_sha256")
            or shadow.get("parent_v2_semantic_sha256")
            != v2.get("semantic_sha256")
            or source_extension.get("status")
            != "BLIND_PRIVATE_SOURCE_EXTENSION_V3_APPROVED"
            or owner.get("status") != "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
            or owner.get("owner_family_key")
            != W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY
            or owner.get("owner_metadata_sha256")
            != W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256
            or owner.get("pair_count") != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or owner.get("source_count") != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or tuple(row.layout_key for row in files)
            != W02_MORPH_V3_PRIVATE_LAYOUTS):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family dependency chain did not close")
    zero_values = (
        candidate.get("formal_private_evaluation_runs"),
        candidate.get("private_payload_reads"),
        v1.get("formal_private_evaluation_runs"),
        v1.get("private_payload_reads"),
        v2.get("formal_private_evaluation_runs"),
        v2.get("private_payload_reads"),
        dev.get("formal_private_evaluation_runs"),
        dev.get("private_payload_reads"),
        shadow.get("formal_private_evaluation_runs"),
        shadow.get("private_payload_reads"),
        owner.get("formal_private_evaluation_runs"),
        owner.get("main_session_private_payload_reads"),
    )
    if any(value != 0 for value in zero_values):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private dependency already consumed private data")
    chain = {
        "boundary_contract_sha256": boundary.sha256(),
        "candidate_artifact_manifest_sha256":
            candidate["candidate_artifact_manifest_sha256"],
        "candidate_receipt_file_sha256":
            identities["candidate_receipt"]["sha256"],
        "candidate_semantic_sha256": candidate["candidate_semantic_sha256"],
        "compile_freeze_sha256": compile_freeze.sha256(),
        "owner_family_key": owner["owner_family_key"],
        "owner_metadata_sha256": owner["owner_metadata_sha256"],
        "owner_receipt_file_sha256": identities["owner_receipt"]["sha256"],
        "source_extension_v3_file_sha256":
            identities["source_extension_v3"]["sha256"],
        "v1_overlay_artifact_manifest_sha256":
            v1["overlay_artifact_manifest_sha256"],
        "v1_overlay_receipt_file_sha256": identities["v1_receipt"]["sha256"],
        "v1_overlay_semantic_sha256": v1["overlay_semantic_sha256"],
        "v2_overlay_artifact_manifest_sha256":
            v2["v2_overlay_artifact_manifest_sha256"],
        "v2_overlay_receipt_file_sha256": identities["v2_receipt"]["sha256"],
        "v2_overlay_semantic_sha256": v2["semantic_sha256"],
        "v3_dev_freeze_file_sha256": identities["v3_dev_freeze"]["sha256"],
        "v3_dev_pass_report_file_sha256":
            identities["v3_dev_report"]["sha256"],
        "v3_shadow_freeze_file_sha256":
            identities["v3_shadow_freeze"]["sha256"],
        "v3_shadow_pass_report_file_sha256":
            identities["v3_shadow_report"]["sha256"],
    }
    return {
        "artifact_chain": chain,
        "artifact_chain_sha256": _hash_value(chain),
        "files": files,
        "identities": identities,
        "owner": owner,
    }


def w02_morphology_successor_v3_private_guard_value(
        *, family_commitment: str, artifact_chain_sha256: str,
        code_freeze_sha256: str) -> dict[str, object]:
    """Build the canonical available guard without reading owner payload."""
    return {
        "artifact_chain_sha256": _sha256(
            artifact_chain_sha256, where="V3 private artifact chain"),
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_FIRST_RUN_GUARD"),
        "code_freeze_sha256": _sha256(
            code_freeze_sha256, where="V3 private code freeze"),
        "family_commitment": _sha256(
            family_commitment, where="V3 private family commitment"),
        "formal_private_evaluation_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "guard_version": W02_MORPH_V3_PRIVATE_FAMILY_VERSION,
        "owner_family_key": W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "run_id_policy": "EXACTLY_ONE",
        "stage_key": "W-02",
        "status": "AVAILABLE",
    }


def _guard_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


def build_w02_morphology_successor_v3_private_family_freeze(
        repository_root: str | Path) -> dict[str, object]:
    """Freeze the V3 family from public metadata and PASS evidence only."""
    repository = Path(repository_root).resolve()
    dependency = _dependency_state(repository)
    owner = dependency["owner"]
    code_rows, code_sha = _code_rows(repository)
    budget = _owner_budget(owner["resource_budget"])
    commitments = owner["commitments"]
    registration = build_v2_private_family_registration(
        "W-02",
        payload_commitment=commitments["payload_commitment"],
        case_commitment=commitments["case_commitment"],
        label_commitment=commitments["label_commitment"],
        cluster_commitment=commitments["cluster_commitment"],
        candidate_freeze_sha256=dependency["artifact_chain_sha256"],
        code_freeze_sha256=code_sha,
        resource_budget=budget,
    )
    guard = w02_morphology_successor_v3_private_guard_value(
        family_commitment=registration.family_commitment,
        artifact_chain_sha256=dependency["artifact_chain_sha256"],
        code_freeze_sha256=code_sha,
    )
    files = dependency["files"]
    return {
        "artifact_chain": dependency["artifact_chain"],
        "artifact_chain_sha256": dependency["artifact_chain_sha256"],
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_FAMILY_FREEZE"),
        "artifact_version": W02_MORPH_V3_PRIVATE_FAMILY_VERSION,
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "dimension_denominator_counts": owner["dimension_denominator_counts"],
        "evaluator_version": W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION,
        "first_run_guard_sha256": _guard_sha(guard),
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_successor_v3_route_dev_runs": 1,
        "formal_successor_v3_route_shadow_runs": 1,
        "formal_training_runs": 1,
        "hard_conjunct_keys": list(registration.policy.hard_conjunct_keys),
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "ne_policy": "BLOCK",
        "next_action": "W02_FORMAL_BLIND_PRIVATE_EVALUATION",
        "owner_commitments": dict(commitments),
        "owner_family_key": owner["owner_family_key"],
        "owner_file_counts": {
            row.layout_key: row.record_count for row in files
        },
        "owner_input_files": [row.to_dict() for row in files],
        "owner_metadata_sha256": owner["owner_metadata_sha256"],
        "owner_pair_count": owner["pair_count"],
        "owner_receipt_file_sha256":
            dependency["identities"]["owner_receipt"]["sha256"],
        "owner_source_count": owner["source_count"],
        "parent_evidence_files": [
            dependency["identities"][key]
            for key in (
                "boundary", "candidate_receipt", "v1_receipt", "v2_receipt",
                "v3_dev_freeze", "v3_dev_report", "v3_shadow_freeze",
                "v3_shadow_report", "source_extension_v3", "owner_receipt",
            )
        ],
        "private_family_registered": 1,
        "private_payload_reads": 0,
        "registration": registration.to_dict(),
        "release_key": "PH2-D03-V2",
        "resource_budget": budget.to_dict(),
        "stage_key": "W-02",
        "status": "W02_SUCCESSOR_V3_BLIND_PRIVATE_FAMILY_FROZEN",
        "support_hard_conjunct_keys": list(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS),
        "teacher_calls": 0,
        "threshold_reduction": 0,
        "zero_call_window_count": 3,
        "zero_write_required": 1,
    }


def publish_w02_morphology_successor_v3_private_family_freeze(
        repository_root: str | Path) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v3_private_family_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family freeze publication drifted")
    return target


def read_w02_morphology_successor_v3_private_family_freeze(
        repository_root: str | Path) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH)
    value = read_canonical_object(target)
    expected = build_w02_morphology_successor_v3_private_family_freeze(
        repository)
    if value != expected or target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family freeze no longer matches live identities")
    return value


def w02_morphology_successor_v3_private_registration_from_freeze(
        value: dict[str, object]) -> V2PrivateFamilyRegistration:
    registration = value.get("registration")
    budget = value.get("resource_budget")
    if not isinstance(registration, dict) or not isinstance(budget, dict):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family registration fields are invalid")
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
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private family registration canonical value drifted")
    return rebuilt


def publish_w02_morphology_successor_v3_private_family_root(
        repository_root: str | Path, family_root: str | Path) -> str:
    """Publish registration, freeze, ledger, and available guard off Git."""
    repository = Path(repository_root).resolve()
    freeze = read_w02_morphology_successor_v3_private_family_freeze(repository)
    registration = w02_morphology_successor_v3_private_registration_from_freeze(
        freeze)
    root = _family_root(family_root, require_exists=False)
    if root.exists():
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private formal family root already exists")
    root.mkdir(parents=False, exist_ok=False)
    write_immutable_json(
        registration.to_dict(), root / W02_MORPH_V3_PRIVATE_REGISTRATION_DOCUMENT)
    write_immutable_json(
        freeze, root / W02_MORPH_V3_PRIVATE_FAMILY_DOCUMENT)
    (root / W02_MORPH_V3_PRIVATE_EXPOSURE_LEDGER).mkdir(exist_ok=False)
    guard = w02_morphology_successor_v3_private_guard_value(
        family_commitment=registration.family_commitment,
        artifact_chain_sha256=str(freeze["artifact_chain_sha256"]),
        code_freeze_sha256=registration.code_freeze_sha256,
    )
    target = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_GUARD_AVAILABLE).parts)
    write_immutable_json(guard, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != freeze["first_run_guard_sha256"]:
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private available guard SHA drifted")
    return digest


def w02_morphology_successor_v3_private_run_identity(
        freeze: dict[str, object], family_freeze_sha256: str) -> str:
    registration = freeze.get("registration")
    if not isinstance(registration, dict):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private run registration is missing")
    return hashlib.sha256(canonical_json_bytes({
        "artifact_chain_sha256": freeze.get("artifact_chain_sha256"),
        "family_commitment": registration.get("family_commitment"),
        "family_freeze_sha256": _sha256(
            family_freeze_sha256, where="V3 private family freeze"),
        "owner_family_key": freeze.get("owner_family_key"),
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
    })).hexdigest()


def _intent_value(
        *, guard_sha256: str, run_identity_sha256: str) -> dict[str, object]:
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_RUN_INTENT"),
        "formal_private_evaluation_runs": 1,
        "guard_sha256": _sha256(guard_sha256, where="V3 private guard"),
        "private_payload_reads_before_guard": 0,
        "run_id": 1,
        "run_identity_sha256": _sha256(
            run_identity_sha256, where="V3 private run identity"),
        "stage_key": "W-02",
        "status": "RUN_INTENT_COMMITTED_BEFORE_PRIVATE_READ",
    }


def consume_w02_morphology_successor_v3_private_guard(
        family_root: str | Path, *, expected_guard_sha256: str,
        run_identity_sha256: str) -> None:
    """Commit intent and consume the only guard before any private open."""
    root = _family_root(family_root, require_exists=True)
    available = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_RUN_INTENT).parts)
    if consumed.exists() or intent.exists() or not available.is_file():
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private guard is unavailable or already consumed")
    if _sha256_file(available)[1] != _sha256(
            expected_guard_sha256, where="expected V3 private guard"):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private available guard bytes drifted")
    write_immutable_json(_intent_value(
        guard_sha256=expected_guard_sha256,
        run_identity_sha256=run_identity_sha256), intent)
    os.replace(available, consumed)


def verify_w02_morphology_successor_v3_private_consumed_guard(
        family_root: str | Path, *, expected_guard_sha256: str,
        run_identity_sha256: str) -> None:
    root = _family_root(family_root, require_exists=True)
    available = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_RUN_INTENT).parts)
    if available.exists() or not consumed.is_file() or not intent.is_file():
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private consumed guard state is incomplete")
    if (_sha256_file(consumed)[1] != expected_guard_sha256
            or read_canonical_object(intent) != _intent_value(
                guard_sha256=expected_guard_sha256,
                run_identity_sha256=run_identity_sha256)):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private consumed guard or run intent drifted")


__all__ = [
    "W02_MORPH_V3_PRIVATE_EXPOSURE_LEDGER",
    "W02_MORPH_V3_PRIVATE_FAMILY_CODE_PATHS",
    "W02_MORPH_V3_PRIVATE_FAMILY_DOCUMENT",
    "W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH",
    "W02_MORPH_V3_PRIVATE_FAMILY_VERSION",
    "W02_MORPH_V3_PRIVATE_FORMAL_FAMILY_NAME",
    "W02_MORPH_V3_PRIVATE_GUARD_AVAILABLE",
    "W02_MORPH_V3_PRIVATE_GUARD_CONSUMED",
    "W02_MORPH_V3_PRIVATE_REGISTRATION_DOCUMENT",
    "W02_MORPH_V3_PRIVATE_RUN_INTENT",
    "W02MorphologySuccessorV3PrivateFamilyError",
    "build_w02_morphology_successor_v3_private_family_freeze",
    "consume_w02_morphology_successor_v3_private_guard",
    "publish_w02_morphology_successor_v3_private_family_freeze",
    "publish_w02_morphology_successor_v3_private_family_root",
    "read_w02_morphology_successor_v3_private_family_freeze",
    "verify_w02_morphology_successor_v3_private_consumed_guard",
    "w02_morphology_successor_v3_private_guard_value",
    "w02_morphology_successor_v3_private_registration_from_freeze",
    "w02_morphology_successor_v3_private_run_identity",
]
