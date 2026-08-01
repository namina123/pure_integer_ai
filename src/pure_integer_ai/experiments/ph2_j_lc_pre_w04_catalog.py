"""构建 J-LC-PRE-W04 gate，并严格回验全部 canonical parents。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_carrier_directional_catalog import (
    CARRIER_DIRECTIONAL_MANIFEST_PATH,
    build_carrier_directional_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_directional_manifest_contract import (
    read_carrier_directional_manifest,
    verify_carrier_directional_files,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_catalog import (
    CARRIER_PROJECTION_MAPPER_MANIFEST_PATH,
    build_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    read_carrier_projection_mapper_manifest,
    verify_carrier_projection_mapper_files,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_catalog import (
    CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH,
    build_carrier_projection_runtime_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    read_carrier_projection_runtime_manifest,
    verify_carrier_projection_runtime_files,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_catalog import (
    build_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
    verify_d03_lc16_overlay_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_j_lc_pre_w04_contract import (
    ARTIFACT_KIND,
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    DEPENDENCY_ROLES,
    FORMAT_VERSION,
    MANIFEST_PATH,
    OPEN_GENERATION_SUFFIX,
    ORIGINAL_W02_RECEIPT_SHA256,
    PARENT_HEAD_SHA1,
    PUBLISHED_STATE,
    SUPPLEMENTAL_RECEIPT_STATUSES,
    W04_BLOCKING_FAILURE_KEYS,
    GateCarrierBinding,
    GateFileIdentity,
    JLcPreW04Error,
    JLcPreW04Gate,
    OpenGenerationBoundary,
    OriginalW02ReceiptCommitment,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_catalog import (
    MANIFEST_PATH as COVERAGE_PATH,
    build_language_capability_coverage_v2,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_catalog import (
    TYPED_CARRIER_PACK_MANIFEST_PATH,
    build_typed_carrier_pack_manifest,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_publication import (
    RECEIPT_RELATIVE_PATH as W02_SUPPLEMENTAL_RECEIPT_PATH,
    read_w02_lc16_supplemental_report,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_publication import (
    RECEIPT_RELATIVE_PATH as W03_SUPPLEMENTAL_RECEIPT_PATH,
    read_w03_lc16_supplemental_report,
)


OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
ORIGINAL_W03_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json")
EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_j_lc_pre_w04_catalog.py", "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_j_lc_pre_w04_contract.py", "CONTRACT"),
    ("tests/test_j_lc_pre_w04_gate.py", "TEST"),
)
DEPENDENCY_PATHS = (
    (str(COVERAGE_PATH).replace("\\", "/"), DEPENDENCY_ROLES[0]),
    (TYPED_CARRIER_PACK_MANIFEST_PATH, DEPENDENCY_ROLES[1]),
    (CARRIER_PROJECTION_MAPPER_MANIFEST_PATH, DEPENDENCY_ROLES[2]),
    (CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH, DEPENDENCY_ROLES[3]),
    (CARRIER_DIRECTIONAL_MANIFEST_PATH, DEPENDENCY_ROLES[4]),
    (OVERLAY_PATH, DEPENDENCY_ROLES[5]),
    (ORIGINAL_W03_RECEIPT_PATH, DEPENDENCY_ROLES[6]),
    (W02_SUPPLEMENTAL_RECEIPT_PATH, DEPENDENCY_ROLES[7]),
    (W03_SUPPLEMENTAL_RECEIPT_PATH, DEPENDENCY_ROLES[8]),
)


def _path(root: Path, relative: str) -> Path:
    target = (root / Path(*relative.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise JLcPreW04Error(f"pre-W04 文件缺失或越界: {relative}")
    return target


def _identity(root: Path, relative: str, role: str) -> GateFileIdentity:
    payload = _path(root, relative).read_bytes()
    return GateFileIdentity(
        role, relative, len(payload), hashlib.sha256(payload).hexdigest())


def _canonical_object(root: Path, relative: str) -> dict[str, Any]:
    payload = _path(root, relative).read_bytes()
    if payload.endswith(b"\n\n"):
        raise JLcPreW04Error(f"canonical parent newline 非法: {relative}")
    body = payload[:-1] if payload.endswith(b"\n") else payload
    value = parse_canonical_json_bytes(body, require_object=True)
    if canonical_json_bytes(value) != body:
        raise JLcPreW04Error(f"canonical parent bytes 漂移: {relative}")
    return value


def _assert_parent_blob(root: Path, relative: str) -> None:
    """证明依赖文件与 W-03 receipt 发布 commit 中的 blob 相同。"""
    current = subprocess.run(
        ("git", "hash-object", "--", relative), cwd=root,
        check=True, capture_output=True, text=True).stdout.strip()
    parent = subprocess.run(
        ("git", "rev-parse", f"{PARENT_HEAD_SHA1}:{relative}"), cwd=root,
        check=True, capture_output=True, text=True).stdout.strip()
    if current != parent:
        raise JLcPreW04Error(f"parent commit blob 漂移: {relative}")


def _verify_canonical_parents(root: Path) -> None:
    coverage = build_language_capability_coverage_v2(root)
    if coverage.canonical_bytes() != _path(root, str(COVERAGE_PATH)).read_bytes():
        raise JLcPreW04Error("LC-COVERAGE-V2 canonical rebuild 漂移")

    typed_path = _path(root, TYPED_CARRIER_PACK_MANIFEST_PATH)
    typed = read_typed_carrier_pack_manifest(typed_path)
    verify_typed_carrier_pack_files(typed, repository_root=root)
    if typed != build_typed_carrier_pack_manifest(root):
        raise JLcPreW04Error("typed carrier parent canonical rebuild 漂移")

    mapper_path = _path(root, CARRIER_PROJECTION_MAPPER_MANIFEST_PATH)
    mapper = read_carrier_projection_mapper_manifest(mapper_path)
    verify_carrier_projection_mapper_files(mapper, repository_root=root)
    if mapper != build_carrier_projection_mapper_manifest(root):
        raise JLcPreW04Error("carrier mapper canonical rebuild 漂移")

    runtime_path = _path(root, CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH)
    runtime = read_carrier_projection_runtime_manifest(runtime_path)
    verify_carrier_projection_runtime_files(runtime, repository_root=root)
    if runtime != build_carrier_projection_runtime_manifest(root):
        raise JLcPreW04Error("projection runtime canonical rebuild 漂移")

    directional_path = _path(root, CARRIER_DIRECTIONAL_MANIFEST_PATH)
    directional = read_carrier_directional_manifest(directional_path)
    verify_carrier_directional_files(directional, repository_root=root)
    if directional != build_carrier_directional_manifest(root):
        raise JLcPreW04Error("directional runtime canonical rebuild 漂移")

    overlay_path = _path(root, OVERLAY_PATH)
    overlay = read_d03_lc16_successor_overlay(overlay_path)
    verify_d03_lc16_overlay_files(overlay, repository_root=root)
    if overlay != build_d03_lc16_successor_overlay(root):
        raise JLcPreW04Error("D-03 LC16 overlay canonical rebuild 漂移")


def _carrier_bindings(root: Path, overlay: Any) -> tuple[GateCarrierBinding, ...]:
    result = []
    for course in overlay.carrier_courses:
        manifest = course.manifest_identity
        sample = course.sample_identity
        result.append(GateCarrierBinding(
            course.carrier_key,
            GateFileIdentity(
                "CARRIER_MANIFEST", manifest.relative_path,
                manifest.size_bytes, manifest.sha256),
            GateFileIdentity(
                "CARRIER_SAMPLE", sample.relative_path,
                sample.size_bytes, sample.sha256),
        ))
    for binding in result:
        for identity in (binding.manifest_identity, binding.sample_identity):
            actual = _identity(root, identity.relative_path, identity.role)
            if actual != identity:
                raise JLcPreW04Error("九载体 manifest/sample identity 漂移")
    return tuple(result)


def build_j_lc_pre_w04_gate(
        repository_root: str | Path,
        ) -> JLcPreW04Gate:
    """只有全部硬合取闭合时构建 PASS gate；否则 fail closed。"""
    root = Path(repository_root).resolve()
    _verify_canonical_parents(root)
    dependencies = tuple(
        _identity(root, relative, role) for relative, role in DEPENDENCY_PATHS)
    for relative, _ in DEPENDENCY_PATHS:
        _assert_parent_blob(root, relative)

    overlay = read_d03_lc16_successor_overlay(_path(root, OVERLAY_PATH))
    carriers = _carrier_bindings(root, overlay)
    for binding in carriers:
        _assert_parent_blob(root, binding.manifest_identity.relative_path)
        _assert_parent_blob(root, binding.sample_identity.relative_path)

    original_w03 = _canonical_object(root, ORIGINAL_W03_RECEIPT_PATH)
    state = original_w03.get("execution_state")
    if (original_w03.get("artifact_kind")
            != "PH2_W03_RUNTIME_EVIDENCE_RECEIPT"
            or original_w03.get("status") != "RUNTIME_EVIDENCED"
            or original_w03.get("w02_receipt_sha256")
            != ORIGINAL_W02_RECEIPT_SHA256
            or not isinstance(state, dict)
            or state.get("W03_RUNTIME_EVIDENCED") != 1
            or state.get("W04_STARTED") != 0
            or state.get("LANGUAGE_CAPABILITY_MASTERED") != 0
            or state.get("LANGUAGE_READINESS") != 0):
        raise JLcPreW04Error("原 W-02/W-03 receipt 状态未闭合")

    w02_value, _ = read_w02_lc16_supplemental_report(
        _path(root, W02_SUPPLEMENTAL_RECEIPT_PATH))
    w03_value, _ = read_w03_lc16_supplemental_report(
        _path(root, W03_SUPPLEMENTAL_RECEIPT_PATH))
    if w02_value.get("status") != "PASS" or w03_value.get("status") != "PASS":
        raise JLcPreW04Error("两道 supplemental receipt 非 PASS")

    failure_by_key = {
        item.failure_key: item.invalidation_suffix
        for item in overlay.failure_dependencies}
    if tuple(key for key in W04_BLOCKING_FAILURE_KEYS
             if "W-04" in failure_by_key.get(key, ())) \
            != W04_BLOCKING_FAILURE_KEYS:
        raise JLcPreW04Error("W-04 failure suffix 冻结边界漂移")
    open_account = next(
        (item for item in overlay.generation_accounts
         if item.account_key == "OPEN_GENERATION"), None)
    if open_account is None:
        raise JLcPreW04Error("OPEN_GENERATION account 缺失")
    open_generation = OpenGenerationBoundary(
        open_account.current_status,
        open_account.failure_suffix,
        open_account.runtime_evidenced,
        open_account.included_in_current_directional_evidence,
        open_account.aggregate_with_other_account,
        int("W-04" in open_account.failure_suffix),
    )
    if open_generation.failure_suffix != OPEN_GENERATION_SUFFIX:
        raise JLcPreW04Error("OPEN_GENERATION failure suffix 漂移")
    overlay_state = overlay.execution_state.to_value()
    if (overlay_state["W04_STARTED"] != 0
            or overlay_state["LANGUAGE_CAPABILITY_MASTERED"] != 0
            or overlay_state["LANGUAGE_READINESS"] != 0
            or overlay_state["open_generation_pass"] != 0):
        raise JLcPreW04Error("overlay execution state 越权")

    original_dependency = dependencies[DEPENDENCY_ROLES.index(
        "ORIGINAL_W03_RECEIPT_WITH_W02_COMMITMENT")]
    evidence = tuple(
        _identity(root, relative, role) for relative, role in EVIDENCE_PATHS)
    return JLcPreW04Gate(
        FORMAT_VERSION,
        ARTIFACT_KIND,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_HEAD_SHA1,
        dependencies,
        carriers,
        evidence,
        OriginalW02ReceiptCommitment(
            original_dependency.relative_path,
            original_dependency.sha256,
            "w02_receipt_sha256",
            original_w03["w02_receipt_sha256"],
        ),
        dict(SUPPLEMENTAL_RECEIPT_STATUSES),
        W04_BLOCKING_FAILURE_KEYS,
        W04_BLOCKING_FAILURE_KEYS,
        (),
        open_generation,
        dict(PUBLISHED_STATE),
    )


__all__ = [
    "DEPENDENCY_PATHS", "EVIDENCE_PATHS", "ORIGINAL_W03_RECEIPT_PATH",
    "OVERLAY_PATH", "build_j_lc_pre_w04_gate",
]
