"""从公开 overlay 构建 W-02 LC-16 supplemental 预注册 manifest。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
    verify_d03_lc16_overlay_files,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    CASE_COUNT,
    DIRECTION_EVALUATION_COUNT,
    EXECUTION_STATE,
    IN_SCOPE_CARRIER_KEYS,
    MANIFEST_PATH,
    MAX_LOGIC_OPERATIONS,
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_READS,
    MAX_WORKERS,
    OVERLAY_PATH,
    OVERLAY_SHA256,
    SAMPLE_KINDS,
    W02Lc16SupplementalManifest,
    SupplementalManifestFile,
    W02_PARENT_RECEIPT_SHA256,
)


W03_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json"
_DEPENDENCY_PATHS = (
    (OVERLAY_PATH, "D03_LC16_OVERLAY"),
    (W03_RECEIPT_PATH, "W02_PARENT_RECEIPT_COMMITMENT"),
)
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_w02_lc16_supplemental_catalog.py", "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_w02_lc16_supplemental_contract.py", "CONTRACT"),
    ("src/pure_integer_ai/experiments/ph2_w02_lc16_supplemental_evaluator.py", "EVALUATOR"),
    ("src/pure_integer_ai/experiments/ph2_w02_lc16_supplemental_publication.py", "PUBLICATION"),
    ("tests/test_w02_lc16_supplemental.py", "TEST"),
)


class W02Lc16SupplementalCatalogError(RuntimeError):
    """supplemental parent、文件身份或公开承诺发生漂移。"""


def _path(root: Path, relative: str) -> Path:
    """在仓库根内解析安全相对路径。"""
    target = (root / Path(*relative.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise W02Lc16SupplementalCatalogError("supplemental 路径逃逸") from error
    if not target.is_file():
        raise W02Lc16SupplementalCatalogError(f"supplemental 文件缺失: {relative}")
    return target


def _identity(root: Path, relative: str) -> tuple[int, str]:
    """读取文件大小和 SHA-256。"""
    payload = _path(root, relative).read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _files(
        root: Path, values: tuple[tuple[str, str], ...],
        ) -> tuple[SupplementalManifestFile, ...]:
    """构建排序后的依赖/证据文件身份。"""
    result = []
    for relative, role in values:
        size, sha256 = _identity(root, relative)
        result.append(SupplementalManifestFile(relative, role, size, sha256))
    return tuple(sorted(result, key=lambda item: item.role))


def build_w02_lc16_supplemental_manifest(
        repository_root: str | Path,
        ) -> W02Lc16SupplementalManifest:
    """回验 overlay 与 W-02 parent 承诺后构建未运行 manifest。"""
    root = Path(repository_root).resolve()
    overlay_path = _path(root, OVERLAY_PATH)
    size, overlay_sha256 = _identity(root, OVERLAY_PATH)
    if overlay_sha256 != OVERLAY_SHA256 or size <= 0:
        raise W02Lc16SupplementalCatalogError("overlay identity 漂移")
    try:
        overlay = read_d03_lc16_successor_overlay(overlay_path)
        verify_d03_lc16_overlay_files(overlay, repository_root=root)
    except Exception as error:
        raise W02Lc16SupplementalCatalogError("overlay 无法严格回验") from error
    if overlay.sha256() != OVERLAY_SHA256:
        raise W02Lc16SupplementalCatalogError("overlay canonical SHA 漂移")
    try:
        w03_receipt = json.loads(_path(root, W03_RECEIPT_PATH).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W02Lc16SupplementalCatalogError("W-03 receipt 无法回读") from error
    if (not isinstance(w03_receipt, dict)
            or w03_receipt.get("w02_receipt_sha256") != W02_PARENT_RECEIPT_SHA256):
        raise W02Lc16SupplementalCatalogError("W-02 parent receipt 承诺漂移")
    scope = {
        "ablation_count": 4,
        "case_count": CASE_COUNT,
        "carrier_count": len(IN_SCOPE_CARRIER_KEYS),
        "direction_count": 3,
        "direction_evaluations": DIRECTION_EVALUATION_COUNT,
        "independent_evaluator_module_separate": 1,
        "max_host_writes": 0,
        "max_logic_operations": MAX_LOGIC_OPERATIONS,
        "max_payload_bytes": MAX_PAYLOAD_BYTES,
        "max_payload_reads": MAX_PAYLOAD_READS,
        "max_workers": MAX_WORKERS,
        "sample_kind_count": len(SAMPLE_KINDS),
        "w02_bearing_dimension_count": 4,
    }
    return W02Lc16SupplementalManifest(
        1,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        OVERLAY_PATH,
        OVERLAY_SHA256,
        W02_PARENT_RECEIPT_SHA256,
        CanonicalJsonObject.from_value(scope),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
        _files(root, _DEPENDENCY_PATHS),
        _files(root, _EVIDENCE_PATHS),
    )


def verify_w02_lc16_supplemental_files(
        manifest: W02Lc16SupplementalManifest,
        *, repository_root: str | Path,
        ) -> None:
    """逐文件回验 supplemental manifest 的依赖和证据身份。"""
    if not isinstance(manifest, W02Lc16SupplementalManifest):
        raise TypeError("manifest 类型非法")
    root = Path(repository_root).resolve()
    for item in (*manifest.dependencies, *manifest.evidence_files):
        size, sha256 = _identity(root, item.relative_path)
        if size != item.byte_count or sha256 != item.sha256:
            raise W02Lc16SupplementalCatalogError(
                f"supplemental 文件身份漂移: {item.relative_path}")
    overlay = read_d03_lc16_successor_overlay(_path(root, OVERLAY_PATH))
    verify_d03_lc16_overlay_files(overlay, repository_root=root)
    if overlay.sha256() != manifest.parent_overlay_sha256:
        raise W02Lc16SupplementalCatalogError("supplemental parent overlay SHA 漂移")


__all__ = [
    "MANIFEST_PATH", "W03_RECEIPT_PATH", "W02Lc16SupplementalCatalogError",
    "build_w02_lc16_supplemental_manifest",
    "verify_w02_lc16_supplemental_files",
]
