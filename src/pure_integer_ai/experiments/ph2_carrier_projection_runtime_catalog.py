"""LC-16 carrier projection runtime 的 parent、核心依赖与证据目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    read_carrier_projection_mapper_manifest,
    verify_carrier_projection_mapper_files,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    EXECUTION_STATE,
    FORMAT_VERSION,
    RUNTIME_SCOPE,
    CarrierProjectionRuntimeFile,
    CarrierProjectionRuntimeManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_carrier_projection_runtime_v1.json")
PARENT_MAPPER_PATH = (
    "data/ph2/manifests/lc16_carrier_projection_mapper_v1.json")
PARENT_MAPPER_SHA256 = (
    "ba1f7d8d2921b6f7e796a0bad916638328235ad8242b0ef80f0fed8d3f405846")
_DEPENDENCY_PATHS = (
    ("src/pure_integer_ai/cognition/shared/artifact_carrier_structure.py",
     "ARTIFACT_PROJECTION"),
    ("src/pure_integer_ai/cognition/shared/candidate_runtime.py",
     "CANDIDATE_RUNTIME"),
    ("src/pure_integer_ai/cognition/shared/evidence_candidate.py",
     "EVIDENCE_ENGINE"),
    ("src/pure_integer_ai/cognition/shared/candidate_verifier.py",
     "INDEPENDENT_VERIFIER"),
    ("src/pure_integer_ai/cognition/shared/candidate_projection.py",
     "PROJECTION_GRAPH"),
    ("src/pure_integer_ai/cognition/shared/hypothesis_resolution.py",
     "RESOLVER"),
)
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_runtime_catalog.py",
     "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_runtime_contract.py",
     "CONTRACT"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_runtime.py",
     "RUNTIME"),
    ("tests/test_lc16_carrier_projection_runtime.py", "TEST"),
)


class CarrierProjectionRuntimeCatalogError(RuntimeError):
    """runtime parent、核心依赖或代码证据发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise CarrierProjectionRuntimeCatalogError("runtime catalog 路径逃逸") from error
    return target


def _identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise CarrierProjectionRuntimeCatalogError(
            f"runtime catalog 文件缺失: {path.name}")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _files(
        root: Path, values: tuple[tuple[str, str], ...],
        ) -> tuple[CarrierProjectionRuntimeFile, ...]:
    result = []
    for relative_path, role in values:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(CarrierProjectionRuntimeFile(
            relative_path, role, byte_count, sha256))
    return tuple(sorted(result, key=lambda item: item.role))


def build_carrier_projection_runtime_manifest(
        repository_root: str | Path,
        ) -> CarrierProjectionRuntimeManifest:
    """严格回验 parent mapper 后冻结共享 runtime 及其既有核心依赖。"""
    root = Path(repository_root).resolve()
    parent_path = _path(root, PARENT_MAPPER_PATH)
    _, parent_sha256 = _identity(parent_path)
    if parent_sha256 != PARENT_MAPPER_SHA256:
        raise CarrierProjectionRuntimeCatalogError(
            "runtime parent mapper 文件身份漂移")
    try:
        parent = read_carrier_projection_mapper_manifest(parent_path)
        verify_carrier_projection_mapper_files(parent, repository_root=root)
    except Exception as error:
        raise CarrierProjectionRuntimeCatalogError(
            "runtime parent mapper 无法严格回验") from error
    if parent.sha256() != parent_sha256:
        raise CarrierProjectionRuntimeCatalogError(
            "runtime parent mapper canonical SHA 漂移")
    return CarrierProjectionRuntimeManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_MAPPER_PATH,
        parent_sha256,
        _files(root, _DEPENDENCY_PATHS),
        CanonicalJsonObject.from_value(RUNTIME_SCOPE),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
        _files(root, _EVIDENCE_PATHS),
    )


__all__ = [
    "CARRIER_PROJECTION_RUNTIME_MANIFEST_PATH", "PARENT_MAPPER_PATH",
    "PARENT_MAPPER_SHA256", "CarrierProjectionRuntimeCatalogError",
    "build_carrier_projection_runtime_manifest",
]
