"""LC-16 方向 runtime 的 parent、依赖与公开证据目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_carrier_directional_manifest_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    DIRECTIONAL_SCOPE,
    EXECUTION_STATE,
    FORMAT_VERSION,
    CarrierDirectionalManifest,
    CarrierDirectionalManifestFile,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    read_carrier_projection_runtime_manifest,
    verify_carrier_projection_runtime_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


CARRIER_DIRECTIONAL_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_carrier_directional_runtime_v1.json")
PARENT_RUNTIME_PATH = (
    "data/ph2/manifests/lc16_carrier_projection_runtime_v1.json")
PARENT_RUNTIME_SHA256 = (
    "8f7b177da817dcc9df0ebfda81d096823bac7aa162a6a5660588bca2fe097419")
_DEPENDENCY_PATHS = (
    ("src/pure_integer_ai/cognition/shared/artifact_carrier_structure.py",
     "ARTIFACT_PROJECTION"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_mapper.py",
     "CARRIER_INPUT"),
    ("src/pure_integer_ai/cognition/shared/hypothesis.py",
     "EVIDENCE_RECORD"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_runtime.py",
     "PARENT_RUNTIME_CODE"),
)
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_carrier_directional_catalog.py",
     "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_carrier_directional_contract.py",
     "DIRECTIONAL_CONTRACT"),
    ("src/pure_integer_ai/experiments/ph2_carrier_directional_evaluator.py",
     "EVALUATOR"),
    ("src/pure_integer_ai/experiments/ph2_carrier_directional_manifest_contract.py",
     "MANIFEST_CONTRACT"),
    ("src/pure_integer_ai/experiments/ph2_carrier_directional_runtime.py",
     "RUNTIME"),
    ("tests/test_lc16_carrier_directional_runtime.py", "TEST"),
)


class CarrierDirectionalCatalogError(RuntimeError):
    """方向 parent、依赖或代码证据发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise CarrierDirectionalCatalogError(
            "directional catalog 路径逃逸") from error
    return target


def _identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise CarrierDirectionalCatalogError(
            f"directional catalog 文件缺失: {path.name}")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _files(
        root: Path, values: tuple[tuple[str, str], ...],
        ) -> tuple[CarrierDirectionalManifestFile, ...]:
    result = []
    for relative_path, role in values:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(CarrierDirectionalManifestFile(
            relative_path, role, byte_count, sha256))
    return tuple(sorted(result, key=lambda item: item.role))


def build_carrier_directional_manifest(
        repository_root: str | Path,
        ) -> CarrierDirectionalManifest:
    """严格回验共享 projection runtime 后冻结三向后继。"""
    root = Path(repository_root).resolve()
    parent_path = _path(root, PARENT_RUNTIME_PATH)
    _, parent_sha256 = _identity(parent_path)
    if parent_sha256 != PARENT_RUNTIME_SHA256:
        raise CarrierDirectionalCatalogError(
            "directional parent runtime 文件身份漂移")
    try:
        parent = read_carrier_projection_runtime_manifest(parent_path)
        verify_carrier_projection_runtime_files(parent, repository_root=root)
    except Exception as error:
        raise CarrierDirectionalCatalogError(
            "directional parent runtime 无法严格回验") from error
    if parent.sha256() != parent_sha256:
        raise CarrierDirectionalCatalogError(
            "directional parent runtime canonical SHA 漂移")
    return CarrierDirectionalManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_RUNTIME_PATH,
        parent_sha256,
        _files(root, _DEPENDENCY_PATHS),
        CanonicalJsonObject.from_value(DIRECTIONAL_SCOPE),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
        _files(root, _EVIDENCE_PATHS),
    )


__all__ = [
    "CARRIER_DIRECTIONAL_MANIFEST_PATH", "PARENT_RUNTIME_PATH",
    "PARENT_RUNTIME_SHA256", "CarrierDirectionalCatalogError",
    "build_carrier_directional_manifest",
]
