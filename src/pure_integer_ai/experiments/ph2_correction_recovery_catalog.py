"""从当前生产实现和直接依赖构建 R-03 修正恢复 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_correction_recovery_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COVERAGE_CONTRACT,
    EXECUTION_STATE,
    MECHANISM_CONTRACT,
    CorrectionRecoveryEvidenceFile,
    CorrectionRecoveryManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


MANIFEST_PATH = Path(
    "data/ph2/manifests/r03_correction_recovery_absorption_v1.json")
SOURCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/hypothesis_resolution.py",
    "src/pure_integer_ai/experiments/memory_isolation_runtime.py",
    "src/pure_integer_ai/experiments/ph2_correction_recovery_catalog.py",
    "src/pure_integer_ai/experiments/ph2_correction_recovery_contract.py",
    "src/pure_integer_ai/storage/backend.py",
    "src/pure_integer_ai/storage/memory_forget.py",
)
TEST_PATHS = (
    "tests/test_h04_hypothesis_resolution.py",
    "tests/test_m03_memory_event.py",
    "tests/test_m04_memory_aggregate.py",
    "tests/test_m08_memory_use.py",
    "tests/test_m09_memory_maintenance.py",
    "tests/test_m10_memory_batch_recovery.py",
    "tests/test_m11_memory_isolation.py",
    "tests/test_r03_correction_recovery_absorption.py",
)


class CorrectionRecoveryCatalogError(RuntimeError):
    """R-03 文件 inventory 无法闭合。"""


def _identity(
        root: Path,
        relative_path: str,
        role: str,
        ) -> CorrectionRecoveryEvidenceFile:
    path = (root / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CorrectionRecoveryCatalogError("R-03 catalog 路径逃逸") from error
    if not path.is_file():
        raise CorrectionRecoveryCatalogError(
            f"R-03 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return CorrectionRecoveryEvidenceFile(
        relative_path,
        role,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def build_correction_recovery_manifest(
        repository_root: str | Path,
        ) -> CorrectionRecoveryManifest:
    """冻结 R-03 实现、直接依赖和零执行状态。"""
    root = Path(repository_root).resolve()
    evidence = (
        *(_identity(root, path, "SOURCE") for path in SOURCE_PATHS),
        *(_identity(root, path, "TEST") for path in TEST_PATHS),
    )
    return CorrectionRecoveryManifest(
        1,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        CanonicalJsonObject.from_value(MECHANISM_CONTRACT),
        CanonicalJsonObject.from_value(COVERAGE_CONTRACT),
        tuple(evidence),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "MANIFEST_PATH",
    "SOURCE_PATHS",
    "TEST_PATHS",
    "CorrectionRecoveryCatalogError",
    "build_correction_recovery_manifest",
]
