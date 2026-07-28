"""从当前主仓文件构建 R-04 授权中心与生成交付吸收 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_authorized_center_generation_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COUNTEREXAMPLE_COVERAGE,
    EXECUTION_STATE,
    FOUR_STATE_MAPPING,
    HONESTY_LIMITS,
    REQUIREMENT_DECISIONS,
    AuthorizedCenterGenerationEvidenceFile,
    AuthorizedCenterGenerationManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


MANIFEST_PATH = Path(
    "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json")
SOURCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/generation_verification.py",
    "src/pure_integer_ai/experiments/authorized_center_runtime.py",
    "src/pure_integer_ai/experiments/authorized_generation_delivery.py",
    "src/pure_integer_ai/experiments/free_text_recall_runtime.py",
    "src/pure_integer_ai/experiments/free_text_revision_runtime.py",
    "src/pure_integer_ai/experiments/generation_verification_runtime.py",
    "src/pure_integer_ai/experiments/ph2_authorized_center_generation_catalog.py",
    "src/pure_integer_ai/experiments/ph2_authorized_center_generation_contract.py",
    "src/pure_integer_ai/experiments/question_answer_runtime.py",
    "src/pure_integer_ai/storage/location_manifest.py",
    "src/pure_integer_ai/storage/tiered_segment_store.py",
)
TEST_PATHS = (
    "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py",
    "tests/test_f00_generation_postcheck.py",
    "tests/test_g04_generation_postcheck.py",
    "tests/test_r04_authorized_center_generation_absorption.py",
)


class AuthorizedCenterGenerationCatalogError(RuntimeError):
    """R-04 evidence 文件缺失或路径逃逸。"""


def _identity(
        root: Path,
        relative_path: str,
        role: str,
        ) -> AuthorizedCenterGenerationEvidenceFile:
    """计算一个仓库内 evidence 文件的精确尺寸和 SHA-256。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AuthorizedCenterGenerationCatalogError(
            "R-04 catalog 路径逃逸") from error
    if not path.is_file():
        raise AuthorizedCenterGenerationCatalogError(
            f"R-04 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return AuthorizedCenterGenerationEvidenceFile(
        relative_path,
        role,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def build_authorized_center_generation_manifest(
        repository_root: str | Path,
        ) -> AuthorizedCenterGenerationManifest:
    """冻结五项裁决、四态、反例覆盖和全部实现/测试文件身份。"""
    root = Path(repository_root).resolve()
    evidence = (
        *(_identity(root, path, "SOURCE") for path in SOURCE_PATHS),
        *(_identity(root, path, "TEST") for path in TEST_PATHS),
    )
    return AuthorizedCenterGenerationManifest(
        1,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        CanonicalJsonObject.from_value(REQUIREMENT_DECISIONS),
        CanonicalJsonObject.from_value(FOUR_STATE_MAPPING),
        CanonicalJsonObject.from_value(COUNTEREXAMPLE_COVERAGE),
        CanonicalJsonObject.from_value(HONESTY_LIMITS),
        tuple(evidence),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "AuthorizedCenterGenerationCatalogError",
    "MANIFEST_PATH",
    "SOURCE_PATHS",
    "TEST_PATHS",
    "build_authorized_center_generation_manifest",
]
