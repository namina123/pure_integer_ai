"""从当前主仓文件构建 R-05 五类专属证明吸收 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_typed_proof_family_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COUNTEREXAMPLE_COVERAGE,
    EXECUTION_STATE,
    FOUR_STATE_MAPPING,
    HONESTY_LIMITS,
    REQUIREMENT_DECISIONS,
    TypedProofFamilyEvidenceFile,
    TypedProofFamilyManifest,
)


MANIFEST_PATH = Path(
    "data/ph2/manifests/r05_typed_proof_family_absorption_v1.json")
SOURCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/causal_execution.py",
    "src/pure_integer_ai/cognition/shared/event_time.py",
    "src/pure_integer_ai/cognition/shared/logic_executor.py",
    "src/pure_integer_ai/cognition/shared/modal_primitives.py",
    "src/pure_integer_ai/experiments/causal_relation_runtime.py",
    "src/pure_integer_ai/experiments/ph2_typed_proof_family_catalog.py",
    "src/pure_integer_ai/experiments/ph2_typed_proof_family_contract.py",
    "src/pure_integer_ai/experiments/typed_proof_family_contracts.py",
    "src/pure_integer_ai/experiments/typed_proof_family_runtime.py",
)
TEST_PATHS = (
    "tests/test_d02_lc05_event_time_aspect_course.py",
    "tests/test_d02c_authored_causes_course.py",
    "tests/test_d02d_authored_condition_course.py",
    "tests/test_d02d_authored_modal_course.py",
    "tests/test_d02d_authored_nested_scope_course.py",
    "tests/test_d02d_authored_not_course.py",
    "tests/test_r05_typed_proof_family_absorption.py",
    "tests/test_r06_event_time.py",
    "tests/test_r07_causal_relation_runtime.py",
    "tests/test_r08_logic_closure.py",
    "tests/test_s04_logic_executor.py",
)


class TypedProofFamilyCatalogError(RuntimeError):
    """R-05 evidence 文件缺失或路径逃逸。"""


def _identity(
        root: Path, relative_path: str, role: str,
        ) -> TypedProofFamilyEvidenceFile:
    """计算一个仓库内 evidence 文件的精确尺寸和 SHA-256。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TypedProofFamilyCatalogError(
            "R-05 catalog 路径逃逸") from error
    if not path.is_file():
        raise TypedProofFamilyCatalogError(
            f"R-05 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return TypedProofFamilyEvidenceFile(
        relative_path,
        role,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def build_typed_proof_family_manifest(
        repository_root: str | Path,
        ) -> TypedProofFamilyManifest:
    """冻结六项裁决、四态、反例覆盖和全部实现/测试文件身份。"""
    root = Path(repository_root).resolve()
    evidence = (
        *(_identity(root, path, "SOURCE") for path in SOURCE_PATHS),
        *(_identity(root, path, "TEST") for path in TEST_PATHS),
    )
    return TypedProofFamilyManifest(
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
    "MANIFEST_PATH",
    "SOURCE_PATHS",
    "TEST_PATHS",
    "TypedProofFamilyCatalogError",
    "build_typed_proof_family_manifest",
]
