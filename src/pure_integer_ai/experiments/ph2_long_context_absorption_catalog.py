"""从当前主仓文件构建 R-06 长上下文三件套吸收 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_long_context_absorption_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COUNTEREXAMPLE_COVERAGE,
    EXECUTION_STATE,
    FOUR_STATE_MAPPING,
    HONESTY_LIMITS,
    REQUIREMENT_DECISIONS,
    LongContextAbsorptionEvidenceFile,
    LongContextAbsorptionManifest,
)


MANIFEST_PATH = Path(
    "data/ph2/manifests/r06_long_context_absorption_v1.json")
SOURCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/generation_verification.py",
    "src/pure_integer_ai/cognition/shared/identity.py",
    "src/pure_integer_ai/cognition/shared/work_memory.py",
    "src/pure_integer_ai/cognition/understanding/span_index.py",
    "src/pure_integer_ai/experiments/authorized_center_runtime.py",
    "src/pure_integer_ai/experiments/authorized_generation_delivery.py",
    "src/pure_integer_ai/experiments/long_generation_checkpoint.py",
    "src/pure_integer_ai/experiments/long_input_hierarchy.py",
    "src/pure_integer_ai/experiments/persistent_conversation_agenda.py",
    "src/pure_integer_ai/experiments/ph2_long_context_absorption_catalog.py",
    "src/pure_integer_ai/experiments/ph2_long_context_absorption_contract.py",
    "src/pure_integer_ai/storage/segment_repository.py",
    "src/pure_integer_ai/storage/tiered_segment_store.py",
)
TEST_PATHS = (
    "tests/test_f00_generation_postcheck.py",
    "tests/test_f00_question_answer_runtime.py",
    "tests/test_g00_generation_plan.py",
    "tests/test_g02_generation_structure_plan.py",
    "tests/test_g03_generation_surface.py",
    "tests/test_g04_generation_postcheck.py",
    "tests/test_k02_sealed_segment_paging.py",
    "tests/test_k04_memory_hot_set.py",
    "tests/test_l04_recursive_span.py",
    "tests/test_r04_authorized_center_generation_absorption.py",
    "tests/test_r06_long_context_absorption.py",
    "tests/test_work_memory_lifecycle.py",
)


class LongContextAbsorptionCatalogError(RuntimeError):
    """R-06 evidence 文件缺失或路径逃逸。"""


def _identity(
        root: Path,
        relative_path: str,
        role: str,
        ) -> LongContextAbsorptionEvidenceFile:
    """计算一个仓库内 evidence 文件的精确尺寸和 SHA-256。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise LongContextAbsorptionCatalogError(
            "R-06 catalog 路径逃逸") from error
    if not path.is_file():
        raise LongContextAbsorptionCatalogError(
            f"R-06 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return LongContextAbsorptionEvidenceFile(
        relative_path,
        role,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def build_long_context_absorption_manifest(
        repository_root: str | Path,
        ) -> LongContextAbsorptionManifest:
    """冻结四项裁决、四态、反例覆盖和全部实现/测试文件身份。"""
    root = Path(repository_root).resolve()
    evidence = (
        *(_identity(root, path, "SOURCE") for path in SOURCE_PATHS),
        *(_identity(root, path, "TEST") for path in TEST_PATHS),
    )
    return LongContextAbsorptionManifest(
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
    "LongContextAbsorptionCatalogError",
    "MANIFEST_PATH",
    "SOURCE_PATHS",
    "TEST_PATHS",
    "build_long_context_absorption_manifest",
]
