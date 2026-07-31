"""LC-16 typed carrier pack 的确定性目录构造器。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_language_coverage_v2_catalog import (
    MANIFEST_PATH as LANGUAGE_COVERAGE_V2_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    IN_SCOPE_CARRIER_KEYS,
    read_language_capability_coverage_v2,
    verify_language_capability_coverage_v2_files,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    DIRECTIONS,
    PACK_ANCHOR_KINDS,
    PACK_EXECUTION_STATE,
    SAMPLE_KINDS,
    SAMPLE_SPLITS,
    TypedCarrierBudget,
    TypedCarrierCase,
    TypedCarrierEvidenceFile,
    TypedCarrierPackManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)


TYPED_CARRIER_PACK_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_typed_carrier_pack_v1.json")
TYPED_CARRIER_PACK_BASE = 16616000
_EVIDENCE_PATHS = (
    (
        "src/pure_integer_ai/experiments/ph2_typed_carrier_pack_contract.py",
        "CONTRACT",
    ),
    (
        "data/ph2/manifests/language_capability_coverage_v2.json",
        "COVERAGE",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_typed_carrier_pack_catalog.py",
        "CATALOG",
    ),
    ("tests/test_lc16_typed_carrier_pack.py", "TEST"),
)


class TypedCarrierPackCatalogError(RuntimeError):
    """typed carrier pack 目录、coverage 或文件证据发生漂移。"""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, relative_path: str | Path) -> Path:
    relative = (relative_path if isinstance(relative_path, Path)
                else Path(*relative_path.split("/")))
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise TypedCarrierPackCatalogError("pack evidence 路径逃逸") from error
    return target


def _cluster_key(
        axis: int,
        carrier_index: int,
        sample_index: int,
        ) -> StableRecordKey:
    return StableRecordKey((
        TYPED_CARRIER_PACK_BASE,
        axis,
        carrier_index + 1,
        sample_index + 1,
    ))


def _budgets() -> tuple[TypedCarrierBudget, ...]:
    return tuple(
        TypedCarrierBudget(
            carrier,
            len(SAMPLE_KINDS),
            4096,
            512,
            1024,
            64,
            16 if carrier == "REFERENCE_LINK_EMBED" else 8,
        )
        for carrier in IN_SCOPE_CARRIER_KEYS
    )


def _cases() -> tuple[TypedCarrierCase, ...]:
    result: list[TypedCarrierCase] = []
    for carrier_index, carrier in enumerate(IN_SCOPE_CARRIER_KEYS):
        for sample_index, sample_kind in enumerate(SAMPLE_KINDS):
            result.append(TypedCarrierCase(
                StableRecordKey((
                    TYPED_CARRIER_PACK_BASE,
                    1,
                    carrier_index + 1,
                    sample_index + 1,
                )),
                carrier,
                sample_kind,
                StableRecordKey((
                    TYPED_CARRIER_PACK_BASE,
                    2,
                    carrier_index + 1,
                    sample_index + 1,
                )),
                SAMPLE_SPLITS[sample_kind],
                _cluster_key(3, carrier_index, sample_index),
                _cluster_key(4, carrier_index, sample_index),
                _cluster_key(5, carrier_index, sample_index),
                _cluster_key(6, carrier_index, sample_index),
                PACK_ANCHOR_KINDS[carrier],
                DIRECTIONS,
            ))
    return tuple(result)


def _evidence_files(root: Path) -> tuple[TypedCarrierEvidenceFile, ...]:
    result = []
    for relative_path, role in _EVIDENCE_PATHS:
        path = _path(root, relative_path)
        if not path.is_file():
            raise TypedCarrierPackCatalogError(
                f"pack evidence 文件缺失: {relative_path}")
        result.append(TypedCarrierEvidenceFile(
            relative_path,
            role,
            path.stat().st_size,
            _sha256_path(path),
        ))
    return tuple(sorted(result, key=lambda item: (item.relative_path, item.role)))


def build_typed_carrier_pack_manifest(
        repository_root: str | Path,
        ) -> TypedCarrierPackManifest:
    """从当前冻结 v2 baseline 构造不含 payload 的 63-case pack。"""
    root = Path(repository_root).resolve()
    coverage_path = _path(root, LANGUAGE_COVERAGE_V2_MANIFEST_PATH)
    try:
        coverage = read_language_capability_coverage_v2(coverage_path)
        verify_language_capability_coverage_v2_files(
            coverage, repository_root=root)
    except Exception as error:
        raise TypedCarrierPackCatalogError(
            "LC-COVERAGE-V2 无法作为 typed carrier pack 基线") from error
    coverage_sha256 = _sha256_path(coverage_path)
    if coverage_sha256 != coverage.sha256():
        raise TypedCarrierPackCatalogError("LC-COVERAGE-V2 SHA-256 回读漂移")
    return TypedCarrierPackManifest(
        1,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        coverage_sha256,
        _budgets(),
        _cases(),
        CanonicalJsonObject.from_value(PACK_EXECUTION_STATE),
        _evidence_files(root),
    )


__all__ = [
    "TYPED_CARRIER_PACK_BASE",
    "TYPED_CARRIER_PACK_MANIFEST_PATH",
    "TypedCarrierPackCatalogError",
    "build_typed_carrier_pack_manifest",
]
