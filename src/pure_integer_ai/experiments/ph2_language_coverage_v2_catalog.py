"""从当前公开代码和冻结 manifest 构建 LC-COVERAGE-V2 缺口基线。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    ARTIFACT_KIND,
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    CARRIER_KEYS,
    DIRECTIONS,
    EXECUTION_STATE,
    INVARIANTS,
    TASK_KEYS,
    W02_RECEIPT_SHA256,
    CoverageV2CarrierRecord,
    CoverageV2Cell,
    CoverageV2EvidenceFile,
    CoverageV2TaskRecord,
    LanguageCapabilityCoverageV2Manifest,
    LanguageCoverageV2Error,
    LegacyCapabilitySplit,
)


MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_coverage_v2.json")

EVIDENCE_PATHS = (
    (
        "data/ph2/manifests/d03_v1/ph2_d03_post_publication_receipt_v1.json",
        "D03_RECEIPT",
    ),
    (
        "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json",
        "D03_GLOBAL",
    ),
    (
        "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json",
        "W03_RECEIPT",
    ),
    (
        "data/ph2/manifests/language_capability_baseline_v39.json",
        "COVERAGE_BASE",
    ),
    (
        "data/ph2/manifests/language_capability_baseline_v41.json",
        "LINEAGE_HEAD",
    ),
    (
        "src/pure_integer_ai/cognition/shared/formal_artifact.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/shared/identity.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/shared/parser_revision.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/shared/representation_rendering.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/shared/structure_order.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/understanding/occurrence_index.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/cognition/understanding/span_index.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_language_baseline_catalog.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_language_coverage_contract.py",
        "CONTRACT",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_language_coverage_v2_catalog.py",
        "IMPLEMENTATION",
    ),
    (
        "src/pure_integer_ai/experiments/ph2_language_coverage_v2_contract.py",
        "CONTRACT",
    ),
    (
        "tests/test_d02_language_coverage_v2.py",
        "TEST",
    ),
)

_BASE_REFS = tuple(sorted((
    "data/ph2/manifests/language_capability_baseline_v39.json",
    "data/ph2/manifests/language_capability_baseline_v41.json",
    "src/pure_integer_ai/experiments/ph2_language_coverage_contract.py",
)))

_CARRIER_AUDITS = {
    "DOCUMENT_CONTAINER": (
        "IN_SCOPE", "ABSENT", "PARTIAL", "ABSENT", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "BINARY_OR_LAYERED_RAW_ENVELOPE_ABSENT",
            "CONTAINER_REGION_AND_MEMBER_IDENTITY_ABSENT",
        ),
    ),
    "HTML": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/structure_order.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "DOM_ATTRIBUTE_AND_NODE_IDENTITY_ABSENT",
            "SOURCE_ORDER_AND_RENDER_ORDER_CONTRACT_ABSENT",
        ),
    ),
    "MARKDOWN": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/structure_order.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "BLOCK_INLINE_AND_ATTRIBUTE_IDENTITY_ABSENT",
            "MARKDOWN_RENDERER_AND_REVISION_CONTRACT_ABSENT",
        ),
    ),
    "MATH_NOTATION": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/formal_artifact.py",
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/shared/structure_order.py",
        ),
        (
            "BINDER_EXTENT_AND_NOTATION_TREE_ADAPTER_ABSENT",
            "EQUIVALENT_SURFACE_PROJECTION_ABSENT",
        ),
    ),
    "PLAIN_TEXT": (
        "IN_SCOPE", "REUSE", "PARTIAL", "PARTIAL", "PARTIAL", "PARTIAL",
        (
            "src/pure_integer_ai/cognition/shared/parser_revision.py",
            "src/pure_integer_ai/cognition/shared/representation_rendering.py",
            "src/pure_integer_ai/cognition/understanding/occurrence_index.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "CARRIER_QUALIFIED_LC16_ABLATION_ABSENT",
            "LC16_INDEPENDENT_VERIFIER_ABSENT",
        ),
    ),
    "REFERENCE_LINK_EMBED": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/understanding/occurrence_index.py",
        ),
        (
            "MISSING_WITHDRAWN_AND_ACCESS_BLOCKED_TARGET_STATES_ABSENT",
            "TARGET_FINGERPRINT_AND_EMBED_BINDING_ABSENT",
        ),
    ),
    "SENSORY_GROUNDING": (
        "WALL", "WALL", "WALL", "WALL", "WALL", "WALL",
        (
            "data/ph2/manifests/language_capability_baseline_v39.json",
            "src/pure_integer_ai/experiments/ph2_language_baseline_catalog.py",
        ),
        (
            "DEFINITIVE_TRUTH_NOT_AUTHORIZED",
            "PHYSICAL_GROUNDING_OUTSIDE_LANGUAGE_SCOPE",
        ),
    ),
    "SOURCE_CODE": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/formal_artifact.py",
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/shared/structure_order.py",
        ),
        (
            "AST_COMMENT_STRING_IMPORT_AND_CALL_ISOLATION_ABSENT",
            "LANGUAGE_AND_PARSER_AUTHORITY_BINDING_ABSENT",
        ),
    ),
    "TABLE_GRID": (
        "IN_SCOPE", "PARTIAL", "PARTIAL", "ABSENT", "ABSENT", "ABSENT",
        (
            "src/pure_integer_ai/cognition/shared/structure_order.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "CELL_HEADER_AND_MERGED_REGION_IDENTITY_ABSENT",
            "GRID_REVISION_AND_READING_ORDER_ADAPTER_ABSENT",
        ),
    ),
    "TRANSCRIBED_OCR_ASR": (
        "IN_SCOPE", "REUSE", "PARTIAL", "PARTIAL", "PARTIAL", "PARTIAL",
        (
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/understanding/occurrence_index.py",
            "src/pure_integer_ai/cognition/understanding/span_index.py",
        ),
        (
            "EXTERNAL_PERCEPTION_RECEIPT_BINDING_ABSENT",
            "PAGE_REGION_TIMECODE_OR_CUE_ALIGNMENT_ABSENT",
        ),
    ),
}


class LanguageCoverageV2CatalogError(RuntimeError):
    """当前仓库无法构建诚实的 LC-COVERAGE-V2。"""


def _path(repository: Path, relative_path: str) -> Path:
    path = (repository / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise LanguageCoverageV2CatalogError("evidence 路径逃逸") from error
    if not path.is_file():
        raise LanguageCoverageV2CatalogError(
            f"evidence 文件缺失: {relative_path}")
    return path


def _identity(
        repository: Path,
        relative_path: str,
        role: str,
        ) -> CoverageV2EvidenceFile:
    payload = _path(repository, relative_path).read_bytes()
    return CoverageV2EvidenceFile(
        relative_path, role, len(payload), hashlib.sha256(payload).hexdigest())


def _task_records() -> tuple[CoverageV2TaskRecord, ...]:
    records = []
    for task_key in TASK_KEYS:
        if task_key == "LC-16":
            records.append(CoverageV2TaskRecord(
                task_key,
                "AUDITED_ABSENT",
                0,
                0,
                tuple(sorted((
                    "data/ph2/manifests/language_capability_baseline_v39.json",
                    "src/pure_integer_ai/cognition/shared/formal_artifact.py",
                    "src/pure_integer_ai/cognition/shared/identity.py",
                    "src/pure_integer_ai/experiments/ph2_language_baseline_catalog.py",
                ))),
            ))
        else:
            records.append(CoverageV2TaskRecord(
                task_key,
                "HISTORICAL_SCOPE_ONLY",
                1,
                0,
                _BASE_REFS,
            ))
    return tuple(records)


def _carrier_records() -> tuple[CoverageV2CarrierRecord, ...]:
    records = []
    for carrier_key in CARRIER_KEYS:
        values = _CARRIER_AUDITS[carrier_key]
        records.append(CoverageV2CarrierRecord(
            carrier_key,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            tuple(sorted(values[6])),
            tuple(sorted(values[7])),
        ))
    return tuple(records)


def _cells(
        carriers: tuple[CoverageV2CarrierRecord, ...],
        ) -> tuple[CoverageV2Cell, ...]:
    by_key = {item.carrier_key: item for item in carriers}
    cells = []
    for task_key in TASK_KEYS:
        for carrier_key in CARRIER_KEYS:
            carrier = by_key[carrier_key]
            for direction in DIRECTIONS:
                if carrier_key == "SENSORY_GROUNDING":
                    cells.append(CoverageV2Cell(
                        task_key,
                        carrier_key,
                        direction,
                        "WALL",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        "WALL_BLOCKED",
                        carrier.evidence_refs,
                        carrier.gap_reasons,
                    ))
                    continue
                reasons = {
                    "CARRIER_QUALIFIED_EVIDENCE_ABSENT",
                    "HISTORICAL_RECEIPTS_DO_NOT_EXTEND_TO_THIS_CELL",
                }
                if task_key == "LC-16":
                    reasons.update({
                        "ARTIFACT_ENVELOPE_RUNTIME_ABSENT",
                        "CARRIER_PROJECTION_AND_REVISION_RUNTIME_ABSENT",
                    })
                cells.append(CoverageV2Cell(
                    task_key,
                    carrier_key,
                    direction,
                    "REQUIRED",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    "ABSENT",
                    carrier.evidence_refs,
                    tuple(sorted(reasons)),
                ))
    return tuple(cells)


def build_language_capability_coverage_v2(
        repository_root: str | Path,
        ) -> LanguageCapabilityCoverageV2Manifest:
    """构建 16×10×3 全显式、无 carrier PASS 的 v2 coverage baseline。"""
    repository = Path(repository_root).resolve()
    evidence = tuple(
        _identity(repository, path, role) for path, role in EVIDENCE_PATHS)
    carriers = _carrier_records()
    try:
        return LanguageCapabilityCoverageV2Manifest(
            2,
            ARTIFACT_KIND,
            ARTIFACT_VERSION,
            ARTIFACT_STATUS,
            W02_RECEIPT_SHA256,
            evidence,
            LegacyCapabilitySplit(
                "NON_TEXT_MEDIA",
                "TYPED_ARTIFACT_CARRIERS",
                "SENSORY_GROUNDING",
                "DEPRECATED_SPLIT_ONLY",
            ),
            _task_records(),
            carriers,
            _cells(carriers),
            CanonicalJsonObject.from_value(INVARIANTS),
            CanonicalJsonObject.from_value(EXECUTION_STATE),
        )
    except LanguageCoverageV2Error as error:
        raise LanguageCoverageV2CatalogError(
            "LC-COVERAGE-V2 构建失败") from error


__all__ = [
    "EVIDENCE_PATHS",
    "MANIFEST_PATH",
    "LanguageCoverageV2CatalogError",
    "build_language_capability_coverage_v2",
]
