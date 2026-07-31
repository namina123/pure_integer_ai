"""LC-16 OCR/ASR 转写载体的冻结 sample、adapter 和 manifest 目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_adapter import (
    adapt_transcribed_ocr_asr_carrier_record,
    deserialize_transcribed_ocr_asr_carrier_materialization,
    serialize_transcribed_ocr_asr_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    CARRIER_KEY,
    EXECUTION_STATE,
    FORMAT_VERSION,
    LICENSE_ID,
    PARSER_PACKAGE,
    PARSER_VERSION,
    TranscribedOcrAsrCarrierEvidenceFile,
    TranscribedOcrAsrCarrierPayloadManifest,
    TranscribedOcrAsrMaterializationIdentity,
    read_transcribed_ocr_asr_carrier_records,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    TypedCarrierBudget,
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
)


TRANSCRIBED_OCR_ASR_CARRIER_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_transcribed_ocr_asr_carrier_v1.json")
TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH = (
    "data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample")
PARENT_PACK_PATH = "data/ph2/manifests/lc16_typed_carrier_pack_v1.json"
PARENT_PACK_SHA256 = (
    "29d1d54e6c547c5f375f807c92c16c9c9c69c97eb143f697df073747be8b0aa2")
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_transcribed_ocr_asr_carrier_adapter.py", "ADAPTER"),
    ("src/pure_integer_ai/experiments/ph2_transcribed_ocr_asr_carrier_catalog.py", "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_transcribed_ocr_asr_carrier_contract.py", "CONTRACT"),
    ("pyproject.toml", "DEPENDENCY"),
    ("tests/test_lc16_transcribed_ocr_asr_carrier_adapter.py", "TEST"),
)


class TranscribedOcrAsrCarrierCatalogError(RuntimeError):
    """parent、sample、adapter 输出或文件身份发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise TranscribedOcrAsrCarrierCatalogError("catalog 路径逃逸") from error
    return target


def _identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise TranscribedOcrAsrCarrierCatalogError(
            f"catalog 文件缺失: {path.name}")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _evidence_files(root: Path) -> tuple[TranscribedOcrAsrCarrierEvidenceFile, ...]:
    result = []
    for relative_path, role in _EVIDENCE_PATHS:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(TranscribedOcrAsrCarrierEvidenceFile(
            relative_path, role, byte_count, sha256))
    return tuple(sorted(result, key=lambda item: (item.relative_path, item.role)))


def _check_budget(budget: TypedCarrierBudget, *, raw_unit_count: int,
                  previous_raw_unit_count: int, envelope_count: int,
                  structure_node_count: int, revision_count: int) -> None:
    if raw_unit_count > budget.max_raw_units:
        raise TranscribedOcrAsrCarrierCatalogError("TRANSCRIBED_OCR_ASR raw 超出 parent budget")
    if previous_raw_unit_count > budget.max_raw_units:
        raise TranscribedOcrAsrCarrierCatalogError("TRANSCRIBED_OCR_ASR previous raw 超出 parent budget")
    if structure_node_count > budget.max_structure_nodes:
        raise TranscribedOcrAsrCarrierCatalogError("TRANSCRIBED_OCR_ASR structure 超出 parent budget")
    if revision_count > budget.max_edges:
        raise TranscribedOcrAsrCarrierCatalogError("TRANSCRIBED_OCR_ASR revision 超出 parent budget")
    if envelope_count > budget.max_depth:
        raise TranscribedOcrAsrCarrierCatalogError("TRANSCRIBED_OCR_ASR envelope 超出 parent budget")


def build_transcribed_ocr_asr_carrier_manifest(
        repository_root: str | Path,
        ) -> TranscribedOcrAsrCarrierPayloadManifest:
    """从冻结 parent 与七行 payload 重建 alignment evidence manifest。"""
    root = Path(repository_root).resolve()
    parent_path = _path(root, PARENT_PACK_PATH)
    parent_byte_count, parent_sha256 = _identity(parent_path)
    if parent_byte_count <= 0 or parent_sha256 != PARENT_PACK_SHA256:
        raise TranscribedOcrAsrCarrierCatalogError("parent pack 文件身份漂移")
    try:
        parent = read_typed_carrier_pack_manifest(parent_path)
        verify_typed_carrier_pack_files(parent, repository_root=root)
    except Exception as error:
        raise TranscribedOcrAsrCarrierCatalogError("parent pack 无法严格回验") from error
    if parent.sha256() != parent_sha256:
        raise TranscribedOcrAsrCarrierCatalogError("parent pack canonical SHA 漂移")
    cases = tuple(item for item in parent.cases if item.carrier_key == CARRIER_KEY)
    budgets = tuple(item for item in parent.budgets if item.carrier_key == CARRIER_KEY)
    if len(budgets) != 1 or len(cases) != budgets[0].max_cases:
        raise TranscribedOcrAsrCarrierCatalogError("parent transcript case/budget 漂移")
    budget = budgets[0]
    expected_anchors = (
        ANCHOR_TEXT_RANGE, ANCHOR_DOCUMENT_REGION, ANCHOR_TRANSCRIPT_ALIGNMENT)
    if any(item.anchor_kinds != expected_anchors for item in cases):
        raise TranscribedOcrAsrCarrierCatalogError("parent transcript anchor 漂移")
    sample_path = _path(root, TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH)
    _, sample_sha256 = _identity(sample_path)
    try:
        records = read_transcribed_ocr_asr_carrier_records(sample_path)
    except Exception as error:
        raise TranscribedOcrAsrCarrierCatalogError("transcript sample 无法严格回读") from error
    if (tuple(item.case_key for item in records)
            != tuple(item.case_key for item in cases)
            or tuple(item.sample_kind for item in records)
            != tuple(item.sample_kind for item in cases)):
        raise TranscribedOcrAsrCarrierCatalogError("sample case 或 sample kind 与 parent 不一致")
    if any(item.payload_state != "UNMATERIALIZED" or item.runtime_state != "NOT_RUN"
           for item in cases):
        raise TranscribedOcrAsrCarrierCatalogError("parent 已越过 payload/runtime 边界")
    materializations = []
    for record in records:
        try:
            adapted = adapt_transcribed_ocr_asr_carrier_record(record)
            payload = serialize_transcribed_ocr_asr_carrier_materialization(adapted)
            restored = deserialize_transcribed_ocr_asr_carrier_materialization(payload, record)
        except Exception as error:
            raise TranscribedOcrAsrCarrierCatalogError(
                f"adapter 无法物化 {record.sample_kind}") from error
        if restored != adapted:
            raise TranscribedOcrAsrCarrierCatalogError("adapter reload identity 漂移")
        _check_budget(
            budget,
            raw_unit_count=record.raw_unit_count,
            previous_raw_unit_count=record.previous_unit_count,
            envelope_count=len(adapted.envelopes),
            structure_node_count=len(adapted.structure_nodes),
            revision_count=len(adapted.revisions),
        )
        materializations.append(TranscribedOcrAsrMaterializationIdentity(
            record.case_key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            record.raw_unit_count,
            record.previous_unit_count,
            len(adapted.envelopes),
            len(adapted.anchors),
            len(adapted.transcript_alignment_anchors),
            len(adapted.structure_nodes),
            len(adapted.revisions),
            len(adapted.revisions[0].mappings) if adapted.revisions else 0,
        ))
    return TranscribedOcrAsrCarrierPayloadManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_PACK_PATH,
        parent_sha256,
        TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH,
        sample_sha256,
        CARRIER_KEY,
        LICENSE_ID,
        PARSER_PACKAGE,
        PARSER_VERSION,
        budget,
        tuple(item.case_key for item in records),
        tuple(item.sample_kind for item in records),
        tuple(materializations),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
        _evidence_files(root),
    )


__all__ = [
    "PARENT_PACK_PATH", "PARENT_PACK_SHA256",
    "TRANSCRIBED_OCR_ASR_CARRIER_MANIFEST_PATH",
    "TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH",
    "TranscribedOcrAsrCarrierCatalogError",
    "build_transcribed_ocr_asr_carrier_manifest",
]
