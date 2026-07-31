"""LC-16 REFERENCE_LINK_EMBED payload、reference adapter 与冻结 parent pack 目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_adapter import (
    adapt_reference_link_embed_carrier_record,
    deserialize_reference_link_embed_carrier_materialization,
    serialize_reference_link_embed_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    CARRIER_KEY,
    EXECUTION_STATE,
    FORMAT_VERSION,
    LICENSE_ID,
    PARSER_PACKAGE,
    PARSER_VERSION,
    ReferenceLinkEmbedCarrierEvidenceFile,
    ReferenceLinkEmbedCarrierPayloadManifest,
    ReferenceLinkEmbedMaterializationIdentity,
    read_reference_link_embed_carrier_records,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    TypedCarrierBudget,
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
)


REFERENCE_LINK_EMBED_CARRIER_MANIFEST_PATH = "data/ph2/manifests/lc16_reference_link_embed_carrier_v1.json"
REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH = "data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample"
PARENT_PACK_PATH = "data/ph2/manifests/lc16_typed_carrier_pack_v1.json"
PARENT_PACK_SHA256 = (
    "29d1d54e6c547c5f375f807c92c16c9c9c69c97eb143f697df073747be8b0aa2")
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_reference_link_embed_carrier_adapter.py", "ADAPTER"),
    ("src/pure_integer_ai/experiments/ph2_reference_link_embed_carrier_catalog.py", "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_reference_link_embed_carrier_contract.py", "CONTRACT"),
    ("pyproject.toml", "DEPENDENCY"),
    ("tests/test_lc16_reference_link_embed_carrier_adapter.py", "TEST"),
)


class ReferenceLinkEmbedCarrierCatalogError(RuntimeError):
    """parent、payload、parser、adapter 输出或文件身份发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ReferenceLinkEmbedCarrierCatalogError("catalog 路径逃逸") from error
    return target


def _identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise ReferenceLinkEmbedCarrierCatalogError(f"catalog 文件缺失: {path.name}")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _evidence_files(root: Path) -> tuple[ReferenceLinkEmbedCarrierEvidenceFile, ...]:
    result = []
    for relative_path, role in _EVIDENCE_PATHS:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(ReferenceLinkEmbedCarrierEvidenceFile(
            relative_path, role, byte_count, sha256))
    return tuple(sorted(result, key=lambda item: (item.relative_path, item.role)))


def _parser_version() -> str:
    # parser 身份属于标准库 JSON 与 data-only reference-slot schema 合同。
    return PARSER_VERSION


def _check_budget(
        budget: TypedCarrierBudget,
        *,
        raw_unit_count: int,
        previous_raw_unit_count: int,
        structure_node_count: int,
        reference_count: int,
        revision_mapping_count: int,
        max_depth: int,
        ) -> None:
    if raw_unit_count > budget.max_raw_units:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED raw 超出 parent budget")
    if previous_raw_unit_count > budget.max_raw_units:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED previous raw 超出 parent budget")
    if structure_node_count > budget.max_structure_nodes:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED structure 超出 parent budget")
    if reference_count > budget.max_references:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED reference 超出 parent budget")
    if reference_count + revision_mapping_count > budget.max_edges:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED edge 超出 parent budget")
    if max_depth > budget.max_depth:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED depth 超出 parent budget")


def build_reference_link_embed_carrier_manifest(
        repository_root: str | Path,
        ) -> ReferenceLinkEmbedCarrierPayloadManifest:
    """从冻结 parent 与七行 payload 重建 raw/reference manifest。"""
    root = Path(repository_root).resolve()
    parent_path = _path(root, PARENT_PACK_PATH)
    parent_byte_count, parent_sha256 = _identity(parent_path)
    if parent_byte_count <= 0 or parent_sha256 != PARENT_PACK_SHA256:
        raise ReferenceLinkEmbedCarrierCatalogError("parent pack 文件身份漂移")
    try:
        parent = read_typed_carrier_pack_manifest(parent_path)
        verify_typed_carrier_pack_files(parent, repository_root=root)
    except Exception as error:
        raise ReferenceLinkEmbedCarrierCatalogError("parent pack 无法严格回验") from error
    if parent.sha256() != parent_sha256:
        raise ReferenceLinkEmbedCarrierCatalogError("parent pack canonical SHA 漂移")
    cases = tuple(item for item in parent.cases if item.carrier_key == CARRIER_KEY)
    budgets = tuple(item for item in parent.budgets
                    if item.carrier_key == CARRIER_KEY)
    if len(budgets) != 1 or len(cases) != budgets[0].max_cases:
        raise ReferenceLinkEmbedCarrierCatalogError("parent REFERENCE_LINK_EMBED case/budget 漂移")
    budget = budgets[0]
    expected_anchors = (
        ANCHOR_TEXT_RANGE, ANCHOR_DOCUMENT_REGION, ANCHOR_REFERENCE_SLOT)
    if any(item.anchor_kinds != expected_anchors for item in cases):
        raise ReferenceLinkEmbedCarrierCatalogError("parent REFERENCE_LINK_EMBED anchor 漂移")
    if any(item.payload_state != "UNMATERIALIZED"
           or item.runtime_state != "NOT_RUN" for item in cases):
        raise ReferenceLinkEmbedCarrierCatalogError("parent 已越过 payload/runtime 边界")

    sample_path = _path(root, REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH)
    _, sample_sha256 = _identity(sample_path)
    try:
        records = read_reference_link_embed_carrier_records(sample_path)
    except Exception as error:
        raise ReferenceLinkEmbedCarrierCatalogError("REFERENCE_LINK_EMBED sample 无法严格回读") from error
    if tuple(item.case_key for item in records) != tuple(
            item.case_key for item in cases):
        raise ReferenceLinkEmbedCarrierCatalogError("sample case_key 与 parent 不一致")
    if tuple(item.sample_kind for item in records) != tuple(
            item.sample_kind for item in cases):
        raise ReferenceLinkEmbedCarrierCatalogError("sample kind 与 parent 不一致")

    _parser_version()
    materializations = []
    for record in records:
        try:
            adapted = adapt_reference_link_embed_carrier_record(record)
            payload = serialize_reference_link_embed_carrier_materialization(adapted)
            restored = deserialize_reference_link_embed_carrier_materialization(payload, record)
        except Exception as error:
            raise ReferenceLinkEmbedCarrierCatalogError(
                f"adapter 无法物化 {record.sample_kind}") from error
        if restored != adapted:
            raise ReferenceLinkEmbedCarrierCatalogError("adapter reload identity 漂移")
        revision_mapping_count = (
            len(adapted.revisions[0].mappings) if adapted.revisions else 0)
        _check_budget(
            budget,
            raw_unit_count=record.raw_unit_count,
            previous_raw_unit_count=record.previous_unit_count,
            structure_node_count=len(adapted.structure_nodes),
            reference_count=len(adapted.references),
            revision_mapping_count=revision_mapping_count,
            max_depth=1,
        )
        materializations.append(ReferenceLinkEmbedMaterializationIdentity(
            record.case_key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            record.raw_unit_count,
            record.previous_unit_count,
            len(adapted.envelopes),
            len(adapted.anchors),
            len(adapted.structure_nodes),
            len(adapted.references),
            len(adapted.revisions),
            revision_mapping_count,
        ))
    return ReferenceLinkEmbedCarrierPayloadManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_PACK_PATH,
        parent_sha256,
        REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH,
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
    "REFERENCE_LINK_EMBED_CARRIER_MANIFEST_PATH",
    "REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH",
    "PARENT_PACK_PATH",
    "PARENT_PACK_SHA256",
    "ReferenceLinkEmbedCarrierCatalogError",
    "build_reference_link_embed_carrier_manifest",
]
