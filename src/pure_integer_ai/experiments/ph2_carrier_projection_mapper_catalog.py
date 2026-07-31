"""LC-16 九类 data-only projection mapper 规则与依赖目录。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    EXECUTION_STATE,
    FORMAT_VERSION,
    CarrierProjectionDependencyFile,
    CarrierProjectionMapperEvidenceFile,
    CarrierProjectionMapperManifest,
    CarrierProjectionRule,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    read_typed_carrier_pack_manifest,
    verify_typed_carrier_pack_files,
)


CARRIER_PROJECTION_MAPPER_MANIFEST_PATH = (
    "data/ph2/manifests/lc16_carrier_projection_mapper_v1.json")
PARENT_PACK_PATH = "data/ph2/manifests/lc16_typed_carrier_pack_v1.json"
PARENT_PACK_SHA256 = (
    "29d1d54e6c547c5f375f807c92c16c9c9c69c97eb143f697df073747be8b0aa2")
_DEPENDENCY_PATHS = (
    ("DOCUMENT_CONTAINER", "data/ph2/manifests/lc16_document_container_carrier_v1.json"),
    ("HTML", "data/ph2/manifests/lc16_html_carrier_v1.json"),
    ("MARKDOWN", "data/ph2/manifests/lc16_markdown_carrier_v2.json"),
    ("MATH_NOTATION", "data/ph2/manifests/lc16_math_notation_carrier_v1.json"),
    ("PLAIN_TEXT", "data/ph2/manifests/lc16_plain_text_carrier_v1.json"),
    ("REFERENCE_LINK_EMBED", "data/ph2/manifests/lc16_reference_link_embed_carrier_v1.json"),
    ("SOURCE_CODE", "data/ph2/manifests/lc16_source_code_carrier_v2.json"),
    ("TABLE_GRID", "data/ph2/manifests/lc16_table_grid_carrier_v1.json"),
    ("TRANSCRIBED_OCR_ASR", "data/ph2/manifests/lc16_transcribed_ocr_asr_carrier_v1.json"),
)
_EVIDENCE_PATHS = (
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_mapper_catalog.py", "CATALOG"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_mapper_contract.py", "CONTRACT"),
    ("src/pure_integer_ai/experiments/ph2_carrier_projection_mapper.py", "MAPPER"),
    ("tests/test_lc16_carrier_projection_mapper.py", "TEST"),
)
_RULE_VALUES = (
    ("DOCUMENT_CONTAINER", "STRUCTURE_NODE", (("family",), ("type",)),
     ("CONTAINER", "DOCUMENT")),
    ("HTML", "STRUCTURE_NODE", (("node_type",), ("name",)),
     ("element", "html")),
    ("MARKDOWN", "STRUCTURE_NODE", (("type",), ("tag",)),
     ("heading_open", "h1")),
    ("MATH_NOTATION", "STRUCTURE_NODE", (("family",), ("type",)),
     ("TOKEN", "COMMAND")),
    ("PLAIN_TEXT", "ANCHOR", (("anchor_kind",),), (1,)),
    ("REFERENCE_LINK_EMBED", "STRUCTURE_NODE", (("family",), ("kind",)),
     ("REFERENCE", "LINK")),
    ("SOURCE_CODE", "STRUCTURE_NODE", (("family",), ("type",)),
     ("TOKEN", "NAME")),
    ("TABLE_GRID", "STRUCTURE_NODE", (("family",), ("type",)),
     ("GRID", "CELL")),
    ("TRANSCRIBED_OCR_ASR", "STRUCTURE_NODE", (("family",), ("type",)),
     ("TRANSCRIPT", "SEGMENT")),
)


class CarrierProjectionMapperCatalogError(RuntimeError):
    """parent、九类 carrier manifest、规则或代码证据发生漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    target = (root / Path(*relative_path.split("/"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise CarrierProjectionMapperCatalogError("catalog 路径逃逸") from error
    return target


def _identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise CarrierProjectionMapperCatalogError(f"catalog 文件缺失: {path.name}")
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _dependencies(root: Path) -> tuple[CarrierProjectionDependencyFile, ...]:
    result = []
    for carrier_key, relative_path in _DEPENDENCY_PATHS:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(CarrierProjectionDependencyFile(
            carrier_key, relative_path, byte_count, sha256))
    return tuple(result)


def _rules() -> tuple[CarrierProjectionRule, ...]:
    result = []
    for index, (carrier_key, input_kind, paths, expected) in enumerate(
            _RULE_VALUES, start=1):
        result.append(CarrierProjectionRule(
            StableRecordKey((16617600, 1, index)),
            carrier_key,
            input_kind,
            paths,
            CanonicalJsonObject.from_value({"values": list(expected)}),
            StableRecordKey((16617600, 2, index)),
        ))
    return tuple(result)


def _evidence(root: Path) -> tuple[CarrierProjectionMapperEvidenceFile, ...]:
    result = []
    for relative_path, role in _EVIDENCE_PATHS:
        byte_count, sha256 = _identity(_path(root, relative_path))
        result.append(CarrierProjectionMapperEvidenceFile(
            relative_path, role, byte_count, sha256))
    return tuple(sorted(result, key=lambda item: (item.relative_path, item.role)))


def build_carrier_projection_mapper_manifest(
        repository_root: str | Path,
        ) -> CarrierProjectionMapperManifest:
    """绑定 parent 与九类已冻结 adapter manifest，构造零执行 mapper 合同。"""
    root = Path(repository_root).resolve()
    parent_path = _path(root, PARENT_PACK_PATH)
    _, parent_sha256 = _identity(parent_path)
    if parent_sha256 != PARENT_PACK_SHA256:
        raise CarrierProjectionMapperCatalogError("parent pack 文件身份漂移")
    try:
        parent = read_typed_carrier_pack_manifest(parent_path)
        verify_typed_carrier_pack_files(parent, repository_root=root)
    except Exception as error:
        raise CarrierProjectionMapperCatalogError("parent pack 无法严格回验") from error
    if parent.sha256() != parent_sha256:
        raise CarrierProjectionMapperCatalogError("parent canonical SHA 漂移")
    return CarrierProjectionMapperManifest(
        FORMAT_VERSION,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        PARENT_PACK_PATH,
        parent_sha256,
        _dependencies(root),
        _rules(),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
        _evidence(root),
    )


__all__ = [
    "CARRIER_PROJECTION_MAPPER_MANIFEST_PATH", "PARENT_PACK_PATH",
    "PARENT_PACK_SHA256", "CarrierProjectionMapperCatalogError",
    "build_carrier_projection_mapper_manifest",
]
