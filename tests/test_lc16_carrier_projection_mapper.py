"""LC-16 九类 carrier-neutral 特征 mapper 与冻结目录测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import concept_identity
from pure_integer_ai.experiments import ph2_carrier_projection_mapper_catalog as catalog
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionMapper,
    CarrierProjectionMapperError,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_catalog import (
    CARRIER_PROJECTION_MAPPER_MANIFEST_PATH,
    PARENT_PACK_PATH,
    PARENT_PACK_SHA256,
    CarrierProjectionMapperCatalogError,
    build_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments.ph2_carrier_projection_mapper_contract import (
    EXECUTION_STATE,
    CarrierProjectionMapperContractError,
    CarrierProjectionMapperManifest,
    CarrierProjectionRule,
    read_carrier_projection_mapper_manifest,
    verify_carrier_projection_mapper_files,
    write_carrier_projection_mapper_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_adapter import (
    adapt_document_container_carrier_record,
)
from pure_integer_ai.experiments.ph2_document_container_carrier_contract import (
    read_document_container_carrier_records,
)
from pure_integer_ai.experiments.ph2_html_carrier_adapter import (
    adapt_html_carrier_record,
)
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    read_html_carrier_records,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_adapter import (
    adapt_markdown_carrier_record,
)
from pure_integer_ai.experiments.ph2_markdown_carrier_contract import (
    read_markdown_carrier_records,
)
from pure_integer_ai.experiments.ph2_math_notation_carrier_adapter import (
    adapt_math_notation_carrier_record,
)
from pure_integer_ai.experiments.ph2_math_notation_carrier_contract import (
    read_math_notation_carrier_records,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_adapter import (
    adapt_plain_text_carrier_record,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_contract import (
    read_plain_text_carrier_records,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_adapter import (
    adapt_reference_link_embed_carrier_record,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    read_reference_link_embed_carrier_records,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_adapter import (
    adapt_source_code_carrier_record,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_contract import (
    read_source_code_carrier_records,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_adapter import (
    adapt_table_grid_carrier_record,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_contract import (
    read_table_grid_carrier_records,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_adapter import (
    adapt_transcribed_ocr_asr_carrier_record,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    read_transcribed_ocr_asr_carrier_records,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    read_typed_carrier_pack_manifest,
)


_ROOT = Path(__file__).resolve().parents[1]
_CARRIERS = (
    ("DOCUMENT_CONTAINER", "data/ph2/lc16_document_container_carrier_v1.jsonl.sample",
     read_document_container_carrier_records, adapt_document_container_carrier_record),
    ("HTML", "data/ph2/lc16_html_carrier_v1.jsonl.sample",
     read_html_carrier_records, adapt_html_carrier_record),
    ("MARKDOWN", "data/ph2/lc16_markdown_carrier_v1.jsonl.sample",
     read_markdown_carrier_records, adapt_markdown_carrier_record),
    ("MATH_NOTATION", "data/ph2/lc16_math_notation_carrier_v1.jsonl.sample",
     read_math_notation_carrier_records, adapt_math_notation_carrier_record),
    ("PLAIN_TEXT", "data/ph2/lc16_plain_text_carrier_v1.jsonl.sample",
     read_plain_text_carrier_records, adapt_plain_text_carrier_record),
    ("REFERENCE_LINK_EMBED", "data/ph2/lc16_reference_link_embed_carrier_v1.jsonl.sample",
     read_reference_link_embed_carrier_records, adapt_reference_link_embed_carrier_record),
    ("SOURCE_CODE", "data/ph2/lc16_source_code_carrier_v1.jsonl.sample",
     read_source_code_carrier_records, adapt_source_code_carrier_record),
    ("TABLE_GRID", "data/ph2/lc16_table_grid_carrier_v1.jsonl.sample",
     read_table_grid_carrier_records, adapt_table_grid_carrier_record),
    ("TRANSCRIBED_OCR_ASR", "data/ph2/lc16_transcribed_ocr_asr_carrier_v1.jsonl.sample",
     read_transcribed_ocr_asr_carrier_records, adapt_transcribed_ocr_asr_carrier_record),
)


@pytest.fixture(scope="module")
def parent():
    return read_typed_carrier_pack_manifest(_ROOT / PARENT_PACK_PATH)


@pytest.fixture(scope="module")
def manifest():
    return build_carrier_projection_mapper_manifest(_ROOT)


def _materializations():
    result = {}
    for carrier_key, relative_path, reader, adapter in _CARRIERS:
        records = reader(_ROOT / relative_path)
        result[carrier_key] = adapter(records[0])
    return result


def _html_custom(text: str):
    records = read_html_carrier_records(
        _ROOT / "data/ph2/lc16_html_carrier_v1.jsonl.sample")
    base = records[0]
    record = replace(
        base,
        raw_text=text,
        raw_unit_count=len(text),
        raw_utf8_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return adapt_html_carrier_record(record)


def _find_html_node(materialization, name: str) -> int:
    for index, node in enumerate(materialization.structure_nodes):
        receipt = parse_canonical_json_bytes(
            bytes(node.qualifiers), require_object=True)
        if receipt.get("name") == name:
            return index
    raise AssertionError(f"未找到 HTML node: {name}")


def test_manifest_freezes_one_data_rule_and_dependency_per_carrier(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert tuple(item.carrier_key for item in manifest.rules) == tuple(
        item.carrier_key for item in manifest.dependencies)
    assert len(manifest.rules) == len(manifest.dependencies) == 9
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert {item.input_kind for item in manifest.rules} == {
        "ANCHOR", "STRUCTURE_NODE"}
    assert {item.role for item in manifest.evidence_files} == {
        "CATALOG", "CONTRACT", "MAPPER", "TEST"}


def test_same_mapper_materializes_all_nine_carrier_inputs(parent, manifest):
    mapper = CarrierProjectionMapper(parent)
    materializations = _materializations()
    for index, rule in enumerate(manifest.rules, start=1):
        materialization = materializations[rule.carrier_key]
        mapped = mapper.map(
            rule.carrier_key,
            materialization,
            rule,
            item_indices=(0,),
            input_key=(16617700, index),
        )
        assert mapped.case_key == materialization.record.case_key
        assert mapped.feature_identities == (
            concept_identity(rule.feature_key.stable_key()),)
        assert mapped.feature_identities[0] in mapped.visible_inputs
        assert mapped.envelope.identity == materialization.envelopes[0].identity
        assert mapped.stable_key()
        if rule.input_kind == "ANCHOR":
            assert mapped.anchor_identities and not mapped.structure_node_identities
        else:
            assert mapped.anchor_identities and mapped.structure_node_identities


def test_unknown_structure_adds_rule_data_without_mapper_code_change(parent):
    first = _html_custom(
        '<future-panel data-mode="first">内容</future-panel>')
    second = _html_custom(
        '<future-panel data-extra="second">新内容</future-panel>')
    rule = CarrierProjectionRule(
        StableRecordKey((16617690, 1)),
        "HTML",
        "STRUCTURE_NODE",
        (("node_type",), ("name",)),
        CanonicalJsonObject.from_value({
            "values": ["element", "future-panel"],
        }),
        StableRecordKey((16617690, 2)),
    )
    mapper = CarrierProjectionMapper(parent)
    mapped = tuple(mapper.map(
        "HTML",
        item,
        rule,
        item_indices=(_find_html_node(item, "future-panel"),),
        input_key=(16617710, index),
    ) for index, item in enumerate((first, second), start=1))
    assert mapped[0].feature_identities == mapped[1].feature_identities
    assert mapped[0].features[0].selected_values == rule.expected_values
    assert mapped[0].structure_node_identities != mapped[1].structure_node_identities


def test_mapper_rejects_carrier_lie_missing_path_and_value_mismatch(parent, manifest):
    mapper = CarrierProjectionMapper(parent)
    materializations = _materializations()
    html = materializations["HTML"]
    html_rule = next(item for item in manifest.rules if item.carrier_key == "HTML")
    with pytest.raises(CarrierProjectionMapperError, match="parent carrier"):
        mapper.map(
            "HTML", materializations["MARKDOWN"], html_rule,
            item_indices=(0,), input_key=(16617720, 1))
    missing = replace(html_rule, selector_paths=(("missing",), ("name",)))
    with pytest.raises(CarrierProjectionMapperError, match="path 缺失"):
        mapper.map(
            "HTML", html, missing,
            item_indices=(0,), input_key=(16617720, 2))
    mismatch = replace(
        html_rule,
        expected_values=CanonicalJsonObject.from_value({
            "values": ["element", "not-html"],
        }),
    )
    with pytest.raises(CarrierProjectionMapperError, match="不匹配"):
        mapper.map(
            "HTML", html, mismatch,
            item_indices=(0,), input_key=(16617720, 3))


def test_plain_text_anchor_route_does_not_require_fake_structure_node(parent, manifest):
    plain = _materializations()["PLAIN_TEXT"]
    rule = next(item for item in manifest.rules
                if item.carrier_key == "PLAIN_TEXT")
    mapped = CarrierProjectionMapper(parent).map(
        "PLAIN_TEXT", plain, rule,
        item_indices=(0,), input_key=(16617730, 1))
    assert mapped.structure_node_identities == ()
    node_rule = replace(rule, input_kind="STRUCTURE_NODE")
    with pytest.raises(CarrierProjectionMapperError, match="越界"):
        CarrierProjectionMapper(parent).map(
            "PLAIN_TEXT", plain, node_rule,
            item_indices=(0,), input_key=(16617730, 2))


def test_manifest_round_trip_files_and_no_overwrite(manifest, tmp_path):
    target = tmp_path / "mapper.json"
    assert write_carrier_projection_mapper_manifest(manifest, target) == target
    assert read_carrier_projection_mapper_manifest(target) == manifest
    assert write_carrier_projection_mapper_manifest(manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(CarrierProjectionMapperContractError, match="内容不同"):
        write_carrier_projection_mapper_manifest(manifest, target)
    verify_carrier_projection_mapper_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_canonical(manifest):
    stored = read_carrier_projection_mapper_manifest(
        _ROOT / CARRIER_PROJECTION_MAPPER_MANIFEST_PATH)
    rebuilt = build_carrier_projection_mapper_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()


def test_catalog_and_manifest_fail_closed(manifest, monkeypatch):
    monkeypatch.setattr(catalog, "PARENT_PACK_SHA256", "0" * 64)
    with pytest.raises(CarrierProjectionMapperCatalogError, match="parent pack"):
        build_carrier_projection_mapper_manifest(_ROOT)
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(CarrierProjectionMapperContractError, match="字段不精确"):
        CarrierProjectionMapperManifest.from_dict(value)
