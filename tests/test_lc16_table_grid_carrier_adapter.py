"""LC-16 TABLE_GRID structure adapter and catalog contract tests."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_GRID_RECT,
    ANCHOR_TEXT_RANGE,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import ph2_table_grid_carrier_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_adapter import (
    TableGridCarrierAdapterError,
    adapt_table_grid_carrier_record,
    deserialize_table_grid_carrier_materialization,
    serialize_table_grid_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_catalog import (
    TABLE_GRID_CARRIER_MANIFEST_PATH,
    TABLE_GRID_CARRIER_SAMPLE_PATH,
    PARENT_PACK_SHA256,
    TableGridCarrierCatalogError,
    build_table_grid_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_table_grid_carrier_contract import (
    EXECUTION_STATE,
    SAMPLE_KINDS,
    TableGridCarrierContractError,
    TableGridCarrierPayloadManifest,
    read_table_grid_carrier_manifest,
    read_table_grid_carrier_records,
    verify_table_grid_carrier_files,
    write_table_grid_carrier_manifest,
    write_table_grid_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*TABLE_GRID_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_table_grid_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_table_grid_carrier_manifest(_ROOT)


def test_sample_freezes_seven_cc0_table_grid_payloads(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert all(item.raw_utf8_sha256 == hashlib.sha256(
        item.raw_text.encode("utf-8")).hexdigest() for item in records)
    assert records[1].raw_text.splitlines()[1] == "Ada,10,extra"
    assert records[3].delimiter == "|"
    assert records[3].read_order == "COLUMN_MAJOR"
    assert '"contains, comma"' in records[5].raw_text
    assert records[6].merged_rectangles == ((0, 0, 0, 1),)


def test_adapter_preserves_raw_grid_rectangles_and_ordinals(records):
    for record in records:
        materialization = adapt_table_grid_carrier_record(record)
        expected = ((record.previous_text, record.raw_text)
                    if record.sample_kind == "REVISION"
                    else (record.raw_text,))
        assert tuple(
            "".join(chr(value) for value in envelope.raw_units)
            for envelope in materialization.envelopes) == expected
        for envelope in materialization.envelopes:
            group = tuple(item for item in materialization.anchors
                          if item.envelope_identity == envelope.identity)
            assert sum(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group) == 1
            assert sum(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group) == 1
            grid = tuple(item for item in group
                         if item.anchor_kind == ANCHOR_GRID_RECT)
            assert grid
            assert all(len(anchor.coordinates) == 4 for anchor in grid)
        assert tuple(item.ordinal for item in materialization.structure_nodes) == tuple(
            range(len(materialization.structure_nodes)))
        assert all(item.qualifiers and all(type(value) is int and value >= 0
                   for value in item.qualifiers)
                   for item in materialization.structure_nodes)
        assert all(item.role is None for item in materialization.structure_nodes)
        payload = serialize_table_grid_carrier_materialization(materialization)
        assert deserialize_table_grid_carrier_materialization(payload, record) == materialization


def test_cell_row_column_merge_read_order_and_parser_states_are_separate(records):
    materializations = tuple(adapt_table_grid_carrier_record(item)
                             for item in records)
    receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for item in materializations for node in item.structure_nodes
    ]
    assert {item["family"] for item in receipts} == {"GRID", "PARSER_STATE"}
    grid_types = {item["type"] for item in receipts
                  if item["family"] == "GRID"}
    parser_states = {item["type"] for item in receipts
                     if item["family"] == "PARSER_STATE"}
    assert {"CELL", "COLUMN", "MERGED_REGION", "READ_ORDER", "ROW"} <= grid_types
    assert {"GRID_OK", "RAGGED_GRID"} <= parser_states


def test_ragged_quoted_unknown_and_merged_layouts_remain_data_observations(records):
    negative = next(item for item in records if item.sample_kind == "NEGATIVE")
    negative_receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for node in adapt_table_grid_carrier_record(negative).structure_nodes
    ]
    ragged = next(item for item in negative_receipts
                  if item["type"] == "RAGGED_GRID")
    assert ragged["details"]["row_widths"] == [2, 3, 1]

    generation = next(item for item in records if item.sample_kind == "GENERATION")
    generation_receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for node in adapt_table_grid_carrier_record(generation).structure_nodes
    ]
    cell_values = [item["details"]["value"] for item in generation_receipts
                   if item["type"] == "CELL"]
    assert "contains, comma" in cell_values

    unknown = next(item for item in records if item.sample_kind == "UNKNOWN")
    unknown_receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for node in adapt_table_grid_carrier_record(unknown).structure_nodes
    ]
    read_order = next(item for item in unknown_receipts
                      if item["type"] == "READ_ORDER")
    assert read_order["details"] == {
        "candidate": "COLUMN_MAJOR",
        "cells": [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
    }
    assert any(item["type"] == "CELL" and item["details"]["value"] == "value"
               for item in unknown_receipts)

    retention = next(item for item in records if item.sample_kind == "RETENTION")
    retention_receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for node in adapt_table_grid_carrier_record(retention).structure_nodes
    ]
    merged = next(item for item in retention_receipts
                  if item["type"] == "MERGED_REGION")
    assert merged["path"] == [0, 0, 0, 1]
    assert merged["details"]["cell_count"] == 2


def test_revision_binds_text_document_grid_and_structure_mappings(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_table_grid_carrier_record(record)
    old_source, new_source = materialization.sources
    assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
    assert old_source.versions.parser.value == 1
    assert new_source.versions.parser.value == 2
    revision = materialization.revisions[0]
    assert ArtifactCarrierRevision.from_stable_key(revision.stable_key()) == revision
    kinds = {item.mapping_kind for item in revision.mappings}
    assert kinds == {1, 2}
    assert any(not item.new_identities for item in revision.mappings)


def test_serializer_rejects_record_drift_and_noncanonical_bytes(records):
    first, second = records[:2]
    payload = serialize_table_grid_carrier_materialization(adapt_table_grid_carrier_record(first))
    with pytest.raises(TableGridCarrierAdapterError, match="record 身份"):
        deserialize_table_grid_carrier_materialization(payload, second)
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(TableGridCarrierAdapterError, match="字段不精确"):
        deserialize_table_grid_carrier_materialization(canonical_json_bytes(value) + b"\n", first)
    with pytest.raises(TableGridCarrierAdapterError, match="newline"):
        deserialize_table_grid_carrier_materialization(payload + b"\n", first)


def test_payload_and_manifest_writers_are_idempotent_never_overwrite(records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_table_grid_carrier_records(records, sample_target) == sample_target
    assert write_table_grid_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(TableGridCarrierContractError, match="内容不同"):
        write_table_grid_carrier_records(records, sample_target)
    manifest_target = tmp_path / "manifest.json"
    assert write_table_grid_carrier_manifest(manifest, manifest_target) == manifest_target
    assert read_table_grid_carrier_manifest(manifest_target) == manifest
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(TableGridCarrierContractError, match="内容不同"):
        write_table_grid_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_parser_budget_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.parser_package == "python-stdlib"
    assert manifest.parser_version == "csv-grid-v1"
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert tuple(item.envelope_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert all(item.anchor_count > item.envelope_count * 2
               for item in manifest.materializations)
    assert all(item.structure_node_count > 0 for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST"}
    verify_table_grid_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_rebuilds_identically(manifest):
    stored = read_table_grid_carrier_manifest(
        _ROOT / Path(*TABLE_GRID_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_table_grid_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_table_grid_carrier_files(stored, repository_root=_ROOT)


def test_catalog_rejects_parser_drift_and_parent_sha(manifest, monkeypatch):
    monkeypatch.setattr(catalog, "_parser_version", lambda: (_ for _ in ()).throw(
        TableGridCarrierCatalogError("table_grid parser version 漂移")))
    with pytest.raises(TableGridCarrierCatalogError, match="parser version"):
        build_table_grid_carrier_manifest(_ROOT)


def test_manifest_rejects_execution_claim_and_extra_fields(manifest):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(TableGridCarrierContractError, match="全零"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(TableGridCarrierContractError, match="字段不精确"):
        TableGridCarrierPayloadManifest.from_dict(value)
