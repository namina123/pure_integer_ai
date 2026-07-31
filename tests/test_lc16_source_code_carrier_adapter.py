"""LC-16 SOURCE_CODE token/AST adapter and catalog contract tests."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import ph2_source_code_carrier_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_adapter import (
    SourceCodeCarrierAdapterError,
    adapt_source_code_carrier_record,
    deserialize_source_code_carrier_materialization,
    serialize_source_code_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_catalog import (
    SOURCE_CODE_CARRIER_MANIFEST_PATH,
    SOURCE_CODE_CARRIER_SAMPLE_PATH,
    PARENT_PACK_SHA256,
    SourceCodeCarrierCatalogError,
    build_source_code_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_source_code_carrier_contract import (
    EXECUTION_STATE,
    SAMPLE_KINDS,
    SourceCodeCarrierContractError,
    SourceCodeCarrierPayloadManifest,
    read_source_code_carrier_manifest,
    read_source_code_carrier_records,
    verify_source_code_carrier_files,
    write_source_code_carrier_manifest,
    write_source_code_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*SOURCE_CODE_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_source_code_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_source_code_carrier_manifest(_ROOT)


def test_sample_freezes_seven_cc0_source_code_payloads(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert all(item.raw_utf8_sha256 == hashlib.sha256(
        item.raw_text.encode("utf-8")).hexdigest() for item in records)
    assert "missing colon" in records[1].raw_text
    assert records[3].raw_text.startswith("@future_transform")
    assert records[5].raw_text.endswith("\n")


def test_adapter_preserves_raw_tree_paths_and_ordinals(records):
    for record in records:
        materialization = adapt_source_code_carrier_record(record)
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
            tree = tuple(item for item in group if item.anchor_kind == ANCHOR_TREE_PATH)
            assert tree
            assert len({anchor.coordinates for anchor in tree}) == len(tree)
        assert tuple(item.ordinal for item in materialization.structure_nodes) == tuple(
            range(len(materialization.structure_nodes)))
        assert all(item.qualifiers and all(type(value) is int and value >= 0
                   for value in item.qualifiers)
                   for item in materialization.structure_nodes)
        assert all(item.role is None for item in materialization.structure_nodes)
        payload = serialize_source_code_carrier_materialization(materialization)
        assert deserialize_source_code_carrier_materialization(payload, record) == materialization


def test_token_ast_comment_string_import_call_and_parser_states_are_separate(records):
    materializations = tuple(adapt_source_code_carrier_record(item)
                             for item in records)
    receipts = [
        parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
        for item in materializations for node in item.structure_nodes
    ]
    assert {item["family"] for item in receipts} == {
        "AST", "PARSER_STATE", "TOKEN"}
    token_types = {item["type"] for item in receipts
                   if item["family"] == "TOKEN"}
    ast_types = {item["type"] for item in receipts
                 if item["family"] == "AST"}
    parser_states = {item["type"] for item in receipts
                     if item["family"] == "PARSER_STATE"}
    assert {"COMMENT", "STRING"} <= token_types
    assert {"Import", "ImportFrom", "Call"} <= ast_types
    assert {"AST_OK", "AST_SYNTAX_ERROR", "TOKENIZE_OK"} <= parser_states


def test_unknown_names_and_invalid_syntax_remain_structural_observations(records):
    unknown = next(item for item in records if item.sample_kind == "UNKNOWN")
    materialization = adapt_source_code_carrier_record(unknown)
    receipts = [bytes(node.qualifiers) for node in materialization.structure_nodes]
    assert any(b"future_transform" in payload for payload in receipts)
    negative = next(item for item in records if item.sample_kind == "NEGATIVE")
    negative_receipts = [bytes(node.qualifiers) for node in
                         adapt_source_code_carrier_record(negative).structure_nodes]
    assert any(b"AST_SYNTAX_ERROR" in payload for payload in negative_receipts)
    assert not any(b'\"type\":\"AST_OK\"' in payload
                   for payload in negative_receipts)


def test_revision_binds_text_document_tree_and_structure_mappings(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_source_code_carrier_record(record)
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
    payload = serialize_source_code_carrier_materialization(adapt_source_code_carrier_record(first))
    with pytest.raises(SourceCodeCarrierAdapterError, match="record 身份"):
        deserialize_source_code_carrier_materialization(payload, second)
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(SourceCodeCarrierAdapterError, match="字段不精确"):
        deserialize_source_code_carrier_materialization(canonical_json_bytes(value) + b"\n", first)
    with pytest.raises(SourceCodeCarrierAdapterError, match="newline"):
        deserialize_source_code_carrier_materialization(payload + b"\n", first)


def test_payload_and_manifest_writers_are_idempotent_never_overwrite(records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_source_code_carrier_records(records, sample_target) == sample_target
    assert write_source_code_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(SourceCodeCarrierContractError, match="内容不同"):
        write_source_code_carrier_records(records, sample_target)
    manifest_target = tmp_path / "manifest.json"
    assert write_source_code_carrier_manifest(manifest, manifest_target) == manifest_target
    assert read_source_code_carrier_manifest(manifest_target) == manifest
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(SourceCodeCarrierContractError, match="内容不同"):
        write_source_code_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_parser_budget_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.parser_package == "python-stdlib"
    assert manifest.parser_version == "tokenize-ast-v1"
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert tuple(item.envelope_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert all(item.anchor_count > item.envelope_count * 2
               for item in manifest.materializations)
    assert all(item.structure_node_count > 0 for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST"}
    verify_source_code_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_rebuilds_identically(manifest):
    stored = read_source_code_carrier_manifest(
        _ROOT / Path(*SOURCE_CODE_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_source_code_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_source_code_carrier_files(stored, repository_root=_ROOT)


def test_catalog_rejects_parser_drift_and_parent_sha(manifest, monkeypatch):
    monkeypatch.setattr(catalog, "_parser_version", lambda: (_ for _ in ()).throw(
        SourceCodeCarrierCatalogError("source_code parser version 漂移")))
    with pytest.raises(SourceCodeCarrierCatalogError, match="parser version"):
        build_source_code_carrier_manifest(_ROOT)


def test_manifest_rejects_execution_claim_and_extra_fields(manifest):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(SourceCodeCarrierContractError, match="全零"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(SourceCodeCarrierContractError, match="字段不精确"):
        SourceCodeCarrierPayloadManifest.from_dict(value)
