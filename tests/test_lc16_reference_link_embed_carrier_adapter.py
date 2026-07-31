"""LC-16 REFERENCE_LINK_EMBED raw/reference adapter 与 catalog 合同测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
    REFERENCE_ACCESS_BLOCKED,
    REFERENCE_RESOLVED,
    REFERENCE_UNRESOLVED,
    REVISION_MAP_ANCHOR,
    REVISION_MAP_REFERENCE,
    REVISION_MAP_STRUCTURE_NODE,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import (
    ph2_reference_link_embed_carrier_catalog as catalog,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_adapter import (
    ReferenceLinkEmbedCarrierAdapterError,
    adapt_reference_link_embed_carrier_record,
    deserialize_reference_link_embed_carrier_materialization,
    serialize_reference_link_embed_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_catalog import (
    PARENT_PACK_SHA256,
    REFERENCE_LINK_EMBED_CARRIER_MANIFEST_PATH,
    REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH,
    ReferenceLinkEmbedCarrierCatalogError,
    build_reference_link_embed_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_reference_link_embed_carrier_contract import (
    EXECUTION_STATE,
    SAMPLE_KINDS,
    ReferenceLinkEmbedCarrierContractError,
    ReferenceLinkEmbedCarrierPayloadManifest,
    read_reference_link_embed_carrier_manifest,
    read_reference_link_embed_carrier_records,
    verify_reference_link_embed_carrier_files,
    write_reference_link_embed_carrier_manifest,
    write_reference_link_embed_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*REFERENCE_LINK_EMBED_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_reference_link_embed_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_reference_link_embed_carrier_manifest(_ROOT)


def _revision_record(record, previous: str, current: str):
    return replace(
        record,
        previous_text=previous,
        previous_unit_count=len(previous),
        previous_utf8_sha256=hashlib.sha256(previous.encode()).hexdigest(),
        raw_text=current,
        raw_unit_count=len(current),
        raw_utf8_sha256=hashlib.sha256(current.encode()).hexdigest(),
    )


def _receipts(materialization):
    return [parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
            for node in materialization.structure_nodes]


def test_sample_freezes_seven_cc0_reference_link_embed_payloads(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert all(item.raw_utf8_sha256 == hashlib.sha256(
        item.raw_text.encode("utf-8")).hexdigest() for item in records)
    assert "#guide-section" in records[0].raw_text
    assert records[1].raw_text == '{"content":"broken"'
    assert "FUTURE_EMBED" in records[3].raw_text
    assert "embed-old" in records[4].previous_text


def test_adapter_preserves_raw_text_document_regions_and_reference_slots(records):
    for record in records:
        materialization = adapt_reference_link_embed_carrier_record(record)
        expected = ((record.previous_text, record.raw_text)
                    if record.sample_kind == "REVISION"
                    else (record.raw_text,))
        assert tuple(
            "".join(chr(value) for value in envelope.raw_units)
            for envelope in materialization.envelopes) == expected
        for envelope in materialization.envelopes:
            group = tuple(item for item in materialization.anchors
                          if item.envelope_identity == envelope.identity)
            assert sum(item.anchor_kind == ANCHOR_TEXT_RANGE
                       for item in group) == 1
            assert sum(item.anchor_kind == ANCHOR_DOCUMENT_REGION
                       for item in group) >= 1
            assert all(item.anchor_kind in {
                ANCHOR_TEXT_RANGE, ANCHOR_DOCUMENT_REGION,
                ANCHOR_REFERENCE_SLOT} for item in group)
        assert all(item.role is None for item in materialization.structure_nodes)
        assert all(item.qualifiers for item in materialization.structure_nodes)
        payload = serialize_reference_link_embed_carrier_materialization(
            materialization)
        assert deserialize_reference_link_embed_carrier_materialization(
            payload, record) == materialization


def test_reference_states_and_unknown_kinds_are_explicit_without_network(records):
    positive = adapt_reference_link_embed_carrier_record(records[0])
    assert {item.target_state for item in positive.references} == {REFERENCE_RESOLVED}
    assert all(item.target_source == positive.sources[0]
               and item.target_anchor is not None for item in positive.references)

    ambiguous = adapt_reference_link_embed_carrier_record(records[2])
    assert {item.target_state for item in ambiguous.references} == {
        REFERENCE_UNRESOLVED}
    unknown = adapt_reference_link_embed_carrier_record(records[3])
    assert {item.target_state for item in unknown.references} == {
        REFERENCE_ACCESS_BLOCKED}
    future = next(item for item in _receipts(unknown)
                  if item["family"] == "REFERENCE")
    assert future["kind"] == "FUTURE_EMBED"

    negative = adapt_reference_link_embed_carrier_record(records[1])
    assert {item["type"] for item in _receipts(negative)} == {"JSON_ERROR"}
    assert not negative.references


@pytest.mark.parametrize(("raw_text", "expected_state"), (
    ('{"content":"x","content":"y","local_targets":[],"references":[]}',
     "JSON_ERROR"),
    ('{"content":"x","local_targets":[],"references":[],"weight":1.5}',
     "JSON_ERROR"),
    ('{"content":"guide","local_targets":[],"references":[{"id":"r","kind":"LINK","span":[0,5],"surface":"other","target":""}]}',
     "REFERENCE_SCHEMA_ERROR"),
))
def test_duplicate_float_and_surface_drift_fail_loud(
        records, raw_text, expected_state):
    record = replace(
        records[0], raw_text=raw_text, raw_unit_count=len(raw_text),
        raw_utf8_sha256=hashlib.sha256(raw_text.encode()).hexdigest())
    materialization = adapt_reference_link_embed_carrier_record(record)
    assert {item["type"] for item in _receipts(materialization)} == {
        expected_state}
    assert not materialization.references


def test_revision_maps_anchor_node_reference_and_preserves_slot_identity(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_reference_link_embed_carrier_record(record)
    old_source, new_source = materialization.sources
    assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
    assert old_source.versions.parser.value == 1
    assert new_source.versions.parser.value == 2
    revision = materialization.revisions[0]
    assert ArtifactCarrierRevision.from_stable_key(revision.stable_key()) == revision
    assert {item.mapping_kind for item in revision.mappings} == {
        REVISION_MAP_ANCHOR, REVISION_MAP_STRUCTURE_NODE, REVISION_MAP_REFERENCE}
    deleted = tuple(item for item in revision.mappings
                    if item.mapping_kind == REVISION_MAP_REFERENCE
                    and not item.new_identities)
    assert len(deleted) == 1


def test_revision_uses_slot_identity_when_span_moves(records):
    base = next(item for item in records if item.sample_kind == "REVISION")
    previous = '{"content":"Old guide.","local_targets":[{"id":"guide-target","span":[4,9]}],"references":[{"id":"link-guide","kind":"LINK","span":[4,9],"surface":"guide","target":"#guide-target"}]}'
    current = '{"content":"New text, guide.","local_targets":[{"id":"guide-target","span":[10,15]}],"references":[{"id":"link-guide","kind":"LINK","span":[10,15],"surface":"guide","target":"#guide-target"}]}'
    materialization = adapt_reference_link_embed_carrier_record(
        _revision_record(base, previous, current))
    assert all(item.new_identities for item in materialization.revisions[0].mappings
               if item.mapping_kind == REVISION_MAP_REFERENCE)


def test_serializer_rejects_record_drift_and_noncanonical_bytes(records):
    first, second = records[:2]
    payload = serialize_reference_link_embed_carrier_materialization(
        adapt_reference_link_embed_carrier_record(first))
    with pytest.raises(ReferenceLinkEmbedCarrierAdapterError, match="record 身份"):
        deserialize_reference_link_embed_carrier_materialization(payload, second)
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(ReferenceLinkEmbedCarrierAdapterError, match="字段不精确"):
        deserialize_reference_link_embed_carrier_materialization(
            canonical_json_bytes(value) + b"\n", first)
    with pytest.raises(ReferenceLinkEmbedCarrierAdapterError, match="newline"):
        deserialize_reference_link_embed_carrier_materialization(payload + b"\n", first)


def test_payload_and_manifest_writers_are_idempotent_never_overwrite(
        records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_reference_link_embed_carrier_records(records, sample_target) == sample_target
    assert write_reference_link_embed_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(ReferenceLinkEmbedCarrierContractError, match="内容不同"):
        write_reference_link_embed_carrier_records(records, sample_target)
    manifest_target = tmp_path / "manifest.json"
    assert write_reference_link_embed_carrier_manifest(manifest, manifest_target) == manifest_target
    assert read_reference_link_embed_carrier_manifest(manifest_target) == manifest
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(ReferenceLinkEmbedCarrierContractError, match="内容不同"):
        write_reference_link_embed_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_parser_budget_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.parser_package == "python-stdlib"
    assert manifest.parser_version == "json-reference-slot-v1"
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert tuple(item.envelope_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert all(item.anchor_count >= item.envelope_count * 2
               for item in manifest.materializations)
    assert all(item.reference_count >= 0 for item in manifest.materializations)
    assert all(item.structure_node_count > 0
               for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST"}
    verify_reference_link_embed_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_rebuilds_identically(manifest):
    stored = read_reference_link_embed_carrier_manifest(
        _ROOT / Path(*REFERENCE_LINK_EMBED_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_reference_link_embed_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_reference_link_embed_carrier_files(stored, repository_root=_ROOT)


def test_catalog_rejects_parser_drift(monkeypatch):
    monkeypatch.setattr(catalog, "_parser_version", lambda: (_ for _ in ()).throw(
        ReferenceLinkEmbedCarrierCatalogError("reference parser version 漂移")))
    with pytest.raises(ReferenceLinkEmbedCarrierCatalogError, match="parser version"):
        build_reference_link_embed_carrier_manifest(_ROOT)


def test_manifest_rejects_execution_claim_and_extra_fields(manifest):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(ReferenceLinkEmbedCarrierContractError, match="全零"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(ReferenceLinkEmbedCarrierContractError, match="字段不精确"):
        ReferenceLinkEmbedCarrierPayloadManifest.from_dict(value)
