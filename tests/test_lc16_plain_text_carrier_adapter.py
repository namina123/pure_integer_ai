"""LC-16 PLAIN_TEXT payload、adapter、serializer 与 manifest 合同测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    RAW_UNIT_UNICODE_SCALAR,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import ph2_plain_text_carrier_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_adapter import (
    PlainTextCarrierAdapterError,
    adapt_plain_text_carrier_record,
    deserialize_plain_text_materialization,
    serialize_plain_text_materialization,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_catalog import (
    PARENT_PACK_SHA256,
    PLAIN_TEXT_CARRIER_MANIFEST_PATH,
    PLAIN_TEXT_CARRIER_SAMPLE_PATH,
    PlainTextCarrierCatalogError,
    build_plain_text_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_plain_text_carrier_contract import (
    ADAPTER_OBLIGATIONS,
    EMPTY_SHA256,
    EXECUTION_STATE,
    SAMPLE_KINDS,
    PlainTextCarrierContractError,
    PlainTextCarrierPayloadManifest,
    read_plain_text_carrier_manifest,
    read_plain_text_carrier_records,
    verify_plain_text_carrier_files,
    write_plain_text_carrier_manifest,
    write_plain_text_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*PLAIN_TEXT_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_plain_text_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_plain_text_carrier_manifest(_ROOT)


def test_sample_freezes_seven_cc0_payloads_without_expected_labels(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert tuple(item.adapter_obligation for item in records) == tuple(
        ADAPTER_OBLIGATIONS[item] for item in SAMPLE_KINDS)
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert all(item.raw_utf8_sha256 == hashlib.sha256(
        item.raw_text.encode("utf-8")).hexdigest() for item in records)
    assert records[1].raw_text == "甲  乙\n\n丙\t丁"
    assert records[3].raw_text.startswith("\U00020000")
    assert "e\u0301" in records[5].raw_text and "é" in records[5].raw_text
    assert records[6].raw_text.endswith(" ")
    assert all(item.previous_utf8_sha256 == EMPTY_SHA256
               for item in records if item.sample_kind != "REVISION")


def test_adapter_preserves_every_scalar_and_full_range_without_preselection(records):
    for record in records:
        materialization = adapt_plain_text_carrier_record(record)
        expected_texts = ((record.previous_text, record.raw_text)
                          if record.sample_kind == "REVISION"
                          else (record.raw_text,))
        assert tuple(
            "".join(chr(item) for item in envelope.raw_units)
            for envelope in materialization.envelopes) == expected_texts
        assert all(envelope.raw_unit_kind == RAW_UNIT_UNICODE_SCALAR
                   for envelope in materialization.envelopes)
        assert tuple(anchor.coordinates for anchor in materialization.anchors) == (
            tuple((0, len(text)) for text in expected_texts))
        payload = serialize_plain_text_materialization(materialization)
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
        assert deserialize_plain_text_materialization(
            payload, record) == materialization
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert canonical_json_bytes(value) + b"\n" == payload


def test_revision_binds_one_old_anchor_to_one_new_anchor_in_same_lineage(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_plain_text_carrier_record(record)
    assert len(materialization.sources) == 2
    old_source, new_source = materialization.sources
    assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
    assert old_source.versions.parser.value == 1
    assert new_source.versions.parser.value == 2
    assert len(materialization.revisions) == 1
    revision = materialization.revisions[0]
    assert ArtifactCarrierRevision.from_stable_key(
        revision.stable_key()) == revision
    assert revision.hypothesis.observation == new_source
    assert revision.mappings[0].old_identity == materialization.anchors[0].identity
    assert revision.mappings[0].new_identities == (
        materialization.anchors[1].identity,)


def test_deserializer_rejects_record_drift_fields_and_noncanonical_bytes(records):
    first, second = records[:2]
    payload = serialize_plain_text_materialization(
        adapt_plain_text_carrier_record(first))
    with pytest.raises(PlainTextCarrierAdapterError, match="record 身份"):
        deserialize_plain_text_materialization(payload, second)

    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(PlainTextCarrierAdapterError, match="字段不精确"):
        deserialize_plain_text_materialization(
            canonical_json_bytes(value) + b"\n", first)
    with pytest.raises(PlainTextCarrierAdapterError, match="newline"):
        deserialize_plain_text_materialization(payload + b"\n", first)
    noncanonical = payload.replace(b'"anchors":', b'"anchors" :', 1)
    with pytest.raises(PlainTextCarrierAdapterError, match="损坏"):
        deserialize_plain_text_materialization(noncanonical, first)


def test_payload_and_manifest_writers_are_idempotent_but_never_overwrite(
        records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_plain_text_carrier_records(records, sample_target) == sample_target
    assert read_plain_text_carrier_records(sample_target) == records
    assert write_plain_text_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(PlainTextCarrierContractError, match="内容不同"):
        write_plain_text_carrier_records(records, sample_target)

    manifest_target = tmp_path / "manifest.json"
    assert write_plain_text_carrier_manifest(
        manifest, manifest_target) == manifest_target
    assert read_plain_text_carrier_manifest(manifest_target) == manifest
    assert write_plain_text_carrier_manifest(
        manifest, manifest_target) == manifest_target
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(PlainTextCarrierContractError, match="内容不同"):
        write_plain_text_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_budget_materializations_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert len(manifest.materializations) == len(SAMPLE_KINDS)
    assert tuple(item.envelope_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert tuple(item.anchor_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert tuple(item.revision_count for item in manifest.materializations) == (
        0, 0, 0, 0, 1, 0, 0)
    assert all(item.raw_unit_count <= manifest.budget.max_raw_units
               for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "TEST"}
    verify_plain_text_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_canonical_and_rebuilds_identically(manifest):
    stored = read_plain_text_carrier_manifest(
        _ROOT / Path(*PLAIN_TEXT_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_plain_text_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_plain_text_carrier_files(stored, repository_root=_ROOT)


def test_manifest_rejects_execution_claim_and_catalog_rejects_parent_sha(
        manifest, monkeypatch):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(PlainTextCarrierContractError, match="全零"):
        replace(
            manifest,
            execution_state=CanonicalJsonObject.from_value(state),
        )

    original = catalog._identity

    def drift(path):
        byte_count, sha256 = original(path)
        if path.name == "lc16_typed_carrier_pack_v1.json":
            sha256 = "0" * 64
        return byte_count, sha256

    monkeypatch.setattr(catalog, "_identity", drift)
    with pytest.raises(PlainTextCarrierCatalogError, match="parent pack"):
        build_plain_text_carrier_manifest(_ROOT)


def test_manifest_parser_rejects_extra_fields(manifest):
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(PlainTextCarrierContractError, match="字段不精确"):
        PlainTextCarrierPayloadManifest.from_dict(value)
