"""LC-16 OCR/ASR 转写载体合同、对齐和 reload identity 测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import ph2_transcribed_ocr_asr_carrier_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_adapter import (
    TranscribedOcrAsrCarrierAdapterError,
    adapt_transcribed_ocr_asr_carrier_record,
    deserialize_transcribed_ocr_asr_carrier_materialization,
    serialize_transcribed_ocr_asr_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_catalog import (
    PARENT_PACK_SHA256,
    TRANSCRIBED_OCR_ASR_CARRIER_MANIFEST_PATH,
    TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH,
    TranscribedOcrAsrCarrierCatalogError,
    build_transcribed_ocr_asr_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_transcribed_ocr_asr_carrier_contract import (
    EXECUTION_STATE,
    SAMPLE_KINDS,
    TranscriptSegment,
    TranscribedOcrAsrCarrierContractError,
    TranscribedOcrAsrCarrierPayloadManifest,
    TranscribedOcrAsrCarrierRecord,
    read_transcribed_ocr_asr_carrier_manifest,
    read_transcribed_ocr_asr_carrier_records,
    verify_transcribed_ocr_asr_carrier_files,
    write_transcribed_ocr_asr_carrier_manifest,
    write_transcribed_ocr_asr_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*TRANSCRIBED_OCR_ASR_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_transcribed_ocr_asr_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_transcribed_ocr_asr_carrier_manifest(_ROOT)


def test_sample_freezes_seven_cc0_transcript_payloads(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert records[0].segments[0].source_mode == "ASR"
    assert records[1].segments[0].temporal_state == "UNAVAILABLE"
    assert records[1].segments[0].time_start_ms == records[1].segments[0].time_end_ms == 0
    assert len(next(item for item in records if item.sample_kind == "AMBIGUOUS").segments[0].speaker_candidates) == 2
    assert next(item for item in records if item.sample_kind == "UNKNOWN").segments[0].source_mode == "UNKNOWN"
    revision = next(item for item in records if item.sample_kind == "REVISION")
    assert revision.previous_segments[0].segment_id != revision.segments[0].segment_id


def test_adapter_preserves_text_time_and_candidates(records):
    for record in records:
        materialization = adapt_transcribed_ocr_asr_carrier_record(record)
        expected = ((record.previous_text, record.raw_text)
                    if record.sample_kind == "REVISION"
                    else (record.raw_text,))
        assert tuple("".join(chr(value) for value in envelope.raw_units)
                     for envelope in materialization.envelopes) == expected
        for envelope in materialization.envelopes:
            group = tuple(item for item in materialization.anchors
                          if item.envelope_identity == envelope.identity)
            assert sum(item.anchor_kind == ANCHOR_TEXT_RANGE for item in group) == 1
            assert sum(item.anchor_kind == ANCHOR_DOCUMENT_REGION for item in group) == 1
            alignment = tuple(item for item in group
                              if item.anchor_kind == ANCHOR_TRANSCRIPT_ALIGNMENT)
            assert alignment
            assert all(len(item.coordinates) == 2 for item in alignment)
        assert tuple(item.ordinal for item in materialization.structure_nodes) == tuple(
            range(len(materialization.structure_nodes)))
        payload = serialize_transcribed_ocr_asr_carrier_materialization(materialization)
        assert deserialize_transcribed_ocr_asr_carrier_materialization(payload, record) == materialization


def test_receipts_keep_ambiguity_and_unknown_as_observations(records):
    ambiguity = next(item for item in records if item.sample_kind == "AMBIGUOUS")
    receipts = [parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
                for node in adapt_transcribed_ocr_asr_carrier_record(ambiguity).structure_nodes]
    segment = next(item for item in receipts if item["type"] == "SEGMENT")
    assert len(segment["details"]["speaker_candidates"]) == 2
    assert len(segment["details"]["confidence_candidates"]) == 2
    unknown = next(item for item in records if item.sample_kind == "UNKNOWN")
    unknown_receipts = [parse_canonical_json_bytes(bytes(node.qualifiers), require_object=True)
                        for node in adapt_transcribed_ocr_asr_carrier_record(unknown).structure_nodes]
    assert any(item["type"] == "UNKNOWN_SOURCE_MODE" for item in unknown_receipts)


def test_revision_maps_segment_ids_and_deleted_alignment(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_transcribed_ocr_asr_carrier_record(record)
    old_source, new_source = materialization.sources
    assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
    assert old_source.versions.parser.value == 1
    assert new_source.versions.parser.value == 2
    revision = materialization.revisions[0]
    assert ArtifactCarrierRevision.from_stable_key(revision.stable_key()) == revision
    assert {item.mapping_kind for item in revision.mappings} == {1, 2}
    assert any(not item.new_identities for item in revision.mappings)


def test_contract_rejects_bad_time_and_float_payload(records):
    segment = records[0].segments[0]
    with pytest.raises(TranscribedOcrAsrCarrierContractError, match="time"):
        replace(segment, time_end_ms=segment.time_start_ms)
    value = parse_canonical_json_bytes(records[0].canonical_line()[:-1], require_object=True)
    value["segments"][0]["time_end_ms"] = 1.5
    with pytest.raises(TranscribedOcrAsrCarrierContractError):
        TranscribedOcrAsrCarrierRecord.from_dict(value)


def test_serializer_rejects_record_drift_and_noncanonical_bytes(records):
    first, second = records[:2]
    payload = serialize_transcribed_ocr_asr_carrier_materialization(
        adapt_transcribed_ocr_asr_carrier_record(first))
    with pytest.raises(TranscribedOcrAsrCarrierAdapterError, match="身份漂移"):
        deserialize_transcribed_ocr_asr_carrier_materialization(payload, second)
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(TranscribedOcrAsrCarrierAdapterError, match="字段不精确"):
        deserialize_transcribed_ocr_asr_carrier_materialization(
            canonical_json_bytes(value) + b"\n", first)


def test_payload_and_manifest_writers_are_idempotent(records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_transcribed_ocr_asr_carrier_records(records, sample_target) == sample_target
    assert write_transcribed_ocr_asr_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(TranscribedOcrAsrCarrierContractError, match="内容不同"):
        write_transcribed_ocr_asr_carrier_records(records, sample_target)
    manifest_target = tmp_path / "manifest.json"
    assert write_transcribed_ocr_asr_carrier_manifest(manifest, manifest_target) == manifest_target
    assert read_transcribed_ocr_asr_carrier_manifest(manifest_target) == manifest
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(TranscribedOcrAsrCarrierContractError, match="内容不同"):
        write_transcribed_ocr_asr_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_parser_budget_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.parser_package == "python-stdlib"
    assert manifest.parser_version == "transcript-alignment-v1"
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert all(item.alignment_count > 0 for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST"}
    verify_transcribed_ocr_asr_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_rebuilds_identically(manifest):
    stored = read_transcribed_ocr_asr_carrier_manifest(
        _ROOT / Path(*TRANSCRIBED_OCR_ASR_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_transcribed_ocr_asr_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()


def test_catalog_rejects_parent_sha_drift(manifest, monkeypatch):
    monkeypatch.setattr(catalog, "PARENT_PACK_SHA256", "0" * 64)
    with pytest.raises(TranscribedOcrAsrCarrierCatalogError, match="parent pack"):
        build_transcribed_ocr_asr_carrier_manifest(_ROOT)


def test_manifest_rejects_execution_claim_and_extra_fields(manifest):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(TranscribedOcrAsrCarrierContractError, match="全零"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(TranscribedOcrAsrCarrierContractError, match="字段不精确"):
        TranscribedOcrAsrCarrierPayloadManifest.from_dict(value)
