"""LC-16 HTML raw/DOM/reference adapter 与 catalog 合同测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TREE_PATH,
    REFERENCE_ACCESS_BLOCKED,
    REFERENCE_RESOLVED,
    REVISION_MAP_ANCHOR,
    REVISION_MAP_REFERENCE,
    REVISION_MAP_STRUCTURE_NODE,
    ArtifactCarrierRevision,
)
from pure_integer_ai.cognition.shared.parser_revision import parser_lineage_key
from pure_integer_ai.experiments import ph2_html_carrier_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_html_carrier_adapter import (
    HtmlCarrierAdapterError,
    adapt_html_carrier_record,
    deserialize_html_carrier_materialization,
    serialize_html_carrier_materialization,
)
from pure_integer_ai.experiments.ph2_html_carrier_catalog import (
    HTML_CARRIER_MANIFEST_PATH,
    HTML_CARRIER_SAMPLE_PATH,
    PARENT_PACK_SHA256,
    HtmlCarrierCatalogError,
    build_html_carrier_manifest,
)
from pure_integer_ai.experiments.ph2_html_carrier_contract import (
    EXECUTION_STATE,
    SAMPLE_KINDS,
    HtmlCarrierContractError,
    HtmlCarrierPayloadManifest,
    read_html_carrier_manifest,
    read_html_carrier_records,
    verify_html_carrier_files,
    write_html_carrier_manifest,
    write_html_carrier_records,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / Path(*HTML_CARRIER_SAMPLE_PATH.split("/"))


@pytest.fixture(scope="module")
def records():
    return read_html_carrier_records(_SAMPLE)


@pytest.fixture(scope="module")
def manifest():
    return build_html_carrier_manifest(_ROOT)


def _revision_record(record, previous: str, current: str):
    return replace(
        record,
        previous_text=previous,
        previous_unit_count=len(previous),
        previous_utf8_sha256=hashlib.sha256(
            previous.encode("utf-8")).hexdigest(),
        raw_text=current,
        raw_unit_count=len(current),
        raw_utf8_sha256=hashlib.sha256(current.encode("utf-8")).hexdigest(),
    )


def test_sample_freezes_seven_cc0_html_payloads(records):
    assert tuple(item.sample_kind for item in records) == SAMPLE_KINDS
    assert all(item.license_id == "CC0-1.0" for item in records)
    assert all(item.raw_unit_count == len(item.raw_text) for item in records)
    assert all(item.raw_utf8_sha256 == hashlib.sha256(
        item.raw_text.encode("utf-8")).hexdigest() for item in records)
    assert "&nbsp;&amp;" in records[1].raw_text
    assert "<future-panel" in records[3].raw_text
    assert records[5].raw_text.endswith("\n")


def test_adapter_preserves_raw_dom_paths_and_never_assigns_semantic_roles(records):
    for record in records:
        materialization = adapt_html_carrier_record(record)
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
                       for item in group) == 1
            tree = tuple(item for item in group
                         if item.anchor_kind == ANCHOR_TREE_PATH)
            assert tree
            assert len({anchor.coordinates for anchor in tree}) == len(tree)
        assert all(item.role is None for item in materialization.structure_nodes)
        assert all(item.qualifiers for item in materialization.structure_nodes)
        payload = serialize_html_carrier_materialization(materialization)
        assert deserialize_html_carrier_materialization(
            payload, record) == materialization


def test_unknown_custom_structure_and_attributes_remain_in_receipts(records):
    unknown = next(item for item in records if item.sample_kind == "UNKNOWN")
    materialization = adapt_html_carrier_record(unknown)
    receipts = [bytes(node.qualifiers)
                for node in materialization.structure_nodes]
    assert any(b"future-panel" in payload for payload in receipts)
    assert any(b"data-mode" in payload for payload in receipts)
    assert any(b"slot-x" in payload for payload in receipts)


def test_reference_states_are_explicit_without_external_access(records):
    retention = next(item for item in records if item.sample_kind == "RETENTION")
    materialization = adapt_html_carrier_record(retention)
    assert len(materialization.reference_anchors) == 2
    assert {item.target_state for item in materialization.references} == {
        REFERENCE_RESOLVED, REFERENCE_ACCESS_BLOCKED}
    resolved = next(item for item in materialization.references
                    if item.target_state == REFERENCE_RESOLVED)
    blocked = next(item for item in materialization.references
                   if item.target_state == REFERENCE_ACCESS_BLOCKED)
    assert resolved.target_source == materialization.sources[0]
    assert resolved.target_anchor is not None
    assert blocked.target_source is None and blocked.target_anchor is None
    assert bytes(blocked.target_fingerprint).startswith(b"https://")


def test_revision_maps_anchor_node_reference_and_preserves_parser_lineage(records):
    record = next(item for item in records if item.sample_kind == "REVISION")
    materialization = adapt_html_carrier_record(record)
    old_source, new_source = materialization.sources
    assert parser_lineage_key(old_source) == parser_lineage_key(new_source)
    assert old_source.versions.parser.value == 1
    assert new_source.versions.parser.value == 2
    revision = materialization.revisions[0]
    assert ArtifactCarrierRevision.from_stable_key(
        revision.stable_key()) == revision
    assert {item.mapping_kind for item in revision.mappings} == {
        REVISION_MAP_ANCHOR,
        REVISION_MAP_STRUCTURE_NODE,
        REVISION_MAP_REFERENCE,
    }


def test_reference_revision_uses_attribute_name_not_attribute_order(records):
    base = next(item for item in records if item.sample_kind == "REVISION")
    record = _revision_record(
        base,
        '<div href="/old" src="/asset"></div>',
        '<div src="/asset-2" href="/new"></div>',
    )
    materialization = adapt_html_carrier_record(record)
    revision = materialization.revisions[0]
    references = {item.identity: item for item in materialization.references}
    mappings = tuple(item for item in revision.mappings
                     if item.mapping_kind == REVISION_MAP_REFERENCE)
    mapped_targets = {
        bytes(references[item.old_identity].target_fingerprint):
        bytes(references[item.new_identities[0]].target_fingerprint)
        for item in mappings if item.new_identities
    }
    assert mapped_targets == {
        b"/old": b"/new",
        b"/asset": b"/asset-2",
    }


def test_deleted_reference_maps_old_to_zero(records):
    base = next(item for item in records if item.sample_kind == "REVISION")
    record = _revision_record(
        base,
        '<div href="/old"><span>保留</span></div>',
        '<div><span>保留</span></div>',
    )
    revision = adapt_html_carrier_record(record).revisions[0]
    reference_mappings = tuple(item for item in revision.mappings
                               if item.mapping_kind == REVISION_MAP_REFERENCE)
    assert len(reference_mappings) == 1
    assert reference_mappings[0].new_identities == ()
    assert any(item.mapping_kind == REVISION_MAP_ANCHOR
               and not item.new_identities for item in revision.mappings)


def test_serializer_rejects_record_drift_and_noncanonical_bytes(records):
    first, second = records[:2]
    payload = serialize_html_carrier_materialization(
        adapt_html_carrier_record(first))
    with pytest.raises(HtmlCarrierAdapterError, match="record 身份"):
        deserialize_html_carrier_materialization(payload, second)
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["unexpected"] = 1
    with pytest.raises(HtmlCarrierAdapterError, match="字段不精确"):
        deserialize_html_carrier_materialization(
            canonical_json_bytes(value) + b"\n", first)
    with pytest.raises(HtmlCarrierAdapterError, match="newline"):
        deserialize_html_carrier_materialization(payload + b"\n", first)


def test_payload_and_manifest_writers_are_idempotent_never_overwrite(
        records, manifest, tmp_path):
    sample_target = tmp_path / "sample.jsonl"
    assert write_html_carrier_records(records, sample_target) == sample_target
    assert write_html_carrier_records(records, sample_target) == sample_target
    sample_target.write_bytes(b"{}\n")
    with pytest.raises(HtmlCarrierContractError, match="内容不同"):
        write_html_carrier_records(records, sample_target)
    manifest_target = tmp_path / "manifest.json"
    assert write_html_carrier_manifest(manifest, manifest_target) == manifest_target
    assert read_html_carrier_manifest(manifest_target) == manifest
    manifest_target.write_bytes(b"{}\n")
    with pytest.raises(HtmlCarrierContractError, match="内容不同"):
        write_html_carrier_manifest(manifest, manifest_target)


def test_manifest_binds_parent_parser_budget_and_zero_execution(manifest):
    assert manifest.parent_pack_sha256 == PARENT_PACK_SHA256
    assert manifest.parser_package == "lxml"
    assert manifest.parser_version == "6.1.1"
    assert manifest.execution_state.to_value() == EXECUTION_STATE
    assert len(manifest.case_keys) == len(SAMPLE_KINDS)
    assert tuple(item.envelope_count for item in manifest.materializations) == (
        1, 1, 1, 1, 2, 1, 1)
    assert all(item.anchor_count > item.envelope_count * 2
               for item in manifest.materializations)
    assert all(item.structure_node_count > 0
               for item in manifest.materializations)
    assert {item.role for item in manifest.evidence_files} == {
        "ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST"}
    verify_html_carrier_files(manifest, repository_root=_ROOT)


def test_stored_manifest_is_current_and_rebuilds_identically(manifest):
    stored = read_html_carrier_manifest(
        _ROOT / Path(*HTML_CARRIER_MANIFEST_PATH.split("/")))
    rebuilt = build_html_carrier_manifest(_ROOT)
    assert stored == manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_html_carrier_files(stored, repository_root=_ROOT)


def test_catalog_rejects_parser_drift(monkeypatch):
    monkeypatch.setattr(catalog, "_parser_version", lambda: (_ for _ in ()).throw(
        HtmlCarrierCatalogError("HTML parser version 漂移")))
    with pytest.raises(HtmlCarrierCatalogError, match="parser version"):
        build_html_carrier_manifest(_ROOT)


def test_manifest_rejects_execution_claim_and_extra_fields(manifest):
    state = dict(EXECUTION_STATE)
    state["formal_training_runs"] = 1
    with pytest.raises(HtmlCarrierContractError, match="全零"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))
    value = manifest.to_dict()
    value["unexpected"] = 1
    with pytest.raises(HtmlCarrierContractError, match="字段不精确"):
        HtmlCarrierPayloadManifest.from_dict(value)
