"""Append-only Kyoto Classical Chinese R4 source extension tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v4 import (
    PUD_WIKIPEDIA_SOURCE_KEY,
    blind_private_source_specs_v4,
    read_blind_private_source_extension_v4_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    CONSUMED_R3_FAILURE_SEAL_SHA256,
    CONSUMED_R3_FAMILY_FREEZE_SHA256,
    BlindPrivateSourceExtensionV5Error,
    KYOTO_SOURCE_KEY,
    blind_private_source_specs_v5,
    read_blind_private_source_extension_v5_manifest,
    validate_blind_private_owner_record_v5,
    validate_blind_private_source_ref_v5,
)


SHA256 = "1" * 64


def _source_ref(*, locator: str = "test:1:lzh-kyoto-0001") -> dict[str, object]:
    spec = blind_private_source_specs_v5()[0]
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    return {
        "artifact_key": [4, 4, 1],
        "attribution": "Universal Dependencies Kyoto attribution retained",
        "course_version": 2,
        "dataset_key": [4, 4],
        "format_version": 2,
        "license_id": spec["license_id"],
        "local_sha256": SHA256,
        "official_url": spec["repository_url"],
        "parser_version": 1,
        "record_kind": "source_ref",
        "record_ordinal": 1,
        "redistribution_policy": "PUBLIC",
        "revision_id": spec["commit_sha1"],
        "schema_version": 2,
        "snapshot_id": spec["snapshot_id"],
        "source_cluster_key": [4, 4, 1, 1],
        "source_identity": f"{KYOTO_SOURCE_KEY}:sentence:{locator}",
        "source_key": KYOTO_SOURCE_KEY,
        "source_span": {
            "document_cluster_key": [4, 4, 1, 2],
            "entity_graph_cluster_key": [4, 4, 1, 3],
            "locator_kind": "sentence",
            "locator_value": locator,
            "span_end": 2,
            "span_start": 0,
        },
        "stable_key": [4, 4, 1, 10, 1],
        "upstream_checksum": "sha1:" + str(data_file["git_blob_sha1"]),
    }


def test_v5_authorizes_only_unused_nonparallel_kyoto_test() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent = read_blind_private_source_extension_v4_manifest(repository)
    value = read_blind_private_source_extension_v5_manifest(repository)
    source = value["sources"][0]
    owner_filter = source["data_file"]["owner_filter"]

    assert parent["sources"][0]["source_key"] == PUD_WIKIPEDIA_SOURCE_KEY
    assert source["source_key"] == KYOTO_SOURCE_KEY
    assert source["language"] == "lzh"
    assert source["license_id"] == "CC-BY-SA-4.0"
    assert source["parallel"] == 0
    assert owner_filter == {
        "available_sentence_count": 5528,
        "contamination_and_duplicate_audit_required": 1,
        "max_private_source_count": 500,
        "selection_policy": (
            "DETERMINISTIC_TEST_ORDER_AFTER_PUBLIC_AND_PRIVATE_"
            "CONTAMINATION_FILTERS"),
    }
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V5_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0
    assert value["main_session_conllu_payload_reads"] == 0


def test_v5_binds_consumed_r3_ne_and_forbids_reuse() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_v5_manifest(repository)
    prior = value["previous_consumed_r3_family"]

    assert value["consumed_r3_owner_reuse_authorized"] == 0
    assert PUD_WIKIPEDIA_SOURCE_KEY in value["consumed_private_source_keys"]
    assert prior["status"] == "NE_NO_RECEIPT"
    assert prior["formal_private_evaluation_runs"] == 1
    assert prior["owner_pair_and_label_stream_consumed"] == 1
    assert prior["reuse_authorized"] == 0
    assert prior["aggregate_report_published"] == 0
    assert prior["family_freeze_sha256"] == CONSUMED_R3_FAMILY_FREEZE_SHA256
    assert prior["failure_seal_sha256"] == CONSUMED_R3_FAILURE_SEAL_SHA256


def test_v5_accepts_kyoto_and_rejects_consumed_or_provenance_drift() -> None:
    value = _source_ref()
    record = validate_blind_private_source_ref_v5(value)
    assert validate_blind_private_owner_record_v5(value) == record

    drifted = _source_ref(locator="dev:1:lzh-kyoto-0001")
    drifted["source_identity"] = (
        f"{KYOTO_SOURCE_KEY}:sentence:dev:1:lzh-kyoto-0001")
    with pytest.raises(BlindPrivateSourceExtensionV5Error, match="provenance"):
        validate_blind_private_source_ref_v5(drifted)

    drifted = _source_ref()
    drifted["source_key"] = PUD_WIKIPEDIA_SOURCE_KEY
    with pytest.raises(BlindPrivateSourceExtensionV5Error,
                       match="not authorized"):
        validate_blind_private_source_ref_v5(drifted)

    drifted = _source_ref()
    drifted["upstream_checksum"] = "sha1:" + "0" * 40
    with pytest.raises(BlindPrivateSourceExtensionV5Error,
                       match="provenance"):
        validate_blind_private_source_ref_v5(drifted)


def test_v5_source_specs_are_deeply_detached() -> None:
    parent_before = deepcopy(blind_private_source_specs_v4())
    first = blind_private_source_specs_v5()
    first[0]["data_file"]["owner_filter"]["available_sentence_count"] = 1
    assert blind_private_source_specs_v5()[0]["data_file"][
        "owner_filter"]["available_sentence_count"] == 5528
    assert blind_private_source_specs_v4() == parent_before
