"""Append-only Kyoto remainder R5 source extension tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    KYOTO_SOURCE_KEY as CONSUMED_R4_KYOTO_SOURCE_KEY,
    read_blind_private_source_extension_v5_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    BASE_LANGUAGE_PROBE_REPORT_SHA256,
    CONSUMED_R4_FAILURE_SEAL_SHA256,
    CONSUMED_R4_FAMILY_FREEZE_SHA256,
    KYOTO_REMAINDER_MINIMUM_ORDINAL,
    KYOTO_REMAINDER_SOURCE_KEY,
    BlindPrivateSourceExtensionV6Error,
    blind_private_source_specs_v6,
    read_blind_private_source_extension_v6_manifest,
    validate_blind_private_owner_record_v6,
    validate_blind_private_source_ref_v6,
)


SHA256 = "1" * 64


def _source_ref(
        *, ordinal: int = KYOTO_REMAINDER_MINIMUM_ORDINAL,
        source_key: str = KYOTO_REMAINDER_SOURCE_KEY,
        ) -> dict[str, object]:
    spec = blind_private_source_specs_v6()[0]
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    locator = f"test:{ordinal}:lzh-kyoto-{ordinal:05d}"
    return {
        "artifact_key": [5, 5, 1],
        "attribution": "Universal Dependencies Kyoto attribution retained",
        "course_version": 2,
        "dataset_key": [5, 5],
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
        "source_cluster_key": [5, 5, 1, 1],
        "source_identity": f"{source_key}:sentence:{locator}",
        "source_key": source_key,
        "source_span": {
            "document_cluster_key": [5, 5, 1, 2],
            "entity_graph_cluster_key": [5, 5, 1, 3],
            "locator_kind": "sentence",
            "locator_value": locator,
            "span_end": 2,
            "span_start": 0,
        },
        "stable_key": [5, 5, 1, 10, 1],
        "upstream_checksum": "sha1:" + str(data_file["git_blob_sha1"]),
    }


def test_v6_authorizes_only_unconsumed_kyoto_remainder() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent = read_blind_private_source_extension_v5_manifest(repository)
    value = read_blind_private_source_extension_v6_manifest(repository)
    source = value["sources"][0]
    owner_filter = source["data_file"]["owner_filter"]

    assert parent["sources"][0]["source_key"] == CONSUMED_R4_KYOTO_SOURCE_KEY
    assert source["source_key"] == KYOTO_REMAINDER_SOURCE_KEY
    assert source["language"] == "lzh"
    assert source["license_id"] == "CC-BY-SA-4.0"
    assert owner_filter["minimum_sentence_ordinal"] == 1001
    assert owner_filter["available_sentence_count"] == 4528
    assert owner_filter["previous_r4_selection_scan_upper_bound"] == 592
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V6_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0


def test_v6_binds_r4_ne_and_public_language_adapter() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_v6_manifest(repository)
    prior = value["previous_consumed_r4_family"]
    adapter = value["public_base_language_family_adapter"]

    assert value["consumed_r4_owner_reuse_authorized"] == 0
    assert CONSUMED_R4_KYOTO_SOURCE_KEY in value["consumed_private_source_keys"]
    assert prior["status"] == "NE_NO_RECEIPT"
    assert prior["formal_private_evaluation_runs"] == 1
    assert prior["owner_pair_and_label_stream_consumed"] == 1
    assert prior["source_ref_records_closed_before_pair_stream"] == 500
    assert prior["family_freeze_sha256"] == CONSUMED_R4_FAMILY_FREEZE_SHA256
    assert prior["failure_seal_sha256"] == CONSUMED_R4_FAILURE_SEAL_SHA256
    assert adapter["status"] == "PASS"
    assert adapter["base_prediction_metamorphic_equal"] == 1
    assert adapter["route_original_lzh_authorized"] == 1
    assert adapter["route_adapted_zh_authorized"] == 0
    assert adapter["probe_report_sha256"] == BASE_LANGUAGE_PROBE_REPORT_SHA256


def test_v6_accepts_ordinal_1001_and_rejects_consumed_prefix() -> None:
    value = _source_ref()
    record = validate_blind_private_source_ref_v6(value)
    assert validate_blind_private_owner_record_v6(value) == record

    with pytest.raises(BlindPrivateSourceExtensionV6Error,
                       match="test sentence locator"):
        validate_blind_private_source_ref_v6(_source_ref(ordinal=1000))

    with pytest.raises(BlindPrivateSourceExtensionV6Error,
                       match="test sentence locator"):
        validate_blind_private_source_ref_v6(_source_ref(ordinal=5529))

    with pytest.raises(BlindPrivateSourceExtensionV6Error,
                       match="not authorized"):
        validate_blind_private_source_ref_v6(_source_ref(
            source_key=CONSUMED_R4_KYOTO_SOURCE_KEY))

    drifted = _source_ref()
    drifted["source_identity"] = (
        f"{KYOTO_REMAINDER_SOURCE_KEY}:sentence:"
        "test:1002:lzh-kyoto-01002")
    with pytest.raises(BlindPrivateSourceExtensionV6Error,
                       match="ordinal identity"):
        validate_blind_private_source_ref_v6(drifted)


def test_v6_source_specs_are_deeply_detached() -> None:
    first = blind_private_source_specs_v6()
    first[0]["data_file"]["owner_filter"]["minimum_sentence_ordinal"] = 1
    assert blind_private_source_specs_v6()[0]["data_file"][
        "owner_filter"]["minimum_sentence_ordinal"] == 1001
    assert (deepcopy(blind_private_source_specs_v6())
            == blind_private_source_specs_v6())
