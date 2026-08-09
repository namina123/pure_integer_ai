"""Append-only unused PUD-Wikipedia R3 source extension tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    PUD_NEWS_SOURCE_KEY,
    blind_private_source_specs_v3,
    read_blind_private_source_extension_v3_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v4 import (
    CONSUMED_R2_FAILURE_SEAL_SHA256,
    CONSUMED_R2_FAMILY_FREEZE_SHA256,
    BlindPrivateSourceExtensionV4Error,
    PUD_WIKIPEDIA_SOURCE_KEY,
    blind_private_source_specs_v4,
    read_blind_private_source_extension_v4_manifest,
    validate_blind_private_owner_record_v4,
    validate_blind_private_source_ref_v4,
)


SHA256 = "1" * 64


def _source_ref(*, locator: str = "w01001") -> dict[str, object]:
    spec = blind_private_source_specs_v4()[0]
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    return {
        "artifact_key": [2, 2, 1],
        "attribution": "Universal Dependencies PUD attribution retained",
        "course_version": 2,
        "dataset_key": [2, 2],
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
        "source_cluster_key": [2, 2, 1, 1],
        "source_identity": f"{PUD_WIKIPEDIA_SOURCE_KEY}:sentence:{locator}",
        "source_key": PUD_WIKIPEDIA_SOURCE_KEY,
        "source_span": {
            "document_cluster_key": [2, 2, 1, 2],
            "entity_graph_cluster_key": [2, 2, 1, 3],
            "locator_kind": "sentence",
            "locator_value": locator,
            "span_end": 2,
            "span_start": 0,
        },
        "stable_key": [2, 2, 1, 10, 1],
        "upstream_checksum": "sha1:" + str(data_file["git_blob_sha1"]),
    }


def test_v4_authorizes_only_unused_wikipedia_domain() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent = read_blind_private_source_extension_v3_manifest(repository)
    value = read_blind_private_source_extension_v4_manifest(repository)
    owner_filter = value["sources"][0]["data_file"]["owner_filter"]

    assert parent["sources"][0]["source_key"] == PUD_NEWS_SOURCE_KEY
    assert value["sources"][0]["source_key"] == PUD_WIKIPEDIA_SOURCE_KEY
    assert owner_filter == {
        "count_basis": "ISOLATED_OWNER_SENT_ID_PREFIX_SCAN_OF_FIXED_R2_18_BLOB",
        "excluded_sentence_id_prefixes": ["n"],
        "fixed_blob_news_sentence_count": 500,
        "fixed_blob_wikipedia_sentence_count": 500,
        "required_sentence_id_prefix": "w",
        "selection_policy": "PUD_WIKIPEDIA_ONLY_NO_NEWS",
    }
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V4_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0


def test_v4_binds_consumed_r2_ne_and_forbids_reuse() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_v4_manifest(repository)
    prior = value["previous_consumed_r2_family"]
    assert value["consumed_r2_owner_reuse_authorized"] == 0
    assert PUD_NEWS_SOURCE_KEY in value["consumed_private_source_keys"]
    assert prior["status"] == "NE_NO_RECEIPT"
    assert prior["formal_private_evaluation_runs"] == 1
    assert prior["reuse_authorized"] == 0
    assert prior["aggregate_report_published"] == 0
    assert prior["family_freeze_sha256"] == CONSUMED_R2_FAMILY_FREEZE_SHA256
    assert prior["failure_seal_sha256"] == CONSUMED_R2_FAILURE_SEAL_SHA256


def test_v4_accepts_wikipedia_and_rejects_news_or_provenance_drift() -> None:
    value = _source_ref()
    record = validate_blind_private_source_ref_v4(value)
    assert validate_blind_private_owner_record_v4(value) == record

    with pytest.raises(BlindPrivateSourceExtensionV4Error, match="Wikipedia"):
        validate_blind_private_source_ref_v4(_source_ref(locator="n01001"))
    drifted = _source_ref()
    drifted["source_key"] = PUD_NEWS_SOURCE_KEY
    with pytest.raises(BlindPrivateSourceExtensionV4Error, match="not authorized"):
        validate_blind_private_source_ref_v4(drifted)
    drifted = _source_ref()
    drifted["upstream_checksum"] = "sha1:" + "0" * 40
    with pytest.raises(BlindPrivateSourceExtensionV4Error, match="provenance"):
        validate_blind_private_source_ref_v4(drifted)


def test_v4_source_specs_are_deeply_detached() -> None:
    parent_before = deepcopy(blind_private_source_specs_v3())
    first = blind_private_source_specs_v4()
    first[0]["data_file"]["owner_filter"][
        "fixed_blob_wikipedia_sentence_count"] = 1
    assert blind_private_source_specs_v4()[0]["data_file"]["owner_filter"][
        "fixed_blob_wikipedia_sentence_count"] == 500
    assert blind_private_source_specs_v3() == parent_before
