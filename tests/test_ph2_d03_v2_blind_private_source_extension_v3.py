"""Append-only PUD count-correction source extension tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v2 import (
    blind_private_source_specs_v2,
    read_blind_private_source_extension_v2_manifest,
    validate_blind_private_source_ref_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    BLOCKED_OWNER_CODE,
    BLOCKED_OWNER_METADATA_SHA256,
    BlindPrivateSourceExtensionV3Error,
    PUD_NEWS_SOURCE_KEY,
    blind_private_source_specs_v3,
    read_blind_private_source_extension_v3_manifest,
    validate_blind_private_owner_record_v3,
    validate_blind_private_source_ref_v3,
)


SHA256 = "1" * 64


def _source_ref(*, locator: str = "n01001") -> dict[str, object]:
    spec = blind_private_source_specs_v3()[0]
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
        "source_identity": f"{PUD_NEWS_SOURCE_KEY}:sentence:{locator}",
        "source_key": PUD_NEWS_SOURCE_KEY,
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


def test_v3_corrects_counts_without_mutating_published_v2() -> None:
    repository = Path(__file__).resolve().parents[1]
    parent = read_blind_private_source_extension_v2_manifest(repository)
    value = read_blind_private_source_extension_v3_manifest(repository)

    assert parent["sources"][0]["data_file"]["owner_filter"] == {
        "documented_news_sentence_count": 750,
        "excluded_sentence_id_prefixes": ["w"],
        "required_sentence_id_prefix": "n",
        "selection_policy": "PUD_NEWS_ONLY_NO_WIKIPEDIA",
    }
    assert value["sources"][0]["data_file"]["owner_filter"] == {
        "count_basis": "ISOLATED_OWNER_SENT_ID_PREFIX_SCAN_OF_FIXED_R2_18_BLOB",
        "excluded_sentence_id_prefixes": ["w"],
        "fixed_blob_news_sentence_count": 500,
        "fixed_blob_wikipedia_sentence_count": 500,
        "required_sentence_id_prefix": "n",
        "selection_policy": "PUD_NEWS_ONLY_NO_WIKIPEDIA",
    }
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V3_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0


def test_v3_binds_blocked_owner_and_forbids_reuse() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_v3_manifest(repository)
    blocked = value["previous_blocked_owner"]
    assert value["blocked_owner_reuse_authorized"] == 0
    assert blocked["status"] == "BLOCKED"
    assert blocked["blocker_code"] == BLOCKED_OWNER_CODE
    assert blocked["metadata_sha256"] == BLOCKED_OWNER_METADATA_SHA256
    assert blocked["metadata_size_bytes"] == 473
    assert blocked["formal_artifact_count"] == 0
    assert blocked["candidate_evaluation_runs"] == 0
    assert blocked["formal_private_evaluation_runs"] == 0
    assert blocked["wikipedia_accepted_count"] == 0


def test_v3_accepts_news_and_rejects_wikipedia_or_provenance_drift() -> None:
    value = _source_ref()
    parent_record = validate_blind_private_source_ref_v2(value)
    record = validate_blind_private_source_ref_v3(value)
    assert record == parent_record
    assert validate_blind_private_owner_record_v3(value) == record

    with pytest.raises(BlindPrivateSourceExtensionV3Error, match="news"):
        validate_blind_private_source_ref_v3(_source_ref(locator="w01001"))
    drifted = _source_ref()
    drifted["upstream_checksum"] = "sha1:" + "0" * 40
    with pytest.raises(BlindPrivateSourceExtensionV3Error, match="provenance"):
        validate_blind_private_source_ref_v3(drifted)


def test_v3_source_specs_are_deeply_detached() -> None:
    parent_before = deepcopy(blind_private_source_specs_v2())
    first = blind_private_source_specs_v3()
    first[0]["data_file"]["owner_filter"]["fixed_blob_news_sentence_count"] = 1
    assert blind_private_source_specs_v3()[0]["data_file"]["owner_filter"][
        "fixed_blob_news_sentence_count"] == 500
    assert blind_private_source_specs_v2() == parent_before
