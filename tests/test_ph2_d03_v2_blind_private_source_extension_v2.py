"""Post-V3 blind-private source extension tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import DatasetContractError
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    BlindPrivateSourceExtensionError,
    validate_blind_private_source_ref,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v2 import (
    BlindPrivateSourceExtensionV2Error,
    PUD_NEWS_SOURCE_KEY,
    blind_private_source_specs_v2,
    read_blind_private_source_extension_v2_manifest,
    validate_blind_private_owner_record_v2,
    validate_blind_private_source_ref_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record


SHA256 = "1" * 64


def _source_ref(*, locator: str = "n01001") -> dict[str, object]:
    spec = blind_private_source_specs_v2()[0]
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


def test_v2_manifest_is_owner_only_and_binds_v3_shadow() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_v2_manifest(repository)
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V2_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0
    assert value["main_session_conllu_payload_reads"] == 0
    assert value["consumed_private_source_keys"] == [
        "UD_ZH_CFL_R2_18_BLIND_PRIVATE",
        "UD_ZH_HK_R2_18_BLIND_PRIVATE",
    ]
    assert value["sources"][0]["data_file"]["owner_filter"] == {
        "documented_news_sentence_count": 750,
        "excluded_sentence_id_prefixes": ["w"],
        "required_sentence_id_prefix": "n",
        "selection_policy": "PUD_NEWS_ONLY_NO_WIKIPEDIA",
    }


def test_v2_extension_accepts_pud_news_but_old_contracts_reject_it() -> None:
    value = _source_ref()
    with pytest.raises(DatasetContractError, match="allowlist"):
        validate_v2_record(value)
    with pytest.raises(BlindPrivateSourceExtensionError, match="not in"):
        validate_blind_private_source_ref(value)
    record = validate_blind_private_source_ref_v2(value)
    assert record.source_key == PUD_NEWS_SOURCE_KEY
    assert validate_blind_private_owner_record_v2(value) == record


def test_v2_extension_rejects_pud_wikipedia_and_checksum_drift() -> None:
    with pytest.raises(BlindPrivateSourceExtensionV2Error, match="news"):
        validate_blind_private_source_ref_v2(_source_ref(locator="w01001"))
    value = _source_ref()
    value["upstream_checksum"] = "sha1:" + "0" * 40
    with pytest.raises(BlindPrivateSourceExtensionV2Error, match="provenance"):
        validate_blind_private_source_ref_v2(value)


def test_v2_owner_rejects_consumed_private_source_refs() -> None:
    value = _source_ref()
    value["source_key"] = "UD_ZH_CFL_R2_18_BLIND_PRIVATE"
    with pytest.raises(
            BlindPrivateSourceExtensionV2Error,
            match="not authorized",
            ):
        validate_blind_private_owner_record_v2(value)
