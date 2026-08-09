"""Append-only blind-private source authorization and parent-freeze tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import DatasetContractError
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    BlindPrivateSourceExtensionError,
    blind_private_source_specs,
    read_blind_private_source_extension_manifest,
    validate_blind_private_owner_record,
    validate_blind_private_source_ref,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record


SHA256 = "1" * 64


def _source_ref(source_key: str) -> dict[str, object]:
    """Build one canonical SourceRef without reading either upstream payload."""
    spec = next(
        row for row in blind_private_source_specs()
        if row["source_key"] == source_key)
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    return {
        "artifact_key": [2, 2, 1],
        "attribution": "Universal Dependencies upstream attribution retained",
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
        "source_identity": f"{source_key}:sentence:fixture-1",
        "source_key": source_key,
        "source_span": {
            "document_cluster_key": [2, 2, 1, 2],
            "entity_graph_cluster_key": [2, 2, 1, 3],
            "locator_kind": "sentence",
            "locator_value": "fixture-1",
            "span_end": 2,
            "span_start": 0,
        },
        "stable_key": [2, 2, 1, 10, 1],
        "upstream_checksum": "sha1:" + str(data_file["git_blob_sha1"]),
    }


def test_manifest_is_payload_free_owner_only_and_canonical() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = read_blind_private_source_extension_manifest(repository)
    assert value["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_APPROVED"
    assert value["private_owner_authorized"] == 1
    assert value["training_authorized"] == 0
    assert value["development_authorized"] == 0
    assert value["shadow_authorized"] == 0
    assert value["main_session_conllu_payload_reads"] == 0
    assert value["old_private_payload_reads"] == 0
    assert len(value["sources"]) == 2
    assert all(row["data_file"]["main_session_content_reads"] == 0
               for row in value["sources"])
    schema_path = repository / "src/pure_integer_ai/experiments/ph2_d03_v2_schema.py"
    assert schema_path.stat().st_size == 21856


@pytest.mark.parametrize("source_key", (
    "UD_ZH_CFL_R2_18_BLIND_PRIVATE",
    "UD_ZH_HK_R2_18_BLIND_PRIVATE",
))
def test_extension_accepts_new_source_without_mutating_parent_schema(
        source_key: str,
        ) -> None:
    value = _source_ref(source_key)
    with pytest.raises(DatasetContractError, match="allowlist"):
        validate_v2_record(value)
    record = validate_blind_private_source_ref(value)
    assert record.source_key == source_key
    assert validate_blind_private_owner_record(value) == record


def test_extension_rejects_upstream_blob_or_usage_drift() -> None:
    value = _source_ref("UD_ZH_CFL_R2_18_BLIND_PRIVATE")
    value["upstream_checksum"] = "sha1:" + "0" * 40
    with pytest.raises(BlindPrivateSourceExtensionError, match="provenance"):
        validate_blind_private_source_ref(value)

    value = _source_ref("UD_ZH_HK_R2_18_BLIND_PRIVATE")
    value["source_span"] = {**value["source_span"], "locator_kind": "record"}
    with pytest.raises(BlindPrivateSourceExtensionError, match="sentence"):
        validate_blind_private_source_ref(value)
