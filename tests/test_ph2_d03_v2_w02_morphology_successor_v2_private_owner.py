"""Metadata-only tests for the successor V2 isolated private owner."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_private_owner import (
    W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_PATH,
    W02_MORPH_V2_PRIVATE_PAIR_COUNT,
    W02MorphologySuccessorV2PrivateOwnerError,
    read_w02_morphology_successor_v2_private_owner_receipt,
    validate_w02_morphology_successor_v2_private_owner_receipt,
)


def _receipt() -> tuple[Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    value = read_canonical_object(
        repository / Path(*W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_PATH.split("/")))
    return repository, value


def test_owner_receipt_closes_inventory_without_payload_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    value, files = read_w02_morphology_successor_v2_private_owner_receipt(
        repository)
    assert value["status"] == "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
    assert value["main_session_private_payload_reads"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert len(files) == 7
    assert sum(row.record_count for row in files if row.record_kind == "observation") == (
        W02_MORPH_V2_PRIVATE_PAIR_COUNT)
    serialized = str(value)
    assert "884d5696c8e244ab" in serialized
    assert ":\\" not in serialized
    assert "/home/" not in serialized


@pytest.mark.parametrize("field", (
    "exact_case_overlap_count",
    "exact_cluster_overlap_count",
    "exact_content_overlap_count",
    "normalized_content_overlap_count",
))
def test_owner_receipt_rejects_any_train_contamination(field: str) -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["contamination_audit"]["train"][field] = 1
    with pytest.raises(W02MorphologySuccessorV2PrivateOwnerError, match="contamination"):
        validate_w02_morphology_successor_v2_private_owner_receipt(
            changed, repository)


def test_owner_receipt_rejects_file_or_source_extension_drift() -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["files"][0]["record_count"] += 1
    with pytest.raises(W02MorphologySuccessorV2PrivateOwnerError, match="source/license"):
        validate_w02_morphology_successor_v2_private_owner_receipt(
            changed, repository)

    changed = deepcopy(value)
    changed["public_repository"]["source_extension_sha256"] = "0" * 64
    with pytest.raises(W02MorphologySuccessorV2PrivateOwnerError, match="extension"):
        validate_w02_morphology_successor_v2_private_owner_receipt(
            changed, repository)


def test_owner_receipt_rejects_duplicate_or_dimension_drift() -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["within_owner_duplicate_audit"]["near_duplicate_pair_count"] = 1
    with pytest.raises(W02MorphologySuccessorV2PrivateOwnerError, match="duplicate"):
        validate_w02_morphology_successor_v2_private_owner_receipt(
            changed, repository)

    changed = deepcopy(value)
    changed["dimension_denominator_counts"][
        "W-02-V2-BOUNDARY-WITHDRAWAL"] -= 1
    with pytest.raises(W02MorphologySuccessorV2PrivateOwnerError, match="denominator"):
        validate_w02_morphology_successor_v2_private_owner_receipt(
            changed, repository)
