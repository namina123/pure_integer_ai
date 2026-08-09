"""Metadata-only tests for the successor V3 PUD-news private owner."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02CompileFreezeError,
    W02FileFreeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner import (
    W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02MorphologySuccessorV3PrivateFileIdentity,
    W02MorphologySuccessorV3PrivateOwnerError,
    read_w02_morphology_successor_v3_private_owner_receipt,
    validate_w02_morphology_successor_v3_private_owner_receipt,
)


def _receipt() -> tuple[Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    value = read_canonical_object(
        repository / Path(*W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH.split("/")))
    return repository, value


def test_v3_owner_receipt_closes_inventory_without_private_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    value, files = read_w02_morphology_successor_v3_private_owner_receipt(
        repository)
    assert value["status"] == "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
    assert value["main_session_private_payload_reads"] == 0
    assert value["main_session_conllu_payload_reads"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert len(files) == 7
    assert all(isinstance(row, W02MorphologySuccessorV3PrivateFileIdentity)
               for row in files)
    assert all(row.license_ids == ("CC-BY-SA-3.0",) for row in files)
    assert sum(row.record_count for row in files
               if row.record_kind == "observation") == (
                   W02_MORPH_V3_PRIVATE_PAIR_COUNT)
    serialized = str(value)
    assert "cf491c57bc5e9868" in serialized
    assert ":\\" not in serialized
    assert "/home/" not in serialized


def test_v3_owner_uses_append_only_license_identity() -> None:
    _, value = _receipt()
    raw = {
        key: item for key, item in value["files"][0].items()
        if key != "relative_path"
    }
    identity = W02MorphologySuccessorV3PrivateFileIdentity.from_dict(raw)
    assert identity.license_ids == ("CC-BY-SA-3.0",)
    with pytest.raises(W02CompileFreezeError, match="license"):
        W02FileFreeze.from_dict(raw)


@pytest.mark.parametrize("field", (
    "exact_case_overlap_count",
    "exact_cluster_overlap_count",
    "exact_content_overlap_count",
    "normalized_content_overlap_count",
))
def test_v3_owner_receipt_rejects_any_train_contamination(field: str) -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["contamination_audit"]["train"][field] = 1
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="contamination",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)


def test_v3_owner_receipt_rejects_file_license_or_domain_drift() -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["files"][0]["license_ids"] = ["CC-BY-SA-4.0"]
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="license",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)

    changed = deepcopy(value)
    changed["pud_domain_audit"]["w_rejected"] = 499
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="PUD domain",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)


def test_v3_owner_receipt_rejects_duplicate_or_blocked_owner_reuse() -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["within_owner_duplicate_audit"]["near_duplicate_pair_count"] = 1
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="duplicate",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)

    changed = deepcopy(value)
    changed["blocked_owner_disjoint_audit"]["blocked_owner_payload_reads"] = 1
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="blocked owner",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)


def test_v3_owner_receipt_rejects_resource_or_extension_drift() -> None:
    repository, value = _receipt()
    changed = deepcopy(value)
    changed["resource_budget"]["usage"]["payload_gets"] = 300_001
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="exceeded",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)

    changed = deepcopy(value)
    changed["public_repository"]["source_extension_v3_sha256"] = "0" * 64
    with pytest.raises(
            W02MorphologySuccessorV3PrivateOwnerError,
            match="dependency",
            ):
        validate_w02_morphology_successor_v3_private_owner_receipt(
            changed, repository)
