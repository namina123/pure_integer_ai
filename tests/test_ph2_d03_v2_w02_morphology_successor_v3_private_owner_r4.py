"""Metadata-only tests for the successor V3 R4 Kyoto owner."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4 import (
    W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH,
    W02MorphologySuccessorV3PrivateOwnerR4Error,
    read_w02_morphology_successor_v3_private_owner_r4_receipt,
    validate_w02_morphology_successor_v3_private_owner_r4_receipt,
)


def _receipt() -> tuple[Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    path = repository / Path(
        *W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH.split("/"))
    return repository, read_canonical_object(path)


def test_r4_owner_receipt_closes_without_private_payload_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    value, files = read_w02_morphology_successor_v3_private_owner_r4_receipt(
        repository)

    assert value["status"] == "OWNER_METADATA_INGESTED_SOURCE_V5_PAYLOAD_UNREAD"
    assert value["owner_metadata_sha256"] == W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256
    assert value["main_session_private_payload_reads"] == 0
    assert value["main_session_source_payload_reads"] == 0
    assert value["candidate_evaluation_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["source_validator_binding_audit"]["status"] == "PASS"
    assert len(files) == 7
    assert sum(
        row.record_count for row in files
        if row.record_kind == "observation") == 500
    assert all(row.license_ids == ("CC-BY-SA-4.0",) for row in files)
    serialized = str(value)
    assert ":\\" not in serialized
    assert "/home/" not in serialized


def test_r4_owner_source_and_label_bindings_are_exact() -> None:
    _, value = _receipt()
    source = value["source_validator_binding_audit"]
    label = value["label_semantic_binding_audit"]

    assert source["source_key"] == "UD_LZH_KYOTO_R2_18_TEST_BLIND_PRIVATE"
    assert source["validation_pass_count"] == 500
    assert source["validation_failure_count"] == 0
    assert source["license_counts"] == {"CC-BY-SA-4.0": 500}
    assert label["expected_true_count"] == 500
    assert label["unknown_dimension_key_count"] == 0
    assert label["unknown_expected_state_count"] == 0
    assert label["double_pass_equal"] == 1


def test_r4_owner_file_layout_and_license_are_exact() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, files = read_w02_morphology_successor_v3_private_owner_r4_receipt(
        repository)
    assert tuple(row.layout_key for row in files) == (
        "PRIVATE_SOURCE",
        "PRIVATE_HELD_OUT_OBSERVATION",
        "PRIVATE_ADVERSARIAL_OBSERVATION",
        "PRIVATE_WALL_OBSERVATION",
        "PRIVATE_HELD_OUT_LABEL",
        "PRIVATE_ADVERSARIAL_LABEL",
        "PRIVATE_WALL_LABEL",
    )
    assert files[0].split == ""
    assert all(row.root_key == "PRIVATE_EVALUATOR_ROOT" for row in files)
    assert all(row.license_ids == ("CC-BY-SA-4.0",) for row in files)


def test_r4_owner_receipt_rejects_old_license_and_v5_drift() -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed["files"][0]["license_ids"] = ["CC-BY-SA-3.0"]
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR4Error,
                       match="license"):
        validate_w02_morphology_successor_v3_private_owner_r4_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["source_validator_binding_audit"]["validator_code_sha256"] = "0" * 64
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR4Error,
                       match="source binding"):
        validate_w02_morphology_successor_v3_private_owner_r4_receipt(
            changed, repository)


def test_r4_owner_receipt_rejects_consumed_domain_or_namespace_reuse() -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed["domain_disjoint_audit"]["old_r3_owner_payload_reads"] = 1
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR4Error,
                       match="domain disjoint"):
        validate_w02_morphology_successor_v3_private_owner_r4_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["namespace_policy"]["namespace_components"][1] += 1
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR4Error,
                       match="projection"):
        validate_w02_morphology_successor_v3_private_owner_r4_receipt(
            changed, repository)
