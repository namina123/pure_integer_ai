"""Metadata-only tests for the successor V3 R3 Wikipedia owner."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3 import (
    W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH,
    W02MorphologySuccessorV3PrivateOwnerR3Error,
    read_w02_morphology_successor_v3_private_owner_r3_receipt,
    validate_w02_morphology_successor_v3_private_owner_r3_receipt,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3_contract import (
    W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS,
)


def _receipt() -> tuple[Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    path = repository / Path(
        *W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH.split("/"))
    return repository, read_canonical_object(path)


def test_r3_owner_receipt_closes_without_private_payload_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    value, files = read_w02_morphology_successor_v3_private_owner_r3_receipt(
        repository)

    assert value["status"] == "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
    assert value["owner_metadata_sha256"] == (
        W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256)
    assert value["main_session_private_payload_reads"] == 0
    assert value["main_session_source_payload_reads"] == 0
    assert value["candidate_evaluation_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["teacher_llm_provenance"] == {
        "llm_calls": 0,
        "teacher_calls": 0,
    }
    assert len(files) == 7
    assert sum(
        row.record_count for row in files
        if row.record_kind == "observation") == 500
    assert all(row.license_ids == ("CC-BY-SA-3.0",) for row in files)
    serialized = str(value)
    assert ":\\" not in serialized
    assert "/home/" not in serialized


def test_r3_owner_label_semantics_are_exactly_bound() -> None:
    _, value = _receipt()
    binding = value["label_semantic_binding_audit"]
    expected = [
        {
            "dimension_key": list(key),
            "dimension_name": name,
            "record_count": count,
        }
        for name, key, count in W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS
    ]

    assert binding["dimension_bindings"] == expected
    assert binding["expected_state_counts"] == {"TRUE": 500}
    assert binding["unknown_dimension_key_count"] == 0
    assert binding["unknown_expected_state_count"] == 0
    assert binding["observation_key_mismatch_count"] == 0
    assert binding["double_pass_equal"] == 1


def test_r3_owner_file_layout_and_license_are_exact() -> None:
    repository = Path(__file__).resolve().parents[1]
    _, files = read_w02_morphology_successor_v3_private_owner_r3_receipt(
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
    assert all(row.license_ids == ("CC-BY-SA-3.0",) for row in files)


@pytest.mark.parametrize(("field", "value", "match"), (
    ("owner_metadata_sha256", "0" * 64, "identity"),
    ("owner_metadata_size_bytes", 10_150, "identity"),
    ("source_snapshot_commitment", "0" * 64, "identity"),
))
def test_r3_owner_receipt_rejects_metadata_identity_drift(
        field: str,
        value: object,
        match: str,
        ) -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed[field] = value

    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match=match):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)


def test_r3_owner_receipt_rejects_path_and_license_drift() -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed["files"][0]["relative_path"] = "source/other.jsonl.gz"
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="relative path"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["files"][0]["license_ids"] = ["CC-BY-SA-4.0"]
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="license"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)


def test_r3_owner_receipt_rejects_namespace_and_audit_drift() -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed["namespace_policy"]["namespace_components"][1] += 1
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="namespace"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["contamination"]["exact_content_overlap_count"] = 1
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="contamination"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["resource_usage"]["logic_operations"] = 9_000_001
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="resource"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)


def test_r3_owner_receipt_rejects_consumed_r2_domain_reuse() -> None:
    repository, receipt = _receipt()
    changed = deepcopy(receipt)
    changed["domain_disjoint_audit"]["r2_formal_family_reads"] = 1
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="domain disjoint"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)

    changed = deepcopy(receipt)
    changed["source_key"] = "UD_ZH_PUD_R2_18_NEWS_BLIND_PRIVATE"
    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR3Error,
                       match="identity"):
        validate_w02_morphology_successor_v3_private_owner_r3_receipt(
            changed, repository)
