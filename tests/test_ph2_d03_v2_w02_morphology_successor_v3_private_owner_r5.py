"""Payload-free public tests for the successor V3 R5 owner receipt."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5 import (
    _metadata_projection,
    read_w02_morphology_successor_v3_private_owner_r5_receipt,
    validate_w02_morphology_successor_v3_private_owner_r5_receipt,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_audit import (
    build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_contract import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH,
    W02MorphologySuccessorV3PrivateOwnerR5Error,
)


def _receipt() -> tuple[Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    return repository, read_canonical_object(
        repository / W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH)


def test_r5_owner_receipt_is_exact_safe_metadata_projection() -> None:
    repository, value = _receipt()
    receipt, files = read_w02_morphology_successor_v3_private_owner_r5_receipt(
        repository)

    assert receipt == value
    assert build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata(
        _metadata_projection(receipt)) == receipt
    assert receipt["status"] == "OWNER_METADATA_INGESTED_SOURCE_V6_PAYLOAD_UNREAD"
    assert receipt["owner_metadata_sha256"] == W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256
    assert tuple(row.layout_key for row in files) == W02_MORPH_V3_PRIVATE_LAYOUTS
    assert sum(row.record_count for row in files) == 1_500
    assert receipt["formal_private_evaluation_runs"] == 0
    assert receipt["main_session_private_payload_reads"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("owner_metadata_sha256",), "0" * 64),
        (("public_identity", "adapter_code_sha256"), "0" * 64),
        (("files", 0, "transport_sha256"), "0" * 64),
        (("status",), "OWNER_METADATA_FROZEN_SOURCE_V6_LABEL_BINDING_VERIFIED"),
    ),
)
def test_r5_owner_receipt_rejects_hash_and_field_drift(
        path: tuple[object, ...], value: object) -> None:
    repository, raw = _receipt()
    drifted = deepcopy(raw)
    target: object = drifted
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(W02MorphologySuccessorV3PrivateOwnerR5Error):
        validate_w02_morphology_successor_v3_private_owner_r5_receipt(
            drifted, repository)


def test_r5_owner_receipt_has_no_private_root_or_payload_path() -> None:
    _repository, receipt = _receipt()
    text = str(receipt).lower()
    assert "d:\\" not in text
    assert "c:\\" not in text
    assert "owner-public-metadata" not in text
    assert "77a594e8813f77876b79c356e9161eb9\\" not in text
