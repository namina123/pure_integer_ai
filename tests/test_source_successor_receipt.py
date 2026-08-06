"""Hasher successor receipt 的 append-only、identity 与 readiness 边界。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.source_successor_receipt import (
    ARTIFACT_KIND,
    PARENT_COMMIT,
    RECEIPT_PATH,
    build_source_successor_receipt,
    publish_source_successor_receipt,
    read_source_successor_receipt,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def test_successor_receipt_is_currently_buildable_without_readiness_transfer(
        tmp_path: Path):
    value = build_source_successor_receipt(ROOT)
    assert value["artifact_kind"] == ARTIFACT_KIND
    assert value["parent_commit"] == PARENT_COMMIT
    assert value["receipt_relative_path"] == RECEIPT_PATH
    assert value["status"] == "SOURCE_SUCCESSOR_EVIDENCED"
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    target = tmp_path / "receipt.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    assert read_source_successor_receipt(ROOT, target) == value


def test_publish_is_append_only_and_duplicate_does_not_rebuild(tmp_path: Path,
                                                               monkeypatch):
    target = tmp_path / "receipt.json"
    published = publish_source_successor_receipt(ROOT, target=target)
    assert read_source_successor_receipt(ROOT, target) == published
    monkeypatch.setattr(
        "pure_integer_ai.experiments.source_successor_receipt.build_source_successor_receipt",
        lambda _root: (_ for _ in ()).throw(AssertionError("duplicate rebuilt")),
    )
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_source_successor_receipt(ROOT, target=target)


def test_reader_rejects_current_identity_and_readiness_drift(tmp_path: Path):
    value = build_source_successor_receipt(ROOT)
    target = tmp_path / "receipt.json"
    payload = json.loads(json.dumps(value))
    payload["source_bindings"][0]["current_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="当前 identity 漂移"):
        read_source_successor_receipt(ROOT, target)

    payload = value.copy()
    payload["readiness_transition"] = {
        "LANGUAGE_READINESS_REPUBLISHED": 1,
        "PW00A_STARTED": 0,
    }
    target.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_source_successor_receipt(ROOT, target)
