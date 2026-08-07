"""首批结构体布局 successor receipt 的身份与权限边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from scripts.struct_layout_successor_receipt import (
    ARTIFACT_KIND,
    CHANGE_COMMIT,
    PARENT_COMMIT,
    RECEIPT_PATH,
    build_struct_layout_successor_receipt,
    publish_struct_layout_successor_receipt,
    read_struct_layout_successor_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SHA256 = (
    "2b34bbed1c7dab67e0cadcfb0ff00f64fd28754e67d023c9e718f8633944d2d7"
)


def test_published_struct_layout_receipt_is_canonical_and_current() -> None:
    target = ROOT / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PUBLISHED_SHA256
    value = read_struct_layout_successor_receipt(ROOT)
    assert value["status"] == "STRUCT_LAYOUT_SUCCESSOR_EVIDENCED"
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }


def test_struct_layout_receipt_builds_without_readiness_transfer(
        tmp_path: Path,
        ) -> None:
    value = build_struct_layout_successor_receipt(ROOT)
    assert value["artifact_kind"] == ARTIFACT_KIND
    assert value["change_commit"] == CHANGE_COMMIT
    assert value["parent_commit"] == PARENT_COMMIT
    assert value["receipt_relative_path"] == RECEIPT_PATH
    assert value["status"] == "STRUCT_LAYOUT_SUCCESSOR_EVIDENCED"
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    assert value["verification"]["full_suite_run"] == 0
    target = tmp_path / "receipt.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    assert read_struct_layout_successor_receipt(ROOT, target) == value


def test_struct_layout_receipt_publish_is_append_only(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    target = tmp_path / "receipt.json"
    published = publish_struct_layout_successor_receipt(ROOT, target=target)
    assert read_struct_layout_successor_receipt(ROOT, target) == published
    monkeypatch.setattr(
        "scripts.struct_layout_successor_receipt.build_struct_layout_successor_receipt",
        lambda _root: (_ for _ in ()).throw(AssertionError("duplicate rebuilt")),
    )
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_struct_layout_successor_receipt(ROOT, target=target)


def test_struct_layout_receipt_rejects_identity_and_readiness_drift(
        tmp_path: Path,
        ) -> None:
    value = build_struct_layout_successor_receipt(ROOT)
    target = tmp_path / "receipt.json"

    altered = json.loads(json.dumps(value))
    altered["source_binding"]["current_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="source binding"):
        read_struct_layout_successor_receipt(ROOT, target)

    altered = json.loads(json.dumps(value))
    altered["readiness_transition"]["LANGUAGE_READINESS_REPUBLISHED"] = 1
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_struct_layout_successor_receipt(ROOT, target)
