"""整数长度直算性能 successor receipt 的身份与权限边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from scripts.performance_successor_receipt import (
    RECEIPT_PATH,
    V2_CHANGE_COMMIT,
    V2_PARENT_COMMIT,
    V2_RECEIPT_PATH,
    build_performance_successor_receipt,
    build_v2_performance_successor_receipt,
    publish_performance_successor_receipt,
    publish_v2_performance_successor_receipt,
    read_performance_successor_receipt,
    read_v2_performance_successor_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SHA256 = (
    "01ecdb29437d3ce7ac88e126cc7a4ccff206fc458290cf7ad65f80457d9ecb17"
)


def test_published_v1_receipt_is_historical_and_strictly_invalidated() -> None:
    target = ROOT / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PUBLISHED_SHA256
    value = read_performance_successor_receipt(ROOT, verify_current=False)
    assert value["status"] == "PERFORMANCE_SUCCESSOR_EVIDENCED"
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    with pytest.raises(ValueError, match="当前 identity 漂移"):
        read_performance_successor_receipt(ROOT)


def test_v1_performance_receipt_build_is_invalidated_by_v2_source() -> None:
    with pytest.raises(ValueError, match="源码 identity 漂移"):
        build_performance_successor_receipt(ROOT)


def test_v1_performance_receipt_publish_remains_append_only() -> None:
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_performance_successor_receipt(ROOT)


def test_performance_receipt_rejects_source_and_readiness_drift(
        tmp_path: Path,
        ) -> None:
    value = read_performance_successor_receipt(ROOT, verify_current=False)
    target = tmp_path / "receipt.json"
    altered = json.loads(json.dumps(value))
    altered["source_bindings"][0]["current_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="源码绑定"):
        read_performance_successor_receipt(
            ROOT, target, verify_current=False)

    altered = json.loads(json.dumps(value))
    altered["readiness_transition"]["LANGUAGE_READINESS_REPUBLISHED"] = 1
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_performance_successor_receipt(
            ROOT, target, verify_current=False)


def test_v2_performance_receipt_builds_with_validated_kernel(
        tmp_path: Path,
        ) -> None:
    value = build_v2_performance_successor_receipt(ROOT)
    assert value["change_commit"] == V2_CHANGE_COMMIT
    assert value["parent_commit"] == V2_PARENT_COMMIT
    assert value["receipt_relative_path"] == V2_RECEIPT_PATH
    assert value["transformation"]["public_validation_changed"] == 0
    assert value["transformation"]["validated_kernel_owner"] == (
        "FROZEN_SLOTS_SEGMENT_RECORD"
    )
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    target = tmp_path / "v2.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    assert read_v2_performance_successor_receipt(ROOT, target) == value


def test_v2_performance_receipt_publish_is_append_only(tmp_path: Path) -> None:
    target = tmp_path / "v2.json"
    published = publish_v2_performance_successor_receipt(ROOT, target=target)
    assert read_v2_performance_successor_receipt(ROOT, target) == published
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_v2_performance_successor_receipt(ROOT, target=target)


def test_v2_performance_receipt_rejects_source_and_readiness_drift(
        tmp_path: Path,
        ) -> None:
    value = build_v2_performance_successor_receipt(ROOT)
    target = tmp_path / "v2.json"
    altered = json.loads(json.dumps(value))
    altered["source_bindings"][0]["current_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="源码绑定"):
        read_v2_performance_successor_receipt(ROOT, target)

    altered = json.loads(json.dumps(value))
    altered["readiness_transition"]["LANGUAGE_READINESS_REPUBLISHED"] = 1
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_v2_performance_successor_receipt(ROOT, target)
