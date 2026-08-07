"""整数长度直算性能 successor receipt 的身份与权限边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from scripts.performance_successor_receipt import (
    ARTIFACT_KIND,
    CHANGE_COMMIT,
    PARENT_COMMIT,
    RECEIPT_PATH,
    build_performance_successor_receipt,
    publish_performance_successor_receipt,
    read_performance_successor_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SHA256 = (
    "01ecdb29437d3ce7ac88e126cc7a4ccff206fc458290cf7ad65f80457d9ecb17"
)


def test_published_performance_receipt_is_canonical_and_current() -> None:
    target = ROOT / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PUBLISHED_SHA256
    value = read_performance_successor_receipt(ROOT)
    assert value["status"] == "PERFORMANCE_SUCCESSOR_EVIDENCED"
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }


def test_performance_receipt_builds_without_readiness_transfer(
        tmp_path: Path,
        ) -> None:
    value = build_performance_successor_receipt(ROOT)
    assert value["artifact_kind"] == ARTIFACT_KIND
    assert value["change_commit"] == CHANGE_COMMIT
    assert value["parent_commit"] == PARENT_COMMIT
    assert value["receipt_relative_path"] == RECEIPT_PATH
    assert value["status"] == "PERFORMANCE_SUCCESSOR_EVIDENCED"
    assert value["transformation"]["cache_cycle"]["delta_per_mille"] == -74
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    target = tmp_path / "receipt.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    assert read_performance_successor_receipt(ROOT, target) == value


def test_performance_receipt_publish_is_append_only(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    published = publish_performance_successor_receipt(ROOT, target=target)
    assert read_performance_successor_receipt(ROOT, target) == published
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_performance_successor_receipt(ROOT, target=target)


def test_performance_receipt_rejects_source_and_readiness_drift(
        tmp_path: Path,
        ) -> None:
    value = build_performance_successor_receipt(ROOT)
    target = tmp_path / "receipt.json"
    altered = json.loads(json.dumps(value))
    altered["source_bindings"][0]["current_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="源码绑定"):
        read_performance_successor_receipt(ROOT, target)

    altered = json.loads(json.dumps(value))
    altered["readiness_transition"]["LANGUAGE_READINESS_REPUBLISHED"] = 1
    target.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_performance_successor_receipt(ROOT, target)
