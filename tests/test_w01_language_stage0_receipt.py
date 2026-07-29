"""W-01 正式 report/receipt 的规范回读、证据绑定和不可覆盖反例。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w01_receipt import (
    W01_FORMAL_RECEIPT_PATH,
    W01_FORMAL_RECEIPT_V1_PATH,
    W01ReceiptError,
    build_w01_formal_receipt,
    read_w01_formal_receipt,
    write_w01_formal_receipt,
)
from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes


_REPOSITORY = Path(__file__).resolve().parents[1]


def test_formal_receipt_round_trip_binds_report_implementation_and_tests():
    """正式 receipt 必须精确回读 run bundle、实现/测试和零学习状态。"""
    receipt = read_w01_formal_receipt(_REPOSITORY)
    rebuilt = build_w01_formal_receipt(_REPOSITORY)

    assert receipt == rebuilt
    assert receipt.to_dict()["status"] == "W01_PROTOCOL_VERIFIED"
    assert receipt.execution_state["W01_PROTOCOL_VERIFIED"] == 1
    assert receipt.execution_state["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert receipt.execution_state["LANGUAGE_READINESS"] == 0
    assert receipt.execution_state["W02_STARTED"] == 0
    assert receipt.execution_state["teacher_calls"] == 0
    assert receipt.execution_state["formal_training_runs"] == 0
    assert receipt.execution_state["w02_semantic_writes"] == 0
    assert len(receipt.formal_run_inventory) == 6
    assert len(receipt.implementation_inventory) == 10
    assert len(receipt.test_inventory) == 4
    assert (receipt.superseded_receipt_identity.relative_path
            == W01_FORMAL_RECEIPT_V1_PATH)


def test_formal_receipt_reader_rejects_noncanonical_or_tampered_state(
        tmp_path: Path):
    """字段重排编码或把 mastered/W-02 改为一均必须 fail closed。"""
    original = json.loads(
        (_REPOSITORY / W01_FORMAL_RECEIPT_PATH).read_text(encoding="utf-8"))
    original["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] = 1
    target = tmp_path / W01_FORMAL_RECEIPT_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_json_bytes(original))

    with pytest.raises(W01ReceiptError, match="execution state"):
        read_w01_formal_receipt(tmp_path)


def test_formal_receipt_writer_is_idempotent_but_never_overwrites(
        tmp_path: Path):
    """同字节重写幂等，任何同路径异字节 receipt 必须拒绝覆盖。"""
    receipt = read_w01_formal_receipt(_REPOSITORY)
    target = tmp_path / "receipt.json"
    assert write_w01_formal_receipt(receipt, target) == target
    assert write_w01_formal_receipt(receipt, target) == target

    changed = replace(
        receipt,
        d03_identity={**receipt.d03_identity, "release_key": "DRIFTED"},
    )
    with pytest.raises(W01ReceiptError, match="不可覆盖"):
        write_w01_formal_receipt(changed, target)
