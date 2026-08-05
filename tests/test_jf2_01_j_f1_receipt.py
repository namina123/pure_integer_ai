"""JF2-01 J-F1 facility receipt 的 canonical、绑定和 append-only 专项。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import pure_integer_ai.experiments.j_f1_facility_receipt as receipt_contract
from pure_integer_ai.experiments.j_f1_facility_receipt import (
    HISTORICAL_REPORT_SHA256,
    JF1FacilityReceipt,
    JF1ReceiptError,
    J_F1_IMPLEMENTATION_PATHS,
    J_F1_RECEIPT_RELATIVE_PATH,
    build_j_f1_facility_receipt,
    read_j_f1_facility_receipt,
    write_j_f1_facility_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def staged_receipt(tmp_path_factory):
    """从一次真实 production report 构造并临时回读 receipt。"""
    receipt = build_j_f1_facility_receipt(ROOT)
    target = tmp_path_factory.mktemp("j-f1-receipt") / "receipt.json"
    write_j_f1_facility_receipt(receipt, target)
    restored = read_j_f1_facility_receipt(
        ROOT, receipt_path=target, verify_runtime=False)
    return receipt, restored, target


def test_jf2_01_receipt_is_canonical_self_excluded_and_live_bound(
        staged_receipt,
        ):
    """临时 receipt 必须逐字节回读，并绑定全部生产实现而不绑定自身。"""
    receipt, restored, target = staged_receipt
    value = restored.to_dict()
    assert restored == receipt
    assert target.read_bytes() == receipt.canonical_bytes()
    assert target.read_bytes().endswith(b"\n")
    assert not target.read_bytes().endswith(b"\n\n")
    assert value["receipt_relative_path"] == J_F1_RECEIPT_RELATIVE_PATH
    assert value["receipt_self_excluded"] == 1
    assert all(
        binding["codec"] == "zlib"
        and binding["size_bytes"] > 0
        and binding["payload_hex"]
        for binding in value["identity_bindings"].values()
    )
    assert tuple(
        item["relative_path"]
        for item in value["implementation_inventory"]
    ) == J_F1_IMPLEMENTATION_PATHS
    assert J_F1_RECEIPT_RELATIVE_PATH not in J_F1_IMPLEMENTATION_PATHS
    assert all(not path.startswith("tests/")
               for path in J_F1_IMPLEMENTATION_PATHS)
    assert HISTORICAL_REPORT_SHA256.encode("ascii") not in target.read_bytes()


def test_jf2_01_receipt_carries_complete_facility_and_honest_boundary(
        staged_receipt,
        ):
    """公开 receipt 必须保存完整合取，同时不授予 readiness 或 J-F2 seal。"""
    value = staged_receipt[1].to_dict()
    assert value["status"] == "FACILITY_EVIDENCED"
    assert value["measurements"]["facility_complete"] == 1
    assert len(value["measurements"]["metrics"]) == 12
    assert len(value["measurements"]["checks"]) == 10
    assert len(value["measurements"]["forbidden_counters"]) == 7
    assert all(item["value"] == 0
               for item in value["measurements"]["forbidden_counters"])
    assert len(value["dimensions"]) == 5
    assert all(item["passed"] == 1 for item in value["dimensions"])
    assert len(value["mechanisms"]) == 13
    assert all(item["passed"] == 1 for item in value["mechanisms"])
    assert value["identity_bindings"]["core_before"] == (
        value["identity_bindings"]["core_after"])
    assert value["identity_bindings"]["host_before"] == (
        value["identity_bindings"]["host_after"])
    assert value["honest_boundary"]["language_readiness"] == 0
    assert value["honest_boundary"][
        "facility_grants_language_readiness"] == 0
    assert value["honest_boundary"]["j_f2_final_seal_published"] == 0


def test_jf2_01_receipt_rejects_duplicate_publish_even_if_bytes_match(
        staged_receipt,
        ):
    """同路径同字节也不得幂等覆盖 append-only receipt。"""
    receipt, _, target = staged_receipt
    with pytest.raises(JF1ReceiptError, match="append-only"):
        write_j_f1_facility_receipt(receipt, target)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value.pop("facility_evidence"),
        lambda value: value["production_evidence"].__setitem__(
            "test_module_import_count", 1),
        lambda value: value["mechanisms"][0].__setitem__(
            "status", "test-only"),
        lambda value: value["identity_bindings"].__setitem__(
            "core_after", value["identity_bindings"]["runtime"]),
        lambda value: value["identity_bindings"]["report"].__setitem__(
            "size_bytes",
            value["identity_bindings"]["report"]["size_bytes"] + 1,
        ),
        lambda value: value["measurements"]["metrics"][0].__setitem__(
            "value", 1),
    ),
)
def test_jf2_01_receipt_tampering_fails_closed(staged_receipt, mutator):
    """缺字段、测试来源、Core 或 report measurement 漂移都必须拒绝。"""
    value = deepcopy(staged_receipt[0].to_dict())
    mutator(value)
    with pytest.raises(JF1ReceiptError):
        JF1FacilityReceipt.from_dict(value)


def test_jf2_01_live_replay_rejects_structurally_valid_report_replacement(
        staged_receipt,
        monkeypatch,
        tmp_path,
        ):
    """结构合法但不属于本次真实 report 的 identity 必须在 live 回验失败。"""
    original = staged_receipt[0]
    value = deepcopy(original.to_dict())
    value["identity_bindings"]["report"] = deepcopy(
        value["identity_bindings"]["runtime"])
    replaced = JF1FacilityReceipt.from_dict(value)
    target = tmp_path / "replaced-report.json"
    write_j_f1_facility_receipt(replaced, target)
    monkeypatch.setattr(
        receipt_contract,
        "build_j_f1_facility_receipt",
        lambda _repository: original,
    )
    with pytest.raises(JF1ReceiptError, match="live production report"):
        read_j_f1_facility_receipt(
            ROOT, receipt_path=target, verify_runtime=True)
