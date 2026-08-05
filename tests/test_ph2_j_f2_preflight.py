"""J-F2 公开 preflight 的只读、canonical 和 fail-closed 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_j_f2_contract as j_f2_contract
from pure_integer_ai.experiments.j_f1_facility_receipt import JF1ReceiptError
from pure_integer_ai.experiments.ph2_j_f2_contract import (
    ARTIFACT_KIND,
    CORE_ARTIFACT_PATH,
    J_F1_RECEIPT_PATH,
    JF2PreflightError,
    build_jf2_preflight,
    read_jf2_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_preflight():
    """只重跑一次真实 J-F1 adapter，供公开 preflight 专项共享。"""
    return build_jf2_preflight(ROOT)


def test_preflight_reads_public_receipts_and_is_ready_for_final_seal(live_preflight):
    """J-F1 与 Core artifact 均通过后，preflight 只允许进入最终封存前态。"""
    report = live_preflight
    assert report.artifact_kind == ARTIFACT_KIND
    assert report.status == "READY_FOR_FORMAL_SEAL"
    assert report.language_capability_mastered == 1
    assert report.language_readiness == 0
    assert "J_F1_FACILITY_MISSING" not in report.blockers
    assert "J_F1_FACILITY_INVALID" not in report.blockers
    assert report.blockers == ()
    assert next(
        item for item in report.dependencies
        if item.role == "J_F1_FACILITY"
    ).status == "PASS"
    assert next(
        item for item in report.dependencies
        if item.role == "CORE_ARTIFACT"
    ).status == "PASS"
    assert all(item.role != "PRIVATE_PAYLOAD" for item in report.dependencies)


def test_preflight_dependency_order_and_canonical_round_trip(
        live_preflight,
        tmp_path,
        ):
    """报告依赖顺序固定，写回后必须逐字节等于 canonical bytes。"""
    report = live_preflight
    target = tmp_path / "preflight.json"
    target.write_bytes(report.canonical_bytes())
    restored = read_jf2_preflight(target)
    assert restored == report
    assert restored.canonical_bytes() == target.read_bytes()


def test_core_artifact_is_published_at_reserved_path():
    """JF2-02 必须在预留路径发布真实 manifest，不允许占位文件。"""
    assert (ROOT / Path(*CORE_ARTIFACT_PATH.split("/"))).is_file()
    assert (ROOT / Path(*J_F1_RECEIPT_PATH.split("/"))).is_file()


def test_preflight_rejects_noncanonical_report(live_preflight, tmp_path):
    """额外换行或字段重排必须 fail closed。"""
    report = live_preflight
    target = tmp_path / "preflight.json"
    target.write_bytes(report.canonical_bytes() + b"\n")
    with pytest.raises(JF2PreflightError):
        read_jf2_preflight(target)


def test_preflight_marks_j_f1_invalid_when_content_replay_fails(monkeypatch):
    """J-F1 文件存在但内容、report 或覆盖回验失败时必须继续阻断。"""
    def reject_content(_repository, *, verify_runtime=False):
        """模拟严格 reader 对内容漂移的拒绝，并核对 live 回验未被关闭。"""
        assert verify_runtime is True
        raise JF1ReceiptError("J-F1 receipt 与 live production report 漂移")

    monkeypatch.setattr(
        j_f2_contract, "read_j_f1_facility_receipt", reject_content)
    report = build_jf2_preflight(ROOT)
    dependency = next(
        item for item in report.dependencies
        if item.role == "J_F1_FACILITY")
    assert dependency.status == "FAIL"
    assert "J_F1_FACILITY_INVALID" in report.blockers
    assert "J_F1_FACILITY_MISSING" not in report.blockers
    assert report.status == "BLOCKED"
    assert report.language_readiness == 0


def test_preflight_marks_core_artifact_invalid_on_content_drift(monkeypatch):
    """Core manifest 存在但严格回读失败时必须 fail closed。"""
    def reject_content(_repository, _path, *, verify_files=True):
        """模拟 Core manifest 的内容身份漂移。"""
        assert verify_files is True
        raise RuntimeError("Core artifact identity drift")

    monkeypatch.setattr(j_f2_contract, "read_core_artifact_manifest", reject_content)
    report = build_jf2_preflight(ROOT)
    dependency = next(
        item for item in report.dependencies
        if item.role == "CORE_ARTIFACT")
    assert dependency.status == "FAIL"
    assert report.blockers == ("CORE_ARTIFACT_INVALID",)
    assert report.status == "BLOCKED"
    assert report.language_readiness == 0
