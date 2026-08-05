"""J-F2 公开 preflight 的只读、canonical 和 fail-closed 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_j_f2_contract import (
    ARTIFACT_KIND,
    CORE_ARTIFACT_PATH,
    J_F1_RECEIPT_PATH,
    JF2PreflightError,
    build_jf2_preflight,
    read_jf2_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_reads_public_receipts_and_stays_blocked():
    """W09 PASS 不得掩盖缺失的 J-F1/Core 最终封存依赖。"""
    report = build_jf2_preflight(ROOT)
    assert report.artifact_kind == ARTIFACT_KIND
    assert report.status == "BLOCKED"
    assert report.language_capability_mastered == 1
    assert report.language_readiness == 0
    assert "J_F1_FACILITY_MISSING" in report.blockers
    assert "CORE_ARTIFACT_MISSING" in report.blockers
    assert all(item.role != "PRIVATE_PAYLOAD" for item in report.dependencies)


def test_preflight_dependency_order_and_canonical_round_trip(tmp_path):
    """报告依赖顺序固定，写回后必须逐字节等于 canonical bytes。"""
    report = build_jf2_preflight(ROOT)
    target = tmp_path / "preflight.json"
    target.write_bytes(report.canonical_bytes())
    restored = read_jf2_preflight(target)
    assert restored == report
    assert restored.canonical_bytes() == target.read_bytes()


@pytest.mark.parametrize("relative_path", (J_F1_RECEIPT_PATH, CORE_ARTIFACT_PATH))
def test_future_final_dependencies_are_not_present(relative_path):
    """最终 J-F2 依赖当前必须缺失，不能用占位文件伪造 ready。"""
    assert not (ROOT / Path(*relative_path.split("/"))).exists()


def test_preflight_rejects_noncanonical_report(tmp_path):
    """额外换行或字段重排必须 fail closed。"""
    report = build_jf2_preflight(ROOT)
    target = tmp_path / "preflight.json"
    target.write_bytes(report.canonical_bytes() + b"\n")
    with pytest.raises(JF2PreflightError):
        read_jf2_preflight(target)
