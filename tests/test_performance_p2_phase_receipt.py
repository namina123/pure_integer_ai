"""PERF-P2 phase profile receipt 的 canonical 与外部证据边界。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import performance_p2_phase_receipt
from scripts.performance_p2_phase_receipt import (
    PROFILE_COMMIT,
    PROFILE_ROOT_NAME,
    RECEIPT_PATH,
    STATUS,
    build_performance_phase_profile_receipt,
    publish_performance_phase_profile_receipt,
    read_performance_phase_profile_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT.parent / PROFILE_ROOT_NAME
PUBLISHED_SHA256 = (
    "cbf94342625485861a38d46bc4146bddfe6668f03c81506ecc9e184590e52eea"
)


def test_published_phase_receipt_is_canonical_without_external_root() -> None:
    target = ROOT / RECEIPT_PATH
    if not target.exists():
        pytest.skip("P2 phase profile receipt 尚未发布")
    payload = target.read_bytes()
    value = read_performance_phase_profile_receipt(
        ROOT, verify_external=False)
    assert hashlib.sha256(payload).hexdigest() == PUBLISHED_SHA256
    assert value["status"] == STATUS
    assert value["profile_commit"] == PROFILE_COMMIT
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    assert payload.endswith(b"\n")


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "memory-attempt-001.json").exists(),
    reason="Git 外 P2 r3 profile artifact 不存在",
)
def test_phase_receipt_build_replays_all_r3_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.performance_p2_phase_receipt.read_head",
        lambda root: PROFILE_COMMIT,
    )
    value = build_performance_phase_profile_receipt(ROOT, ARTIFACT_ROOT)
    assert value["verification"] == {
        "canonical_report_count": 10,
        "empty_stderr_count": 10,
        "float_value_count": 0,
        "sqlite_database_count": 5,
    }
    restored = read_performance_phase_profile_receipt(
        ROOT, ARTIFACT_ROOT, path=ROOT / RECEIPT_PATH,
        verify_external=False,
    ) if (ROOT / RECEIPT_PATH).exists() else None
    if restored is not None:
        assert restored["scenario_summaries"] == value["scenario_summaries"]


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "memory-attempt-001.json").exists(),
    reason="Git 外 P2 r3 profile artifact 不存在",
)
def test_phase_receipt_publish_is_append_only(
        tmp_path: Path,
        monkeypatch,
        ) -> None:
    monkeypatch.setattr(
        "scripts.performance_p2_phase_receipt.read_head",
        lambda root: PROFILE_COMMIT,
    )
    target = tmp_path / "performance_phase_profile_receipt.json"
    value = publish_performance_phase_profile_receipt(
        ROOT, ARTIFACT_ROOT, target=target)
    assert read_performance_phase_profile_receipt(
        ROOT, ARTIFACT_ROOT, target) == value
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_performance_phase_profile_receipt(
            ROOT, ARTIFACT_ROOT, target=target)


@pytest.mark.skipif(
    not (ROOT.parent / ".perf-p2-phase-s128-r2" / "memory-attempt-001.json").exists(),
    reason="Git 外 P2 r2 invalid capture 不存在",
)
def test_phase_receipt_rejects_utf16_capture(monkeypatch) -> None:
    invalid_root = ROOT.parent / ".perf-p2-phase-s128-r2"
    monkeypatch.setattr(
        performance_p2_phase_receipt, "PROFILE_ROOT_NAME",
        invalid_root.name,
    )
    monkeypatch.setattr(
        performance_p2_phase_receipt, "read_head",
        lambda root: PROFILE_COMMIT,
    )
    with pytest.raises(ValueError, match="canonical"):
        build_performance_phase_profile_receipt(ROOT, invalid_root)
