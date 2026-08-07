"""Canonical and external-evidence checks for the PERF-P3 trial receipt."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.performance_p3_sqlite_trial_receipt import (
    RECEIPT_PATH,
    STATUS,
    TRIAL_COMMIT,
    TRIAL_ROOT_NAME,
    build_performance_p3_sqlite_trial_receipt,
    publish_performance_p3_sqlite_trial_receipt,
    read_performance_p3_sqlite_trial_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT.parent / TRIAL_ROOT_NAME
PUBLISHED_SHA256 = (
    "fe82107dd792ac868564999f5f5dfde130319db81dcbec5a6a3420aeffb3f605"
)


def test_published_trial_receipt_is_canonical_without_external_root() -> None:
    target = ROOT / RECEIPT_PATH
    if not target.exists():
        pytest.skip("P3 SQLite trial receipt not published yet")
    payload = target.read_bytes()
    value = read_performance_p3_sqlite_trial_receipt(
        ROOT, verify_external=False)
    assert hashlib.sha256(payload).hexdigest() == PUBLISHED_SHA256
    assert value["status"] == STATUS
    assert value["trial_commit"] == TRIAL_COMMIT
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "reports" / "r01-s32.json").exists(),
    reason="Git-external P3 SQLite trial artifact is absent",
)
def test_trial_receipt_replays_all_external_artifacts() -> None:
    value = build_performance_p3_sqlite_trial_receipt(ROOT, ARTIFACT_ROOT)
    assert value["verification"] == {
        "canonical_report_count": 15,
        "empty_stderr_count": 15,
        "exception_path_pass_count": 15,
        "float_value_count": 0,
        "sqlite_database_count": 15,
    }
    assert [item["scale"] for item in value["scale_summaries"]] == [32, 128, 512]


@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "reports" / "r01-s32.json").exists(),
    reason="Git-external P3 SQLite trial artifact is absent",
)
def test_trial_receipt_publish_is_append_only(tmp_path: Path) -> None:
    target = tmp_path / "trial-receipt.json"
    value = publish_performance_p3_sqlite_trial_receipt(
        ROOT, ARTIFACT_ROOT, target=target)
    assert read_performance_p3_sqlite_trial_receipt(
        ROOT, ARTIFACT_ROOT, target) == value
    with pytest.raises(ValueError, match="overwrite forbidden"):
        publish_performance_p3_sqlite_trial_receipt(
            ROOT, ARTIFACT_ROOT, target=target)
