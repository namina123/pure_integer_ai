"""公开 24 问广域问答主证据审计与结构残片回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.broad_qa_dev_surface_audit import (
    build_broad_qa_dev_audit,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATABASE = Path(
    "K:/pure_integer_ai_work/broad_qa_week_v1/indexes/"
    "broad-qa-20k-from-100k-target-v2.sqlite3")


def test_real_k_dev_questions_require_readable_primary_evidence() -> None:
    if not _DATABASE.is_file():
        pytest.skip("K: broad QA database is not present")
    report = build_broad_qa_dev_audit(
        project_root=_ROOT, database_path=_DATABASE)
    assert report.status == "PASS"
    assert report.question_count == 24
    assert report.answer_count == 23
    assert report.unknown_count == 1
    assert report.evidence_expected_count == 23
    assert report.evidence_hit_count == 23
    assert report.primary_surface_clean_count == 23
    assert report.long_answer_count == 23
    assert report.replay_bit_identical is True
    assert "银牌" in report.observations[7].primary_evidence
    assert "每年7月第二个周末" in report.observations[13].primary_evidence
    assert "6,000" in report.observations[14].primary_evidence
    assert "爱德华·斯特凡" in report.observations[15].primary_evidence
    assert "80万" in report.observations[18].primary_evidence
    assert not report.observations[7].primary_evidence.startswith("Category:")
