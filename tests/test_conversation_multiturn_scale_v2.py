"""扩大来源覆盖与主证据命中开发切片。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_multiturn_scale_v2 import (
    build_expanded_multiturn_report,
    expanded_multiturn_questions,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATABASE = Path(
    "K:/pure_integer_ai_work/broad_qa_week_v1/indexes/"
    "broad-qa-20k-from-100k-target-v2.sqlite3")
_RUN_ROOT = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")
_PACK_SHA = "1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d"


def test_expanded_slice_covers_five_source_domains_and_recovery() -> None:
    questions = expanded_multiturn_questions()
    assert len(questions) == 19
    assert "保满铁路" in questions[0]
    assert "矮寨大桥" in questions[3]
    assert "维基数据" in questions[6]
    assert "儋州西庆机场" in questions[9]
    assert "火星上的矮寨大桥" in questions[13]
    assert "黄山松" in questions[16]


def test_real_k_expanded_slice_requires_evidence_tokens_and_replay() -> None:
    if not _DATABASE.is_file() or not _RUN_ROOT.is_dir():
        pytest.skip("K: public dialogue artifacts are not present")
    report = build_expanded_multiturn_report(
        project_root=_ROOT,
        database_path=_DATABASE,
        training_run_root=_RUN_ROOT,
        expected_pack_sha256=_PACK_SHA,
    )
    assert report.status == "PASS"
    assert report.scenario_count == 6
    assert report.question_count == 19
    assert report.answer_count == 17
    assert report.unknown_count == 1
    assert report.clarify_count == 1
    assert report.evidence_expected_count == 17
    assert report.evidence_hit_count == 17
    assert report.long_answer_count == 16
    assert report.trained_surface_used_count == 1
    assert report.focus_injection_count == 10
    assert report.focus_not_crossed_unknown == 1
    assert report.replay_bit_identical is True
