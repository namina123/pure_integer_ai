"""公开多轮来源焦点与表层消费者开发切片。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_broad_qa_runtime import DialogueTurn
from pure_integer_ai.experiments.conversation_multiturn_scale import (
    MULTITURN_PASS,
    build_multiturn_scale_report,
    multiturn_questions,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATABASE = Path(
    "K:/pure_integer_ai_work/broad_qa_week_v1/indexes/"
    "broad-qa-20k-from-100k-target-v2.sqlite3")
_RUN_ROOT = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")
_PACK_SHA = "1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d"


def test_multiturn_questions_have_unknown_recovery_sequence() -> None:
    questions = multiturn_questions()
    assert len(questions) == 6
    assert questions[1].startswith("火星上的")
    assert questions[2].startswith("它")
    assert "完整句子" in questions[5]


def test_dialogue_turn_keeps_retrieval_question_as_optional_audit_value() -> None:
    turn = DialogueTurn(
        0, "问题", "答案", "答案", "ANSWER", "来源", None, (1, 2, 3))
    assert turn.retrieval_question is None


def test_real_k_multiturn_slice_passes_focus_and_recovery() -> None:
    if not _DATABASE.is_file() or not _RUN_ROOT.is_dir():
        pytest.skip("K: public dialogue artifacts are not present")
    report = build_multiturn_scale_report(
        project_root=_ROOT,
        database_path=_DATABASE,
        training_run_root=_RUN_ROOT,
        expected_pack_sha256=_PACK_SHA,
    )
    assert report.status == MULTITURN_PASS
    assert report.question_count == 6
    assert report.answer_count == 4
    assert report.unknown_count == 1
    assert report.clarify_count == 1
    assert report.focus_injection_count == 1
    assert report.focus_not_crossed_unknown == 1
    assert report.trained_surface_used_count == 1
    assert report.replay_bit_identical is True
    assert report.turns[2].retrieval_question == report.turns[2].question
    assert report.turns[5].retrieval_question != report.turns[5].question
