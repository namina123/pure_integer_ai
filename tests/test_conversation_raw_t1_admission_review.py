"""T1-G21：admission review 的中文可读投影与整数稳定性测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_admission_review import (
    REVIEW_ELIGIBLE,
    REVIEW_NEGATIVE,
    RawT1AdmissionReviewError,
    build_raw_t1_admission_review,
    render_raw_t1_admission_review_zh,
)
from pure_integer_ai.experiments.conversation_raw_t1_training_admission import (
    admit_raw_t1_training_pack,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data/ph2"


def _admission():
    names = (
        "dlg_raw_text_observation_v1.jsonl.sample",
        "dlg_raw_lexical_evidence_v1.jsonl.sample",
        "dlg_raw_proposition_relation_evidence_v1.jsonl.sample",
        "dlg_raw_proposition_qualification_v1.jsonl.sample",
    )
    return admit_raw_t1_training_pack(*tuple((_DATA / name).read_bytes() for name in names))


def test_review_preserves_physical_rows_and_renders_chinese_text() -> None:
    review = build_raw_t1_admission_review(_admission())
    repeat = build_raw_t1_admission_review(_admission())

    assert review.eligible_count == 2
    assert review.negative_count == 1
    assert review.rejected_count == 0
    assert tuple(item.review_status for item in review.rows) == (
        REVIEW_ELIGIBLE, REVIEW_NEGATIVE, REVIEW_ELIGIBLE)
    assert review.canonical_record() == repeat.canonical_record()
    text = render_raw_t1_admission_review_zh(review)
    assert "T1 原始文本 admission 观察报告" in text
    assert "可送独立标注审核" in text
    assert "仅保留为负例见证" in text
    assert "词义、命题或现实真值" in text


def test_review_rejects_wrong_input_without_writing() -> None:
    with pytest.raises(TypeError, match="RawT1TrainingAdmission"):
        build_raw_t1_admission_review(object())
    with pytest.raises(TypeError, match="RawT1AdmissionReview"):
        render_raw_t1_admission_review_zh(object())
