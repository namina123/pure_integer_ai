"""T1-G22：独立标注提交绑定与 fail-closed 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    ANNOTATION_DEFER,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
    RawT1AnnotationSubmissionError,
    validate_raw_t1_annotation_submission,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"


def _bound():
    observation = load_raw_text_observation_jsonl(_OBS.read_bytes())[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    candidate = extraction.candidates[0]
    decisions = (
        RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, ANNOTATION_ACCEPT,
            "sentence_candidate", 1),
    )
    submission = RawT1AnnotationSubmission(
        "ann-g22-01", "reviewer-public-a", observation.observation_id,
        observation.source_namespace, decisions)
    return extraction, observation, submission


def test_submission_round_trips_and_binds_exact_candidate_span() -> None:
    extraction, observation, submission = _bound()
    result = validate_raw_t1_annotation_submission(extraction, observation, submission)

    assert result is submission
    assert result.canonical_record() == submission.canonical_record()
    assert all(type(value) is int and value >= 0 for value in result.canonical_record())


def test_submission_rejects_span_or_identity_drift() -> None:
    extraction, observation, submission = _bound()
    changed = RawT1AnnotationSubmission(
        submission.annotation_id, submission.reviewer_scope,
        submission.observation_id, submission.source_namespace,
        (RawT1AnnotationDecision(0, 0, 1, 0, 1, ANNOTATION_ACCEPT,
                                 "sentence_candidate", 1),))
    with pytest.raises(RawT1AnnotationSubmissionError, match="span 漂移"):
        validate_raw_t1_annotation_submission(extraction, observation, changed)
    changed_identity = RawT1AnnotationSubmission(
        submission.annotation_id, submission.reviewer_scope, "other-observation",
        submission.source_namespace, submission.decisions)
    with pytest.raises(RawT1AnnotationSubmissionError, match="observation identity"):
        validate_raw_t1_annotation_submission(extraction, observation, changed_identity)


def test_defer_may_omit_label_but_unknown_candidate_fails_closed() -> None:
    extraction, observation, submission = _bound()
    candidate = extraction.candidates[0]
    deferred = RawT1AnnotationSubmission(
        "ann-g22-defer", "reviewer-public-b", observation.observation_id,
        observation.source_namespace,
        (RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, ANNOTATION_DEFER, "", 7),))
    assert validate_raw_t1_annotation_submission(extraction, observation, deferred)
    unknown = RawT1AnnotationSubmission(
        "ann-g22-unknown", "reviewer-public-b", observation.observation_id,
        observation.source_namespace,
        (RawT1AnnotationDecision(99, 0, 1, 0, 1, ANNOTATION_ACCEPT,
                                 "sentence_candidate", 1),))
    with pytest.raises(RawT1AnnotationSubmissionError, match="ordinal 未知"):
        validate_raw_t1_annotation_submission(extraction, observation, unknown)
