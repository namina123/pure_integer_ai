"""T1-G23：多标注者共识/冲突/缺标合并测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    CONSENSUS_ACCEPT,
    CONSENSUS_CONFLICT,
    CONSENSUS_INCOMPLETE,
    RawT1AnnotationConsensusError,
    merge_raw_t1_annotation_submissions,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    ANNOTATION_DEFER,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextSpanUnit,
    compile_raw_text_observation,
    load_raw_text_observation_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"


def _base():
    observation = load_raw_text_observation_jsonl(_OBS.read_bytes())[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    candidate = extraction.candidates[0]

    def submission(scope: str, decision: int = ANNOTATION_ACCEPT,
                   role: str = "sentence_candidate", reason: int = 1):
        item = RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, decision, role, reason)
        return RawT1AnnotationSubmission(
            f"ann-{scope}", scope, observation.observation_id,
            observation.source_namespace, (item,))
    return observation, extraction, submission


def test_identical_independent_submissions_form_ready_consensus() -> None:
    observation, extraction, submission = _base()
    result = merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))

    assert result.reviewer_scopes == ("a", "b")
    assert result.decisions[0].status == CONSENSUS_ACCEPT
    assert result.ready_for_training_review is True
    assert result.decisions[0].reviewer_count == 2
    assert all(type(value) is int and value >= 0 for value in result.canonical_record())


def test_different_roles_are_conflict_not_majority_vote() -> None:
    observation, extraction, submission = _base()
    result = merge_raw_t1_annotation_submissions(
        extraction, observation,
        (submission("a"), submission("b", role="other_role")))

    assert result.decisions[0].status == CONSENSUS_CONFLICT
    assert result.ready_for_training_review is False
    assert result.conflict_count == 1


def test_missing_candidate_or_duplicate_scope_fails_closed() -> None:
    observation, extraction, submission = _base()
    first = submission("a")
    with pytest.raises(RawT1AnnotationConsensusError, match="至少需要两个"):
        merge_raw_t1_annotation_submissions(extraction, observation, (first,))
    with pytest.raises(RawT1AnnotationConsensusError, match="不得重复"):
        merge_raw_t1_annotation_submissions(extraction, observation, (first, first))

    deferred = submission("b", decision=ANNOTATION_DEFER, role="", reason=7)
    result = merge_raw_t1_annotation_submissions(
        extraction, observation, (first, deferred))
    assert result.decisions[0].status == CONSENSUS_CONFLICT


def test_unsubmitted_candidate_is_incomplete_not_implicitly_rejected() -> None:
    raw = tuple("甲。乙".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g23-two", source_id="src-g23-two",
        context_id="ctx-g23-two", family_id="fam-g23-two",
        source_namespace="t1-g23-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "opaque", 0, 1, 0, 3),))
    extraction = extract_raw_text_candidate_spans(raw)
    candidate = extraction.candidates[0]
    item = RawT1AnnotationDecision(
        candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
        candidate.start_byte, candidate.end_byte, ANNOTATION_ACCEPT,
        "sentence_candidate", 1)
    one = RawT1AnnotationSubmission(
        "ann-g23-one", "a", observation.observation_id,
        observation.source_namespace, (item,))
    other = RawT1AnnotationSubmission(
        "ann-g23-other", "b", observation.observation_id,
        observation.source_namespace, (item,))
    result = merge_raw_t1_annotation_submissions(extraction, observation, (one, other))
    assert result.decisions[1].status == CONSENSUS_INCOMPLETE
    assert result.ready_for_training_review is False


def test_one_reviewer_omitting_a_candidate_cannot_form_ready_consensus() -> None:
    observation, extraction, submission = _base()
    first = submission("a")
    second = RawT1AnnotationSubmission(
        "ann-b-empty", "b", observation.observation_id,
        observation.source_namespace, (RawT1AnnotationDecision(
            0, extraction.candidates[0].start_scalar,
            extraction.candidates[0].end_scalar,
            extraction.candidates[0].start_byte,
            extraction.candidates[0].end_byte,
            ANNOTATION_DEFER, "", 9),))
    # Both submit candidate 0, so this remains a conflict; the separate
    # two-candidate fixture below covers a genuinely omitted ordinal.
    result = merge_raw_t1_annotation_submissions(
        extraction, observation, (first, second))
    assert result.decisions[0].status == CONSENSUS_CONFLICT
