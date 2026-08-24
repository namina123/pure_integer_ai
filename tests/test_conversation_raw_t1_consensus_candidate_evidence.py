"""T1-G24：ready consensus 的 candidate-level evidence 投影测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    merge_raw_t1_annotation_submissions,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    ANNOTATION_DEFER,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    EVIDENCE_DECISION_ACCEPT,
    RawT1ConsensusCandidateEvidenceError,
    project_raw_t1_consensus_candidate_evidence,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"


def _consensus():
    observation = load_raw_text_observation_jsonl(_OBS.read_bytes())[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    candidate = extraction.candidates[0]

    def submission(scope: str, decision: int = ANNOTATION_ACCEPT):
        item = RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, decision,
            "sentence_candidate", 1)
        return RawT1AnnotationSubmission(
            f"ann-{scope}", scope, observation.observation_id,
            observation.source_namespace, (item,))

    return observation, extraction, merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))


def test_ready_consensus_projects_without_collapsing_candidate_granularity() -> None:
    observation, extraction, consensus = _consensus()
    evidence = project_raw_t1_consensus_candidate_evidence(
        consensus, observation, evidence_namespace="g24-public")

    assert len(evidence) == len(extraction.candidates)
    assert evidence[0].decision == EVIDENCE_DECISION_ACCEPT
    assert evidence[0].evidence_id == "g24-public:obs-g0-train-01:0"
    assert evidence[0].start_scalar == extraction.candidates[0].start_scalar
    assert all(type(value) is int and value >= 0
               for item in evidence for value in item.canonical_record())


def test_unready_consensus_cannot_be_projected() -> None:
    observation = load_raw_text_observation_jsonl(_OBS.read_bytes())[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    candidate = extraction.candidates[0]
    item = RawT1AnnotationDecision(
        candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
        candidate.start_byte, candidate.end_byte, ANNOTATION_DEFER, "", 7)
    one = RawT1AnnotationSubmission(
        "ann-a-defer", "a", observation.observation_id,
        observation.source_namespace, (item,))
    two = RawT1AnnotationSubmission(
        "ann-b-defer", "b", observation.observation_id,
        observation.source_namespace, (item,))
    consensus = merge_raw_t1_annotation_submissions(extraction, observation, (one, two))
    with pytest.raises(RawT1ConsensusCandidateEvidenceError, match="尚未 ready"):
        project_raw_t1_consensus_candidate_evidence(
            consensus, observation, evidence_namespace="g24-public")
