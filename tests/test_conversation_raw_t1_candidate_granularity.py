"""T1-G25：candidate evidence 与 lexical unit 粒度审计测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import merge_raw_t1_annotation_submissions
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import ANNOTATION_ACCEPT, RawT1AnnotationDecision, RawT1AnnotationSubmission
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    GRANULARITY_COVERS_MULTIPLE_UNITS, GRANULARITY_EXACT_UNIT,
    RawT1CandidateGranularityError, audit_raw_t1_candidate_granularity,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import project_raw_t1_consensus_candidate_evidence
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import extract_raw_text_candidate_spans
from pure_integer_ai.experiments.conversation_raw_text_observation import RawTextSpanUnit, compile_raw_text_observation, load_raw_text_observation_jsonl

_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"


def _candidate_evidence_for(observation, extraction, candidate):
    def submission(scope: str):
        item = RawT1AnnotationDecision(candidate.ordinal, candidate.start_scalar,
            candidate.end_scalar, candidate.start_byte, candidate.end_byte,
            ANNOTATION_ACCEPT, "sentence_candidate", 1)
        return RawT1AnnotationSubmission(f"ann-{scope}", scope,
            observation.observation_id, observation.source_namespace, (item,))
    consensus = merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))
    return project_raw_t1_consensus_candidate_evidence(
        consensus, observation, evidence_namespace="g25-public")[candidate.ordinal]


def test_full_sentence_candidate_covering_three_units_is_not_downcast() -> None:
    observation = load_raw_text_observation_jsonl(_OBS.read_bytes())[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    evidence = _candidate_evidence_for(observation, extraction, extraction.candidates[0])
    audit = audit_raw_t1_candidate_granularity(evidence, observation)
    assert audit.status == GRANULARITY_COVERS_MULTIPLE_UNITS
    assert audit.can_downcast_to_lexical_unit is False
    assert audit.matched_unit_ids == ("u1", "u2", "u3")


def test_exact_candidate_unit_span_is_the_only_downcastable_case() -> None:
    raw = tuple("甲。".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g25-exact", source_id="src-g25-exact",
        context_id="ctx-g25-exact", family_id="fam-g25-exact",
        source_namespace="t1-g25-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, 2, 0, len(raw)),))
    extraction = extract_raw_text_candidate_spans(raw)
    evidence = _candidate_evidence_for(observation, extraction, extraction.candidates[0])
    audit = audit_raw_t1_candidate_granularity(evidence, observation)
    assert audit.status == GRANULARITY_EXACT_UNIT
    assert audit.can_downcast_to_lexical_unit is True


def test_identity_drift_fails_closed() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    extraction = extract_raw_text_candidate_spans(observations[0].raw_bytes)
    evidence = _candidate_evidence_for(observations[0], extraction, extraction.candidates[0])
    with pytest.raises(RawT1CandidateGranularityError, match="identity 漂移"):
        audit_raw_t1_candidate_granularity(evidence, observations[1])
