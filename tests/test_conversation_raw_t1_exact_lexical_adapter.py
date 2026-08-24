"""T1-G26：exact candidate→G1 lexical evidence 适配边界测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    bind_raw_lexical_evidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    merge_raw_t1_annotation_submissions,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    ANNOTATION_REJECT,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
)
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    audit_raw_t1_candidate_granularity,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    project_raw_t1_consensus_candidate_evidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_exact_lexical_adapter import (
    ADAPTER_ACCEPTED,
    ADAPTER_GRANULARITY_MISMATCH,
    ADAPTER_REJECTED_NEGATIVE,
    RawT1ExactLexicalAdapterError,
    adapt_exact_candidate_to_lexical_evidence,
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


def _project(observation, extraction, decision: int):
    candidate = extraction.candidates[0]

    def submission(scope: str):
        item = RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, decision,
            "sentence_candidate" if decision == ANNOTATION_ACCEPT else "negative_candidate", 1)
        return RawT1AnnotationSubmission(
            f"ann-{scope}", scope, observation.observation_id,
            observation.source_namespace, (item,))

    consensus = merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))
    evidence = project_raw_t1_consensus_candidate_evidence(
        consensus, observation, evidence_namespace="g26-public")[0]
    audit = audit_raw_t1_candidate_granularity(evidence, observation)
    return evidence, audit


def test_exact_accept_projects_to_bindable_g1_lexical_evidence() -> None:
    raw = tuple("甲。".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g26-exact", source_id="src-g26-exact",
        context_id="ctx-g26-exact", family_id="fam-g26-exact",
        source_namespace="t1-g26-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, 2, 0, len(raw)),))
    extraction = extract_raw_text_candidate_spans(raw)
    evidence, audit = _project(observation, extraction, ANNOTATION_ACCEPT)
    result = adapt_exact_candidate_to_lexical_evidence(
        evidence, audit, observation, authority="independent-consensus-v1")

    assert result.status == ADAPTER_ACCEPTED
    assert result.lexical_evidence is not None
    assert bind_raw_lexical_evidence(observation, result.lexical_evidence).unit_id == "u1"


def test_reject_is_negative_and_multi_unit_candidate_is_not_downcast() -> None:
    exact_raw = tuple("甲。".encode("utf-8"))
    exact_observation = compile_raw_text_observation(
        exact_raw, observation_id="obs-g26-reject-exact", source_id="src-g26-r",
        context_id="ctx-g26-r", family_id="fam-g26-r",
        source_namespace="t1-g26-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, 2, 0, len(exact_raw)),))
    exact_extraction = extract_raw_text_candidate_spans(exact_raw)
    rejected_exact, rejected_exact_audit = _project(
        exact_observation, exact_extraction, ANNOTATION_REJECT)
    exact_negative = adapt_exact_candidate_to_lexical_evidence(
        rejected_exact, rejected_exact_audit, exact_observation,
        authority="independent-consensus-v1")
    assert exact_negative.status == ADAPTER_REJECTED_NEGATIVE

    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    observation = observations[0]
    extraction = extract_raw_text_candidate_spans(observation.raw_bytes)
    rejected, rejected_audit = _project(observation, extraction, ANNOTATION_REJECT)
    negative = adapt_exact_candidate_to_lexical_evidence(
        rejected, rejected_audit, observation, authority="independent-consensus-v1")
    assert negative.status == ADAPTER_GRANULARITY_MISMATCH
    # This public candidate covers three units, so even ACCEPT cannot be downcast.
    accepted, accepted_audit = _project(observation, extraction, ANNOTATION_ACCEPT)
    result = adapt_exact_candidate_to_lexical_evidence(
        accepted, accepted_audit, observation, authority="independent-consensus-v1")
    assert result.status == ADAPTER_GRANULARITY_MISMATCH


def test_identity_drift_is_rejected() -> None:
    raw = tuple("甲。".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g26-id", source_id="src-g26-id",
        context_id="ctx-g26-id", family_id="fam-g26-id",
        source_namespace="t1-g26-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, 2, 0, len(raw)),))
    extraction = extract_raw_text_candidate_spans(raw)
    evidence, audit = _project(observation, extraction, ANNOTATION_ACCEPT)
    other = load_raw_text_observation_jsonl(_OBS.read_bytes())[1]
    with pytest.raises(RawT1ExactLexicalAdapterError, match="identity"):
        adapt_exact_candidate_to_lexical_evidence(
            evidence, audit, other, authority="independent-consensus-v1")
