"""T1-G27：exact candidate→G1/G2/G3 admission probe。"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
    RawRelationArgument,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    merge_raw_t1_annotation_submissions,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
)
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    audit_raw_t1_candidate_granularity,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    project_raw_t1_consensus_candidate_evidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_exact_relation_admission import (
    EXACT_RELATION_ADMISSION_ACCEPTED,
    RawT1ExactRelationAdmissionError,
    admit_exact_candidate_relation,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextSpanUnit,
    compile_raw_text_observation,
)


def _fixture():
    raw = tuple("甲。".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g27-exact", source_id="src-g27-exact",
        context_id="ctx-g27-exact", family_id="fam-g27-exact",
        source_namespace="t1-g27-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, 2, 0, len(raw)),))
    extraction = extract_raw_text_candidate_spans(raw)
    candidate = extraction.candidates[0]

    def submission(scope: str):
        decision = RawT1AnnotationDecision(
            candidate.ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, ANNOTATION_ACCEPT,
            "sentence", 1)
        return RawT1AnnotationSubmission(
            f"ann-{scope}", scope, observation.observation_id,
            observation.source_namespace, (decision,))

    consensus = merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))
    candidate_evidence = project_raw_t1_consensus_candidate_evidence(
        consensus, observation, evidence_namespace="g27-public")
    audits = tuple(audit_raw_t1_candidate_granularity(item, observation)
                   for item in candidate_evidence)
    evidence_id = candidate_evidence[0].evidence_id
    proposition = RawPropositionRelationEvidence(
        "p-g27-exact", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id, observation.source_namespace,
        observation.split, "annotated_sentence", "independent-proposition-v1",
        (RawRelationArgument(evidence_id, "u1", "sentence", 1),))
    qualification = RawPropositionQualification(
        "q-g27-exact", proposition.proposition_id, observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "SUPPORTED",
        "two-reviewer-consensus-v1", (evidence_id,), "independent-qualification-v1")
    return observation, candidate_evidence, audits, proposition, qualification


def test_exact_candidate_closes_g1_g2_g3_and_preserves_candidate_trace() -> None:
    values = _fixture()
    result = admit_exact_candidate_relation(*values, authority="g26-adapter-v1")

    assert result.status == EXACT_RELATION_ADMISSION_ACCEPTED
    assert result.consumer.state == "SUPPORTED"
    assert result.consumer.response_act == "ANSWER"
    assert result.candidate_evidence_ids == ("g27-public:obs-g27-exact:0",)
    assert result.lexical_evidence[0].unit_id == "u1"
    assert all(type(value) is int and value >= 0 for value in result.canonical_record())


def test_wrong_proposition_evidence_chain_cannot_be_hidden_by_adapter() -> None:
    observation, candidate_evidence, audits, proposition, qualification = _fixture()
    changed = RawPropositionQualification(
        qualification.qualification_id, qualification.proposition_id,
        qualification.observation_id, qualification.source_id,
        qualification.context_id, qualification.family_id,
        qualification.source_namespace, qualification.split, qualification.state,
        qualification.reason_id, ("unknown-evidence",), qualification.authority)
    with pytest.raises(RawT1ExactRelationAdmissionError, match="闭合失败"):
        admit_exact_candidate_relation(
            observation, candidate_evidence, audits, proposition, changed,
            authority="g26-adapter-v1")
