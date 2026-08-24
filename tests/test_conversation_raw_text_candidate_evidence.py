"""T1-G18：候选与 lexical evidence 的物理对齐测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_evidence import (
    RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE,
    RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY,
    RawTextCandidateEvidenceError,
    audit_raw_text_candidate_lexical_evidence,
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
_LEX = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"


def test_public_lexical_evidence_is_covered_by_mechanical_sentence_candidates() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    evidence = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    for observation in observations:
        selected = tuple(item for item in evidence
                         if item.observation_id == observation.observation_id)
        result = audit_raw_text_candidate_lexical_evidence(
            extract_raw_text_candidate_spans(observation.raw_bytes),
            observation, selected)
        assert result.status == RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE
        assert result.complete is True
        assert result.covered_evidence_count == len(selected)
        assert all(type(value) is int and value >= 0
                   for value in result.canonical_record())


def test_lexical_evidence_crossing_sentence_boundary_is_not_accepted() -> None:
    raw = tuple("甲。乙".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-e-cross", source_id="src-e-cross",
        context_id="ctx-e-cross", family_id="fam-e-cross",
        source_namespace="t1-g18-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "opaque", 0, 3, 0, len(raw)),))
    from pure_integer_ai.experiments.conversation_raw_lexical_evidence import RawLexicalEvidence
    evidence = RawLexicalEvidence(
        "e-cross", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id, observation.source_namespace,
        observation.split, "u1", "explicit_span", "public-test-v1",
        0, 3, 0, len(raw))
    result = audit_raw_text_candidate_lexical_evidence(
        extract_raw_text_candidate_spans(raw), observation, (evidence,))
    assert result.status == RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY
    assert result.crossing_evidence_count == 1


def test_evidence_identity_drift_fails_closed() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    evidence = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    extraction = extract_raw_text_candidate_spans(observations[0].raw_bytes)
    with pytest.raises(RawTextCandidateEvidenceError, match="物理绑定失败"):
        audit_raw_text_candidate_lexical_evidence(
            extraction, observations[0], (evidence[3],))


def test_empty_evidence_cannot_be_reported_as_complete() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    extraction = extract_raw_text_candidate_spans(observations[0].raw_bytes)
    with pytest.raises(RawTextCandidateEvidenceError, match="evidence 不能为空"):
        audit_raw_text_candidate_lexical_evidence(extraction, observations[0], ())
