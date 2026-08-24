"""T1-G19：候选进入后续训练审核前的只读资格闸门。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_promotion_audit import (
    PROMOTION_ELIGIBLE_FOR_REVIEW,
    PROMOTION_NEGATIVE_WITNESS_ONLY,
    RawTextCandidatePromotionError,
    audit_raw_text_candidate_promotion,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"
_LEX = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"


def test_train_and_heldout_are_eligible_but_negative_stays_witness() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    evidence = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    audits = tuple(
        audit_raw_text_candidate_promotion(
            extract_raw_text_candidate_spans(item.raw_bytes), item,
            tuple(value for value in evidence
                  if value.observation_id == item.observation_id))
        for item in observations)

    assert tuple(item.status for item in audits) == (
        PROMOTION_ELIGIBLE_FOR_REVIEW,
        PROMOTION_ELIGIBLE_FOR_REVIEW,
        PROMOTION_NEGATIVE_WITNESS_ONLY,
    )
    assert audits[0].eligible_for_review is True
    assert audits[2].eligible_for_review is False
    assert all(type(value) is int and value >= 0
               for audit in audits for value in audit.canonical_record())


def test_promotion_audit_is_not_a_training_writer() -> None:
    observations = load_raw_text_observation_jsonl(_OBS.read_bytes())
    evidence = load_raw_lexical_evidence_jsonl(_LEX.read_bytes())
    audit = audit_raw_text_candidate_promotion(
        extract_raw_text_candidate_spans(observations[0].raw_bytes), observations[0],
        tuple(value for value in evidence
              if value.observation_id == observations[0].observation_id))
    assert not hasattr(audit, "promoted_case")
    with pytest.raises(TypeError, match="RawTextCandidateExtraction"):
        audit_raw_text_candidate_promotion(object(), observations[0], evidence)
