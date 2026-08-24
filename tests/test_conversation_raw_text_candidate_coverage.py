"""T1-G17：候选与显式 unit 的物理覆盖审计。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_text_candidate_coverage import (
    RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE,
    RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY,
    RawTextCandidateCoverageError,
    audit_raw_text_candidate_coverage,
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
_SAMPLE = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"


def test_public_train_and_heldout_units_are_physically_covered() -> None:
    observations = load_raw_text_observation_jsonl(_SAMPLE.read_bytes())
    for observation in observations[:2]:
        result = audit_raw_text_candidate_coverage(
            extract_raw_text_candidate_spans(observation.raw_bytes), observation)
        assert result.status == RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE
        assert result.complete is True
        assert result.covered_unit_count == len(observation.units)
        assert result.canonical_record()


def test_unit_crossing_sentence_boundary_is_not_silently_covered() -> None:
    raw = tuple("甲。乙".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-cross", source_id="src-cross",
        context_id="ctx-cross", family_id="fam-cross",
        source_namespace="t1-g17-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "opaque", 0, 3, 0, len(raw)),))
    result = audit_raw_text_candidate_coverage(
        extract_raw_text_candidate_spans(raw), observation)
    assert result.status == RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY
    assert result.crossing_unit_count == 1
    assert result.covered_unit_count == 0


def test_candidate_coverage_rejects_raw_identity_drift() -> None:
    observations = load_raw_text_observation_jsonl(_SAMPLE.read_bytes())
    extraction = extract_raw_text_candidate_spans(observations[0].raw_bytes)
    with pytest.raises(RawTextCandidateCoverageError, match="raw input 不一致"):
        audit_raw_text_candidate_coverage(extraction, observations[1])
