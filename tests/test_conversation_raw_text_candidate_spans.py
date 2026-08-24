"""T1-G16：机械句界候选的物理边界测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_UTF8,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL,
    RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL,
    RawTextCandidateError,
    extract_raw_text_candidate_spans,
)


def _surface(item, scalars: tuple[int, ...]) -> str:
    return "".join(chr(value) for value in scalars[item.start_scalar:item.end_scalar])


def test_sentence_boundaries_are_mechanical_and_non_overlapping() -> None:
    raw = tuple("暴雨导致河水上涨。风大？！北川站".encode("utf-8"))
    result = extract_raw_text_candidate_spans(raw)

    assert result.accepted is True
    assert [_surface(item, result.intake.unicode_scalars) for item in result.candidates] == [
        "暴雨导致河水上涨。", "风大？！", "北川站"]
    assert [item.boundary_kind for item in result.candidates] == [
        RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL,
        RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL,
        RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL,
    ]
    assert result.candidates[0].end_scalar == result.candidates[1].start_scalar
    assert result.candidates[1].end_scalar == result.candidates[2].start_scalar
    assert all(type(value) is int and value >= 0 for value in result.canonical_record())


def test_scalar_and_byte_spans_reconstruct_exact_utf8() -> None:
    raw = tuple("台风导致港口封闭。".encode("utf-8"))
    result = extract_raw_text_candidate_spans(raw)
    candidate = result.candidates[0]

    assert tuple(result.intake.unicode_scalars[
        candidate.start_scalar:candidate.end_scalar]) == tuple(map(ord, "台风导致港口封闭。"))
    assert result.intake.raw_input_bytes[candidate.start_byte:candidate.end_byte] == raw


@pytest.mark.parametrize("raw", [(0xE0, 0x80, 0x80), (0x61, 0x0A, 0x62)])
def test_rejected_raw_input_never_materializes_candidates(raw: tuple[int, ...]) -> None:
    result = extract_raw_text_candidate_spans(raw)
    assert result.candidates == ()
    assert result.accepted is False
    if raw[0] == 0xE0:
        assert result.intake.result_code == DLG_RAW_REJECT_UTF8


def test_candidate_span_rejects_unknown_boundary_kind() -> None:
    from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import RawTextCandidateSpan
    with pytest.raises(RawTextCandidateError, match="boundary_kind"):
        RawTextCandidateSpan(0, 0, 1, 0, 1, 99)
