from __future__ import annotations

import pytest

from pure_integer_ai.experiments.text_segments import sentence_bounds


def test_sentence_bounds_without_evidence_keeps_whole_segment() -> None:
    assert sentence_bounds(5) == [(0, 5)]


def test_sentence_bounds_uses_only_injected_cut_positions() -> None:
    assert sentence_bounds(7, cut_after=[5, 2, 5]) == [(0, 2), (2, 5), (5, 7)]


def test_sentence_bounds_empty_input_has_no_span() -> None:
    assert sentence_bounds(0, cut_after=[]) == []


@pytest.mark.parametrize("cut", [0, -1, 4])
def test_sentence_bounds_rejects_out_of_range_cut(cut: int) -> None:
    with pytest.raises(ValueError):
        sentence_bounds(3, cut_after=[cut])


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_sentence_bounds_rejects_non_integer_cut(value: object) -> None:
    with pytest.raises(TypeError):
        sentence_bounds(3, cut_after=[value])
