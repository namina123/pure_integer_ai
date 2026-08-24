"""训练表层消费者的独立新组合评估。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_trained_surface_heldout import (
    BASELINE_NO_LEARNED_SURFACE,
    HELDOUT_NE,
    HELDOUT_PASS,
    HeldOutSurfaceEvaluationError,
    build_public_heldout_cases,
    run_trained_surface_heldout,
    validate_cases_independent,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)


_ROOT = Path(__file__).resolve().parents[1]
_RUN_ROOT = Path(
    "K:/pure_integer_ai_work/dialogue_training_week_v1/"
    "dialogue-pack-v6-clean-surface")
_PACK_SHA = "1c907caac90c6edb687ad45e0db490da9188028374d90757af8fc28b720ce03d"


def _runtime_or_skip():
    if not _RUN_ROOT.is_dir():
        pytest.skip("K: dialogue training run is not present in this environment")
    return load_trained_surface_runtime(
        project_root=_ROOT,
        training_run_root=_RUN_ROOT,
        expected_pack_sha256=_PACK_SHA,
    )


def test_public_cases_are_independent_and_cover_all_four_response_paths() -> None:
    cases = build_public_heldout_cases()
    assert len(cases) == 10
    assert len({item.case_id for item in cases}) == 10
    assert {item.response_act for item in cases} == {
        "ANSWER", "UNKNOWN", "CLARIFY", "REPAIR"}
    assert all(item.is_long for item in cases)
    assert validate_cases_independent(_ROOT, cases) == cases


def test_trained_consumer_passes_new_entities_qualifiers_and_combinations() -> None:
    runtime = _runtime_or_skip()
    report = run_trained_surface_heldout(runtime, _ROOT)

    assert report.status == HELDOUT_PASS
    assert report.total_cases == 10
    assert report.passed_cases == 10
    assert report.failed_cases == 0
    assert report.long_cases == 10
    assert report.baseline_no_consumer_cases == 10
    assert report.ready == 0
    assert all(item.baseline_status == BASELINE_NO_LEARNED_SURFACE
               for item in report.observations)
    assert all(item.trained_used and item.trained_status == HELDOUT_PASS
               for item in report.observations)


def test_empty_development_evaluation_is_ne_not_ready() -> None:
    runtime = _runtime_or_skip()
    report = run_trained_surface_heldout(runtime, _ROOT, ())
    assert report.status == HELDOUT_NE
    assert report.total_cases == 0
    assert report.ready == 0


def test_course_surface_cannot_be_relabelled_as_heldout() -> None:
    case = build_public_heldout_cases()[0]
    leaked = replace(case, expected_surface="暴雨导致河水上涨。")
    with pytest.raises(HeldOutSurfaceEvaluationError, match="expected surface 泄漏"):
        validate_cases_independent(_ROOT, (leaked,))
