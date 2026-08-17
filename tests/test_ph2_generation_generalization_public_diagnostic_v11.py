"""GG-03 V11 public diagnostic inventory tests."""
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_public_diagnostic_v11 import (
    PUBLIC_V11_DIAGNOSTIC_CASE_IDS,
    build_generation_generalization_public_v11_diagnostic_observations,
)


_SOURCE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_v11_diagnostic_is_label_free_and_isolates_expanded_boundaries() -> None:
    """诊断只扩展候选/source 组合，不携带 evaluator label。"""
    observations = build_generation_generalization_public_v11_diagnostic_observations(
        _SOURCE)
    assert len(observations) == 8
    assert {item.episode_id for item in observations} == set(
        PUBLIC_V11_DIAGNOSTIC_CASE_IDS)
    assert all(len(item.question.evidence) == 4 for item in observations)
    assert all(len(item.dialogue.turns) == 3 for item in observations)
    assert all(item.question.answer_plan.response_act in {
        "CLARIFY", "CONFLICT"} for item in observations)
    assert all(item.reference_course is None for item in observations)
