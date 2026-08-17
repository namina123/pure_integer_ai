"""GG-03 V12 public diagnostic inventory tests."""
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_public_diagnostic_v12 import (
    PUBLIC_V12_DIAGNOSTIC_CASE_IDS,
    build_generation_generalization_public_v12_diagnostic_observations,
)


_SOURCE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_v12_diagnostic_is_label_free_and_stays_in_contract() -> None:
    """V12 只变化候选基数和 scope 历史，不携带 evaluator label。"""
    observations = build_generation_generalization_public_v12_diagnostic_observations(
        _SOURCE)
    assert len(observations) == 8
    assert {item.episode_id for item in observations} == set(
        PUBLIC_V12_DIAGNOSTIC_CASE_IDS)
    assert all(len(item.question.evidence) <= 4 for item in observations)
    assert all(len(item.dialogue.turns) in {3, 5} for item in observations)
    assert all(item.question.answer_plan.response_act in {
        "CLARIFY", "CONFLICT"} for item in observations)
    assert all(item.reference_course is None for item in observations)
    assert all(
        len({item.proposition_id for item in observation.question.evidence})
        == (len(observation.question.evidence)
            if observation.question.answer_plan.response_act == "CLARIFY"
            else 1)
        for observation in observations
    )
