"""E-04 assessment-aware actual reference choice 单纵切专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.ph2_generation_choice_assessment_selector import (
    GenerationChoiceAssessmentSelectorPolicy,
    select_generation_choice_by_assessment,
)
from pure_integer_ai.experiments.ph2_generation_generalization_assessment_choice_closure import (
    GenerationGeneralizationAssessmentChoiceClosureError,
    build_generation_generalization_assessment_aware_train_slice,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    apply_grounded_answer_reference_assessment_selection,
    build_grounded_answer_reference_selection,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
)

from tests.test_ph2_grounded_answer_reference_compile import (
    _reference_assessment_runtime,
    _reference_selection,
    _run_reference_strategy,
    _with_reference_verdict,
)


_BASE = 21120


def test_assessment_choice_binds_next_actual_chain_and_disabled_fallback():
    """assessment 改变下一次全链；关闭层不读 projection 并回退 baseline。"""
    antecedent_prepared = _reference_selection("ANTECEDENT_REFERENCE")
    antecedent_selection = antecedent_prepared[3]
    explicit_selection = build_grounded_answer_reference_selection(
        tuple(item.compilation for item in antecedent_selection.options),
        "EXPLICIT_REPETITION",
        (_BASE, 1),
    )
    explicit_prepared = (*antecedent_prepared[:3], explicit_selection)
    baseline_training = _run_reference_strategy(
        "ANTECEDENT_REFERENCE", antecedent_prepared)
    alternative_training = _run_reference_strategy(
        "EXPLICIT_REPETITION", explicit_prepared)
    reference_selections = (
        baseline_training[0], alternative_training[0])
    choices = tuple(item.choice for item in reference_selections)
    baseline = baseline_training[0].choice

    backend, mapper, learning, consumer = _reference_assessment_runtime(_BASE)
    try:
        baseline_update = consumer.apply(_with_reference_verdict(
            baseline_training[4].outcome, VERDICT_REFUTE))
        alternative_update = consumer.apply(
            alternative_training[4].outcome)
        assessment = select_generation_choice_by_assessment(
            mapper,
            learning,
            GenerationChoiceAssessmentSelectorPolicy((_BASE, 2)),
            choices,
            baseline,
        )
        assessed_selection = (
            apply_grounded_answer_reference_assessment_selection(
                reference_selections, assessment))
        actual = _run_reference_strategy(
            "EXPLICIT_REPETITION",
            (*explicit_prepared[:3], assessed_selection),
        )

        backend_before = backend.snapshot()
        learning_before = learning.state_key()
        disabled = select_generation_choice_by_assessment(
            mapper,
            learning,
            GenerationChoiceAssessmentSelectorPolicy(
                (_BASE, 3), ("DISCOURSE_REFERENCE_CHOICE",)),
            choices,
            baseline,
        )
        assert backend.snapshot() == backend_before
        assert learning.state_key() == learning_before

        closure = build_generation_generalization_assessment_aware_train_slice(
            baseline_training[4],
            alternative_training[4],
            baseline_update,
            alternative_update,
            assessment,
            actual[4],
            disabled,
        )
        assert closure.status == "PASS_ASSESSMENT_AWARE_ACTUAL_CHOICE"
        assert closure.complete == 1
        assert closure.stable_key()
        assert actual[2] == (
            "北川站东门于2024年启用。北川站东门的启用事项已登记入档。")

        with pytest.raises(
                GenerationGeneralizationAssessmentChoiceClosureError):
            build_generation_generalization_assessment_aware_train_slice(
                baseline_training[4],
                alternative_training[4],
                baseline_update,
                alternative_update,
                assessment,
                baseline_training[4],
                disabled,
            )
    finally:
        backend.close()
