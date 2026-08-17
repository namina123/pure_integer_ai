"""Owner-independent GG-03 semantic-label contract tests."""
from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    GenerationGeneralizationSemanticProjection,
    build_expected_generation_generalization_semantic_projection,
    build_generation_generalization_semantic_label_record,
    generation_generalization_semantic_verdict_contract_sha256,
    semantic_projection_from_realization,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BUDGET = GenerationGeneralizationEvaluationBudget(512, 4, 4, 96, 16)


def _observation(episode):
    return GenerationGeneralizationEvaluationObservation.from_held_out_episode(
        replace(
            episode,
            episode_id=f"semantic-{episode.episode_id}",
            split="held_out",
        ),
        _BUDGET,
    )


def test_legal_paraphrases_share_one_semantic_commitment() -> None:
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act == "ANSWER"
        and item.reference_course is None)
    observation = _observation(episode)
    projections = tuple(
        semantic_projection_from_realization(item)
        for item in episode.surfaces.accepted)
    expected = build_expected_generation_generalization_semantic_projection(
        observation)
    assert len({item.sha256() for item in projections}) == 1
    assert projections[0] == expected
    assert GenerationGeneralizationSemanticProjection.from_dict(
        expected.to_dict()) == expected

    label = build_generation_generalization_semantic_label_record(observation)
    assert type(label).from_dict(label.to_dict()) == label
    assert label.verdict_for_projection(projections[0]) == "PASS"
    assert label.verdict_for_projection(None) == "NE"
    assert len(generation_generalization_semantic_verdict_contract_sha256()) == 64


def test_complete_but_different_meaning_is_fail_not_ne() -> None:
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act == "CLARIFY")
    observation = _observation(episode)
    expected = build_expected_generation_generalization_semantic_projection(
        observation)
    label = build_generation_generalization_semantic_label_record(observation)
    different = GenerationGeneralizationSemanticProjection(
        expected.carrier_kind,
        "CONFLICT",
        expected.scope_id,
        expected.claim_ids,
        expected.cited_source_ids,
    )
    assert different.sha256() != expected.sha256()
    assert label.verdict_for_projection(different) == "FAIL"


def test_reference_claim_order_remains_part_of_meaning() -> None:
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.reference_course is not None)
    observation = _observation(episode)
    expected = build_expected_generation_generalization_semantic_projection(
        observation)
    reversed_claims = replace(
        expected, claim_ids=tuple(reversed(expected.claim_ids)))
    assert reversed_claims.sha256() != expected.sha256()
    assert build_generation_generalization_semantic_label_record(
        observation).verdict_for_projection(reversed_claims) == "FAIL"
