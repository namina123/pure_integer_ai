"""E-05 label-free executable Observation 与 reference compiler 防泄漏专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import language_branch_identity
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_answer_verification import (
    GenerationGeneralizationAnswerVerificationInput,
    run_generation_generalization_answer_verification,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    default_answer_verification_protocol,
    default_source_conflict_protocol,
)
from pure_integer_ai.experiments.ph2_generation_generalization_source_conflict import (
    GenerationGeneralizationSourceConflictInput,
    run_generation_generalization_source_conflict_verification,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
    GroundedAnswerSplitClusters,
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerClaimCandidateBinding,
    GroundedAnswerReferenceCompileRequest,
    compile_grounded_answer_reference_connector,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningError,
    compile_grounded_answer_reference_planning,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)

from tests.test_g03_generation_surface import _surface_protocol
from tests.test_ph2_grounded_answer_connector_runtime import (
    _rehearse_answer_requirement,
)
from tests.test_ph2_grounded_answer_response_act_runtime import (
    _rehearse_response_act,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BASE = 21140


def test_label_free_observation_compiles_and_full_held_out_episode_is_rejected():
    """候选只读无 surface Observation；完整 held-out episode 不能进 planning。"""
    training = read_grounded_answer_episodes(_SAMPLE)[-1]
    held_out = replace(
        training,
        episode_id="heldout-grounded-reference-event-v1",
        split="held_out",
        clusters=GroundedAnswerSplitClusters(
            "heldout-station-records-s1",
            "heldout-east-gate-event-chain-p1",
            "heldout-event-and-record-q1",
            "heldout-reference-event-p1",
        ),
    )
    budget = GenerationGeneralizationEvaluationBudget(
        512, 4, 4, 96, 16)
    observation = (
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            held_out, budget))
    payload = canonical_json_line(observation.to_dict())
    assert all(marker not in payload for marker in (
        b'"surfaces"', b'"accepted"', b'"rejected"',
        b'"minimum_legal_surfaces"', b'"challenges"',
    ))
    restored = GenerationGeneralizationEvaluationObservation.from_dict(
        parse_canonical_json_bytes(payload[:-1], require_object=True))
    assert restored == observation

    branch = language_branch_identity((_BASE, 1))
    with pytest.raises(
            GroundedResponseActPlanningError,
            match="surface label"):
        compile_grounded_answer_reference_planning(held_out, branch)

    planning = compile_grounded_answer_reference_planning(
        observation, branch)
    claims = tuple(
        GroundedAnswerClaimCandidateBinding(
            proposition_id,
            planning.candidate_for(proposition_id),
        )
        for proposition_id in observation.reference_course.ordered_proposition_ids
    )
    compilations = tuple(
        compile_grounded_answer_reference_connector(
            GroundedAnswerReferenceCompileRequest(
                observation,
                planning.planning,
                claims,
                branch,
                (_BASE, 2),
                strategy,
                (observation.stable_key(),),
            ),
            _surface_protocol(_BASE + 1),
        )
        for strategy in REFERENCE_STRATEGIES
    )
    assert tuple(item.strategy for item in compilations) == (
        REFERENCE_STRATEGIES)
    assert tuple(
        observation.reference_course.surface_for(strategy)
        for strategy in REFERENCE_STRATEGIES) == ("前述", "北川站东门的")
    assert all(len(item.sentences) == 2 for item in compilations)

    _course, answer_planning, answer, _run = _rehearse_answer_requirement(
        "INDEPENDENT_UNDERSTANDING_READBACK", 31)
    answer_observation = (
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            replace(answer.source_episode, split="held_out"), budget))
    evaluation_answer_planning = replace(
        answer_planning, episode=answer_observation)
    answer_input = GenerationGeneralizationAnswerVerificationInput(
        "INDEPENDENT_UNDERSTANDING_READBACK",
        answer_observation,
        evaluation_answer_planning,
        answer.choice,
        answer.use,
        answer.execution,
        answer.parse_request,
        answer.postcheck,
    )
    answer_report = run_generation_generalization_answer_verification(
        default_answer_verification_protocol(), answer_input)
    assert answer_report.results[0].verdict == VERDICT_SUPPORT
    assert answer_input.stable_key()[1:len(
        answer_observation.stable_key()) + 2] == (
            len(answer_observation.stable_key()),
            *answer_observation.stable_key(),
        )

    _course, conflict_planning, conflict, _run = _rehearse_response_act(
        "SOURCE_UNCERTAINTY_CITATION", "CONFLICT", 32)
    conflict_observation = (
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            replace(conflict.source_episode, split="held_out"), budget))
    evaluation_conflict_planning = replace(
        conflict_planning, episode=conflict_observation)
    conflict_input = GenerationGeneralizationSourceConflictInput(
        conflict_observation,
        evaluation_conflict_planning,
        conflict.choice,
        conflict.use,
        conflict.execution,
        conflict.parse_request,
        conflict.postcheck,
    )
    conflict_report = (
        run_generation_generalization_source_conflict_verification(
            default_source_conflict_protocol(), conflict_input))
    assert conflict_report.results[0].verdict == VERDICT_SUPPORT
    assert conflict_input.stable_key()[:len(
        conflict_observation.stable_key()) + 1] == (
            len(conflict_observation.stable_key()),
            *conflict_observation.stable_key(),
        )
