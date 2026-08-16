from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerClaimCandidateBinding,
    GroundedAnswerReferenceCompileRequest,
    compile_grounded_answer_reference_connector,
)

from tests.test_g02_generation_structure_plan import (
    _request,
    _selection,
)
from tests.test_g03_generation_surface import (
    _surface_protocol,
)


SAMPLE_PATH = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BASE = 20970


def test_reference_course_compiles_to_two_sentences_and_one_anaphora():
    """第五条 grounded 课程形成逐 Proposition 句和前序 antecedent。"""
    episode = read_grounded_answer_episodes(SAMPLE_PATH)[-1]
    request, _unused = _request(count=2)
    branch = language_branch_identity((_BASE, 1))
    planning = GenerationPlanningRequest(
        replace(request.goal, target_branch=branch),
        request.candidates,
    )
    selection, _unused_first, _unused_second = _selection(planning)
    claims = tuple(
        GroundedAnswerClaimCandidateBinding(proposition_id, candidate)
        for proposition_id, candidate in zip(
            episode.question.answer_plan.ordered_claim_ids,
            planning.candidates,
            strict=True,
        )
    )
    compilation = compile_grounded_answer_reference_connector(
        GroundedAnswerReferenceCompileRequest(
            episode,
            planning,
            claims,
            branch,
            (_BASE, 2),
            "ANTECEDENT_REFERENCE",
        ),
        _surface_protocol(_BASE + 1),
    )

    plan = compilation.connector.structure_planner().plan(selection)

    assert len(compilation.sentences) == 2
    assert len(compilation.connector.registry.templates) == 2
    assert len(plan.syntax.sentences) == 2
    assert len(plan.syntax.anaphora) == 1
    assert tuple(
        sentence.instance.candidate_key
        for sentence in plan.syntax.sentences) == tuple(
            candidate.stable_key() for candidate in planning.candidates)
    requirement = plan.syntax.anaphora[0]
    assert requirement.address == plan.syntax.sentences[1].address
    assert requirement.slot == compilation.reference_slot
    assert requirement.antecedent_candidate_key == (
        planning.candidates[0].stable_key())
    assert compilation.connector.anaphora_declarations is not None
