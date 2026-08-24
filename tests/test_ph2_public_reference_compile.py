"""DLG-RAW-05B 无标签 public reference connector 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    PublicGroundedAnswerReferenceClaimSurface,
    PublicGroundedAnswerReferenceCompileRequest,
    GroundedAnswerReferenceCompileError,
    compile_public_grounded_answer_reference_connector,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_public_reference_planning,
    public_response_act_planning_input_from_episode,
)


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BASE = 65001


def _scalars(value: str) -> tuple[int, ...]:
    """测试边界把已知 public observation 文本显式写成 scalar tuple。"""
    return tuple(ord(item) for item in value)


def _surface_protocol() -> GenerationSurfaceProtocol:
    """建立本专项独立的 G-03 surface protocol，不复用训练 fixture。"""
    return GenerationSurfaceProtocol(*tuple(
        minimal_instruction_identity((_BASE, 70, index))
        for index in range(1, 10)
    ))


def _public_build_and_claims():
    """从第五条公开课程形成不携带 episode 的 V3 planning/scalar input。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.episode_id == "train-grounded-reference-event-v2")
    original_plan = episode.question.answer_plan
    object.__setattr__(episode.question, "answer_plan", object())
    try:
        planning_input = public_response_act_planning_input_from_episode(
            episode)
    finally:
        object.__setattr__(episode.question, "answer_plan", original_plan)
    order = ("p-year", "p-registration")
    build = compile_public_reference_planning(
        planning_input,
        language_branch_identity((_BASE, 71)),
        order,
    )
    by_proposition = {
        item.proposition_id: item.claim_text for item in planning_input.evidence
    }
    claims = tuple(PublicGroundedAnswerReferenceClaimSurface(
        proposition_id,
        _scalars(by_proposition[proposition_id]),
    ) for proposition_id in order)
    evidence_keys = tuple(sorted({
        evidence.stable_key()
        for candidate in build.planning.candidates
        for evidence in candidate.evidence
    }))
    return build, claims, evidence_keys


def test_public_reference_compiler_uses_scalar_input_without_episode_or_labels() -> None:
    """两个策略来自同一无标签 planning，且 anaphora 仅由结构 option 决定。"""
    build, claims, evidence_keys = _public_build_and_claims()
    protocol = _surface_protocol()
    request = PublicGroundedAnswerReferenceCompileRequest(
        build,
        claims,
        (_BASE, 72),
        "ANTECEDENT_REFERENCE",
        _scalars("前述"),
        _scalars("北川站东门的"),
        evidence_keys,
    )
    antecedent = compile_public_grounded_answer_reference_connector(
        request,
        protocol,
    )
    explicit = compile_public_grounded_answer_reference_connector(
        PublicGroundedAnswerReferenceCompileRequest(
            build,
            claims,
            (_BASE, 72),
            "EXPLICIT_REPETITION",
            _scalars("前述"),
            _scalars("北川站东门的"),
            evidence_keys,
        ),
        protocol,
    )

    assert not hasattr(request, "episode")
    assert antecedent.planning == build.planning
    assert tuple(item.proposition_id for item in antecedent.claims) == (
        "p-year", "p-registration")
    assert antecedent.connector.anaphora_declarations is not None
    assert explicit.connector.anaphora_declarations is None
    antecedent_reference = next(
        item for item in antecedent.sentences[1].aliases
        if item.filler == antecedent.reference_origin)
    explicit_reference = next(
        item for item in explicit.sentences[1].aliases
        if item.filler == explicit.reference_origin)
    assert representation_parts(antecedent_reference.representation)[1] == _scalars("前述")
    assert representation_parts(explicit_reference.representation)[1] == _scalars("北川站东门的")


def test_public_reference_compiler_rejects_forged_claim_scalar() -> None:
    """V3 connector 只能接受当前无标签 planning Evidence 重派生的 claim scalar。"""
    build, claims, evidence_keys = _public_build_and_claims()
    forged = (replace(claims[0], scalars=_scalars("伪造的启用信息")), claims[1])

    with pytest.raises(
            GroundedAnswerReferenceCompileError,
            match="claim scalar 不等于来源 Evidence"):
        PublicGroundedAnswerReferenceCompileRequest(
            build,
            forged,
            (_BASE, 72),
            "ANTECEDENT_REFERENCE",
            _scalars("前述"),
            _scalars("北川站东门的"),
            evidence_keys,
        )
