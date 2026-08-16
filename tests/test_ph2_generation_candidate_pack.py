"""E-05 production generation candidate pack 与 R-01 owner 聚焦专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.ph2_generation_candidate_alias_runtime import (
    ProductionGenerationAliasRuntimeFactory,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    REPRESENTATION_RULES,
    build_generation_candidate_pack,
    publish_generation_candidate_pack,
    read_generation_candidate_pack,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g03_generation_surface import _surface_protocol
from tests.test_ph2_grounded_answer_course import (
    _connector_question_and_candidate,
)
from tests.test_ph2_grounded_answer_reference_compile import (
    _reference_selection,
)


_BASE = 22030
_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def _assert_realizations(runtime, branch, aliases) -> None:
    """逐条证明 production R-01 owner 唯一恢复请求的 Representation。"""
    for alias in aliases:
        proposal = runtime.preview_surface(
            alias.filler,
            branch,
            budget=AliasRouteSearchBudget(64, 64, 64),
            allowed_prefix_steps=(),
        )
        assert proposal.result.selected is not None
        assert proposal.result.selected.value == alias.representation


def test_candidate_pack_roundtrips_and_rebuilds_all_alias_inputs(tmp_path):
    """pack 双发布一致，三类输入可装配，answer fresh/resume/clone 状态相同。"""
    model, question, planning, candidate, branch = (
        _connector_question_and_candidate())
    training_sha = hashlib.sha256(_SAMPLE.read_bytes()).hexdigest()
    pack = build_generation_candidate_pack(model, training_sha)
    first = publish_generation_candidate_pack(pack, tmp_path / "first")
    second = publish_generation_candidate_pack(pack, tmp_path / "second")
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert b'"surfaces"' not in first.manifest_path.read_bytes()
    assert b'"accepted"' not in first.manifest_path.read_bytes()
    assert b'"rejected"' not in first.manifest_path.read_bytes()

    loaded = read_generation_candidate_pack(
        first.pack_root, expected_sha256=pack.sha256())
    assert loaded.pack == pack
    assert loaded.pack.model == model
    assert loaded.pack.representation_rules == REPRESENTATION_RULES

    surface_protocol = _surface_protocol(_BASE + 1)
    answer_compilation = compile_grounded_answer_connectors(
        loaded.pack.model,
        question,
        GroundedAnswerConnectorTarget(
            candidate.proposition, branch, (_BASE, 2)),
        surface_protocol,
    )
    answer_variant = answer_compilation.variants[0]
    visible_keys = (candidate.stable_key(),)
    backend = DictBackend()
    cloned_backend = None
    try:
        ctx = make_train_context(backend)
        fresh_factory = ProductionGenerationAliasRuntimeFactory(
            loaded.pack, ctx, visible_evidence_keys=visible_keys)
        fresh = fresh_factory.build(answer_variant)
        fresh_snapshot = backend.snapshot()
        _assert_realizations(fresh, branch, answer_variant.aliases)

        resume_factory = ProductionGenerationAliasRuntimeFactory(
            loaded.pack, ctx, visible_evidence_keys=visible_keys)
        resumed = resume_factory.build(answer_variant)
        assert backend.snapshot() == fresh_snapshot
        assert resumed.state_key() == fresh.state_key()

        cloned_backend = clone_backend(backend)
        clone_ctx = make_train_context(cloned_backend)
        evaluation_factory = fresh_factory.clone_for_evaluation(clone_ctx)
        evaluation = evaluation_factory.build(answer_variant)
        assert evaluation is not fresh
        assert evaluation.closure.use_owner is not fresh.closure.use_owner
        assert evaluation.state_key() == fresh.state_key()
        _assert_realizations(evaluation, branch, answer_variant.aliases)
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()

    response_branch = language_branch_identity((_BASE, 3))
    response_compilation = compile_grounded_response_act_patterns(
        loaded.pack.model,
        GroundedResponseActCompileTarget(
            "CLARIFY",
            minimal_instruction_identity((_BASE, 4)),
            response_branch,
            (_BASE, 5),
        ),
    )
    response_variant = response_compilation.variants[0]
    response_backend = DictBackend()
    try:
        response = ProductionGenerationAliasRuntimeFactory(
            loaded.pack, make_train_context(response_backend)).build(
                response_variant)
        proposal = response.preview_surface(
            response_variant.template.stance,
            response_branch,
            budget=response_variant.surface_budget,
            allowed_prefix_steps=(),
        )
        assert proposal.result.selected is not None
        assert proposal.result.selected.value == response_variant.representation
    finally:
        response_backend.close()

    _episode, _planning, reference_branch, selection = _reference_selection()
    reference_compilation = selection.compilation
    reference_aliases = tuple(
        alias
        for sentence in reference_compilation.sentences
        for alias in sentence.aliases
    )
    reference_backend = DictBackend()
    try:
        reference = ProductionGenerationAliasRuntimeFactory(
            loaded.pack, make_train_context(reference_backend)).build(
                reference_compilation)
        _assert_realizations(
            reference, reference_branch, reference_aliases)
        proposal = reference.preview_reference(
            reference_compilation.reference_origin,
            target_kinds=(OBJECT_PROPOSITION,),
            budget=AliasRouteSearchBudget(64, 64, 64),
        )
        assert proposal.result.selected is not None
        assert proposal.result.selected.value == (
            reference_compilation.claims[0].candidate.proposition.template)
    finally:
        reference_backend.close()
