"""W05-03 独立 Understanding/Reasoning/Generation consumer 专项。"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05_IDENTITY_VERSIONS,
    adapt_w05_training_payload,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_STAGE_KEY,
    W05_W04_BASE_RUN_ID,
    W05RunRequest,
    digest_value,
)
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.experiments.ph2_w05_generation import (
    build_w05_generation_runtime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_HARD_CASES,
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_READY,
    W05_GENERATION_REJECTED,
    W05_GENERATION_UNKNOWN,
    W05GenerationCaseResult,
    W05GenerationProtocol,
    run_w05_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_reasoning import (
    W05_REASONING_AUTHORIZED,
    W05_REASONING_CONFLICT,
    W05_REASONING_OUTCOME_SUPPORT,
    W05_REASONING_REJECTED,
    W05_REASONING_SUPERSEDED,
    W05ReasoningProtocol,
    build_w05_reasoning_runtime,
    reasoning_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    W05_UNDERSTANDING_CONFLICT,
    W05_UNDERSTANDING_OUTCOME_SUPPORT,
    W05_UNDERSTANDING_UNIQUE,
    W05_UNDERSTANDING_UNKNOWN,
    W05UnderstandingProtocol,
    build_w05_understanding_runtime,
    understanding_request_for_candidate,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w05_historical_context import open_historical_w05_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"


@pytest.fixture(scope="module")
def consumer_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("w05-consumers")
    probe = SQLiteBackend(str(root / "context.sqlite"))
    try:
        context = open_historical_w05_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=probe.storage_capabilities().stable_key(),
        )
    finally:
        probe.close()
    request = W05RunRequest(
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        stage_key=W05_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(W05_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W05PayloadFirewall.open(
        ROOT, context, request).read_training_payload()
    adapter = adapt_w05_training_payload(payload)
    backend = SQLiteBackend(str(root / "learning.sqlite"))
    learning = build_w05_learning_runtime(backend, adapter)
    try:
        yield adapter, learning
    finally:
        backend.close()


def _candidate(adapter, perturbation):
    values = adapter.candidates_for_perturbation(perturbation)
    assert len(values) == 1
    return values[0]


def test_understanding_and_reasoning_consume_exact_structure_and_evidence(
        consumer_bundle):
    """理解返回五态，推理只授权完整 Proposition/Role/Scope/Evidence。"""
    adapter, learning = consumer_bundle
    supported = _candidate(adapter, "NONE")
    role_swap = _candidate(adapter, "ROLE_SWAP")
    scope_shift = _candidate(adapter, "SCOPE_SHIFT")
    omission = _candidate(adapter, "OCCURRENCE_OMISSION")

    understanding = build_w05_understanding_runtime(learning)
    resolution = understanding.resolve(understanding_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50513, 1)),
    ))
    assert resolution.status == W05_UNDERSTANDING_UNIQUE
    assert resolution.selected == supported
    assert resolution.evidence_keys
    understanding_use = understanding.adopt(resolution, supported)
    understanding_outcome = understanding.verify_use(understanding_use)
    assert understanding_outcome.verdict == W05_UNDERSTANDING_OUTCOME_SUPPORT

    conflict = understanding.resolve(understanding_request_for_candidate(
        scope_shift,
        request_key=LosslessIntegerKey((50513, 2)),
    ))
    assert conflict.status == W05_UNDERSTANDING_CONFLICT
    assert conflict.candidates == (scope_shift,)
    rejected = understanding.resolve(understanding_request_for_candidate(
        role_swap,
        request_key=LosslessIntegerKey((50513, 3)),
    ))
    assert rejected.status == W05_UNDERSTANDING_UNKNOWN
    superseded = understanding.resolve(understanding_request_for_candidate(
        omission,
        request_key=LosslessIntegerKey((50513, 4)),
    ))
    assert superseded.status == W05_UNDERSTANDING_UNKNOWN

    no_role = build_w05_understanding_runtime(
        learning,
        protocol=W05UnderstandingProtocol(role_bridge_connected=False),
    )
    assert no_role.resolve(understanding_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50513, 5)),
    )).status == W05_UNDERSTANDING_UNKNOWN

    reasoning = build_w05_reasoning_runtime(learning)
    authorized = reasoning.authorize(reasoning_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50514, 1)),
    ))
    assert authorized.status == W05_REASONING_AUTHORIZED
    assert authorized.candidate == supported
    assert authorized.evidence_keys
    assert reasoning.verify_use(
        authorized).verdict == W05_REASONING_OUTCOME_SUPPORT
    assert reasoning.authorize(reasoning_request_for_candidate(
        role_swap,
        request_key=LosslessIntegerKey((50514, 2)),
    )).status == W05_REASONING_REJECTED
    assert reasoning.authorize(reasoning_request_for_candidate(
        scope_shift,
        request_key=LosslessIntegerKey((50514, 3)),
    )).status == W05_REASONING_CONFLICT
    assert reasoning.authorize(reasoning_request_for_candidate(
        omission,
        request_key=LosslessIntegerKey((50514, 4)),
    )).status == W05_REASONING_SUPERSEDED

    no_scope = build_w05_reasoning_runtime(
        learning,
        protocol=W05ReasoningProtocol(scope_projection_connected=False),
    )
    assert no_scope.authorize(reasoning_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50514, 5)),
    )).status == W05_REASONING_REJECTED


def test_generation_writes_exact_choice_use_outcome_and_independent_recovery(
        consumer_bundle):
    """Generation 保留 target structure，并由独立 Understanding 回读。"""
    adapter, learning = consumer_bundle
    supported = _candidate(adapter, "NONE")
    branch = language_branch_identity(
        (50515, 1), versions=W05_IDENTITY_VERSIONS)
    uncertainty = concept_identity(
        (50515, 2), versions=W05_IDENTITY_VERSIONS)
    constraints = GenerationExpressionConstraints(
        branch,
        (),
        (),
        0,
        0,
        0,
        128,
    )
    request = generation_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50515, 10)),
        uncertainty=uncertainty,
        constraints=constraints,
    )
    generation = build_w05_generation_runtime(learning)
    choice = generation.choose(request)
    assert choice.status == W05_GENERATION_READY
    assert choice.options
    assert all(item.target_proposition == supported.candidate
               for item in choice.options)
    assert all(item.role_bindings == request.role_bindings
               for item in choice.options)
    assert all(item.context == request.target.context for item in choice.options)
    assert all(item.surface == supported.surface for item in choice.options)

    selected = choice.options[0]
    uses = generation.adopt(choice, (selected.stable_key(),))
    adopted = next(
        item for item in uses
        if item.decision.action == W05_GENERATION_ADOPTED
    )
    independent_understanding = build_w05_understanding_runtime(learning)
    outcome = generation.verify_use(
        adopted, understanding=independent_understanding)
    assert outcome.verdict == W05_GENERATION_OUTCOME_SUPPORT
    assert outcome.understanding_status == W05_UNDERSTANDING_UNIQUE
    assert outcome.occurrence_preserved
    assert outcome.role_preserved
    assert outcome.scope_preserved
    assert outcome.proposition_preserved
    assert outcome.ref.use_key == adopted.ref.use_key

    case_values = (
        choice.status == W05_GENERATION_READY,
        outcome.occurrence_preserved,
        outcome.role_preserved,
        outcome.scope_preserved,
        outcome.understanding_status == W05_UNDERSTANDING_UNIQUE,
        outcome.verdict == W05_GENERATION_OUTCOME_SUPPORT,
    )
    cases = tuple(
        W05GenerationCaseResult(
            name,
            passed,
            LosslessIntegerKey((50515, 100 + ordinal)),
        )
        for ordinal, (name, passed) in enumerate(
            zip(W05_GENERATION_HARD_CASES, case_values, strict=True),
            start=1,
        )
    )
    report = run_w05_generation_hard_conjunct(
        cases, protocol=generation.protocol)
    assert report.status == "PASS"
    assert report.passed == 1


def test_consumer_and_generation_ablation_boundaries_are_directional(
        consumer_bundle):
    """关闭 Proposition/Role/Scope/Generation bridge 时不从其它 consumer 代答。"""
    adapter, learning = consumer_bundle
    supported = _candidate(adapter, "NONE")
    branch = language_branch_identity(
        (50515, 20), versions=W05_IDENTITY_VERSIONS)
    uncertainty = concept_identity(
        (50515, 21), versions=W05_IDENTITY_VERSIONS)
    constraints = GenerationExpressionConstraints(
        branch, (), (), 0, 0, 0, 128)

    no_proposition = build_w05_understanding_runtime(
        learning,
        protocol=W05UnderstandingProtocol(
            proposition_consumer_connected=False),
    )
    assert no_proposition.resolve(understanding_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50513, 20)),
    )).status == W05_UNDERSTANDING_UNKNOWN

    no_reasoning_proposition = build_w05_reasoning_runtime(
        learning,
        protocol=W05ReasoningProtocol(
            proposition_consumer_connected=False),
    )
    use = no_reasoning_proposition.authorize(reasoning_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50514, 20)),
    ))
    assert use.status != W05_REASONING_AUTHORIZED

    no_generation = build_w05_generation_runtime(
        learning,
        protocol=W05GenerationProtocol(generation_bridge_connected=False),
    )
    request = generation_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50515, 20)),
        uncertainty=uncertainty,
        constraints=constraints,
    )
    assert no_generation.choose(request).status == W05_GENERATION_UNKNOWN

    no_role = build_w05_generation_runtime(
        learning,
        protocol=W05GenerationProtocol(role_bridge_connected=False),
    )
    request_role = generation_request_for_candidate(
        supported,
        request_key=LosslessIntegerKey((50515, 21)),
        uncertainty=uncertainty,
        constraints=constraints,
    )
    assert no_role.choose(request_role).status == W05_GENERATION_REJECTED

    cases = tuple(
        W05GenerationCaseResult(
            name,
            True,
            LosslessIntegerKey((50515, 300 + ordinal)),
        )
        for ordinal, name in enumerate(W05_GENERATION_HARD_CASES, start=1)
    )
    failed = run_w05_generation_hard_conjunct(
        cases, protocol=no_generation.protocol)
    assert failed.status == "FAIL"
    assert failed.passed == 0

    import pure_integer_ai.experiments.ph2_w05_generation as generation_module
    import pure_integer_ai.experiments.ph2_w05_reasoning as reasoning_module
    import pure_integer_ai.experiments.ph2_w05_understanding as understanding_module

    source = "\n".join((
        inspect.getsource(generation_module),
        inspect.getsource(reasoning_module),
        inspect.getsource(understanding_module),
    ))
    assert "小猫" not in source
    assert "小鸟" not in source
    assert "ROLE_ACTOR" not in source
    assert "ROLE_PATIENT" not in source
