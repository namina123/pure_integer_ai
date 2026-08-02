"""W06-R07 CAUSES direct relation、独立 witness 与生成闭环专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_FORMAL_RUN_ID,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_W05_BASE_RUN_ID,
    W06RunRequest,
    open_w06_frozen_context,
)
from pure_integer_ai.experiments.ph2_w06_firewall import W06PayloadFirewall
from pure_integer_ai.experiments.ph2_w06_learning import (
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_r07 import (
    W06R07Runtime,
    generation_request_for_candidate,
    slice_w06_r07_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07Budget,
    W06R07BudgetExceeded,
    W06R07CausalQuery,
    W06R07ConsumerProtocol,
    W06R07ContractError,
    W06_R07_GENERATION_READY,
    W06_R07_GENERATION_REJECTED,
    W06_R07_GENERATION_UNKNOWN,
    W06_R07_OUTCOME_REFUTE,
    W06_R07_OUTCOME_SUPPORT,
    W06_R07_REFUTED,
    W06_R07_SUPPORTED,
    W06_R07_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r07_endpoint_projection import (
    W06_R07_ENDPOINT_PROJECTION_PATH,
    canonical_w06_r07_endpoint_projection_bytes,
    read_w06_r07_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r07_shared import (
    candidate_causal_protocol,
    w06_r07_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = W06R07Budget(32, 64, 512)
EXPECTED_PROJECTION_SHA256 = (
    "a3ec39a0b0eeea750d53b08277f0bef0cf39338994cabead45dea163a176e4b8"
)
EXPECTED_RELATION_DIGEST = (
    122, 141, 85, 59, 198, 105, 161, 8, 223, 168, 197, 34, 164, 145,
    5, 118, 128, 93, 88, 134, 39, 200, 24, 207, 111, 3, 12, 55, 166,
    254, 98, 222,
)
EXPECTED_EVIDENCE_DIGEST = (
    255, 87, 37, 77, 72, 67, 247, 118, 210, 89, 80, 1, 199, 71, 0,
    95, 231, 6, 13, 25, 161, 251, 162, 193, 8, 24, 117, 221, 145,
    173, 140, 213,
)
EXPECTED_ACTIVE_DIGEST = (
    65, 222, 20, 138, 211, 219, 178, 230, 214, 25, 66, 88, 116, 15,
    101, 20, 82, 247, 218, 250, 24, 26, 251, 189, 149, 209, 53, 180,
    168, 116, 128, 87,
)


@pytest.fixture(scope="module")
def adapted():
    """只经 public train firewall 构建一次完整 adapter 输出。"""
    backend = DictBackend()
    try:
        context = open_w06_frozen_context(
            ROOT,
            current_remote_commit_sha1=PUBLIC_BASELINE_HEAD,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        request = W06RunRequest(
            run_id=W06_FORMAL_RUN_ID,
            parent_run_id=W06_W05_BASE_RUN_ID,
            base_run_id=W06_W05_BASE_RUN_ID,
            stage_key=W06_STAGE_KEY,
            owner_key=context.owner_key,
            runner_key=W06_RUNNER_KEY,
            current_remote_commit_sha1=context.current_remote_commit_sha1,
            source_overlay_sha256=context.source_overlay_sha256,
            context_key=context.stable_key(),
            backend_profile_key=context.backend_profile_key,
            base_fence_key=context.base_fence_key,
            worker_count=1,
            mode="fresh",
            resource_budget=tuple(sorted(W06_RESOURCE_BUDGET.items())),
            candidate_payload_paths=tuple(
                item.relative_path
                for item in context.candidate_payload_bindings),
            teacher_evidence_paths=tuple(
                item.relative_path
                for item in context.teacher_evidence_bindings),
        )
        payload = W06PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
        return adapt_w06_training_payload(payload)
    finally:
        backend.close()


def _projection():
    return read_w06_r07_endpoint_projection(
        ROOT / W06_R07_ENDPOINT_PROJECTION_PATH)


def _build(adapted, protocol=W06R07ConsumerProtocol()):
    sliced = slice_w06_r07_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R07Runtime(
        learning, sliced, _projection(), protocol=protocol)


def _active_candidates(sliced, learning):
    return tuple(sorted(
        (
            item for item in sliced.candidates
            if learning.snapshot_for(
                item.proposition.proposition).active_fact is not None
        ),
        key=lambda item: item.proposition.proposition.stable_key(),
    ))


def _constraints(candidate):
    branch = w06_r07_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _query(runtime, candidate, *, key, budget=BUDGET):
    return runtime.query_for_candidate(
        candidate,
        request_key=LosslessIntegerKey(key),
        budget=budget,
    )


def _support_account(learning, candidate):
    for application in learning.applications():
        for account in application.accounts:
            if (account.candidate == candidate.proposition.proposition
                    and account.stance == EVIDENCE_SUPPORT
                    and not account.derived_supersede):
                return account
    raise AssertionError("未找到普通 support Evidence")


def _exercise_positive(runtime, sliced, learning):
    """两个 canonical pair 做 U/R，三个 active statement 做 G。"""
    active = _active_candidates(sliced, learning)
    representatives = {}
    for candidate in active:
        representatives.setdefault(runtime.view.endpoints_for(candidate), candidate)
    query_results = []
    for ordinal, candidate in enumerate(representatives.values(), start=1):
        understanding = runtime.resolve_understanding(
            _query(runtime, candidate, key=(1, ordinal)))
        understanding_use = runtime.adopt_understanding(understanding)
        understanding_outcome = runtime.verify_understanding(understanding_use)
        reasoning = runtime.resolve_reasoning(
            _query(runtime, candidate, key=(2, ordinal)))
        reasoning_use = runtime.adopt_reasoning(reasoning)
        reasoning_outcome = runtime.verify_reasoning(reasoning_use)
        query_results.append((
            understanding,
            understanding_use,
            understanding_outcome,
            reasoning,
            reasoning_use,
            reasoning_outcome,
        ))
    generation_results = []
    for ordinal, candidate in enumerate(active, start=1):
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((3, ordinal)),
            constraints=_constraints(candidate),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        generation_results.append((candidate, choice, use, outcome))
    return tuple(query_results), tuple(generation_results)


def test_w06_r07_slice_and_endpoint_projection_are_train_only(adapted):
    sliced = slice_w06_r07_adapter(adapted)
    assert len(sliced.candidates) == len(sliced.evidence) == 10
    assert not sliced.rejections
    assert {item.relation_family for item in sliced.candidates} == {"CAUSES"}
    projection = _projection()
    assert len(projection.entries) == 20
    assert len({item.canonical_endpoint for item in projection.entries}) == 14
    canonical = canonical_w06_r07_endpoint_projection_bytes(ROOT)
    assert canonical == (ROOT / W06_R07_ENDPOINT_PROJECTION_PATH).read_bytes()
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_PROJECTION_SHA256


def test_w06_r07_direct_urg_use_generation_and_digests(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        queries, generations = _exercise_positive(runtime, sliced, learning)
        assert len(queries) == 2
        assert all(
            item.status == W06_R07_SUPPORTED
            and outcome.verdict == W06_R07_OUTCOME_SUPPORT
            for result in queries
            for item, outcome in ((result[0], result[2]), (result[3], result[5]))
        )
        assert len(generations) == 3
        assert all(
            choice.status == W06_R07_GENERATION_READY
            and outcome.verdict == W06_R07_OUTCOME_SUPPORT
            and not outcome.effect_execution_authorized
            for _candidate, choice, _use, outcome in generations
        )
        report = runtime.report()
        assert report.relation_digest == EXPECTED_RELATION_DIGEST
        assert report.source_evidence_digest == EXPECTED_EVIDENCE_DIGEST
        assert report.active_projection_digest == EXPECTED_ACTIVE_DIGEST
        assert (
            report.candidate_count,
            report.active_count,
            report.refuted_count,
            report.conflict_count,
            report.superseded_count,
        ) == (10, 3, 6, 0, 1)
        assert (
            report.understanding_use_count,
            report.reasoning_use_count,
            report.generation_use_count,
            report.consumed_premise_count,
        ) == (2, 2, 3, 9)
        assert (
            report.effect_execution_count,
            report.event_time_fact_write_count,
            report.causal_implies_event_time_fact,
            report.precedence_implies_causation,
            report.temporal_support_sufficient,
        ) == (0, 0, 0, 0, 0)
    finally:
        backend.close()


def test_w06_r07_canonical_pair_keeps_two_direct_active_premises(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)
        groups = {}
        for candidate in active:
            groups.setdefault(runtime.view.endpoints_for(candidate), []).append(candidate)
        assert sorted(len(value) for value in groups.values()) == [1, 2]
        pair = next(key for key, value in groups.items() if len(value) == 2)
        candidate = groups[pair][0]
        resolution = runtime.resolve_understanding(
            _query(runtime, candidate, key=(10, 1)))
        use = runtime.adopt_understanding(resolution)
        assert len(resolution.propositions) == len(use.relation_uses) == 2
        assert {item.proposition for item in use.relation_uses} == {
            item.proposition.proposition for item in groups[pair]
        }
    finally:
        backend.close()


def test_w06_r07_refute_families_and_supersede_remain_separated(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        expected = {
            "CORRELATION_CONFUSION",
            "PSEUDO_RELATION",
            "COUNTERFACTUAL_OVERCLAIM",
            "CONFOUNDING_CONFUSION",
            "TEMPORAL_ONLY",
            "DIRECTION_REVERSAL",
        }
        refuted = tuple(
            item for item in sliced.candidates
            if learning.snapshot_for(
                item.proposition.proposition).snapshot.epistemic_status
            == EPISTEMIC_REFUTED
        )
        assert {item.perturbation_kind for item in refuted} == expected
        for ordinal, candidate in enumerate(refuted, start=1):
            resolution = runtime.resolve_understanding(
                _query(runtime, candidate, key=(20, ordinal)))
            assert resolution.status == W06_R07_REFUTED
            assert not resolution.evaluation.effect_execution_authorized
        conflict = next(
            item for item in sliced.candidates
            if item.perturbation_kind == "CONFLICT_SOURCE")
        snapshot = learning.snapshot_for(conflict.proposition.proposition)
        assert snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert snapshot.active_fact is None
    finally:
        backend.close()


def test_w06_r07_precedes_and_order_shortcuts_fail_closed(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        precedes = adapted.candidates_for_substage("PRECEDES")[0]
        with pytest.raises(W06R07ContractError):
            runtime.query_for_candidate(
                precedes, request_key=LosslessIntegerKey((30, 1)))
        active = _active_candidates(sliced, learning)
        cause, effect = runtime.view.endpoints_for(active[0])
        reverse = runtime.resolve_reasoning(W06R07CausalQuery(
            LosslessIntegerKey((30, 2)),
            effect,
            cause,
            BUDGET,
            active[0].source_ref,
        ))
        assert reverse.status == W06_R07_REFUTED
        assert reverse.evaluation.direct_only
        assert not reverse.evaluation.effect_execution_authorized
        protocol = candidate_causal_protocol(active[0])
        assert protocol.relation == active[0].proposition.predicate
    finally:
        backend.close()


def test_w06_r07_does_not_invent_cross_pair_or_effect_path(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)
        first = runtime.view.endpoints_for(active[0])
        second = next(
            runtime.view.endpoints_for(item) for item in active
            if runtime.view.endpoints_for(item) != first)
        resolution = runtime.resolve_understanding(W06R07CausalQuery(
            LosslessIntegerKey((40, 1)),
            first[0],
            second[1],
            BUDGET,
            active[0].source_ref,
        ))
        assert resolution.status == W06_R07_UNKNOWN
        assert resolution.evaluation.witnesses == ()
        assert not resolution.evaluation.effect_execution_authorized
    finally:
        backend.close()


@pytest.mark.parametrize("field", [
    "causes_connected",
    "witness_connected",
    "temporal_boundary_connected",
])
def test_w06_r07_query_ablations_are_orthogonal(adapted, field):
    protocol = replace(W06R07ConsumerProtocol(), **{field: False})
    backend, sliced, learning, runtime = _build(adapted, protocol)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        resolution = runtime.resolve_understanding(
            _query(runtime, candidate, key=(50, len(field))))
        assert resolution.status == W06_R07_UNKNOWN
        assert not resolution.propositions
    finally:
        backend.close()


@pytest.mark.parametrize("field, expected", [
    ("generation_connected", W06_R07_GENERATION_UNKNOWN),
    ("source_scope_connected", W06_R07_GENERATION_REJECTED),
])
def test_w06_r07_generation_ablations_fail_closed(adapted, field, expected):
    protocol = replace(W06R07ConsumerProtocol(), **{field: False})
    backend, sliced, learning, runtime = _build(adapted, protocol)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((51, len(field))),
            constraints=_constraints(candidate),
        ))
        assert choice.status == expected
        assert not choice.options
    finally:
        backend.close()


def test_w06_r07_postcheck_ablation_refutes_generation_only(adapted):
    protocol = replace(W06R07ConsumerProtocol(), postcheck_connected=False)
    backend, sliced, learning, runtime = _build(adapted, protocol)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((52, 1)),
            constraints=_constraints(candidate),
        ))
        assert choice.status == W06_R07_GENERATION_READY
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R07_OUTCOME_REFUTE
        assert outcome.causal_query_status == W06_R07_UNKNOWN
        assert not outcome.recovered_target
        assert not outcome.effect_execution_authorized
    finally:
        backend.close()


def test_w06_r07_withdrawal_refutes_stale_urg_use(adapted):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)
        candidate = next(
            item for item in active
            if sum(
                runtime.view.endpoints_for(other)
                == runtime.view.endpoints_for(item)
                for other in active
            ) == 1
        )
        resolution = runtime.resolve_understanding(
            _query(runtime, candidate, key=(60, 1)))
        relation_use = runtime.adopt_understanding(resolution)
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((60, 2)),
            constraints=_constraints(candidate),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        account = _support_account(learning, candidate)
        withdrawal = learning.withdraw_evidence(account, withdrawal_level=1)
        assert withdrawal.prior.trace.outcome.evidence.evidence_id == (
            withdrawal.evidence.supersedes_evidence_id)
        current = learning.snapshot_for(candidate.proposition.proposition)
        assert current.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert current.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        assert current.active_fact is None
        assert runtime.verify_understanding(
            relation_use).verdict == W06_R07_OUTCOME_REFUTE
        generation_outcome = runtime.verify_generation(generation_use)
        assert generation_outcome.verdict == W06_R07_OUTCOME_REFUTE
        assert not generation_outcome.authorization_current
        assert not generation_outcome.witness_current
    finally:
        backend.close()


@pytest.mark.parametrize("budget", [
    W06R07Budget(9, 64, 512),
    W06R07Budget(32, 1, 512),
    W06R07Budget(32, 64, 1),
])
def test_w06_r07_budget_fails_without_partial_result(adapted, budget):
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)
        candidate = next(
            item for item in active
            if sum(
                runtime.view.endpoints_for(other)
                == runtime.view.endpoints_for(item)
                for other in active
            ) == 2
        )
        with pytest.raises(W06R07BudgetExceeded):
            runtime.resolve_understanding(
                _query(runtime, candidate, key=(70, *budget.stable_key()),
                       budget=budget))
        assert not runtime.understanding.resolutions
        assert not runtime.understanding.uses
    finally:
        backend.close()


def test_w06_r07_replay_is_bit_identical(adapted):
    results = []
    for _ordinal in range(2):
        backend, sliced, learning, runtime = _build(adapted)
        try:
            _exercise_positive(runtime, sliced, learning)
            results.append((runtime.state_key(), runtime.report()))
        finally:
            backend.close()
    assert results[0] == results[1]
