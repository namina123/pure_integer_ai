"""W06-R03 PROPERTY 的公开有界 direct fact 闭环专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.property_relation import (
    PropertyQueryBudget,
    PropertyRelationBudgetExceeded,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06_IDENTITY_VERSIONS,
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
from pure_integer_ai.experiments.ph2_w06_r03 import (
    W06R03Runtime,
    generation_request_for_candidate,
    slice_w06_r03_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ConsumerProtocol,
    W06R03ContractError,
    W06R03ReasoningRequest,
    W06R03UnderstandingRequest,
    W06_R03_GENERATION_READY,
    W06_R03_GENERATION_UNKNOWN,
    W06_R03_OUTCOME_REFUTE,
    W06_R03_OUTCOME_SUPPORT,
    W06_R03_REASONING_CONFLICT,
    W06_R03_REASONING_REFUTED,
    W06_R03_REASONING_SUPPORTED,
    W06_R03_REASONING_UNRESOLVED,
    W06_R03_UNDERSTANDING_CONFLICT,
    W06_R03_UNDERSTANDING_UNIQUE,
    W06_R03_UNDERSTANDING_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    w06_r03_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = PropertyQueryBudget(32, 32)


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


def _build(adapted, protocol=W06R03ConsumerProtocol()):
    """建立只含 PROPERTY train truth 的共享 W-06 learning/runtime。"""
    sliced = slice_w06_r03_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R03Runtime(
        learning, sliced, protocol=protocol)


def _active_candidates(sliced, learning):
    """返回 current active 的两个 supported PROPERTY candidate。"""
    return tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )


def _constraints(candidate):
    """按 candidate 语言分支构造最小 generation 约束。"""
    branch = w06_r03_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _understanding_request(runtime, candidate, *, key):
    """从 candidate 的 subject/attribute 构造 Understanding 请求。"""
    claim = runtime.view.claim_for(candidate)
    return W06R03UnderstandingRequest(
        LosslessIntegerKey(key),
        claim.subject,
        claim.attribute,
        BUDGET,
    )


def _reasoning_request(runtime, candidate, *, key):
    """从 candidate 完整六维 claim 构造 Reasoning 请求。"""
    return W06R03ReasoningRequest(
        LosslessIntegerKey(key),
        runtime.view.claim_for(candidate),
        BUDGET,
    )


def _exercise_positive(runtime, candidates):
    """对两个 active direct PROPERTY 执行 U/R/G 正向闭环。"""
    results = []
    for ordinal, candidate in enumerate(candidates, start=1):
        understanding = runtime.resolve_understanding(
            _understanding_request(runtime, candidate, key=(1, ordinal)))
        understanding_use = runtime.adopt_understanding(understanding)
        understanding_outcome = runtime.verify_understanding(understanding_use)
        reasoning = runtime.resolve_reasoning(
            _reasoning_request(runtime, candidate, key=(2, ordinal)))
        reasoning_use = runtime.adopt_reasoning(reasoning)
        reasoning_outcome = runtime.verify_reasoning(reasoning_use)
        claim = runtime.view.claim_for(candidate)
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            claim=claim,
            request_key=LosslessIntegerKey((3, ordinal)),
            constraints=_constraints(candidate),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        generation_outcome = runtime.verify_generation(generation_use)
        results.append((
            candidate,
            understanding,
            understanding_use,
            understanding_outcome,
            reasoning,
            reasoning_use,
            reasoning_outcome,
            choice,
            generation_use,
            generation_outcome,
        ))
    return tuple(results)


def test_w06_r03_slice_keeps_seven_property_candidates(adapted):
    """七条 train PROPERTY 必须全部成为 candidate，且无 schema rejection。"""
    sliced = slice_w06_r03_adapter(adapted)
    assert len(sliced.candidates) == 7
    assert len(sliced.evidence) == 7
    assert len(sliced.observations) == 7
    assert len(sliced.rejections) == 0
    assert len(sliced.rejection_evidence) == 0
    assert len(sliced.schemas) == 1
    assert {item.relation_family for item in sliced.candidates} == {"PROPERTY"}
    assert {item.observation.split for item in sliced.candidates} == {"train"}
    assert {item.substage_key for item in sliced.candidates} == {"PROPERTY"}
    assert {item.sample_role for item in sliced.candidates} == {
        "support", "refute", "conflict", "supersede"}
    assert all(item.rational_role_values is not None
               for item in sliced.candidates)


def test_w06_r03_direct_property_urg_use_and_generation_postcheck(adapted):
    """两个 active PROPERTY direct fact 均能被 U/R/G 采用并独立重验。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidates = _active_candidates(sliced, learning)
        assert len(candidates) == 2
        results = _exercise_positive(runtime, candidates)
        for (candidate, understanding, understanding_use, understanding_outcome,
             reasoning, reasoning_use, reasoning_outcome, choice,
             generation_use, generation_outcome) in results:
            claim = runtime.view.claim_for(candidate)
            assert understanding.status == W06_R03_UNDERSTANDING_UNIQUE
            assert understanding.selected == claim
            assert len(understanding.propositions) == 1
            assert len(understanding_use.relation_uses) == 1
            assert understanding_outcome.verdict == W06_R03_OUTCOME_SUPPORT
            assert reasoning.status == W06_R03_REASONING_SUPPORTED
            assert len(reasoning.propositions) == 1
            assert len(reasoning_use.relation_uses) == 1
            assert reasoning_outcome.verdict == W06_R03_OUTCOME_SUPPORT
            assert choice.status == W06_R03_GENERATION_READY
            assert len(choice.options) == 1
            assert len(generation_use.relation_uses) == 1
            assert generation_use.relation_uses[0].proposition == (
                candidate.proposition.proposition)
            assert generation_outcome.verdict == W06_R03_OUTCOME_SUPPORT
            assert generation_outcome.property_query_status == (
                W06_R03_REASONING_SUPPORTED)
            assert generation_outcome.recovered_target is True
            assert generation_outcome.surface_structure_valid is True
            assert not hasattr(choice.request, "expected_surface")
            assert not hasattr(choice.request, "expected_label")

        report = runtime.report()
        assert report.candidate_count == 7
        assert report.rejection_count == 0
        assert report.active_count == 2
        assert report.refuted_count == 3
        assert report.conflict_count == 1
        assert report.superseded_count == 1
        assert report.understanding_use_count == 2
        assert report.reasoning_use_count == 2
        assert report.generation_use_count == 2
        assert report.generation_outcome_count == 2
        assert report.consumed_premise_count == 6
        assert report.private_read_count == 0
        assert report.formal_guard_read_count == 0
        assert report.future_relation_claim_count == 0
        assert report.w06_started == 0
    finally:
        backend.close()


def test_w06_r03_refute_conflict_and_supersede_are_separated(adapted):
    """role/value/intensity refute、current conflict 与 parser supersede 分账。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        original = next(
            item for item in sliced.candidates
            if item.sample_role == "support"
            and item.source_record.revision_id == "teacher-maple-red-v1")
        revised = next(
            item for item in sliced.candidates
            if item.sample_role == "supersede")
        conflict_value = next(
            item for item in sliced.candidates
            if item.sample_role == "conflict")
        refutes = tuple(
            item for item in sliced.candidates if item.sample_role == "refute")

        original_snapshot = learning.snapshot_for(
            original.proposition.proposition)
        assert original_snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert original_snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED

        current = runtime.resolve_reasoning(
            _reasoning_request(runtime, revised, key=(20, 1)))
        assert current.status == W06_R03_REASONING_SUPPORTED
        assert current.propositions == (revised.proposition.proposition,)

        conflict = runtime.resolve_understanding(
            _understanding_request(runtime, conflict_value, key=(20, 2)))
        assert conflict.status == W06_R03_UNDERSTANDING_CONFLICT
        assert conflict.propositions == (conflict_value.proposition.proposition,)
        assert learning.snapshot_for(
            conflict_value.proposition.proposition,
        ).snapshot.lifecycle == LIFECYCLE_ACTIVE

        for ordinal, candidate in enumerate(refutes, start=1):
            resolution = runtime.resolve_reasoning(
                _reasoning_request(runtime, candidate, key=(21, ordinal)))
            assert resolution.status == W06_R03_REASONING_REFUTED
            assert resolution.propositions == (candidate.proposition.proposition,)
            assert learning.snapshot_for(
                candidate.proposition.proposition,
            ).snapshot.epistemic_status == EPISTEMIC_REFUTED
    finally:
        backend.close()


def test_w06_r03_unknown_future_and_structure_fail_closed(adapted):
    """未知 identity、未来 relation 与结构漂移不得被表层相似放行。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        claim = runtime.view.claim_for(candidate)
        unknown_subject = ObjectIdentity(
            OBJECT_ENTITY,
            (50603, 990, 1),
            versions=W06_IDENTITY_VERSIONS,
        )
        unknown_attribute = ObjectIdentity(
            OBJECT_CONCEPT,
            (50603, 990, 2),
            versions=W06_IDENTITY_VERSIONS,
        )
        unknown = runtime.resolve_understanding(W06R03UnderstandingRequest(
            LosslessIntegerKey((30, 1)),
            unknown_subject,
            unknown_attribute,
            BUDGET,
        ))
        assert unknown.status == W06_R03_UNDERSTANDING_UNKNOWN

        swapped = replace(claim, value=claim.attribute, attribute=claim.value)
        wrong = runtime.resolve_reasoning(W06R03ReasoningRequest(
            LosslessIntegerKey((30, 2)),
            swapped,
            BUDGET,
        ))
        assert wrong.status == W06_R03_REASONING_UNRESOLVED

        later = next(
            item for item in adapted.candidates
            if item.substage_key == "MEREOLOGY")
        with pytest.raises(W06R03ContractError, match="不属于 R03"):
            generation_request_for_candidate(
                later,
                claim=claim,
                request_key=LosslessIntegerKey((30, 3)),
                constraints=_constraints(candidate),
            )
    finally:
        backend.close()


def test_w06_r03_target_and_generation_ablations_are_orthogonal(adapted):
    """关闭 PROPERTY bridge 击穿三向，只关闭 generation 不得击穿 U/R。"""
    target_backend, sliced, learning, target = _build(
        adapted,
        W06R03ConsumerProtocol(property_bridge_connected=False),
    )
    try:
        candidate = _active_candidates(sliced, learning)[0]
        assert target.resolve_understanding(
            _understanding_request(target, candidate, key=(40, 1))
        ).status == W06_R03_UNDERSTANDING_UNKNOWN
        assert target.resolve_reasoning(
            _reasoning_request(target, candidate, key=(40, 2))
        ).status == W06_R03_REASONING_UNRESOLVED
        choice = target.choose_generation(generation_request_for_candidate(
            candidate,
            claim=target.view.claim_for(candidate),
            request_key=LosslessIntegerKey((40, 3)),
            constraints=_constraints(candidate),
        ))
        assert choice.status == W06_R03_GENERATION_UNKNOWN
    finally:
        target_backend.close()

    generation_backend, sliced, learning, generation = _build(
        adapted,
        W06R03ConsumerProtocol(generation_connected=False),
    )
    try:
        candidate = _active_candidates(sliced, learning)[0]
        assert generation.resolve_understanding(
            _understanding_request(generation, candidate, key=(41, 1))
        ).status == W06_R03_UNDERSTANDING_UNIQUE
        assert generation.resolve_reasoning(
            _reasoning_request(generation, candidate, key=(41, 2))
        ).status == W06_R03_REASONING_SUPPORTED
        choice = generation.choose_generation(generation_request_for_candidate(
            candidate,
            claim=generation.view.claim_for(candidate),
            request_key=LosslessIntegerKey((41, 3)),
            constraints=_constraints(candidate),
        ))
        assert choice.status == W06_R03_GENERATION_UNKNOWN
    finally:
        generation_backend.close()


def test_w06_r03_postcheck_ablation_refutes_generation_only(adapted):
    """关闭 postcheck 后 option 可形成，但不能伪装 generation SUPPORT。"""
    backend, sliced, learning, runtime = _build(
        adapted, W06R03ConsumerProtocol(postcheck_connected=False))
    try:
        candidate = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            claim=runtime.view.claim_for(candidate),
            request_key=LosslessIntegerKey((50, 1)),
            constraints=_constraints(candidate),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R03_OUTCOME_REFUTE
        assert outcome.property_query_status == W06_R03_REASONING_UNRESOLVED
        assert outcome.authorization_current is True
        assert outcome.relation_structure_preserved is True
    finally:
        backend.close()


def test_w06_r03_withdrawal_demotes_direct_property_and_stale_use(adapted):
    """撤回 active PROPERTY 后 U/R/G 均不得继续 SUPPORT。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        before = runtime.resolve_reasoning(
            _reasoning_request(runtime, candidate, key=(60, 1)))
        use = runtime.adopt_reasoning(before)
        assert runtime.verify_reasoning(use).verdict == W06_R03_OUTCOME_SUPPORT

        application = next(
            item for item in learning.applications()
            if item.binding.candidate == candidate.proposition.proposition)
        account = next(
            item for item in application.accounts
            if item.stance == EVIDENCE_SUPPORT and not item.derived_supersede)
        prior = account.trace.outcome.evidence
        withdrawal = learning.withdraw_evidence(account, withdrawal_level=1)
        assert withdrawal.evidence.supersedes_evidence_id == prior.evidence_id

        after_reasoning = runtime.resolve_reasoning(
            _reasoning_request(runtime, candidate, key=(60, 2)))
        assert after_reasoning.status == W06_R03_REASONING_UNRESOLVED
        after_understanding = runtime.resolve_understanding(
            _understanding_request(runtime, candidate, key=(60, 3)))
        assert after_understanding.status == W06_R03_UNDERSTANDING_UNKNOWN
        stale = runtime.verify_reasoning(use)
        assert stale.verdict == W06_R03_OUTCOME_REFUTE
        assert stale.current_status == W06_R03_REASONING_UNRESOLVED
        snapshot = learning.snapshot_for(candidate.proposition.proposition)
        assert snapshot.active_fact is None
        assert snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        history = learning.learning.engine.ledger.evidence_history(
            snapshot.formation.hypothesis)
        assert prior in history
        assert withdrawal.evidence in history
    finally:
        backend.close()


def test_w06_r03_budget_fails_closed_without_partial_result(adapted):
    """调用方预算不足时必须抛出，不返回部分 PROPERTY 结果。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidate = _active_candidates(sliced, learning)[0]
        tiny = replace(
            _reasoning_request(runtime, candidate, key=(70, 1)),
            budget=PropertyQueryBudget(1, 32),
        )
        with pytest.raises(PropertyRelationBudgetExceeded):
            runtime.resolve_reasoning(tiny)
    finally:
        backend.close()


def test_w06_r03_replay_is_bit_identical(adapted):
    """相同 public train 与 U/R/G 请求必须得到 bit-identical state/report。"""
    left_backend, left_slice, left_learning, left = _build(adapted)
    right_backend, right_slice, right_learning, right = _build(adapted)
    try:
        left_candidates = _active_candidates(left_slice, left_learning)
        right_candidates = _active_candidates(right_slice, right_learning)
        _exercise_positive(left, left_candidates)
        _exercise_positive(right, right_candidates)
        assert left.state_key() == right.state_key()
        assert left.report() == right.report()
        assert left_backend.snapshot() == right_backend.snapshot()
    finally:
        left_backend.close()
        right_backend.close()
