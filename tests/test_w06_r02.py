"""W06-R02 SUBSET/MEMBER 的公开有界集合闭环专项。"""
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
    OBJECT_ENTITY,
    OBJECT_SET_EXPR,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.set_relation import (
    SetRelationBudget,
    SetRelationBudgetExceeded,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    adapt_w06_training_payload,
    W06_IDENTITY_VERSIONS,
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
from pure_integer_ai.experiments.ph2_w06_r02 import (
    W06R02Runtime,
    generation_request_for_candidate,
    query_for_candidate,
    slice_w06_r02_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r02_contract import (
    W06R02ConsumerProtocol,
    W06R02ContractError,
    W06R02SetQuery,
    W06_R02_CONFLICT,
    W06_R02_GENERATION_READY,
    W06_R02_GENERATION_UNKNOWN,
    W06_R02_OUTCOME_REFUTE,
    W06_R02_OUTCOME_SUPPORT,
    W06_R02_REFUTED,
    W06_R02_SUPPORTED,
    W06_R02_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r02_endpoint_projection import (
    W06R02EndpointProjectionError,
    W06_R02_ENDPOINT_PROJECTION_PATH,
    canonical_w06_r02_endpoint_projection_bytes,
    publish_w06_r02_endpoint_projection,
    read_w06_r02_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r02_shared import (
    w06_r02_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = SetRelationBudget(32, 128, 512, 32, 32)


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


def _build(adapted, protocol=W06R02ConsumerProtocol()):
    """建立只含 R02 train truth 的共享 W-06 learning/runtime。"""
    sliced = slice_w06_r02_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    endpoint_projection = read_w06_r02_endpoint_projection(
        ROOT / W06_R02_ENDPOINT_PROJECTION_PATH)
    return backend, sliced, learning, W06R02Runtime(
        learning, sliced, endpoint_projection, protocol=protocol)


def _active_candidates(sliced, learning):
    active = tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )
    return (
        next(item for item in active if item.relation_family == "SUBSET"),
        next(item for item in active if item.relation_family == "MEMBER"),
    )


def _constraints(candidate):
    branch = w06_r02_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _derived_member_query(runtime, subset, member, *, key):
    subset_child, subset_parent = runtime.view.endpoints_for(subset)
    element, member_set = runtime.view.endpoints_for(member)
    assert member_set == subset_child
    return W06R02SetQuery(
        LosslessIntegerKey(key),
        "MEMBER",
        element,
        subset_parent,
        BUDGET,
    )


def _exercise_positive(runtime, candidates):
    subset, member = candidates
    understanding = runtime.resolve_understanding(
        _derived_member_query(runtime, subset, member, key=(1, 1)))
    understanding_use = runtime.adopt_understanding(understanding)
    understanding_outcome = runtime.verify_understanding(understanding_use)
    reasoning = runtime.resolve_reasoning(
        _derived_member_query(runtime, subset, member, key=(1, 2)))
    reasoning_use = runtime.adopt_reasoning(reasoning)
    reasoning_outcome = runtime.verify_reasoning(reasoning_use)
    generations = []
    for ordinal, candidate in enumerate(candidates, start=1):
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((1, 10 + ordinal)),
            constraints=_constraints(candidate),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        generations.append((choice, use, outcome))
    return (
        understanding, understanding_use, understanding_outcome,
        reasoning, reasoning_use, reasoning_outcome,
        tuple(generations),
    )


def test_w06_r02_endpoint_projection_rebuild_and_readback_are_canonical(
        tmp_path,
        ):
    """endpoint overlay 必须可由公开 parent 重建且严格回读。"""
    path = ROOT / W06_R02_ENDPOINT_PROJECTION_PATH
    canonical = canonical_w06_r02_endpoint_projection_bytes(ROOT)
    assert path.read_bytes() == canonical
    projection = read_w06_r02_endpoint_projection(path)
    clone = projection.clone_for_evaluation()
    assert len(projection.entries) == 10
    assert len({item.canonical_endpoint for item in projection.entries}) == 5
    assert clone is not projection
    assert clone.state_key() == projection.state_key()
    published = publish_w06_r02_endpoint_projection(
        ROOT, tmp_path / "endpoint-projection.json")
    assert published.read_bytes() == canonical
    assert read_w06_r02_endpoint_projection(published).state_key() == (
        projection.state_key())
    with pytest.raises(W06R02EndpointProjectionError, match="禁止覆盖"):
        publish_w06_r02_endpoint_projection(ROOT, published)


def test_w06_r02_slice_keeps_five_candidates_and_type_rejection(adapted):
    """六条 train 必须分成五个 hypothesis 与一个 schema rejection。"""
    sliced = slice_w06_r02_adapter(adapted)
    assert len(sliced.candidates) == 5
    assert len(sliced.evidence) == 5
    assert len(sliced.observations) == 5
    assert len(sliced.rejections) == len(sliced.rejection_evidence) == 1
    assert len(sliced.schemas) == 5
    assert {item.relation_family for item in sliced.candidates} == {
        "SUBSET", "MEMBER"}
    assert {item.observation.split for item in sliced.candidates} == {"train"}
    assert all(item.substage_key == "SUBSET_MEMBER"
               for item in sliced.candidates)
    rejection = sliced.rejections[0]
    assert rejection.relation_family == "MEMBER"
    assert rejection.proposition not in {
        item.proposition.proposition for item in sliced.candidates}


def test_w06_r02_derived_member_and_direct_generation_use_set_runtime(adapted):
    """MEMBER lift 必须消费 MEMBER+SUBSET 两前提，direct U/R/G 均可追溯。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidates = _active_candidates(sliced, learning)
        values = _exercise_positive(runtime, candidates)
        understanding, understanding_use, understanding_outcome = values[:3]
        reasoning, reasoning_use, reasoning_outcome = values[3:6]
        generations = values[6]
        assert understanding.status == W06_R02_SUPPORTED
        assert reasoning.status == W06_R02_SUPPORTED
        assert len(understanding.propositions) == 2
        assert len(reasoning.propositions) == 2
        assert len(understanding_use.relation_uses) == 2
        assert len(reasoning_use.relation_uses) == 2
        assert understanding_outcome.verdict == W06_R02_OUTCOME_SUPPORT
        assert reasoning_outcome.verdict == W06_R02_OUTCOME_SUPPORT
        for candidate, (choice, use, outcome) in zip(
                candidates, generations, strict=True):
            assert choice.status == W06_R02_GENERATION_READY
            assert len(choice.options) == 1
            assert len(use.relation_uses) == 1
            assert use.relation_uses[0].proposition == (
                candidate.proposition.proposition)
            assert outcome.verdict == W06_R02_OUTCOME_SUPPORT
            assert outcome.recovered_target is True
            assert outcome.surface_structure_valid is True
            assert not hasattr(choice.request, "expected_surface")
            assert not hasattr(choice.request, "expected_label")

        report = runtime.report()
        assert report.candidate_count == 5
        assert report.rejection_count == 1
        assert report.active_count == 2
        assert report.refuted_count == 1
        assert report.conflict_count == 1
        assert report.superseded_count == 1
        assert report.understanding_use_count == 1
        assert report.reasoning_use_count == 1
        assert report.generation_use_count == 2
        assert report.generation_outcome_count == 2
        assert report.derived_query_count == 2
        assert report.consumed_premise_count == 6
        assert report.private_read_count == 0
        assert report.formal_guard_read_count == 0
        assert report.future_relation_claim_count == 0
        assert report.w06_started == 0
    finally:
        backend.close()


def test_w06_r02_current_view_separates_lifecycle_states(adapted):
    """revised truth、反向 refute、current conflict 与 rejection 必须分账。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        original = next(
            item for item in sliced.candidates
            if item.relation_family == "SUBSET" and item.sample_role == "support")
        revised = next(
            item for item in sliced.candidates if item.sample_role == "supersede")
        reversed_value = next(
            item for item in sliced.candidates
            if item.perturbation_kind == "DIRECTION_REVERSAL")
        conflict_value = next(
            item for item in sliced.candidates if item.sample_role == "conflict")
        original_snapshot = learning.snapshot_for(
            original.proposition.proposition)
        assert original_snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert original_snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED

        current = runtime.resolve_reasoning(query_for_candidate(
            revised,
            request_key=LosslessIntegerKey((20, 1)),
            endpoint_resolver=runtime.view.endpoint_resolver,
            budget=BUDGET,
        ))
        assert current.status == W06_R02_SUPPORTED
        assert current.propositions == (revised.proposition.proposition,)

        refuted = runtime.resolve_reasoning(query_for_candidate(
            reversed_value,
            request_key=LosslessIntegerKey((20, 2)),
            endpoint_resolver=runtime.view.endpoint_resolver,
            budget=BUDGET,
        ))
        assert refuted.status == W06_R02_REFUTED
        assert learning.snapshot_for(
            reversed_value.proposition.proposition,
        ).snapshot.epistemic_status == EPISTEMIC_REFUTED

        conflict = runtime.resolve_understanding(query_for_candidate(
            conflict_value,
            request_key=LosslessIntegerKey((20, 3)),
            endpoint_resolver=runtime.view.endpoint_resolver,
            budget=BUDGET,
        ))
        assert conflict.status == W06_R02_CONFLICT
        assert conflict.propositions == (conflict_value.proposition.proposition,)
        assert learning.snapshot_for(
            conflict_value.proposition.proposition,
        ).snapshot.lifecycle == LIFECYCLE_ACTIVE
    finally:
        backend.close()


def test_w06_r02_direction_type_and_structural_unknown_fail_closed(adapted):
    """方向、端点类型、未知内容和后续 relation 均不得被表层相似放行。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        subset, member = _active_candidates(sliced, learning)
        subset_child, subset_parent = runtime.view.endpoints_for(subset)
        element, _member_set = runtime.view.endpoints_for(member)
        reverse = runtime.resolve_reasoning(W06R02SetQuery(
            LosslessIntegerKey((30, 1)),
            "SUBSET",
            subset_parent,
            subset_child,
            BUDGET,
        ))
        assert reverse.status == W06_R02_REFUTED
        with pytest.raises(W06R02ContractError, match="SUBSET"):
            W06R02SetQuery(
                LosslessIntegerKey((30, 2)),
                "SUBSET",
                element,
                subset_parent,
                BUDGET,
            )

        unknown_set = ObjectIdentity(
            OBJECT_SET_EXPR,
            (50602, 990, 1),
            versions=W06_IDENTITY_VERSIONS,
        )
        unknown_member = ObjectIdentity(
            OBJECT_ENTITY,
            (50602, 990, 2),
            versions=W06_IDENTITY_VERSIONS,
        )
        unknown = runtime.resolve_understanding(W06R02SetQuery(
            LosslessIntegerKey((30, 3)),
            "MEMBER",
            unknown_member,
            unknown_set,
            BUDGET,
        ))
        assert unknown.status == W06_R02_UNKNOWN

        request = generation_request_for_candidate(
            subset,
            request_key=LosslessIntegerKey((30, 4)),
            constraints=_constraints(subset),
        )
        swapped = tuple(sorted((
            (request.role_fillers[0][0], request.role_fillers[1][1]),
            (request.role_fillers[1][0], request.role_fillers[0][1]),
        ), key=lambda item: item[0].stable_key()))
        choice = runtime.choose_generation(replace(
            request,
            request_key=LosslessIntegerKey((30, 5)),
            role_fillers=swapped,
        ))
        assert choice.status == W06_R02_GENERATION_UNKNOWN
        later = next(
            item for item in adapted.candidates
            if item.substage_key == "PROPERTY")
        with pytest.raises(W06R02ContractError, match="不属于 R02"):
            generation_request_for_candidate(
                later,
                request_key=LosslessIntegerKey((30, 6)),
                constraints=_constraints(subset),
            )
    finally:
        backend.close()


def test_w06_r02_target_and_generation_ablations_are_orthogonal(adapted):
    """关闭 R02 bridge 击穿三向，只关闭 generation 不得击穿 U/R。"""
    target_backend, sliced, learning, target = _build(
        adapted,
        W06R02ConsumerProtocol(set_relation_bridge_connected=False),
    )
    try:
        subset, member = _active_candidates(sliced, learning)
        query = _derived_member_query(target, subset, member, key=(40, 1))
        assert target.resolve_understanding(query).status == W06_R02_UNKNOWN
        assert target.resolve_reasoning(replace(
            query, request_key=LosslessIntegerKey((40, 2))
        )).status == W06_R02_UNKNOWN
        choice = target.choose_generation(generation_request_for_candidate(
            subset,
            request_key=LosslessIntegerKey((40, 3)),
            constraints=_constraints(subset),
        ))
        assert choice.status == W06_R02_GENERATION_UNKNOWN
    finally:
        target_backend.close()

    generation_backend, sliced, learning, generation = _build(
        adapted,
        W06R02ConsumerProtocol(generation_connected=False),
    )
    try:
        subset, member = _active_candidates(sliced, learning)
        query = _derived_member_query(
            generation, subset, member, key=(41, 1))
        assert generation.resolve_understanding(query).status == W06_R02_SUPPORTED
        assert generation.resolve_reasoning(replace(
            query, request_key=LosslessIntegerKey((41, 2))
        )).status == W06_R02_SUPPORTED
        choice = generation.choose_generation(generation_request_for_candidate(
            subset,
            request_key=LosslessIntegerKey((41, 3)),
            constraints=_constraints(subset),
        ))
        assert choice.status == W06_R02_GENERATION_UNKNOWN
    finally:
        generation_backend.close()


def test_w06_r02_postcheck_ablation_refutes_generation_only(adapted):
    """关闭 postcheck 后 option 可形成，但不能伪装 generation SUPPORT。"""
    backend, sliced, learning, runtime = _build(
        adapted, W06R02ConsumerProtocol(postcheck_connected=False))
    try:
        subset, member = _active_candidates(sliced, learning)
        query = _derived_member_query(runtime, subset, member, key=(50, 1))
        assert runtime.resolve_understanding(query).status == W06_R02_SUPPORTED
        choice = runtime.choose_generation(generation_request_for_candidate(
            subset,
            request_key=LosslessIntegerKey((50, 2)),
            constraints=_constraints(subset),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R02_OUTCOME_REFUTE
        assert outcome.set_query_status == W06_R02_UNKNOWN
        assert outcome.authorization_current is True
        assert outcome.relation_structure_preserved is True
    finally:
        backend.close()


def test_w06_r02_withdrawal_demotes_member_lift_and_stale_use(adapted):
    """撤回 direct MEMBER 后派生 MEMBER 必须退出，历史 Evidence 仍保留。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        subset, member = _active_candidates(sliced, learning)
        before = runtime.resolve_understanding(
            _derived_member_query(runtime, subset, member, key=(60, 1)))
        use = runtime.adopt_understanding(before)
        assert runtime.verify_understanding(use).verdict == (
            W06_R02_OUTCOME_SUPPORT)
        application = next(
            item for item in learning.applications()
            if item.binding.candidate == member.proposition.proposition)
        account = next(
            item for item in application.accounts
            if item.stance == EVIDENCE_SUPPORT and not item.derived_supersede)
        prior = account.trace.outcome.evidence
        withdrawal = learning.withdraw_evidence(account, withdrawal_level=1)
        assert withdrawal.evidence.supersedes_evidence_id == prior.evidence_id

        after = runtime.resolve_understanding(
            _derived_member_query(runtime, subset, member, key=(60, 2)))
        assert after.status == W06_R02_UNKNOWN
        stale = runtime.verify_understanding(use)
        assert stale.verdict == W06_R02_OUTCOME_REFUTE
        assert stale.current_status == W06_R02_UNKNOWN
        snapshot = learning.snapshot_for(member.proposition.proposition)
        assert snapshot.active_fact is None
        assert snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        history = learning.learning.engine.ledger.evidence_history(
            snapshot.formation.hypothesis)
        assert prior in history
        assert withdrawal.evidence in history
    finally:
        backend.close()


def test_w06_r02_budget_fails_closed_without_partial_result(adapted):
    """调用方预算不足时必须抛出，不返回部分集合闭包。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        subset, member = _active_candidates(sliced, learning)
        query = _derived_member_query(runtime, subset, member, key=(70, 1))
        tiny = replace(query, budget=SetRelationBudget(1, 4, 16, 4, 4))
        with pytest.raises(SetRelationBudgetExceeded):
            runtime.resolve_reasoning(tiny)
    finally:
        backend.close()


def test_w06_r02_replay_is_bit_identical(adapted):
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
