"""W06-R01 稳定 PURE_ALIAS/REFERS 的公开有界闭环专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import concept_identity
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
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
from pure_integer_ai.experiments.ph2_w06_r01 import (
    W06R01Runtime,
    generation_request_for_candidate,
    slice_w06_r01_adapter,
    w06_r01_alias_protocol,
    w06_r01_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ConsumerProtocol,
    W06R01ContractError,
    W06R01ReasoningRequest,
    W06R01UnderstandingRequest,
    W06_R01_GENERATION_READY,
    W06_R01_GENERATION_UNKNOWN,
    W06_R01_OUTCOME_REFUTE,
    W06_R01_OUTCOME_SUPPORT,
    W06_R01_REASONING_CONFLICT,
    W06_R01_REASONING_REFUTED,
    W06_R01_REASONING_SUPPORTED,
    W06_R01_REASONING_UNRESOLVED,
    W06_R01_UNDERSTANDING_CONFLICT,
    W06_R01_UNDERSTANDING_UNIQUE,
    W06_R01_UNDERSTANDING_UNKNOWN,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"


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


def _build(adapted, protocol=W06R01ConsumerProtocol()):
    """建立只含 R01 train truth 的共享 W-06 learning/runtime。"""
    sliced = slice_w06_r01_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R01Runtime(
        learning, sliced, protocol=protocol)


def _roles(candidate):
    """返回 relation family 对应的冻结起终 Role。"""
    return (
        (ROLE_ALIAS_LEFT, ROLE_ALIAS_RIGHT)
        if candidate.relation_family == "PURE_ALIAS"
        else (ROLE_REFERS_FROM, ROLE_REFERS_TO)
    )


def _endpoints(candidate):
    """从 typed RoleBinding 读取关系端点，不按 surface 识别。"""
    values = []
    for role in _roles(candidate):
        identity = authored_relation_role_identity(role)
        values.append(next(
            item.filler
            for item in candidate.proposition.canonical_bindings()
            if item.role == identity
        ))
    return tuple(values)


def _active_candidates(sliced, learning):
    """按 family 返回当前 active alias 与 revised refers。"""
    active = tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )
    return (
        next(item for item in active if item.relation_family == "PURE_ALIAS"),
        next(item for item in active if item.relation_family == "REFERS"),
    )


def _constraints(candidate):
    """建立不含 expected surface 的目标语言和输出预算。"""
    branch = w06_r01_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _exercise_positive(runtime, candidates):
    """对两个 active family 分别执行 U/R/G exact Use 与 postcheck。"""
    outcomes = []
    for ordinal, candidate in enumerate(candidates, start=1):
        source, target = _endpoints(candidate)
        understanding = runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((1, ordinal)),
                source,
                (target.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        understanding_use = runtime.adopt_understanding(understanding)
        understanding_outcome = runtime.verify_understanding(understanding_use)

        reasoning = runtime.resolve_reasoning(W06R01ReasoningRequest(
            LosslessIntegerKey((2, ordinal)),
            candidate.relation_family,
            source,
            target,
        ))
        reasoning_use = runtime.adopt_reasoning(reasoning)
        reasoning_outcome = runtime.verify_reasoning(reasoning_use)

        generation = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((3, ordinal)),
            constraints=_constraints(candidate),
        ))
        generation_use = runtime.adopt_generation(
            generation, generation.options[0].stable_key())
        generation_outcome = runtime.verify_generation(generation_use)
        outcomes.append((
            understanding,
            understanding_use,
            understanding_outcome,
            reasoning,
            reasoning_use,
            reasoning_outcome,
            generation,
            generation_use,
            generation_outcome,
        ))
    return tuple(outcomes)


def test_w06_r01_slice_binds_only_stable_train_alias_refers(adapted):
    """R01 不得读取后续关系、held-out label 或旧 occurrence REFERS。"""
    sliced = slice_w06_r01_adapter(adapted)
    assert len(sliced.candidates) == 5
    assert len(sliced.evidence) == 5
    assert len(sliced.observations) == 5
    assert len(sliced.source_bindings) == 5
    assert sliced.rejections == ()
    assert sliced.rejection_evidence == ()
    assert {item.relation_family for item in sliced.candidates} == {
        "PURE_ALIAS", "REFERS"}
    assert {item.observation.split for item in sliced.candidates} == {"train"}
    assert all(
        item.substage_key == "PURE_ALIAS_REFERS"
        for item in sliced.candidates)
    assert all(
        endpoint.object_kind != 3
        for item in sliced.candidates for endpoint in item.endpoints)
    protocol = w06_r01_alias_protocol(sliced)
    assert len(protocol.alias_schemas) == 1
    assert len(protocol.refers_schemas) == 1
    assert set(protocol.alias_schemas + protocol.refers_schemas) == {
        item.schema.schema for item in sliced.candidates}


def test_w06_r01_positive_u_r_g_consumes_current_active_evidence(adapted):
    """alias 与 revised refers 都必须形成三向 exact Use 和独立 SUPPORT。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        candidates = _active_candidates(sliced, learning)
        outcomes = _exercise_positive(runtime, candidates)
        for candidate, values in zip(candidates, outcomes, strict=True):
            (understanding, understanding_use, understanding_outcome,
             reasoning, reasoning_use, reasoning_outcome,
             generation, generation_use, generation_outcome) = values
            _source, target = _endpoints(candidate)
            assert understanding.status == W06_R01_UNDERSTANDING_UNIQUE
            assert understanding.selected == target
            assert understanding_outcome.verdict == W06_R01_OUTCOME_SUPPORT
            assert understanding_use.alias_use.relation_uses
            assert reasoning.status == W06_R01_REASONING_SUPPORTED
            assert reasoning_outcome.verdict == W06_R01_OUTCOME_SUPPORT
            assert reasoning_use.relation_uses
            assert generation.status == W06_R01_GENERATION_READY
            assert len(generation.options) == 1
            assert generation_outcome.verdict == W06_R01_OUTCOME_SUPPORT
            assert generation_outcome.recovered_target is True
            assert generation_outcome.surface_structure_valid is True
            assert generation_use.option.target_proposition == (
                candidate.proposition.proposition)
            assert not hasattr(generation.request, "expected_surface")
            assert not hasattr(generation.request, "expected_label")

        report = runtime.report()
        assert report.candidate_count == 5
        assert report.active_count == 2
        assert report.refuted_count == 1
        assert report.conflict_count == 1
        assert report.superseded_count == 1
        assert report.understanding_use_count == 2
        assert report.reasoning_use_count == 2
        assert report.generation_use_count == 2
        assert report.generation_outcome_count == 2
        assert report.explored_state_count == 2
        assert report.considered_fact_count == 2
        assert report.route_count == 2
        assert report.private_read_count == 0
        assert report.formal_guard_read_count == 0
        assert report.future_relation_claim_count == 0
        assert report.w06_started == 0
    finally:
        backend.close()


def test_w06_r01_current_view_separates_supersede_refute_and_conflict(adapted):
    """历史冲突不得污染 revised current fact，反向与真实冲突保持分态。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        original = next(
            item for item in sliced.candidates
            if item.relation_family == "REFERS"
            and item.sample_role == "support")
        revised = next(
            item for item in sliced.candidates
            if item.relation_family == "REFERS"
            and item.sample_role == "supersede")
        reversed_value = next(
            item for item in sliced.candidates
            if item.perturbation_kind == "DIRECTION_REVERSAL")
        conflict_value = next(
            item for item in sliced.candidates
            if item.sample_role == "conflict")

        original_snapshot = learning.snapshot_for(
            original.proposition.proposition)
        assert original_snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert original_snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
        source, target = _endpoints(revised)
        current = runtime.resolve_understanding(W06R01UnderstandingRequest(
            LosslessIntegerKey((10, 1)),
            source,
            (target.object_kind,),
            AliasRouteSearchBudget(16, 16, 16),
            False,
        ))
        assert current.status == W06_R01_UNDERSTANDING_UNIQUE
        assert current.propositions == (revised.proposition.proposition,)

        reversed_source, reversed_target = _endpoints(reversed_value)
        refuted = runtime.resolve_reasoning(W06R01ReasoningRequest(
            LosslessIntegerKey((10, 2)),
            "REFERS",
            reversed_source,
            reversed_target,
        ))
        assert refuted.status == W06_R01_REASONING_REFUTED
        assert learning.snapshot_for(
            reversed_value.proposition.proposition,
        ).snapshot.epistemic_status == EPISTEMIC_REFUTED

        conflict_source, conflict_target = _endpoints(conflict_value)
        conflict = runtime.resolve_reasoning(W06R01ReasoningRequest(
            LosslessIntegerKey((10, 3)),
            "REFERS",
            conflict_source,
            conflict_target,
        ))
        assert conflict.status == W06_R01_REASONING_CONFLICT
        conflict_understanding = runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((10, 4)),
                conflict_source,
                (conflict_target.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        assert conflict_understanding.status == W06_R01_UNDERSTANDING_CONFLICT
        assert conflict_understanding.selected is None
    finally:
        backend.close()


def test_w06_r01_alias_is_symmetric_but_refers_is_not_reversed(adapted):
    """PURE_ALIAS 可双向消费，REFERS 的反向 active route 必须缺失。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        alias, refers = _active_candidates(sliced, learning)
        alias_left, alias_right = _endpoints(alias)
        reverse_alias = runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((20, 1)),
                alias_right,
                (alias_left.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        assert reverse_alias.status == W06_R01_UNDERSTANDING_UNIQUE
        assert reverse_alias.selected == alias_left

        refers_from, refers_to = _endpoints(refers)
        reverse_refers = runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((20, 2)),
                refers_to,
                (refers_from.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        assert reverse_refers.status == W06_R01_UNDERSTANDING_UNKNOWN
    finally:
        backend.close()


def test_w06_r01_structural_unknown_does_not_read_held_out_labels(adapted):
    """内容替换、逆结构和伪 relation 只做无标签 fail-closed 探针。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        alias, refers = _active_candidates(sliced, learning)
        unknown_left = concept_identity((50601, 990, 1))
        unknown_right = concept_identity((50601, 990, 2))
        unknown = runtime.resolve_reasoning(W06R01ReasoningRequest(
            LosslessIntegerKey((30, 1)),
            "PURE_ALIAS",
            unknown_left,
            unknown_right,
        ))
        assert unknown.status == W06_R01_REASONING_UNRESOLVED
        with pytest.raises(W06R01ContractError, match="未注册"):
            W06R01ReasoningRequest(
                LosslessIntegerKey((30, 2)),
                "SUBSET",
                unknown_left,
                unknown_right,
            )

        generation_request = generation_request_for_candidate(
            refers,
            request_key=LosslessIntegerKey((30, 3)),
            constraints=_constraints(refers),
        )
        swapped = tuple(
            sorted(
                (
                    (generation_request.role_fillers[0][0],
                     generation_request.role_fillers[1][1]),
                    (generation_request.role_fillers[1][0],
                     generation_request.role_fillers[0][1]),
                ),
                key=lambda item: item[0].stable_key(),
            ))
        reversed_choice = runtime.choose_generation(replace(
            generation_request,
            request_key=LosslessIntegerKey((30, 4)),
            role_fillers=swapped,
        ))
        assert reversed_choice.status == W06_R01_GENERATION_UNKNOWN
        later = next(
            item for item in adapted.candidates
            if item.substage_key == "SUBSET_MEMBER")
        with pytest.raises(W06R01ContractError, match="不属于 R01"):
            generation_request_for_candidate(
                later,
                request_key=LosslessIntegerKey((30, 5)),
                constraints=_constraints(alias),
            )
    finally:
        backend.close()


def test_w06_r01_target_and_generation_ablations_are_orthogonal(adapted):
    """关闭 R01 bridge 击穿三向；只关 generation 不得击穿 U/R。"""
    target_backend, sliced, learning, target_runtime = _build(
        adapted,
        W06R01ConsumerProtocol(alias_refers_bridge_connected=False),
    )
    try:
        alias, _refers = _active_candidates(sliced, learning)
        source, target = _endpoints(alias)
        understanding = target_runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((40, 1)),
                source,
                (target.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        reasoning = target_runtime.resolve_reasoning(W06R01ReasoningRequest(
            LosslessIntegerKey((40, 2)), "PURE_ALIAS", source, target))
        generation = target_runtime.choose_generation(
            generation_request_for_candidate(
                alias,
                request_key=LosslessIntegerKey((40, 3)),
                constraints=_constraints(alias),
            ))
        assert understanding.status == W06_R01_UNDERSTANDING_UNKNOWN
        assert reasoning.status == W06_R01_REASONING_UNRESOLVED
        assert generation.status == W06_R01_GENERATION_UNKNOWN
        assert target_runtime.report().future_relation_claim_count == 0
    finally:
        target_backend.close()

    generation_backend, sliced, learning, generation_runtime = _build(
        adapted,
        W06R01ConsumerProtocol(generation_connected=False),
    )
    try:
        alias, _refers = _active_candidates(sliced, learning)
        source, target = _endpoints(alias)
        understanding = generation_runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((41, 1)),
                source,
                (target.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        reasoning = generation_runtime.resolve_reasoning(
            W06R01ReasoningRequest(
                LosslessIntegerKey((41, 2)), "PURE_ALIAS", source, target))
        generation = generation_runtime.choose_generation(
            generation_request_for_candidate(
                alias,
                request_key=LosslessIntegerKey((41, 3)),
                constraints=_constraints(alias),
            ))
        assert understanding.status == W06_R01_UNDERSTANDING_UNIQUE
        assert reasoning.status == W06_R01_REASONING_SUPPORTED
        assert generation.status == W06_R01_GENERATION_UNKNOWN
    finally:
        generation_backend.close()


def test_w06_r01_postcheck_ablation_refutes_generation_only(adapted):
    """关闭 postcheck 后 option 可形成，但不能伪装 generation SUPPORT。"""
    backend, sliced, learning, runtime = _build(
        adapted,
        W06R01ConsumerProtocol(postcheck_connected=False),
    )
    try:
        alias, _refers = _active_candidates(sliced, learning)
        source, target = _endpoints(alias)
        understanding = runtime.resolve_understanding(
            W06R01UnderstandingRequest(
                LosslessIntegerKey((50, 1)),
                source,
                (target.object_kind,),
                AliasRouteSearchBudget(16, 16, 16),
                False,
            ))
        assert understanding.status == W06_R01_UNDERSTANDING_UNIQUE
        choice = runtime.choose_generation(generation_request_for_candidate(
            alias,
            request_key=LosslessIntegerKey((50, 2)),
            constraints=_constraints(alias),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R01_OUTCOME_REFUTE
        assert outcome.understanding_status == W06_R01_UNDERSTANDING_UNKNOWN
        assert outcome.authorization_current is True
        assert outcome.relation_structure_preserved is True
    finally:
        backend.close()


def test_w06_r01_withdrawal_demotes_route_and_refutes_stale_use(adapted):
    """R01 withdrawal 必须保留 Evidence 历史并使旧选择退出 current view。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        alias, _refers = _active_candidates(sliced, learning)
        source, target = _endpoints(alias)
        before = runtime.resolve_understanding(W06R01UnderstandingRequest(
            LosslessIntegerKey((60, 1)),
            source,
            (target.object_kind,),
            AliasRouteSearchBudget(16, 16, 16),
            False,
        ))
        use = runtime.adopt_understanding(before)
        assert runtime.verify_understanding(use).verdict == (
            W06_R01_OUTCOME_SUPPORT)
        application = next(
            item for item in learning.applications()
            if item.binding.candidate == alias.proposition.proposition)
        account = next(
            item for item in application.accounts
            if item.stance == EVIDENCE_SUPPORT and not item.derived_supersede)
        prior = account.trace.outcome.evidence
        withdrawal = learning.withdraw_evidence(account, withdrawal_level=1)
        assert withdrawal.evidence.supersedes_evidence_id == prior.evidence_id

        after = runtime.resolve_understanding(W06R01UnderstandingRequest(
            LosslessIntegerKey((60, 2)),
            source,
            (target.object_kind,),
            AliasRouteSearchBudget(16, 16, 16),
            False,
        ))
        assert after.status == W06_R01_UNDERSTANDING_UNKNOWN
        stale = runtime.verify_understanding(use)
        assert stale.verdict == W06_R01_OUTCOME_REFUTE
        assert stale.current_status == W06_R01_UNDERSTANDING_UNKNOWN
        snapshot = learning.snapshot_for(alias.proposition.proposition)
        assert snapshot.active_fact is None
        assert snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        history = learning.learning.engine.ledger.evidence_history(
            snapshot.formation.hypothesis)
        assert prior in history
        assert withdrawal.evidence in history
    finally:
        backend.close()


def test_w06_r01_replay_is_bit_identical(adapted):
    """相同 public train 与三向请求必须得到 bit-identical state/report。"""
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
