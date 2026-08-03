"""W07-L06 MODAL public bounded closure 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_adapter import adapt_w07_training_payload
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.experiments.ph2_w07_l06 import (
    W07_L06_PREFIX,
    build_w07_l06_runtime,
    generation_request_for_proposal,
    logic_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicBudget,
    W07LogicConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    role_tree_key,
    w07_logic_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapted():
    backend = DictBackend()
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        request = W07RunRequest(
            W07_FORMAL_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_STAGE_KEY,
            context.owner_key,
            W07_RUNNER_KEY,
            context.baseline_commit_sha1,
            context.stable_key(),
            context.backend_profile_key,
            context.base_fence_key,
            1,
            "fresh",
            tuple(sorted(W07_RESOURCE_BUDGET.items())),
            tuple(item.relative_path
                  for item in context.candidate_payload_bindings),
            tuple(item.relative_path
                  for item in context.teacher_evidence_bindings),
        )
        return adapt_w07_training_payload(W07PayloadFirewall.open(
            ROOT, context, request).read_training_payload())
    finally:
        backend.close()


def _build(adapted, *, protocol=None):
    backend = DictBackend()
    runtime = build_w07_l06_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, *, perturbation="NONE", state=None):
    for index, proposal in enumerate(runtime.proposals, start=1):
        if proposal.observation.perturbation_kind != perturbation:
            continue
        execution = runtime.execute(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((850, index))))
        if state is None or (
                execution is not None
                and execution.evaluation.state.stable_key() == state):
            return proposal
    raise LookupError("未找到 MODAL proposal")


def _constraints(proposal):
    branch = w07_logic_language_branch(proposal)
    return GenerationExpressionConstraints(
        branch,
        (proposal.bound_root.structure,),
        (branch,),
        0,
        0,
        0,
        256,
    )


def test_l06_prefix_and_modal_plan_identity(adapted):
    """L06 只打开严格 prefix，每个 MODAL 保留独立 resolver plan。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L06_PREFIX
        assert len(runtime.proposals) == 10
        assert len(runtime.adapter.rejections) == 2
        for proposal in runtime.proposals:
            plan = runtime.modal_plan_for(proposal)
            assert plan.source == proposal.source_binding.source_ref
            assert plan.input_scope == proposal.request_scope
            if plan.status == "RESOLVED":
                assert plan.output_scope is not None
                assert plan.evidence_ids
            else:
                assert plan.output_scope is None
                assert plan.evidence_ids == ()
        report = runtime.report()
        assert (
            report.candidate_count,
            report.active_candidate_count,
            report.operator_profile_count,
            report.executable_proposal_count,
            report.resolved_count,
            report.unresolved_count,
        ) == (10, 4, 1, 9, 7, 3)
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert report.reality_fact_count == 0
    finally:
        backend.close()


def test_modal_resolver_preserves_four_states_and_output_scope(adapted):
    """resolved plan 的四态/source/output scope 原样进入 S-04，非现实 fact。"""
    backend, runtime = _build(adapted)
    try:
        observed = set()
        for index, proposal in enumerate(runtime.proposals, start=1):
            plan = runtime.modal_plan_for(proposal)
            if plan.status != "RESOLVED":
                continue
            execution = runtime.execute(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((860, index))))
            if execution is None:
                continue
            assert execution.evaluation.state == plan.state
            assert execution.evaluation.source == plan.source
            assert execution.evaluation.scope == plan.output_scope
            assert set(plan.evidence_ids).issubset(execution.evaluation.evidence_ids)
            observed.add(plan.state.stable_key())
        assert observed == {(1, 0), (0, 1), (0, 0), (1, 1)}
    finally:
        backend.close()


def test_missing_denied_and_budget_resolvers_are_direct_unknown(adapted):
    """缺 resolver/certificate 直接 UNKNOWN，不先生成 child derivation。"""
    backend, runtime = _build(adapted)
    try:
        for index, kind in enumerate((
                "RESOLVER_MISSING", "RESOLVER_DENIED", "BUDGET_UNDECIDED"
                ), start=1):
            proposal = _proposal(runtime, perturbation=kind)
            execution = runtime.execute(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((870, index))))
            assert execution is not None
            assert execution.evaluation.state.stable_key() == (0, 0)
            assert execution.evaluation.derivation == ()
            assert execution.evaluation.evidence_ids == ()
            assert execution.evaluation.failures
            assert runtime.understanding.preview(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((871, index)),
            )).status == "UNKNOWN"
    finally:
        backend.close()


def test_modal_urg_generation_scope_and_family_ablation(adapted):
    """U/R/G exact Use 与 generation modal tree/scope，family ablation 正交。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((880, 1)))
        u_use = runtime.adopt_understanding(runtime.resolve_understanding(request))
        r_use = runtime.adopt_reasoning(runtime.resolve_reasoning(
            replace(request, request_key=LosslessIntegerKey((880, 2)))))
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((880, 3)),
            logic_request_key=LosslessIntegerKey((880, 4)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "READY"
        option = choice.options[0]
        assert option.operator_families == ("MODAL",)
        assert option.role_tree_key == role_tree_key(
            proposal.bound_root, include_bound_provenance=True)
        assert option.scope == runtime.modal_plan_for(proposal).output_scope
        assert runtime.verify_generation(runtime.adopt_generation(
            choice, option.stable_key())).verdict == "SUPPORT"
    finally:
        backend.close()

    disabled = W07LogicConsumerProtocol(
        W07_L06_PREFIX, disabled_operator_families=("MODAL",))
    backend, disabled_runtime = _build(adapted, protocol=disabled)
    try:
        proposal = next(item for item in disabled_runtime.proposals
                        if item.observation.perturbation_kind == "NONE")
        assert disabled_runtime.understanding.preview(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((881, 1)))).status == "NO_ADOPTION"
    finally:
        backend.close()


def test_modal_scope_shift_withdrawal_budget_and_replay(adapted):
    """scope shift 不冒充现实、预算/withdrawal/replay 全部 fail closed。"""
    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.state_key() == right.state_key()
        assert left.report() == right.report()
        shifted = _proposal(left, perturbation="MODAL_SCOPE_SHIFT")
        shifted_execution = left.execute(logic_request_for_proposal(
            shifted, request_key=LosslessIntegerKey((890, 1))))
        assert shifted_execution is not None
        assert shifted_execution.evaluation.scope == left.modal_plan_for(
            shifted).output_scope
        proposal = _proposal(left, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((890, 2)))
        with pytest.raises(RuntimeError, match="max_steps.*超限"):
            left.execute(replace(request, budget=W07LogicBudget(8, 32, 1, 32)))
        resolution = left.resolve_understanding(request)
        use = left.adopt_understanding(resolution)
        account = next(
            account for application in left.learning.applications()
            if application.binding.proposal == proposal
            for account in application.accounts
            if account.stance == EVIDENCE_SUPPORT and not account.derived_supersede)
        left.learning.withdraw_evidence(account, withdrawal_level=1)
        assert left.verify_understanding(use).verdict == "REFUTE"
    finally:
        left_backend.close()
        right_backend.close()
