"""W07-L05 FORALL public bounded closure 专项。"""
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
from pure_integer_ai.experiments.ph2_w07_l05 import (
    W07_L05_PREFIX,
    build_w07_l05_runtime,
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
    runtime = build_w07_l05_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, *, perturbation="NONE", state=None):
    for index, proposal in enumerate(
            runtime.view.executable_proposals("FORALL"), start=1):
        if proposal.observation.perturbation_kind != perturbation:
            continue
        execution = runtime.execute(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((750, index))))
        if state is None or (
                execution is not None
                and execution.evaluation.state.stable_key() == state):
            return proposal
    raise LookupError("未找到 FORALL proposal")


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


def test_l05_prefix_and_forall_typed_identity(adapted):
    """L05 只打开严格 prefix，Binder/Variable/domain/body 均不丢失。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L05_PREFIX
        assert len(runtime.proposals) == 8
        assert len(runtime.adapter.rejections) == 2
        for proposal in runtime.proposals:
            quantifier = runtime.quantifier_for(proposal)
            assert quantifier.definition.binder in proposal.bound_root.introduced_binders
            assert quantifier.definition.body_slot == proposal.specs[0].definition.slots[0]
            assert quantifier.value_role in {
                binding.role for binding in proposal.bound_root.bindings[0].filler.bindings
            }
        report = runtime.report()
        assert (
            report.candidate_count,
            report.active_candidate_count,
            report.operator_profile_count,
            report.executable_proposal_count,
            report.branch_count,
        ) == (8, 4, 1, 6, 8)
        assert report.private_read_count == report.formal_guard_read_count == 0
    finally:
        backend.close()


def test_forall_support_requires_closed_domain_and_counterexample_refutes(
        adapted):
    """全支持开放域保持 UNKNOWN；显式单 counterexample 可受限 REFUTE。"""
    backend, runtime = _build(adapted)
    try:
        closed_support = _proposal(runtime, state=(1, 0))
        closed_execution = runtime.execute(logic_request_for_proposal(
            closed_support, request_key=LosslessIntegerKey((760, 1))))
        assert closed_execution is not None
        assert closed_execution.evaluation.state.stable_key() == (1, 0)
        open_support = _proposal(runtime, perturbation="DOMAIN_CLOSURE_CONFUSION")
        open_execution = runtime.execute(logic_request_for_proposal(
            open_support, request_key=LosslessIntegerKey((760, 2))))
        assert open_execution is not None
        assert open_execution.evaluation.state.stable_key() == (0, 0)
        counterexample = _proposal(runtime, state=(0, 1))
        counterexample_execution = runtime.execute(logic_request_for_proposal(
            counterexample, request_key=LosslessIntegerKey((760, 3))))
        assert counterexample_execution is not None
        assert counterexample_execution.evaluation.state.stable_key() == (0, 1)
        assert any(branch.state.refute
                   for branch in counterexample_execution.evaluation.branches)
    finally:
        backend.close()


def test_forall_vacuous_truth_and_four_state_branch_trace(adapted):
    """显式 closed empty domain 才允许 vacuous support，conflict 保留。"""
    backend, runtime = _build(adapted)
    try:
        empty = _proposal(runtime, perturbation="EMPTY_DOMAIN_CONFUSION")
        empty_execution = runtime.execute(logic_request_for_proposal(
            empty, request_key=LosslessIntegerKey((770, 1))))
        assert empty_execution is not None
        assert empty_execution.evaluation.branches == ()
        assert empty_execution.evaluation.state.stable_key() == (1, 0)
        conflict = _proposal(runtime, state=(1, 1))
        conflict_execution = runtime.execute(logic_request_for_proposal(
            conflict, request_key=LosslessIntegerKey((770, 2))))
        assert conflict_execution is not None
        assert conflict_execution.evaluation.state.stable_key() == (1, 1)
        assert len(conflict_execution.evaluation.branches) == 1
    finally:
        backend.close()


def test_forall_urg_generation_scope_and_ablations(adapted):
    """U/R/G exact Use 与 generation tree/source/scope，family 消融正交。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((780, 1)))
        u_use = runtime.adopt_understanding(runtime.resolve_understanding(request))
        r_use = runtime.adopt_reasoning(runtime.resolve_reasoning(
            replace(request, request_key=LosslessIntegerKey((780, 2)))))
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((780, 3)),
            logic_request_key=LosslessIntegerKey((780, 4)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "READY"
        option = choice.options[0]
        assert option.operator_families == ("FORALL",)
        assert option.role_tree_key == role_tree_key(
            proposal.bound_root, include_bound_provenance=True)
        assert runtime.verify_generation(runtime.adopt_generation(
            choice, option.stable_key())).verdict == "SUPPORT"
    finally:
        backend.close()

    disabled = W07LogicConsumerProtocol(
        W07_L05_PREFIX, disabled_operator_families=("FORALL",))
    backend, disabled_runtime = _build(adapted, protocol=disabled)
    try:
        proposal = next(item for item in disabled_runtime.proposals
                        if item.observation.perturbation_kind == "NONE")
        assert disabled_runtime.understanding.preview(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((781, 1)))).status == "NO_ADOPTION"
    finally:
        backend.close()


def test_forall_counterexample_withdrawal_budget_and_replay(adapted):
    """预算、counterexample withdrawal 和双 owner replay 均保持 fail closed。"""
    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.state_key() == right.state_key()
        assert left.report() == right.report()
        proposal = _proposal(left, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((790, 1)))
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
