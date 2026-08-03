"""W07-L04 EXISTS public bounded closure 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_adapter import (
    adapt_w07_training_payload,
)
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
from pure_integer_ai.experiments.ph2_w07_l04 import (
    W07_L04_PREFIX,
    build_w07_l04_runtime,
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
    runtime = build_w07_l04_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, *, perturbation="NONE", state=None):
    for index, proposal in enumerate(
            runtime.view.executable_proposals("EXISTS"), start=1):
        if proposal.observation.perturbation_kind != perturbation:
            continue
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((700, index)))
        execution = runtime.execute(request)
        if state is None or (
                execution is not None
                and execution.evaluation.state.stable_key() == state):
            return proposal
    raise LookupError("未找到 EXISTS proposal")


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


def test_l04_prefix_and_quantifier_identity_are_closed(adapted):
    """L04 只打开严格 prefix，且每个 proposal 保留量词 typed identity。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L04_PREFIX
        assert len(runtime.proposals) == 8
        assert len(runtime.adapter.rejections) == 1
        for proposal in runtime.proposals:
            quantifier = runtime.quantifier_for(proposal)
            assert quantifier.definition.binder in proposal.bound_root.introduced_binders
            assert quantifier.definition.variable.object_kind
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
        ) == (8, 4, 1, 6)
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert report.w07_started == 0
    finally:
        backend.close()


def test_exists_branch_trace_keeps_witness_and_four_states(adapted):
    """每个分支保留 assignment/value/state；聚合遵循开放域 EXISTS 公式。"""
    backend, runtime = _build(adapted)
    try:
        observed = set()
        for index, proposal in enumerate(
                runtime.view.executable_proposals("EXISTS"), start=1):
            request = logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((100, index)))
            execution = runtime.execute(request)
            assert execution is not None
            quantifier = runtime.quantifier_for(proposal)
            assert len(execution.evaluation.branches) == len(
                quantifier.definition.domain.values)
            assert tuple(item.ordinal for item in execution.evaluation.branches) == tuple(
                range(len(execution.evaluation.branches)))
            states = tuple(item.state for item in execution.evaluation.branches)
            expected = (
                (1, int(quantifier.definition.domain.closed
                         and all(item.refute for item in states)))
                if any(item.support for item in states)
                else ((0, 1) if quantifier.definition.domain.closed
                      and all(item.refute for item in states)
                      else (0, 0)))
            assert execution.evaluation.state.stable_key() == expected
            observed.add(expected)
            assert all(item.assignment is not None for item in execution.evaluation.branches)
            assert all(item.source == request.source and item.scope == request.scope
                       for item in execution.evaluation.branches)
        assert observed == {(1, 0), (0, 1), (0, 0), (1, 1)}
    finally:
        backend.close()


def test_open_domain_does_not_turn_missing_witness_into_false(adapted):
    """开放域全 refute 仍是 unknown；closed finite domain 才允许 refute。"""
    backend, runtime = _build(adapted)
    try:
        open_proposal = _proposal(runtime, perturbation="DOMAIN_CLOSURE_CONFUSION")
        open_execution = runtime.execute(logic_request_for_proposal(
            open_proposal, request_key=LosslessIntegerKey((200, 1))))
        assert open_execution is not None
        assert open_execution.evaluation.state.stable_key() == (0, 0)
        assert open_execution.evaluation.failures

        closed_proposal = _proposal(runtime, state=(0, 1))
        closed_execution = runtime.execute(logic_request_for_proposal(
            closed_proposal, request_key=LosslessIntegerKey((200, 2))))
        assert closed_execution is not None
        assert closed_execution.evaluation.state.stable_key() == (0, 1)
        empty = _proposal(runtime, perturbation="EMPTY_DOMAIN_CONFUSION")
        empty_execution = runtime.execute(logic_request_for_proposal(
            empty, request_key=LosslessIntegerKey((200, 3))))
        assert empty_execution is not None
        assert empty_execution.evaluation.state.stable_key() == (0, 1)
    finally:
        backend.close()


def test_exists_uses_are_separate_and_generation_restores_scope(adapted):
    """U/R/G 各自产生 exact Use，生成保留量词 tree/source/scope。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((300, 1)))
        u_use = runtime.adopt_understanding(runtime.resolve_understanding(request))
        r_use = runtime.adopt_reasoning(runtime.resolve_reasoning(
            replace(request, request_key=LosslessIntegerKey((300, 2)))))
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((300, 3)),
            logic_request_key=LosslessIntegerKey((300, 4)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "READY"
        option = choice.options[0]
        assert option.operator_families == ("EXISTS",)
        assert option.role_tree_key == role_tree_key(
            proposal.bound_root, include_bound_provenance=True)
        assert option.source == proposal.source_binding.source_ref
        assert option.scope == proposal.request_scope
        generation_use = runtime.adopt_generation(choice, option.stable_key())
        assert runtime.verify_generation(generation_use).verdict == "SUPPORT"
    finally:
        backend.close()


def test_exists_confusion_ablations_budget_withdrawal_and_replay(adapted):
    """operator confusion、consumer 消融、预算、withdrawal 和 replay 均 fail closed。"""
    backend, runtime = _build(adapted)
    try:
        confused = next(
            item for item in runtime.proposals
            if item.observation.perturbation_kind == "QUANTIFIER_SWAP")
        assert runtime.understanding.preview(logic_request_for_proposal(
            confused, request_key=LosslessIntegerKey((400, 1)))).status == "NO_ADOPTION"
        proposal = _proposal(runtime, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((400, 2)))
        with pytest.raises(RuntimeError, match="max_steps.*超限"):
            runtime.execute(replace(
                request,
                budget=W07LogicBudget(8, 32, 1, 32),
            ))
        resolution = runtime.resolve_understanding(request)
        use = runtime.adopt_understanding(resolution)
        account = next(
            account for application in runtime.learning.applications()
            if application.binding.proposal == proposal
            for account in application.accounts
            if account.stance == 1 and not account.derived_supersede)
        runtime.learning.withdraw_evidence(account, withdrawal_level=1)
        assert runtime.verify_understanding(use).verdict == "REFUTE"
    finally:
        backend.close()

    disabled = W07LogicConsumerProtocol(
        W07_L04_PREFIX, disabled_operator_families=("EXISTS",))
    backend, disabled_runtime = _build(adapted, protocol=disabled)
    try:
        proposal = next(item for item in disabled_runtime.proposals
                        if item.observation.perturbation_kind == "NONE")
        assert disabled_runtime.understanding.preview(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((401, 1)))).status == "NO_ADOPTION"
    finally:
        backend.close()

    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.state_key() == right.state_key()
        assert left.report() == right.report()
    finally:
        left_backend.close()
        right_backend.close()
