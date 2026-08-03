"""W07-L02 AND_OR public bounded closure 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
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
from pure_integer_ai.experiments.ph2_w07_l02 import (
    W07_L02_PREFIX,
    build_w07_l02_runtime,
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
    runtime = build_w07_l02_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, family, *, perturbation="NONE"):
    return next(
        item for item in runtime.view.executable_proposals("AND_OR")
        if item.operator_families == (family,)
        and item.observation.perturbation_kind == perturbation)


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


def test_l02_prefix_retains_not_and_learns_two_independent_profiles(adapted):
    """L02 owner 保留 L01，AND 与 OR 分别形成 profile。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L02_PREFIX
        assert len(runtime.proposals) == 11
        assert len(runtime.view.executable_proposals("NOT")) == 7
        report = runtime.report()
        assert (
            report.candidate_count,
            report.active_candidate_count,
            report.operator_profile_count,
            report.executable_proposal_count,
            report.and_execution_count,
            report.or_execution_count,
        ) == (11, 5, 2, 9, 5, 4)
        assert (
            report.supported_count,
            report.refuted_count,
            report.unknown_count,
            report.conflict_count,
        ) == (3, 2, 2, 2)
        assert report.algebraic_rewrite_count == 0
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert bytes(report.operator_digest).hex() == (
            "040e73610ed1b2e09f994fb6eaf55c2871fac5ede22dcd181a6b44ad6c7d8302")
        assert bytes(report.source_evidence_digest).hex() == (
            "0d9b4b409869f2c1e9a62b9a41c852f6b93ac9415cefc61dd456273fb6c79eef")
        assert bytes(report.execution_digest).hex() == (
            "883d4f231262145e507ca61d5076efd721868728831334ac4bf9db3750e0e384")
    finally:
        backend.close()


def test_and_or_execute_distinct_open_world_four_state_tables(adapted):
    """AND/OR 使用各自 handler，unknown/conflict 不被压成二值。"""
    backend, runtime = _build(adapted)
    try:
        seen = {"AND": set(), "OR": set()}
        for index, proposal in enumerate(
                runtime.view.executable_proposals("AND_OR"), start=1):
            states = tuple(
                (item["support"], item["refute"])
                for item in proposal.observation.typed_payload.to_value()[
                    "operand_evidence"])
            family = proposal.operator_families[0]
            expected = (
                (int(all(item[0] for item in states)),
                 int(any(item[1] for item in states)))
                if family == "AND"
                else (int(any(item[0] for item in states)),
                      int(all(item[1] for item in states)))
            )
            execution = runtime.view.execute(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((100, index))))
            assert execution is not None
            assert execution.evaluation.state.stable_key() == expected
            assert len(execution.evaluation.derivation) == 1
            seen[family].add(expected)
        assert seen["AND"] == {(1, 0), (0, 1), (0, 0), (1, 1)}
        assert seen["OR"] == {(1, 0), (0, 1), (0, 0), (1, 1)}
    finally:
        backend.close()


def test_and_or_each_form_independent_u_r_g_use_and_postcheck(adapted):
    """两个 profile 各自形成 U/R/G Use，不能由另一 operator 代替。"""
    backend, runtime = _build(adapted)
    try:
        for index, family in enumerate(("AND", "OR"), start=1):
            proposal = _proposal(runtime, family)
            u = runtime.resolve_understanding(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((200, index))))
            r = runtime.resolve_reasoning(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((210, index))))
            u_use = runtime.adopt_understanding(u)
            r_use = runtime.adopt_reasoning(r)
            assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
            assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
            choice = runtime.choose_generation(generation_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((220, index)),
                logic_request_key=LosslessIntegerKey((221, index)),
                constraints=_constraints(proposal),
            ))
            assert choice.status == "READY"
            assert choice.options[0].operator_families == (family,)
            use = runtime.adopt_generation(
                choice, choice.options[0].stable_key())
            assert runtime.verify_generation(use).verdict == "SUPPORT"
        report = runtime.report()
        assert (
            report.understanding_use_count,
            report.reasoning_use_count,
            report.generation_use_count,
            report.generation_outcome_count,
        ) == (2, 2, 2, 2)
    finally:
        backend.close()


def test_operand_order_is_preserved_even_when_semantics_are_commutative(adapted):
    """Role/ordinal/source provenance 保留，不因 AND/OR 可交换而重排。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, "OR", perturbation="OPERAND_ORDER_SWAP")
        original = proposal.bound_root
        assert len(original.bindings) == 2
        swapped = BoundProposition(
            original.template,
            original.instruction,
            original.predicate,
            original.structure,
            original.source_anchor,
            original.context,
            original.introduced_binders,
            tuple(BoundRoleBinding(
                binding.role,
                original.bindings[1 - index].filler,
                binding.ordinal,
            ) for index, binding in enumerate(original.bindings)),
            original.applied_variables,
        )
        assert role_tree_key(
            original, include_bound_provenance=True) != role_tree_key(
                swapped, include_bound_provenance=True)
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((300, 1)),
            logic_request_key=LosslessIntegerKey((300, 2)),
            constraints=_constraints(proposal),
        ))
        assert choice.options[0].role_tree_key == role_tree_key(
            original, include_bound_provenance=True)
        assert choice.options[0].role_tree_key != role_tree_key(
            swapped, include_bound_provenance=True)
    finally:
        backend.close()


def test_operator_confusion_and_orthogonal_family_ablations_fail_closed(adapted):
    """operator confusion 不执行，family 与 G/postcheck 消融保持正交。"""
    backend, runtime = _build(adapted)
    try:
        confused = next(
            item for item in runtime.proposals
            if item.observation.perturbation_kind == "OPERATOR_CONFUSION")
        assert runtime.understanding.preview(logic_request_for_proposal(
            confused,
            request_key=LosslessIntegerKey((400, 1)),
        )).status == "NO_ADOPTION"
    finally:
        backend.close()

    for disabled, survivor in (("AND", "OR"), ("OR", "AND")):
        protocol = W07LogicConsumerProtocol(
            W07_L02_PREFIX,
            disabled_operator_families=(disabled,),
        )
        backend, runtime = _build(adapted, protocol=protocol)
        try:
            target = next(
                item for item in runtime.proposals
                if item.operator_families == (disabled,)
                and item.observation.sample_role == "support")
            assert runtime.understanding.preview(logic_request_for_proposal(
                target,
                request_key=LosslessIntegerKey((410, 1)),
            )).status == "NO_ADOPTION"
            assert runtime.view.execute(logic_request_for_proposal(
                _proposal(runtime, survivor),
                request_key=LosslessIntegerKey((410, 2)),
            )) is not None
        finally:
            backend.close()

    no_generation = W07LogicConsumerProtocol(
        W07_L02_PREFIX, generation_connected=False)
    backend, runtime = _build(adapted, protocol=no_generation)
    try:
        proposal = _proposal(runtime, "AND")
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((420, 1)),
            logic_request_key=LosslessIntegerKey((420, 2)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "UNKNOWN"
    finally:
        backend.close()

    no_postcheck = W07LogicConsumerProtocol(
        W07_L02_PREFIX, postcheck_connected=False)
    backend, runtime = _build(adapted, protocol=no_postcheck)
    try:
        proposal = _proposal(runtime, "OR")
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((430, 1)),
            logic_request_key=LosslessIntegerKey((430, 2)),
            constraints=_constraints(proposal),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        assert runtime.verify_generation(use).verdict == "REFUTE"
    finally:
        backend.close()


def test_withdrawal_budget_and_replay_close_l02(adapted):
    """withdrawal 使 stale Use 失效，预算 fail closed，clean replay 相同。"""
    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.report() == right.report()
        assert left.state_key() == right.state_key()
        proposal = _proposal(left, "AND")
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((500, 1)))
        with pytest.raises(RuntimeError, match="max_steps.*超限"):
            left.view.execute(replace(
                request,
                budget=W07LogicBudget(
                    max_depth=8,
                    max_branches=32,
                    max_steps=2,
                    max_resolver_calls=32,
                ),
            ))
        resolution = left.resolve_understanding(request)
        use = left.adopt_understanding(resolution)
        account = next(
            account
            for application in left.learning.applications()
            if application.binding.proposal == proposal
            for account in application.accounts
            if account.stance == EVIDENCE_SUPPORT
            and not account.derived_supersede)
        before = left.learning.learning.engine.ledger.evidence_history(
            account.outcome.evidence.hypothesis)
        left.learning.withdraw_evidence(account, withdrawal_level=1)
        after = left.learning.learning.engine.ledger.evidence_history(
            account.outcome.evidence.hypothesis)
        assert len(after) == len(before) + 1
        assert left.verify_understanding(use).verdict == "REFUTE"
    finally:
        left_backend.close()
        right_backend.close()
