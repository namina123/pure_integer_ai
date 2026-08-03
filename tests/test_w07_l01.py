"""W07-L01 NOT public bounded closure 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.scope_identity import query_scope
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
from pure_integer_ai.experiments.ph2_w07_l01 import (
    build_w07_l01_runtime,
    generation_request_for_proposal,
    logic_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicBudget,
    W07LogicConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    w07_logic_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapted():
    """唯一读取 public train firewall；不触及 private/formal root。"""
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
        payload = W07PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
        return adapt_w07_training_payload(payload)
    finally:
        backend.close()


def _build(adapted, *, protocol=None):
    backend = DictBackend()
    runtime = build_w07_l01_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


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


def _proposal(runtime, *, perturbation="NONE", state=None):
    for item in runtime.view.executable_proposals("NOT"):
        raw = item.observation.typed_payload.to_value()
        operand = raw["operand_evidence"][0]
        child_state = (operand["support"], operand["refute"])
        if (item.observation.perturbation_kind == perturbation
                and (state is None or state == child_state)):
            return item
    raise AssertionError("缺少目标 NOT proposal")


def test_l01_slice_learns_one_not_profile_without_future_substage(adapted):
    """只形成 NOT，多个来源化 adoption 聚合为同一 learned definition。"""
    backend, runtime = _build(adapted)
    try:
        assert {item.observation.substage for item in runtime.proposals} == {"NOT"}
        assert len(runtime.proposals) == 9
        assert len(runtime.adapter.specs) == 9
        assert len(runtime.learning.active_specs()) == 4
        definitions = {
            item.definition.stable_key() for item in runtime.learning.active_specs()}
        assert len(definitions) == 1
        assert runtime.adapter.rejections == ()
        report = runtime.report()
        assert report.candidate_count == 9
        assert report.active_candidate_count == 4
        assert report.executable_proposal_count == 7
        assert (
            report.supported_count,
            report.refuted_count,
            report.unknown_count,
            report.conflict_count,
        ) == (2, 2, 2, 1)
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert report.future_substage_claim_count == report.w07_started == 0
        assert bytes(report.operator_digest).hex() == (
            "b11d5a5b36e4bbacbac2cb7fc679d6d019364e3b60aae166242e434ef6ba3329")
        assert bytes(report.source_evidence_digest).hex() == (
            "264a9d23d401f1b7b0b7776525da8fbbe5733573a0db6a27e50f207883626f67")
        assert bytes(report.execution_digest).hex() == (
            "1b3fc59ab180d68d47c8f9a51a3cbb03a94d93953eb954e20c1407aaadaf14ca")
    finally:
        backend.close()


def test_not_executes_four_state_table_with_separate_operator_content_evidence(
        adapted):
    """NOT 真交换 support/refute，并保持 unknown/conflict 与 Evidence 分账。"""
    backend, runtime = _build(adapted)
    try:
        observed = set()
        for index, proposal in enumerate(
                runtime.view.executable_proposals("NOT"), start=1):
            raw = proposal.observation.typed_payload.to_value()
            child = raw["operand_evidence"][0]
            execution = runtime.view.execute(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((1, index))))
            assert execution is not None
            expected = (
                (child["support"], child["refute"])
                if proposal.observation.perturbation_kind == "DOUBLE_NEGATION"
                else (child["refute"], child["support"])
            )
            assert execution.evaluation.state.stable_key() == expected
            assert execution.evaluation.derivation
            assert execution.operator_adoptions
            assert execution.operator_premise_keys
            assert execution.content_premise_keys
            assert set(execution.operator_premise_keys).isdisjoint(
                execution.content_premise_keys)
            observed.add(execution.evaluation.state.stable_key())
        assert observed == {(1, 0), (0, 1), (0, 0), (1, 1)}
    finally:
        backend.close()


def test_understanding_reasoning_generation_are_independent_exact_uses(adapted):
    """U/R/G 分别执行、记 Use；generation postcheck 恢复结构和四态。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(0, 1))
        u_request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((10, 1)))
        r_request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((20, 1)))
        u_resolution = runtime.resolve_understanding(u_request)
        r_resolution = runtime.resolve_reasoning(r_request)
        assert u_resolution.status == r_resolution.status == "SUPPORTED"
        u_use = runtime.adopt_understanding(u_resolution)
        r_use = runtime.adopt_reasoning(r_resolution)
        assert u_use.use_key != r_use.use_key
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"

        request = generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((30, 1)),
            logic_request_key=LosslessIntegerKey((31, 1)),
            constraints=_constraints(proposal),
        )
        assert not hasattr(request, "expected_surface")
        assert not hasattr(request, "label")
        choice = runtime.choose_generation(request)
        assert choice.status == "READY"
        use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == "SUPPORT"
        assert all((
            outcome.adoption_current,
            outcome.structure_preserved,
            outcome.role_order_preserved,
            outcome.state_preserved,
            outcome.source_scope_preserved,
            outcome.surface_valid,
            outcome.recovered_target,
        ))
        report = runtime.report()
        assert (
            report.understanding_use_count,
            report.reasoning_use_count,
            report.generation_choice_count,
            report.generation_use_count,
            report.generation_outcome_count,
        ) == (1, 1, 1, 1, 1)
    finally:
        backend.close()


def test_double_negation_is_two_direct_steps_without_rewrite(adapted):
    """DOUBLE_NEGATION 保留两层 trace，不物化自动约简规则。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, perturbation="DOUBLE_NEGATION")
        execution = runtime.view.execute(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((40, 1))))
        assert execution is not None
        assert len(execution.evaluation.derivation) == 2
        assert [item.operator for item in execution.evaluation.derivation] == [
            proposal.bound_root.structure,
            proposal.bound_root.structure,
        ]
        assert runtime.report().double_negation_rewrite_count == 0
    finally:
        backend.close()


def test_no_adoption_scope_flip_target_and_generation_ablations_fail_closed(
        adapted):
    """cue/handler、错 scope、目标与 generation/postcheck 消融不能冒充 PASS。"""
    disabled = W07LogicConsumerProtocol(("NOT",), ("NOT",))
    backend, runtime = _build(adapted, protocol=disabled)
    try:
        proposal = runtime.proposals[0]
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((50, 1)))
        assert runtime.resolve_understanding(request).status == "NO_ADOPTION"
    finally:
        backend.close()

    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime)
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((50, 2)))
        flipped = replace(request, scope=query_scope(99, parent=request.scope))
        assert runtime.understanding.preview(flipped).status == "NO_ADOPTION"
        pseudo = next(
            item for item in runtime.proposals
            if item.observation.perturbation_kind == "PSEUDO_OPERATOR")
        pseudo_request = logic_request_for_proposal(
            pseudo, request_key=LosslessIntegerKey((50, 3)))
        assert runtime.understanding.preview(pseudo_request).status == (
            "NO_ADOPTION")
    finally:
        backend.close()

    no_generation = W07LogicConsumerProtocol(
        ("NOT",), (), True, True, False, True)
    backend, runtime = _build(adapted, protocol=no_generation)
    try:
        proposal = _proposal(runtime)
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((50, 4)),
            logic_request_key=LosslessIntegerKey((50, 5)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "UNKNOWN"
    finally:
        backend.close()

    no_postcheck = W07LogicConsumerProtocol(
        ("NOT",), (), True, True, True, False)
    backend, runtime = _build(adapted, protocol=no_postcheck)
    try:
        proposal = _proposal(runtime)
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((50, 6)),
            logic_request_key=LosslessIntegerKey((50, 7)),
            constraints=_constraints(proposal),
        ))
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        assert runtime.verify_generation(use).verdict == "REFUTE"
    finally:
        backend.close()


def test_withdrawal_invalidates_stale_u_r_g_use_but_preserves_history(adapted):
    """append-only withdrawal 使旧 exact Use 失效，旧 Evidence 仍在 history。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(0, 1))
        u_resolution = runtime.resolve_understanding(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((60, 1))))
        r_resolution = runtime.resolve_reasoning(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((60, 2))))
        u_use = runtime.adopt_understanding(u_resolution)
        r_use = runtime.adopt_reasoning(r_resolution)
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((60, 3)),
            logic_request_key=LosslessIntegerKey((60, 4)),
            constraints=_constraints(proposal),
        ))
        g_use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        account = next(
            account
            for application in runtime.learning.applications()
            if application.binding.proposal == proposal
            for account in application.accounts
            if account.stance == EVIDENCE_SUPPORT
            and not account.derived_supersede)
        before = runtime.learning.learning.engine.ledger.evidence_history(
            account.outcome.evidence.hypothesis)
        runtime.learning.withdraw_evidence(account, withdrawal_level=1)
        after = runtime.learning.learning.engine.ledger.evidence_history(
            account.outcome.evidence.hypothesis)
        assert len(after) == len(before) + 1
        assert runtime.verify_understanding(u_use).verdict == "REFUTE"
        assert runtime.verify_reasoning(r_use).verdict == "REFUTE"
        assert runtime.verify_generation(g_use).verdict == "REFUTE"
    finally:
        backend.close()


def test_budget_and_replay_are_fail_closed_and_bit_identical(adapted):
    """递归预算收紧会失败；两个 clean owner 的 report/state 完全一致。"""
    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        proposal = _proposal(left, perturbation="DOUBLE_NEGATION")
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((70, 1)))
        with pytest.raises(RuntimeError, match="max_depth.*超限"):
            left.view.execute(replace(
                request,
                budget=W07LogicBudget(
                    max_depth=1,
                    max_branches=32,
                    max_steps=128,
                    max_resolver_calls=32,
                ),
            ))
        assert left.report() == right.report()
        assert left.state_key() == right.state_key()
    finally:
        left_backend.close()
        right_backend.close()
