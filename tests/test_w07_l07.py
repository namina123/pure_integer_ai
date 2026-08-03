"""W07-L07 NESTED_SCOPE public bounded closure 专项。"""
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
from pure_integer_ai.experiments.ph2_w07_l07 import (
    W07_L07_PREFIX,
    build_w07_l07_runtime,
    generation_request_for_proposal,
    logic_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicBudget,
    W07LogicConsumerProtocol,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    role_tree_key,
    structure_tree_key,
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
    runtime = build_w07_l07_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, *, perturbation="NONE"):
    for proposal in runtime.proposals:
        if proposal.observation.perturbation_kind == perturbation:
            return proposal
    raise LookupError("未找到 nested proposal")


def _constraints(proposal):
    branch = w07_logic_language_branch(proposal)
    return GenerationExpressionConstraints(
        branch,
        tuple(item.definition.structure for item in proposal.specs),
        (branch,),
        0,
        0,
        0,
        256,
    )


def test_l07_prefix_layers_and_nested_identity(adapted):
    """L07 保留 ordered layers、父子 tree、scope 和各层 candidate。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L07_PREFIX
        assert len(runtime.proposals) == 8
        assert len(runtime.adapter.rejections) == 3
        for proposal in runtime.proposals:
            raw = proposal.observation.typed_payload.to_value()
            layers = raw["layers"]
            assert len(layers) == 2
            assert raw["derivation_order"] == [
                "inner-" + layers[-1]["operator_family"].lower(),
                "outer-" + layers[0]["operator_family"].lower(),
            ]
            assert len(proposal.specs) == len(layers)
            assert proposal.operator_families == tuple(
                layer["operator_family"] for layer in layers)
            assert proposal.bound_root.template == proposal.specs[0].candidate
            assert structure_tree_key(proposal.bound_root)
            assert role_tree_key(
                proposal.bound_root, include_bound_provenance=True)
        report = runtime.report()
        assert (
            report.candidate_count,
            report.active_candidate_count,
            report.operator_profile_count,
            report.executable_proposal_count,
        ) == (16, 12, 4, 7)
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert report.reality_fact_count == report.carrier_projection_count == 0
        assert report.schema_rejection_count >= 1
        assert report.conflict_candidate_count >= 1
        assert report.superseded_candidate_count >= 1
    finally:
        backend.close()


def test_nested_order_scope_and_quantifier_inheritance_are_distinct(adapted):
    """NOT/EXISTS、EXISTS/NOT 和 modal scope flip 不可交换且保留 branch frame。"""
    backend, runtime = _build(adapted)
    try:
        first = _proposal(runtime, perturbation="QUANTIFIER_SWAP")
        second = next(item for item in runtime.proposals
                      if item.observation.perturbation_kind == "QUANTIFIER_SWAP"
                      and item.operator_families != first.operator_families)
        assert first.operator_families != second.operator_families
        assert structure_tree_key(first.bound_root) != structure_tree_key(second.bound_root)
        assert role_tree_key(first.bound_root, include_bound_provenance=True) != (
            role_tree_key(second.bound_root, include_bound_provenance=True))
        for index, proposal in enumerate((first, second), start=1):
            execution = runtime.execute(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((920, index))))
            assert execution is not None
            assert len(execution.evaluation.derivation) >= 2
            assert all(item.source == proposal.source_binding.source_ref
                       for item in execution.evaluation.derivation)
            assert all(item.scope.source == proposal.source_binding.source_ref
                       for item in execution.evaluation.derivation)
            if proposal.quantifiers and proposal.operator_families[0] == "EXISTS":
                assert execution.evaluation.branches
                assert all(branch.assignment is not None
                           for branch in execution.evaluation.branches)
                assert all(branch.source == proposal.source_binding.source_ref
                           and branch.scope == proposal.request_scope
                           for branch in execution.evaluation.branches)
            elif proposal.quantifiers:
                assert any(item.branch_key for item in execution.evaluation.derivation)
        shifted = _proposal(runtime, perturbation="MODAL_SCOPE_SHIFT")
        shifted_execution = runtime.execute(logic_request_for_proposal(
            shifted, request_key=LosslessIntegerKey((920, 3))))
        assert shifted_execution is not None
        assert shifted_execution.evaluation.scope != shifted.request_scope
    finally:
        backend.close()


def test_nested_each_actual_layer_forms_independent_urg_use(adapted):
    """同一 root 的每个实际 derivation layer 都有独立 U/R/G exact Use。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, perturbation="NONE")
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((930, 1)))
        u_use = runtime.adopt_understanding(runtime.resolve_understanding(request))
        r_use = runtime.adopt_reasoning(runtime.resolve_reasoning(
            replace(request, request_key=LosslessIntegerKey((930, 2)))))
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
        assert len(runtime.layer_uses) == 4
        assert {item.consumer for item in runtime.layer_uses} == {
            "UNDERSTANDING", "REASONING"}
        assert all(item.operator_premise_keys for item in runtime.layer_uses)
        assert tuple(item.ordinal for item in runtime.layer_uses[:2]) == (0, 1)
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((930, 3)),
            logic_request_key=LosslessIntegerKey((930, 4)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "READY"
        option = choice.options[0]
        generation_use = runtime.adopt_generation(choice, option.stable_key())
        assert runtime.verify_generation(generation_use).verdict == "SUPPORT"
        assert len(runtime.layer_uses) == 6
        assert runtime.report().generation_layer_use_count == 2
    finally:
        backend.close()


def test_nested_missing_operator_ablation_budget_withdrawal_and_replay(adapted):
    """missing layer、family ablation、depth/branch budget、withdrawal/replay fail closed。"""
    backend, runtime = _build(adapted)
    try:
        assert any(item.reason == "MISSING_INNER_OPERATOR"
                   for item in runtime.adapter.rejections)
        disabled = W07LogicConsumerProtocol(
            W07_L07_PREFIX, disabled_operator_families=("MODAL",))
        disabled_backend, disabled_runtime = _build(adapted, protocol=disabled)
        try:
            proposal = _proposal(disabled_runtime, perturbation="NONE")
            assert disabled_runtime.understanding.preview(logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((940, 1)))).status == "NO_ADOPTION"
        finally:
            disabled_backend.close()
        proposal = _proposal(runtime, perturbation="NONE")
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((940, 2)))
        with pytest.raises(RuntimeError, match="max_depth.*超限"):
            runtime.execute(replace(
                request, budget=W07LogicBudget(1, 32, 128, 32)))
        with pytest.raises(RuntimeError, match="max_steps.*超限"):
            runtime.execute(replace(
                request, budget=W07LogicBudget(8, 32, 1, 32)))
        assert runtime.execute(request) is not None
    finally:
        backend.close()

    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.state_key() == right.state_key()
        assert left.report() == right.report()
        proposal = _proposal(left, perturbation="NONE")
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((941, 1)))
        use = left.adopt_understanding(left.resolve_understanding(request))
        account = next(
            account for application in left.learning.applications()
            if application.binding.proposal == proposal
            for account in application.accounts
            if account.stance == EVIDENCE_SUPPORT and not account.derived_supersede)
        left.learning.withdraw_evidence(account, withdrawal_level=1)
        assert left.verify_understanding(use).verdict == "REFUTE"
        assert left.report().cycle_claim_count == 0
    finally:
        left_backend.close()
        right_backend.close()
