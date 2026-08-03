"""W07-L03 CONDITION public bounded closure 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.scope_identity import query_scope
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
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
from pure_integer_ai.experiments.ph2_w07_l03 import (
    W07ConditionProof,
    W07_L03_PREFIX,
    build_w07_l03_runtime,
    generation_request_for_proposal,
    logic_request_for_proposal,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicBudget,
    W07LogicConsumerProtocol,
    W07LogicContractError,
    W07LogicRequest,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    role_tree_key,
    w07_logic_language_branch,
)
from pure_integer_ai.experiments.typed_proof_family_contracts import (
    CONDITION_AFFIRMING_CONSEQUENT,
    CONDITION_MATERIAL,
    CONDITION_NECESSARY,
    CONDITION_SUFFICIENT,
    PROOF_ACCEPTED,
    PROOF_BUDGET_EXHAUSTED,
    PROOF_CONFLICTED,
    PROOF_FAIL_CLOSED,
    PROOF_FAMILY_MISMATCH,
    PROOF_FAMILY_NOT,
    PROOF_REJECTED,
    PROOF_UNKNOWN,
    ProofWorkBudget,
)
from pure_integer_ai.experiments.typed_proof_family_runtime import (
    TypedProofFamilyDispatcher,
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
    runtime = build_w07_l03_runtime(
        backend,
        adapted,
        **({} if protocol is None else {"protocol": protocol}),
    )
    return backend, runtime


def _proposal(runtime, *, perturbation="NONE", state=None):
    for index, proposal in enumerate(
            runtime.view.executable_proposals("CONDITION"), start=1):
        if proposal.observation.perturbation_kind != perturbation:
            continue
        if state is None:
            return proposal
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((700, index)))
        execution = runtime.view.execute(request)
        if execution is not None and execution.evaluation.state.stable_key() == state:
            return proposal
    raise LookupError("未找到 CONDITION proposal")


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


def _dispatch(certificate, *, limit=128):
    return TypedProofFamilyDispatcher().check(
        certificate, ProofWorkBudget(limit))


def test_l03_prefix_counts_four_states_and_zero_reality_claims(adapted):
    """L03 累积前缀只形成一个 CONDITION profile 和 provisional 结果。"""
    backend, runtime = _build(adapted)
    try:
        assert tuple(dict.fromkeys(
            item.observation.substage for item in runtime.adapter.proposals
        )) == W07_L03_PREFIX
        assert len(runtime.proposals) == 9
        assert len(runtime.view.executable_proposals("NOT")) == 7
        assert len(runtime.view.executable_proposals("AND_OR")) == 9
        report = runtime.report()
        assert (
            report.candidate_count,
            report.active_candidate_count,
            report.operator_profile_count,
            report.executable_proposal_count,
        ) == (9, 3, 1, 5)
        assert (
            report.supported_count,
            report.refuted_count,
            report.unknown_count,
            report.conflict_count,
        ) == (2, 1, 1, 1)
        assert (
            report.material_certificate_count,
            report.sufficient_certificate_count,
            report.necessary_certificate_count,
            report.provisional_content_evidence_count,
        ) == (5, 5, 5, 2)
        assert bytes(report.operator_digest).hex() == (
            "dc9799cb4462805a6b53329d7e8d3afabf9581a31241f889f8ad6e4d85b577c8")
        assert bytes(report.source_evidence_digest).hex() == (
            "d492bbf7ed41ee8187e38a0a4b02485b17f86f5096a55750816c2cd1f0133b5d")
        assert bytes(report.execution_digest).hex() == (
            "33b32e4caba6be417dc1cb0469b10f9840c283818846a08d54349e6dabb9988d")
        assert bytes(report.certificate_digest).hex() == (
            "db2d16212afaeef395978f18e622825420c6fa1aad5c3f018367b412c4b9f996")
        assert (
            report.modus_ponens_count,
            report.global_truth_count,
            report.causal_fact_count,
            report.temporal_fact_count,
            report.action_fact_count,
            report.private_read_count,
            report.formal_guard_read_count,
        ) == (0, 0, 0, 0, 0, 0, 0)
    finally:
        backend.close()


def test_condition_executes_ordered_open_world_table_and_trace(adapted):
    """前件/后件顺序进入真实 S-04，unknown/conflict 不压成二值。"""
    backend, runtime = _build(adapted)
    try:
        observed = set()
        for index, proposal in enumerate(
                runtime.view.executable_proposals("CONDITION"), start=1):
            raw_states = tuple(
                (item["support"], item["refute"])
                for item in proposal.observation.typed_payload.to_value()[
                    "operand_evidence"])
            expected = (
                int(bool(raw_states[0][1] or raw_states[1][0])),
                int(bool(raw_states[0][0] and raw_states[1][1])),
            )
            request = logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((100, index)))
            execution = runtime.view.execute(request)
            assert execution is not None
            assert execution.evaluation.state.stable_key() == expected
            assert len(execution.evaluation.derivation) == 1
            definition = execution.operator_adoptions[0].spec.definition
            by_slot = {
                (item.role, item.ordinal): item.filler
                for item in proposal.bound_root.bindings}
            ordered = tuple(
                by_slot[(slot.role, slot.ordinal)].template
                for slot in definition.slots)
            assert execution.evaluation.derivation[-1].premises == ordered
            observed.add(expected)
        assert observed == {(1, 0), (0, 1), (0, 0), (1, 1)}
    finally:
        backend.close()


def test_condition_proofs_bind_material_sufficient_necessary_and_adoption(
        adapted):
    """三类 P2-G certificate 保持方向，kind Evidence 绑定当前 adoption。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        proofs = []
        for index, kind in enumerate((
                CONDITION_MATERIAL,
                CONDITION_SUFFICIENT,
                CONDITION_NECESSARY,
                ), start=1):
            request = logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((200, index)))
            proof = runtime.prove(request, kind)
            assert proof is not None
            assert proof.receipt.result.status == PROOF_ACCEPTED
            assert proof.provisional_content_evidence
            assert proof.certificate.evaluation.source == request.source
            assert proof.certificate.evaluation.scope == request.scope
            assert proof.certificate.kind_evidence_ids == (
                () if kind == CONDITION_MATERIAL
                else proof.operator_evidence_ids)
            proofs.append(proof)
        assert proofs[0].certificate.condition.proposition == (
            proofs[1].certificate.condition.proposition)
        assert proofs[2].certificate.condition.proposition == (
            proofs[1].certificate.conditioned.proposition)
        assert proofs[2].certificate.conditioned.proposition == (
            proofs[1].certificate.condition.proposition)
    finally:
        backend.close()


def test_condition_certificate_ablations_fail_closed_or_unknown(adapted):
    """缺 kind Evidence、错向、错 scope、肯定后件、跨 family 与预算均拒绝。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((300, 1)))
        sufficient = runtime.certificate_for(request, CONDITION_SUFFICIENT)
        material = runtime.certificate_for(request, CONDITION_MATERIAL)
        assert sufficient is not None and material is not None
        assert _dispatch(replace(
            sufficient, kind_evidence_ids=())).result.status == PROOF_UNKNOWN
        assert _dispatch(replace(
            material,
            kind_evidence_ids=sufficient.kind_evidence_ids,
        )).result.status == PROOF_FAIL_CLOSED
        assert _dispatch(replace(
            sufficient,
            condition=sufficient.conditioned,
            conditioned=sufficient.condition,
        )).result.status == PROOF_FAIL_CLOSED
        drifted_scope = query_scope(991, parent=sufficient.condition.scope)
        assert _dispatch(replace(
            sufficient,
            condition=replace(sufficient.condition, scope=drifted_scope),
        )).result.status == PROOF_FAIL_CLOSED
        assert _dispatch(replace(
            sufficient,
            inference_kind=CONDITION_AFFIRMING_CONSEQUENT,
        )).result.status == PROOF_REJECTED
        assert _dispatch(replace(
            sufficient,
            declared_family=PROOF_FAMILY_NOT,
        )).result.status == PROOF_FAMILY_MISMATCH
        assert _dispatch(sufficient, limit=1).result.status == (
            PROOF_BUDGET_EXHAUSTED)

        forged = replace(sufficient, kind_evidence_ids=(991001,))
        with pytest.raises(W07LogicContractError, match="当前 adoption"):
            W07ConditionProof(
                CONDITION_SUFFICIENT,
                runtime.view.execute(request),
                forged,
                _dispatch(forged),
                sufficient.kind_evidence_ids,
            )
    finally:
        backend.close()


def test_condition_proof_preserves_all_four_checker_statuses(adapted):
    """certificate checker 原样保留 provisional/refuted/unknown/conflicted。"""
    expected = {
        (1, 0): PROOF_ACCEPTED,
        (0, 1): PROOF_REJECTED,
        (0, 0): PROOF_UNKNOWN,
        (1, 1): PROOF_CONFLICTED,
    }
    backend, runtime = _build(adapted)
    try:
        observed = set()
        for index, proposal in enumerate(
                runtime.view.executable_proposals("CONDITION"), start=1):
            request = logic_request_for_proposal(
                proposal, request_key=LosslessIntegerKey((400, index)))
            proof = runtime.prove(request, CONDITION_MATERIAL)
            assert proof is not None
            state = proof.execution.evaluation.state.stable_key()
            assert proof.receipt.result.status == expected[state]
            observed.add(proof.receipt.result.status)
        assert observed == set(expected.values())
    finally:
        backend.close()


def test_condition_urg_generation_and_postcheck_keep_direction_and_source(
        adapted):
    """U/R/G 各自持有 Use，generation 保留 conditional tree/source/scope。"""
    backend, runtime = _build(adapted)
    try:
        proposal = _proposal(runtime, state=(1, 0))
        u = runtime.resolve_understanding(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((500, 1))))
        r = runtime.resolve_reasoning(logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((500, 2))))
        u_use = runtime.adopt_understanding(u)
        r_use = runtime.adopt_reasoning(r)
        assert runtime.verify_understanding(u_use).verdict == "SUPPORT"
        assert runtime.verify_reasoning(r_use).verdict == "SUPPORT"
        choice = runtime.choose_generation(generation_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((500, 3)),
            logic_request_key=LosslessIntegerKey((500, 4)),
            constraints=_constraints(proposal),
        ))
        assert choice.status == "READY"
        option = choice.options[0]
        assert option.operator_families == ("CONDITION",)
        assert option.role_tree_key == role_tree_key(
            proposal.bound_root, include_bound_provenance=True)
        assert option.source == proposal.source_binding.source_ref
        assert option.scope == proposal.request_scope
        use = runtime.adopt_generation(choice, option.stable_key())
        assert runtime.verify_generation(use).verdict == "SUPPORT"
        report = runtime.report()
        assert (
            report.understanding_use_count,
            report.reasoning_use_count,
            report.generation_use_count,
            report.generation_outcome_count,
        ) == (1, 1, 1, 1)
    finally:
        backend.close()


def test_condition_confusions_and_orthogonal_ablations_do_not_execute(adapted):
    """Role reversal、CAUSES/PRECEDES confusion 与独立 bridge 消融均 fail closed。"""
    backend, runtime = _build(adapted)
    try:
        invalid = tuple(
            item for item in runtime.proposals
            if item.observation.perturbation_kind in {
                "ANTECEDENT_CONSEQUENT_SWAP",
                "CAUSAL_CONFUSION",
                "TEMPORAL_CONFUSION",
            })
        assert len(invalid) == 3
        for index, proposal in enumerate(invalid, start=1):
            assert runtime.understanding.preview(logic_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((600, index)),
            )).status == "NO_ADOPTION"
    finally:
        backend.close()

    disabled = W07LogicConsumerProtocol(
        W07_L03_PREFIX,
        disabled_operator_families=("CONDITION",),
    )
    backend, runtime = _build(adapted, protocol=disabled)
    try:
        proposal = next(
            item for item in runtime.proposals
            if item.observation.perturbation_kind == "NONE")
        assert runtime.prove(logic_request_for_proposal(
            proposal,
            request_key=LosslessIntegerKey((610, 1)),
        ), CONDITION_MATERIAL) is None
        and_proposal = next(
            item for item in runtime.view.executable_proposals("AND_OR")
            if item.operator_families == ("AND",))
        assert runtime.view.execute(W07LogicRequest(
            LosslessIntegerKey((610, 2)),
            "AND_OR",
            and_proposal.bound_root.template,
            and_proposal.source_binding.source_ref,
            and_proposal.request_scope,
        )) is not None
    finally:
        backend.close()

    for field, expected in (
            ("generation_connected", "UNKNOWN"),
            ("postcheck_connected", "REFUTE")):
        protocol = W07LogicConsumerProtocol(
            W07_L03_PREFIX, **{field: False})
        backend, runtime = _build(adapted, protocol=protocol)
        try:
            proposal = _proposal(runtime, state=(1, 0))
            choice = runtime.choose_generation(generation_request_for_proposal(
                proposal,
                request_key=LosslessIntegerKey((620, 1)),
                logic_request_key=LosslessIntegerKey((620, 2)),
                constraints=_constraints(proposal),
            ))
            if field == "generation_connected":
                assert choice.status == expected
            else:
                use = runtime.adopt_generation(
                    choice, choice.options[0].stable_key())
                assert runtime.verify_generation(use).verdict == expected
        finally:
            backend.close()


def test_condition_withdrawal_budget_and_replay_close_l03(adapted):
    """withdrawal 使 stale Use 失效，预算 fail closed，clean replay 相同。"""
    left_backend, left = _build(adapted)
    right_backend, right = _build(adapted)
    try:
        assert left.report() == right.report()
        assert left.state_key() == right.state_key()
        proposal = _proposal(left, state=(1, 0))
        request = logic_request_for_proposal(
            proposal, request_key=LosslessIntegerKey((700, 1)))
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
