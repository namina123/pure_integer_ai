"""W07-02 共享 LogicClosure/H-04/H-05 lifecycle 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
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
from pure_integer_ai.experiments.ph2_w07_learning import (
    W07LearningError,
    W07LogicLearningRuntime,
    build_w07_learning_runtime,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]


def _request(context):
    return W07RunRequest(
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
        tuple(
            item.relative_path
            for item in context.candidate_payload_bindings
        ),
        tuple(
            item.relative_path
            for item in context.teacher_evidence_bindings
        ),
    )


@pytest.fixture(scope="module")
def adapter_output(tmp_path_factory):
    path = tmp_path_factory.mktemp("w07-learning") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        payload = W07PayloadFirewall.open(
            ROOT, context, _request(context)).read_training_payload()
        return adapt_w07_training_payload(payload)
    finally:
        backend.close()


@pytest.fixture(scope="module")
def learning_runtime(adapter_output):
    backend = DictBackend()
    runtime = build_w07_learning_runtime(backend, adapter_output)
    yield runtime
    backend.close()


def _application_for(runtime, predicate):
    return next(
        item for item in runtime.applications()
        if predicate(item.binding)
    )


def test_w07_learning_uses_one_shared_owner_and_deduplicates_replay(
        learning_runtime, adapter_output):
    """七类 proposal 共用一个 H-04/H-05 owner，重复 apply 零写。"""
    assert learning_runtime.logic.candidate_runtime is learning_runtime.learning
    assert len(learning_runtime.proposals()) == 63
    assert len(learning_runtime.registered_specs()) == 71
    assert len({
        item.candidate for item in learning_runtime.registered_specs()
    }) == 71
    before = learning_runtime.learning.state_key()
    applications = learning_runtime.apply_all(adapter_output)
    assert len(applications) == 63
    assert learning_runtime.learning.state_key() == before


def test_w07_learning_rejects_forged_registered_binding(learning_runtime):
    """同一 teacher route 不能替换 reason 或 supersede 控制语义。"""
    binding = learning_runtime.applications()[0].binding
    forged = replace(binding, reason_key=(*binding.reason_key, 1))
    with pytest.raises(W07LearningError, match="不属于已登记 adapter"):
        learning_runtime.apply_evidence(forged)


def test_w07_learning_preserves_operator_four_state_and_active_projection(
        learning_runtime):
    """support/refute/unknown/conflict 由当前 Evidence 派生，不压成 bool。"""
    report = learning_runtime.report()
    assert report.candidate_count == 71
    assert report.schema_rejection_count == 3
    assert report.evidence_application_count == 63
    assert report.operator_evidence_account_count == 94
    assert report.active_operator_count == 36
    assert report.refuted_candidate_count == 7
    assert report.conflict_candidate_count == 15
    assert report.unknown_candidate_count == 13
    assert report.archived_candidate_count == 7
    assert report.superseded_candidate_count == 8
    assert report.reparse_count == 7
    assert report.withdrawal_count == 0

    snapshots = tuple(
        learning_runtime.snapshot_for(item.candidate)
        for item in learning_runtime.registered_specs()
    )
    assert {item.epistemic_status for item in snapshots} == {
        EPISTEMIC_CONFLICTED,
        EPISTEMIC_REFUTED,
        EPISTEMIC_UNKNOWN,
        EPISTEMIC_SUPPORTED,
    }
    assert len(learning_runtime.active_specs()) == 36
    assert all(
        learning_runtime.logic.adoption(item) is not None
        for item in learning_runtime.active_specs()
    )
    registry = learning_runtime.logic.registry_snapshot()
    assert registry.adoptions == ()
    assert len(registry.conflicted_structures) == 7


def test_w07_learning_separates_operator_adoption_from_content_evidence(
        learning_runtime):
    """合法 FALSE 支持结构采用，内容 refute 不写入 operator Hypothesis。"""
    application = _application_for(
        learning_runtime,
        lambda item: (
            item.expected_state == "FALSE"
            and item.stances == (EVIDENCE_SUPPORT,)
            and item.supersedes_observation_key is None
        ),
    )
    assert application.binding.content_stances == (EVIDENCE_REFUTE,)
    account = application.accounts[0]
    assert not account.derived_supersede
    assert account.outcome.evidence.stance == EVIDENCE_SUPPORT
    adoption = learning_runtime.logic.adoption(account.spec)
    assert adoption is not None
    assert account.outcome.evidence in adoption.evidence
    assert all(
        item.hypothesis == adoption.hypothesis for item in adoption.evidence)


def test_w07_learning_reparse_supersedes_old_candidates_and_keeps_history(
        learning_runtime):
    """七条 parser revision 先形成 replacement，再退出八个旧 layer candidate。"""
    reparses = tuple(
        item for item in learning_runtime.applications() if item.reparse)
    assert len(reparses) == 7
    assert sum(len(item.superseded_candidates) for item in reparses) == 8
    for application in reparses:
        derived = tuple(
            item for item in application.accounts if item.derived_supersede)
        assert len(derived) == len(application.superseded_candidates)
        assert all(item.stance == EVIDENCE_REFUTE for item in derived)
        for account in derived:
            snapshot = learning_runtime.snapshot_for(account.spec.candidate)
            history = learning_runtime.learning.engine.ledger.evidence_history(
                account.outcome.evidence.hypothesis)
            assert snapshot.lifecycle == LIFECYCLE_SUPERSEDED
            assert account.outcome.evidence in history


def test_w07_learning_refute_archives_and_replay_is_idempotent(
        learning_runtime):
    """PSEUDO_OPERATOR 形成 candidate+refute，并由现役 H-04 自动 archive。"""
    application = _application_for(
        learning_runtime,
        lambda item: (
            item.proposal.observation.perturbation_kind
            == "PSEUDO_OPERATOR"
        ),
    )
    account = application.accounts[0]
    assert account.stance == EVIDENCE_REFUTE
    assert learning_runtime.snapshot_for(
        account.spec.candidate).lifecycle == LIFECYCLE_ARCHIVED
    before = learning_runtime.learning.state_key()
    first = learning_runtime.archive_refuted(account)
    second = learning_runtime.archive_refuted(account)
    assert first is second
    assert first.automatic
    assert first.evidence == account.outcome.evidence
    assert learning_runtime.learning.state_key() == before


def test_w07_learning_withdrawal_is_append_only_and_demotes_projection(
        adapter_output):
    """UNKNOWN revision 撤回 support，旧 Evidence 留存但退出 active set。"""
    backend = DictBackend()
    runtime = W07LogicLearningRuntime(backend)
    try:
        runtime.register_adapter_output(adapter_output)
        binding = next(
            item for item in adapter_output.evidence
            if (item.expected_state == "FALSE"
                and item.stances == (EVIDENCE_SUPPORT,)
                and item.supersedes_observation_key is None
                and len(item.proposal.specs) == 1)
        )
        application = runtime.apply_evidence(binding)
        account = application.accounts[0]
        prior = account.outcome.evidence
        hypothesis = prior.hypothesis
        assert runtime.logic.adoption(account.spec) is not None
        history_before = runtime.learning.engine.ledger.evidence_history(
            hypothesis)

        withdrawal = runtime.withdraw_evidence(
            account, withdrawal_level=1)
        history_after = runtime.learning.engine.ledger.evidence_history(
            hypothesis)
        snapshot = runtime.snapshot_for(account.spec.candidate)
        assert len(history_after) == len(history_before) + 1
        assert prior in history_after
        assert withdrawal.evidence in history_after
        assert withdrawal.evidence.stance == EVIDENCE_UNKNOWN
        assert withdrawal.evidence.supersedes_evidence_id == prior.evidence_id
        assert prior.evidence_id not in snapshot.support_evidence_ids
        assert len(snapshot.unknown_evidence_ids) == 2
        assert withdrawal.evidence.evidence_id in snapshot.unknown_evidence_ids
        assert runtime.logic.adoption(account.spec) is None
        assert runtime.withdraw_evidence(
            account, withdrawal_level=1) is withdrawal
        with pytest.raises(W07LearningError, match="不同等级"):
            runtime.withdraw_evidence(account, withdrawal_level=2)
    finally:
        backend.close()
