"""W06-02 SemanticGraph、H-05、R-00 lifecycle 与 withdrawal 专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
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
    W06LearningError,
    W06LearningResult,
    W06RelationLearningRuntime,
    build_w06_learning_runtime,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "4d57305bc4474081c9304a05287ab4783f49a849"


@pytest.fixture(scope="module")
def adapted(tmp_path_factory):
    """通过一次性 public firewall 构建当前 W-06 adapter 输出。"""
    path = tmp_path_factory.mktemp("w06-learning") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        backend_key = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    context = open_w06_frozen_context(
        ROOT,
        current_remote_commit_sha1=HEAD,
        backend_profile_key=backend_key,
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
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W06PayloadFirewall.open(
        ROOT, context, request).read_training_payload()
    return adapt_w06_training_payload(payload)


@pytest.fixture(scope="module")
def learned(adapted):
    """在一个可回读 DictBackend 中完成全部 accepted train lifecycle。"""
    backend = DictBackend()
    try:
        yield backend, build_w06_learning_runtime(backend, adapted)
    finally:
        backend.close()


def test_w06_learning_closes_current_train_lifecycle_without_rejection_pollution(
        learned):
    """当前 50 条 accepted Evidence 必须形成真实 H-05/R-00 分态结果。"""
    _backend, runtime = learned
    assert runtime.report() == W06LearningResult(
        candidate_count=50,
        schema_rejection_count=1,
        relation_family_count=14,
        evidence_application_count=50,
        evidence_account_count=64,
        active_candidate_count=17,
        archived_candidate_count=19,
        superseded_candidate_count=7,
        conflict_candidate_count=13,
        unknown_candidate_count=1,
        reparse_count=7,
        withdrawal_count=0,
    )
    candidate_report = runtime.learning.report()
    assert candidate_report.candidate_count == 50
    assert candidate_report.prediction_count == 64
    assert candidate_report.active_projection_count == 17
    assert len(runtime.active_candidates()) == 17


def test_w06_learning_materializes_semantic_definitions_and_fourteen_families(
        learned, adapted):
    """所有合法 Proposition 必须由 SemanticGraph 回读，14 family 均保留 typed 坐标。"""
    _backend, runtime = learned
    for candidate in adapted.candidates:
        proposition_ref = runtime.semantic_graph.ontology.resolve(
            candidate.proposition.proposition)
        assert proposition_ref is not None
        restored = runtime.semantic_graph.read_atomic(
            proposition_ref)
        assert restored.definition == candidate.proposition
    assert {
        item.relation_family for item in adapted.candidates
    } == {
        item.relation_family
        for item in runtime.registered_candidates()
    }
    registered = {
        item.proposition.proposition
        for item in runtime.registered_candidates()
    }
    assert all(
        rejection.proposition not in registered
        for rejection in adapted.rejections
    )


def test_w06_learning_preserves_refute_conflict_unknown_archive_and_projection(
        learned, adapted):
    """refute、conflict 与 unknown 必须保持不同四态和 active projection 后果。"""
    _backend, runtime = learned
    refuted = next(
        item for item in adapted.candidates
        if item.sample_role == "refute"
    )
    refuted_snapshot = runtime.snapshot_for(
        refuted.proposition.proposition)
    assert refuted_snapshot.snapshot.lifecycle == LIFECYCLE_ARCHIVED
    assert refuted_snapshot.snapshot.epistemic_status == EPISTEMIC_REFUTED
    assert refuted_snapshot.active_fact is None

    conflicted, conflict_snapshot = next(
        (item, snapshot)
        for item in adapted.candidates
        if item.sample_role == "conflict"
        for snapshot in (runtime.snapshot_for(item.proposition.proposition),)
        if snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
    )
    assert conflict_snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
    assert conflict_snapshot.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
    assert conflict_snapshot.active_fact is None

    unknown_binding = next(
        item for item in adapted.evidence
        if item.stances == (EVIDENCE_UNKNOWN,)
    )
    unknown_snapshot = runtime.snapshot_for(unknown_binding.candidate)
    assert unknown_snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
    assert unknown_snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
    assert unknown_snapshot.active_fact is None

    for candidate in runtime.active_candidates():
        assert runtime.consumer is not None
        assert len(runtime.consumer.lookup_proposition(
            candidate.proposition.proposition)) == 1


def test_w06_parser_revision_supports_replacement_before_superseding_old_candidate(
        learned):
    """七个 reparse 必须先激活 replacement，再以派生 refute 退出旧候选。"""
    _backend, runtime = learned
    reparses = tuple(
        item for item in runtime.applications() if item.reparse)
    assert len(reparses) == 7
    for application in reparses:
        assert len(application.superseded_candidates) == 1
        target = application.superseded_candidates[0]
        replacement = application.binding.candidate
        target_snapshot = runtime.snapshot_for(target)
        replacement_snapshot = runtime.snapshot_for(replacement)
        assert target_snapshot.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert replacement_snapshot.active_fact is not None
        transition = runtime.learning.engine.ledger.transition_history(
            target_snapshot.formation.hypothesis)[-1]
        assert transition.replacement == replacement_snapshot.formation.hypothesis
        assert application.accounts[0].candidate == replacement
        assert application.accounts[-1].candidate == target
        assert application.accounts[-1].derived_supersede is True


def test_w06_register_and_evidence_routes_are_deduplicated(learned, adapted):
    """重复登记和重复 apply_all 必须保持 backend 与 owner 状态逐字节不变。"""
    backend, runtime = learned
    before_backend = backend.snapshot()
    before_state = runtime.closure.state_key()
    before_report = runtime.report()
    runtime.apply_all(adapted)
    assert backend.snapshot() == before_backend
    assert runtime.closure.state_key() == before_state
    assert runtime.report() == before_report


def test_w06_withdrawal_supersedes_support_and_demotes_active_projection(adapted):
    """隔离 probe 必须以 UNKNOWN revision 撤回 support，并保留旧 Evidence 历史。"""
    backend = DictBackend()
    try:
        runtime = W06RelationLearningRuntime(backend)
        runtime.register_adapter_output(adapted)
        binding = next(
            item for item in adapted.evidence
            if (item.stances == (EVIDENCE_SUPPORT,)
                and item.supersedes_observation_key is None)
        )
        application = runtime.apply_evidence(binding)
        account = application.accounts[0]
        assert runtime.consumer is not None
        assert runtime.consumer.lookup_proposition(account.candidate)
        prior = account.trace.outcome.evidence

        withdrawal = runtime.withdraw_evidence(
            account, withdrawal_level=1)
        assert withdrawal.evidence.supersedes_evidence_id == prior.evidence_id
        assert withdrawal.evidence.stance == EVIDENCE_UNKNOWN
        snapshot = runtime.snapshot_for(account.candidate)
        assert snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert snapshot.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        assert snapshot.snapshot.support_evidence_ids == ()
        assert withdrawal.evidence.evidence_id in (
            snapshot.snapshot.unknown_evidence_ids)
        history = runtime.learning.engine.ledger.evidence_history(
            snapshot.formation.hypothesis)
        assert prior in history and withdrawal.evidence in history
        assert runtime.consumer.lookup_proposition(account.candidate) == ()
        assert runtime.learning.state_key()[-1] >= (
            withdrawal.evidence.timestamp_seq + 2)

        before = runtime.learning.state_key()
        assert runtime.withdraw_evidence(
            account, withdrawal_level=1) == withdrawal
        assert runtime.learning.state_key() == before
        with pytest.raises(W06LearningError, match="不同等级"):
            runtime.withdraw_evidence(account, withdrawal_level=2)
    finally:
        backend.close()
