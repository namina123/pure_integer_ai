"""W06-R05 SIMILAR/ANTONYM 的公开有界双 channel 闭环专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricRelationBudget,
    SymmetricRelationBudgetExceeded,
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
from pure_integer_ai.experiments.ph2_w06_r05 import (
    W06R05Runtime,
    generation_request_for_candidate,
    query_for_candidate,
    slice_w06_r05_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ConsumerProtocol,
    W06R05ContractError,
    W06R05PairQuery,
    W06_R05_CONFLICT,
    W06_R05_GENERATION_READY,
    W06_R05_GENERATION_REJECTED,
    W06_R05_GENERATION_UNKNOWN,
    W06_R05_OUTCOME_REFUTE,
    W06_R05_OUTCOME_SUPPORT,
    W06_R05_REFUTED,
    W06_R05_SUPPORTED,
    W06_R05_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r05_shared import (
    candidate_endpoints,
    w06_r05_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = SymmetricRelationBudget(32, 32)


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


def _build(adapted, protocol=W06R05ConsumerProtocol()):
    """建立只含 SIMILAR_ANTONYM train truth 的 learning/runtime。"""
    sliced = slice_w06_r05_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R05Runtime(
        learning, sliced, protocol=protocol)


def _active_candidates(sliced, learning):
    """返回 current active 的三个 R05 candidate。"""
    return tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )


def _supported_representatives(sliced, learning):
    """每个 supported canonical pair+channel 只返回一个代表 candidate。"""
    result = {}
    for candidate in _active_candidates(sliced, learning):
        pair = tuple(sorted(
            candidate_endpoints(candidate), key=lambda item: item.stable_key()))
        result.setdefault((candidate.relation_family, pair), candidate)
    return tuple(
        result[key]
        for key in sorted(result, key=lambda item: (
            item[0], *(endpoint.stable_key() for endpoint in item[1])))
    )


def _constraints(candidate):
    """按 candidate 语言分支构造最小 generation 约束。"""
    branch = w06_r05_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _query(runtime, candidate, *, key, budget=BUDGET):
    """从 candidate 的 typed pair/channel 构造 R05 查询。"""
    return query_for_candidate(
        candidate,
        request_key=LosslessIntegerKey(key),
        budget=budget,
    )


def _support_account(learning, candidate):
    """找到当前 candidate 的普通 support Evidence account。"""
    for application in learning.applications():
        for account in application.accounts:
            if (account.candidate == candidate.proposition.proposition
                    and account.stance == EVIDENCE_SUPPORT
                    and not account.derived_supersede):
                return account
    raise AssertionError("未找到普通 support Evidence")


def _exercise_positive(runtime, sliced, learning):
    """对两个 supported pair/channel 执行 U/R，并生成三个 active fact。"""
    query_results = []
    for ordinal, candidate in enumerate(
            _supported_representatives(sliced, learning), start=1):
        understanding = runtime.resolve_understanding(
            _query(runtime, candidate, key=(1, ordinal)))
        understanding_use = runtime.adopt_understanding(understanding)
        understanding_outcome = runtime.verify_understanding(understanding_use)
        reasoning = runtime.resolve_reasoning(
            _query(runtime, candidate, key=(2, ordinal)))
        reasoning_use = runtime.adopt_reasoning(reasoning)
        reasoning_outcome = runtime.verify_reasoning(reasoning_use)
        query_results.append((
            candidate,
            understanding,
            understanding_use,
            understanding_outcome,
            reasoning,
            reasoning_use,
            reasoning_outcome,
        ))

    generation_results = []
    active = tuple(sorted(
        _active_candidates(sliced, learning),
        key=lambda item: item.proposition.proposition.stable_key(),
    ))
    for ordinal, candidate in enumerate(active, start=1):
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((3, ordinal)),
            constraints=_constraints(candidate),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        generation_outcome = runtime.verify_generation(generation_use)
        generation_results.append((
            candidate, choice, generation_use, generation_outcome))
    return tuple(query_results), tuple(generation_results)


def test_w06_r05_slice_keeps_seven_candidates_and_needs_no_projection(adapted):
    """七条 train truth 均保留，canonical endpoint 已跨相关记录一致。"""
    sliced = slice_w06_r05_adapter(adapted)
    assert len(sliced.candidates) == len(sliced.evidence) == 7
    assert len(sliced.rejections) == len(sliced.rejection_evidence) == 0
    assert {item.relation_family for item in sliced.candidates} == {
        "SIMILAR", "ANTONYM"}
    assert len(sliced.schemas) == 2
    assert dict(sliced.execution_state) == dict(adapted.execution_state)

    related_revisions = {
        "teacher-fast-rapid-reversed-v1",
        "teacher-similar-as-antonym-v1",
        "teacher-rapid-fast-similar-v2",
        "teacher-rapid-fast-similar-v1",
    }
    related = tuple(
        item for item in sliced.candidates
        if item.source_record.revision_id in related_revisions
    )
    assert len(related) == 4
    pairs = {
        tuple(sorted(
            candidate_endpoints(item), key=lambda value: value.stable_key()))
        for item in related
    }
    assert len(pairs) == 1


def test_w06_r05_pair_channel_urg_use_and_generation_postcheck(adapted):
    """SIMILAR/ANTONYM 均须形成 U/R/G exact Use 与独立 postcheck。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        assert len(_active_candidates(sliced, learning)) == 3
        query_results, generation_results = _exercise_positive(
            runtime, sliced, learning)
        assert {item[0].relation_family for item in query_results} == {
            "SIMILAR", "ANTONYM"}
        for (_candidate, understanding, understanding_use,
             understanding_outcome, reasoning, reasoning_use,
             reasoning_outcome) in query_results:
            assert understanding.status == W06_R05_SUPPORTED
            assert reasoning.status == W06_R05_SUPPORTED
            assert understanding_outcome.verdict == W06_R05_OUTCOME_SUPPORT
            assert reasoning_outcome.verdict == W06_R05_OUTCOME_SUPPORT
            assert understanding_use.relation_uses
            assert reasoning_use.relation_uses
        for _candidate, choice, generation_use, outcome in generation_results:
            assert choice.status == W06_R05_GENERATION_READY
            assert len(generation_use.relation_uses) == 1
            assert outcome.verdict == W06_R05_OUTCOME_SUPPORT
            assert outcome.pair_channel_preserved

        report = runtime.report()
        assert report.candidate_count == 7
        assert report.rejection_count == 0
        assert report.active_count == 3
        assert report.refuted_count == 2
        assert report.conflict_count == 1
        assert report.superseded_count == 1
        assert report.understanding_use_count == 2
        assert report.reasoning_use_count == 2
        assert report.generation_use_count == 3
        assert report.generation_outcome_count == 3
        assert report.consumed_premise_count == 9
        assert report.private_read_count == 0
        assert report.formal_guard_read_count == 0
        assert report.w06_started == 0
        assert bytes(report.relation_digest).hex() == (
            "005f24aef0b360f9837e1fc5af2398e44520becfb3ad6c63c4eb997d5acbdb92")
        assert bytes(report.source_evidence_digest).hex() == (
            "1c9f5b252b8845e34e3665e2cc85e562806f92237d4a507685ba74f8bb4f3069")
        assert bytes(report.active_projection_digest).hex() == (
            "50ac4e17ac560c4667f26d1e6636c5386b131b467701435ddcec677578ab3f92")
    finally:
        backend.close()


def test_w06_r05_symmetry_refute_conflict_and_supersede_are_separated(adapted):
    """反向查询、refute、current conflict 和 parser supersede 必须分态。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        snapshots = {
            revision: learning.snapshot_for(item.proposition.proposition)
            for revision, item in by_revision.items()
        }
        assert snapshots["teacher-alias-as-similar-v1"].snapshot.epistemic_status == (
            EPISTEMIC_REFUTED)
        assert snapshots["teacher-similar-as-antonym-v1"].snapshot.epistemic_status == (
            EPISTEMIC_REFUTED)
        assert (snapshots["teacher-open-close-conflict-v1"].snapshot.lifecycle
                == LIFECYCLE_ACTIVE)
        assert (snapshots[
            "teacher-open-close-conflict-v1"].snapshot.epistemic_status
            == EPISTEMIC_CONFLICTED)
        assert (snapshots["teacher-rapid-fast-similar-v1"].snapshot.lifecycle
                == LIFECYCLE_SUPERSEDED)

        refuted = runtime.resolve_reasoning(_query(
            runtime, by_revision["teacher-alias-as-similar-v1"], key=(10, 1)))
        assert refuted.status == W06_R05_REFUTED
        conflict = runtime.resolve_reasoning(_query(
            runtime, by_revision["teacher-open-close-conflict-v1"],
            key=(10, 2)))
        assert conflict.status == W06_R05_CONFLICT

        active = by_revision["teacher-fast-rapid-reversed-v1"]
        direct = _query(runtime, active, key=(10, 3))
        reverse = replace(
            direct,
            request_key=LosslessIntegerKey((10, 4)),
            left=direct.right,
            right=direct.left,
        )
        direct_result = runtime.resolve_understanding(direct)
        reverse_result = runtime.resolve_understanding(reverse)
        assert direct_result.status == reverse_result.status == W06_R05_SUPPORTED
        assert direct_result.evaluation.pair == reverse_result.evaluation.pair

        superseded = by_revision["teacher-rapid-fast-similar-v1"]
        revised = runtime.resolve_reasoning(
            _query(runtime, superseded, key=(10, 5)))
        assert revised.status == W06_R05_SUPPORTED
        assert superseded.proposition.proposition not in revised.propositions
    finally:
        backend.close()


def test_w06_r05_channel_alias_future_and_neighbor_relations_fail_closed(adapted):
    """两个 channel 不互推，alias/future/相邻 relation 不能进入 R05 truth。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        similar = by_revision["teacher-fast-rapid-reversed-v1"]
        similar_query = _query(runtime, similar, key=(20, 1))
        antonym_same_pair = replace(
            similar_query,
            request_key=LosslessIntegerKey((20, 2)),
            channel="ANTONYM",
        )
        assert runtime.resolve_reasoning(similar_query).status == (
            W06_R05_SUPPORTED)
        assert runtime.resolve_reasoning(antonym_same_pair).status == (
            W06_R05_REFUTED)

        antonym = by_revision["teacher-high-low-antonym-v1"]
        antonym_query = _query(runtime, antonym, key=(20, 3))
        similar_other_pair = replace(
            antonym_query,
            request_key=LosslessIntegerKey((20, 4)),
            channel="SIMILAR",
        )
        assert runtime.resolve_understanding(antonym_query).status == (
            W06_R05_SUPPORTED)
        assert runtime.resolve_understanding(similar_other_pair).status == (
            W06_R05_UNKNOWN)

        alias_confusion = by_revision["teacher-alias-as-similar-v1"]
        assert runtime.reasoning.preview(
            _query(runtime, alias_confusion, key=(20, 5))).status == (
                W06_R05_REFUTED)
        for family in (
                "PURE_ALIAS", "PROPERTY", "SUBSET", "MEMBER",
                "PART_OF", "HAS_PART", "PRECEDES"):
            with pytest.raises(W06R05ContractError):
                W06R05PairQuery(
                    LosslessIntegerKey((20, 6)),
                    family,
                    *candidate_endpoints(similar),
                    BUDGET,
                    similar.source_ref,
                )

        non_r05 = adapted.candidates_for_substage("MEREOLOGY")[0]
        with pytest.raises(W06R05ContractError):
            generation_request_for_candidate(
                non_r05,
                request_key=LosslessIntegerKey((20, 7)),
                constraints=_constraints(similar),
            )
    finally:
        backend.close()


def test_w06_r05_symmetry_does_not_add_transitive_closure(adapted):
    """R05 view 只有 symmetry；无 direct pair 的跨对象查询保持 UNKNOWN。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        protocols = runtime.view.protocols.values()
        assert all(not hasattr(item, "transitive_rules") for item in protocols)
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        similar = by_revision["teacher-fast-rapid-reversed-v1"]
        antonym = by_revision["teacher-high-low-antonym-v1"]
        left = candidate_endpoints(similar)[0]
        unrelated = candidate_endpoints(antonym)[0]
        query = W06R05PairQuery(
            LosslessIntegerKey((25, 1)),
            "SIMILAR",
            left,
            unrelated,
            BUDGET,
            similar.source_ref,
        )
        assert runtime.resolve_understanding(query).status == W06_R05_UNKNOWN
    finally:
        backend.close()


def test_w06_r05_similar_antonym_and_generation_ablations_are_orthogonal(adapted):
    """SIMILAR、ANTONYM、generation 和 channel identity 分别击穿目标。"""
    backend, sliced, learning, _runtime = _build(adapted)
    try:
        representatives = {
            item.relation_family: item
            for item in _supported_representatives(sliced, learning)
        }
        similar_off = W06R05Runtime(
            learning,
            sliced,
            protocol=W06R05ConsumerProtocol(similar_connected=False),
        )
        assert similar_off.resolve_understanding(
            _query(similar_off, representatives["SIMILAR"], key=(30, 1))
        ).status == W06_R05_UNKNOWN
        assert similar_off.resolve_understanding(
            _query(similar_off, representatives["ANTONYM"], key=(30, 2))
        ).status == W06_R05_SUPPORTED

        antonym_off = W06R05Runtime(
            learning,
            sliced,
            protocol=W06R05ConsumerProtocol(antonym_connected=False),
        )
        assert antonym_off.resolve_reasoning(
            _query(antonym_off, representatives["SIMILAR"], key=(30, 3))
        ).status == W06_R05_SUPPORTED
        assert antonym_off.resolve_reasoning(
            _query(antonym_off, representatives["ANTONYM"], key=(30, 4))
        ).status == W06_R05_UNKNOWN

        generation_off = W06R05Runtime(
            learning,
            sliced,
            protocol=W06R05ConsumerProtocol(generation_connected=False),
        )
        target = representatives["SIMILAR"]
        assert generation_off.resolve_reasoning(
            _query(generation_off, target, key=(30, 5))
        ).status == W06_R05_SUPPORTED
        choice = generation_off.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((30, 6)),
                constraints=_constraints(target),
            ))
        assert choice.status == W06_R05_GENERATION_UNKNOWN

        channel_off = W06R05Runtime(
            learning,
            sliced,
            protocol=W06R05ConsumerProtocol(
                channel_identity_connected=False),
        )
        assert channel_off.resolve_understanding(
            _query(channel_off, target, key=(30, 7))
        ).status == W06_R05_UNKNOWN
        rejected = channel_off.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((30, 8)),
                constraints=_constraints(target),
            ))
        assert rejected.status == W06_R05_GENERATION_REJECTED
    finally:
        backend.close()


def test_w06_r05_postcheck_ablation_refutes_generation_only(adapted):
    """postcheck 断开只让 generation outcome REFUTE，不影响 choice ready。"""
    backend, sliced, learning, runtime = _build(
        adapted,
        protocol=W06R05ConsumerProtocol(postcheck_connected=False),
    )
    try:
        active = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            active,
            request_key=LosslessIntegerKey((40, 1)),
            constraints=_constraints(active),
        ))
        assert choice.status == W06_R05_GENERATION_READY
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R05_OUTCOME_REFUTE
        assert outcome.pair_query_status == W06_R05_UNKNOWN
    finally:
        backend.close()


def test_w06_r05_withdrawal_refutes_stale_pair_and_target_use(adapted):
    """撤回一个 active SIMILAR 后 pair 可仍支持，但旧 exact Use 必须失效。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        target = next(
            item for item in sliced.candidates
            if item.source_record.revision_id == "teacher-rapid-fast-similar-v2")
        resolution = runtime.resolve_understanding(
            _query(runtime, target, key=(50, 1)))
        use = runtime.adopt_understanding(resolution)
        assert len(use.relation_uses) == 2
        choice = runtime.choose_generation(generation_request_for_candidate(
            target,
            request_key=LosslessIntegerKey((50, 2)),
            constraints=_constraints(target),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())

        learning.withdraw_evidence(
            _support_account(learning, target), withdrawal_level=1)
        current = runtime.understanding.preview(
            _query(runtime, target, key=(50, 3)))
        assert current.status == W06_R05_SUPPORTED
        assert target.proposition.proposition not in current.propositions
        assert runtime.verify_understanding(use).verdict == (
            W06_R05_OUTCOME_REFUTE)
        assert runtime.verify_generation(generation_use).verdict == (
            W06_R05_OUTCOME_REFUTE)
        stale_choice = runtime.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((50, 4)),
                constraints=_constraints(target),
            ))
        assert stale_choice.status == W06_R05_GENERATION_UNKNOWN
    finally:
        backend.close()


def test_w06_r05_budget_fails_closed_without_partial_result(adapted):
    """预算耗尽时不得追加 resolution 或 Use 前缀。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        similar = next(
            item for item in _active_candidates(sliced, learning)
            if item.relation_family == "SIMILAR")
        tight = _query(
            runtime,
            similar,
            key=(60, 1),
            budget=SymmetricRelationBudget(1, 1),
        )
        with pytest.raises(SymmetricRelationBudgetExceeded):
            runtime.resolve_understanding(tight)
        assert runtime.understanding.resolutions == ()
        assert runtime.understanding.uses == ()
    finally:
        backend.close()


def test_w06_r05_replay_is_bit_identical(adapted):
    """同一 adapter 和请求序列必须产生 bit-identical 状态。"""
    backend_a, sliced_a, learning_a, runtime_a = _build(adapted)
    backend_b, sliced_b, learning_b, runtime_b = _build(adapted)
    try:
        _exercise_positive(runtime_a, sliced_a, learning_a)
        _exercise_positive(runtime_b, sliced_b, learning_b)
        assert runtime_a.state_key() == runtime_b.state_key()
        assert runtime_a.report() == runtime_b.report()
    finally:
        backend_a.close()
        backend_b.close()
