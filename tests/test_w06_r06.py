"""W06-R06 PRECEDES/event-time 的公开有界闭环专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
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
from pure_integer_ai.experiments.ph2_w06_r06 import (
    W06R06Runtime,
    generation_request_for_candidate,
    slice_w06_r06_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r06_contract import (
    W06R06Budget,
    W06R06BudgetExceeded,
    W06R06ConsumerProtocol,
    W06R06ContractError,
    W06R06EventTimeQuery,
    W06_R06_CONFLICT,
    W06_R06_GENERATION_READY,
    W06_R06_GENERATION_REJECTED,
    W06_R06_GENERATION_UNKNOWN,
    W06_R06_OUTCOME_REFUTE,
    W06_R06_OUTCOME_SUPPORT,
    W06_R06_REFUTED,
    W06_R06_SUPPORTED,
    W06_R06_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r06_endpoint_projection import (
    W06_R06_ENDPOINT_PROJECTION_PATH,
    canonical_w06_r06_endpoint_projection_bytes,
    read_w06_r06_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r06_shared import (
    candidate_event_time_qualifier,
    w06_r06_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = W06R06Budget(32, 64)


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


def _projection():
    return read_w06_r06_endpoint_projection(
        ROOT / W06_R06_ENDPOINT_PROJECTION_PATH)


def _build(adapted, protocol=W06R06ConsumerProtocol()):
    """建立只含 PRECEDES/train truth 的 learning/runtime。"""
    sliced = slice_w06_r06_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R06Runtime(
        learning, sliced, _projection(), protocol=protocol)


def _active_candidates(sliced, learning):
    return tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )


def _constraints(candidate):
    branch = w06_r06_language_branch(candidate)
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
    return runtime.query_for_candidate(
        candidate,
        request_key=LosslessIntegerKey(key),
        budget=budget,
    )


def _support_account(learning, candidate):
    for application in learning.applications():
        for account in application.accounts:
            if (account.candidate == candidate.proposition.proposition
                    and account.stance == EVIDENCE_SUPPORT
                    and not account.derived_supersede):
                return account
    raise AssertionError("未找到普通 support Evidence")


def _exercise_positive(runtime, sliced, learning):
    """对 BEFORE/AFTER/SAME 分别形成 U/R/G exact Use。"""
    query_results = []
    generation_results = []
    active = tuple(sorted(
        _active_candidates(sliced, learning),
        key=lambda item: item.proposition.proposition.stable_key(),
    ))
    for ordinal, candidate in enumerate(active, start=1):
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


def test_w06_r06_slice_and_endpoint_projection_are_train_only(adapted):
    """九条 train truth 均保留，18 个 local event 映射到 12 个端点。"""
    sliced = slice_w06_r06_adapter(adapted)
    assert len(sliced.candidates) == len(sliced.evidence) == 9
    assert len(sliced.rejections) == len(sliced.rejection_evidence) == 0
    assert {item.relation_family for item in sliced.candidates} == {
        "EVENT_BEFORE", "EVENT_AFTER", "EVENT_SAME", "EVENT_UNKNOWN"}
    assert len(sliced.schemas) == 4
    projection = _projection()
    assert len(projection.entries) == 18
    assert len({item.canonical_endpoint for item in projection.entries}) == 12
    assert canonical_w06_r06_endpoint_projection_bytes(ROOT) == (
        ROOT / W06_R06_ENDPOINT_PROJECTION_PATH).read_bytes()


def test_w06_r06_event_time_urg_use_generation_and_digests(adapted):
    """BEFORE/AFTER/SAME 均形成 U/R/G exact Use 与独立 postcheck。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        query_results, generation_results = _exercise_positive(
            runtime, sliced, learning)
        assert {item[0].relation_family for item in query_results} == {
            "EVENT_BEFORE", "EVENT_AFTER", "EVENT_SAME"}
        for (_candidate, understanding, understanding_use,
             understanding_outcome, reasoning, reasoning_use,
             reasoning_outcome) in query_results:
            assert understanding.status == W06_R06_SUPPORTED
            assert reasoning.status == W06_R06_SUPPORTED
            assert understanding_outcome.verdict == W06_R06_OUTCOME_SUPPORT
            assert reasoning_outcome.verdict == W06_R06_OUTCOME_SUPPORT
            assert len(understanding_use.relation_uses) == 1
            assert len(reasoning_use.relation_uses) == 1
        for _candidate, choice, generation_use, outcome in generation_results:
            assert choice.status == W06_R06_GENERATION_READY
            assert len(generation_use.relation_uses) == 1
            assert outcome.verdict == W06_R06_OUTCOME_SUPPORT
            assert outcome.relation_qualifier_preserved

        report = runtime.report()
        assert (report.candidate_count, report.rejection_count) == (9, 0)
        assert (
            report.active_count,
            report.refuted_count,
            report.unknown_count,
            report.conflict_count,
            report.superseded_count,
        ) == (3, 3, 1, 1, 1)
        assert report.understanding_use_count == 3
        assert report.reasoning_use_count == 3
        assert report.generation_use_count == 3
        assert report.generation_outcome_count == 3
        assert report.consumed_premise_count == 9
        assert report.occurrence_order_consumed == 0
        assert report.structure_order_consumed == 0
        assert report.causes_effect_count == 0
        assert report.private_read_count == report.formal_guard_read_count == 0
        assert report.w06_started == 0
        assert bytes(report.relation_digest).hex() == (
            "c74c80b7fcc36ed72a2d93815b9c1e893abf0bcd56e430864e2b53d5856f9b2b")
        assert bytes(report.source_evidence_digest).hex() == (
            "d591f14a2397d1b28f2be88f44e1ff60ea7d01b30c5820a9efd4f420c2224a10")
        assert bytes(report.active_projection_digest).hex() == (
            "64b5132e91b0577e4bba779ab8ac8cef3cf18f0838542d262e8a8b1daadba7dd")
    finally:
        backend.close()


def test_w06_r06_before_after_same_and_unknown_remain_distinct(adapted):
    """raw family 不合并；BEFORE/AFTER 仅共享 normalized before edge。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        before = runtime.resolve_reasoning(_query(
            runtime, by_revision["teacher-open-before-enter-v2"], key=(10, 1)))
        after = runtime.resolve_reasoning(_query(
            runtime, by_revision["teacher-enter-after-open-v1"], key=(10, 2)))
        same = runtime.resolve_reasoning(_query(
            runtime, by_revision["teacher-light-bell-same-v1"], key=(10, 3)))
        unknown = runtime.resolve_understanding(_query(
            runtime, by_revision["teacher-event-time-unknown-v1"], key=(10, 4)))
        assert before.status == after.status == same.status == W06_R06_SUPPORTED
        assert before.evaluation.normalized_before_edge == (
            after.evaluation.normalized_before_edge)
        assert before.request.relation_family != after.request.relation_family
        assert same.evaluation.normalized_before_edge == ()
        assert len(same.evaluation.same_group) == 2
        assert unknown.status == W06_R06_UNKNOWN
        assert unknown.evaluation.explicit_unknown
        assert unknown.propositions == (
            by_revision["teacher-event-time-unknown-v1"].proposition.proposition,)

        wrong_raw_family = replace(
            before.request,
            request_key=LosslessIntegerKey((10, 5)),
            relation_family="EVENT_AFTER",
            qualifier=candidate_event_time_qualifier(
                by_revision["teacher-enter-after-open-v1"]),
        )
        assert runtime.resolve_reasoning(wrong_raw_family).status == W06_R06_UNKNOWN
    finally:
        backend.close()


def test_w06_r06_lifecycle_refutes_conflict_and_supersede_are_separated(adapted):
    """三类 refute、current conflict、unknown 和 parser supersede 必须分态。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        for revision in (
                "teacher-open-before-enter-reversed-v1",
                "teacher-occurrence-order-confusion-v1",
                "teacher-structure-order-confusion-v1"):
            snapshot = learning.snapshot_for(
                by_revision[revision].proposition.proposition)
            assert snapshot.snapshot.lifecycle == LIFECYCLE_ARCHIVED
            assert snapshot.snapshot.epistemic_status == EPISTEMIC_REFUTED
            assert runtime.resolve_reasoning(_query(
                runtime, by_revision[revision],
                key=(20, len(runtime.reasoning.resolutions) + 1),
            )).status == W06_R06_REFUTED

        unknown = learning.snapshot_for(
            by_revision["teacher-event-time-unknown-v1"].proposition.proposition)
        assert unknown.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert unknown.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        conflict = learning.snapshot_for(
            by_revision[
                "teacher-depart-arrive-conflict-v1"].proposition.proposition)
        assert conflict.snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert conflict.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
        assert runtime.resolve_understanding(_query(
            runtime,
            by_revision["teacher-depart-arrive-conflict-v1"],
            key=(20, 4),
        )).status == W06_R06_CONFLICT
        superseded = learning.snapshot_for(
            by_revision["teacher-open-before-enter-v1"].proposition.proposition)
        assert superseded.snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        revised = runtime.resolve_understanding(_query(
            runtime, by_revision["teacher-open-before-enter-v1"], key=(20, 5)))
        assert revised.status == W06_R06_SUPPORTED
        assert by_revision[
            "teacher-open-before-enter-v1"].proposition.proposition not in (
                revised.propositions)
    finally:
        backend.close()


def test_w06_r06_neighbor_causes_and_order_shortcuts_fail_closed(adapted):
    """CAUSES、occurrence order、structure order 和无注册 family 不进时序 truth。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        for revision in (
                "teacher-occurrence-order-confusion-v1",
                "teacher-structure-order-confusion-v1"):
            assert runtime.resolve_understanding(_query(
                runtime, by_revision[revision],
                key=(30, len(runtime.understanding.resolutions) + 1),
            )).status == W06_R06_REFUTED
        assert runtime.report().occurrence_order_consumed == 0
        assert runtime.report().structure_order_consumed == 0
        assert runtime.report().causes_effect_count == 0

        active = by_revision["teacher-open-before-enter-v2"]
        query = _query(runtime, active, key=(30, 3))
        with pytest.raises(W06R06ContractError):
            W06R06EventTimeQuery(
                LosslessIntegerKey((30, 4)),
                "CAUSES",
                query.subject,
                query.object_identity,
                query.qualifier,
                BUDGET,
                query.source,
            )
        causes = adapted.candidates_for_substage("CAUSES")[0]
        with pytest.raises(W06R06ContractError):
            generation_request_for_candidate(
                causes,
                request_key=LosslessIntegerKey((30, 5)),
                constraints=_constraints(active),
            )
    finally:
        backend.close()


def test_w06_r06_does_not_invent_transitive_or_cross_state_closure(adapted):
    """无 direct raw family/endpoints/qualifier 的组合保持 UNKNOWN。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        by_revision = {item.source_record.revision_id: item
                       for item in sliced.candidates}
        before = _query(
            runtime, by_revision["teacher-open-before-enter-v2"], key=(40, 1))
        same = _query(
            runtime, by_revision["teacher-light-bell-same-v1"], key=(40, 2))
        unsupported = replace(
            before,
            request_key=LosslessIntegerKey((40, 3)),
            object_identity=same.object_identity,
        )
        resolution = runtime.resolve_reasoning(unsupported)
        assert resolution.status == W06_R06_UNKNOWN
        assert resolution.propositions == resolution.evidence_keys == ()
    finally:
        backend.close()


def test_w06_r06_relation_qualifier_and_generation_ablations_are_orthogonal(adapted):
    """PRECEDES state、qualifier 和 generation 门分别只击穿目标。"""
    backend, sliced, learning, _runtime = _build(adapted)
    try:
        active = {item.relation_family: item
                  for item in _active_candidates(sliced, learning)}
        before_off = W06R06Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R06ConsumerProtocol(before_connected=False),
        )
        assert before_off.resolve_understanding(
            _query(before_off, active["EVENT_BEFORE"], key=(50, 1))
        ).status == W06_R06_UNKNOWN
        assert before_off.resolve_understanding(
            _query(before_off, active["EVENT_AFTER"], key=(50, 2))
        ).status == W06_R06_SUPPORTED

        qualifier_off = W06R06Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R06ConsumerProtocol(qualifier_connected=False),
        )
        target = active["EVENT_BEFORE"]
        assert qualifier_off.resolve_reasoning(
            _query(qualifier_off, target, key=(50, 3))
        ).status == W06_R06_UNKNOWN
        choice = qualifier_off.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((50, 4)),
                constraints=_constraints(target),
            ))
        assert choice.status == W06_R06_GENERATION_REJECTED

        generation_off = W06R06Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R06ConsumerProtocol(generation_connected=False),
        )
        assert generation_off.resolve_reasoning(
            _query(generation_off, target, key=(50, 5))
        ).status == W06_R06_SUPPORTED
        choice = generation_off.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((50, 6)),
                constraints=_constraints(target),
            ))
        assert choice.status == W06_R06_GENERATION_UNKNOWN
    finally:
        backend.close()


def test_w06_r06_postcheck_ablation_refutes_generation_only(adapted):
    """postcheck 断开只让 generation outcome REFUTE，不影响 choice ready。"""
    backend, sliced, learning, runtime = _build(
        adapted,
        protocol=W06R06ConsumerProtocol(postcheck_connected=False),
    )
    try:
        active = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            active,
            request_key=LosslessIntegerKey((60, 1)),
            constraints=_constraints(active),
        ))
        assert choice.status == W06_R06_GENERATION_READY
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R06_OUTCOME_REFUTE
        assert outcome.event_time_query_status == W06_R06_UNKNOWN
    finally:
        backend.close()


def test_w06_r06_withdrawal_refutes_stale_query_and_generation_use(adapted):
    """撤回 active BEFORE 后旧 U/G exact Use 均必须失效。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        target = next(
            item for item in sliced.candidates
            if item.source_record.revision_id == "teacher-open-before-enter-v2")
        resolution = runtime.resolve_understanding(
            _query(runtime, target, key=(70, 1)))
        use = runtime.adopt_understanding(resolution)
        choice = runtime.choose_generation(generation_request_for_candidate(
            target,
            request_key=LosslessIntegerKey((70, 2)),
            constraints=_constraints(target),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())

        learning.withdraw_evidence(
            _support_account(learning, target), withdrawal_level=1)
        current = runtime.understanding.preview(
            _query(runtime, target, key=(70, 3)))
        assert current.status == W06_R06_UNKNOWN
        assert runtime.verify_understanding(use).verdict == W06_R06_OUTCOME_REFUTE
        assert runtime.verify_generation(generation_use).verdict == (
            W06_R06_OUTCOME_REFUTE)
        stale_choice = runtime.choose_generation(
            generation_request_for_candidate(
                target,
                request_key=LosslessIntegerKey((70, 4)),
                constraints=_constraints(target),
            ))
        assert stale_choice.status == W06_R06_GENERATION_UNKNOWN
    finally:
        backend.close()


def test_w06_r06_budget_fails_closed_without_partial_result(adapted):
    """预算耗尽时不得追加 resolution 或 Use 前缀。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        target = _active_candidates(sliced, learning)[0]
        tight = _query(
            runtime,
            target,
            key=(80, 1),
            budget=W06R06Budget(1, 1),
        )
        with pytest.raises(W06R06BudgetExceeded):
            runtime.resolve_understanding(tight)
        assert runtime.understanding.resolutions == ()
        assert runtime.understanding.uses == ()
    finally:
        backend.close()


def test_w06_r06_replay_is_bit_identical(adapted):
    """同一 adapter、projection 和请求序列必须产生 bit-identical 状态。"""
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
