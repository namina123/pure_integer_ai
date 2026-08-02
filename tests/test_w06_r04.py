"""W06-R04 MEREOLOGY 的公开有界部分整体闭环专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyBudgetExceeded,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06_IDENTITY_VERSIONS,
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
from pure_integer_ai.experiments.ph2_w06_r04 import (
    W06R04Runtime,
    generation_request_for_candidate,
    query_for_candidate,
    slice_w06_r04_adapter,
)
from pure_integer_ai.experiments.ph2_w06_r04_contract import (
    W06R04ConsumerProtocol,
    W06R04ContractError,
    W06R04MereologyQuery,
    W06_R04_GENERATION_READY,
    W06_R04_GENERATION_REJECTED,
    W06_R04_GENERATION_UNKNOWN,
    W06_R04_OUTCOME_REFUTE,
    W06_R04_OUTCOME_SUPPORT,
    W06_R04_SUPPORTED,
    W06_R04_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r04_endpoint_projection import (
    W06_R04_ENDPOINT_PROJECTION_PATH,
    build_w06_r04_endpoint_projection,
    canonical_w06_r04_endpoint_projection_bytes,
    read_w06_r04_endpoint_projection,
)
from pure_integer_ai.experiments.ph2_w06_r04_generation import (
    W06_R04_POSTCHECK_BUDGET,
)
from pure_integer_ai.experiments.ph2_w06_r04_shared import (
    w06_r04_language_branch,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASELINE_HEAD = "6a1555857d194af758a7229de8f736accb3fc5db"
BUDGET = MereologyBudget(32, 128, 512, 32)


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
    """回读已发布 R04 endpoint projection。"""
    return read_w06_r04_endpoint_projection(
        ROOT / W06_R04_ENDPOINT_PROJECTION_PATH)


def _build(adapted, protocol=W06R04ConsumerProtocol()):
    """建立只含 MEREOLOGY train truth 的共享 W-06 learning/runtime。"""
    sliced = slice_w06_r04_adapter(adapted)
    backend = DictBackend()
    learning = build_w06_learning_runtime(backend, sliced)
    return backend, sliced, learning, W06R04Runtime(
        learning, sliced, _projection(), protocol=protocol)


def _active_candidates(sliced, learning):
    """返回 current active 的两个 supported MEREOLOGY candidate。"""
    return tuple(
        item for item in sliced.candidates
        if learning.snapshot_for(
            item.proposition.proposition).active_fact is not None
    )


def _constraints(candidate):
    """按 candidate 语言分支构造最小 generation 约束。"""
    branch = w06_r04_language_branch(candidate)
    return GenerationExpressionConstraints(
        branch,
        (),
        (branch,),
        0,
        0,
        0,
        128,
    )


def _query(runtime, candidate, *, key):
    """从 candidate 的 canonical part/whole 构造 R04 查询。"""
    return query_for_candidate(
        candidate,
        request_key=LosslessIntegerKey(key),
        endpoint_resolver=runtime.view.endpoint_resolver,
        budget=BUDGET,
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


def _exercise_positive(runtime, candidates):
    """对 active direct MEREOLOGY 执行 U/R/G 正向闭环。"""
    results = []
    for ordinal, candidate in enumerate(candidates, start=1):
        understanding = runtime.resolve_understanding(
            _query(runtime, candidate, key=(1, ordinal)))
        understanding_use = runtime.adopt_understanding(understanding)
        understanding_outcome = runtime.verify_understanding(understanding_use)
        reasoning = runtime.resolve_reasoning(
            _query(runtime, candidate, key=(2, ordinal)))
        reasoning_use = runtime.adopt_reasoning(reasoning)
        reasoning_outcome = runtime.verify_reasoning(reasoning_use)
        choice = runtime.choose_generation(generation_request_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((3, ordinal)),
            constraints=_constraints(candidate),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        generation_outcome = runtime.verify_generation(generation_use)
        results.append((
            candidate,
            understanding,
            understanding_use,
            understanding_outcome,
            reasoning,
            reasoning_use,
            reasoning_outcome,
            choice,
            generation_use,
            generation_outcome,
        ))
    return tuple(results)


def test_w06_r04_endpoint_projection_rebuild_and_readback_are_canonical():
    """R04 endpoint projection 必须只覆盖 train，且可从 parent 重建。"""
    target = ROOT / W06_R04_ENDPOINT_PROJECTION_PATH
    value = build_w06_r04_endpoint_projection(ROOT)
    assert target.read_bytes() == canonical_w06_r04_endpoint_projection_bytes(
        ROOT)
    assert value["projection_policy"] == {
        "canonical_endpoint_count": 5,
        "held_out_mapping_count": 0,
        "identity_basis": "AUTHORED_ENDPOINT_ID_NOT_SURFACE",
        "local_endpoint_count": 14,
        "rejected_type_mismatch_count": 0,
        "train_seed_count": 7,
    }
    projection = read_w06_r04_endpoint_projection(target)
    assert len(projection.entries) == 14
    assert len({item.canonical_endpoint for item in projection.entries}) == 5


def test_w06_r04_slice_keeps_seven_candidates_and_no_rejection(adapted):
    """七条 train MEREOLOGY 必须全部成为 candidate，且无 schema rejection。"""
    sliced = slice_w06_r04_adapter(adapted)
    assert len(sliced.candidates) == len(sliced.evidence) == 7
    assert len(sliced.rejections) == len(sliced.rejection_evidence) == 0
    assert {item.relation_family for item in sliced.candidates} == {
        "PART_OF", "HAS_PART"}
    assert len(sliced.schemas) == 2
    assert dict(sliced.execution_state) == dict(adapted.execution_state)


def test_w06_r04_direct_mereology_urg_use_and_generation_postcheck(adapted):
    """active PART_OF/HAS_PART 均须形成 U/R/G exact Use 与 postcheck。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)
        assert len(active) == 2
        assert {item.relation_family for item in active} == {
            "PART_OF", "HAS_PART"}
        results = _exercise_positive(runtime, active)
        for (_candidate, understanding, understanding_use,
             understanding_outcome, reasoning, reasoning_use,
             reasoning_outcome, choice, generation_use,
             generation_outcome) in results:
            assert understanding.status == W06_R04_SUPPORTED
            assert reasoning.status == W06_R04_SUPPORTED
            assert choice.status == W06_R04_GENERATION_READY
            assert understanding_outcome.verdict == W06_R04_OUTCOME_SUPPORT
            assert reasoning_outcome.verdict == W06_R04_OUTCOME_SUPPORT
            assert generation_outcome.verdict == W06_R04_OUTCOME_SUPPORT
            assert len(understanding_use.relation_uses) == 1
            assert len(reasoning_use.relation_uses) == 1
            assert len(generation_use.relation_uses) == 1
        report = runtime.report()
        assert report.candidate_count == 7
        assert report.rejection_count == 0
        assert report.active_count == 2
        assert report.refuted_count == 3
        assert report.conflict_count == 1
        assert report.superseded_count == 1
        assert report.understanding_use_count == 2
        assert report.reasoning_use_count == 2
        assert report.generation_use_count == 2
        assert report.generation_outcome_count == 2
        assert report.consumed_premise_count == 6
        assert report.private_read_count == 0
        assert report.formal_guard_read_count == 0
        assert report.w06_started == 0
    finally:
        backend.close()


def test_w06_r04_refute_conflict_supersede_and_inverse_are_separated(adapted):
    """方向反转、冲突、parser supersede 和 inverse closure 必须分态。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        snapshots = {
            item.source_record.revision_id: learning.snapshot_for(
                item.proposition.proposition)
            for item in sliced.candidates
        }
        assert snapshots["teacher-wheel-part-reversed-v1"].snapshot.epistemic_status == (
            EPISTEMIC_REFUTED)
        assert snapshots["teacher-car-contains-as-part-of-v1"].snapshot.epistemic_status == (
            EPISTEMIC_REFUTED)
        assert snapshots["teacher-wheel-category-confusion-v1"].snapshot.epistemic_status == (
            EPISTEMIC_REFUTED)
        assert (snapshots["teacher-battery-part-conflict-v1"].snapshot.lifecycle
                == LIFECYCLE_ACTIVE)
        assert (snapshots["teacher-battery-part-conflict-v1"].snapshot.epistemic_status
                == EPISTEMIC_CONFLICTED)
        assert (snapshots["teacher-wheel-part-of-car-v1"].snapshot.lifecycle
                == LIFECYCLE_SUPERSEDED)

        refuted = next(
            item for item in sliced.candidates
            if item.source_record.revision_id == "teacher-wheel-part-reversed-v1")
        resolution = runtime.resolve_reasoning(
            _query(runtime, refuted, key=(10, 1)))
        assert resolution.status == W06_R04_UNKNOWN
        assert not resolution.propositions

        has_part = next(
            item for item in _active_candidates(sliced, learning)
            if item.relation_family == "HAS_PART")
        learning.withdraw_evidence(
            _support_account(learning, has_part), withdrawal_level=1)
        inverse = runtime.reasoning.resolve(
            _query(runtime, has_part, key=(10, 2)))
        inverse_use = runtime.reasoning.adopt(inverse)
        assert inverse.status == W06_R04_SUPPORTED
        assert len(inverse_use.relation_uses) == 1
        assert inverse.evaluation.support_proof is not None
        assert len(inverse.evaluation.support_proof.applications) == 1
        assert has_part.proposition.proposition not in inverse.propositions
    finally:
        backend.close()


def test_w06_r04_unknown_future_and_structure_fail_closed(adapted):
    """未知端点、未来 relation family 与非 R04 candidate 必须 fail closed。"""
    backend, sliced, _learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, runtime.learning)[0]
        unknown = ObjectIdentity(
            OBJECT_ENTITY,
            (50604, 999, 1),
            versions=W06_IDENTITY_VERSIONS,
        )
        query = W06R04MereologyQuery(
            LosslessIntegerKey((20, 1)),
            active.relation_family,
            runtime.view.endpoints_for(active)[0],
            unknown,
            BUDGET,
        )
        resolution = runtime.resolve_understanding(query)
        assert resolution.status == W06_R04_UNKNOWN
        assert not resolution.propositions
        with pytest.raises(W06R04ContractError):
            W06R04MereologyQuery(
                LosslessIntegerKey((20, 2)),
                "PROPERTY",
                runtime.view.endpoints_for(active)[0],
                runtime.view.endpoints_for(active)[1],
                BUDGET,
            )
    finally:
        backend.close()


def test_w06_r04_target_and_generation_ablations_are_orthogonal(adapted):
    """查询桥、生成桥、方向和来源开关必须击穿对应维度。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)[0]
        query_off = W06R04Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R04ConsumerProtocol(
                mereology_bridge_connected=False),
        )
        assert query_off.resolve_understanding(
            _query(query_off, active, key=(30, 1))).status == W06_R04_UNKNOWN
        generation_off = W06R04Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R04ConsumerProtocol(generation_connected=False),
        )
        assert generation_off.resolve_reasoning(
            _query(generation_off, active, key=(30, 2))).status == (
                W06_R04_SUPPORTED)
        choice = generation_off.choose_generation(
            generation_request_for_candidate(
                active,
                request_key=LosslessIntegerKey((30, 3)),
                constraints=_constraints(active),
            ))
        assert choice.status == W06_R04_GENERATION_UNKNOWN
        direction_off = W06R04Runtime(
            learning,
            sliced,
            _projection(),
            protocol=W06R04ConsumerProtocol(direction_connected=False),
        )
        assert direction_off.resolve_understanding(
            _query(direction_off, active, key=(30, 4))).status == (
                W06_R04_UNKNOWN)
        rejected = direction_off.choose_generation(
            generation_request_for_candidate(
                active,
                request_key=LosslessIntegerKey((30, 5)),
                constraints=_constraints(active),
            ))
        assert rejected.status == W06_R04_GENERATION_REJECTED
    finally:
        backend.close()


def test_w06_r04_postcheck_ablation_refutes_generation_only(adapted):
    """postcheck 断开只应让 generation outcome REFUTE，不影响 choice ready。"""
    backend, sliced, learning, runtime = _build(
        adapted,
        protocol=W06R04ConsumerProtocol(postcheck_connected=False),
    )
    try:
        active = _active_candidates(sliced, learning)[0]
        choice = runtime.choose_generation(generation_request_for_candidate(
            active,
            request_key=LosslessIntegerKey((40, 1)),
            constraints=_constraints(active),
        ))
        assert choice.status == W06_R04_GENERATION_READY
        use = runtime.adopt_generation(choice, choice.options[0].stable_key())
        outcome = runtime.verify_generation(use)
        assert outcome.verdict == W06_R04_OUTCOME_REFUTE
        assert outcome.mereology_query_status == W06_R04_UNKNOWN
    finally:
        backend.close()


def test_w06_r04_withdrawal_demotes_direct_fact_and_stale_use(adapted):
    """append-only withdrawal 后 direct fact 退出且旧 Use 不得继续通过。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = next(
            item for item in _active_candidates(sliced, learning)
            if item.relation_family == "PART_OF")
        resolution = runtime.resolve_understanding(
            _query(runtime, active, key=(50, 1)))
        use = runtime.adopt_understanding(resolution)
        choice = runtime.choose_generation(generation_request_for_candidate(
            active,
            request_key=LosslessIntegerKey((50, 2)),
            constraints=_constraints(active),
        ))
        generation_use = runtime.adopt_generation(
            choice, choice.options[0].stable_key())
        learning.withdraw_evidence(
            _support_account(learning, active), withdrawal_level=1)
        current = runtime.understanding.preview(
            _query(runtime, active, key=(50, 3)))
        assert current.status == W06_R04_UNKNOWN
        assert runtime.verify_understanding(use).verdict == (
            W06_R04_OUTCOME_REFUTE)
        assert runtime.verify_generation(generation_use).verdict == (
            W06_R04_OUTCOME_REFUTE)
    finally:
        backend.close()


def test_w06_r04_budget_fails_closed_without_partial_result(adapted):
    """预算耗尽时不得追加 resolution 或 Use 前缀。"""
    backend, sliced, learning, runtime = _build(adapted)
    try:
        active = _active_candidates(sliced, learning)[0]
        tight = query_for_candidate(
            active,
            request_key=LosslessIntegerKey((60, 1)),
            endpoint_resolver=runtime.view.endpoint_resolver,
            budget=MereologyBudget(1, 1, 1, 1),
        )
        with pytest.raises(MereologyBudgetExceeded):
            runtime.resolve_understanding(tight)
        assert runtime.understanding.resolutions == ()
        assert runtime.understanding.uses == ()
    finally:
        backend.close()


def test_w06_r04_replay_is_bit_identical(adapted):
    """同一 projection、adapter 和请求序列必须产生 bit-identical 状态。"""
    backend_a, sliced_a, learning_a, runtime_a = _build(adapted)
    backend_b, sliced_b, learning_b, runtime_b = _build(adapted)
    try:
        active_a = _active_candidates(sliced_a, learning_a)
        active_b = _active_candidates(sliced_b, learning_b)
        _exercise_positive(runtime_a, active_a)
        _exercise_positive(runtime_b, active_b)
        assert runtime_a.state_key() == runtime_b.state_key()
        assert runtime_a.report() == runtime_b.report()
    finally:
        backend_a.close()
        backend_b.close()
