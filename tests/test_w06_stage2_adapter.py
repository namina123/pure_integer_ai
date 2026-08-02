"""W06-02 typed relation adapter 与 schema rejection 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_ENTITY
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06_REJECTION_TYPE_MISMATCH,
    W06TypedAdapterError,
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
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "4d57305bc4474081c9304a05287ab4783f49a849"


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> W06TrainingPayload:
    """通过一次性 public firewall 取得当前 18-pack train-only payload。"""
    path = tmp_path_factory.mktemp("w06-adapter") / "probe.sqlite"
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
    return W06PayloadFirewall.open(
        ROOT, context, request).read_training_payload()


@pytest.fixture(scope="module")
def adapted(payload):
    """执行不写 learning state 的确定性 adapter。"""
    return adapt_w06_training_payload(payload)


def _replace_observation(payload, original, replacement):
    """在不可变 payload 中只替换一个 Observation，供 fail-closed 探针使用。"""
    return W06TrainingPayload(
        payload.source_refs,
        tuple(
            replacement if item.stable_key == original.stable_key else item
            for item in payload.observations
        ),
        payload.teacher_evidence,
    )


def test_w06_adapter_splits_fifty_candidates_and_one_schema_rejection(adapted):
    """TYPE_MISMATCH 必须在 candidate 形成前分流，且全部执行状态保持零。"""
    assert len(adapted.candidates) == len(adapted.observations) == 50
    assert len(adapted.evidence) == 50
    assert len(adapted.rejections) == len(adapted.rejection_evidence) == 1
    assert len(adapted.schemas) == 14
    assert dict(adapted.execution_state) == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_STARTED": 0,
        "W07_STARTED": 0,
        "formal_w06_training_runs": 0,
        "teacher_calls": 0,
    }


def test_w06_adapter_keeps_fourteen_relation_families_and_one_schema_definition(
        adapted):
    """accepted 与 rejected 分账合并后覆盖 14 family，合法 schema identity 不分叉。"""
    families = {
        item.relation_family for item in adapted.candidates
    } | {
        item.relation_family for item in adapted.rejections
    }
    assert families == set(W06_RELATION_PROFILES)
    assert {item.relation_family for item in adapted.candidates} == families
    schema_identities = [item.schema for item in adapted.schemas]
    assert len(schema_identities) == len(set(schema_identities)) == 14
    assert all(
        item.schema.schema in set(schema_identities)
        for item in adapted.candidates
    )


def test_w06_type_mismatch_rejection_never_becomes_h05_evidence(adapted):
    """错误 MEMBER set filler 只形成可审计拒绝，不获得 hypothesis/Evidence 身份。"""
    rejection = adapted.rejections[0]
    assert rejection.relation_family == "MEMBER"
    assert rejection.observation.sample_role == "refute"
    assert rejection.observation.perturbation_kind == "TYPE_MISMATCH"
    assert rejection.rejection_state == W06_REJECTION_TYPE_MISMATCH
    assert len(rejection.violations) == 1
    violation = rejection.violations[0]
    assert violation.filler.object_kind == OBJECT_ENTITY
    assert violation.allowed_object_kinds == (18,)
    assert violation.declared_object_kinds == (OBJECT_ENTITY,)
    assert rejection.proposition not in {
        item.proposition.proposition for item in adapted.candidates
    }
    assert rejection.observation.stable_key not in {
        item.observation.stable_key for item in adapted.evidence
    }

    evidence = adapted.rejection_evidence[0]
    assert evidence.observation == rejection.observation
    assert evidence.expected_state == "FALSE"
    assert evidence.decision == "reject_type_mismatch"
    assert evidence.rejection_state == W06_REJECTION_TYPE_MISMATCH


def test_w06_adapter_preserves_source_revision_scope_direction_role_and_consumer(
        adapted):
    """accepted candidate 必须逐字段保留 payload 身份与来源，不从 surface 猜关系。"""
    for candidate in adapted.candidates:
        value = candidate.observation.typed_payload.to_value()
        assert candidate.relation_family == value["relation_family"]
        assert candidate.directionality == value["directionality"]
        assert candidate.surface == value["surface"]
        assert candidate.consumer_request.to_value() == value["consumer_request"]
        assert candidate.proposition.context == candidate.spec.proposition.context
        assert candidate.source_ref == candidate.proposition.source
        source_revision = candidate.provenance.to_value()["source_revision"]
        assert source_revision == {
            "parser_version": candidate.source_record.parser_version,
            "revision_id": candidate.source_record.revision_id,
            "snapshot_id": candidate.source_record.snapshot_id,
        }
        assert tuple(
            (
                list(item.identity.stable_key()),
                item.object_kind,
                item.start,
                item.end,
                item.ordinal,
                item.surface_fragment,
            )
            for item in candidate.endpoints
        ) == tuple(
            (
                item["endpoint_key"],
                item["object_kind"],
                item["start"],
                item["end"],
                item["ordinal"],
                item["surface_fragment"],
            )
            for item in value["endpoints"]
        )


def test_w06_adapter_is_independent_of_payload_record_order(payload, adapted):
    """输入物理顺序不得决定 schema identity 采用哪一份定义。"""
    reversed_payload = W06TrainingPayload(
        tuple(reversed(payload.source_refs)),
        tuple(reversed(payload.observations)),
        tuple(reversed(payload.teacher_evidence)),
    )
    assert adapt_w06_training_payload(reversed_payload) == adapted


def test_w06_adapter_fails_closed_on_unknown_relation(payload):
    """未知 relation family 不得绕过 registry 形成候选或 rejection。"""
    original = next(
        item for item in payload.observations
        if item.w_stage == "W-06" and item.perturbation_kind == "NONE"
    )
    value = original.typed_payload.to_value()
    value["relation_family"] = "UNKNOWN_RELATION"
    changed = replace(
        original,
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    with pytest.raises(W06TypedAdapterError, match="relation family"):
        adapt_w06_training_payload(
            _replace_observation(payload, original, changed))


def test_w06_adapter_rejects_schema_drift_outside_type_mismatch(payload):
    """普通样本的 slot 类型漂移必须 hard fail，不能降级成负例信封。"""
    original = next(
        item for item in payload.observations
        if (item.w_stage == "W-06"
            and item.substage == "SUBSET_MEMBER"
            and item.perturbation_kind == "NONE")
    )
    value = original.typed_payload.to_value()
    value["relation_schema"]["slots"][-1][
        "allowed_object_kinds"] = [OBJECT_ENTITY]
    changed = replace(
        original,
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    with pytest.raises(W06TypedAdapterError):
        adapt_w06_training_payload(
            _replace_observation(payload, original, changed))


def test_w06_adapter_rejects_empty_type_mismatch_marker(payload):
    """没有真实 Role 类型破坏的 TYPE_MISMATCH 标记不得制造 rejection PASS。"""
    original = next(
        item for item in payload.observations
        if (item.w_stage == "W-06"
            and item.substage == "SUBSET_MEMBER"
            and item.perturbation_kind == "NONE")
    )
    changed = replace(original, perturbation_kind="TYPE_MISMATCH")
    with pytest.raises(W06TypedAdapterError, match="没有真实类型破坏"):
        adapt_w06_training_payload(
            _replace_observation(payload, original, changed))


def test_w06_adapter_requires_explicit_false_rejection_evidence(payload):
    """schema rejection 缺少明确 FALSE teacher 决策时必须停止。"""
    observation = next(
        item for item in payload.observations
        if (item.w_stage == "W-06"
            and item.perturbation_kind == "TYPE_MISMATCH")
    )
    teacher = next(
        item for item in payload.teacher_evidence
        if item.observation_key == observation.stable_key
    )
    value = teacher.typed_evidence.to_value()
    value["expected_state"] = "TRUE"
    value["expected_payload"]["decision"] = "accept_member"
    changed = replace(
        teacher,
        typed_evidence=CanonicalJsonObject.from_value(value),
    )
    changed_payload = W06TrainingPayload(
        payload.source_refs,
        payload.observations,
        tuple(
            changed if item.stable_key == teacher.stable_key else item
            for item in payload.teacher_evidence
        ),
    )
    with pytest.raises(W06TypedAdapterError, match="明确 FALSE"):
        adapt_w06_training_payload(changed_payload)
