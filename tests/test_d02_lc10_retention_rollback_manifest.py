"""LC-10 retention、回滚、范围收缩与零运行边界测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_retention_rollback_catalog import (
    LC10_MANIFEST_PATH,
    build_retention_protocol_fixtures,
    build_retention_rollback_manifest,
)
from pure_integer_ai.experiments.ph2_retention_rollback_contract import (
    EXECUTION_STATE,
    RETENTION_PHASE_KEYS,
    RetentionRollbackContractError,
    RetentionRollbackManifest,
    evaluate_retention_fixture,
    read_retention_rollback_manifest,
    write_retention_rollback_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST_SHA256 = (
    "f2f23eebabb2de30cebeed6a71ee2365cc590280d36e00fa90bf49feb3b29143")


@pytest.fixture(scope="module")
def formal_manifest() -> RetentionRollbackManifest:
    """只读构建一次当前 LC-10 manifest。"""
    return build_retention_rollback_manifest(REPOSITORY)


def _fixture(key: str):
    return next(
        item for item in build_retention_protocol_fixtures()
        if item.fixture_key == key)


def test_phase_order_and_three_direct_fixture_verdicts(formal_manifest):
    """A→B→重验 A 到范围收缩的顺序及正负结果必须直接可判。"""
    assert formal_manifest.retention_sequence == RETENTION_PHASE_KEYS
    assert {
        item.fixture_key: (
            item.expected_verdict, item.expected_failure_code)
        for item in formal_manifest.fixtures
    } == {
        "LC10_FORGETTING_REJECT_V1": (
            "REJECT", "OLD_CAPABILITY_FORGOTTEN"),
        "LC10_NO_CHANGE_ACCEPT_V1": ("PASS", "NONE"),
        "LC10_SCOPE_CONTRACTION_ACCEPT_V1": ("PASS", "NONE"),
    }
    for item in formal_manifest.fixtures:
        assert evaluate_retention_fixture(
            item.dimension_results, item.checkpoints) == (
                item.expected_verdict, item.expected_failure_code)


def test_no_mean_masking_and_scope_contraction_are_dimension_local():
    """旧维遗忘不得被正迁移平均掩盖，收缩必须逐维 NE。"""
    accepted = _fixture("LC10_NO_CHANGE_ACCEPT_V1")
    assert {item.classification for item in accepted.dimension_results} == {
        "INTERFERENCE", "NO_CHANGE", "POSITIVE_TRANSFER"}
    forgotten = _fixture("LC10_FORGETTING_REJECT_V1")
    assert evaluate_retention_fixture(
        accepted.dimension_results + forgotten.dimension_results,
        accepted.checkpoints,
    ) == ("REJECT", "OLD_CAPABILITY_FORGOTTEN")
    contracted = _fixture("LC10_SCOPE_CONTRACTION_ACCEPT_V1")
    removed = next(
        item for item in contracted.dimension_results
        if item.classification == "SCOPE_CONTRACTION")
    assert (removed.after_verdict, removed.in_scope_after) == ("NE", 0)


@pytest.mark.parametrize(("index", "field", "value", "failure"), (
    (2, "core_state_sha256", "0" * 64, "CORE_STATE_DRIFT"),
    (2, "unaffected_state_sha256", "1" * 64,
     "UNAFFECTED_CAPABILITY_DRIFT"),
    (4, "dump_artifact_sha256", "2" * 64,
     "DUMP_RESUME_DIGEST_DRIFT"),
    (5, "source_b_visible", 1, "SOURCE_WITHDRAWAL_VISIBILITY_DRIFT"),
    (5, "dependent_state_sha256", None,
     "SOURCE_WITHDRAWAL_NOT_LOCALIZED"),
    (4, "host_learning_writes", 1, "HOST_LEARNING_WRITE_NONZERO"),
))
def test_state_drift_and_nonzero_host_writes_reject(
        index, field, value, failure):
    """Core、未受影响维、恢复、撤回和宿主写的破坏必须 fail closed。"""
    fixture = _fixture("LC10_NO_CHANGE_ACCEPT_V1")
    checkpoints = list(fixture.checkpoints)
    if value is None:
        value = checkpoints[1].dependent_state_sha256
    checkpoints[index] = replace(checkpoints[index], **{field: value})
    assert evaluate_retention_fixture(
        fixture.dimension_results, tuple(checkpoints)) == ("REJECT", failure)


def test_rollback_locality_and_scope_receipt_mismatch_reject():
    """rollback 必须回到 A；scope 改动与逐维 receipt 必须一致且可重放。"""
    fixture = _fixture("LC10_SCOPE_CONTRACTION_ACCEPT_V1")
    checkpoints = list(fixture.checkpoints)
    checkpoints[6] = replace(
        checkpoints[6], dependent_state_sha256="3" * 64)
    assert evaluate_retention_fixture(
        fixture.dimension_results, tuple(checkpoints)) == (
            "REJECT", "ROLLBACK_NOT_BASELINE_EQUIVALENT")

    checkpoints = list(fixture.checkpoints)
    checkpoints[9] = replace(
        checkpoints[9], declared_scope_sha256="4" * 64)
    assert evaluate_retention_fixture(
        fixture.dimension_results, tuple(checkpoints)) == (
            "REJECT", "SCOPE_CONTRACTION_NOT_REPLAYABLE")

    no_change = _fixture("LC10_NO_CHANGE_ACCEPT_V1")
    assert evaluate_retention_fixture(
        no_change.dimension_results, fixture.checkpoints) == (
            "REJECT", "SCOPE_CONTRACTION_RECEIPT_MISMATCH")


def test_runtime_bindings_and_execution_boundary_are_explicit(formal_manifest):
    """现有设施可绑定但未执行，通用撤回和未来 clone 不得冒充 PASS。"""
    states = {
        item.binding_key: item.binding_state
        for item in formal_manifest.runtime_bindings
    }
    assert states["GENERAL_SOURCE_WITHDRAWAL"] == "PROTOCOL_ONLY_NE"
    assert states["V06_RETENTION_ISOLATION_CLONE"] == "FUTURE_REQUIRED"
    assert set(states.values()) == {
        "AVAILABLE_NOT_EXECUTED", "FUTURE_REQUIRED", "PROTOCOL_ONLY_NE"}
    assert formal_manifest.artifact_status == "COURSE_FROZEN"
    assert formal_manifest.runtime_status == "NOT_STARTED"
    assert formal_manifest.actual_retention_evidenced == 0
    assert formal_manifest.runtime_pass_authority == 0
    assert formal_manifest.execution_state.to_value() == EXECUTION_STATE
    assert all(value == 0 for value in EXECUTION_STATE.values())


def test_evidence_file_identity_is_bound_to_current_repository(formal_manifest):
    """所有 runtime binding 引用必须闭合到当前文件 byte/hash 身份。"""
    referenced = {
        path
        for binding in formal_manifest.runtime_bindings
        for path in binding.evidence_refs
    }
    inventoried = {item.relative_path for item in formal_manifest.evidence_files}
    assert referenced == inventoried
    for item in formal_manifest.evidence_files:
        path = REPOSITORY / Path(*item.relative_path.split("/"))
        payload = path.read_bytes()
        assert len(payload) == item.byte_count
        assert hashlib.sha256(payload).hexdigest() == item.sha256


def test_manifest_round_trip_nonoverwrite_and_corruption(tmp_path, formal_manifest):
    """LC-10 manifest 必须规范回读、幂等发布并拒绝覆盖和损坏。"""
    path = tmp_path / "lc10.json"
    assert write_retention_rollback_manifest(formal_manifest, path) == path
    assert write_retention_rollback_manifest(formal_manifest, path) == path
    restored = read_retention_rollback_manifest(path)
    assert restored == formal_manifest
    path.write_bytes(b"{}\n")
    with pytest.raises(RetentionRollbackContractError):
        write_retention_rollback_manifest(formal_manifest, path)
    with pytest.raises(RetentionRollbackContractError):
        read_retention_rollback_manifest(path)


def test_manifest_rejects_future_execution_and_bad_evidence(formal_manifest):
    """任何 retention/V-06 执行声明或 evidence inventory 漏项都必须拒绝。"""
    with pytest.raises(RetentionRollbackContractError):
        replace(formal_manifest, actual_retention_evidenced=1)
    bad_state = dict(EXECUTION_STATE)
    bad_state["teacher_calls"] = 1
    with pytest.raises(RetentionRollbackContractError):
        replace(
            formal_manifest,
            execution_state=CanonicalJsonObject.from_value(bad_state),
        )
    with pytest.raises(RetentionRollbackContractError):
        replace(formal_manifest, evidence_files=formal_manifest.evidence_files[:-1])


def test_repository_formal_artifact_matches_current_builder(formal_manifest):
    """正式不可覆盖 artifact 必须逐字节等于当前 builder 并绑定固定 hash。"""
    path = REPOSITORY / LC10_MANIFEST_PATH
    assert path.is_file()
    payload = path.read_bytes()
    assert payload == formal_manifest.canonical_bytes()
    assert hashlib.sha256(payload).hexdigest() == FORMAL_MANIFEST_SHA256
    assert read_retention_rollback_manifest(path) == formal_manifest
