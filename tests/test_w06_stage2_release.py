"""PH2 W-06 公开 runtime receipt 的安全和不可覆盖发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_candidate import (
    W06_CANDIDATE_HOST_FREEZE_KIND,
    W06_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_EVALUATION_ORDER,
    W06_OPEN_GENERATION_STATE,
    W06_PRIVATE_ABLATION_KEYS,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_contract import (
    W06PrivateDimensionResult,
    evidence_commitment,
    public_safe_w06_aggregate,
)
from pure_integer_ai.experiments.ph2_w06_release import (
    W06_EXPECTED_PARENT_IDENTITIES,
    W06_EXPECTED_RETENTION_IDENTITIES,
    W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
    W06_REQUIRED_VERIFICATION_JOBS,
    W06ReleaseError,
    publish_w06_runtime_receipt,
    read_w06_runtime_receipt,
)


def _fixture(tmp_path: Path):
    dimensions = tuple(W06PrivateDimensionResult(
        dimension,
        "PASS",
        1,
        1,
        0,
        0,
        evidence_commitment({"dimension": dimension}),
    ) for dimension in W06_EVALUATION_ORDER)
    aggregate = public_safe_w06_aggregate(
        dimensions,
        family_commitment="1" * 64,
        payload_commitment="2" * 64,
        case_commitment="3" * 64,
        label_commitment="4" * 64,
        cluster_commitment="5" * 64,
        failure_phase="NONE",
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )
    aggregate["ablation_results"] = [
        {
            "ablation_key": key,
            "dimension_statuses": [
                "FAIL" if index == ordinal else "PASS"
                for index in range(len(W06_EVALUATION_ORDER))
            ],
        }
        for ordinal, key in enumerate(W06_PRIVATE_ABLATION_KEYS)
    ]
    aggregate["generation_ablation_statuses"] = [
        *("PASS" for _ in range(len(W06_EVALUATION_ORDER) - 1)),
        "FAIL",
    ]
    aggregate["infrastructure"] = {
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
        "relation_scope_cell_count": 27,
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))
    aggregate_sha = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()

    artifact_counts = [
        ["ACTIVE_RELATION", 17],
        ["CANDIDATE", 50],
        ["CARRIER_PROJECTION", 9],
        ["EVIDENCE_ACCOUNT", 64],
        ["EVIDENCE_APPLICATION", 50],
        ["LOGICAL_SHARD", 16],
        ["RELATION_FAMILY", 14],
        ["RELATION_SCOPE_CELL", 27],
        ["RELATION_USE", 3],
        ["SCHEMA_REJECTION", 1],
        ["SUBSTAGE", 7],
    ]
    host_digests = {
        "active_projection": "a" * 64,
        "candidate": "b" * 64,
        "carrier_scope": "c" * 64,
        "logical": "d" * 64,
        "relation": "e" * 64,
        "source_evidence": "f" * 64,
        "transaction": "0" * 64,
    }
    resource_report = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": 14_000,
        "actual_payload_bytes": 199_296,
        "actual_payload_gets": 54,
        "actual_recompute_objects": 123,
        "actual_records": 541,
        "actual_segments": 59,
        "actual_workers": 4,
        "teacher_calls": 0,
    }
    host_evidence = {
        "artifact_counts": artifact_counts,
        "dump_manifest_sha256": "7" * 64,
        "dump_readback": 0,
        "execution_state": dict(W06_FORMAL_EXECUTION_STATE),
        "host_digests": host_digests,
        "learning_attempt_count": 1,
        "new_learning_write_count": 126,
        "owned_tables": ["graph_object", "ph2_w06_transaction_event"],
        "payload_bytes_this_call": 199_296,
        "payload_gets_this_call": 54,
        "resource_report": resource_report,
        "retention_sha256": [
            list(item) for item in W06_EXPECTED_RETENTION_IDENTITIES],
        "teacher_calls": 0,
        "transaction_event_count": 5,
    }
    readback = dict(host_evidence)
    readback.update({
        "dump_readback": 1,
        "new_learning_write_count": 0,
        "payload_bytes_this_call": 0,
        "payload_gets_this_call": 0,
    })
    host_path = tmp_path / "candidate_host_freeze.json"
    host_path.write_bytes(canonical_json_bytes({
        "artifact_kind": W06_CANDIDATE_HOST_FREEZE_KIND,
        "candidate_contract_sha256": "6" * 64,
        "dump_readback_evidence": readback,
        "execution_state": dict(W06_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "host_evidence": host_evidence,
        "open_generation_state": W06_OPEN_GENERATION_STATE,
        "owner_write_counts": {
            "artifact_writes": 258,
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": 0,
            "teacher_calls": 0,
        },
        "remote_commit_sha1": "c" * 40,
        "self_excluded": 1,
    }))
    host_sha = hashlib.sha256(host_path.read_bytes()).hexdigest()
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_bytes(canonical_json_bytes({
        "aggregate_sha256": aggregate_sha,
        "artifact_kind": "PH2_W06_RUNTIME_RECEIPT_RECOMMENDATION",
        "candidate_host_freeze_sha256": host_sha,
        "family_commitment": "1" * 64,
        "formal_run_count": 1,
        "format_version": 1,
        "recommend_runtime_receipt": 1,
    }))
    recommendation_sha = hashlib.sha256(
        recommendation_path.read_bytes()).hexdigest()
    return (
        aggregate_path,
        aggregate_sha,
        recommendation_path,
        recommendation_sha,
        host_path,
        host_sha,
    )


def _kwargs(tmp_path: Path):
    aggregate, aggregate_sha, recommendation, recommendation_sha, host, host_sha = (
        _fixture(tmp_path))
    return {
        "aggregate_path": aggregate,
        "aggregate_sha256": aggregate_sha,
        "recommendation_path": recommendation,
        "recommendation_sha256": recommendation_sha,
        "candidate_host_freeze_path": host,
        "candidate_contract_sha256": "6" * 64,
        "candidate_first_run_guard_sha256": "7" * 64,
        "candidate_host_freeze_sha256": host_sha,
        "private_family_freeze_sha256": "8" * 64,
        "private_first_run_guard_sha256": "9" * 64,
        "w05_receipt_sha256": W06_EXPECTED_PARENT_IDENTITIES[
            "w05_receipt_sha256"],
        "d03_global_manifest_sha256": W06_EXPECTED_PARENT_IDENTITIES[
            "d03_global_manifest_sha256"],
        "d03_stage_manifest_sha256": W06_EXPECTED_PARENT_IDENTITIES[
            "d03_stage_manifest_sha256"],
        "invalidation_graph_sha256": W06_EXPECTED_PARENT_IDENTITIES[
            "invalidation_graph_sha256"],
        "source_overlay_sha256": W06_EXPECTED_PARENT_IDENTITIES[
            "source_overlay_sha256"],
        "publication_commit_sha1": "c" * 40,
        "verification_run_id": 20260803,
        "verification_jobs": tuple(
            (name, "PASS") for name in W06_REQUIRED_VERIFICATION_JOBS),
    }


def test_w06_public_receipt_is_safe_and_non_overwritable(tmp_path: Path):
    """PASS receipt 精确保留 W06 状态且不可覆盖。"""
    kwargs = _kwargs(tmp_path)
    path, digest = publish_w06_runtime_receipt(tmp_path, **kwargs)
    before = path.read_bytes()
    receipt = json.loads(before.decode("utf-8"))
    assert path.relative_to(tmp_path).as_posix() == (
        W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH)
    assert hashlib.sha256(before).hexdigest() == digest
    assert receipt["execution_state"]["W06_STARTED"] == 1
    assert receipt["execution_state"]["W07_STARTED"] == 0
    assert receipt["execution_state"]["W06_RUNTIME_EVIDENCED"] == 1
    assert receipt["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert receipt["candidate_evidence"]["artifact_counts"][-3] == [
        "RELATION_USE", 3]
    assert len(receipt["dimension_results"]) == 8
    assert len(receipt["ablation_results"]) == 8
    assert read_w06_runtime_receipt(tmp_path) == receipt
    assert receipt["open_generation_state"] == W06_OPEN_GENERATION_STATE
    assert all(token not in before for token in (
        b"surface", b"expected", b"private_path", b"message"))
    with pytest.raises(W06ReleaseError, match="不可覆盖"):
        publish_w06_runtime_receipt(tmp_path, **kwargs)
    assert path.read_bytes() == before


def test_w06_public_receipt_rejects_non_pass_aggregate(tmp_path: Path):
    """FAIL/NE aggregate 不得发布为 runtime evidence。"""
    kwargs = _kwargs(tmp_path)
    aggregate_path = Path(kwargs["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["fail_count"] = 1
    aggregate_path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(
        aggregate_path.read_bytes()).hexdigest()
    with pytest.raises(W06ReleaseError, match="PASS hard conjunct"):
        publish_w06_runtime_receipt(tmp_path, **kwargs)


def test_w06_public_receipt_rejects_nonorthogonal_ablation(tmp_path: Path):
    """任一消融未精确击穿目标维度时不得发布。"""
    kwargs = _kwargs(tmp_path)
    aggregate_path = Path(kwargs["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["ablation_results"][0]["dimension_statuses"][1] = "FAIL"
    aggregate_path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(
        aggregate_path.read_bytes()).hexdigest()
    with pytest.raises(W06ReleaseError, match="正交击穿"):
        publish_w06_runtime_receipt(tmp_path, **kwargs)


def test_w06_public_receipt_rejects_candidate_isolation_drift(tmp_path: Path):
    """candidate readback 或 owner write 漂移不得进入 receipt。"""
    kwargs = _kwargs(tmp_path)
    host_path = Path(kwargs["candidate_host_freeze_path"])
    value = json.loads(host_path.read_text(encoding="utf-8"))
    value["dump_readback_evidence"]["new_learning_write_count"] = 1
    host_path.write_bytes(canonical_json_bytes(value))
    kwargs["candidate_host_freeze_sha256"] = hashlib.sha256(
        host_path.read_bytes()).hexdigest()
    recommendation_path = Path(kwargs["recommendation_path"])
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["candidate_host_freeze_sha256"] = kwargs[
        "candidate_host_freeze_sha256"]
    recommendation_path.write_bytes(canonical_json_bytes(recommendation))
    kwargs["recommendation_sha256"] = hashlib.sha256(
        recommendation_path.read_bytes()).hexdigest()
    with pytest.raises(W06ReleaseError, match="host freeze 状态"):
        publish_w06_runtime_receipt(tmp_path, **kwargs)


def test_w06_public_receipt_rejects_parent_identity_drift(tmp_path: Path):
    """任一冻结 parent SHA 漂移都不得进入公开 receipt。"""
    kwargs = _kwargs(tmp_path)
    kwargs["source_overlay_sha256"] = "2" * 64
    with pytest.raises(W06ReleaseError, match="parent identity"):
        publish_w06_runtime_receipt(tmp_path, **kwargs)
