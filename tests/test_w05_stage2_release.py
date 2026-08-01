"""PH2 W-05 公开 runtime receipt 的安全和不可覆盖发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_candidate import (
    W05_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_EVALUATION_ORDER,
    W05_OPEN_GENERATION_STATE,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_contract import (
    W05PrivateDimensionResult,
    evidence_commitment,
    public_safe_w05_aggregate,
)
from pure_integer_ai.experiments.ph2_w05_contract import W05_PRIVATE_ABLATION_KEYS
from pure_integer_ai.experiments.ph2_w05_release import (
    W05_EXPECTED_PARENT_IDENTITIES,
    W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
    W05_REQUIRED_VERIFICATION_JOBS,
    W05ReleaseError,
    publish_w05_runtime_receipt,
    read_w05_runtime_receipt,
)


def _fixture(tmp_path: Path):
    dimensions = tuple(W05PrivateDimensionResult(
        dimension,
        "PASS",
        1,
        1,
        0,
        0,
        evidence_commitment({"dimension": dimension}),
    ) for dimension in W05_EVALUATION_ORDER)
    aggregate = public_safe_w05_aggregate(
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
                for index in range(5)
            ],
        }
        for ordinal, key in enumerate(W05_PRIVATE_ABLATION_KEYS)
    ]
    aggregate["generation_ablation_statuses"] = [
        "PASS", "PASS", "PASS", "PASS", "FAIL"]
    aggregate["infrastructure"] = {
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
        "role_proposition_scope_cell_count": 27,
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))
    aggregate_sha = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    host_path = tmp_path / "candidate_host_freeze.json"
    artifact_counts = [
        ["CANDIDATE", 6],
        ["CARRIER_PROJECTION", 9],
        ["EVIDENCE_ACCOUNT", 8],
        ["EVIDENCE_APPLICATION", 6],
        ["GENERATION_CHOICE", 1],
        ["GENERATION_DECISION", 1],
        ["GENERATION_OUTCOME", 1],
        ["GENERATION_USE", 1],
        ["LOGICAL_SHARD", 16],
        ["OCCURRENCE", 19],
        ["REASONING_OUTCOME", 1],
        ["REASONING_USE", 1],
        ["ROLE_BINDING", 11],
        ["ROLE_PROPOSITION_SCOPE_CELL", 27],
        ["UNDERSTANDING_OUTCOME", 1],
        ["UNDERSTANDING_USE", 1],
    ]
    host_digests = {
        "candidate": "a" * 64,
        "carrier_scope": "b" * 64,
        "generation": "c" * 64,
        "logical": "d" * 64,
        "reasoning": "e" * 64,
        "transaction": "f" * 64,
        "understanding": "0" * 64,
    }
    resource_report = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": 4_800,
        "actual_payload_bytes": 167_589,
        "actual_payload_gets": 33,
        "actual_recompute_objects": 45,
        "actual_records": 360,
        "actual_segments": 15,
        "actual_workers": 4,
        "teacher_calls": 0,
    }
    host_evidence = {
        "artifact_counts": artifact_counts,
        "dump_manifest_sha256": "1" * 64,
        "dump_readback": 0,
        "execution_state": dict(W05_FORMAL_EXECUTION_STATE),
        "host_digests": host_digests,
        "learning_attempt_count": 1,
        "new_learning_write_count": 23,
        "owned_tables": ["graph_object", "ph2_w05_transaction_event"],
        "payload_bytes_this_call": 167_589,
        "payload_gets_this_call": 33,
        "resource_report": resource_report,
        "teacher_calls": 0,
        "transaction_event_count": 5,
    }
    readback_evidence = dict(host_evidence)
    readback_evidence.update({
        "dump_readback": 1,
        "new_learning_write_count": 0,
        "payload_bytes_this_call": 0,
        "payload_gets_this_call": 0,
    })
    host_path.write_bytes(canonical_json_bytes({
        "candidate_contract_sha256": "6" * 64,
        "dump_readback_evidence": readback_evidence,
        "execution_state": dict(W05_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "host_evidence": host_evidence,
        "open_generation_state": W05_OPEN_GENERATION_STATE,
        "owner_write_counts": {
            "artifact_writes": 110,
            "evaluator_label_writes": 0,
            "formal_training_runs": 1,
            "readback_learning_writes": 0,
            "teacher_calls": 0,
        },
        "self_excluded": 1,
    }))
    host_sha = hashlib.sha256(host_path.read_bytes()).hexdigest()
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_bytes(canonical_json_bytes({
        "aggregate_sha256": aggregate_sha,
        "artifact_kind": "PH2_W05_RUNTIME_RECEIPT_RECOMMENDATION",
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
        "candidate_host_freeze_sha256": host_sha,
        "w04_receipt_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "w04_receipt_sha256"],
        "d03_receipt_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "d03_receipt_sha256"],
        "d03_global_manifest_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "d03_global_manifest_sha256"],
        "d03_stage_manifest_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "d03_stage_manifest_sha256"],
        "invalidation_graph_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "invalidation_graph_sha256"],
        "atomic_pack_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "atomic_pack_sha256"],
        "pre_w04_gate_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "pre_w04_gate_sha256"],
        "lc16_overlay_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "lc16_overlay_sha256"],
        "lc16_mapper_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "lc16_mapper_sha256"],
        "lc16_projection_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "lc16_projection_sha256"],
        "lc16_directional_sha256": W05_EXPECTED_PARENT_IDENTITIES[
            "lc16_directional_sha256"],
        "publication_commit_sha1": "c" * 40,
        "verification_run_id": 20260801,
        "verification_jobs": tuple(
            (name, "PASS") for name in W05_REQUIRED_VERIFICATION_JOBS),
    }


def test_w05_public_receipt_is_safe_and_non_overwritable(tmp_path: Path):
    """PASS receipt 精确保留 W05 状态且不可覆盖。"""
    kwargs = _kwargs(tmp_path)
    path, digest = publish_w05_runtime_receipt(tmp_path, **kwargs)
    before = path.read_bytes()
    receipt = json.loads(before.decode("utf-8"))
    assert path.relative_to(tmp_path).as_posix() == (
        W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH)
    assert hashlib.sha256(before).hexdigest() == digest
    assert receipt["execution_state"]["W05_STARTED"] == 1
    assert receipt["execution_state"]["W06_STARTED"] == 0
    assert receipt["execution_state"]["W05_RUNTIME_EVIDENCED"] == 1
    assert receipt["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert receipt["candidate_evidence"]["artifact_counts"][-3][1] == 27
    assert read_w05_runtime_receipt(tmp_path) == receipt
    assert receipt["open_generation_state"] == W05_OPEN_GENERATION_STATE
    assert all(token not in before for token in (
        b"surface", b"expected", b"private_path"))
    with pytest.raises(W05ReleaseError, match="不可覆盖"):
        publish_w05_runtime_receipt(tmp_path, **kwargs)
    assert path.read_bytes() == before


def test_w05_public_receipt_rejects_non_pass_aggregate(tmp_path: Path):
    """FAIL/NE aggregate 不得发布为 runtime evidence。"""
    kwargs = _kwargs(tmp_path)
    aggregate_path = Path(kwargs["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["fail_count"] = 1
    aggregate_path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(
        aggregate_path.read_bytes()).hexdigest()
    with pytest.raises(W05ReleaseError, match="PASS hard conjunct"):
        publish_w05_runtime_receipt(tmp_path, **kwargs)


def test_w05_public_receipt_rejects_parent_identity_drift(tmp_path: Path):
    """任一冻结 parent SHA 漂移都不得进入公开 receipt。"""
    kwargs = _kwargs(tmp_path)
    kwargs["atomic_pack_sha256"] = "2" * 64
    with pytest.raises(W05ReleaseError, match="parent identity"):
        publish_w05_runtime_receipt(tmp_path, **kwargs)
