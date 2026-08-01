"""PH2 W-04 公开 runtime receipt 的安全和不可覆盖发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w04_candidate import (
    W04_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_EVALUATION_ORDER,
    W04_OPEN_GENERATION_STATE,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_contract import (
    W04_PRIVATE_ABLATION_KEYS,
    W04PrivateDimensionResult,
    evidence_commitment,
    public_safe_w04_aggregate,
)
from pure_integer_ai.experiments.ph2_w04_release import (
    W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
    W04_REQUIRED_VERIFICATION_JOBS,
    W04ReleaseError,
    publish_w04_runtime_receipt,
)


def _fixture(tmp_path: Path):
    dimensions = tuple(W04PrivateDimensionResult(
        dimension,
        "PASS",
        1,
        1,
        0,
        0,
        evidence_commitment({"dimension": dimension}),
    ) for dimension in W04_EVALUATION_ORDER)
    aggregate = public_safe_w04_aggregate(
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
        for ordinal, key in enumerate(W04_PRIVATE_ABLATION_KEYS)
    ]
    aggregate["generation_ablation_statuses"] = [
        "PASS", "PASS", "PASS", "PASS", "FAIL"]
    aggregate["infrastructure"] = {
        "candidate_inventory_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_bytes(canonical_json_bytes(aggregate))
    aggregate_sha = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    host_path = tmp_path / "candidate_host_freeze.json"
    host_path.write_bytes(canonical_json_bytes({
        "candidate_contract_sha256": "6" * 64,
        "execution_state": dict(W04_FORMAL_EXECUTION_STATE),
        "formal_run_count": 1,
        "open_generation_state": W04_OPEN_GENERATION_STATE,
        "self_excluded": 1,
    }))
    host_sha = hashlib.sha256(host_path.read_bytes()).hexdigest()
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_bytes(canonical_json_bytes({
        "aggregate_sha256": aggregate_sha,
        "artifact_kind": "PH2_W04_RUNTIME_RECEIPT_RECOMMENDATION",
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
        "w03_receipt_sha256": "7" * 64,
        "d03_receipt_sha256": "8" * 64,
        "d03_global_manifest_sha256": "9" * 64,
        "d03_stage_manifest_sha256": "a" * 64,
        "pre_w04_gate_sha256": "b" * 64,
        "publication_commit_sha1": "c" * 40,
        "verification_run_id": 20260801,
        "verification_jobs": tuple(
            (name, "PASS") for name in W04_REQUIRED_VERIFICATION_JOBS),
    }


def test_w04_public_receipt_is_safe_and_non_overwritable(tmp_path: Path):
    """PASS receipt 精确保留 W04 状态且不可覆盖。"""
    kwargs = _kwargs(tmp_path)
    path, digest = publish_w04_runtime_receipt(tmp_path, **kwargs)
    before = path.read_bytes()
    receipt = json.loads(before.decode("utf-8"))
    assert path.relative_to(tmp_path).as_posix() == (
        W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH)
    assert hashlib.sha256(before).hexdigest() == digest
    assert receipt["execution_state"]["W04_STARTED"] == 1
    assert receipt["execution_state"]["W05_STARTED"] == 0
    assert receipt["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert receipt["open_generation_state"] == W04_OPEN_GENERATION_STATE
    assert all(token not in before for token in (
        b"surface", b"expected", b"private_path"))
    with pytest.raises(W04ReleaseError, match="不可覆盖"):
        publish_w04_runtime_receipt(tmp_path, **kwargs)
    assert path.read_bytes() == before


def test_w04_public_receipt_rejects_non_pass_aggregate(tmp_path: Path):
    """FAIL/NE aggregate 不得发布为 runtime evidence。"""
    kwargs = _kwargs(tmp_path)
    aggregate_path = Path(kwargs["aggregate_path"])
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["fail_count"] = 1
    aggregate_path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(
        aggregate_path.read_bytes()).hexdigest()
    with pytest.raises(W04ReleaseError, match="PASS hard conjunct"):
        publish_w04_runtime_receipt(tmp_path, **kwargs)
