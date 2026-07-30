"""PH2 W-03 公开 runtime receipt 的安全和不可覆盖发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_EVALUATION_ORDER,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_contract import (
    W03PrivateDimensionResult,
    evidence_commitment,
    public_safe_w03_aggregate,
)
from pure_integer_ai.experiments.ph2_w03_release import (
    W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
    W03ReleaseError,
    publish_w03_runtime_receipt,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str, str]:
    """形成不含 private surface/case 的安全 release fixture。"""
    dimensions = tuple(
        W03PrivateDimensionResult(
            dimension,
            "PASS",
            1,
            1,
            0,
            0,
            evidence_commitment({"dimension": dimension}),
        )
        for dimension in W03_EVALUATION_ORDER
    )
    aggregate = public_safe_w03_aggregate(
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
    aggregate.update({
        "ablation_results": [
            {
                "ablation_key": key,
                "dimension_statuses": [
                    "FAIL" if index == ordinal else "PASS"
                    for index in range(len(W03_EVALUATION_ORDER))],
            }
            for ordinal, key in enumerate(W03_ABLATION_KEYS)
        ],
        "generation_ablation_statuses": ["PASS", "PASS", "PASS", "PASS", "FAIL"],
        "infrastructure": {
            "candidate_inventory_match": 1,
            "clone_dump_readback": 1,
            "clone_host_copy_match": 1,
            "host_copy_unchanged": 1,
            "label_writes": 0,
            "restore_learning_writes": 0,
        },
    })
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_bytes = canonical_json_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_bytes)
    aggregate_sha = hashlib.sha256(aggregate_bytes).hexdigest()
    host_path = tmp_path / "candidate_host_freeze.json"
    host = canonical_json_bytes({
        "candidate_contract_sha256": "6" * 64,
        "execution_state": {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W03_STARTED": 1,
            "W04_STARTED": 0,
            "formal_w03_training_runs": 1,
            "teacher_calls": 0,
        },
        "formal_run_count": 1,
        "self_excluded": 1,
    })
    host_path.write_bytes(host)
    host_sha = hashlib.sha256(host).hexdigest()
    recommendation_path = tmp_path / "runtime_receipt_recommendation.json"
    recommendation = canonical_json_bytes({
        "aggregate_sha256": aggregate_sha,
        "artifact_kind": "PH2_W03_RUNTIME_RECEIPT_RECOMMENDATION",
        "candidate_host_freeze_sha256": host_sha,
        "family_commitment": "1" * 64,
        "formal_run_count": 1,
        "format_version": 1,
        "recommend_runtime_receipt": 1,
    })
    recommendation_path.write_bytes(recommendation)
    return (
        aggregate_path,
        recommendation_path,
        host_path,
        aggregate_sha,
        hashlib.sha256(recommendation).hexdigest(),
        host_sha,
    )


def test_public_receipt_is_safe_and_non_overwritable(tmp_path: Path):
    """PASS receipt 只含 commitment/count，重复发布不覆盖原 bytes。"""
    aggregate, recommendation, host, aggregate_sha, recommendation_sha, host_sha = _fixture(tmp_path)
    kwargs = {
        "aggregate_path": aggregate,
        "aggregate_sha256": aggregate_sha,
        "recommendation_path": recommendation,
        "recommendation_sha256": recommendation_sha,
        "candidate_host_freeze_path": host,
        "candidate_contract_sha256": "6" * 64,
        "candidate_host_freeze_sha256": host_sha,
        "w02_receipt_sha256": "8" * 64,
        "d03_receipt_sha256": "9" * 64,
        "d03_global_manifest_sha256": "a" * 64,
        "d03_stage_manifest_sha256": "b" * 64,
        "publication_commit_sha1": "c" * 40,
        "publication_ci_run_id": 123,
        "publication_ci_jobs": (
            ("Python 3.11 on ubuntu-latest", "success"),
            ("Python 3.14 on ubuntu-latest", "success"),
            ("Python 3.14 on windows-latest", "success"),
            ("Secret scan", "success"),
        ),
    }
    path, digest = publish_w03_runtime_receipt(tmp_path, **kwargs)
    before = path.read_bytes()
    receipt = json.loads(before.decode("utf-8"))
    assert path.relative_to(tmp_path).as_posix() == W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH
    assert hashlib.sha256(before).hexdigest() == digest
    assert receipt["status"] == "RUNTIME_EVIDENCED"
    assert receipt["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert receipt["execution_state"]["LANGUAGE_READINESS"] == 0
    assert receipt["execution_state"]["W04_STARTED"] == 0
    assert all(token not in before for token in (b"surface", b"expected", b"private_path"))
    with pytest.raises(W03ReleaseError, match="不可覆盖"):
        publish_w03_runtime_receipt(tmp_path, **kwargs)
    assert path.read_bytes() == before


def test_public_receipt_rejects_non_pass_aggregate(tmp_path: Path):
    """FAIL aggregate 不能被 release owner 伪装成 runtime evidence。"""
    aggregate, recommendation, host, aggregate_sha, recommendation_sha, host_sha = _fixture(tmp_path)
    value = json.loads(aggregate.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["fail_count"] = 1
    aggregate.write_bytes(canonical_json_bytes(value))
    aggregate_sha = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    with pytest.raises(W03ReleaseError, match="PASS|hard conjunct"):
        publish_w03_runtime_receipt(
            tmp_path,
            aggregate_path=aggregate,
            aggregate_sha256=aggregate_sha,
            recommendation_path=recommendation,
            recommendation_sha256=recommendation_sha,
            candidate_host_freeze_path=host,
            candidate_contract_sha256="6" * 64,
            candidate_host_freeze_sha256=host_sha,
            w02_receipt_sha256="8" * 64,
            d03_receipt_sha256="9" * 64,
            d03_global_manifest_sha256="a" * 64,
            d03_stage_manifest_sha256="b" * 64,
            publication_commit_sha1="c" * 40,
            publication_ci_run_id=123,
            publication_ci_jobs=(
                ("Python 3.11 on ubuntu-latest", "success"),
                ("Python 3.14 on ubuntu-latest", "success"),
                ("Python 3.14 on windows-latest", "success"),
                ("Secret scan", "success"),
            ),
        )
