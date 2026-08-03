"""PH2 W-07 公开 runtime receipt 的安全和不可覆盖发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_candidate import (
    W07_CANDIDATE_CONTRACT_KIND,
    W07_CANDIDATE_FIRST_RUN_GUARD_KIND,
    W07_CANDIDATE_HOST_FREEZE_KIND,
    W07_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_GENERATION_CHOICE_PATH,
    W07_GENERATION_OUTCOME_PATH,
    W07_GLOBAL_MANIFEST_PATH,
    W07_INVALIDATION_GRAPH_PATH,
    W07_LC13_DIRECTIONAL_PATH,
    W07_LC16_DIRECTIONAL_PATH,
    W07_LC16_OVERLAY_PATH,
    W07_OPEN_GENERATION_STATE,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
    W07_STAGE_MANIFEST_PATH,
    W07_SUBSTAGE_ORDER,
    W07_W06_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07PrivateDimensionResult,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_contract import (
    W07V2AblationProgress,
    W07V2DiagnosticCursor,
    public_safe_w07_v2_aggregate,
)
from pure_integer_ai.experiments.ph2_w07_release import (
    W07_EXPECTED_RETENTION_IDENTITIES,
    W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
    W07_REQUIRED_VERIFICATION_JOBS,
    W07ReleaseError,
    publish_w07_runtime_receipt,
    read_w07_runtime_receipt,
)


CANDIDATE_HEAD = "b" * 40
EVALUATOR_HEAD = "e" * 40


def _write(path: Path, value: dict) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _result(dimension: str, status: str) -> W07PrivateDimensionResult:
    passed = int(status == "PASS")
    return W07PrivateDimensionResult(
        dimension, status, passed, 1, int(not passed), 0, "a" * 64)


def _fixture(tmp_path: Path, monkeypatch):
    baseline = tuple(_result(item, "PASS") for item in W07_PUBLIC_DIMENSION_KEYS)
    ablations = tuple(
        W07V2AblationProgress(
            key,
            tuple(_result(
                dimension,
                "FAIL" if index == ordinal else "PASS",
            ) for index, dimension in enumerate(W07_PUBLIC_DIMENSION_KEYS)),
        )
        for ordinal, key in enumerate(W07_PUBLIC_ABLATION_KEYS)
    )
    aggregate = public_safe_w07_v2_aggregate(
        baseline,
        ablations,
        family_commitment="1" * 64,
        payload_commitment="2" * 64,
        case_commitment="3" * 64,
        label_commitment="4" * 64,
        cluster_commitment="5" * 64,
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
        public_repo_writes=0,
        failure_kind="NONE",
        cursor=W07V2DiagnosticCursor(
            "REPORT_SAFETY", "PUBLISH_REPORT"),
        ablation_gates_passed=True,
    )
    baseline_audit = {
        "generation_choices": 15,
        "generation_outcomes": 15,
        "generation_uses": 15,
        "nested_generation_layer_uses": 2,
        "nested_reasoning_layer_uses": 2,
        "nested_understanding_layer_uses": 2,
        "reasoning_outcomes": 8,
        "reasoning_uses": 8,
        "understanding_outcomes": 8,
        "understanding_uses": 8,
    }
    ledger_audit = [
        {
            "nonempty_table_count": 7,
            "row_count": 100 + ordinal,
            "substage": substage,
            "table_count": 53,
        }
        for ordinal, substage in enumerate(W07_SUBSTAGE_ORDER)
    ]
    aggregate["infrastructure"] = {
        "baseline_consumer_audit": baseline_audit,
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_public_head_commit_sha1": EVALUATOR_HEAD,
        "host_copy_unchanged": 1,
        "ledger_audit": ledger_audit,
        "ledger_audit_commitment": hashlib.sha256(
            canonical_json_bytes(ledger_audit)).hexdigest(),
        "ledger_count": 7,
        "ledgers_closed": 1,
        "ledgers_committed": 1,
        "logic_scope_cell_count": 189,
        "logic_use_count": 21,
        "owner_audit_complete": 1,
    }
    aggregate["generation_ablation_statuses"] = [
        * ("PASS" for _ in range(len(W07_PUBLIC_DIMENSION_KEYS) - 1)),
        "FAIL",
    ]
    aggregate_path, aggregate_sha = _write(
        tmp_path / "formal" / "aggregate.json", aggregate)

    repository = tmp_path / "repository"
    contract_path, contract_sha = _write(
        tmp_path / "candidate" / "candidate_contract_freeze.json",
        {
            "artifact_kind": W07_CANDIDATE_CONTRACT_KIND,
            "public_head_commit_sha1": CANDIDATE_HEAD,
        },
    )
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_w07_release."
        "verify_w07_candidate_contract_freeze",
        lambda *_args, **_kwargs: json.loads(
            contract_path.read_text(encoding="utf-8")),
    )
    guard_path, candidate_guard_sha = _write(
        contract_path.parent / "formal_first_run_guard.json",
        {
            "artifact_kind": W07_CANDIDATE_FIRST_RUN_GUARD_KIND,
            "candidate_contract_sha256": contract_sha,
            "formal_run_count_after": 1,
            "formal_run_count_before": 0,
            "public_head_commit_sha1": CANDIDATE_HEAD,
        },
    )
    artifact_counts = [
        ["ACTIVE_OPERATOR", 36],
        ["CANDIDATE", 71],
        ["CARRIER_PROJECTION", 9],
        ["EVIDENCE_ACCOUNT", 94],
        ["EVIDENCE_APPLICATION", 63],
        ["LOGICAL_SHARD", 16],
        ["LOGIC_SCOPE_CELL", 189],
        ["LOGIC_USE", 21],
        ["OPERATOR_PROFILE", 7],
        ["SCHEMA_REJECTION", 3],
        ["SUBSTAGE", 7],
    ]
    host_digests = {
        "active_projection": "a" * 64,
        "candidate": "b" * 64,
        "carrier_scope": "c" * 64,
        "logic": "d" * 64,
        "logical": "e" * 64,
        "source_evidence": "f" * 64,
        "transaction": "0" * 64,
    }
    dump_path = contract_path.parent / "run" / "w07_dump_manifest.json"
    dump_path.parent.mkdir(parents=True)
    dump_path.write_bytes(b"public-dump")
    dump_sha = hashlib.sha256(dump_path.read_bytes()).hexdigest()
    resource_report = {
        "actual_checkpoint_count": 1,
        "actual_logic_operations": 36_240,
        "actual_payload_bytes": 36_741,
        "actual_payload_gets": 21,
        "actual_recompute_objects": 363,
        "actual_records": 222,
        "actual_segments": 80,
        "actual_workers": 4,
        "teacher_calls": 0,
    }
    host_evidence = {
        "artifact_counts": artifact_counts,
        "dump_manifest_sha256": dump_sha,
        "dump_readback": 0,
        "execution_state": dict(W07_FORMAL_EXECUTION_STATE),
        "host_digests": host_digests,
        "learning_attempt_count": 1,
        "new_learning_write_count": 174,
        "owned_tables": ["graph_object", "ph2_w07_transaction_event"],
        "payload_bytes_this_call": 36_741,
        "payload_gets_this_call": 21,
        "resource_report": resource_report,
        "retention_sha256": [
            list(item) for item in W07_EXPECTED_RETENTION_IDENTITIES],
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
    host_path, host_sha = _write(
        contract_path.parent / "candidate_host_freeze.json",
        {
            "artifact_kind": W07_CANDIDATE_HOST_FREEZE_KIND,
            "candidate_contract_sha256": contract_sha,
            "candidate_first_run_guard_sha256": candidate_guard_sha,
            "dump_readback_evidence": readback,
            "execution_state": dict(W07_FORMAL_EXECUTION_STATE),
            "formal_run_count": 1,
            "format_version": 1,
            "host_evidence": host_evidence,
            "open_generation_state": W07_OPEN_GENERATION_STATE,
            "owner_write_counts": {
                "artifact_writes": 174,
                "evaluator_label_writes": 0,
                "formal_training_runs": 1,
                "readback_learning_writes": 0,
                "teacher_calls": 0,
            },
            "public_head_commit_sha1": CANDIDATE_HEAD,
            "self_excluded": 1,
        },
    )
    family_freeze_path, family_freeze_sha = _write(
        tmp_path / "formal" / "private_family_freeze.json",
        {
            "artifact_kind": "PH2_W07_PRIVATE_FAMILY_V2_FREEZE",
            "candidate_contract_sha256": contract_sha,
            "candidate_host_freeze_sha256": host_sha,
            "case_commitment": "3" * 64,
            "cluster_commitment": "5" * 64,
            "evaluator_public_head_commit_sha1": EVALUATOR_HEAD,
            "evaluator_version": 2,
            "family_key": "1" * 64,
            "formal_run_count": 0,
            "format_version": 2,
            "label_commitment": "4" * 64,
            "payload_commitment": "2" * 64,
            "self_excluded": 1,
        },
    )
    private_guard_path, private_guard_sha = _write(
        tmp_path / "formal" / "formal_first_run_guard.json",
        {
            "artifact_kind": "PH2_W07_PRIVATE_V2_FIRST_RUN_GUARD",
            "evaluator_version": 2,
            "family_freeze_sha256": family_freeze_sha,
            "formal_run_count_after": 1,
            "formal_run_count_before": 0,
            "format_version": 2,
        },
    )
    recommendation_path, recommendation_sha = _write(
        tmp_path / "formal" / "recommendation.json",
        {
            "aggregate_sha256": aggregate_sha,
            "artifact_kind": "PH2_W07_RUNTIME_RECEIPT_V2_RECOMMENDATION",
            "candidate_contract_sha256": contract_sha,
            "candidate_host_freeze_sha256": host_sha,
            "evaluator_public_head_commit_sha1": EVALUATOR_HEAD,
            "evaluator_version": 2,
            "family_commitment": "1" * 64,
            "formal_run_count": 1,
            "format_version": 2,
            "recommend_runtime_receipt": 1,
        },
    )

    parent_paths = {
        "d03_global_manifest_sha256": W07_GLOBAL_MANIFEST_PATH,
        "d03_stage_manifest_sha256": W07_STAGE_MANIFEST_PATH,
        "generation_choice_sha256": W07_GENERATION_CHOICE_PATH,
        "generation_outcome_sha256": W07_GENERATION_OUTCOME_PATH,
        "invalidation_graph_sha256": W07_INVALIDATION_GRAPH_PATH,
        "lc13_directional_sha256": W07_LC13_DIRECTIONAL_PATH,
        "lc16_directional_sha256": W07_LC16_DIRECTIONAL_PATH,
        "lc16_overlay_sha256": W07_LC16_OVERLAY_PATH,
        "w06_receipt_sha256": W07_W06_RECEIPT_PATH,
    }
    observed_parents = {}
    for key, relative in parent_paths.items():
        path = repository / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
        observed_parents[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_w07_release."
        "W07_EXPECTED_PARENT_IDENTITIES",
        observed_parents,
    )
    return {
        "repository_root": repository,
        "aggregate_path": aggregate_path,
        "aggregate_sha256": aggregate_sha,
        "recommendation_path": recommendation_path,
        "recommendation_sha256": recommendation_sha,
        "candidate_contract_freeze_path": contract_path,
        "candidate_contract_sha256": contract_sha,
        "candidate_first_run_guard_path": guard_path,
        "candidate_first_run_guard_sha256": candidate_guard_sha,
        "candidate_host_freeze_path": host_path,
        "candidate_host_freeze_sha256": host_sha,
        "candidate_dump_manifest_path": dump_path,
        "private_family_freeze_path": family_freeze_path,
        "private_family_freeze_sha256": family_freeze_sha,
        "private_first_run_guard_path": private_guard_path,
        "private_first_run_guard_sha256": private_guard_sha,
        "candidate_public_head_commit_sha1": CANDIDATE_HEAD,
        "evaluator_public_head_commit_sha1": EVALUATOR_HEAD,
        "verification_run_id": 20260804,
        "verification_jobs": tuple(
            (name, "PASS") for name in W07_REQUIRED_VERIFICATION_JOBS),
        "parent_paths": parent_paths,
    }


def _publish(kwargs):
    values = dict(kwargs)
    values.pop("parent_paths")
    repository = values.pop("repository_root")
    return publish_w07_runtime_receipt(repository, **values)


def test_w07_public_receipt_is_safe_and_non_overwritable(
        tmp_path: Path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    path, digest = _publish(kwargs)
    before = path.read_bytes()
    receipt = json.loads(before.decode("utf-8"))
    assert path.relative_to(kwargs["repository_root"]).as_posix() == (
        W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH)
    assert hashlib.sha256(before).hexdigest() == digest
    assert receipt["execution_state"]["W07_RUNTIME_EVIDENCED"] == 1
    assert receipt["execution_state"]["W08_STARTED"] == 0
    assert receipt["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert len(receipt["dimension_results"]) == 8
    assert len(receipt["ablation_results"]) == 8
    assert read_w07_runtime_receipt(kwargs["repository_root"]) == receipt
    assert all(token not in before for token in (
        b"case_key", b"label_key", b"surface", b"expected",
        b"private_path", b"message", b":\\\\"))
    with pytest.raises(W07ReleaseError, match="不可覆盖"):
        _publish(kwargs)
    assert path.read_bytes() == before


def test_w07_public_receipt_rejects_non_pass_aggregate(
        tmp_path: Path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    path = Path(kwargs["aggregate_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["fail_count"] = 1
    path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(W07ReleaseError, match="PASS hard conjunct"):
        _publish(kwargs)


def test_w07_public_receipt_rejects_nonorthogonal_ablation(
        tmp_path: Path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    path = Path(kwargs["aggregate_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    result = value["ablation_results"][0]["dimension_results"][1]
    result.update({"status": "FAIL", "passed": 0, "fail_count": 1})
    path.write_bytes(canonical_json_bytes(value))
    kwargs["aggregate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(W07ReleaseError, match="hard conjunct"):
        _publish(kwargs)


def test_w07_public_receipt_rejects_candidate_isolation_drift(
        tmp_path: Path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    host_path = Path(kwargs["candidate_host_freeze_path"])
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["dump_readback_evidence"]["new_learning_write_count"] = 1
    host_path.write_bytes(canonical_json_bytes(host))
    new_host_sha = hashlib.sha256(host_path.read_bytes()).hexdigest()
    kwargs["candidate_host_freeze_sha256"] = new_host_sha
    recommendation_path = Path(kwargs["recommendation_path"])
    recommendation = json.loads(
        recommendation_path.read_text(encoding="utf-8"))
    recommendation["candidate_host_freeze_sha256"] = new_host_sha
    recommendation_path.write_bytes(canonical_json_bytes(recommendation))
    kwargs["recommendation_sha256"] = hashlib.sha256(
        recommendation_path.read_bytes()).hexdigest()
    with pytest.raises(W07ReleaseError, match="host freeze 状态"):
        _publish(kwargs)


def test_w07_public_receipt_rejects_parent_identity_drift(
        tmp_path: Path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    relative = kwargs["parent_paths"]["lc16_overlay_sha256"]
    path = kwargs["repository_root"] / Path(*relative.split("/"))
    path.write_bytes(b"drift")
    with pytest.raises(W07ReleaseError, match="parent identity"):
        _publish(kwargs)
