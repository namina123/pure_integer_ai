from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.batch_contract_artifact_audit import (
    BatchArtifactEvidenceError,
    inspect_completed_batch_contract,
)
from pure_integer_ai.experiments.batch_run_recovery import ProcessProbe


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _fixture(root: Path) -> None:
    identity = root.name
    course_sha = "a" * 64
    _write(root / "run_identity.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_PREFLIGHT_V1",
        "identity": identity, "source_identity": [1],
        "license_identity": [2], "course_sha256": course_sha,
        "free_dialogue_complete": 0,
    })
    _write(root / "source_license.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_SOURCE_LICENSE_V1",
        "source_identity": [1], "license_identity": [2],
        "license": "CC0-1.0", "course_sha256": course_sha,
        "source_body_rows_added": 0, "whole_sentence_or_answer_table": 0,
        "free_dialogue_complete": 0,
    })
    _write(root / "split_manifest.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_SPLIT_V1",
        "identity": identity,
        "train": {"status": "BOUNDED_ACTIVE_FACTS"},
        "held_out": {"status": "UNKNOWN_REUSED_ACTIVE_TRAINING_PROPOSITIONS"},
        "negative": {"status": "BOUNDED_NOT_RUN"},
        "conflict": {"status": "BOUNDED_NOT_RUN"},
        "free_dialogue_complete": 0,
    })
    row = {
        "case": 1, "proposition_key": [11], "operation_key": [22],
        "query_values": [33, 65306, 44], "operation_keys_inferred": [[22]],
        "operation_keys_consumed": [[22]], "active_spaces": [[1]],
        "core_hop_count": 1, "memory_route_count": 1,
        "termination": 1, "response_plan_present": 1,
        "generation_present": 1, "claim_count": 1,
        "claim_evidence_count": 1, "factual_closed": 1,
        "delivery_called": 1, "surface_values": [55],
        "memory_database": "memory.sqlite3", "memory_scope": "shared_session",
    }
    _write(root / "case_checkpoint.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_CASE_CHECKPOINT_V1",
        "identity": identity, "completed_case_count": 1,
        "next_case": 2, "cases": [row], "free_dialogue_complete": 0,
    })
    config = {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_CASE_WORKER_V1",
        "identity": identity, "operation_key": [22],
        "operation_cue": [33], "proposition_key": [11],
        "case": 1, "free_dialogue_complete": 0,
    }
    _write(root / "case_worker_001.json", config)
    _write(root / "case_result_001.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_CASE_RESULT_V1",
        "identity": identity, "case": 1,
        "result": {k: v for k, v in row.items()
                   if k not in {"memory_database", "memory_scope"}},
        "free_dialogue_complete": 0,
    })
    _write(root / "run_state.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_RUN_STATE_V1",
        "identity": identity, "status": "COMPLETE", "pid": 101,
        "worker_pid": 102, "free_dialogue_complete": 0,
    })
    receipt = {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_PREFLIGHT_V1",
        "identity": identity, "status": "PASS_BATCH_CONNECTOR_QUERY_DELIVERY",
        "course_sha256": course_sha, "positive_case_count": 1,
        "positive_closed_count": 1, "covered_fact_count": 1,
        "uncovered_fact_count": 0, "active_fact_count": 1,
        "uncovered_facts": [], "cases": [row],
        "production_not_started": 1, "core_training_performed": 1,
        "database_copy_count": 0, "source_body_rows_added": 0,
        "whole_sentence_or_answer_table": 0, "free_dialogue_complete": 0,
    }
    _write(root / "preflight.json", receipt)
    receipt_sha = hashlib.sha256((root / "preflight.json").read_bytes()).hexdigest()
    _write(root / "checkpoint.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_PREFLIGHT_V1",
        "identity": identity, "stage": receipt["status"],
        "preflight_sha256": receipt_sha, "free_dialogue_complete": 0,
    })
    _write(root / "trace.int.json", {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_TRACE_V1",
        "cases": [row], "free_dialogue_complete": 0,
    })


def _exited(pid: int) -> ProcessProbe:
    return ProcessProbe(pid, "EXITED", reason="fixture_exit")


def test_completed_contract_is_consistent_but_not_language_qualified(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = inspect_completed_batch_contract(
        tmp_path, owner_pids=(101, 102), expected_covered_count=1,
        expected_uncovered_count=0, probe=_exited)
    assert result["status"] == "CONSISTENT_AGGREGATES_ONLY"
    assert result["wave1_qualification"] == "UNKNOWN_not_a_language_qualification_run"
    assert result["full_chain_provenance"] == "UNKNOWN_not_saved_by_v1_runner"
    assert result["completed_case_count"] == 1


def test_live_owner_blocks_artifact_audit(tmp_path: Path) -> None:
    with pytest.raises(BatchArtifactEvidenceError, match="not proven inactive"):
        inspect_completed_batch_contract(
            tmp_path, owner_pids=(101,), expected_covered_count=1,
            expected_uncovered_count=0,
            probe=lambda pid: ProcessProbe(pid, "LIVE"))


def test_missing_terminal_artifacts_are_unknown(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "preflight.json").unlink()
    result = inspect_completed_batch_contract(
        tmp_path, owner_pids=(101, 102), expected_covered_count=1,
        expected_uncovered_count=0, probe=_exited)
    assert result["status"] == "UNKNOWN_INCOMPLETE_ARTIFACTS"
    assert "preflight.json" in result["missing_artifacts"]


def test_receipt_claim_change_is_a_contradiction_not_a_new_closed_case(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "preflight.json"
    receipt = json.loads(path.read_text("ascii"))
    receipt["cases"][0]["proposition_key"] = [999]
    _write(path, receipt)
    result = inspect_completed_batch_contract(
        tmp_path, owner_pids=(101, 102), expected_covered_count=1,
        expected_uncovered_count=0, probe=_exited)
    assert result["status"] == "FAIL_ARTIFACT_CONTRADICTION"
    assert "preflight.json:coverage_checkpoint_drift" in result["issues"]
    assert "checkpoint.json:preflight_sha256_drift" in result["issues"]


def test_current_failure_is_distinguished_from_old_failure(tmp_path: Path) -> None:
    _fixture(tmp_path)
    for name in ("preflight.json", "checkpoint.json", "trace.int.json"):
        (tmp_path / name).unlink()
    state_path = tmp_path / "run_state.json"
    state = json.loads(state_path.read_text("ascii"))
    state.update(status="CASE_WORKER_FAILED", updated_epoch_ns=100)
    _write(state_path, state)
    failure_path = tmp_path / "failure.json"
    failure = {
        "format": "PURE_INTEGER_RELATION_GENERATION_BATCH_FAILURE_V1",
        "identity": tmp_path.name, "stage": "queries", "error_type": "RuntimeError",
        "updated_epoch_ns": 99, "free_dialogue_complete": 0,
    }
    _write(failure_path, failure)
    old = inspect_completed_batch_contract(
        tmp_path, owner_pids=(101, 102), expected_covered_count=1,
        expected_uncovered_count=0, probe=_exited)
    assert old["status"] == "UNKNOWN_INCOMPLETE_ARTIFACTS"
    failure["updated_epoch_ns"] = 101
    _write(failure_path, failure)
    current = inspect_completed_batch_contract(
        tmp_path, owner_pids=(101, 102), expected_covered_count=1,
        expected_uncovered_count=0, probe=_exited)
    assert current["status"] == "FAIL_CURRENT_CASE_WORKER"
    assert current["failure_attempt_qualification"] == (
        "CURRENT_FAILURE_CORRELATED_WITH_TERMINAL_STATE")
