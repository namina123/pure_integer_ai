"""W07-05 candidate freeze、八维合同与唯一 guard 专项。"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_w07_candidate import (
    W07_CANDIDATE_FORMAL_MODE,
    W07_CANDIDATE_FORMAL_WORKER_COUNT,
    W07_CASE_FAMILIES,
    W07_EXPECTED_COUNTS,
    W07_FORMAL_EXECUTION_STATE,
    build_w07_candidate_contract,
    consume_w07_candidate_first_run_guard,
    publish_w07_candidate_contract_freeze,
    verify_w07_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
    W07_SUBSTAGE_ORDER,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
TEST_HEAD = "a" * 40


@pytest.fixture
def external_tmp_path():
    with tempfile.TemporaryDirectory(
            prefix="w07-candidate-test-", dir=ROOT.parent) as value:
        yield Path(value)


def _contract(root: Path):
    backend = SQLiteBackend(str(root / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w07_candidate_contract(
        ROOT,
        backend_profile_key=profile,
        public_head_commit_sha1=TEST_HEAD,
    )


def test_w07_candidate_contract_freezes_eight_dimensions_before_guard(
        external_tmp_path):
    contract = _contract(external_tmp_path)
    evaluation = contract["evaluation_contract"]
    assert tuple(evaluation["dimension_order"]) == W07_PUBLIC_DIMENSION_KEYS
    assert tuple(evaluation["ablation_order"]) == W07_PUBLIC_ABLATION_KEYS
    assert tuple(evaluation["substage_order"]) == W07_SUBSTAGE_ORDER
    assert tuple(tuple(item) for item in evaluation["case_families"]) == (
        W07_CASE_FAMILIES)
    assert len(W07_CASE_FAMILIES) == 8
    assert contract["expected_counts"] == W07_EXPECTED_COUNTS
    assert contract["execution_state"]["W07_STARTED"] == 0
    assert contract["formal_w07_training_runs"] == 0
    assert contract["guard_consumed"] == 0
    request = contract["candidate_request"]
    assert request["worker_count"] == W07_CANDIDATE_FORMAL_WORKER_COUNT
    assert request["mode"] == W07_CANDIDATE_FORMAL_MODE
    assert request["logical_shard_count"] == 16


def test_w07_candidate_freeze_and_guard_are_non_overwritable(
        external_tmp_path):
    contract = _contract(external_tmp_path)
    root = external_tmp_path / "candidate"
    path, digest = publish_w07_candidate_contract_freeze(ROOT, root, contract)
    assert path.is_file() and len(digest) == 64
    assert verify_w07_candidate_contract_freeze(
        ROOT, root, candidate_contract_sha256=digest) == contract
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w07_candidate_contract_freeze(ROOT, root, contract)
    guard, guard_sha = consume_w07_candidate_first_run_guard(
        root,
        candidate_contract_sha256=digest,
        public_head_commit_sha1=TEST_HEAD,
    )
    assert guard.is_file() and len(guard_sha) == 64
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w07_candidate_first_run_guard(
            root,
            candidate_contract_sha256=digest,
            public_head_commit_sha1=TEST_HEAD,
        )
    assert W07_FORMAL_EXECUTION_STATE["W07_STARTED"] == 1
    assert W07_FORMAL_EXECUTION_STATE["formal_w07_training_runs"] == 1
    assert W07_FORMAL_EXECUTION_STATE["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert W07_FORMAL_EXECUTION_STATE["LANGUAGE_READINESS"] == 0


def test_w07_candidate_rejects_inventory_and_external_root_drift(
        external_tmp_path):
    contract = _contract(external_tmp_path)
    contract["code_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity 漂移"):
        publish_w07_candidate_contract_freeze(
            ROOT, external_tmp_path / "drift", contract)
    contract = _contract(external_tmp_path)
    with pytest.raises(RuntimeError, match="公开 Git 外"):
        publish_w07_candidate_contract_freeze(
            ROOT, ROOT / "candidate-forbidden", contract)
