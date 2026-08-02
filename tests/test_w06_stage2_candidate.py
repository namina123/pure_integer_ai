"""W06-05 candidate 合同、八项消融、唯一 guard 和 host freeze 专项。"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_w06_candidate import (
    W06_CANDIDATE_FORMAL_MODE,
    W06_CANDIDATE_FORMAL_WORKER_COUNT,
    W06_CASE_FAMILIES,
    W06_EXPECTED_COUNTS,
    W06_FORMAL_EXECUTION_STATE,
    build_w06_candidate_contract,
    consume_w06_candidate_first_run_guard,
    execute_w06_candidate_once,
    publish_w06_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_ABLATION_KEYS,
    W06_DIMENSION_KEYS,
    W06_FORMAL_RUN_ID,
    W06_PRIVATE_ABLATION_KEYS,
    W06_W05_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w06_runtime import W06RuntimeConfig
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "2ceb8f955c81204bdc194b962911053de0133bbb"


@pytest.fixture
def external_tmp_path():
    """在公开 Git 同级目录建立并自动清理物理隔离的测试 root。"""
    with tempfile.TemporaryDirectory(
            prefix="w06-candidate-test-", dir=ROOT.parent) as value:
        yield Path(value)


def _contract(tmp_path: Path):
    """用当前 SQLite capability 构造 zero-state candidate contract。"""
    backend = SQLiteBackend(str(tmp_path / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w06_candidate_contract(
        ROOT,
        backend_profile_key=profile,
        current_remote_commit_sha1=HEAD,
    )


def test_w06_candidate_contract_freezes_relations_ablations_before_guard(
        external_tmp_path: Path):
    """freeze 保持零状态，并冻结七 bearing、八 ablation 和七 case family。"""
    contract = _contract(external_tmp_path)
    evaluation = contract["evaluation_contract"]
    assert tuple(evaluation["dimension_order"]) == W06_DIMENSION_KEYS
    assert tuple(evaluation["ablation_order"]) == W06_ABLATION_KEYS
    assert tuple(evaluation["formal_ablation_order"]) == (
        W06_PRIVATE_ABLATION_KEYS)
    assert len(W06_PRIVATE_ABLATION_KEYS) == 8
    relation = contract["relation_contract"]
    assert tuple(tuple(item) for item in relation["case_families"]) == (
        W06_CASE_FAMILIES)
    assert len(relation["profiles"]) == 14
    assert contract["expected_counts"] == W06_EXPECTED_COUNTS
    assert contract["execution_state"]["W06_STARTED"] == 0
    assert contract["formal_w06_training_runs"] == 0
    root = external_tmp_path / "candidate"
    path, digest = publish_w06_candidate_contract_freeze(ROOT, root, contract)
    assert path.is_file() and len(digest) == 64
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w06_candidate_contract_freeze(ROOT, root, contract)
    guard, _ = consume_w06_candidate_first_run_guard(
        root, candidate_contract_sha256=digest)
    assert guard.is_file()
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w06_candidate_first_run_guard(
            root, candidate_contract_sha256=digest)


def test_w06_candidate_contract_rejects_inventory_and_relation_drift(
        external_tmp_path: Path):
    """code/test/artifact 与 relation profile 任一漂移都不得冻结。"""
    contract = _contract(external_tmp_path)
    root = external_tmp_path / "candidate"
    contract["code_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity 漂移"):
        publish_w06_candidate_contract_freeze(ROOT, root, contract)
    contract = _contract(external_tmp_path)
    contract["relation_contract"]["profiles"][0]["closure_policy"] = "DRIFT"
    with pytest.raises(RuntimeError, match="relation/case/profile"):
        publish_w06_candidate_contract_freeze(ROOT, root, contract)


def test_w06_candidate_formal_run_sets_started_only_after_guard(
        external_tmp_path: Path):
    """唯一 candidate run 后 host/readback、关系、载体、资源和状态闭合。"""
    contract = _contract(external_tmp_path)
    root = external_tmp_path / "formal_candidate"
    _, contract_sha = publish_w06_candidate_contract_freeze(ROOT, root, contract)
    request = contract["candidate_request"]
    config = W06RuntimeConfig(
        repository_root=ROOT,
        run_root=root / "run",
        sqlite_path=root / "coordinator.sqlite",
        run_id=W06_FORMAL_RUN_ID,
        parent_run_id=W06_W05_BASE_RUN_ID,
        base_run_id=W06_W05_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=W06_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W06_CANDIDATE_FORMAL_MODE,
        current_remote_commit_sha1=HEAD,
    )
    outcome, readback, host, _, guard, _ = execute_w06_candidate_once(
        ROOT,
        root,
        config=config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=root / "readback.sqlite",
    )
    assert guard.is_file() and host.is_file()
    assert outcome.execution_state == W06_FORMAL_EXECUTION_STATE
    assert readback.execution_state == W06_FORMAL_EXECUTION_STATE
    assert outcome.candidate_count == 50
    assert outcome.active_candidate_count == 17
    assert dict(outcome.artifact_counts)["EVIDENCE_ACCOUNT"] == 64
    assert outcome.transaction_event_count == 5
    assert outcome.payload_gets_this_call > 0
    assert readback.payload_gets_this_call == 0
    assert readback.new_learning_write_count == 0
    assert outcome.resource_report["teacher_calls"] == 0
    host_value = json.loads(host.read_text(encoding="utf-8"))
    assert host_value["formal_run_count"] == 1
    assert host_value["execution_state"] == W06_FORMAL_EXECUTION_STATE
    assert host_value["owner_write_counts"]["evaluator_label_writes"] == 0
    assert host_value["owner_write_counts"]["readback_learning_writes"] == 0
