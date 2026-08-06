"""W05-05 candidate 合同、五项消融、唯一 guard 和 host freeze 专项。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_w05_candidate as candidate_owner
import pure_integer_ai.experiments.ph2_w05_runtime as runtime_owner
from pure_integer_ai.experiments.ph2_w05_candidate import (
    W05_CANDIDATE_FORMAL_MODE,
    W05_CANDIDATE_FORMAL_WORKER_COUNT,
    W05_FORMAL_EXECUTION_STATE,
    build_w05_candidate_contract,
    consume_w05_candidate_first_run_guard,
    execute_w05_candidate_once,
    publish_w05_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_ABLATION_KEYS,
    W05_FORMAL_RUN_ID,
    W05_PRIVATE_ABLATION_KEYS,
    W05_W04_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w05_runtime import W05RuntimeConfig
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w05_historical_context import open_historical_w05_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"


@pytest.fixture(autouse=True)
def _historical_candidate_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """candidate/runtime 行为消费冻结 gate，不改变生产 authority。"""
    for owner in (candidate_owner, runtime_owner):
        monkeypatch.setattr(
            owner,
            "open_w05_frozen_context",
            open_historical_w05_context,
        )


def _contract(tmp_path: Path):
    """用当前 SQLite capability 构造 zero-state candidate contract。"""
    backend = SQLiteBackend(str(tmp_path / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w05_candidate_contract(
        ROOT,
        global_manifest_path=GLOBAL,
        backend_profile_key=profile,
        current_remote_commit_sha1=HEAD,
    )


def test_w05_candidate_contract_freezes_five_formal_ablations_before_guard(
        tmp_path: Path,
        ):
    """freeze 保持 zero state，四项 D-03 加 W05-G 形成五项 formal 顺序。"""
    contract = _contract(tmp_path)
    evaluation = contract["evaluation_contract"]
    assert tuple(evaluation["d03_ablation_order"]) == W05_ABLATION_KEYS
    assert tuple(evaluation["formal_ablation_order"]) == W05_PRIVATE_ABLATION_KEYS
    assert len(W05_PRIVATE_ABLATION_KEYS) == 5
    assert contract["execution_state"]["W05_STARTED"] == 0
    assert contract["execution_state"]["W06_STARTED"] == 0
    assert contract["formal_w05_training_runs"] == 0
    root = tmp_path / "candidate"
    path, digest = publish_w05_candidate_contract_freeze(ROOT, root, contract)
    assert path.is_file() and len(digest) == 64
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w05_candidate_contract_freeze(ROOT, root, contract)
    guard, _ = consume_w05_candidate_first_run_guard(
        root, candidate_contract_sha256=digest)
    assert guard.is_file()
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w05_candidate_first_run_guard(
            root, candidate_contract_sha256=digest)


def test_w05_candidate_formal_run_sets_started_only_after_guard(tmp_path: Path):
    """唯一 candidate run 后 host/readback、scope、资源和状态闭合。"""
    contract = _contract(tmp_path)
    root = tmp_path / "formal_candidate"
    _, contract_sha = publish_w05_candidate_contract_freeze(ROOT, root, contract)
    request = contract["candidate_request"]
    config = W05RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=root / "run",
        sqlite_path=root / "coordinator.sqlite",
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=W05_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W05_CANDIDATE_FORMAL_MODE,
        current_remote_commit_sha1=HEAD,
    )
    outcome, readback, host, _, guard, _ = execute_w05_candidate_once(
        ROOT,
        root,
        config=config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=root / "readback.sqlite",
    )
    assert guard.is_file() and host.is_file()
    assert outcome.execution_state == W05_FORMAL_EXECUTION_STATE
    assert readback.execution_state == W05_FORMAL_EXECUTION_STATE
    assert outcome.active_candidate_count == 2
    assert outcome.transaction_event_count == 5
    assert outcome.payload_gets_this_call > 0
    assert readback.payload_gets_this_call == 0
    assert readback.new_learning_write_count == 0
    assert outcome.resource_report["teacher_calls"] == 0
    host_value = json.loads(host.read_text(encoding="utf-8"))
    assert host_value["formal_run_count"] == 1
    assert host_value["execution_state"] == W05_FORMAL_EXECUTION_STATE
    assert host_value["owner_write_counts"]["evaluator_label_writes"] == 0
    assert host_value["owner_write_counts"]["readback_learning_writes"] == 0
