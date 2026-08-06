"""W04-05 candidate 合同、唯一 guard 和 host freeze 专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_w04_candidate as candidate_owner
import pure_integer_ai.experiments.ph2_w04_runtime as runtime_owner
from pure_integer_ai.experiments.ph2_w04_candidate import (
    W04_CANDIDATE_FORMAL_MODE,
    W04_CANDIDATE_FORMAL_WORKER_COUNT,
    W04_FORMAL_EXECUTION_STATE,
    build_w04_candidate_contract,
    consume_w04_candidate_first_run_guard,
    execute_w04_candidate_once,
    publish_w04_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_FORMAL_RUN_ID,
    W04_W03_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w04_runtime import W04RuntimeConfig
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w04_historical_context import open_historical_w04_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "da69958c1f149a2f264053f7b7407a53f575cd93"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"


@pytest.fixture(autouse=True)
def _historical_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选行为使用冻结上下文，同时保留生产 opener 的独立拒绝测试。"""
    monkeypatch.setattr(
        candidate_owner,
        "open_w04_frozen_context",
        open_historical_w04_context,
    )
    monkeypatch.setattr(
        runtime_owner,
        "open_w04_frozen_context",
        open_historical_w04_context,
    )


def _contract(tmp_path: Path):
    backend = SQLiteBackend(str(tmp_path / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w04_candidate_contract(
        ROOT,
        global_manifest_path=GLOBAL,
        backend_profile_key=profile,
        current_remote_commit_sha1=HEAD,
    )


def test_w04_candidate_contract_is_zero_before_guard(tmp_path: Path):
    """freeze 只声明 zero execution，重复 publication 不覆盖。"""
    contract = _contract(tmp_path)
    root = tmp_path / "candidate"
    path, digest = publish_w04_candidate_contract_freeze(
        ROOT, root, contract)
    assert path.is_file()
    assert len(digest) == 64
    assert contract["execution_state"]["W04_STARTED"] == 0
    assert contract["formal_w04_training_runs"] == 0
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w04_candidate_contract_freeze(ROOT, root, contract)
    guard, _ = consume_w04_candidate_first_run_guard(
        root, candidate_contract_sha256=digest)
    assert guard.is_file()
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w04_candidate_first_run_guard(
            root, candidate_contract_sha256=digest)


def test_w04_candidate_formal_run_sets_started_only_after_guard(tmp_path: Path):
    """正式 candidate 一次运行后 host/readback 与资源身份闭合。"""
    contract = _contract(tmp_path)
    root = tmp_path / "formal_candidate"
    _, contract_sha = publish_w04_candidate_contract_freeze(ROOT, root, contract)
    request = contract["candidate_request"]
    config = W04RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=root / "run",
        sqlite_path=root / "host.sqlite",
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=W04_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W04_CANDIDATE_FORMAL_MODE,
        current_remote_commit_sha1=HEAD,
    )
    outcome, readback, host, _, guard, _ = execute_w04_candidate_once(
        ROOT,
        root,
        config=config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=root / "readback.sqlite",
    )
    assert guard.is_file() and host.is_file()
    assert outcome.execution_state == W04_FORMAL_EXECUTION_STATE
    assert readback.execution_state == W04_FORMAL_EXECUTION_STATE
    assert outcome.active_candidate_count == 1
    assert readback.dump_readback
    assert outcome.resource_report["teacher_calls"] == 0
