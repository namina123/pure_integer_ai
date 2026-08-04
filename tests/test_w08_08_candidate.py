"""W08-08 Candidate freeze、唯一 guard、正式 host 和封存专项。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_candidate import (
    W08_CANDIDATE_HOST_FREEZE_NAME,
    W08_CANDIDATE_TERMINAL_SEAL_NAME,
    execute_w08_candidate_once,
)
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_CLUSTER_AXES,
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
    W08_EXPECTED_COUNTS,
    build_w08_candidate_contract,
    consume_w08_candidate_first_run_guard,
    publish_w08_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_FORMAL_EXECUTION_STATE,
    W08_OPEN_GENERATION_PREFORMAL_STATE,
    W08RuntimeConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def external_tmp_path():
    """在 public Git 同级创建物理隔离且自动清理的测试根。"""
    with tempfile.TemporaryDirectory(
        prefix="w08-candidate-test-", dir=ROOT.parent
    ) as value:
        yield Path(value)


def _contract():
    return build_w08_candidate_contract(
        ROOT,
        current_public_head_commit_sha1=_head(),
    )


def test_candidate_contract_freezes_all_axes_before_guard(external_tmp_path):
    contract = _contract()
    evaluation = contract["evaluation_contract"]
    assert tuple(evaluation["dimension_order"]) == W08_DIMENSION_KEYS
    assert tuple(evaluation["ablation_order"]) == W08_ABLATION_KEYS
    assert tuple(contract["cluster_contract"]["axis_keys"]) == W08_CANDIDATE_CLUSTER_AXES
    assert contract["expected_counts"] == W08_EXPECTED_COUNTS
    assert contract["execution_state"]["W08_STARTED"] == 0
    assert contract["formal_w08_training_runs"] == 0
    assert contract["future_firewall"]["future_payload_reads"] == 0
    assert contract["future_firewall"]["forbidden_inventory"]
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True)
    assert '"label"' not in encoded and '"expected"' not in encoded

    root = external_tmp_path / "candidate"
    path, digest = publish_w08_candidate_contract_freeze(ROOT, root, contract)
    assert path.is_file() and len(digest) == 64
    with pytest.raises(RuntimeError, match="root 必须全新"):
        publish_w08_candidate_contract_freeze(ROOT, root, contract)

    guard, _ = consume_w08_candidate_first_run_guard(
        root,
        candidate_contract_sha256=digest,
    )
    guard_value = json.loads(guard.read_bytes())
    assert guard_value["execution_state_after_start"] == W08_FORMAL_EXECUTION_STATE
    assert guard_value["formal_run_count_before"] == 0
    assert guard_value["formal_run_count_after"] == 1
    assert guard_value["open_generation_state_after_start"] == (
        W08_OPEN_GENERATION_PREFORMAL_STATE
    )
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w08_candidate_first_run_guard(
            root,
            candidate_contract_sha256=digest,
        )


def test_candidate_rejects_public_identity_and_cluster_drift(external_tmp_path):
    contract = _contract()
    contract["code_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="public identity 漂移"):
        publish_w08_candidate_contract_freeze(
            ROOT,
            external_tmp_path / "identity",
            contract,
        )
    contract = _contract()
    contract["cluster_contract"]["axis_keys"][0] = "DRIFT"
    with pytest.raises(RuntimeError, match="cluster 合同漂移"):
        publish_w08_candidate_contract_freeze(
            ROOT,
            external_tmp_path / "cluster",
            contract,
        )


def test_unique_formal_candidate_closes_host_readback_and_seal(external_tmp_path):
    contract = _contract()
    root = external_tmp_path / "formal_candidate"
    _, contract_sha = publish_w08_candidate_contract_freeze(ROOT, root, contract)
    run_root = root / "host"
    config = W08RuntimeConfig(
        ROOT,
        run_root,
        run_root / "coordinator.sqlite",
        worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W08_CANDIDATE_FORMAL_MODE,
    )
    outcome, readback, host, host_sha, guard, _, seal, _ = execute_w08_candidate_once(
        ROOT,
        root,
        config=config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
    )
    assert host.name == W08_CANDIDATE_HOST_FREEZE_NAME and host.is_file()
    assert seal.name == W08_CANDIDATE_TERMINAL_SEAL_NAME and seal.is_file()
    assert guard.is_file()
    assert len(host_sha) == 64
    assert dict(outcome.execution_state) == W08_FORMAL_EXECUTION_STATE
    assert dict(readback.execution_state) == W08_FORMAL_EXECUTION_STATE
    assert outcome.compiled_artifact_count == 5 and len(outcome.uses) == 15
    assert outcome.payload_gets_this_call > 0
    assert readback.payload_gets_this_call == 0
    assert outcome.future_payload_reads == outcome.evaluator_label_reads == 0
    assert outcome.teacher_calls == outcome.memory_learning_writes == 0
    host_value = json.loads(host.read_bytes())
    seal_value = json.loads(seal.read_bytes())
    assert host_value["formal_run_count"] == 1
    assert host_value["candidate_sealed"] == 1
    assert host_value["owner_write_counts"]["evaluator_label_writes"] == 0
    assert seal_value["terminal_state"] == "PASS"
    assert seal_value["candidate_host_freeze_sha256"] == hashlib.sha256(
        host.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="不可重跑"):
        execute_w08_candidate_once(
            ROOT,
            root,
            config=config,
            contract=contract,
            candidate_contract_sha256=contract_sha,
        )


def test_candidate_root_must_be_git_external(tmp_path):
    contract = _contract()
    with pytest.raises(RuntimeError, match="物理隔离"):
        publish_w08_candidate_contract_freeze(
            ROOT,
            ROOT / ".pytest_tmp_safe" / tmp_path.name,
            contract,
        )
