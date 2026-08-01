"""W05-04 transaction、恢复、九载体 scope、资源与 dump/readback 专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_LC16_CARRIER_KEYS,
    W05_LC16_DIRECTIONS,
    W05_LC16_SCOPE_KEY,
    W05_OPEN_GENERATION_STATE,
    W05_W04_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w05_faults import (
    W05_FAILURE_POINT_KEYS,
    W05InjectedFault,
)
from pure_integer_ai.experiments.ph2_w05_runtime import (
    W05RuntimeConfig,
    load_w05_candidate_dump,
    run_language_stage5,
)
from pure_integer_ai.experiments.ph2_w05_transaction import (
    W05TransactionError,
    W05TransactionStore,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
W04_RECEIPT = ROOT / "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json"
W04_RECEIPT_SHA256 = (
    "153db3d7f3c0fca04642f4198df16e3c1adb0f5c78e4d6c7c59d35122989727b"
)


def _config(root, *, worker=1, mode="fresh", fault=None):
    """在同一物理 W-05 root/coordinator 上构造调度变体。"""
    return W05RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=root / "run",
        sqlite_path=root / "coordinator.sqlite",
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        base_fence_key=None,
        worker_count=worker,
        mode=mode,
        current_remote_commit_sha1=HEAD,
        fault_point=fault,
    )


def _logic_key(outcome):
    """返回排除物理 worker/attempt 的完整 W-05 逻辑身份。"""
    return (
        outcome.logical_state_digest,
        outcome.candidate_digest,
        outcome.understanding_digest,
        outcome.reasoning_digest,
        outcome.generation_digest,
        outcome.carrier_scope_digest,
        outcome.active_candidate_count,
        outcome.artifact_counts,
    )


def _dump(config):
    """回读测试 root 的 canonical dump object。"""
    path = (
        Path(config.run_root)
        / f"w05_run_{config.run_id:020d}"
        / "w05_dump_manifest.json"
    )
    value = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(value, dict)
    return value


def test_w05_runtime_worker_and_mode_are_bit_identical(tmp_path):
    """1/2/4 worker 与 fresh/restart/resume 的逻辑产物必须 bit-identical。"""
    outcomes = [
        run_language_stage5(_config(
            tmp_path / f"host-{worker}-{mode}", worker=worker, mode=mode))
        for worker in (1, 2, 4)
        for mode in ("fresh", "restart", "resume")
    ]
    assert len({_logic_key(item) for item in outcomes}) == 1
    for outcome in outcomes:
        assert outcome.active_candidate_count == 2
        assert outcome.transaction_event_count == 5
        assert outcome.payload_gets_this_call > 0
        assert outcome.payload_bytes_this_call > 0
        assert outcome.teacher_calls == 0
        assert outcome.open_generation_state == W05_OPEN_GENERATION_STATE
        assert outcome.execution_state == {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W05_STARTED": 0,
            "W06_STARTED": 0,
            "formal_w05_training_runs": 0,
            "teacher_calls": 0,
        }


def test_w05_runtime_dump_readback_is_zero_transport_and_zero_learning(tmp_path):
    """dump/readback 保持全逻辑 digest，且本次不读 payload、不写学习态。"""
    config = _config(tmp_path)
    outcome = run_language_stage5(config)
    readback = load_w05_candidate_dump(config)
    assert readback.dump_readback
    assert _logic_key(readback) == _logic_key(outcome)
    assert readback.transaction_digest == outcome.transaction_digest
    assert readback.dump_manifest_sha256 == outcome.dump_manifest_sha256
    assert readback.payload_gets_this_call == 0
    assert readback.payload_bytes_this_call == 0
    assert readback.new_learning_write_count == 0
    assert readback.teacher_calls == 0


@pytest.mark.parametrize("fault", W05_FAILURE_POINT_KEYS)
def test_w05_runtime_six_faults_recover_in_same_root(tmp_path, fault):
    """六故障点必须在同一 coordinator/root 上 restart 到干净逻辑状态。"""
    root = tmp_path / "faulted"
    with pytest.raises(W05InjectedFault):
        run_language_stage5(_config(root, mode="fresh", fault=fault))
    recovered = run_language_stage5(_config(root, mode="restart"))
    clean = run_language_stage5(_config(tmp_path / "clean", mode="fresh"))
    assert _logic_key(recovered) == _logic_key(clean)
    assert recovered.transaction_event_count == 5
    assert recovered.learning_attempt_count == 2
    assert recovered.dump_manifest_sha256 == clean.dump_manifest_sha256


def test_w05_transaction_rejects_duplicate_commit_drift():
    """恢复可精确重放同 commit，但异内容重复 commit 必须拒绝。"""
    backend = DictBackend()
    try:
        store = W05TransactionStore(
            backend, run_id=W05_FORMAL_RUN_ID, execution_identity_key=(1, 2, 3))
        store.begin({"request": [1]})
        store.preview({"shards": 16})
        first = store.commit({"value": 1})
        assert store.commit({"value": 1}) == first
        with pytest.raises(W05TransactionError, match="内容漂移"):
            store.commit({"value": 2})
        with pytest.raises(W05TransactionError, match="不得跳级"):
            store.published({"manifest": "early"})
    finally:
        backend.close()


def test_w05_runtime_has_16_shards_five_events_and_resource_bounds(tmp_path):
    """preview/cursor/manifest-last 次序和全部冻结资源预算必须闭合。"""
    config = _config(tmp_path)
    outcome = run_language_stage5(config)
    dump = _dump(config)
    transaction = dump["transaction"]
    assert len(transaction) == 4
    preview = transaction[1]
    cursor = transaction[3]
    assert preview["logical_shard_count"] == 16
    assert len(preview["shards"]) == 16
    assert [item["shard_ordinal"] for item in preview["shards"]] == list(range(16))
    assert cursor["completed_shards"] == list(range(16))
    assert dump["transaction_event_count"] == 5
    mapping = {
        "actual_checkpoint_count": "max_checkpoint_count",
        "actual_logic_operations": "max_logic_operations",
        "actual_payload_bytes": "max_payload_bytes",
        "actual_payload_gets": "max_payload_gets",
        "actual_recompute_objects": "max_recompute_objects",
        "actual_records": "max_records",
        "actual_segments": "max_segments",
        "actual_workers": "max_workers",
    }
    for actual, maximum in mapping.items():
        assert outcome.resource_report[actual] <= outcome.resource_budget[maximum]


def test_w05_nine_carriers_share_one_proposition_and_27_direction_cells(tmp_path):
    """九载体必须经同一 projection path 绑定同一 Proposition/Role/Scope。"""
    config = _config(tmp_path)
    run_language_stage5(config)
    scope = _dump(config)["commit"]["carrier_scope"]
    records = scope["records"]
    assert scope["scope_key"] == W05_LC16_SCOPE_KEY
    assert scope["carrier_count"] == 9
    assert scope["direction_cell_count"] == 27
    assert tuple(item["carrier_key"] for item in records) == W05_LC16_CARRIER_KEYS
    assert len({tuple(item["proposition_key"]) for item in records}) == 1
    assert len({str(item["occurrence_keys"]) for item in records}) == 1
    assert len({str(item["role_binding_keys"]) for item in records}) == 1
    assert len({tuple(item["proposition_context_key"]) for item in records}) == 1
    for record in records:
        assert record["retained_projection_equal"] == 1
        assert tuple(
            item["direction"] for item in record["direction_cells"]
        ) == W05_LC16_DIRECTIONS
        assert all(item["scope_key"] == W05_LC16_SCOPE_KEY
                   for item in record["direction_cells"])
        assert all(item["use_key"] and item["outcome_key"]
                   for item in record["direction_cells"])


def test_w05_runtime_preserves_w04_receipt_and_owns_no_w04_table(tmp_path):
    """W-04 receipt/root 保持只读，W-05 coordinator 不注册 W-04 owner 表。"""
    before = hashlib.sha256(W04_RECEIPT.read_bytes()).hexdigest()
    outcome = run_language_stage5(_config(tmp_path))
    after = hashlib.sha256(W04_RECEIPT.read_bytes()).hexdigest()
    assert before == after == W04_RECEIPT_SHA256
    assert "ph2_w05_transaction_event" in outcome.owned_tables
    assert not any(item.startswith("ph2_w04_") for item in outcome.owned_tables)
