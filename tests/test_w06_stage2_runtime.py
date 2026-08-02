"""W06-04 transaction、恢复、九载体 relation、资源与 retention 专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_LC16_CARRIER_KEYS,
    W05_LC16_DIRECTIONS,
    W05_LC16_SCOPE_KEY,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_FAILURE_POINT_KEYS,
    W06_FORMAL_RUN_ID,
    W06_OPEN_GENERATION_STATE,
    W06_W05_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w06_faults import W06InjectedFault
from pure_integer_ai.experiments.ph2_w06_runtime import (
    W06RuntimeConfig,
    load_w06_public_dump,
    run_language_stage6_public,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.experiments.ph2_w06_transaction import (
    W06TransactionError,
    W06TransactionStore,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "307bf185698551434107ae95169576d26d61762f"
RETENTION_SHA256 = {
    "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json": (
        "141a6c2341671d4d92d9974a355b8081fd12dff17315f5d1f60913a45c31c8f1"
    ),
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json": (
        "ef64636ab287eacbacae4040f59da74bb4105374cba31d756e1ddefaf86043f6"
    ),
    "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json": (
        "153db3d7f3c0fca04642f4198df16e3c1adb0f5c78e4d6c7c59d35122989727b"
    ),
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json": (
        "64c2fff496e766df880d2db1b184e2b8a009abd3b37b1a1b1331900458ccff78"
    ),
}


def _config(root, *, worker=1, mode="fresh", fault=None):
    """在一个隔离 W-06 public root/coordinator 上构造调度变体。"""
    return W06RuntimeConfig(
        repository_root=ROOT,
        run_root=root / "run",
        sqlite_path=root / "coordinator.sqlite",
        run_id=W06_FORMAL_RUN_ID,
        parent_run_id=W06_W05_BASE_RUN_ID,
        base_run_id=W06_W05_BASE_RUN_ID,
        base_fence_key=None,
        worker_count=worker,
        mode=mode,
        current_remote_commit_sha1=HEAD,
        fault_point=fault,
    )


def _logic_key(outcome):
    """返回排除物理 worker/attempt 的完整 W-06 逻辑身份。"""
    return (
        outcome.logical_state_digest,
        outcome.candidate_digest,
        outcome.relation_digest,
        outcome.source_evidence_digest,
        outcome.active_projection_digest,
        outcome.carrier_scope_digest,
        outcome.candidate_count,
        outcome.active_candidate_count,
        outcome.relation_summary_digests,
        outcome.artifact_counts,
    )


def _dump(config):
    """回读测试 root 的 canonical W-06 public dump。"""
    path = (
        Path(config.run_root)
        / f"w06_run_{config.run_id:020d}"
        / "w06_dump_manifest.json"
    )
    value = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def runtime_evidence(tmp_path_factory):
    """共享一份隔离 clean run，避免非组合断言重复重建同一逻辑态。"""
    before = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in RETENTION_SHA256
    }
    config = _config(tmp_path_factory.mktemp("w06-runtime-evidence"))
    outcome = run_language_stage6_public(config)
    return config, outcome, _dump(config), before


def test_w06_runtime_worker_and_mode_are_bit_identical(tmp_path):
    """1/2/4 worker 与 fresh/restart/resume 的逻辑产物必须 bit-identical。"""
    outcomes = [
        run_language_stage6_public(_config(
            tmp_path / f"host-{worker}-{mode}", worker=worker, mode=mode))
        for worker in (1, 2, 4)
        for mode in ("fresh", "restart", "resume")
    ]
    assert len({_logic_key(item) for item in outcomes}) == 1
    for outcome in outcomes:
        assert outcome.candidate_count == 50
        assert outcome.active_candidate_count == 17
        assert outcome.transaction_event_count == 5
        assert outcome.payload_gets_this_call > 0
        assert outcome.payload_bytes_this_call > 0
        assert outcome.new_learning_write_count > 0
        assert outcome.teacher_calls == 0
        assert outcome.open_generation_state == W06_OPEN_GENERATION_STATE
        assert outcome.execution_state == {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W06_STARTED": 0,
            "W07_STARTED": 0,
            "formal_w06_training_runs": 0,
            "teacher_calls": 0,
        }


def test_w06_runtime_dump_readback_is_zero_transport_and_zero_learning(
        runtime_evidence):
    """dump/readback 保持全逻辑 digest，且本次不读 payload、不写学习态。"""
    config, outcome, _, _ = runtime_evidence
    readback = load_w06_public_dump(config)
    assert readback.dump_readback
    assert _logic_key(readback) == _logic_key(outcome)
    assert readback.transaction_digest == outcome.transaction_digest
    assert readback.dump_manifest_sha256 == outcome.dump_manifest_sha256
    assert readback.payload_gets_this_call == 0
    assert readback.payload_bytes_this_call == 0
    assert readback.new_learning_write_count == 0
    assert readback.teacher_calls == 0


@pytest.mark.parametrize("fault", W06_FAILURE_POINT_KEYS)
def test_w06_runtime_six_faults_recover_in_same_root(
        tmp_path, runtime_evidence, fault):
    """六故障点必须在同一 coordinator/root 上 restart 到干净逻辑状态。"""
    root = tmp_path / "faulted"
    with pytest.raises(W06InjectedFault):
        run_language_stage6_public(_config(root, mode="fresh", fault=fault))
    recovered = run_language_stage6_public(_config(root, mode="restart"))
    clean = runtime_evidence[1]
    assert _logic_key(recovered) == _logic_key(clean)
    assert recovered.transaction_event_count == 5
    assert recovered.learning_attempt_count == 2
    assert recovered.dump_manifest_sha256 == clean.dump_manifest_sha256


def test_w06_transaction_rejects_duplicate_commit_drift():
    """恢复可精确重放同 commit，但异内容重复 commit 必须拒绝。"""
    backend = DictBackend()
    try:
        store = W06TransactionStore(
            backend, run_id=W06_FORMAL_RUN_ID,
            execution_identity_key=(1, 2, 3))
        store.begin({"request": [1]})
        store.preview({"shards": 16})
        first = store.commit({"value": 1})
        assert store.commit({"value": 1}) == first
        with pytest.raises(W06TransactionError, match="内容漂移"):
            store.commit({"value": 2})
        with pytest.raises(W06TransactionError, match="不得跳级"):
            store.published({"manifest": "early"})
    finally:
        backend.close()


def test_w06_runtime_has_16_shards_five_events_and_resource_bounds(
        runtime_evidence):
    """preview/cursor/manifest-last 次序和全部冻结资源预算必须闭合。"""
    _, outcome, dump, _ = runtime_evidence
    transaction = dump["transaction"]
    assert len(transaction) == 4
    preview = transaction[1]
    cursor = transaction[3]
    assert preview["logical_shard_count"] == 16
    assert len(preview["shards"]) == 16
    assert [item["shard_ordinal"] for item in preview["shards"]] == list(
        range(16))
    assert sum(len(item["candidate_keys"])
               for item in preview["shards"]) == 50
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


def test_w06_nine_carriers_share_one_relation_and_27_direction_cells(
        runtime_evidence):
    """九载体必须经同一 projection path 绑定同一 relation/Role/Scope。"""
    scope = runtime_evidence[2]["commit"]["carrier_scope"]
    records = scope["records"]
    assert scope["scope_key"] == W05_LC16_SCOPE_KEY
    assert scope["carrier_count"] == 9
    assert scope["direction_cell_count"] == 27
    assert tuple(item["carrier_key"] for item in records) == (
        W05_LC16_CARRIER_KEYS)
    assert len({tuple(item["proposition_key"]) for item in records}) == 1
    assert len({tuple(item["predicate_key"]) for item in records}) == 1
    assert len({str(item["endpoint_keys"]) for item in records}) == 1
    assert len({str(item["role_binding_keys"]) for item in records}) == 1
    assert len({tuple(item["proposition_context_key"])
                for item in records}) == 1
    for record in records:
        assert record["retained_projection_equal"] == 1
        assert tuple(
            item["direction"] for item in record["direction_cells"]
        ) == W05_LC16_DIRECTIONS
        assert all(item["scope_key"] == W05_LC16_SCOPE_KEY
                   for item in record["direction_cells"])
        assert all(item["status"] == "SUPPORT"
                   for item in record["direction_cells"])
        assert all(item["use_key"] and item["outcome_key"]
                   for item in record["direction_cells"])


def test_w06_runtime_preserves_receipts_and_owns_no_historical_table(
        runtime_evidence):
    """W02--W05 receipt 只读，W-06 coordinator 不注册历史 owner 表。"""
    _, outcome, _, before = runtime_evidence
    after = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in RETENTION_SHA256
    }
    assert before == after == RETENTION_SHA256
    assert dict(outcome.retention_sha256) == RETENTION_SHA256
    assert "ph2_w06_transaction_event" in outcome.owned_tables
    assert not any(
        item.startswith(("ph2_w02_", "ph2_w03_", "ph2_w04_", "ph2_w05_"))
        for item in outcome.owned_tables
    )


def test_w06_runtime_reports_seven_substages_and_frozen_counts(
        runtime_evidence):
    """七关系 summary 必须保持 50 candidate、64 account 与 17 active。"""
    _, outcome, dump, _ = runtime_evidence
    learning = dump["commit"]["learning"]
    summaries = dump["commit"]["relation_summaries"]
    assert tuple(summaries) == tuple(sorted(W06_RELATION_SUBSTAGE_ORDER))
    assert learning["candidate_count"] == 50
    assert learning["schema_rejection_count"] == 1
    assert learning["relation_family_count"] == 14
    assert learning["evidence_account_count"] == 64
    assert learning["active_candidate_count"] == 17
    expected = {
        "PURE_ALIAS_REFERS": (5, 2),
        "SUBSET_MEMBER": (5, 2),
        "PROPERTY": (7, 2),
        "MEREOLOGY": (7, 2),
        "SIMILAR_ANTONYM": (7, 3),
        "PRECEDES": (9, 3),
        "CAUSES": (10, 3),
    }
    assert {
        key: (value["candidate_count"], value["active_count"])
        for key, value in summaries.items()
    } == expected
    artifacts = dict(outcome.artifact_counts)
    assert artifacts["CANDIDATE"] == 50
    assert artifacts["EVIDENCE_ACCOUNT"] == 64
    assert artifacts["ACTIVE_RELATION"] == 17
    assert artifacts["SCHEMA_REJECTION"] == 1
    assert artifacts["SUBSTAGE"] == 7
