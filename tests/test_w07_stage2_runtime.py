"""W07-04 transaction、恢复、九载体 logic、资源与 retention 专项。"""
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
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_FAILURE_POINT_KEYS,
    W07_FORMAL_RUN_ID,
    W07_OPEN_GENERATION_STATE,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
)
from pure_integer_ai.experiments.ph2_w07_faults import (
    W07InjectedFault,
    hit_w07_fault,
)
from pure_integer_ai.experiments.ph2_w07_runtime import (
    W07RuntimeConfig,
    load_w07_public_dump,
    run_language_stage7_public,
)
from pure_integer_ai.experiments.ph2_w07_transaction import (
    W07TransactionError,
    W07TransactionStore,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]
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
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json": (
        "aaf35a8346446e80d71f057ae391d9a734a864ced317fa06f2ea01f99efbc0e7"
    ),
}


def _config(root, *, worker=1, mode="fresh", fault=None, fence=None):
    return W07RuntimeConfig(
        repository_root=ROOT,
        run_root=root / "run",
        sqlite_path=root / "coordinator.sqlite",
        run_id=W07_FORMAL_RUN_ID,
        parent_run_id=W07_W06_BASE_RUN_ID,
        base_run_id=W07_W06_BASE_RUN_ID,
        base_fence_key=fence,
        worker_count=worker,
        mode=mode,
        fault_point=fault,
    )


def _logic_key(outcome):
    return (
        outcome.logical_state_digest,
        outcome.candidate_digest,
        outcome.logic_digest,
        outcome.source_evidence_digest,
        outcome.active_projection_digest,
        outcome.carrier_scope_digest,
        outcome.candidate_count,
        outcome.active_candidate_count,
        outcome.logic_summary_digests,
        outcome.artifact_counts,
    )


def _dump(config):
    path = (
        Path(config.run_root)
        / f"w07_run_{config.run_id:020d}"
        / "w07_dump_manifest.json"
    )
    value = parse_canonical_json_bytes(path.read_bytes(), require_object=True)
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def runtime_evidence(tmp_path_factory):
    before = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in RETENTION_SHA256
    }
    config = _config(tmp_path_factory.mktemp("w07-runtime-evidence"))
    outcome = run_language_stage7_public(config)
    return config, outcome, _dump(config), before


def test_w07_runtime_dump_readback_is_zero_transport_and_zero_learning(
        runtime_evidence):
    config, outcome, _, _ = runtime_evidence
    readback = load_w07_public_dump(config)
    assert readback.dump_readback
    assert _logic_key(readback) == _logic_key(outcome)
    assert readback.transaction_digest == outcome.transaction_digest
    assert readback.payload_gets_this_call == 0
    assert readback.payload_bytes_this_call == 0
    assert readback.new_learning_write_count == 0
    assert readback.teacher_calls == 0


def test_w07_transaction_rejects_duplicate_commit_and_publish_drift():
    backend = DictBackend()
    try:
        store = W07TransactionStore(
            backend,
            run_id=W07_FORMAL_RUN_ID,
            execution_identity_key=(1, 2, 3),
        )
        store.begin({"request": [1]})
        store.preview({"shards": 16})
        first = store.commit({"value": 1})
        assert store.commit({"value": 1}) == first
        with pytest.raises(W07TransactionError, match="内容漂移"):
            store.commit({"value": 2})
        with pytest.raises(W07TransactionError, match="不得跳级"):
            store.published({"manifest": "early"})
    finally:
        backend.close()


@pytest.mark.parametrize("fault", W07_FAILURE_POINT_KEYS)
def test_w07_six_fault_points_are_registered_and_fail_closed(fault):
    with pytest.raises(W07InjectedFault, match=fault):
        hit_w07_fault(fault, fault)
    hit_w07_fault(fault, None)


def test_w07_after_commit_fault_recovers_in_same_root(
        tmp_path, runtime_evidence):
    root = tmp_path / "faulted"
    with pytest.raises(W07InjectedFault, match="AFTER_COMMIT_BEFORE_CURSOR"):
        run_language_stage7_public(_config(
            root, mode="fresh", fault="AFTER_COMMIT_BEFORE_CURSOR"))
    recovered = run_language_stage7_public(_config(root, mode="restart"))
    assert _logic_key(recovered) == _logic_key(runtime_evidence[1])
    assert recovered.transaction_event_count == 5
    assert recovered.learning_attempt_count == 2


def test_w07_worker_and_mode_are_physical_scheduling_only():
    common = dict(
        run_id=W07_FORMAL_RUN_ID,
        parent_run_id=W07_W06_BASE_RUN_ID,
        base_run_id=W07_W06_BASE_RUN_ID,
        stage_key=W07_STAGE_KEY,
        owner_key="PH2_W07_TRANSACTION_OWNER",
        runner_key=W07_RUNNER_KEY,
        baseline_commit_sha1="a" * 40,
        context_key=(1,),
        backend_profile_key=(2,),
        base_fence_key=(3,),
        resource_budget=tuple(sorted(W07_RESOURCE_BUDGET.items())),
        candidate_payload_paths=("candidate",),
        teacher_evidence_paths=("evidence",),
    )
    requests = tuple(W07RunRequest(
        **common, worker_count=worker, mode=mode)
        for worker, mode in ((1, "fresh"), (2, "restart"), (4, "resume")))
    assert len({item.execution_identity_key() for item in requests}) == 1
    assert len({item.scheduling_key() for item in requests}) == 3


def test_w07_runtime_has_16_shards_five_events_and_resource_bounds(
        runtime_evidence):
    _, outcome, dump, _ = runtime_evidence
    transaction = dump["transaction"]
    assert len(transaction) == 4
    assert transaction[1]["logical_shard_count"] == 16
    assert [item["shard_ordinal"]
            for item in transaction[1]["shards"]] == list(range(16))
    assert sum(len(item["candidate_keys"])
               for item in transaction[1]["shards"]) == 71
    assert transaction[3]["completed_shards"] == list(range(16))
    assert outcome.transaction_event_count == 5
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


def test_w07_nine_carriers_bind_seven_logic_and_189_urg_cells(
        runtime_evidence):
    scope = runtime_evidence[2]["commit"]["carrier_scope"]
    assert scope["scope_key"] == W05_LC16_SCOPE_KEY
    assert scope["carrier_count"] == 9
    assert scope["carrier_projection_count"] == 9
    assert scope["logic_projection_binding_count"] == 63
    assert scope["logic_cell_count"] == 189
    assert tuple(item["carrier_key"] for item in scope["records"]) == (
        W05_LC16_CARRIER_KEYS)
    for record in scope["records"]:
        assert tuple(item["substage"] for item in record["projections"]) == (
            W07_SUBSTAGE_ORDER)
        assert all(item["retained_projection_equal"] == 1
                   for item in record["projections"])
        assert {
            (item["substage"], item["direction"])
            for item in record["logic_cells"]
        } == {
            (substage, direction)
            for substage in W07_SUBSTAGE_ORDER
            for direction in W05_LC16_DIRECTIONS
        }
        assert all(item["scope_key"] == W05_LC16_SCOPE_KEY
                   and item["status"] == "SUPPORT"
                   and item["use_commitment"]
                   and item["outcome_commitment"]
                   for item in record["logic_cells"])
    targets = runtime_evidence[2]["commit"]["logic_targets"]
    assert tuple(item["substage"] for item in targets) == W07_SUBSTAGE_ORDER
    assert all(len(item["direction_artifacts"]) == 3 for item in targets)
    assert all(cell["use_key"] and cell["outcome_key"]
               for item in targets for cell in item["direction_artifacts"])


def test_w07_runtime_preserves_receipts_and_owns_no_historical_table(
        runtime_evidence):
    _, outcome, _, before = runtime_evidence
    after = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in RETENTION_SHA256
    }
    assert before == after == RETENTION_SHA256
    assert dict(outcome.retention_sha256) == RETENTION_SHA256
    assert "ph2_w07_transaction_event" in outcome.owned_tables
    assert not any(item.startswith((
        "ph2_w02_", "ph2_w03_", "ph2_w04_", "ph2_w05_", "ph2_w06_",
    )) for item in outcome.owned_tables)


def test_w07_runtime_reports_frozen_learning_and_zero_formal_state(
        runtime_evidence):
    _, outcome, dump, _ = runtime_evidence
    learning = dump["commit"]["learning"]
    assert learning == {
        "active_operator_count": 36,
        "archived_candidate_count": 7,
        "candidate_count": 71,
        "conflict_candidate_count": 15,
        "evidence_application_count": 63,
        "operator_evidence_account_count": 94,
        "refuted_candidate_count": 7,
        "reparse_count": 7,
        "schema_rejection_count": 3,
        "superseded_candidate_count": 8,
        "unknown_candidate_count": 13,
        "withdrawal_count": 0,
    }
    assert outcome.execution_state == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_RUNTIME_EVIDENCED": 1,
        "W07_STARTED": 0,
        "W08_STARTED": 0,
        "formal_w07_training_runs": 0,
        "teacher_calls": 0,
    }
    assert outcome.open_generation_state == W07_OPEN_GENERATION_STATE
    assert dict(outcome.artifact_counts) == {
        "ACTIVE_OPERATOR": 36,
        "CANDIDATE": 71,
        "CARRIER_PROJECTION": 9,
        "EVIDENCE_ACCOUNT": 94,
        "EVIDENCE_APPLICATION": 63,
        "LOGICAL_SHARD": 16,
        "LOGIC_SCOPE_CELL": 189,
        "LOGIC_USE": 21,
        "OPERATOR_PROFILE": 7,
        "SCHEMA_REJECTION": 3,
        "SUBSTAGE": 7,
    }


def test_w07_base_fence_and_partial_dump_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="base fence"):
        run_language_stage7_public(_config(tmp_path / "fence", fence=(9, 9)))
    config = _config(tmp_path / "partial")
    path = (
        Path(config.run_root)
        / f"w07_run_{config.run_id:020d}"
        / "w07_dump_manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"artifact_kind":"PH2_W07_PUBLIC_RUNTIME_DUMP"')
    with pytest.raises((RuntimeError, ValueError)):
        load_w07_public_dump(config)
