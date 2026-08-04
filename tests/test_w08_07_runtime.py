"""W08-07 独立 transaction、恢复、资源、retention 与 dump 专项。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_FAILURE_POINT_KEYS,
    W08_OWNER_KEY,
    W08_RESOURCE_BUDGET,
    W08ContractError,
    make_w08_request,
    open_w08_frozen_contract,
    validate_w08_request,
)
from pure_integer_ai.experiments.ph2_w08_faults import W08InjectedFault
from pure_integer_ai.experiments.ph2_w08_runtime import (
    W08_PUBLIC_DUMP_NAME,
    _retention,
    load_w08_public_dump,
    run_language_stage8_public,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_OPEN_GENERATION_PREFORMAL_STATE,
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
    W08RuntimeConfig,
    W08RuntimeError,
    W08RuntimeResourceReceipt,
)
from pure_integer_ai.experiments.ph2_w08_transaction import (
    W08_EVENT_SEQUENCE,
    W08_TRANSACTION_EVENT_TABLE,
    W08TransactionError,
    W08TransactionStore,
    register_w08_transaction_table,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]


def _config(root: Path, *, worker_count: int = 1, mode: str = "fresh", fault=None):
    return W08RuntimeConfig(
        ROOT,
        root,
        root / "coordinator.sqlite",
        worker_count=worker_count,
        mode=mode,
        fault_point=fault,
    )


def _manifest(config: W08RuntimeConfig) -> Path:
    return (
        Path(config.run_root)
        / f"w08_run_{config.run_id:020d}"
        / W08_PUBLIC_DUMP_NAME
    )


@pytest.fixture(scope="module")
def public_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("w08-07-public")
    config = _config(root)
    outcome = run_language_stage8_public(config)
    readback = load_w08_public_dump(config)
    return config, outcome, readback


def test_public_runtime_commits_five_dimensions_fifteen_uses_and_hard_conjuncts(
    public_run,
):
    _config_value, outcome, readback = public_run
    assert tuple(item.dimension_key for item in outcome.artifacts) == W08_DIMENSION_KEYS
    assert tuple((item.dimension_key, item.consumer_key) for item in outcome.uses) == tuple(
        (dimension, consumer)
        for dimension in W08_DIMENSION_KEYS
        for consumer in W08_CONSUMER_KEYS
    )
    assert all(item.outcome_state == "RESOLVED" for item in outcome.uses)
    assert all(item.evidence_keys for item in outcome.uses)
    assert tuple(item.conjunct_key for item in outcome.hard_conjuncts) == (
        W08_RUNTIME_HARD_CONJUNCT_KEYS
    )
    assert all(item.state == "PUBLIC_BOUNDED_PASS" for item in outcome.hard_conjuncts)
    assert outcome.transaction_event_count == 5
    assert outcome.compiled_artifact_count == 5
    assert outcome.owned_tables == (W08_TRANSACTION_EVENT_TABLE,)
    assert outcome.payload_gets_this_call > 0
    assert outcome.payload_bytes_this_call > 0
    assert outcome.canonical_key() == readback.canonical_key()
    assert readback.dump_readback
    assert readback.payload_gets_this_call == 0
    assert readback.payload_bytes_this_call == 0


def test_public_runtime_keeps_preformal_state_and_zero_forbidden_accounts(public_run):
    _config_value, outcome, readback = public_run
    for current in (outcome, readback):
        assert dict(current.execution_state)["W08_STARTED"] == 0
        assert dict(current.execution_state)["formal_w08_training_runs"] == 0
        assert current.open_generation_state == W08_OPEN_GENERATION_PREFORMAL_STATE
        assert current.teacher_calls == 0
        assert current.evaluator_label_reads == 0
        assert current.future_payload_reads == 0
        assert current.host_learning_writes == 0
        assert current.memory_learning_writes == 0


def test_dump_is_metadata_only_and_w08_owns_a_distinct_transaction_table(public_run):
    config, _outcome, _readback = public_run
    payload = json.loads(_manifest(config).read_bytes())
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert "document_text" not in encoded
    assert "raw_text" not in encoded
    assert '"surface"' not in encoded
    connection = sqlite3.connect(str(Path(config.sqlite_path)))
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert W08_TRANSACTION_EVENT_TABLE in tables
    assert "ph2_w07_transaction_event" not in tables


def test_worker_1_2_4_runs_are_semantically_canonical(tmp_path):
    outcomes = tuple(
        run_language_stage8_public(
            _config(tmp_path / f"worker-{worker}", worker_count=worker)
        )
        for worker in (1, 2, 4)
    )
    assert len({item.canonical_key() for item in outcomes}) == 1
    assert len({item.scheduling_key for item in outcomes}) == 3
    assert tuple(item.resource_report.actual_workers for item in outcomes) == (1, 2, 4)
    assert len({item.artifact_commitment_key for item in outcomes}) == 1
    assert len({item.use_commitment_key for item in outcomes}) == 1


def test_completed_run_restart_and_resume_are_zero_payload_equivalent(tmp_path):
    root = tmp_path / "completed"
    fresh = run_language_stage8_public(_config(root))
    restart = run_language_stage8_public(_config(root, mode="restart"))
    resume = run_language_stage8_public(_config(root, mode="resume"))
    assert fresh.canonical_key() == restart.canonical_key() == resume.canonical_key()
    assert restart.payload_gets_this_call == resume.payload_gets_this_call == 0
    assert restart.payload_bytes_this_call == resume.payload_bytes_this_call == 0
    assert fresh.transaction_commitment_key == restart.transaction_commitment_key
    assert restart.transaction_commitment_key == resume.transaction_commitment_key


@pytest.mark.parametrize("fault_point", W08_FAILURE_POINT_KEYS)
@pytest.mark.parametrize("recovery_mode", ("restart", "resume"))
def test_six_fault_points_recover_to_fresh_semantics(
    tmp_path,
    public_run,
    fault_point,
    recovery_mode,
):
    _baseline_config, baseline, _readback = public_run
    root = tmp_path / recovery_mode / fault_point
    with pytest.raises(W08InjectedFault, match=fault_point):
        run_language_stage8_public(_config(root, fault=fault_point))
    recovered = run_language_stage8_public(_config(root, mode=recovery_mode))
    assert recovered.canonical_key() == baseline.canonical_key()
    assert recovered.transaction_event_count == 5
    if fault_point in {"AFTER_COMMIT_BEFORE_CURSOR", "AFTER_MANIFEST_PUBLISH"}:
        assert recovered.payload_gets_this_call == 0
        assert recovered.payload_bytes_this_call == 0
    assert load_w08_public_dump(_config(root, mode=recovery_mode)).canonical_key() == (
        baseline.canonical_key()
    )


def test_fresh_rejects_an_existing_transaction(public_run):
    config, _outcome, _readback = public_run
    with pytest.raises(W08RuntimeError, match="fresh mode"):
        run_language_stage8_public(config)


def test_transaction_rejects_duplicate_drift_skip_and_bad_payload_json():
    backend = DictBackend()
    try:
        tx = W08TransactionStore(
            backend,
            run_id=9,
            owner_key=W08_OWNER_KEY,
            execution_identity_key=(1, 2, 3),
        )
        begin = tx.begin({"request": [1]})
        assert tx.begin({"request": [1]}) == begin
        with pytest.raises(W08TransactionError, match="内容漂移"):
            tx.begin({"request": [2]})
        with pytest.raises(W08TransactionError, match="不得跳级"):
            tx.commit({"commit": [1]})
        tx.preview({"preview": [1]})
        commit = tx.commit({"commit": [1]})
        cursor = tx.cursor({"cursor": [1]})
        published = tx.published({"published": [1]})
        assert tx.commit({"commit": [1]}) == commit
        assert tx.cursor({"cursor": [1]}) == cursor
        assert tx.published({"published": [1]}) == published
        with pytest.raises(W08TransactionError, match="内容漂移"):
            tx.commit({"commit": [2]})
        with pytest.raises(W08TransactionError, match="内容漂移"):
            tx.cursor({"cursor": [2]})
        with pytest.raises(W08TransactionError, match="内容漂移"):
            tx.published({"published": [2]})
        assert tuple(item.event_kind for item in tx.events()) == W08_EVENT_SEQUENCE
    finally:
        backend.close()

    corrupt = DictBackend()
    try:
        register_w08_transaction_table(corrupt)
        corrupt.insert(
            W08_TRANSACTION_EVENT_TABLE,
            {
                "run_id": 9,
                "event_seq": 1,
                "event_kind": 1,
                "owner_key": W08_OWNER_KEY,
                "identity_sha256": "0" * 64,
                "payload_sha256": "0" * 64,
                "payload_json": "{",
            },
        )
        corrupt.commit()
        with pytest.raises(W08TransactionError, match="JSON 损坏"):
            W08TransactionStore(
                corrupt,
                run_id=9,
                owner_key=W08_OWNER_KEY,
                execution_identity_key=(1, 2, 3),
            )
    finally:
        corrupt.close()


def test_partial_or_tampered_dump_fails_closed(tmp_path):
    root = tmp_path / "partial"
    config = _config(root, fault="AFTER_COMMIT_BEFORE_CURSOR")
    with pytest.raises(W08InjectedFault):
        run_language_stage8_public(config)
    target = _manifest(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(W08RuntimeError, match="identity 漂移"):
        run_language_stage8_public(_config(root, mode="resume"))

    clean_root = tmp_path / "tampered"
    clean = _config(clean_root)
    run_language_stage8_public(clean)
    _manifest(clean).write_text("{}\n", encoding="utf-8")
    with pytest.raises(W08RuntimeError, match="identity/state 漂移"):
        load_w08_public_dump(clean)


def test_base_fence_and_future_path_are_rejected_before_payload_read():
    context = open_w08_frozen_contract(ROOT)
    request = make_w08_request(context)
    with pytest.raises(W08ContractError, match="base fence"):
        validate_w08_request(
            context,
            replace(request, base_fence_key=(80807, 999, 1)),
        )
    with pytest.raises(W08ContractError, match="forbidden paths"):
        validate_w08_request(
            context,
            replace(
                request,
                forbidden_payload_paths=(context.future_forbidden_paths[0],),
            ),
        )


def test_resource_overrun_is_rejected():
    with pytest.raises(W08RuntimeError, match="resource budget"):
        W08RuntimeResourceReceipt(
            W08_RESOURCE_BUDGET["max_records"] + 1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        )


def test_retention_drift_is_detected_without_modifying_public_receipts(monkeypatch):
    original = __import__(
        "pure_integer_ai.experiments.ph2_w08_runtime",
        fromlist=["read_w08_authority"],
    ).read_w08_authority

    def drifted(repository):
        value = original(repository)
        copied = dict(value)
        identities = [dict(item) for item in value["retention_identities"]]
        identities[0]["sha256"] = "0" * 64
        copied["retention_identities"] = identities
        return copied

    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_w08_runtime.read_w08_authority",
        drifted,
    )
    with pytest.raises(W08RuntimeError, match="retention identity 漂移"):
        _retention(ROOT)
