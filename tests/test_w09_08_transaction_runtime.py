"""W09-08 单 owner 事务、恢复、worker 规范和 metadata dump 专项。"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_DIMENSION_KEYS,
    W09_FAILURE_POINT_KEYS,
    W09_ZERO_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w09_contract import W09_OWNER_KEY
from pure_integer_ai.experiments.ph2_w09_faults import W09InjectedFault
from pure_integer_ai.experiments.ph2_w09_runtime import (
    load_w09_public_dump,
    run_w09_public_transaction,
)
from pure_integer_ai.experiments.ph2_w09_runtime_contract import (
    W09RuntimeComponentReceipt,
    W09RuntimeConfig,
    W09RuntimeError,
    W09_RUNTIME_DUMP_NAME,
    W09_RUNTIME_OWNED_TABLES,
)
from pure_integer_ai.experiments.ph2_w09_transaction import (
    W09_EVENT_SEQUENCE,
    W09_TRANSACTION_EVENT_TABLE,
    W09TransactionError,
    W09TransactionStore,
    register_w09_transaction_table,
)
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]


def _config(
    root: Path,
    *,
    worker_count: int = 1,
    mode: str = "fresh",
    fault_point: str | None = None,
) -> W09RuntimeConfig:
    """创建隔离 run root 下的 W09-08 配置。"""
    return W09RuntimeConfig(
        ROOT,
        root,
        root / "coordinator.sqlite",
        worker_count=worker_count,
        mode=mode,
        fault_point=fault_point,
    )


def _manifest(config: W09RuntimeConfig) -> Path:
    """返回测试配置对应的 metadata dump。"""
    return (
        Path(config.run_root)
        / f"w09_run_{config.run_id:020d}"
        / W09_RUNTIME_DUMP_NAME
    )


@pytest.fixture(scope="module")
def public_run(tmp_path_factory: pytest.TempPathFactory):
    """只执行一次完整 public transaction，供只读断言共享。"""
    root = tmp_path_factory.mktemp("w09-08-public")
    config = _config(root)
    outcome = run_w09_public_transaction(config)
    readback = load_w09_public_dump(config)
    return config, outcome, readback


def test_one_owner_binds_five_dimensions_three_windows_jlc_clone_and_rollback(
    public_run,
) -> None:
    """五维与辅助结果必须共 owner，但 result/receipt identity 逐项独立。"""
    _config_value, outcome, readback = public_run
    evidence = outcome.evidence
    assert tuple(item.component_key for item in evidence.dimension_receipts) == W09_DIMENSION_KEYS
    assert tuple(item.component_key for item in evidence.window_receipts) == (
        "WINDOW-1",
        "WINDOW-2",
        "WINDOW-3",
    )
    assert tuple(item.component_key for item in evidence.all_receipts[-3:]) == (
        "J-LC-W09",
        "V-06-CLONE",
        "ROLLBACK-AUDIT",
    )
    assert {item.owner_key for item in evidence.all_receipts} == {W09_OWNER_KEY}
    assert len({item.result_key for item in evidence.all_receipts}) == len(evidence.all_receipts)
    assert len({item.receipt_key for item in evidence.all_receipts}) == len(evidence.all_receipts)
    assert len(evidence.learning_event_keys) == 27
    assert len(evidence.logical_shards) == 16
    assert outcome.transaction_event_count == 5
    assert outcome.canonical_key() == readback.canonical_key()
    assert outcome.dump_manifest_sha256 == readback.dump_manifest_sha256
    assert outcome.payload_gets_this_call > 0
    assert outcome.payload_bytes_this_call > 0
    assert readback.payload_gets_this_call == 0
    assert readback.payload_bytes_this_call == 0


def test_public_freeze_keeps_formal_mastered_and_readiness_zero(public_run) -> None:
    """W09-08 只能冻结 public bounded 结果，不得提前发布正式状态。"""
    _config_value, outcome, readback = public_run
    for current in (outcome, readback):
        assert dict(current.execution_state) == W09_ZERO_EXECUTION_STATE
        assert current.formal_evidenced == 0
        assert current.language_capability_mastered == 0
        assert current.language_readiness == 0
        assert current.evidence.teacher_calls == 0
        assert current.evidence.api_calls == 0
        assert current.evidence.llm_calls == 0
        assert current.evidence.host_write_count == 0


def test_dump_is_metadata_only_and_w09_owns_distinct_table(public_run) -> None:
    """dump 不得包含 surface/label/path，事务表也不得借用 W08 owner。"""
    config, _outcome, _readback = public_run
    payload = json.loads(_manifest(config).read_bytes())
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    for forbidden in (
        '"surface"',
        '"expected"',
        '"label"',
        "raw_text",
        "document_text",
        "ph2_w08_transaction_event",
    ):
        assert forbidden not in encoded
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
    assert W09_TRANSACTION_EVENT_TABLE in tables
    assert tuple(payload["owned_tables"]) == W09_RUNTIME_OWNED_TABLES


def test_worker_1_2_4_have_identical_host_dimensions_resources_and_dump(tmp_path: Path) -> None:
    """物理 worker 只影响 scheduling，不得改变任何规范发布结果。"""
    outcomes = tuple(
        run_w09_public_transaction(
            _config(tmp_path / f"worker-{worker}", worker_count=worker)
        )
        for worker in (1, 2, 4)
    )
    assert len({item.canonical_key() for item in outcomes}) == 1
    assert len({item.dump_manifest_sha256 for item in outcomes}) == 1
    assert len({item.evidence.host_state_key for item in outcomes}) == 1
    assert len({item.evidence.dimension_receipts for item in outcomes}) == 1
    assert len({item.evidence.resource_normalization for item in outcomes}) == 1
    assert len({item.scheduling_key for item in outcomes}) == 3


def test_completed_restart_resume_are_zero_payload_and_do_not_duplicate_events(
    tmp_path: Path,
) -> None:
    """已 commit 的恢复只补发布尾部，不得重读 payload 或重复学习事件。"""
    root = tmp_path / "completed"
    fresh = run_w09_public_transaction(_config(root))
    restart = run_w09_public_transaction(_config(root, mode="restart"))
    resume = run_w09_public_transaction(_config(root, mode="resume"))
    assert fresh.canonical_key() == restart.canonical_key() == resume.canonical_key()
    assert restart.payload_gets_this_call == resume.payload_gets_this_call == 0
    assert restart.payload_bytes_this_call == resume.payload_bytes_this_call == 0
    assert fresh.evidence.learning_event_keys == restart.evidence.learning_event_keys
    assert restart.evidence.learning_event_keys == resume.evidence.learning_event_keys
    connection = sqlite3.connect(str(root / "coordinator.sqlite"))
    try:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {W09_TRANSACTION_EVENT_TABLE} WHERE run_id = ?",
            (10,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 5


@pytest.mark.parametrize("fault_point", W09_FAILURE_POINT_KEYS)
@pytest.mark.parametrize("recovery_mode", ("restart", "resume"))
def test_six_failure_points_recover_without_half_receipt_or_duplicate_learning(
    tmp_path: Path,
    public_run,
    fault_point: str,
    recovery_mode: str,
) -> None:
    """六处注入都必须恢复到同一 dump，且 commit 后恢复为零 payload。"""
    _baseline_config, baseline, _readback = public_run
    root = tmp_path / recovery_mode / fault_point
    with pytest.raises(W09InjectedFault, match=fault_point):
        run_w09_public_transaction(_config(root, fault_point=fault_point))
    recovered = run_w09_public_transaction(_config(root, mode=recovery_mode))
    assert recovered.canonical_key() == baseline.canonical_key()
    assert recovered.dump_manifest_sha256 == baseline.dump_manifest_sha256
    assert recovered.transaction_event_count == 5
    assert len(recovered.evidence.learning_event_keys) == 27
    if fault_point in {"AFTER_COMMIT_BEFORE_CURSOR", "AFTER_MANIFEST_PUBLISH"}:
        assert recovered.payload_gets_this_call == 0
        assert recovered.payload_bytes_this_call == 0
    readback = load_w09_public_dump(_config(root, mode=recovery_mode))
    assert readback.canonical_key() == baseline.canonical_key()


def test_fresh_rejects_existing_transaction_and_tampered_dump(public_run, tmp_path: Path) -> None:
    """fresh 重放与半发布或被篡改的 dump 都必须 fail closed。"""
    config, _outcome, _readback = public_run
    with pytest.raises(W09RuntimeError, match="fresh mode"):
        run_w09_public_transaction(config)

    partial_root = tmp_path / "partial"
    partial = _config(partial_root, fault_point="AFTER_COMMIT_BEFORE_CURSOR")
    with pytest.raises(W09InjectedFault):
        run_w09_public_transaction(partial)
    target = _manifest(partial)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(W09RuntimeError, match="identity 漂移"):
        run_w09_public_transaction(_config(partial_root, mode="resume"))

    clean_root = tmp_path / "tampered"
    clean = _config(clean_root)
    run_w09_public_transaction(clean)
    _manifest(clean).write_text("{}", encoding="utf-8")
    with pytest.raises(W09RuntimeError, match="identity/state"):
        load_w09_public_dump(clean)


def test_transaction_rejects_owner_drift_skip_duplicate_drift_and_bad_json() -> None:
    """底层 owner 必须拒绝混写、越级、异内容重放和损坏物理行。"""
    backend = DictBackend()
    try:
        tx = W09TransactionStore(
            backend,
            run_id=10,
            owner_key=W09_OWNER_KEY,
            execution_identity_key=(1, 2, 3),
        )
        begin = tx.begin({"request": [1]})
        assert tx.begin({"request": [1]}) == begin
        with pytest.raises(W09TransactionError, match="内容漂移"):
            tx.begin({"request": [2]})
        with pytest.raises(W09TransactionError, match="不得跳级"):
            tx.commit({"commit": [1]})
        tx.preview({"preview": [1]})
        tx.commit({"commit": [1]})
        tx.cursor({"cursor": [1]})
        tx.published({"published": [1]})
        assert tuple(item.event_kind for item in tx.events()) == W09_EVENT_SEQUENCE
    finally:
        backend.close()
    with pytest.raises(W09TransactionError, match="owner"):
        W09TransactionStore(
            DictBackend(),
            run_id=10,
            owner_key="PH2_W08_TRANSACTION_OWNER",
            execution_identity_key=(1, 2, 3),
        )
    with pytest.raises(W09RuntimeError, match="owner"):
        W09RuntimeComponentReceipt(
            W09_DIMENSION_KEYS[0],
            "PH2_W08_TRANSACTION_OWNER",
            (1,),
            (2,),
            "PUBLIC_BOUNDED_PASS",
        )

    corrupt = DictBackend()
    try:
        register_w09_transaction_table(corrupt)
        corrupt.insert(
            W09_TRANSACTION_EVENT_TABLE,
            {
                "run_id": 10,
                "event_seq": 1,
                "event_kind": 1,
                "owner_key": W09_OWNER_KEY,
                "identity_sha256": "0" * 64,
                "payload_sha256": "0" * 64,
                "payload_json": "{",
            },
        )
        corrupt.commit()
        with pytest.raises(W09TransactionError, match="JSON 损坏"):
            W09TransactionStore(
                corrupt,
                run_id=10,
                owner_key=W09_OWNER_KEY,
                execution_identity_key=(1, 2, 3),
            )
    finally:
        corrupt.close()


@pytest.mark.parametrize(
    "updates",
    (
        {"run_id": 0},
        {"run_id": 9},
        {"parent_run_id": 8},
        {"worker_count": 3},
        {"mode": "continue"},
        {"fault_point": "UNKNOWN"},
    ),
)
def test_invalid_recovery_config_fails_before_transaction(tmp_path: Path, updates) -> None:
    """run/base/worker/mode/fault 配置漂移必须在创建事务前拒绝。"""
    values = dict(
        repository_root=ROOT,
        run_root=tmp_path,
        sqlite_path=tmp_path / "coordinator.sqlite",
    )
    values.update(updates)
    with pytest.raises(W09RuntimeError):
        run_w09_public_transaction(W09RuntimeConfig(**values))
    assert not (tmp_path / "coordinator.sqlite").exists()
