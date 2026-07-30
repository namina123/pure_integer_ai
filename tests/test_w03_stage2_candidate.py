"""PH2 W-03 candidate 合同冻结、不可覆盖发布与唯一运行守卫。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_w03_candidate as candidate_owner
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_w03_candidate import (
    W03_CANDIDATE_CODE_PATHS,
    W03_CANDIDATE_CONTRACT_FREEZE_NAME,
    W03_CANDIDATE_FIRST_RUN_GUARD_NAME,
    W03_CANDIDATE_HOST_FREEZE_NAME,
    W03_CANDIDATE_TEST_PATHS,
    build_w03_candidate_contract,
    consume_w03_candidate_first_run_guard,
    execute_w03_candidate_once,
    publish_w03_candidate_contract_freeze,
    verify_w03_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationObservation,
    formal_w03_publication_baseline,
)
from pure_integer_ai.experiments.ph2_w03_runtime import (
    W03RunOutcome,
    W03RuntimeConfig,
)
from pure_integer_ai.storage.backend import SQLiteBackend


REPOSITORY = Path(__file__).resolve().parents[1]
W02_ARTIFACTS = REPOSITORY.parent / "w02_artifacts"


def _publication_observation() -> W03PublicationObservation:
    """构造与已冻结 W03-00C 四项 CI 完全相同的离线观测。"""
    baseline = formal_w03_publication_baseline()
    return W03PublicationObservation(
        local_head_sha1=baseline.head_sha1,
        tracking_head_sha1=baseline.head_sha1,
        remote_head_sha1=baseline.head_sha1,
        ci_run_id=baseline.ci_run_id,
        ci_head_sha1=baseline.head_sha1,
        ci_status="completed",
        ci_conclusion="success",
        ci_jobs=baseline.ci_jobs,
    )


def _backend_profile_key() -> tuple[int, ...]:
    """读取正式 SQLite backend profile，不产生持久候选状态。"""
    backend = SQLiteBackend(":memory:")
    try:
        return backend.storage_capabilities().stable_key()
    finally:
        backend.close()


def _contract() -> dict[str, object]:
    """建立零 payload、零正式运行的候选冻结合同。"""
    return build_w03_candidate_contract(
        REPOSITORY,
        W02_ARTIFACTS,
        global_manifest_path=FORMAL_GLOBAL_MANIFEST_PATH,
        backend_profile_key=_backend_profile_key(),
        current_remote_commit_sha1=formal_w03_publication_baseline().head_sha1,
        publication_observation=_publication_observation(),
    )


def test_candidate_contract_binds_complete_inventory_and_pre_run_state():
    """冻结合同必须显式绑定能力、资料、门槛、恢复和 run-count=0。"""
    contract = _contract()

    assert contract["artifact_kind"] == "PH2_W03_CANDIDATE_CONTRACT_FREEZE"
    assert contract["formal_w03_training_runs"] == 0
    assert contract["execution_state"] == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W03_STARTED": 0,
        "W04_STARTED": 0,
        "formal_w03_training_runs": 0,
        "teacher_calls": 0,
    }
    assert tuple(item["path"] for item in contract["code_inventory"]) == (
        W03_CANDIDATE_CODE_PATHS)
    assert tuple(item["path"] for item in contract["test_inventory"]) == (
        W03_CANDIDATE_TEST_PATHS)
    assert contract["w02_binding"]["capability_code_count"] == 10
    assert contract["w02_binding"]["host_artifact_count"] == 11
    assert contract["w02_binding"]["formal_training_runs"] == 1
    assert contract["d03_w03_binding"]["train_pack_count"] == 6
    assert contract["visibility_counts"] == {
        "candidate": 12,
        "evaluator": 29,
        "teacher": 18,
    }
    assert contract["candidate_request"]["run_id"] == 4
    assert contract["candidate_request"]["parent_run_id"] == 3
    assert contract["candidate_request"]["base_run_id"] == 3
    assert contract["candidate_request"]["candidate_payload_count"] == 12
    assert contract["candidate_request"]["teacher_evidence_count"] == 6
    assert contract["recovery_protocol"]["logical_shard_count"] == 16
    assert contract["recovery_protocol"]["worker_counts"] == [1, 2, 4]
    assert len(contract["recovery_protocol"]["failure_points"]) == 6
    assert contract["evaluation_contract"]["threshold"] == {
        "fail_allowed": 0,
        "ne_policy": "BLOCK",
        "required_pass_denominator": 1,
        "required_pass_numerator": 1,
    }
    assert len(contract["evaluation_contract"]["evaluation_order"]) == 5


def test_contract_freeze_and_first_run_guard_are_exclusive(tmp_path: Path):
    """freeze 与 first-run guard 均只允许首次写入，重复调用不得改字节。"""
    contract = _contract()
    root = tmp_path / "w03_candidate"
    freeze_path, freeze_sha = publish_w03_candidate_contract_freeze(
        REPOSITORY,
        W02_ARTIFACTS,
        root,
        contract,
    )
    assert freeze_path.name == W03_CANDIDATE_CONTRACT_FREEZE_NAME
    assert verify_w03_candidate_contract_freeze(freeze_path, contract) == freeze_sha
    before = freeze_path.read_bytes()
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w03_candidate_contract_freeze(
            REPOSITORY,
            W02_ARTIFACTS,
            root,
            contract,
        )
    assert freeze_path.read_bytes() == before

    guard_path, guard_sha = consume_w03_candidate_first_run_guard(
        root,
        candidate_contract_sha256=freeze_sha,
    )
    assert guard_path.name == W03_CANDIDATE_FIRST_RUN_GUARD_NAME
    assert hashlib.sha256(guard_path.read_bytes()).hexdigest() == guard_sha
    guard_before = guard_path.read_bytes()
    with pytest.raises(RuntimeError, match="已经消费|不可重跑"):
        consume_w03_candidate_first_run_guard(
            root,
            candidate_contract_sha256=freeze_sha,
        )
    assert guard_path.read_bytes() == guard_before


def test_contract_root_must_be_git_external_and_w02_isolated(tmp_path: Path):
    """candidate root 不能位于公开 Git 或不可变 W-02 artifact 树内。"""
    contract = _contract()
    with pytest.raises(RuntimeError, match="Git 外|隔离"):
        publish_w03_candidate_contract_freeze(
            REPOSITORY,
            W02_ARTIFACTS,
            REPOSITORY / "candidate-forbidden",
            contract,
        )
    with pytest.raises(RuntimeError, match="Git 外|隔离"):
        publish_w03_candidate_contract_freeze(
            REPOSITORY,
            W02_ARTIFACTS,
            W02_ARTIFACTS / "candidate-forbidden",
            contract,
        )


def test_contract_or_freeze_identity_drift_fails_before_run(tmp_path: Path):
    """合同或 freeze 任一字节漂移都必须在 first-run guard 前失败。"""
    contract = _contract()
    root = tmp_path / "w03_candidate"
    freeze_path, freeze_sha = publish_w03_candidate_contract_freeze(
        REPOSITORY,
        W02_ARTIFACTS,
        root,
        contract,
    )
    changed = dict(contract)
    changed["formal_w03_training_runs"] = 1
    with pytest.raises(RuntimeError, match="identity|合同|漂移"):
        verify_w03_candidate_contract_freeze(freeze_path, changed)
    with pytest.raises(RuntimeError, match="SHA-256|identity|漂移"):
        consume_w03_candidate_first_run_guard(
            root,
            candidate_contract_sha256="0" * 64,
        )
    assert not (root / W03_CANDIDATE_FIRST_RUN_GUARD_NAME).exists()
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == freeze_sha


def _synthetic_outcome(
        root: Path,
        *,
        dump_readback: bool,
        ) -> W03RunOutcome:
    """形成不含任何训练 payload 的 host/dump 编排测试结果。"""
    digest = "1" * 64
    counts = tuple(sorted({
        "EVIDENCE_ACCOUNT": 64,
        "GENERATION_CHOICE": 2,
        "GENERATION_DECISION": 3,
        "GENERATION_OUTCOME": 4,
        "GENERATION_USE": 3,
        "PROJECTION": 59,
        "TRAIN_ENVELOPE": 163,
        "W02_RETENTION": 1,
    }.items()))
    return W03RunOutcome(
        digest,
        digest,
        digest,
        digest,
        digest,
        digest,
        digest,
        digest,
        counts,
        {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W03_STARTED": 0,
            "W04_STARTED": 0,
            "formal_w03_training_runs": 0,
            "teacher_calls": 0,
        },
        {
            "actual_checkpoint_count": 1,
            "actual_logic_operations": 100,
            "actual_payload_bytes": 0 if dump_readback else 100,
            "actual_payload_gets": 0 if dump_readback else 18,
            "actual_recompute_objects": 10,
            "actual_records": 100,
            "actual_segments": 5,
            "actual_workers": 4,
            "logical_shards": 16,
            "merged_records": 163,
            "requested_workers": 4,
            "teacher_calls": 0,
        },
        {
            "max_checkpoint_count": 768,
            "max_logic_operations": 3_000_000,
            "max_payload_bytes": 201_326_592,
            "max_payload_gets": 196_608,
            "max_recompute_objects": 300_000,
            "max_records": 300_000,
            "max_segments": 12_288,
            "max_workers": 4,
        },
        3 if dump_readback else 4,
        1,
        1,
        0 if dump_readback else 100,
        0,
        True,
        str((root / ("readback.sqlite3" if dump_readback
                     else "candidate.sqlite3")).resolve()),
        ("ph2_w03_artifact",),
        dump_readback,
    )


def test_atomic_candidate_orchestration_guards_run_and_freezes_host(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """test-local runtime 必须在 guard 后只执行一次并以 fresh dump 封存。"""
    contract = _contract()
    root = tmp_path / "w03_candidate"
    _, contract_sha = publish_w03_candidate_contract_freeze(
        REPOSITORY,
        W02_ARTIFACTS,
        root,
        contract,
    )
    request = contract["candidate_request"]
    config = W03RuntimeConfig(
        repository_root=REPOSITORY,
        global_manifest_path=FORMAL_GLOBAL_MANIFEST_PATH,
        w02_artifacts_root=W02_ARTIFACTS,
        run_root=root / "runs",
        sqlite_path=root / "candidate.sqlite3",
        run_id=4,
        parent_run_id=3,
        base_run_id=3,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=4,
        mode="fresh",
        current_remote_commit_sha1=formal_w03_publication_baseline().head_sha1,
    )
    calls = []

    def fake_run(actual: W03RuntimeConfig) -> W03RunOutcome:
        """确认 guard 先于唯一 test-local host 执行，并留下可冻结文件。"""
        assert (root / W03_CANDIDATE_FIRST_RUN_GUARD_NAME).is_file()
        calls.append("run")
        (root / "runs").mkdir()
        (root / "runs" / "synthetic.segment").write_bytes(b"segment")
        (root / "candidate.sqlite3").write_bytes(b"candidate")
        return _synthetic_outcome(root, dump_readback=False)

    def fake_load(
            actual: W03RuntimeConfig,
            *,
            target_sqlite_path: str | Path,
            ) -> W03RunOutcome:
        """只在 host 完成后写 test-local fresh readback 文件。"""
        assert calls == ["run"]
        calls.append("readback")
        Path(target_sqlite_path).write_bytes(b"readback")
        return _synthetic_outcome(root, dump_readback=True)

    monkeypatch.setattr(candidate_owner, "run_language_stage2", fake_run)
    monkeypatch.setattr(candidate_owner, "load_w03_candidate_dump", fake_load)
    outcome, readback, freeze_path, freeze_sha, guard_path, guard_sha = (
        execute_w03_candidate_once(
            REPOSITORY,
            W02_ARTIFACTS,
            root,
            config=config,
            contract=contract,
            candidate_contract_sha256=contract_sha,
            dump_readback_sqlite_path=root / "readback.sqlite3",
        )
    )
    assert calls == ["run", "readback"]
    assert outcome.execution_state["formal_w03_training_runs"] == 1
    assert readback.dump_readback is True
    assert freeze_path.name == W03_CANDIDATE_HOST_FREEZE_NAME
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == freeze_sha
    assert hashlib.sha256(guard_path.read_bytes()).hexdigest() == guard_sha

    with pytest.raises(RuntimeError, match="已经消费|不可重跑"):
        execute_w03_candidate_once(
            REPOSITORY,
            W02_ARTIFACTS,
            root,
            config=config,
            contract=contract,
            candidate_contract_sha256=contract_sha,
            dump_readback_sqlite_path=root / "readback-2.sqlite3",
        )
    assert calls == ["run", "readback"]
