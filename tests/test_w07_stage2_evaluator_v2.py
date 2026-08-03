"""W07 evaluator v2 diagnostics, persistence, and safe one-shot closure."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_adapter import adapt_w07_training_payload
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07_PUBLIC_DIMENSION_KEYS,
    W07RunRequest,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_evaluator import (
    evaluate_w07_case,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07PrivateDimensionResult,
    decode_w07_private_documents,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_family import (
    build_w07_private_family_documents,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_contract import (
    W07_V2_NONE,
    W07V2AblationProgress,
    W07V2DiagnosticCursor,
    public_safe_w07_v2_aggregate,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_family import (
    build_w07_v2_private_family_documents,
    consume_w07_v2_private_first_run_guard,
    publish_w07_v2_private_family,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_runtime import (
    W07V2PrivateEvaluatorRuntimeConfig,
    run_w07_v2_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_consumers import (
    W07EvaluatorConsumerSuite,
    _LogicBundle,
)
from pure_integer_ai.experiments.ph2_w07_l02 import W07_L02_PREFIX
from pure_integer_ai.experiments.ph2_w07_learning import build_w07_learning_runtime
from pure_integer_ai.experiments.ph2_w07_logic_shared import slice_w07_adapter
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_HEAD = "c" * 40


def _adapted(tmp_path: Path):
    profile_backend = SQLiteBackend(str(tmp_path / "profile.sqlite"))
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=profile_backend.storage_capabilities().stable_key(),
        )
        request = W07RunRequest(
            W07_FORMAL_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_STAGE_KEY,
            context.owner_key,
            W07_RUNNER_KEY,
            context.baseline_commit_sha1,
            context.stable_key(),
            context.backend_profile_key,
            context.base_fence_key,
            1,
            "fresh",
            tuple(sorted(W07_RESOURCE_BUDGET.items())),
            tuple(item.relative_path
                  for item in context.candidate_payload_bindings),
            tuple(item.relative_path
                  for item in context.teacher_evidence_bindings),
        )
        payload = W07PayloadFirewall.open(ROOT, context, request).read_training_payload()
        return adapt_w07_training_payload(payload)
    finally:
        profile_backend.close()


def _documents(head: str = SYNTHETIC_HEAD):
    return build_w07_v2_private_family_documents(
        candidate_contract_sha256="a" * 64,
        candidate_host_freeze_sha256="b" * 64,
        evaluator_public_head_commit_sha1=head,
        nonce=(7, 11, 19, 31),
    )


def _public_result(case, passed: bool):
    return W07PrivateDimensionResult(
        case.dimension_key,
        "PASS" if passed else "FAIL",
        int(passed),
        1,
        int(not passed),
        0,
        "c" * 64,
    )


def test_v2_family_freeze_and_safe_partial_are_immutable(tmp_path: Path):
    documents = _documents()
    family = tmp_path / "family"
    freeze, freeze_sha = publish_w07_v2_private_family(family, documents)
    value = json.loads(freeze.read_text(encoding="utf-8"))
    assert value["evaluator_version"] == 2
    assert value["diagnostic_contract"]["operations"]
    guard, guard_sha = consume_w07_v2_private_first_run_guard(
        family, family_freeze_sha256=freeze_sha)
    assert hashlib.sha256(guard.read_bytes()).hexdigest() == guard_sha
    with pytest.raises(RuntimeError, match="already consumed"):
        consume_w07_v2_private_first_run_guard(
            family, family_freeze_sha256=freeze_sha)
    cursor = W07V2DiagnosticCursor(
        "ABLATION_AND_OR",
        "EVALUATE_CASE",
        "W-07-AND_OR-ABLATION",
        "W-07-AND_OR",
    )
    aggregate = public_safe_w07_v2_aggregate(
        (_public_result(SimpleNamespace(dimension_key="W-07-AND_OR"), True),),
        (W07V2AblationProgress(
            "W-07-AND_OR-ABLATION",
            (_public_result(SimpleNamespace(dimension_key="W-07-AND_OR"), False),),
        ),),
        family_commitment=documents.family_key,
        payload_commitment=documents.payload_commitment,
        case_commitment=documents.case_commitment,
        label_commitment=documents.label_commitment,
        cluster_commitment=documents.cluster_commitment,
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
        public_repo_writes=0,
        failure_kind="UNEXPECTED",
        cursor=cursor,
        ablation_gates_passed=False,
    )
    encoded = canonical_json_bytes(aggregate)
    assert aggregate["status"] == "NE"
    assert b"case_key" not in encoded
    assert b"message" not in encoded


def test_v2_persistent_sqlite_and_or_keeps_committed_ledger(tmp_path: Path):
    adapted = _adapted(tmp_path)
    sliced = slice_w07_adapter(adapted, W07_L02_PREFIX)
    backend = SQLiteBackend(str(tmp_path / "logic_and_or.sqlite"))
    learning = build_w07_learning_runtime(backend, sliced)

    class _DummyBackend:
        def commit(self):
            return None

        def schema_snapshot(self):
            return ()

        def close(self):
            return None

    bundles = tuple(
        _LogicBundle(
            substage,
            backend if substage == "AND_OR" else _DummyBackend(),
            sliced if substage == "AND_OR" else None,
            learning if substage == "AND_OR" else None,
        )
        for substage in W07_SUBSTAGE_ORDER
    )
    suite = W07EvaluatorConsumerSuite(bundles)
    documents = _documents()
    payload = decode_w07_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    case = next(item for item in payload.cases
                if item.dimension_key == "W-07-AND_OR")
    try:
        suite.commit()
        audit = suite.ledger_audit()[1]
        assert audit["row_count"] > 0
        assert evaluate_w07_case(
            suite, case, evaluation_ordinal=0).status == "PASS"
        assert evaluate_w07_case(
            suite,
            case,
            disabled_dimension=case.dimension_key,
            evaluation_ordinal=1,
        ).status == "FAIL"
    finally:
        suite.close()
    database = tmp_path / "logic_and_or.sqlite"
    with sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro", uri=True) as reopened:
        tables = tuple(
            row[0] for row in reopened.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
        assert tables
        assert sum(
            reopened.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in tables
        ) > 0


def test_v2_runtime_fault_publishes_exact_safe_cursor(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    contract_bytes = canonical_json_bytes({"candidate_request": {}})
    host_bytes = canonical_json_bytes({})
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    family = tmp_path / "family"
    documents = build_w07_v2_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        evaluator_public_head_commit_sha1=SYNTHETIC_HEAD,
        nonce=(41, 43, 47),
    )
    _, family_sha = publish_w07_v2_private_family(
        family, documents, forbidden_roots=(repository, candidate))
    from pure_integer_ai.experiments import ph2_w07_evaluator_v2_runtime as runtime
    monkeypatch.setattr(
        runtime,
        "_candidate_documents",
        lambda _repository, _candidate: (
            {"candidate_request": {}}, contract_bytes, {}, host_bytes),
    )
    fault = W07V2DiagnosticCursor("PAYLOAD_DECODE", "ENTER_PHASE")
    result = run_w07_v2_private_evaluation_once(
        W07V2PrivateEvaluatorRuntimeConfig(
            repository_root=repository,
            candidate_root=candidate,
            family_root=family,
            execution_root=family / "execution",
            evaluator_public_head_commit_sha1=SYNTHETIC_HEAD,
            fault_cursor=fault,
        ),
        family_freeze_sha256=family_sha,
    )
    assert result.status == "NE"
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["failure_kind"] == "INJECTED"
    assert aggregate["diagnostic_cursor"] == fault.to_safe_dict()
    assert aggregate["baseline_results"] == []
    assert result.recommendation_path is None


@pytest.mark.parametrize(
    ("fault_cursor", "expected_status"),
    (
        (None, "PASS"),
        (
            W07V2DiagnosticCursor(
                "ABLATION_CONDITION",
                "EVALUATE_CASE",
                "W-07-CONDITION-ABLATION",
                "W-07-EXISTS",
            ),
            "NE",
        ),
    ),
)
def test_v2_runtime_full_orchestration_keeps_exact_safe_progress(
        tmp_path: Path, monkeypatch, fault_cursor, expected_status):
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    contract_bytes = canonical_json_bytes({"candidate_request": {
        "base_fence_key": [1], "worker_count": 4,
    }})
    host_bytes = canonical_json_bytes({})
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    (candidate / "candidate_contract_freeze.json").write_bytes(contract_bytes)
    family = tmp_path / "family"
    docs = build_w07_v2_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        evaluator_public_head_commit_sha1=SYNTHETIC_HEAD,
        nonce=(53, 59, 61),
    )
    _, family_sha = publish_w07_v2_private_family(
        family, docs, forbidden_roots=(repository, candidate))
    dump_root = candidate / "run"
    dump_path = dump_root / f"w07_run_{W07_FORMAL_RUN_ID:020d}"
    dump_path.mkdir(parents=True)
    (dump_path / "w07_dump_manifest.json").write_bytes(b"{}")

    class _FakeSuite:
        def commit(self):
            return None

        def ledger_audit(self):
            return tuple({
                "nonempty_table_count": 1,
                "row_count": 1,
                "substage": item,
                "table_count": 1,
            } for item in W07_SUBSTAGE_ORDER)

        def audit(self):
            return {"understanding_uses": 0}

        def close(self):
            return None

    from pure_integer_ai.experiments import ph2_w07_evaluator_v2_runtime as runtime
    monkeypatch.setattr(
        runtime,
        "_candidate_documents",
        lambda _repository, _candidate: (
            {"candidate_request": {"base_fence_key": [1], "worker_count": 4}},
            contract_bytes,
            {"host_evidence": {"host_digests": {
                "logical": "l", "candidate": "c", "logic": "g",
                "source_evidence": "s", "active_projection": "a",
                "carrier_scope": "r", "transaction": "t",
            }}},
            host_bytes,
        ),
    )
    monkeypatch.setattr(runtime, "_dump_root", lambda _candidate: dump_root)
    monkeypatch.setattr(
        runtime,
        "_candidate_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime,
        "load_w07_public_dump",
        lambda _config: SimpleNamespace(
            dump_readback=1,
            logical_state_digest="l",
            candidate_digest="c",
            logic_digest="g",
            source_evidence_digest="s",
            active_projection_digest="a",
            carrier_scope_digest="r",
            transaction_digest="t",
            artifact_counts={
                "CARRIER_PROJECTION": 9,
                "LOGIC_SCOPE_CELL": 189,
                "LOGIC_USE": 21,
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_build_consumer_suite",
        lambda *_args: (_FakeSuite(), object()),
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_w07_case",
        lambda _suite, case, **kwargs: _public_result(
            case,
            kwargs.get("disabled_dimension") != case.dimension_key),
    )
    result = run_w07_v2_private_evaluation_once(
        W07V2PrivateEvaluatorRuntimeConfig(
            repository_root=repository,
            candidate_root=candidate,
            family_root=family,
            execution_root=family / "execution",
            evaluator_public_head_commit_sha1=SYNTHETIC_HEAD,
            fault_cursor=fault_cursor,
        ),
        family_freeze_sha256=family_sha,
    )
    assert result.status == expected_status
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["evaluator_version"] == 2
    assert len(aggregate["baseline_results"]) == 8
    assert aggregate["infrastructure"]["ledgers_committed"] == 1
    if fault_cursor is None:
        assert len(aggregate["ablation_results"]) == 8
        assert result.recommendation_path is not None
    else:
        assert tuple(
            len(item["dimension_results"])
            for item in aggregate["ablation_results"]
        ) == (8, 2)
        assert aggregate["failure_kind"] == "INJECTED"
        assert aggregate["diagnostic_cursor"] == fault_cursor.to_safe_dict()
        assert result.recommendation_path is None
