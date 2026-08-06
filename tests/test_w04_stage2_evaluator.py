"""W04-06 private family、五维评测、消融和 owner isolation 专项。"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

import pure_integer_ai.experiments.ph2_w04_candidate as candidate_owner
import pure_integer_ai.experiments.ph2_w04_runtime as runtime_owner
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w04_adapter import adapt_w04_training_payload
from pure_integer_ai.experiments.ph2_w04_candidate import (
    build_w04_candidate_contract,
    execute_w04_candidate_once,
    publish_w04_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_FORMAL_RUN_ID,
    W04_RESOURCE_BUDGET,
    W04_RUNNER_KEY,
    W04_STAGE_KEY,
    W04_W03_BASE_RUN_ID,
    W04RunRequest,
)
from pure_integer_ai.experiments.ph2_w04_evaluator import (
    W04EvaluatorAblation,
    evaluate_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_contract import (
    W04_PRIVATE_ABLATION_KEYS,
    W04_PRIVATE_FIRST_RUN_GUARD_NAME,
    decode_w04_private_documents,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_family import (
    build_w04_private_family_documents,
    consume_w04_private_first_run_guard,
    publish_w04_private_family,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_runtime import (
    W04PrivateEvaluatorRuntimeConfig,
    run_w04_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w04_firewall import W04PayloadFirewall
from pure_integer_ai.experiments.ph2_w04_learning import (
    build_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_runtime import W04RuntimeConfig
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w04_historical_context import open_historical_w04_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "da69958c1f149a2f264053f7b7407a53f575cd93"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"


@pytest.fixture(autouse=True)
def _historical_candidate_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选编排只消费冻结上下文，不改生产 authority 的 fail-closed。"""
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


def _documents(contract_sha="a" * 64, host_sha="b" * 64):
    return build_w04_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(8, 4, 2, 1),
    )


def _learning(tmp_path: Path):
    backend = SQLiteBackend(str(tmp_path / "learning.sqlite"))
    context = open_historical_w04_context(
        ROOT,
        GLOBAL,
        current_remote_commit_sha1=HEAD,
        backend_profile_key=backend.storage_capabilities().stable_key(),
    )
    request = W04RunRequest(
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        stage_key=W04_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W04_RUNNER_KEY,
        current_remote_commit_sha1=HEAD,
        pre_w04_gate_key=context.pre_w04_gate_key,
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(W04_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W04PayloadFirewall.open(
        ROOT, context, request).read_training_payload()
    learning = build_w04_learning_runtime(
        backend, adapt_w04_training_payload(payload))
    return backend, learning


def _candidate_contract(tmp_path: Path):
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


def test_w04_private_documents_and_five_ablations_are_orthogonal(tmp_path: Path):
    """baseline 五维全过；每个消融只击穿目标维度。"""
    documents = _documents()
    payload = decode_w04_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    backend, learning = _learning(tmp_path)
    try:
        baseline = evaluate_w04_learning_runtime(learning, payload.cases)
        assert tuple(item.status for item in baseline) == ("PASS",) * 5
        for ordinal, key in enumerate(W04_PRIVATE_ABLATION_KEYS):
            values = evaluate_w04_learning_runtime(
                learning,
                payload.cases,
                ablation=W04EvaluatorAblation(key),
            )
            assert tuple(item.status for item in values) == tuple(
                "FAIL" if index == ordinal else "PASS"
                for index in range(5)
            )
            assert all(item.ne_count == 0 for item in values)
    finally:
        backend.close()


def test_w04_private_family_and_guard_are_non_overwritable(tmp_path: Path):
    """private family 与正式 guard 都只能排他写一次。"""
    documents = _documents()
    root = tmp_path / "family"
    _, freeze_sha = publish_w04_private_family(root, documents)
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w04_private_family(root, documents)
    guard, _ = consume_w04_private_first_run_guard(
        root, family_freeze_sha256=freeze_sha)
    assert guard.name == W04_PRIVATE_FIRST_RUN_GUARD_NAME
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w04_private_first_run_guard(
            root, family_freeze_sha256=freeze_sha)


def test_w04_private_runtime_seals_ne_once_without_host_or_label_writes(
        tmp_path: Path,
        ):
    """历史 gate 漂移后的 private 重跑只能封存安全 NE，且仍不可重跑。"""
    contract = _candidate_contract(tmp_path)
    candidate_root = tmp_path / "candidate"
    _, contract_sha = publish_w04_candidate_contract_freeze(
        ROOT, candidate_root, contract)
    request = contract["candidate_request"]
    candidate_config = W04RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=candidate_root / "run",
        sqlite_path=candidate_root / "host.sqlite",
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=4,
        mode="fresh",
        current_remote_commit_sha1=HEAD,
    )
    _, _, _, host_sha, _, _ = execute_w04_candidate_once(
        ROOT,
        candidate_root,
        config=candidate_config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=candidate_root / "readback.sqlite",
    )
    family_root = tmp_path / "private_family"
    documents = build_w04_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(9, 7, 5, 3),
    )
    _, family_sha = publish_w04_private_family(
        family_root,
        documents,
        forbidden_roots=(ROOT, candidate_root),
    )
    result = run_w04_private_evaluation_once(
        W04PrivateEvaluatorRuntimeConfig(
            repository_root=ROOT,
            global_manifest_path=GLOBAL,
            candidate_root=candidate_root,
            family_root=family_root,
            execution_root=family_root / "execution",
            current_remote_commit_sha1=HEAD,
        ),
        family_freeze_sha256=family_sha,
    )
    assert result.status == "NE"
    assert result.aggregate_path.is_file()
    aggregate = json.loads(result.aggregate_path.read_text("utf-8"))
    assert aggregate["failure_phase"] == "BASELINE"
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w04_private_evaluation_once(
            W04PrivateEvaluatorRuntimeConfig(
                repository_root=ROOT,
                global_manifest_path=GLOBAL,
                candidate_root=candidate_root,
                family_root=family_root,
                execution_root=family_root / "execution-2",
                current_remote_commit_sha1=HEAD,
            ),
            family_freeze_sha256=family_sha,
        )


def test_w04_private_fault_seals_safe_ne_and_consumes_guard(tmp_path: Path):
    """正式 phase 故障只发布枚举 NE，且同 family 不可重跑。"""
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    contract_bytes = canonical_json_bytes({
        "candidate_request": {
            "base_fence_key": [1],
            "worker_count": 4,
        },
    })
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    (candidate / "candidate_contract_freeze.json").write_bytes(contract_bytes)
    host_bytes = canonical_json_bytes({
        "candidate_contract_sha256": contract_sha,
        "execution_state": {"W04_STARTED": 1},
        "formal_run_count": 1,
        "self_excluded": 1,
    })
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    (candidate / "candidate_host_freeze.json").write_bytes(host_bytes)
    family = tmp_path / "family"
    documents = build_w04_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(6, 4, 2),
    )
    _, family_sha = publish_w04_private_family(
        family,
        documents,
        forbidden_roots=(repository, candidate),
    )
    result = run_w04_private_evaluation_once(
        W04PrivateEvaluatorRuntimeConfig(
            repository_root=repository,
            global_manifest_path=GLOBAL,
            candidate_root=candidate,
            family_root=family,
            execution_root=family / "execution",
            current_remote_commit_sha1=HEAD,
            fault_phase="PAYLOAD_DECODE",
        ),
        family_freeze_sha256=family_sha,
    )
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert result.status == "NE"
    assert aggregate["failure_phase"] == "PAYLOAD_DECODE"
    assert aggregate["dimension_results"] == []
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w04_private_evaluation_once(
            W04PrivateEvaluatorRuntimeConfig(
                repository_root=repository,
                global_manifest_path=GLOBAL,
                candidate_root=candidate,
                family_root=family,
                execution_root=family / "execution-2",
                current_remote_commit_sha1=HEAD,
            ),
            family_freeze_sha256=family_sha,
        )
