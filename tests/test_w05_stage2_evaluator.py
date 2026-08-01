"""W05-06 private family、五维评测、消融和 owner isolation 专项。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_adapter import adapt_w05_training_payload
from pure_integer_ai.experiments.ph2_w05_candidate import (
    W05_CANDIDATE_FORMAL_MODE,
    W05_CANDIDATE_FORMAL_WORKER_COUNT,
    build_w05_candidate_contract,
    execute_w05_candidate_once,
    publish_w05_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_PRIVATE_ABLATION_KEYS,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_STAGE_KEY,
    W05_W04_BASE_RUN_ID,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
)
from pure_integer_ai.experiments.ph2_w05_evaluator import (
    W05EvaluatorAblation,
    evaluate_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_contract import (
    W05_PRIVATE_FIRST_RUN_GUARD_NAME,
    decode_w05_private_documents,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_family import (
    build_w05_private_family_documents,
    consume_w05_private_first_run_guard,
    publish_w05_private_family,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_runtime import (
    W05PrivateEvaluatorRuntimeConfig,
    run_w05_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.experiments.ph2_w05_learning import build_w05_learning_runtime
from pure_integer_ai.experiments.ph2_w05_runtime import W05RuntimeConfig
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"


def _documents(contract_sha: str = "a" * 64, host_sha: str = "b" * 64):
    """构造只绑定 commitment 的 W-05 private family 文档。"""
    return build_w05_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(5, 7, 11, 13),
    )


def _learning(tmp_path: Path):
    """通过 W-05 public firewall 构造临时只读评估学习态。"""
    backend = SQLiteBackend(str(tmp_path / "learning.sqlite"))
    context = open_w05_frozen_context(
        ROOT,
        GLOBAL,
        current_remote_commit_sha1=HEAD,
        backend_profile_key=backend.storage_capabilities().stable_key(),
    )
    request = W05RunRequest(
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        stage_key=W05_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=HEAD,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(W05_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W05PayloadFirewall.open(ROOT, context, request).read_training_payload()
    return backend, build_w05_learning_runtime(
        backend, adapt_w05_training_payload(payload))


def _candidate_contract(tmp_path: Path):
    """按当前 SQLite capability 构造 zero-state candidate contract。"""
    backend = SQLiteBackend(str(tmp_path / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w05_candidate_contract(
        ROOT,
        global_manifest_path=GLOBAL,
        backend_profile_key=profile,
        current_remote_commit_sha1=HEAD,
    )


def test_w05_private_documents_and_five_ablations_are_orthogonal(
        tmp_path: Path,
        ):
    """baseline 五维全过；每个 bridge 消融只击穿目标维度。"""
    documents = _documents()
    payload = decode_w05_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    backend, learning = _learning(tmp_path)
    try:
        baseline = evaluate_w05_learning_runtime(learning, payload.cases)
        assert tuple(item.status for item in baseline) == ("PASS",) * 5
        for ordinal, key in enumerate(W05_PRIVATE_ABLATION_KEYS):
            values = evaluate_w05_learning_runtime(
                learning,
                payload.cases,
                ablation=W05EvaluatorAblation(key),
            )
            assert tuple(item.status for item in values) == tuple(
                "FAIL" if index == ordinal else "PASS"
                for index in range(5)
            )
            assert all(item.ne_count == 0 for item in values)
    finally:
        backend.close()


def test_w05_private_family_and_guard_are_non_overwritable(tmp_path: Path):
    """private family、phase/ablation freeze 与正式 guard 均只写一次。"""
    documents = _documents()
    root = tmp_path / "family"
    freeze, freeze_sha = publish_w05_private_family(root, documents)
    freeze_value = json.loads(freeze.read_text(encoding="utf-8"))
    assert tuple(freeze_value["ablation_order"]) == W05_PRIVATE_ABLATION_KEYS
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w05_private_family(root, documents)
    guard, _ = consume_w05_private_first_run_guard(
        root, family_freeze_sha256=freeze_sha)
    assert guard.name == W05_PRIVATE_FIRST_RUN_GUARD_NAME
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w05_private_first_run_guard(
            root, family_freeze_sha256=freeze_sha)


def test_w05_private_runtime_passes_once_with_scope_and_owner_isolation(
        tmp_path: Path,
        ):
    """一次 private run 闭合 clone/readback、九载体 scope 与零越权写。"""
    contract = _candidate_contract(tmp_path)
    candidate_root = tmp_path / "candidate"
    _, contract_sha = publish_w05_candidate_contract_freeze(
        ROOT, candidate_root, contract)
    request = contract["candidate_request"]
    candidate_config = W05RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=candidate_root / "run",
        sqlite_path=candidate_root / "coordinator.sqlite",
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=W05_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W05_CANDIDATE_FORMAL_MODE,
        current_remote_commit_sha1=HEAD,
    )
    _, _, _, host_sha, _, _ = execute_w05_candidate_once(
        ROOT,
        candidate_root,
        config=candidate_config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=candidate_root / "readback.sqlite",
    )
    family_root = tmp_path / "private_family"
    documents = build_w05_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(17, 19, 23, 29),
    )
    _, family_sha = publish_w05_private_family(
        family_root,
        documents,
        forbidden_roots=(ROOT, candidate_root),
    )
    config = W05PrivateEvaluatorRuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        candidate_root=candidate_root,
        family_root=family_root,
        execution_root=family_root / "execution",
        current_remote_commit_sha1=HEAD,
    )
    result = run_w05_private_evaluation_once(
        config, family_freeze_sha256=family_sha)
    assert result.status == "PASS"
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["infrastructure"]["carrier_projection_count"] == 9
    assert aggregate["infrastructure"]["role_proposition_scope_cell_count"] == 27
    assert aggregate["infrastructure"]["carrier_scope_digest_match"] == 1
    assert aggregate["infrastructure"]["evaluator_label_writes"] == 0
    assert aggregate["infrastructure"]["public_repo_writes"] == 0
    assert result.recommendation_path is not None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w05_private_evaluation_once(
            config, family_freeze_sha256=family_sha)


def test_w05_private_fault_seals_safe_ne_and_consumes_guard(tmp_path: Path):
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
        "execution_state": {"W05_STARTED": 1},
        "formal_run_count": 1,
        "self_excluded": 1,
    })
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    (candidate / "candidate_host_freeze.json").write_bytes(host_bytes)
    family = tmp_path / "family"
    documents = build_w05_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(31, 37, 41),
    )
    _, family_sha = publish_w05_private_family(
        family,
        documents,
        forbidden_roots=(repository, candidate),
    )
    config = W05PrivateEvaluatorRuntimeConfig(
        repository_root=repository,
        global_manifest_path=GLOBAL,
        candidate_root=candidate,
        family_root=family,
        execution_root=family / "execution",
        current_remote_commit_sha1=HEAD,
        fault_phase="PAYLOAD_DECODE",
    )
    result = run_w05_private_evaluation_once(
        config, family_freeze_sha256=family_sha)
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert result.status == "NE"
    assert aggregate["failure_phase"] == "PAYLOAD_DECODE"
    assert aggregate["dimension_results"] == []
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w05_private_evaluation_once(
            config, family_freeze_sha256=family_sha)
