"""W06-06 private family、八维评测、消融和 owner isolation 专项。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_adapter import adapt_w06_training_payload
from pure_integer_ai.experiments.ph2_w06_candidate import (
    W06_CANDIDATE_FORMAL_MODE,
    W06_CANDIDATE_FORMAL_WORKER_COUNT,
    build_w06_candidate_contract,
    execute_w06_candidate_once,
    publish_w06_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_FORMAL_RUN_ID,
    W06_PRIVATE_ABLATION_KEYS,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_W05_BASE_RUN_ID,
    W06RunRequest,
    open_w06_frozen_context,
    validate_w06_request,
)
from pure_integer_ai.experiments.ph2_w06_evaluator import (
    W06EvaluatorAblation,
    evaluate_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_consumers import (
    build_w06_evaluator_consumer_suite,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_contract import (
    W06_PRIVATE_FIRST_RUN_GUARD_NAME,
    decode_w06_private_documents,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_family import (
    build_w06_private_family_documents,
    consume_w06_private_first_run_guard,
    publish_w06_private_family,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_runtime import (
    W06PrivateEvaluatorRuntimeConfig,
    run_w06_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w06_firewall import W06PayloadFirewall
from pure_integer_ai.experiments.ph2_w06_runtime import W06RuntimeConfig
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "2ceb8f955c81204bdc194b962911053de0133bbb"


@pytest.fixture
def external_tmp_path():
    """在公开 Git 同级目录建立并自动清理隔离 evaluator root。"""
    with tempfile.TemporaryDirectory(
            prefix="w06-evaluator-test-", dir=ROOT.parent) as value:
        yield Path(value)


def _documents(contract_sha: str = "a" * 64, host_sha: str = "b" * 64):
    return build_w06_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(6, 7, 11, 13),
    )


def _consumer_suite(root: Path):
    """经 public firewall 建立七个隔离 relation consumer owner。"""
    backend = SQLiteBackend(str(root / "learning.sqlite"))
    try:
        context = open_w06_frozen_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        request = validate_w06_request(context, W06RunRequest(
            run_id=W06_FORMAL_RUN_ID,
            parent_run_id=W06_W05_BASE_RUN_ID,
            base_run_id=W06_W05_BASE_RUN_ID,
            stage_key=W06_STAGE_KEY,
            owner_key=context.owner_key,
            runner_key=W06_RUNNER_KEY,
            current_remote_commit_sha1=HEAD,
            source_overlay_sha256=context.source_overlay_sha256,
            context_key=context.stable_key(),
            backend_profile_key=context.backend_profile_key,
            base_fence_key=context.base_fence_key,
            worker_count=1,
            mode="fresh",
            resource_budget=tuple(sorted(W06_RESOURCE_BUDGET.items())),
            candidate_payload_paths=tuple(
                item.relative_path
                for item in context.candidate_payload_bindings),
            teacher_evidence_paths=tuple(
                item.relative_path
                for item in context.teacher_evidence_bindings),
        ))
        payload = W06PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
    finally:
        backend.close()
    return build_w06_evaluator_consumer_suite(
        ROOT,
        adapt_w06_training_payload(payload),
        backend_factory=lambda _substage: DictBackend(),
    )


def _candidate_contract(root: Path):
    backend = SQLiteBackend(str(root / "profile.sqlite"))
    try:
        profile = backend.storage_capabilities().stable_key()
    finally:
        backend.close()
    return build_w06_candidate_contract(
        ROOT,
        backend_profile_key=profile,
        current_remote_commit_sha1=HEAD,
    )


def test_w06_private_documents_and_eight_ablations_are_orthogonal(
        external_tmp_path: Path):
    """baseline 八维全过；每个 bridge 消融只击穿目标维度。"""
    documents = _documents()
    payload = decode_w06_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    suite = _consumer_suite(external_tmp_path)
    try:
        baseline = evaluate_w06_learning_runtime(suite, payload.cases)
        assert tuple(item.status for item in baseline) == ("PASS",) * 8
        assert suite.audit() == {
            "generation_choices": 24,
            "generation_outcomes": 24,
            "generation_uses": 24,
            "reasoning_outcomes": 15,
            "reasoning_uses": 15,
            "understanding_outcomes": 15,
            "understanding_uses": 15,
        }
        for ordinal, key in enumerate(W06_PRIVATE_ABLATION_KEYS):
            values = evaluate_w06_learning_runtime(
                suite,
                payload.cases,
                ablation=W06EvaluatorAblation(key),
                evaluation_ordinal=ordinal + 1,
            )
            assert tuple(item.status for item in values) == tuple(
                "FAIL" if index == ordinal else "PASS"
                for index in range(8)
            )
            assert all(item.ne_count == 0 for item in values)
    finally:
        suite.close()


def test_w06_private_family_and_guard_are_non_overwritable(
        external_tmp_path: Path):
    """private family、phase/ablation freeze 与正式 guard 均只写一次。"""
    documents = _documents()
    root = external_tmp_path / "family"
    freeze, freeze_sha = publish_w06_private_family(root, documents)
    freeze_value = json.loads(freeze.read_text(encoding="utf-8"))
    assert tuple(freeze_value["ablation_order"]) == W06_PRIVATE_ABLATION_KEYS
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w06_private_family(root, documents)
    guard, _ = consume_w06_private_first_run_guard(
        root, family_freeze_sha256=freeze_sha)
    assert guard.name == W06_PRIVATE_FIRST_RUN_GUARD_NAME
    with pytest.raises(RuntimeError, match="不可重跑"):
        consume_w06_private_first_run_guard(
            root, family_freeze_sha256=freeze_sha)


def test_w06_private_runtime_passes_once_with_relation_owner_isolation(
        external_tmp_path: Path):
    """一次 private run 闭合 clone/readback、九载体 relation 与零越权写。"""
    contract = _candidate_contract(external_tmp_path)
    candidate_root = external_tmp_path / "candidate"
    _, contract_sha = publish_w06_candidate_contract_freeze(
        ROOT, candidate_root, contract)
    request = contract["candidate_request"]
    candidate_config = W06RuntimeConfig(
        repository_root=ROOT,
        run_root=candidate_root / "run",
        sqlite_path=candidate_root / "coordinator.sqlite",
        run_id=W06_FORMAL_RUN_ID,
        parent_run_id=W06_W05_BASE_RUN_ID,
        base_run_id=W06_W05_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=W06_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W06_CANDIDATE_FORMAL_MODE,
        current_remote_commit_sha1=HEAD,
    )
    _, _, _, host_sha, _, _ = execute_w06_candidate_once(
        ROOT,
        candidate_root,
        config=candidate_config,
        contract=contract,
        candidate_contract_sha256=contract_sha,
        dump_readback_sqlite_path=candidate_root / "readback.sqlite",
    )
    family_root = external_tmp_path / "private_family"
    documents = build_w06_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(17, 19, 23, 29),
    )
    _, family_sha = publish_w06_private_family(
        family_root,
        documents,
        forbidden_roots=(ROOT, candidate_root),
    )
    config = W06PrivateEvaluatorRuntimeConfig(
        repository_root=ROOT,
        candidate_root=candidate_root,
        family_root=family_root,
        execution_root=family_root / "execution",
        current_remote_commit_sha1=HEAD,
    )
    result = run_w06_private_evaluation_once(
        config, family_freeze_sha256=family_sha)
    assert result.status == "PASS"
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["pass_count"] == 8
    assert aggregate["fail_count"] == 0
    assert aggregate["ne_count"] == 0
    assert aggregate["infrastructure"]["carrier_projection_count"] == 9
    assert aggregate["infrastructure"]["relation_scope_cell_count"] == 27
    assert aggregate["infrastructure"]["carrier_scope_digest_match"] == 1
    assert aggregate["infrastructure"]["evaluator_label_writes"] == 0
    assert aggregate["infrastructure"]["public_repo_writes"] == 0
    assert result.recommendation_path is not None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w06_private_evaluation_once(
            config, family_freeze_sha256=family_sha)


def test_w06_private_fault_seals_safe_ne_and_consumes_guard(
        external_tmp_path: Path):
    """正式 phase 故障只发布枚举 NE，且同 family 不可重跑。"""
    repository = external_tmp_path / "repository"
    candidate = external_tmp_path / "candidate"
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
        "execution_state": {
            "W06_STARTED": 1,
            "formal_w06_training_runs": 1,
        },
        "formal_run_count": 1,
        "self_excluded": 1,
    })
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    (candidate / "candidate_host_freeze.json").write_bytes(host_bytes)
    family = external_tmp_path / "family"
    documents = build_w06_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        nonce=(31, 37, 41),
    )
    _, family_sha = publish_w06_private_family(
        family,
        documents,
        forbidden_roots=(repository, candidate),
    )
    config = W06PrivateEvaluatorRuntimeConfig(
        repository_root=repository,
        candidate_root=candidate,
        family_root=family,
        execution_root=family / "execution",
        current_remote_commit_sha1=HEAD,
        fault_phase="PAYLOAD_DECODE",
    )
    result = run_w06_private_evaluation_once(
        config, family_freeze_sha256=family_sha)
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert result.status == "NE"
    assert aggregate["failure_phase"] == "PAYLOAD_DECODE"
    assert aggregate["dimension_results"] == []
    assert result.recommendation_path is None
    with pytest.raises(RuntimeError, match="不可重跑"):
        run_w06_private_evaluation_once(
            config, family_freeze_sha256=family_sha)
