"""W07-06 evaluator contract, real facade consumers, and fault sealing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_adapter import adapt_w07_training_payload
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_W06_BASE_RUN_ID,
    W07_PUBLIC_ABLATION_KEYS,
    W07RunRequest,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_evaluator import (
    W07EvaluatorAblation,
    evaluate_w07_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_consumers import (
    build_w07_evaluator_consumer_suite,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_PRIVATE_FIRST_RUN_GUARD_NAME,
    decode_w07_private_documents,
    public_safe_w07_aggregate,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_family import (
    build_w07_private_family_documents,
    consume_w07_private_first_run_guard,
    publish_w07_private_family,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_runtime import (
    W07PrivateEvaluatorRuntimeConfig,
    run_w07_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.storage.backend import DictBackend


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapted():
    backend = DictBackend()
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=backend.storage_capabilities().stable_key(),
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
        payload = W07PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
        return adapt_w07_training_payload(payload)
    finally:
        backend.close()


def _documents():
    return build_w07_private_family_documents(
        candidate_contract_sha256="a" * 64,
        candidate_host_freeze_sha256="b" * 64,
        evaluator_public_head_commit_sha1="c" * 40,
        nonce=(7, 11, 19, 31),
    )


def test_w07_private_documents_and_eight_ablations_are_orthogonal(
        adapted):
    """Baseline calls each real L01-L07 facade; one ablation fails one dimension."""
    documents = _documents()
    payload = decode_w07_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    suite = build_w07_evaluator_consumer_suite(
        ROOT, adapted, backend_factory=lambda _substage: DictBackend())
    try:
        baseline = evaluate_w07_learning_runtime(
            suite, payload.cases, evaluation_ordinal=0)
        assert tuple(item.status for item in baseline) == ("PASS",) * 8
        audit = suite.audit()
        assert audit["understanding_uses"] == 8
        assert audit["reasoning_uses"] == 8
        assert audit["generation_choices"] == 15
        assert audit["generation_uses"] == 15
        assert audit["generation_outcomes"] == 15
        assert audit["nested_understanding_layer_uses"] >= 2
        assert audit["nested_reasoning_layer_uses"] >= 2
        for ordinal, key in enumerate(W07_PUBLIC_ABLATION_KEYS):
            values = evaluate_w07_learning_runtime(
                suite,
                payload.cases,
                ablation=W07EvaluatorAblation(key),
                evaluation_ordinal=ordinal + 1,
            )
            assert tuple(item.status for item in values) == tuple(
                "FAIL" if index == ordinal else "PASS"
                for index in range(8)
            )
            assert all(item.ne_count == 0 for item in values)
    finally:
        suite.close()


def test_w07_private_family_guard_and_safe_aggregate_are_immutable(tmp_path):
    documents = _documents()
    root = tmp_path / "private-family"
    freeze, freeze_sha = publish_w07_private_family(root, documents)
    value = json.loads(freeze.read_text(encoding="utf-8"))
    assert value["formal_run_count"] == 0
    assert value["hard_requirements"]
    with pytest.raises(RuntimeError, match="immutable"):
        publish_w07_private_family(root, documents)
    guard, guard_sha = consume_w07_private_first_run_guard(
        root, family_freeze_sha256=freeze_sha)
    assert guard.name == W07_PRIVATE_FIRST_RUN_GUARD_NAME
    assert hashlib.sha256(guard.read_bytes()).hexdigest() == guard_sha
    with pytest.raises(RuntimeError, match="already consumed"):
        consume_w07_private_first_run_guard(
            root, family_freeze_sha256=freeze_sha)
    aggregate = public_safe_w07_aggregate(
        (),
        family_commitment=documents.family_key,
        payload_commitment=documents.payload_commitment,
        case_commitment=documents.case_commitment,
        label_commitment=documents.label_commitment,
        cluster_commitment=documents.cluster_commitment,
        failure_phase="PAYLOAD_DECODE",
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )
    encoded = canonical_json_bytes(aggregate)
    assert aggregate["status"] == "NE"
    assert b"case_key" not in encoded
    assert b"expected" not in encoded


def test_w07_private_fault_consumes_guard_and_seals_enum_ne(
        tmp_path, monkeypatch):
    """A preregistered phase fault is terminal for only its new test family."""
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    contract_bytes = canonical_json_bytes({"candidate_request": {}})
    host_bytes = canonical_json_bytes({})
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    family = tmp_path / "family"
    documents = build_w07_private_family_documents(
        candidate_contract_sha256=contract_sha,
        candidate_host_freeze_sha256=host_sha,
        evaluator_public_head_commit_sha1="d" * 40,
        nonce=(41, 43, 47),
    )
    _, family_sha = publish_w07_private_family(
        family, documents, forbidden_roots=(repository, candidate))
    from pure_integer_ai.experiments import ph2_w07_evaluator_runtime as runtime
    monkeypatch.setattr(
        runtime,
        "_candidate_documents",
        lambda _repository, _candidate: (
            {"candidate_request": {}}, contract_bytes, {}, host_bytes),
    )
    result = run_w07_private_evaluation_once(
        W07PrivateEvaluatorRuntimeConfig(
            repository_root=repository,
            candidate_root=candidate,
            family_root=family,
            execution_root=family / "execution",
            evaluator_public_head_commit_sha1="d" * 40,
            fault_phase="PAYLOAD_DECODE",
        ),
        family_freeze_sha256=family_sha,
    )
    assert result.status == "NE"
    aggregate = json.loads(result.aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["failure_phase"] == "PAYLOAD_DECODE"
    assert aggregate["dimension_results"] == []
