"""PH2 W-03 独立 private evaluator 的合同、消融与安全发布。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import ph2_w03_evaluator_runtime as evaluator_runtime
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_EVALUATION_ORDER,
)
from pure_integer_ai.experiments.ph2_w03_evaluator import (
    W03EvaluatorAblation,
    evaluate_w03_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_contract import (
    W03_EVALUATOR_PHASES,
    W03_PRIVATE_AGGREGATE_NAME,
    W03_PRIVATE_FAMILY_FREEZE_NAME,
    W03_PRIVATE_FIRST_RUN_GUARD_NAME,
    W03PrivateDimensionResult,
    W03PrivateEvaluationError,
    decode_w03_private_documents,
    public_safe_w03_aggregate,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_family import (
    build_w03_private_family_documents,
    consume_w03_private_first_run_guard,
    publish_w03_private_family,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_runtime import (
    W03EvaluatorInjectedFault,
    W03PrivateEvaluatorRuntimeConfig,
    _apply_capability_gates,
    _validate_owner_roots,
    run_w03_private_evaluation_once,
)
from tests.test_w03_stage2_understanding import _runtime


_CANDIDATE_CONTRACT_SHA = "1" * 64
_CANDIDATE_HOST_SHA = "2" * 64


def _documents(nonce: tuple[int, ...] = (20260730, 603, 1)):
    """形成 candidate 冻结后才可创建的 test-local private family 文档。"""
    return build_w03_private_family_documents(
        candidate_contract_sha256=_CANDIDATE_CONTRACT_SHA,
        candidate_host_freeze_sha256=_CANDIDATE_HOST_SHA,
        family_nonce=nonce,
    )


def test_private_documents_freeze_source_case_label_cluster_and_run_zero():
    """private family 必须先冻结五维顺序、owner、cluster 与 run-count=0。"""
    documents = _documents()
    payload = decode_w03_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )

    assert tuple(item.dimension_key for item in payload.cases) == (
        W03_EVALUATION_ORDER)
    assert tuple(item.dimension_key for item in payload.labels) == (
        W03_EVALUATION_ORDER)
    assert all(item.expected_status == "PASS" for item in payload.labels)
    assert all(item.required == 1 and item.fail_allowed == 0
               and item.ne_policy == "BLOCK" for item in payload.labels)
    assert payload.source.owner_key == "PH2_W03_PRIVATE_EVALUATOR_OWNER"
    assert payload.source.license_id == "CC0-1.0"
    assert payload.candidate_contract_sha256 == _CANDIDATE_CONTRACT_SHA
    assert payload.candidate_host_freeze_sha256 == _CANDIDATE_HOST_SHA
    assert payload.formal_run_count == 0
    assert len(payload.cluster_bindings) == 5
    assert len({item.case_key for item in payload.cases}) == 5
    assert len({item.label_key for item in payload.labels}) == 5


def test_private_document_order_extra_fields_and_label_drift_fail_closed():
    """case 顺序、字段集合或 1/1 标签被改动时必须在候选 clone 前拒绝。"""
    documents = _documents()
    cases = json.loads(documents.case_bytes.decode("utf-8"))
    cases["cases"] = list(reversed(cases["cases"]))
    with pytest.raises(W03PrivateEvaluationError, match="顺序|dimension"):
        decode_w03_private_documents(
            documents.source_bytes,
            documents.schema_bytes,
            canonical_json_bytes(cases),
            documents.label_bytes,
            documents.cluster_bytes,
        )

    labels = json.loads(documents.label_bytes.decode("utf-8"))
    labels["labels"][0]["fail_allowed"] = 1
    with pytest.raises(W03PrivateEvaluationError, match="阈值|label|1/1"):
        decode_w03_private_documents(
            documents.source_bytes,
            documents.schema_bytes,
            documents.case_bytes,
            canonical_json_bytes(labels),
            documents.cluster_bytes,
        )

    source = json.loads(documents.source_bytes.decode("utf-8"))
    source["private_surface"] = "forbidden"
    with pytest.raises(W03PrivateEvaluationError, match="字段|source"):
        decode_w03_private_documents(
            canonical_json_bytes(source),
            documents.schema_bytes,
            documents.case_bytes,
            documents.label_bytes,
            documents.cluster_bytes,
        )


def test_five_way_runtime_and_each_ablation_have_exact_bearing_effect():
    """baseline 五维全过；四 bearing 消融仅击穿目标，generation 消融硬失败。"""
    documents = _documents()
    payload = decode_w03_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    backend, understanding = _runtime()
    try:
        understanding.apply_all_evidence()
        persisted_outcomes = (
            {"use_key": [1, 2, 3], "verdict": "SUPPORT"},
            {"use_key": [1, 2, 3], "verdict": "REFUTE"},
        )
        baseline = evaluate_w03_learning_runtime(
            understanding,
            payload.cases,
            persisted_generation_outcomes=persisted_outcomes,
        )
        assert tuple(item.status for item in baseline) == ("PASS",) * 5
        assert all((item.passed, item.required, item.fail_count, item.ne_count)
                   == (1, 1, 0, 0) for item in baseline)

        for ablation_key, dimension_key in zip(
                W03_ABLATION_KEYS,
                W03_EVALUATION_ORDER[:4],
                strict=True):
            ablated = evaluate_w03_learning_runtime(
                understanding,
                payload.cases,
                persisted_generation_outcomes=persisted_outcomes,
                ablation=W03EvaluatorAblation(ablation_key),
            )
            assert tuple(item.status for item in ablated) == tuple(
                "FAIL" if item.dimension_key == dimension_key else "PASS"
                for item in ablated
            )
            assert all(item.ne_count == 0 for item in ablated)

        generation_disabled = evaluate_w03_learning_runtime(
            understanding,
            payload.cases,
            persisted_generation_outcomes=persisted_outcomes,
            sense_consumer_connected=False,
        )
        assert tuple(item.status for item in generation_disabled[:4]) == (
            "PASS",) * 4
        assert generation_disabled[4].status == "FAIL"
        assert generation_disabled[4].ne_count == 0
    finally:
        backend.close()


def test_public_aggregate_contains_only_safe_counts_commitments_and_phase():
    """安全 aggregate 不得包含 private case/label、surface、路径或异常文本。"""
    documents = _documents()
    payload = decode_w03_private_documents(
        documents.source_bytes,
        documents.schema_bytes,
        documents.case_bytes,
        documents.label_bytes,
        documents.cluster_bytes,
    )
    backend, understanding = _runtime()
    try:
        understanding.apply_all_evidence()
        results = evaluate_w03_learning_runtime(
            understanding,
            payload.cases,
            persisted_generation_outcomes=(
                {"use_key": [9], "verdict": "SUPPORT"},
                {"use_key": [9], "verdict": "REFUTE"},
            ),
        )
    finally:
        backend.close()
    aggregate = public_safe_w03_aggregate(
        results,
        family_commitment=documents.family_commitment,
        payload_commitment=documents.payload_commitment,
        case_commitment=documents.case_commitment,
        label_commitment=documents.label_commitment,
        cluster_commitment=documents.cluster_commitment,
        failure_phase="NONE",
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )
    encoded = canonical_json_bytes(aggregate)
    forbidden = (
        *[item.case_key for item in payload.cases],
        *[item.label_key for item in payload.labels],
        "surface",
        "expected",
        "private_path",
        "exception",
    )
    assert all(item.encode("utf-8") not in encoded for item in forbidden)
    assert aggregate["status"] == "PASS"
    assert aggregate["failure_phase"] == "NONE"
    assert aggregate["formal_run_count"] == 1
    assert aggregate["host_writes"] == aggregate["label_writes"] == 0
    assert aggregate["pass_count"] == 5
    assert aggregate["fail_count"] == aggregate["ne_count"] == 0


def test_private_family_and_first_run_guard_are_non_overwritable(tmp_path: Path):
    """family freeze 必须先于唯一 formal guard，二者重复 publication 均不改 SHA。"""
    documents = _documents()
    root = tmp_path / "private_family"
    freeze_path, freeze_sha = publish_w03_private_family(root, documents)
    assert freeze_path.name == W03_PRIVATE_FAMILY_FREEZE_NAME
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == freeze_sha
    before = freeze_path.read_bytes()
    with pytest.raises(RuntimeError, match="不可覆盖"):
        publish_w03_private_family(root, documents)
    assert freeze_path.read_bytes() == before

    guard_path, guard_sha = consume_w03_private_first_run_guard(
        root,
        family_freeze_sha256=freeze_sha,
    )
    assert guard_path.name == W03_PRIVATE_FIRST_RUN_GUARD_NAME
    assert hashlib.sha256(guard_path.read_bytes()).hexdigest() == guard_sha
    guard_before = guard_path.read_bytes()
    with pytest.raises(RuntimeError, match="已经消费|不可重跑"):
        consume_w03_private_first_run_guard(
            root,
            family_freeze_sha256=freeze_sha,
        )
    assert guard_path.read_bytes() == guard_before


def test_phase_and_fault_registry_are_complete_and_do_not_encode_messages():
    """正式 failure phase 只允许冻结枚举，不允许动态异常文本进入报告。"""
    assert W03_EVALUATOR_PHASES == (
        "PAYLOAD_DECODE",
        "CLONE_LOAD",
        "HOST_COPY",
        "CLONE_COMPARE",
        "BASELINE",
        "ABLATION_CONCEPT_SPLIT",
        "ABLATION_POLYSEMY_COMPETITION",
        "ABLATION_SOURCE_CONFLICT",
        "ABLATION_SUPERSEDE",
        "GENERATION",
        "INTEGRITY",
        "REPORT_SAFETY",
    )
    with pytest.raises(W03PrivateEvaluationError, match="failure phase"):
        public_safe_w03_aggregate(
            (),
            family_commitment="3" * 64,
            payload_commitment="4" * 64,
            case_commitment="5" * 64,
            label_commitment="6" * 64,
            cluster_commitment="7" * 64,
            failure_phase="secret exception message",
            formal_run_count=1,
            host_writes=0,
            label_writes=0,
        )


def test_capability_gate_fail_is_distinct_from_infrastructure_ne():
    """消融不承重是 capability FAIL；冻结 phase 异常才是 infrastructure NE。"""
    baseline = tuple(
        W03PrivateDimensionResult(
            dimension,
            "PASS",
            1,
            1,
            0,
            0,
            str(ordinal) * 64,
        )
        for ordinal, dimension in enumerate(W03_EVALUATION_ORDER, start=1)
    )
    capability = _apply_capability_gates(
        baseline, (True, False, True, True, True))
    capability_aggregate = public_safe_w03_aggregate(
        capability,
        family_commitment="6" * 64,
        payload_commitment="7" * 64,
        case_commitment="8" * 64,
        label_commitment="9" * 64,
        cluster_commitment="a" * 64,
        failure_phase="NONE",
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )
    infrastructure_aggregate = public_safe_w03_aggregate(
        (),
        family_commitment="6" * 64,
        payload_commitment="7" * 64,
        case_commitment="8" * 64,
        label_commitment="9" * 64,
        cluster_commitment="a" * 64,
        failure_phase="BASELINE",
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )

    assert capability_aggregate["status"] == "FAIL"
    assert capability_aggregate["fail_count"] == 1
    assert capability_aggregate["ne_count"] == 0
    assert capability_aggregate["failure_phase"] == "NONE"
    assert infrastructure_aggregate["status"] == "NE"
    assert infrastructure_aggregate["fail_count"] == 0
    assert infrastructure_aggregate["ne_count"] == 1
    assert infrastructure_aggregate["dimension_results"] == []


def test_runtime_roots_require_execution_strictly_inside_family(tmp_path: Path):
    """execution 只属于 private family，不能落入 repo、W-02 或 candidate。"""
    repository = (tmp_path / "repository").resolve()
    w02_root = (tmp_path / "w02").resolve()
    candidate_root = (tmp_path / "candidate").resolve()
    family_root = (tmp_path / "family").resolve()
    _validate_owner_roots(
        repository,
        w02_root,
        candidate_root,
        family_root,
        family_root / "execution",
    )

    for bad_execution in (
            family_root,
            repository / "execution",
            w02_root / "execution",
            candidate_root / "execution"):
        with pytest.raises(evaluator_runtime.W03EvaluatorInfrastructureError,
                           match="root|隔离"):
            _validate_owner_roots(
                repository,
                w02_root,
                candidate_root,
                family_root,
                bad_execution,
            )
    with pytest.raises(evaluator_runtime.W03EvaluatorInfrastructureError,
                       match="root|隔离"):
        _validate_owner_roots(
            repository,
            w02_root,
            candidate_root,
            repository / "private_family",
            repository / "private_family" / "execution",
        )


def test_payload_phase_fault_consumes_guard_and_publishes_safe_ne(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """private decode phase 已进入正式 run 时，故障只发布枚举 NE 且不可重跑。"""
    documents = _documents((20260730, 603, 2))
    family_root = tmp_path / "family"
    _, freeze_sha = publish_w03_private_family(family_root, documents)
    monkeypatch.setattr(
        evaluator_runtime,
        "_verify_candidate",
        lambda *args, **kwargs: ({}, {}, ()),
    )
    config = W03PrivateEvaluatorRuntimeConfig(
        repository_root=tmp_path / "repository",
        w02_artifacts_root=tmp_path / "w02",
        candidate_root=tmp_path / "candidate",
        family_root=family_root,
        execution_root=family_root / "execution",
        fault_phase="PAYLOAD_DECODE",
    )

    with pytest.raises(W03EvaluatorInjectedFault):
        run_w03_private_evaluation_once(
            config, family_freeze_sha256=freeze_sha)

    guard = family_root / W03_PRIVATE_FIRST_RUN_GUARD_NAME
    aggregate_path = family_root / "publication" / W03_PRIVATE_AGGREGATE_NAME
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    encoded = aggregate_path.read_bytes()
    assert guard.is_file()
    assert aggregate["status"] == "NE"
    assert aggregate["failure_phase"] == "PAYLOAD_DECODE"
    assert aggregate["fail_count"] == 0
    assert aggregate["ne_count"] == 1
    assert aggregate["dimension_results"] == []
    assert b"synthetic phase fault" not in encoded
    with pytest.raises(RuntimeError, match="已经消费|不可重跑"):
        run_w03_private_evaluation_once(
            config, family_freeze_sha256=freeze_sha)
