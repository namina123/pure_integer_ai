"""W09-10 private adapter 的 public train-only 合同预检。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_evaluator import (
    W09_CONTENT_FAILURE_KINDS,
    W09PrivateEvaluationPair,
    W09EvaluatorSnapshot,
    assess_w09_orthogonal_ablations,
    assess_w09_private_j_lc,
    assess_w09_private_open_generation,
    assess_w09_private_resource,
    assess_w09_private_rollback,
    assess_w09_private_v06,
    assess_w09_private_windows,
    evaluate_w09_private_pairs,
    summarize_w09_content_failures,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_contract import (
    public_safe_w09_aggregate,
    validate_w09_safe_report,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09CandidateInferenceAdapter,
    compile_w09_inference_state,
)
from pure_integer_ai.experiments.ph2_w09_rotation import build_w09_rotation_records
from pure_integer_ai.experiments.ph2_w09_inference import W09InferenceOutcome


def _sha(value: object) -> str:
    """为 synthetic metadata 生成稳定 commitment。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class _Preflight:
    """保存一次 train-only rotation 预检的安全结果。"""

    dimensions: tuple[Any, ...]
    ablations: tuple[dict[str, object], ...]
    open_generation: dict[str, object]
    j_lc: dict[str, object]
    windows: tuple[dict[str, object], ...]
    v06: dict[str, object]
    rollback: dict[str, object]
    resource: dict[str, object]
    aggregate: dict[str, object]
    unseen_selector_dimensions: tuple[Any, ...]
    content_failure_summary: tuple[dict[str, object], ...]
    synthetic_failure_summary: tuple[dict[str, object], ...]


@pytest.fixture(scope="module")
def public_preflight() -> _Preflight:
    """只从公开 W09 train payload 构造 rotation 并完成五维预检。"""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    context = open_w09_frozen_contract(root)
    payload = W09PayloadFirewall.open(
        root, context, make_w09_request(context)
    ).read_training_payload()
    records = build_w09_rotation_records(payload)
    state = compile_w09_inference_state(payload)
    adapter = W09CandidateInferenceAdapter(state)
    snapshot = W09EvaluatorSnapshot(
        dimension_artifact_keys=tuple((index + 1,) for index in range(len(W09_DIMENSION_KEYS))),
        use_cells=tuple(
            (dimension, consumer, "PUBLIC_BOUNDED_PASS")
            for dimension in W09_DIMENSION_KEYS
            for consumer in W09_CONSUMER_KEYS
        ),
        hard_conjunct_states=(
            tuple((dimension, "PUBLIC_BOUNDED_PASS") for dimension in W09_DIMENSION_KEYS)
            + tuple((f"WINDOW-{ordinal}", "PUBLIC_BOUNDED_PASS") for ordinal in range(1, 4))
            + (("J-LC-W09", "PUBLIC_BOUNDED_PASS"), ("V-06-CLONE", "PUBLIC_BOUNDED_PASS"), ("ROLLBACK-AUDIT", "PUBLIC_BOUNDED_PASS"))
        ),
        semantic_state_key=(1,),
        dump_manifest_sha256=_sha("dump"),
        inference_state_key=state.state_key,
        inference_state_sha256=state.sha256(),
        inference_interface_version=state.interface_version,
        inference_rule_count=len(state.rules),
        logical_shard_count=16,
        learning_event_count=27,
        future_payload_reads=0,
        evaluator_label_reads=0,
        host_learning_writes=0,
        memory_learning_writes=0,
        teacher_calls=0,
        api_calls=0,
        llm_calls=0,
        host_write_count=0,
        resource_counts=tuple((key, 0) for key in sorted(W09_RESOURCE_BUDGET)),
    )
    pairs = tuple(
        W09PrivateEvaluationPair("SYNTHETIC-ROTATION", observation, label, "ROTATION")
        for observation, label in zip(records.observations, records.labels)
    )

    def infer(disabled: tuple[str, ...] = ()) -> tuple[Any, ...]:
        """执行一个不读 label 的五维 inference family。"""
        return tuple(
            adapter.infer(observation, dimension_key=dimension, disabled_components=disabled)
            for observation in records.observations
            for dimension in W09_DIMENSION_KEYS
        )

    baseline = infer()
    content_failure_summary = summarize_w09_content_failures(
        snapshot, pairs, case_outcomes=baseline,
    )
    synthetic_failures = list(baseline)
    first_expected_state = pairs[0].label.expected_state
    mismatched_state = next(
        state for state in ("TRUE", "FALSE", "CONFLICT", "UNKNOWN")
        if state != first_expected_state
    )
    for index in range(len(W09_DIMENSION_KEYS)):
        synthetic_failures[index] = replace(
            synthetic_failures[index], actual_state=mismatched_state,
        )
    for index in range(len(W09_DIMENSION_KEYS), 2 * len(W09_DIMENSION_KEYS)):
        synthetic_failures[index] = replace(
            synthetic_failures[index],
            actual_payload=CanonicalJsonObject.from_value({"accepted": 2}),
        )
    synthetic_failure_summary = summarize_w09_content_failures(
        snapshot, pairs, case_outcomes=tuple(synthetic_failures),
    )
    unseen_observations = tuple(
        replace(
            observation,
            sample_role=("support" if observation.sample_role == "supersede" else "supersede"),
        )
        for observation in records.observations
    )
    unseen_baseline = tuple(
        adapter.infer(observation, dimension_key=dimension)
        for observation in unseen_observations
        for dimension in W09_DIMENSION_KEYS
    )
    families = []
    for index, ablation in enumerate(W09_ABLATION_KEYS):
        values = infer((W09_DIMENSION_KEYS[index],)) if index < len(W09_DIMENSION_KEYS) else ()
        families.append((ablation, values))
    dimensions = evaluate_w09_private_pairs(snapshot, pairs, case_outcomes=baseline)
    ablations = tuple(
        assess_w09_orthogonal_ablations(snapshot, pairs, outcome_families=tuple(families))
    )
    open_generation = assess_w09_private_open_generation(snapshot, pairs, case_outcomes=baseline)
    j_lc = assess_w09_private_j_lc(snapshot, pairs, case_outcomes=baseline)
    windows = tuple(assess_w09_private_windows(snapshot))
    v06 = assess_w09_private_v06(
        snapshot,
        independent_probe_count=len(pairs),
        improved_probe_count=len(pairs),
        isolated_learning_write_count=snapshot.learning_event_count,
    )
    rollback = assess_w09_private_rollback(
        snapshot, invalidated_count=3, preserved_count=snapshot.learning_event_count, leaked_write_count=0
    )
    resource = assess_w09_private_resource(snapshot)
    aggregate = public_safe_w09_aggregate(
        dimensions,
        family_commitment=_sha("family"),
        payload_commitment=_sha("payload"),
        case_commitment=_sha("case"),
        label_commitment=_sha("label"),
        cluster_commitment=_sha("cluster"),
        rotation_package_commitment=_sha("rotation"),
        failure_phase="NONE",
        formal_run_count=1,
        write_counts={
            "candidate_writes": 0,
            "label_writes": 0,
            "public_writes": 0,
            "host_writes": 0,
            "core_writes": 0,
            "evidence_writes": 0,
            "use_writes": 0,
            "memory_writes": 0,
            "assessment_writes": 0,
            "clock_writes": 0,
        },
        ablation_results=list(ablations),
        windows=list(windows),
        open_generation=open_generation,
        j_lc=j_lc,
        v06=v06,
        rollback=rollback,
        resource=resource,
    )
    unseen_dimensions = evaluate_w09_private_pairs(
        snapshot,
        tuple(
            W09PrivateEvaluationPair("SYNTHETIC-UNSEEN", observation, label, "ROTATION")
            for observation, label in zip(unseen_observations, records.labels)
        ),
        case_outcomes=unseen_baseline,
    )
    return _Preflight(
        dimensions,
        ablations,
        open_generation,
        j_lc,
        windows,
        v06,
        rollback,
        resource,
        aggregate,
        unseen_dimensions,
        content_failure_summary,
        synthetic_failure_summary,
    )


def test_public_rotation_closes_all_bearing_dimensions(public_preflight: _Preflight) -> None:
    """公开 rotation 预检必须逐项得到 309/309 PASS。"""
    assert len(public_preflight.dimensions) == len(W09_DIMENSION_KEYS)
    assert all(item.status == "PASS" and item.passed_count == 309 for item in public_preflight.dimensions)


def test_unseen_selector_uses_typed_semantics(public_preflight: _Preflight) -> None:
    """改变 selector 元数据后仍须按 typed 结构完成五维验证。"""
    assert all(item.status == "PASS" and item.passed_count == 309 for item in public_preflight.unseen_selector_dimensions)


def test_content_failure_summary_is_empty_for_public_pass(public_preflight: _Preflight) -> None:
    """全 PASS rotation 不得发布虚假失败分组。"""
    assert public_preflight.content_failure_summary == ()


def test_content_failure_summary_only_publishes_enumerated_counts(public_preflight: _Preflight) -> None:
    """失败遥测只允许 schema 枚举和五维重复计数。"""
    summary = public_preflight.synthetic_failure_summary
    validate_w09_safe_report(summary)
    assert {item["failure_kind"] for item in summary} == {
        "STATE_MISMATCH", "PAYLOAD_MISMATCH",
    }
    assert sum(item["count"] for item in summary if item["failure_kind"] == "STATE_MISMATCH") == len(W09_DIMENSION_KEYS)
    assert sum(item["count"] for item in summary if item["failure_kind"] == "PAYLOAD_MISMATCH") == len(W09_DIMENSION_KEYS)
    assert all(
        set(item) == {
            "count", "failure_kind", "family_kind", "payload_kind",
            "perturbation_kind",
        }
        and item["failure_kind"] in W09_CONTENT_FAILURE_KINDS
        and item["family_kind"] == "ROTATION"
        and type(item["count"]) is int
        and item["count"] > 0
        for item in summary
    )
    serialized_keys = {
        str(key).lower()
        for item in summary
        for key in item
    }
    assert serialized_keys.isdisjoint({
        "case", "case_key", "expected", "expected_payload", "expected_state",
        "label", "label_key", "message", "path", "private_path",
        "relative_path", "surface", "surface_form", "text", "typed_payload",
        "raw_text", "raw_observation", "observed_surface", "exception",
        "error_message",
    })


def test_train_kind_accepts_new_typed_shape_without_answer_input() -> None:
    """同一登记 kind 的新 typed 形状不得因 shape 漂移越过 data-only 推理。"""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    context = open_w09_frozen_contract(root)
    payload = W09PayloadFirewall.open(
        root, context, make_w09_request(context)
    ).read_training_payload()
    records = build_w09_rotation_records(payload)
    state = compile_w09_inference_state(payload)
    adapter = W09CandidateInferenceAdapter(state)
    original = records.observations[0]
    typed = original.typed_payload.to_value()
    typed["unseen_typed_shape_marker"] = 1
    altered = replace(
        original,
        typed_payload=original.typed_payload.from_value(typed),
    )
    outcome = adapter.infer(altered, dimension_key=W09_DIMENSION_KEYS[0])
    assert isinstance(outcome, W09InferenceOutcome)
    assert outcome.observation_key == tuple(altered.stable_key.components)


def test_held_out_selector_collision_uses_typed_projection() -> None:
    """held-out 结构变化不得复用同 selector 的 train 状态。"""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    context = open_w09_frozen_contract(root)
    payload = W09PayloadFirewall.open(
        root, context, make_w09_request(context)
    ).read_training_payload()
    observations = build_w09_rotation_records(payload).observations
    original = next(
        item for item in observations
        if item.payload_kind == "FreeTextHierarchyRecallObservationV1"
        and item.perturbation_kind == "PARAGRAPH_MOVE"
    )
    typed = original.typed_payload.to_value()
    typed["phenomena"] = ["UNKNOWN"]
    altered = replace(
        original,
        typed_payload=original.typed_payload.from_value(typed),
    )
    adapter = W09CandidateInferenceAdapter(compile_w09_inference_state(payload))
    outcome = adapter.infer(altered, dimension_key=W09_DIMENSION_KEYS[0])
    assert outcome.actual_state == "UNKNOWN"


def test_public_rotation_ablations_are_orthogonal_and_walls_stay_ne(public_preflight: _Preflight) -> None:
    """五个承重消融各自击穿目标，两个墙消融保持 NE。"""
    assert [item["status"] for item in public_preflight.ablations] == ["PASS"] * 5 + ["NE", "NE"]
    for item in public_preflight.ablations[:5]:
        assert item["dimension_statuses"].count("FAIL") == 1


def test_public_aggregate_requires_open_generation_and_jlc(public_preflight: _Preflight) -> None:
    """aggregate 必须把开放生成与 J-LC 纳入正式硬合取。"""
    assert public_preflight.open_generation["status"] == "PASS"
    assert public_preflight.j_lc["status"] == "PASS"
    assert [item["status"] for item in public_preflight.windows] == ["PASS", "PASS", "PASS"]
    assert public_preflight.v06["status"] == "PASS"
    assert public_preflight.rollback["status"] == "PASS"
    assert public_preflight.resource["status"] == "PASS"
    assert public_preflight.aggregate["status"] == "PASS"
    assert public_preflight.aggregate["pre_wean_language_learning_capability_evidenced"] == 1
    assert public_preflight.aggregate["open_generation"]["status"] == "PASS"
