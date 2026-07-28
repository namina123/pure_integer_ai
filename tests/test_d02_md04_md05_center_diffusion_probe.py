"""MD-04/05 四基线、K-04 规模 probe、独立 evaluator 和 artifact T0。"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    MD_BASELINE_KEYS,
    MD_SAMPLE_GROUP_KEYS,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
)
from pure_integer_ai.experiments.ph2_md04_probe_contract import (
    MD04_ABLATION_KEYS,
    MD04ProbeContractError,
    MD04ProbePlan,
    MD04ProbeRunArtifact,
    MD05DecisionArtifact,
    ProbeCaseOutcome,
    ProbeMemoryCandidate,
    read_md04_probe_plan,
    read_md04_probe_runs,
    read_md05_decision,
    write_immutable_artifact,
)
from pure_integer_ai.experiments.ph2_md04_probe_fixture import (
    build_md04_fixture_bundle,
)
from pure_integer_ai.experiments import ph2_md04_probe_runtime
from pure_integer_ai.experiments.ph2_md04_probe_runtime import run_md04_probe
from pure_integer_ai.experiments.ph2_md05_probe_evaluator import (
    build_md05_labels,
    evaluate_md05_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MD03_PATH = REPO_ROOT / "data/ph2/manifests/md03_directional_center_adapter_v1.json"
BASELINE_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v21.json"
PLAN_PATH = REPO_ROOT / "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json"
RUN_PATH = REPO_ROOT / "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json"
DECISION_PATH = REPO_ROOT / "data/ph2/manifests/md05_center_diffusion_decision_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _bundle():
    return build_md04_fixture_bundle(
        md03_manifest_sha256=_sha(MD03_PATH),
        baseline_manifest_sha256=_sha(BASELINE_PATH),
    )


@lru_cache(maxsize=1)
def _runs() -> MD04ProbeRunArtifact:
    return run_md04_probe(
        _bundle(),
        plan_relative_path="data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
        plan_sha256=_sha(PLAN_PATH),
    )


@lru_cache(maxsize=1)
def _decision() -> MD05DecisionArtifact:
    runs = _runs()
    return evaluate_md05_probe(
        _bundle().plan,
        runs,
        plan_relative_path="data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
        plan_sha256=_sha(PLAN_PATH),
        run_relative_path="data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
        run_sha256=hashlib.sha256(runs.canonical_bytes()).hexdigest(),
    )


def _primary_outcomes() -> tuple[ProbeCaseOutcome, ...]:
    return tuple(
        item for item in _runs().strategy_outcomes
        if item.strategy_key == "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP")


def test_plan_is_result_blind_complete_and_bound_to_md03_v21():
    """首次运行 plan 列全四基线/十组/五消融/规模且不藏 expected。"""
    plan = _bundle().plan
    assert read_md04_probe_plan(PLAN_PATH) == plan
    assert plan.strategy_keys == MD_BASELINE_KEYS
    assert plan.sample_group_keys == MD_SAMPLE_GROUP_KEYS
    assert plan.ablation_keys == MD04_ABLATION_KEYS
    assert plan.scale_factors == (1, 10, 100)
    assert {item.sample_group_key for item in plan.cases} == set(
        MD_SAMPLE_GROUP_KEYS)
    assert len(plan.cases) == 12
    assert plan.md03_manifest_sha256 == _sha(MD03_PATH)
    assert plan.baseline_manifest_sha256 == _sha(BASELINE_PATH)
    assert plan.results_observed == 0
    payload = plan.canonical_bytes()
    assert b"expected" not in payload
    assert b"correct_candidate" not in payload
    assert MD04ProbePlan.from_dict(plan.to_dict()) == plan


def test_fixture_binds_real_md03_centers_and_no_learning_write():
    """每个引用都绑定真实 DirectionalMemoryCenter，activation 仍不授权采用。"""
    bundle = _bundle()
    for case, binding in zip(bundle.plan.cases, bundle.bindings):
        assert binding.case_key == case.case_key
        assert all(isinstance(item, DirectionalMemoryCenter)
                   for item in binding.centers)
        for center in binding.centers:
            assert center.center.activation_only == 1
            assert center.write_boundary.activation_authorizes_adoption == 0
            assert center.write_boundary.host_learning_write_count == 0
    assert all(value == 0
               for value in bundle.plan.execution_state.to_value().values())


def test_plan_and_candidate_contracts_fail_closed_and_nonoverwrite(tmp_path):
    """坏 hash、层级/通道漂移、伪结果和覆盖都在执行前拒绝。"""
    plan = _bundle().plan
    output = tmp_path / "plan.json"
    write_immutable_artifact(plan, output)
    write_immutable_artifact(plan, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(MD04ProbeContractError, match="内容不同"):
        write_immutable_artifact(plan, output)
    with pytest.raises(MD04ProbeContractError, match="不得携带结果"):
        replace(plan, results_observed=1)
    with pytest.raises(MD04ProbeContractError, match="SHA-256"):
        replace(plan, md03_manifest_sha256="bad")
    candidate = plan.cases[0].cold_candidates[0]
    assert ProbeMemoryCandidate.from_dict(candidate.to_dict()) == candidate
    with pytest.raises(MD04ProbeContractError, match="placement/channel"):
        replace(candidate, placement="HOT")


def test_four_strategies_use_identical_fixture_and_primary_stops_exactly():
    """四基线逐 case 对拍；主策略精确区分三阻断、冲突和 unknown。"""
    runs = _runs()
    case_sets = {
        strategy: tuple(
            item.case_key for item in runs.strategy_outcomes
            if item.strategy_key == strategy)
        for strategy in MD_BASELINE_KEYS
    }
    assert len(set(case_sets.values())) == 1
    status = {
        (item.case_key.components[-1], decision.center_key.components[-1]):
            decision.status
        for item in _primary_outcomes()
        for decision in item.stop_decisions
    }
    assert status[(1, 1)] == "ACCESS_BLOCKED"
    assert status[(2, 1)] == "BUDGET_EXHAUSTED"
    assert status[(3, 1)] == "GROUNDING_BLOCKED"
    assert status[(7, 1)] == "CLARIFY"
    assert status[(12, 1)] == "UNKNOWN"
    assert all(value == "RESOLVED" for key, value in status.items()
               if key not in {(1, 1), (2, 1), (3, 1), (6, 1),
                              (7, 1), (12, 1)})
    assert status[(6, 1)] == "GROUNDING_BLOCKED"


def test_k04_receipts_share_physical_read_and_keep_hot_set_bounded():
    """多中心共享一次物理读取但 receipt/decision 身份不合并。"""
    outcome = next(item for item in _primary_outcomes()
                   if item.case_key.components[-1] == 11)
    cold = tuple(item for item in outcome.receipt_records
                 if item.channel_key == "L4_SEALED_PAGE")
    assert len(cold) == 3
    assert len({item.center_key for item in cold}) == 3
    assert len({item.physical_read_key for item in cold}) == 1
    assert all(item.page_read_count == 1 for item in cold)
    assert len({item.stop_decision_key for item in cold}) == 3
    metrics = outcome.query_metrics.to_value()
    assert metrics["segment_reads"] == 1
    assert metrics["page_in_records"] == 3
    assert metrics["peak_hot_objects"] <= 4
    assert outcome.audit_values.to_value()["reader_epoch_leak_count"] == 0


def test_unrelated_1x_10x_100x_does_not_grow_reads_or_page_in():
    """相关候选固定时，无关总量扩大不增加 cold bytes/page-in/peak/segment read。"""
    grouped = {}
    for outcome in _runs().scale_outcomes:
        grouped.setdefault(outcome.case_key.components[-1], []).append(outcome)
    for case_ordinal, outcomes in grouped.items():
        ordered = sorted(outcomes, key=lambda item: item.scale_factor)
        assert [item.scale_factor for item in ordered] == [1, 10, 100]
        for key in (
                "cold_read_bytes", "page_in_records", "peak_hot_objects",
                "segment_reads"):
            assert len({item.query_metrics.to_value()[key]
                        for item in ordered}) == 1
        assert all(item.audit_values.to_value()["reader_epoch_leak_count"] == 0
                   for item in ordered)
        if case_ordinal == 9:
            assert all(item.query_metrics.to_value()["cold_read_bytes"] == 0
                       for item in ordered)
            assert all(item.query_metrics.to_value()["segment_reads"] == 0
                       for item in ordered)


def test_far_candidate_restores_evidence_source_chain_and_time_is_lazy():
    """冷远线索带回 Evidence/Source，逻辑时间推进不改写 descriptor。"""
    case = next(item for item in _bundle().plan.cases
                if item.case_key.components[-1] == 4)
    correct = next(item for item in case.cold_candidates
                   if item.candidate_key.components[-1] == 102)
    outcome = next(item for item in _primary_outcomes()
                   if item.case_key == case.case_key)
    assert correct.candidate_key in outcome.adopted_candidate_keys
    assert any(correct.evidence_key in item.evidence_keys
               and correct.source_key in item.dependency_keys
               for item in outcome.receipt_records)
    assert all(item.audit_values.to_value()["full_store_rewrite_count"] == 0
               for item in _primary_outcomes())
    assert all(item.audit_values.to_value()["old_evidence_preserved"] == 1
               for item in _primary_outcomes())


def test_runtime_has_no_evaluator_label_or_expected_answer_dependency():
    """runtime 模块不能看到 expected/evaluator；标签变化不改变 raw run 字节。"""
    source = inspect.getsource(ph2_md04_probe_runtime)
    assert "ProbeEvaluatorLabel" not in source
    assert "expected_status" not in source
    assert "ph2_md05" not in source
    before = _runs().canonical_bytes()
    labels = build_md05_labels(_bundle().plan)
    assert labels
    replace(labels[0], expected_status="UNKNOWN")
    assert _runs().canonical_bytes() == before


def test_md05_independent_decision_passes_exact_conjunction():
    """独立 evaluator 给主策略 PASS、三基线 REJECT，并列出直接比较证据。"""
    decision = _decision()
    assert decision.verdict == "PASS"
    reports = {item.strategy_key: item for item in decision.strategy_reports}
    assert reports["OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"].probe_decision == "PASS"
    assert reports["OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"].hard_invariant_failures == ()
    assert all(item.probe_decision == "REJECT"
               for key, item in reports.items()
               if key != "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP")
    comparison = decision.comparison_evidence.to_value()
    assert comparison["no_quality_regression"] == 1
    assert comparison["challenge_improvement_count"] >= 1
    assert comparison["far_source_chain_recovered"] == 1
    assert comparison["irrelevant_query_cold_read_bytes"] == 0
    assert comparison["resource_growth_violation_count"] == 0
    assert comparison["time_advance_full_store_rewrites"] == 0
    assert comparison["held_out_combination_overlap_count"] == 0


def test_every_ablation_degrades_preregistered_dimension():
    """五个部件各自命中至少一项预注册维度，不允许 theater。"""
    evidence = _decision().ablation_evidence.to_value()
    assert tuple(evidence) == MD04_ABLATION_KEYS
    assert all(values for values in evidence.values())
    assert "UNRELATED_REVISION_CHANGES" in evidence[
        "DEPENDENCY_INVALIDATION"]
    assert "UNAUTHORIZED_GENERATION" in evidence["LAYERED_ATTRIBUTION"]
    assert "LOGIC_STEPS" in evidence["STOP_DECISION"]
    assert "ADOPTED_CORRECT" in evidence["TYPED_CENTER"]
    assert "COLD_READ_BYTES" in evidence["TYPED_CHANNEL_SELECTION"]


def test_run_decision_roundtrip_strict_zero_write_and_tamper_rejection(tmp_path):
    """raw run/decision 可规范回读、不可覆盖，伪 verdict/host write 失败。"""
    runs = _runs()
    decision = _decision()
    assert MD04ProbeRunArtifact.from_dict(runs.to_dict()) == runs
    assert MD05DecisionArtifact.from_dict(decision.to_dict()) == decision
    run_path = tmp_path / "runs.json"
    decision_path = tmp_path / "decision.json"
    write_immutable_artifact(runs, run_path)
    write_immutable_artifact(decision, decision_path)
    assert read_md04_probe_runs(run_path) == runs
    assert read_md05_decision(decision_path) == decision
    assert runs.host_learning_write_count == 0
    assert decision.evaluator_host_write_count == 0
    assert all(value == 0 for value in runs.execution_state.to_value().values())
    with pytest.raises(MD04ProbeContractError, match="宿主学习写"):
        replace(runs, host_learning_write_count=1)
    with pytest.raises(MD04ProbeContractError, match="verdict"):
        replace(decision, verdict="REJECT")
    bad_metrics = runs.strategy_outcomes[0].query_metrics.to_value()
    bad_metrics.pop("cold_read_bytes")
    with pytest.raises(MD04ProbeContractError, match="metrics 字段"):
        replace(
            runs.strategy_outcomes[0],
            query_metrics=CanonicalJsonObject.from_value(bad_metrics),
        )


def test_repository_md04_md05_artifacts_match_frozen_plan_and_fresh_run():
    """正式 artifact 必须逐字节等于当前 plan 的 fresh raw run 和独立决断。"""
    assert read_md04_probe_plan(PLAN_PATH) == _bundle().plan
    assert read_md04_probe_runs(RUN_PATH) == _runs()
    assert read_md05_decision(DECISION_PATH) == _decision()
