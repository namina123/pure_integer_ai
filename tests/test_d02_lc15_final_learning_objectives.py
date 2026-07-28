"""D-02 LC-15 最终分型学习目标、owner 和消融冻结 T0。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_learning_objective_coverage import (
    BASELINE_ABLATION_KEYS,
    CORE_CAPABILITY_KEYS,
    FINAL_OBJECTIVE_MANIFEST_PATH,
    LearningObjectiveCoverageError,
    build_final_learning_objective_manifest,
    read_final_learning_objective_manifest,
    verify_final_learning_objective_sources,
    write_final_learning_objective_manifest,
)


FORMAL_MANIFEST_SHA256 = (
    "d69815ea6acd9e068ab88dd321993b95b3d6e0fad02a6c9c5101bdca9a40bd44")


def test_final_manifest_binds_every_core_capability_and_seven_families():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    assert manifest.course_status == "COURSE_FROZEN"
    assert manifest.objective_taxonomy_status == "FINAL_FROZEN"
    assert manifest.runtime_status == "NOT_STARTED"
    assert manifest.course_source_count == 9
    assert tuple(item.capability_key for item in manifest.capability_bindings) == (
        CORE_CAPABILITY_KEYS)
    expected_families = {key: "FROZEN" for key in SAMPLE_FAMILIES}
    for binding in manifest.capability_bindings:
        assert binding.sample_family_states.to_value() == expected_families
        assert binding.runtime_elimination_status == "NOT_STARTED"
        assert binding.runtime_pass_authority == 0
    assert manifest.capability_exit_states.to_value()[
        "TYPED_LEARNING_OBJECTIVES"] == "COURSE_FROZEN"


def test_eleven_objectives_have_independent_lifecycle_signals_and_owner_split():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    assert tuple(item.objective_key for item in manifest.objectives) == (
        LANGUAGE_OBJECTIVE_KEYS)
    signals = {item.elimination_signal for item in manifest.objectives}
    assert len(signals) == len(LANGUAGE_OBJECTIVE_KEYS)
    for objective in manifest.objectives:
        assert objective.training_evidence_owner == "TEACHER_EVIDENCE"
        assert objective.evaluation_evidence_owner == "EVALUATOR_LABEL"
        assert objective.external_signal_allowed == 0
        assert objective.evaluator_signal_can_train == 0
        assert objective.runtime_pass_authority == 0
        assert set(objective.candidate_lifecycle_outcomes) == {
            "ARCHIVED", "CONSUMER_EXIT", "REFUTED", "SUPERSEDED"}


def test_each_capability_has_objective_evaluator_negative_and_ablation():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    all_objectives = set(LANGUAGE_OBJECTIVE_KEYS)
    covered = set()
    for binding in manifest.capability_bindings:
        assert binding.objective_keys
        assert set(binding.objective_keys) <= all_objectives
        covered.update(binding.objective_keys)
        assert len(binding.elimination_signals) == len(binding.objective_keys)
        assert binding.evaluator_dimensions
        assert binding.baseline_kinds
        assert binding.course_ablation_keys
        assert binding.training_evidence_owner == "TEACHER_EVIDENCE"
        assert binding.evaluation_evidence_owner == "EVALUATOR_LABEL"
        assert binding.external_signal_allowed == 0
        assert binding.evaluator_signal_can_train == 0
        assert len(binding.evidence_refs) == 2
    assert covered == all_objectives


def test_frequency_boot_structure_and_shuffle_ablations_are_result_blind():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    assert tuple(item.ablation_key for item in manifest.baseline_ablations) == (
        BASELINE_ABLATION_KEYS)
    for ablation in manifest.baseline_ablations:
        assert ablation.expected_effect == (
            "DEGRADE_AT_LEAST_ONE_PRE_REGISTERED_DIMENSION")
        assert ablation.runtime_status == "NOT_STARTED"
        assert ablation.results_observed == 0
        assert ablation.runtime_pass_authority == 0


def test_course_and_pack_hashes_round_trip_to_existing_artifacts():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    verify_final_learning_objective_sources(
        manifest, repository_root=Path.cwd(), workspace_root=Path.cwd().parent)
    paths = {item.course_manifest_relative_path
             for item in manifest.capability_bindings}
    assert len(paths) == 9
    assert "data/ph2/manifests/lc01_lc15_initial_course_v1.json" in paths
    assert "data/ph2/manifests/lc14_attribution_quotation_course_v1.json" in paths


def test_manifest_round_trip_idempotent_and_nonoverwriting(tmp_path):
    manifest = build_final_learning_objective_manifest(Path.cwd())
    path = tmp_path / "lc15.json"
    write_final_learning_objective_manifest(manifest, path)
    assert read_final_learning_objective_manifest(path) == manifest
    write_final_learning_objective_manifest(manifest, path)
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(LearningObjectiveCoverageError, match="内容不同"):
        write_final_learning_objective_manifest(manifest, path)


def test_bad_owner_external_signal_and_fake_execution_fail_closed():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    binding = manifest.capability_bindings[0]
    with pytest.raises(LearningObjectiveCoverageError, match="训练 owner"):
        replace(binding, training_evidence_owner="EVALUATOR_LABEL")
    with pytest.raises(LearningObjectiveCoverageError, match="EXTERNAL"):
        replace(binding, external_signal_allowed=1)
    with pytest.raises(LearningObjectiveCoverageError, match="不得冒充"):
        replace(binding, runtime_elimination_status="PASSED")
    objective = manifest.objectives[0]
    with pytest.raises(LearningObjectiveCoverageError, match="evaluator 不得训练"):
        replace(objective, evaluator_signal_can_train=1)


def test_missing_family_objective_ablation_and_nonzero_state_fail_closed():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    binding = manifest.capability_bindings[0]
    states = binding.sample_family_states.to_value()
    states["UNKNOWN"] = "MISSING"
    with pytest.raises(LearningObjectiveCoverageError, match="七类"):
        replace(binding, sample_family_states=CanonicalJsonObject.from_value(states))
    with pytest.raises(LearningObjectiveCoverageError, match="淘汰信号"):
        replace(binding, elimination_signals=binding.elimination_signals[:-1])
    with pytest.raises(LearningObjectiveCoverageError, match="course_ablation_keys"):
        replace(binding, course_ablation_keys=())
    state = manifest.execution_state.to_value()
    state["candidate_eliminations_executed"] = 1
    with pytest.raises(LearningObjectiveCoverageError, match="execution state"):
        replace(manifest, execution_state=CanonicalJsonObject.from_value(state))


def test_missing_capability_objective_or_baseline_ablation_fails_closed():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    with pytest.raises(LearningObjectiveCoverageError, match="核心能力"):
        replace(manifest, capability_bindings=manifest.capability_bindings[:-1])
    with pytest.raises(LearningObjectiveCoverageError, match="十一类"):
        replace(manifest, objectives=manifest.objectives[:-1])
    with pytest.raises(LearningObjectiveCoverageError, match="四基线"):
        replace(manifest, baseline_ablations=manifest.baseline_ablations[:-1])


def test_source_hash_drift_is_detected_without_runtime_execution():
    manifest = build_final_learning_objective_manifest(Path.cwd())
    bad = replace(
        manifest.capability_bindings[0], course_manifest_sha256="0" * 64)
    mutated = replace(
        manifest, capability_bindings=(bad, *manifest.capability_bindings[1:]))
    with pytest.raises(LearningObjectiveCoverageError, match="hash 漂移"):
        verify_final_learning_objective_sources(
            mutated, repository_root=Path.cwd(), workspace_root=Path.cwd().parent)


def test_formal_repository_manifest_is_exact_and_all_execution_is_zero():
    manifest = read_final_learning_objective_manifest(FINAL_OBJECTIVE_MANIFEST_PATH)
    assert manifest.sha256() == FORMAL_MANIFEST_SHA256
    assert manifest.execution_state.to_value() == {
        "ablation_results_observed": 0,
        "candidate_eliminations_executed": 0,
        "companion_writes": 0,
        "core_learning_writes": 0,
        "d03_published": 0,
        "formal_training_runs": 0,
        "mastered_claims": 0,
        "memory_learning_writes": 0,
        "readiness_claims": 0,
        "teacher_calls": 0,
        "use_learning_writes": 0,
        "w01_started": 0,
    }
    verify_final_learning_objective_sources(
        manifest, repository_root=Path.cwd(), workspace_root=Path.cwd().parent)
