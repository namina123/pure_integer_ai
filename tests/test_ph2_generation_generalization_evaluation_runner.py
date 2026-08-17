"""E-05 production evaluation actual runner 的 synthetic public dry-run。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_generation_generalization_evaluation_runner as evaluation_runner,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    build_generation_candidate_pack,
    publish_generation_candidate_pack,
    read_generation_candidate_pack,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationBatch,
    GenerationGeneralizationEvaluationBatchRunError,
    run_generation_generalization_evaluation_actual,
    run_generation_generalization_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    learn_grounded_answer_surface_model,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
)
from pure_integer_ai.storage.backend import DictBackend


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def _observations():
    """剥离四条 synthetic held-out 输入，覆盖三类 actual path。"""
    budget = GenerationGeneralizationEvaluationBudget(
        512, 4, 4, 96, 16)
    episodes = read_grounded_answer_episodes(_SAMPLE)
    selected = tuple(
        item for item in episodes
        if item.question.answer_plan.response_act in {
            "ANSWER", "CLARIFY", "CONFLICT"}
    )
    return tuple(
        GenerationGeneralizationEvaluationObservation.from_held_out_episode(
            replace(
                item,
                episode_id=f"synthetic-e05-{index}-{item.episode_id}",
                split="held_out",
            ),
            budget,
        )
        for index, item in enumerate(selected, start=1)
    )


def _refute_first_requirement(batch):
    """只改写首项独立 result/report，保留 actual input 与 route。"""
    run = batch.runs[0]
    evidence = run.requirements[0]
    old = evidence.verification.result
    new = replace(old, verdict=VERDICT_REFUTE)
    reports = tuple(
        replace(
            report,
            results=tuple(new if item == old else item
                          for item in report.results),
        )
        for report in evidence.verification_reports
    )
    failed_evidence = replace(
        evidence,
        verification_reports=reports,
        verification=replace(evidence.verification, result=new),
    )
    failed_run = replace(
        run,
        requirements=(failed_evidence, *run.requirements[1:]),
    )
    return GenerationGeneralizationEvaluationBatch(
        (failed_run, *batch.runs[1:]))


def test_evaluation_runner_executes_three_paths_and_aggregates_pass_fail_ne(
        tmp_path):
    """四条 actual run 六路 PASS；改写 refute 为 FAIL；缺路为 NE。"""
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    pack = build_generation_candidate_pack(
        model, hashlib.sha256(_SAMPLE.read_bytes()).hexdigest())
    published = publish_generation_candidate_pack(pack, tmp_path / "candidate")
    loaded = read_generation_candidate_pack(
        published.pack_root, expected_sha256=pack.sha256())

    backend = DictBackend()
    try:
        host = make_train_context(backend)
        baseline = backend.snapshot()
        observations = _observations()
        batch = run_generation_generalization_evaluation_batch(
            host, loaded, observations)

        assert batch.status == "PASS"
        assert batch.coverage == INDEPENDENT_VERIFIER_REQUIREMENTS
        assert {run.path for run in batch.runs} == {
            "ANSWER", "RESPONSE_ACT", "REFERENCE"}
        assert all(run.runtime_status
                   == "PASS_EVALUATION_ACTUAL_CONJUNCTION"
                   for run in batch.runs)
        assert all(item.status == "PASS" for item in batch.evidence)
        assert all(run.surface_text for run in batch.runs)
        assert all(run.teacher_call_count == 0
                   and run.label_read_count == 0
                   and run.host_learning_write_count == 0
                   for run in batch.runs)
        assert len({run.evaluation_owner_key for run in batch.runs}) == 4
        assert backend.snapshot() == baseline

        answer = next(
            item for item in observations
            if (item.question.answer_plan.response_act == "ANSWER"
                and item.reference_course is None))
        repeated = run_generation_generalization_evaluation_actual(
            host, loaded, answer)
        original = next(
            item for item in batch.runs
            if item.observation == answer)
        assert repeated.stable_key() == original.stable_key()
        assert repeated.surface_text == original.surface_text
        assert backend.snapshot() == baseline

        failed = _refute_first_requirement(batch)
        assert failed.status == "FAIL"
        assert failed.requirement_status(
            "INDEPENDENT_UNDERSTANDING_READBACK") == "FAIL"

        incomplete = GenerationGeneralizationEvaluationBatch(tuple(
            run for run in batch.runs
            if run.observation.question.answer_plan.response_act != "CONFLICT"
        ))
        assert incomplete.status == "NE"
        assert incomplete.requirement_status(
            "SOURCE_UNCERTAINTY_CITATION") == "NE"
    finally:
        backend.close()


def test_evaluation_batch_failure_records_boundary_without_forwarding_message(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """batch 异常只暴露 ordinal/path，保留 cause 供安全诊断器取代码位置。"""
    observations = _observations()

    def fail_actual(*_args, **_kwargs):
        raise RuntimeError("private observation text must not escape")

    monkeypatch.setattr(
        evaluation_runner,
        "run_generation_generalization_evaluation_actual",
        fail_actual,
    )
    with pytest.raises(
            GenerationGeneralizationEvaluationBatchRunError) as caught:
        run_generation_generalization_evaluation_batch(
            object(), object(), observations)
    assert caught.value.observation_ordinal == 1
    assert caught.value.evaluation_path in {
        "ANSWER", "RESPONSE_ACT", "REFERENCE"}
    assert str(caught.value) == "evaluation batch item failed"
    assert "private observation" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
