"""来源约束问答、多表面和 split 防泄漏合同专项。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    CARRIER_KINDS,
    GroundedAnswerPlan,
    GroundedAnswerCourseError,
    GroundedEvidence,
    audit_grounded_answer_course,
    read_grounded_answer_episodes,
    verify_surface_realization,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    learn_grounded_answer_surface_model,
    realize_grounded_answer_surfaces,
)


SAMPLE_PATH = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_train_sample_closes_four_response_acts_and_multi_surface_contract():
    """四类 response act 均有多个合法表面，负例保留分型失败。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    audit = audit_grounded_answer_course(episodes)
    assert audit.episode_count == 4
    assert audit.accepted_surface_count == 8
    assert audit.rejected_surface_count == 10
    assert {item.question.answer_plan.response_act for item in episodes} == {
        "ANSWER", "UNKNOWN", "CLARIFY", "CONFLICT",
    }
    assert all(item.split == "train" for item in episodes)
    assert all(len(item.surfaces.accepted) >= 2 for item in episodes)
    assert {"MARKDOWN", "HTML", "CODE", "TABLE"} <= CARRIER_KINDS


def test_declared_negative_dimensions_are_recomputed_not_trusted():
    """每个 rejected surface 的失败维度由 typed plan 重新计算。"""
    for episode in read_grounded_answer_episodes(SAMPLE_PATH):
        for accepted in episode.surfaces.accepted:
            assert verify_surface_realization(
                episode.question, accepted).passed
        for rejected in episode.surfaces.rejected:
            result = verify_surface_realization(
                episode.question, rejected.realization)
            assert not result.passed
            assert result.violations == rejected.expected_violations


def test_each_split_axis_rejects_cross_split_leakage():
    """source、命题、问句构造和 paraphrase 任一复用均不得跨 split。"""
    first = read_grounded_answer_episodes(SAMPLE_PATH)[0]
    for axis in ("source", "proposition", "question_construction", "paraphrase"):
        clusters = replace(
            first.clusters,
            **{name: f"isolated-{name}" for name in (
                "source", "proposition", "question_construction", "paraphrase")
               if name != axis},
        )
        leaked = replace(
            first,
            episode_id=f"held-out-leak-{axis}",
            split="held_out",
            clusters=clusters,
        )
        with pytest.raises(GroundedAnswerCourseError, match="跨 split 泄漏"):
            audit_grounded_answer_course((first, leaked), train_only=False)


def test_reader_rejects_noncanonical_json_before_course_use(tmp_path):
    """有内容但非规范字节的 JSONL 不得进入训练切片。"""
    value = json.loads(SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    path = tmp_path / "noncanonical.jsonl"
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(GroundedAnswerCourseError, match="不是规范"):
        read_grounded_answer_episodes(path)


def test_training_compiler_separates_observation_from_teacher_labels():
    """学生 Observation 不得看到 intent、plan、合法 surface 或负例维度。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    bundle = compile_grounded_answer_training_records(SAMPLE_PATH)
    assert bundle.validation.source_ref_count == 4
    assert bundle.validation.observation_count == 4
    assert bundle.validation.teacher_evidence_count == 4
    assert bundle.validation.evaluator_label_count == 0
    assert bundle.validation.splits == ("train",)
    for episode, observation, teacher in zip(
            episodes, bundle.observations, bundle.teacher_evidence):
        visible = observation.typed_payload.to_value()
        hidden = teacher.typed_evidence.to_value()
        assert "typed_intent" not in visible
        assert "answer_plan" not in visible
        assert "surface_realizations" not in visible
        assert hidden["typed_intent"] == episode.question.typed_intent
        assert hidden["answer_plan"] == episode.question.answer_plan.to_dict()
        visible_text = json.dumps(visible, ensure_ascii=False, sort_keys=True)
        for realization in (
                *episode.surfaces.accepted,
                *(item.realization for item in episode.surfaces.rejected)):
            assert realization.surface not in visible_text


def test_training_compiler_is_deterministic_without_writing_artifacts():
    """同一 sample 重编译得到相同记录，且首切片不发布 held-out/evaluator。"""
    first = compile_grounded_answer_training_records(SAMPLE_PATH)
    second = compile_grounded_answer_training_records(SAMPLE_PATH)
    assert first == second
    assert all(item.split == "train" for item in first.observations)


def test_learned_claim_slot_generates_two_surfaces_for_unseen_proposition():
    """已学字面+claim 槽可填入新来源命题，不回放旧实体或唯一答案。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    bundle = compile_grounded_answer_training_records(SAMPLE_PATH)
    model, report = learn_grounded_answer_surface_model(bundle)
    assert report.episode_count == 4
    assert report.accepted_surface_count == 8
    assert report.pattern_count == 8
    assert report.slotted_pattern_count == 2
    assert report.response_act_count == 4

    base = episodes[0].question
    question = replace(
        base,
        context_surface="云岭站运行档案记载：西门于2026年启用。",
        question_surface="云岭站西门何时启用？",
        evidence_scope_id=301,
        response_scope_id=401,
        evidence=(GroundedEvidence(
            "ev-new-year",
            "p-new-year",
            "src-new-station-record",
            301,
            "云岭站西门于2026年启用",
            "西门于2026年启用",
            1,
            0,
        ),),
        answer_plan=GroundedAnswerPlan(
            "ANSWER",
            ("p-new-year",),
            ("p-new-year",),
            (),
            ("src-new-station-record",),
        ),
    )
    generated = realize_grounded_answer_surfaces(model, question)
    assert {item.surface for item in generated} == {
        "云岭站西门于2026年启用。",
        "档案显示，云岭站西门于2026年启用。",
    }
    assert all("北川站" not in item.surface for item in generated)
    assert all(verify_surface_realization(question, item).passed
               for item in generated)


def test_learned_nonanswer_patterns_remain_domain_neutral():
    """UNKNOWN 复用的是 response-act 句式，不携带训练问题中的实体和数值。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    model, _ = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(SAMPLE_PATH))
    base = episodes[1].question
    question = replace(
        base,
        context_surface="当前资料没有提供目标设施的维护成本。",
        question_surface="目标设施的维护成本是多少？",
        evidence_scope_id=302,
        response_scope_id=402,
    )
    generated = realize_grounded_answer_surfaces(model, question)
    assert len(generated) == 2
    assert all("东门" not in item.surface and "预算" not in item.surface
               for item in generated)
    assert all(verify_surface_realization(question, item).passed
               for item in generated)
