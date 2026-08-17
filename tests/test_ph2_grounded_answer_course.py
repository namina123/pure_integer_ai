"""来源约束问答、多表面和 split 防泄漏合同专项。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    CARRIER_KINDS,
    REFERENCE_CHALLENGE_KINDS,
    REFERENCE_GRANULARITY,
    REFERENCE_STRATEGIES,
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
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorError,
    GroundedAnswerConnectorTarget,
    build_grounded_answer_connector,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
    GroundedAnswerSurfaceParser,
    build_grounded_answer_parser_catalog,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from tests.test_g02_generation_structure_plan import _request
from tests.test_g03_generation_surface import _surface_protocol


SAMPLE_PATH = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def test_train_sample_closes_four_response_acts_and_multi_surface_contract():
    """四类 response act 均有多个合法表面，负例保留分型失败。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    audit = audit_grounded_answer_course(episodes)
    assert audit.episode_count == 5
    assert audit.accepted_surface_count == 11
    assert audit.rejected_surface_count == 12
    assert {item.question.answer_plan.response_act for item in episodes} == {
        "ANSWER", "UNKNOWN", "CLARIFY", "CONFLICT",
    }
    assert all(item.split == "train" for item in episodes)
    assert all(len(item.surfaces.accepted) >= 2 for item in episodes)
    assert {"MARKDOWN", "HTML", "CODE", "TABLE"} <= CARRIER_KINDS


def test_clarify_plan_rejects_hidden_conflict_before_runtime():
    """CLARIFY 不能夹带会使 G-01 必然改判 CONFLICT 的第三候选。"""
    base = next(
        item.question for item in read_grounded_answer_episodes(SAMPLE_PATH)
        if item.question.answer_plan.response_act == "CLARIFY")
    conflict = (
        GroundedEvidence(
            "ev-clarify-conflict-support",
            "p-clarify-conflict",
            "src-clarify-conflict-a",
            base.evidence_scope_id,
            "第三候选处于开放状态",
            "来源甲记录第三候选处于开放状态",
            1,
            0,
        ),
        GroundedEvidence(
            "ev-clarify-conflict-refute",
            "p-clarify-conflict",
            "src-clarify-conflict-b",
            base.evidence_scope_id,
            "第三候选处于开放状态",
            "来源乙否认第三候选处于开放状态",
            0,
            1,
        ),
    )

    with pytest.raises(GroundedAnswerCourseError, match="不得包含冲突"):
        replace(base, evidence=(*base.evidence, *conflict))


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
    assert bundle.validation.source_ref_count == 5
    assert bundle.validation.observation_count == 5
    assert bundle.validation.teacher_evidence_count == 5
    assert bundle.validation.evaluator_label_count == 0
    assert bundle.validation.splits == ("train",)
    for episode, observation, teacher in zip(
            episodes, bundle.observations, bundle.teacher_evidence):
        visible = observation.typed_payload.to_value()
        hidden = teacher.typed_evidence.to_value()
        assert "typed_intent" not in visible
        assert "answer_plan" not in visible
        assert "surface_realizations" not in visible
        assert "reference_course" not in visible
        assert hidden["typed_intent"] == episode.question.typed_intent
        assert hidden["answer_plan"] == episode.question.answer_plan.to_dict()
        if episode.reference_course is None:
            assert "reference_course" not in hidden
        else:
            assert hidden["reference_course"] == (
                episode.reference_course.to_dict())
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


def test_reference_course_is_teacher_only_and_bound_to_real_surfaces():
    """双句 reference 标签精确绑定 Evidence、顺序、slot 与表面 span。"""
    episode = read_grounded_answer_episodes(SAMPLE_PATH)[-1]
    course = episode.reference_course
    assert course is not None
    assert course.granularity == REFERENCE_GRANULARITY
    assert tuple(item.strategy for item in course.surface_labels) == (
        REFERENCE_STRATEGIES)
    assert tuple(item.kind for item in course.challenges) == (
        REFERENCE_CHALLENGE_KINDS)
    accepted = {
        item.realization_id: item for item in episode.surfaces.accepted
    }
    for label in course.surface_labels:
        surface = accepted[label.realization_id].surface
        assert surface[label.span_start:label.span_end] == (
            label.reference_surface)
        assert label.span_start == 14


def test_learned_claim_slot_generates_two_surfaces_for_unseen_proposition():
    """已学字面+claim 槽可填入新来源命题，不回放旧实体或唯一答案。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    bundle = compile_grounded_answer_training_records(SAMPLE_PATH)
    model, report = learn_grounded_answer_surface_model(bundle)
    assert report.episode_count == 5
    assert report.accepted_surface_count == 11
    assert report.pattern_count == 11
    assert report.slotted_pattern_count == 5
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
        "记录表明，云岭站西门于2026年启用。",
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


def _connector_question_and_candidate():
    """建立未见 claim、typed candidate 和共享 connector 编译输入。"""
    episodes = read_grounded_answer_episodes(SAMPLE_PATH)
    model, _ = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(SAMPLE_PATH))
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
    request, _unused = _request(count=1)
    branch = language_branch_identity((20916, 900, 1))
    planning = GenerationPlanningRequest(
        replace(request.goal, target_branch=branch),
        request.candidates,
    )
    return model, question, planning, planning.candidates[0], branch


def test_grounded_patterns_compile_to_explicit_single_template_variants():
    """多个合法 pattern 必须显式选择，且每个模板保留 literal/claim 分槽。"""
    model, question, _planning, candidate, branch = (
        _connector_question_and_candidate())
    surface_protocol = _surface_protocol(20916)
    compilation = compile_grounded_answer_connectors(
        model,
        question,
        GroundedAnswerConnectorTarget(
            candidate.proposition, branch, (20916, 901)),
        surface_protocol,
    )
    assert len(compilation.variants) == 3
    assert len(compilation.structures) == 2
    assert sorted(len(item.pattern_ids) for item in compilation.structures) == [
        1, 2]
    assert all(item.option.support_teacher_keys
               for item in compilation.variants)
    assert all(len(item.template.slots) >= 2
               for item in compilation.variants)
    assert all(sum(
        binding.source == compilation.value_protocol.proposition_source
        for binding in item.template.bindings) == 1
        for item in compilation.variants)
    other_request, _unused = _request(count=2)
    other_candidate = other_request.candidates[1]
    other = compile_grounded_answer_connectors(
        model,
        question,
        GroundedAnswerConnectorTarget(
            other_candidate.proposition, branch, (20916, 901)),
        surface_protocol,
    )
    assert {item.template.connector for item in compilation.variants}.isdisjoint(
        {item.template.connector for item in other.variants})
    selected = compilation.variants[0]
    variant, connector = build_grounded_answer_connector(
        compilation,
        selected.option.structure_id,
        selected.option.pattern_id,
        surface_protocol,
    )
    assert variant == selected
    assert connector.registry.templates == (selected.template,)
    with pytest.raises(GroundedAnswerConnectorError, match="不属于"):
        build_grounded_answer_connector(
            compilation,
            selected.option.structure_id,
            999999999,
            surface_protocol,
        )
    other_structure = next(
        item for item in compilation.structures
        if item.structure_id != selected.option.structure_id)
    with pytest.raises(GroundedAnswerConnectorError, match="不属于已选"):
        build_grounded_answer_connector(
            compilation,
            other_structure.structure_id,
            selected.option.pattern_id,
            surface_protocol,
        )


def test_restricted_parser_recovers_claim_and_classifies_surface_damage():
    """parser 只凭 units/catalog 恢复命题，并分型遗漏、重复和未知损坏。"""
    model, question, _planning, candidate, branch = (
        _connector_question_and_candidate())
    surface_protocol = _surface_protocol(20917)
    compilation = compile_grounded_answer_connectors(
        model,
        question,
        GroundedAnswerConnectorTarget(
            candidate.proposition, branch, (20917, 901)),
        surface_protocol,
    )
    renderer = minimal_instruction_identity((20917, 902, 1))
    catalog = build_grounded_answer_parser_catalog(
        compilation, candidate, renderer)
    protocol = GroundedAnswerParserProtocol(*tuple(
        minimal_instruction_identity((20917, 903, index))
        for index in range(1, 7)
    ))
    parser = GroundedAnswerSurfaceParser(protocol, catalog)
    grammar = catalog.grammars[0]
    request = GenerationSurfaceParseRequest(
        renderer,
        grammar.units,
        branch,
        candidate.source,
        candidate.scope,
    )

    parsed = parser.parse(request)

    assert parsed.succeeded
    assert parsed.observation.representations == grammar.representations
    assert parsed.observation.propositions[0].proposition == (
        candidate.proposition)
    claim = next(
        item.units for item in grammar.slots
        if item.part_kind == "CLAIM")
    missing = parser.parse(replace(request, units=(0x3002,)))
    duplicate = parser.parse(replace(
        request, units=(*grammar.units, *claim)))
    damaged = parser.parse(replace(
        request, units=(*grammar.units[:-1], 0xFF01)))
    assert missing.reason == protocol.missing_claim
    assert duplicate.reason == protocol.duplicate_claim
    assert damaged.reason == protocol.no_match
