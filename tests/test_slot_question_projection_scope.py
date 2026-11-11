"""WH 的已观察角色不得因别名闭包而占用尚未出现的答案槽。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_concept_links import TrainedConceptLink
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_OPEN,
    TrainedInputMember, TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from pure_integer_ai.cognition.understanding.query_slot_question_scope import slot_question_projection_roles
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge


def _fixture(*, competing=False):
    """构造真实别名边、两角色来源及精确 WH 语法，全部采用开放整数符号。"""
    proposition, predicate, source = (1, 50), (5, 60), (9,)
    members = (
        TrainedInputMember((11,), (4, 101), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 0, 1, 0, (8, 1)),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                           proposition, predicate, source, 1, 2, 1),
        TrainedInputMember((22,), (4, 102), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 2, 3, 2, (8, 2)),
    )
    fact = TrainedInputStructure(proposition, predicate, source, (7, 1), members)
    frame = TrainedRelationQuestionFrame(
        (92002, 90), (92004, 1), predicate, ((8, 1),),
        ((44,), (55, 88)), (8, 2), (88,))
    frames = (frame,)
    if competing:
        frames += (replace(frame, source_case_key=(92002, 91),
                           roles=((8, 2),), answer_role_key=(8, 1)),)
    opens = tuple(OpenRoleFrame(
        proposition, predicate, source, item.roles, item.gaps, (0, 4))
        for item in frames)
    link = TrainedConceptLink((4, 101), (4, 102), proposition, predicate, source, (6, 1))
    projector = TrainedInputStructureProjector(
        (fact,), concept_links=(link,), open_role_frames=opens,
        slot_question_frames=frames)
    plain = TrainedInputStructureProjector(
        (fact,), concept_links=(link,), open_role_frames=opens)
    return fact, frames, projector, plain


def test_exact_wh_keeps_alias_history_but_answer_role_remains_open():
    """真实投影保留同指路径，同时仅计算当前已观察角色及后续语法证明。"""
    fact, frames, projector, plain = _fixture()
    values = (44, 11, 55, 88)
    projected = projector.project(values)
    old = plain.project(values)
    relation, = projected.relation_candidates
    assert relation.state == STRUCTURE_OPEN
    assert (relation.required_count, relation.support_count, relation.role_coverage,
            relation.required_open) == (3, 1, 1, 2)
    assert projected.concept_links == old.concept_links
    assert projected.concept_routes == old.concept_routes
    assert len(projected.concept_routes) == 1
    assert {(item.role_key, item.candidate_key) for item in projected.semantic_candidates} == {
        ((8, 1), (4, 101))}
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    closed = bridge._close_relation_question_predicate_scope(projected)
    binding, = match_relation_slot_question_frames(frames, closed)
    assert binding.answer_role_key == (8, 2)
    assert binding.slot_span_refs == relation.span_refs
    promoted, = closed.relation_candidates
    assert promoted.proposition_key == fact.proposition_key
    assert (promoted.required_open, promoted.predicate_coverage) == (1, 1)


@pytest.mark.parametrize("values", [(44, 11, 55, 89), (44, 11, 55, 88, 99), (11, 33, 22)])
def test_untrained_or_incomplete_wh_does_not_restrict_alias_projection(values):
    """未许可标记、未解析后缀及普通陈述不获得新角色裁剪资格。"""
    _fact, _frames, projector, plain = _fixture()
    assert projector.project(values) == plain.project(values)


def test_competing_observed_role_contracts_preserve_alias_candidates():
    """同谓词的相反角色解释不能任选一个，再靠删除另一角色消除竞争。"""
    _fact, frames, projector, plain = _fixture(competing=True)
    projected = projector.project((44, 11, 55, 88))
    assert projected == plain.project((44, 11, 55, 88))
    assert match_relation_slot_question_frames(frames, projected) == ()


def test_explicit_alias_can_still_ground_the_observed_role():
    """只限制角色位置，不能让合法已观察槽位失去原有别名来源。"""
    fact, frames, _projector, _plain = _fixture()
    origin = replace(fact, proposition_key=(1, 51), members=tuple(
        replace(item, proposition_key=(1, 51),
                candidate_key=(4, 111) if item.role_key == (8, 1) else item.candidate_key,
                values=(66,) if item.role_key == (8, 1) else item.values)
        for item in fact.members))
    links = (TrainedConceptLink((4, 111), (4, 101), origin.proposition_key,
                               origin.predicate_key, origin.source_ref, (6, 1)),)
    frame = replace(frames[0], gaps=((44,), (33, 88)))
    open_frame = OpenRoleFrame(fact.proposition_key, fact.predicate_key, fact.source_ref,
                               frame.roles, frame.gaps, (0, 4))
    projector = TrainedInputStructureProjector(
        (fact, origin), concept_links=links, open_role_frames=(open_frame,),
        slot_question_frames=(frame,))
    projected = projector.project((44, 66, 33, 88))
    relation = next(item for item in projected.relation_candidates
                    if item.proposition_key == fact.proposition_key)
    alias, = (item for item in projected.semantic_candidates
              if item.proposition_key == fact.proposition_key and item.role_key == (8, 1))
    assert alias.concept_route_key
    assert alias.candidate_key == (4, 101)
    bindings = match_relation_slot_question_frames((frame,), projected,
                                                    allow_exact_alternatives=True)
    binding, = (item for item in bindings if item.relation_key == relation.stable_key())
    assert binding.relation_key == relation.stable_key()


def _nested_fixture(*, competing=False):
    """答案名称是观察名称的真子串；没有别名边，不能借闭包存在才启用语法。"""
    proposition, predicate, source = (1, 150), (5, 160), (19,)
    members = (
        TrainedInputMember((11, 22), (4, 201), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 0, 2, 0, (8, 1)),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                           proposition, predicate, source, 2, 3, 1),
        TrainedInputMember((22,), (4, 202), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 3, 4, 2, (8, 2)),
    )
    fact = TrainedInputStructure(proposition, predicate, source, (7, 101), members)
    frame = TrainedRelationQuestionFrame((92002, 190), (92004, 1), predicate,
        ((8, 1),), ((44,), (55, 88)), (8, 2), (88,))
    frames = (frame,)
    if competing:
        frames += (replace(frame, source_case_key=(92002, 191),
                           roles=((8, 2),), answer_role_key=(8, 1)),)
    opens = tuple(OpenRoleFrame(proposition, predicate, source, item.roles, item.gaps, (0, 5))
                  for item in frames)
    return (frames, TrainedInputStructureProjector((fact,), open_role_frames=opens,
                slot_question_frames=frames),
            TrainedInputStructureProjector((fact,), open_role_frames=opens))


def test_nested_literal_observation_cannot_fill_the_unobserved_wh_answer_role():
    """保留两个精确字面候选，但只把完整已观察跨度计入该关系的角色证据。"""
    frames, projector, plain = _nested_fixture()
    values = (44, 11, 22, 55, 88)
    projected, original = projector.project(values), plain.project(values)
    assert projected.semantic_candidates == original.semantic_candidates
    assert projected.spans == original.spans
    relation, = projected.relation_candidates
    assert relation.state == STRUCTURE_OPEN
    assert (relation.role_coverage, relation.support_count, relation.required_open) == (1, 1, 2)
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    closed = bridge._close_relation_question_predicate_scope(projected)
    binding, = match_relation_slot_question_frames(frames, closed)
    assert binding.answer_role_key == (8, 2)
    assert len(binding.slot_span_refs) == 1
    observed, = (span for span in projected.spans if span.ref_key == binding.slot_span_refs[0])
    assert (observed.start, observed.end, observed.values) == (1, 3, (11, 22))


@pytest.mark.parametrize("values", ((44, 11, 22, 55, 89), (44, 11, 22, 55, 88, 99), (11, 22, 33, 22)))
def test_nested_literal_without_complete_wh_keeps_original_role_evidence(values):
    """非问式、未知后缀和额外输入不取得角色限制资格，普通事实仍保留原合同。"""
    _frames, projector, plain = _nested_fixture()
    assert projector.project(values) == plain.project(values)


def test_nested_literal_with_competing_role_contracts_remains_ambiguous():
    """完整问式的角色解释竞争时保留全部原始关系支持，不选择更长名称。"""
    frames, projector, plain = _nested_fixture(competing=True)
    projected = projector.project((44, 11, 22, 55, 88))
    assert projected == plain.project((44, 11, 22, 55, 88))
    assert match_relation_slot_question_frames(frames, projected) == ()


def _prefix_alternatives(projector, frames):
    """同合同的空前缀会把请求间隔吞入观察槽；两个解析均保留来源。"""
    alternatives = frames + tuple(replace(frame,
        source_case_key=(*frame.source_case_key, 1), gaps=((), *frame.gaps[1:]))
        for frame in frames)
    opens = tuple(OpenRoleFrame(fact.proposition_key, fact.predicate_key,
        fact.source_ref, frame.roles, frame.gaps, (0, 5))
        for fact in projector.structures for frame in alternatives)
    return alternatives, TrainedInputStructureProjector(projector.structures,
        concept_links=projector.concept_graph.links, open_role_frames=opens,
        slot_question_frames=alternatives)


def test_grounded_wh_span_is_not_blocked_by_a_prefix_absorbing_parse():
    """精确图角色支持当前完整名称，空前缀的未知大跨度不占用答案槽。"""
    frames, projector, _plain = _nested_fixture()
    frames, projector = _prefix_alternatives(projector, frames)
    projected = projector.project((44, 11, 22, 55, 88))
    relation, = projected.relation_candidates
    assert relation.state == STRUCTURE_OPEN
    assert (relation.role_coverage, relation.support_count, relation.required_open) == (1, 1, 2)
    assert len(projected.semantic_candidates) == 2
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    closed = bridge._close_relation_question_predicate_scope(projected)
    binding, = match_relation_slot_question_frames(frames, closed)
    span, = (item for item in closed.spans if item.ref_key in binding.slot_span_refs)
    assert (span.start, span.end, span.values) == (1, 3, (11, 22))


def test_grounded_alias_span_also_resolves_only_same_contract_prefix_parses():
    """别名经过完整图路径落地时与直接字面角色采用相同的语法资格。"""
    fact, frames, original_projector, _plain = _fixture()
    origin = replace(fact, proposition_key=(1, 51), members=tuple(
        replace(item, proposition_key=(1, 51),
                candidate_key=(4, 111) if item.role_key == (8, 1) else item.candidate_key,
                values=(66,) if item.role_key == (8, 1) else item.values)
        for item in fact.members))
    link = TrainedConceptLink((4, 111), (4, 101), origin.proposition_key,
                             origin.predicate_key, origin.source_ref, (6, 1))
    frame = replace(frames[0], gaps=((44,), (33, 88)))
    projector = TrainedInputStructureProjector((fact, origin),
        concept_links=(link, *original_projector.concept_graph.links))
    frames, projector = _prefix_alternatives(projector, (frame,))
    projected = projector.project((44, 66, 33, 88))
    relation = next(item for item in projected.relation_candidates
                    if item.proposition_key == fact.proposition_key)
    alias, = (item for item in projected.semantic_candidates
              if item.proposition_key == fact.proposition_key and item.role_key == (8, 1))
    assert alias.concept_route_key
    assert relation.role_coverage == 1
    assert match_relation_slot_question_frames(frames, projected,
        allow_exact_alternatives=True)


def test_prefix_alternatives_preserve_different_role_contract_competition():
    """即使只有一个 Role 有当前图支持，也不能取消相反 Role 的语法合同。"""
    frames, projector, _plain = _nested_fixture(competing=True)
    _frames, projector = _prefix_alternatives(projector, frames)
    plain = TrainedInputStructureProjector(projector.structures,
        open_role_frames=projector.open_role_frames)
    assert projector.project((44, 11, 22, 55, 88)) == plain.project((44, 11, 22, 55, 88))


def test_two_grounded_prefix_spans_are_not_ranked_by_length():
    """两个来源分别支持完整/含前缀跨度时保留竞争，不选更短或更长名称。"""
    frames, projector, _plain = _nested_fixture()
    fact, = projector.structures
    other = replace(fact, proposition_key=(1, 151), source_ref=(29,),
        members=tuple(replace(item, proposition_key=(1, 151), source_ref=(29,),
            values=(44, 11, 22) if item.role_key == (8, 1) else item.values,
            trained_start=0 if item.role_key == (8, 1) else item.trained_start + 1,
            trained_end=item.trained_end + 1) for item in fact.members))
    base = TrainedInputStructureProjector((fact, other))
    _frames, projector = _prefix_alternatives(base, frames)
    plain = TrainedInputStructureProjector(base.structures,
        open_role_frames=projector.open_role_frames)
    assert projector.project((44, 11, 22, 55, 88)) == plain.project((44, 11, 22, 55, 88))


def test_unresolved_prefix_parses_do_not_ground_an_unknown_observed_name():
    """只看见答案名称的片段时，两个未知观察跨度仍保留原来的竞争。"""
    frames, projector, _plain = _nested_fixture()
    _frames, projector = _prefix_alternatives(projector, frames)
    plain = TrainedInputStructureProjector(projector.structures,
        open_role_frames=projector.open_role_frames)
    assert projector.project((44, 77, 22, 55, 88)) == plain.project((44, 77, 22, 55, 88))


@pytest.mark.parametrize("same_source", (False, True))
def test_multiple_observed_roles_require_support_from_one_complete_source(same_source):
    """不同命题各自命中一个槽不能拼成当前解析的完整图支持。"""
    predicate, first, second = (5, 160), ((8, 1), 1, 2), ((8, 2), 3, 4)
    frame = TrainedRelationQuestionFrame((92002, 201), (92004, 1), predicate,
        ((8, 1), (8, 2)), ((44,), (55,), (66, 88)), (8, 3), (88,))
    alternative = replace(frame, source_case_key=(92002, 202),
                          gaps=((), *frame.gaps[1:]))
    groups = ((frozenset((first, second)),) if same_source else
              (frozenset((first,)), frozenset((second,))))
    result = slot_question_projection_roles((frame, alternative),
        (44, 11, 55, 22, 66, 88), (19,), role_span_groups={predicate: groups})
    assert result == ({predicate: frozenset((first, second))} if same_source else {})
