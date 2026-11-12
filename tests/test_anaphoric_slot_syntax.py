"""受训纯语法可保留未决指代，不能从未知名称或空跨度制造事实。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_anaphoric_slot_scope import attach_anaphoric_slot_syntax
from pure_integer_ai.cognition.understanding.query_input_structure import STRUCTURE_CLOSED, STRUCTURE_OPEN
from pure_integer_ai.cognition.understanding.query_reference_context import extended_reference_context_required
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from test_slot_question_projection_scope import _fixture


def _syntax_fixture():
    """实际投影器没有字面语义成员，完整问式仍精确解析一个来源化marker。"""
    core, frames, projector, _plain = _fixture()
    marker = TrainedRelationQuestionFrame((92002, 200), frames[0].operation_key,
        frames[0].predicate_key, ((8, 2),), ((99, 77), (78,)), (8, 1), (99,), SLOT_MODE_REFERENCE)
    inventory = (*frames, marker)
    prepared = projector.project((44, 99, 55, 88))
    assert not prepared.semantic_candidates and not prepared.relation_candidates
    assert len(prepared.open_relations) == 1
    return core, inventory, prepared


def test_syntax_only_projection_reaches_wh_and_reference_gates_without_filler():
    """完整真实投影→语法候选→WH绑定→历史义务，仍缺全部实体和答案。"""
    core, frames, initial = _syntax_fixture()
    prepared = attach_anaphoric_slot_syntax(initial, frames, (core,))
    candidate, = prepared.relation_candidates
    assert (candidate.span_refs, candidate.support_count, candidate.role_coverage,
            candidate.predicate_coverage, candidate.required_open, candidate.state) == ((), 1, 0, 1, 2, STRUCTURE_OPEN)
    assert prepared.semantic_candidates == initial.semantic_candidates == ()
    assert prepared.open_relations == initial.open_relations
    binding, = match_relation_slot_question_frames(frames, prepared, allow_dynamic_alternatives=True)
    prepared = replace(prepared, operation_keys=(binding.operation_key,), relation_question_bindings=(binding,))
    assert extended_reference_context_required(prepared, frames)
    assert prepared.relation_candidates[0].required_open == 2


@pytest.mark.parametrize('drift', ('marker', 'predicate', 'role', 'source', 'core-role', 'no-core', 'no-question'))
def test_syntax_candidate_requires_all_current_training_and_core_witnesses(drift):
    """缺任一完整授权、来源或角色合同均保持原输入，不扩大历史权限。"""
    core, frames, initial = _syntax_fixture()
    plain, marker = frames
    if drift == 'marker': marker = replace(marker, wh_values=(98,), gaps=((98, 77), (78,)))
    elif drift == 'predicate': marker = replace(marker, predicate_key=(5, 999))
    elif drift == 'role': marker = replace(marker, roles=((8, 1),), answer_role_key=(8, 2))
    elif drift == 'source': core = replace(core, source_ref=(999,), members=tuple(replace(m, source_ref=(999,)) for m in core.members))
    elif drift == 'core-role': core = replace(core, members=tuple(replace(m, role_key=(8, 3)) if m.role_key == (8, 2) else m for m in core.members))
    inventory = (marker,) if drift == 'no-question' else (plain, marker)
    assert attach_anaphoric_slot_syntax(initial, inventory, () if drift == 'no-core' else (core,)) == initial


@pytest.mark.parametrize('changes', ({'role_coverage': 1}, {'predicate_coverage': 0},
    {'state': STRUCTURE_CLOSED}, {'required_count': 2, 'required_open': 1}))
def test_empty_semantic_span_list_cannot_claim_objects_or_closure(changes):
    """空语义跨度只表示严格的未决语法支持，不能宣称实体或闭合事实。"""
    core, frames, initial = _syntax_fixture()
    candidate, = attach_anaphoric_slot_syntax(initial, frames, (core,)).relation_candidates
    with pytest.raises(ValueError):
        replace(candidate, **changes)


def test_distinct_complete_sources_remain_competing_candidates():
    """同形语法保留全部完整命题和来源，不按稳定键选择答案。"""
    core, frames, initial = _syntax_fixture()
    other = replace(core, proposition_key=(1, 51), source_ref=(10,), members=tuple(
        replace(member, proposition_key=(1, 51), source_ref=(10,)) for member in core.members))
    opened, = initial.open_relations
    alternative = replace(opened, frame=replace(opened.frame, proposition=other.proposition_key, source_ref=other.source_ref))
    initial = replace(initial, open_relations=tuple(sorted((opened, alternative), key=lambda row: row.stable_key())))
    result = attach_anaphoric_slot_syntax(initial, frames, (core, other))
    assert len(result.relation_candidates) == 2
    assert len(match_relation_slot_question_frames(frames, result, allow_dynamic_alternatives=True)) == 2
    assert not match_relation_slot_question_frames(frames, result)
