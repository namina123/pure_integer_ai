"""从真实整数投影区分缺角色、完整关系和来源授权的指代义务。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector, graph_query_input_structure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from pure_integer_ai.cognition.understanding.query_unlicensed_relation import (
    unlicensed_incomplete_relation_input,
)


def _grammar(*, marker_predicate=(5, 60), marker_role=(8, 2)):
    """教师整数只提供双角色语法和一个精确指代标记，不内置自然语言。"""
    proposition, predicate, source = (1, 50), (5, 60), (9, 70)
    roles = ((8, 1), (8, 2))
    trained = TrainedInputStructure(proposition, predicate, source, (7, 1), (
        TrainedInputMember((11,), (4, 101), 4, PROJECTION_FILLER,
            proposition, predicate, source, 0, 1, 0, roles[0]),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
            proposition, predicate, source, 1, 2, 1),
        TrainedInputMember((22,), (4, 102), 4, PROJECTION_FILLER,
            proposition, predicate, source, 2, 3, 2, roles[1]),
    ))
    marker = TrainedRelationQuestionFrame(
        (92002, 71), (92004, 71), marker_predicate, (roles[0],),
        ((), (33, 44)), marker_role, (44,), SLOT_MODE_REFERENCE)
    projector = TrainedInputStructureProjector((trained,), open_role_frames=(
        OpenRoleFrame(proposition, predicate, source, roles, ((), (33,), ()), (0, 3)),),
        slot_question_frames=(marker,))
    return projector, (marker,)


@pytest.mark.parametrize('values', (
    (11, 33, 99), (77, 11, 33, 99), (11, 33, 99, 77),
    (11, 33, 44, 77), (77, 11, 33, 44),
))
def test_unknown_role_and_extra_input_require_same_call_clarification(values):
    """未知值及已学标记的额外前后缀均不得授权历史替代角色。"""
    projector, frames = _grammar()
    structure = projector.project(values)
    original = structure.stable_key()
    assert unlicensed_incomplete_relation_input(structure, frames)
    assert structure.stable_key() == original


def test_licensed_reference_keeps_original_resolution_even_without_antecedent():
    """有训练许可即保留原指代路径，不能用先行词暂缺作为提前拒绝条件。"""
    projector, frames = _grammar()
    structure = projector.project((11, 33, 44))
    assert structure.open_relations
    assert not unlicensed_incomplete_relation_input(structure, frames)
    assert unlicensed_incomplete_relation_input(structure, ())


@pytest.mark.parametrize('change', ('predicate', 'role'))
def test_reference_permission_cannot_cross_predicate_or_role(change):
    projector, frames = _grammar(**({'marker_predicate': (5, 99)}
        if change == 'predicate' else {'marker_role': (8, 99)}))
    assert unlicensed_incomplete_relation_input(projector.project((11, 33, 44)), frames)


def test_complete_factual_path_and_graph_only_requests_keep_original_contract():
    projector, frames = _grammar()
    assert not unlicensed_incomplete_relation_input(projector.project((11, 33, 22)), frames)
    incomplete = projector.project((11, 33, 99))
    assert not unlicensed_incomplete_relation_input(incomplete, frames, complete_relation=True)
    assert not unlicensed_incomplete_relation_input(graph_query_input_structure(((1, 5),)), frames)
    assert not unlicensed_incomplete_relation_input(projector.project((99,)), frames)


def test_bound_wh_and_explicit_operation_keep_independent_contract():
    projector, frames = _grammar()
    wh = replace(frames[0], slot_mode=0)
    structure = projector.project((11, 33, 44))
    bindings = match_relation_slot_question_frames((wh,), structure, allow_dynamic_alternatives=True)
    assert bindings
    assert not unlicensed_incomplete_relation_input(replace(
        structure, operation_keys=(wh.operation_key,), relation_question_bindings=bindings), ())
    assert not unlicensed_incomplete_relation_input(replace(
        structure, operation_keys=((92004, 92),)), ())


@pytest.mark.parametrize('field,value', (
    ('proposition_key', (1, 99)), ('predicate_key', (5, 99)), ('source_ref', (9, 99)),
))
def test_foreign_projection_cannot_claim_an_observed_current_role(field, value):
    projector, frames = _grammar()
    structure = projector.project((11, 33, 99))
    changed = tuple(sorted((replace(item, **{field: value})
        if item.projection_kind == PROJECTION_FILLER else item
        for item in structure.semantic_candidates), key=lambda item: item.stable_key()))
    assert not unlicensed_incomplete_relation_input(replace(
        structure, semantic_candidates=changed), frames)


def test_complete_path_signal_is_a_strict_boolean():
    projector, frames = _grammar()
    with pytest.raises(TypeError):
        unlicensed_incomplete_relation_input(projector.project((11, 33, 99)), frames,
            complete_relation=1)
