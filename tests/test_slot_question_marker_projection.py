"""普通WH问词边界不同不改变已观察角色，reference授权继续严格隔离。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, match_relation_slot_question_frames,
)
from pure_integer_ai.cognition.understanding.query_slot_question_scope import slot_question_projection_roles
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from test_slot_question_projection_scope import _nested_fixture


def test_plain_wh_marker_annotations_keep_answer_substring_out_of_observed_role():
    """真实投影及绑定消费完整帧；答案名称的重叠子串不能关闭答案槽。"""
    frames, projector, _plain = _nested_fixture()
    variant = replace(frames[0], source_case_key=(92002, 991), wh_values=(55, 88))
    inventory = (*frames, variant)
    current = TrainedInputStructureProjector(projector.structures,
        open_role_frames=projector.open_role_frames, slot_question_frames=inventory)
    prepared = current.project((44, 11, 22, 55, 88))
    assert prepared.semantic_candidates == projector.project((44, 11, 22, 55, 88)).semantic_candidates
    assert all(r.required_open == 2 for r in prepared.relation_candidates)
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = inventory
    closed = bridge._close_relation_question_predicate_scope(prepared)
    binding, = match_relation_slot_question_frames(inventory, closed)
    assert binding.answer_role_key == (8, 2)
    assert len(inventory) == 2 and inventory[0].stable_key() != inventory[1].stable_key()


def test_reference_marker_boundaries_do_not_share_plain_projection_equivalence():
    """两个不同reference原子仍是不同授权，不因相同角色和gaps而合并。"""
    frames, _projector, _plain = _nested_fixture()
    frame = replace(frames[0], slot_mode=SLOT_MODE_REFERENCE)
    other = replace(frame, source_case_key=(92002, 992), wh_values=(55, 88))
    predicate = frame.predicate_key
    groups = {predicate: (frozenset((((8, 1), 1, 3),)),)}
    assert slot_question_projection_roles((frame, other), (44, 11, 22, 55, 88), (19,), role_span_groups=groups) == {}


@pytest.mark.parametrize('changes', ({'operation_key': (92004, 99)},
    {'answer_role_key': (8, 3)}, {'roles': ((8, 3),)}))
def test_marker_equivalence_cannot_cross_operation_answer_or_observed_role(changes):
    """仅WH标注边界可分账；实际语义合同漂移继续拒绝位置裁剪。"""
    frames, _projector, _plain = _nested_fixture()
    frame = frames[0]
    other = replace(frame, source_case_key=(92002, 993), wh_values=(55, 88), **changes)
    groups = {frame.predicate_key: (frozenset((((8, 1), 1, 3),)),)}
    assert slot_question_projection_roles((frame, other), (44, 11, 22, 55, 88), (19,), role_span_groups=groups) == {}
