"""多观察角色WH沿完整训练次序消费，答案与Core来源成员不能被重排替代。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    InputRelationQuestionBinding, PROJECTION_FILLER, PROJECTION_PREDICATE,
    TrainedInputMember, TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_question_open_frames import (
    bound_question_open_frames, question_observed_role_contract,
)


def _identity(key):
    """测试身份只公开规范稳定键，不引入宿主对象语义。"""
    return SimpleNamespace(stable_key=lambda: key)


def _prepared(reverse):
    """用六角色原事实、五个独立观察跨度及单答案WH建立真实生产绑定。"""
    proposition, predicate, source = (1, 501), (5, 601), (70,)
    members = [TrainedInputMember((11,), (4, 101), 4, PROJECTION_FILLER,
                proposition, predicate, source, 0, 1, 0, (8, 1)),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                proposition, predicate, source, 1, 2, 1)]
    members.extend(TrainedInputMember((10 + role,), (4, 100 + role), 4, PROJECTION_FILLER,
        proposition, predicate, source, role, role + 1, role, (8, role)) for role in range(2, 7))
    fact = TrainedInputStructure(proposition, predicate, source, (7, 1), tuple(members))
    surface = SimpleNamespace(proposition=_identity(proposition), predicate=_identity(predicate), source_hash=70,
        bindings=tuple(SimpleNamespace(role=_identity(item.role_key), filler=_identity(item.candidate_key))
                       for item in members if item.projection_kind == PROJECTION_FILLER))
    order = (6, 5, 4, 2, 1) if reverse else (1, 2, 4, 5, 6)
    frame = TrainedRelationQuestionFrame((92002, 501), (92004, 1), predicate,
        tuple((8, role) for role in order), ((), (44,), (44,), (44,), (44,), (88,)), (8, 3), (88,))
    opens = bound_question_open_frames((frame,), (surface,))
    assert len(opens) == 1
    values = tuple(value for index, role in enumerate(order)
                   for value in ((10 + role,) if index == 0 else (44, 10 + role))) + (88,)
    projector = TrainedInputStructureProjector((fact,), open_role_frames=opens, slot_question_frames=(frame,))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge._relation_question_open_frames = opens
    bridge.core_fact_index = {proposition: surface}
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {}
    projected = projector.project(values)
    prepared = bridge._attach_relation_question_binding(bridge._filter_relation_question_open_scope(projected))
    return bridge, frame, prepared, surface


@pytest.mark.parametrize('reverse', (False, True))
def test_multi_observed_wh_binds_and_validates_current_roles_in_trained_order(reverse):
    """完整生产投影、作用域、绑定及槽验证共同支持五观察角色和唯一答案。"""
    bridge, frame, prepared, surface = _prepared(reverse)
    binding, = prepared.relation_question_bindings
    assert len(binding.slot_span_refs) == 5
    assert binding.answer_role_key == (8, 3)
    parts = bridge._relation_slot_question_parts(prepared, binding)
    assert parts is not None and parts[3] == (4, 103) and parts[4] == ()
    spans = {span.ref_key: span for span in prepared.spans}
    assert tuple(spans[ref].values for ref in binding.slot_span_refs) == tuple(
        (10 + role[1],) for role in frame.roles)
    assert tuple(item.role.stable_key() for item in surface.bindings) == tuple((8, role) for role in range(1, 7))
    key = binding.stable_key()
    assert key[0] == 92032
    cursor = 1
    for _ in range(3):
        cursor += key[cursor] + 1
    assert key[cursor] == 5


def test_multi_observed_wh_does_not_adopt_incomplete_or_wrong_answer_binding():
    """类型允许多槽不等于事实授权，消费者仍要求完整有序槽和原答案角色。"""
    bridge, _frame, prepared, _surface = _prepared(True)
    binding, = prepared.relation_question_bindings
    assert bridge._relation_slot_question_parts(prepared,
        replace(binding, slot_span_refs=binding.slot_span_refs[:-1])) is None
    assert bridge._relation_slot_question_parts(prepared,
        replace(binding, answer_role_key=(8, 99))) is None


def test_plain_role_order_extension_does_not_expand_reference_mode():
    """普通受训顺序可改变，显式指代模式仍要求原Core观察角色序。"""
    _bridge, frame, _prepared_input, surface = _prepared(True)
    roles = tuple(item.role.stable_key() for item in surface.bindings)
    assert question_observed_role_contract(frame, roles)
    assert not question_observed_role_contract(replace(frame, slot_mode=SLOT_MODE_REFERENCE), roles)


@pytest.mark.parametrize('roles', [((8, 1),), ((8, 1), (8, 2), (8, 3)), ((8, 1), (8, 1))])
def test_multi_role_contract_rejects_missing_extra_answer_or_duplicate_source(roles):
    """省略一个答案的合同不能接受缺观察角色、答案混入或重复来源角色。"""
    _bridge, frame, _prepared_input, _surface = _prepared(False)
    assert not question_observed_role_contract(frame, roles)


@pytest.mark.parametrize('spans', [(), ((1,), (1,))])
def test_multi_wh_binding_keeps_empty_and_duplicate_span_rejection(spans):
    """解除二元假设后，空槽和重复槽仍在整数绑定构造时拒绝。"""
    with pytest.raises(ValueError, match='slot spans are invalid'):
        InputRelationQuestionBinding((1,), (2,), (3,), spans, (8, 1))
