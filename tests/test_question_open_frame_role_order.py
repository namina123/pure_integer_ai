"""生产问式装配保持已训练的表面次序，不借来源顺序拒绝合法角色绑定。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, TrainedInputStructureProjector,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_question_open_frames import bound_question_open_frames
from pure_integer_ai.experiments.trained_relation_question_graph import MODE_RELATION_PAIR_QUERY
from tests.test_relation_question_graph import _fact, _frame


def _identity(key):
    """只为测试提供固定完整整数身份视图，不创建宿主对象语义。"""
    return SimpleNamespace(stable_key=lambda: key)


def _surface(fact):
    """从同一整数来源成员构造生产装配所消费的只读事实视图。"""
    return SimpleNamespace(proposition=_identity(fact.proposition_key),
        predicate=_identity(fact.predicate_key), source_hash=fact.source_ref[0],
        bindings=tuple(SimpleNamespace(role=_identity(item.role_key),filler=_identity(item.candidate_key))
            for item in fact.members if item.projection_kind == PROJECTION_FILLER))


def test_reverse_complete_frame_survives_production_open_scope_and_binds():
    """真实装配函数生成反向框架，过滤及谓词闭合共同保留精确角色来源。"""
    fact = _fact(source=(70,))
    surface = _surface(fact)
    frame = replace(_frame(), roles=_frame().roles[::-1])
    original_roles = tuple(item.role.stable_key() for item in surface.bindings)
    opens = bound_question_open_frames((frame,), (surface,))
    assert len(opens) == 1 and opens[0].roles == frame.roles
    assert opens[0].gaps == frame.gaps and opens[0].source_ref == fact.source_ref
    assert original_roles == tuple(item.role.stable_key() for item in surface.bindings)
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge._relation_question_open_frames = opens
    bridge.core_fact_index = {fact.proposition_key: surface}
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {frame.operation_key:MODE_RELATION_PAIR_QUERY}
    projector = TrainedInputStructureProjector((fact,),open_role_frames=opens)
    projected = projector.project((22,44,11,55))
    prepared = bridge._attach_relation_question_binding(bridge._filter_relation_question_open_scope(projected))
    assert len(prepared.relation_question_bindings) == 1
    assert prepared.concept_evidence_closed(fact.proposition_key,frozenset({fact.proposition_key}))
    wrong = projector.project((11,44,22,55))
    rejected = bridge._attach_relation_question_binding(bridge._filter_relation_question_open_scope(wrong))
    assert rejected.relation_question_bindings == ()


@pytest.mark.parametrize('drift', ('predicate','missing','extra','duplicate'))
def test_complete_frame_cannot_create_a_different_role_contract(drift):
    """角色集合核验拒绝不同谓词、缺失、多余和重复来源角色。"""
    surface = _surface(_fact(source=(70,)))
    if drift == 'predicate':
        surface.predicate = _identity((5,999))
    elif drift == 'missing':
        surface.bindings = surface.bindings[:1]
    elif drift == 'extra':
        surface.bindings += (SimpleNamespace(role=_identity((8,3))),)
    else:
        surface.bindings = (surface.bindings[0],surface.bindings[0])
    assert bound_question_open_frames((_frame(),),(surface,)) == ()


def test_same_role_order_and_separate_sources_keep_original_members():
    """顺序未变时结果保持；同形不同来源和命题不能合并为一个框架。"""
    first = _surface(_fact(source=(70,)))
    second = _surface(_fact(source=(71,),proposition=(1,51)))
    opens = bound_question_open_frames((_frame(),),(first,second))
    assert {(x.proposition,x.source_ref,x.roles,x.gaps) for x in opens} == {
        (x.proposition.stable_key(),(x.source_hash,),_frame().roles,_frame().gaps)
        for x in (first,second)}


def test_wh_omission_and_source_order_contract_is_unchanged():
    """WH仍须省略一个真实角色，不能凭完整问式修复扩大其许可。"""
    surface = _surface(_fact(source=(70,)))
    frame = replace(_frame(),roles=((8,1),),gaps=((),(44,88)),answer_role_key=(8,2),wh_values=(88,))
    assert len(bound_question_open_frames((frame,),(surface,))) == 1
    assert bound_question_open_frames((replace(frame,answer_role_key=(8,9)),),(surface,)) == ()
