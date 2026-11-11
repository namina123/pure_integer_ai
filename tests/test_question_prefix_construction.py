"""请求前缀构造须携带完整双见证，不扩大事实、指代或组合预算权限。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, QuestionCompositionBudgetExceeded, TrainedRelationQuestionFrame,
)
from pure_integer_ai.cognition.understanding.query_question_prefix import (
    PrefixedSlotQuestionFrame, QuestionPrefixRule, prefixed_slot_question_frames,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    available_slot_question_frames, composed_slot_open_role_frames, restore_slot_question_frame,
)


def _inventory():
    """纯整数来源同时教完整及另一答案角色的前缀对，目标WH组合未直接训练。"""
    full = TrainedRelationQuestionFrame((1,), (2,), (3,), ((4,), (5,)),
                                        ((), (70,), (71,)))
    full_extended = replace(full, source_case_key=(2,), gaps=((90,), *full.gaps[1:]))
    other = TrainedRelationQuestionFrame((3,), (2,), (3,), ((5,),),
                                         ((), (81, 82)), (4,), (82,))
    other_extended = replace(other, source_case_key=(4,), gaps=((90,), *other.gaps[1:]))
    target = TrainedRelationQuestionFrame((5,), (2,), (3,), ((4,),),
                                          ((), (83, 82)), (5,), (82,))
    return full, full_extended, other, other_extended, target


def _prefixed(frames=None, values=(90, 55, 83, 82)):
    """只取新类型；所有旧训练帧仍由现役入口原样提供。"""
    return tuple(frame for frame in available_slot_question_frames(
        _inventory() if frames is None else frames, values)
        if isinstance(frame, PrefixedSlotQuestionFrame))


def test_prefix_composes_new_answer_role_with_complete_integer_sources():
    """新答案角色沿base继承，前缀证据完整编码并冷恢复相同对象。"""
    frame, = _prefixed()
    assert frame.base == _inventory()[-1]
    assert frame.gaps == ((90,), (83, 82))
    assert frame.roles == ((4,),) and frame.answer_role_key == (5,)
    assert frame.wh_values == (82,) and not frame.is_reference_slot
    assert PrefixedSlotQuestionFrame.from_stable_key(frame.stable_key()) == frame
    assert QuestionPrefixRule.from_stable_key(frame.rule.stable_key()) == frame.rule
    assert restore_slot_question_frame(frame.stable_key(), trained_frames=_inventory()) == frame
    assert set(frame.source_frames) == set(_inventory())


@pytest.mark.parametrize('missing', range(5))
def test_prefix_cold_consumer_requires_every_original_model_member(missing):
    """任一base或双见证父帧缺失，都不能借完整stable_key取得挂载资格。"""
    frame, = _prefixed()
    frames = _inventory()
    with pytest.raises(ValueError, match='absent from the mounted model'):
        restore_slot_question_frame(frame.stable_key(), trained_frames=frames[:missing] + frames[missing + 1:])


@pytest.mark.parametrize('missing', range(4))
def test_prefix_discovery_requires_both_complete_and_wh_pairs(missing):
    """只有单类或单端来源时，不把表面差分猜成通用前缀。"""
    frames = _inventory()
    assert not _prefixed(frames[:missing] + frames[missing + 1:])


@pytest.mark.parametrize('change', [
    {'predicate_key': (99,)}, {'operation_key': (99,)},
    {'answer_role_key': (99,)}, {'wh_values': (81,)},
    {'gaps': ((90,), (84, 82))},
    {'slot_mode': SLOT_MODE_REFERENCE},
])
def test_prefix_discovery_rejects_witness_contract_drift(change):
    """谓词、操作、答案、问词、后续间隔或指代模式漂移均破坏成对资格。"""
    frames = _inventory()
    changed = replace(frames[3], **change)
    assert not _prefixed((*frames[:3], changed, frames[4]))


def test_prefix_does_not_authorize_reference_or_other_predicate_target():
    """同一个前缀不能扩大历史指代或授权另一个关系的WH。"""
    frames = _inventory()
    for target in (replace(frames[-1], slot_mode=SLOT_MODE_REFERENCE),
                   replace(frames[-1], predicate_key=(99,))):
        assert not _prefixed((*frames[:-1], target))


def test_prefix_preserves_bound_core_proposition_and_source():
    """生产开放框架只更换问式间隔，原Core命题、来源及观察角色保持。"""
    from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
    frames = _inventory()
    base = frames[-1]
    bound = OpenRoleFrame((301,), base.predicate_key, (401,), base.roles,
                          base.gaps, (0, 4))
    generated, = composed_slot_open_role_frames(frames, (bound,), (90, 55, 83, 82))
    assert generated.proposition == bound.proposition
    assert generated.source_ref == bound.source_ref
    assert generated.roles == bound.roles
    assert generated.gaps == ((90,), (83, 82))


def test_prefix_requires_full_input_and_keeps_original_frame_inventory():
    """未知或非首位前缀不被剥离；已训练帧原样保留且不重复构造原形状。"""
    frames = _inventory()
    assert not _prefixed(values=(91, 55, 83, 82))
    assert not _prefixed(values=(55, 90, 83, 82))
    frame, = _prefixed()
    learned = replace(frames[-1], source_case_key=(6,), gaps=frame.gaps)
    assert not _prefixed((*frames, learned))
    assert set(item for item in frames if item.is_slot_question).issubset(
        set(available_slot_question_frames(frames, (90, 55, 83, 82))))


def test_prefix_consumes_the_existing_composition_budget():
    """旧组合已经花完预算时，新语法构造必须关闭，不能另开一份预算。"""
    with pytest.raises(QuestionCompositionBudgetExceeded):
        prefixed_slot_question_frames(_inventory(), (90, 55, 83, 82),
                                      max_combinations=1, initial_combinations=1)


@pytest.mark.parametrize('prefix', [(), (True,), (0x110000,)])
def test_prefix_rule_rejects_missing_or_non_integer_prefix(prefix):
    """语法对象仍服从严格整数及码点边界，不借Python相等性接受bool。"""
    frames = _inventory()
    with pytest.raises(ValueError):
        QuestionPrefixRule(prefix, ((frames[0], frames[1]),), ((frames[2], frames[3]),))


def test_prefix_grammar_representative_is_independent_of_frame_enumeration_order():
    """相同语法的不同物理来源必须按稳定键选择代表，输入枚举序不改变结果。"""
    frames = (*_inventory(), replace(_inventory()[-1], source_case_key=(6,)))
    forward, = _prefixed(frames)
    reverse, = _prefixed(tuple(reversed(frames)))
    assert forward == reverse
    assert forward.base.source_case_key == (5,)


def test_real_projector_consumes_complete_witnesses_before_role_scope_and_binding():
    """实际投影器须接到完整问式见证，再产生原Core来源的开放角色及WH绑定。"""
    from pure_integer_ai.cognition.understanding.query_input_structure import (
        PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
        TrainedInputStructure, TrainedInputStructureProjector,
    )
    from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
    from pure_integer_ai.cognition.understanding.query_relation_questions import match_relation_slot_question_frames
    from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
    frames = tuple(replace(frame, predicate_key=(5, 3)) for frame in _inventory())
    proposition, predicate, source = (301,), (5, 3), (401,)
    members = (
        TrainedInputMember((55,), (4, 501), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 0, 1, 0, (4,)),
        TrainedInputMember((70,), predicate, 5, PROJECTION_PREDICATE,
                           proposition, predicate, source, 1, 2, 1),
        TrainedInputMember((56,), (4, 502), 4, PROJECTION_FILLER,
                           proposition, predicate, source, 2, 3, 2, (5,)),
    )
    fact = TrainedInputStructure(proposition, predicate, source, (601,), members)
    slots = tuple(frame for frame in frames if frame.is_slot_question)
    opens = tuple(OpenRoleFrame(proposition, predicate, source, frame.roles, frame.gaps, (0, 4))
                  for frame in slots)
    arguments = dict(open_role_frames=opens, slot_question_frames=slots)
    old = TrainedInputStructureProjector((fact,), **arguments)
    projector = TrainedInputStructureProjector((fact,), **arguments,
        complete_question_witness_frames=tuple(frame for frame in frames if not frame.is_slot_question))
    values = (90, 55, 83, 82)
    assert not any(item.frame.gaps == ((90,), (83, 82)) for item in old.project(values).open_relations)
    projected = projector.project(values)
    assert any(item.frame.gaps == ((90,), (83, 82)) and item.frame.proposition == proposition
               and item.frame.source_ref == source for item in projected.open_relations)
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    closed = bridge._close_relation_question_predicate_scope(projected)
    binding, = match_relation_slot_question_frames(frames, closed)
    assert binding.answer_role_key == (5,)
    recovered = restore_slot_question_frame(binding.frame_key, trained_frames=frames)
    assert isinstance(recovered, PrefixedSlotQuestionFrame)


def test_projector_rejects_wh_frames_in_the_complete_witness_parameter():
    """明确分离完整见证与WH帧，不能由新参数绕过原类型门。"""
    from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
    with pytest.raises(TypeError, match='complete_question_witness_frames'):
        TrainedInputStructureProjector((), complete_question_witness_frames=(_inventory()[-1],))
