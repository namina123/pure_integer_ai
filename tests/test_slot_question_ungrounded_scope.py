"""单个未落地 WH 解析不能删除原始字面关系而改变缺槽语义。"""
from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from test_slot_question_projection_scope import _nested_fixture


def test_single_ungrounded_parse_keeps_all_original_direct_relation_contributions():
    """训练语法把额外前缀吞入槽时，实际直接角色跨度仍须保留为原候选。"""
    frames, base, _plain = _nested_fixture()
    fact, = base.structures
    frame = replace(frames[0],gaps=((),frames[0].gaps[1]))
    opened = OpenRoleFrame(fact.proposition_key,fact.predicate_key,fact.source_ref,
                           frame.roles,frame.gaps,(0,5))
    projector = TrainedInputStructureProjector((fact,),open_role_frames=(opened,),slot_question_frames=(frame,))
    plain = TrainedInputStructureProjector((fact,),open_role_frames=(opened,))
    values = (44,11,22,55,88)
    assert projector.project(values) == plain.project(values)
