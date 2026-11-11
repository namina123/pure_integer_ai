"""完整落地WH只排除零锚点吞入，不选择同名来源或扩大指代权限。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE,
    TrainedInputMember, TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_question_syntax_scope import grounded_question_open_scope
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import available_slot_question_frames
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_question_open_frames import bound_question_open_frames


def _identity(key):
    """测试仅用完整整数身份，不依赖宿主对象比较。"""
    return SimpleNamespace(stable_key=lambda: key)


def _fixture():
    """两个同名不同对象及一个无关来源，各有精确和吞入式WH语法。"""
    facts, surfaces = [], []
    predicate = (5, 60)
    for ordinal, value in ((1, 11), (2, 11), (3, 66)):
        proposition, source = (1, 50 + ordinal), (90 + ordinal,)
        members = (
            TrainedInputMember((value,), (4, 100 + ordinal), 4, PROJECTION_FILLER,
                proposition, predicate, source, 0, 1, 0, (8, 1)),
            TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                proposition, predicate, source, 1, 2, 1),
            TrainedInputMember((20 + ordinal,), (4, 200 + ordinal), 4, PROJECTION_FILLER,
                proposition, predicate, source, 2, 3, 2, (8, 2)),
        )
        facts.append(TrainedInputStructure(proposition, predicate, source, (7, 1), members))
        surfaces.append(SimpleNamespace(proposition=_identity(proposition), predicate=_identity(predicate),
            source_hash=source[0], bindings=tuple(SimpleNamespace(role=_identity(m.role_key),
                filler=_identity(m.candidate_key)) for m in members if m.projection_kind == PROJECTION_FILLER)))
    exact = TrainedRelationQuestionFrame((92002, 1), (92004, 1), predicate,
        ((8, 1),), ((44,), (55, 88)), (8, 2), (88,))
    swallowed = replace(exact, source_case_key=(92002, 2), gaps=((), (88,)))
    frames = (exact, swallowed)
    opened = bound_question_open_frames(frames, surfaces)
    projector = TrainedInputStructureProjector(tuple(facts), open_role_frames=opened,
        slot_question_frames=frames)
    structure = projector.project((44, 11, 55, 88))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    bridge._relation_question_open_frames = opened
    bridge.core_fact_index = {s.proposition.stable_key(): s for s in surfaces}
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {}
    by_shape = {(f.predicate_key, f.roles, f.gaps): (f,)
                for f in available_slot_question_frames(frames, structure.token_values)}
    return bridge, structure, frames, by_shape


def test_production_filter_keeps_both_grounded_sources_and_no_arbitrary_answer():
    """生产过滤只保留两个精确来源，绑定仍因不同对象竞争而为空，并且幂等。"""
    bridge, structure, _frames, _shapes = _fixture()
    assert len(structure.open_relations) == 12
    filtered = bridge._filter_relation_question_open_scope(structure)
    assert {c.frame.proposition for c in filtered.open_relations} == {(1, 51), (1, 52)}
    assert len(filtered.open_relations) == 2
    assert all(c.bindings[0].values == (11,) for c in filtered.open_relations)
    assert filtered.token_values == structure.token_values and filtered.spans == structure.spans
    prepared = bridge._attach_relation_question_binding(filtered)
    assert prepared.relation_question_bindings == ()
    assert bridge._filter_relation_question_open_scope(prepared) == prepared
    assert bridge._attach_relation_question_binding(prepared) == prepared


def test_without_complete_grounded_witness_all_unknown_candidates_remain():
    """无完整独立filler见证时不删除未知问式、不凭纯语法提供新事实。"""
    _bridge, structure, frames, shapes = _fixture()
    unknown = replace(structure, semantic_candidates=())
    assert grounded_question_open_scope(structure.open_relations, unknown, shapes, frames) == structure.open_relations


def test_same_interval_without_its_own_source_filler_does_not_borrow_another_fact():
    """同一输入跨度也必须有该命题、谓词、来源和Role自己的独立投影。"""
    _bridge, structure, frames, shapes = _fixture()
    selected = grounded_question_open_scope(structure.open_relations, structure, shapes, frames)
    assert len(selected) == 2
    assert all(c.frame.proposition != (1, 53) for c in selected)


@pytest.mark.parametrize('reference_mode', (False, True))
def test_trained_reference_candidates_are_preserved(reference_mode):
    """显式reference和受训marker授权的plain动态观察均不能被落地问式裁掉。"""
    _bridge, structure, frames, shapes = _fixture()
    unknown = next(c for c in structure.open_relations if c.bindings[0].values == (44, 11, 55))
    marker = replace(frames[1], source_case_key=(92002, 3), slot_mode=SLOT_MODE_REFERENCE,
        answer_role_key=(8, 1), roles=((8, 2),), wh_values=unknown.bindings[0].values,
        gaps=((), unknown.bindings[0].values))
    if reference_mode:
        shapes = dict(shapes)
        shapes[(unknown.frame.predicate, unknown.frame.roles, unknown.frame.gaps)] = (replace(frames[1], slot_mode=SLOT_MODE_REFERENCE),)
    selected = grounded_question_open_scope(structure.open_relations, structure, shapes, (*frames, marker))
    assert unknown in selected


def test_partial_grounding_remains_a_candidate_and_no_witness_keeps_order():
    """只排除零对齐，两个角色只落地一个时仍保留原动态解析。"""
    _bridge, structure, frames, shapes = _fixture()
    known = next(c for c in structure.open_relations if c.frame.proposition == (1, 51)
                 and c.bindings[0].values == (11,))
    unknown = next(c for c in structure.open_relations if c.bindings[0].values == (44, 11, 55))
    role = replace(unknown.bindings[0], role=(8, 3))
    mixed = SimpleNamespace(frame=replace(known.frame, roles=(*known.frame.roles, (8, 3)),
        gaps=(*known.frame.gaps, ())), bindings=(*known.bindings, role))
    shape = (mixed.frame.predicate, mixed.frame.roles, mixed.frame.gaps)
    shapes = dict(shapes)
    shapes[shape] = (replace(frames[0], roles=((8, 1), (8, 3)), gaps=mixed.frame.gaps),)
    candidates = (*structure.open_relations, mixed)
    assert mixed in grounded_question_open_scope(candidates, structure, shapes, frames)
    assert grounded_question_open_scope(iter(candidates), replace(structure, semantic_candidates=()), shapes, frames) == candidates
