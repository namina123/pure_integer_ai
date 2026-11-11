"""闭合子串、合法训练间隔和完整当前输入的事实资格回归。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
    graph_query_input_structure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_coverage import (
    has_uncovered_closed_relation_modifier,
)


def _projector(*, gap=(33,)):
    """以无语言词表的整数训练角色和间隔构造真实投影器。"""
    proposition, predicate, source = (1, 50), (5, 60), (9, 70)
    roles = ((8, 1), (8, 2))
    trained = TrainedInputStructure(
        proposition, predicate, source, (7, 1), (
            TrainedInputMember((11,), (4, 101), 4, PROJECTION_FILLER,
                               proposition, predicate, source, 0, 1, 0, roles[0]),
            TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
                               proposition, predicate, source, 1, 2, 1),
            TrainedInputMember((22,), (4, 102), 4, PROJECTION_FILLER,
                               proposition, predicate, source, 2, 3, 2, roles[1]),
        ))
    frame = OpenRoleFrame(proposition, predicate, source, roles,
                          ((), gap, ()), (0, 3))
    return TrainedInputStructureProjector((trained,), open_role_frames=(frame,))


def test_complete_members_and_trained_gap_have_input_coverage():
    """合法训练间隔不要求成为独立谓词或 filler，但角色必须精确绑定。"""
    assert not has_uncovered_closed_relation_modifier(_projector().project((11, 33, 22)))
    assert not has_uncovered_closed_relation_modifier(
        _projector(gap=(77, 33)).project((11, 77, 33, 22)))


def test_closed_substring_does_not_consume_unknown_modifier_or_quoted_context():
    """前缀、内部间隔和后缀都不能靠较短的已知角色取得闭合资格。"""
    projector = _projector()
    for values in ((11, 77, 33, 22), (77, 11, 33, 22),
                   (11, 33, 22, 77), (77, 11, 33, 22, 88)):
        structure = projector.project(values)
        assert any(item.predicate_coverage and item.required_open == 0
                   for item in structure.relation_candidates)
        assert has_uncovered_closed_relation_modifier(structure)


def test_open_relation_and_graph_only_input_are_not_closed_substring_errors():
    """该门只检查错误闭合，不把未闭合输入或图输入当作修饰词问题。"""
    assert not has_uncovered_closed_relation_modifier(_projector().project((11, 33)))
    assert not has_uncovered_closed_relation_modifier(graph_query_input_structure(((1, 2),)))


def test_foreign_frame_source_does_not_license_an_extra_gap():
    """完整框架匹配不得跨物理来源为当前命题补语法资格。"""
    structure = _projector(gap=(77, 33)).project((11, 77, 33, 22))
    assert not has_uncovered_closed_relation_modifier(structure)
    foreign = tuple(replace(item, frame=replace(item.frame, source_ref=(9, 99)))
                    for item in structure.open_relations)
    assert has_uncovered_closed_relation_modifier(replace(structure, open_relations=foreign))


def test_pair_closure_without_predicate_cannot_ignore_untrained_input():
    """无谓词的双端点闭合也必须消费完整输入，不能借局部角色读事实。"""
    projector = _projector()
    for values in ((11, 77, 22), (77, 11, 22), (11, 22, 77)):
        structure = projector.project(values)
        relation, = structure.relation_candidates
        assert relation.required_open == 0 and relation.predicate_coverage == 0
        assert relation.role_coverage == 2
        assert has_uncovered_closed_relation_modifier(structure)
    assert not has_uncovered_closed_relation_modifier(projector.project((11, 22)))
    trained = _projector(gap=(77,)).project((11, 77, 22))
    assert not has_uncovered_closed_relation_modifier(trained)


def test_overlapping_pair_fillers_do_not_cover_unknown_suffix():
    """两个独立角色共用同一表面时，角色数量不能代替区间覆盖。"""
    base = _projector()
    original, = base.structures
    structure = replace(original, members=(
        *original.members[:-1], replace(original.members[-1], values=(11,))))
    projector = TrainedInputStructureProjector((structure,))
    for values in ((11, 77), (77, 11), (77, 11, 88)):
        projected = projector.project(values)
        relation, = projected.relation_candidates
        assert relation.predicate_coverage == 0 and relation.role_coverage == 2
        assert relation.required_count == relation.support_count == 2
        assert has_uncovered_closed_relation_modifier(projected)
    assert not has_uncovered_closed_relation_modifier(projector.project((11,)))


def test_pair_closure_uncovered_input_persists_unknown_memory(tmp_path):
    """共享门在真实Memory写入及冷恢复中保持无谓词局部闭合为UNKNOWN。"""
    from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_UNKNOWN
    from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
        TrainedDialogueMemoryGraph,
    )
    values = (11, 77, 22)
    structure = _projector().project(values)
    database = tmp_path / "uncovered-pair-memory.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=51) as memory:
        appended = memory.append("".join(map(chr, values)), speaker_kind=1,
                                 input_structure=structure)
        observation = memory.intake.result_for_source(appended.source).observation_ref
        active = memory.active_structures(observation_refs=(observation,))
        assert active.hypotheses and active.evidence
        assert all(item.stance == EVIDENCE_UNKNOWN for item in active.evidence)
    with TrainedDialogueMemoryGraph(database, session_id=51) as restored:
        active = restored.active_structures(observation_refs=(observation,))
        assert active.hypotheses and active.evidence
        assert all(item.stance == EVIDENCE_UNKNOWN for item in active.evidence)


def test_missing_outer_carrier_keeps_all_trained_inner_gaps():
    """省略来源终止符时，必须仍消费内部训练间隔并精确绑定两个角色。"""
    base = _projector(gap=(77, 33))
    inner = base.open_role_frames[0]
    full = replace(inner, gaps=(*inner.gaps[:-1], (88,)))
    projector = TrainedInputStructureProjector(
        base.structures, open_role_frames=(full,), envelope_role_frames=(inner,))
    body = projector.project((11, 77, 33, 22))
    assert not has_uncovered_closed_relation_modifier(body)
    assert body.open_relations and all(item.frame == inner for item in body.open_relations)
    full_input = projector.project((11, 77, 33, 22, 88))
    assert not has_uncovered_closed_relation_modifier(full_input)
    assert all(item.frame == full for item in full_input.open_relations)
    assert has_uncovered_closed_relation_modifier(projector.project((11, 99, 77, 33, 22)))


def test_uncovered_input_keeps_unknown_memory_evidence_after_reopen(tmp_path):
    """未消费区间不能作为正事实写进 O/H/E，且冷恢复保持其未知状态。"""
    from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_UNKNOWN
    from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
        TrainedDialogueMemoryGraph,
    )
    values = (11, 77, 33, 22)
    structure = _projector().project(values)
    database = tmp_path / "uncovered-memory.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=50) as memory:
        appended = memory.append("".join(map(chr, values)), speaker_kind=1,
                                 input_structure=structure)
        observation = memory.intake.result_for_source(appended.source).observation_ref
        active = memory.active_structures(observation_refs=(observation,))
        assert active.hypotheses and active.evidence
        assert all(item.stance == EVIDENCE_UNKNOWN for item in active.evidence)
    with TrainedDialogueMemoryGraph(database, session_id=50) as restored:
        active = restored.active_structures(observation_refs=(observation,))
        assert active.hypotheses and active.evidence
        assert all(item.stance == EVIDENCE_UNKNOWN for item in active.evidence)
