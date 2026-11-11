"""当前指代义务决定历史窗口，无锚点猜测不能扩大 Memory 根。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_reference_context import (
    extended_reference_context_required, reference_marker_sources,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph, _RESPONSE_CONTEXT_HISTORY_LIMIT,
)


def _structure(kind):
    """用真实整数投影区分无锚点、单缺角色、已闭合及三种 WH 合同。"""
    prop, pred, source, roles = (1, 71), (5, 72), (9, 73), ((8, 1), (8, 2))
    members = (
        TrainedInputMember((65,), (4, 81), 4, PROJECTION_FILLER,
                           prop, pred, source, 0, 1, 0, roles[0]),
        TrainedInputMember((82,), pred, 5, PROJECTION_PREDICATE,
                           prop, pred, source, 1, 2, 1),
        TrainedInputMember((66,), (4, 82), 4, PROJECTION_FILLER,
                           prop, pred, source, 2, 3, 2, roles[1]),
    )
    full_frame = OpenRoleFrame(prop, pred, source, roles, ((), (82,), ()), (0, 3))
    frame = TrainedRelationQuestionFrame(
        (92002, 71), (92004, 71), pred, (roles[0],),
        ((), (82, 87)), roles[1], (87,),
        **({"slot_mode": SLOT_MODE_REFERENCE} if kind == "reference" else {}))
    reference_frames = (
        TrainedRelationQuestionFrame((92002, 72), frame.operation_key, pred, (roles[0],),
            ((900, 89), (901,)), roles[1], (89,), SLOT_MODE_REFERENCE),
        TrainedRelationQuestionFrame((92002, 73), frame.operation_key, pred, (roles[1],),
            ((900, 88), (901,)), roles[0], (88,), SLOT_MODE_REFERENCE),
    )
    projector = TrainedInputStructureProjector((
        TrainedInputStructure(prop, pred, source, (7, 1), members),),
        open_role_frames=(full_frame, OpenRoleFrame(
            prop, pred, source, frame.roles, frame.gaps, (0, 3))),
        slot_question_frames=(frame, *reference_frames))
    values = {"unknown": (88, 82, 89), "known": (65, 82, 89),
              "closed": (65, 82, 66), "plain": (65, 82, 87),
              "dynamic": (88, 82, 87), "reference": (65, 82, 87)}[kind]
    projected = projector.project(values)
    if kind in {"plain", "dynamic", "reference"}:
        bindings = match_relation_slot_question_frames(
            (frame,), projected, allow_dynamic_alternatives=True)
        assert len(bindings) == 1
        projected = replace(projected, operation_keys=(frame.operation_key,),
                            relation_question_bindings=bindings)
    return projected, projector


@pytest.mark.parametrize("kind,expected", (("unknown", False), ("known", True),
    ("closed", False), ("plain", False), ("dynamic", True), ("reference", True)))
def test_extended_window_requires_current_reference_evidence(kind, expected):
    structure, projector = _structure(kind)
    assert structure.open_relations
    assert extended_reference_context_required(structure, projector.slot_question_frames) is expected


@pytest.mark.parametrize("field,value", (("source_ref", (9, 999)),
                                        ("proposition_key", (1, 999)),
                                        ("predicate_key", (5, 999))))
def test_unrelated_projection_cannot_authorize_wider_context(field, value):
    structure, projector = _structure("known")
    structure = replace(structure, semantic_candidates=tuple(sorted((
        replace(item, **{field: value}) if item.projection_kind == PROJECTION_FILLER else item
        for item in structure.semantic_candidates), key=lambda item: item.stable_key())))
    assert not extended_reference_context_required(structure, projector.slot_question_frames)


@pytest.mark.parametrize("kind", ("known", "dynamic", "reference"))
def test_untrained_marker_cannot_borrow_history(kind):
    """动态跨度本身和普通 WH 都不能授权历史实体替代未知名称。"""
    structure, projector = _structure(kind)
    plain = tuple(item for item in projector.slot_question_frames if not item.is_reference_slot)
    assert not extended_reference_context_required(structure, plain)


@pytest.mark.parametrize("drift", ("values", "role", "predicate"))
def test_marker_authorization_keeps_exact_role_predicate_and_full_span(drift):
    """已训练指代原子不能授权另一谓词、角色或含额外字符的未知名称。"""
    _structure_value, projector = _structure("dynamic")
    values, role, predicate = (88,), (8, 1), (5, 72)
    assert reference_marker_sources(values, role, predicate, projector.slot_question_frames)
    if drift == "values":
        values = (88, 99)
    elif drift == "role":
        role = (8, 2)
    else:
        predicate = (5, 999)
    assert not reference_marker_sources(values, role, predicate, projector.slot_question_frames)


@pytest.mark.parametrize("kind,limit", (("unknown", 1), ("plain", 1),
    ("known", _RESPONSE_CONTEXT_HISTORY_LIMIT), ("dynamic", _RESPONSE_CONTEXT_HISTORY_LIMIT),
    ("reference", _RESPONSE_CONTEXT_HISTORY_LIMIT)))
def test_memory_append_consumes_reference_window_policy(tmp_path, monkeypatch, kind, limit):
    """真实 SQLite append 用同一当前结构选择历史，观察及 O/H/E 仍然写入。"""
    structure, projector = _structure(kind)
    requested = []
    original = TrainedDialogueMemoryGraph._preceding_response_context_links

    def read_links(memory, *, history_limit=1):
        """记录真实读者收到的窗口，并沿用其原来源核验。"""
        requested.append(history_limit)
        return original(memory, history_limit=history_limit)

    monkeypatch.setattr(TrainedDialogueMemoryGraph, "_preceding_response_context_links", read_links)
    with TrainedDialogueMemoryGraph(tmp_path / "companion.sqlite3", session_id=71,
                                   input_projector=projector) as memory:
        appended = memory.append("".join(map(chr, structure.token_values)),
                                 speaker_kind=1, input_structure=structure)
        assert appended.turn_seq == 1
        assert memory.backend.count("memory_event") > 0
    assert requested == [limit]


@pytest.mark.parametrize("kind,depth", (("unknown", 1), ("known", 1),
    ("closed", 0), ("plain", 0), ("dynamic", 1), ("reference", 1)))
def test_query_restore_consumes_current_reference_obligation(monkeypatch, kind, depth):
    """生产恢复读者不能仅凭开放跨度把普通 WH 升级为历史指代。"""
    from pure_integer_ai.experiments import trained_graph_query_bridge as module

    structure, projector = _structure(kind)
    requested = []

    class RestoreObserved(Exception):
        """在实际恢复调用边界截取参数，避免运行无关的图查询后半段。"""

    def restore(**arguments):
        """捕获真正传给 Memory 读者的图深度，不伪造恢复结果。"""
        requested.append(arguments["context_depth_limit"])
        raise RestoreObserved

    monkeypatch.setattr(module, "derive_discourse_topic_candidates", lambda _structure: ())
    monkeypatch.setattr(module, "graph_backed_discourse_topic_keys", lambda *_arguments: ())
    bridge = module.TrainedGraphQueryBridge.__new__(module.TrainedGraphQueryBridge)
    bridge._reference_runtime = None
    bridge._input_observation_ref = None
    bridge.memory = SimpleNamespace(session_id=71, active_structures=restore)
    bridge.core_runtime = SimpleNamespace(context=SimpleNamespace(graph_ontology=None))
    bridge.relation_question_frames = projector.slot_question_frames
    with pytest.raises(RestoreObserved):
        bridge._restore_memory_structures_impl(structure)
    assert requested == [depth]
