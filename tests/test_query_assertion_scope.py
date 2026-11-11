"""完整作用域、输入来源与持久 O/H/E 立场的专项回归。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE, EVIDENCE_UNKNOWN
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED, TrainedAssertionScopeFrame,
    attach_assertion_scope_frames,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_coverage import has_uncovered_closed_relation_modifier
from pure_integer_ai.experiments.trained_assertion_scope_graph import (
    materialize_assertion_scope_graph, read_assertion_scope_graph,
)
from pure_integer_ai.experiments.trained_assertion_scope_runtime import (
    assertion_scope_feature, assertion_scope_query_records,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_FEATURE_ASSERTION_NEGATED, RESPONSE_FEATURE_ASSERTION_POSSIBLE,
    RESPONSE_FEATURE_ASSERTION_QUOTED,
)


def _fixture(mode):
    """无自然词表的整数训练关系；作用域间隔与已知成员分别提供。"""
    proposition, predicate, source = (1, 50), (5, 60), (9, 70)
    roles = ((8, 1), (8, 2))
    train = TrainedInputStructure(proposition, predicate, source, (7, 1), (
        TrainedInputMember((11,), (4, 101), 4, PROJECTION_FILLER,
            proposition, predicate, source, 0, 1, 0, roles[0]),
        TrainedInputMember((33,), predicate, 5, PROJECTION_PREDICATE,
            proposition, predicate, source, 1, 2, 1),
        TrainedInputMember((22,), (4, 102), 4, PROJECTION_FILLER,
            proposition, predicate, source, 2, 3, 2, roles[1]),
    ))
    projector = TrainedInputStructureProjector((train,))
    syntax = OpenRoleFrame((92386, mode), predicate, (92385, 1), roles,
        ((), (77, 33), ()), (0, 4))
    return projector, TrainedAssertionScopeFrame(syntax, mode, (92384, 1), (91696, 1))


@pytest.mark.parametrize("mode,feature", (
    (SCOPE_NEGATED, RESPONSE_FEATURE_ASSERTION_NEGATED),
    (SCOPE_POSSIBLE, RESPONSE_FEATURE_ASSERTION_POSSIBLE),
    (SCOPE_QUOTED, RESPONSE_FEATURE_ASSERTION_QUOTED),
))
def test_scope_requires_full_input_and_exact_roles(mode, feature):
    """新语法资格与真值分离；额外修饰、未知 filler 和来源漂移不得闭合。"""
    projector, frame = _fixture(mode)
    original = projector.project((11, 77, 33, 22))
    assert has_uncovered_closed_relation_modifier(original)
    bound = attach_assertion_scope_frames((frame,), original)
    assert len(bound.assertion_scope_bindings) == 1
    assert bound.stable_key() != original.stable_key()
    assert not has_uncovered_closed_relation_modifier(bound)
    assert assertion_scope_feature(bound) == feature
    for values in ((11, 77, 33, 22, 88), (88, 11, 77, 33, 22),
                   (11, 77, 77, 33, 22), (99, 77, 33, 22)):
        assert not attach_assertion_scope_frames((frame,), projector.project(values)).assertion_scope_bindings
    foreign = replace(bound.assertion_scope_bindings[0], source_ref=(9, 999))
    with pytest.raises(ValueError, match="来源"):
        replace(bound, assertion_scope_bindings=(foreign,))
    binding, evidence = assertion_scope_query_records(bound, bound.source_ref)[0]
    assert evidence.polarity == EVIDENCE_UNKNOWN
    assert evidence.evidence_key == bound.assertion_scope_bindings[0].stable_key()
    assert binding.filler_key == evidence.evidence_key


def test_competing_modes_and_removed_model_do_not_authorize_scope():
    """不能覆盖并列解释或信任调用方预装的作用域证明。"""
    projector, frame = _fixture(SCOPE_NEGATED)
    raw = projector.project((11, 77, 33, 22))
    competing = replace(frame, mode=SCOPE_POSSIBLE)
    assert not attach_assertion_scope_frames((frame, competing), raw).assertion_scope_bindings
    bound = attach_assertion_scope_frames((frame,), raw)
    assert attach_assertion_scope_frames((), bound) == raw


@pytest.mark.parametrize("mode", (SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED))
def test_scope_memory_keeps_only_source_local_refutation_after_reopen(tmp_path, mode):
    """否定不是事实升级，可能与引述无正支持，冷恢复保留全部立场。"""
    projector, frame = _fixture(mode)
    values = (11, 77, 33, 22)
    bound = attach_assertion_scope_frames((frame,), projector.project(values))
    database = tmp_path / "scoped-memory.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=52) as memory:
        appended = memory.append("".join(map(chr, values)), speaker_kind=1, input_structure=bound)
        observation = memory.intake.result_for_source(appended.source).observation_ref
        original = memory.active_structures(observation_refs=(observation,), context_depth_limit=0)
        stances = [item.stance for item in original.evidence]
        assert EVIDENCE_UNKNOWN in stances and 1 not in stances
        assert len(original.hypotheses) == 3
        assert stances.count(EVIDENCE_REFUTE) == int(mode == SCOPE_NEGATED)
        positive = projector.project((11, 33, 22))
        next_turn = memory.append("".join(map(chr, (11, 33, 22))), speaker_kind=1,
                                  input_structure=positive)
        next_ref = memory.intake.result_for_source(next_turn.source).observation_ref
        later = memory.active_structures(observation_refs=(next_ref,), context_depth_limit=0)
        assert any(item.stance == 1 for item in later.evidence)
        assert all(item.stance in (1, EVIDENCE_UNKNOWN) for item in later.evidence)
        assert not ({item.competition_key for item in original.hypotheses}
                    & {item.competition_key for item in later.hypotheses})
    with TrainedDialogueMemoryGraph(database, session_id=52) as memory:
        restored = memory.active_structures(observation_refs=(observation,), context_depth_limit=0)
        assert restored == original


def test_scope_model_cold_restore_rejects_missing_integer_member(tmp_path):
    """实际整数 SQLite 模型可冷恢复，删去权威成员后必须失败。"""
    import sqlite3

    _, frame = _fixture(SCOPE_NEGATED)
    target = tmp_path / "scope.sqlite3"
    materialize_assertion_scope_graph((frame,), target)
    assert read_assertion_scope_graph(target) == (frame,)
    with sqlite3.connect(target) as connection:
        connection.execute("DELETE FROM assertion_scope_part WHERE ordinal=2")
    with pytest.raises(ValueError, match="连续"):
        read_assertion_scope_graph(target)
