"""自然纠错否定的物理模型资格、兼容身份及追加证据闭环。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE, EVIDENCE_SUPPORT
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED, attach_assertion_scope_frames,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import memory_input_candidate_keys
from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_PREDICATE, TrainedInputStructureProjector
from pure_integer_ai.experiments.natural_revision_runtime import NaturalRevisionBindingError, NaturalRevisionRuntime
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_revision_cue_graph import RevisionCueGraphRecord, TrainedRevisionCue
from test_query_assertion_scope import _fixture


def _setup(mode=SCOPE_NEGATED):
    """来源化整数语法组合，不在运行时新增自然语言判断规则。"""
    projector, frame = _fixture(mode)
    frame = replace(frame, syntax=replace(frame.syntax, gaps=((91,), *frame.syntax.gaps[1:])))
    record = RevisionCueGraphRecord(TrainedRevisionCue((91690, 1), (91,), (91690, 10)),
                                    (91660, 1), (91696, 1))
    runtime = NaturalRevisionRuntime((record,), scope_frames=(frame,))
    raw = projector.project((91, 11, 77, 33, 22))
    return projector, frame, record, runtime, attach_assertion_scope_frames((frame,), raw)


def test_negated_revision_supersedes_support_and_keeps_original_after_cold_restore(tmp_path):
    """同一命题的用户肯定撤回只追加反驳，原支持及来源仍可完整恢复。"""
    projector, _, _, runtime, corrected = _setup()
    binding = runtime.bind_input(corrected, memory_input_structure=corrected)
    assert binding.replacement_stance == EVIDENCE_REFUTE
    assert binding.stable_key()[0] == 4
    assert binding.scope_binding_key == corrected.assertion_scope_bindings[0].stable_key()
    database = tmp_path / "negated-revision.sqlite3"
    with TrainedDialogueMemoryGraph(database, session_id=14, input_projector=projector) as memory:
        memory.append("".join(map(chr, (11, 33, 22))), speaker_kind=1,
                      input_structure=projector.project((11, 33, 22)))
        target = runtime.resolve_target(memory, corrected, binding)
        appended = memory.append("".join(map(chr, corrected.token_values)), speaker_kind=1,
                                 input_structure=corrected)
        commit = runtime.commit(memory, target, appended.source)
    with TrainedDialogueMemoryGraph(database, session_id=14, input_projector=projector) as memory:
        snapshot = memory.active_structures(candidate_keys=memory_input_candidate_keys(
            corrected, session_id=14, speaker_kind=1))
        prior = next(item for item in snapshot.evidence if item.evidence_key == target.prior_evidence_key)
        current = next(item for item in snapshot.evidence if item.evidence_key == commit.replacement_evidence_key)
        assert prior.stance == EVIDENCE_SUPPORT
        assert current.stance == EVIDENCE_REFUTE
        assert current.supersedes_key == prior.evidence_key
        assert current.hypothesis_key == prior.hypothesis_key
        assert current.source_ref[1:] == appended.source.stable_key()


@pytest.mark.parametrize("mode", (SCOPE_POSSIBLE, SCOPE_QUOTED))
def test_nonassertive_revision_is_rejected_before_memory_write(tmp_path, mode):
    """可能与引述不是撤回；拒绝具有明确整数原因且不产生事件。"""
    _, _, _, runtime, corrected = _setup(mode)
    with TrainedDialogueMemoryGraph(tmp_path / "nonassertive.sqlite3", session_id=15) as memory:
        before = memory.backend.count("memory_event"), memory._next_turn_seq
        with pytest.raises(NaturalRevisionBindingError) as error:
            runtime.bind_input(corrected, memory_input_structure=corrected)
        assert error.value.reason_code == 4
        assert (memory.backend.count("memory_event"), memory._next_turn_seq) == before


def test_revision_rejects_unmounted_missing_competing_or_inexact_scope():
    """预装、删模型、双解释和额外输入都不能替代独立完整重匹配。"""
    projector, frame, record, runtime, corrected = _setup()
    for frames in ((), (replace(frame, source_identity=(92384, 2)),),
                   (frame, replace(frame, mode=SCOPE_POSSIBLE))):
        other = NaturalRevisionRuntime((record,), scope_frames=frames)
        with pytest.raises(NaturalRevisionBindingError) as error:
            other.bind_input(corrected)
        assert error.value.reason_code == 4
    with pytest.raises(NaturalRevisionBindingError):
        runtime.bind_input(replace(corrected, assertion_scope_bindings=()))
    competing_frames = (frame, replace(frame, mode=SCOPE_POSSIBLE))
    competing_input = attach_assertion_scope_frames(competing_frames, corrected)
    assert not competing_input.assertion_scope_bindings
    with pytest.raises(NaturalRevisionBindingError, match="competing"):
        NaturalRevisionRuntime((record,), scope_frames=competing_frames).bind_input(competing_input)
    extra = projector.project((*corrected.token_values, 88))
    assert not attach_assertion_scope_frames((frame,), extra).assertion_scope_bindings
    binding = runtime.bind_input(corrected)
    with pytest.raises(ValueError):
        replace(binding, scope_binding_key=())
    with pytest.raises(ValueError):
        replace(binding, input_source_ref=(9, 888))
    with pytest.raises(ValueError):
        replace(binding, replacement_stance=True)


def test_support_binding_keeps_original_canonical_bytes():
    """旧无作用域支持的规范身份逐项保持原版本一，不能因新字段漂移。"""
    projector, _, _, runtime, _ = _setup()
    binding = runtime.bind_input(projector.project((91, 11, 33, 22)))
    pack = lambda key: (len(key), *key)
    expected = (1, *pack(binding.cue_identity), *pack(binding.action_key),
        *pack(binding.proposition_key), *pack(binding.relation_key),
        *pack(binding.input_source_ref), binding.cue_start, binding.cue_end,
        *pack(binding.source_identity), *pack(binding.license_identity))
    assert binding.replacement_stance == EVIDENCE_SUPPORT
    assert not binding.scope_binding_key
    assert binding.stable_key() == expected


def test_negated_missing_target_and_substituted_binding_reject_before_write(tmp_path):
    """缺先前目标与输入绑定漂移分别拒绝，禁止在解析失败后追加记忆。"""
    _, _, _, runtime, corrected = _setup()
    binding = runtime.bind_input(corrected)
    with TrainedDialogueMemoryGraph(tmp_path / "empty.sqlite3", session_id=16) as memory:
        before = memory.backend.count("memory_event"), memory._next_turn_seq
        with pytest.raises(NaturalRevisionBindingError) as error:
            runtime.resolve_target(memory, corrected, binding)
        assert error.value.reason_code == 2
        with pytest.raises(NaturalRevisionBindingError, match="actual input"):
            runtime.resolve_target(memory, corrected, replace(binding, cue_identity=(91690, 999)))
        assert (memory.backend.count("memory_event"), memory._next_turn_seq) == before


def test_unknown_only_later_input_does_not_mask_the_preceding_assertion(tmp_path):
    """原失败输入的初始UNKNOWN保留，但不能遮蔽真实先前断言。"""
    projector, _, _, runtime, corrected = _setup()
    with TrainedDialogueMemoryGraph(tmp_path / "unknown-between.sqlite3", session_id=17,
                                    input_projector=projector) as memory:
        original = memory.append("".join(map(chr, (11,33,22))), speaker_kind=1,
                                 input_structure=projector.project((11,33,22)))
        failed = projector.project((91,11,77,33,22))
        memory.append("".join(map(chr, failed.token_values)), speaker_kind=1,input_structure=failed)
        binding = runtime.bind_input(corrected)
        target = runtime.resolve_target(memory, corrected, binding)
        assert target.prior_source_ref[1:] == original.source.stable_key()
        before = memory.backend.count("memory_event")
        memory.retract("".join(map(chr,(11,33,22))), speaker_kind=1)
        with pytest.raises(NaturalRevisionBindingError, match="competing"):
            runtime.resolve_target(memory, corrected, binding)
        assert memory.backend.count("memory_event") > before


def test_complete_scope_selects_its_relation_without_discarding_other_projections():
    """独立完整作用域授权关系，不把其它未授权投影总数当作竞争断言。"""
    projector, frame, record, _, _ = _setup()
    trained = projector.structures[0]
    other = replace(trained, proposition_key=(1,51), predicate_key=(5,61),
        members=tuple(replace(member, proposition_key=(1,51),predicate_key=(5,61),
            candidate_key=(5,61) if member.projection_kind==PROJECTION_PREDICATE else member.candidate_key,
            values=(44,) if member.projection_kind==PROJECTION_PREDICATE else member.values)
            for member in trained.members))
    combined = TrainedInputStructureProjector((trained,other))
    raw = combined.project((91,11,77,33,22))
    bound = attach_assertion_scope_frames((frame,),raw)
    assert len(bound.relation_candidates)>1
    assert len(bound.assertion_scope_bindings)==1
    runtime=NaturalRevisionRuntime((record,),scope_frames=(frame,))
    binding=runtime.bind_input(bound,memory_input_structure=bound)
    assert binding.relation_key==bound.assertion_scope_bindings[0].relation_key
    assert binding.proposition_key==trained.proposition_key
    assert len(bound.relation_candidates)==len(raw.relation_candidates)
    competing=replace(other,predicate_key=trained.predicate_key,
        members=tuple(replace(member,predicate_key=trained.predicate_key,
            candidate_key=trained.predicate_key if member.projection_kind==PROJECTION_PREDICATE else member.candidate_key,
            values=(33,) if member.projection_kind==PROJECTION_PREDICATE else member.values)
            for member in other.members))
    raw_competing=TrainedInputStructureProjector((trained,competing)).project(raw.token_values)
    unresolved=attach_assertion_scope_frames((frame,),raw_competing)
    assert not unresolved.assertion_scope_bindings
    with pytest.raises(NaturalRevisionBindingError):runtime.bind_input(unresolved)
