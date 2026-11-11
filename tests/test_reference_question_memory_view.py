"""真实投影和 SQLite 角色发生验证多来源指代候选交接。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_PROPOSITION, ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_question_memory_view import (
    merge_reference_question_views,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_response_occurrence import (
    materialize_input_roles, read_input_roles,
)


def _fixture(*, same_answer=False, distinct_topology=False, reverse_source_members=False,
             predicate_in_gap=False, composed_question=False, ordinary_parse=False):
    """两个不同来源和答案共享观察端点，不能在 Memory 前删掉竞争。"""
    roles = tuple(role_identity((value,)) for value in (1, 2))
    predicate = ObjectIdentity(OBJECT_CONCEPT, (60,))
    observed = ObjectIdentity(OBJECT_ENTITY, (102,))
    question = TrainedRelationQuestionFrame(
        (92002, 1), (92004, 1), predicate.stable_key(), (roles[1].stable_key(),),
        ((99, 33), (88,)), roles[0].stable_key(), (99,), SLOT_MODE_REFERENCE)
    questions = (question,)
    if composed_question:
        questions = (*questions, replace(question, source_case_key=(92002, 2),
                                        gaps=((77, 99, 33), (66,))))
    structures, open_frames, facts = [], [], {}
    for index, value in enumerate((11, 66), 1):
        proposition = ObjectIdentity(OBJECT_PROPOSITION, (50 + index,))
        answer = ObjectIdentity(OBJECT_ENTITY, (201 if same_answer else 200 + index,))
        prop, pred, source = proposition.stable_key(), predicate.stable_key(), (70 + index,)
        members = (
            TrainedInputMember((value,), answer.stable_key(), OBJECT_ENTITY, PROJECTION_FILLER,
                               prop, pred, source, 0, 1, 0, roles[0].stable_key()),
            TrainedInputMember((22,), observed.stable_key(), OBJECT_ENTITY, PROJECTION_FILLER,
                               prop, pred, source, 1, 2, 1, roles[1].stable_key()),
            TrainedInputMember((44,) if predicate_in_gap else (33,), pred,
                               OBJECT_CONCEPT, PROJECTION_PREDICATE,
                               prop, pred, source, 2, 3, 2),
        )
        if reverse_source_members and index == 2:
            members = (replace(members[1], trained_start=0, trained_end=1, member_ordinal=0),
                       replace(members[0], trained_start=1, trained_end=2, member_ordinal=1),
                       members[2])
        structures.append(TrainedInputStructure(
            prop, pred, source, (7, index if distinct_topology else 1), members))
        open_frames.append(OpenRoleFrame(prop, pred, source, question.roles,
                                         question.gaps, (0, 5)))
        if composed_question:
            second = questions[1]
            open_frames.append(OpenRoleFrame(
                prop, pred, source, second.roles, second.gaps,
                (0, len(second.roles) + sum(map(len, second.gaps)))))
        if ordinary_parse:
            open_frames.append(OpenRoleFrame(
                prop, pred, source, tuple(role.stable_key() for role in roles),
                ((), (33,), (88,)), (0, 4)))
        facts[prop] = SimpleNamespace(
            proposition=proposition, predicate=predicate, source_hash=source[0],
            bindings=(SimpleNamespace(role=roles[0], filler=answer),
                      SimpleNamespace(role=roles[1], filler=observed)))
        if reverse_source_members and index == 2:
            facts[prop].bindings = tuple(reversed(facts[prop].bindings))
    projector = TrainedInputStructureProjector(tuple(structures),
        open_role_frames=tuple(open_frames), slot_question_frames=questions)
    bridge = TrainedGraphQueryBridge.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = questions
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {}
    bridge.core_fact_index = facts
    return bridge, projector, roles


@pytest.mark.parametrize("predicate_in_gap", (False, True))
def test_exact_reference_question_replaces_initial_broad_core_parse(predicate_in_gap):
    """首次由精确问式保留代词边界，之后仍拒绝较宽完整切分混入复核。"""
    from pure_integer_ai.cognition.understanding.query_question_memory_view import (
        retained_reference_question_gap_witness,
    )
    bridge, projector, roles = _fixture(
        predicate_in_gap=predicate_in_gap, composed_question=True, ordinary_parse=True)
    raw = projector.project((77, 99, 33, 22, 88))
    broad = tuple(item for item in raw.open_relations
                  if any(part.role == roles[0].stable_key() and part.values == (77, 99)
                         for part in item.bindings))
    assert len(broad) == 2
    prepared = bridge._attach_relation_question_binding(raw)
    assert len(prepared.relation_question_bindings) == 2
    assert len(prepared.open_relations) == 2
    assert all(any(part.role == roles[0].stable_key() and part.values == (99,)
                   for part in item.bindings) for item in prepared.open_relations)
    assert bridge._attach_relation_question_binding(prepared) == prepared
    binding = prepared.relation_question_bindings[0]
    frame, relation, *_rest = bridge._relation_slot_question_parts(prepared, binding)
    assert retained_reference_question_gap_witness(prepared, frame, relation, binding)
    competing = next(item for item in broad
                     if item.frame.proposition == relation.proposition_key)
    changed = replace(prepared, open_relations=tuple(sorted(
        (*prepared.open_relations, competing), key=lambda item: item.stable_key())))
    assert not retained_reference_question_gap_witness(changed, frame, relation, binding)
    assert bridge._relation_slot_question_view(changed, binding) is None


@pytest.mark.parametrize("predicate_in_gap", (False, True))
@pytest.mark.parametrize("composed_question", (False, True))
def test_multiple_reference_bindings_reach_memory_and_cold_restore(
        tmp_path, predicate_in_gap, composed_question):
    """重复 prepare/intake 不丢指代角色，两来源的角色发生可独立冷恢复。"""
    bridge, projector, roles = _fixture(predicate_in_gap=predicate_in_gap,
                                      composed_question=composed_question)
    values = (77, 99, 33, 22, 88) if composed_question else (99, 33, 22, 88)
    prepared = bridge._attach_relation_question_binding(projector.project(values))
    assert len(prepared.relation_question_bindings) == 2
    assert len(prepared.open_relations) == 2
    assert len({item.frame.source_ref for item in prepared.open_relations}) == 2
    assert all(any(item.role == roles[0].stable_key() and item.values == (99,)
                   for item in candidate.bindings) for candidate in prepared.open_relations)
    assert bridge._attach_relation_question_binding(prepared) == prepared
    assert bridge.active_memory_input_structure(prepared) == prepared
    assert not any(item.role_key == roles[0].stable_key()
                   for item in prepared.semantic_candidates if item.projection_kind == PROJECTION_FILLER)
    database = tmp_path / "companion.sqlite3"
    saved = []
    with TrainedDialogueMemoryGraph(database, session_id=71, input_projector=projector) as memory:
        appended = memory.append("".join(map(chr, prepared.token_values)),
                                 speaker_kind=1, input_structure=prepared)
        observation = memory.intake.result_for_source(appended.source).observation_ref
        selected = memory.active_structures(observation_refs=(observation,))
        for candidate in prepared.open_relations:
            key = frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,),
                            candidate.stable_key())
            hypothesis, = (MemoryObjectRef.from_stable_key(item.hypothesis_key)
                           for item in selected.hypotheses if item.candidate_key == key)
            occurrences = materialize_input_roles(memory, candidate, observation, hypothesis)
            assert {item.role for item in occurrences} == set(roles)
            saved.append((candidate, observation, hypothesis, occurrences))
    with TrainedDialogueMemoryGraph(database, session_id=71, input_projector=projector) as memory:
        for candidate, observation, hypothesis, occurrences in saved:
            assert read_input_roles(memory, candidate, observation, hypothesis,
                                    required=True) == occurrences


@pytest.mark.parametrize("drift", (
    "physical_source", "input_source", "marker", "marker_span", "role_pairing",
    "missing_projection", "competing_projection", "slot_span", "unbound",
))
def test_reference_gap_witness_rejects_incomplete_or_drifted_current_projection(drift):
    """谓词只在 gap 中时，当前完整候选、marker 和来源仍必须逐项闭合。"""
    from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles
    from pure_integer_ai.cognition.understanding.query_question_memory_view import (
        retained_reference_question_gap_witness,
    )
    bridge, projector, _roles = _fixture(predicate_in_gap=True)
    prepared = bridge._attach_relation_question_binding(projector.project((99, 33, 22, 88)))
    binding = prepared.relation_question_bindings[0]
    frame, relation, *_rest = bridge._relation_slot_question_parts(prepared, binding)
    assert retained_reference_question_gap_witness(prepared, frame, relation, binding)
    original = next(item for item in prepared.open_relations
                    if item.frame.proposition == relation.proposition_key)
    changed = original
    if drift == "physical_source":
        changed = replace(original, frame=replace(original.frame, source_ref=(999,)))
    elif drift == "input_source":
        changed = replace(original, source_ref=(999,), bindings=tuple(
            replace(item, source_ref=(999,)) for item in original.bindings))
    elif drift == "marker":
        changed, = match_open_roles((original.frame,), (100, 33, 22, 88), original.source_ref)
    elif drift == "marker_span":
        changed, = match_open_roles((replace(original.frame,
            gaps=((99,), (), (88,))),), original.values, original.source_ref)
    elif drift == "role_pairing":
        changed, = match_open_roles((replace(original.frame,
            roles=tuple(reversed(original.frame.roles))),), original.values, original.source_ref)
    if drift in {"input_source", "marker"}:
        # 完整输入合同先于问式见证生效，不能绕过其校验来构造测试输入。
        with pytest.raises(ValueError, match="开放角色假设不得跨来源或丢弃输入"):
            replace(prepared, open_relations=tuple(sorted(
                (changed if item == original else item for item in prepared.open_relations),
                key=lambda item: item.stable_key())))
        return
    if drift == "missing_projection":
        prepared = replace(prepared, open_relations=tuple(
            item for item in prepared.open_relations if item != original))
    elif drift == "competing_projection":
        changed = replace(original, frame=replace(original.frame, source_ref=(999,)))
        prepared = replace(prepared, open_relations=tuple(sorted(
            (*prepared.open_relations, changed), key=lambda item: item.stable_key())))
    elif drift == "slot_span":
        binding = replace(binding, slot_span_refs=((999,),))
    elif drift == "unbound":
        prepared = replace(prepared, relation_question_bindings=())
    else:
        prepared = replace(prepared, open_relations=tuple(sorted(
            (changed if item == original else item for item in prepared.open_relations),
            key=lambda item: item.stable_key())))
    assert not retained_reference_question_gap_witness(prepared, frame, relation, binding)
    if drift != "unbound":
        assert bridge._relation_slot_question_parts(prepared, binding) is None


@pytest.mark.parametrize("kind", ("missing", "duplicate", "foreign"))
def test_reference_view_merge_rejects_partial_or_foreign_scope(kind):
    """不能用部分覆盖或另一输入的视图悄悄裁决候选。"""
    bridge, projector, _roles = _fixture()
    prepared = bridge._attach_relation_question_binding(projector.project((99, 33, 22, 88)))
    views = tuple(bridge._relation_slot_question_view(prepared, binding)
                  for binding in prepared.relation_question_bindings)
    if kind == "missing":
        views = views[:1]
    elif kind == "duplicate":
        views = (views[0], views[0])
    else:
        views = (replace(views[0], graph_object_keys=((91, 92),)), views[1])
    with pytest.raises(ValueError):
        merge_reference_question_views(prepared, views)


@pytest.mark.parametrize("case,expected_consensus", (
    ("same_antecedent", True), ("distinct_targets", False),
    ("distinct_antecedents", False), ("multiple_supporting_occurrences", True),
    ("distinct_support_sets", False),
    ("distinct_topology", False),
    ("source_member_order", True),
))
def test_memory_grounded_source_consensus_requires_complete_same_history_support(case, expected_consensus):
    """来源可不同，目标及完整历史支持集合必须一致，不能按最近/最小键消歧。"""
    bridge, projector, roles = _fixture(same_answer=case != "distinct_targets",
                                      distinct_topology=case == "distinct_topology",
                                      reverse_source_members=case == "source_member_order")
    prepared = bridge._attach_relation_question_binding(projector.project((99, 33, 22, 88)))
    first_target = next(iter(bridge.core_fact_index.values())).bindings[0].filler
    targets = tuple(fact.bindings[0].filler for fact in bridge.core_fact_index.values())
    candidate_sets = []
    contexts = {}
    for index, candidate in enumerate(prepared.open_relations):
        antecedent = SimpleNamespace(values=(700 + index if case == "distinct_targets" else 700,))
        antecedent.stable_key = lambda index=index: (
            91535, index if case == "distinct_antecedents" else 1)
        antecedents = (antecedent,)
        if case == "multiple_supporting_occurrences" or (case == "distinct_support_sets" and index == 0):
            extra = SimpleNamespace(values=(700,), stable_key=lambda: (91535, 2))
            antecedents = (antecedent, extra)
        candidates = SimpleNamespace(
            reference=SimpleNamespace(role=roles[0], ordinal=0, values=(99,)),
            antecedents=antecedents, stable_key=lambda index=index: (91535, 100 + index))
        candidate_sets.append(candidates)
        contexts[id(candidates)] = candidate
    bridge._reference_candidate_sets = tuple(candidate_sets)
    bridge._reference_runtime = SimpleNamespace(
        _occurrence_values=lambda item: item.values,
        _semantic_objects=lambda values: (targets[values[0] - 700],)
            if case == "distinct_targets" else (first_target,),
        semantic_objects_for=lambda item, _contexts: (targets[item.values[0] - 700],)
            if case == "distinct_targets" else (first_target,))
    bridge._response_contexts = ()
    bridge._input_observation_ref = object()
    bridge._reference_relation_context = lambda item, _structure: (contexts[id(item)], ())
    selected = bridge._memory_grounded_slot_binding(prepared)
    if expected_consensus:
        consensus = bridge._memory_grounded_reference_source_consensus
        assert selected.stable_key() == consensus.selected_binding_key
        assert len(consensus.members) == 2
        assert {item.binding_key for item in consensus.members} == {
            item.stable_key() for item in prepared.relation_question_bindings}
        expected = ((91535, 1), (91535, 2)) if case == "multiple_supporting_occurrences" else ((91535, 1),)
        assert all(item.antecedent_keys == expected for item in consensus.members)
    else:
        assert selected is None
        assert getattr(bridge, "_memory_grounded_reference_source_consensus", None) is None


@pytest.mark.parametrize("drift", ("contract_key", "antecedent_keys"))
def test_source_consensus_record_refuses_contract_or_history_drift(drift):
    """记录不能仅凭相同答案合并不同作用域或历史实体解释。"""
    from pure_integer_ai.experiments.query_reference_source_consensus import (
        GroundedReferenceSource, reference_source_consensus,
    )
    first = GroundedReferenceSource((1,), (11,), ((21,),), (31,))
    second = GroundedReferenceSource((2,), (12,), ((21,),), (31,))
    consensus = reference_source_consensus((second, first))
    assert consensus is not None
    assert consensus == reference_source_consensus((first, second))
    assert consensus.selected_binding_key == (1,)
    value = ((99,),) if drift == "antecedent_keys" else (99,)
    assert reference_source_consensus((first, replace(second, **{drift: value}))) is None
