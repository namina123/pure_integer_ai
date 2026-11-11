"""完整 Core 来源仲裁必须保留真实答案、Role 次序与结构冲突。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OwnerScope, SourceRef, VersionBundle, VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity, proposition_identity,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_input_provenance import (
    aggregate_input_provenance_duplicates,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_CAUSES, RELATION_SIMILAR, ROLE_CAUSE, ROLE_EFFECT,
    ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
)
from pure_integer_ai.experiments.query_relation_arbitration import (
    core_relation_endpoint_identity, input_relation_competition,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


def _fixture(*, different_answer=False, different_known=False,
             symmetric=True, composed=False, dynamic_reference=False):
    """经真实投影器建立同形、反向 Role、独立来源的两个普通 WH 候选。"""
    predicate = authored_relation_identity(
        RELATION_SIMILAR if symmetric else RELATION_CAUSES)
    roles = tuple(authored_relation_role_identity(value)
                  for value in ((ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT)
                                if symmetric else (ROLE_CAUSE, ROLE_EFFECT)))
    structures, open_frames, question_frames, facts = [], [], [], {}
    for index, pair in enumerate((roles, roles[::-1] if symmetric else roles), 101):
        source = SourceRef(206, index, 0, OwnerScope(), VersionBundle())
        proposition = proposition_identity(source, (1, 1))
        known = entity_identity(source, (
            10, 98 if different_known and index == 102 else 10))
        answer = entity_identity(source, (
            10, 99 if different_answer and index == 102 else 11))
        prop, pred, ref = (proposition.stable_key(), predicate.stable_key(), (index,))
        members = (
            TrainedInputMember((11,), known.stable_key(), known.object_kind,
                PROJECTION_FILLER, prop, pred, ref, 0, 1, 0, pair[0].stable_key()),
            TrainedInputMember((33,), pred, predicate.object_kind,
                PROJECTION_PREDICATE, prop, pred, ref, 1, 2, 1),
            TrainedInputMember((22,), answer.stable_key(), answer.object_kind,
                PROJECTION_FILLER, prop, pred, ref, 2, 3, 2, pair[1].stable_key()),
        )
        structures.append(TrainedInputStructure(prop, pred, ref, (7,), members))
        frame = TrainedRelationQuestionFrame(
            (92002, index), (92004, 1), pred, (pair[0].stable_key(),),
            ((88, 33), ()), pair[1].stable_key(), (88,))
        local_frames = (frame,)
        if composed:
            local_frames = (
                replace(frame, source_case_key=(92002, index, 1),
                        gaps=((), (100, 200)), wh_values=(200,)),
                replace(frame, source_case_key=(92002, index, 2),
                        gaps=((300,), (400, 200)), wh_values=(200,)),
            )
        question_frames.extend(local_frames)
        open_frames.extend(OpenRoleFrame(
            prop, pred, ref, item.roles, item.gaps, (0, 3))
            for item in local_frames)
        facts[prop] = SimpleNamespace(
            proposition=proposition, predicate=predicate, source_hash=index,
            bindings=tuple(SimpleNamespace(role=role, filler=filler)
                           for role, filler in zip(pair, (known, answer))),
        )
    structure = TrainedInputStructureProjector(
        tuple(structures), open_role_frames=tuple(open_frames),
        slot_question_frames=tuple(question_frames) if composed else (),
    ).project((300, 11, 100, 200) if composed
              else (88, 33, 55) if dynamic_reference else (88, 33, 11))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = tuple(question_frames)
    bridge._relation_question_open_frames = tuple(open_frames)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.core_fact_index = facts
    return bridge, structure


def test_symmetric_wh_reversed_sources_bind_one_answer_through_bridge():
    bridge, structure = _fixture()
    assert match_relation_slot_question_frames(
        bridge.relation_question_frames, structure) == ()
    alternatives = match_relation_slot_question_frames(
        bridge.relation_question_frames, structure, allow_exact_alternatives=True)
    assert len(alternatives) == 2
    facts = tuple(bridge.core_fact_index.values())
    assert core_relation_endpoint_identity(facts[0]) == core_relation_endpoint_identity(facts[1])
    assert not input_relation_competition(
        structure, bridge.core_fact_index,
        question_predicate_keys=frozenset({facts[0].predicate.stable_key()}))

    prepared = bridge._attach_relation_question_binding(structure)
    assert len(prepared.relation_question_bindings) == 1
    assert len(prepared.relation_candidates) == 1
    assert bridge._attach_relation_question_binding(prepared) == prepared
    binding = prepared.relation_question_bindings[0]
    assert binding == min(alternatives, key=lambda item: item.stable_key())
    parts = bridge._relation_slot_question_parts(prepared, binding)
    assert parts is not None and parts[4] == ()


def test_symmetric_wh_different_answer_remains_competing_and_unbound():
    bridge, structure = _fixture(different_answer=True)
    predicate = next(iter(bridge.core_fact_index.values())).predicate.stable_key()
    assert input_relation_competition(
        structure, bridge.core_fact_index,
        question_predicate_keys=frozenset({predicate}))
    prepared = bridge._attach_relation_question_binding(structure)
    assert prepared.relation_question_bindings == ()
    assert len(prepared.relation_candidates) == 2


@pytest.mark.parametrize("change", ("structure", "carrier", "source", "dynamic"))
def test_symmetric_wh_does_not_discard_other_binding_conflicts(change):
    bridge, structure = _fixture()
    candidate = structure.relation_candidates[-1]
    if change in {"structure", "carrier"}:
        changes = ({"structure_key": (999, *candidate.structure_key)}
                   if change == "structure" else {"carrier_key": (99,)})
        structure = replace(structure, relation_candidates=tuple(sorted((
            replace(item, **changes) if item == candidate else item
            for item in structure.relation_candidates), key=lambda item: item.stable_key())))
    elif change == "source":
        bridge.core_fact_index[candidate.proposition_key].source_hash += 1
    else:
        bridge.relation_question_frames = tuple(
            replace(frame, slot_mode=1) for frame in bridge.relation_question_frames)
    prepared = bridge._attach_relation_question_binding(structure)
    if change == "dynamic":
        assert len(prepared.relation_question_bindings) == 2
        assert len(prepared.relation_candidates) == 2
    else:
        assert prepared.relation_question_bindings == ()


def test_core_endpoint_comparison_retains_order_for_other_predicates():
    bridge, _structure = _fixture()
    facts = tuple(bridge.core_fact_index.values())
    for fact in facts:
        fact.predicate = entity_identity(
            SourceRef(206, 1, 0, OwnerScope(), VersionBundle()), (40, 1))
    assert core_relation_endpoint_identity(facts[0]) != core_relation_endpoint_identity(facts[1])


def test_ordered_wh_equivalent_core_sources_bind_one_answer_through_bridge():
    """非对称关系的同一完整有序声明可选物理代表，重复准备仍幂等。"""
    bridge, structure = _fixture(symmetric=False)
    assert match_relation_slot_question_frames(
        bridge.relation_question_frames, structure) == ()
    facts = tuple(bridge.core_fact_index.values())
    assert core_relation_endpoint_identity(facts[0]) == core_relation_endpoint_identity(facts[1])
    prepared = bridge._attach_relation_question_binding(structure)
    assert len(prepared.relation_question_bindings) == 1
    assert len(prepared.relation_candidates) == 1
    assert bridge._attach_relation_question_binding(prepared) == prepared


@pytest.mark.parametrize("drift", ("equivalent", "answer", "known"))
@pytest.mark.parametrize("source_proof", (False, True))
def test_partial_wh_aggregation_keeps_full_core_competitors(drift, source_proof):
    """真实输入全链不得按相同已观察槽位提前删除未观察答案或同名实体。"""
    bridge, structure = _fixture(
        symmetric=False, different_answer=drift == "answer",
        different_known=drift == "known")
    proofs = ({item.source_ref: (1, 77) for item in structure.relation_candidates}
              if source_proof else None)
    aggregated = aggregate_input_provenance_duplicates(
        structure, source_provenance=proofs)
    assert len(aggregated.relation_candidates) == 2
    assert len(aggregated.open_relations) == 2
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated
    filtered = bridge._filter_relation_question_open_scope(aggregated)
    assert len(filtered.relation_candidates) == 2
    prepared = bridge._attach_relation_question_binding(filtered)
    if drift == "equivalent":
        assert len(prepared.relation_question_bindings) == 1
        assert len(prepared.relation_candidates) == 1
    else:
        assert not prepared.relation_question_bindings
        assert len(prepared.relation_candidates) == 2
        assert input_relation_competition(
            prepared, bridge.core_fact_index,
            question_predicate_keys=frozenset(
                fact.predicate.stable_key() for fact in bridge.core_fact_index.values()))


@pytest.mark.parametrize("drift", ("answer", "known", "roles", "owner", "version"))
def test_ordered_wh_preserves_every_full_endpoint_identity(drift):
    """同形观察不能消除不同答案、已知端点、Role 或身份作用域。"""
    bridge, structure = _fixture(
        symmetric=False, different_answer=drift == "answer",
        different_known=drift == "known")
    fact = tuple(bridge.core_fact_index.values())[-1]
    if drift == "roles":
        first, second = fact.bindings
        fact.bindings = (
            SimpleNamespace(role=first.role, filler=second.filler),
            SimpleNamespace(role=second.role, filler=first.filler))
    elif drift in {"owner", "version"}:
        answer = fact.bindings[-1]
        filler = answer.filler
        answer.filler = replace(filler, **(
            {"owner": replace(filler.owner, tenant_id=1, user_id=77,
                              visibility=VISIBILITY_USER)} if drift == "owner"
            else {"versions": replace(filler.versions,
                  parser=type(filler.versions.parser)(77))}))
    prepared = bridge._attach_relation_question_binding(structure)
    assert not prepared.relation_question_bindings
    assert len(prepared.relation_candidates) == 2


def test_composed_wh_uses_full_parents_and_equivalent_ordered_core_sources():
    """完整投影到桥接消费新间隔组合，答案仍来自已验证的 Core。"""
    from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
        ComposedSlotQuestionFrame, restore_slot_question_frame,
    )

    bridge, structure = _fixture(symmetric=False, composed=True)
    structure = bridge._filter_relation_question_open_scope(structure)
    prepared = bridge._attach_relation_question_binding(structure)
    binding, = prepared.relation_question_bindings
    frame = restore_slot_question_frame(
        binding.frame_key, trained_frames=bridge.relation_question_frames)
    assert isinstance(frame, ComposedSlotQuestionFrame)
    assert frame.gaps == ((300,), (100, 200))
    assert len(prepared.relation_candidates) == 1
    assert bridge._relation_slot_question_parts(prepared, binding) is not None
    assert bridge._attach_relation_question_binding(prepared) == prepared
    bridge.relation_question_frames = tuple(
        item for item in bridge.relation_question_frames if item.gaps[0] == ())
    assert bridge._relation_slot_question_parts(prepared, binding) is None


def _pending_reference_fixture():
    """使用真实开放投影与训练帧建立两个来源不同的待解观察槽。"""
    bridge, structure = _fixture(
        symmetric=False, different_known=True, different_answer=True,
        dynamic_reference=True)
    frame = bridge.relation_question_frames[0]
    marker = replace(frame, source_case_key=(92002, 999),
                     roles=(frame.answer_role_key,),
                     answer_role_key=frame.roles[0], gaps=((55, 33), ()),
                     wh_values=(55,), slot_mode=1)
    bridge.relation_question_frames = (*bridge.relation_question_frames, marker)
    prepared = bridge._attach_relation_question_binding(structure)
    assert len(prepared.relation_question_bindings) == 2
    assert all(bridge._relation_slot_question_parts(prepared, item)[4]
               for item in prepared.relation_question_bindings)
    return bridge, prepared, marker


def test_pending_trained_reference_defers_endpoints_without_selecting_a_fact():
    """不同模板端点不等于已观察冲突，全部候选仍交给原指代与事实门。"""
    from pure_integer_ai.experiments.query_pending_reference import (
        pending_reference_relation_keys,
    )

    bridge, prepared, _marker = _pending_reference_fixture()
    original = prepared.integer_trace()
    assert pending_reference_relation_keys(
        prepared, bridge.core_fact_index, bridge.relation_question_frames
    ) == frozenset(item.relation_key for item in prepared.relation_question_bindings)
    assert not bridge._input_relation_competition(
        prepared, question_predicate_keys=frozenset(
            frame.predicate_key for frame in bridge.relation_question_frames))
    assert len(prepared.relation_candidates) == 2
    assert prepared.integer_trace() == original


def test_unselected_reference_question_keeps_complete_candidates_for_clarification():
    """同一问式全部待解时可澄清，不能选择模板端点或改写原输入。"""
    from pure_integer_ai.experiments.query_pending_reference import unselected_reference_question

    bridge, prepared, _marker = _pending_reference_fixture()
    original = prepared.integer_trace()
    assert unselected_reference_question(
        prepared, bridge.core_fact_index, bridge.relation_question_frames)
    assert prepared.integer_trace() == original
    assert not unselected_reference_question(replace(prepared,
        relation_question_bindings=prepared.relation_question_bindings[:1]),
        bridge.core_fact_index, bridge.relation_question_frames)


@pytest.mark.parametrize("drift", ("syntax", "source", "marker", "parse"))
def test_unselected_reference_question_rejects_incomplete_or_mixed_candidates(drift):
    """任一完整语法、来源、训练指代原子或开放解析缺失均不能绕到澄清。"""
    from pure_integer_ai.experiments.query_pending_reference import unselected_reference_question

    bridge, prepared, marker = _pending_reference_fixture()
    if drift == "syntax":
        first, second = prepared.relation_question_bindings
        prepared = replace(prepared, relation_question_bindings=(first,
            replace(second, answer_role_key=(99999,))))
    elif drift == "source":
        prepared = replace(prepared, open_relations=tuple(sorted((
            replace(item, frame=replace(item.frame, source_ref=(99999,)))
            for item in prepared.open_relations), key=lambda item: item.stable_key())))
    elif drift == "marker":
        bridge.relation_question_frames = tuple(
            item for item in bridge.relation_question_frames if item != marker)
    else:
        prepared = replace(prepared, open_relations=prepared.open_relations[:1])
    assert not unselected_reference_question(
        prepared, bridge.core_fact_index, bridge.relation_question_frames)


@pytest.mark.parametrize("missing", (
    "marker", "frame", "role", "values", "parse", "source", "gap"))
def test_pending_reference_without_exact_training_or_parse_keeps_conflict(missing):
    """缺训练原子、完整父帧、精确角色/跨度/来源/间隔时不得推迟竞争。"""
    bridge, prepared, marker = _pending_reference_fixture()
    if missing == "marker":
        bridge.relation_question_frames = bridge.relation_question_frames[:-1]
    elif missing == "frame":
        bridge.relation_question_frames = (marker,)
    elif missing in {"role", "values"}:
        replacement = (replace(marker, answer_role_key=(999,))
                       if missing == "role" else
                       replace(marker, gaps=((56, 33), ()), wh_values=(56,)))
        bridge.relation_question_frames = (*bridge.relation_question_frames[:-1], replacement)
    elif missing == "parse":
        prepared = replace(prepared, open_relations=())
    elif missing == "source":
        prepared = replace(prepared, open_relations=tuple(sorted((
            replace(item, frame=replace(item.frame, source_ref=(999,)))
            for item in prepared.open_relations), key=lambda item: item.stable_key())))
    else:
        prepared = replace(prepared, open_relations=tuple(sorted((
            replace(item, frame=replace(item.frame, gaps=((88,), ())),
                    bindings=tuple(replace(span, start=1, values=(33, 55))
                                   for span in item.bindings))
            for item in prepared.open_relations), key=lambda item: item.stable_key())))
    assert bridge._input_relation_competition(
        prepared, question_predicate_keys=frozenset(
            frame.predicate_key for frame in bridge.relation_question_frames))
