"""Variable-slot question evidence and train-only graph boundary tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE,
    STRUCTURE_CLOSED, STRUCTURE_OPEN, TrainedInputMember, TrainedInputStructure,
    TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    ComposedRelationQuestionFrame, TrainedRelationQuestionFrame,
    match_relation_question_frames, match_relation_slot_question_frames,
    relation_question_syntax_predicates,
)
from pure_integer_ai.experiments.connected_relation_question_course import (
    compile_connected_pair_questions, compile_train_relation_question,
    compile_train_slot_relation_question,
)
from pure_integer_ai.experiments.integer_relation_connected_course import (
    compile_connected_course, prepare_connected_course,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge, _selected_relation_view,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    MODE_RELATION_PAIR_QUERY, materialize_relation_question_graph,
    read_relation_question_graph,
)
from pure_integer_ai.experiments.integer_relation_generation_bridge import (
    IntegerRelationGenerationBridgeError,
    materialize_integer_relation_generation,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES
from pure_integer_ai.experiments.ph2_authored_subset_member_course import (
    read_authored_subset_member_seeds,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_role_identity,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ph2/authored_connected_wave1_facts_v1.jsonl.sample"
CASES = ROOT / "data/ph2/authored_connected_wave1_cases_v1.jsonl.sample"


def _fact(*, proposition=(1, 50), predicate=(5, 60), source=(9, 70),
          fillers=(11, 22), identities=((4, 101), (4, 102))):
    members = (
        TrainedInputMember(
            (fillers[0],), identities[0], 4, PROJECTION_FILLER,
            proposition, predicate, source, 0, 1, 0, (8, 1)),
        TrainedInputMember(
            (33,), predicate, 5, PROJECTION_PREDICATE,
            proposition, predicate, source, 1, 2, 1),
        TrainedInputMember(
            (fillers[1],), identities[1], 4, PROJECTION_FILLER,
            proposition, predicate, source, 2, 3, 2, (8, 2)),
    )
    return TrainedInputStructure(proposition, predicate, source, (7, 1), members)


def _frame(**kwargs):
    return TrainedRelationQuestionFrame(
        kwargs.get("source", (92002, 1)), (92004, 1),
        kwargs.get("predicate", (5, 60)), ((8, 1), (8, 2)),
        ((), (44,), (55,)),
    )


def _property_fixture():
    proposition = (1, 60)
    predicate = (5, 61)
    source = (9, 80)
    roles = tuple((8, index) for index in range(1, 7))
    members = [TrainedInputMember(
        (10,), (4, 201), 4, PROJECTION_FILLER,
        proposition, predicate, source, 0, 1, 0, roles[0])]
    members.append(TrainedInputMember(
        (90,), predicate, 5, PROJECTION_PREDICATE,
        proposition, predicate, source, 1, 2, 1))
    for index in range(1, 6):
        members.append(TrainedInputMember(
            (10 + index,), (4, 201 + index), 4, PROJECTION_FILLER,
            proposition, predicate, source, index + 1, index + 2,
            index + 1, roles[index]))
    structure = TrainedInputStructure(
        proposition, predicate, source, (7, 2), tuple(members))
    gaps = ((77,), (90,), (70,), (71,), (72,), (73,), (88,))
    frame = TrainedRelationQuestionFrame(
        (92002, 90), (92004, 1), predicate, roles, gaps)
    values = (77, 10, 90, 11, 70, 12, 71, 13, 72, 14, 73, 15, 88)
    return structure, frame, values


def test_six_role_question_requires_every_current_graph_slot(tmp_path):
    structure, frame, values = _property_fixture()
    database = tmp_path / "six-role-questions.sqlite3"
    materialize_relation_question_graph(
        (frame,), database, source_identity=(92026, 2), license_identity=(1, 2))
    assert read_relation_question_graph(database)[0] == (frame,)
    projector = TrainedInputStructureProjector((structure,))
    projected = projector.project(values)
    matches = match_relation_question_frames((frame,), projected)
    assert len(matches) == 1
    assert len(matches[0].slot_span_refs) == 6
    assert matches[0].relation_key in {
        item.stable_key() for item in projected.relation_candidates
        if item.proposition_key == structure.proposition_key
    }
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_operation_modes = {
        frame.operation_key: MODE_RELATION_PAIR_QUERY}
    prepared = bridge._attach_relation_question_binding(projected)
    memory_view = bridge.active_memory_input_structure(prepared)
    assert memory_view.relation_question_bindings == matches
    assert memory_view.relation_candidates == tuple(
        item for item in projected.relation_candidates
        if item.stable_key() == matches[0].relation_key)
    assert match_relation_question_frames((frame,), projector.project(
        (77, 10, 90, 12, 70, 11, 71, *values[7:]))) == ()
    assert match_relation_question_frames((frame,), projector.project(
        values[:9] + values[10:])) == ()
    assert match_relation_question_frames((frame,), projector.project(
        (*values, 99))) == ()


def test_connected_question_compiler_accepts_complete_six_role_train_case():
    fragments = ("甲", "乙", "丙", "丁", "戊", "己")
    query = "请核对甲、乙、丙、丁、戊、己。"
    profile = W06_RELATION_PROFILES["PROPERTY"]
    seed = SimpleNamespace(
        seed_id="p1", sample_role="support", split="train", label_owner="teacher",
        relation_kind=profile.relation_kind,
        endpoints=tuple(SimpleNamespace(surface_fragment=value)
                        for value in fragments),
        bindings=tuple(SimpleNamespace(role_kind=role)
                       for role, _allowed in profile.role_object_kinds),
    )
    assignment = SimpleNamespace(
        case_key=(92002, 1), split_code=1,
        operation_key=(92004, 1), input_values=tuple(map(ord, query)))
    draft = SimpleNamespace(
        case_sha256="case", source_sha256="source", seeds=(seed,),
        cases=({"split": "train", "input": query,
                "operation": "ASSERT_RELATION", "seed_ids": ["p1"]},))
    compilation = SimpleNamespace(
        case_sha256="case", assignments=(assignment,),
        source_audit={"source_sha256": "source",
                      "status": "PASS_SOURCE_AND_FOUR_WAY_ISOLATION_ONLY"})
    frames = compile_connected_pair_questions(draft, compilation)
    assert len(frames) == 1
    assert len(frames[0].roles) == 6
    assert len(frames[0].gaps) == 7
    assert frames[0].source_case_key == (92002, 1)


def test_connected_question_compiler_orders_roles_by_observed_input_spans():
    seeds = {
        item.seed_id: item for item in read_authored_subset_member_seeds(
            ROOT / "data/ph2/authored_relation_subset_member_seed_v1.jsonl.sample")}
    seed = seeds["teacher-sparrow-member-v1"]
    frame = compile_train_relation_question(
        seed, "麻雀集合里有小麻雀吗？", source_case_key=(92029, 15, 1),
        operation_key=(92004, 1))
    assert frame is not None
    assert frame.roles == tuple(
        authored_relation_role_identity(item.role_kind).stable_key() for item in (
            next(binding for binding in seed.bindings
                 if binding.role_kind == 8),
            next(binding for binding in seed.bindings
                 if binding.role_kind == 7)))
    assert tuple("".join(map(chr, gap)) for gap in frame.gaps) == (
        "", "里有", "吗？")


def test_question_graph_roundtrip_and_integer_only_tamper_rejection(tmp_path):
    frame = _frame()
    database = tmp_path / "questions.sqlite3"
    materialize_relation_question_graph(
        (frame,), database, source_identity=(92026, 1), license_identity=(1, 2))
    restored, metadata = read_relation_question_graph(database)
    assert restored == (frame,)
    assert metadata["source_identity"] == (92026, 1)
    assert metadata["license_identity"] == (1, 2)
    with pytest.raises(FileExistsError):
        materialize_relation_question_graph(
            (frame,), database, source_identity=(92026, 1), license_identity=(1, 2))
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE relation_question_frame SET frame_hash=1")
    with pytest.raises(RuntimeError, match="identity drifted"):
        read_relation_question_graph(database)


def test_variable_question_binds_current_fact_not_training_example():
    new_fact = _fact(proposition=(1, 51), source=(9, 71), fillers=(66, 77),
                     identities=((4, 201), (4, 202)))
    projected = TrainedInputStructureProjector((_fact(), new_fact)).project(
        (66, 44, 77, 55))
    matches = match_relation_question_frames((_frame(),), projected)
    assert len(matches) == 1
    assert matches[0].relation_key == next(
        item.stable_key() for item in projected.relation_candidates
        if item.proposition_key == (1, 51))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (_frame(),)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_operation_modes = {(92004, 1): MODE_RELATION_PAIR_QUERY}
    prepared = bridge._attach_relation_question_binding(projected)
    memory_view = bridge.active_memory_input_structure(prepared)
    assert memory_view.operation_keys == ((92004, 1),)
    assert len(memory_view.relation_candidates) == 1
    assert memory_view.relation_candidates[0].proposition_key == (1, 51)
    assert memory_view.relation_question_bindings == matches
    assert memory_view.integer_trace()["relation_question_bindings"]
    assert _selected_relation_view(prepared, matches[0].relation_key)[0] == memory_view


def test_unbound_pair_operation_is_retained_for_query_fail_closed():
    """A licensed pair marker with no current binding must not raise at intake."""
    frame = _frame()
    projected = TrainedInputStructureProjector((_fact(),)).project(
        (66, 44, 77, 55))
    projected = replace(projected, operation_keys=(frame.operation_key,))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {
        frame.operation_key: MODE_RELATION_PAIR_QUERY}

    prepared = bridge._attach_relation_question_binding(projected)

    assert prepared == projected
    assert prepared.operation_keys == (frame.operation_key,)
    assert not prepared.relation_question_bindings


def _slot_frame():
    return TrainedRelationQuestionFrame(
        (92002, 90), (92004, 1), (5, 60), ((8, 1),),
        ((), (33, 88)), (8, 2), (88,))


def _slot_projection(*facts, values=(11, 33, 88)):
    from pure_integer_ai.cognition.understanding.query_open_roles import (
        OpenRoleFrame,
    )

    frame = _slot_frame()
    open_frames = tuple(OpenRoleFrame(
        fact.proposition_key, fact.predicate_key, fact.source_ref,
        frame.roles, frame.gaps,
        (0, len(frame.roles) + sum(map(len, frame.gaps))),
    ) for fact in facts)
    projector = TrainedInputStructureProjector(
        tuple(facts), open_role_frames=open_frames)
    return frame, projector.project(values)


def test_single_slot_wh_question_binds_only_current_open_core_relation():
    frame, projected = _slot_projection(_fact(source=(9,)))
    matches = match_relation_slot_question_frames((frame,), projected)

    assert len(matches) == 1
    assert matches[0].answer_role_key == (8, 2)
    assert matches[0].slot_span_refs == (projected.spans[0].ref_key,)
    relation = next(item for item in projected.relation_candidates
                    if item.stable_key() == matches[0].relation_key)
    assert relation.state == STRUCTURE_OPEN
    assert relation.required_open == 1
    assert relation.predicate_coverage == 1


def test_single_slot_wh_prefers_unique_exact_core_candidate_over_dynamic():
    exact_fact = _fact(source=(9, 70), fillers=(11, 22))
    unrelated_fact = _fact(
        proposition=(1, 51), source=(9, 71), fillers=(66, 77),
        identities=((4, 201), (4, 202)))
    frame, projected = _slot_projection(exact_fact, unrelated_fact)

    matches = match_relation_slot_question_frames((frame,), projected)

    exact_relation = next(
        item for item in projected.relation_candidates
        if item.proposition_key == exact_fact.proposition_key
        and item.state == STRUCTURE_OPEN)
    assert len(matches) == 1
    assert matches[0].relation_key == exact_relation.stable_key()


def test_single_slot_wh_dynamic_alternatives_require_explicit_bridge_mode():
    first = _fact(source=(9,), fillers=(11, 22))
    second = _fact(
        proposition=(1, 51), source=(10,), fillers=(66, 77),
        identities=((4, 201), (4, 202)))
    frame, projected = _slot_projection(
        first, second, values=(99, 33, 88))

    assert match_relation_slot_question_frames((frame,), projected) == ()
    alternatives = match_relation_slot_question_frames(
        (frame,), projected, allow_dynamic_alternatives=True)
    assert len(alternatives) == 2
    assert {item.relation_key for item in alternatives} == {
        item.stable_key() for item in projected.relation_candidates
        if item.state == STRUCTURE_OPEN}


@pytest.mark.parametrize("values", [
    (22, 33, 88),  # Known endpoint has the opposite trained role.
    (11, 33, 88, 77),  # Unparsed input suffix.
    (11, 33, 89),  # Untrained WH marker.
])
def test_single_slot_wh_question_fails_closed_on_nonmatching_input(values):
    frame, projected = _slot_projection(_fact(source=(9,)), values=values)
    assert match_relation_slot_question_frames((frame,), projected) == ()


def test_single_slot_wh_question_keeps_unresolved_role_span_without_entity():
    frame, projected = _slot_projection(
        _fact(source=(9,)), values=(99, 33, 88))

    matches = match_relation_slot_question_frames((frame,), projected)

    assert len(matches) == 1
    dynamic_ref = matches[0].slot_span_refs[0]
    dynamic_span = next(item for item in projected.spans
                        if item.ref_key == dynamic_ref)
    assert (dynamic_span.start, dynamic_span.end, dynamic_span.values) == (
        0, 1, (99,))
    assert not any(item.span_ref == dynamic_ref
                   and item.projection_kind == PROJECTION_FILLER
                   for item in projected.semantic_candidates)
    assert len(projected.open_relations) == 1


def test_single_slot_wh_question_rejects_competing_core_facts():
    first = _fact(source=(9,), fillers=(11, 22))
    second = _fact(proposition=(1, 51), source=(9, 71),
                   fillers=(11, 66), identities=((4, 201), (4, 202)))
    frame, projected = _slot_projection(first, second)

    assert match_relation_slot_question_frames((frame,), projected) == ()
    assert match_relation_slot_question_frames(
        (frame,), projected, allow_dynamic_alternatives=True) == ()


@pytest.mark.parametrize(("antecedent_identity", "expected_index"), [
    ((4, 201), 1),
    ((4, 999), None),
])
def test_memory_grounded_dynamic_slot_selects_only_exact_core_identity(
        antecedent_identity, expected_index):
    from types import SimpleNamespace

    first = _fact(source=(9,), fillers=(11, 22))
    second = _fact(
        proposition=(1, 51), source=(10,), fillers=(66, 77),
        identities=((4, 201), (4, 202)))
    frame, projected = _slot_projection(
        first, second, values=(99, 33, 88))
    alternatives = match_relation_slot_question_frames(
        (frame,), projected, allow_dynamic_alternatives=True)
    assert len(alternatives) == 2
    projected = replace(
        projected,
        operation_keys=(frame.operation_key,),
        relation_question_bindings=alternatives,
    )

    class Identity:
        def __init__(self, key):
            self.key = key

        def stable_key(self):
            return self.key

    def core_fact(source_hash, proposition, first_filler):
        return SimpleNamespace(
            proposition=Identity(proposition),
            predicate=Identity((5, 60)),
            source_hash=source_hash,
            bindings=(
                SimpleNamespace(role=Identity((8, 1)), filler=Identity(first_filler)),
                SimpleNamespace(role=Identity((8, 2)), filler=Identity((4, 202))),
            ),
        )

    target_fact = core_fact(10, (1, 51), (4, 201))
    other_fact = core_fact(9, (1, 50), (4, 200))
    target_open = SimpleNamespace(frame=SimpleNamespace(
        proposition=(1, 51), predicate=(5, 60)))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.core_fact_index = {
        (1, 50): other_fact,
        (1, 51): target_fact,
    }
    bridge._input_observation_ref = object()
    bridge._reference_candidate_sets = (SimpleNamespace(
        reference=SimpleNamespace(
            role=Identity((8, 1)), values=(99,)),
        antecedents=(Identity((1, 1)),),
    ),)
    bridge._response_contexts = ()

    class ReferenceRuntime:
        @staticmethod
        def _occurrence_values(_antecedent):
            return (99,)

        @staticmethod
        def semantic_objects_for(_antecedent, _contexts):
            if antecedent_identity == (4, 201):
                return (target_fact.bindings[0].filler,)
            return (Identity(antecedent_identity),)

    bridge._reference_runtime = ReferenceRuntime()
    bridge._reference_relation_context = lambda _candidate, _structure: (
        target_open, None)
    target_binding = next(
        item for item in alternatives
        if next(candidate for candidate in projected.relation_candidates
                if candidate.stable_key() == item.relation_key).proposition_key
        == (1, 51))
    parts = bridge._relation_slot_question_parts(projected, target_binding)
    assert parts is not None
    assert parts[4] == ((8, 1),)
    assert {item.filler.stable_key() for item in parts[2].bindings
            if item.role.stable_key() == parts[4][0]} == {(4, 201)}
    assert bridge._reference_relation_context(
        bridge._reference_candidate_sets[0], projected)[0].frame.proposition == (1, 51)
    assert bridge._reference_candidate_sets[0].reference.role.stable_key() == (8, 1)

    selected = bridge._memory_grounded_slot_binding(projected)

    if expected_index is None:
        assert selected is None
    else:
        assert selected in alternatives
        assert selected.relation_key == next(
            item.stable_key() for item in projected.relation_candidates
            if item.proposition_key == (1, 51))


def test_single_slot_wh_bridge_rejects_competing_core_facts():
    first = _fact(source=(9,), fillers=(11, 22))
    second = _fact(proposition=(1, 51), source=(9, 71),
                   fillers=(11, 66), identities=((4, 201), (4, 202)))
    frame, projected = _slot_projection(first, second)
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.core_fact_index = {}

    prepared = bridge._attach_relation_question_binding(projected)

    assert prepared.relation_question_bindings == ()
    assert prepared.operation_keys == ()


def test_single_slot_wh_bridge_binds_core_answer_role_and_scopes_memory():
    from types import SimpleNamespace

    frame, projected = _slot_projection(_fact(source=(9,)))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()

    class Identity:
        def __init__(self, key):
            self.key = key

        def stable_key(self):
            return self.key

    bridge.core_fact_index = {
        (1, 50): SimpleNamespace(
            predicate=Identity((5, 60)), source_hash=9,
            bindings=(
                SimpleNamespace(role=Identity((8, 1)),
                                filler=Identity((4, 101))),
                SimpleNamespace(role=Identity((8, 2)),
                                filler=Identity((4, 102))),
            ),
        ),
    }
    prepared = bridge._attach_relation_question_binding(projected)
    memory_view = bridge.active_memory_input_structure(prepared)

    assert prepared.operation_keys == ((92004, 1),)
    assert prepared.relation_question_bindings[0].answer_role_key == (8, 2)
    assert len(memory_view.relation_candidates) == 1
    assert memory_view.relation_candidates[0].proposition_key == (1, 50)
    assert memory_view.open_relations == ()


def test_slot_question_compiler_uses_train_teacher_role_and_exact_wh_source():
    from pure_integer_ai.experiments.ph2_authored_relation_compile import (
        authored_relation_role_identity,
    )

    draft = prepare_connected_course(SOURCE, CASES)
    seed = next(item for item in draft.seeds if item.seed_id == "t02a")
    frame = compile_train_slot_relation_question(
        seed, "代号甲表示什么？", answer_binding_index=1,
        wh_surface="什么", source_case_key=(92002, 91),
        operation_key=(92004, 1))

    assert frame.roles == (authored_relation_role_identity(
        seed.bindings[0].role_kind).stable_key(),)
    assert frame.answer_role_key == authored_relation_role_identity(
        seed.bindings[1].role_kind).stable_key()
    assert frame.wh_values == tuple(map(ord, "什么"))
    assert frame.gaps == ((), tuple(map(ord, "表示什么？")))
    assert TrainedRelationQuestionFrame.from_stable_key(
        frame.stable_key()) == frame
    with pytest.raises(ValueError, match="omit the answer endpoint"):
        compile_train_slot_relation_question(
            seed, "代号甲表示东侧信标是什么？", answer_binding_index=1,
            wh_surface="什么", source_case_key=(92002, 92),
            operation_key=(92004, 1))


def test_slot_question_compiler_records_wh_gap_with_observed_surface_order():
    from pure_integer_ai.experiments.ph2_authored_relation_compile import (
        authored_relation_role_identity,
    )

    draft = prepare_connected_course(SOURCE, CASES)
    seed = next(item for item in draft.seeds if item.seed_id == "t02a")
    frame = compile_train_slot_relation_question(
        seed, "什么代号甲表示晨泽瓶？", answer_binding_index=1,
        wh_surface="什么", source_case_key=(92002, 93),
        operation_key=(92004, 1))

    assert frame.roles == (authored_relation_role_identity(
        seed.bindings[0].role_kind).stable_key(),)
    assert frame.gaps == (tuple(map(ord, "什么")),
                          tuple(map(ord, "表示晨泽瓶？")))


def test_slot_question_compiler_uses_surface_order_for_answer_gap():
    from pure_integer_ai.experiments.ph2_authored_relation_compile import (
        authored_relation_role_identity,
    )

    seed = SimpleNamespace(
        split="train", label_owner="teacher", sample_role="support",
        relation_kind=6,
        bindings=(
            SimpleNamespace(role_kind=17, endpoint_id="part"),
            SimpleNamespace(role_kind=18, endpoint_id="whole"),
        ),
        endpoints=(
            SimpleNamespace(endpoint_id="part", surface_fragment="车轮"),
            SimpleNamespace(endpoint_id="whole", surface_fragment="汽车"),
        ),
    )

    frame = compile_train_slot_relation_question(
        seed, "汽车的一部分是什么？", answer_binding_index=0,
        wh_surface="什么", source_case_key=(92002, 94),
        operation_key=(92004, 1))

    assert frame.roles == (authored_relation_role_identity(18).stable_key(),)
    assert frame.answer_role_key == authored_relation_role_identity(17).stable_key()
    assert frame.gaps == ((), tuple(map(ord, "的一部分是什么？")))
    assert frame.wh_values == tuple(map(ord, "什么"))


def test_slot_question_frame_survives_integer_graph_roundtrip(tmp_path):
    frame = _slot_frame()
    graph = tmp_path / "slot-questions.sqlite3"
    materialize_relation_question_graph(
        (frame,), graph, source_identity=(92026, 90),
        license_identity=tuple(map(ord, "CC0-1.0")))

    restored, metadata = read_relation_question_graph(graph)
    assert restored == (frame,)
    assert metadata["license_identity"] == tuple(map(ord, "CC0-1.0"))


@pytest.mark.parametrize("values", [
    (66, 44, 77, 55),  # Unseen endpoints have no Core fact evidence.
    (22, 44, 11, 55),  # Role order cannot borrow inverse authority.
    (11, 44, 22, 55, 99),  # Unparsed suffix must not be discarded.
    (11, 44, 55),  # Missing second slot remains open.
])
def test_question_never_converts_open_roles_or_partial_syntax_to_fact(values):
    projected = TrainedInputStructureProjector((_fact(),)).project(values)
    assert match_relation_question_frames((_frame(),), projected) == ()


def test_question_syntax_predicate_witness_survives_unclosed_core_relation():
    projected = TrainedInputStructureProjector((_fact(),)).project(
        (11, 44, 22, 55))
    requested = _frame(predicate=(5, 61))
    assert match_relation_question_frames((requested,), projected) == ()
    assert relation_question_syntax_predicates(
        (requested,), projected) == ((5, 61),)


def test_pair_inverse_rejects_ambiguous_predicates_for_one_filler_pair():
    """An unseen relation cue cannot borrow one of two same-endpoint facts."""
    from types import SimpleNamespace

    first = _fact(predicate=(5, 60), proposition=(1, 50), source=(9, 70))
    second = _fact(predicate=(5, 61), proposition=(1, 51), source=(9, 71))
    projected = TrainedInputStructureProjector((first, second)).project(
        (11, 44, 22, 55))
    relations = tuple(
        replace(item, predicate_coverage=0, state=STRUCTURE_OPEN)
        for item in projected.relation_candidates)
    assert len(relations) >= 2
    structure = replace(projected, relation_candidates=relations)

    def fact_stub(proposition, predicate):
        identity = lambda key: SimpleNamespace(stable_key=lambda: key)
        return SimpleNamespace(
            proposition=identity(proposition), predicate=identity(predicate),
            bindings=(SimpleNamespace(filler=identity((11,))),
                      SimpleNamespace(filler=identity((22,)))),
        )

    facts = (
        fact_stub((1, 50), (5, 60)),
        fact_stub((1, 51), (5, 61)),
    )
    assert TrainedGraphQueryBridge._pair_inverse_candidates(
        structure, facts) == ()


def test_complete_question_gap_predicate_closes_evidence_without_predicate_span():
    """A trained question gap may carry the predicate contract by itself."""
    from dataclasses import replace

    from pure_integer_ai.cognition.understanding.query_open_roles import (
        OpenRoleFrame,
    )

    fact = _fact()
    frame = _frame()
    open_frame = OpenRoleFrame(
        fact.proposition_key, fact.predicate_key, fact.source_ref,
        frame.roles, frame.gaps,
        (0, len(frame.roles) + sum(map(len, frame.gaps))),
    )
    projected = TrainedInputStructureProjector(
        (fact,), open_role_frames=(open_frame,)).project(
            (11, 44, 22, 55))
    relation = projected.relation_candidates[0]
    projected = replace(
        projected,
        relation_candidates=(replace(
            relation, required_count=3, support_count=2,
            predicate_coverage=0, required_open=1,
            state=STRUCTURE_OPEN),),
    )

    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = (frame,)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.input_discourse_operation_modes = {
        frame.operation_key: MODE_RELATION_PAIR_QUERY}

    prepared = bridge._attach_relation_question_binding(projected)
    assert len(prepared.relation_question_bindings) == 1
    assert prepared.relation_candidates[0].state == STRUCTURE_CLOSED
    assert not any(
        item.projection_kind == PROJECTION_PREDICATE
        for item in prepared.semantic_candidates)
    assert prepared.concept_evidence_closed(
        fact.proposition_key, frozenset({fact.proposition_key}))


@pytest.mark.parametrize("change", (None, "unbound", "wrong_role", "wrong_gap",
                                    "wrong_operation", "missing_evidence"))
def test_complete_reverse_question_uses_trained_slot_order(change):
    """反向问式保留来源成员序，只有精确训练 Role/语法与实际证据闭合才通过。"""
    fact = _fact()
    forward = _frame()
    reverse = replace(forward, roles=forward.roles[::-1])
    projected = TrainedInputStructureProjector((fact,)).project((22, 44, 11, 55))
    relation = projected.relation_candidates[0]
    projected = replace(projected, relation_candidates=(replace(
        relation, required_count=2, support_count=2, required_open=0,
        predicate_coverage=0, state=STRUCTURE_CLOSED),))
    matches = match_relation_question_frames((reverse,), projected)
    assert len(matches) == 1
    prepared = replace(projected, operation_keys=(reverse.operation_key,),
                       relation_question_bindings=matches)
    claims = frozenset({fact.proposition_key})
    if change == "unbound":
        prepared = replace(prepared, relation_question_bindings=())
    elif change == "wrong_role":
        prepared = replace(prepared, relation_question_bindings=(replace(
            matches[0], frame_key=forward.stable_key()),))
    elif change == "wrong_gap":
        prepared = replace(prepared, relation_question_bindings=(replace(
            matches[0], frame_key=replace(reverse, gaps=((), (77,), (55,))).stable_key()),))
    elif change == "wrong_operation":
        prepared = replace(prepared, operation_keys=((999,),),
                           relation_question_bindings=(replace(
                               matches[0], operation_key=(999,)),))
    elif change == "missing_evidence":
        claims = frozenset()
    assert prepared.concept_evidence_closed(fact.proposition_key, claims) == (change is None)
    if change is None:
        from pure_integer_ai.cognition.understanding.query_question_evidence import (
            complete_question_evidence_closed,
        )
        # 相同 Span 的其他投影不能替代该 Role 自己未闭合的概念路径。
        closed_keys = frozenset(item.stable_key() for item in prepared.semantic_candidates)
        assert complete_question_evidence_closed(prepared, fact.proposition_key, claims, closed_keys)
        missing_role = next(item for item in prepared.semantic_candidates
                            if item.projection_kind == PROJECTION_FILLER)
        assert not complete_question_evidence_closed(
            prepared, fact.proposition_key, claims, closed_keys - {missing_role.stable_key()})


def test_question_composes_two_train_gap_families_without_fact_replay(tmp_path):
    first = _frame()
    second = TrainedRelationQuestionFrame(
        (92002, 2), first.operation_key, first.predicate_key,
        first.roles, ((66,), (44,), (77,)))
    structure = _fact(proposition=(1, 51), source=(9, 71), fillers=(11, 22))
    values = (66, 11, 44, 22, 55)
    projected = TrainedInputStructureProjector((structure,)).project(values)
    question_graph = tmp_path / "composable-questions.sqlite3"
    materialize_relation_question_graph(
        (first, second), question_graph,
        source_identity=(92026, 3), license_identity=(1, 2))
    restored, metadata = read_relation_question_graph(question_graph)
    assert metadata["source_identity"] == (92026, 3)
    matches = match_relation_question_frames(restored, projected)
    assert len(matches) == 1
    assert matches[0].relation_key == next(
        item.stable_key() for item in projected.relation_candidates
        if item.proposition_key == (1, 51))
    composed = ComposedRelationQuestionFrame.from_stable_key(
        matches[0].frame_key)
    assert composed.gaps == ((66,), (44,), (55,))
    assert composed.gap_sources == (
        ((92002, 2),), ((92002, 1), (92002, 2)), ((92002, 1),))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = restored
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_operation_modes = {
        first.operation_key: MODE_RELATION_PAIR_QUERY}
    prepared = bridge._attach_relation_question_binding(projected)
    assert prepared.operation_keys == (first.operation_key,)
    assert bridge.active_memory_input_structure(
        prepared).relation_question_bindings == matches
    assert match_relation_question_frames(restored, TrainedInputStructureProjector(
        (structure,)).project((66, 22, 44, 11, 55))) == ()
    assert match_relation_question_frames(restored, TrainedInputStructureProjector(
        (structure,)).project((*values, 99))) == ()


def test_question_composition_rejects_budget_exhaustion():
    first = _frame()
    second = TrainedRelationQuestionFrame(
        (92002, 2), first.operation_key, first.predicate_key,
        first.roles, ((66,), (44,), (77,)))
    projected = TrainedInputStructureProjector((_fact(),)).project(
        (66, 11, 44, 22, 55))
    with pytest.raises(ValueError, match="composition exceeds"):
        match_relation_question_frames((first, second), projected,
                                       max_combinations=1)


@pytest.mark.parametrize("different", ["operation", "predicate", "role_order"])
def test_question_gap_composition_never_crosses_semantic_contract(different):
    first = _frame()
    second = TrainedRelationQuestionFrame(
        (92002, 2),
        (92004, 2) if different == "operation" else first.operation_key,
        (5, 61) if different == "predicate" else first.predicate_key,
        tuple(reversed(first.roles)) if different == "role_order" else first.roles,
        ((66,), (44,), (77,)),
    )
    projected = TrainedInputStructureProjector((_fact(),)).project(
        (66, 11, 44, 22, 55))
    assert match_relation_question_frames((first, second), projected) == ()


def test_question_rejects_competing_graph_frames_and_propositions():
    projected = TrainedInputStructureProjector((_fact(),)).project((11, 44, 22, 55))
    assert match_relation_question_frames(
        (_frame(), _frame(source=(92002, 2))), projected) == ()
    competing = _fact(proposition=(1, 51), source=(9, 71))
    projected = TrainedInputStructureProjector((_fact(), competing)).project(
        (11, 44, 22, 55))
    assert match_relation_question_frames((_frame(),), projected) == ()


def test_question_binding_requires_its_graph_and_no_unresolved_discourse_gap():
    projected = TrainedInputStructureProjector((_fact(),)).project((11, 44, 22, 55))
    binding = match_relation_question_frames((_frame(),), projected)
    assert len(binding) == 1
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = ()
    bridge.pending_relation_compositions = ()
    with pytest.raises(ValueError, match="no trained graph"):
        bridge._attach_relation_question_binding(replace(
            projected, operation_keys=((92004, 1),),
            relation_question_bindings=binding))
    from pure_integer_ai.cognition.understanding.query_input_structure import (
        InputDiscourseGap, STRUCTURE_OPEN,
    )

    gap = InputDiscourseGap(
        projected.spans[0].ref_key,
        (projected.relation_candidates[0].stable_key(),),
        (projected.relation_candidates[0].stable_key(),),
        STRUCTURE_OPEN,
    )
    gapped = replace(projected, discourse_gaps=(gap,))
    assert match_relation_question_frames((_frame(),), gapped) == ()


def test_connected_compiler_uses_train_only_and_keeps_unsupported_operations_unknown(tmp_path):
    draft = prepare_connected_course(SOURCE, CASES)
    compilation = compile_connected_course(SOURCE, CASES, tmp_path / "pack")
    frames = compile_connected_pair_questions(draft, compilation)
    assert {frame.source_case_key for frame in frames} == {(92002, 1), (92002, 8)}
    assert {frame.operation_key for frame in frames} == {(92004, 1)}
    for frame in frames:
        original = draft.cases[frame.source_case_key[1] - 1]
        assert original["split"] == "train"
        assert all(tuple(map(ord, word)) not in frame.gaps
                   for word in ("北港", "前湾", "城市公交集", "公共交通集"))
    with pytest.raises(ValueError, match="identity drifted"):
        compile_connected_pair_questions(
            replace(draft, case_sha256="wrong"), compilation)
    with pytest.raises(ValueError, match="source identity drifted"):
        compile_connected_pair_questions(
            replace(draft, source_sha256="wrong"), compilation)


def test_full_generation_budget_fails_before_any_graph_write(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pure_integer_ai.experiments import integer_relation_generation_bridge as module

    class ReadOnlyFacts:
        def __init__(self, _path, *, surface_artifact_database=None):
            assert surface_artifact_database is None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def active_surface_facts(self):
            return (SimpleNamespace(), SimpleNamespace())

    database = tmp_path / "bounded.sqlite3"
    database.write_bytes(b"unchanged graph boundary")
    monkeypatch.setattr(module, "TrainedRelationGraphRuntime", ReadOnlyFacts)
    with pytest.raises(IntegerRelationGenerationBridgeError, match="declared budget"):
        materialize_integer_relation_generation(
            database, branch_key=(1,), family_key=(2,), representative_limit=1,
            require_full_active_coverage=True)
    assert database.read_bytes() == b"unchanged graph boundary"
