"""PROPERTY 部分事实与未绑定关系问式的查询路由回归。"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    InputRelationCandidate,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
    OpenRoleFrame,
    OpenRoleSpan,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    _has_unbound_event_time_question_candidate,
    _has_unbound_relation_question_candidate,
    _partial_property_fact_matches,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceBinding,
)


def _property_fixture(
        proposition_value: int = 50,
        *,
        filler_values: tuple[int, ...] = (11, 12, 13, 14),
        ) -> tuple[object, ActiveRelationSurface]:
    """构造带四个角色的训练后 property 事实及其三角色部分输入。"""
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (proposition_value,))
    predicate = ObjectIdentity(OBJECT_CONCEPT, (60,))
    roles = tuple(ObjectIdentity(OBJECT_ROLE, (index,)) for index in range(1, 5))
    fillers = tuple(
        ObjectIdentity(OBJECT_ENTITY, (100 + index,))
        for index in range(1, 5))
    proposition_key = proposition.stable_key()
    predicate_key = predicate.stable_key()
    source_ref = (9,)
    members = [TrainedInputMember(
        (33,), predicate_key, OBJECT_CONCEPT, PROJECTION_PREDICATE,
        proposition_key, predicate_key, source_ref, 0, 1, 0)]
    for ordinal, value in enumerate(filler_values):
        members.append(TrainedInputMember(
            (value,), fillers[ordinal].stable_key(), OBJECT_ENTITY,
            PROJECTION_FILLER, proposition_key, predicate_key, source_ref,
            ordinal + 1, ordinal + 2, ordinal + 1,
            roles[ordinal].stable_key()))
    trained = TrainedInputStructure(
        proposition_key, predicate_key, source_ref, (7, 1), tuple(members))
    # 留下最后一个 value 未观察，模拟“主体+属性+谓词”的部分 PROPERTY 输入。
    projected = TrainedInputStructureProjector((trained,)).project(
        (33, *filler_values[:3]))
    fact = ActiveRelationSurface(
        proposition, predicate, "", tuple(
            RelationSurfaceBinding(role, filler, "", 9, index, index + 1)
            for index, (role, filler) in enumerate(zip(roles, fillers))),
        "", 9, 0, 1)
    return projected, fact


def test_unique_partial_property_fact_requires_exact_role_and_filler_identity():
    projected, fact = _property_fixture()
    matches = _partial_property_fact_matches(projected, (fact,))
    assert len(matches) == 1
    relation, matched_fact = matches[0]
    assert relation.proposition_key == fact.proposition.stable_key()
    assert matched_fact == fact

    wrong_filler = replace(
        fact.bindings[0],
        filler=ObjectIdentity(OBJECT_ENTITY, (999,)),
    )
    assert not _partial_property_fact_matches(
        projected, (replace(fact, bindings=(wrong_filler, *fact.bindings[1:])),))


def test_partial_property_rejects_source_drift_and_competing_propositions():
    projected, fact = _property_fixture()
    assert not _partial_property_fact_matches(
        projected, (replace(fact, source_hash=10),))

    other_projected, other_fact = _property_fixture(51)
    combined = replace(
        projected,
        semantic_candidates=tuple(sorted(
            (*projected.semantic_candidates, *other_projected.semantic_candidates),
            key=lambda item: item.stable_key())),
        relation_candidates=tuple(sorted(
            (*projected.relation_candidates, *other_projected.relation_candidates),
            key=lambda item: item.stable_key())),
    )
    assert not _partial_property_fact_matches(combined, (fact, other_fact))


def test_unbound_trained_relation_question_is_detected_before_alias_fallback():
    proposition = (7, 100)
    predicate = (4, 200)
    source_ref = (9,)
    relation = InputRelationCandidate(
        proposition, predicate, (1, 3), (7, 1), source_ref,
        ((1, 1), (1, 2), (1, 3)),
        required_count=3, support_count=2, role_coverage=2,
        predicate_coverage=1, required_open=1,
        state=STRUCTURE_OPEN, conflict_count=0)
    structure = SimpleNamespace(
        relation_candidates=(relation,),
        relation_question_bindings=(),
        pending_relation_composition_bindings=(),
        discourse_compositions=(),
        discourse_gaps=(),
    )
    frame = OpenRoleFrame(
        proposition, predicate, source_ref,
        ((20, 1), (20, 2)), ((), (), ()), (0, 3))
    assert _has_unbound_relation_question_candidate(structure, (frame,))

    closed = replace(
        relation, support_count=3, required_open=0, state=2)
    assert not _has_unbound_relation_question_candidate(
        SimpleNamespace(
            relation_candidates=(closed,),
            relation_question_bindings=(),
            pending_relation_composition_bindings=(),
            discourse_compositions=(),
            discourse_gaps=(),
        ), (frame,))


def test_unbound_question_ending_blocks_closed_substring_fact_candidate():
    proposition = (7, 101)
    predicate = (4, 201)
    source_ref = (9,)
    relation = InputRelationCandidate(
        proposition, predicate, (1, 4), (7, 1), source_ref,
        ((1, 1), (1, 2), (1, 3)),
        required_count=3, support_count=3, role_coverage=2,
        predicate_coverage=1, required_open=0,
        state=STRUCTURE_CLOSED, conflict_count=0)
    frame = OpenRoleFrame(
        proposition, predicate, source_ref,
        ((20, 1), (20, 2)),
        ((), tuple(map(ord, "指向")), tuple(map(ord, "吗？"))),
        (0, 6))
    structure = SimpleNamespace(
        token_values=tuple(map(ord, "词项甲直接指向概念甲吗？")),
        relation_candidates=(relation,),
        relation_question_bindings=(),
        pending_relation_composition_bindings=(),
        discourse_compositions=(),
        discourse_gaps=(),
    )

    assert _has_unbound_relation_question_candidate(structure, (frame,))
    assert not _has_unbound_relation_question_candidate(
        SimpleNamespace(**{
            **structure.__dict__,
            "token_values": tuple(map(ord, "词项甲指向概念甲")),
        }),
        (frame,),
    )
    assert not _has_unbound_relation_question_candidate(
        SimpleNamespace(**{
            **structure.__dict__,
            "relation_question_bindings": ((1,),),
        }), (frame,))


def test_unbound_event_time_question_is_detected_when_predicate_span_is_missing():
    """Event/Time open roles must stop before the generic frontier route."""
    proposition = (7, 101)
    predicate = (4, 201)
    source_ref = (9,)
    relation = InputRelationCandidate(
        proposition, predicate, (1, 4), (7, 1), source_ref,
        ((1, 1), (1, 2)),
        required_count=3, support_count=1, role_coverage=1,
        predicate_coverage=0, required_open=2,
        state=STRUCTURE_OPEN, conflict_count=0)
    frame = OpenRoleFrame(
        proposition, predicate, source_ref,
        ((20, 1), (20, 2)), ((), (), ()), (0, 2))
    open_relation = OpenRelationCandidate(
        frame, source_ref, (5, 6), (
            OpenRoleSpan((20, 1), source_ref, 0, 0, 1, (5,)),
            OpenRoleSpan((20, 2), source_ref, 1, 1, 2, (6,)),
        ))
    structure = SimpleNamespace(
        relation_candidates=(relation,),
        relation_question_bindings=(),
        pending_relation_composition_bindings=(),
        discourse_compositions=(),
        discourse_gaps=(),
        open_relations=(open_relation,),
    )
    assert _has_unbound_event_time_question_candidate(
        structure, frozenset({predicate}))


def test_unbound_event_time_question_gate_rejects_closed_or_scoped_inputs():
    proposition = (7, 102)
    predicate = (4, 202)
    source_ref = (9,)
    relation = InputRelationCandidate(
        proposition, predicate, (1, 5), (7, 1), source_ref,
        ((1, 1), (1, 2)),
        required_count=3, support_count=1, role_coverage=1,
        predicate_coverage=0, required_open=2,
        state=STRUCTURE_OPEN, conflict_count=0)
    frame = OpenRoleFrame(
        proposition, predicate, source_ref,
        ((20, 1), (20, 2)), ((), (), ()), (0, 2))
    open_relation = OpenRelationCandidate(
        frame, source_ref, (5, 6), (
            OpenRoleSpan((20, 1), source_ref, 0, 0, 1, (5,)),
            OpenRoleSpan((20, 2), source_ref, 1, 1, 2, (6,)),
        ))
    base = dict(
        relation_candidates=(relation,),
        relation_question_bindings=(),
        pending_relation_composition_bindings=(),
        discourse_compositions=(),
        discourse_gaps=(),
        open_relations=(open_relation,),
    )
    assert not _has_unbound_event_time_question_candidate(
        SimpleNamespace(**{**base, "discourse_gaps": (object(),)}),
        frozenset({predicate}))
    assert not _has_unbound_event_time_question_candidate(
        SimpleNamespace(**{**base, "relation_question_bindings": (object(),)}),
        frozenset({predicate}))
    closed = replace(relation, support_count=3, required_open=0, state=2)
    assert not _has_unbound_event_time_question_candidate(
        SimpleNamespace(**{**base, "relation_candidates": (closed,)}),
        frozenset({predicate}))
