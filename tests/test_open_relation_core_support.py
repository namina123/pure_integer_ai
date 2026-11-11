"""Fail-closed projection of active Core support into open role queries."""
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE as OBJECT_ROLE_KIND,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_OBSERVATION,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    SPAN_SOURCE,
    STRUCTURE_OPEN,
    InputSemanticCandidate,
    InputSpan,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
    OpenRoleFrame,
    OpenRoleSpan,
)
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry,
    EvidenceEntry,
)
from pure_integer_ai.experiments import trained_graph_query_bridge as bridge_module
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    _CrossSpaceExpander,
    _generic_response_adoption_allowed,
    _observed_input_role_keys,
    _root_response_context_inputs,
    _validate_response_context_memory_closure,
)
from pure_integer_ai.experiments.trained_open_response_runtime import (
    _discourse_memory_refs,
    _reference_generation_provenance,
    generate_open_responses,
)
from pure_integer_ai.experiments.run_open_role_memory_slice import (
    _validate_resolved_reference_generation,
)
from pure_integer_ai.experiments.trained_reference_resolution import (
    CrossTurnReferenceCandidate,
    CrossTurnReferenceResolution,
    REFERENCE_RESOLVED,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_REFERS,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
)
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredFactualResponseContext,
    DeliveredFactualSlot,
)
from pure_integer_ai.experiments.trained_response_delivery import (
    resolve_factual_claim_input_hypotheses,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
    MemoryActiveStructures,
    MemoryEvidenceStructure,
    MemoryHypothesisStructure,
    MemoryObservationStructure,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    InputRoleOccurrence,
    ReferenceCandidateSet,
    ResponseRoleOccurrence,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import ResolverDecision
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


SOURCE_REF = (41, 7001)
PROPOSITION = (9001, 1)
PREDICATE = (9002, 1)
SUBJECT_ROLE = (9010, 1)
OBJECT_ROLE = (9010, 2)
SUBJECT_FILLER = (9020, 1)
OBJECT_FILLER = (9020, 2)
SCOPE = (9030, 1)


def _fixture(*, subject_candidates=(SUBJECT_FILLER,), relation_count=1,
             open_candidate_count=1, include_object_span=True):
    frame = OpenRoleFrame(
        PROPOSITION,
        PREDICATE,
        SOURCE_REF,
        (SUBJECT_ROLE, OBJECT_ROLE),
        ((), (71,), ()),
        (0, 5),
    )
    candidate = OpenRelationCandidate(
        frame,
        SOURCE_REF,
        (11, 71, 12, 13, 14),
        (
            OpenRoleSpan(SUBJECT_ROLE, SOURCE_REF, 0, 0, 1, (11,)),
            OpenRoleSpan(OBJECT_ROLE, SOURCE_REF, 1, 2, 5, (12, 13, 14)),
        ),
    )
    spans = (InputSpan(SOURCE_REF, 0, 1, SPAN_SOURCE, (11,)),)
    if include_object_span:
        spans += (InputSpan(SOURCE_REF, 2, 5, SPAN_SOURCE, (12, 13, 14)),)
    semantic = tuple(sorted((
        InputSemanticCandidate(
            key, key[0], PROJECTION_FILLER, spans[0].ref_key,
            PROPOSITION, PREDICATE, SUBJECT_ROLE, SOURCE_REF, 0,
        )
        for key in subject_candidates
    ), key=lambda item: item.stable_key()))
    relation_candidate = SimpleNamespace(
        proposition_key=PROPOSITION,
        predicate_coverage=1,
        state=STRUCTURE_OPEN,
    )
    input_structure = SimpleNamespace(
        spans=spans,
        semantic_candidates=semantic,
        relation_candidates=tuple(
            relation_candidate for _ in range(relation_count)),
        open_relations=tuple(candidate for _ in range(open_candidate_count)),
    )
    source = SimpleNamespace(
        hypothesis=SimpleNamespace(scope=SimpleNamespace(
            stable_key=lambda: SCOPE)))
    owner = SimpleNamespace(core_runtime=SimpleNamespace(
        generation_input=lambda _proposition: source,
        evidence_history=lambda _proposition: (),
        context=SimpleNamespace(scoped_identity_store=SimpleNamespace(
            registry=object())),
    ))
    expander = object.__new__(_CrossSpaceExpander)
    expander.owner = owner
    expander.input_structure = input_structure
    expander.open_candidates_by_frame = {
        frame.stable_key(): input_structure.open_relations,
    }
    return expander, frame, SimpleNamespace(
        proposition=SimpleNamespace(stable_key=lambda: PROPOSITION),
        bindings=(
            SimpleNamespace(role=SimpleNamespace(
                stable_key=lambda: SUBJECT_ROLE), filler=SimpleNamespace(
                    stable_key=lambda: SUBJECT_FILLER)),
            SimpleNamespace(role=SimpleNamespace(
                stable_key=lambda: OBJECT_ROLE), filler=SimpleNamespace(
                    stable_key=lambda: OBJECT_FILLER)),
        ),
    )


def test_open_role_keys_are_cached_without_changing_value_semantics():
    _expander, frame, _fact = _fixture()
    bindings = (
        OpenRoleSpan(SUBJECT_ROLE, SOURCE_REF, 0, 0, 1, (11,)),
        OpenRoleSpan(OBJECT_ROLE, SOURCE_REF, 1, 2, 5, (12, 13, 14)),
    )
    candidate = OpenRelationCandidate(
        frame, SOURCE_REF, (11, 71, 12, 13, 14), bindings)

    frame_key = frame.stable_key()
    schema_key = frame.schema_key()
    candidate_key = candidate.stable_key()
    binding_keys = tuple(item.stable_key() for item in bindings)
    assert frame.stable_key() is frame_key
    assert frame.schema_key() is schema_key
    assert candidate.stable_key() is candidate_key
    assert all(item.stable_key() is key
               for item, key in zip(bindings, binding_keys, strict=True))

    restored_frame = OpenRoleFrame.from_stable_key(frame_key)
    restored_candidate = OpenRelationCandidate.from_stable_key(candidate_key)
    assert restored_frame == frame
    assert hash(restored_frame) == hash(frame)
    assert restored_frame.stable_key() == frame_key
    assert restored_candidate == candidate
    assert hash(restored_candidate) == hash(candidate)
    assert restored_candidate.stable_key() == candidate_key


def test_open_support_projects_only_known_matching_roles(monkeypatch):
    expander, frame, fact = _fixture()
    support = EvidenceEntry((1,), PROPOSITION, polarity=1)
    monkeypatch.setattr(
        bridge_module, "core_query_evidence", lambda *_args: (support,))
    bindings = []
    evidence = []

    accepted = expander._record_open_relation_support(
        frame, fact, bindings, evidence)

    assert accepted
    assert [(item.role_key, item.filler_key) for item in bindings] == [
        (SUBJECT_ROLE, SUBJECT_FILLER),
    ]
    assert OBJECT_ROLE not in {item.role_key for item in bindings}
    assert evidence == [support]


def test_open_support_keeps_unindexed_dynamic_role_unknown(monkeypatch):
    expander, frame, fact = _fixture(include_object_span=False)
    support = EvidenceEntry((1,), PROPOSITION, polarity=1)
    monkeypatch.setattr(
        bridge_module, "core_query_evidence", lambda *_args: (support,))
    bindings = []
    evidence = []

    accepted = expander._record_open_relation_support(
        frame, fact, bindings, evidence)

    assert accepted
    assert [(item.role_key, item.filler_key) for item in bindings] == [
        (SUBJECT_ROLE, SUBJECT_FILLER),
    ]
    assert OBJECT_ROLE not in {item.role_key for item in bindings}
    assert evidence == [support]


def test_open_support_rejects_conflict_missing_predicate_and_ambiguity(monkeypatch):
    support = EvidenceEntry((1,), PROPOSITION, polarity=1)
    monkeypatch.setattr(
        bridge_module, "core_query_evidence", lambda *_args: (support,))

    for options in (
        {"subject_candidates": ((9020, 9),)},
        {"subject_candidates": ()},
        {"subject_candidates": (SUBJECT_FILLER, (9020, 9))},
        {"relation_count": 0},
        {"relation_count": 2},
        {"open_candidate_count": 2},
    ):
        expander, frame, fact = _fixture(**options)
        bindings = []
        evidence = []
        assert not expander._record_open_relation_support(
            frame, fact, bindings, evidence)
        assert bindings == []
        assert evidence == []


def test_observed_open_role_is_not_a_cross_turn_candidate():
    """Only an unfilled integer role may create a REFERS candidate set."""
    source = SourceRef(41, 7001, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    source_key = source.stable_key()
    subject_span = InputSpan(source_key, 0, 1, SPAN_SOURCE, (11,))
    target_span = InputSpan(source_key, 2, 5, SPAN_SOURCE, (12, 13, 14))
    subject = InputSemanticCandidate(
        SUBJECT_FILLER, SUBJECT_FILLER[0], PROJECTION_FILLER,
        subject_span.ref_key, PROPOSITION, PREDICATE, SUBJECT_ROLE,
        source_key, 0)
    structure = SimpleNamespace(
        spans=(subject_span, target_span),
        semantic_candidates=(subject,),
    )

    observed = _observed_input_role_keys(structure)

    assert SUBJECT_ROLE in observed
    assert OBJECT_ROLE not in observed


def test_reference_projection_uses_unique_observed_open_relation():
    """A unique observed subject may authorize only that REFERS frame."""
    source = SourceRef(41, 7001, 0, GLOBAL_OWNER_SCOPE, VersionBundle()).stable_key()
    subject_role = ObjectIdentity(OBJECT_ROLE_KIND, (9010, 1))
    target_role = ObjectIdentity(OBJECT_ROLE_KIND, (9010, 2))
    subject = ObjectIdentity(OBJECT_ENTITY, (9020, 1))
    frame = OpenRoleFrame(
        PROPOSITION,
        PREDICATE,
        source,
        (subject_role.stable_key(), target_role.stable_key()),
        ((), (71,), ()),
        (0, 5),
    )
    candidate = OpenRelationCandidate(
        frame,
        source,
        (11, 71, 12, 13, 14),
        (
            OpenRoleSpan(subject_role.stable_key(), source, 0, 0, 1, (11,)),
            OpenRoleSpan(target_role.stable_key(), source, 1, 2, 5, (12, 13, 14)),
        ),
    )
    subject_span = InputSpan(source, 0, 1, SPAN_SOURCE, (11,))
    target_span = InputSpan(source, 2, 5, SPAN_SOURCE, (12, 13, 14))
    semantic = InputSemanticCandidate(
        subject.stable_key(), OBJECT_ENTITY, PROJECTION_FILLER,
        subject_span.ref_key, PROPOSITION, PREDICATE,
        subject_role.stable_key(), source, 0,
    )
    structure = SimpleNamespace(
        source_ref=source,
        spans=(subject_span, target_span),
        semantic_candidates=(semantic,),
        open_relations=(candidate,),
    )
    reference = SimpleNamespace(
        role=target_role,
        ordinal=1,
        values=(12, 13, 14),
        source_ref=source,
    )
    candidate_set = SimpleNamespace(reference=reference)
    bridge = object.__new__(bridge_module.TrainedGraphQueryBridge)

    assert bridge._reference_current_objects(candidate_set, structure) == (subject,)

    ambiguous = SimpleNamespace(
        source_ref=source,
        spans=structure.spans,
        semantic_candidates=structure.semantic_candidates,
        open_relations=(candidate, candidate),
    )
    assert bridge._reference_current_objects(candidate_set, ambiguous) == ()

    missing_semantic = SimpleNamespace(
        source_ref=source,
        spans=structure.spans,
        semantic_candidates=(),
        open_relations=structure.open_relations,
    )
    assert bridge._reference_current_objects(candidate_set, missing_semantic) == ()


def test_factual_delivery_maps_closed_relation_hypothesis_to_claim():
    """Closed relation input keeps exact claim/H provenance for delivery roles."""
    memory_space = SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1)

    def memory_ref(kind, key):
        return MemoryObjectRef(memory_space, GLOBAL_OWNER_SCOPE,
                               VersionBundle(), kind, key)

    observation = memory_ref(MEMORY_OBJECT_OBSERVATION, (1,))
    hypothesis = memory_ref(MEMORY_OBJECT_HYPOTHESIS, (2,))
    candidate = (
        91502,
        1, 5,
        2, *PROPOSITION,
        2, *PREDICATE,
        2, 9003, 1,
        2, 9004, 1,
        7, 2, 2, 2, 1, 0, 2, 0,
    )
    row = SimpleNamespace(
        observation_key=observation.stable_key(),
        hypothesis_key=hypothesis.stable_key(),
        candidate_key=candidate,
    )

    class Memory(TrainedDialogueMemoryGraph):
        def active_structures(self, **kwargs):
            assert kwargs == {
                "candidate_keys": (),
                "observation_refs": (observation,),
                "context_depth_limit": 0,
            }
            return SimpleNamespace(hypotheses=(row,))

    memory = object.__new__(Memory)
    assert resolve_factual_claim_input_hypotheses(
        memory, observation, (PROPOSITION,)) == ((PROPOSITION, hypothesis),)


def test_reference_target_adoption_joins_missing_target_occurrence():
    """The resolved REFERS occurrence is the open ``to`` role, not ``from``."""
    source = SourceRef(41, 7001, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    source_key = source.stable_key()
    predicate = authored_relation_identity(RELATION_REFERS)
    from_role = authored_relation_role_identity(ROLE_REFERS_FROM)
    to_role = authored_relation_role_identity(ROLE_REFERS_TO)
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (9901, 1))
    from_filler = ObjectIdentity(OBJECT_ENTITY, (9902, 1))
    target = ObjectIdentity(OBJECT_ENTITY, (9902, 2))
    values = (11, 71, 12, 13, 14)
    frame = OpenRoleFrame(
        proposition.stable_key(), predicate.stable_key(), (7001,),
        (from_role.stable_key(), to_role.stable_key()), ((), (71,), ()),
        (0, len(values)),
    )
    open_candidate = OpenRelationCandidate(
        frame, source_key, values,
        (
            OpenRoleSpan(from_role.stable_key(), source_key, 0, 0, 1, (11,)),
            OpenRoleSpan(to_role.stable_key(), source_key, 1, 2, 5,
                         (12, 13, 14)),
        ),
    )
    spans = (
        InputSpan(source_key, 0, 1, SPAN_SOURCE, (11,)),
        InputSpan(source_key, 1, 2, SPAN_SOURCE, (71,)),
        InputSpan(source_key, 2, 5, SPAN_SOURCE, (12, 13, 14)),
    )
    fillers = (
        InputSemanticCandidate(
            from_filler.stable_key(), from_filler.object_kind, PROJECTION_FILLER,
            spans[0].ref_key, proposition.stable_key(), predicate.stable_key(),
            from_role.stable_key(), source_key, 0),
    )
    predicate_candidate = InputSemanticCandidate(
        predicate.stable_key(), predicate.object_kind, PROJECTION_PREDICATE,
        spans[1].ref_key, proposition.stable_key(), predicate.stable_key(),
        (), source_key, 0)
    relation = SimpleNamespace(
        proposition_key=proposition.stable_key(),
        predicate_key=predicate.stable_key(),
        state=STRUCTURE_OPEN,
        conflict_count=0,
        predicate_coverage=1,
        required_count=3,
        support_count=2,
        role_coverage=1,
        required_open=1,
        span_refs=tuple(sorted((spans[0].ref_key, spans[1].ref_key))),
    )
    input_structure = SimpleNamespace(
        source_ref=source_key,
        token_values=values,
        spans=spans,
        open_relations=(open_candidate,),
        relation_candidates=(relation,),
        semantic_candidates=fillers + (predicate_candidate,),
    )
    fact = SimpleNamespace(
        proposition=proposition,
        predicate=predicate,
        source_hash=7001,
        bindings=(
            SimpleNamespace(role=from_role, filler=from_filler),
            SimpleNamespace(role=to_role, filler=target),
        ),
    )
    hypothesis_key = (91536, 1, 7001)
    reference = SimpleNamespace(
        role=to_role,
        ordinal=1,
        values=(12, 13, 14),
        source_ref=(7001, *source_key),
        occurrence=occurrence_identity(source, start=2, end=5, ordinal=0),
        origin=span_identity(source, members=((2, 5),)),
    )
    winner = SimpleNamespace(
        hypothesis=SimpleNamespace(stable_key=lambda: hypothesis_key),
    )
    resolution = SimpleNamespace(
        requires_resolution=1,
        status=REFERENCE_RESOLVED,
        winner=winner,
        supporting_candidates=(winner,),
        winner_target=target,
        winner_hypothesis_key=hypothesis_key,
        adoption_support_key=hypothesis_key,
        winner_target_key=target.stable_key(),
        candidate_set=SimpleNamespace(reference=reference),
    )
    bridge = object.__new__(_CrossSpaceExpander)
    bridge.owner = SimpleNamespace(
        _reference_resolutions=(resolution,),
        facts=(fact,),
    )
    bridge.input_structure = input_structure

    adopted = bridge._validated_reference_target_adoption(SimpleNamespace())

    assert adopted is not None
    assert adopted[-1] == target


def test_materialized_reference_target_adoption_survives_closed_input_relation():
    """A closed adopted relation is recovered only from its QueryState proof."""
    predicate = authored_relation_identity(RELATION_REFERS)
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (9911, 1))
    target = ObjectIdentity(OBJECT_ENTITY, (9912, 1))
    fact = SimpleNamespace(proposition=proposition, predicate=predicate)
    hypothesis_key = (91536, 1, 7002)
    target_key = target.stable_key()
    resolution_key = (91538, 2, 7002)
    resolution = SimpleNamespace(
        requires_resolution=1,
        status=REFERENCE_RESOLVED,
        stable_key=lambda: resolution_key,
        winner_hypothesis_key=hypothesis_key,
        adoption_support_key=hypothesis_key,
        winner_target_key=target_key,
    )
    winner = SimpleNamespace(
        hypothesis=SimpleNamespace(stable_key=lambda: hypothesis_key))
    candidate = SimpleNamespace(stable_key=lambda: (91539, 1, 7002))
    open_candidate = SimpleNamespace(stable_key=lambda: (91540, 1, 7002))
    record = _CrossSpaceExpander._reference_adoption_record(
        resolution, fact, candidate, open_candidate, target)
    marker_key = (*bridge_module._REFERENCE_TARGET_ADOPTION_EVIDENCE, *record)
    bridge = object.__new__(_CrossSpaceExpander)
    bridge.owner = SimpleNamespace(
        _reference_resolutions=(resolution,),
        core_fact_index={proposition.stable_key(): fact},
        dialogue_frame_index={proposition.stable_key(): object()},
    )
    bridge._reference_adoption_record_key = record
    state = SimpleNamespace(
        evidence=(EvidenceEntry(
            (7002,), proposition.stable_key(), space=1, polarity=3,
            evidence_key=marker_key, payload_key=record),),
        bindings=(BindingEntry(
            bridge_module._REFERENCE_TARGET_ADOPTION_ROLE, record,
            space=1),),
    )

    assert bridge._materialized_reference_target_adoption_proposition(
        state) == proposition.stable_key()

    detached = SimpleNamespace(
        evidence=state.evidence, bindings=())
    assert bridge._materialized_reference_target_adoption_proposition(
        detached) is None


def test_open_typed_projection_skips_unrelated_role_shape():
    from pure_integer_ai.cognition.understanding.query_open_roles import (
        OpenRelationCandidate, OpenRoleFrame, OpenRoleSpan,
    )
    from pure_integer_ai.experiments.trained_typed_relation_bridge import (
        project_open_typed_relation,
    )

    source_ref = (9, 10)
    role = (8, 31)
    frame = OpenRoleFrame(
        (5, 1), (6, 14), source_ref, (role,), ((), ()), (0, 1))
    binding = OpenRoleSpan(role, source_ref, 0, 0, 1, (65,))
    candidate = OpenRelationCandidate(
        frame, source_ref, (65,), (binding,))

    assert project_open_typed_relation(
        candidate, SimpleNamespace(relation_kind=10)) is None


def test_unresolved_required_reference_stays_unknown_without_exception():
    unresolved = SimpleNamespace(requires_resolution=True, status=2)
    assert generate_open_responses(
        None,
        None,
        None,
        (object(),),
        (),
        {},
        (object(),),
        reference_resolutions=(unresolved,),
    ) == ()


def test_resolved_reference_cannot_be_replaced_by_generic_companion():
    """The delivered plan must retain REFERS provenance after arbitration."""
    generic_plan = object()
    assert _generic_response_adoption_allowed(
        None, generic_plan, generic_structural_preferred=True,
        resolved_reference_resolutions=((1,),)) is True
    assert _generic_response_adoption_allowed(
        object(), generic_plan, generic_structural_preferred=True,
        resolved_reference_resolutions=((1,),)) is False
    assert _generic_response_adoption_allowed(
        object(), generic_plan, generic_structural_preferred=True,
        resolved_reference_resolutions=()) is True


def test_resolved_reference_accepts_factual_delivery_provenance():
    """Resolved REFERS may carry a factual delivery context without `.context`."""
    source = SourceRef(41, 7002, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    input_source = SourceRef(41, 7003, 2, GLOBAL_OWNER_SCOPE, VersionBundle())
    memory_space = SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1)

    def memory_ref(kind, key):
        return MemoryObjectRef(memory_space, GLOBAL_OWNER_SCOPE,
                               VersionBundle(), kind, key)

    output_observation = memory_ref(MEMORY_OBJECT_OBSERVATION, (1,))
    output_hypothesis = memory_ref(MEMORY_OBJECT_HYPOTHESIS, (2,))
    input_observation = memory_ref(MEMORY_OBJECT_OBSERVATION, (3,))
    role = ObjectIdentity(OBJECT_ROLE_KIND, (1,))
    target = ObjectIdentity(OBJECT_ENTITY, (2,))
    proof = ObjectIdentity(OBJECT_CONTEXT_SCOPE, (3,))
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (4,))
    occurrence = ResponseRoleOccurrence(
        ObjectIdentity(OBJECT_ENTITY, (5,)), ObjectIdentity(OBJECT_CONTEXT_SCOPE, (6,)),
        role, proof, ObjectIdentity(OBJECT_CONTEXT_SCOPE, (7,)),
        output_observation, output_hypothesis, 0,
        (11, *source.stable_key()), document_scope(source).stable_key(),
    )
    factual = DeliveredFactualResponseContext(
        output_observation, output_hypothesis, (11, *source.stable_key()),
        document_scope(source).stable_key(), proof, input_observation,
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (8,)), proposition.stable_key(), (proposition,),
        (DeliveredFactualSlot(role, target, 1, 11, (65,), (OBJECT_ENTITY,)),),
        (ObjectIdentity(OBJECT_CONTEXT_SCOPE, (9,)),), ((10,),), ((12,),), 0,
        (occurrence,),
    )
    reference = InputRoleOccurrence(
        ObjectIdentity(OBJECT_ENTITY, (13,)), ObjectIdentity(OBJECT_CONTEXT_SCOPE, (14,)),
        role, input_observation, memory_ref(MEMORY_OBJECT_HYPOTHESIS, (15,)),
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (16,)), 0,
        (12, *input_source.stable_key()), document_scope(input_source).stable_key(),
        (65,),
    )
    hypothesis = HypothesisKey(
        (91536, 1), (91536, 2), (91536, 3),
        document_scope(input_source), input_source,
    )
    candidate_set = ReferenceCandidateSet(reference, (occurrence,), (0,))
    candidate = CrossTurnReferenceCandidate(
        occurrence, factual.stable_key(), 0, (target,), hypothesis,
    )
    resolution = CrossTurnReferenceResolution(
        candidate_set, (candidate,), (), (),
        ResolverDecision(1, (91536, 4), 0, 0, (), (), ()), 1,
        REFERENCE_RESOLVED, hypothesis.stable_key(), target.stable_key(),
    )
    state = SimpleNamespace(bindings=resolution.query_bindings(), evidence=())

    objects, _evidence, reference_memory, targets = _reference_generation_provenance(
        state, (factual,), (resolution,))
    delivered_memory = _discourse_memory_refs((factual,))

    assert target in targets
    assert target in objects
    assert {output_observation, output_hypothesis} <= set(reference_memory)
    assert {output_observation, output_hypothesis, input_observation} <= set(delivered_memory)

    # Factual kind=1 generation carries typed representations rather than the
    # open-role-only ``generation.discourse_context`` field.  The checker must
    # validate the same winner through the integer ResponsePlan provenance.
    trace = {
        "generation": {"kind": 1, "representations": [[1]]},
        "response_plan": {
            "discourse_links": [list(item.stable_key())
                                 for item in resolution.discourse_objects()],
            "event_refs": [],
            "memory_refs": [list(output_observation.stable_key()),
                            list(output_hypothesis.stable_key())],
        },
    }
    assert _validate_resolved_reference_generation(trace, (resolution,)) == 1


def test_response_context_input_observation_is_a_bounded_memory_root():
    source = SourceRef(41, 7002, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    memory_space = SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1)

    def memory_ref(kind, key):
        return MemoryObjectRef(memory_space, GLOBAL_OWNER_SCOPE,
                               VersionBundle(), kind, key)

    output_observation = memory_ref(MEMORY_OBJECT_OBSERVATION, (31,))
    output_hypothesis = memory_ref(MEMORY_OBJECT_HYPOTHESIS, (32,))
    input_observation = memory_ref(MEMORY_OBJECT_OBSERVATION, (33,))
    role = ObjectIdentity(OBJECT_ROLE_KIND, (1,))
    target = ObjectIdentity(OBJECT_ENTITY, (2,))
    proof = ObjectIdentity(OBJECT_CONTEXT_SCOPE, (3,))
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (4,))
    factual = DeliveredFactualResponseContext(
        output_observation, output_hypothesis, (11, *source.stable_key()),
        document_scope(source).stable_key(), proof, input_observation,
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (8,)), proposition.stable_key(),
        (proposition,),
        (DeliveredFactualSlot(role, target, 1, 11, (65,), (OBJECT_ENTITY,)),),
        (ObjectIdentity(OBJECT_CONTEXT_SCOPE, (9,)),), ((10,),), ((12,),), 1,
    )

    def closure(observation, hypothesis, turn_seq):
        observation_key = observation.stable_key()
        hypothesis_key = hypothesis.stable_key()
        evidence_key = (turn_seq + 100,)
        source_ref = (turn_seq + 200, *source.stable_key())
        scope_key = document_scope(source).stable_key()
        return MemoryActiveStructures(
            (MemoryObservationStructure(
                observation_key, source_ref, scope_key, turn_seq + 1, turn_seq,
                1, (hypothesis_key,)),),
            (MemoryHypothesisStructure(
                observation_key, hypothesis_key, (91532, 1), (91532, 2),
                (91532, 3), source_ref, scope_key, (evidence_key,)),),
            (MemoryEvidenceStructure(
                evidence_key, hypothesis_key, source_ref, scope_key, 3, (),
                (turn_seq + 300,)),),
            ((91532, turn_seq),),
        )

    original = closure(output_observation, output_hypothesis, 1)
    added = closure(input_observation,
                    memory_ref(MEMORY_OBJECT_HYPOTHESIS, (34,)), 2)

    class Memory:
        def active_structures(self, **kwargs):
            assert kwargs == {
                "candidate_keys": (),
                "observation_refs": (input_observation,),
                "context_depth_limit": 0,
            }
            return added

    rooted = _root_response_context_inputs(Memory(), original, (factual,))
    _validate_response_context_memory_closure(rooted, (factual,))

    assert {item.observation_key for item in rooted.observations} == {
        output_observation.stable_key(), input_observation.stable_key(),
    }
    assert len(rooted.hypotheses) == 1
    assert len(rooted.evidence) == 1
