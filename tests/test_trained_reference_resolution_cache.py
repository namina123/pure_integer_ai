from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    HypothesisKey, EVIDENCE_SUPPORT, EVIDENCE_REFUTE,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_ROLE,
    SourceRef,
    VersionBundle,
    ObjectIdentity,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_OBSERVATION,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity
from pure_integer_ai.experiments.trained_reference_resolution import (
    CrossTurnReferenceCandidate,
    REFERENCE_DIMENSION_REFERS,
    REFERENCE_CONFLICT,
    REFERENCE_OPEN,
    REFERENCE_RESOLVED,
    TrainedCrossTurnReferenceRuntime,
    TrainedReferenceEvidence,
    _PreparedReference,
    _MAX_SEMANTIC_OBJECT_CACHE_ENTRIES,
)
from pure_integer_ai.experiments.reference_target_support import (
    REFERENCE_TARGET_SUPPORT_VERSION, ReferenceTargetSupport, reference_target_support,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    InputRoleOccurrence,
    ReferenceCandidateSet,
    ResponseRoleOccurrence,
)


def _reference_fixture():
    """构造带真实角色身份、训练图证据与 H-00/H-04 消费者的有界参考状态。"""
    source = SourceRef(41, 7301, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    antecedent_source = SourceRef(
        41, 7302, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    memory_space = SpaceIdentity(SPACE_TYPE_MEMORY, 1, 1)

    def memory_ref(kind, key):
        return MemoryObjectRef(
            memory_space, GLOBAL_OWNER_SCOPE, VersionBundle(), kind, key)

    role = ObjectIdentity(OBJECT_ROLE, (1,))
    antecedent = ResponseRoleOccurrence(
        ObjectIdentity(OBJECT_ENTITY, (2,)),
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (3,)), role,
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (4,)),
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (5,)),
        memory_ref(MEMORY_OBJECT_OBSERVATION, (6,)),
        memory_ref(MEMORY_OBJECT_HYPOTHESIS, (7,)), 0,
        (11, *antecedent_source.stable_key()),
        document_scope(antecedent_source).stable_key(),
    )
    reference = InputRoleOccurrence(
        ObjectIdentity(OBJECT_ENTITY, (8,)),
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (9,)), role,
        memory_ref(MEMORY_OBJECT_OBSERVATION, (10,)),
        memory_ref(MEMORY_OBJECT_HYPOTHESIS, (11,)),
        ObjectIdentity(OBJECT_CONTEXT_SCOPE, (12,)), 0,
        (12, *source.stable_key()), document_scope(source).stable_key(), (65,),
    )
    candidate_set = ReferenceCandidateSet(reference, (antecedent,), (0,))
    antecedent_key = antecedent.stable_key()
    reference_key = reference.stable_key()
    candidate_set_key = candidate_set.stable_key()
    assert antecedent.stable_key() is antecedent_key
    assert replace(antecedent) == antecedent
    assert hash(replace(antecedent)) == hash(antecedent)
    assert reference.stable_key() is reference_key
    assert replace(reference) == reference
    assert hash(replace(reference)) == hash(reference)
    assert candidate_set.stable_key() is candidate_set_key
    assert hash(replace(candidate_set)) == hash(candidate_set)
    hypothesis = HypothesisKey(
        (91536, 1), (91536, 2), (91536, 3), document_scope(source), source)
    target = ObjectIdentity(OBJECT_ENTITY, (13,))
    candidate = CrossTurnReferenceCandidate(
        antecedent, (91535, 1), 0, (target,), hypothesis)
    proposition_key = (91536, 100)
    evidence = TrainedReferenceEvidence(
        antecedent.stable_key(), target, REFERENCE_DIMENSION_REFERS,
        EVIDENCE_SUPPORT, 1, (proposition_key,), ((91536, 101),),
        antecedent_source, 7302, (91536, 102),
    )
    prepared = _PreparedReference(
        candidate_set, (candidate,), (evidence,), 1, ())

    runtime = object.__new__(TrainedCrossTurnReferenceRuntime)
    runtime._prepared = OrderedDict()
    runtime._resolved = OrderedDict()
    runtime._prepare = lambda *_args, **_kwargs: prepared
    return runtime, candidate_set, target, proposition_key, prepared


def test_reference_resolution_cache_rechecks_new_core_support():
    runtime, candidate_set, target, proposition_key, _prepared = _reference_fixture()

    unresolved = runtime.resolve(
        candidate_set, (), core_support=frozenset(), timestamp_seq=1)
    resolved = runtime.resolve(
        candidate_set, (), core_support=frozenset({proposition_key}),
        timestamp_seq=1)
    replay = runtime.resolve(
        candidate_set, (), core_support=frozenset({proposition_key}),
        timestamp_seq=1)

    assert unresolved.status == REFERENCE_OPEN
    assert unresolved.winner is None
    assert resolved.status == REFERENCE_RESOLVED
    assert resolved.winner_target == target
    assert replay is resolved


def test_factual_reference_reads_exact_delivery_slot_without_surface_projection():
    """同名表面不能扩张已经交付的整数对象，角色/证明漂移则失败关闭。"""
    from pure_integer_ai.experiments.trained_response_context import (
        DeliveredFactualResponseContext, DeliveredFactualSlot,
    )

    runtime, candidate_set, target, _proposition, _prepared = _reference_fixture()
    occurrence, = candidate_set.antecedents
    slot = DeliveredFactualSlot(occurrence.role, target, 1, 1, (1,),
                                (OBJECT_ENTITY,), occurrence.hypothesis)
    context = DeliveredFactualResponseContext(
        occurrence.observation, occurrence.hypothesis, occurrence.source_ref,
        occurrence.scope_key, occurrence.proof_ref, occurrence.observation,
        occurrence.role, (1,), (), (slot,), (), (), ((1,),), 1, (occurrence,))
    runtime._semantic_objects = lambda *_a: (_ for _ in ()).throw(AssertionError('surface projection forbidden'))
    runtime._occurrence_values = lambda *_a: (_ for _ in ()).throw(AssertionError('surface read forbidden'))
    assert runtime.semantic_objects_for(occurrence, (context,)) == (target,)
    malformed = replace(context, slots=(replace(slot, role=ObjectIdentity(OBJECT_ROLE, (999,))),))
    with pytest.raises(ValueError, match='original typed slot'):
        runtime.semantic_objects_for(occurrence, (malformed,))


def _repeated_reference_fixture(mode="same"):
    """两个历史 occurrence 独立保留，改变目标/证据而不改变解析或来源门。"""
    runtime, original, target, proposition, prepared = _reference_fixture()
    first = prepared.candidates[0]
    occurrence = first.occurrence
    second_occurrence = replace(occurrence,
        occurrence=ObjectIdentity(OBJECT_ENTITY, (21,)),
        observation=replace(occurrence.observation, object_key=(22,)),
        hypothesis=replace(occurrence.hypothesis, object_key=(23,)))
    other_target = ObjectIdentity(OBJECT_ENTITY, (24,)) if mode == "different" else target
    second = replace(first, occurrence=second_occurrence, context_key=(91535, 2),
        context_depth=3, semantic_objects=(other_target,),
        hypothesis=replace(first.hypothesis, candidate_key=(91536, 21)))
    candidates = tuple(sorted((first, second), key=lambda item: item.stable_key()))
    candidate_set = replace(original,
        antecedents=tuple(item.occurrence for item in candidates),
        context_depths=tuple(item.context_depth for item in candidates))
    second_evidence = replace(prepared.evidence[0], antecedent_key=second_occurrence.stable_key(),
                              target=other_target, decisive=0 if mode == "nondecisive" else 1)
    evidence = (*prepared.evidence, second_evidence)
    if mode == "refuted":
        evidence = (*evidence, replace(second_evidence, stance=EVIDENCE_REFUTE, decisive=0))
    prepared = _PreparedReference(candidate_set, candidates,
        tuple(sorted(evidence, key=lambda item: item.stable_key())), 1, ())
    runtime._prepare = lambda *_args, **_kwargs: prepared
    support = frozenset() if mode == "missing_core" else frozenset({proposition})
    result = runtime.resolve(candidate_set, (), core_support=support, timestamp_seq=1)
    return result, target


@pytest.mark.parametrize("mode,status", (
    ("same", REFERENCE_RESOLVED), ("different", REFERENCE_CONFLICT),
    ("refuted", REFERENCE_CONFLICT), ("nondecisive", REFERENCE_CONFLICT),
    ("missing_core", REFERENCE_OPEN),
))
def test_repeated_history_requires_one_exact_target_and_complete_h00_h04_support(mode, status):
    """共同目标可保留全部历史支持，异目标/反驳/非决定性/缺读取不产生支持记录。"""
    result, target = _repeated_reference_fixture(mode)
    assert result.status == status
    assert result.winner is None
    if mode != "same":
        assert result.target_support is None
        assert not result.winner_target_key
        return
    assert result.winner_target == target
    assert not result.winner_hypothesis_key
    assert len(result.supporting_candidates) == len(result.target_support.members) == 2
    assert set(result.support_context_keys) == {item.context_key for item in result.candidates}
    assert ReferenceTargetSupport.from_stable_key(result.target_support.stable_key()) == result.target_support
    assert result.adoption_support_key == result.target_support.stable_key()
    assert any(item.role_key == (REFERENCE_TARGET_SUPPORT_VERSION, 1)
               and item.filler_key == result.target_support.stable_key()
               for item in result.query_bindings())
    assert {x.occurrence.occurrence for x in result.candidates} <= set(result.discourse_objects())
    assert {x.occurrence.stable_key() for x in result.candidates} == {
        x.antecedent_key for x in result.target_support.members}
    assert {x.stable_key() for x in result.evidence_records} == {
        key for x in result.target_support.members for key in x.evidence_record_keys}


@pytest.mark.parametrize("drift", ("missing_record", "graph_source", "h04_adoption", "target"))
def test_target_support_refuses_partial_records_or_decision_drift(drift):
    """已有一致目标不能代替完整来源证据、H-04 采用集合和精确目标身份。"""
    result, _target = _repeated_reference_fixture()
    graphs, records, decision = result.evidence, result.evidence_records, result.decision
    if drift == "missing_record":
        records = records[1:]
    elif drift == "graph_source":
        graphs = (replace(graphs[0], source=replace(graphs[0].source, source_id=999)), *graphs[1:])
    elif drift == "h04_adoption":
        decision = replace(decision, adopted_hypotheses=decision.adopted_hypotheses[:1])
    else:
        changed = replace(result.target_support, target_key=ObjectIdentity(OBJECT_ENTITY, (99,)).stable_key())
        with pytest.raises(ValueError, match="目标支持与当前完整候选"):
            replace(result, target_support=changed, winner_target_key=changed.target_key)
        return
    assert reference_target_support(result.candidate_set, result.candidates, graphs, records, decision) is None


def test_repeated_history_generation_requires_all_contexts_bindings_and_evidence():
    """生成保留两个独立历史成员的 O/H 和上下文，删掉任何成员即关闭。"""
    from pure_integer_ai.experiments.trained_open_response_runtime import _reference_generation_provenance
    result, target = _repeated_reference_fixture()
    contexts = tuple(SimpleNamespace(stable_key=lambda item=item: item.context_key,
        role_occurrences=(item.occurrence,)) for item in result.supporting_candidates)
    state = SimpleNamespace(bindings=result.query_bindings(), evidence=result.query_evidence())
    objects, evidence, memory, targets = _reference_generation_provenance(state, contexts, (result,))
    assert targets == (target,)
    assert set(result.discourse_objects()) <= set(objects)
    assert result.stable_key() in evidence
    assert {ref for item in result.supporting_candidates
            for ref in (item.occurrence.observation, item.occurrence.hypothesis)} <= set(memory)
    with pytest.raises(ValueError, match="精确上下文"):
        _reference_generation_provenance(state, contexts[:1], (result,))
    with pytest.raises(ValueError, match="同一次 QueryState"):
        _reference_generation_provenance(replace_namespace(state, evidence=()), contexts, (result,))
    with pytest.raises(ValueError, match="同一次 QueryState"):
        _reference_generation_provenance(replace_namespace(state, bindings=tuple(
            x for x in state.bindings if x.role_key != (REFERENCE_TARGET_SUPPORT_VERSION, 1))), contexts, (result,))


def replace_namespace(value, **changes):
    """只改变测试状态指定字段，保留其余消费者输入。"""
    return SimpleNamespace(**(vars(value) | changes))


def test_semantic_projection_cache_is_bounded_and_context_scoped():
    identity = ObjectIdentity(OBJECT_ENTITY, (91,))

    class Projector:
        def __init__(self):
            self.calls = 0

        def project(self, values):
            self.calls += 1
            span_ref = (7,)
            return SimpleNamespace(
                spans=(SimpleNamespace(
                    ref_key=span_ref, start=0, end=len(values)),),
                semantic_candidates=(SimpleNamespace(
                    candidate_key=identity.stable_key(),
                    projection_kind=PROJECTION_FILLER,
                    span_ref=span_ref),),
            )

    runtime = object.__new__(TrainedCrossTurnReferenceRuntime)
    runtime.projector = Projector()
    runtime._semantic_objects_cache = OrderedDict()
    runtime._semantic_objects_cache_ints = 0

    values = (65, 66)
    first = runtime._semantic_objects(values)
    assert runtime._semantic_objects(values) is first
    assert runtime.projector.calls == 1

    for value in range(_MAX_SEMANTIC_OBJECT_CACHE_ENTRIES + 1):
        runtime._semantic_objects((value + 1000,))
    assert len(runtime._semantic_objects_cache) == (
        _MAX_SEMANTIC_OBJECT_CACHE_ENTRIES)
    assert runtime._semantic_objects_cache_ints <= 131072

    runtime._prepared = OrderedDict()
    runtime._resolved = OrderedDict()
    runtime.clear_context_cache()
    assert not runtime._semantic_objects_cache
    assert runtime._semantic_objects_cache_ints == 0
