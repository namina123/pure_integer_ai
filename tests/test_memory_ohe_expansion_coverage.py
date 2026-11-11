"""同一 O/H/E 内容可达性去重必须保留原上下文关系和全部声明。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import OwnerScope, VersionBundle
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_EVIDENCE, MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION,
    MemoryLinkedRef, memory_object_ref,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_MEMORY, FrontierEntry, QueryRoot, QueryState, ROOT_PENDING,
)
from pure_integer_ai.experiments.memory_ohe_expansion_coverage import memory_ohe_expansion_coverage
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    MemoryEvidenceAnchor, MemoryHypothesisAnchor, MemoryObservationAnchor, _CrossSpaceExpander,
)
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


def _chain():
    """用独立整数身份构造完整来源内 O/H/E 链，不含语言或评测答案。"""
    space = SpaceIdentity(SPACE_TYPE_MEMORY, 31, 32)
    refs = tuple(memory_object_ref(space, kind, (700 + kind,),
        owner=OwnerScope(), versions=VersionBundle()) for kind in (
            MEMORY_OBJECT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_EVIDENCE))
    o, h, e = (ref.stable_key() for ref in refs)
    observation = MemoryObservationAnchor(o, (91,), (92,), 91, 1, 1, (h,))
    hypothesis = MemoryHypothesisAnchor(o, h, (93,), (94,), (95,), (91,), (92,), (e,))
    evidence = MemoryEvidenceAnchor(e, h, (91,), (92,), 3, (96,), (97,), ())
    root = QueryRoot(SPACE_MEMORY, o, ROOT_PENDING, 1, source_ref=(91,), scope_key=(92,))
    return refs, observation, hypothesis, evidence, root


def test_only_full_current_root_and_both_member_edges_cover_the_chain():
    """覆盖只能由实际根及双向归属推出，未根定观察不能授权子声明。"""
    refs, observation, hypothesis, evidence, root = _chain()
    expected = frozenset((ref.object_kind, ref.stable_key()) for ref in refs)
    assert memory_ohe_expansion_coverage((root,), (observation,), (hypothesis,), (evidence,)) == expected
    assert not memory_ohe_expansion_coverage((), (observation,), (hypothesis,), (evidence,))
    assert not memory_ohe_expansion_coverage((replace(root, source_ref=(999,)),),
        (observation,), (hypothesis,), (evidence,))
    assert not memory_ohe_expansion_coverage((replace(root, scope_key=(999,)),),
        (observation,), (hypothesis,), (evidence,))
    covered = memory_ohe_expansion_coverage((root,), (replace(observation, hypothesis_keys=()),),
        (hypothesis,), (evidence,))
    assert covered == frozenset({(refs[0].object_kind, refs[0].stable_key())})


@pytest.mark.parametrize('field,value', (
    ('hypothesis_key', (999,)), ('source_ref', (999,)), ('scope_key', (999,)),
    ('evidence_key', (999,)),
))
def test_foreign_or_unregistered_evidence_is_not_suppressed(field, value):
    """来源、scope、反向归属或成员身份不符时，仍须走原上下文路径。"""
    _refs, observation, hypothesis, evidence, root = _chain()
    evidence = replace(evidence, **{field: value})
    covered = memory_ohe_expansion_coverage((root,), (observation,), (hypothesis,), (evidence,))
    assert (MEMORY_OBJECT_EVIDENCE, evidence.evidence_key) not in covered


@pytest.mark.parametrize('covered_target', (False, True))
def test_production_observation_expansion_retains_link_without_fake_visits(covered_target):
    """调用真实扩展器，验证去重保留 link/QueryHop，未覆盖目标仍被排程。"""
    refs, target, hypothesis, evidence, target_root = _chain()
    origin = replace(target, observation_key=(101,), source_ref=(102,), scope_key=(103,),
        source_hash=102, hypothesis_keys=(), relation_refs=(MemoryLinkedRef.memory(refs[0]),))
    origin_root = QueryRoot(SPACE_MEMORY, origin.observation_key, ROOT_PENDING, 1,
        source_ref=origin.source_ref, scope_key=origin.scope_key)
    roots = (origin_root, target_root) if covered_target else (origin_root,)
    owner = SimpleNamespace(
        _memory_observations=(origin, target), _memory_hypotheses=(hypothesis,),
        _memory_evidence=(evidence,), _memory_observations_by_key={origin.observation_key: origin},
        _memory_hypotheses_by_key={}, _memory_evidence_by_key={},
        _memory_hypotheses_by_observation={}, _memory_evidence_by_hypothesis={},
        _memory_active_object_keys=frozenset({(refs[0].object_kind, refs[0].stable_key())}),
        space_weights=(1, 1, 1))
    expander = object.__new__(_CrossSpaceExpander)
    expander.owner, expander.hops, expander._memory_ohe_expansion_coverage = owner, [], None
    expander._with_query_status = lambda state, edge, **options: state
    edge = FrontierEntry((SPACE_MEMORY, 101), SPACE_MEMORY,
        target_key=origin.observation_key, root_key=origin.observation_key)
    state = QueryState((104,), roots=roots, frontier=(edge,))
    outcome = expander._expand_memory(state, (), edge, origin.observation_key)
    result = outcome.state
    assert len(result.frontier) == int(not covered_target)
    assert result.roots == state.roots and outcome.reads == 1
    assert tuple(item.node_key for item in result.visited) == (origin.observation_key,)
    assert any(item.role_key == (91533, 1) and item.filler_key == origin.relation_refs[0].stable_key()
               for item in result.bindings)
    assert any(item.predicate == (91533, 1) and item.to_filler == refs[0].stable_key()
               for item in expander.hops)
