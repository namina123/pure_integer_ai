"""内容H/E延后后，O只凭当前匹配冷授权保留，实际全集相等门不变。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.memory_event import MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS
from pure_integer_ai.cognition.shared.query_state import BindingEntry, QueryRoot, ROOT_EXPANDED
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, frame_key, memory_candidate_routes,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError, verify_query_memory_ohe
from pure_integer_ai.experiments.saved_response_context_audit import (
    read_saved_factual_response_contexts, restore_saved_factual_response_contexts,
)
from test_query_memory_ohe_audit import _fixture


def _records():
    """当前和旧来源各有真实类型OHE声明；查询只读取旧O，旧H/E保持未消费。"""
    candidate = frame_key(MEMORY_CANDIDATE_VERSION, (1,), (1, 1, 0), (77,), (78,), (79,), (80,))
    captured, declarations, current, _ = _fixture(document_id=2, candidate_key=candidate)
    old_captured, old_declarations, old, _ = _fixture(document_id=1, candidate_key=candidate)
    declarations.update(old_declarations)
    manifests = {current.source: current, old.source: old}
    old_oref = next(ref for ref, row in old_declarations.items() if row.event.event_kind == MEMORY_EVENT_OBSERVATION)
    old_href = next(ref for ref in old_declarations if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS)
    input_oref = next(ref for ref, row in declarations.items()
        if row.event.event_kind == MEMORY_EVENT_OBSERVATION and row.event.payload.source == current.source)
    scope = declarations[old_oref].event.scope.stable_key()
    value = captured['trace']

    def intern(key):
        """测试直接保存完整规范键，不按hash缩减成员。"""
        raw = list(key)
        if raw not in value['integer_key_table']:
            value['integer_key_table'].append(raw)
        return value['integer_key_table'].index(raw) + 1

    value['hops'].append(dict(space=2, proposition_ref=intern(old_oref.stable_key()),
        predicate_ref=intern((MEMORY_EVENT_OBSERVATION,)), source_hash=19))
    value['state_key']['roots'].append(intern(QueryRoot(2, old_oref.stable_key(), ROOT_EXPANDED, 1,
        (19, *old.source.stable_key()), scope).stable_key()))
    value['state_key']['bindings'].append(intern(BindingEntry((1,), old_oref.stable_key(), 2, scope).stable_key()))
    route = frame_key(1, candidate, memory_candidate_routes(candidate)[0], old_href.stable_key())
    trace = SimpleNamespace(trace={'response_context_refs': []}, memory_routes=lambda: (route,))
    memory = SimpleNamespace(owner=current.source.owner, intake=SimpleNamespace(require_current_manifest=manifests.__getitem__))
    readers = dict(declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
        source_hash_reader=lambda source: 1019 if source == current.source else 19)
    restored = restore_saved_factual_response_contexts(memory, trace, input_observation=input_oref,
        input_candidate_keys=(candidate,), **readers)
    return captured, declarations, manifests, old_oref, restored, readers, memory, trace, input_oref, candidate


def _verify(records, *, captured=None, refs=None, maximum=128):
    """真实OHE入口核对所有声明、根、绑定和完整跳集合。"""
    original, _declarations, manifests, _oref, restored, readers, *_ = records
    return verify_query_memory_ohe(original if captured is None else captured,
        current_manifest_reader=manifests.__getitem__,
        recall_observations=restored.recall_observations if refs is None else refs,
        max_ohe_declarations=maximum, **readers)


def test_independently_restored_source_only_observation_keeps_exact_ohe_sets():
    records = _records()
    captured, _declarations, _manifests, oref, restored, *_ = records
    assert restored.contexts == () and restored.recall_observations == (oref,)
    result = _verify(records)
    assert result['cold_recall_observation_count'] == 1
    assert result['independent_evidence_count'] == 1
    assert len(result['chains']) == 1
    with pytest.raises(QueryMemoryOheEvidenceError, match='hops differ'):
        _verify(records, refs=())


@pytest.mark.parametrize('change', ('missing_root', 'wrong_scope', 'wrong_source', 'missing_binding', 'duplicate_root',
    'duplicate_ref', 'wrong_kind', 'list_ref', 'missing_declaration', 'declaration_budget', 'extra_hypothesis'))
def test_source_only_does_not_relax_physical_or_complete_set_obligations(change):
    records = _records()
    captured, declarations, _manifests, oref, _restored, *_ = records
    changed = deepcopy(captured)
    value = changed['trace']
    refs, maximum = (oref,), 128
    if change in ('missing_root', 'duplicate_root'):
        if change == 'missing_root':
            value['state_key']['roots'].pop()
        else:
            value['state_key']['roots'].append(value['state_key']['roots'][-1])
    elif change in ('wrong_scope', 'wrong_source'):
        key = value['integer_key_table'][value['state_key']['roots'][-1] - 1]
        key[-1 if change == 'wrong_scope' else 4 + len(oref.stable_key()) + 1] += 1
    elif change == 'missing_binding':
        value['state_key']['bindings'].pop()
    elif change == 'duplicate_ref':
        refs = (oref, oref)
    elif change == 'wrong_kind':
        refs = (next(ref for ref in declarations if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS),)
    elif change == 'list_ref':
        refs = [oref]
    elif change == 'missing_declaration':
        del declarations[oref]
    elif change == 'declaration_budget':
        maximum = 3
    else:
        ref = next(ref for ref in declarations if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS)
        key_ref = value['integer_key_table'].index(list(ref.stable_key())) + 1
        value['hops'].append(dict(space=2, proposition_ref=key_ref,
            predicate_ref=value['hops'][1]['predicate_ref'], source_hash=19))
    with pytest.raises((ValueError, QueryMemoryOheEvidenceError)):
        _verify(records, captured=changed, refs=refs, maximum=maximum)


@pytest.mark.parametrize('change', ('missing_route', 'wrong_candidate', 'wrong_shared_route', 'future_source'))
def test_cold_reader_requires_current_matching_route_and_original_temporal_scope(change):
    records = _records()
    _captured, _decls, _manifests, old, _restored, readers, memory, trace, current, candidate = records
    if change == 'missing_route':
        trace.memory_routes = lambda: ()
    elif change == 'wrong_candidate':
        candidate = frame_key(MEMORY_CANDIDATE_VERSION, (1,), (1, 1, 0), (90,), (91,), (92,), (93,))
    elif change == 'wrong_shared_route':
        route = trace.memory_routes()[0]
        trace.memory_routes = lambda: (frame_key(1, candidate, (99,), route),)
    else:
        current = old
        # 原当前较新的H/O只在反向身份中参与，必须被非未来来源门拒绝。
        new_href = next(ref for ref, row in _decls.items()
            if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS and row.event.payload.hypothesis.observation.document_id == 2)
        trace.memory_routes = lambda: (frame_key(1, candidate, memory_candidate_routes(candidate)[0], new_href.stable_key()),)
    if change == 'missing_route':
        result = restore_saved_factual_response_contexts(memory, trace, input_observation=current,
            input_candidate_keys=(candidate,), **readers)
        assert result.recall_observations == ()
    else:
        with pytest.raises(QueryMemoryOheEvidenceError):
            restore_saved_factual_response_contexts(memory, trace, input_observation=current,
                input_candidate_keys=(candidate,), **readers)


def test_legacy_context_interface_does_not_export_new_source_objects():
    records = _records()
    *_, readers, memory, trace, current, candidate = records
    assert read_saved_factual_response_contexts(memory, trace, input_observation=current,
        input_candidate_keys=(candidate,), **readers) == ()
