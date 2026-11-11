"""无历史指代门不得吞掉已有候选、读取事实或跨轮借用角色。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_ROLE, ObjectIdentity, occurrence_identity, span_identity
from pure_integer_ai.cognition.shared.query_state import QueryAnchor, QueryRoot, ROOT_PENDING, SPACE_MEMORY, seed_query
from pure_integer_ai.experiments.trained_input_clarification import (
    clarification_query_state, unanchored_reference_question,
)
from pure_integer_ai.experiments.trained_response_occurrence import InputRoleOccurrence
from pure_integer_ai.experiments.batch_case_provenance import CaseProvenanceError, _integer_projection
from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, verify_query_memory_ohe_nonfactual,
)
from test_query_memory_ohe_audit import _fixture


def test_no_history_gate_requires_unique_validated_reference_slot_and_preserves_any_candidate():
    """普通 WH、多动态槽及任一保留历史候选不得获无历史资格。"""
    parts = (SimpleNamespace(is_reference_slot=True), None, None, None, ((81,),))
    assert unanchored_reference_question(parts, ())
    assert not unanchored_reference_question(parts, (object(),))
    assert not unanchored_reference_question(None, ())
    assert not unanchored_reference_question((SimpleNamespace(is_reference_slot=False), *parts[1:]), ())
    assert not unanchored_reference_question((*parts[:4], ()), ())
    assert not unanchored_reference_question((*parts[:4], ((81,), (82,))), ())


def test_clarification_retains_exact_current_role_but_denies_foreign_observation():
    """原观察、角色和来源完整保留；别轮角色不能成为当前澄清证据。"""
    _captured, declarations, manifest, evidence = _fixture(3)
    observation = manifest.bindings[0].object_ref
    hypothesis = manifest.bindings[1].object_ref
    source = manifest.source
    role = InputRoleOccurrence(occurrence_identity(source, start=0, end=1, ordinal=0),
        span_identity(source, members=((0, 1),)), ObjectIdentity(OBJECT_ROLE, (81,), source.owner, source.versions),
        observation, hypothesis, declarations[observation].event.payload.context.value(),
        0, evidence.source_ref, evidence.scope_key, (65,))
    rroot, ranchor, redge = role.seed(1)
    roots = (QueryRoot(2, observation.stable_key(), ROOT_PENDING, 1),
             QueryRoot(3, (31,), ROOT_PENDING, 1), QueryRoot(1, (41,), ROOT_PENDING, 1), rroot)
    state = seed_query((99,), roots=roots, anchors=(ranchor,))
    from pure_integer_ai.cognition.shared.query_state import FrontierEntry
    frontier = tuple(FrontierEntry((root.owner_space, *root.root_key), root.owner_space,
        target_key=root.root_key, root_key=root.root_key) for root in roots)
    result = clarification_query_state(state, frontier, (SimpleNamespace(observation_key=observation.stable_key()),),
        (SimpleNamespace(key=(31,)),), input_role_occurrences=(role,))
    assert result is not None and result.budget == state.budget and result.read_count == 0
    assert rroot in result.roots and redge.target_key == role.stable_key()
    assert any(binding.filler_key == roots[2].stable_key() for binding in result.bindings)
    foreign = replace(role, observation=manifest.bindings[2].object_ref)
    assert clarification_query_state(state, frontier,
        (SimpleNamespace(observation_key=observation.stable_key()),), (SimpleNamespace(key=(31,)),),
        input_role_occurrences=(foreign,)) is None


def test_trace_projection_retains_null_reason_and_only_strips_known_diagnostic_text():
    """诊断字段是边界元数据，未知文本及错误字段类型继续关闭。"""
    assert _integer_projection({'failure_reason':None, 'termination':3}) == {'failure_reason':None, 'termination':3}
    assert _integer_projection({'failure_reason':'missing source-qualified antecedent', 'termination':3}) == {'termination':3}
    for value in (1, True, [], {}):
        with pytest.raises(CaseProvenanceError):
            _integer_projection({'failure_reason':value})
    with pytest.raises(CaseProvenanceError):
        _integer_projection({'new_reason':'text'})


def test_saved_nonfactual_plan_never_accepts_a_truth_claim_or_invented_generation():
    """有 Plan 的非事实调用复用完整 O/H/E 门，事实声明和缺生成均拒绝。"""
    captured, declarations, manifest, _evidence = _fixture(3)
    trace = captured['trace']
    trace.update(integer_key_encoding='INTEGER_KEY_TABLE_V1', termination=3, generation={'kind':3})
    trace['state_key']['scalars'][15] = 1
    report = {'termination':3, 'delivery_claim_count':0, 'response_plan_present':1, 'generation_present':1,
              'input_source':captured['input_source'], 'observation_ref':captured['input_observation']}
    readers = dict(declaration_reader=lambda ref:(declarations[ref],) if ref in declarations else (),
                   current_manifest_reader=lambda source:manifest, source_hash_reader=lambda source:19)
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_query_memory_ohe_nonfactual(trace, report, **readers)
    trace['response_plan']['claim_refs'] = [[7, 1]]
    with pytest.raises(QueryMemoryOheEvidenceError, match='non-claim Plan'):
        verify_query_memory_ohe_nonfactual(trace, report, **readers)
    trace['response_plan']['claim_refs'] = []
    trace['generation'] = None
    with pytest.raises(QueryMemoryOheEvidenceError, match='non-claim Plan'):
        verify_query_memory_ohe_nonfactual(trace, report, **readers)
