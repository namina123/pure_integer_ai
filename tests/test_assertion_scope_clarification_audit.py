"""作用域保存消费者必须保留输入、证明和完整 Core 历史。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord, HypothesisKey
from pure_integer_ai.cognition.shared.identity import OBJECT_PROPOSITION, ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED, ROOT_PENDING, QueryRoot
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED, attach_assertion_scope_frames,
)
from pure_integer_ai.experiments.input_conflict_clarification_audit import verify_input_conflict_clarification
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.trained_assertion_scope_runtime import assertion_scope_query_records
from pure_integer_ai.experiments.trained_graph_query_bridge import _selected_relation_view
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence
from test_query_assertion_scope import _fixture


def _context(mode=SCOPE_NEGATED):
    """用整数课程和真实证据记录提供冷上下文，不运行查询或生成。"""
    projector, frame = _fixture(mode)
    prepared = attach_assertion_scope_frames((frame,), projector.project((11, 77, 33, 22)))
    active = _selected_relation_view(prepared, prepared.assertion_scope_bindings[0].relation_key)[0]
    query = (991,)
    binding, scope_evidence = assertion_scope_query_records(active, query)[0]
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (50,))
    source = SourceRef(1, 2, 3, proposition.owner, proposition.versions)
    hypothesis = HypothesisKey((1,), proposition.stable_key(), (99,), document_scope(source), source)
    history = (EvidenceRecord(1, hypothesis, 1, (7,), source, 1),
               EvidenceRecord(2, hypothesis, 2, (8,), source, 2, supersedes_evidence_id=1))
    registry = SimpleNamespace(find=lambda *args: 71)
    core = SimpleNamespace(evidence_history=lambda key: history if key == proposition else (),
        context=SimpleNamespace(scoped_identity_store=SimpleNamespace(registry=registry)))
    scalars = [0] * 17
    scalars[8], scalars[15], scalars[16] = 1, 1, 2
    trace = SimpleNamespace(trace={'state_key': {'scalars': scalars}, 'termination': 2,
        'query_ref': query, 'input_structure': active.integer_trace()},
        resolve_ref=lambda ref: ref, hops=lambda: (SimpleNamespace(space=1),))
    context = dict(roots=[QueryRoot(1, proposition.stable_key(), ROOT_EXPANDED, 1)],
        evidence=[scope_evidence, *core_query_evidence(proposition, history, registry)],
        bindings={binding.stable_key()}, error=QueryMemoryOheEvidenceError,
        assertion_scope_frames=(frame,), core_runtime=core, authorized_input=active)
    return trace, prepared, context


@pytest.mark.parametrize('mode', (SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED))
def test_scope_qualification_preserves_actual_core_hops_and_history(mode):
    trace, prepared, context = _context(mode)
    original = deepcopy(trace.trace)
    result = verify_input_conflict_clarification(trace, prepared, None, (), **context)
    assert result['scope_mode'] == mode
    assert result['core_hop_count'] == 1
    assert result['core_truth_evidence_count'] == 2
    assert trace.trace == original


@pytest.mark.parametrize('drift', (
    'model', 'competition', 'preinstalled', 'view', 'input', 'proof', 'extra_proof',
    'binding', 'extra_binding', 'pending', 'truth', 'history', 'core', 'state',
))
def test_scope_qualification_rejects_missing_or_drifting_cold_evidence(drift):
    trace, prepared, context = _context()
    if drift == 'model':
        context['assertion_scope_frames'] = ()
    elif drift == 'competition':
        frame = context['assertion_scope_frames'][0]
        context['assertion_scope_frames'] += (replace(frame, mode=SCOPE_POSSIBLE),)
    elif drift == 'preinstalled':
        frame = context['assertion_scope_frames'][0]
        scope = replace(prepared.assertion_scope_bindings[0], frame_key=replace(frame, source_identity=(99,)).stable_key())
        prepared = replace(prepared, assertion_scope_bindings=(scope,))
    elif drift == 'view':
        context['authorized_input'] = prepared
    elif drift == 'input':
        trace.trace['input_structure']['token_values'].append(9)
    elif drift == 'proof':
        context['evidence'][0] = replace(context['evidence'][0], polarity=1)
    elif drift == 'extra_proof':
        context['evidence'].append(replace(context['evidence'][0], source_ref=(99,)))
    elif drift == 'binding':
        context['bindings'].clear()
    elif drift == 'extra_binding':
        key = next(iter(context['bindings']))
        context['bindings'].add((*key[:-1], 1))
    elif drift == 'pending':
        context['roots'][0] = replace(context['roots'][0], status=ROOT_PENDING)
    elif drift == 'truth':
        context['evidence'][1] = replace(context['evidence'][1], payload_key=(99,))
    elif drift == 'history':
        context['evidence'].pop()
    elif drift == 'core':
        context['core_runtime'] = None
    else:
        trace.trace['state_key']['scalars'][7] = 1
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_input_conflict_clarification(trace, prepared, None, (), **context)
