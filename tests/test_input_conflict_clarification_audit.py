"""冲突回应必须有独立完整端点竞争，且不能消费任何 Core 真值。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, QueryRoot, ROOT_EMPTY, ROOT_EXPANDED,
    ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY,
)
from pure_integer_ai.experiments.input_conflict_clarification_audit import (
    verify_input_conflict_clarification,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.trained_input_clarification import INPUT_CLARIFICATION_VERSION
from tests.test_query_relation_arbitration import _fixture


def _context(*, different_answer=True):
    """通过真实投影、范围过滤和问题绑定建立冷竞争上下文。"""
    bridge, projected = _fixture(symmetric=False, different_answer=different_answer)
    prepared = bridge._attach_relation_question_binding(
        bridge._filter_relation_question_open_scope(projected))
    scalars = [0] * 17
    scalars[8], scalars[15], scalars[16] = 1, 1, 2
    hops = [SimpleNamespace(space=SPACE_MEMORY), SimpleNamespace(space=SPACE_DIALOGUE)]
    trace = SimpleNamespace(trace={
        'state_key': {'scalars': scalars}, 'termination': 2,
        'input_structure': prepared.integer_trace(),
    }, hops=lambda: tuple(hops))
    roots = [QueryRoot(SPACE_CORE, (INPUT_CLARIFICATION_VERSION, 0), ROOT_EMPTY, 1),
             QueryRoot(SPACE_MEMORY, (3,), ROOT_EXPANDED, 1),
             QueryRoot(SPACE_DIALOGUE, (4,), ROOT_EXPANDED, 1)]
    context = dict(roots=roots, evidence=[], bindings=set(), error=QueryMemoryOheEvidenceError)
    return trace, prepared, bridge.core_fact_index, bridge.relation_question_frames, context, hops


def test_competing_full_endpoints_qualify_without_changing_original_trace():
    trace, prepared, facts, frames, context, _ = _context()
    original = deepcopy(trace.trace)
    result = verify_input_conflict_clarification(trace, prepared, facts, frames, **context)
    assert result['retained_input_relation_count'] == 2
    assert result['status'].startswith('MATCHED_INDEPENDENT_INPUT_COMPETITION')
    assert trace.trace == original


def test_equivalent_complete_endpoints_cannot_masquerade_as_conflict():
    trace, prepared, facts, frames, context, _ = _context(different_answer=False)
    with pytest.raises(QueryMemoryOheEvidenceError, match='competing complete Core'):
        verify_input_conflict_clarification(trace, prepared, facts, frames, **context)


@pytest.mark.parametrize('drift', (
    'input', 'conflict', 'closure', 'generation', 'termination', 'cold_input',
    'cold_facts', 'core_hop', 'support', 'refutation', 'binding', 'root',
    'pending', 'duplicate_core_root',
))
def test_conflict_qualification_rejects_input_state_and_truth_drift(drift):
    trace, prepared, facts, frames, context, hops = _context()
    if drift == 'input':
        trace.trace['input_structure']['token_values'].append(999)
    elif drift in {'conflict', 'closure', 'generation', 'termination'}:
        index, value = {'conflict': (8, 0), 'closure': (7, 1),
                        'generation': (15, 0), 'termination': (16, 3)}[drift]
        trace.trace['state_key']['scalars'][index] = value
    elif drift == 'cold_input':
        prepared = None
    elif drift == 'cold_facts':
        facts = None
    elif drift == 'core_hop':
        hops.append(SimpleNamespace(space=SPACE_CORE))
    elif drift in {'support', 'refutation'}:
        context['evidence'].append(EvidenceEntry((1,), (2,), SPACE_CORE,
            1 if drift == 'support' else 2))
    elif drift == 'binding':
        context['bindings'].add(BindingEntry((1,), (2,), conflict_kept=1).stable_key())
    elif drift == 'root':
        context['roots'][0] = replace(context['roots'][0], root_key=(999,))
    elif drift == 'pending':
        context['roots'][1] = replace(context['roots'][1], status=ROOT_PENDING)
    else:
        context['roots'].append(context['roots'][0])
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_input_conflict_clarification(trace, prepared, facts, frames, **context)
