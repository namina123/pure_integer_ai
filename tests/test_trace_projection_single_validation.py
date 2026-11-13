"""投影复用完整成员校验，须与独立原消费者保持相同通过及拒绝集合。"""
from copy import deepcopy
import json

import pytest

from pure_integer_ai.experiments.batch_case_provenance import _integer_projection
from pure_integer_ai.experiments.batch_trace_storage import TraceStorageError, _trace_projection
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_batch_case_storage import _captured


def _original_projection(raw):
    """独立执行原普通trace、完整整数树、key table及规范JSON全部消费者。"""
    value = json.loads(raw)
    projection = _integer_projection(value)
    GraphQueryTraceEvidence(value)
    GraphQueryTraceEvidence.from_integer_artifact(projection)
    return projection, canonical_json_bytes(projection)


@pytest.mark.parametrize('extra', [None, [], [0, -1, 1 << 256],
    {'序': [None, 0, -3, {'成员': [1 << 512]}]}])
def test_projection_preserves_complete_original_members_and_canonical_bytes(extra):
    """全部非头成员、负值、任意精度及Unicode键继续逐字节保留。"""
    _capture, raw = _captured()
    value = json.loads(raw)
    value['extra_integer_record'] = extra
    payload = json.dumps(value, ensure_ascii=True).encode('utf-8')
    assert _trace_projection(payload) == _original_projection(payload)


@pytest.mark.parametrize('mutation', [
    'schema3', 'schema_bool', 'encoding', 'format_type', 'duplicate_key',
    'negative_key', 'bool_key', 'float_key', 'empty_key', 'nonsequence_key',
    'invalid_query_ref', 'empty_query_ref', 'query_state_drift', 'missing_query',
    'state_type', 'input_type', 'frontier_type', 'steps_type', 'scalar_count',
    'scalar_bool', 'unexpected_text', 'nested_float', 'failure_reason_type',
    'bad_string_header', 'nested_bool', 'empty_object_key',
])
def test_projection_cannot_lose_any_original_rejection_gate(mutation):
    """类型、唯一性、全成员及投影envelope任一漂移仍被两条独立路径拒绝。"""
    _capture, raw = _captured()
    value = deepcopy(json.loads(raw))
    if mutation == 'schema3':
        value['schema_version'] = 3
    elif mutation == 'schema_bool':
        value['schema_version'] = True
    elif mutation == 'encoding':
        value['integer_key_encoding'] = 'UNREGISTERED'
    elif mutation == 'format_type':
        value['format'] = 7
    elif mutation == 'duplicate_key':
        value['integer_key_table'].append(value['integer_key_table'][0])
    elif mutation == 'negative_key':
        value['integer_key_table'][0][0] = -1
    elif mutation == 'bool_key':
        value['integer_key_table'][0][0] = True
    elif mutation == 'float_key':
        value['integer_key_table'][0][0] = 1.0
    elif mutation == 'empty_key':
        value['integer_key_table'].append([])
    elif mutation == 'nonsequence_key':
        value['integer_key_table'].append(9)
    elif mutation == 'invalid_query_ref':
        value['query_ref'] = len(value['integer_key_table']) + 1
    elif mutation == 'empty_query_ref':
        value['query_ref'] = 0
    elif mutation == 'query_state_drift':
        value['state_key']['query_ref'] = 0
    elif mutation == 'missing_query':
        del value['query_ref']
    elif mutation == 'state_type':
        value['state_key'] = []
    elif mutation == 'input_type':
        value['input_structure'] = []
    elif mutation == 'frontier_type':
        value['frontier_trace'] = []
    elif mutation == 'steps_type':
        value['frontier_trace']['steps'] = {}
    elif mutation == 'scalar_count':
        value['state_key']['scalars'].append(0)
    elif mutation == 'scalar_bool':
        value['state_key']['scalars'][0] = True
    elif mutation == 'unexpected_text':
        value['unregistered_text'] = '正文不可偷偷进入整数投影'
    elif mutation == 'nested_float':
        value['extra'] = {'nested': [1, 2.0]}
    elif mutation == 'failure_reason_type':
        value['failure_reason'] = 9
    elif mutation == 'bad_string_header':
        value['state_key']['encoding'] = []
    elif mutation == 'nested_bool':
        value['extra'] = [None, {'nested': False}]
    else:
        value['extra'] = {'': 1}
    payload = json.dumps(value).encode('utf-8')
    with pytest.raises((TypeError, ValueError)):
        _original_projection(payload)
    with pytest.raises(TraceStorageError):
        _trace_projection(payload)
