"""同次捕获复用完整验证时，语义漂移与物理篡改仍须拒绝。"""
from copy import deepcopy
import json

import pytest

from pure_integer_ai.experiments.batch_case_storage import (
    CaseArtifactStorageError, build_compact_case_provenance,
    restore_compact_case_provenance,
)
from pure_integer_ai.experiments.batch_trace_storage import save_case_trace
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_batch_case_storage import _captured


def test_same_call_reference_preserves_canonical_capture(tmp_path):
    """有损压缩或重复投影不得改变新旧捕获的完整规范字节。"""
    captured, original = _captured()
    ref, raw = save_case_trace(tmp_path, json.loads(original), compressed=True)
    compact = build_compact_case_provenance(captured, trace_bytes=raw,
        trace_ref=ref, trace_root=tmp_path)
    restored = restore_compact_case_provenance(compact, root=tmp_path)
    assert canonical_json_bytes(restored) == canonical_json_bytes(captured)
    assert raw == original


@pytest.mark.parametrize('replacement', [True, 1.0, 2])
def test_same_call_rejects_projection_type_or_value_drift(tmp_path, replacement):
    """宿主数值相等不能掩盖bool、float或整数值漂移。"""
    captured, original = _captured()
    ref, raw = save_case_trace(tmp_path, json.loads(original), compressed=True)
    captured = deepcopy(captured)
    captured['trace']['state_key']['active_spaces'][0] = replacement
    with pytest.raises((CaseArtifactStorageError, ValueError)):
        build_compact_case_provenance(captured, trace_bytes=raw,
            trace_ref=ref, trace_root=tmp_path)


@pytest.mark.parametrize('mutation', ['blob', 'descriptor', 'projection', 'raw'])
def test_same_call_rejects_changed_storage_or_reference(tmp_path, mutation):
    """本地复用不能绕过blob、descriptor、投影或当前原始bytes的精确核对。"""
    captured, original = _captured()
    ref, raw = save_case_trace(tmp_path, json.loads(original), compressed=True)
    if mutation == 'blob':
        path = tmp_path / ref['path']
        path.write_bytes(path.read_bytes()[:-1] + b'\x00')
    elif mutation == 'descriptor':
        path = tmp_path / 'trace.int.json'
        descriptor = json.loads(path.read_bytes())
        descriptor['trace_ref']['sha256'] = '0' * 64
        path.write_bytes(canonical_json_bytes(descriptor))
    elif mutation == 'projection':
        ref['integer_projection_sha256'] = '0' * 64
    else:
        value = json.loads(raw)
        value['response_surface'] = 'Y'
        raw = canonical_json_bytes(value) + b'\n'
    with pytest.raises(CaseArtifactStorageError):
        build_compact_case_provenance(captured, trace_bytes=raw,
            trace_ref=ref, trace_root=tmp_path)
