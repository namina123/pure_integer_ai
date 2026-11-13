"""同次不可变trace快照须保留完整捕获字节和全部漂移拒绝门。"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest

from pure_integer_ai.experiments.batch_case_storage import (
    CaseArtifactStorageError, build_compact_case_provenance, restore_compact_case_provenance,
)
from pure_integer_ai.experiments.batch_trace_storage import (
    TraceStorageError, ValidatedTraceSnapshot, save_case_trace_snapshot,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_batch_case_storage import _captured


@pytest.mark.parametrize('compressed', [False, True])
def test_snapshot_preserves_full_legacy_capture_bytes(tmp_path, compressed):
    """新旧封装与冷恢复的完整规范bytes均须相同，普通物理保存协议不变。"""
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=compressed)
    options = dict(trace_bytes=snapshot.raw, trace_ref=ref, trace_root=tmp_path)
    baseline = build_compact_case_provenance(captured, **options)
    actual = build_compact_case_provenance(captured, **options, validated_trace=snapshot)
    assert snapshot.raw == original
    assert canonical_json_bytes(actual) == canonical_json_bytes(baseline)
    assert canonical_json_bytes(restore_compact_case_provenance(actual, root=tmp_path)) == canonical_json_bytes(captured)


@pytest.mark.parametrize('replacement', [True, 1.0, 2])
def test_snapshot_rejects_captured_integer_type_or_value_drift(tmp_path, replacement):
    """bool/float的宿主相等和真实值漂移均不能借快照获得资格。"""
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=True)
    captured = deepcopy(captured)
    captured['trace']['state_key']['active_spaces'][0] = replacement
    with pytest.raises((CaseArtifactStorageError, ValueError)):
        build_compact_case_provenance(captured, trace_bytes=snapshot.raw,
            trace_ref=ref, trace_root=tmp_path, validated_trace=snapshot)


@pytest.mark.parametrize('mutation', ['blob', 'descriptor', 'projection', 'raw'])
def test_snapshot_rejects_physical_or_reference_drift(tmp_path, mutation):
    """不可变本地快照不能替代当前物理blob、descriptor和引用逐项核验。"""
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=True)
    raw = snapshot.raw
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
            trace_ref=ref, trace_root=tmp_path, validated_trace=snapshot)


@pytest.mark.parametrize('mutation', ['bool', 'duplicate', 'header'])
def test_snapshot_constructor_requires_full_valid_trace(mutation):
    """构造不能用现成投影或hash跳过原整数树、键表和头部合同。"""
    _captured_record, raw = _captured()
    value = json.loads(raw)
    if mutation == 'bool':
        value['integer_key_table'][0][0] = True
    elif mutation == 'duplicate':
        value['integer_key_table'].append(value['integer_key_table'][0])
    else:
        value['integer_key_encoding'] = 'UNREGISTERED'
    with pytest.raises(TraceStorageError):
        ValidatedTraceSnapshot(json.dumps(value).encode('utf-8'))


def test_snapshot_has_no_external_projection_argument_and_is_immutable():
    """原始bytes与验证所得投影在构造之后均不可替换。"""
    _captured_record, raw = _captured()
    snapshot = ValidatedTraceSnapshot(raw)
    with pytest.raises(TypeError):
        ValidatedTraceSnapshot(raw, projection_bytes=b'{}')
    with pytest.raises(FrozenInstanceError):
        snapshot.raw = b'{}'
    with pytest.raises(FrozenInstanceError):
        snapshot.projection_bytes = b'{}'


def test_snapshot_from_another_raw_record_cannot_supply_equal_projection(tmp_path):
    """已剥离的边界字符串相同投影不能掩盖原始记录身份变化。"""
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=True)
    other = json.loads(original)
    other['response_surface'] = 'Y'
    other_snapshot = ValidatedTraceSnapshot(canonical_json_bytes(other) + b'\n')
    assert other_snapshot.projection_bytes == snapshot.projection_bytes
    with pytest.raises(CaseArtifactStorageError):
        build_compact_case_provenance(captured, trace_bytes=snapshot.raw,
            trace_ref=ref, trace_root=tmp_path, validated_trace=other_snapshot)


def test_snapshot_argument_rejects_unvalidated_lookalike(tmp_path):
    """同名属性或鸭子类型对象不具备完整构造校验资格。"""
    from types import SimpleNamespace
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=True)
    lookalike = SimpleNamespace(raw=original, projection_bytes=snapshot.projection_bytes)
    with pytest.raises(CaseArtifactStorageError):
        build_compact_case_provenance(captured, trace_bytes=original,
            trace_ref=ref, trace_root=tmp_path, validated_trace=lookalike)


def test_snapshot_encoding_must_match_actual_storage(tmp_path):
    """相同逻辑bytes和投影也不能借用另一物理编码的见证。"""
    captured, original = _captured()
    ref, snapshot = save_case_trace_snapshot(tmp_path, json.loads(original), compressed=True)
    other_encoding = ValidatedTraceSnapshot(snapshot.raw)
    with pytest.raises(CaseArtifactStorageError):
        build_compact_case_provenance(captured, trace_bytes=original,
            trace_ref=ref, trace_root=tmp_path, validated_trace=other_encoding)


def test_snapshot_physical_chunks_cannot_be_injected_or_mutated():
    """全部规范chunk只能由原始bytes实际编码得到，且不可变。"""
    _captured_record, raw = _captured()
    snapshot = ValidatedTraceSnapshot(raw, compressed=True)
    with pytest.raises(TypeError):
        ValidatedTraceSnapshot(raw, compressed=True, stored_chunks=(b'forged',))
    with pytest.raises(FrozenInstanceError):
        snapshot.stored_chunks = (b'forged',)
    with pytest.raises(TypeError):
        snapshot.stored_chunks[0] = b'forged'
