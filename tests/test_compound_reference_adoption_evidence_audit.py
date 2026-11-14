"""Core采用复合记录保留完整冷读见证和每字段上限，未知协议失败关闭。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_PROPOSITION, ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.compound_reference_adoption_evidence_audit import (
    verify_compound_reference_adoption_evidence,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, _key, _rows, verify_query_memory_ohe,
)
from pure_integer_ai.experiments.reference_question_transition_audit import ReferenceQuestionTransitionAudit
from test_query_memory_ohe_audit import _fixture


def _pack(parts):
    """测试独立按七字段长度帧构造记录，不调用生产记录构造器。"""
    return (91599, 7, *(value for part in parts for value in (len(part), *part)))


@pytest.fixture(scope='module')
def records():
    """用合成决策隔离结构冷读；真正决策及全转换由生产保存回归验证。"""
    _captured, _declarations, manifest, _entry = _fixture()
    source = manifest.source
    claim = ObjectIdentity(OBJECT_PROPOSITION, (901,), source.owner, source.versions)
    resolution = SimpleNamespace(stable_key=lambda: (91536, *((91,) * 500000)),
        status=3, adoption_support_key=(902,), winner_target_key=(903,))
    parts = (resolution.stable_key(), claim.stable_key(), (904,), (905,), (906,), (902,), (903,))
    payload = _pack(parts)
    scope = document_scope(source).stable_key()
    entry = EvidenceEntry((19, *source.stable_key()), claim.stable_key(), 1, 3, 1,
        (91599, 7, *payload), scope, payload)
    binding = BindingEntry((91599, 6), payload, 1, scope).stable_key()
    value = dict(input_structure=dict(stable_key=(907,)),
        state_key=dict(bindings=(binding,)))
    witness = ReferenceQuestionTransitionAudit((('target_adoption_count', 1),), (907,), (entry.stable_key(),))
    frame = SimpleNamespace(proposition=claim, source_hash=19)
    physical = SimpleNamespace(proposition=SimpleNamespace(definition=SimpleNamespace(
        source=source, proposition=claim)), hypothesis=SimpleNamespace(scope=document_scope(source)))
    core = SimpleNamespace(generation_input=lambda _: physical, active_surface_frames=lambda: (frame,))
    return entry, value, witness, resolution, core, parts


def _verify(records, *, entry=None, value=None, witness=None, resolutions=None, core=None, maximum=256):
    """保持正式原子读取器，不把合成决策记作运行时能力。"""
    original, original_value, transition, resolution, physical, _parts = records
    trace = SimpleNamespace(trace=original_value if value is None else value, resolve_ref=lambda item: tuple(item))
    return verify_compound_reference_adoption_evidence((original if entry is None else entry).stable_key(),
        trace, transition=transition if witness is None else witness,
        resolutions=(resolution,) if resolutions is None else resolutions,
        core_runtime=physical if core is None else core, key_reader=_key, rows_reader=_rows,
        max_state_members=maximum, error=QueryMemoryOheEvidenceError)


def test_compound_core_keeps_all_members_and_original_atomic_limit(records):
    entry = records[0]
    with pytest.raises(QueryMemoryOheEvidenceError, match='bounded integer'):
        _key(entry.stable_key(), 'oversized compound')
    assert _verify(records) == entry.stable_key()


@pytest.mark.parametrize('field,value', (
    ('source_ref', (99,)), ('hypothesis_key', (98,)), ('scope_key', (97,)),
    ('evidence_key', (96,)), ('payload_key', (95,)), ('supersedes_key', (94,)),
    ('space', 2), ('polarity', 1), ('trust', 2),
))
def test_changed_members_cannot_borrow_another_transition(records, field, value):
    with pytest.raises((ValueError, QueryMemoryOheEvidenceError)):
        _verify(records, entry=replace(records[0], **{field: value}))


@pytest.mark.parametrize('change', ('missing', 'duplicate', 'status', 'support', 'target', 'identity'))
def test_complete_cold_resolution_remains_required(records, change):
    resolution = SimpleNamespace(**vars(records[3]))
    if change == 'missing':
        resolutions = ()
    elif change == 'duplicate':
        resolutions = (resolution, resolution)
    else:
        if change == 'status':
            resolution.status = 2
        elif change == 'identity':
            resolution.stable_key = lambda: (99,)
        else:
            setattr(resolution, 'adoption_support_key' if change == 'support' else 'winner_target_key', (99,))
        resolutions = (resolution,)
    with pytest.raises(QueryMemoryOheEvidenceError, match='cold resolution'):
        _verify(records, resolutions=resolutions)


@pytest.mark.parametrize('change', ('missing', 'wrong_input', 'wrong_evidence', 'count', 'binding', 'duplicate_binding', 'scope', 'source_hash', 'frame', 'budget'))
def test_complete_transition_current_binding_and_physical_graph_are_required(records, change):
    entry, original, witness, _resolution, core, _parts = records
    value = deepcopy(original)
    maximum = 256
    if change == 'missing':
        witness = {}
    elif change == 'wrong_input':
        value['input_structure']['stable_key'] = (99,)
    elif change == 'wrong_evidence':
        witness = replace(witness, adoption_evidence_keys=())
    elif change == 'count':
        witness = replace(witness, summary=(('target_adoption_count', 0),))
    elif change == 'binding':
        value['state_key']['bindings'] = ()
    elif change == 'duplicate_binding':
        value['state_key']['bindings'] *= 2
    elif change == 'budget':
        maximum = 6
    elif change == 'scope':
        physical = core.generation_input(None)
        core = SimpleNamespace(active_surface_frames=core.active_surface_frames,
            generation_input=lambda _: SimpleNamespace(proposition=physical.proposition,
                hypothesis=SimpleNamespace(scope=SimpleNamespace(stable_key=lambda: (99,)))))
    else:
        frames = () if change == 'frame' else (SimpleNamespace(proposition=core.active_surface_frames()[0].proposition, source_hash=99),)
        core = SimpleNamespace(generation_input=core.generation_input, active_surface_frames=lambda: frames)
    with pytest.raises(QueryMemoryOheEvidenceError):
        _verify(records, entry=entry, value=value, witness=witness, core=core, maximum=maximum)


@pytest.mark.parametrize('change', ('truncated', 'trailing', 'negative_length', 'bool_length', 'negative_atom', 'bool_atom', 'over_budget_atom', 'wrong_version'))
def test_each_nested_atom_and_all_framing_stay_strict(records, change):
    entry, _value, witness, _resolution, _core, parts = records
    payload = entry.payload_key
    if change == 'truncated':
        payload = payload[:-1]
    elif change == 'trailing':
        payload = (*payload, 0)
    elif change in ('negative_length', 'bool_length'):
        payload = (*payload[:2], -1 if change == 'negative_length' else True, *payload[3:])
    elif change == 'wrong_version':
        payload = (91598, *payload[1:])
    else:
        atom = (-1,) if change == 'negative_atom' else (True,) if change == 'bool_atom' else (0,) * 1000001
        payload = _pack((*parts[:3], atom, *parts[4:]))
    fields = (entry.source_ref, entry.hypothesis_key, (91599, 7, *payload), entry.scope_key, payload, ())
    raw = (1, 3, 1, *(value for part in fields for value in (len(part), *part)))
    changed = SimpleNamespace(stable_key=lambda: raw)
    witness = replace(witness, adoption_evidence_keys=(raw,))
    with pytest.raises(QueryMemoryOheEvidenceError):
        _verify(records, entry=changed, witness=witness)


@pytest.mark.parametrize('plan_change', (None, 'unknown_type', 'changed_member'))
def test_real_ohe_entry_dispatches_state_and_plan_before_independent_resolution_checks(records, plan_change):
    """真实入口检查两处复合键；合成决策在后续独立类型门明确拒绝。"""
    entry, original, witness, resolution, core, _parts = records
    captured, declarations, manifest, _evidence = _fixture()
    value = captured['trace']
    value['input_structure'] = dict(original['input_structure'])
    for kind, key in (('evidence', entry.stable_key()), ('bindings', original['state_key']['bindings'][0])):
        value['integer_key_table'].append(list(key))
        value['state_key'][kind].append(len(value['integer_key_table']))
    planned = entry
    if plan_change:
        planned = replace(entry, **({'space': 2} if plan_change == 'unknown_type' else {'source_ref': (99,)}))
    value['response_plan']['evidence_refs'].append(list(planned.stable_key()))
    match = 'bounded unique cold records' if plan_change is None else 'unregistered compound|current input transition'
    with pytest.raises(QueryMemoryOheEvidenceError, match=match):
        verify_query_memory_ohe(captured,
            declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
            current_manifest_reader=lambda source: manifest, source_hash_reader=lambda source: 19,
            core_runtime=core, reference_transition=witness, reference_resolutions=(resolution,))
