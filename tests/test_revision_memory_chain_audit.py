"""真实追加链冷恢复及缺来源、错序、丢替代边的拒绝核验。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_EVIDENCE, MEMORY_RECORD_IDENTITY_VERSION, MemoryObjectRef, declaration_object_key,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.revision_memory_chain_audit import verify_revision_memory_chain
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from test_natural_revision_scope import _setup


def _records(tmp_path):
    """由真实支持更正及否定更正生成两段替代链，之后只读原记录。"""
    projector, _, cue, runtime, corrected = _setup()
    sources = []
    with TrainedDialogueMemoryGraph(tmp_path/'revision-chain.sqlite3', session_id=23,
                                   input_projector=projector) as memory:
        original = projector.project((11,33,22))
        sources.append(memory.append(''.join(map(chr,original.token_values)), speaker_kind=1,
                                     input_structure=original).source)
        for structure in (projector.project((91,11,33,22)), corrected):
            binding = runtime.bind_input(structure)
            target = runtime.resolve_target(memory, structure, binding)
            current = memory.append(''.join(map(chr,structure.token_values)), speaker_kind=1,
                                    input_structure=structure)
            sources.append(current.source)
            commit = runtime.commit(memory, target, current.source)
        declarations = {row.event.object_ref:row for row in memory.intake.event_log.query(
            access=memory.candidate_index.access) if row.event.is_declaration}
        manifests = {source:memory.intake.require_current_manifest(source) for source in sources}
    row = declarations[MemoryObjectRef.from_stable_key(commit.replacement_evidence_key)]
    hrow = declarations[row.event.payload.hypothesis_ref]

    def declaration(ref):
        """模拟共同消费者的唯一声明读取，缺项明确关闭。"""
        if ref not in declarations:
            raise QueryMemoryOheEvidenceError('missing unique declaration')
        return declarations[ref]

    return row, hrow, declarations, manifests, dict(declaration=declaration,
        manifest=lambda source:manifests[source], observation=lambda source:manifests[source],
        cue_records=(cue,), max_steps=64, error=QueryMemoryOheEvidenceError)


def test_real_two_append_chain_is_recovered_without_mutating_history(tmp_path):
    row, hrow, declarations, manifests, context = _records(tmp_path)
    before = tuple(declarations.items()), tuple(manifests.items())
    assert verify_revision_memory_chain(row, hrow, **context) == 2
    assert (tuple(declarations.items()), tuple(manifests.items())) == before


@pytest.mark.parametrize('drift', (
    'cue', 'reason', 'detail', 'supersedes', 'source_order', 'clock_order',
    'missing', 'lineage', 'budget',
))
def test_append_chain_cannot_borrow_initial_intake_or_unrelated_history(tmp_path, drift):
    row, hrow, declarations, manifests, context = _records(tmp_path)
    payload = row.event.payload
    if drift == 'cue':
        context['cue_records'] = ()
    elif drift in ('reason','detail','supersedes','source_order','clock_order'):
        updates = {'reason':dict(legacy_reason_key=(999,)), 'detail':dict(detail=(999,)),
            'supersedes':dict(supersedes_ref=None),
            'source_order':dict(source=hrow.event.payload.hypothesis.observation),
            'clock_order':dict(observed_at=replace(payload.observed_at,seq=1))}
        changed = replace(payload,**updates[drift])
        ref = row.event.object_ref
        if drift == 'clock_order':
            ref = replace(ref,object_key=declaration_object_key(row.event.event_kind,changed,
                identity_version=MEMORY_RECORD_IDENTITY_VERSION))
        row = replace(row,event=replace(row.event,payload=changed,object_ref=ref))
    elif drift == 'missing':
        declarations.pop(payload.supersedes_ref)
    elif drift == 'lineage':
        source = hrow.event.payload.hypothesis.observation
        manifest = manifests[source]
        initial = next(ref for ref,item in declarations.items()
            if getattr(item.event.payload,'hypothesis_ref',None)==hrow.event.object_ref
            and getattr(item.event.payload,'source',None)==source)
        first = next(item for item in manifest.bindings if item.object_ref==initial)
        other = next(item for item in manifest.bindings
            if item.binding_kind==INTAKE_DERIVED_EVIDENCE and item.object_ref!=initial)
        manifests[source] = replace(manifest, bindings=tuple(
            replace(item,lineage_key=other.lineage_key) if item==first else
            replace(item,lineage_key=first.lineage_key) if item==other else item
            for item in manifest.bindings))
    else:
        context['max_steps'] = 1
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_revision_memory_chain(row, hrow, **context)
