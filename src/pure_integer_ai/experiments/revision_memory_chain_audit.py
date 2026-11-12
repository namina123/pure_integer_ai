"""冷核验追加纠错证据的物理替代链，不追认历史更正的语义正确性。"""
from pure_integer_ai.cognition.shared.memory_event import (
    EvidencePayload, INTAKE_DERIVED_EVIDENCE, INTAKE_DERIVED_HYPOTHESIS,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.query_memory_candidates import memory_topic_candidate
from pure_integer_ai.experiments.natural_revision_runtime import (
    REVISION_DETAIL_NAMESPACE, REVISION_REASON_NAMESPACE,
)


def verify_revision_memory_chain(row, hypothesis_row, *, declaration, manifest,
                                  observation, cue_records, max_steps, error):
    """在原声明预算内逐项回到首次 H/E lineage，所有来源和序均原位核验。"""
    hypothesis = hypothesis_row.event.payload.hypothesis
    href = hypothesis_row.event.object_ref
    original_source = hypothesis.observation
    hbindings = tuple(item for item in manifest(original_source).bindings
        if item.binding_kind == INTAKE_DERIVED_HYPOTHESIS and item.object_ref == href)
    cues = {(*REVISION_REASON_NAMESPACE, *item.cue.cue_identity) for item in cue_records}
    if len(hbindings) != 1 or not cues:
        raise error('initial intake O/H/E bindings differ; appended Evidence is unqualified')
    try:
        topic = memory_topic_candidate(hypothesis.candidate_key)
    except ValueError as cause:
        raise error('revision Evidence lacks a complete topic identity') from cause
    if topic is None:
        raise error('revision Evidence lacks a complete topic identity')
    seen = set()
    while True:
        ref, payload = row.event.object_ref, row.event.payload
        if (ref in seen or len(seen) >= max_steps or not isinstance(payload, EvidencePayload)
                or payload.hypothesis_ref != href or payload.source is None):
            raise error('revision Evidence chain cycles, exceeds budget or changes its Hypothesis')
        seen.add(ref)
        source = payload.source
        ebindings = tuple(item for item in manifest(source).bindings
            if item.binding_kind == INTAKE_DERIVED_EVIDENCE and item.object_ref == ref)
        if source == original_source:
            if (len(ebindings) != 1 or ebindings[0].lineage_key != hbindings[0].lineage_key
                    or payload.supersedes_ref is not None):
                raise error('revision Evidence chain lacks its exact initial intake lineage')
            return len(seen) - 1
        if (ebindings or payload.supersedes_ref is None or payload.stance not in (1, 2)
                or payload.legacy_reason_key not in cues
                or payload.detail != (*REVISION_DETAIL_NAMESPACE, *topic.proposition_key)
                or source.owner != original_source.owner or source.versions != original_source.versions
                or source.source_kind != original_source.source_kind
                or source.source_id != original_source.source_id
                or row.event.scope != document_scope(source)):
            raise error('revision Evidence chain lacks a mounted cue or complete source-qualified append')
        observation(source)
        prior = declaration(payload.supersedes_ref)
        earlier = prior.event.payload
        if (not isinstance(earlier, EvidencePayload) or earlier.source is None
                or earlier.hypothesis_ref != href or earlier.stance != 1
                or earlier.source.owner != source.owner or earlier.source.versions != source.versions
                or earlier.source.source_id != source.source_id
                or earlier.source.source_kind != source.source_kind
                or earlier.source.document_id >= source.document_id
                or earlier.observed_at.clock != payload.observed_at.clock
                or earlier.observed_at.seq >= payload.observed_at.seq):
            raise error('revision Evidence supersedes changes source, target, stance or logical order')
        row = prior
