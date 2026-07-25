"""把 A-01 singleton adopted 决策投影到 A-02 transient 内容。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentItem,
    WorkMemoryOccurrenceAnchor,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
)
from pure_integer_ai.cognition.understanding.occurrence_reference import (
    OccurrenceReferenceResolution,
)


def project_occurrence_reference_to_work_memory(
        work_memory: WorkMemory,
        occurrence_index: OccurrenceIndex,
        resolution: OccurrenceReferenceResolution,
        *,
        role: ObjectIdentity,
        logical_seq: int,
        trace: tuple[int, ...],
        supersedes: tuple[tuple[int, ...], ...] = (),
        ) -> WorkMemoryContentItem | None:
    """核验 A-01 图身份，仅将唯一 adopted antecedent 写入指定开放 Role。"""
    if not isinstance(work_memory, WorkMemory):
        raise TypeError("A-01 WorkMemory adapter 需要 WorkMemory")
    if not isinstance(occurrence_index, OccurrenceIndex):
        raise TypeError("A-01 WorkMemory adapter 需要 OccurrenceIndex")
    if not isinstance(resolution, OccurrenceReferenceResolution):
        raise TypeError("A-01 WorkMemory adapter resolution 类型错误")
    winner = resolution.winner
    if winner is None:
        return None
    reference_record = occurrence_index.read(resolution.request.reference)
    antecedent_record = occurrence_index.read(winner.antecedent)
    reference_identity = occurrence_index.ontology.identity_of(
        reference_record.occurrence)
    antecedent_identity = occurrence_index.ontology.identity_of(
        antecedent_record.occurrence)
    if (reference_identity != winner.reference_identity
            or antecedent_identity != winner.antecedent_identity):
        raise ValueError("A-01 WorkMemory adapter 图身份与 decision 漂移")
    if reference_record.source != antecedent_record.source:
        raise ValueError("A-01 WorkMemory adapter 不接受跨来源 winner")
    lifespan_scope = work_memory.require_content_store().scope_for_role(role)
    item = WorkMemoryContentItem(
        role,
        antecedent_identity,
        WorkMemoryOccurrenceAnchor(
            reference_record.occurrence,
            reference_identity,
            reference_record.source,
            reference_record.scope,
        ),
        lifespan_scope,
        logical_seq,
        trace,
        supersedes,
    )
    return work_memory.put_content(item)


__all__ = ["project_occurrence_reference_to_work_memory"]
