"""W-06 candidate train-only payload 的不可变容器。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)


@dataclass(frozen=True)
class W06TrainingPayload:
    """保存 firewall 核准的 source、train observation 和 train Evidence。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


__all__ = ["W06TrainingPayload"]
