"""W-08 firewall 核准后的不可变训练 payload。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)


@dataclass(frozen=True)
class W08TrainingPayload:
    """保存来源、train Observation 与既有 train Evidence。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


__all__ = ["W08TrainingPayload"]
