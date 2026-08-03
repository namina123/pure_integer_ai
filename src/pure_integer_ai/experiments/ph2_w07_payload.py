"""W-07 train-only logic payload 的不可变公开容器。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)


@dataclass(frozen=True)
class W07TrainingPayload:
    """保存 firewall 核准的 source、train observation 和 train Evidence。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


__all__ = ["W07TrainingPayload"]
