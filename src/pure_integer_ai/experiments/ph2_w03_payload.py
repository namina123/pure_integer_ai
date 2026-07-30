"""W-03 firewall 与纯 adapter 共用的 train-only 值合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)


@dataclass(frozen=True)
class W03TrainingPayload:
    """完整校验后原子交付的 typed train-only 记录。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


__all__ = ["W03TrainingPayload"]
