"""W-05 train-only 原子 payload 容器。

本文件只定义 firewall 成功后交付给 adapter 的只读记录集合；它不承担
读取、解析路径、学习或 evaluator label 逻辑，避免把 owner 边界混入
候选语义层。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)


@dataclass(frozen=True)
class W05TrainingPayload:
    """W-05 candidate 可见的 source、train observation 与 train Evidence。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


__all__ = ["W05TrainingPayload"]
