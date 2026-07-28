"""PH2 teacher Evidence 与只读 evaluator label 的独立 owner 合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pure_integer_ai.experiments.ph2_dataset_core import (
    EXPECTED_STATES,
    OWNER_MODES,
    RECORD_EVALUATOR_LABEL,
    RECORD_TEACHER_EVIDENCE,
    W_STAGES,
    CanonicalJsonObject,
    DatasetContractError,
    StableRecordKey,
    _enum_text,
    _nonempty_text,
    _nonnegative_int,
    _positive_int,
)


@dataclass(frozen=True)
class TeacherEvidenceRecord:
    """由独立 teacher owner 持有、按阶段和退场级别受控的 typed Evidence。"""

    RECORD_KIND: ClassVar[str] = RECORD_TEACHER_EVIDENCE

    format_version: int
    schema_version: int
    course_version: int
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    stable_key: StableRecordKey
    observation_key: StableRecordKey
    evidence_kind: str
    typed_evidence: CanonicalJsonObject
    source_ref_key: StableRecordKey
    visible_from_stage: str
    withdrawal_level: int
    owner_key: StableRecordKey

    def __post_init__(self) -> None:
        for name, value in (
                ("format_version", self.format_version),
                ("schema_version", self.schema_version),
                ("course_version", self.course_version)):
            _positive_int(value, where=f"TeacherEvidenceRecord.{name}")
        if any(not isinstance(value, StableRecordKey) for value in (
                self.dataset_key, self.artifact_key,
                self.stable_key, self.observation_key,
                self.source_ref_key, self.owner_key)):
            raise DatasetContractError("TeacherEvidenceRecord 整数键字段类型错误")
        _nonempty_text(self.evidence_kind, where="TeacherEvidenceRecord.evidence_kind")
        if not isinstance(self.typed_evidence, CanonicalJsonObject):
            raise DatasetContractError("TeacherEvidenceRecord.typed_evidence 类型错误")
        _enum_text(
            self.visible_from_stage,
            W_STAGES,
            where="TeacherEvidenceRecord.visible_from_stage",
        )
        _nonnegative_int(
            self.withdrawal_level,
            where="TeacherEvidenceRecord.withdrawal_level",
        )
        if self.withdrawal_level > 3:
            raise DatasetContractError("TeacherEvidenceRecord.withdrawal_level 必须在 0..3")

    def to_dict(self) -> dict[str, Any]:
        """导出 TeacherEvidenceRecord 的规范 JSON object。"""
        return {
            "artifact_key": self.artifact_key.to_list(),
            "course_version": self.course_version,
            "dataset_key": self.dataset_key.to_list(),
            "evidence_kind": self.evidence_kind,
            "format_version": self.format_version,
            "observation_key": self.observation_key.to_list(),
            "owner_key": self.owner_key.to_list(),
            "record_kind": self.RECORD_KIND,
            "schema_version": self.schema_version,
            "source_ref_key": self.source_ref_key.to_list(),
            "stable_key": self.stable_key.to_list(),
            "typed_evidence": self.typed_evidence.to_value(),
            "visible_from_stage": self.visible_from_stage,
            "withdrawal_level": self.withdrawal_level,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherEvidenceRecord":
        """从已解析 JSON object 恢复 TeacherEvidenceRecord。"""
        return cls(
            value["format_version"],
            value["schema_version"],
            value["course_version"],
            StableRecordKey.from_value(value["dataset_key"], where="teacher.dataset_key"),
            StableRecordKey.from_value(value["artifact_key"], where="teacher.artifact_key"),
            StableRecordKey.from_value(value["stable_key"], where="teacher.stable_key"),
            StableRecordKey.from_value(
                value["observation_key"], where="teacher.observation_key"),
            str(value["evidence_kind"]),
            CanonicalJsonObject.from_value(value["typed_evidence"]),
            StableRecordKey.from_value(
                value["source_ref_key"], where="teacher.source_ref_key"),
            str(value["visible_from_stage"]),
            value["withdrawal_level"],
            StableRecordKey.from_value(value["owner_key"], where="teacher.owner_key"),
        )


@dataclass(frozen=True)
class EvaluatorLabelRecord:
    """由只读 evaluator owner 持有的分维四态/结构标签与预算。"""

    RECORD_KIND: ClassVar[str] = RECORD_EVALUATOR_LABEL

    format_version: int
    schema_version: int
    course_version: int
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    stable_key: StableRecordKey
    observation_key: StableRecordKey
    dimension_key: StableRecordKey
    expected_state: str
    expected_payload: CanonicalJsonObject
    budget_units: int
    evaluator_version: int
    visible_stage: str
    owner_key: StableRecordKey
    owner_mode: str = "read_only"

    def __post_init__(self) -> None:
        for name, value in (
                ("format_version", self.format_version),
                ("schema_version", self.schema_version),
                ("course_version", self.course_version),
                ("evaluator_version", self.evaluator_version)):
            _positive_int(value, where=f"EvaluatorLabelRecord.{name}")
        if any(not isinstance(value, StableRecordKey) for value in (
                self.dataset_key, self.artifact_key,
                self.stable_key, self.observation_key,
                self.dimension_key, self.owner_key)):
            raise DatasetContractError("EvaluatorLabelRecord 整数键字段类型错误")
        _enum_text(
            self.expected_state,
            EXPECTED_STATES,
            where="EvaluatorLabelRecord.expected_state",
        )
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise DatasetContractError("EvaluatorLabelRecord.expected_payload 类型错误")
        _nonnegative_int(self.budget_units, where="EvaluatorLabelRecord.budget_units")
        _enum_text(self.visible_stage, W_STAGES, where="EvaluatorLabelRecord.visible_stage")
        _enum_text(self.owner_mode, OWNER_MODES, where="EvaluatorLabelRecord.owner_mode")

    def to_dict(self) -> dict[str, Any]:
        """导出 EvaluatorLabelRecord 的规范 JSON object。"""
        return {
            "artifact_key": self.artifact_key.to_list(),
            "budget_units": self.budget_units,
            "course_version": self.course_version,
            "dataset_key": self.dataset_key.to_list(),
            "dimension_key": self.dimension_key.to_list(),
            "evaluator_version": self.evaluator_version,
            "expected_payload": self.expected_payload.to_value(),
            "expected_state": self.expected_state,
            "format_version": self.format_version,
            "observation_key": self.observation_key.to_list(),
            "owner_key": self.owner_key.to_list(),
            "owner_mode": self.owner_mode,
            "record_kind": self.RECORD_KIND,
            "schema_version": self.schema_version,
            "stable_key": self.stable_key.to_list(),
            "visible_stage": self.visible_stage,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluatorLabelRecord":
        """从已解析 JSON object 恢复 EvaluatorLabelRecord。"""
        return cls(
            value["format_version"],
            value["schema_version"],
            value["course_version"],
            StableRecordKey.from_value(value["dataset_key"], where="evaluator.dataset_key"),
            StableRecordKey.from_value(value["artifact_key"], where="evaluator.artifact_key"),
            StableRecordKey.from_value(value["stable_key"], where="evaluator.stable_key"),
            StableRecordKey.from_value(
                value["observation_key"], where="evaluator.observation_key"),
            StableRecordKey.from_value(
                value["dimension_key"], where="evaluator.dimension_key"),
            str(value["expected_state"]),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            value["budget_units"],
            value["evaluator_version"],
            str(value["visible_stage"]),
            StableRecordKey.from_value(value["owner_key"], where="evaluator.owner_key"),
            str(value["owner_mode"]),
        )


__all__ = ["EvaluatorLabelRecord", "TeacherEvidenceRecord"]
