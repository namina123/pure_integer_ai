"""PH2 断奶资料统一合同的薄 facade 与 record_kind dispatch。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.ph2_dataset_core import (
    ALLOWED_LICENSE_IDS,
    EPISTEMIC_ROLES,
    EXPECTED_STATES,
    FORMAT_VERSION,
    JSONL_RECORD_KINDS,
    LOCAL_ONLY_LICENSE_IDS,
    OWNER_KINDS,
    PUBLIC_LICENSE_IDS,
    RECORD_ARTIFACT_MANIFEST,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    REDISTRIBUTION_POLICIES,
    SAMPLE_ROLES,
    SCHEMA_VERSION,
    SPLITS,
    W_STAGES,
    CanonicalJsonObject,
    DatasetContractError,
    StableRecordKey,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_manifest import (
    ArtifactFileIdentity,
    ArtifactManifest,
)
from pure_integer_ai.experiments.ph2_dataset_owner_records import (
    EvaluatorLabelRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_records import (
    ObservationRecord,
    SourceRefRecord,
)


DatasetRecord = (
    SourceRefRecord
    | ObservationRecord
    | TeacherEvidenceRecord
    | EvaluatorLabelRecord
    | ArtifactManifest
)


def record_from_dict(value: dict[str, Any]) -> DatasetRecord:
    """按冻结 record_kind 从 JSON object 恢复统一资料记录。"""
    if not isinstance(value, dict):
        raise DatasetContractError("资料记录根必须是 object")
    kind = value.get("record_kind")
    readers = {
        RECORD_SOURCE_REF: SourceRefRecord.from_dict,
        RECORD_OBSERVATION: ObservationRecord.from_dict,
        RECORD_TEACHER_EVIDENCE: TeacherEvidenceRecord.from_dict,
        RECORD_EVALUATOR_LABEL: EvaluatorLabelRecord.from_dict,
        RECORD_ARTIFACT_MANIFEST: ArtifactManifest.from_dict,
    }
    reader = readers.get(kind)
    if reader is None:
        raise DatasetContractError(f"未知 record_kind: {kind!r}")
    return reader(value)


def record_kind(record: DatasetRecord) -> str:
    """返回统一资料记录的冻结种类。"""
    kind = getattr(record, "RECORD_KIND", None)
    if kind not in JSONL_RECORD_KINDS + (RECORD_ARTIFACT_MANIFEST,):
        raise DatasetContractError("对象不是统一资料记录")
    return kind


__all__ = [
    "ALLOWED_LICENSE_IDS",
    "ArtifactFileIdentity",
    "ArtifactManifest",
    "CanonicalJsonObject",
    "DatasetContractError",
    "DatasetRecord",
    "EPISTEMIC_ROLES",
    "EXPECTED_STATES",
    "EvaluatorLabelRecord",
    "FORMAT_VERSION",
    "JSONL_RECORD_KINDS",
    "LOCAL_ONLY_LICENSE_IDS",
    "ObservationRecord",
    "OWNER_KINDS",
    "PUBLIC_LICENSE_IDS",
    "RECORD_ARTIFACT_MANIFEST",
    "RECORD_EVALUATOR_LABEL",
    "RECORD_OBSERVATION",
    "RECORD_SOURCE_REF",
    "RECORD_TEACHER_EVIDENCE",
    "REDISTRIBUTION_POLICIES",
    "SAMPLE_ROLES",
    "SCHEMA_VERSION",
    "SPLITS",
    "SourceRefRecord",
    "StableRecordKey",
    "TeacherEvidenceRecord",
    "W_STAGES",
    "canonical_json_bytes",
    "canonical_json_line",
    "parse_canonical_json_bytes",
    "record_from_dict",
    "record_kind",
]
