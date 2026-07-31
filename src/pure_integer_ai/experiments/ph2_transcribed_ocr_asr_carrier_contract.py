"""LC-16 OCR/ASR 转写载体的 data-only payload 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
    TypedCarrierBudget,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_LC16_TRANSCRIBED_OCR_ASR_CARRIER_PAYLOAD"
ARTIFACT_VERSION = "LC16-TRANSCRIBED-OCR-ASR-CARRIER-20260731-A"
ARTIFACT_STATUS = "PAYLOAD_FROZEN_ALIGNMENT_ADAPTER_ONLY"
CARRIER_KEY = "TRANSCRIBED_OCR_ASR"
LICENSE_ID = "CC0-1.0"
PARSER_PACKAGE = "python-stdlib"
PARSER_VERSION = "transcript-alignment-v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SOURCE_MODES = ("ASR", "OCR", "UNKNOWN")
TEMPORAL_STATES = ("ALIGNED", "UNAVAILABLE")
READ_ORDERS = ("CHRONOLOGICAL", "SOURCE_ORDER")
ADAPTER_OBLIGATIONS = {
    "POSITIVE": "SEGMENT_ALIGNMENT_ROUND_TRIP",
    "NEGATIVE": "RAW_TRANSCRIPT_AND_INVALID_STATE_PRESERVATION",
    "AMBIGUOUS": "CANDIDATE_SPEAKER_AND_CONFIDENCE_PRESERVATION",
    "UNKNOWN": "UNKNOWN_SOURCE_PRESERVATION",
    "REVISION": "SEGMENT_REVISION_MAPPING",
    "GENERATION": "EXACT_TRANSCRIPT_SERIALIZATION",
    "RETENTION": "TRANSCRIPT_RELOAD_IDENTITY",
}
EVIDENCE_ROLES = ("ADAPTER", "CATALOG", "CONTRACT", "DEPENDENCY", "TEST")
EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "carrier_qualified_runtime_authority": 0,
    "formal_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}


class TranscribedOcrAsrCarrierContractError(RuntimeError):
    """转写 payload、manifest 或文件证据不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TranscribedOcrAsrCarrierContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False,
          strip: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TranscribedOcrAsrCarrierContractError(f"{where} 必须是文本")
    if strip and value and value.strip() != value:
        raise TranscribedOcrAsrCarrierContractError(f"{where} 含首尾空白")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise TranscribedOcrAsrCarrierContractError(
            f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise TranscribedOcrAsrCarrierContractError(
            f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise TranscribedOcrAsrCarrierContractError(
            f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in text or path.as_posix() != text
            or ":" in path.parts[0]):
        raise TranscribedOcrAsrCarrierContractError(
            f"{where} 必须是安全 POSIX 相对路径")
    return text


def _raw_identity(text: str, unit_count: Any, sha256: Any, *, where: str,
                  allow_empty: bool = False) -> None:
    _text(text, where=f"{where} text", allow_empty=allow_empty, strip=False)
    if text:
        try:
            validate_unicode_scalars(tuple(ord(item) for item in text))
        except Exception as error:
            raise TranscribedOcrAsrCarrierContractError(
                f"{where} 含非法 Unicode scalar") from error
    if _nonnegative(unit_count, where=f"{where} unit_count") != len(text):
        raise TranscribedOcrAsrCarrierContractError(f"{where} unit_count 漂移")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _sha256(sha256, where=f"{where} utf8_sha256") != expected:
        raise TranscribedOcrAsrCarrierContractError(f"{where} UTF-8 SHA-256 漂移")


@dataclass(frozen=True, order=True)
class TranscriptSegment:
    """单一转写片段；时间、置信度和序号均保持严格整数。"""

    segment_id: int
    ordinal: int
    text_start: int
    text_end: int
    time_start_ms: int
    time_end_ms: int
    speaker_candidates: tuple[str, ...]
    confidence_candidates: tuple[int, ...]
    source_mode: str
    temporal_state: str

    def __post_init__(self) -> None:
        _positive(self.segment_id, where="segment_id")
        _nonnegative(self.ordinal, where="segment ordinal")
        _nonnegative(self.text_start, where="segment text_start")
        _positive(self.text_end, where="segment text_end")
        if self.text_start >= self.text_end:
            raise TranscribedOcrAsrCarrierContractError(
                "segment text range 必须递增")
        _nonnegative(self.time_start_ms, where="segment time_start_ms")
        _nonnegative(self.time_end_ms, where="segment time_end_ms")
        if self.temporal_state not in TEMPORAL_STATES:
            raise TranscribedOcrAsrCarrierContractError(
                "segment temporal_state 未登记")
        if (self.temporal_state == "ALIGNED"
                and self.time_start_ms >= self.time_end_ms):
            raise TranscribedOcrAsrCarrierContractError(
                "segment time range 必须递增")
        if (self.temporal_state == "UNAVAILABLE"
                and (self.time_start_ms, self.time_end_ms) != (0, 0)):
            raise TranscribedOcrAsrCarrierContractError(
                "UNAVAILABLE temporal state 不得伪造时间")
        if (not isinstance(self.speaker_candidates, tuple)
                or any(_text(item, where="speaker candidate") != item
                       for item in self.speaker_candidates)
                or self.speaker_candidates != tuple(
                    dict.fromkeys(self.speaker_candidates))):
            raise TranscribedOcrAsrCarrierContractError(
                "speaker_candidates 非法")
        if (not isinstance(self.confidence_candidates, tuple)
                or any(type(item) is not int or not 0 <= item <= 1000
                       for item in self.confidence_candidates)
                or self.confidence_candidates != tuple(
                    dict.fromkeys(self.confidence_candidates))):
            raise TranscribedOcrAsrCarrierContractError(
                "confidence_candidates 非法")
        if self.source_mode not in SOURCE_MODES:
            raise TranscribedOcrAsrCarrierContractError(
                "segment source_mode 未登记")

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_candidates": list(self.confidence_candidates),
            "ordinal": self.ordinal,
            "segment_id": self.segment_id,
            "source_mode": self.source_mode,
            "speaker_candidates": list(self.speaker_candidates),
            "text_end": self.text_end,
            "text_start": self.text_start,
            "temporal_state": self.temporal_state,
            "time_end_ms": self.time_end_ms,
            "time_start_ms": self.time_start_ms,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscriptSegment":
        raw = _exact(value, {
            "confidence_candidates", "ordinal", "segment_id", "source_mode",
            "speaker_candidates", "text_end", "text_start", "time_end_ms",
            "time_start_ms", "temporal_state",
        }, where="TranscriptSegment")
        return cls(
            raw["segment_id"], raw["ordinal"], raw["text_start"],
            raw["text_end"], raw["time_start_ms"], raw["time_end_ms"],
            tuple(str(item) for item in raw["speaker_candidates"]),
            tuple(raw["confidence_candidates"]), str(raw["source_mode"]),
            str(raw["temporal_state"]),
        )


def _segments(value: tuple[TranscriptSegment, ...], text: str, *, where: str,
              allow_empty: bool = False) -> None:
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(not isinstance(item, TranscriptSegment) for item in value)):
        raise TranscribedOcrAsrCarrierContractError(f"{where} segments 非法")
    if tuple(item.ordinal for item in value) != tuple(range(len(value))):
        raise TranscribedOcrAsrCarrierContractError(f"{where} ordinal 不连续")
    if len({item.segment_id for item in value}) != len(value):
        raise TranscribedOcrAsrCarrierContractError(f"{where} segment_id 重复")
    for item in value:
        if item.text_end > len(text):
            raise TranscribedOcrAsrCarrierContractError(
                f"{where} text range 超出 raw text")


@dataclass(frozen=True, order=True)
class TranscribedOcrAsrCarrierRecord:
    """绑定 parent case 的 OCR/ASR 原始转写与来源化对齐候选。"""

    format_version: int
    case_key: StableRecordKey
    sample_kind: str
    adapter_obligation: str
    license_id: str
    read_order: str
    raw_text: str
    raw_unit_count: int
    raw_utf8_sha256: str
    segments: tuple[TranscriptSegment, ...]
    previous_text: str
    previous_unit_count: int
    previous_utf8_sha256: str
    previous_segments: tuple[TranscriptSegment, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise TranscribedOcrAsrCarrierContractError("record format_version 非法")
        if not isinstance(self.case_key, StableRecordKey):
            raise TranscribedOcrAsrCarrierContractError("record case_key 非法")
        if self.sample_kind not in SAMPLE_KINDS:
            raise TranscribedOcrAsrCarrierContractError("record sample_kind 未登记")
        if self.adapter_obligation != ADAPTER_OBLIGATIONS[self.sample_kind]:
            raise TranscribedOcrAsrCarrierContractError("adapter obligation 漂移")
        if self.license_id != LICENSE_ID:
            raise TranscribedOcrAsrCarrierContractError(
                "TRANSCRIBED_OCR_ASR payload 必须为 CC0-1.0")
        if self.read_order not in READ_ORDERS:
            raise TranscribedOcrAsrCarrierContractError("read_order 未登记")
        _raw_identity(self.raw_text, self.raw_unit_count, self.raw_utf8_sha256,
                      where="record raw")
        _segments(self.segments, self.raw_text, where="record")
        if self.read_order == "CHRONOLOGICAL":
            aligned = tuple(item for item in self.segments
                            if item.temporal_state == "ALIGNED")
            if aligned != tuple(sorted(
                    aligned,
                    key=lambda item: (
                        item.time_start_ms, item.time_end_ms, item.ordinal))):
                raise TranscribedOcrAsrCarrierContractError(
                    "CHRONOLOGICAL segments 时间顺序漂移")
        _raw_identity(self.previous_text, self.previous_unit_count,
                      self.previous_utf8_sha256, where="record previous",
                      allow_empty=True)
        _segments(self.previous_segments, self.previous_text,
                  where="record previous", allow_empty=True)
        if self.sample_kind == "REVISION":
            if (not self.previous_text or self.previous_text == self.raw_text
                    or not self.previous_segments):
                raise TranscribedOcrAsrCarrierContractError(
                    "REVISION 必须携带不同的 previous transcript")
        elif (self.previous_text or self.previous_unit_count
              or self.previous_utf8_sha256 != EMPTY_SHA256
              or self.previous_segments):
            raise TranscribedOcrAsrCarrierContractError(
                "非 REVISION 不得携带 previous payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_obligation": self.adapter_obligation,
            "case_key": self.case_key.to_list(),
            "format_version": self.format_version,
            "license_id": self.license_id,
            "previous_segments": [item.to_dict() for item in self.previous_segments],
            "previous_text": self.previous_text,
            "previous_unit_count": self.previous_unit_count,
            "previous_utf8_sha256": self.previous_utf8_sha256,
            "raw_text": self.raw_text,
            "raw_unit_count": self.raw_unit_count,
            "raw_utf8_sha256": self.raw_utf8_sha256,
            "read_order": self.read_order,
            "sample_kind": self.sample_kind,
            "segments": [item.to_dict() for item in self.segments],
        }

    def canonical_line(self) -> bytes:
        return canonical_json_line(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscribedOcrAsrCarrierRecord":
        raw = _exact(value, {
            "adapter_obligation", "case_key", "format_version", "license_id",
            "previous_segments", "previous_text", "previous_unit_count",
            "previous_utf8_sha256", "raw_text", "raw_unit_count",
            "raw_utf8_sha256", "read_order", "sample_kind", "segments",
        }, where="TranscribedOcrAsrCarrierRecord")
        try:
            key = StableRecordKey.from_value(raw["case_key"], where="case_key")
            segments = tuple(TranscriptSegment.from_dict(item)
                            for item in raw["segments"])
            previous_segments = tuple(TranscriptSegment.from_dict(item)
                                     for item in raw["previous_segments"])
        except Exception as error:
            raise TranscribedOcrAsrCarrierContractError(
                "record nested field 损坏") from error
        return cls(
            raw["format_version"], key, str(raw["sample_kind"]),
            str(raw["adapter_obligation"]), str(raw["license_id"]),
            str(raw["read_order"]), str(raw["raw_text"]), raw["raw_unit_count"],
            str(raw["raw_utf8_sha256"]), segments, str(raw["previous_text"]),
            raw["previous_unit_count"], str(raw["previous_utf8_sha256"]),
            previous_segments,
        )


def read_transcribed_ocr_asr_carrier_records(
        path: str | Path,
        ) -> tuple[TranscribedOcrAsrCarrierRecord, ...]:
    """严格读取七行 canonical JSONL 转写 payload。"""
    try:
        payload = Path(path).read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise TranscribedOcrAsrCarrierContractError("payload 缺少终止换行")
        lines = payload.splitlines(keepends=True)
        if any(not line.endswith(b"\n") or line == b"\n" for line in lines):
            raise TranscribedOcrAsrCarrierContractError("payload 含坏行或空行")
        records = []
        for line in lines:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            if canonical_json_line(value) != line:
                raise TranscribedOcrAsrCarrierContractError(
                    "payload 行不是 canonical JSON")
            records.append(TranscribedOcrAsrCarrierRecord.from_dict(value))
    except TranscribedOcrAsrCarrierContractError:
        raise
    except Exception as error:
        raise TranscribedOcrAsrCarrierContractError("payload 损坏") from error
    result = tuple(records)
    if (len(result) != len(SAMPLE_KINDS)
            or tuple(item.sample_kind for item in result) != SAMPLE_KINDS
            or result != tuple(sorted(result, key=lambda item: item.case_key))
            or len({item.case_key for item in result}) != len(result)):
        raise TranscribedOcrAsrCarrierContractError("payload 七类样本覆盖或顺序漂移")
    return result


def write_transcribed_ocr_asr_carrier_records(
        records: tuple[TranscribedOcrAsrCarrierRecord, ...],
        path: str | Path,
        ) -> Path:
    if (not isinstance(records, tuple)
            or any(not isinstance(item, TranscribedOcrAsrCarrierRecord)
                   for item in records)):
        raise TranscribedOcrAsrCarrierContractError("records 类型非法")
    payload = b"".join(item.canonical_line() for item in records)
    target = Path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise TranscribedOcrAsrCarrierContractError(
                "payload 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TranscribedOcrAsrCarrierContractError("payload 无法写入") from error
    return target


@dataclass(frozen=True, order=True)
class TranscribedOcrAsrMaterializationIdentity:
    case_key: StableRecordKey
    byte_count: int
    sha256: str
    raw_unit_count: int
    previous_raw_unit_count: int
    envelope_count: int
    anchor_count: int
    alignment_count: int
    structure_node_count: int
    revision_count: int
    revision_mapping_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise TranscribedOcrAsrCarrierContractError("materialization case_key 非法")
        _positive(self.byte_count, where="materialization byte_count")
        _sha256(self.sha256, where="materialization sha256")
        _positive(self.raw_unit_count, where="materialization raw_unit_count")
        _nonnegative(self.previous_raw_unit_count,
                     where="materialization previous_raw_unit_count")
        for name in ("envelope_count", "anchor_count", "alignment_count",
                     "structure_node_count", "revision_count",
                     "revision_mapping_count"):
            _nonnegative(getattr(self, name), where=f"materialization {name}")
        if self.alignment_count <= 0 or self.structure_node_count <= 0:
            raise TranscribedOcrAsrCarrierContractError(
                "materialization 缺少 alignment 或 structure nodes")
        if self.anchor_count <= self.envelope_count * 2:
            raise TranscribedOcrAsrCarrierContractError(
                "materialization 未保留 transcript alignment anchors")
        if self.revision_count == 0:
            if (self.previous_raw_unit_count != 0 or self.envelope_count != 1
                    or self.revision_mapping_count != 0):
                raise TranscribedOcrAsrCarrierContractError(
                    "普通 materialization 对象计数漂移")
        elif self.revision_count == 1:
            if (self.previous_raw_unit_count <= 0 or self.envelope_count != 2
                    or self.revision_mapping_count <= 0):
                raise TranscribedOcrAsrCarrierContractError(
                    "revision materialization 对象计数漂移")
        else:
            raise TranscribedOcrAsrCarrierContractError("revision_count 必须为 0/1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment_count": self.alignment_count,
            "anchor_count": self.anchor_count,
            "byte_count": self.byte_count,
            "case_key": self.case_key.to_list(),
            "envelope_count": self.envelope_count,
            "previous_raw_unit_count": self.previous_raw_unit_count,
            "raw_unit_count": self.raw_unit_count,
            "revision_count": self.revision_count,
            "revision_mapping_count": self.revision_mapping_count,
            "sha256": self.sha256,
            "structure_node_count": self.structure_node_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscribedOcrAsrMaterializationIdentity":
        raw = _exact(value, {
            "alignment_count", "anchor_count", "byte_count", "case_key",
            "envelope_count", "previous_raw_unit_count", "raw_unit_count",
            "revision_count", "revision_mapping_count", "sha256",
            "structure_node_count",
        }, where="TranscribedOcrAsrMaterializationIdentity")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            raw["byte_count"], str(raw["sha256"]), raw["raw_unit_count"],
            raw["previous_raw_unit_count"], raw["envelope_count"],
            raw["anchor_count"], raw["alignment_count"],
            raw["structure_node_count"], raw["revision_count"],
            raw["revision_mapping_count"],
        )


@dataclass(frozen=True, order=True)
class TranscribedOcrAsrCarrierEvidenceFile:
    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise TranscribedOcrAsrCarrierContractError("evidence role 未登记")
        _positive(self.byte_count, where="evidence byte_count")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"byte_count": self.byte_count, "relative_path": self.relative_path,
                "role": self.role, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscribedOcrAsrCarrierEvidenceFile":
        raw = _exact(value, {"byte_count", "relative_path", "role", "sha256"},
                     where="TranscribedOcrAsrCarrierEvidenceFile")
        return cls(str(raw["relative_path"]), str(raw["role"]),
                   raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class TranscribedOcrAsrCarrierPayloadManifest:
    format_version: int
    artifact_version: str
    artifact_status: str
    parent_pack_relative_path: str
    parent_pack_sha256: str
    sample_relative_path: str
    sample_sha256: str
    carrier_key: str
    license_id: str
    parser_package: str
    parser_version: str
    budget: TypedCarrierBudget
    case_keys: tuple[StableRecordKey, ...]
    sample_kinds: tuple[str, ...]
    materializations: tuple[TranscribedOcrAsrMaterializationIdentity, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[TranscribedOcrAsrCarrierEvidenceFile, ...]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise TranscribedOcrAsrCarrierContractError("manifest artifact identity 漂移")
        _relative_path(self.parent_pack_relative_path,
                       where="parent_pack_relative_path")
        _sha256(self.parent_pack_sha256, where="parent_pack_sha256")
        _relative_path(self.sample_relative_path, where="sample_relative_path")
        _sha256(self.sample_sha256, where="sample_sha256")
        if (self.carrier_key != CARRIER_KEY or self.license_id != LICENSE_ID
                or self.parser_package != PARSER_PACKAGE
                or self.parser_version != PARSER_VERSION):
            raise TranscribedOcrAsrCarrierContractError("manifest carrier/parser identity 漂移")
        if (not isinstance(self.budget, TypedCarrierBudget)
                or self.budget.carrier_key != CARRIER_KEY
                or self.budget.max_cases != len(SAMPLE_KINDS)):
            raise TranscribedOcrAsrCarrierContractError("manifest budget 非法")
        if (not isinstance(self.case_keys, tuple)
                or len(self.case_keys) != len(SAMPLE_KINDS)
                or self.case_keys != tuple(sorted(set(self.case_keys)))):
            raise TranscribedOcrAsrCarrierContractError("manifest case_keys 非法")
        if self.sample_kinds != SAMPLE_KINDS:
            raise TranscribedOcrAsrCarrierContractError("manifest sample_kinds 漂移")
        if (not isinstance(self.materializations, tuple)
                or tuple(item.case_key for item in self.materializations)
                != self.case_keys):
            raise TranscribedOcrAsrCarrierContractError(
                "manifest materializations 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise TranscribedOcrAsrCarrierContractError(
                "manifest execution_state 必须全零")
        if (not isinstance(self.evidence_files, tuple)
                or self.evidence_files != tuple(sorted(
                    self.evidence_files,
                    key=lambda item: (item.relative_path, item.role)))
                or {item.role for item in self.evidence_files} != set(EVIDENCE_ROLES)
                or len({item.relative_path for item in self.evidence_files})
                != len(self.evidence_files)):
            raise TranscribedOcrAsrCarrierContractError("manifest evidence_files 未闭合")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "budget": self.budget.to_dict(),
            "carrier_key": self.carrier_key,
            "case_keys": [item.to_list() for item in self.case_keys],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "license_id": self.license_id,
            "materializations": [item.to_dict() for item in self.materializations],
            "parent_pack_relative_path": self.parent_pack_relative_path,
            "parent_pack_sha256": self.parent_pack_sha256,
            "parser_package": self.parser_package,
            "parser_version": self.parser_version,
            "sample_kinds": list(self.sample_kinds),
            "sample_relative_path": self.sample_relative_path,
            "sample_sha256": self.sample_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscribedOcrAsrCarrierPayloadManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version", "budget",
            "carrier_key", "case_keys", "evidence_files", "execution_state",
            "format_version", "license_id", "materializations",
            "parent_pack_relative_path", "parent_pack_sha256", "parser_package",
            "parser_version", "sample_kinds", "sample_relative_path",
            "sample_sha256",
        }, where="TranscribedOcrAsrCarrierPayloadManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise TranscribedOcrAsrCarrierContractError("manifest artifact_kind 非法")
        try:
            return cls(
                raw["format_version"], str(raw["artifact_version"]),
                str(raw["artifact_status"]), str(raw["parent_pack_relative_path"]),
                str(raw["parent_pack_sha256"]), str(raw["sample_relative_path"]),
                str(raw["sample_sha256"]), str(raw["carrier_key"]),
                str(raw["license_id"]), str(raw["parser_package"]),
                str(raw["parser_version"]), TypedCarrierBudget.from_dict(raw["budget"]),
                tuple(StableRecordKey.from_value(item, where="case_key")
                      for item in raw["case_keys"]),
                tuple(str(item) for item in raw["sample_kinds"]),
                tuple(TranscribedOcrAsrMaterializationIdentity.from_dict(item)
                      for item in raw["materializations"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(TranscribedOcrAsrCarrierEvidenceFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except TranscribedOcrAsrCarrierContractError:
            raise
        except Exception as error:
            raise TranscribedOcrAsrCarrierContractError(
                "manifest nested field 损坏") from error


def read_transcribed_ocr_asr_carrier_manifest(
        path: str | Path,
        ) -> TranscribedOcrAsrCarrierPayloadManifest:
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise TranscribedOcrAsrCarrierContractError("manifest newline 非法")
        manifest = TranscribedOcrAsrCarrierPayloadManifest.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except TranscribedOcrAsrCarrierContractError:
        raise
    except Exception as error:
        raise TranscribedOcrAsrCarrierContractError("manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise TranscribedOcrAsrCarrierContractError("manifest 不是 canonical 字节")
    return manifest


def write_transcribed_ocr_asr_carrier_manifest(
        manifest: TranscribedOcrAsrCarrierPayloadManifest,
        path: str | Path,
        ) -> Path:
    if not isinstance(manifest, TranscribedOcrAsrCarrierPayloadManifest):
        raise TranscribedOcrAsrCarrierContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise TranscribedOcrAsrCarrierContractError(
                "manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TranscribedOcrAsrCarrierContractError("manifest 无法写入") from error
    return target


def verify_transcribed_ocr_asr_carrier_files(
        manifest: TranscribedOcrAsrCarrierPayloadManifest,
        *, repository_root: str | Path,
        ) -> None:
    root = Path(repository_root).resolve()
    paths = [(manifest.sample_relative_path, manifest.sample_sha256, None)]
    paths.extend((item.relative_path, item.sha256, item.byte_count)
                 for item in manifest.evidence_files)
    for relative_path, sha256, byte_count in paths:
        target = (root / Path(*relative_path.split("/"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise TranscribedOcrAsrCarrierContractError("evidence 路径逃逸") from error
        if not target.is_file():
            raise TranscribedOcrAsrCarrierContractError("evidence 文件缺失")
        payload = target.read_bytes()
        if ((byte_count is not None and len(payload) != byte_count)
                or hashlib.sha256(payload).hexdigest() != sha256):
            raise TranscribedOcrAsrCarrierContractError("evidence 文件身份漂移")


__all__ = [
    "ADAPTER_OBLIGATIONS", "ARTIFACT_KIND", "ARTIFACT_STATUS",
    "ARTIFACT_VERSION", "CARRIER_KEY", "EMPTY_SHA256", "EVIDENCE_ROLES",
    "EXECUTION_STATE", "FORMAT_VERSION", "LICENSE_ID", "PARSER_PACKAGE",
    "PARSER_VERSION", "READ_ORDERS", "SOURCE_MODES", "TEMPORAL_STATES",
    "TranscriptSegment", "TranscribedOcrAsrCarrierContractError",
    "TranscribedOcrAsrCarrierEvidenceFile", "TranscribedOcrAsrCarrierPayloadManifest",
    "TranscribedOcrAsrCarrierRecord", "TranscribedOcrAsrMaterializationIdentity",
    "read_transcribed_ocr_asr_carrier_manifest",
    "read_transcribed_ocr_asr_carrier_records",
    "verify_transcribed_ocr_asr_carrier_files",
    "write_transcribed_ocr_asr_carrier_manifest",
    "write_transcribed_ocr_asr_carrier_records",
]
