"""LC-16 REFERENCE_LINK_EMBED 七类 payload 与 reference adapter evidence 合同。"""
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
ARTIFACT_KIND = "PH2_LC16_REFERENCE_LINK_EMBED_CARRIER_PAYLOAD"
ARTIFACT_VERSION = "LC16-REFERENCE_LINK_EMBED-CARRIER-20260731-A"
ARTIFACT_STATUS = "PAYLOAD_FROZEN_REFERENCE_ADAPTER_ONLY"
CARRIER_KEY = "REFERENCE_LINK_EMBED"
LICENSE_ID = "CC0-1.0"
PARSER_PACKAGE = "python-stdlib"
PARSER_VERSION = "json-reference-slot-v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CASE_KEYS = tuple(
    StableRecordKey((16616000, 1, 6, ordinal))
    for ordinal in range(1, len(SAMPLE_KINDS) + 1)
)
ADAPTER_OBLIGATIONS = {
    "POSITIVE": "LOCAL_LINK_AND_EMBED_ROUND_TRIP",
    "NEGATIVE": "RAW_SOURCE_AND_REFERENCE_FAILURE_PRESERVATION",
    "AMBIGUOUS": "UNRESOLVED_TARGET_WITHOUT_SEMANTIC_PRESELECTION",
    "UNKNOWN": "UNKNOWN_REFERENCE_KIND_PRESERVATION",
    "REVISION": "REFERENCE_BINDING_REVISION_MAPPING",
    "GENERATION": "EXACT_REFERENCE_LINK_EMBED_SERIALIZATION",
    "RETENTION": "REFERENCE_BINDING_RELOAD_IDENTITY",
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


class ReferenceLinkEmbedCarrierContractError(RuntimeError):
    """REFERENCE_LINK_EMBED payload、manifest 或 evidence 身份不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 字段不精确")
    return value


def _text(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        strip: bool = True,
        ) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 必须是文本")
    if strip and value and value.strip() != value:
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 含首尾空白")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if (len(text) != 64
            or any(item not in "0123456789abcdef" for item in text)):
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in text or path.as_posix() != text
            or ":" in path.parts[0]):
        raise ReferenceLinkEmbedCarrierContractError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _raw_identity(
        text: str,
        unit_count: Any,
        sha256: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> None:
    _text(text, where=f"{where} text", allow_empty=allow_empty, strip=False)
    if text:
        try:
            validate_unicode_scalars(tuple(ord(item) for item in text))
        except Exception as error:
            raise ReferenceLinkEmbedCarrierContractError(
                f"{where} 含非法 Unicode scalar") from error
    if _nonnegative(unit_count, where=f"{where} unit_count") != len(text):
        raise ReferenceLinkEmbedCarrierContractError(f"{where} unit_count 漂移")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _sha256(sha256, where=f"{where} utf8_sha256") != expected:
        raise ReferenceLinkEmbedCarrierContractError(f"{where} UTF-8 SHA-256 漂移")


@dataclass(frozen=True, order=True)
class ReferenceLinkEmbedCarrierRecord:
    """一个绑定 parent case 且不含语义 expected label 的 REFERENCE_LINK_EMBED payload。"""

    format_version: int
    case_key: StableRecordKey
    sample_kind: str
    adapter_obligation: str
    license_id: str
    raw_text: str
    raw_unit_count: int
    raw_utf8_sha256: str
    previous_text: str
    previous_unit_count: int
    previous_utf8_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ReferenceLinkEmbedCarrierContractError("record format_version 非法")
        if not isinstance(self.case_key, StableRecordKey):
            raise ReferenceLinkEmbedCarrierContractError("record case_key 非法")
        if self.case_key not in CASE_KEYS:
            raise ReferenceLinkEmbedCarrierContractError(
                "record case_key 不属于 REFERENCE_LINK_EMBED 七类 case")
        if self.sample_kind not in SAMPLE_KINDS:
            raise ReferenceLinkEmbedCarrierContractError("record sample_kind 未登记")
        expected_index = SAMPLE_KINDS.index(self.sample_kind)
        if self.case_key != CASE_KEYS[expected_index]:
            raise ReferenceLinkEmbedCarrierContractError("record case_key/sample_kind 错配")
        if self.adapter_obligation != ADAPTER_OBLIGATIONS[self.sample_kind]:
            raise ReferenceLinkEmbedCarrierContractError("record adapter obligation 漂移")
        if self.license_id != LICENSE_ID:
            raise ReferenceLinkEmbedCarrierContractError("REFERENCE_LINK_EMBED payload 必须为 CC0-1.0")
        _raw_identity(
            self.raw_text, self.raw_unit_count, self.raw_utf8_sha256,
            where="record raw")
        _raw_identity(
            self.previous_text,
            self.previous_unit_count,
            self.previous_utf8_sha256,
            where="record previous",
            allow_empty=True,
        )
        if self.sample_kind == "REVISION":
            if not self.previous_text or self.previous_text == self.raw_text:
                raise ReferenceLinkEmbedCarrierContractError(
                    "REVISION 必须携带不同的 previous reference payload")
        elif (self.previous_text or self.previous_unit_count
              or self.previous_utf8_sha256 != EMPTY_SHA256):
            raise ReferenceLinkEmbedCarrierContractError(
                "非 REVISION 不得携带 previous payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_obligation": self.adapter_obligation,
            "case_key": self.case_key.to_list(),
            "format_version": self.format_version,
            "license_id": self.license_id,
            "previous_text": self.previous_text,
            "previous_unit_count": self.previous_unit_count,
            "previous_utf8_sha256": self.previous_utf8_sha256,
            "raw_text": self.raw_text,
            "raw_unit_count": self.raw_unit_count,
            "raw_utf8_sha256": self.raw_utf8_sha256,
            "sample_kind": self.sample_kind,
        }

    def canonical_line(self) -> bytes:
        return canonical_json_line(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceLinkEmbedCarrierRecord":
        raw = _exact(value, {
            "adapter_obligation", "case_key", "format_version", "license_id",
            "previous_text", "previous_unit_count", "previous_utf8_sha256",
            "raw_text", "raw_unit_count", "raw_utf8_sha256", "sample_kind",
        }, where="ReferenceLinkEmbedCarrierRecord")
        try:
            case_key = StableRecordKey.from_value(
                raw["case_key"], where="record case_key")
        except Exception as error:
            raise ReferenceLinkEmbedCarrierContractError("record case_key 损坏") from error
        return cls(
            raw["format_version"],
            case_key,
            str(raw["sample_kind"]),
            str(raw["adapter_obligation"]),
            str(raw["license_id"]),
            str(raw["raw_text"]),
            raw["raw_unit_count"],
            str(raw["raw_utf8_sha256"]),
            str(raw["previous_text"]),
            raw["previous_unit_count"],
            str(raw["previous_utf8_sha256"]),
        )


def read_reference_link_embed_carrier_records(
        path: str | Path,
        ) -> tuple[ReferenceLinkEmbedCarrierRecord, ...]:
    """严格读取七行 canonical JSONL，不容忍 BOM、空行或尾随字节。"""
    try:
        payload = Path(path).read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise ReferenceLinkEmbedCarrierContractError("payload 缺少终止换行")
        lines = payload.splitlines(keepends=True)
        if any(not line.endswith(b"\n") or line == b"\n" for line in lines):
            raise ReferenceLinkEmbedCarrierContractError("payload 含坏行或空行")
        records = []
        for line in lines:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            if canonical_json_line(value) != line:
                raise ReferenceLinkEmbedCarrierContractError("payload 行不是 canonical JSON")
            records.append(ReferenceLinkEmbedCarrierRecord.from_dict(value))
    except ReferenceLinkEmbedCarrierContractError:
        raise
    except Exception as error:
        raise ReferenceLinkEmbedCarrierContractError("REFERENCE_LINK_EMBED payload 损坏") from error
    result = tuple(records)
    if len(result) != len(SAMPLE_KINDS):
        raise ReferenceLinkEmbedCarrierContractError("payload 必须精确包含七行")
    if tuple(item.sample_kind for item in result) != SAMPLE_KINDS:
        raise ReferenceLinkEmbedCarrierContractError("payload sample_kind 顺序或覆盖漂移")
    if tuple(item.case_key for item in result) != CASE_KEYS:
        raise ReferenceLinkEmbedCarrierContractError("payload REFERENCE_LINK_EMBED case_key 顺序或覆盖漂移")
    return result


def write_reference_link_embed_carrier_records(
        records: tuple[ReferenceLinkEmbedCarrierRecord, ...],
        path: str | Path,
        ) -> Path:
    """排他或幂等写七行 payload，禁止异内容覆盖。"""
    if (not isinstance(records, tuple)
            or any(not isinstance(item, ReferenceLinkEmbedCarrierRecord)
                   for item in records)
            or tuple(item.sample_kind for item in records) != SAMPLE_KINDS
            or tuple(item.case_key for item in records) != CASE_KEYS):
        raise ReferenceLinkEmbedCarrierContractError("records 类型、顺序或覆盖非法")
    payload = b"".join(item.canonical_line() for item in records)
    target = Path(path)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise ReferenceLinkEmbedCarrierContractError("payload 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise ReferenceLinkEmbedCarrierContractError("payload 无法写入") from error
    return target


@dataclass(frozen=True, order=True)
class ReferenceLinkEmbedMaterializationIdentity:
    """一个 reference case 的 adapter 输出摘要及对象计数。"""

    case_key: StableRecordKey
    byte_count: int
    sha256: str
    raw_unit_count: int
    previous_raw_unit_count: int
    envelope_count: int
    anchor_count: int
    structure_node_count: int
    reference_count: int
    revision_count: int
    revision_mapping_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise ReferenceLinkEmbedCarrierContractError("materialization case_key 非法")
        if self.case_key not in CASE_KEYS:
            raise ReferenceLinkEmbedCarrierContractError("materialization case_key 非 REFERENCE_LINK_EMBED case")
        _positive(self.byte_count, where="materialization byte_count")
        _sha256(self.sha256, where="materialization sha256")
        _positive(self.raw_unit_count, where="materialization raw_unit_count")
        _nonnegative(
            self.previous_raw_unit_count,
            where="materialization previous_raw_unit_count")
        for name in (
                "envelope_count", "anchor_count", "structure_node_count",
                "reference_count", "revision_count",
                "revision_mapping_count"):
            _nonnegative(getattr(self, name), where=f"materialization {name}")
        if self.anchor_count < self.envelope_count * 2:
            raise ReferenceLinkEmbedCarrierContractError(
                "materialization 未保留 text/document anchors")
        if self.structure_node_count <= 0:
            raise ReferenceLinkEmbedCarrierContractError(
                "materialization 缺少 reference structure nodes")
        if self.revision_count == 0:
            if (self.previous_raw_unit_count != 0 or self.envelope_count != 1
                    or self.revision_mapping_count != 0):
                raise ReferenceLinkEmbedCarrierContractError(
                    "普通 materialization 对象计数漂移")
        elif self.revision_count == 1:
            if (self.previous_raw_unit_count <= 0 or self.envelope_count != 2
                    or self.revision_mapping_count <= 0):
                raise ReferenceLinkEmbedCarrierContractError(
                    "revision materialization 对象计数漂移")
        else:
            raise ReferenceLinkEmbedCarrierContractError("revision_count 必须为 0/1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_count": self.anchor_count,
            "byte_count": self.byte_count,
            "case_key": self.case_key.to_list(),
            "envelope_count": self.envelope_count,
            "previous_raw_unit_count": self.previous_raw_unit_count,
            "raw_unit_count": self.raw_unit_count,
            "reference_count": self.reference_count,
            "revision_count": self.revision_count,
            "revision_mapping_count": self.revision_mapping_count,
            "sha256": self.sha256,
            "structure_node_count": self.structure_node_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceLinkEmbedMaterializationIdentity":
        raw = _exact(value, {
            "anchor_count", "byte_count", "case_key", "envelope_count",
            "previous_raw_unit_count", "raw_unit_count", "reference_count",
            "revision_count", "revision_mapping_count", "sha256",
            "structure_node_count",
        }, where="ReferenceLinkEmbedMaterializationIdentity")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            raw["byte_count"],
            str(raw["sha256"]),
            raw["raw_unit_count"],
            raw["previous_raw_unit_count"],
            raw["envelope_count"],
            raw["anchor_count"],
            raw["structure_node_count"],
            raw["reference_count"],
            raw["revision_count"],
            raw["revision_mapping_count"],
        )


@dataclass(frozen=True, order=True)
class ReferenceLinkEmbedCarrierEvidenceFile:
    """contract、adapter、catalog、dependency 和测试的文件身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise ReferenceLinkEmbedCarrierContractError("evidence role 未登记")
        _positive(self.byte_count, where="evidence byte_count")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceLinkEmbedCarrierEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="ReferenceLinkEmbedCarrierEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True, order=True)
class ReferenceLinkEmbedCarrierPayloadManifest:
    """绑定 parent、解析器、七行 payload、adapter 输出与零执行状态。"""

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
    materializations: tuple[ReferenceLinkEmbedMaterializationIdentity, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[ReferenceLinkEmbedCarrierEvidenceFile, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ReferenceLinkEmbedCarrierContractError("manifest format_version 非法")
        if (self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise ReferenceLinkEmbedCarrierContractError("manifest artifact identity 漂移")
        _relative_path(
            self.parent_pack_relative_path, where="parent_pack_relative_path")
        _sha256(self.parent_pack_sha256, where="parent_pack_sha256")
        _relative_path(self.sample_relative_path, where="sample_relative_path")
        _sha256(self.sample_sha256, where="sample_sha256")
        if self.carrier_key != CARRIER_KEY or self.license_id != LICENSE_ID:
            raise ReferenceLinkEmbedCarrierContractError("manifest carrier/license 非法")
        if (self.parser_package != PARSER_PACKAGE
                or self.parser_version != PARSER_VERSION):
            raise ReferenceLinkEmbedCarrierContractError("manifest parser identity 漂移")
        if (not isinstance(self.budget, TypedCarrierBudget)
                or self.budget.carrier_key != CARRIER_KEY
                or self.budget.max_cases != len(SAMPLE_KINDS)):
            raise ReferenceLinkEmbedCarrierContractError("manifest budget 非法")
        if self.case_keys != CASE_KEYS:
            raise ReferenceLinkEmbedCarrierContractError("manifest REFERENCE_LINK_EMBED case_keys 漂移")
        if self.sample_kinds != SAMPLE_KINDS:
            raise ReferenceLinkEmbedCarrierContractError("manifest sample_kinds 漂移")
        if (not isinstance(self.materializations, tuple)
                or any(not isinstance(item, ReferenceLinkEmbedMaterializationIdentity)
                       for item in self.materializations)
                or tuple(item.case_key for item in self.materializations)
                != self.case_keys):
            raise ReferenceLinkEmbedCarrierContractError("manifest materializations 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise ReferenceLinkEmbedCarrierContractError("manifest execution_state 必须全零")
        if (not isinstance(self.evidence_files, tuple)
                or any(not isinstance(item, ReferenceLinkEmbedCarrierEvidenceFile)
                       for item in self.evidence_files)
                or self.evidence_files != tuple(sorted(
                    self.evidence_files,
                    key=lambda item: (item.relative_path, item.role)))
                or {item.role for item in self.evidence_files}
                != set(EVIDENCE_ROLES)
                or len({item.relative_path for item in self.evidence_files})
                != len(self.evidence_files)):
            raise ReferenceLinkEmbedCarrierContractError("manifest evidence_files 未闭合")

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
            "materializations": [
                item.to_dict() for item in self.materializations],
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
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceLinkEmbedCarrierPayloadManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version", "budget",
            "carrier_key", "case_keys", "evidence_files", "execution_state",
            "format_version", "license_id", "materializations",
            "parent_pack_relative_path", "parent_pack_sha256", "parser_package",
            "parser_version", "sample_kinds", "sample_relative_path",
            "sample_sha256",
        }, where="ReferenceLinkEmbedCarrierPayloadManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise ReferenceLinkEmbedCarrierContractError("manifest artifact_kind 非法")
        try:
            return cls(
                raw["format_version"],
                str(raw["artifact_version"]),
                str(raw["artifact_status"]),
                str(raw["parent_pack_relative_path"]),
                str(raw["parent_pack_sha256"]),
                str(raw["sample_relative_path"]),
                str(raw["sample_sha256"]),
                str(raw["carrier_key"]),
                str(raw["license_id"]),
                str(raw["parser_package"]),
                str(raw["parser_version"]),
                TypedCarrierBudget.from_dict(raw["budget"]),
                tuple(StableRecordKey.from_value(item, where="case_key")
                      for item in raw["case_keys"]),
                tuple(str(item) for item in raw["sample_kinds"]),
                tuple(ReferenceLinkEmbedMaterializationIdentity.from_dict(item)
                      for item in raw["materializations"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(ReferenceLinkEmbedCarrierEvidenceFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except ReferenceLinkEmbedCarrierContractError:
            raise
        except Exception as error:
            raise ReferenceLinkEmbedCarrierContractError("manifest 嵌套字段损坏") from error


def read_reference_link_embed_carrier_manifest(
        path: str | Path,
        ) -> ReferenceLinkEmbedCarrierPayloadManifest:
    """严格回读 canonical REFERENCE_LINK_EMBED manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise ReferenceLinkEmbedCarrierContractError("manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = ReferenceLinkEmbedCarrierPayloadManifest.from_dict(value)
    except ReferenceLinkEmbedCarrierContractError:
        raise
    except Exception as error:
        raise ReferenceLinkEmbedCarrierContractError("manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise ReferenceLinkEmbedCarrierContractError("manifest 不是 canonical 字节")
    return manifest


def write_reference_link_embed_carrier_manifest(
        manifest: ReferenceLinkEmbedCarrierPayloadManifest,
        path: str | Path,
        ) -> Path:
    """排他或幂等写 manifest，不允许异内容覆盖。"""
    if not isinstance(manifest, ReferenceLinkEmbedCarrierPayloadManifest):
        raise ReferenceLinkEmbedCarrierContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise ReferenceLinkEmbedCarrierContractError("manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise ReferenceLinkEmbedCarrierContractError("manifest 无法写入") from error
    return target


def verify_reference_link_embed_carrier_files(
        manifest: ReferenceLinkEmbedCarrierPayloadManifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐字节回验 sample 和五类代码/依赖证据。"""
    root = Path(repository_root).resolve()
    paths = [(manifest.sample_relative_path, manifest.sample_sha256, None)]
    paths.extend((item.relative_path, item.sha256, item.byte_count)
                 for item in manifest.evidence_files)
    for relative_path, sha256, byte_count in paths:
        target = (root / Path(*relative_path.split("/"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ReferenceLinkEmbedCarrierContractError("evidence 路径逃逸") from error
        if not target.is_file():
            raise ReferenceLinkEmbedCarrierContractError("evidence 文件缺失")
        payload = target.read_bytes()
        if ((byte_count is not None and len(payload) != byte_count)
                or hashlib.sha256(payload).hexdigest() != sha256):
            raise ReferenceLinkEmbedCarrierContractError("evidence 文件身份漂移")


__all__ = [
    "ADAPTER_OBLIGATIONS",
    "ARTIFACT_KIND",
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "CASE_KEYS",
    "CARRIER_KEY",
    "EMPTY_SHA256",
    "EVIDENCE_ROLES",
    "EXECUTION_STATE",
    "FORMAT_VERSION",
    "LICENSE_ID",
    "PARSER_PACKAGE",
    "PARSER_VERSION",
    "ReferenceLinkEmbedCarrierContractError",
    "ReferenceLinkEmbedCarrierEvidenceFile",
    "ReferenceLinkEmbedCarrierPayloadManifest",
    "ReferenceLinkEmbedCarrierRecord",
    "ReferenceLinkEmbedMaterializationIdentity",
    "read_reference_link_embed_carrier_manifest",
    "read_reference_link_embed_carrier_records",
    "verify_reference_link_embed_carrier_files",
    "write_reference_link_embed_carrier_manifest",
    "write_reference_link_embed_carrier_records",
]
