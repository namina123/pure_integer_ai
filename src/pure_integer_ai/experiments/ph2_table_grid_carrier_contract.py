"""LC-16 TABLE_GRID 七类 payload 与 grid adapter evidence 合同。"""
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
ARTIFACT_KIND = "PH2_LC16_TABLE_GRID_CARRIER_PAYLOAD"
ARTIFACT_VERSION = "LC16-TABLE_GRID-CARRIER-20260731-A"
ARTIFACT_STATUS = "PAYLOAD_FROZEN_GRID_ADAPTER_ONLY"
CARRIER_KEY = "TABLE_GRID"
LICENSE_ID = "CC0-1.0"
PARSER_PACKAGE = "python-stdlib"
PARSER_VERSION = "csv-grid-v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ADAPTER_OBLIGATIONS = {
    "POSITIVE": "CELL_ROW_COLUMN_HEADER_ROUND_TRIP",
    "NEGATIVE": "RAW_SOURCE_AND_RAGGED_GRID_PRESERVATION",
    "AMBIGUOUS": "STRUCTURE_WITHOUT_SEMANTIC_PRESELECTION",
    "UNKNOWN": "UNKNOWN_LAYOUT_PRESERVATION",
    "REVISION": "GRID_REVISION_MAPPING",
    "GENERATION": "EXACT_TABLE_GRID_SERIALIZATION",
    "RETENTION": "GRID_RELOAD_IDENTITY",
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


class TableGridCarrierContractError(RuntimeError):
    """TABLE_GRID payload、manifest 或 evidence 身份不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TableGridCarrierContractError(f"{where} 字段不精确")
    return value


def _text(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        strip: bool = True,
        ) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TableGridCarrierContractError(f"{where} 必须是文本")
    if strip and value and value.strip() != value:
        raise TableGridCarrierContractError(f"{where} 含首尾空白")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise TableGridCarrierContractError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise TableGridCarrierContractError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if (len(text) != 64
            or any(item not in "0123456789abcdef" for item in text)):
        raise TableGridCarrierContractError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in text or path.as_posix() != text
            or ":" in path.parts[0]):
        raise TableGridCarrierContractError(f"{where} 必须是安全 POSIX 相对路径")
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
            raise TableGridCarrierContractError(
                f"{where} 含非法 Unicode scalar") from error
    if _nonnegative(unit_count, where=f"{where} unit_count") != len(text):
        raise TableGridCarrierContractError(f"{where} unit_count 漂移")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _sha256(sha256, where=f"{where} utf8_sha256") != expected:
        raise TableGridCarrierContractError(f"{where} UTF-8 SHA-256 漂移")


@dataclass(frozen=True, order=True)
class TableGridCarrierRecord:
    """一个绑定 parent case 且不含语义 expected label 的 TableGrid payload。"""

    format_version: int
    case_key: StableRecordKey
    sample_kind: str
    adapter_obligation: str
    license_id: str
    delimiter: str
    header_rows: tuple[int, ...]
    header_columns: tuple[int, ...]
    merged_rectangles: tuple[tuple[int, int, int, int], ...]
    read_order: str
    raw_text: str
    raw_unit_count: int
    raw_utf8_sha256: str
    previous_text: str
    previous_unit_count: int
    previous_utf8_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise TableGridCarrierContractError("record format_version 非法")
        if not isinstance(self.case_key, StableRecordKey):
            raise TableGridCarrierContractError("record case_key 非法")
        if self.sample_kind not in SAMPLE_KINDS:
            raise TableGridCarrierContractError("record sample_kind 未登记")
        if self.adapter_obligation != ADAPTER_OBLIGATIONS[self.sample_kind]:
            raise TableGridCarrierContractError("record adapter obligation 漂移")
        if self.license_id != LICENSE_ID:
            raise TableGridCarrierContractError("TABLE_GRID payload 必须为 CC0-1.0")
        if (not isinstance(self.delimiter, str) or len(self.delimiter) != 1
                or self.delimiter in {"\r", "\n", '"'}):
            raise TableGridCarrierContractError("record delimiter 非法")
        for name in ("header_rows", "header_columns"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(type(item) is not int or item < 0 for item in values)
                    or values != tuple(sorted(set(values)))):
                raise TableGridCarrierContractError(f"record {name} 非法")
        if (not isinstance(self.merged_rectangles, tuple)
                or any(not isinstance(item, tuple) or len(item) != 4
                       or any(type(value) is not int or value < 0
                              for value in item)
                       or item[0] > item[1] or item[2] > item[3]
                       for item in self.merged_rectangles)
                or self.merged_rectangles != tuple(sorted(
                    set(self.merged_rectangles)))):
            raise TableGridCarrierContractError("record merged_rectangles 非法")
        if self.read_order not in {"COLUMN_MAJOR", "ROW_MAJOR"}:
            raise TableGridCarrierContractError("record read_order 非法")
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
                raise TableGridCarrierContractError(
                    "REVISION 必须携带不同的 previous TableGrid")
        elif (self.previous_text or self.previous_unit_count
              or self.previous_utf8_sha256 != EMPTY_SHA256):
            raise TableGridCarrierContractError(
                "非 REVISION 不得携带 previous payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_obligation": self.adapter_obligation,
            "case_key": self.case_key.to_list(),
            "delimiter": self.delimiter,
            "format_version": self.format_version,
            "header_columns": list(self.header_columns),
            "header_rows": list(self.header_rows),
            "license_id": self.license_id,
            "merged_rectangles": [list(item) for item in self.merged_rectangles],
            "previous_text": self.previous_text,
            "previous_unit_count": self.previous_unit_count,
            "previous_utf8_sha256": self.previous_utf8_sha256,
            "raw_text": self.raw_text,
            "raw_unit_count": self.raw_unit_count,
            "raw_utf8_sha256": self.raw_utf8_sha256,
            "read_order": self.read_order,
            "sample_kind": self.sample_kind,
        }

    def canonical_line(self) -> bytes:
        return canonical_json_line(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TableGridCarrierRecord":
        raw = _exact(value, {
            "adapter_obligation", "case_key", "delimiter", "format_version",
            "header_columns", "header_rows", "license_id",
            "merged_rectangles",
            "previous_text", "previous_unit_count", "previous_utf8_sha256",
            "raw_text", "raw_unit_count", "raw_utf8_sha256", "read_order",
            "sample_kind",
        }, where="TableGridCarrierRecord")
        try:
            case_key = StableRecordKey.from_value(
                raw["case_key"], where="record case_key")
        except Exception as error:
            raise TableGridCarrierContractError("record case_key 损坏") from error
        return cls(
            raw["format_version"],
            case_key,
            str(raw["sample_kind"]),
            str(raw["adapter_obligation"]),
            str(raw["license_id"]),
            str(raw["delimiter"]),
            tuple(raw["header_rows"]),
            tuple(raw["header_columns"]),
            tuple(tuple(item) for item in raw["merged_rectangles"]),
            str(raw["read_order"]),
            str(raw["raw_text"]),
            raw["raw_unit_count"],
            str(raw["raw_utf8_sha256"]),
            str(raw["previous_text"]),
            raw["previous_unit_count"],
            str(raw["previous_utf8_sha256"]),
        )


def read_table_grid_carrier_records(
        path: str | Path,
        ) -> tuple[TableGridCarrierRecord, ...]:
    """严格读取七行 canonical JSONL，不容忍 BOM、空行或尾随字节。"""
    try:
        payload = Path(path).read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise TableGridCarrierContractError("payload 缺少终止换行")
        lines = payload.splitlines(keepends=True)
        if any(not line.endswith(b"\n") or line == b"\n" for line in lines):
            raise TableGridCarrierContractError("payload 含坏行或空行")
        records = []
        for line in lines:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            if canonical_json_line(value) != line:
                raise TableGridCarrierContractError("payload 行不是 canonical JSON")
            records.append(TableGridCarrierRecord.from_dict(value))
    except TableGridCarrierContractError:
        raise
    except Exception as error:
        raise TableGridCarrierContractError("TABLE_GRID payload 损坏") from error
    result = tuple(records)
    if len(result) != len(SAMPLE_KINDS):
        raise TableGridCarrierContractError("payload 必须精确包含七行")
    if tuple(item.sample_kind for item in result) != SAMPLE_KINDS:
        raise TableGridCarrierContractError("payload sample_kind 顺序或覆盖漂移")
    if result != tuple(sorted(result, key=lambda item: item.case_key)):
        raise TableGridCarrierContractError("payload 必须按 case_key 排序")
    if len({item.case_key for item in result}) != len(result):
        raise TableGridCarrierContractError("payload case_key 不得重复")
    return result


def write_table_grid_carrier_records(
        records: tuple[TableGridCarrierRecord, ...],
        path: str | Path,
        ) -> Path:
    """排他或幂等写七行 payload，禁止异内容覆盖。"""
    if (not isinstance(records, tuple)
            or any(not isinstance(item, TableGridCarrierRecord)
                   for item in records)):
        raise TableGridCarrierContractError("records 类型非法")
    payload = b"".join(item.canonical_line() for item in records)
    target = Path(path)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TableGridCarrierContractError("payload 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TableGridCarrierContractError("payload 无法写入") from error
    return target


@dataclass(frozen=True, order=True)
class TableGridMaterializationIdentity:
    """一个 TableGrid case 的 adapter 输出摘要及对象计数。"""

    case_key: StableRecordKey
    byte_count: int
    sha256: str
    raw_unit_count: int
    previous_raw_unit_count: int
    envelope_count: int
    anchor_count: int
    structure_node_count: int
    revision_count: int
    revision_mapping_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, StableRecordKey):
            raise TableGridCarrierContractError("materialization case_key 非法")
        _positive(self.byte_count, where="materialization byte_count")
        _sha256(self.sha256, where="materialization sha256")
        _positive(self.raw_unit_count, where="materialization raw_unit_count")
        _nonnegative(
            self.previous_raw_unit_count,
            where="materialization previous_raw_unit_count")
        for name in (
                "envelope_count", "anchor_count", "structure_node_count",
                "revision_count", "revision_mapping_count"):
            _nonnegative(getattr(self, name), where=f"materialization {name}")
        if self.anchor_count <= self.envelope_count * 2:
            raise TableGridCarrierContractError(
                "materialization 未保留 grid anchors")
        if self.structure_node_count <= 0:
            raise TableGridCarrierContractError(
                "materialization 缺少 structure nodes")
        if self.revision_count == 0:
            if (self.previous_raw_unit_count != 0 or self.envelope_count != 1
                    or self.revision_mapping_count != 0):
                raise TableGridCarrierContractError(
                    "普通 materialization 对象计数漂移")
        elif self.revision_count == 1:
            if (self.previous_raw_unit_count <= 0 or self.envelope_count != 2
                    or self.revision_mapping_count <= 0):
                raise TableGridCarrierContractError(
                    "revision materialization 对象计数漂移")
        else:
            raise TableGridCarrierContractError("revision_count 必须为 0/1")

    def to_dict(self) -> dict[str, Any]:
        return {
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
    def from_dict(cls, value: dict[str, Any]) -> "TableGridMaterializationIdentity":
        raw = _exact(value, {
            "anchor_count", "byte_count", "case_key", "envelope_count",
            "previous_raw_unit_count", "raw_unit_count", "revision_count",
            "revision_mapping_count", "sha256", "structure_node_count",
        }, where="TableGridMaterializationIdentity")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            raw["byte_count"],
            str(raw["sha256"]),
            raw["raw_unit_count"],
            raw["previous_raw_unit_count"],
            raw["envelope_count"],
            raw["anchor_count"],
            raw["structure_node_count"],
            raw["revision_count"],
            raw["revision_mapping_count"],
        )


@dataclass(frozen=True, order=True)
class TableGridCarrierEvidenceFile:
    """contract、adapter、catalog、dependency 和测试的文件身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise TableGridCarrierContractError("evidence role 未登记")
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
    def from_dict(cls, value: dict[str, Any]) -> "TableGridCarrierEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="TableGridCarrierEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True, order=True)
class TableGridCarrierPayloadManifest:
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
    materializations: tuple[TableGridMaterializationIdentity, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[TableGridCarrierEvidenceFile, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise TableGridCarrierContractError("manifest format_version 非法")
        if (self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise TableGridCarrierContractError("manifest artifact identity 漂移")
        _relative_path(
            self.parent_pack_relative_path, where="parent_pack_relative_path")
        _sha256(self.parent_pack_sha256, where="parent_pack_sha256")
        _relative_path(self.sample_relative_path, where="sample_relative_path")
        _sha256(self.sample_sha256, where="sample_sha256")
        if self.carrier_key != CARRIER_KEY or self.license_id != LICENSE_ID:
            raise TableGridCarrierContractError("manifest carrier/license 非法")
        if (self.parser_package != PARSER_PACKAGE
                or self.parser_version != PARSER_VERSION):
            raise TableGridCarrierContractError("manifest parser identity 漂移")
        if (not isinstance(self.budget, TypedCarrierBudget)
                or self.budget.carrier_key != CARRIER_KEY
                or self.budget.max_cases != len(SAMPLE_KINDS)):
            raise TableGridCarrierContractError("manifest budget 非法")
        if (not isinstance(self.case_keys, tuple)
                or len(self.case_keys) != len(SAMPLE_KINDS)
                or any(not isinstance(item, StableRecordKey)
                       for item in self.case_keys)
                or self.case_keys != tuple(sorted(set(self.case_keys)))):
            raise TableGridCarrierContractError("manifest case_keys 非法")
        if self.sample_kinds != SAMPLE_KINDS:
            raise TableGridCarrierContractError("manifest sample_kinds 漂移")
        if (not isinstance(self.materializations, tuple)
                or any(not isinstance(item, TableGridMaterializationIdentity)
                       for item in self.materializations)
                or tuple(item.case_key for item in self.materializations)
                != self.case_keys):
            raise TableGridCarrierContractError("manifest materializations 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise TableGridCarrierContractError("manifest execution_state 必须全零")
        if (not isinstance(self.evidence_files, tuple)
                or any(not isinstance(item, TableGridCarrierEvidenceFile)
                       for item in self.evidence_files)
                or self.evidence_files != tuple(sorted(
                    self.evidence_files,
                    key=lambda item: (item.relative_path, item.role)))
                or {item.role for item in self.evidence_files}
                != set(EVIDENCE_ROLES)
                or len({item.relative_path for item in self.evidence_files})
                != len(self.evidence_files)):
            raise TableGridCarrierContractError("manifest evidence_files 未闭合")

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
    def from_dict(cls, value: dict[str, Any]) -> "TableGridCarrierPayloadManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version", "budget",
            "carrier_key", "case_keys", "evidence_files", "execution_state",
            "format_version", "license_id", "materializations",
            "parent_pack_relative_path", "parent_pack_sha256", "parser_package",
            "parser_version", "sample_kinds", "sample_relative_path",
            "sample_sha256",
        }, where="TableGridCarrierPayloadManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise TableGridCarrierContractError("manifest artifact_kind 非法")
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
                tuple(TableGridMaterializationIdentity.from_dict(item)
                      for item in raw["materializations"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(TableGridCarrierEvidenceFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except TableGridCarrierContractError:
            raise
        except Exception as error:
            raise TableGridCarrierContractError("manifest 嵌套字段损坏") from error


def read_table_grid_carrier_manifest(
        path: str | Path,
        ) -> TableGridCarrierPayloadManifest:
    """严格回读 canonical TABLE_GRID manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise TableGridCarrierContractError("manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = TableGridCarrierPayloadManifest.from_dict(value)
    except TableGridCarrierContractError:
        raise
    except Exception as error:
        raise TableGridCarrierContractError("manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise TableGridCarrierContractError("manifest 不是 canonical 字节")
    return manifest


def write_table_grid_carrier_manifest(
        manifest: TableGridCarrierPayloadManifest,
        path: str | Path,
        ) -> Path:
    """排他或幂等写 manifest，不允许异内容覆盖。"""
    if not isinstance(manifest, TableGridCarrierPayloadManifest):
        raise TableGridCarrierContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TableGridCarrierContractError("manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TableGridCarrierContractError("manifest 无法写入") from error
    return target


def verify_table_grid_carrier_files(
        manifest: TableGridCarrierPayloadManifest,
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
            raise TableGridCarrierContractError("evidence 路径逃逸") from error
        if not target.is_file():
            raise TableGridCarrierContractError("evidence 文件缺失")
        payload = target.read_bytes()
        if ((byte_count is not None and len(payload) != byte_count)
                or hashlib.sha256(payload).hexdigest() != sha256):
            raise TableGridCarrierContractError("evidence 文件身份漂移")


__all__ = [
    "ADAPTER_OBLIGATIONS",
    "ARTIFACT_KIND",
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "CARRIER_KEY",
    "EMPTY_SHA256",
    "EVIDENCE_ROLES",
    "EXECUTION_STATE",
    "FORMAT_VERSION",
    "LICENSE_ID",
    "PARSER_PACKAGE",
    "PARSER_VERSION",
    "TableGridCarrierContractError",
    "TableGridCarrierEvidenceFile",
    "TableGridCarrierPayloadManifest",
    "TableGridCarrierRecord",
    "TableGridMaterializationIdentity",
    "read_table_grid_carrier_manifest",
    "read_table_grid_carrier_records",
    "verify_table_grid_carrier_files",
    "write_table_grid_carrier_manifest",
    "write_table_grid_carrier_records",
]
