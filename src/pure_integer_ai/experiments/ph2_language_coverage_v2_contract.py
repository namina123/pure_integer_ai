"""LC-COVERAGE-V2 的只追加能力、载体和方向覆盖基线合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 2
ARTIFACT_KIND = "PH2_LANGUAGE_CAPABILITY_COVERAGE_V2"
ARTIFACT_VERSION = "LC-COVERAGE-V2-20260731-A"
ARTIFACT_STATUS = "BASELINE_FROZEN"

TASK_KEYS = tuple(f"LC-{index:02d}" for index in range(1, 17))
CARRIER_KEYS = (
    "DOCUMENT_CONTAINER",
    "HTML",
    "MARKDOWN",
    "MATH_NOTATION",
    "PLAIN_TEXT",
    "REFERENCE_LINK_EMBED",
    "SENSORY_GROUNDING",
    "SOURCE_CODE",
    "TABLE_GRID",
    "TRANSCRIBED_OCR_ASR",
)
IN_SCOPE_CARRIER_KEYS = tuple(
    key for key in CARRIER_KEYS if key != "SENSORY_GROUNDING")
DIRECTIONS = ("GENERATION", "REASONING", "UNDERSTANDING")

TASK_BASELINE_STATES = ("AUDITED_ABSENT", "HISTORICAL_SCOPE_ONLY")
CARRIER_SCOPE_KINDS = ("IN_SCOPE", "WALL")
AUDIT_STATES = ("ABSENT", "PARTIAL", "REUSE", "WALL")
CELL_APPLICABILITY = ("REQUIRED", "WALL")
CELL_STATES = ("ABSENT", "WALL_BLOCKED")

EVIDENCE_ROLES = (
    "CONTRACT",
    "COVERAGE_BASE",
    "D03_GLOBAL",
    "D03_RECEIPT",
    "IMPLEMENTATION",
    "LINEAGE_HEAD",
    "TEST",
    "W03_RECEIPT",
)

W02_RECEIPT_SHA256 = (
    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb")

INVARIANTS = {
    "base_artifacts_preserved": 1,
    "carrier_qualified_passes": 0,
    "d03_v1_preserved": 1,
    "historical_receipts_not_extended": 1,
    "non_text_media_split_required": 1,
    "sensory_grounding_wall_preserved": 1,
    "w04_blocked": 1,
}

EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "companion_writes": 0,
    "formal_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}


class LanguageCoverageV2Error(RuntimeError):
    """LC-COVERAGE-V2 字段、身份或诚实状态不闭合。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LanguageCoverageV2Error(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise LanguageCoverageV2Error(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise LanguageCoverageV2Error(f"{where} 必须是小写 SHA-256")
    return text


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise LanguageCoverageV2Error(f"{where} 必须是非负严格整数")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LanguageCoverageV2Error(f"{where} 字段不精确")
    return value


def _texts(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise LanguageCoverageV2Error(f"{where} 必须是 tuple")
    result = tuple(_text(item, where=where) for item in value)
    if not allow_empty and not result:
        raise LanguageCoverageV2Error(f"{where} 不得为空")
    if tuple(sorted(set(result))) != result:
        raise LanguageCoverageV2Error(f"{where} 必须排序且去重")
    return result


@dataclass(frozen=True)
class CoverageV2EvidenceFile:
    """一个纳入 v2 基线的公开仓文件身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise LanguageCoverageV2Error("evidence role 未登记")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise LanguageCoverageV2Error("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoverageV2EvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="CoverageV2EvidenceFile")
        return cls(
            str(raw["relative_path"]), str(raw["role"]),
            raw["byte_count"], str(raw["sha256"]),
        )


@dataclass(frozen=True)
class LegacyCapabilitySplit:
    """只在 v2 中解释旧 NON_TEXT_MEDIA 的墙内/墙外拆分。"""

    legacy_capability_key: str
    in_scope_capability_key: str
    wall_capability_key: str
    migration_state: str

    def __post_init__(self) -> None:
        if self.legacy_capability_key != "NON_TEXT_MEDIA":
            raise LanguageCoverageV2Error("legacy capability key 漂移")
        if self.in_scope_capability_key != "TYPED_ARTIFACT_CARRIERS":
            raise LanguageCoverageV2Error("墙内 capability key 漂移")
        if self.wall_capability_key != "SENSORY_GROUNDING":
            raise LanguageCoverageV2Error("墙外 capability key 漂移")
        if self.migration_state != "DEPRECATED_SPLIT_ONLY":
            raise LanguageCoverageV2Error("legacy split 不得改写历史基线")

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_scope_capability_key": self.in_scope_capability_key,
            "legacy_capability_key": self.legacy_capability_key,
            "migration_state": self.migration_state,
            "wall_capability_key": self.wall_capability_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LegacyCapabilitySplit":
        raw = _exact(value, {
            "in_scope_capability_key", "legacy_capability_key",
            "migration_state", "wall_capability_key",
        }, where="LegacyCapabilitySplit")
        return cls(
            str(raw["legacy_capability_key"]),
            str(raw["in_scope_capability_key"]),
            str(raw["wall_capability_key"]),
            str(raw["migration_state"]),
        )


@dataclass(frozen=True)
class CoverageV2TaskRecord:
    """一个 LC 任务在 carrier-qualified v2 前沿中的诚实基线。"""

    task_key: str
    baseline_state: str
    historical_scope_authority: int
    carrier_qualified_runtime_authority: int
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_key not in TASK_KEYS:
            raise LanguageCoverageV2Error("task_key 未登记")
        if self.baseline_state not in TASK_BASELINE_STATES:
            raise LanguageCoverageV2Error("task baseline_state 非法")
        if type(self.historical_scope_authority) is not int or (
                self.historical_scope_authority not in {0, 1}):
            raise LanguageCoverageV2Error("historical_scope_authority 非法")
        if type(self.carrier_qualified_runtime_authority) is not int or (
                self.carrier_qualified_runtime_authority not in {0, 1}):
            raise LanguageCoverageV2Error(
                "carrier_qualified_runtime_authority 非法")
        if self.task_key == "LC-16":
            if (self.baseline_state != "AUDITED_ABSENT"
                    or self.historical_scope_authority != 0):
                raise LanguageCoverageV2Error("LC-16 必须保持审计后缺失")
        elif (self.baseline_state != "HISTORICAL_SCOPE_ONLY"
              or self.historical_scope_authority != 1):
            raise LanguageCoverageV2Error("LC-01..15 只能继承历史范围")
        if self.carrier_qualified_runtime_authority != 0:
            raise LanguageCoverageV2Error("v2 基线不得签发 carrier runtime")
        object.__setattr__(self, "evidence_refs", _texts(
            self.evidence_refs, where="task evidence_refs"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_state": self.baseline_state,
            "carrier_qualified_runtime_authority": (
                self.carrier_qualified_runtime_authority),
            "evidence_refs": list(self.evidence_refs),
            "historical_scope_authority": self.historical_scope_authority,
            "task_key": self.task_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoverageV2TaskRecord":
        raw = _exact(value, {
            "baseline_state", "carrier_qualified_runtime_authority",
            "evidence_refs", "historical_scope_authority", "task_key",
        }, where="CoverageV2TaskRecord")
        return cls(
            str(raw["task_key"]), str(raw["baseline_state"]),
            raw["historical_scope_authority"],
            raw["carrier_qualified_runtime_authority"],
            tuple(str(item) for item in raw["evidence_refs"]),
        )


@dataclass(frozen=True)
class CoverageV2CarrierRecord:
    """一个 carrier 对 raw、结构、revision、投影和 consumer 的装载审计。"""

    carrier_key: str
    scope_kind: str
    raw_state: str
    structure_state: str
    revision_state: str
    projection_state: str
    consumer_state: str
    evidence_refs: tuple[str, ...]
    gap_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.carrier_key not in CARRIER_KEYS:
            raise LanguageCoverageV2Error("carrier_key 未登记")
        if self.scope_kind not in CARRIER_SCOPE_KINDS:
            raise LanguageCoverageV2Error("carrier scope_kind 非法")
        for name in (
                "raw_state", "structure_state", "revision_state",
                "projection_state", "consumer_state"):
            if getattr(self, name) not in AUDIT_STATES:
                raise LanguageCoverageV2Error(f"carrier {name} 非法")
        states = (
            self.raw_state, self.structure_state, self.revision_state,
            self.projection_state, self.consumer_state,
        )
        if self.carrier_key == "SENSORY_GROUNDING":
            if self.scope_kind != "WALL" or set(states) != {"WALL"}:
                raise LanguageCoverageV2Error("感知 grounding 必须保持 WALL")
        elif self.scope_kind != "IN_SCOPE" or "WALL" in states:
            raise LanguageCoverageV2Error("typed carrier 不得藏入 WALL")
        object.__setattr__(self, "evidence_refs", _texts(
            self.evidence_refs, where="carrier evidence_refs"))
        object.__setattr__(self, "gap_reasons", _texts(
            self.gap_reasons, where="carrier gap_reasons"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "consumer_state": self.consumer_state,
            "evidence_refs": list(self.evidence_refs),
            "gap_reasons": list(self.gap_reasons),
            "projection_state": self.projection_state,
            "raw_state": self.raw_state,
            "revision_state": self.revision_state,
            "scope_kind": self.scope_kind,
            "structure_state": self.structure_state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoverageV2CarrierRecord":
        raw = _exact(value, {
            "carrier_key", "consumer_state", "evidence_refs", "gap_reasons",
            "projection_state", "raw_state", "revision_state",
            "scope_kind", "structure_state",
        }, where="CoverageV2CarrierRecord")
        return cls(
            str(raw["carrier_key"]), str(raw["scope_kind"]),
            str(raw["raw_state"]), str(raw["structure_state"]),
            str(raw["revision_state"]), str(raw["projection_state"]),
            str(raw["consumer_state"]),
            tuple(str(item) for item in raw["evidence_refs"]),
            tuple(str(item) for item in raw["gap_reasons"]),
        )


@dataclass(frozen=True)
class CoverageV2Cell:
    """一个 LC task、carrier 和方向的 fail-closed 覆盖单元。"""

    task_key: str
    carrier_key: str
    direction: str
    applicability: str
    observation_state: str
    representation_state: str
    adapter_state: str
    projection_state: str
    consumer_state: str
    verifier_state: str
    retention_state: str
    coverage_state: str
    evidence_refs: tuple[str, ...]
    ne_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_key not in TASK_KEYS:
            raise LanguageCoverageV2Error("cell task_key 未登记")
        if self.carrier_key not in CARRIER_KEYS:
            raise LanguageCoverageV2Error("cell carrier_key 未登记")
        if self.direction not in DIRECTIONS:
            raise LanguageCoverageV2Error("cell direction 未登记")
        if self.applicability not in CELL_APPLICABILITY:
            raise LanguageCoverageV2Error("cell applicability 非法")
        state_names = (
            "observation_state", "representation_state", "adapter_state",
            "projection_state", "consumer_state", "verifier_state",
            "retention_state", "coverage_state",
        )
        states = tuple(getattr(self, name) for name in state_names)
        if any(state not in CELL_STATES for state in states):
            raise LanguageCoverageV2Error("cell state 非法")
        if self.carrier_key == "SENSORY_GROUNDING":
            if (self.applicability != "WALL"
                    or set(states) != {"WALL_BLOCKED"}):
                raise LanguageCoverageV2Error("感知 cell 必须全部 WALL_BLOCKED")
        elif (self.applicability != "REQUIRED"
              or set(states) != {"ABSENT"}):
            raise LanguageCoverageV2Error("typed carrier v2 基线必须显式 ABSENT")
        object.__setattr__(self, "evidence_refs", _texts(
            self.evidence_refs, where="cell evidence_refs"))
        object.__setattr__(self, "ne_reasons", _texts(
            self.ne_reasons, where="cell ne_reasons"))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.task_key, self.carrier_key, self.direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_state": self.adapter_state,
            "applicability": self.applicability,
            "carrier_key": self.carrier_key,
            "consumer_state": self.consumer_state,
            "coverage_state": self.coverage_state,
            "direction": self.direction,
            "evidence_refs": list(self.evidence_refs),
            "ne_reasons": list(self.ne_reasons),
            "observation_state": self.observation_state,
            "projection_state": self.projection_state,
            "representation_state": self.representation_state,
            "retention_state": self.retention_state,
            "task_key": self.task_key,
            "verifier_state": self.verifier_state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoverageV2Cell":
        raw = _exact(value, {
            "adapter_state", "applicability", "carrier_key",
            "consumer_state", "coverage_state", "direction",
            "evidence_refs", "ne_reasons", "observation_state",
            "projection_state", "representation_state", "retention_state",
            "task_key", "verifier_state",
        }, where="CoverageV2Cell")
        return cls(
            str(raw["task_key"]), str(raw["carrier_key"]),
            str(raw["direction"]), str(raw["applicability"]),
            str(raw["observation_state"]),
            str(raw["representation_state"]), str(raw["adapter_state"]),
            str(raw["projection_state"]), str(raw["consumer_state"]),
            str(raw["verifier_state"]), str(raw["retention_state"]),
            str(raw["coverage_state"]),
            tuple(str(item) for item in raw["evidence_refs"]),
            tuple(str(item) for item in raw["ne_reasons"]),
        )


@dataclass(frozen=True)
class LanguageCapabilityCoverageV2Manifest:
    """绑定历史全账与链头、但不扩张旧 receipt 权限的 v2 baseline。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    w02_receipt_sha256: str
    evidence_files: tuple[CoverageV2EvidenceFile, ...]
    legacy_split: LegacyCapabilitySplit
    task_records: tuple[CoverageV2TaskRecord, ...]
    carrier_records: tuple[CoverageV2CarrierRecord, ...]
    cells: tuple[CoverageV2Cell, ...]
    invariants: CanonicalJsonObject
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise LanguageCoverageV2Error("format_version 非法")
        if self.artifact_kind != ARTIFACT_KIND:
            raise LanguageCoverageV2Error("artifact_kind 漂移")
        if self.artifact_version != ARTIFACT_VERSION:
            raise LanguageCoverageV2Error("artifact_version 漂移")
        if self.artifact_status != ARTIFACT_STATUS:
            raise LanguageCoverageV2Error("artifact_status 漂移")
        if self.w02_receipt_sha256 != W02_RECEIPT_SHA256:
            raise LanguageCoverageV2Error("W-02 receipt commitment 漂移")
        if not isinstance(self.legacy_split, LegacyCapabilitySplit):
            raise LanguageCoverageV2Error("legacy_split 类型非法")

        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or not all(isinstance(item, CoverageV2EvidenceFile)
                           for item in self.evidence_files)):
            raise LanguageCoverageV2Error("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.relative_path, item.role),
        ))
        object.__setattr__(self, "evidence_files", evidence)
        paths = tuple(item.relative_path for item in evidence)
        if len(paths) != len(set(paths)):
            raise LanguageCoverageV2Error("evidence path 重复")
        roles = {item.role for item in evidence}
        required_roles = {
            "CONTRACT", "COVERAGE_BASE", "D03_GLOBAL", "D03_RECEIPT",
            "IMPLEMENTATION", "LINEAGE_HEAD", "TEST", "W03_RECEIPT",
        }
        if roles != required_roles:
            raise LanguageCoverageV2Error("evidence role 未精确闭合")

        if (not isinstance(self.task_records, tuple)
                or not all(isinstance(item, CoverageV2TaskRecord)
                           for item in self.task_records)):
            raise LanguageCoverageV2Error("task_records 非法")
        tasks = tuple(sorted(self.task_records, key=lambda item: item.task_key))
        object.__setattr__(self, "task_records", tasks)
        if tuple(item.task_key for item in tasks) != TASK_KEYS:
            raise LanguageCoverageV2Error("LC-01..16 必须逐项列全")

        if (not isinstance(self.carrier_records, tuple)
                or not all(isinstance(item, CoverageV2CarrierRecord)
                           for item in self.carrier_records)):
            raise LanguageCoverageV2Error("carrier_records 非法")
        carriers = tuple(sorted(
            self.carrier_records, key=lambda item: item.carrier_key))
        object.__setattr__(self, "carrier_records", carriers)
        if tuple(item.carrier_key for item in carriers) != CARRIER_KEYS:
            raise LanguageCoverageV2Error("carrier 必须逐项列全")

        if (not isinstance(self.cells, tuple)
                or not all(isinstance(item, CoverageV2Cell)
                           for item in self.cells)):
            raise LanguageCoverageV2Error("cells 非法")
        cells = tuple(sorted(self.cells, key=lambda item: item.key))
        object.__setattr__(self, "cells", cells)
        expected = tuple(
            (task, carrier, direction)
            for task in TASK_KEYS
            for carrier in CARRIER_KEYS
            for direction in DIRECTIONS
        )
        if tuple(item.key for item in cells) != expected:
            raise LanguageCoverageV2Error(
                "task×carrier×direction 单元必须精确列全")

        if (not isinstance(self.invariants, CanonicalJsonObject)
                or self.invariants.to_value() != INVARIANTS):
            raise LanguageCoverageV2Error("invariants 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise LanguageCoverageV2Error("execution_state 必须保持冻结状态")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "carrier_records": [item.to_dict() for item in self.carrier_records],
            "cells": [item.to_dict() for item in self.cells],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "invariants": self.invariants.to_value(),
            "legacy_split": self.legacy_split.to_dict(),
            "task_records": [item.to_dict() for item in self.task_records],
            "w02_receipt_sha256": self.w02_receipt_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "LanguageCapabilityCoverageV2Manifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "carrier_records", "cells", "evidence_files", "execution_state",
            "format_version", "invariants", "legacy_split", "task_records",
            "w02_receipt_sha256",
        }, where="LanguageCapabilityCoverageV2Manifest")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["artifact_status"]),
            str(raw["w02_receipt_sha256"]),
            tuple(CoverageV2EvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            LegacyCapabilitySplit.from_dict(raw["legacy_split"]),
            tuple(CoverageV2TaskRecord.from_dict(item)
                  for item in raw["task_records"]),
            tuple(CoverageV2CarrierRecord.from_dict(item)
                  for item in raw["carrier_records"]),
            tuple(CoverageV2Cell.from_dict(item) for item in raw["cells"]),
            CanonicalJsonObject.from_value(raw["invariants"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_language_capability_coverage_v2(
        path: str | Path,
        ) -> LanguageCapabilityCoverageV2Manifest:
    """严格回读 canonical LC-COVERAGE-V2。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise LanguageCoverageV2Error("v2 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = LanguageCapabilityCoverageV2Manifest.from_dict(value)
    except LanguageCoverageV2Error:
        raise
    except Exception as error:
        raise LanguageCoverageV2Error("v2 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise LanguageCoverageV2Error("v2 manifest 非规范字节")
    return manifest


def write_language_capability_coverage_v2(
        manifest: LanguageCapabilityCoverageV2Manifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 v2 baseline，禁止同版本异内容覆盖。"""
    if not isinstance(manifest, LanguageCapabilityCoverageV2Manifest):
        raise LanguageCoverageV2Error("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise LanguageCoverageV2Error("v2 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise LanguageCoverageV2Error("v2 manifest 无法写入") from error
    return target


def verify_language_capability_coverage_v2_files(
        manifest: LanguageCapabilityCoverageV2Manifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐字节回验 v2 baseline 绑定的全部公开仓证据。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise LanguageCoverageV2Error("evidence 路径逃逸") from error
        if not path.is_file():
            raise LanguageCoverageV2Error("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise LanguageCoverageV2Error("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "AUDIT_STATES",
    "CARRIER_KEYS",
    "CELL_APPLICABILITY",
    "CELL_STATES",
    "CoverageV2CarrierRecord",
    "CoverageV2Cell",
    "CoverageV2EvidenceFile",
    "CoverageV2TaskRecord",
    "DIRECTIONS",
    "EXECUTION_STATE",
    "FORMAT_VERSION",
    "INVARIANTS",
    "IN_SCOPE_CARRIER_KEYS",
    "LanguageCapabilityCoverageV2Manifest",
    "LanguageCoverageV2Error",
    "LegacyCapabilitySplit",
    "TASK_KEYS",
    "W02_RECEIPT_SHA256",
    "read_language_capability_coverage_v2",
    "verify_language_capability_coverage_v2_files",
    "write_language_capability_coverage_v2",
]
