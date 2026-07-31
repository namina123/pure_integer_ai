"""LC-16 跨载体结构特征 mapper 的公开冻结合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    IN_SCOPE_CARRIER_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_LC16_CARRIER_PROJECTION_MAPPER"
ARTIFACT_VERSION = "LC16-CARRIER-PROJECTION-MAPPER-20260731-A"
ARTIFACT_STATUS = "MAPPER_CONTRACT_FROZEN_NO_CAPABILITY_PASS"
INPUT_KINDS = ("ANCHOR", "STRUCTURE_NODE")
EVIDENCE_ROLES = ("CATALOG", "CONTRACT", "MAPPER", "TEST")
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


class CarrierProjectionMapperContractError(RuntimeError):
    """mapper 规则、依赖、manifest 或文件证据不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise CarrierProjectionMapperContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CarrierProjectionMapperContractError(
            f"{where} 必须是无首尾空白的非空文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise CarrierProjectionMapperContractError(
            f"{where} 必须是正严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise CarrierProjectionMapperContractError(
            f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in text or path.as_posix() != text
            or ":" in path.parts[0]):
        raise CarrierProjectionMapperContractError(
            f"{where} 必须是安全 POSIX 相对路径")
    return text


@dataclass(frozen=True, order=True)
class CarrierProjectionRule:
    """以数据路径匹配 carrier-local receipt 并映射到共享 feature key。"""

    rule_key: StableRecordKey
    carrier_key: str
    input_kind: str
    selector_paths: tuple[tuple[str, ...], ...]
    expected_values: CanonicalJsonObject
    feature_key: StableRecordKey

    def __post_init__(self) -> None:
        if not isinstance(self.rule_key, StableRecordKey):
            raise CarrierProjectionMapperContractError("rule_key 非法")
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise CarrierProjectionMapperContractError("carrier_key 未登记")
        if self.input_kind not in INPUT_KINDS:
            raise CarrierProjectionMapperContractError("input_kind 未登记")
        if (not isinstance(self.selector_paths, tuple)
                or not self.selector_paths
                or any(not isinstance(path, tuple) or not path
                       or any(not isinstance(item, str) or not item
                              or item.strip() != item for item in path)
                       for path in self.selector_paths)
                or self.selector_paths != tuple(dict.fromkeys(
                    self.selector_paths))):
            raise CarrierProjectionMapperContractError("selector_paths 非法")
        if not isinstance(self.expected_values, CanonicalJsonObject):
            raise CarrierProjectionMapperContractError("expected_values 非法")
        expected = self.expected_values.to_value()
        if set(expected) != {"values"} or not isinstance(expected["values"], list):
            raise CarrierProjectionMapperContractError(
                "expected_values 必须精确包装 values 列表")
        if len(expected["values"]) != len(self.selector_paths):
            raise CarrierProjectionMapperContractError(
                "expected_values 与 selector_paths 数量漂移")
        if not isinstance(self.feature_key, StableRecordKey):
            raise CarrierProjectionMapperContractError("feature_key 非法")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "expected_values": self.expected_values.to_value(),
            "feature_key": self.feature_key.to_list(),
            "input_kind": self.input_kind,
            "rule_key": self.rule_key.to_list(),
            "selector_paths": [list(item) for item in self.selector_paths],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CarrierProjectionRule":
        raw = _exact(value, {
            "carrier_key", "expected_values", "feature_key", "input_kind",
            "rule_key", "selector_paths",
        }, where="CarrierProjectionRule")
        try:
            return cls(
                StableRecordKey.from_value(raw["rule_key"], where="rule_key"),
                str(raw["carrier_key"]),
                str(raw["input_kind"]),
                tuple(tuple(path) for path in raw["selector_paths"]),
                CanonicalJsonObject.from_value(raw["expected_values"]),
                StableRecordKey.from_value(
                    raw["feature_key"], where="feature_key"),
            )
        except CarrierProjectionMapperContractError:
            raise
        except Exception as error:
            raise CarrierProjectionMapperContractError("rule nested field 损坏") from error


@dataclass(frozen=True, order=True)
class CarrierProjectionDependencyFile:
    carrier_key: str
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise CarrierProjectionMapperContractError("dependency carrier 未登记")
        _relative_path(self.relative_path, where="dependency relative_path")
        _positive(self.byte_count, where="dependency byte_count")
        _sha256(self.sha256, where="dependency sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "carrier_key": self.carrier_key,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CarrierProjectionDependencyFile":
        raw = _exact(value, {
            "byte_count", "carrier_key", "relative_path", "sha256",
        }, where="CarrierProjectionDependencyFile")
        return cls(str(raw["carrier_key"]), str(raw["relative_path"]),
                   raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class CarrierProjectionMapperEvidenceFile:
    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise CarrierProjectionMapperContractError("evidence role 未登记")
        _positive(self.byte_count, where="evidence byte_count")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"byte_count": self.byte_count, "relative_path": self.relative_path,
                "role": self.role, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CarrierProjectionMapperEvidenceFile":
        raw = _exact(value, {"byte_count", "relative_path", "role", "sha256"},
                     where="CarrierProjectionMapperEvidenceFile")
        return cls(str(raw["relative_path"]), str(raw["role"]),
                   raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class CarrierProjectionMapperManifest:
    format_version: int
    artifact_version: str
    artifact_status: str
    parent_pack_relative_path: str
    parent_pack_sha256: str
    dependencies: tuple[CarrierProjectionDependencyFile, ...]
    rules: tuple[CarrierProjectionRule, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[CarrierProjectionMapperEvidenceFile, ...]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise CarrierProjectionMapperContractError("manifest artifact identity 漂移")
        _relative_path(self.parent_pack_relative_path,
                       where="parent_pack_relative_path")
        _sha256(self.parent_pack_sha256, where="parent_pack_sha256")
        if (not isinstance(self.dependencies, tuple)
                or tuple(item.carrier_key for item in self.dependencies)
                != IN_SCOPE_CARRIER_KEYS
                or len({item.relative_path for item in self.dependencies})
                != len(self.dependencies)):
            raise CarrierProjectionMapperContractError(
                "dependencies 必须精确覆盖九类 carrier")
        if (not isinstance(self.rules, tuple)
                or tuple(item.carrier_key for item in self.rules)
                != IN_SCOPE_CARRIER_KEYS
                or len({item.rule_key for item in self.rules}) != len(self.rules)
                or len({item.feature_key for item in self.rules}) != len(self.rules)):
            raise CarrierProjectionMapperContractError(
                "rules 必须精确覆盖九类 carrier 且身份唯一")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise CarrierProjectionMapperContractError(
                "manifest execution_state 必须全零")
        if (not isinstance(self.evidence_files, tuple)
                or self.evidence_files != tuple(sorted(
                    self.evidence_files,
                    key=lambda item: (item.relative_path, item.role)))
                or {item.role for item in self.evidence_files} != set(EVIDENCE_ROLES)
                or len({item.relative_path for item in self.evidence_files})
                != len(self.evidence_files)):
            raise CarrierProjectionMapperContractError("manifest evidence_files 未闭合")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "parent_pack_relative_path": self.parent_pack_relative_path,
            "parent_pack_sha256": self.parent_pack_sha256,
            "rules": [item.to_dict() for item in self.rules],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CarrierProjectionMapperManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "dependencies", "evidence_files", "execution_state",
            "format_version", "parent_pack_relative_path",
            "parent_pack_sha256", "rules",
        }, where="CarrierProjectionMapperManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise CarrierProjectionMapperContractError("manifest artifact_kind 非法")
        try:
            return cls(
                raw["format_version"], str(raw["artifact_version"]),
                str(raw["artifact_status"]),
                str(raw["parent_pack_relative_path"]),
                str(raw["parent_pack_sha256"]),
                tuple(CarrierProjectionDependencyFile.from_dict(item)
                      for item in raw["dependencies"]),
                tuple(CarrierProjectionRule.from_dict(item)
                      for item in raw["rules"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(CarrierProjectionMapperEvidenceFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except CarrierProjectionMapperContractError:
            raise
        except Exception as error:
            raise CarrierProjectionMapperContractError(
                "manifest nested field 损坏") from error


def read_carrier_projection_mapper_manifest(
        path: str | Path,
        ) -> CarrierProjectionMapperManifest:
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CarrierProjectionMapperContractError("manifest newline 非法")
        manifest = CarrierProjectionMapperManifest.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except CarrierProjectionMapperContractError:
        raise
    except Exception as error:
        raise CarrierProjectionMapperContractError("manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise CarrierProjectionMapperContractError("manifest 不是 canonical 字节")
    return manifest


def write_carrier_projection_mapper_manifest(
        manifest: CarrierProjectionMapperManifest,
        path: str | Path,
        ) -> Path:
    if not isinstance(manifest, CarrierProjectionMapperManifest):
        raise CarrierProjectionMapperContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise CarrierProjectionMapperContractError(
                "manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise CarrierProjectionMapperContractError("manifest 无法写入") from error
    return target


def verify_carrier_projection_mapper_files(
        manifest: CarrierProjectionMapperManifest,
        *, repository_root: str | Path,
        ) -> None:
    root = Path(repository_root).resolve()
    files = [
        (item.relative_path, item.byte_count, item.sha256)
        for item in (*manifest.dependencies, *manifest.evidence_files)
    ]
    files.append((manifest.parent_pack_relative_path, None,
                  manifest.parent_pack_sha256))
    for relative_path, byte_count, sha256 in files:
        target = (root / Path(*relative_path.split("/"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise CarrierProjectionMapperContractError("evidence 路径逃逸") from error
        if not target.is_file():
            raise CarrierProjectionMapperContractError("evidence 文件缺失")
        payload = target.read_bytes()
        if ((byte_count is not None and len(payload) != byte_count)
                or hashlib.sha256(payload).hexdigest() != sha256):
            raise CarrierProjectionMapperContractError("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_STATUS", "ARTIFACT_VERSION", "EVIDENCE_ROLES",
    "EXECUTION_STATE", "FORMAT_VERSION", "INPUT_KINDS",
    "CarrierProjectionDependencyFile", "CarrierProjectionMapperContractError",
    "CarrierProjectionMapperEvidenceFile", "CarrierProjectionMapperManifest",
    "CarrierProjectionRule", "read_carrier_projection_mapper_manifest",
    "verify_carrier_projection_mapper_files",
    "write_carrier_projection_mapper_manifest",
]
