"""LC-16 九载体方向 runtime 的追加式 manifest 合同。"""
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


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_LC16_CARRIER_DIRECTIONAL_RUNTIME"
ARTIFACT_VERSION = "LC16-CARRIER-DIRECTIONAL-RUNTIME-20260731-A"
ARTIFACT_STATUS = "DIRECTIONAL_RUNTIME_FROZEN_NO_LC16_CAPABILITY_PASS"
DEPENDENCY_ROLES = (
    "ARTIFACT_PROJECTION",
    "CARRIER_INPUT",
    "EVIDENCE_RECORD",
    "PARENT_RUNTIME_CODE",
)
EVIDENCE_ROLES = (
    "CATALOG",
    "DIRECTIONAL_CONTRACT",
    "EVALUATOR",
    "MANIFEST_CONTRACT",
    "RUNTIME",
    "TEST",
)
DIRECTIONAL_SCOPE = {
    "carrier_count": 9,
    "correction_supersede_retention": 1,
    "direction_ablation": 1,
    "direction_count": 3,
    "directional_consumer_implemented": 1,
    "directional_evaluator_implemented": 1,
    "directional_evaluator_module_separate": 1,
    "exact_use_outcome": 1,
    "generation_input_leakage_channels": 0,
    "generation_surface_units": 1,
    "lc16_capability_pass": 0,
    "open_generation_pass": 0,
    "support_refute": 1,
}
EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "directional_consumer_evaluated": 1,
    "directional_hard_conjunct_pass": 0,
    "formal_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}


class CarrierDirectionalManifestContractError(RuntimeError):
    """方向 manifest、parent 或文件身份未闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise CarrierDirectionalManifestContractError(
            f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CarrierDirectionalManifestContractError(
            f"{where} 必须是无首尾空白的非空文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise CarrierDirectionalManifestContractError(
            f"{where} 必须是正严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise CarrierDirectionalManifestContractError(
            f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in text or path.as_posix() != text
            or ":" in path.parts[0]):
        raise CarrierDirectionalManifestContractError(
            f"{where} 必须是安全 POSIX 相对路径")
    return text


@dataclass(frozen=True, order=True)
class CarrierDirectionalManifestFile:
    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="directional file relative_path")
        if self.role not in set(DEPENDENCY_ROLES) | set(EVIDENCE_ROLES):
            raise CarrierDirectionalManifestContractError(
                "directional file role 未登记")
        _positive(self.byte_count, where="directional file byte_count")
        _sha256(self.sha256, where="directional file sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "CarrierDirectionalManifestFile":
        raw = _exact(
            value,
            {"byte_count", "relative_path", "role", "sha256"},
            where="CarrierDirectionalManifestFile",
        )
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True, order=True)
class CarrierDirectionalManifest:
    format_version: int
    artifact_version: str
    artifact_status: str
    parent_runtime_relative_path: str
    parent_runtime_sha256: str
    dependencies: tuple[CarrierDirectionalManifestFile, ...]
    directional_scope: CanonicalJsonObject
    execution_state: CanonicalJsonObject
    evidence_files: tuple[CarrierDirectionalManifestFile, ...]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise CarrierDirectionalManifestContractError(
                "directional manifest artifact identity 漂移")
        _relative_path(
            self.parent_runtime_relative_path,
            where="parent_runtime_relative_path",
        )
        _sha256(self.parent_runtime_sha256, where="parent_runtime_sha256")
        if (not isinstance(self.dependencies, tuple)
                or self.dependencies != tuple(sorted(
                    self.dependencies, key=lambda item: item.role))
                or tuple(item.role for item in self.dependencies)
                != DEPENDENCY_ROLES
                or len({item.relative_path for item in self.dependencies})
                != len(self.dependencies)):
            raise CarrierDirectionalManifestContractError(
                "directional dependencies 未精确闭合")
        if (not isinstance(self.evidence_files, tuple)
                or self.evidence_files != tuple(sorted(
                    self.evidence_files, key=lambda item: item.role))
                or tuple(item.role for item in self.evidence_files)
                != EVIDENCE_ROLES
                or len({item.relative_path for item in self.evidence_files})
                != len(self.evidence_files)):
            raise CarrierDirectionalManifestContractError(
                "directional evidence_files 未精确闭合")
        if (not isinstance(self.directional_scope, CanonicalJsonObject)
                or self.directional_scope.to_value() != DIRECTIONAL_SCOPE):
            raise CarrierDirectionalManifestContractError(
                "directional_scope 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise CarrierDirectionalManifestContractError(
                "directional execution_state 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "directional_scope": self.directional_scope.to_value(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "parent_runtime_relative_path": self.parent_runtime_relative_path,
            "parent_runtime_sha256": self.parent_runtime_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CarrierDirectionalManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "dependencies", "directional_scope", "evidence_files",
            "execution_state", "format_version",
            "parent_runtime_relative_path", "parent_runtime_sha256",
        }, where="CarrierDirectionalManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise CarrierDirectionalManifestContractError(
                "directional manifest artifact_kind 非法")
        try:
            return cls(
                raw["format_version"],
                str(raw["artifact_version"]),
                str(raw["artifact_status"]),
                str(raw["parent_runtime_relative_path"]),
                str(raw["parent_runtime_sha256"]),
                tuple(CarrierDirectionalManifestFile.from_dict(item)
                      for item in raw["dependencies"]),
                CanonicalJsonObject.from_value(raw["directional_scope"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(CarrierDirectionalManifestFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except CarrierDirectionalManifestContractError:
            raise
        except Exception as error:
            raise CarrierDirectionalManifestContractError(
                "directional manifest nested field 损坏") from error


def read_carrier_directional_manifest(
        path: str | Path,
        ) -> CarrierDirectionalManifest:
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CarrierDirectionalManifestContractError(
                "directional manifest newline 非法")
        manifest = CarrierDirectionalManifest.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except CarrierDirectionalManifestContractError:
        raise
    except Exception as error:
        raise CarrierDirectionalManifestContractError(
            "directional manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise CarrierDirectionalManifestContractError(
            "directional manifest 不是 canonical 字节")
    return manifest


def write_carrier_directional_manifest(
        manifest: CarrierDirectionalManifest,
        path: str | Path,
        ) -> Path:
    if not isinstance(manifest, CarrierDirectionalManifest):
        raise CarrierDirectionalManifestContractError(
            "directional manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise CarrierDirectionalManifestContractError(
                "directional manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise CarrierDirectionalManifestContractError(
            "directional manifest 无法写入") from error
    return target


def verify_carrier_directional_files(
        manifest: CarrierDirectionalManifest,
        *, repository_root: str | Path,
        ) -> None:
    root = Path(repository_root).resolve()
    files = [
        (item.relative_path, item.byte_count, item.sha256)
        for item in (*manifest.dependencies, *manifest.evidence_files)
    ]
    files.append((
        manifest.parent_runtime_relative_path,
        None,
        manifest.parent_runtime_sha256,
    ))
    for relative_path, byte_count, sha256 in files:
        target = (root / Path(*relative_path.split("/"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise CarrierDirectionalManifestContractError(
                "directional evidence 路径逃逸") from error
        if not target.is_file():
            raise CarrierDirectionalManifestContractError(
                "directional evidence 文件缺失")
        payload = target.read_bytes()
        if ((byte_count is not None and len(payload) != byte_count)
                or hashlib.sha256(payload).hexdigest() != sha256):
            raise CarrierDirectionalManifestContractError(
                "directional evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_STATUS", "ARTIFACT_VERSION",
    "DEPENDENCY_ROLES", "DIRECTIONAL_SCOPE", "EVIDENCE_ROLES",
    "EXECUTION_STATE", "FORMAT_VERSION", "CarrierDirectionalManifest",
    "CarrierDirectionalManifestContractError",
    "CarrierDirectionalManifestFile", "read_carrier_directional_manifest",
    "verify_carrier_directional_files", "write_carrier_directional_manifest",
]
