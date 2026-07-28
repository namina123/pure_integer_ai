"""R-02 存储生产吸收、资源指标与文件身份纯合同。"""
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
ARTIFACT_VERSION = "R-02-storage-absorption-v1"
ARTIFACT_STATUS = "PRODUCTION_EVIDENCED"
OBJECT_KIND_REGISTRY = {
    "LOCATION_MANIFEST": 2,
    "MIGRATION_COMMIT": 3,
    "SEGMENT": 1,
    "SEGMENT_RELEASE": 4,
    "SEGMENT_WRITE_INTENT": 5,
}
COMPATIBILITY = {
    "legacy_unknown_orphan_policy": "EXPLICIT_AUDIT_ONLY",
    "normal_startup_segment_enumeration": 0,
    "old_format_readable": 1,
    "write_intent_required_before_target_payload": 1,
}
EXECUTION_STATE = {
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}
METRIC_KEYS = {
    "active_readers_after",
    "active_write_intents_after",
    "audit_content_matches_build",
    "audit_max_page_records",
    "audit_record_count",
    "audit_segment_payload_bytes",
    "audit_segment_payload_gets",
    "build_content_sha256",
    "exact_query_count",
    "exact_query_record_count",
    "exact_query_segment_payload_bytes",
    "exact_query_segment_payload_gets",
    "manifest_entry_count",
    "max_segment_bytes",
    "record_count",
    "records_per_segment",
    "segment_count",
    "startup_segment_payload_bytes",
    "startup_segment_payload_gets",
}
EVIDENCE_ROOTS = ("REPOSITORY", "WORKSPACE")
EVIDENCE_ROLES = ("PROFILE_REPORT", "PROFILE_STATE", "SOURCE", "TEST")


class StorageAbsorptionContractError(RuntimeError):
    """R-02 资源门、兼容边界或文件证据不闭合。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise StorageAbsorptionContractError(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise StorageAbsorptionContractError(f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise StorageAbsorptionContractError(f"{where} 必须是 SHA-256")
    return text


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise StorageAbsorptionContractError(f"{where} 必须是非负严格整数")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise StorageAbsorptionContractError(f"{where} 字段不精确")
    return value


@dataclass(frozen=True)
class StorageEvidenceFile:
    """一个仓库或工作区相对的 R-02 证据文件身份。"""

    root_key: str
    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.root_key not in EVIDENCE_ROOTS:
            raise StorageAbsorptionContractError("evidence root 未登记")
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise StorageAbsorptionContractError("evidence role 未登记")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise StorageAbsorptionContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    @property
    def identity_key(self) -> str:
        return f"{self.root_key}/{self.relative_path}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "root_key": self.root_key,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StorageEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "root_key", "sha256",
        }, where="StorageEvidenceFile")
        return cls(
            str(raw["root_key"]), str(raw["relative_path"]),
            str(raw["role"]), raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True)
class StorageAbsorptionManifest:
    """R-02 生产机制、十万级资源门和零训练状态。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    object_kind_registry: CanonicalJsonObject
    compatibility: CanonicalJsonObject
    profile_metrics: CanonicalJsonObject
    evidence_files: tuple[StorageEvidenceFile, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise StorageAbsorptionContractError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise StorageAbsorptionContractError("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise StorageAbsorptionContractError("artifact_status 非法")
        if (not isinstance(self.object_kind_registry, CanonicalJsonObject)
                or self.object_kind_registry.to_value() != OBJECT_KIND_REGISTRY):
            raise StorageAbsorptionContractError("object kind registry 漂移")
        if (not isinstance(self.compatibility, CanonicalJsonObject)
                or self.compatibility.to_value() != COMPATIBILITY):
            raise StorageAbsorptionContractError("compatibility 漂移")
        if not isinstance(self.profile_metrics, CanonicalJsonObject):
            raise StorageAbsorptionContractError("profile_metrics 类型非法")
        metrics = _exact(
            self.profile_metrics.to_value(), METRIC_KEYS,
            where="profile_metrics")
        for key in METRIC_KEYS - {"build_content_sha256"}:
            _nonnegative(metrics[key], where=f"profile metric {key}")
        _sha256(metrics["build_content_sha256"], where="build content")
        expected = {
            "active_readers_after": 0,
            "active_write_intents_after": 0,
            "audit_content_matches_build": 1,
            "audit_max_page_records": 256,
            "audit_record_count": 100_000,
            "audit_segment_payload_gets": 100,
            "exact_query_count": 10,
            "exact_query_record_count": 10,
            "exact_query_segment_payload_gets": 10,
            "manifest_entry_count": 100,
            "record_count": 100_000,
            "records_per_segment": 1_000,
            "segment_count": 100,
            "startup_segment_payload_bytes": 0,
            "startup_segment_payload_gets": 0,
        }
        if any(metrics[key] != value for key, value in expected.items()):
            raise StorageAbsorptionContractError("十万级 profile 硬门未通过")
        if (metrics["max_segment_bytes"] <= 0
                or metrics["audit_segment_payload_bytes"] <= 0
                or metrics["exact_query_segment_payload_bytes"] <= 0):
            raise StorageAbsorptionContractError("profile payload 指标非法")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or not all(isinstance(item, StorageEvidenceFile)
                           for item in self.evidence_files)):
            raise StorageAbsorptionContractError("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.root_key, item.relative_path, item.role)))
        object.__setattr__(self, "evidence_files", evidence)
        identities = tuple(item.identity_key for item in evidence)
        if len(identities) != len(set(identities)):
            raise StorageAbsorptionContractError("evidence 文件重复")
        roles = {item.role for item in evidence}
        if roles != set(EVIDENCE_ROLES):
            raise StorageAbsorptionContractError("evidence role 未闭合")
        if sum(item.role == "PROFILE_REPORT" for item in evidence) != 3:
            raise StorageAbsorptionContractError("三阶段 profile report 未列全")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise StorageAbsorptionContractError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_R02_STORAGE_ABSORPTION",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "compatibility": self.compatibility.to_value(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "object_kind_registry": self.object_kind_registry.to_value(),
            "profile_metrics": self.profile_metrics.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StorageAbsorptionManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "compatibility", "evidence_files", "execution_state",
            "format_version", "object_kind_registry", "profile_metrics",
        }, where="StorageAbsorptionManifest")
        if raw["artifact_kind"] != "PH2_R02_STORAGE_ABSORPTION":
            raise StorageAbsorptionContractError("artifact_kind 非法")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            CanonicalJsonObject.from_value(raw["object_kind_registry"]),
            CanonicalJsonObject.from_value(raw["compatibility"]),
            CanonicalJsonObject.from_value(raw["profile_metrics"]),
            tuple(StorageEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_storage_absorption_manifest(
        path: str | Path,
        ) -> StorageAbsorptionManifest:
    """严格回读规范 R-02 artifact。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise StorageAbsorptionContractError("R-02 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = StorageAbsorptionManifest.from_dict(value)
    except StorageAbsorptionContractError:
        raise
    except Exception as error:
        raise StorageAbsorptionContractError("R-02 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise StorageAbsorptionContractError("R-02 manifest 非规范字节")
    return manifest


def write_storage_absorption_manifest(
        manifest: StorageAbsorptionManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 R-02 artifact，禁止同版本覆盖。"""
    if not isinstance(manifest, StorageAbsorptionManifest):
        raise StorageAbsorptionContractError("R-02 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise StorageAbsorptionContractError("R-02 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise StorageAbsorptionContractError("R-02 manifest 无法写入") from error
    return target


def verify_storage_absorption_files(
        manifest: StorageAbsorptionManifest,
        *,
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> None:
    """逐字节回验仓内实现和仓外 profile 状态。"""
    roots = {
        "REPOSITORY": Path(repository_root).resolve(),
        "WORKSPACE": Path(workspace_root).resolve(),
    }
    for item in manifest.evidence_files:
        root = roots[item.root_key]
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise StorageAbsorptionContractError("evidence 路径逃逸") from error
        if not path.is_file():
            raise StorageAbsorptionContractError("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise StorageAbsorptionContractError("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "COMPATIBILITY",
    "EXECUTION_STATE",
    "OBJECT_KIND_REGISTRY",
    "StorageAbsorptionContractError",
    "StorageAbsorptionManifest",
    "StorageEvidenceFile",
    "read_storage_absorption_manifest",
    "verify_storage_absorption_files",
    "write_storage_absorption_manifest",
]
