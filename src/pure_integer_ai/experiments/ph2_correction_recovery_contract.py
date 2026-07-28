"""R-03 持续修正、精确来源撤回和恢复生产吸收纯合同。"""
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
ARTIFACT_VERSION = "R-03-correction-recovery-absorption-v1"
ARTIFACT_STATUS = "PRODUCTION_EVIDENCED"
MECHANISM_CONTRACT = {
    "backend_close_implicit_commit": 0,
    "evidence_lifecycle_visibility_retention_use_ledgers": 5,
    "forget_selection_owner_exact": 1,
    "forget_selection_owner_subtree": 2,
    "forget_selection_source_exact": 3,
    "forget_transaction_commit_owner": "MEMORY_ISOLATION_RUNTIME",
    "h04_recovery_evidence_progression": "APPEND_ONLY_ALLOWED",
    "h04_recovery_lifecycle_policy": "EXACT_FAIL_CLOSED",
    "source_forget_target_policy": "EXACT_SOURCE_REF",
}
COVERAGE_CONTRACT = {
    "dump_and_recovery_dependency": 1,
    "exact_source_forget": 1,
    "h04_append_evidence_recovery": 1,
    "h04_lifecycle_drift_rejected": 1,
    "h04_new_candidate_recovery": 1,
    "h04_supersede_recovery": 1,
    "history_physical_append_only": 1,
    "owner_forget_preserved": 1,
    "source_reason_idempotence": 1,
    "sqlite_runtime_owned_commit": 1,
    "three_fault_points": 3,
    "unsealed_source_reappearance_rejected": 1,
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
EVIDENCE_ROLES = ("SOURCE", "TEST")


class CorrectionRecoveryContractError(RuntimeError):
    """R-03 合同、文件身份或零执行边界不闭合。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CorrectionRecoveryContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CorrectionRecoveryContractError(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise CorrectionRecoveryContractError(f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise CorrectionRecoveryContractError(f"{where} 必须是 SHA-256")
    return text


@dataclass(frozen=True)
class CorrectionRecoveryEvidenceFile:
    """一个仓库相对的 R-03 实现或回归测试文件身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise CorrectionRecoveryContractError("evidence role 未登记")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise CorrectionRecoveryContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
            cls,
            value: dict[str, Any],
            ) -> "CorrectionRecoveryEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="CorrectionRecoveryEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True)
class CorrectionRecoveryManifest:
    """R-03 生产机制、覆盖合同、文件证据和零执行状态。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    mechanism_contract: CanonicalJsonObject
    coverage_contract: CanonicalJsonObject
    evidence_files: tuple[CorrectionRecoveryEvidenceFile, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise CorrectionRecoveryContractError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise CorrectionRecoveryContractError("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise CorrectionRecoveryContractError("artifact_status 非法")
        if (not isinstance(self.mechanism_contract, CanonicalJsonObject)
                or self.mechanism_contract.to_value() != MECHANISM_CONTRACT):
            raise CorrectionRecoveryContractError("mechanism contract 漂移")
        if (not isinstance(self.coverage_contract, CanonicalJsonObject)
                or self.coverage_contract.to_value() != COVERAGE_CONTRACT):
            raise CorrectionRecoveryContractError("coverage contract 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or any(not isinstance(item, CorrectionRecoveryEvidenceFile)
                       for item in self.evidence_files)):
            raise CorrectionRecoveryContractError("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.relative_path, item.role),
        ))
        object.__setattr__(self, "evidence_files", evidence)
        paths = tuple(item.relative_path for item in evidence)
        if len(paths) != len(set(paths)):
            raise CorrectionRecoveryContractError("evidence 文件重复")
        if {item.role for item in evidence} != set(EVIDENCE_ROLES):
            raise CorrectionRecoveryContractError("evidence role 未闭合")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise CorrectionRecoveryContractError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_R03_CORRECTION_RECOVERY_ABSORPTION",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "coverage_contract": self.coverage_contract.to_value(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "mechanism_contract": self.mechanism_contract.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CorrectionRecoveryManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "coverage_contract", "evidence_files", "execution_state",
            "format_version", "mechanism_contract",
        }, where="CorrectionRecoveryManifest")
        if raw["artifact_kind"] != "PH2_R03_CORRECTION_RECOVERY_ABSORPTION":
            raise CorrectionRecoveryContractError("artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            CanonicalJsonObject.from_value(raw["mechanism_contract"]),
            CanonicalJsonObject.from_value(raw["coverage_contract"]),
            tuple(CorrectionRecoveryEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_correction_recovery_manifest(
        path: str | Path,
        ) -> CorrectionRecoveryManifest:
    """严格回读规范 R-03 artifact。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CorrectionRecoveryContractError("R-03 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = CorrectionRecoveryManifest.from_dict(value)
    except CorrectionRecoveryContractError:
        raise
    except Exception as error:
        raise CorrectionRecoveryContractError("R-03 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise CorrectionRecoveryContractError("R-03 manifest 非规范字节")
    return manifest


def write_correction_recovery_manifest(
        manifest: CorrectionRecoveryManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 R-03 artifact，禁止同版本覆盖。"""
    if not isinstance(manifest, CorrectionRecoveryManifest):
        raise CorrectionRecoveryContractError("R-03 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise CorrectionRecoveryContractError("R-03 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise CorrectionRecoveryContractError("R-03 manifest 无法写入") from error
    return target


def verify_correction_recovery_files(
        manifest: CorrectionRecoveryManifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐字节回验 R-03 实现与直接依赖测试。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CorrectionRecoveryContractError("evidence 路径逃逸") from error
        if not path.is_file():
            raise CorrectionRecoveryContractError("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise CorrectionRecoveryContractError("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "COVERAGE_CONTRACT",
    "CorrectionRecoveryContractError",
    "CorrectionRecoveryEvidenceFile",
    "CorrectionRecoveryManifest",
    "EXECUTION_STATE",
    "MECHANISM_CONTRACT",
    "read_correction_recovery_manifest",
    "verify_correction_recovery_files",
    "write_correction_recovery_manifest",
]
