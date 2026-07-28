"""R-01..R-06 吸收后的 v41 能力基线修订合同。"""
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
ARTIFACT_KIND = "PH2_LANGUAGE_CAPABILITY_BASELINE_R01_R06_REVISION"
ARTIFACT_VERSION = "LG-LC-MD-GG-baseline-v41-supersedes-v40"
ARTIFACT_STATUS = "BASELINE_FROZEN"
MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_baseline_v41.json")
SUPERSEDES_PATH = "data/ph2/manifests/language_capability_baseline_v40.json"
VERSION_KEYS = {
    "backend_version": "BACKEND-EXPLICIT-TRANSACTION-OWNER-V2-R03",
    "code_version": "PH2-PRODUCTION-ABSORPTION-CODE-V4-R01-R06",
    "evaluator_version": "P3IA-PRIVATE-LABEL-EVALUATOR-V2-R01",
    "location_version": "EXACT-RECORD-ACL-LOCATION-V2-R04",
    "schema_version": "PH2-TYPED-ABSORPTION-SCHEMA-V4-R01-R06",
    "segment_version": "SEALED-SEGMENT-INTENT-SINGLE-GET-V2-R02",
}
VERSION_EVIDENCE = {
    "backend_version": [
        "src/pure_integer_ai/storage/backend.py",
        "src/pure_integer_ai/experiments/memory_isolation_runtime.py",
    ],
    "code_version": [
        "src/pure_integer_ai/experiments/free_text_recall_runtime.py",
        "src/pure_integer_ai/experiments/typed_proof_family_runtime.py",
        "src/pure_integer_ai/experiments/long_generation_checkpoint.py",
    ],
    "evaluator_version": [
        "src/pure_integer_ai/experiments/free_text_hierarchy_recall_evaluator.py",
        "src/pure_integer_ai/experiments/ph2_md05_probe_evaluator.py",
    ],
    "location_version": [
        "src/pure_integer_ai/storage/location_manifest.py",
        "src/pure_integer_ai/experiments/authorized_center_runtime.py",
    ],
    "schema_version": [
        "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
        "src/pure_integer_ai/experiments/ph2_typed_proof_family_contract.py",
        "src/pure_integer_ai/experiments/ph2_long_context_absorption_contract.py",
    ],
    "segment_version": [
        "src/pure_integer_ai/storage/segment_repository.py",
        "src/pure_integer_ai/storage/segment_write_intent.py",
        "src/pure_integer_ai/storage/tiered_segment_store.py",
    ],
}
CAPABILITY_FACTS = {
    "code_switch_status": "NE",
    "d02_source_entry_count": 7,
    "gg03_course_status": "COURSE_FROZEN",
    "md05_decision": "PASS",
    "p3ia_course_status": "COURSE_FROZEN",
    "p3ia_production_contract_status": "CONTRACT_READY",
    "p3ib_phase": "PH3",
    "p3ib_status": "NE",
    "r02_storage_status": "PRODUCTION_EVIDENCED",
    "r03_recovery_status": "PRODUCTION_EVIDENCED",
    "r04_authorized_generation_status": "PRODUCTION_EVIDENCED",
    "r05_typed_proof_status": "PRODUCTION_EVIDENCED",
    "r06_long_context_status": "PRODUCTION_EVIDENCED",
}
EXECUTION_STATE = {
    "assessment_updates": 0,
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "evaluator_label_writes": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}
REQUIRED_EVIDENCE_ROLES = {
    "data/ph2/manifests/d02_source_pack_coverage_v1.json": "ARTIFACT",
    "data/ph2/manifests/gg01_generation_choice_contract_v2.json": "ARTIFACT",
    "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json": "ARTIFACT",
    "data/ph2/manifests/gg03_generation_generalization_course_v1.json": "ARTIFACT",
    "data/ph2/manifests/language_capability_baseline_v40.json": "ARTIFACT",
    "data/ph2/manifests/lc07_discourse_information_course_v2.json": "ARTIFACT",
    "data/ph2/manifests/lc09_transfer_axis_manifest_v2.json": "ARTIFACT",
    "data/ph2/manifests/lc13_directional_consumer_manifest_v2.json": "ARTIFACT",
    "data/ph2/manifests/lc15_final_learning_objectives_v2.json": "ARTIFACT",
    "data/ph2/manifests/md01_memory_dynamics_contract_v1.json": "ARTIFACT",
    "data/ph2/manifests/md02_situation_state_adapter_v1.json": "ARTIFACT",
    "data/ph2/manifests/md03_directional_center_adapter_v1.json": "ARTIFACT",
    "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json": "ARTIFACT",
    "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json": "ARTIFACT",
    "data/ph2/manifests/md05_center_diffusion_decision_v1.json": "ARTIFACT",
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json": "ARTIFACT",
    "data/ph2/manifests/r02_storage_absorption_v1.json": "ARTIFACT",
    "data/ph2/manifests/r03_correction_recovery_absorption_v1.json": "ARTIFACT",
    "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json": "ARTIFACT",
    "data/ph2/manifests/r05_typed_proof_family_absorption_v1.json": "ARTIFACT",
    "data/ph2/manifests/r06_long_context_absorption_v1.json": "ARTIFACT",
    "src/pure_integer_ai/experiments/authorized_center_runtime.py": "SOURCE",
    "src/pure_integer_ai/experiments/free_text_hierarchy_recall_evaluator.py": "SOURCE",
    "src/pure_integer_ai/experiments/free_text_recall_runtime.py": "SOURCE",
    "src/pure_integer_ai/experiments/long_generation_checkpoint.py": "SOURCE",
    "src/pure_integer_ai/experiments/memory_isolation_runtime.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_capability_baseline_v41_catalog.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_capability_baseline_v41_contract.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_long_context_absorption_contract.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_md05_probe_evaluator.py": "SOURCE",
    "src/pure_integer_ai/experiments/ph2_typed_proof_family_contract.py": "SOURCE",
    "src/pure_integer_ai/experiments/typed_proof_family_runtime.py": "SOURCE",
    "src/pure_integer_ai/storage/backend.py": "SOURCE",
    "src/pure_integer_ai/storage/location_manifest.py": "SOURCE",
    "src/pure_integer_ai/storage/segment_repository.py": "SOURCE",
    "src/pure_integer_ai/storage/segment_write_intent.py": "SOURCE",
    "src/pure_integer_ai/storage/tiered_segment_store.py": "SOURCE",
    "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py": "TEST",
    "tests/test_d02_p3ia_ledger_revisions.py": "TEST",
    "tests/test_r02_storage_absorption.py": "TEST",
    "tests/test_r03_correction_recovery_absorption.py": "TEST",
    "tests/test_r04_authorized_center_generation_absorption.py": "TEST",
    "tests/test_r05_typed_proof_family_absorption.py": "TEST",
    "tests/test_r06_long_context_absorption.py": "TEST",
}


class CapabilityBaselineV41Error(RuntimeError):
    """v41 能力基线依赖、版本或零执行边界不闭合。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CapabilityBaselineV41Error(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise CapabilityBaselineV41Error(f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise CapabilityBaselineV41Error(f"{where} 必须是 SHA-256")
    return text


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CapabilityBaselineV41Error(f"{where} 字段不精确")
    return value


@dataclass(frozen=True)
class BaselineV41EvidenceFile:
    """一个仓库内 v41 上游 artifact、实现或反例测试身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in {"ARTIFACT", "SOURCE", "TEST"}:
            raise CapabilityBaselineV41Error("evidence role 未登记")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise CapabilityBaselineV41Error("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BaselineV41EvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="BaselineV41EvidenceFile")
        return cls(
            str(raw["relative_path"]), str(raw["role"]),
            raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True)
class CapabilityBaselineV41Manifest:
    """绑定 v40、R-01..06、MD/GG/D-02 和六类版本键的修订。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    supersedes_relative_path: str
    supersedes_sha256: str
    version_keys: CanonicalJsonObject
    version_evidence: CanonicalJsonObject
    capability_facts: CanonicalJsonObject
    evidence_files: tuple[BaselineV41EvidenceFile, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise CapabilityBaselineV41Error("format_version 非法")
        if self.artifact_kind != ARTIFACT_KIND:
            raise CapabilityBaselineV41Error("artifact_kind 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise CapabilityBaselineV41Error("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise CapabilityBaselineV41Error("artifact_status 非法")
        if self.supersedes_relative_path != SUPERSEDES_PATH:
            raise CapabilityBaselineV41Error("supersedes path 非法")
        _sha256(self.supersedes_sha256, where="supersedes sha256")
        for value, expected, label in (
                (self.version_keys, VERSION_KEYS, "version keys"),
                (self.version_evidence, VERSION_EVIDENCE, "version evidence"),
                (self.capability_facts, CAPABILITY_FACTS, "capability facts"),
                (self.execution_state, EXECUTION_STATE, "execution state")):
            if (not isinstance(value, CanonicalJsonObject)
                    or value.to_value() != expected):
                raise CapabilityBaselineV41Error(f"{label} 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or any(not isinstance(item, BaselineV41EvidenceFile)
                       for item in self.evidence_files)):
            raise CapabilityBaselineV41Error("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files, key=lambda item: item.relative_path))
        object.__setattr__(self, "evidence_files", evidence)
        actual = {item.relative_path: item.role for item in evidence}
        if actual != REQUIRED_EVIDENCE_ROLES:
            raise CapabilityBaselineV41Error("evidence 文件集合不完整")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "capability_facts": self.capability_facts.to_value(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "supersedes_relative_path": self.supersedes_relative_path,
            "supersedes_sha256": self.supersedes_sha256,
            "version_evidence": self.version_evidence.to_value(),
            "version_keys": self.version_keys.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "CapabilityBaselineV41Manifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "capability_facts", "evidence_files", "execution_state",
            "format_version", "supersedes_relative_path",
            "supersedes_sha256", "version_evidence", "version_keys",
        }, where="CapabilityBaselineV41Manifest")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["artifact_status"]),
            str(raw["supersedes_relative_path"]),
            str(raw["supersedes_sha256"]),
            CanonicalJsonObject.from_value(raw["version_keys"]),
            CanonicalJsonObject.from_value(raw["version_evidence"]),
            CanonicalJsonObject.from_value(raw["capability_facts"]),
            tuple(BaselineV41EvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_capability_baseline_v41(
        path: str | Path,
        ) -> CapabilityBaselineV41Manifest:
    """严格回读 canonical v41 能力基线。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CapabilityBaselineV41Error("v41 newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = CapabilityBaselineV41Manifest.from_dict(value)
    except CapabilityBaselineV41Error:
        raise
    except Exception as error:
        raise CapabilityBaselineV41Error("v41 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise CapabilityBaselineV41Error("v41 manifest 非规范字节")
    return manifest


def write_capability_baseline_v41(
        manifest: CapabilityBaselineV41Manifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 v41，禁止同版本异内容覆盖。"""
    if not isinstance(manifest, CapabilityBaselineV41Manifest):
        raise CapabilityBaselineV41Error("v41 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise CapabilityBaselineV41Error("v41 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise CapabilityBaselineV41Error("v41 无法写入") from error
    return target


def verify_capability_baseline_v41_files(
        manifest: CapabilityBaselineV41Manifest,
        *, repository_root: str | Path,
        ) -> None:
    """逐字节回验 v41 的全部仓内依赖、实现和测试。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CapabilityBaselineV41Error("evidence 路径逃逸") from error
        if not path.is_file():
            raise CapabilityBaselineV41Error("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise CapabilityBaselineV41Error("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "CAPABILITY_FACTS",
    "EXECUTION_STATE",
    "MANIFEST_PATH",
    "REQUIRED_EVIDENCE_ROLES",
    "SUPERSEDES_PATH",
    "VERSION_EVIDENCE",
    "VERSION_KEYS",
    "BaselineV41EvidenceFile",
    "CapabilityBaselineV41Error",
    "CapabilityBaselineV41Manifest",
    "read_capability_baseline_v41",
    "verify_capability_baseline_v41_files",
    "write_capability_baseline_v41",
]
