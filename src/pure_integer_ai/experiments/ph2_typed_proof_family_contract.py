"""R-05 P2-G 五类专属证明生产吸收的不可覆盖 artifact 合同。"""
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
ARTIFACT_VERSION = "R-05-typed-proof-family-absorption-v1"
ARTIFACT_STATUS = "PRODUCTION_EVIDENCED"
REQUIREMENT_DECISIONS = {
    "causal_counterfactual_direction_witness_pair": "PASS",
    "condition_kind_direction_no_affirming_consequent": "PASS",
    "modal_kind_frame_domain_resolver": "PASS",
    "strict_dispatch_budget_unknown_conflict_fail_closed": "PASS",
    "structural_not_open_world": "PASS",
    "temporal_typed_endpoints_direction_scope_premises": "PASS",
}
FOUR_STATE_MAPPING = {
    "causal_execution_runtime": "PRODUCTION_TRUE_LIVE",
    "event_time_verifier": "PRODUCTION_TRUE_LIVE",
    "logic_executor": "PRODUCTION_TRUE_LIVE",
    "p2g_probe_course_and_ablation": "TEST_ONLY",
    "typed_proof_family_dispatcher": "OPT_IN_GATE",
}
COUNTEREXAMPLE_COVERAGE = {
    "causal_direction_relabel_rejected": 1,
    "causal_inhibiting_requires_effect_transition": 1,
    "causal_witness_source_reuse_fail_closed": 1,
    "condition_affirming_consequent_rejected": 1,
    "condition_kind_evidence_missing_unknown": 1,
    "condition_necessary_direction_preserved": 1,
    "dispatcher_budget_atomic_exhaustion": 1,
    "dispatcher_family_mismatch_rejected": 1,
    "modal_domain_incomplete_unknown": 1,
    "modal_frame_mismatch_fail_closed": 1,
    "modal_resolver_evidence_missing_fail_closed": 1,
    "not_conflict_preserved": 1,
    "not_open_world_unknown_preserved": 1,
    "not_structure_mismatch_fail_closed": 1,
    "temporal_conflict_preserved": 1,
    "temporal_reverse_direction_rejected": 1,
    "temporal_unknown_preserved": 1,
}
HONESTY_LIMITS = {
    "default_training_caller_installed": 0,
    "definitive_truth_claimed": 0,
    "free_text_certificate_formation_claimed": 0,
    "lexical_not_equivalence_claimed": 0,
    "material_implication_causal_equivalence_claimed": 0,
    "probe_fixture_or_ablation_copied_to_production": 0,
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


class TypedProofFamilyContractError(RuntimeError):
    """R-05 固定裁决、文件身份或零执行边界不闭合。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段精确相等。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise TypedProofFamilyContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求 manifest 文本非空且首尾无空白。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypedProofFamilyContractError(
            f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    """要求 evidence 路径是仓库内规范 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise TypedProofFamilyContractError(
            f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    """要求文件摘要是小写完整 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise TypedProofFamilyContractError(f"{where} 必须是 SHA-256")
    return text


@dataclass(frozen=True)
class TypedProofFamilyEvidenceFile:
    """一个仓库相对的 R-05 实现、生产依赖或反例测试身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        """核验路径、角色、非空尺寸和摘要。"""
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise TypedProofFamilyContractError("evidence role 未登记")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise TypedProofFamilyContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        """返回规范 JSON 对象。"""
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "TypedProofFamilyEvidenceFile":
        """从精确 JSON 对象恢复 evidence identity。"""
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="TypedProofFamilyEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True)
class TypedProofFamilyManifest:
    """R-05 六项裁决、四态、反例、诚实边界和文件级证据。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    requirement_decisions: CanonicalJsonObject
    four_state_mapping: CanonicalJsonObject
    counterexample_coverage: CanonicalJsonObject
    honesty_limits: CanonicalJsonObject
    evidence_files: tuple[TypedProofFamilyEvidenceFile, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        """要求固定裁决逐项相等、角色闭合且执行状态全零。"""
        if self.format_version != FORMAT_VERSION:
            raise TypedProofFamilyContractError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise TypedProofFamilyContractError("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise TypedProofFamilyContractError("artifact_status 非法")
        for value, expected, label in (
                (self.requirement_decisions, REQUIREMENT_DECISIONS,
                 "requirement decisions"),
                (self.four_state_mapping, FOUR_STATE_MAPPING,
                 "four-state mapping"),
                (self.counterexample_coverage, COUNTEREXAMPLE_COVERAGE,
                 "counterexample coverage"),
                (self.honesty_limits, HONESTY_LIMITS, "honesty limits"),
                (self.execution_state, EXECUTION_STATE, "execution state")):
            if (not isinstance(value, CanonicalJsonObject)
                    or value.to_value() != expected):
                raise TypedProofFamilyContractError(f"{label} 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or any(not isinstance(item, TypedProofFamilyEvidenceFile)
                       for item in self.evidence_files)):
            raise TypedProofFamilyContractError("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.relative_path, item.role),
        ))
        object.__setattr__(self, "evidence_files", evidence)
        paths = tuple(item.relative_path for item in evidence)
        if len(paths) != len(set(paths)):
            raise TypedProofFamilyContractError("evidence 文件重复")
        if {item.role for item in evidence} != set(EVIDENCE_ROLES):
            raise TypedProofFamilyContractError("evidence role 未闭合")

    def to_dict(self) -> dict[str, Any]:
        """返回规范 artifact JSON 对象。"""
        return {
            "artifact_kind": "PH2_R05_TYPED_PROOF_FAMILY_ABSORPTION",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "counterexample_coverage": self.counterexample_coverage.to_value(),
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "four_state_mapping": self.four_state_mapping.to_value(),
            "honesty_limits": self.honesty_limits.to_value(),
            "requirement_decisions": self.requirement_decisions.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        """返回带单个尾换行的 canonical JSON 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回规范 artifact SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "TypedProofFamilyManifest":
        """从精确 artifact JSON 恢复并重验全部固定合同。"""
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "counterexample_coverage", "evidence_files", "execution_state",
            "format_version", "four_state_mapping", "honesty_limits",
            "requirement_decisions",
        }, where="TypedProofFamilyManifest")
        if raw["artifact_kind"] != "PH2_R05_TYPED_PROOF_FAMILY_ABSORPTION":
            raise TypedProofFamilyContractError("artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            CanonicalJsonObject.from_value(raw["requirement_decisions"]),
            CanonicalJsonObject.from_value(raw["four_state_mapping"]),
            CanonicalJsonObject.from_value(raw["counterexample_coverage"]),
            CanonicalJsonObject.from_value(raw["honesty_limits"]),
            tuple(TypedProofFamilyEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_typed_proof_family_manifest(
        path: str | Path,
        ) -> TypedProofFamilyManifest:
    """严格回读规范 R-05 artifact，并拒绝非 canonical 字节。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise TypedProofFamilyContractError("R-05 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = TypedProofFamilyManifest.from_dict(value)
    except TypedProofFamilyContractError:
        raise
    except Exception as error:
        raise TypedProofFamilyContractError("R-05 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise TypedProofFamilyContractError("R-05 manifest 非规范字节")
    return manifest


def write_typed_proof_family_manifest(
        manifest: TypedProofFamilyManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 R-05 artifact，禁止同版本异内容覆盖。"""
    if not isinstance(manifest, TypedProofFamilyManifest):
        raise TypedProofFamilyContractError("R-05 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TypedProofFamilyContractError(
                "R-05 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TypedProofFamilyContractError(
            "R-05 manifest 无法写入") from error
    return target


def verify_typed_proof_family_files(
        manifest: TypedProofFamilyManifest,
        *, repository_root: str | Path,
        ) -> None:
    """逐字节回验 R-05 checker、生产依赖和反例测试。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise TypedProofFamilyContractError("evidence 路径逃逸") from error
        if not path.is_file():
            raise TypedProofFamilyContractError("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise TypedProofFamilyContractError("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "COUNTEREXAMPLE_COVERAGE",
    "EXECUTION_STATE",
    "FOUR_STATE_MAPPING",
    "HONESTY_LIMITS",
    "REQUIREMENT_DECISIONS",
    "TypedProofFamilyContractError",
    "TypedProofFamilyEvidenceFile",
    "TypedProofFamilyManifest",
    "read_typed_proof_family_manifest",
    "verify_typed_proof_family_files",
    "write_typed_proof_family_manifest",
]
