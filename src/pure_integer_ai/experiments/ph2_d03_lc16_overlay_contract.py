"""D-03 v1 之后的 LC-16 typed Artifact 追加式总账合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_records import (
    D03Lc16OverlayError,
    OverlayCarrierCourse,
    OverlayCoverageCell,
    OverlayEvaluatorBoundary,
    OverlayEvidenceFile,
    OverlayFailureDependency,
    OverlayFileIdentity,
    OverlayGenerationAccount,
    OverlayResourceBudget,
    OverlayScopeRecord,
    exact_dict,
    sha256,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import (
    ARTIFACT_KIND,
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    EVALUATOR_SPECS,
    EVIDENCE_ROLES,
    EXECUTION_STATE,
    FAILURE_DEPENDENCY_SPECS,
    FORMAT_VERSION,
    GENERATION_ACCOUNT_SPECS,
    RESOURCE_BUDGET_SPECS,
    SCOPE_KEYS,
    SCOPE_SPECS,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    DIRECTIONS,
    IN_SCOPE_CARRIER_KEYS,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
)


def expected_scope_records() -> tuple[OverlayScopeRecord, ...]:
    """从冻结常量构造八个阶段 scope。"""
    return tuple(OverlayScopeRecord(*item) for item in SCOPE_SPECS)


def expected_coverage_cells() -> tuple[OverlayCoverageCell, ...]:
    """构造 9×3×8 的全显式载体、方向与阶段 scope 矩阵。"""
    scopes = {item.scope_key: item for item in expected_scope_records()}
    return tuple(
        OverlayCoverageCell(
            carrier_key,
            direction,
            scope_key,
            scopes[scope_key].evaluator_key,
            scopes[scope_key].current_state,
        )
        for carrier_key in IN_SCOPE_CARRIER_KEYS
        for direction in DIRECTIONS
        for scope_key in SCOPE_KEYS
    )


def expected_evaluator_boundaries() -> tuple[OverlayEvaluatorBoundary, ...]:
    """构造三向公开 evaluator 与 W-02/W-03 私评预注册边界。"""
    return tuple(OverlayEvaluatorBoundary(*item) for item in EVALUATOR_SPECS)


def expected_resource_budgets() -> tuple[OverlayResourceBudget, ...]:
    """构造两个 supplemental family 的硬资源预算。"""
    return tuple(OverlayResourceBudget(*item) for item in RESOURCE_BUDGET_SPECS)


def expected_failure_dependencies() -> tuple[OverlayFailureDependency, ...]:
    """构造 overlay、资格与开放生成的完整失效后缀。"""
    return tuple(OverlayFailureDependency(*item)
                 for item in FAILURE_DEPENDENCY_SPECS)


def expected_generation_accounts() -> tuple[OverlayGenerationAccount, ...]:
    """构造互不聚合的来源化回放与开放生成账户。"""
    return tuple(OverlayGenerationAccount(*item)
                 for item in GENERATION_ACCOUNT_SPECS)


@dataclass(frozen=True, order=True)
class D03Lc16SuccessorOverlay:
    """冻结 D-03 v1 后新增 LC-16 课程与补充资格的追加式总账。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    parent_directional_manifest: OverlayFileIdentity
    d03_v1_global_manifest: OverlayFileIdentity
    d03_v1_publication_receipt: OverlayFileIdentity
    typed_carrier_pack: OverlayFileIdentity
    w02_parent_receipt_sha256: str
    w03_parent_receipt: OverlayFileIdentity
    carrier_courses: tuple[OverlayCarrierCourse, ...]
    scope_records: tuple[OverlayScopeRecord, ...]
    coverage_cells: tuple[OverlayCoverageCell, ...]
    evaluator_boundaries: tuple[OverlayEvaluatorBoundary, ...]
    resource_budgets: tuple[OverlayResourceBudget, ...]
    failure_dependencies: tuple[OverlayFailureDependency, ...]
    generation_accounts: tuple[OverlayGenerationAccount, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[OverlayEvidenceFile, ...]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise D03Lc16OverlayError("overlay artifact identity 漂移")
        for name in (
                "parent_directional_manifest", "d03_v1_global_manifest",
                "d03_v1_publication_receipt", "typed_carrier_pack",
                "w03_parent_receipt"):
            if not isinstance(getattr(self, name), OverlayFileIdentity):
                raise D03Lc16OverlayError(f"overlay {name} 类型非法")
        sha256(self.w02_parent_receipt_sha256,
               where="w02_parent_receipt_sha256")
        if (not isinstance(self.carrier_courses, tuple)
                or tuple(item.carrier_key for item in self.carrier_courses)
                != IN_SCOPE_CARRIER_KEYS):
            raise D03Lc16OverlayError("overlay 九类 carrier 课程未精确闭合")
        all_cases = tuple(
            case for course in self.carrier_courses for case in course.cases)
        if (len(all_cases) != len(IN_SCOPE_CARRIER_KEYS) * len(SAMPLE_KINDS)
                or len({item.case_key for item in all_cases}) != len(all_cases)
                or len({item.owner_key for item in all_cases}) != len(all_cases)):
            raise D03Lc16OverlayError("overlay 63 个 owner 隔离 case 未闭合")
        if self.scope_records != expected_scope_records():
            raise D03Lc16OverlayError("overlay scope_records 漂移")
        if self.coverage_cells != expected_coverage_cells():
            raise D03Lc16OverlayError("overlay carrier×direction×scope 漂移")
        if self.evaluator_boundaries != expected_evaluator_boundaries():
            raise D03Lc16OverlayError("overlay evaluator boundary 漂移")
        if self.resource_budgets != expected_resource_budgets():
            raise D03Lc16OverlayError("overlay resource budget 漂移")
        if self.failure_dependencies != expected_failure_dependencies():
            raise D03Lc16OverlayError("overlay failure dependency 漂移")
        if self.generation_accounts != expected_generation_accounts():
            raise D03Lc16OverlayError("overlay generation 分账漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise D03Lc16OverlayError("overlay execution_state 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or tuple(item.role for item in self.evidence_files)
                != EVIDENCE_ROLES
                or len({item.file_identity.relative_path
                        for item in self.evidence_files})
                != len(self.evidence_files)):
            raise D03Lc16OverlayError("overlay evidence_files 未精确闭合")

    def to_dict(self) -> dict[str, Any]:
        """导出 canonical JSON 所需的完整 overlay object。"""
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "carrier_courses": [item.to_dict() for item in self.carrier_courses],
            "coverage_cells": [item.to_dict() for item in self.coverage_cells],
            "d03_v1_global_manifest": self.d03_v1_global_manifest.to_dict(),
            "d03_v1_publication_receipt": (
                self.d03_v1_publication_receipt.to_dict()),
            "evaluator_boundaries": [
                item.to_dict() for item in self.evaluator_boundaries],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "failure_dependencies": [
                item.to_dict() for item in self.failure_dependencies],
            "format_version": self.format_version,
            "generation_accounts": [
                item.to_dict() for item in self.generation_accounts],
            "parent_directional_manifest": (
                self.parent_directional_manifest.to_dict()),
            "resource_budgets": [item.to_dict() for item in self.resource_budgets],
            "scope_records": [item.to_dict() for item in self.scope_records],
            "typed_carrier_pack": self.typed_carrier_pack.to_dict(),
            "w02_parent_receipt_sha256": self.w02_parent_receipt_sha256,
            "w03_parent_receipt": self.w03_parent_receipt.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """返回单尾换行的 canonical JSON 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回 overlay canonical 字节的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "D03Lc16SuccessorOverlay":
        """从字段精确的 object 恢复 overlay。"""
        raw = exact_dict(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "carrier_courses", "coverage_cells", "d03_v1_global_manifest",
            "d03_v1_publication_receipt", "evaluator_boundaries",
            "evidence_files", "execution_state", "failure_dependencies",
            "format_version", "generation_accounts",
            "parent_directional_manifest", "resource_budgets", "scope_records",
            "typed_carrier_pack", "w02_parent_receipt_sha256",
            "w03_parent_receipt",
        }, where="D03Lc16SuccessorOverlay")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise D03Lc16OverlayError("overlay artifact_kind 漂移")
        try:
            return cls(
                raw["format_version"], str(raw["artifact_version"]),
                str(raw["artifact_status"]),
                OverlayFileIdentity.from_dict(raw["parent_directional_manifest"]),
                OverlayFileIdentity.from_dict(raw["d03_v1_global_manifest"]),
                OverlayFileIdentity.from_dict(
                    raw["d03_v1_publication_receipt"]),
                OverlayFileIdentity.from_dict(raw["typed_carrier_pack"]),
                str(raw["w02_parent_receipt_sha256"]),
                OverlayFileIdentity.from_dict(raw["w03_parent_receipt"]),
                tuple(OverlayCarrierCourse.from_dict(item)
                      for item in raw["carrier_courses"]),
                tuple(OverlayScopeRecord.from_dict(item)
                      for item in raw["scope_records"]),
                tuple(OverlayCoverageCell.from_dict(item)
                      for item in raw["coverage_cells"]),
                tuple(OverlayEvaluatorBoundary.from_dict(item)
                      for item in raw["evaluator_boundaries"]),
                tuple(OverlayResourceBudget.from_dict(item)
                      for item in raw["resource_budgets"]),
                tuple(OverlayFailureDependency.from_dict(item)
                      for item in raw["failure_dependencies"]),
                tuple(OverlayGenerationAccount.from_dict(item)
                      for item in raw["generation_accounts"]),
                CanonicalJsonObject.from_value(raw["execution_state"]),
                tuple(OverlayEvidenceFile.from_dict(item)
                      for item in raw["evidence_files"]),
            )
        except D03Lc16OverlayError:
            raise
        except Exception as error:
            raise D03Lc16OverlayError("overlay nested field 损坏") from error


def read_d03_lc16_successor_overlay(
        path: str | Path,
        ) -> D03Lc16SuccessorOverlay:
    """严格回读 canonical overlay。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise D03Lc16OverlayError("overlay newline 非法")
        manifest = D03Lc16SuccessorOverlay.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except D03Lc16OverlayError:
        raise
    except Exception as error:
        raise D03Lc16OverlayError("overlay 文件损坏") from error
    if manifest.canonical_bytes() != payload:
        raise D03Lc16OverlayError("overlay 不是 canonical 字节")
    return manifest


def write_d03_lc16_successor_overlay(
        manifest: D03Lc16SuccessorOverlay,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 overlay，拒绝同路径异内容覆盖。"""
    if not isinstance(manifest, D03Lc16SuccessorOverlay):
        raise D03Lc16OverlayError("overlay 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise D03Lc16OverlayError("overlay 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise D03Lc16OverlayError("overlay 无法写入") from error
    return target


def verify_d03_lc16_overlay_files(
        manifest: D03Lc16SuccessorOverlay,
        *, repository_root: str | Path,
        ) -> None:
    """逐字节回验 overlay 直接绑定的历史、课程和代码证据。"""
    root = Path(repository_root).resolve()
    identities = [
        manifest.parent_directional_manifest,
        manifest.d03_v1_global_manifest,
        manifest.d03_v1_publication_receipt,
        manifest.typed_carrier_pack,
        manifest.w03_parent_receipt,
        *(item.manifest_identity for item in manifest.carrier_courses),
        *(item.sample_identity for item in manifest.carrier_courses),
        *(item.file_identity for item in manifest.evidence_files),
    ]
    if len({item.relative_path for item in identities}) != len(identities):
        raise D03Lc16OverlayError("overlay 直接文件路径重复")
    for identity in identities:
        target = (root / Path(*identity.relative_path.split("/"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise D03Lc16OverlayError("overlay 文件路径逃逸") from error
        if not target.is_file():
            raise D03Lc16OverlayError("overlay 文件缺失")
        payload = target.read_bytes()
        if (len(payload) != identity.size_bytes
                or hashlib.sha256(payload).hexdigest() != identity.sha256):
            raise D03Lc16OverlayError("overlay 文件身份漂移")


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_STATUS", "ARTIFACT_VERSION",
    "D03Lc16OverlayError", "D03Lc16SuccessorOverlay", "FORMAT_VERSION",
    "expected_coverage_cells", "expected_evaluator_boundaries",
    "expected_failure_dependencies", "expected_generation_accounts",
    "expected_resource_budgets", "expected_scope_records",
    "read_d03_lc16_successor_overlay", "verify_d03_lc16_overlay_files",
    "write_d03_lc16_successor_overlay",
]
