"""R-06 P2-H 长输入、持久 agenda 与续写 checkpoint 吸收 artifact 合同。"""
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
ARTIFACT_VERSION = "R-06-long-context-absorption-v1"
ARTIFACT_STATUS = "PRODUCTION_EVIDENCED"
REQUIREMENT_DECISIONS = {
    "generation_checkpoint_cursor_prefix_cross_process": "PASS",
    "generation_exact_pagein_postcheck_before_atomic_commit": "PASS",
    "long_input_source_absolute_hierarchy_prefix_content_digest": "PASS",
    "persistent_agenda_metadata_only_center_record_dependency": "PASS",
}
FOUR_STATE_MAPPING = {
    "authorized_center_and_generation_delivery": "OPT_IN_GATE",
    "generation_g04_postcheck": "PRODUCTION_TRUE_LIVE",
    "long_generation_checkpoint_store": "OPT_IN_GATE",
    "long_input_hierarchy_builder": "OPT_IN_GATE",
    "p2h_probe_fixture_renderer_ablation": "TEST_ONLY",
    "persistent_conversation_agenda_store": "OPT_IN_GATE",
    "span_index": "PRODUCTION_TRUE_LIVE",
    "work_memory_transient_hot_region": "PRODUCTION_TRUE_LIVE",
}
COUNTEREXAMPLE_COVERAGE = {
    "agenda_cross_process_metadata_resume": 1,
    "agenda_missing_cannot_bind_payload": 1,
    "agenda_record_identity_drift_zero_read": 1,
    "agenda_shared_read_independent_receipts": 1,
    "checkpoint_budget_exceeded_atomic_failure": 1,
    "checkpoint_cross_process_exact_cursor_resume": 1,
    "checkpoint_future_antecedent_atomic_failure": 1,
    "checkpoint_prefix_drift_atomic_failure": 1,
    "checkpoint_stale_cursor_atomic_failure": 1,
    "generation_missing_pagein_atomic_failure": 1,
    "generation_missing_postcheck_atomic_failure": 1,
    "input_chunk_local_offset_rejected": 1,
    "input_chunk_order_and_width_invariant": 1,
    "input_duplicate_surface_scope_distinct": 1,
    "input_parser_source_drift_distinct": 1,
    "input_prefix_content_digest_drift_rejected": 1,
}
HONESTY_LIMITS = {
    "default_training_caller_installed": 0,
    "fixed_64_sentence_plan_copied": 0,
    "free_text_hierarchy_formation_claimed": 0,
    "synthetic_fact_formula_copied": 0,
    "unbounded_context_or_memory_claimed": 0,
    "work_memory_persistent_agenda_equivalence_claimed": 0,
    "controlled_renderer_copied": 0,
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


class LongContextAbsorptionContractError(RuntimeError):
    """R-06 固定裁决、文件身份或零执行边界不闭合。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段精确相等。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise LongContextAbsorptionContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求 manifest 文本非空且首尾无空白。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LongContextAbsorptionContractError(
            f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    """要求 evidence 路径是仓库内规范 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise LongContextAbsorptionContractError(
            f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    """要求文件摘要是小写完整 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise LongContextAbsorptionContractError(f"{where} 必须是 SHA-256")
    return text


@dataclass(frozen=True)
class LongContextAbsorptionEvidenceFile:
    """一个仓库相对的 R-06 实现、生产依赖或反例测试身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        """核验路径、角色、非空尺寸和摘要。"""
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise LongContextAbsorptionContractError("evidence role 未登记")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise LongContextAbsorptionContractError("evidence 文件不得为空")
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
            ) -> "LongContextAbsorptionEvidenceFile":
        """从精确 JSON 对象恢复 evidence identity。"""
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="LongContextAbsorptionEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True)
class LongContextAbsorptionManifest:
    """R-06 四项裁决、四态、反例、诚实边界和文件级证据。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    requirement_decisions: CanonicalJsonObject
    four_state_mapping: CanonicalJsonObject
    counterexample_coverage: CanonicalJsonObject
    honesty_limits: CanonicalJsonObject
    evidence_files: tuple[LongContextAbsorptionEvidenceFile, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        """要求固定裁决逐项相等、角色闭合且执行状态全零。"""
        if self.format_version != FORMAT_VERSION:
            raise LongContextAbsorptionContractError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise LongContextAbsorptionContractError("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise LongContextAbsorptionContractError("artifact_status 非法")
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
                raise LongContextAbsorptionContractError(f"{label} 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or any(not isinstance(item, LongContextAbsorptionEvidenceFile)
                       for item in self.evidence_files)):
            raise LongContextAbsorptionContractError("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.relative_path, item.role),
        ))
        object.__setattr__(self, "evidence_files", evidence)
        paths = tuple(item.relative_path for item in evidence)
        if len(paths) != len(set(paths)):
            raise LongContextAbsorptionContractError("evidence 文件重复")
        if {item.role for item in evidence} != set(EVIDENCE_ROLES):
            raise LongContextAbsorptionContractError("evidence role 未闭合")

    def to_dict(self) -> dict[str, Any]:
        """返回规范 artifact JSON 对象。"""
        return {
            "artifact_kind": "PH2_R06_LONG_CONTEXT_ABSORPTION",
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
            ) -> "LongContextAbsorptionManifest":
        """从精确 artifact JSON 恢复并重验全部固定合同。"""
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "counterexample_coverage", "evidence_files", "execution_state",
            "format_version", "four_state_mapping", "honesty_limits",
            "requirement_decisions",
        }, where="LongContextAbsorptionManifest")
        if raw["artifact_kind"] != "PH2_R06_LONG_CONTEXT_ABSORPTION":
            raise LongContextAbsorptionContractError("artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            CanonicalJsonObject.from_value(raw["requirement_decisions"]),
            CanonicalJsonObject.from_value(raw["four_state_mapping"]),
            CanonicalJsonObject.from_value(raw["counterexample_coverage"]),
            CanonicalJsonObject.from_value(raw["honesty_limits"]),
            tuple(LongContextAbsorptionEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_long_context_absorption_manifest(
        path: str | Path,
        ) -> LongContextAbsorptionManifest:
    """严格回读规范 R-06 artifact，并拒绝非 canonical 字节。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise LongContextAbsorptionContractError(
                "R-06 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = LongContextAbsorptionManifest.from_dict(value)
    except LongContextAbsorptionContractError:
        raise
    except Exception as error:
        raise LongContextAbsorptionContractError(
            "R-06 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise LongContextAbsorptionContractError("R-06 manifest 非规范字节")
    return manifest


def write_long_context_absorption_manifest(
        manifest: LongContextAbsorptionManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 R-06 artifact，禁止同版本异内容覆盖。"""
    if not isinstance(manifest, LongContextAbsorptionManifest):
        raise LongContextAbsorptionContractError("R-06 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise LongContextAbsorptionContractError(
                "R-06 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise LongContextAbsorptionContractError(
            "R-06 manifest 无法写入") from error
    return target


def verify_long_context_absorption_files(
        manifest: LongContextAbsorptionManifest,
        *, repository_root: str | Path,
        ) -> None:
    """逐字节回验 R-06 三件套、生产依赖和反例测试。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise LongContextAbsorptionContractError(
                "evidence 路径逃逸") from error
        if not path.is_file():
            raise LongContextAbsorptionContractError("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise LongContextAbsorptionContractError(
                "evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "COUNTEREXAMPLE_COVERAGE",
    "EXECUTION_STATE",
    "FOUR_STATE_MAPPING",
    "HONESTY_LIMITS",
    "LongContextAbsorptionContractError",
    "LongContextAbsorptionEvidenceFile",
    "LongContextAbsorptionManifest",
    "REQUIREMENT_DECISIONS",
    "read_long_context_absorption_manifest",
    "verify_long_context_absorption_files",
    "write_long_context_absorption_manifest",
]
