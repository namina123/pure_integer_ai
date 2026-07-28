"""RI-00 五类额外推理模式的 bounded probe 与范围裁决合同。"""
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
ARTIFACT_STATUS = "PROBE_DECIDED"
RUNTIME_STATUS = "NOT_CONNECTED"
MODE_KEYS = (
    "ABDUCTION",
    "COUNTERFACTUAL",
    "DEFEASIBLE_DEFAULT",
    "DEONTIC_NORMATIVE",
    "TEMPORAL",
)
PROBE_VERDICTS = ("NE", "PASS", "REJECT")
MODE_REPRESENTATION_STATES = (
    "ABSENT",
    "AVAILABLE_NOT_EXECUTED",
    "PARTIAL_SCAFFOLD",
)
REQUIRED_INVARIANTS = {
    "ABDUCTION": (
        "NO_NEW_CAUSES",
        "TYPED_ABDUCTIVE_BRANCH",
    ),
    "COUNTERFACTUAL": (
        "COUNTERFACTUAL_BRANCH_RUNTIME",
        "CURRENT_PROJECTION_POLLUTION_ZERO",
    ),
    "DEFEASIBLE_DEFAULT": (
        "DEFAULT_EXCEPTION_REVERSAL",
        "SOURCE_SCOPE_PRIORITY",
    ),
    "DEONTIC_NORMATIVE": (
        "NORMATIVE_FACT_PROJECTION_SEPARATION",
        "NORM_CONFLICT_STATUS",
    ),
    "TEMPORAL": (
        "SOURCE_SCOPE_PRESERVED",
        "SURFACE_ORDER_NOT_TRUTH",
        "TEMPORAL_FOUR_STATE",
    ),
}
EXPECTED_MODE_VERDICTS = {
    "ABDUCTION": "REJECT",
    "COUNTERFACTUAL": "REJECT",
    "DEFEASIBLE_DEFAULT": "REJECT",
    "DEONTIC_NORMATIVE": "REJECT",
    "TEMPORAL": "PASS",
}
VERIFIER_DIMENSIONS = (
    "ABDUCTION_NO_NEW_CAUSES",
    "COUNTERFACTUAL_CURRENT_PROJECTION_ISOLATION",
    "DEFEASIBLE_EXCEPTION_REVERSAL",
    "DEONTIC_NORMATIVE_FACT_SEPARATION",
    "MODE_SPECIFIC_SCOPE",
    "NO_AGGREGATE_MASKING",
    "SOURCE_SCOPE_PRESERVATION",
    "TEMPORAL_FOUR_STATE",
    "ZERO_HOST_LEARNING_WRITE",
)
VERIFIER_NE_CONDITIONS = (
    "COUNTERFACTUAL_BRANCH_RUNTIME_ABSENT",
    "DEFEASIBLE_PRIORITY_RUNTIME_ABSENT",
    "DEONTIC_PROJECTION_RUNTIME_ABSENT",
    "FORMAL_W07_RUNTIME_NOT_STARTED",
    "TYPED_ABDUCTIVE_BRANCH_ABSENT",
)
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


class ReasoningModeProbeContractError(RuntimeError):
    """RI-00 模式、invariant、范围或零写边界不满足合同。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise ReasoningModeProbeContractError(f"{where} 必须是规范文本")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise ReasoningModeProbeContractError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ReasoningModeProbeContractError(f"{where} 必须是 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise ReasoningModeProbeContractError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise ReasoningModeProbeContractError(f"{where} 必须是安全相对路径")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ReasoningModeProbeContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise ReasoningModeProbeContractError(f"{where} 不能为空")
    for item in value:
        _text(item, where=where)
    if tuple(sorted(set(value))) != value:
        raise ReasoningModeProbeContractError(f"{where} 必须排序去重")
    return value


def evaluate_reasoning_mode_probe(
        invariant_results: CanonicalJsonObject,
        *,
        forbidden_side_effect_count: int,
        host_learning_writes: int,
        ) -> str:
    """以最坏 invariant 裁决；REJECT/NE 不得被其他 PASS 平均掩盖。"""
    if not isinstance(invariant_results, CanonicalJsonObject):
        raise ReasoningModeProbeContractError("invariant_results 类型非法")
    values = invariant_results.to_value()
    if not isinstance(values, dict) or not values:
        raise ReasoningModeProbeContractError("invariant_results 不能为空")
    if any(value not in PROBE_VERDICTS for value in values.values()):
        raise ReasoningModeProbeContractError("invariant verdict 未登记")
    side_effects = _nonnegative(
        forbidden_side_effect_count, where="forbidden_side_effect_count")
    writes = _nonnegative(host_learning_writes, where="host_learning_writes")
    if side_effects or writes or "REJECT" in values.values():
        return "REJECT"
    if "NE" in values.values():
        return "NE"
    return "PASS"


@dataclass(frozen=True)
class ReasoningModeProbeDecision:
    """一个额外推理模式的 typed 支撑、直接 invariant 和窄域结论。"""

    mode_key: str
    representation_state: str
    typed_mode_available: int
    invariant_results: CanonicalJsonObject
    forbidden_side_effect_count: int
    host_learning_writes: int
    verdict: str
    scope_decision: str
    evidence_refs: tuple[str, ...]
    ne_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mode_key not in MODE_KEYS:
            raise ReasoningModeProbeContractError("mode_key 未登记")
        if self.representation_state not in MODE_REPRESENTATION_STATES:
            raise ReasoningModeProbeContractError("representation_state 未登记")
        _flag(self.typed_mode_available, where="typed_mode_available")
        if not isinstance(self.invariant_results, CanonicalJsonObject):
            raise ReasoningModeProbeContractError("invariant_results 类型非法")
        values = self.invariant_results.to_value()
        if tuple(sorted(values)) != REQUIRED_INVARIANTS[self.mode_key]:
            raise ReasoningModeProbeContractError("模式 invariant 未列全")
        actual = evaluate_reasoning_mode_probe(
            self.invariant_results,
            forbidden_side_effect_count=self.forbidden_side_effect_count,
            host_learning_writes=self.host_learning_writes,
        )
        if self.verdict != actual or self.verdict not in PROBE_VERDICTS:
            raise ReasoningModeProbeContractError("模式 verdict 与直接结果漂移")
        if self.verdict != EXPECTED_MODE_VERDICTS[self.mode_key]:
            raise ReasoningModeProbeContractError("RI-00 冻结 verdict 漂移")
        _text(self.scope_decision, where="scope_decision")
        _strict_text_tuple(self.evidence_refs, where="evidence_refs")
        for item in self.evidence_refs:
            _relative_path(item, where="evidence_ref")
        _strict_text_tuple(
            self.ne_conditions, where="ne_conditions", allow_empty=True)
        if self.forbidden_side_effect_count != 0 or self.host_learning_writes != 0:
            raise ReasoningModeProbeContractError("RI-00 禁止副作用和学习写")
        if self.mode_key == "TEMPORAL":
            if not (
                    self.typed_mode_available == 1
                    and self.representation_state == "AVAILABLE_NOT_EXECUTED"
                    and self.verdict == "PASS"
                    and not self.ne_conditions):
                raise ReasoningModeProbeContractError("temporal 窄域 PASS 边界漂移")
        else:
            if (self.typed_mode_available != 0 or self.verdict != "REJECT"
                    or not self.ne_conditions):
                raise ReasoningModeProbeContractError("缺失模式必须诚实 REJECT")
        if (self.mode_key == "ABDUCTION"
                and values["NO_NEW_CAUSES"] != "PASS"):
            raise ReasoningModeProbeContractError("abduction 不得新造 CAUSES")
        if (self.mode_key == "COUNTERFACTUAL"
                and values["CURRENT_PROJECTION_POLLUTION_ZERO"] == "PASS"):
            raise ReasoningModeProbeContractError(
                "counterfactual runtime 缺失时不得伪造零污染 PASS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_refs": list(self.evidence_refs),
            "forbidden_side_effect_count": self.forbidden_side_effect_count,
            "host_learning_writes": self.host_learning_writes,
            "invariant_results": self.invariant_results.to_value(),
            "mode_key": self.mode_key,
            "ne_conditions": list(self.ne_conditions),
            "representation_state": self.representation_state,
            "scope_decision": self.scope_decision,
            "typed_mode_available": self.typed_mode_available,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReasoningModeProbeDecision":
        expected = {
            "evidence_refs", "forbidden_side_effect_count",
            "host_learning_writes", "invariant_results", "mode_key",
            "ne_conditions", "representation_state", "scope_decision",
            "typed_mode_available", "verdict",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ReasoningModeProbeContractError(
                "ReasoningModeProbeDecision 字段不精确")
        return cls(
            str(value["mode_key"]),
            str(value["representation_state"]),
            value["typed_mode_available"],
            CanonicalJsonObject.from_value(value["invariant_results"]),
            value["forbidden_side_effect_count"],
            value["host_learning_writes"],
            str(value["verdict"]),
            str(value["scope_decision"]),
            tuple(str(item) for item in value["evidence_refs"]),
            tuple(str(item) for item in value["ne_conditions"]),
        )


@dataclass(frozen=True)
class ReasoningModeEvidenceFile:
    """RI-00 现有 typed facility 或缺口审计文件身份。"""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise ReasoningModeProbeContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReasoningModeEvidenceFile":
        if not isinstance(value, dict) or set(value) != {
                "byte_count", "relative_path", "sha256"}:
            raise ReasoningModeProbeContractError(
                "ReasoningModeEvidenceFile 字段不精确")
        return cls(
            str(value["relative_path"]), value["byte_count"],
            str(value["sha256"]),
        )


@dataclass(frozen=True)
class ReasoningModeProbeManifest:
    """RI-00 五类模式的 bounded verdict、范围和零运行正式 artifact。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    runtime_status: str
    task_key: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    decisions: tuple[ReasoningModeProbeDecision, ...]
    evidence_files: tuple[ReasoningModeEvidenceFile, ...]
    pass_count: int
    reject_count: int
    ne_count: int
    runtime_pass_authority: int
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ReasoningModeProbeContractError("format_version 漂移")
        _text(self.artifact_version, where="artifact_version")
        if self.artifact_status != ARTIFACT_STATUS:
            raise ReasoningModeProbeContractError("artifact_status 非 PROBE_DECIDED")
        if self.runtime_status != RUNTIME_STATUS:
            raise ReasoningModeProbeContractError("runtime_status 非 NOT_CONNECTED")
        if self.task_key != "RI-00":
            raise ReasoningModeProbeContractError("task_key 非 RI-00")
        _relative_path(
            self.baseline_manifest_relative_path,
            where="baseline_manifest_relative_path")
        _sha256(self.baseline_manifest_sha256, where="baseline_manifest_sha256")
        if (not isinstance(self.decisions, tuple)
                or not all(isinstance(item, ReasoningModeProbeDecision)
                           for item in self.decisions)):
            raise ReasoningModeProbeContractError("decisions 类型非法")
        decisions = tuple(sorted(self.decisions, key=lambda item: item.mode_key))
        object.__setattr__(self, "decisions", decisions)
        if tuple(item.mode_key for item in decisions) != MODE_KEYS:
            raise ReasoningModeProbeContractError("五类 reasoning mode 未闭合")
        if (not isinstance(self.evidence_files, tuple)
                or not all(isinstance(item, ReasoningModeEvidenceFile)
                           for item in self.evidence_files)):
            raise ReasoningModeProbeContractError("evidence_files 类型非法")
        evidence = tuple(sorted(
            self.evidence_files, key=lambda item: item.relative_path))
        object.__setattr__(self, "evidence_files", evidence)
        if len({item.relative_path for item in evidence}) != len(evidence):
            raise ReasoningModeProbeContractError("evidence file 重复")
        referenced = {path for item in decisions for path in item.evidence_refs}
        inventoried = {item.relative_path for item in evidence}
        if referenced != inventoried:
            raise ReasoningModeProbeContractError("evidence inventory 未闭合")
        counts = {
            "pass_count": sum(item.verdict == "PASS" for item in decisions),
            "reject_count": sum(item.verdict == "REJECT" for item in decisions),
            "ne_count": sum(item.verdict == "NE" for item in decisions),
        }
        for name, actual in counts.items():
            if _nonnegative(getattr(self, name), where=name) != actual:
                raise ReasoningModeProbeContractError(f"{name} 漂移")
        _flag(self.runtime_pass_authority, where="runtime_pass_authority")
        if self.runtime_pass_authority != 0:
            raise ReasoningModeProbeContractError("bounded probe 不得签发 runtime PASS")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise ReasoningModeProbeContractError("verifier_dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise ReasoningModeProbeContractError("verifier_ne_conditions 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise ReasoningModeProbeContractError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_RI00_REASONING_MODE_PROBE_MANIFEST",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "baseline_manifest_relative_path": (
                self.baseline_manifest_relative_path),
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "decisions": [item.to_dict() for item in self.decisions],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "ne_count": self.ne_count,
            "pass_count": self.pass_count,
            "reject_count": self.reject_count,
            "runtime_pass_authority": self.runtime_pass_authority,
            "runtime_status": self.runtime_status,
            "task_key": self.task_key,
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReasoningModeProbeManifest":
        expected = {
            "artifact_kind", "artifact_status", "artifact_version",
            "baseline_manifest_relative_path", "baseline_manifest_sha256",
            "decisions", "evidence_files", "execution_state",
            "format_version", "ne_count", "pass_count", "reject_count",
            "runtime_pass_authority", "runtime_status", "task_key",
            "verifier_dimensions", "verifier_ne_conditions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ReasoningModeProbeContractError(
                "ReasoningModeProbeManifest 字段不精确")
        if value["artifact_kind"] != "PH2_RI00_REASONING_MODE_PROBE_MANIFEST":
            raise ReasoningModeProbeContractError("artifact_kind 非法")
        return cls(
            value["format_version"], str(value["artifact_version"]),
            str(value["artifact_status"]), str(value["runtime_status"]),
            str(value["task_key"]),
            str(value["baseline_manifest_relative_path"]),
            str(value["baseline_manifest_sha256"]),
            tuple(ReasoningModeProbeDecision.from_dict(item)
                  for item in value["decisions"]),
            tuple(ReasoningModeEvidenceFile.from_dict(item)
                  for item in value["evidence_files"]),
            value["pass_count"], value["reject_count"], value["ne_count"],
            value["runtime_pass_authority"],
            tuple(str(item) for item in value["verifier_dimensions"]),
            tuple(str(item) for item in value["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(value["execution_state"]),
        )


def write_reasoning_mode_probe_manifest(
        manifest: ReasoningModeProbeManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 RI-00 manifest，拒绝覆盖不同内容。"""
    if not isinstance(manifest, ReasoningModeProbeManifest):
        raise ReasoningModeProbeContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise ReasoningModeProbeContractError("RI-00 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise ReasoningModeProbeContractError("RI-00 manifest 无法发布") from error
    return target


def read_reasoning_mode_probe_manifest(
        path: str | Path,
        ) -> ReasoningModeProbeManifest:
    """严格回读规范 RI-00 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise ReasoningModeProbeContractError("RI-00 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = ReasoningModeProbeManifest.from_dict(value)
    except ReasoningModeProbeContractError:
        raise
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        raise ReasoningModeProbeContractError("RI-00 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise ReasoningModeProbeContractError("RI-00 manifest 非规范 JSON")
    return manifest


__all__ = [
    "ARTIFACT_STATUS",
    "EXECUTION_STATE",
    "EXPECTED_MODE_VERDICTS",
    "FORMAT_VERSION",
    "MODE_KEYS",
    "MODE_REPRESENTATION_STATES",
    "PROBE_VERDICTS",
    "REQUIRED_INVARIANTS",
    "RUNTIME_STATUS",
    "ReasoningModeEvidenceFile",
    "ReasoningModeProbeContractError",
    "ReasoningModeProbeDecision",
    "ReasoningModeProbeManifest",
    "VERIFIER_DIMENSIONS",
    "VERIFIER_NE_CONDITIONS",
    "evaluate_reasoning_mode_probe",
    "read_reasoning_mode_probe_manifest",
    "write_reasoning_mode_probe_manifest",
]
