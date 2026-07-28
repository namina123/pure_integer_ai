"""LC-02 及后续能力课程共用的 D-03 前冻结 manifest 合同。"""
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
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    SAMPLE_FAMILIES,
)


FORMAT_VERSION = 1
COURSE_STATUS = "COURSE_FROZEN"
RUNTIME_STATUS = "NOT_STARTED"
COURSE_SPLITS = ("train", "held_out")
COURSE_SPLIT_AXES = (
    "COMBINATION_CLUSTER",
    "CONTENT_CLUSTER",
    "EVIDENCE_OWNER",
    "SHAPE_CLUSTER",
    "SOURCE_CLUSTER",
    "SPLIT",
    "TEMPLATE_CLUSTER",
)
COURSE_EXECUTION_STATE = {
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
COURSE_INVARIANTS = {
    "compiler_or_record_count_is_not_learning_evidence": 1,
    "course_frozen_is_not_runtime_learned": 1,
    "evaluator_and_held_out_host_writes_zero": 1,
    "expected_payload_private_to_label_owner": 1,
    "formal_training_forbidden_before_d03": 1,
    "observation_selection_state_unselected": 1,
    "runtime_pass_authority": 0,
    "student_reads_observation_only": 1,
    "teacher_call_forbidden": 1,
}


class CapabilityCourseContractError(RuntimeError):
    """能力课程 manifest 缺字段、越权或伪造运行事实。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求对象字段与声明集合精确一致。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise CapabilityCourseContractError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求字段为无首尾空白的非空文本。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CapabilityCourseContractError(f"{where} 必须是非空规范文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求字段为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise CapabilityCourseContractError(f"{where} 必须是正严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求字段为小写 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CapabilityCourseContractError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求公开 artifact 路径可迁移且不含环境私有位置。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or ":" in path.parts[0]):
        raise CapabilityCourseContractError(f"{where} 必须是安全相对路径")
    lowered = text.casefold()
    if any(item in lowered for item in (
            "cookie", "proxy", "client_secret", "authorization")):
        raise CapabilityCourseContractError(f"{where} 泄漏环境字段")
    return text


def _sorted_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False) -> tuple[str, ...]:
    """要求文本 tuple 已排序去重，避免规范字节漂移。"""
    if not isinstance(value, tuple):
        raise CapabilityCourseContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise CapabilityCourseContractError(f"{where} 不能为空")
    if any(not isinstance(item, str) or not item or item.strip() != item
           for item in value):
        raise CapabilityCourseContractError(f"{where} 含非法文本")
    if tuple(sorted(set(value))) != value:
        raise CapabilityCourseContractError(f"{where} 必须排序去重")
    return value


@dataclass(frozen=True)
class CapabilityCourseManifest:
    """冻结一个能力课程的来源、split、verifier、消融和零执行事实。"""

    format_version: int
    artifact_version: str
    course_status: str
    runtime_status: str
    task_keys: tuple[str, ...]
    capability_keys: tuple[str, ...]
    stage: str
    substage: str
    source_key: str
    license_id: str
    sample_relative_path: str
    sample_sha256: str
    sample_count: int
    sample_families: tuple[str, ...]
    split_axes: tuple[str, ...]
    teacher_families: tuple[str, ...]
    evaluator_families: tuple[str, ...]
    teacher_template_families: tuple[str, ...]
    evaluator_template_families: tuple[str, ...]
    payload_kind: str
    payload_keys: tuple[str, ...]
    objective_keys: tuple[str, ...]
    evaluator_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    retention_protocols: tuple[str, ...]
    combination_axes: tuple[str, ...]
    baseline_kinds: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    pack_manifest_relative_path: str
    pack_manifest_sha256: str
    pack_record_count: int
    pack_splits: tuple[str, ...]
    invariants: CanonicalJsonObject
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        """校验课程冻结不被误写成 runtime、训练或 mastered 证据。"""
        if self.format_version != FORMAT_VERSION:
            raise CapabilityCourseContractError("course format_version 非法")
        _text(self.artifact_version, where="artifact_version")
        if self.course_status != COURSE_STATUS:
            raise CapabilityCourseContractError("course_status 必须 COURSE_FROZEN")
        if self.runtime_status != RUNTIME_STATUS:
            raise CapabilityCourseContractError("runtime_status 必须 NOT_STARTED")
        _sorted_tuple(self.task_keys, where="task_keys")
        if any(not item.startswith(("LC-", "GG-")) for item in self.task_keys):
            raise CapabilityCourseContractError(
                "task_keys 必须是 LC/GG 能力课程任务")
        _sorted_tuple(self.capability_keys, where="capability_keys")
        if any(item not in CAPABILITY_KEYS for item in self.capability_keys):
            raise CapabilityCourseContractError("capability_keys 未登记")
        _text(self.stage, where="stage")
        _text(self.substage, where="substage")
        _text(self.source_key, where="source_key")
        _text(self.license_id, where="license_id")
        _relative_path(self.sample_relative_path, where="sample_relative_path")
        _sha256(self.sample_sha256, where="sample_sha256")
        _positive(self.sample_count, where="sample_count")
        if self.sample_families != SAMPLE_FAMILIES:
            raise CapabilityCourseContractError("七类 sample family 未冻结")
        if self.split_axes != COURSE_SPLIT_AXES:
            raise CapabilityCourseContractError("split_axes 未冻结")
        for name in (
                "teacher_families", "evaluator_families",
                "teacher_template_families", "evaluator_template_families",
                "payload_keys", "objective_keys", "evaluator_dimensions",
                "verifier_ne_conditions", "retention_protocols",
                "combination_axes", "baseline_kinds", "ablation_keys"):
            _sorted_tuple(getattr(self, name), where=name)
        if set(self.teacher_families) & set(self.evaluator_families):
            raise CapabilityCourseContractError("teacher/evaluator family 泄漏")
        if (set(self.teacher_template_families)
                & set(self.evaluator_template_families)):
            raise CapabilityCourseContractError("teacher/evaluator template 泄漏")
        _text(self.payload_kind, where="payload_kind")
        if any(item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise CapabilityCourseContractError("objective_keys 未登记")
        _relative_path(
            self.pack_manifest_relative_path,
            where="pack_manifest_relative_path",
        )
        _sha256(self.pack_manifest_sha256, where="pack_manifest_sha256")
        _positive(self.pack_record_count, where="pack_record_count")
        if self.pack_record_count != self.sample_count * 3:
            raise CapabilityCourseContractError("pack_record_count 与四 owner 记录数不闭合")
        if self.pack_splits != COURSE_SPLITS:
            raise CapabilityCourseContractError("pack_splits 未冻结")
        if (not isinstance(self.invariants, CanonicalJsonObject)
                or self.invariants.to_value() != COURSE_INVARIANTS):
            raise CapabilityCourseContractError("course invariants 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != COURSE_EXECUTION_STATE):
            raise CapabilityCourseContractError("course execution_state 非零或缺项")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "ablation_keys": list(self.ablation_keys),
            "artifact_kind": "PH2_CAPABILITY_COURSE_FREEZE",
            "artifact_version": self.artifact_version,
            "baseline_kinds": list(self.baseline_kinds),
            "capability_keys": list(self.capability_keys),
            "combination_axes": list(self.combination_axes),
            "course_status": self.course_status,
            "evaluator_dimensions": list(self.evaluator_dimensions),
            "evaluator_families": list(self.evaluator_families),
            "evaluator_template_families": list(
                self.evaluator_template_families),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "invariants": self.invariants.to_value(),
            "license_id": self.license_id,
            "objective_keys": list(self.objective_keys),
            "pack_manifest_relative_path": self.pack_manifest_relative_path,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "pack_record_count": self.pack_record_count,
            "pack_splits": list(self.pack_splits),
            "payload_keys": list(self.payload_keys),
            "payload_kind": self.payload_kind,
            "retention_protocols": list(self.retention_protocols),
            "runtime_status": self.runtime_status,
            "sample_count": self.sample_count,
            "sample_families": list(self.sample_families),
            "sample_relative_path": self.sample_relative_path,
            "sample_sha256": self.sample_sha256,
            "source_key": self.source_key,
            "split_axes": list(self.split_axes),
            "stage": self.stage,
            "substage": self.substage,
            "task_keys": list(self.task_keys),
            "teacher_families": list(self.teacher_families),
            "teacher_template_families": list(
                self.teacher_template_families),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        """返回带单一结尾换行的规范 manifest 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回规范 manifest 的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapabilityCourseManifest":
        """从精确字段对象恢复课程 manifest。"""
        raw = _exact_keys(value, {
            "ablation_keys", "artifact_kind", "artifact_version",
            "baseline_kinds", "capability_keys", "combination_axes",
            "course_status", "evaluator_dimensions", "evaluator_families",
            "evaluator_template_families", "execution_state",
            "format_version", "invariants", "license_id", "objective_keys",
            "pack_manifest_relative_path", "pack_manifest_sha256",
            "pack_record_count", "pack_splits", "payload_keys", "payload_kind",
            "retention_protocols", "runtime_status", "sample_count",
            "sample_families", "sample_relative_path", "sample_sha256",
            "source_key", "split_axes", "stage", "substage", "task_keys",
            "teacher_families", "teacher_template_families",
            "verifier_ne_conditions",
        }, where="CapabilityCourseManifest")
        if raw["artifact_kind"] != "PH2_CAPABILITY_COURSE_FREEZE":
            raise CapabilityCourseContractError("artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["course_status"]),
            str(raw["runtime_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            tuple(str(item) for item in raw["capability_keys"]),
            str(raw["stage"]),
            str(raw["substage"]),
            str(raw["source_key"]),
            str(raw["license_id"]),
            str(raw["sample_relative_path"]),
            str(raw["sample_sha256"]),
            raw["sample_count"],
            tuple(str(item) for item in raw["sample_families"]),
            tuple(str(item) for item in raw["split_axes"]),
            tuple(str(item) for item in raw["teacher_families"]),
            tuple(str(item) for item in raw["evaluator_families"]),
            tuple(str(item) for item in raw["teacher_template_families"]),
            tuple(str(item) for item in raw["evaluator_template_families"]),
            str(raw["payload_kind"]),
            tuple(str(item) for item in raw["payload_keys"]),
            tuple(str(item) for item in raw["objective_keys"]),
            tuple(str(item) for item in raw["evaluator_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            tuple(str(item) for item in raw["retention_protocols"]),
            tuple(str(item) for item in raw["combination_axes"]),
            tuple(str(item) for item in raw["baseline_kinds"]),
            tuple(str(item) for item in raw["ablation_keys"]),
            str(raw["pack_manifest_relative_path"]),
            str(raw["pack_manifest_sha256"]),
            raw["pack_record_count"],
            tuple(str(item) for item in raw["pack_splits"]),
            CanonicalJsonObject.from_value(raw["invariants"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def write_capability_course_manifest(
        manifest: CapabilityCourseManifest,
        path: str | Path) -> Path:
    """独占或幂等写课程 manifest，禁止原地覆盖不同内容。"""
    if not isinstance(manifest, CapabilityCourseManifest):
        raise CapabilityCourseContractError("course manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise CapabilityCourseContractError("course manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise CapabilityCourseContractError("course manifest 无法发布") from error
    return target


def read_capability_course_manifest(
        path: str | Path) -> CapabilityCourseManifest:
    """严格回读规范课程 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CapabilityCourseContractError("course manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = CapabilityCourseManifest.from_dict(value)
    except CapabilityCourseContractError:
        raise
    except Exception as error:
        raise CapabilityCourseContractError("course manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise CapabilityCourseContractError("course manifest 非规范字节")
    return manifest


__all__ = [
    "COURSE_EXECUTION_STATE",
    "COURSE_INVARIANTS",
    "COURSE_SPLIT_AXES",
    "COURSE_SPLITS",
    "CapabilityCourseContractError",
    "CapabilityCourseManifest",
    "read_capability_course_manifest",
    "write_capability_course_manifest",
]
