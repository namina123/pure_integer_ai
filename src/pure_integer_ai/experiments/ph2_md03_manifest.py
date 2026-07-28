"""MD-03 三向 center adapter 的不可覆盖审计 manifest。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import PAYLOAD_KINDS
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    CENTER_STRENGTHS,
    DIRECTIONS,
    zero_execution_state,
)


FORMAT_VERSION = 1
MD03_ADAPTER_KEYS = (
    "DIRECTIONAL_CENTER_PROFILE",
    "DIRECTIONAL_MEMORY_CENTER",
    "DIRECTIONAL_WRITE_BOUNDARY",
    "MEMORY_CENTER_FORMATION_REPORT",
)
MD03_INVARIANT_KEYS = (
    "ACTIVATION_NOT_ADOPTION",
    "DIFFERENT_BOUNDARY_PERMISSION_NOT_MERGED",
    "EXACT_DEDUP_PRESERVES_ORIGINS_DEPENDENCIES",
    "HOST_LEARNING_WRITE_ZERO",
    "NO_COMMITTER_IN_ADAPTER",
    "OWNER_SCOPE_VERSION_FAIL_CLOSED",
    "SAME_QUERY_THREE_DISTINCT_PAYLOADS",
)
MD03_VERIFIER_DIMENSIONS = (
    "ACTIVATION_ADOPTION_SEPARATION",
    "DIRECTIONAL_PAYLOAD_DISTINCTION",
    "EXACT_DEDUP_AND_PROVENANCE_MERGE",
    "OWNER_SCOPE_VERSION_ISOLATION",
    "STRENGTH_PRESERVATION",
    "WRITE_PERMISSION_ORTHOGONALITY",
    "ZERO_HOST_LEARNING_WRITE",
)
MD03_NE_CONDITIONS = (
    "CENTER_DIFFUSION_QUALITY_REQUESTED",
    "MD04_PROBE_NOT_EXECUTED",
    "RETENTION_GENERALIZATION_REQUESTED",
    "RUNTIME_CONSUMPTION_REQUESTED",
)


class MD03ManifestError(RuntimeError):
    """MD-03 manifest 字段、前置或零执行事实不完整。"""


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段集合精确相等。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise MD03ManifestError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求无首尾空白的非空文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MD03ManifestError(f"{where} 必须是非空无首尾空白文本")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256。"""
    text = _text(value, where=where)
    if (len(text) != 64
            or any(char not in "0123456789abcdef" for char in text)):
        raise MD03ManifestError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可迁移的安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise MD03ManifestError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _text_tuple(
        value: tuple[str, ...],
        *,
        where: str,
        expected: tuple[str, ...] | None = None,
        ) -> tuple[str, ...]:
    """要求文本 tuple 稳定有序、唯一，可选精确列全。"""
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in value)):
        raise MD03ManifestError(f"{where} 必须是非空文本 tuple")
    if value != tuple(sorted(value)) or len(set(value)) != len(value):
        raise MD03ManifestError(f"{where} 必须稳定有序且唯一")
    if expected is not None and value != expected:
        raise MD03ManifestError(f"{where} 未精确列全")
    return value


@dataclass(frozen=True)
class MD03AdapterManifest:
    """冻结三向 payload、写权限、去重边界、前置和 probe 盲区。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_keys: tuple[str, ...]
    md02_manifest_relative_path: str
    md02_manifest_sha256: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    adapter_type_keys: tuple[str, ...]
    direction_keys: tuple[str, ...]
    payload_kind_keys: tuple[str, ...]
    strength_keys: tuple[str, ...]
    invariant_keys: tuple[str, ...]
    reused_component_refs: tuple[str, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    adapter_status: str
    probe_status: str
    results_observed: int
    host_learning_write_count: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise MD03ManifestError("MD-03 format_version 非法")
        _text(self.artifact_version, where="MD-03 artifact version")
        if self.artifact_status != "ADAPTER_FROZEN":
            raise MD03ManifestError("MD-03 artifact status 非法")
        if self.task_keys != ("MD-03",):
            raise MD03ManifestError("MD-03 task keys 非法")
        _relative_path(self.md02_manifest_relative_path, where="MD-02 path")
        _sha256(self.md02_manifest_sha256, where="MD-02 hash")
        _relative_path(self.baseline_manifest_relative_path, where="baseline path")
        _sha256(self.baseline_manifest_sha256, where="baseline hash")
        for actual, expected, label in (
                (self.adapter_type_keys, MD03_ADAPTER_KEYS, "adapter types"),
                (self.direction_keys, DIRECTIONS, "directions"),
                (self.payload_kind_keys, PAYLOAD_KINDS, "payload kinds"),
                (self.strength_keys, CENTER_STRENGTHS, "strengths"),
                (self.invariant_keys, MD03_INVARIANT_KEYS, "invariants"),
                (self.verifier_dimensions, MD03_VERIFIER_DIMENSIONS,
                 "verifier dimensions"),
                (self.verifier_ne_conditions, MD03_NE_CONDITIONS,
                 "verifier NE")):
            _text_tuple(actual, where=f"MD-03 {label}", expected=expected)
        _text_tuple(self.reused_component_refs, where="MD-03 reused refs")
        if self.adapter_status != "T0_EVIDENCED":
            raise MD03ManifestError("MD-03 adapter status 非法")
        if self.probe_status != "NOT_STARTED":
            raise MD03ManifestError("MD-03 不得冒充 probe 已运行")
        if type(self.results_observed) is not int or self.results_observed != 0:
            raise MD03ManifestError("MD-03 results observed 必须为 0")
        if (type(self.host_learning_write_count) is not int
                or self.host_learning_write_count != 0):
            raise MD03ManifestError("MD-03 host learning write 必须为 0")
        state = self.execution_state.to_value()
        expected_state = zero_execution_state().to_value()
        if tuple(state) != tuple(expected_state) or state != expected_state:
            raise MD03ManifestError("MD-03 execution state 必须精确全零")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 MD-03 manifest。"""
        return {
            "adapter_status": self.adapter_status,
            "adapter_type_keys": list(self.adapter_type_keys),
            "artifact_kind": "PH2_MD03_DIRECTIONAL_CENTER_ADAPTER",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "baseline_manifest_relative_path": (
                self.baseline_manifest_relative_path),
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "direction_keys": list(self.direction_keys),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "host_learning_write_count": self.host_learning_write_count,
            "invariant_keys": list(self.invariant_keys),
            "md02_manifest_relative_path": self.md02_manifest_relative_path,
            "md02_manifest_sha256": self.md02_manifest_sha256,
            "payload_kind_keys": list(self.payload_kind_keys),
            "probe_status": self.probe_status,
            "results_observed": self.results_observed,
            "reused_component_refs": list(self.reused_component_refs),
            "strength_keys": list(self.strength_keys),
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD03AdapterManifest":
        """从精确 JSON object 恢复 MD-03 manifest。"""
        raw = _exact_keys(value, {
            "adapter_status", "adapter_type_keys", "artifact_kind",
            "artifact_status", "artifact_version",
            "baseline_manifest_relative_path", "baseline_manifest_sha256",
            "direction_keys", "execution_state", "format_version",
            "host_learning_write_count", "invariant_keys",
            "md02_manifest_relative_path", "md02_manifest_sha256",
            "payload_kind_keys", "probe_status", "results_observed",
            "reused_component_refs", "strength_keys", "task_keys",
            "verifier_dimensions", "verifier_ne_conditions",
        }, where="MD03AdapterManifest")
        if raw["artifact_kind"] != "PH2_MD03_DIRECTIONAL_CENTER_ADAPTER":
            raise MD03ManifestError("MD-03 artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            str(raw["md02_manifest_relative_path"]),
            str(raw["md02_manifest_sha256"]),
            str(raw["baseline_manifest_relative_path"]),
            str(raw["baseline_manifest_sha256"]),
            tuple(str(item) for item in raw["adapter_type_keys"]),
            tuple(str(item) for item in raw["direction_keys"]),
            tuple(str(item) for item in raw["payload_kind_keys"]),
            tuple(str(item) for item in raw["strength_keys"]),
            tuple(str(item) for item in raw["invariant_keys"]),
            tuple(str(item) for item in raw["reused_component_refs"]),
            tuple(str(item) for item in raw["verifier_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            str(raw["adapter_status"]),
            str(raw["probe_status"]),
            raw["results_observed"],
            raw["host_learning_write_count"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        """返回规范单行 JSON 字节。"""
        return canonical_json_line(self.to_dict())


def build_md03_adapter_manifest(
        *,
        md02_manifest_relative_path: str,
        md02_manifest_sha256: str,
        baseline_manifest_relative_path: str,
        baseline_manifest_sha256: str,
        ) -> MD03AdapterManifest:
    """绑定 MD-02/v19 并冻结三向 adapter 的 T0 事实。"""
    return MD03AdapterManifest(
        FORMAT_VERSION,
        "MD-03-directional-center-adapter-v1",
        "ADAPTER_FROZEN",
        ("MD-03",),
        md02_manifest_relative_path,
        md02_manifest_sha256,
        baseline_manifest_relative_path,
        baseline_manifest_sha256,
        MD03_ADAPTER_KEYS,
        DIRECTIONS,
        PAYLOAD_KINDS,
        CENTER_STRENGTHS,
        MD03_INVARIANT_KEYS,
        tuple(sorted((
            "src/pure_integer_ai/cognition/shared/generation_plan.py",
            "src/pure_integer_ai/cognition/shared/memory_query.py",
            "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
            "src/pure_integer_ai/experiments/attractor_runtime.py",
            "src/pure_integer_ai/experiments/memory_generation_runtime.py",
            "src/pure_integer_ai/experiments/memory_query_runtime.py",
            "src/pure_integer_ai/experiments/ph2_memory_dynamics_contract.py",
        ))),
        MD03_VERIFIER_DIMENSIONS,
        MD03_NE_CONDITIONS,
        "T0_EVIDENCED",
        "NOT_STARTED",
        0,
        0,
        zero_execution_state(),
    )


def write_md03_adapter_manifest(
        manifest: MD03AdapterManifest,
        path: str | Path,
        ) -> Path:
    """独占或逐字节幂等发布 MD-03 manifest。"""
    if not isinstance(manifest, MD03AdapterManifest):
        raise MD03ManifestError("MD-03 manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise MD03ManifestError("MD-03 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise MD03ManifestError("MD-03 manifest 无法发布") from error
    return target


def read_md03_adapter_manifest(path: str | Path) -> MD03AdapterManifest:
    """严格回读规范 MD-03 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MD03ManifestError("MD-03 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = MD03AdapterManifest.from_dict(value)
    except MD03ManifestError:
        raise
    except Exception as error:
        raise MD03ManifestError("MD-03 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise MD03ManifestError("MD-03 manifest 非规范字节")
    return manifest


__all__ = [
    "MD03AdapterManifest",
    "MD03ManifestError",
    "MD03_ADAPTER_KEYS",
    "MD03_INVARIANT_KEYS",
    "MD03_NE_CONDITIONS",
    "MD03_VERIFIER_DIMENSIONS",
    "build_md03_adapter_manifest",
    "read_md03_adapter_manifest",
    "write_md03_adapter_manifest",
]
