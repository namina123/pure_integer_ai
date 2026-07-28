"""MD-02 篇章状态三件套 adapter 的不可覆盖审计 manifest。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    zero_execution_state,
)


FORMAT_VERSION = 1
MD02_ADAPTER_KEYS = (
    "CURRENT_SITUATION_PROJECTION",
    "SITUATION_DEPENDENCY_INDEX",
    "SITUATION_EVENT_LOG",
    "SITUATION_REBUILD_RECEIPT",
)
MD02_INVARIANT_KEYS = (
    "HOST_LEARNING_WRITE_ZERO",
    "LOCAL_DEPENDENCY_INVALIDATION",
    "NO_SECOND_MEMORY_DATASET_EVIDENCE_ENGINE",
    "ORIGINAL_EVENT_APPEND_ONLY",
    "OWNER_SCOPE_VERSION_FAIL_CLOSED",
    "UNAFFECTED_PROJECTION_BIT_IDENTITY",
    "WORK_MEMORY_TRANSIENT_ONLY",
)
MD02_VERIFIER_DIMENSIONS = (
    "BACKING_EVENT_IDENTITY",
    "DEPENDENCY_INDEX_REBUILD",
    "LOCAL_INVALIDATION_SCOPE",
    "ORIGINAL_EVENT_PRESERVATION",
    "OWNER_SCOPE_VERSION_ISOLATION",
    "UNAFFECTED_PROJECTION_BIT_IDENTITY",
    "ZERO_HOST_LEARNING_WRITE",
)
MD02_NE_CONDITIONS = (
    "CENTER_DIFFUSION_QUALITY_REQUESTED",
    "MD03_DIRECTIONAL_ADAPTER_NOT_EXECUTED",
    "MD04_PROBE_NOT_EXECUTED",
    "RETENTION_GENERALIZATION_REQUESTED",
)


class MD02ManifestError(RuntimeError):
    """MD-02 manifest 字段、前置或零执行边界不完整。"""


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段集合精确相等。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise MD02ManifestError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求无首尾空白的非空文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MD02ManifestError(f"{where} 必须是非空无首尾空白文本")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256。"""
    text = _text(value, where=where)
    if (len(text) != 64
            or any(char not in "0123456789abcdef" for char in text)):
        raise MD02ManifestError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可迁移的安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise MD02ManifestError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _text_tuple(
        value: tuple[str, ...],
        *,
        where: str,
        expected: tuple[str, ...] | None = None,
        ) -> tuple[str, ...]:
    """要求文本 tuple 稳定有序、唯一，可选精确集合。"""
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in value)):
        raise MD02ManifestError(f"{where} 必须是非空文本 tuple")
    if value != tuple(sorted(value)) or len(set(value)) != len(value):
        raise MD02ManifestError(f"{where} 必须稳定有序且唯一")
    if expected is not None and value != expected:
        raise MD02ManifestError(f"{where} 未精确列全")
    return value


@dataclass(frozen=True)
class MD02AdapterManifest:
    """冻结 MD-02 实现边界、复用对象、T0 维度和 probe 盲区。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_keys: tuple[str, ...]
    md01_manifest_relative_path: str
    md01_manifest_sha256: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    adapter_type_keys: tuple[str, ...]
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
            raise MD02ManifestError("MD-02 format_version 非法")
        _text(self.artifact_version, where="MD-02 artifact version")
        if self.artifact_status != "ADAPTER_FROZEN":
            raise MD02ManifestError("MD-02 artifact status 非法")
        if self.task_keys != ("MD-02",):
            raise MD02ManifestError("MD-02 task keys 非法")
        _relative_path(self.md01_manifest_relative_path, where="MD-01 path")
        _sha256(self.md01_manifest_sha256, where="MD-01 hash")
        _relative_path(self.baseline_manifest_relative_path, where="baseline path")
        _sha256(self.baseline_manifest_sha256, where="baseline hash")
        _text_tuple(
            self.adapter_type_keys,
            where="MD-02 adapter types",
            expected=MD02_ADAPTER_KEYS,
        )
        _text_tuple(
            self.invariant_keys,
            where="MD-02 invariants",
            expected=MD02_INVARIANT_KEYS,
        )
        _text_tuple(self.reused_component_refs, where="MD-02 reused refs")
        _text_tuple(
            self.verifier_dimensions,
            where="MD-02 verifier dimensions",
            expected=MD02_VERIFIER_DIMENSIONS,
        )
        _text_tuple(
            self.verifier_ne_conditions,
            where="MD-02 verifier NE",
            expected=MD02_NE_CONDITIONS,
        )
        if self.adapter_status != "T0_EVIDENCED":
            raise MD02ManifestError("MD-02 adapter status 非法")
        if self.probe_status != "NOT_STARTED":
            raise MD02ManifestError("MD-02 不得冒充 probe 已运行")
        if type(self.results_observed) is not int or self.results_observed != 0:
            raise MD02ManifestError("MD-02 results observed 必须为 0")
        if (type(self.host_learning_write_count) is not int
                or self.host_learning_write_count != 0):
            raise MD02ManifestError("MD-02 host learning write 必须为 0")
        state = self.execution_state.to_value()
        expected_state = zero_execution_state().to_value()
        if tuple(state) != tuple(expected_state):
            raise MD02ManifestError("MD-02 execution state 字段不精确")
        if state != expected_state:
            raise MD02ManifestError("MD-02 execution state 必须全零")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 MD-02 manifest。"""
        return {
            "adapter_status": self.adapter_status,
            "adapter_type_keys": list(self.adapter_type_keys),
            "artifact_kind": "PH2_MD02_SITUATION_STATE_ADAPTER",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "baseline_manifest_relative_path": (
                self.baseline_manifest_relative_path),
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "host_learning_write_count": self.host_learning_write_count,
            "invariant_keys": list(self.invariant_keys),
            "md01_manifest_relative_path": self.md01_manifest_relative_path,
            "md01_manifest_sha256": self.md01_manifest_sha256,
            "probe_status": self.probe_status,
            "results_observed": self.results_observed,
            "reused_component_refs": list(self.reused_component_refs),
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD02AdapterManifest":
        """从精确 JSON object 恢复 MD-02 manifest。"""
        raw = _exact_keys(value, {
            "adapter_status", "adapter_type_keys", "artifact_kind",
            "artifact_status", "artifact_version",
            "baseline_manifest_relative_path", "baseline_manifest_sha256",
            "execution_state", "format_version", "host_learning_write_count",
            "invariant_keys", "md01_manifest_relative_path",
            "md01_manifest_sha256", "probe_status", "results_observed",
            "reused_component_refs", "task_keys", "verifier_dimensions",
            "verifier_ne_conditions",
        }, where="MD02AdapterManifest")
        if raw["artifact_kind"] != "PH2_MD02_SITUATION_STATE_ADAPTER":
            raise MD02ManifestError("MD-02 artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            str(raw["md01_manifest_relative_path"]),
            str(raw["md01_manifest_sha256"]),
            str(raw["baseline_manifest_relative_path"]),
            str(raw["baseline_manifest_sha256"]),
            tuple(str(item) for item in raw["adapter_type_keys"]),
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


def build_md02_adapter_manifest(
        *,
        md01_manifest_relative_path: str,
        md01_manifest_sha256: str,
        baseline_manifest_relative_path: str,
        baseline_manifest_sha256: str,
        ) -> MD02AdapterManifest:
    """绑定 MD-01/v18 基线并冻结三件套 adapter 的 T0 范围。"""
    return MD02AdapterManifest(
        FORMAT_VERSION,
        "MD-02-situation-state-adapter-v1",
        "ADAPTER_FROZEN",
        ("MD-02",),
        md01_manifest_relative_path,
        md01_manifest_sha256,
        baseline_manifest_relative_path,
        baseline_manifest_sha256,
        MD02_ADAPTER_KEYS,
        MD02_INVARIANT_KEYS,
        tuple(sorted((
            "src/pure_integer_ai/cognition/shared/attractor_state.py",
            "src/pure_integer_ai/cognition/shared/memory_event_log.py",
            "src/pure_integer_ai/cognition/shared/parser_revision.py",
            "src/pure_integer_ai/cognition/shared/work_memory_content.py",
            "src/pure_integer_ai/cognition/shared/work_memory_discourse.py",
            "src/pure_integer_ai/cognition/understanding/memory_intake.py",
        ))),
        MD02_VERIFIER_DIMENSIONS,
        MD02_NE_CONDITIONS,
        "T0_EVIDENCED",
        "NOT_STARTED",
        0,
        0,
        zero_execution_state(),
    )


def write_md02_adapter_manifest(
        manifest: MD02AdapterManifest,
        path: str | Path,
        ) -> Path:
    """独占或逐字节幂等发布 MD-02 manifest。"""
    if not isinstance(manifest, MD02AdapterManifest):
        raise MD02ManifestError("MD-02 manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise MD02ManifestError("MD-02 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise MD02ManifestError("MD-02 manifest 无法发布") from error
    return target


def read_md02_adapter_manifest(path: str | Path) -> MD02AdapterManifest:
    """严格回读规范 MD-02 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MD02ManifestError("MD-02 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = MD02AdapterManifest.from_dict(value)
    except MD02ManifestError:
        raise
    except Exception as error:
        raise MD02ManifestError("MD-02 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise MD02ManifestError("MD-02 manifest 非规范字节")
    return manifest


__all__ = [
    "MD02AdapterManifest",
    "MD02ManifestError",
    "MD02_ADAPTER_KEYS",
    "MD02_INVARIANT_KEYS",
    "MD02_NE_CONDITIONS",
    "MD02_VERIFIER_DIMENSIONS",
    "build_md02_adapter_manifest",
    "read_md02_adapter_manifest",
    "write_md02_adapter_manifest",
]
