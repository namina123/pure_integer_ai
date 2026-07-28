"""LC-13 三向 consumer、exact Use/outcome 与分层 postcheck 纯合同。"""
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
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_STATUS = "COURSE_FROZEN"
RUNTIME_STATUS = "NOT_STARTED"
DIRECTIONS = ("GENERATION", "REASONING", "UNDERSTANDING")
APPLICABILITY_STATES = ("N_A", "REQUIRED")
COURSE_FACT_STATES = ("ABSENT", "COURSE_FROZEN", "DESIGNED", "OUT_OF_SCOPE")
CONSUMER_STATES = (
    "AVAILABLE_NOT_EXECUTED",
    "MISSING_NE",
    "OUT_OF_SCOPE",
)
EXACT_USE_OUTCOME_STATES = (
    "CONTRACT_FROZEN_NOT_CONSUMED",
    "OUT_OF_SCOPE",
    "REQUIRED_NOT_CONNECTED",
)
POSTCHECK_STATES = (
    "AVAILABLE_NOT_EXECUTED",
    "OUT_OF_SCOPE",
    "REQUIRED_NOT_CONNECTED",
)
DIRECTIONAL_VERDICTS = ("NE", "OUT_OF_SCOPE")
POSTCHECK_DIMENSIONS = {
    "GENERATION": (
        "ADDRESSEE_RECOVERABILITY",
        "LAYER_OUTCOME_LOCALITY",
        "SEMANTIC_CONTENT",
        "SOURCE_UNCERTAINTY",
        "ZERO_HOST_LEARNING_WRITE",
    ),
    "REASONING": (
        "CURRENT_PROJECTION",
        "PREMISE_SCOPE",
        "PROOF_DIRECTION",
        "ZERO_HOST_LEARNING_WRITE",
    ),
    "UNDERSTANDING": (
        "OBJECT_IDENTITY",
        "SOURCE_SCOPE",
        "UNKNOWN_AMBIGUITY",
        "ZERO_HOST_LEARNING_WRITE",
    ),
}
VERIFIER_DIMENSIONS = (
    "ALL_CAPABILITY_DIRECTIONS_INVENTORIED",
    "APPLICABILITY_EXPLICIT",
    "CONSUMER_FILE_IDENTITY",
    "CONSUMER_STATE_HONEST",
    "DIRECTION_OWNER_EXPLICIT",
    "EXACT_USE_OUTCOME_STATE_EXPLICIT",
    "LAYERED_POSTCHECK_ROUTE",
    "NO_CROSS_DIRECTION_BROADCAST",
    "NE_FOR_MISSING_CONSUMER",
    "WRITE_PERMISSION_LEAST_PRIVILEGE",
    "ZERO_HOST_LEARNING_WRITE",
)
VERIFIER_NE_CONDITIONS = (
    "ASSESSMENT_CONSUMER_NOT_CONNECTED",
    "DIRECTIONAL_CONSUMER_NOT_CONNECTED",
    "FORMAL_W_RUNTIME_NOT_STARTED",
    "LC13_ROUTE_NOT_EXECUTED",
    "NON_TEXT_WALL_OUT_OF_SCOPE",
    "POSTCHECK_NOT_EXECUTED",
)
EXECUTION_STATE = {
    "assessment_updates": 0,
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "directional_runtime_runs": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "postcheck_runtime_runs": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}


class DirectionalConsumerContractError(RuntimeError):
    """LC-13 路由、文件身份、NE 或零运行边界不满足冻结合同。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise DirectionalConsumerContractError(f"{where} 必须是规范文本")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise DirectionalConsumerContractError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise DirectionalConsumerContractError(f"{where} 必须是 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise DirectionalConsumerContractError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise DirectionalConsumerContractError(f"{where} 必须是安全相对路径")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise DirectionalConsumerContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise DirectionalConsumerContractError(f"{where} 不能为空")
    for item in value:
        _text(item, where=where)
    if tuple(sorted(set(value))) != value:
        raise DirectionalConsumerContractError(f"{where} 必须排序去重")
    return value


@dataclass(frozen=True)
class DirectionalConsumerRoute:
    """一个能力在一个方向上的 applicability、consumer 和 postcheck。"""

    capability_key: str
    direction: str
    applicability: str
    course_fact_state: str
    consumer_state: str
    consumer_refs: tuple[str, ...]
    owner_key: str
    write_permissions: tuple[str, ...]
    exact_use_outcome_state: str
    postcheck_key: str
    postcheck_state: str
    postcheck_dimensions: tuple[str, ...]
    directional_verdict: str
    ne_conditions: tuple[str, ...]
    host_learning_writes: int

    def __post_init__(self) -> None:
        if self.capability_key not in CAPABILITY_KEYS:
            raise DirectionalConsumerContractError("capability_key 未登记")
        if self.direction not in DIRECTIONS:
            raise DirectionalConsumerContractError("direction 未登记")
        if self.applicability not in APPLICABILITY_STATES:
            raise DirectionalConsumerContractError("applicability 未登记")
        if self.course_fact_state not in COURSE_FACT_STATES:
            raise DirectionalConsumerContractError("course_fact_state 未登记")
        if self.consumer_state not in CONSUMER_STATES:
            raise DirectionalConsumerContractError("consumer_state 未登记")
        _strict_text_tuple(
            self.consumer_refs, where="consumer_refs", allow_empty=True)
        for item in self.consumer_refs:
            _relative_path(item, where="consumer_ref")
        _text(self.owner_key, where="owner_key")
        _strict_text_tuple(
            self.write_permissions, where="write_permissions",
            allow_empty=True)
        if self.exact_use_outcome_state not in EXACT_USE_OUTCOME_STATES:
            raise DirectionalConsumerContractError(
                "exact_use_outcome_state 未登记")
        _text(self.postcheck_key, where="postcheck_key")
        if self.postcheck_state not in POSTCHECK_STATES:
            raise DirectionalConsumerContractError("postcheck_state 未登记")
        if self.postcheck_dimensions != POSTCHECK_DIMENSIONS[self.direction]:
            raise DirectionalConsumerContractError("postcheck_dimensions 漂移")
        if self.directional_verdict not in DIRECTIONAL_VERDICTS:
            raise DirectionalConsumerContractError("directional_verdict 未登记")
        _strict_text_tuple(self.ne_conditions, where="ne_conditions")
        _nonnegative(self.host_learning_writes, where="host_learning_writes")
        if self.host_learning_writes != 0:
            raise DirectionalConsumerContractError("LC-13 禁止宿主学习写")

        if self.applicability == "N_A":
            if not (
                    self.course_fact_state == "OUT_OF_SCOPE"
                    and self.consumer_state == "OUT_OF_SCOPE"
                    and not self.consumer_refs
                    and self.owner_key == "OUT_OF_SCOPE"
                    and not self.write_permissions
                    and self.exact_use_outcome_state == "OUT_OF_SCOPE"
                    and self.postcheck_state == "OUT_OF_SCOPE"
                    and self.directional_verdict == "OUT_OF_SCOPE"
                    and self.ne_conditions == (
                        "NON_TEXT_WALL_OUT_OF_SCOPE",)):
                raise DirectionalConsumerContractError("N_A 路由边界漂移")
            return
        if self.directional_verdict != "NE":
            raise DirectionalConsumerContractError(
                "D-03 前方向能力 verdict 必须为 NE")
        if "FORMAL_W_RUNTIME_NOT_STARTED" not in self.ne_conditions:
            raise DirectionalConsumerContractError("方向路由缺 W runtime NE")
        if self.consumer_state == "AVAILABLE_NOT_EXECUTED":
            if (not self.consumer_refs
                    or self.owner_key in {"OUT_OF_SCOPE", "UNASSIGNED_NE"}
                    or "NO_HOST_LEARNING_WRITE" not in self.write_permissions
                    or self.postcheck_state != "AVAILABLE_NOT_EXECUTED"
                    or "LC13_ROUTE_NOT_EXECUTED" not in self.ne_conditions
                    or "POSTCHECK_NOT_EXECUTED" not in self.ne_conditions):
                raise DirectionalConsumerContractError(
                    "可用 consumer 的零执行边界不完整")
        elif self.consumer_state == "MISSING_NE":
            if (self.consumer_refs or self.owner_key != "UNASSIGNED_NE"
                    or self.write_permissions
                    or self.exact_use_outcome_state
                    != "REQUIRED_NOT_CONNECTED"
                    or self.postcheck_state != "REQUIRED_NOT_CONNECTED"
                    or "DIRECTIONAL_CONSUMER_NOT_CONNECTED"
                    not in self.ne_conditions):
                raise DirectionalConsumerContractError(
                    "缺失 consumer 未诚实登记 NE")
        else:
            raise DirectionalConsumerContractError("REQUIRED 路由状态非法")
        if (self.exact_use_outcome_state
                == "CONTRACT_FROZEN_NOT_CONSUMED"):
            if not (
                    self.capability_key == "LAYERED_GENERATION"
                    and self.direction == "GENERATION"
                    and "ASSESSMENT_CONSUMER_NOT_CONNECTED"
                    in self.ne_conditions):
                raise DirectionalConsumerContractError(
                    "exact Use/outcome 合同权限越界")
        elif (self.consumer_state == "AVAILABLE_NOT_EXECUTED"
              and self.exact_use_outcome_state != "REQUIRED_NOT_CONNECTED"):
            raise DirectionalConsumerContractError(
                "其他设施不得冒充 exact Use/outcome 已冻结")

    @property
    def route_key(self) -> str:
        return f"{self.capability_key}/{self.direction}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicability": self.applicability,
            "capability_key": self.capability_key,
            "consumer_refs": list(self.consumer_refs),
            "consumer_state": self.consumer_state,
            "course_fact_state": self.course_fact_state,
            "direction": self.direction,
            "directional_verdict": self.directional_verdict,
            "exact_use_outcome_state": self.exact_use_outcome_state,
            "host_learning_writes": self.host_learning_writes,
            "ne_conditions": list(self.ne_conditions),
            "owner_key": self.owner_key,
            "postcheck_dimensions": list(self.postcheck_dimensions),
            "postcheck_key": self.postcheck_key,
            "postcheck_state": self.postcheck_state,
            "route_key": self.route_key,
            "write_permissions": list(self.write_permissions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DirectionalConsumerRoute":
        expected = {
            "applicability", "capability_key", "consumer_refs",
            "consumer_state", "course_fact_state", "direction",
            "directional_verdict", "exact_use_outcome_state",
            "host_learning_writes", "ne_conditions", "owner_key",
            "postcheck_dimensions", "postcheck_key", "postcheck_state",
            "route_key", "write_permissions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise DirectionalConsumerContractError(
                "DirectionalConsumerRoute 字段不精确")
        result = cls(
            str(value["capability_key"]),
            str(value["direction"]),
            str(value["applicability"]),
            str(value["course_fact_state"]),
            str(value["consumer_state"]),
            tuple(str(item) for item in value["consumer_refs"]),
            str(value["owner_key"]),
            tuple(str(item) for item in value["write_permissions"]),
            str(value["exact_use_outcome_state"]),
            str(value["postcheck_key"]),
            str(value["postcheck_state"]),
            tuple(str(item) for item in value["postcheck_dimensions"]),
            str(value["directional_verdict"]),
            tuple(str(item) for item in value["ne_conditions"]),
            value["host_learning_writes"],
        )
        if value["route_key"] != result.route_key:
            raise DirectionalConsumerContractError("route_key 漂移")
        return result


@dataclass(frozen=True)
class DirectionalConsumerEvidenceFile:
    """一个现有 consumer 文件的固定身份。"""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise DirectionalConsumerContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any],
            ) -> "DirectionalConsumerEvidenceFile":
        if not isinstance(value, dict) or set(value) != {
                "byte_count", "relative_path", "sha256"}:
            raise DirectionalConsumerContractError(
                "DirectionalConsumerEvidenceFile 字段不精确")
        return cls(
            str(value["relative_path"]),
            value["byte_count"],
            str(value["sha256"]),
        )


@dataclass(frozen=True)
class DirectionalConsumerManifest:
    """LC-13 的 20×3 方向账、consumer 文件身份和零运行 artifact。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    runtime_status: str
    task_key: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    routes: tuple[DirectionalConsumerRoute, ...]
    evidence_files: tuple[DirectionalConsumerEvidenceFile, ...]
    route_count: int
    available_not_executed_count: int
    missing_ne_count: int
    out_of_scope_count: int
    exact_use_outcome_contract_count: int
    runtime_connected_count: int
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise DirectionalConsumerContractError("format_version 漂移")
        _text(self.artifact_version, where="artifact_version")
        if self.artifact_status != ARTIFACT_STATUS:
            raise DirectionalConsumerContractError("artifact_status 非 COURSE_FROZEN")
        if self.runtime_status != RUNTIME_STATUS:
            raise DirectionalConsumerContractError("runtime_status 非 NOT_STARTED")
        if self.task_key != "LC-13":
            raise DirectionalConsumerContractError("task_key 非 LC-13")
        _relative_path(
            self.baseline_manifest_relative_path,
            where="baseline_manifest_relative_path")
        _sha256(self.baseline_manifest_sha256, where="baseline_manifest_sha256")
        if (not isinstance(self.routes, tuple)
                or not all(isinstance(item, DirectionalConsumerRoute)
                           for item in self.routes)):
            raise DirectionalConsumerContractError("routes 类型非法")
        routes = tuple(sorted(
            self.routes, key=lambda item: (item.capability_key, item.direction)))
        object.__setattr__(self, "routes", routes)
        expected_keys = tuple(
            f"{capability}/{direction}"
            for capability in CAPABILITY_KEYS for direction in DIRECTIONS)
        if tuple(item.route_key for item in routes) != expected_keys:
            raise DirectionalConsumerContractError("20×3 方向 inventory 未闭合")
        if (not isinstance(self.evidence_files, tuple)
                or not all(isinstance(item, DirectionalConsumerEvidenceFile)
                           for item in self.evidence_files)):
            raise DirectionalConsumerContractError("evidence_files 类型非法")
        evidence = tuple(sorted(
            self.evidence_files, key=lambda item: item.relative_path))
        object.__setattr__(self, "evidence_files", evidence)
        if len({item.relative_path for item in evidence}) != len(evidence):
            raise DirectionalConsumerContractError("evidence file 重复")
        referenced = {path for route in routes for path in route.consumer_refs}
        inventoried = {item.relative_path for item in evidence}
        if referenced != inventoried:
            raise DirectionalConsumerContractError(
                "consumer refs 与文件 inventory 未闭合")
        actual_counts = {
            "route_count": len(routes),
            "available_not_executed_count": sum(
                item.consumer_state == "AVAILABLE_NOT_EXECUTED"
                for item in routes),
            "missing_ne_count": sum(
                item.consumer_state == "MISSING_NE" for item in routes),
            "out_of_scope_count": sum(
                item.consumer_state == "OUT_OF_SCOPE" for item in routes),
            "exact_use_outcome_contract_count": sum(
                item.exact_use_outcome_state
                == "CONTRACT_FROZEN_NOT_CONSUMED" for item in routes),
            "runtime_connected_count": 0,
        }
        for name, actual in actual_counts.items():
            if _nonnegative(getattr(self, name), where=name) != actual:
                raise DirectionalConsumerContractError(f"{name} 漂移")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise DirectionalConsumerContractError("verifier_dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise DirectionalConsumerContractError("verifier_ne_conditions 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise DirectionalConsumerContractError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_LC13_DIRECTIONAL_CONSUMER_MANIFEST",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "available_not_executed_count": self.available_not_executed_count,
            "baseline_manifest_relative_path": (
                self.baseline_manifest_relative_path),
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "exact_use_outcome_contract_count": (
                self.exact_use_outcome_contract_count),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "missing_ne_count": self.missing_ne_count,
            "out_of_scope_count": self.out_of_scope_count,
            "route_count": self.route_count,
            "routes": [item.to_dict() for item in self.routes],
            "runtime_connected_count": self.runtime_connected_count,
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
    def from_dict(cls, value: dict[str, Any]) -> "DirectionalConsumerManifest":
        expected = {
            "artifact_kind", "artifact_status", "artifact_version",
            "available_not_executed_count", "baseline_manifest_relative_path",
            "baseline_manifest_sha256", "evidence_files",
            "exact_use_outcome_contract_count", "execution_state",
            "format_version", "missing_ne_count", "out_of_scope_count",
            "route_count", "routes", "runtime_connected_count",
            "runtime_status", "task_key", "verifier_dimensions",
            "verifier_ne_conditions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise DirectionalConsumerContractError(
                "DirectionalConsumerManifest 字段不精确")
        if value["artifact_kind"] != "PH2_LC13_DIRECTIONAL_CONSUMER_MANIFEST":
            raise DirectionalConsumerContractError("artifact_kind 非法")
        return cls(
            value["format_version"],
            str(value["artifact_version"]),
            str(value["artifact_status"]),
            str(value["runtime_status"]),
            str(value["task_key"]),
            str(value["baseline_manifest_relative_path"]),
            str(value["baseline_manifest_sha256"]),
            tuple(DirectionalConsumerRoute.from_dict(item)
                  for item in value["routes"]),
            tuple(DirectionalConsumerEvidenceFile.from_dict(item)
                  for item in value["evidence_files"]),
            value["route_count"],
            value["available_not_executed_count"],
            value["missing_ne_count"],
            value["out_of_scope_count"],
            value["exact_use_outcome_contract_count"],
            value["runtime_connected_count"],
            tuple(str(item) for item in value["verifier_dimensions"]),
            tuple(str(item) for item in value["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(value["execution_state"]),
        )


def write_directional_consumer_manifest(
        manifest: DirectionalConsumerManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 LC-13 manifest，拒绝覆盖不同内容。"""
    if not isinstance(manifest, DirectionalConsumerManifest):
        raise DirectionalConsumerContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise DirectionalConsumerContractError(
                "LC-13 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise DirectionalConsumerContractError("LC-13 manifest 无法发布") from error
    return target


def read_directional_consumer_manifest(
        path: str | Path,
        ) -> DirectionalConsumerManifest:
    """严格回读规范 LC-13 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise DirectionalConsumerContractError("LC-13 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = DirectionalConsumerManifest.from_dict(value)
    except DirectionalConsumerContractError:
        raise
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        raise DirectionalConsumerContractError("LC-13 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise DirectionalConsumerContractError("LC-13 manifest 非规范 JSON")
    return manifest


__all__ = [
    "APPLICABILITY_STATES",
    "ARTIFACT_STATUS",
    "CONSUMER_STATES",
    "COURSE_FACT_STATES",
    "DIRECTIONS",
    "DIRECTIONAL_VERDICTS",
    "DirectionalConsumerContractError",
    "DirectionalConsumerEvidenceFile",
    "DirectionalConsumerManifest",
    "DirectionalConsumerRoute",
    "EXACT_USE_OUTCOME_STATES",
    "EXECUTION_STATE",
    "FORMAT_VERSION",
    "POSTCHECK_DIMENSIONS",
    "POSTCHECK_STATES",
    "RUNTIME_STATUS",
    "VERIFIER_DIMENSIONS",
    "VERIFIER_NE_CONDITIONS",
    "read_directional_consumer_manifest",
    "write_directional_consumer_manifest",
]
