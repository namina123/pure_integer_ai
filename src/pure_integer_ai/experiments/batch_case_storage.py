"""保存自然查询证据的一份 trace 与可验证 provenance 引用。

运行时 trace 仍是唯一规范整数 record；provenance 只保存相对路径、长度和
SHA-256。读取方必须先核对文件边界、hash 和整数投影，才可恢复现有 V1
审计器需要的内存结构。该模块不执行查询、生成或 Memory 写入。
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.batch_trace_storage import (
    TraceStorageError, _relative_path, _trace_projection,
    read_trace_reference, sha256_bytes, validate_trace_reference,
)
from pure_integer_ai.experiments.graph_query_trace_contract import (
    GraphQueryTraceEvidence,
    GraphQueryTraceContractError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


COMPACT_PROVENANCE_FORMAT = "PURE_INTEGER_BATCH_CASE_PROVENANCE_REF_V1"
CAPTURED_PROVENANCE_FORMAT = "PURE_INTEGER_BATCH_CASE_PROVENANCE_V1"
CAPTURED_STATUS = "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS"


# 兼容既有冷审调用者；trace 与 provenance 共享一种物理存储错误。
CaseArtifactStorageError = TraceStorageError


def build_compact_case_provenance(
        captured: Mapping[str, Any], *, trace_bytes: bytes,
        trace_path: str | None = None, trace_ref: Mapping[str, Any] | None = None,
        trace_root: Path | None = None,
        ) -> dict[str, Any]:
    """把完整V1捕获转换成不复制trace的可验证引用记录。"""
    if not isinstance(captured, Mapping):
        raise CaseArtifactStorageError("captured provenance must be a mapping")
    if (captured.get("format") != CAPTURED_PROVENANCE_FORMAT
            or captured.get("status") != CAPTURED_STATUS):
        raise CaseArtifactStorageError("captured provenance format or status is invalid")
    if (type(captured.get("free_dialogue_complete")) is not int
            or captured.get("free_dialogue_complete") != 0):
        raise CaseArtifactStorageError("captured free_dialogue_complete must be the strict integer zero")
    if "trace" not in captured:
        raise CaseArtifactStorageError("captured provenance has no trace to externalize")
    _projection, projection_bytes = _trace_projection(trace_bytes)
    if canonical_json_bytes(captured["trace"]) != projection_bytes:
        raise CaseArtifactStorageError("captured trace projection differs from the original trace artifact")
    if trace_ref is None:
        reference = {
            "path": _relative_path(trace_path), "byte_count": len(trace_bytes),
            "sha256": sha256_bytes(trace_bytes),
            "integer_projection_sha256": sha256_bytes(projection_bytes),
        }
    else:
        if trace_path is not None or trace_root is None:
            raise CaseArtifactStorageError("saved trace_ref requires trace_root and excludes trace_path")
        validate_trace_reference(trace_ref)
        if read_trace_reference(trace_root, trace_ref) != trace_bytes:
            raise CaseArtifactStorageError("saved trace differs from the captured original bytes")
        reference = dict(trace_ref)
    without_trace = json.loads(canonical_json_bytes({
        key: value for key, value in captured.items() if key != "trace"}))
    result = {
        "format": COMPACT_PROVENANCE_FORMAT,
        "captured_format": CAPTURED_PROVENANCE_FORMAT,
        "status": CAPTURED_STATUS,
        "provenance": without_trace,
        "trace_ref": reference,
        "free_dialogue_complete": 0,
    }
    return result


def compact_case_provenance_bytes(
        captured: Mapping[str, Any], *, trace_bytes: bytes,
        trace_path: str | None = None, trace_ref: Mapping[str, Any] | None = None,
        trace_root: Path | None = None,
        ) -> bytes:
    """返回可直接写入K身份的规范紧凑引用bytes。"""
    return canonical_json_bytes(build_compact_case_provenance(
        captured, trace_path=trace_path, trace_bytes=trace_bytes,
        trace_ref=trace_ref, trace_root=trace_root)) + b"\n"


def _resolve_trace(root: Path, trace_ref: Mapping[str, Any]) -> dict[str, Any]:
    """在声明root内读取并完整核验trace引用。"""
    raw = read_trace_reference(root, trace_ref)
    projection, _projection_bytes = _trace_projection(raw)
    return projection


def restore_compact_case_provenance(
        compact: Mapping[str, Any], *, root: Path,
        ) -> dict[str, Any]:
    """核对引用后恢复旧V1捕获结构，供既有冷审器继续消费。"""
    if not isinstance(compact, Mapping):
        raise CaseArtifactStorageError("compact provenance must be a mapping")
    if set(compact) != {"format", "captured_format", "status", "provenance",
                        "trace_ref", "free_dialogue_complete"}:
        raise CaseArtifactStorageError("compact provenance envelope fields differ")
    if (compact.get("format") != COMPACT_PROVENANCE_FORMAT
            or compact.get("captured_format") != CAPTURED_PROVENANCE_FORMAT
            or compact.get("status") != CAPTURED_STATUS
            or type(compact.get("free_dialogue_complete")) is not int
            or compact.get("free_dialogue_complete") != 0):
        raise CaseArtifactStorageError("compact provenance envelope is invalid")
    provenance = compact.get("provenance")
    if not isinstance(provenance, Mapping) or "trace" in provenance:
        raise CaseArtifactStorageError("compact provenance payload must omit trace")
    if (provenance.get("format") != CAPTURED_PROVENANCE_FORMAT
            or provenance.get("status") != CAPTURED_STATUS
            or type(provenance.get("free_dialogue_complete")) is not int
            or provenance.get("free_dialogue_complete") != 0):
        raise CaseArtifactStorageError("restored provenance format or status is invalid")
    projection = _resolve_trace(root, compact.get("trace_ref"))
    return {**provenance, "trace": projection}


def load_case_provenance(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """核验存储后返回V1结构；元数据与Plan/Memory的语义连接仍由既有冷审器核对。"""
    path = path.resolve(strict=True)
    artifact_root = path.parent if root is None else root.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(artifact_root):
        raise CaseArtifactStorageError("saved provenance path is outside the declared artifact root")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CaseArtifactStorageError("saved provenance must be an object")
    if value.get("format") == CAPTURED_PROVENANCE_FORMAT:
        if (value.get("status") != CAPTURED_STATUS
                or type(value.get("free_dialogue_complete")) is not int
                or value.get("free_dialogue_complete") != 0):
            raise CaseArtifactStorageError("saved provenance status or completion flag is invalid")
        if "trace" not in value:
            raise CaseArtifactStorageError("full provenance record has no trace")
        try:
            GraphQueryTraceEvidence.from_integer_artifact(value["trace"])
        except (GraphQueryTraceContractError, TypeError, ValueError) as error:
            raise CaseArtifactStorageError("saved provenance trace is invalid") from error
        return value
    if value.get("format") != COMPACT_PROVENANCE_FORMAT:
        raise CaseArtifactStorageError("unsupported saved provenance format")
    return restore_compact_case_provenance(value, root=artifact_root)


__all__ = [
    "CAPTURED_PROVENANCE_FORMAT",
    "CAPTURED_STATUS",
    "COMPACT_PROVENANCE_FORMAT",
    "CaseArtifactStorageError",
    "build_compact_case_provenance",
    "compact_case_provenance_bytes",
    "load_case_provenance",
    "restore_compact_case_provenance",
    "sha256_bytes",
]
