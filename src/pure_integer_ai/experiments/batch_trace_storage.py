"""唯一 trace blob 的有界物理保存、身份引用与冷恢复；不执行图查询。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
import hashlib
import json
from pathlib import Path
import posixpath
from typing import Any

from pure_integer_ai.experiments.batch_case_provenance import CaseProvenanceError, _integer_projection
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceContractError, GraphQueryTraceEvidence
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.storage.trace_byte_codec import (
    TRACE_BYTE_ENCODING, TraceByteCodecError, decode_trace_bytes, iter_encoded_trace_bytes,
)


TRACE_STORAGE_FORMAT = "PURE_INTEGER_QUERY_TRACE_STORAGE_REF_V1"
MAX_TRACE_BYTES = 128 * 1024 * 1024
_REF_FIELDS = {"path", "byte_count", "sha256", "integer_projection_sha256"}
_STORED_FIELDS = {"encoding", "stored_byte_count", "stored_sha256"}
MAX_DESCRIPTOR_BYTES = 8192


class TraceStorageError(ValueError):
    """trace 引用、规范 record 或物理 blob 违反合同。"""


def sha256_bytes(raw: bytes) -> str:
    """以原始 bytes 计算规范 SHA-256。"""
    if not isinstance(raw, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(raw).hexdigest()


def _relative_path(value: object, label: str = "trace_path") -> str:
    """只接受 root 内规范 POSIX 相对路径，拒绝路径逃逸。"""
    if (not isinstance(value, str) or not value or "\\" in value
            or "\0" in value or ":" in value):
        raise TraceStorageError(f"{label} must be a nonempty POSIX relative path")
    normalized = posixpath.normpath(value)
    if (normalized != value or normalized in {".", ".."}
            or normalized.startswith("../") or normalized.startswith("/")):
        raise TraceStorageError(f"{label} escapes the declared artifact root")
    return normalized


def _bounded_sha(value: object, label: str) -> str:
    """核验固定宽度小写 SHA 身份。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise TraceStorageError(f"{label} must be a lowercase SHA-256")
    return value


def load_case_query_trace(path: Path) -> dict[str, Any]:
    """冷审共同 caller 的 trace 与同目录 receipt，兼容新旧物理格式。"""
    with (path.parent / "call_receipt.json").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise TraceStorageError("call receipt exceeds byte budget")
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise TraceStorageError("call receipt must be an object")
    return load_saved_query_trace(path, receipt=receipt)


def _positive_int(value: object, label: str) -> int:
    """核验严格正整数字节数。"""
    if type(value) is not int or value <= 0:
        raise TraceStorageError(f"{label} must be a strict positive integer")
    return value


def _trace_projection(raw: bytes) -> tuple[dict[str, Any], bytes]:
    """完整验证原 trace 及整数投影；规范身份仍由原 record 决定。"""
    if not isinstance(raw, bytes) or not raw:
        raise TraceStorageError("trace bytes must be nonempty")
    try:
        value = json.loads(raw)
        projection = _integer_projection(value)
        GraphQueryTraceEvidence(value)
        GraphQueryTraceEvidence.from_integer_artifact(projection)
    except (CaseProvenanceError, GraphQueryTraceContractError, TypeError, ValueError) as error:
        raise TraceStorageError("trace is not a valid integer QueryBridge artifact") from error
    return projection, canonical_json_bytes(projection)


@dataclass(frozen=True, slots=True)
class ValidatedTraceSnapshot:
    """仅本次调用复用完整校验；两个不可变bytes不能由外部注入投影。"""

    raw: bytes = field(repr=False)
    compressed: bool = False
    checkpoint: InitVar[Callable[[], None] | None] = None
    projection_bytes: bytes = field(init=False, repr=False)
    stored_chunks: tuple[bytes, ...] = field(init=False, repr=False)

    def __post_init__(self, checkpoint: Callable[[], None] | None) -> None:
        """构造即完整验证逻辑记录并生成全部规范物理字节，不能外部注入。"""
        if type(self.compressed) is not bool:
            raise TraceStorageError("snapshot encoding choice must be a strict boolean")
        _projection, projection_bytes = _trace_projection(self.raw)
        object.__setattr__(self, "projection_bytes", projection_bytes)
        chunks = tuple(iter_encoded_trace_bytes(self.raw, checkpoint=checkpoint)) if self.compressed else (self.raw,)
        object.__setattr__(self, "stored_chunks", chunks)


def validate_trace_reference(ref: Mapping[str, Any]) -> bool:
    """核验 V1 或物理编码引用，返回是否使用新压缩编码。"""
    if not isinstance(ref, Mapping) or set(ref) not in (_REF_FIELDS, _REF_FIELDS | _STORED_FIELDS):
        raise TraceStorageError("trace_ref fields differ from the declared path, size and hashes")
    _relative_path(ref.get("path"))
    _positive_int(ref.get("byte_count"), "trace_ref.byte_count")
    _bounded_sha(ref.get("sha256"), "trace_ref.sha256")
    _bounded_sha(ref.get("integer_projection_sha256"), "trace_ref.integer_projection_sha256")
    compressed = "encoding" in ref
    if compressed:
        if ref["encoding"] != TRACE_BYTE_ENCODING:
            raise TraceStorageError("trace_ref encoding is unsupported")
        _positive_int(ref.get("stored_byte_count"), "trace_ref.stored_byte_count")
        _bounded_sha(ref.get("stored_sha256"), "trace_ref.stored_sha256")
    return compressed


def _descriptor_reference(path: Path, root: Path) -> dict[str, Any]:
    """有界读取小 descriptor；符号链接仍须位于声明 root 内。"""
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise TraceStorageError("trace descriptor is outside the declared artifact root")
    with path.open("rb") as stream:
        raw = stream.read(MAX_DESCRIPTOR_BYTES + 1)
    if len(raw) > MAX_DESCRIPTOR_BYTES:
        raise TraceStorageError("trace descriptor exceeds byte budget")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise TraceStorageError("trace descriptor is invalid JSON") from error
    if (not isinstance(value, dict)
            or set(value) != {"format", "trace_ref", "free_dialogue_complete"}
            or value["format"] != TRACE_STORAGE_FORMAT
            or type(value["free_dialogue_complete"]) is not int or value["free_dialogue_complete"] != 0):
        raise TraceStorageError("trace descriptor envelope differs")
    if not validate_trace_reference(value["trace_ref"]):
        raise TraceStorageError("trace descriptor must reference the compressed blob")
    return value["trace_ref"]


def _read_stored_trace_bytes(root: Path, ref: Mapping[str, Any], *, max_trace_bytes: int = MAX_TRACE_BYTES) -> bytes:
    """完整核验当前物理路径、descriptor、长度和字节身份，不据此宣称已解码。"""
    compressed = validate_trace_reference(ref)
    if type(max_trace_bytes) is not int or max_trace_bytes <= 0:
        raise TraceStorageError("trace byte limit must be a strict positive integer")
    expected_size = ref["byte_count"]
    if expected_size > max_trace_bytes:
        raise TraceStorageError("trace exceeds decoded byte budget")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise TraceStorageError("trace artifact root must be an existing directory")
    path = (root / Path(*ref["path"].split("/"))).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise TraceStorageError("trace_ref path is outside the declared artifact root")
    stored_size = ref["stored_byte_count"] if compressed else expected_size
    maximum_stored = expected_size + 16 + 4 * ((expected_size + 65535) // 65536)
    if stored_size > maximum_stored or path.stat().st_size != stored_size:
        raise TraceStorageError("trace_ref byte count or SHA-256 differs")
    if compressed and _descriptor_reference(path.parent / "trace.int.json", root) != ref:
        raise TraceStorageError("trace descriptor and provenance references differ")
    with path.open("rb") as stream:
        raw = stream.read(stored_size + 1)
    expected_sha = ref["stored_sha256"] if compressed else ref["sha256"]
    if len(raw) != stored_size or sha256_bytes(raw) != expected_sha:
        raise TraceStorageError("trace_ref byte count or SHA-256 differs")
    return raw


def _read_trace_storage(root: Path, ref: Mapping[str, Any], *, max_trace_bytes: int = MAX_TRACE_BYTES) -> bytes:
    """独立冷读保持完整解码、规范重编码及原始字节身份核验。"""
    raw = _read_stored_trace_bytes(root, ref, max_trace_bytes=max_trace_bytes)
    compressed, expected_size = "encoding" in ref, ref["byte_count"]
    if compressed:
        try:
            raw = decode_trace_bytes(raw, max_decoded_bytes=expected_size)
        except TraceByteCodecError as error:
            raise TraceStorageError("stored trace codec is invalid") from error
    if len(raw) != expected_size or sha256_bytes(raw) != ref["sha256"]:
        raise TraceStorageError("decoded trace byte count or SHA-256 differs")
    return raw


def _read_snapshot_trace_storage(root: Path, ref: Mapping[str, Any], snapshot: ValidatedTraceSnapshot,
                                 *, max_trace_bytes: int = MAX_TRACE_BYTES) -> bytes:
    """逐字节回读本次已实际产生的规范编码，原冷解码路径不变。"""
    if type(snapshot) is not ValidatedTraceSnapshot:
        raise TraceStorageError("physical trace witness must be a validated snapshot")
    stored = _read_stored_trace_bytes(root, ref, max_trace_bytes=max_trace_bytes)
    if (snapshot.compressed != ("encoding" in ref) or len(snapshot.raw) != ref["byte_count"]
            or sha256_bytes(snapshot.raw) != ref["sha256"]
            or sha256_bytes(snapshot.projection_bytes) != ref["integer_projection_sha256"]
            or stored != b"".join(snapshot.stored_chunks)):
        raise TraceStorageError("saved trace differs from its complete same-call canonical encoding witness")
    return snapshot.raw


def read_trace_reference(root: Path, ref: Mapping[str, Any], *, max_trace_bytes: int = MAX_TRACE_BYTES) -> bytes:
    """先验证物理存储，再完整核验整数投影；普通冷读取不信任派生hash。"""
    raw = _read_trace_storage(root, ref, max_trace_bytes=max_trace_bytes)
    _projection, projection_bytes = _trace_projection(raw)
    if sha256_bytes(projection_bytes) != ref["integer_projection_sha256"]:
        raise TraceStorageError("trace_ref integer projection SHA-256 differs")
    return raw


def save_case_trace_snapshot(root: Path, value: Mapping[str, Any], *, compressed: bool = False,
                    max_trace_bytes: int = MAX_TRACE_BYTES,
                    checkpoint: Callable[[], None] | None = None,
                    before_write: Callable[[Path, int], None] | None = None
                    ) -> tuple[dict[str, Any], ValidatedTraceSnapshot]:
    """保存唯一blob和descriptor，并返回本次完整校验的不可变快照。"""
    root = root.resolve(strict=True)
    if root.drive.upper() != "K:" or not root.is_dir() or type(compressed) is not bool:
        raise TraceStorageError("trace save requires an explicit K case directory and boolean encoding choice")
    _positive_int(max_trace_bytes, "max_trace_bytes")
    raw = canonical_json_bytes(value) + b"\n"
    if len(raw) > max_trace_bytes:
        raise TraceStorageError("trace exceeds decoded byte budget")
    snapshot = ValidatedTraceSnapshot(raw, compressed=compressed, checkpoint=checkpoint)
    ref = {"path": "trace.int.lz4" if compressed else "trace.int.json",
           "byte_count": len(raw), "sha256": sha256_bytes(raw),
           "integer_projection_sha256": sha256_bytes(snapshot.projection_bytes)}
    path = root / ref["path"]
    stored_size = 0
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        for chunk in snapshot.stored_chunks:
            if before_write is not None:
                before_write(path, stored_size + len(chunk))
            stream.write(chunk)
            stored_size += len(chunk)
            digest.update(chunk)
    if compressed:
        ref.update(encoding=TRACE_BYTE_ENCODING, stored_byte_count=stored_size, stored_sha256=digest.hexdigest())
        descriptor = canonical_json_bytes({"format": TRACE_STORAGE_FORMAT, "trace_ref": ref,
                                            "free_dialogue_complete": 0}) + b"\n"
        path = root / "trace.int.json"
        if before_write is not None:
            before_write(path, len(descriptor))
        with path.open("xb") as stream:
            stream.write(descriptor)
    return ref, snapshot


def save_case_trace(root: Path, value: Mapping[str, Any], *, compressed: bool = False,
                    max_trace_bytes: int = MAX_TRACE_BYTES,
                    checkpoint: Callable[[], None] | None = None,
                    before_write: Callable[[Path, int], None] | None = None
                    ) -> tuple[dict[str, Any], bytes]:
    """保留原调用者的ref/bytes接口和全部物理保存校验。"""
    reference, snapshot = save_case_trace_snapshot(root, value, compressed=compressed,
        max_trace_bytes=max_trace_bytes, checkpoint=checkpoint, before_write=before_write)
    return reference, snapshot.raw


def load_saved_query_trace(path: Path, *, root: Path | None = None,
                           max_trace_bytes: int = MAX_TRACE_BYTES,
                           receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """兼容旧 JSON 与新 descriptor，返回原完整 trace 供独立冷审。"""
    path = path.resolve(strict=True)
    root = path.parent if root is None else root.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise TraceStorageError("saved trace path is outside the declared artifact root")
    if type(max_trace_bytes) is not int or max_trace_bytes <= 0 or path.stat().st_size > max_trace_bytes:
        raise TraceStorageError("saved trace exceeds byte budget")
    with path.open("rb") as stream:
        raw = stream.read(max_trace_bytes + 1)
    if len(raw) > max_trace_bytes:
        raise TraceStorageError("saved trace exceeds byte budget")
    stored_descriptor_sha = sha256_bytes(raw)
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise TraceStorageError("saved trace is invalid JSON") from error
    ref = None
    if isinstance(value, dict) and value.get("format") == TRACE_STORAGE_FORMAT:
        ref = _descriptor_reference(path, root)
        raw = read_trace_reference(root, ref, max_trace_bytes=max_trace_bytes)
        value = json.loads(raw)
    else:
        _trace_projection(raw)
    if receipt is not None:
        if (receipt.get("trace_sha256") != sha256_bytes(raw)
                or receipt.get("trace_byte_count", len(raw)) != len(raw)
                or receipt.get("trace_descriptor_sha256", stored_descriptor_sha) != stored_descriptor_sha):
            raise TraceStorageError("call receipt differs from the original trace or descriptor identity")
        stored_size = len(raw) if ref is None else ref["stored_byte_count"]
        stored_sha = sha256_bytes(raw) if ref is None else ref["stored_sha256"]
        if (receipt.get("trace_stored_byte_count", stored_size) != stored_size
                or receipt.get("trace_stored_sha256", stored_sha) != stored_sha):
            raise TraceStorageError("call receipt differs from the physical trace identity")
    return value
