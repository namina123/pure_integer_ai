"""DLG-RAW-07 公开 payload closure 的唯一物理读取 host adapter。

这里允许使用本机文件系统；返回值只保留纯 core 定义的 logical payload 与读取
trace。物理位置、链接、OS 错误和安装布局不会流入 closure identity 或后续运行时。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_PATH_ESCAPE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
    PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1,
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    PublicSourcePayloadReadTraceV1,
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
    require_public_source_payload_closure_identity_v1,
)


# object-model: exception; interop=DLG-RAW-07-host
class PublicSourcePayloadHostError(RuntimeError):
    """物理读取失败到固定 provider result code 的 host-only 映射。"""

    def __init__(self, result_code: int, logical_key: bytes = b"") -> None:
        """保存可运输失败码与可选 logical key，不把 OS 文本当作语义。"""
        self.result_code = result_code
        self.logical_key = logical_key
        super().__init__("DLG-RAW-07 public payload host read rejected")


# object-model: value; representation=struct; interop=DLG-RAW-07-host
@dataclass(frozen=True, slots=True)
class PublicSourcePayloadHostReadResultV1:
    """一次完整无写入读取的 closure 与其有序纯 trace。"""

    closure: PublicSourcePayloadClosureV1
    read_trace: tuple[PublicSourcePayloadReadTraceV1, ...]

    def __post_init__(self) -> None:
        """使 host 返回的 trace 与 closure 同键、同序且无遗漏。"""
        if type(self.closure) is not PublicSourcePayloadClosureV1:
            raise TypeError("payload host result closure 类型错误")
        if not isinstance(self.read_trace, tuple):
            raise TypeError("payload host result trace 必须是 tuple")
        if len(self.read_trace) != len(self.closure.records):
            raise ValueError("payload host result trace 数量漂移")
        for ordinal, (trace, record) in enumerate(
                zip(self.read_trace, self.closure.records), start=1):
            if type(trace) is not PublicSourcePayloadReadTraceV1:
                raise TypeError("payload host result 含非法 trace")
            if (trace.request_ordinal != ordinal
                    or trace.logical_key != record.logical_key
                    or trace.raw_payload != record.raw_payload
                    or trace.payload_length != record.payload_length
                    or trace.raw_sha256 != record.raw_sha256
                    or trace.result_code != PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1
                    or trace.write_effect_code
                    != PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1):
                raise ValueError("payload host result trace/closure 漂移")


def _host_error(result_code: int, logical_key: bytes = b"") -> PublicSourcePayloadHostError:
    """构造不含路径和 OS 异常文本的固定 host failure。"""
    return PublicSourcePayloadHostError(result_code, logical_key)


def _resolve_host_root(resource_root: str | Path) -> Path:
    """在 host 层解析一个必须存在的目录，失败不泄漏为 core 输入。"""
    if not isinstance(resource_root, (str, Path)):
        raise _host_error(PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1)
    try:
        root = Path(resource_root).resolve(strict=True)
        if not root.is_dir():
            raise _host_error(PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1)
    except PublicSourcePayloadHostError:
        raise
    except OSError as error:
        raise _host_error(PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1) from error
    return root


def _logical_key_parts(logical_key: bytes) -> tuple[str, ...]:
    """把 provider 已冻结的 ASCII key 变为 host 路径片段。"""
    try:
        value = logical_key.decode("ascii")
    except UnicodeDecodeError as error:
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
            logical_key,
        ) from error
    parts = tuple(value.split("/"))
    if (len(parts) != 3 or parts[:2] != ("data", "ph2")
            or any(not part or part in (".", "..") for part in parts)):
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
            logical_key,
        )
    return parts


def _resolve_host_payload_path(root: Path, logical_key: bytes) -> Path:
    """拒绝 closure 目录中的任意组件符号链接或 root 外逃逸。"""
    current = root
    try:
        for part in _logical_key_parts(logical_key):
            current = current / part
            if current.is_symlink():
                raise _host_error(
                    PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1,
                    logical_key,
                )
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except PublicSourcePayloadHostError:
        raise
    except FileNotFoundError as error:
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
            logical_key,
        ) from error
    except ValueError as error:
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_PATH_ESCAPE_V1,
            logical_key,
        ) from error
    except OSError as error:
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1,
            logical_key,
        ) from error
    if not resolved.is_file():
        raise _host_error(
            PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
            logical_key,
        )
    return resolved


def read_public_source_payload_closure_from_root(
        resource_root: str | Path,
        *,
        expected_closure_identity: bytes | None = None,
        ) -> PublicSourcePayloadHostReadResultV1:
    """读取精确登记的 public resources，建立纯 closure 与无写入 trace。

    ``expected_closure_identity`` 用于恢复前绑定：任一 raw 内容、长度或 digest
    改变都会在 host 边界 fail closed，且不会返回可被 runtime 消费的 closure。
    """
    root = _resolve_host_root(resource_root)
    records = []
    trace = []
    for ordinal, logical_key in enumerate(PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1, start=1):
        path = _resolve_host_payload_path(root, logical_key)
        try:
            raw_payload = path.read_bytes()
        except FileNotFoundError as error:
            raise _host_error(
                PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
                logical_key,
            ) from error
        except OSError as error:
            raise _host_error(
                PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1,
                logical_key,
            ) from error
        try:
            record = public_source_payload_record_from_u8_v1(
                logical_key,
                raw_payload,
            )
            entry = PublicSourcePayloadReadTraceV1(
                ordinal,
                record.logical_key,
                record.raw_payload,
                record.payload_length,
                record.raw_sha256,
                record.result_code,
                PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1,
            )
        except PublicSourcePayloadProviderError as error:
            raise _host_error(
                PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
                logical_key,
            ) from error
        records.append(record)
        trace.append(entry)
    try:
        closure = build_public_source_payload_closure_v1(tuple(records))
        if expected_closure_identity is not None:
            require_public_source_payload_closure_identity_v1(
                closure,
                expected_closure_identity,
            )
    except PublicSourcePayloadProviderError as error:
        raise _host_error(PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1) from error
    return PublicSourcePayloadHostReadResultV1(closure, tuple(trace))


def load_public_source_payload_closure_from_root(
        resource_root: str | Path,
        *,
        expected_closure_identity: bytes | None = None,
        ) -> PublicSourcePayloadClosureV1:
    """仅返回 closure 的便利 host API；完整读取 trace 可用 ``read_*`` 取得。"""
    return read_public_source_payload_closure_from_root(
        resource_root,
        expected_closure_identity=expected_closure_identity,
    ).closure


__all__ = [
    "PublicSourcePayloadHostError",
    "PublicSourcePayloadHostReadResultV1",
    "load_public_source_payload_closure_from_root",
    "read_public_source_payload_closure_from_root",
]
