"""DLG-RAW-12：V5 discourse-focus outer state 的逻辑与 bytes snapshot。

旧 V3/V4 snapshot 不承担 focus 语义。本模块以独立版本冻结 V4 inner snapshot、
focus record、runtime binding 与 unsigned count/length bytes transport。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_provider_origin_context_snapshot import (
    MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_chain import (
    ProviderOriginDiscourseFocusV1,
    ProviderOriginFocusChainError,
    provider_origin_discourse_focus_schema_identity_v1,
    provider_origin_discourse_focus_schema_record_v1,
    restore_provider_origin_discourse_focus_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_snapshot import (
    ConversationRawMixedDialogueSnapshotError,
    mixed_dialogue_runtime_binding_v4,
    restore_public_mixed_frame_dialogue_state_v4,
    snapshot_public_mixed_frame_dialogue_state_v4,
)
from pure_integer_ai.experiments.conversation_raw_provider_focus_dialogue_session import (
    RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE,
    RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT,
    RAW_PROVIDER_FOCUS_DIALOGUE_STATE_RECORD_V1,
    RAW_PROVIDER_FOCUS_DIALOGUE_TURN_RECORD_V1,
    ConversationRawProviderFocusDialogueStateV1,
)


RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V5 = 5
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V5 = 5
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5 = 5
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V5 = 1
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1 = 8
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1 = 1
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1 = 1
RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5 = (
    2 * MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2)

RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V5 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/PROVIDER-FOCUS-DIALOGUE-RUNTIME/V5")

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-12
class ConversationRawProviderFocusDialogueSnapshotError(ValueError):
    """V5 runtime binding、outer record 或 bytes transport 不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """核验显式 count/length/version 使用的无符号 64-bit 数学整数。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            f"{label} 必须是非负 u64")
    return value


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验 logical snapshot 的有限非负整数记录。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawProviderFocusDialogueSnapshotError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _pack(
        result: list[int],
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool = True,
        ) -> None:
    """写入显式 u64 count 加 payload segment。"""
    record = _record(value, label=label, allow_empty=allow_empty)
    result.extend((_u64(len(record), label=f"{label} count"), *record))


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """读取一个 logical record 标量并拒绝截断。"""
    if cursor >= len(record):
        raise ConversationRawProviderFocusDialogueSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], int]:
    """读取 count-framed segment，拒绝越界和非法空值。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    _u64(count, label=f"{label} count")
    if count > len(record) - cursor:
        raise ConversationRawProviderFocusDialogueSnapshotError(f"{label} 长度越界")
    value = _record(
        record[cursor:cursor + count], label=label, allow_empty=allow_empty)
    return value, cursor + count


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """使用 portable SHA raw-u8 framing 形成 V5 runtime identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            f"{label} 无法形成") from error


def provider_focus_dialogue_snapshot_transport_record_v5() -> tuple[int, ...]:
    """冻结 V5 bytes transport 的版本、预算、宽度、字节序与最短整数规则。"""
    return (
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V5,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1,
    )


def provider_focus_dialogue_runtime_binding_v5(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 V5 必须精确匹配的 V4/focus/schema/transport binding。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("provider focus dialogue snapshot runtime 类型错误")
    result = [RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V5]
    for label, value in (
            ("mixed dialogue V4 runtime binding", mixed_dialogue_runtime_binding_v4(runtime)),
            ("discourse focus schema", provider_origin_discourse_focus_schema_record_v1()),
            ("discourse focus schema identity", provider_origin_discourse_focus_schema_identity_v1()),
            ("provider focus state record", (RAW_PROVIDER_FOCUS_DIALOGUE_STATE_RECORD_V1,)),
            ("provider focus turn record", (RAW_PROVIDER_FOCUS_DIALOGUE_TURN_RECORD_V1,)),
            ("provider focus focus none tag", (RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE,)),
            ("provider focus focus present tag", (RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT,)),
            ("provider focus snapshot transport", provider_focus_dialogue_snapshot_transport_record_v5()),
    ):
        _pack(result, value, label=label, allow_empty=False)
    return tuple(result)


def provider_focus_dialogue_runtime_identity_v5(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """返回 V5 runtime binding 的 raw identity，不混用旧 V4 domain。"""
    return _identity(
        RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V5,
        provider_focus_dialogue_runtime_binding_v5(runtime),
        label="provider focus dialogue runtime identity",
    )


def snapshot_public_provider_focus_dialogue_state_v5(
        state: ConversationRawProviderFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 V5 logical snapshot；inner state 始终以独立 V4 record 嵌入。"""
    if type(state) is not ConversationRawProviderFocusDialogueStateV1:
        raise TypeError("provider focus dialogue V5 snapshot state 类型错误")
    binding = provider_focus_dialogue_runtime_binding_v5(runtime)
    inner = snapshot_public_mixed_frame_dialogue_state_v4(
        state.mixed_state, runtime)
    result = [RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V5]
    _pack(result, binding, label="provider focus dialogue V5 runtime binding", allow_empty=False)
    _pack(result, inner, label="provider focus dialogue V4 inner snapshot", allow_empty=False)
    if state.focus is None:
        result.append(RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE)
    else:
        result.append(RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT)
        _pack(
            result,
            state.focus.canonical_record(),
            label="provider focus dialogue focus record",
            allow_empty=False,
        )
    restored = restore_public_provider_focus_dialogue_state_v5(tuple(result), runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot encoder readback 漂移")
    return tuple(result)


def restore_public_provider_focus_dialogue_state_v5(
        record: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawProviderFocusDialogueStateV1:
    """严格恢复 V5 outer snapshot；V3/V4 记录一律拒绝、不猜测升级。"""
    record = _record(
        record, label="provider focus dialogue V5 snapshot", allow_empty=False)
    expected_binding = provider_focus_dialogue_runtime_binding_v5(runtime)
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="provider focus dialogue V5 snapshot version")
    if version != RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V5:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot version 未注册")
    binding, cursor = _read_segment(
        record,
        cursor,
        label="provider focus dialogue V5 runtime binding",
        allow_empty=False,
    )
    if binding != expected_binding:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 runtime binding 漂移")
    inner_record, cursor = _read_segment(
        record,
        cursor,
        label="provider focus dialogue V4 inner snapshot",
        allow_empty=False,
    )
    focus_tag, cursor = _read_scalar(
        record, cursor, label="provider focus dialogue focus tag")
    if focus_tag == RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE:
        focus = None
    elif focus_tag == RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT:
        focus_record, cursor = _read_segment(
            record,
            cursor,
            label="provider focus dialogue focus record",
            allow_empty=False,
        )
        try:
            focus = restore_provider_origin_discourse_focus_v1(focus_record)
        except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
            raise ConversationRawProviderFocusDialogueSnapshotError(
                "provider focus dialogue focus 无法恢复") from error
    else:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue focus tag 未注册")
    if cursor != len(record):
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot 含尾随整数")
    try:
        inner = restore_public_mixed_frame_dialogue_state_v4(inner_record, runtime)
    except (ConversationRawMixedDialogueSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V4 inner state 无法恢复") from error
    try:
        return ConversationRawProviderFocusDialogueStateV1(inner, focus)
    except (TypeError, ValueError) as error:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 state 无法恢复") from error


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """以最短 unsigned big-endian byte vector 表示单个非负数学整数。"""
    if type(value) is not int or value < 0:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            f"{label} 必须是非负严格整数")
    size = max(1, (value.bit_length() + 7) // 8)
    _u64(size, label=f"{label} byte length")
    if size > RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            f"{label} 超出 V5 snapshot bytes 预算")
    return value.to_bytes(size, "big")


def _read_u64_bytes(
        payload: bytes,
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """从 raw bytes 显式读取一个 big-endian u64。"""
    if cursor > len(payload) - 8:
        raise ConversationRawProviderFocusDialogueSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def encode_public_provider_focus_dialogue_snapshot_v5_bytes(
        state: ConversationRawProviderFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> bytes:
    """以唯一 count/length unsigned transport 编码 V5 logical snapshot。"""
    record = snapshot_public_provider_focus_dialogue_state_v5(state, runtime)
    result = bytearray()
    result.extend(_u64(
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5,
        label="provider focus dialogue V5 bytes version",
    ).to_bytes(8, "big"))
    result.extend(_u64(
        len(record),
        label="provider focus dialogue V5 bytes integer count",
    ).to_bytes(8, "big"))
    for index, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value, label=f"provider focus dialogue V5 integer[{index}]")
        result.extend(_u64(
            len(encoded),
            label=f"provider focus dialogue V5 integer[{index}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5:
            raise ConversationRawProviderFocusDialogueSnapshotError(
                "provider focus dialogue V5 snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_provider_focus_dialogue_snapshot_v5_bytes(payload, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 bytes encoder readback 漂移")
    return payload


def decode_public_provider_focus_dialogue_snapshot_v5_bytes(
        payload: bytes,
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawProviderFocusDialogueStateV1:
    """严格解码 V5 bytes transport，拒绝非最短整数、截断和尾随 bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot bytes 必须是非空 raw bytes")
    if len(payload) > RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="provider focus dialogue V5 bytes version")
    if version != RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload, cursor, label="provider focus dialogue V5 bytes integer count")
    if count > (len(payload) - cursor) // 9:
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 bytes integer count 越界")
    values: list[int] = []
    for index in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"provider focus dialogue V5 integer[{index}] length",
        )
        if (size < 1 or size > len(payload) - cursor
                or size > RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5):
            raise ConversationRawProviderFocusDialogueSnapshotError(
                f"provider focus dialogue V5 integer[{index}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawProviderFocusDialogueSnapshotError(
                f"provider focus dialogue V5 integer[{index}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawProviderFocusDialogueSnapshotError(
            "provider focus dialogue V5 snapshot bytes 含尾随 bytes")
    return restore_public_provider_focus_dialogue_state_v5(tuple(values), runtime)


__all__ = [
    "RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V5",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1",
    "RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1",
    "ConversationRawProviderFocusDialogueSnapshotError",
    "decode_public_provider_focus_dialogue_snapshot_v5_bytes",
    "encode_public_provider_focus_dialogue_snapshot_v5_bytes",
    "provider_focus_dialogue_runtime_binding_v5",
    "provider_focus_dialogue_runtime_identity_v5",
    "provider_focus_dialogue_snapshot_transport_record_v5",
    "restore_public_provider_focus_dialogue_state_v5",
    "snapshot_public_provider_focus_dialogue_state_v5",
]
