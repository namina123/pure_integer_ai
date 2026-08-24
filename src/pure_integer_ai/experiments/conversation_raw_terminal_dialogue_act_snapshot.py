"""DLG-RAW-13：终端话语行为 outer state 的版本化 logical/bytes snapshot。

本 snapshot 不升级 DLG-RAW-12；它把一个已验证的 inner snapshot 嵌入新 record，并
额外锁定 terminal-act course/runtime binding。act 本身是 dialogue-state no-op，故
历史 response 不伪装为 V3 ledger event；恢复后以同一冻结 course 继续输入时必须
重新得到相同的 response bytes 和 outer turn record。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_snapshot import (
    RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1,
    ConversationRawMixedFocusDialogueSnapshotError,
    mixed_focus_dialogue_runtime_binding_v1,
    mixed_focus_dialogue_runtime_identity_v1,
    restore_public_mixed_focus_dialogue_state_v1,
    snapshot_public_mixed_focus_dialogue_state_v1,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    TERMINAL_DIALOGUE_ACT_RUNTIME_BINDING_RECORD_V1,
    TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1,
    TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1,
    TerminalDialogueActRuntimeV1,
    TerminalDialogueActStateV1,
)


TERMINAL_DIALOGUE_ACT_SNAPSHOT_RECORD_V1 = 1
TERMINAL_DIALOGUE_ACT_SNAPSHOT_BYTES_V1 = 1
TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_RECORD_V1 = 1
TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_U64_WIDTH_V1 = 8
TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1 = 1
TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1 = 1
TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1 = (
    RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 + 65536)
TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_INTEGER_COUNT_V1 = (
    TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1 // 9)

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-13
class ConversationRawTerminalDialogueActSnapshotError(ValueError):
    """DLG-RAW-13 snapshot、runtime binding 或 bytes transport 不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """验证 transport count/length/version 的明确 unsigned 64-bit 范围。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise ConversationRawTerminalDialogueActSnapshotError(
            f"{label} 必须是非负严格 u64")
    return value


def _record(value: tuple[int, ...], *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """验证预算内有限非负整数 record，拒绝 Python list 和隐式数字。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or len(value) > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawTerminalDialogueActSnapshotError(
            f"{label} 不是预算内的{'可空' if allow_empty else '非空'}整数 record")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以显式 count 追加一段 nonnegative integer record。"""
    record = _record(value, label=label, allow_empty=False)
    _u64(len(record), label=f"{label} count")
    result.extend((len(record), *record))


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """顺序读取一个有界 nonnegative integer，拒绝截断。"""
    if cursor >= len(record):
        raise ConversationRawTerminalDialogueActSnapshotError(f"{label} 截断")
    value = _u64(record[cursor], label=label)
    return value, cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[tuple[int, ...], int]:
    """按 count framing 恢复一段非空整数 record，拒绝越界。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    if count < 1 or count > len(record) - cursor:
        raise ConversationRawTerminalDialogueActSnapshotError(
            f"{label} count 越界")
    value = tuple(record[cursor:cursor + count])
    _record(value, label=label, allow_empty=False)
    return value, cursor + count


def terminal_dialogue_act_snapshot_transport_record_v1() -> tuple[int, ...]:
    """冻结 DLG-RAW-13 outer bytes transport 的版本、预算、宽度与字节序。"""
    return (
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_RECORD_V1,
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_BYTES_V1,
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1,
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_U64_WIDTH_V1,
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1,
        TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1,
    )


def terminal_dialogue_act_runtime_binding_v1(
        runtime: TerminalDialogueActRuntimeV1,
        ) -> tuple[int, ...]:
    """导出同时锁定 inner DLG-RAW-12 与 terminal course 的 outer binding。"""
    if type(runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("terminal dialogue act snapshot runtime 类型错误")
    result = [
        TERMINAL_DIALOGUE_ACT_RUNTIME_BINDING_RECORD_V1,
        TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1,
        TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1,
    ]
    for label, value in (
            ("terminal dialogue act runtime binding", runtime.binding_record()),
            ("terminal dialogue act runtime identity", runtime.runtime_identity_u8()),
            ("mixed focus inner runtime binding",
             mixed_focus_dialogue_runtime_binding_v1(runtime.inner_runtime)),
            ("mixed focus inner runtime identity",
             mixed_focus_dialogue_runtime_identity_v1(runtime.inner_runtime)),
            ("terminal dialogue act bytes transport",
             terminal_dialogue_act_snapshot_transport_record_v1())):
        _pack(result, value, label=label)
    return tuple(result)


def snapshot_public_terminal_dialogue_act_state_v1(
        state: TerminalDialogueActStateV1,
        runtime: TerminalDialogueActRuntimeV1,
        ) -> tuple[int, ...]:
    """导出新 outer logical snapshot，内嵌而不重写 DLG-RAW-12 state snapshot。"""
    if type(state) is not TerminalDialogueActStateV1:
        raise TypeError("terminal dialogue act snapshot state 类型错误")
    binding = terminal_dialogue_act_runtime_binding_v1(runtime)
    try:
        inner = snapshot_public_mixed_focus_dialogue_state_v1(
            state.inner_state,
            runtime.inner_runtime,
        )
    except (ConversationRawMixedFocusDialogueSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act inner snapshot 无法形成") from error
    result = [TERMINAL_DIALOGUE_ACT_SNAPSHOT_RECORD_V1]
    _pack(result, binding, label="terminal dialogue act runtime binding")
    _pack(result, inner, label="terminal dialogue act inner snapshot")
    record = tuple(result)
    restored = restore_public_terminal_dialogue_act_state_v1(record, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot encoder readback 漂移")
    return record


def restore_public_terminal_dialogue_act_state_v1(
        record: tuple[int, ...],
        runtime: TerminalDialogueActRuntimeV1,
        ) -> TerminalDialogueActStateV1:
    """严格恢复 outer state；binding/version/尾随值不匹配时全部拒绝。"""
    values = _record(record, label="terminal dialogue act snapshot",
                     allow_empty=False)
    if type(runtime) is not TerminalDialogueActRuntimeV1:
        raise TypeError("terminal dialogue act snapshot runtime 类型错误")
    expected_binding = terminal_dialogue_act_runtime_binding_v1(runtime)
    cursor = 0
    version, cursor = _read_scalar(values, cursor,
                                   label="terminal dialogue act snapshot version")
    if version != TERMINAL_DIALOGUE_ACT_SNAPSHOT_RECORD_V1:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot version 未注册")
    binding, cursor = _read_segment(values, cursor,
                                    label="terminal dialogue act runtime binding")
    if binding != expected_binding:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act runtime binding 漂移")
    inner_record, cursor = _read_segment(values, cursor,
                                         label="terminal dialogue act inner snapshot")
    if cursor != len(values):
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot 含尾随整数")
    try:
        inner = restore_public_mixed_focus_dialogue_state_v1(
            inner_record,
            runtime.inner_runtime,
        )
    except (ConversationRawMixedFocusDialogueSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act inner snapshot 无法恢复") from error
    return TerminalDialogueActStateV1(inner)


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """用最短 unsigned big-endian bytes 编码一个非负数学整数。"""
    if type(value) is not int or value < 0:
        raise ConversationRawTerminalDialogueActSnapshotError(
            f"{label} 必须是非负严格整数")
    count = max(1, (value.bit_length() + 7) // 8)
    _u64(count, label=f"{label} byte length")
    if count > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawTerminalDialogueActSnapshotError(
            f"{label} 超出 snapshot bytes 预算")
    return bytes((value >> shift) & 0xFF
                 for shift in range((count - 1) * 8, -1, -8))


def _read_u64_bytes(
        payload: bytes,
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """逐 byte 读取固定大端 u64，不依赖宿主 struct/Pickle。"""
    if cursor > len(payload) - 8:
        raise ConversationRawTerminalDialogueActSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def encode_public_terminal_dialogue_act_snapshot_v1_bytes(
        state: TerminalDialogueActStateV1,
        runtime: TerminalDialogueActRuntimeV1,
        ) -> bytes:
    """以固定 count/length/minimal-unsigned 规则编码 outer logical snapshot。"""
    record = snapshot_public_terminal_dialogue_act_state_v1(state, runtime)
    result = bytearray()
    for label, value in (
            ("terminal dialogue act bytes version",
             TERMINAL_DIALOGUE_ACT_SNAPSHOT_BYTES_V1),
            ("terminal dialogue act bytes integer count", len(record))):
        result.extend(_u64(value, label=label).to_bytes(8, "big"))
    for ordinal, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value,
            label=f"terminal dialogue act integer[{ordinal}]",
        )
        result.extend(_u64(len(encoded), label=(
            f"terminal dialogue act integer[{ordinal}] length")).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1:
            raise ConversationRawTerminalDialogueActSnapshotError(
                "terminal dialogue act snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_terminal_dialogue_act_snapshot_v1_bytes(
        payload,
        runtime,
    )
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act bytes encoder readback 漂移")
    return payload


def decode_public_terminal_dialogue_act_snapshot_v1_bytes(
        payload: bytes,
        runtime: TerminalDialogueActRuntimeV1,
        ) -> TerminalDialogueActStateV1:
    """严格解码 bytes snapshot，拒绝截断、leading zero 与尾随 physical bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot bytes 必须是非空 raw bytes")
    if len(payload) > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(payload, cursor,
                                      label="terminal dialogue act bytes version")
    if version != TERMINAL_DIALOGUE_ACT_SNAPSHOT_BYTES_V1:
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act bytes version 未注册")
    count, cursor = _read_u64_bytes(payload, cursor,
                                    label="terminal dialogue act bytes integer count")
    if (count > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or count > (len(payload) - cursor) // 9):
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act bytes integer count 越界")
    values: list[int] = []
    for ordinal in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"terminal dialogue act integer[{ordinal}] length",
        )
        if (size < 1 or size > len(payload) - cursor
                or size > TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1):
            raise ConversationRawTerminalDialogueActSnapshotError(
                f"terminal dialogue act integer[{ordinal}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawTerminalDialogueActSnapshotError(
                f"terminal dialogue act integer[{ordinal}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawTerminalDialogueActSnapshotError(
            "terminal dialogue act snapshot bytes 含尾随 bytes")
    return restore_public_terminal_dialogue_act_state_v1(tuple(values), runtime)


__all__ = [
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_BYTES_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_RECORD_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_RECORD_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_U64_WIDTH_V1",
    "TERMINAL_DIALOGUE_ACT_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1",
    "ConversationRawTerminalDialogueActSnapshotError",
    "decode_public_terminal_dialogue_act_snapshot_v1_bytes",
    "encode_public_terminal_dialogue_act_snapshot_v1_bytes",
    "restore_public_terminal_dialogue_act_state_v1",
    "snapshot_public_terminal_dialogue_act_state_v1",
    "terminal_dialogue_act_runtime_binding_v1",
    "terminal_dialogue_act_snapshot_transport_record_v1",
]
