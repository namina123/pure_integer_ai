"""DLG-RAW-11B：V2 mixed dialogue session 的 V3 binding 与快照。

此模块只在逻辑 snapshot 边界组合 runtime binding、V2 context codec 和 session
operation ordinal。它不读取路径、终端、SQLite 或 provider runtime；物理 bytes
transport 使用固定的 ``u64 count/length || unsigned big-endian integer``。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    provider_origin_anchor_schema_record_v1,
    provider_origin_provider_binding_from_public_provider_v1,
    provider_origin_relation_enum_identity_v1,
    provider_origin_relation_enum_record_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_APPEND_ACCEPTED,
    MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
    MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
    MIXED_CONTEXT_APPEND_RESULT_RECORD_V1,
    MIXED_CONTEXT_FRAME_TURN_RECORD_V2,
    MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
    MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
)
from pure_integer_ai.experiments.conversation_provider_origin_context_snapshot import (
    MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2,
    MixedContextSnapshotError,
    mixed_context_snapshot_codec_identity_v2,
    mixed_context_snapshot_codec_revision_v2,
    restore_mixed_conversation_context_v2,
    snapshot_mixed_conversation_context_v2,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    provider_origin_followup_schema_identity_v1,
    provider_origin_followup_schema_record_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    ConversationRawMixedDialogueStateV2,
    RAW_MIXED_DIALOGUE_TURN_RECORD_V4,
)


RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V3 = 3
RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V4 = 4
RAW_MIXED_DIALOGUE_PROJECTION_ADMISSION_RECORD_V1 = 1
RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V3 = 3
RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V3 = 3
RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V4 = 4
RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4 = 4
RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V3 = MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2
RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4 = MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2
RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V4 = 1
RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1 = 8
RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1 = 1
RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1 = 1

RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V3 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/MIXED-DIALOGUE-RUNTIME/V3")
RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V4 = (
    b"PURE-INTEGER-AI/DLG-RAW-11C/MIXED-DIALOGUE-RUNTIME/V4")
RAW_MIXED_DIALOGUE_PROJECTION_ADMISSION_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/PROJECTION-ADMISSION/V1")

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-11B
class ConversationRawMixedDialogueSnapshotError(ValueError):
    """V3 runtime binding、mixed session record 或 bytes transport 不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """固定 record count、length 和版本为显式 unsigned 64-bit 数学整数。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise ConversationRawMixedDialogueSnapshotError(
            f"{label} 必须是非负 u64")
    return value


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证所有 logical record 使用有限、规范的非负严格整数序列。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawMixedDialogueSnapshotError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str,
          allow_empty: bool = True) -> None:
    """写入 ``u64 count || payload``，不借用 Python 容器序列化。"""
    record = _record(value, label=label, allow_empty=allow_empty)
    result.extend((_u64(len(record), label=f"{label} count"), *record))


def _read_scalar(
        record: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[int, int]:
    """读取一项已全局验证的非负整数，显式拒绝截断。"""
    if cursor >= len(record):
        raise ConversationRawMixedDialogueSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], int]:
    """读取 count-framed integer segment，拒绝溢出、截断和空值偷换。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    _u64(count, label=f"{label} count")
    if count > len(record) - cursor:
        raise ConversationRawMixedDialogueSnapshotError(f"{label} 长度越界")
    value = record[cursor:cursor + count]
    cursor += count
    return _record(value, label=label, allow_empty=allow_empty), cursor


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """以已冻结 portable SHA raw-u8 framing 生成跨语言 binding identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawMixedDialogueSnapshotError(
            f"{label} 无法形成") from error


def mixed_dialogue_projection_admission_record_v1() -> tuple[int, ...]:
    """冻结 provider-origin 投影 admission 的分型、写入来源和失败码。"""
    return (
        RAW_MIXED_DIALOGUE_PROJECTION_ADMISSION_RECORD_V1,
        MIXED_CONTEXT_APPEND_RESULT_RECORD_V1,
        MIXED_CONTEXT_FRAME_TURN_RECORD_V2,
        MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
        MIXED_CONTEXT_WRITE_ORIGIN_NONE,
        MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
        MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
        MIXED_CONTEXT_APPEND_ACCEPTED,
        MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
        MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
    )


def mixed_dialogue_projection_admission_identity_v1() -> tuple[int, ...]:
    """返回完整 admission schema 的 raw u8 identity，供 V3 binding 逐项锁定。"""
    return _identity(
        RAW_MIXED_DIALOGUE_PROJECTION_ADMISSION_IDENTITY_DOMAIN_V1,
        mixed_dialogue_projection_admission_record_v1(),
        label="mixed dialogue projection admission identity",
    )


def mixed_dialogue_runtime_binding_v3(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 V3 session snapshot 必须精确匹配的完整 runtime/schema binding。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed dialogue snapshot runtime 类型错误")
    provider = runtime.proof_sentence_provider
    if provider is None:
        raise ConversationRawMixedDialogueSnapshotError(
            "V3 mixed dialogue snapshot 必须绑定完整 proof provider")
    provider_binding = provider_origin_provider_binding_from_public_provider_v1(
        provider).canonical_record()
    result = [RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V3]
    for label, value in (
            ("public dialogue runtime binding", runtime.binding_record()),
            ("provider origin binding", provider_binding),
            ("provider origin anchor schema", provider_origin_anchor_schema_record_v1()),
            ("provider origin relation enum", provider_origin_relation_enum_record_v1()),
            ("provider origin relation identity", provider_origin_relation_enum_identity_v1()),
            ("projection admission schema", mixed_dialogue_projection_admission_record_v1()),
            ("projection admission identity", mixed_dialogue_projection_admission_identity_v1()),
            ("mixed context codec revision", mixed_context_snapshot_codec_revision_v2()),
            ("mixed context codec identity", mixed_context_snapshot_codec_identity_v2()),
    ):
        _pack(result, value, label=label, allow_empty=False)
    return tuple(result)


def mixed_dialogue_runtime_identity_v3(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """返回 V3 binding 的 raw u8 identity，不以 Python runtime 对象充当身份。"""
    return _identity(
        RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V3,
        mixed_dialogue_runtime_binding_v3(runtime),
        label="mixed dialogue runtime identity",
    )


def mixed_dialogue_snapshot_transport_record_v4() -> tuple[int, ...]:
    """冻结 V4 outer bytes transport 的全部跨语言可观察规则。"""
    return (
        RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V4,
        RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4,
        RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4,
        RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1,
        RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1,
        RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1,
    )


def mixed_dialogue_runtime_binding_v4(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 V4 binding，额外锁定 11C follow-up catalog 与 reducer schema。

    V3 从不观察 follow-up grammar，因此它的 binding 保持不变。默认 V4 session
    只能使用此处完整锁定的 catalog；缺失或漂移时不允许 snapshot/restore。
    """
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed dialogue V4 snapshot runtime 类型错误")
    followup_catalog = runtime.provider_origin_followup_catalog
    if followup_catalog is None:
        raise ConversationRawMixedDialogueSnapshotError(
            "V4 mixed dialogue snapshot 缺 provider-origin follow-up catalog")
    result = [RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V4]
    for label, value in (
            ("mixed dialogue V3 runtime binding", mixed_dialogue_runtime_binding_v3(runtime)),
            ("provider-origin follow-up catalog", followup_catalog.canonical_record()),
            ("provider-origin follow-up catalog identity", followup_catalog.catalog_identity_u8),
            ("provider-origin follow-up schema", provider_origin_followup_schema_record_v1()),
            ("provider-origin follow-up schema identity", provider_origin_followup_schema_identity_v1()),
            ("mixed dialogue V4 turn record", (RAW_MIXED_DIALOGUE_TURN_RECORD_V4,)),
            ("mixed dialogue V4 snapshot transport", mixed_dialogue_snapshot_transport_record_v4()),
    ):
        _pack(result, value, label=label, allow_empty=False)
    return tuple(result)


def mixed_dialogue_runtime_identity_v4(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """返回 V4 runtime binding identity，不混用 V3 domain。"""
    return _identity(
        RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V4,
        mixed_dialogue_runtime_binding_v4(runtime),
        label="mixed dialogue V4 runtime identity",
    )


def snapshot_public_mixed_frame_dialogue_state(
        state: ConversationRawMixedDialogueStateV2,
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 V2 session 的 V3 logical snapshot，encoder 先执行内层 readback。"""
    if type(state) is not ConversationRawMixedDialogueStateV2:
        raise TypeError("mixed dialogue snapshot state 类型错误")
    binding = mixed_dialogue_runtime_binding_v3(runtime)
    context_record = snapshot_mixed_conversation_context_v2(state.context)
    result = [RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V3]
    for label, value in (
            ("mixed dialogue runtime binding", binding),
            ("mixed dialogue conversation key", state.conversation_key),
            ("mixed dialogue context", context_record),
    ):
        _pack(result, value, label=label, allow_empty=False)
    result.append(_u64(
        state.next_operation_ordinal,
        label="mixed dialogue next operation ordinal",
    ))
    restored = restore_public_mixed_frame_dialogue_state(tuple(result), runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot encoder readback 漂移")
    return tuple(result)


def restore_public_mixed_frame_dialogue_state(
        record: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedDialogueStateV2:
    """严格恢复 V3 session snapshot；V1/V2 outer record 一律拒绝且不猜测升级。"""
    record = _record(record, label="mixed dialogue snapshot", allow_empty=False)
    expected_binding = mixed_dialogue_runtime_binding_v3(runtime)
    cursor = 0
    version, cursor = _read_scalar(record, cursor, label="mixed dialogue snapshot version")
    if version != RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V3:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot version 未注册")
    binding, cursor = _read_segment(
        record, cursor, label="mixed dialogue runtime binding", allow_empty=False)
    if binding != expected_binding:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue runtime binding 漂移")
    conversation_key, cursor = _read_segment(
        record, cursor, label="mixed dialogue conversation key", allow_empty=False)
    context_record, cursor = _read_segment(
        record, cursor, label="mixed dialogue context", allow_empty=False)
    ordinal, cursor = _read_scalar(
        record, cursor, label="mixed dialogue next operation ordinal")
    _u64(ordinal, label="mixed dialogue next operation ordinal")
    if ordinal < 1:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue next operation ordinal 必须大于零")
    if cursor != len(record):
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot 含尾随整数")
    try:
        context = restore_mixed_conversation_context_v2(context_record)
    except (MixedContextSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V2 context 无法恢复") from error
    if context.conversation_key != conversation_key:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue state/context conversation key 漂移")
    try:
        state = ConversationRawMixedDialogueStateV2(
            conversation_key,
            ordinal,
            context,
        )
    except (TypeError, ValueError) as error:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue state 无法恢复") from error
    return state


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """把一个非负数学整数编码为最短 unsigned big-endian byte vector。"""
    _u64((value.bit_length() + 7) // 8, label=f"{label} byte length")
    size = max(1, (value.bit_length() + 7) // 8)
    if size > RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V3:
        raise ConversationRawMixedDialogueSnapshotError(
            f"{label} 超出 snapshot bytes 预算")
    return value.to_bytes(size, "big")


def encode_public_mixed_frame_dialogue_snapshot_bytes(
        state: ConversationRawMixedDialogueStateV2,
        runtime: PublicDialogueRuntimeV1,
        ) -> bytes:
    """将 V3 logical snapshot 编为唯一的 count/length unsigned bytes transport。"""
    record = snapshot_public_mixed_frame_dialogue_state(state, runtime)
    result = bytearray()
    result.extend(_u64(
        RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V3,
        label="mixed dialogue bytes version",
    ).to_bytes(8, "big"))
    result.extend(_u64(
        len(record), label="mixed dialogue bytes integer count",
    ).to_bytes(8, "big"))
    for index, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value, label=f"mixed dialogue integer[{index}]")
        result.extend(_u64(
            len(encoded), label=f"mixed dialogue integer[{index}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4:
            raise ConversationRawMixedDialogueSnapshotError(
                "mixed dialogue snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_mixed_frame_dialogue_snapshot_bytes(payload, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue bytes encoder readback 漂移")
    return payload


def _read_u64_bytes(
        payload: bytes, cursor: int, *, label: str,
        ) -> tuple[int, int]:
    """从 raw bytes 显式读取一项 big-endian u64，不使用宿主 unpack 格式。"""
    if cursor > len(payload) - 8:
        raise ConversationRawMixedDialogueSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def decode_public_mixed_frame_dialogue_snapshot_bytes(
        payload: bytes,
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedDialogueStateV2:
    """严格解码 V3 bytes transport 后恢复 session，拒绝 leading zero 与尾随 bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot bytes 必须是非空 raw bytes")
    if len(payload) > RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="mixed dialogue bytes version")
    if version != RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V3:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload, cursor, label="mixed dialogue bytes integer count")
    if count > (len(payload) - cursor) // 9:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue bytes integer count 越界")
    values: list[int] = []
    for index in range(count):
        size, cursor = _read_u64_bytes(
            payload, cursor, label=f"mixed dialogue integer[{index}] length")
        if size < 1 or size > len(payload) - cursor:
            raise ConversationRawMixedDialogueSnapshotError(
                f"mixed dialogue integer[{index}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawMixedDialogueSnapshotError(
                f"mixed dialogue integer[{index}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue snapshot bytes 含尾随 bytes")
    return restore_public_mixed_frame_dialogue_state(tuple(values), runtime)


def snapshot_public_mixed_frame_dialogue_state_v4(
        state: ConversationRawMixedDialogueStateV2,
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出默认 11C session 的 V4 logical snapshot，不重编码 inner V2 context。"""
    if type(state) is not ConversationRawMixedDialogueStateV2:
        raise TypeError("mixed dialogue V4 snapshot state 类型错误")
    binding = mixed_dialogue_runtime_binding_v4(runtime)
    context_record = snapshot_mixed_conversation_context_v2(state.context)
    result = [RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V4]
    for label, value in (
            ("mixed dialogue V4 runtime binding", binding),
            ("mixed dialogue V4 conversation key", state.conversation_key),
            ("mixed dialogue V4 context", context_record),
    ):
        _pack(result, value, label=label, allow_empty=False)
    result.append(_u64(
        state.next_operation_ordinal,
        label="mixed dialogue V4 next operation ordinal",
    ))
    restored = restore_public_mixed_frame_dialogue_state_v4(tuple(result), runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot encoder readback 漂移")
    return tuple(result)


def restore_public_mixed_frame_dialogue_state_v4(
        record: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedDialogueStateV2:
    """严格恢复 V4 state，明确拒绝 V1/V2/V3 记录而不猜测升级。"""
    record = _record(record, label="mixed dialogue V4 snapshot", allow_empty=False)
    expected_binding = mixed_dialogue_runtime_binding_v4(runtime)
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="mixed dialogue V4 snapshot version")
    if version != RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V4:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot version 未注册")
    binding, cursor = _read_segment(
        record,
        cursor,
        label="mixed dialogue V4 runtime binding",
        allow_empty=False,
    )
    if binding != expected_binding:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 runtime binding 漂移")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="mixed dialogue V4 conversation key",
        allow_empty=False,
    )
    context_record, cursor = _read_segment(
        record,
        cursor,
        label="mixed dialogue V4 context",
        allow_empty=False,
    )
    ordinal, cursor = _read_scalar(
        record,
        cursor,
        label="mixed dialogue V4 next operation ordinal",
    )
    _u64(ordinal, label="mixed dialogue V4 next operation ordinal")
    if ordinal < 1:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 next operation ordinal 必须大于零")
    if cursor != len(record):
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot 含尾随整数")
    try:
        context = restore_mixed_conversation_context_v2(context_record)
    except (MixedContextSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 inner context 无法恢复") from error
    if context.conversation_key != conversation_key:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 state/context conversation key 漂移")
    try:
        return ConversationRawMixedDialogueStateV2(
            conversation_key,
            ordinal,
            context,
        )
    except (TypeError, ValueError) as error:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 state 无法恢复") from error


def encode_public_mixed_frame_dialogue_snapshot_v4_bytes(
        state: ConversationRawMixedDialogueStateV2,
        runtime: PublicDialogueRuntimeV1,
        ) -> bytes:
    """以同一整数 bytes transport 写出 V4 outer snapshot。"""
    record = snapshot_public_mixed_frame_dialogue_state_v4(state, runtime)
    result = bytearray()
    result.extend(_u64(
        RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4,
        label="mixed dialogue V4 bytes version",
    ).to_bytes(8, "big"))
    result.extend(_u64(
        len(record), label="mixed dialogue V4 bytes integer count",
    ).to_bytes(8, "big"))
    for index, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value, label=f"mixed dialogue V4 integer[{index}]")
        result.extend(_u64(
            len(encoded), label=f"mixed dialogue V4 integer[{index}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V3:
            raise ConversationRawMixedDialogueSnapshotError(
                "mixed dialogue V4 snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_mixed_frame_dialogue_snapshot_v4_bytes(payload, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 bytes encoder readback 漂移")
    return payload


def decode_public_mixed_frame_dialogue_snapshot_v4_bytes(
        payload: bytes,
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedDialogueStateV2:
    """严格解码 V4 count/length transport，不接受 V3 或尾随 bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot bytes 必须是非空 raw bytes")
    if len(payload) > RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V3:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="mixed dialogue V4 bytes version")
    if version != RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload, cursor, label="mixed dialogue V4 bytes integer count")
    if count > (len(payload) - cursor) // 9:
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 bytes integer count 越界")
    values: list[int] = []
    for index in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"mixed dialogue V4 integer[{index}] length",
        )
        if size < 1 or size > len(payload) - cursor:
            raise ConversationRawMixedDialogueSnapshotError(
                f"mixed dialogue V4 integer[{index}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawMixedDialogueSnapshotError(
                f"mixed dialogue V4 integer[{index}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawMixedDialogueSnapshotError(
            "mixed dialogue V4 snapshot bytes 含尾随 bytes")
    return restore_public_mixed_frame_dialogue_state_v4(tuple(values), runtime)


__all__ = [
    "RAW_MIXED_DIALOGUE_PROJECTION_ADMISSION_RECORD_V1",
    "RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V3",
    "RAW_MIXED_DIALOGUE_RUNTIME_BINDING_RECORD_V4",
    "RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V3",
    "RAW_MIXED_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V4",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V3",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V3",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V3",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_RECORD_V4",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V4",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1",
    "RAW_MIXED_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1",
    "ConversationRawMixedDialogueSnapshotError",
    "decode_public_mixed_frame_dialogue_snapshot_bytes",
    "decode_public_mixed_frame_dialogue_snapshot_v4_bytes",
    "encode_public_mixed_frame_dialogue_snapshot_bytes",
    "encode_public_mixed_frame_dialogue_snapshot_v4_bytes",
    "mixed_dialogue_projection_admission_identity_v1",
    "mixed_dialogue_projection_admission_record_v1",
    "mixed_dialogue_runtime_binding_v3",
    "mixed_dialogue_runtime_binding_v4",
    "mixed_dialogue_runtime_identity_v3",
    "mixed_dialogue_runtime_identity_v4",
    "mixed_dialogue_snapshot_transport_record_v4",
    "restore_public_mixed_frame_dialogue_state",
    "restore_public_mixed_frame_dialogue_state_v4",
    "snapshot_public_mixed_frame_dialogue_state",
    "snapshot_public_mixed_frame_dialogue_state_v4",
]
