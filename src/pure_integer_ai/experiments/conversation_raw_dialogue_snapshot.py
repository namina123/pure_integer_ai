"""DLG-RAW-07：RAW-04 typed session 的可移植快照 codec。

本模块不读取文件、不接收路径、不调用 terminal/runtime，也不保存问题或回答的表层 bytes。
它只把已经完成的 RAW-04 state 转换为有序非负整数 record，并从同一 record 重建
``ConversationContextRead`` 的显式前缀依赖。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationContextState,
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    ConversationRawDialogueState,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)


RAW_DIALOGUE_SNAPSHOT_RECORD_V1 = 1
RAW_DIALOGUE_CONTEXT_SNAPSHOT_RECORD_V1 = 1
RAW_DIALOGUE_TURN_SNAPSHOT_RECORD_V1 = 1
RAW_DIALOGUE_SNAPSHOT_BYTES_V1 = 1
RAW_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 = 4 * 1024 * 1024
_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-07
class ConversationRawDialogueSnapshotError(ValueError):
    """DLG-RAW-07 快照 record/bytes 或恢复 binding 不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """固定所有计数/序号为非负 u64，拒绝宿主整数溢出语义。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise ConversationRawDialogueSnapshotError(f"{label} 必须是非负 u64")
    return value


def _nonnegative_sequence(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """核验快照中每个变长字段都只含规范非负整数。"""
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawDialogueSnapshotError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str,
          allow_empty: bool = True) -> None:
    """写入 count + payload 的 record 段，不依赖 Python 容器序列化。"""
    checked = _nonnegative_sequence(value, label=label, allow_empty=allow_empty)
    result.extend((_u64(len(checked), label=f"{label} count"), *checked))


def _read_scalar(record: tuple[int, ...], cursor: int, *, label: str) -> tuple[int, int]:
    """读取一个已全局核验过的非负整数，并显式处理截断。"""
    if cursor >= len(record):
        raise ConversationRawDialogueSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], int]:
    """读取 length-prefixed 整数段，禁止截断、越界和空值偷换。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    if count > len(record) - cursor:
        raise ConversationRawDialogueSnapshotError(f"{label} 长度越界")
    value = record[cursor:cursor + count]
    cursor += count
    return _nonnegative_sequence(value, label=label, allow_empty=allow_empty), cursor


def _identity_from_key(value: tuple[int, ...], *, label: str) -> ObjectIdentity:
    """从完整稳定键重建 ObjectIdentity，错误不泄露 host exception 语义。"""
    try:
        return ObjectIdentity.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise ConversationRawDialogueSnapshotError(
            f"{label} 不是完整 ObjectIdentity stable key") from error


def _source_from_key(value: tuple[int, ...], *, label: str) -> SourceRef:
    """从完整稳定键重建 SourceRef，禁止只保留局部 source id。"""
    try:
        return SourceRef.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise ConversationRawDialogueSnapshotError(
            f"{label} 不是完整 SourceRef stable key") from error


def _turn_snapshot_record(turn: ConversationTurnState) -> tuple[int, ...]:
    """编码一个 context turn 及其 read descriptor，绝不递归复制历史 turns。"""
    if not isinstance(turn, ConversationTurnState):
        raise TypeError("snapshot turn 类型错误")
    read = turn.context_read
    if not isinstance(read, ConversationContextRead):
        raise ConversationRawDialogueSnapshotError(
            "context snapshot turn 必须携带显式 context read")
    result = [RAW_DIALOGUE_TURN_SNAPSHOT_RECORD_V1, _u64(
        turn.turn_ordinal, label="snapshot turn ordinal")]
    for label, value in (
            ("request key", turn.request_key),
            ("target key", turn.target_key),
            ("query key", turn.query_key),
            ("planning key", turn.planning_key),
            ("response stance", turn.response_stance.stable_key()),
            ("parser revision", turn.parser_revision),
            ("readback key", turn.readback_key)):
        _pack(result, value, label=label, allow_empty=False)
    result.append(_u64(len(turn.selected_candidate_keys),
                       label="snapshot selected candidate count"))
    for ordinal, value in enumerate(turn.selected_candidate_keys):
        _pack(result, value, label=f"selected candidate[{ordinal}]",
              allow_empty=False)
    result.append(_u64(len(turn.cited_sources),
                       label="snapshot cited source count"))
    for ordinal, source in enumerate(turn.cited_sources):
        _pack(result, source.stable_key(), label=f"cited source[{ordinal}]",
              allow_empty=False)
    result.append(_u64(len(turn.discourse_sentence_keys),
                       label="snapshot discourse sentence count"))
    for ordinal, value in enumerate(turn.discourse_sentence_keys):
        _pack(result, value, label=f"discourse sentence[{ordinal}]",
              allow_empty=False)
    result.append(_u64(len(read.turns), label="snapshot context read limit"))
    result.append(_u64(read.revision, label="snapshot context read revision"))
    _pack(result, read.digest, label="snapshot context read digest",
          allow_empty=False)
    _pack(result, read.stable_key(), label="snapshot context read witness",
          allow_empty=False)
    return tuple(result)


def _context_snapshot_record(context: ConversationContextState) -> tuple[int, ...]:
    """编码完整 append-only typed context；每个 turn 保持固定逻辑顺序。"""
    if not isinstance(context, ConversationContextState):
        raise TypeError("snapshot context 类型错误")
    result = [RAW_DIALOGUE_CONTEXT_SNAPSHOT_RECORD_V1]
    _pack(result, context.conversation_key, label="snapshot context key",
          allow_empty=False)
    result.append(_u64(context.revision, label="snapshot context revision"))
    _pack(result, context.previous_digest, label="snapshot previous digest")
    result.append(_u64(len(context.turns), label="snapshot context turn count"))
    for ordinal, turn in enumerate(context.turns):
        _pack(result, _turn_snapshot_record(turn), label=f"snapshot turn[{ordinal}]",
              allow_empty=False)
    return tuple(result)


def _runtime_binding(runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
    """从已构造 logical runtime 取得完整 binding，禁止调用者手工遗漏字段。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("snapshot public dialogue runtime 类型错误")
    return _nonnegative_sequence(
        runtime.binding_record(),
        label="snapshot runtime binding",
        allow_empty=False,
    )


def snapshot_public_frame_dialogue_state(
        state: ConversationRawDialogueState,
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 RAW-04 state 的完整 logical snapshot record。

    ``runtime`` 必须是已经由单一 public source closure 构造的完整逻辑 runtime；其
    binding 同时锁定 source closure、active/base/source-bound catalog 与 protocol revision。
    """
    if not isinstance(state, ConversationRawDialogueState):
        raise TypeError("snapshot state 类型错误")
    binding = _runtime_binding(runtime)
    result = [RAW_DIALOGUE_SNAPSHOT_RECORD_V1]
    _pack(result, binding, label="snapshot runtime binding", allow_empty=False)
    _pack(result, state.conversation_key, label="snapshot conversation key",
          allow_empty=False)
    result.append(_u64(state.next_operation_ordinal,
                       label="snapshot next operation ordinal"))
    _pack(result, _context_snapshot_record(state.context),
          label="snapshot context", allow_empty=False)
    return tuple(result)


def _decode_turn(
        record: tuple[int, ...],
        context: ConversationContextState,
        ) -> ConversationTurnState:
    """从一个 turn record 恢复字段，并用已恢复前缀重建 context read。"""
    cursor = 0
    version, cursor = _read_scalar(record, cursor, label="snapshot turn version")
    if version != RAW_DIALOGUE_TURN_SNAPSHOT_RECORD_V1:
        raise ConversationRawDialogueSnapshotError("snapshot turn version 未注册")
    ordinal, cursor = _read_scalar(record, cursor, label="snapshot turn ordinal")
    fields = []
    for label in (
            "snapshot request key",
            "snapshot target key",
            "snapshot query key",
            "snapshot planning key",
            "snapshot response stance",
            "snapshot parser revision",
            "snapshot readback key"):
        value, cursor = _read_segment(record, cursor, label=label,
                                      allow_empty=False)
        fields.append(value)
    selected_count, cursor = _read_scalar(
        record, cursor, label="snapshot selected candidate count")
    selected = []
    for item_ordinal in range(selected_count):
        value, cursor = _read_segment(
            record, cursor, label=f"snapshot selected candidate[{item_ordinal}]",
            allow_empty=False)
        selected.append(value)
    if tuple(selected) != tuple(sorted(selected)) or len(set(selected)) != len(selected):
        raise ConversationRawDialogueSnapshotError(
            "snapshot selected candidate 未规范排序或重复")
    source_count, cursor = _read_scalar(record, cursor,
                                        label="snapshot cited source count")
    cited_sources = []
    for item_ordinal in range(source_count):
        value, cursor = _read_segment(
            record, cursor, label=f"snapshot cited source[{item_ordinal}]",
            allow_empty=False)
        cited_sources.append(_source_from_key(
            value, label=f"snapshot cited source[{item_ordinal}]"))
    cited_source_keys = tuple(item.stable_key() for item in cited_sources)
    if (cited_source_keys != tuple(sorted(cited_source_keys))
            or len(set(cited_source_keys)) != len(cited_source_keys)):
        raise ConversationRawDialogueSnapshotError(
            "snapshot cited source 未规范排序或重复")
    sentence_count, cursor = _read_scalar(
        record, cursor, label="snapshot discourse sentence count")
    sentences = []
    for item_ordinal in range(sentence_count):
        value, cursor = _read_segment(
            record, cursor, label=f"snapshot discourse sentence[{item_ordinal}]",
            allow_empty=False)
        sentences.append(value)
    read_limit, cursor = _read_scalar(record, cursor,
                                      label="snapshot context read limit")
    read_revision, cursor = _read_scalar(record, cursor,
                                         label="snapshot context read revision")
    read_digest, cursor = _read_segment(record, cursor,
                                        label="snapshot context read digest",
                                        allow_empty=False)
    read_witness, cursor = _read_segment(record, cursor,
                                         label="snapshot context read witness",
                                         allow_empty=False)
    if cursor != len(record):
        raise ConversationRawDialogueSnapshotError("snapshot turn 含尾随整数")
    if ordinal != context.revision:
        raise ConversationRawDialogueSnapshotError("snapshot turn ordinal 不连续")
    if read_limit > context.revision:
        raise ConversationRawDialogueSnapshotError(
            "snapshot context read limit 超出已恢复前缀")
    expected_read = context.read(read_limit)
    if (read_revision != expected_read.revision
            or read_digest != expected_read.digest
            or read_witness != expected_read.stable_key()):
        raise ConversationRawDialogueSnapshotError(
            "snapshot context read witness 与已恢复前缀不一致")
    request_key, target_key, query_key, planning_key, stance_key, parser_revision, readback_key = fields
    turn = ConversationTurnState(
        ordinal,
        request_key,
        target_key,
        query_key,
        planning_key,
        _identity_from_key(stance_key, label="snapshot response stance"),
        tuple(selected),
        tuple(cited_sources),
        tuple(sentences),
        parser_revision,
        readback_key,
        expected_read,
    )
    return turn


def _decode_context_snapshot(record: tuple[int, ...]) -> ConversationContextState:
    """严格 decode/replay complete typed context，逐 turn 校验 digest/read 前驱链。"""
    cursor = 0
    version, cursor = _read_scalar(record, cursor, label="snapshot context version")
    if version != RAW_DIALOGUE_CONTEXT_SNAPSHOT_RECORD_V1:
        raise ConversationRawDialogueSnapshotError("snapshot context version 未注册")
    conversation_key, cursor = _read_segment(record, cursor,
                                             label="snapshot context key",
                                             allow_empty=False)
    revision, cursor = _read_scalar(record, cursor, label="snapshot context revision")
    previous_digest, cursor = _read_segment(record, cursor,
                                             label="snapshot previous digest")
    turn_count, cursor = _read_scalar(record, cursor,
                                      label="snapshot context turn count")
    context = start_conversation_context(conversation_key)
    for ordinal in range(turn_count):
        turn_record, cursor = _read_segment(record, cursor,
                                            label=f"snapshot turn[{ordinal}]",
                                            allow_empty=False)
        turn = _decode_turn(turn_record, context)
        context = ConversationContextState(
            context.conversation_key,
            context.revision + 1,
            context.digest(),
            (*context.turns, turn),
        )
    if cursor != len(record):
        raise ConversationRawDialogueSnapshotError("snapshot context 含尾随整数")
    if revision != context.revision or previous_digest != context.previous_digest:
        raise ConversationRawDialogueSnapshotError(
            "snapshot context revision 或 previous digest 漂移")
    return context


def restore_public_frame_dialogue_state(
        record: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawDialogueState:
    """按给定 logical runtime binding 恢复 RAW-04 state，不产生任何副作用。"""
    if (not isinstance(record, tuple)
            or any(type(item) is not int or item < 0 for item in record)):
        raise ConversationRawDialogueSnapshotError(
            "snapshot record 必须是非负严格整数 tuple")
    expected_binding = _runtime_binding(runtime)
    cursor = 0
    version, cursor = _read_scalar(record, cursor, label="snapshot version")
    if version != RAW_DIALOGUE_SNAPSHOT_RECORD_V1:
        raise ConversationRawDialogueSnapshotError("snapshot version 未注册")
    binding, cursor = _read_segment(record, cursor,
                                    label="snapshot runtime binding",
                                    allow_empty=False)
    if binding != expected_binding:
        raise ConversationRawDialogueSnapshotError("snapshot runtime binding 漂移")
    conversation_key, cursor = _read_segment(record, cursor,
                                             label="snapshot conversation key",
                                             allow_empty=False)
    next_operation_ordinal, cursor = _read_scalar(
        record, cursor, label="snapshot next operation ordinal")
    if next_operation_ordinal < 1:
        raise ConversationRawDialogueSnapshotError(
            "snapshot next operation ordinal 必须大于零")
    context_record, cursor = _read_segment(record, cursor,
                                            label="snapshot context",
                                            allow_empty=False)
    if cursor != len(record):
        raise ConversationRawDialogueSnapshotError("snapshot 含尾随整数")
    context = _decode_context_snapshot(context_record)
    if context.conversation_key != conversation_key:
        raise ConversationRawDialogueSnapshotError(
            "snapshot state/context conversation key 漂移")
    try:
        return ConversationRawDialogueState(
            conversation_key,
            next_operation_ordinal,
            context,
        )
    except (TypeError, ValueError) as error:
        raise ConversationRawDialogueSnapshotError(
            "snapshot RAW-04 state 无法恢复") from error


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """把一个非负数学整数显式转换为最短 unsigned-big-endian byte sequence。"""
    size = (value.bit_length() + 7) // 8
    _u64(size, label=f"{label} byte length")
    if size > RAW_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawDialogueSnapshotError(
            f"{label} 超出 snapshot bytes 预算")
    return value.to_bytes(max(1, size), "big")


def encode_public_frame_dialogue_snapshot_bytes(
        state: ConversationRawDialogueState,
        runtime: PublicDialogueRuntimeV1,
        ) -> bytes:
    """以固定 u64 count/length + unsigned-big-endian 整数导出 snapshot bytes。"""
    record = snapshot_public_frame_dialogue_state(state, runtime)
    result = bytearray()
    result.extend(_u64(RAW_DIALOGUE_SNAPSHOT_BYTES_V1,
                       label="snapshot bytes version").to_bytes(8, "big"))
    result.extend(_u64(len(record), label="snapshot bytes integer count").to_bytes(
        8, "big"))
    for ordinal, value in enumerate(record):
        encoded = _unsigned_integer_bytes(value, label=f"snapshot integer[{ordinal}]")
        result.extend(_u64(len(encoded), label=f"snapshot integer[{ordinal}] length").to_bytes(
            8, "big"))
        result.extend(encoded)
        if len(result) > RAW_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
            raise ConversationRawDialogueSnapshotError("snapshot bytes 超出固定预算")
    return bytes(result)


def _read_u64_bytes(payload: bytes, cursor: int, *, label: str) -> tuple[int, int]:
    """从 bytes transport 读取一个 u64，截断不借用 Python slicing 行为。"""
    if cursor > len(payload) - 8:
        raise ConversationRawDialogueSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def decode_public_frame_dialogue_snapshot_bytes(
        payload: bytes,
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawDialogueState:
    """严格解码 snapshot bytes 后恢复 state；任何 transport 漂移都不返回部分状态。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawDialogueSnapshotError("snapshot bytes 必须是非空 raw bytes")
    if len(payload) > RAW_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawDialogueSnapshotError("snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(payload, cursor,
                                      label="snapshot bytes version")
    if version != RAW_DIALOGUE_SNAPSHOT_BYTES_V1:
        raise ConversationRawDialogueSnapshotError("snapshot bytes version 未注册")
    count, cursor = _read_u64_bytes(payload, cursor,
                                    label="snapshot bytes integer count")
    if count > (len(payload) - cursor) // 9:
        raise ConversationRawDialogueSnapshotError("snapshot bytes integer count 越界")
    values = []
    for ordinal in range(count):
        size, cursor = _read_u64_bytes(payload, cursor,
                                       label=f"snapshot integer[{ordinal}] length")
        if size < 1 or size > len(payload) - cursor:
            raise ConversationRawDialogueSnapshotError(
                f"snapshot integer[{ordinal}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawDialogueSnapshotError(
                f"snapshot integer[{ordinal}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawDialogueSnapshotError("snapshot bytes 含尾随 bytes")
    return restore_public_frame_dialogue_state(tuple(values), runtime)


__all__ = [
    "RAW_DIALOGUE_SNAPSHOT_RECORD_V1",
    "RAW_DIALOGUE_CONTEXT_SNAPSHOT_RECORD_V1",
    "RAW_DIALOGUE_TURN_SNAPSHOT_RECORD_V1",
    "RAW_DIALOGUE_SNAPSHOT_BYTES_V1",
    "RAW_DIALOGUE_SNAPSHOT_MAX_BYTES_V1",
    "ConversationRawDialogueSnapshotError",
    "decode_public_frame_dialogue_snapshot_bytes",
    "encode_public_frame_dialogue_snapshot_bytes",
    "restore_public_frame_dialogue_state",
    "snapshot_public_frame_dialogue_state",
]
