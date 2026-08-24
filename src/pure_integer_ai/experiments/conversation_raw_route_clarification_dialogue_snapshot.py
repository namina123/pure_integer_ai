"""DLG-RAW-14：来源绑定澄清状态的 logical/bytes snapshot。

本模块不修改 DLG-RAW-12/13。它将已经冻结的 DLG-RAW-13 snapshot 嵌入新的
outer record，并以显式整数 framing 保存一条 active pending 与 append-only
selection event history。恢复时重新绑定当前公开 selector course，不能由 display
text、Python object identity 或宿主序列化补充语义。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_public_route_clarification_catalog import (
    PublicRouteClarificationCatalogError,
    RouteClarificationOutputReadbackV1,
    route_clarification_output_readback_v1,
    validate_public_route_clarification_catalog_cached_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_RECORD_V1,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    UTF8_STRICT_V1,
    ConversationRawIntake,
    ConversationRawIntakeError,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1,
)
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue import (
    ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1,
    ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_BINDING_RECORD_V1,
    ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1,
    ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1,
    ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1,
    ROUTE_CLARIFICATION_PENDING_RECORD_V1,
    ROUTE_CLARIFICATION_RESPONSE_RECORD_V1,
    ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1,
    ROUTE_CLARIFICATION_STATE_IDENTITY_DOMAIN_V1,
    ConversationRawRouteClarificationDialogueError,
    RouteAmbiguityCandidateProjectionV1,
    RouteAmbiguityPendingV1,
    RouteClarificationDialogueRuntimeV1,
    RouteClarificationDialogueStateV1,
    RouteClarificationSelectionEventV1,
    _pending_still_binds_runtime,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1,
    TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1,
    TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1,
    TERMINAL_DIALOGUE_RESPONSE_RECORD_V1,
    TerminalDialogueActStateV1,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act_snapshot import (
    TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1,
    ConversationRawTerminalDialogueActSnapshotError,
    restore_public_terminal_dialogue_act_state_v1,
    snapshot_public_terminal_dialogue_act_state_v1,
    terminal_dialogue_act_runtime_binding_v1,
)


ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_BINDING_RECORD_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RECORD_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_BYTES_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1 = 8
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1 = 1
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1 = 1

# 每条 selection 保存 offer/selection 两段 DLG-RAW-13 trace；这是有意显式的
# 持久化预算，而不是按 Python object graph 大小隐式增长。
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 = (
    TERMINAL_DIALOGUE_ACT_SNAPSHOT_MAX_BYTES_V1
    * (2 * ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1 + 1)
    + 4 * 1024 * 1024
)
ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1 = (
    ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 // 9
)

ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-CLARIFICATION-SNAPSHOT-RUNTIME/V1"
)

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-14
class ConversationRawRouteClarificationDialogueSnapshotError(ValueError):
    """DLG-RAW-14 snapshot、nested state 或 selector linkage 不闭合。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证 framing 标量使用明确、可移植的 unsigned 64-bit 范围。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 必须是范围内的严格 u64")
    return value


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证预算内有限非负整数 record，拒绝 list、bool 与负数。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or len(value) > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 不是预算内的{'可空' if allow_empty else '非空'}整数 record")
    return value


def _u8(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证跨语言 raw-u8 vector，不让 bytes/str 成为逻辑状态。"""
    record = _record(value, label=label, allow_empty=allow_empty)
    if any(item > 255 for item in record):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 不是 raw-u8 vector")
    return record


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """用 ``u64 count || record`` 追加一个显式长度子 record。"""
    record = _record(value, label=label, allow_empty=True)
    result.extend((_u64(len(record), label=f"{label} count"), *record))


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """顺序读取一个 u64 protocol scalar，拒绝截断。"""
    if cursor >= len(record):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 截断")
    return _u64(record[cursor], label=label), cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[tuple[int, ...], int]:
    """按 count framing 读取一个子 record，拒绝越界或非规范容器。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    if count > len(record) - cursor:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} count 越界")
    value = _record(
        tuple(record[cursor:cursor + count]),
        label=label,
        allow_empty=allow_empty,
    )
    return value, cursor + count


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """用冻结 portable SHA-256 framing 形成 raw-u8[32] identity。"""
    try:
        identity = tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} identity 无法形成") from error
    identity = _u8(identity, label=f"{label} identity", allow_empty=False)
    if len(identity) != 32:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} identity 长度漂移")
    return identity


def _state_identity_from_record(
        state_record: tuple[int, ...],
        ) -> tuple[int, ...]:
    """按 DLG-RAW-14 既有 domain 重算 terminal-act state linkage。"""
    return _identity(
        ROUTE_CLARIFICATION_STATE_IDENTITY_DOMAIN_V1,
        _record(state_record, label="route terminal state", allow_empty=False),
        label="route terminal state",
    )


def _restore_intake_record(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> ConversationRawIntake:
    """从 RAW-00 canonical record 恢复 intake，并逐字段 canonical readback。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != DLG_RAW_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    result_code, cursor = _read_scalar(values, cursor, label=f"{label} result code")
    utf8_rule, cursor = _read_scalar(values, cursor, label=f"{label} UTF-8 rule")
    if utf8_rule != UTF8_STRICT_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} UTF-8 rule 未注册")
    fields: list[tuple[int, ...]] = []
    for field in (
            "raw input", "canonical body", "unicode scalars", "typed record",
            "output bytes", "state delta"):
        value, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} {field}",
            allow_empty=True,
        )
        fields.append(value)
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    try:
        intake = ConversationRawIntake(
            result_code,
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            fields[5],
        )
    except (ConversationRawIntakeError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 无法恢复") from error
    if intake.canonical_record() != values:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return intake


def _restore_output_readback(
        record: tuple[int, ...],
        output_u8: tuple[int, ...],
        *,
        label: str,
        ) -> RouteClarificationOutputReadbackV1:
    """从 output bytes 重建多行 readback，绝不复用单行 RAW-00 ingress。"""
    output = _u8(output_u8, label=f"{label} output", allow_empty=False)
    try:
        readback = route_clarification_output_readback_v1(output)
    except (PublicRouteClarificationCatalogError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 无法重建") from error
    if readback.canonical_record() != _record(
            record, label=f"{label} record", allow_empty=False):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return readback


def _terminal_state_operation(
        state_record: tuple[int, ...],
        *,
        label: str,
        ) -> int:
    """读取 terminal-act state 所嵌 DLG-RAW-12 operation 序，验证两层 framing。"""
    state = _record(state_record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(state, cursor, label=f"{label} version")
    if version != TERMINAL_DIALOGUE_ACT_STATE_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    mixed_state, cursor = _read_segment(
        state,
        cursor,
        label=f"{label} mixed focus state",
        allow_empty=False,
    )
    if cursor != len(state):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    mixed_cursor = 0
    mixed_version, mixed_cursor = _read_scalar(
        mixed_state,
        mixed_cursor,
        label=f"{label} mixed focus version",
    )
    if mixed_version != RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} mixed focus version 未注册")
    ordinal, mixed_cursor = _read_scalar(
        mixed_state,
        mixed_cursor,
        label=f"{label} next operation ordinal",
    )
    if ordinal < 1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} next operation ordinal 非正")
    for field in ("mixed state", "focus context"):
        _unused, mixed_cursor = _read_segment(
            mixed_state,
            mixed_cursor,
            label=f"{label} {field}",
            allow_empty=False,
        )
    if mixed_cursor != len(mixed_state):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} mixed focus state 含尾随整数")
    return ordinal


def _terminal_response_trace(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, int, ConversationRawIntake, tuple[int, ...]]:
    """解析 DLG-RAW-13 response 的可见 linkage，不重写其封存 union。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != TERMINAL_DIALOGUE_RESPONSE_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    _response_kind, cursor = _read_scalar(
        values, cursor, label=f"{label} response kind")
    carrier_kind, cursor = _read_scalar(
        values, cursor, label=f"{label} base carrier kind")
    base_result_code, cursor = _read_scalar(
        values, cursor, label=f"{label} base result code")
    _state_effect, cursor = _read_scalar(
        values, cursor, label=f"{label} state effect")
    _act_code, cursor = _read_scalar(values, cursor, label=f"{label} act code")
    fields: list[tuple[int, ...]] = []
    for field in (
            "base carrier", "input intake", "catalog", "catalog identity",
            "form", "form identity", "output", "output readback"):
        value, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} {field}",
            allow_empty=True,
        )
        fields.append(value)
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    intake = _restore_intake_record(fields[1], label=f"{label} input intake")
    output = _u8(fields[6], label=f"{label} output", allow_empty=False)
    output_readback = _restore_intake_record(
        fields[7], label=f"{label} output readback")
    if (not output_readback.accepted
            or output_readback.raw_input_bytes != output):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} output readback 漂移")
    return carrier_kind, base_result_code, intake, output


def _terminal_turn_trace(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[
            tuple[int, ...], int, tuple[int, ...],
            tuple[int, ...], tuple[int, int, ConversationRawIntake, tuple[int, ...]],
            tuple[int, ...], int,
        ]:
    """读取 DLG-RAW-13 turn 的前后 state、inner trace 与 response linkage。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != TERMINAL_DIALOGUE_ACT_TURN_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    before, cursor = _read_segment(
        values, cursor, label=f"{label} before", allow_empty=False)
    inner_turn, cursor = _read_segment(
        values, cursor, label=f"{label} inner turn", allow_empty=False)
    response, cursor = _read_segment(
        values, cursor, label=f"{label} response", allow_empty=False)
    after, cursor = _read_segment(
        values, cursor, label=f"{label} after", allow_empty=False)
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    before_operation = _terminal_state_operation(before, label=f"{label} before")
    after_operation = _terminal_state_operation(after, label=f"{label} after")
    if after_operation != before_operation + 1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} operation 序不连续")
    return (
        before,
        before_operation,
        inner_turn,
        response,
        _terminal_response_trace(response, label=f"{label} response"),
        after,
        after_operation,
    )


def _restore_candidate_projection_record(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> RouteAmbiguityCandidateProjectionV1:
    """恢复一个 candidate projection，并重算其 canonical identity。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    fields: list[tuple[int, ...]] = []
    for field in (
            "candidate", "candidate identity", "base frame key", "base frame SHA-256",
            "target key", "recipe", "option", "option identity", "option surface"):
        value, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} {field}",
            allow_empty=False,
        )
        fields.append(value)
    identity, cursor = _read_segment(
        values,
        cursor,
        label=f"{label} projection identity",
        allow_empty=False,
    )
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    try:
        projection = RouteAmbiguityCandidateProjectionV1(*fields)
    except (ConversationRawRouteClarificationDialogueError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 无法恢复") from error
    if (projection.projection_identity_u8 != identity
            or projection.canonical_record() != values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return projection


def _restore_pending_record(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> RouteAmbiguityPendingV1:
    """恢复一轮 pending offer，并重算每个 candidate 与 pending identity。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != ROUTE_CLARIFICATION_PENDING_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    created, cursor = _read_scalar(values, cursor, label=f"{label} created operation")
    selection, cursor = _read_scalar(
        values, cursor, label=f"{label} selection operation")
    fields: list[tuple[int, ...]] = []
    for field in (
            "input intake", "inner turn", "inner response",
            "inner before state identity", "inner after state identity",
            "resolution", "route identity", "selector catalog",
            "selector catalog identity", "selector form", "selector form identity"):
        value, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} {field}",
            allow_empty=False,
        )
        fields.append(value)
    candidate_count, cursor = _read_scalar(
        values, cursor, label=f"{label} candidate count")
    if candidate_count < 2 or candidate_count > (len(values) - cursor) // 2:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} candidate count 越界")
    candidates: list[RouteAmbiguityCandidateProjectionV1] = []
    for ordinal in range(candidate_count):
        candidate_record, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} candidate {ordinal + 1}",
            allow_empty=False,
        )
        candidates.append(_restore_candidate_projection_record(
            candidate_record, label=f"{label} candidate {ordinal + 1}"))
    output, cursor = _read_segment(
        values, cursor, label=f"{label} output", allow_empty=False)
    readback_record, cursor = _read_segment(
        values,
        cursor,
        label=f"{label} output readback",
        allow_empty=False,
    )
    identity, cursor = _read_segment(
        values, cursor, label=f"{label} pending identity", allow_empty=False)
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    intake = _restore_intake_record(fields[0], label=f"{label} input intake")
    readback = _restore_output_readback(
        readback_record,
        output,
        label=f"{label} output readback",
    )
    try:
        pending = RouteAmbiguityPendingV1(
            created,
            selection,
            intake,
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            fields[5],
            fields[6],
            fields[7],
            fields[8],
            fields[9],
            fields[10],
            tuple(candidates),
            output,
            readback,
        )
    except (ConversationRawRouteClarificationDialogueError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 无法恢复") from error
    if (pending.pending_identity_u8 != identity
            or pending.canonical_record() != values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return pending


def _restore_selection_event_record(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[
            RouteClarificationSelectionEventV1,
            RouteAmbiguityPendingV1,
            RouteAmbiguityCandidateProjectionV1,
        ]:
    """恢复 selection event，并同时恢复其独立 parent pending/candidate evidence。"""
    values = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(values, cursor, label=f"{label} version")
    if version != ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} version 未注册")
    ordinal, cursor = _read_scalar(values, cursor, label=f"{label} ordinal")
    fields: list[tuple[int, ...]] = []
    for field in (
            "predecessor identity", "pending", "pending identity",
            "selected candidate", "selected candidate identity", "selection intake",
            "inner turn", "inner response", "inner before state identity",
            "inner after state identity", "output", "output readback"):
        value, cursor = _read_segment(
            values,
            cursor,
            label=f"{label} {field}",
            allow_empty=(field == "predecessor identity"),
        )
        fields.append(value)
    identity, cursor = _read_segment(
        values, cursor, label=f"{label} event identity", allow_empty=False)
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 含尾随整数")
    pending = _restore_pending_record(fields[1], label=f"{label} pending")
    candidate = _restore_candidate_projection_record(
        fields[3], label=f"{label} selected candidate")
    intake = _restore_intake_record(fields[5], label=f"{label} selection intake")
    readback = _restore_output_readback(
        fields[11], fields[10], label=f"{label} output readback")
    try:
        event = RouteClarificationSelectionEventV1(
            ordinal,
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            fields[4],
            intake,
            fields[6],
            fields[7],
            fields[8],
            fields[9],
            fields[10],
            readback,
        )
    except (ConversationRawRouteClarificationDialogueError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 无法恢复") from error
    if (event.event_identity_u8 != identity
            or event.canonical_record() != values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return event, pending, candidate


def _current_selector_catalog(
        runtime: RouteClarificationDialogueRuntimeV1,
        ):
    """从当前 closure 重建 selector catalog，拒绝 runtime/cached course 漂移。"""
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification snapshot runtime 类型错误")
    try:
        catalog = validate_public_route_clarification_catalog_cached_v1(
            runtime.selector_catalog,
            runtime.terminal_runtime.inner_runtime.source_payload_closure,
            runtime.validation_cache,
        )
    except (PublicRouteClarificationCatalogError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot selector catalog 无法重建") from error
    if (catalog.canonical_record() != runtime.selector_catalog.canonical_record()
            or catalog.catalog_identity_u8
            != runtime.selector_catalog.catalog_identity_u8):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot selector catalog identity 漂移")
    return catalog


def _validate_pending_linkage(
        pending: RouteAmbiguityPendingV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        selector_catalog,
        *,
        label: str,
        ) -> None:
    """逐 record 验证 offer trace、state identity 和当前 selector course 重演。"""
    if not _pending_still_binds_runtime(pending, runtime, selector_catalog):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} selector course 或 candidate linkage 漂移")
    (
        before,
        before_operation,
        _inner_turn,
        response_record,
        response_trace,
        after,
        after_operation,
    ) = _terminal_turn_trace(pending.inner_turn_record, label=f"{label} inner turn")
    carrier_kind, base_result_code, intake, _output = response_trace
    if (before_operation != pending.created_operation_ordinal
            or after_operation != pending.selection_operation_ordinal
            or _state_identity_from_record(before)
            != pending.inner_before_state_identity_u8
            or _state_identity_from_record(after)
            != pending.inner_after_state_identity_u8
            or response_record != pending.inner_response_record
            or carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
            or base_result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
            or intake.canonical_record() != pending.input_intake.canonical_record()):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} offer terminal trace/linkage 漂移")


def _validate_selection_event_linkage(
        event: RouteClarificationSelectionEventV1,
        pending: RouteAmbiguityPendingV1,
        candidate: RouteAmbiguityCandidateProjectionV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        selector_catalog,
        *,
        label: str,
        ) -> None:
    """逐 record 验证 event 的 parent pending、完整重输及实际 Frame answer trace。"""
    _validate_pending_linkage(
        pending,
        runtime,
        selector_catalog,
        label=f"{label} pending",
    )
    selected = pending.candidate_for_identity(event.selected_candidate_identity_u8)
    by_intake = pending.candidate_for_intake(event.selection_intake)
    if (event.pending_identity_u8 != pending.pending_identity_u8
            or event.pending_record != pending.canonical_record()
            or selected is None
            or by_intake is None
            or selected.canonical_record() != candidate.canonical_record()
            or by_intake.canonical_record() != candidate.canonical_record()
            or event.selected_candidate_record != candidate.canonical_record()
            or event.selected_candidate_identity_u8
            != candidate.candidate_identity_u8):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} parent pending/candidate/selection input linkage 漂移")
    (
        before,
        before_operation,
        _inner_turn,
        response_record,
        response_trace,
        after,
        after_operation,
    ) = _terminal_turn_trace(event.inner_turn_record, label=f"{label} inner turn")
    carrier_kind, base_result_code, intake, output = response_trace
    if (before_operation != pending.selection_operation_ordinal
            or after_operation != before_operation + 1
            or _state_identity_from_record(before)
            != event.inner_before_state_identity_u8
            or _state_identity_from_record(after)
            != event.inner_after_state_identity_u8
            or event.inner_before_state_identity_u8
            != pending.inner_after_state_identity_u8
            or response_record != event.inner_response_record
            or carrier_kind != TERMINAL_DIALOGUE_BASE_CARRIER_FRAME_V1
            or base_result_code != DLG_RAW_ACCEPT
            or intake.canonical_record()
            != event.selection_intake.canonical_record()
            or output != event.output_u8):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} actual answer terminal trace/linkage 漂移")


def route_clarification_dialogue_snapshot_transport_record_v1() -> tuple[int, ...]:
    """冻结 DLG-RAW-14 bytes transport 的版本、预算、字节序与最短整数规则。"""
    return (
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1,
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_BYTES_V1,
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1,
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1,
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1,
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1,
    )


def route_clarification_dialogue_snapshot_runtime_binding_v1(
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """显式锁定 DLG-RAW-14/13 runtime、selector catalog、schema 与 transport。"""
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification snapshot runtime 类型错误")
    catalog = _current_selector_catalog(runtime)
    terminal = runtime.terminal_runtime
    result = [ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_BINDING_RECORD_V1]
    for label, value in (
            ("DLG-RAW-14 runtime binding", runtime.binding_record()),
            ("DLG-RAW-14 runtime identity", runtime.runtime_identity_u8()),
            ("DLG-RAW-13 runtime binding", terminal.binding_record()),
            ("DLG-RAW-13 runtime identity", terminal.runtime_identity_u8()),
            ("DLG-RAW-13 snapshot binding",
             terminal_dialogue_act_runtime_binding_v1(terminal)),
            ("selector catalog", catalog.canonical_record()),
            ("selector catalog identity", catalog.catalog_identity_u8),
            ("DLG-RAW-14 state/turn/response/pending/event schema", (
                ROUTE_CLARIFICATION_DIALOGUE_RUNTIME_BINDING_RECORD_V1,
                ROUTE_CLARIFICATION_DIALOGUE_STATE_RECORD_V1,
                ROUTE_CLARIFICATION_DIALOGUE_TURN_RECORD_V1,
                ROUTE_CLARIFICATION_RESPONSE_RECORD_V1,
                ROUTE_CLARIFICATION_PENDING_RECORD_V1,
                ROUTE_CLARIFICATION_CANDIDATE_PROJECTION_RECORD_V1,
                ROUTE_CLARIFICATION_SELECTION_EVENT_RECORD_V1,
                ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1,
            )),
            ("DLG-RAW-14 snapshot transport",
             route_clarification_dialogue_snapshot_transport_record_v1())):
        _pack(result, value, label=label)
    return tuple(result)


def route_clarification_dialogue_snapshot_runtime_identity_v1(
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 snapshot binding 的独立 portable identity，供 logical record 冗余锁定。"""
    return _identity(
        ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_IDENTITY_DOMAIN_V1,
        route_clarification_dialogue_snapshot_runtime_binding_v1(runtime),
        label="route clarification snapshot runtime",
    )


def snapshot_public_route_clarification_dialogue_state_v1(
        state: RouteClarificationDialogueStateV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 DLG-RAW-14 logical snapshot，嵌入未改写的 DLG-RAW-13 snapshot。"""
    if type(state) is not RouteClarificationDialogueStateV1:
        raise TypeError("route clarification snapshot state 类型错误")
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification snapshot runtime 类型错误")
    catalog = _current_selector_catalog(runtime)
    if state.pending is not None:
        _validate_pending_linkage(
            state.pending,
            runtime,
            catalog,
            label="route clarification active pending",
        )
    for ordinal, event in enumerate(state.selection_events, start=1):
        event_record = event.canonical_record()
        restored_event, parent_pending, candidate = _restore_selection_event_record(
            event_record,
            label=f"route clarification selection event {ordinal}",
        )
        if restored_event.canonical_record() != event_record:
            raise ConversationRawRouteClarificationDialogueSnapshotError(
                "route clarification selection event encoder readback 漂移")
        _validate_selection_event_linkage(
            restored_event,
            parent_pending,
            candidate,
            runtime,
            catalog,
            label=f"route clarification selection event {ordinal}",
        )
    binding = route_clarification_dialogue_snapshot_runtime_binding_v1(runtime)
    binding_identity = route_clarification_dialogue_snapshot_runtime_identity_v1(runtime)
    try:
        inner = snapshot_public_terminal_dialogue_act_state_v1(
            state.inner_state,
            runtime.terminal_runtime,
        )
    except (ConversationRawTerminalDialogueActSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification inner DLG-RAW-13 snapshot 无法形成") from error
    result = [ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RECORD_V1]
    _pack(result, binding, label="route clarification snapshot runtime binding")
    _pack(result, binding_identity,
          label="route clarification snapshot runtime identity")
    _pack(result, inner, label="route clarification DLG-RAW-13 inner snapshot")
    if state.pending is None:
        result.append(0)
    else:
        result.append(1)
        _pack(result, state.pending.canonical_record(),
              label="route clarification active pending")
    result.append(_u64(
        len(state.selection_events),
        label="route clarification selection event count",
    ))
    for ordinal, event in enumerate(state.selection_events, start=1):
        _pack(result, event.canonical_record(),
              label=f"route clarification selection event {ordinal}")
    record = tuple(result)
    restored = restore_public_route_clarification_dialogue_state_v1(record, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot encoder readback 漂移")
    return record


def restore_public_route_clarification_dialogue_state_v1(
        record: tuple[int, ...],
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> RouteClarificationDialogueStateV1:
    """严格恢复 active pending/event ledger，逐 record 验证 runtime 与 selector linkage。"""
    values = _record(record, label="route clarification snapshot", allow_empty=False)
    if type(runtime) is not RouteClarificationDialogueRuntimeV1:
        raise TypeError("route clarification snapshot runtime 类型错误")
    expected_binding = route_clarification_dialogue_snapshot_runtime_binding_v1(runtime)
    expected_identity = route_clarification_dialogue_snapshot_runtime_identity_v1(runtime)
    cursor = 0
    version, cursor = _read_scalar(
        values, cursor, label="route clarification snapshot version")
    if version != ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RECORD_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot version 未注册")
    binding, cursor = _read_segment(
        values,
        cursor,
        label="route clarification snapshot runtime binding",
        allow_empty=False,
    )
    if binding != expected_binding:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot runtime binding 漂移")
    binding_identity, cursor = _read_segment(
        values,
        cursor,
        label="route clarification snapshot runtime identity",
        allow_empty=False,
    )
    if binding_identity != expected_identity:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot runtime identity 漂移")
    inner_record, cursor = _read_segment(
        values,
        cursor,
        label="route clarification DLG-RAW-13 inner snapshot",
        allow_empty=False,
    )
    pending_present, cursor = _read_scalar(
        values, cursor, label="route clarification active pending present")
    if pending_present not in (0, 1):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification active pending presence 未注册")
    pending = None
    if pending_present:
        pending_record, cursor = _read_segment(
            values,
            cursor,
            label="route clarification active pending",
            allow_empty=False,
        )
        pending = _restore_pending_record(
            pending_record,
            label="route clarification active pending",
        )
    event_count, cursor = _read_scalar(
        values, cursor, label="route clarification selection event count")
    if event_count > ROUTE_CLARIFICATION_MAX_SELECTION_EVENTS_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification selection event count 超出预算")
    event_parts: list[
        tuple[
            RouteClarificationSelectionEventV1,
            RouteAmbiguityPendingV1,
            RouteAmbiguityCandidateProjectionV1,
        ]
    ] = []
    for ordinal in range(event_count):
        event_record, cursor = _read_segment(
            values,
            cursor,
            label=f"route clarification selection event {ordinal + 1}",
            allow_empty=False,
        )
        event_parts.append(_restore_selection_event_record(
            event_record,
            label=f"route clarification selection event {ordinal + 1}",
        ))
    if cursor != len(values):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot 含尾随整数")
    try:
        inner_state = restore_public_terminal_dialogue_act_state_v1(
            inner_record,
            runtime.terminal_runtime,
        )
    except (ConversationRawTerminalDialogueActSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification inner DLG-RAW-13 snapshot 无法恢复") from error
    if type(inner_state) is not TerminalDialogueActStateV1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification inner DLG-RAW-13 state 类型漂移")
    catalog = _current_selector_catalog(runtime)
    if pending is not None:
        _validate_pending_linkage(
            pending,
            runtime,
            catalog,
            label="route clarification active pending",
        )
    events = tuple(part[0] for part in event_parts)
    for ordinal, (event, parent_pending, candidate) in enumerate(
            event_parts, start=1):
        _validate_selection_event_linkage(
            event,
            parent_pending,
            candidate,
            runtime,
            catalog,
            label=f"route clarification selection event {ordinal}",
        )
    try:
        state = RouteClarificationDialogueStateV1(inner_state, pending, events)
    except (ConversationRawRouteClarificationDialogueError, TypeError, ValueError) as error:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification outer state 无法恢复") from error
    return state


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """按最短 unsigned big-endian 规则编码一个非负数学整数。"""
    if type(value) is not int or value < 0:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 必须是非负严格整数")
    count = max(1, (value.bit_length() + 7) // 8)
    _u64(count, label=f"{label} byte length")
    if count > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 超出 snapshot bytes 预算")
    return bytes((value >> shift) & 0xFF
                 for shift in range((count - 1) * 8, -1, -8))


def _read_u64_bytes(
        payload: bytes,
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """逐 byte 读取固定 big-endian u64，不依赖 struct/Pickle。"""
    if cursor > len(payload) - 8:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def encode_public_route_clarification_dialogue_snapshot_v1_bytes(
        state: RouteClarificationDialogueStateV1,
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> bytes:
    """把 logical snapshot 编码为固定 framing/minimal-unsigned raw bytes。"""
    record = snapshot_public_route_clarification_dialogue_state_v1(state, runtime)
    result = bytearray()
    for label, value in (
            ("route clarification snapshot bytes version",
             ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_BYTES_V1),
            ("route clarification snapshot bytes integer count", len(record))):
        result.extend(_u64(value, label=label).to_bytes(8, "big"))
    for ordinal, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value, label=f"route clarification snapshot integer[{ordinal}]")
        result.extend(_u64(
            len(encoded),
            label=f"route clarification snapshot integer[{ordinal}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
            raise ConversationRawRouteClarificationDialogueSnapshotError(
                "route clarification snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_route_clarification_dialogue_snapshot_v1_bytes(
        payload, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes encoder readback 漂移")
    return payload


def decode_public_route_clarification_dialogue_snapshot_v1_bytes(
        payload: bytes,
        runtime: RouteClarificationDialogueRuntimeV1,
        ) -> RouteClarificationDialogueStateV1:
    """严格解码 bytes snapshot，拒绝截断、leading zero 与尾随 physical bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes 必须是非空 raw bytes")
    if len(payload) > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="route clarification snapshot bytes version")
    if version != ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_BYTES_V1:
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload,
        cursor,
        label="route clarification snapshot bytes integer count",
    )
    if (count > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or count > (len(payload) - cursor) // 9):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes integer count 越界")
    values: list[int] = []
    for ordinal in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"route clarification snapshot integer[{ordinal}] length",
        )
        if (size < 1 or size > len(payload) - cursor
                or size > ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1):
            raise ConversationRawRouteClarificationDialogueSnapshotError(
                f"route clarification snapshot integer[{ordinal}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawRouteClarificationDialogueSnapshotError(
                f"route clarification snapshot integer[{ordinal}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawRouteClarificationDialogueSnapshotError(
            "route clarification snapshot bytes 含尾随 bytes")
    return restore_public_route_clarification_dialogue_state_v1(
        tuple(values), runtime)


__all__ = [
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_BYTES_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_MAX_BYTES_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_BINDING_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_RUNTIME_IDENTITY_DOMAIN_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1",
    "ROUTE_CLARIFICATION_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1",
    "ConversationRawRouteClarificationDialogueSnapshotError",
    "decode_public_route_clarification_dialogue_snapshot_v1_bytes",
    "encode_public_route_clarification_dialogue_snapshot_v1_bytes",
    "restore_public_route_clarification_dialogue_state_v1",
    "route_clarification_dialogue_snapshot_runtime_binding_v1",
    "route_clarification_dialogue_snapshot_runtime_identity_v1",
    "route_clarification_dialogue_snapshot_transport_record_v1",
    "snapshot_public_route_clarification_dialogue_state_v1",
]
