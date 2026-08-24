"""DLG-RAW-12：V3 append-only focus context 的独立快照与 bytes codec。

V2/V4 snapshot 不在本模块的输入域内。本 codec 只恢复 V3 canonical records，bytes
transport 固定为 ``u64 version/count/length || minimal unsigned big-endian``。恢复
Frame 时只由此前已恢复的 V3 Frame 前缀重放 legacy compatibility chain；provider 与
focus event 永远不进入该旧链。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationContextState,
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    PROVIDER_ORIGIN_ANCHOR_RECORD_V1,
    ProviderOriginAnchorError,
    ProviderOriginAnchorProjectionV1,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1,
    MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1,
    MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3,
    MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3,
    MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3,
    MIXED_FOCUS_CONTEXT_SCHEMA_V3,
    MIXED_FOCUS_CONTEXT_STATE_RECORD_V3,
    MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS,
    MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    FocusContextReadWitnessV3,
    FrameQuestionAnswerTurnV3,
    MixedConversationFocusContextStateV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFollowupFocusTurnV1,
    ProviderOriginFocusAdmissionV1,
    ProviderOriginFocusContextError,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)


MIXED_FOCUS_CONTEXT_SNAPSHOT_RECORD_V3 = 3
MIXED_FOCUS_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V3 = MIXED_FOCUS_CONTEXT_STATE_RECORD_V3
MIXED_FOCUS_CONTEXT_SNAPSHOT_BYTES_V3 = 3
MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_REVISION_V3 = 3
MIXED_FOCUS_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V3 = 1
MIXED_FOCUS_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V3 = 1
MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3 = 4 * 1024 * 1024
MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V3 = (
    MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3 // 9)

MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V3 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/MIXED-FOCUS-CONTEXT-SNAPSHOT-CODEC/V3")

_U64_EXCLUSIVE = 1 << 64
_DIGEST_SIZE = 32


# object-model: exception; interop=DLG-RAW-12
class MixedFocusContextSnapshotError(ValueError):
    """V3 focus snapshot record、bytes transport 或重放链不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """核验 transport count、length 与版本使用显式无符号 64-bit 范围。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise MixedFocusContextSnapshotError(f"{label} 必须是非负 u64")
    return value


def _record(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 codec 输入是预算内、非空的有限非负整数 record。"""
    if (type(value) is not tuple
            or not value
            or len(value) > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V3
            or any(type(item) is not int or item < 0 for item in value)):
        raise MixedFocusContextSnapshotError(
            f"{label} 必须是预算内的非空非负严格整数 tuple")
    return value


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """读取一项已整体校验的整数，显式拒绝截断。"""
    if cursor >= len(record):
        raise MixedFocusContextSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_count(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """读取 count 字段，不依赖宿主容器长度作为协议。"""
    value, cursor = _read_scalar(record, cursor, label=label)
    return _u64(value, label=label), cursor


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], int]:
    """读取 count-framed record 段并拒绝越界或空值偷换。"""
    count, cursor = _read_count(record, cursor, label=f"{label} count")
    if count > len(record) - cursor:
        raise MixedFocusContextSnapshotError(f"{label} 长度越界")
    value = record[cursor:cursor + count]
    cursor += count
    if not allow_empty and not value:
        raise MixedFocusContextSnapshotError(f"{label} 不得为空")
    return value, cursor


def _read_digest(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[tuple[int, ...], int]:
    """读取 raw u8[32]，拒绝 hex、变长或非 byte identity。"""
    value, cursor = _read_segment(
        record,
        cursor,
        label=label,
        allow_empty=False,
    )
    if len(value) != _DIGEST_SIZE or any(item > 255 for item in value):
        raise MixedFocusContextSnapshotError(f"{label} 必须是 raw u8[32]")
    return value, cursor


def _object_identity(value: tuple[int, ...], *, label: str) -> ObjectIdentity:
    """由完整 stable key 恢复 response stance，不保留 host object identity。"""
    try:
        return ObjectIdentity.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            f"{label} 不是完整 ObjectIdentity stable key") from error


def _source_ref(value: tuple[int, ...], *, label: str) -> SourceRef:
    """由完整 stable key 恢复 citation，不降解为局部 source id。"""
    try:
        return SourceRef.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            f"{label} 不是完整 SourceRef stable key") from error


def _legacy_read_descriptor(
        record: tuple[int, ...],
        ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
    """解析 legacy Frame read 的 key/revision/digest/visible-count 描述。"""
    record = _record(record, label="focus V3 legacy frame context read")
    cursor = 0
    _version, cursor = _read_scalar(
        record, cursor, label="focus V3 legacy frame read version")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 legacy frame read conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="focus V3 legacy frame read revision")
    digest, cursor = _read_digest(
        record, cursor, label="focus V3 legacy frame read digest")
    visible_count, cursor = _read_count(
        record, cursor, label="focus V3 legacy frame read visible count")
    for ordinal in range(visible_count):
        _fingerprint, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 legacy frame turn fingerprint[{ordinal}]",
            allow_empty=False,
        )
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 legacy frame read 含尾随整数")
    return conversation_key, revision, digest, visible_count


def _find_legacy_context(
        contexts: tuple[ConversationContextState, ...],
        conversation_key: tuple[int, ...],
        ) -> ConversationContextState | None:
    """按 stable conversation key 获取此前 Frame 前缀重放出的旧链。"""
    for context in contexts:
        if context.conversation_key == conversation_key:
            return context
    return None


def _replace_legacy_context(
        contexts: tuple[ConversationContextState, ...],
        replacement: ConversationContextState,
        ) -> tuple[ConversationContextState, ...]:
    """以稳定 key 替换一条旧链；首次 key 以输入顺序追加。"""
    result: list[ConversationContextState] = []
    replaced = False
    for context in contexts:
        if context.conversation_key == replacement.conversation_key:
            result.append(replacement)
            replaced = True
        else:
            result.append(context)
    return tuple(result) if replaced else (*contexts, replacement)


def _replay_legacy_read(
        read_record: tuple[int, ...],
        contexts: tuple[ConversationContextState, ...],
        ) -> tuple[ConversationContextRead, ConversationContextState]:
    """由已恢复 Frame 前缀重建 legacy read，并逐整数对照其 stable record。"""
    key, revision, _digest, visible_count = _legacy_read_descriptor(read_record)
    context = _find_legacy_context(contexts, key)
    if context is None:
        if revision != 0:
            raise MixedFocusContextSnapshotError(
                "focus V3 legacy Frame read 缺少可重放前缀")
        try:
            context = start_conversation_context(key)
        except (TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError(
                "focus V3 legacy Frame read conversation key 非法") from error
    if context.revision != revision:
        raise MixedFocusContextSnapshotError(
            "focus V3 legacy Frame read revision 与前缀漂移")
    try:
        expected = context.read(visible_count)
    except (TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 legacy Frame read 无法形成") from error
    if expected.stable_key() != read_record:
        raise MixedFocusContextSnapshotError(
            "focus V3 legacy Frame read digest 或尾部漂移")
    return expected, context


def _decode_legacy_frame_turn(
        record: tuple[int, ...],
        contexts: tuple[ConversationContextState, ...],
        ) -> tuple[ConversationTurnState, tuple[ConversationContextState, ...]]:
    """显式重建 Frame typed record 及其仅由 Frame 组成的 compatibility chain。"""
    record = _record(record, label="focus V3 Frame legacy typed record")
    cursor = 0
    _version, cursor = _read_scalar(
        record, cursor, label="focus V3 Frame legacy turn version")
    ordinal, cursor = _read_scalar(
        record, cursor, label="focus V3 Frame legacy turn ordinal")
    fields: list[tuple[int, ...]] = []
    for label in (
            "request key", "target key", "query key", "planning key",
            "parser revision", "readback key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 Frame legacy {label}",
            allow_empty=False,
        )
        fields.append(value)
    stance_key, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 Frame legacy response stance",
        allow_empty=False,
    )
    selected_count, cursor = _read_count(
        record,
        cursor,
        label="focus V3 Frame legacy selected candidate count",
    )
    selected: list[tuple[int, ...]] = []
    for item_ordinal in range(selected_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=("focus V3 Frame legacy selected candidate"
                   f"[{item_ordinal}]"),
            allow_empty=False,
        )
        selected.append(value)
    source_count, cursor = _read_count(
        record,
        cursor,
        label="focus V3 Frame legacy cited source count",
    )
    sources: list[SourceRef] = []
    for item_ordinal in range(source_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 Frame legacy cited source[{item_ordinal}]",
            allow_empty=False,
        )
        sources.append(_source_ref(
            value,
            label=f"focus V3 Frame legacy cited source[{item_ordinal}]",
        ))
    sentence_count, cursor = _read_count(
        record,
        cursor,
        label="focus V3 Frame legacy discourse sentence count",
    )
    sentences: list[tuple[int, ...]] = []
    for item_ordinal in range(sentence_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=("focus V3 Frame legacy discourse sentence"
                   f"[{item_ordinal}]"),
            allow_empty=False,
        )
        sentences.append(value)
    has_context_read, cursor = _read_scalar(
        record,
        cursor,
        label="focus V3 Frame legacy has context read",
    )
    if has_context_read != 0 and has_context_read != 1:
        raise MixedFocusContextSnapshotError(
            "focus V3 Frame legacy context read flag 未注册")
    prior_read: ConversationContextRead | None = None
    prior_context: ConversationContextState | None = None
    if has_context_read:
        read_record, cursor = _read_segment(
            record,
            cursor,
            label="focus V3 Frame legacy context read",
            allow_empty=False,
        )
        prior_read, prior_context = _replay_legacy_read(read_record, contexts)
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 Frame legacy typed record 含尾随整数")
    request_key, target_key, query_key, planning_key, parser_revision, readback_key = fields
    try:
        turn = ConversationTurnState(
            ordinal,
            request_key,
            target_key,
            query_key,
            planning_key,
            _object_identity(
                stance_key,
                label="focus V3 Frame legacy response stance",
            ),
            tuple(selected),
            tuple(sources),
            tuple(sentences),
            parser_revision,
            readback_key,
            prior_read,
        )
    except (TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 Frame legacy typed record 无法恢复") from error
    if turn.stable_key() != record:
        raise MixedFocusContextSnapshotError(
            "focus V3 Frame legacy typed record 非规范或字段漂移")
    if prior_context is None:
        return turn, contexts
    try:
        updated = ConversationContextState(
            prior_context.conversation_key,
            prior_context.revision + 1,
            prior_context.digest(),
            (*prior_context.turns, turn),
        )
    except (TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 Frame legacy compatibility chain 无法追加") from error
    return turn, _replace_legacy_context(contexts, updated)


def _decode_role_bindings(
        record: tuple[int, ...],
        ) -> tuple[ProviderOriginRoleBindingV1, ...]:
    """由 anchor 保序 record 恢复 role bindings，不借助 map 或 host 排序。"""
    record = _record(record, label="focus V3 anchor role binding record")
    cursor = 0
    count, cursor = _read_count(
        record, cursor, label="focus V3 anchor role binding count")
    result: list[ProviderOriginRoleBindingV1] = []
    for ordinal in range(count):
        binding_key, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 anchor binding[{ordinal}] key",
            allow_empty=False,
        )
        role_key, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 anchor binding[{ordinal}] role key",
            allow_empty=False,
        )
        filler_key, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 anchor binding[{ordinal}] filler key",
            allow_empty=False,
        )
        item_ordinal, cursor = _read_scalar(
            record,
            cursor,
            label=f"focus V3 anchor binding[{ordinal}] ordinal",
        )
        try:
            result.append(ProviderOriginRoleBindingV1(
                binding_key,
                role_key,
                filler_key,
                item_ordinal,
            ))
        except (TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError(
                f"focus V3 anchor binding[{ordinal}] 非法") from error
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 anchor role binding record 含尾随整数")
    return tuple(result)


def _decode_occurrences(
        record: tuple[int, ...],
        ) -> tuple[ProviderOriginOccurrenceV1, ...]:
    """由 anchor 保序 record 恢复 occurrence，保留 source-local ordinal 与 span。"""
    record = _record(record, label="focus V3 anchor occurrence record")
    cursor = 0
    count, cursor = _read_count(
        record, cursor, label="focus V3 anchor occurrence count")
    result: list[ProviderOriginOccurrenceV1] = []
    for ordinal in range(count):
        occurrence_key, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 anchor occurrence[{ordinal}] key",
            allow_empty=False,
        )
        semantic_object_key, cursor = _read_segment(
            record,
            cursor,
            label=("focus V3 anchor occurrence"
                   f"[{ordinal}] semantic object key"),
            allow_empty=False,
        )
        item_ordinal, cursor = _read_scalar(
            record,
            cursor,
            label=f"focus V3 anchor occurrence[{ordinal}] ordinal",
        )
        start, cursor = _read_scalar(
            record,
            cursor,
            label=f"focus V3 anchor occurrence[{ordinal}] start",
        )
        end, cursor = _read_scalar(
            record,
            cursor,
            label=f"focus V3 anchor occurrence[{ordinal}] end",
        )
        try:
            result.append(ProviderOriginOccurrenceV1(
                occurrence_key,
                semantic_object_key,
                item_ordinal,
                start,
                end,
            ))
        except (TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError(
                f"focus V3 anchor occurrence[{ordinal}] 非法") from error
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 anchor occurrence record 含尾随整数")
    return tuple(result)


def _decode_anchor(record: tuple[int, ...]) -> ProviderOriginAnchorProjectionV1:
    """由完整 canonical record 重建 source anchor，不访问 provider/runtime。"""
    record = _record(record, label="focus V3 provider origin anchor")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 provider origin anchor version")
    if version != PROVIDER_ORIGIN_ANCHOR_RECORD_V1:
        raise MixedFocusContextSnapshotError(
            "focus V3 provider origin anchor version 未注册")
    status, cursor = _read_scalar(
        record, cursor, label="focus V3 provider origin anchor status")
    provider_kind, cursor = _read_scalar(
        record, cursor, label="focus V3 provider origin anchor provider kind")
    first: list[tuple[int, ...]] = []
    for label in (
            "provider identity", "runtime identity", "catalog record identity",
            "provider result identity", "input intake identity",
            "output readback identity", "source record key", "source ref stable key",
            "source commitment", "w03 observation key", "w04 observation key",
            "w05 observation key", "proposition key", "predicate key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 provider origin anchor {label}",
        )
        first.append(value)
    relation_kind, cursor = _read_scalar(
        record,
        cursor,
        label="focus V3 provider origin anchor relation kind",
    )
    focus: list[tuple[int, ...]] = []
    for label in (
            "generation construction key", "focus role binding key",
            "focus role key", "focus filler key", "focus occurrence key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 provider origin anchor {label}",
        )
        focus.append(value)
    focus_start, cursor = _read_scalar(
        record, cursor, label="focus V3 provider origin anchor focus start")
    focus_end, cursor = _read_scalar(
        record, cursor, label="focus V3 provider origin anchor focus end")
    binding_record, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 provider origin anchor role bindings",
        allow_empty=False,
    )
    occurrence_record, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 provider origin anchor occurrences",
        allow_empty=False,
    )
    output_scalars, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 provider origin anchor output scalars",
    )
    output_u8, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 provider origin anchor output u8",
    )
    anchor_identity, cursor = _read_digest(
        record,
        cursor,
        label="focus V3 provider origin anchor identity",
    )
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 provider origin anchor 含尾随整数")
    try:
        anchor = ProviderOriginAnchorProjectionV1(
            status,
            provider_kind,
            first[0], first[1], first[2], first[3], first[4], first[5],
            first[6], first[7], first[8], first[9], first[10], first[11],
            first[12], first[13], relation_kind,
            focus[0], focus[1], focus[2], focus[3], focus[4],
            focus_start, focus_end,
            _decode_role_bindings(binding_record),
            _decode_occurrences(occurrence_record),
            output_scalars,
            output_u8,
            anchor_identity,
        )
    except (ProviderOriginAnchorError, TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 provider origin anchor 无法按 canonical record 恢复") from error
    if anchor.canonical_record() != record:
        raise MixedFocusContextSnapshotError(
            "focus V3 provider origin anchor canonical record 漂移")
    return anchor


def _decode_read_witness(record: tuple[int, ...]) -> FocusContextReadWitnessV3:
    """解析 V3 read witness，并让结构体重新计算 self identity。"""
    record = _record(record, label="focus V3 read witness")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 read witness version")
    if version != MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3:
        raise MixedFocusContextSnapshotError("focus V3 read witness version 未注册")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 read witness conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="focus V3 read witness revision")
    snapshot_digest, cursor = _read_digest(
        record,
        cursor,
        label="focus V3 read witness snapshot digest",
    )
    requested_limit, cursor = _read_scalar(
        record, cursor, label="focus V3 read witness requested limit")
    visible_start, cursor = _read_scalar(
        record, cursor, label="focus V3 read witness visible start ordinal")
    count, cursor = _read_count(
        record, cursor, label="focus V3 read witness visible identity count")
    identities: list[tuple[int, ...]] = []
    for ordinal in range(count):
        value, cursor = _read_digest(
            record,
            cursor,
            label=f"focus V3 read witness visible identity[{ordinal}]",
        )
        identities.append(value)
    witness_identity, cursor = _read_digest(
        record, cursor, label="focus V3 read witness identity")
    if cursor != len(record):
        raise MixedFocusContextSnapshotError("focus V3 read witness 含尾随整数")
    try:
        witness = FocusContextReadWitnessV3(
            conversation_key,
            revision,
            snapshot_digest,
            requested_limit,
            visible_start,
            tuple(identities),
            witness_identity,
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 read witness 无法恢复") from error
    if witness.canonical_record() != record:
        raise MixedFocusContextSnapshotError(
            "focus V3 read witness canonical record 漂移")
    return witness


def _decode_admission(record: tuple[int, ...]) -> ProviderOriginFocusAdmissionV1:
    """严格恢复完整 follow-up admission carrier，不采用对象引用或 JSON。"""
    record = _record(record, label="focus V3 admission")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 admission version")
    if version != MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1:
        raise MixedFocusContextSnapshotError("focus V3 admission version 未注册")
    provider_kind, cursor = _read_scalar(
        record, cursor, label="focus V3 admission provider kind")
    relation_kind, cursor = _read_scalar(
        record, cursor, label="focus V3 admission relation kind")
    reference_start, cursor = _read_scalar(
        record, cursor, label="focus V3 admission reference start")
    reference_end, cursor = _read_scalar(
        record, cursor, label="focus V3 admission reference end")
    target_start, cursor = _read_scalar(
        record, cursor, label="focus V3 admission target start")
    target_end, cursor = _read_scalar(
        record, cursor, label="focus V3 admission target end")
    segments: list[tuple[int, ...]] = []
    for label in (
            "catalog record", "catalog identity", "form record", "form identity",
            "candidate record", "candidate identity", "input intake record",
            "input intake identity", "output readback record",
            "output readback identity", "provider identity", "runtime identity",
            "provider catalog identity", "provider result identity", "anchor identity",
            "source record key", "source ref stable key", "source commitment",
            "w03 observation key", "w04 observation key", "w05 observation key",
            "generation construction key", "proposition key", "predicate key",
            "reference role binding key", "reference role key", "reference filler key",
            "reference occurrence key", "target role binding key", "target role key",
            "target filler key", "target occurrence key", "target output scalars",
            "target output u8"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 admission {label}",
            allow_empty=False,
        )
        segments.append(value)
    admission_identity, cursor = _read_digest(
        record, cursor, label="focus V3 admission identity")
    if cursor != len(record):
        raise MixedFocusContextSnapshotError("focus V3 admission 含尾随整数")
    try:
        admission = ProviderOriginFocusAdmissionV1(
            segments[0], segments[1], segments[2], segments[3],
            segments[4], segments[5], segments[6], segments[7],
            segments[8], segments[9], provider_kind,
            segments[10], segments[11], segments[12], segments[13], segments[14],
            segments[15], segments[16], segments[17], segments[18], segments[19],
            segments[20], segments[21], segments[22], segments[23], relation_kind,
            segments[24], segments[25], segments[26], segments[27],
            reference_start, reference_end,
            segments[28], segments[29], segments[30], segments[31],
            target_start, target_end, segments[32], segments[33], admission_identity,
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError("focus V3 admission 无法恢复") from error
    if admission.canonical_record() != record:
        raise MixedFocusContextSnapshotError(
            "focus V3 admission canonical record 漂移")
    return admission


def _decode_turn(
        record: tuple[int, ...],
        legacy_contexts: tuple[ConversationContextState, ...],
        ) -> tuple[
            FrameQuestionAnswerTurnV3
            | ProviderOriginContextTurnV3
            | ProviderOriginFollowupFocusTurnV1,
            tuple[ConversationContextState, ...],
        ]:
    """按 V3 kind 解码一条 turn，且只让 Frame 接触 legacy compatibility chain。"""
    record = _record(record, label="focus V3 context turn")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 turn record version")
    turn_kind, cursor = _read_scalar(
        record, cursor, label="focus V3 turn kind")
    ordinal, cursor = _read_scalar(
        record, cursor, label="focus V3 turn append ordinal")
    previous_digest, cursor = _read_digest(
        record,
        cursor,
        label="focus V3 turn previous snapshot digest",
    )
    witness_record, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 turn prior read witness",
        allow_empty=False,
    )
    witness = _decode_read_witness(witness_record)
    if turn_kind == MIXED_FOCUS_CONTEXT_TURN_KIND_FRAME_QA_RUN:
        if version != MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3:
            raise MixedFocusContextSnapshotError("focus V3 Frame turn version 未注册")
        frame_record, cursor = _read_segment(
            record,
            cursor,
            label="focus V3 Frame legacy payload",
            allow_empty=False,
        )
        write_origin, cursor = _read_scalar(
            record, cursor, label="focus V3 Frame write origin")
        turn_identity, cursor = _read_digest(
            record, cursor, label="focus V3 Frame turn identity")
        if cursor != len(record):
            raise MixedFocusContextSnapshotError("focus V3 Frame turn 含尾随整数")
        frame_turn, next_contexts = _decode_legacy_frame_turn(
            frame_record,
            legacy_contexts,
        )
        try:
            turn = FrameQuestionAnswerTurnV3(
                ordinal,
                previous_digest,
                witness,
                frame_turn,
                write_origin,
                turn_identity,
            )
        except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError("focus V3 Frame turn 无法恢复") from error
    elif turn_kind == MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION:
        if version != MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3:
            raise MixedFocusContextSnapshotError("focus V3 provider turn version 未注册")
        anchor_record, cursor = _read_segment(
            record,
            cursor,
            label="focus V3 provider anchor payload",
            allow_empty=False,
        )
        provider_result_identity, cursor = _read_digest(
            record,
            cursor,
            label="focus V3 provider result identity",
        )
        write_origin, cursor = _read_scalar(
            record, cursor, label="focus V3 provider write origin")
        turn_identity, cursor = _read_digest(
            record, cursor, label="focus V3 provider turn identity")
        if cursor != len(record):
            raise MixedFocusContextSnapshotError(
                "focus V3 provider turn 含尾随整数")
        try:
            turn = ProviderOriginContextTurnV3(
                ordinal,
                previous_digest,
                witness,
                _decode_anchor(anchor_record),
                provider_result_identity,
                write_origin,
                turn_identity,
            )
        except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError(
                "focus V3 provider turn 无法恢复") from error
        next_contexts = legacy_contexts
    elif turn_kind == MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_FOLLOWUP_FOCUS:
        if version != MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1:
            raise MixedFocusContextSnapshotError("focus V3 follow-up turn version 未注册")
        anchor_record, cursor = _read_segment(
            record,
            cursor,
            label="focus V3 follow-up parent anchor",
            allow_empty=False,
        )
        parent_identity, cursor = _read_digest(
            record,
            cursor,
            label="focus V3 follow-up parent turn identity",
        )
        admission_record, cursor = _read_segment(
            record,
            cursor,
            label="focus V3 follow-up admission",
            allow_empty=False,
        )
        write_origin, cursor = _read_scalar(
            record, cursor, label="focus V3 follow-up write origin")
        turn_identity, cursor = _read_digest(
            record, cursor, label="focus V3 follow-up turn identity")
        if cursor != len(record):
            raise MixedFocusContextSnapshotError(
                "focus V3 follow-up turn 含尾随整数")
        try:
            turn = ProviderOriginFollowupFocusTurnV1(
                ordinal,
                previous_digest,
                witness,
                _decode_anchor(anchor_record),
                parent_identity,
                _decode_admission(admission_record),
                write_origin,
                turn_identity,
            )
        except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
            raise MixedFocusContextSnapshotError(
                "focus V3 follow-up turn 无法恢复") from error
        next_contexts = legacy_contexts
    else:
        raise MixedFocusContextSnapshotError("focus V3 turn kind 未注册")
    if turn.canonical_record() != record:
        raise MixedFocusContextSnapshotError("focus V3 turn canonical record 漂移")
    return turn, next_contexts


def _restore_context(record: tuple[int, ...]) -> MixedConversationFocusContextStateV3:
    """重放 complete V3 context，逐 event 交给 state 构造器核验 parent/read 链。"""
    record = _record(record, label="focus V3 snapshot context")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 snapshot context version")
    if version != MIXED_FOCUS_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V3:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot context version 未注册")
    schema, cursor = _read_scalar(
        record, cursor, label="focus V3 snapshot schema")
    if schema != MIXED_FOCUS_CONTEXT_SCHEMA_V3:
        raise MixedFocusContextSnapshotError("focus V3 snapshot schema 未注册")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 snapshot conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="focus V3 snapshot revision")
    previous_digest, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 snapshot previous digest",
    )
    if previous_digest and (len(previous_digest) != _DIGEST_SIZE
                            or any(item > 255 for item in previous_digest)):
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot previous digest 非法")
    turn_count, cursor = _read_count(
        record, cursor, label="focus V3 snapshot turn count")
    turns: list[
        FrameQuestionAnswerTurnV3
        | ProviderOriginContextTurnV3
        | ProviderOriginFollowupFocusTurnV1
    ] = []
    legacy_contexts: tuple[ConversationContextState, ...] = ()
    for ordinal in range(turn_count):
        turn_record, cursor = _read_segment(
            record,
            cursor,
            label=f"focus V3 snapshot turn[{ordinal}]",
            allow_empty=False,
        )
        turn, legacy_contexts = _decode_turn(turn_record, legacy_contexts)
        turns.append(turn)
    if cursor != len(record):
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot context 含尾随整数")
    try:
        state = MixedConversationFocusContextStateV3(
            conversation_key,
            revision,
            previous_digest,
            tuple(turns),
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot state 无法恢复") from error
    if state.canonical_record() != record:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot context canonical record 漂移")
    return state


def snapshot_mixed_conversation_focus_context_v3(
        state: MixedConversationFocusContextStateV3,
        ) -> tuple[int, ...]:
    """导出 V3 top-level snapshot record；不包含 runtime binding 或物理路径。"""
    if type(state) is not MixedConversationFocusContextStateV3:
        raise TypeError("focus V3 snapshot 需要 MixedConversationFocusContextStateV3")
    context_record = _record(
        state.canonical_record(),
        label="focus V3 snapshot state record",
    )
    _u64(len(context_record), label="focus V3 snapshot context count")
    result = (
        MIXED_FOCUS_CONTEXT_SNAPSHOT_RECORD_V3,
        len(context_record),
        *context_record,
    )
    restored = _restore_context(context_record)
    if restored.canonical_record() != context_record:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot state readback 漂移")
    return result


def restore_mixed_conversation_focus_context_v3(
        record: tuple[int, ...],
        ) -> MixedConversationFocusContextStateV3:
    """严格恢复 V3 snapshot；拒绝 V2/V4、截断、升级猜测与尾随整数。"""
    record = _record(record, label="focus V3 snapshot")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="focus V3 snapshot version")
    if version != MIXED_FOCUS_CONTEXT_SNAPSHOT_RECORD_V3:
        raise MixedFocusContextSnapshotError("focus V3 snapshot version 未注册")
    context_record, cursor = _read_segment(
        record,
        cursor,
        label="focus V3 snapshot context",
        allow_empty=False,
    )
    if cursor != len(record):
        raise MixedFocusContextSnapshotError("focus V3 snapshot 含尾随整数")
    return _restore_context(context_record)


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """用最短 unsigned-big-endian byte sequence 表示一个非负数学整数。"""
    if type(value) is not int or value < 0:
        raise MixedFocusContextSnapshotError(f"{label} 必须是非负严格整数")
    size = max(1, (value.bit_length() + 7) // 8)
    _u64(size, label=f"{label} byte length")
    if size > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3:
        raise MixedFocusContextSnapshotError(f"{label} 超出 snapshot bytes 预算")
    return value.to_bytes(size, "big")


def encode_mixed_conversation_focus_context_snapshot_v3_bytes(
        state: MixedConversationFocusContextStateV3,
        ) -> bytes:
    """以固定 u64 count/length 与最短无符号大端整数导出 V3 bytes transport。"""
    record = snapshot_mixed_conversation_focus_context_v3(state)
    _u64(len(record), label="focus V3 snapshot bytes integer count")
    result = bytearray()
    result.extend(_u64(
        MIXED_FOCUS_CONTEXT_SNAPSHOT_BYTES_V3,
        label="focus V3 snapshot bytes version",
    ).to_bytes(8, "big"))
    result.extend(len(record).to_bytes(8, "big"))
    for ordinal, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value,
            label=f"focus V3 snapshot integer[{ordinal}]",
        )
        result.extend(_u64(
            len(encoded),
            label=f"focus V3 snapshot integer[{ordinal}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3:
            raise MixedFocusContextSnapshotError(
                "focus V3 snapshot bytes 超出固定预算")
    return bytes(result)


def _read_u64_bytes(
        payload: bytes,
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """逐 byte 读取 u64，拒绝截断并不借用 slicing 的部分结果。"""
    if cursor > len(payload) - 8:
        raise MixedFocusContextSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def decode_mixed_conversation_focus_context_snapshot_v3_bytes(
        payload: bytes,
        ) -> MixedConversationFocusContextStateV3:
    """严格 decode V3 bytes 并恢复 state；不返回部分状态。"""
    if type(payload) is not bytes or not payload:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot bytes 必须是非空 raw bytes")
    if len(payload) > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3:
        raise MixedFocusContextSnapshotError("focus V3 snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="focus V3 snapshot bytes version")
    if version != MIXED_FOCUS_CONTEXT_SNAPSHOT_BYTES_V3:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload,
        cursor,
        label="focus V3 snapshot bytes integer count",
    )
    if (count > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V3
            or count > (len(payload) - cursor) // 9):
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot bytes integer count 越界")
    values: list[int] = []
    for ordinal in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"focus V3 snapshot integer[{ordinal}] length",
        )
        if (size < 1
                or size > MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3
                or size > len(payload) - cursor):
            raise MixedFocusContextSnapshotError(
                f"focus V3 snapshot integer[{ordinal}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise MixedFocusContextSnapshotError(
                f"focus V3 snapshot integer[{ordinal}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot bytes 含尾随 bytes")
    return restore_mixed_conversation_focus_context_v3(tuple(values))


def mixed_focus_context_snapshot_codec_revision_v3() -> tuple[int, ...]:
    """返回 runtime binding 可逐整数锁定的 V3 codec layout revision。"""
    return (
        MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_REVISION_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_RECORD_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V3,
        MIXED_FOCUS_CONTEXT_SCHEMA_V3,
        MIXED_FOCUS_CONTEXT_FRAME_TURN_RECORD_V3,
        MIXED_FOCUS_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V3,
        MIXED_FOCUS_CONTEXT_FOLLOWUP_FOCUS_TURN_RECORD_V1,
        MIXED_FOCUS_CONTEXT_FOCUS_ADMISSION_RECORD_V1,
        MIXED_FOCUS_CONTEXT_READ_WITNESS_RECORD_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_BYTES_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3,
        MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V3,
    )


def mixed_focus_context_snapshot_codec_identity_v3() -> tuple[int, ...]:
    """返回 V3 codec layout revision 的 raw u8[32] identity。"""
    try:
        return tuple(portable_sha256_v1(
            MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V3,
            (mixed_focus_context_snapshot_codec_revision_v3(),),
        ))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise MixedFocusContextSnapshotError(
            "focus V3 snapshot codec identity 无法形成") from error


__all__ = [
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_BYTES_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_CODEC_REVISION_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V3",
    "MIXED_FOCUS_CONTEXT_SNAPSHOT_RECORD_V3",
    "MixedFocusContextSnapshotError",
    "decode_mixed_conversation_focus_context_snapshot_v3_bytes",
    "encode_mixed_conversation_focus_context_snapshot_v3_bytes",
    "mixed_focus_context_snapshot_codec_identity_v3",
    "mixed_focus_context_snapshot_codec_revision_v3",
    "restore_mixed_conversation_focus_context_v3",
    "snapshot_mixed_conversation_focus_context_v3",
]
