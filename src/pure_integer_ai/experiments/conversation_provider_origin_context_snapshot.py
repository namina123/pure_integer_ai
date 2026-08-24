"""DLG-RAW-11B：mixed context V2 的独立可迁移 snapshot codec。

本模块只处理已形成的 ``MixedConversationContextStateV2``。它不读取路径、
runtime、provider、SQLite 或表层文本；所有 snapshot 都是有序的非负整数
record，bytes transport 使用 ``u64 count/length || unsigned-big-endian integer``。

Frame payload 引用既有 ``ConversationTurnState.stable_key()``。其中 legacy
``ConversationContextRead`` 的指纹不可被反推，因此 decoder 会由已恢复的 V2
前缀中全部 ``FRAME_QA_RUN`` 重建独立 legacy compatibility chain（provider turn
不参与该旧链），然后逐整数比较 read witness。缺少前缀或任何漂移一律拒绝。
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
    ProviderOriginAnchorProjectionV1,
    ProviderOriginAnchorError,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_FRAME_TURN_RECORD_V2,
    MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
    MIXED_CONTEXT_READ_WITNESS_RECORD_V2,
    MIXED_CONTEXT_SCHEMA_V2,
    MIXED_CONTEXT_STATE_RECORD_V2,
    MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    FrameQuestionAnswerTurnV2,
    MixedContextReadWitnessV2,
    MixedConversationContextStateV2,
    ProviderOriginContextError,
    ProviderOriginContextTurnV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)


MIXED_CONTEXT_SNAPSHOT_RECORD_V2 = 2
MIXED_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V2 = MIXED_CONTEXT_STATE_RECORD_V2
MIXED_CONTEXT_SNAPSHOT_BYTES_V2 = 2
MIXED_CONTEXT_SNAPSHOT_CODEC_REVISION_V2 = 2
MIXED_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V2 = 1
MIXED_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V2 = 1
MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2 = 4 * 1024 * 1024
MIXED_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V2 = (
    MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2 // 9)

MIXED_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V2 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/MIXED-CONTEXT-SNAPSHOT-CODEC/V2")

_U64_EXCLUSIVE = 1 << 64
_DIGEST_SIZE = 32


# object-model: exception; interop=DLG-RAW-11B
class MixedContextSnapshotError(ValueError):
    """V2 mixed context snapshot record、transport 或历史重放不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """验证 transport count、length 与版本使用显式无符号 64-bit 范围。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise MixedContextSnapshotError(f"{label} 必须是非负 u64")
    return value


def _record(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 codec 输入只包含有限、严格非负的数学整数。"""
    if (type(value) is not tuple
            or not value
            or len(value) > MIXED_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V2
            or any(type(item) is not int or item < 0 for item in value)):
        raise MixedContextSnapshotError(
            f"{label} 必须是预算内的非空非负严格整数 tuple")
    return value


def _read_scalar(
        record: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[int, int]:
    """读取一项已整体校验过的整数，并显式拒绝截断。"""
    if cursor >= len(record):
        raise MixedContextSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_count(
        record: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[int, int]:
    """读取 record 的 count 字段，避免宿主整数长度语义泄漏。"""
    value, cursor = _read_scalar(record, cursor, label=label)
    return _u64(value, label=label), cursor


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], int]:
    """读取 ``u64 count || ordered integers`` 段，拒绝越界与空值偷换。"""
    count, cursor = _read_count(record, cursor, label=f"{label} count")
    if count > len(record) - cursor:
        raise MixedContextSnapshotError(f"{label} 长度越界")
    value = record[cursor:cursor + count]
    cursor += count
    if not allow_empty and not value:
        raise MixedContextSnapshotError(f"{label} 不得为空")
    return value, cursor


def _read_digest(
        record: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[tuple[int, ...], int]:
    """读取 raw u8[32]，绝不接受 hex 或变长 identity。"""
    value, cursor = _read_segment(
        record,
        cursor,
        label=label,
        allow_empty=False,
    )
    if len(value) != _DIGEST_SIZE or any(item > 255 for item in value):
        raise MixedContextSnapshotError(f"{label} 必须是 raw u8[32]")
    return value, cursor


def _object_identity(value: tuple[int, ...], *, label: str) -> ObjectIdentity:
    """由完整 stable key 恢复 response stance，不保留 host object identity。"""
    try:
        return ObjectIdentity.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            f"{label} 不是完整 ObjectIdentity stable key") from error


def _source_ref(value: tuple[int, ...], *, label: str) -> SourceRef:
    """由完整 stable key 恢复 citation，禁止仅保存局部 source id。"""
    try:
        return SourceRef.from_stable_key(value)
    except (TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            f"{label} 不是完整 SourceRef stable key") from error


def _legacy_read_descriptor(
        record: tuple[int, ...],
        ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
    """显式解析 legacy read stable key，返回其 key、revision、digest 与可见数。"""
    record = _record(record, label="legacy frame context read")
    cursor = 0
    _legacy_version, cursor = _read_scalar(
        record, cursor, label="legacy frame context read version")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="legacy frame context read conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="legacy frame context read revision")
    digest, cursor = _read_digest(
        record,
        cursor,
        label="legacy frame context read digest",
    )
    visible_count, cursor = _read_count(
        record,
        cursor,
        label="legacy frame context read visible count",
    )
    for ordinal in range(visible_count):
        _fingerprint, cursor = _read_segment(
            record,
            cursor,
            label=("legacy frame context read "
                   f"turn fingerprint[{ordinal}]"),
            allow_empty=False,
        )
    if cursor != len(record):
        raise MixedContextSnapshotError("legacy frame context read 含尾随整数")
    return conversation_key, revision, digest, visible_count


def _find_legacy_context(
        contexts: tuple[ConversationContextState, ...],
        conversation_key: tuple[int, ...],
        ) -> ConversationContextState | None:
    """按显式 stable key 查找已由 V2 前缀重放的 legacy compatibility chain。"""
    for context in contexts:
        if context.conversation_key == conversation_key:
            return context
    return None


def _replace_legacy_context(
        contexts: tuple[ConversationContextState, ...],
        replacement: ConversationContextState,
        ) -> tuple[ConversationContextState, ...]:
    """以稳定 key 替换一条 legacy chain；没有旧项时附加新链。"""
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
    """由已恢复 Frame 前缀重新形成 legacy read，并逐整数核验其稳定记录。"""
    key, revision, _digest, visible_count = _legacy_read_descriptor(read_record)
    context = _find_legacy_context(contexts, key)
    if context is None:
        if revision != 0:
            raise MixedContextSnapshotError(
                "legacy frame context read 缺少可重放的 V2 Frame 前缀")
        try:
            context = start_conversation_context(key)
        except (TypeError, ValueError) as error:
            raise MixedContextSnapshotError(
                "legacy frame context read conversation key 非法") from error
    if context.revision != revision:
        raise MixedContextSnapshotError(
            "legacy frame context read revision 与已恢复前缀漂移")
    try:
        expected = context.read(visible_count)
    except (TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "legacy frame context read 无法由已恢复前缀形成") from error
    if expected.stable_key() != read_record:
        raise MixedContextSnapshotError(
            "legacy frame context read digest、fingerprint 或可见尾部漂移")
    return expected, context


def _decode_legacy_frame_turn(
        record: tuple[int, ...],
        contexts: tuple[ConversationContextState, ...],
        ) -> tuple[ConversationTurnState, tuple[ConversationContextState, ...]]:
    """显式重建 Frame 的 legacy typed record 与其已验证 compatibility read。"""
    record = _record(record, label="mixed frame legacy typed record")
    cursor = 0
    _legacy_version, cursor = _read_scalar(
        record, cursor, label="mixed frame legacy turn version")
    ordinal, cursor = _read_scalar(
        record, cursor, label="mixed frame legacy turn ordinal")
    fields: list[tuple[int, ...]] = []
    for label in (
            "request key",
            "target key",
            "query key",
            "planning key",
            "parser revision",
            "readback key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"mixed frame legacy {label}",
            allow_empty=False,
        )
        fields.append(value)
    stance_key, cursor = _read_segment(
        record,
        cursor,
        label="mixed frame legacy response stance",
        allow_empty=False,
    )
    selected_count, cursor = _read_count(
        record,
        cursor,
        label="mixed frame legacy selected candidate count",
    )
    selected: list[tuple[int, ...]] = []
    for item_ordinal in range(selected_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=("mixed frame legacy selected candidate"
                   f"[{item_ordinal}]"),
            allow_empty=False,
        )
        selected.append(value)
    source_count, cursor = _read_count(
        record,
        cursor,
        label="mixed frame legacy cited source count",
    )
    sources: list[SourceRef] = []
    for item_ordinal in range(source_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"mixed frame legacy cited source[{item_ordinal}]",
            allow_empty=False,
        )
        sources.append(_source_ref(
            value,
            label=f"mixed frame legacy cited source[{item_ordinal}]",
        ))
    sentence_count, cursor = _read_count(
        record,
        cursor,
        label="mixed frame legacy discourse sentence count",
    )
    sentences: list[tuple[int, ...]] = []
    for item_ordinal in range(sentence_count):
        value, cursor = _read_segment(
            record,
            cursor,
            label=("mixed frame legacy discourse sentence"
                   f"[{item_ordinal}]"),
            allow_empty=False,
        )
        sentences.append(value)
    has_context_read, cursor = _read_scalar(
        record,
        cursor,
        label="mixed frame legacy has context read",
    )
    if has_context_read not in {0, 1}:
        raise MixedContextSnapshotError(
            "mixed frame legacy context read flag 未注册")
    prior_read: ConversationContextRead | None = None
    prior_context: ConversationContextState | None = None
    if has_context_read:
        read_record, cursor = _read_segment(
            record,
            cursor,
            label="mixed frame legacy context read",
            allow_empty=False,
        )
        prior_read, prior_context = _replay_legacy_read(read_record, contexts)
    if cursor != len(record):
        raise MixedContextSnapshotError("mixed frame legacy typed record 含尾随整数")
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
                label="mixed frame legacy response stance",
            ),
            tuple(selected),
            tuple(sources),
            tuple(sentences),
            parser_revision,
            readback_key,
            prior_read,
        )
    except (TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "mixed frame legacy typed record 无法恢复") from error
    if turn.stable_key() != record:
        raise MixedContextSnapshotError(
            "mixed frame legacy typed record 非规范或字段漂移")
    if prior_context is None:
        return turn, contexts
    try:
        updated_context = ConversationContextState(
            prior_context.conversation_key,
            prior_context.revision + 1,
            prior_context.digest(),
            (*prior_context.turns, turn),
        )
    except (TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "mixed frame legacy compatibility chain 无法追加") from error
    return turn, _replace_legacy_context(contexts, updated_context)


def _decode_role_bindings(
        record: tuple[int, ...],
        ) -> tuple[ProviderOriginRoleBindingV1, ...]:
    """由 anchor 内的保序 record 恢复 RoleBinding，不按宿主 map 重排。"""
    record = _record(record, label="provider anchor role binding record")
    cursor = 0
    count, cursor = _read_count(
        record, cursor, label="provider anchor role binding count")
    result: list[ProviderOriginRoleBindingV1] = []
    for ordinal in range(count):
        binding_key, cursor = _read_segment(
            record,
            cursor,
            label=f"provider anchor binding[{ordinal}] key",
            allow_empty=False,
        )
        role_key, cursor = _read_segment(
            record,
            cursor,
            label=f"provider anchor binding[{ordinal}] role key",
            allow_empty=False,
        )
        filler_key, cursor = _read_segment(
            record,
            cursor,
            label=f"provider anchor binding[{ordinal}] filler key",
            allow_empty=False,
        )
        item_ordinal, cursor = _read_scalar(
            record,
            cursor,
            label=f"provider anchor binding[{ordinal}] ordinal",
        )
        try:
            result.append(ProviderOriginRoleBindingV1(
                binding_key,
                role_key,
                filler_key,
                item_ordinal,
            ))
        except (TypeError, ValueError) as error:
            raise MixedContextSnapshotError(
                f"provider anchor binding[{ordinal}] 非法") from error
    if cursor != len(record):
        raise MixedContextSnapshotError("provider anchor role binding record 含尾随整数")
    return tuple(result)


def _decode_occurrences(
        record: tuple[int, ...],
        ) -> tuple[ProviderOriginOccurrenceV1, ...]:
    """由 anchor 内的保序 record 恢复 occurrence，保留 source-local ordinal。"""
    record = _record(record, label="provider anchor occurrence record")
    cursor = 0
    count, cursor = _read_count(
        record, cursor, label="provider anchor occurrence count")
    result: list[ProviderOriginOccurrenceV1] = []
    for ordinal in range(count):
        occurrence_key, cursor = _read_segment(
            record,
            cursor,
            label=f"provider anchor occurrence[{ordinal}] key",
            allow_empty=False,
        )
        semantic_object_key, cursor = _read_segment(
            record,
            cursor,
            label=("provider anchor occurrence"
                   f"[{ordinal}] semantic object key"),
            allow_empty=False,
        )
        item_ordinal, cursor = _read_scalar(
            record,
            cursor,
            label=f"provider anchor occurrence[{ordinal}] ordinal",
        )
        start, cursor = _read_scalar(
            record,
            cursor,
            label=f"provider anchor occurrence[{ordinal}] start",
        )
        end, cursor = _read_scalar(
            record,
            cursor,
            label=f"provider anchor occurrence[{ordinal}] end",
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
            raise MixedContextSnapshotError(
                f"provider anchor occurrence[{ordinal}] 非法") from error
    if cursor != len(record):
        raise MixedContextSnapshotError("provider anchor occurrence record 含尾随整数")
    return tuple(result)


def _decode_anchor(
        record: tuple[int, ...],
        ) -> ProviderOriginAnchorProjectionV1:
    """从 complete canonical record 重建 anchor，不接触 provider 或输出文本。"""
    record = _record(record, label="provider origin anchor")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="provider origin anchor version")
    if version != PROVIDER_ORIGIN_ANCHOR_RECORD_V1:
        raise MixedContextSnapshotError("provider origin anchor version 未注册")
    status, cursor = _read_scalar(
        record, cursor, label="provider origin anchor status")
    provider_kind, cursor = _read_scalar(
        record, cursor, label="provider origin anchor provider kind")
    first_segments: list[tuple[int, ...]] = []
    for label in (
            "provider identity",
            "runtime identity",
            "catalog record identity",
            "provider result identity",
            "input intake identity",
            "output readback identity",
            "source record key",
            "source ref stable key",
            "source commitment",
            "w03 observation key",
            "w04 observation key",
            "w05 observation key",
            "proposition key",
            "predicate key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"provider origin anchor {label}",
        )
        first_segments.append(value)
    relation_kind, cursor = _read_scalar(
        record,
        cursor,
        label="provider origin anchor relation kind",
    )
    focus_segments: list[tuple[int, ...]] = []
    for label in (
            "generation construction key",
            "focus role binding key",
            "focus role key",
            "focus filler key",
            "focus occurrence key"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"provider origin anchor {label}",
        )
        focus_segments.append(value)
    focus_start, cursor = _read_scalar(
        record, cursor, label="provider origin anchor focus start")
    focus_end, cursor = _read_scalar(
        record, cursor, label="provider origin anchor focus end")
    binding_record, cursor = _read_segment(
        record,
        cursor,
        label="provider origin anchor ordered role bindings",
        allow_empty=False,
    )
    occurrence_record, cursor = _read_segment(
        record,
        cursor,
        label="provider origin anchor ordered occurrences",
        allow_empty=False,
    )
    output_scalars, cursor = _read_segment(
        record,
        cursor,
        label="provider origin anchor output scalars",
    )
    output_u8, cursor = _read_segment(
        record,
        cursor,
        label="provider origin anchor output u8",
    )
    anchor_identity, cursor = _read_digest(
        record,
        cursor,
        label="provider origin anchor identity",
    )
    if cursor != len(record):
        raise MixedContextSnapshotError("provider origin anchor 含尾随整数")
    try:
        anchor = ProviderOriginAnchorProjectionV1(
            status,
            provider_kind,
            first_segments[0],
            first_segments[1],
            first_segments[2],
            first_segments[3],
            first_segments[4],
            first_segments[5],
            first_segments[6],
            first_segments[7],
            first_segments[8],
            first_segments[9],
            first_segments[10],
            first_segments[11],
            first_segments[12],
            first_segments[13],
            relation_kind,
            focus_segments[0],
            focus_segments[1],
            focus_segments[2],
            focus_segments[3],
            focus_segments[4],
            focus_start,
            focus_end,
            _decode_role_bindings(binding_record),
            _decode_occurrences(occurrence_record),
            output_scalars,
            output_u8,
            anchor_identity,
        )
    except (ProviderOriginAnchorError, TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "provider origin anchor 无法按 canonical record 恢复") from error
    if anchor.canonical_record() != record:
        raise MixedContextSnapshotError("provider origin anchor canonical record 漂移")
    return anchor


def _decode_read_witness(
        record: tuple[int, ...],
        ) -> MixedContextReadWitnessV2:
    """解析完整 V2 read witness，self identity 由构造器重新计算。"""
    record = _record(record, label="mixed context read witness")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="mixed context read witness version")
    if version != MIXED_CONTEXT_READ_WITNESS_RECORD_V2:
        raise MixedContextSnapshotError("mixed context read witness version 未注册")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="mixed context read witness conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="mixed context read witness revision")
    snapshot_digest, cursor = _read_digest(
        record,
        cursor,
        label="mixed context read witness snapshot digest",
    )
    requested_limit, cursor = _read_scalar(
        record,
        cursor,
        label="mixed context read witness requested limit",
    )
    visible_start, cursor = _read_scalar(
        record,
        cursor,
        label="mixed context read witness visible start ordinal",
    )
    identity_count, cursor = _read_count(
        record,
        cursor,
        label="mixed context read witness visible identity count",
    )
    identities: list[tuple[int, ...]] = []
    for ordinal in range(identity_count):
        value, cursor = _read_digest(
            record,
            cursor,
            label=("mixed context read witness visible turn identity"
                   f"[{ordinal}]"),
        )
        identities.append(value)
    witness_identity, cursor = _read_digest(
        record,
        cursor,
        label="mixed context read witness identity",
    )
    if cursor != len(record):
        raise MixedContextSnapshotError("mixed context read witness 含尾随整数")
    try:
        witness = MixedContextReadWitnessV2(
            conversation_key,
            revision,
            snapshot_digest,
            requested_limit,
            visible_start,
            tuple(identities),
            witness_identity,
        )
    except (ProviderOriginContextError, TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "mixed context read witness 无法恢复") from error
    if witness.canonical_record() != record:
        raise MixedContextSnapshotError("mixed context read witness canonical record 漂移")
    return witness


def _decode_turn(
        record: tuple[int, ...],
        legacy_contexts: tuple[ConversationContextState, ...],
        ) -> tuple[
            FrameQuestionAnswerTurnV2 | ProviderOriginContextTurnV1,
            tuple[ConversationContextState, ...],
        ]:
    """按显式 kind 解码一条 tagged turn，并保留 Frame compatibility chain。"""
    record = _record(record, label="mixed context turn")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="mixed context turn record version")
    turn_kind, cursor = _read_scalar(
        record, cursor, label="mixed context turn kind")
    ordinal, cursor = _read_scalar(
        record, cursor, label="mixed context turn append ordinal")
    previous_digest, cursor = _read_digest(
        record,
        cursor,
        label="mixed context turn previous snapshot digest",
    )
    witness_record, cursor = _read_segment(
        record,
        cursor,
        label="mixed context turn prior read witness",
        allow_empty=False,
    )
    witness = _decode_read_witness(witness_record)
    if turn_kind == MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN:
        if version != MIXED_CONTEXT_FRAME_TURN_RECORD_V2:
            raise MixedContextSnapshotError("mixed Frame turn version 未注册")
        frame_record, cursor = _read_segment(
            record,
            cursor,
            label="mixed Frame legacy typed payload",
            allow_empty=False,
        )
        write_origin, cursor = _read_scalar(
            record,
            cursor,
            label="mixed Frame write origin",
        )
        turn_identity, cursor = _read_digest(
            record,
            cursor,
            label="mixed Frame turn identity",
        )
        if cursor != len(record):
            raise MixedContextSnapshotError("mixed Frame turn 含尾随整数")
        frame_turn, next_contexts = _decode_legacy_frame_turn(
            frame_record,
            legacy_contexts,
        )
        try:
            turn = FrameQuestionAnswerTurnV2(
                ordinal,
                previous_digest,
                witness,
                frame_turn,
                write_origin,
                turn_identity,
            )
        except (ProviderOriginContextError, TypeError, ValueError) as error:
            raise MixedContextSnapshotError("mixed Frame turn 无法恢复") from error
    elif turn_kind == MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION:
        if version != MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1:
            raise MixedContextSnapshotError("mixed provider turn version 未注册")
        anchor_record, cursor = _read_segment(
            record,
            cursor,
            label="mixed provider anchor payload",
            allow_empty=False,
        )
        provider_result_identity, cursor = _read_digest(
            record,
            cursor,
            label="mixed provider result identity",
        )
        write_origin, cursor = _read_scalar(
            record,
            cursor,
            label="mixed provider write origin",
        )
        consumed_count, cursor = _read_count(
            record,
            cursor,
            label="mixed provider consumed reference count",
        )
        if consumed_count != 0:
            raise MixedContextSnapshotError(
                "mixed provider 11B consumed reference 必须为空")
        turn_identity, cursor = _read_digest(
            record,
            cursor,
            label="mixed provider turn identity",
        )
        if cursor != len(record):
            raise MixedContextSnapshotError("mixed provider turn 含尾随整数")
        try:
            turn = ProviderOriginContextTurnV1(
                ordinal,
                previous_digest,
                witness,
                _decode_anchor(anchor_record),
                provider_result_identity,
                (),
                write_origin,
                turn_identity,
            )
        except (ProviderOriginContextError, TypeError, ValueError) as error:
            raise MixedContextSnapshotError("mixed provider turn 无法恢复") from error
        next_contexts = legacy_contexts
    else:
        raise MixedContextSnapshotError("mixed context turn kind 未注册")
    if turn.canonical_record() != record:
        raise MixedContextSnapshotError("mixed context turn canonical record 漂移")
    return turn, next_contexts


def _restore_context(
        record: tuple[int, ...],
        ) -> MixedConversationContextStateV2:
    """重放 complete V2 context，逐条交给 typed 构造器校验前驱和 read witness。"""
    record = _record(record, label="mixed context snapshot context")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="mixed context snapshot context version")
    if version != MIXED_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V2:
        raise MixedContextSnapshotError("mixed context snapshot context version 未注册")
    schema, cursor = _read_scalar(
        record, cursor, label="mixed context snapshot schema")
    if schema != MIXED_CONTEXT_SCHEMA_V2:
        raise MixedContextSnapshotError("mixed context snapshot schema 未注册")
    conversation_key, cursor = _read_segment(
        record,
        cursor,
        label="mixed context snapshot conversation key",
        allow_empty=False,
    )
    revision, cursor = _read_scalar(
        record, cursor, label="mixed context snapshot revision")
    previous_digest, cursor = _read_segment(
        record,
        cursor,
        label="mixed context snapshot previous digest",
    )
    if previous_digest and (len(previous_digest) != _DIGEST_SIZE
                            or any(item > 255 for item in previous_digest)):
        raise MixedContextSnapshotError(
            "mixed context snapshot previous digest 非法")
    turn_count, cursor = _read_count(
        record,
        cursor,
        label="mixed context snapshot turn count",
    )
    turns: list[FrameQuestionAnswerTurnV2 | ProviderOriginContextTurnV1] = []
    legacy_contexts: tuple[ConversationContextState, ...] = ()
    for ordinal in range(turn_count):
        turn_record, cursor = _read_segment(
            record,
            cursor,
            label=f"mixed context snapshot turn[{ordinal}]",
            allow_empty=False,
        )
        turn, legacy_contexts = _decode_turn(turn_record, legacy_contexts)
        turns.append(turn)
    if cursor != len(record):
        raise MixedContextSnapshotError("mixed context snapshot context 含尾随整数")
    try:
        state = MixedConversationContextStateV2(
            conversation_key,
            revision,
            previous_digest,
            tuple(turns),
        )
    except (ProviderOriginContextError, TypeError, ValueError) as error:
        raise MixedContextSnapshotError(
            "mixed context snapshot state 无法恢复") from error
    if state.canonical_record() != record:
        raise MixedContextSnapshotError("mixed context snapshot context canonical record 漂移")
    return state


def snapshot_mixed_conversation_context_v2(
        state: MixedConversationContextStateV2,
        ) -> tuple[int, ...]:
    """导出 V2 snapshot top-level record；不含 runtime binding 或物理路径。"""
    if type(state) is not MixedConversationContextStateV2:
        raise TypeError("mixed context snapshot 需要 MixedConversationContextStateV2")
    context_record = _record(
        state.canonical_record(),
        label="mixed context snapshot state record",
    )
    _u64(len(context_record), label="mixed context snapshot context count")
    result = (
        MIXED_CONTEXT_SNAPSHOT_RECORD_V2,
        len(context_record),
        *context_record,
    )
    # encoder 也必须证明生成物可恢复，避免输出未来必定失效的 bytes。
    restored = _restore_context(context_record)
    if restored.canonical_record() != context_record:
        raise MixedContextSnapshotError("mixed context snapshot state readback 漂移")
    return result


def restore_mixed_conversation_context_v2(
        record: tuple[int, ...],
        ) -> MixedConversationContextStateV2:
    """严格恢复 top-level V2 snapshot record，拒绝旧版、截断与尾随整数。"""
    record = _record(record, label="mixed context snapshot")
    cursor = 0
    version, cursor = _read_scalar(
        record, cursor, label="mixed context snapshot version")
    if version != MIXED_CONTEXT_SNAPSHOT_RECORD_V2:
        raise MixedContextSnapshotError("mixed context snapshot version 未注册")
    context_record, cursor = _read_segment(
        record,
        cursor,
        label="mixed context snapshot context",
        allow_empty=False,
    )
    if cursor != len(record):
        raise MixedContextSnapshotError("mixed context snapshot 含尾随整数")
    return _restore_context(context_record)


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """把非负数学整数编码为最短 unsigned-big-endian byte sequence。"""
    if type(value) is not int or value < 0:
        raise MixedContextSnapshotError(f"{label} 必须是非负严格整数")
    size = max(1, (value.bit_length() + 7) // 8)
    _u64(size, label=f"{label} byte length")
    if size > MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2:
        raise MixedContextSnapshotError(f"{label} 超出 snapshot bytes 预算")
    return value.to_bytes(size, "big")


def encode_mixed_conversation_context_snapshot_v2_bytes(
        state: MixedConversationContextStateV2,
        ) -> bytes:
    """用固定 u64 count/length 与无符号大端整数导出 V2 snapshot bytes。"""
    record = snapshot_mixed_conversation_context_v2(state)
    _u64(len(record), label="mixed context snapshot bytes integer count")
    result = bytearray()
    result.extend(_u64(
        MIXED_CONTEXT_SNAPSHOT_BYTES_V2,
        label="mixed context snapshot bytes version",
    ).to_bytes(8, "big"))
    result.extend(len(record).to_bytes(8, "big"))
    for ordinal, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value,
            label=f"mixed context snapshot integer[{ordinal}]",
        )
        result.extend(_u64(
            len(encoded),
            label=f"mixed context snapshot integer[{ordinal}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2:
            raise MixedContextSnapshotError("mixed context snapshot bytes 超出固定预算")
    return bytes(result)


def _read_u64_bytes(
        payload: bytes, cursor: int, *, label: str,
        ) -> tuple[int, int]:
    """从 transport 逐字节读取 u64，截断不借用 Python slicing 语义。"""
    if cursor > len(payload) - 8:
        raise MixedContextSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def decode_mixed_conversation_context_snapshot_v2_bytes(
        payload: bytes,
        ) -> MixedConversationContextStateV2:
    """严格 decode V2 bytes 后恢复 state；任何 transport 漂移都不返回部分状态。"""
    if type(payload) is not bytes or not payload:
        raise MixedContextSnapshotError(
            "mixed context snapshot bytes 必须是非空 raw bytes")
    if len(payload) > MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2:
        raise MixedContextSnapshotError("mixed context snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload, cursor, label="mixed context snapshot bytes version")
    if version != MIXED_CONTEXT_SNAPSHOT_BYTES_V2:
        raise MixedContextSnapshotError("mixed context snapshot bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload,
        cursor,
        label="mixed context snapshot bytes integer count",
    )
    if (count > MIXED_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V2
            or count > (len(payload) - cursor) // 9):
        raise MixedContextSnapshotError(
            "mixed context snapshot bytes integer count 越界")
    values: list[int] = []
    for ordinal in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"mixed context snapshot integer[{ordinal}] length",
        )
        if (size < 1
                or size > MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2
                or size > len(payload) - cursor):
            raise MixedContextSnapshotError(
                f"mixed context snapshot integer[{ordinal}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise MixedContextSnapshotError(
                f"mixed context snapshot integer[{ordinal}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise MixedContextSnapshotError("mixed context snapshot bytes 含尾随 bytes")
    return restore_mixed_conversation_context_v2(tuple(values))


def mixed_context_snapshot_codec_revision_v2() -> tuple[int, ...]:
    """返回 V3 runtime binding 可逐整数锁定的 codec revision record。"""
    return (
        MIXED_CONTEXT_SNAPSHOT_CODEC_REVISION_V2,
        MIXED_CONTEXT_SNAPSHOT_RECORD_V2,
        MIXED_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V2,
        MIXED_CONTEXT_SCHEMA_V2,
        MIXED_CONTEXT_FRAME_TURN_RECORD_V2,
        MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
        MIXED_CONTEXT_READ_WITNESS_RECORD_V2,
        MIXED_CONTEXT_SNAPSHOT_BYTES_V2,
        MIXED_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V2,
        MIXED_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V2,
        MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2,
        MIXED_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V2,
    )


def mixed_context_snapshot_codec_identity_v2() -> tuple[int, ...]:
    """返回 codec revision record 的 raw u8[32] identity，供 runtime binding 使用。"""
    try:
        return tuple(portable_sha256_v1(
            MIXED_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V2,
            (mixed_context_snapshot_codec_revision_v2(),),
        ))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise MixedContextSnapshotError("mixed context snapshot codec identity 无法形成") from error


__all__ = [
    "MIXED_CONTEXT_SNAPSHOT_BYTES_V2",
    "MIXED_CONTEXT_SNAPSHOT_CODEC_IDENTITY_DOMAIN_V2",
    "MIXED_CONTEXT_SNAPSHOT_CODEC_REVISION_V2",
    "MIXED_CONTEXT_SNAPSHOT_CONTEXT_RECORD_V2",
    "MIXED_CONTEXT_SNAPSHOT_INTEGER_BYTES_ENCODING_V2",
    "MIXED_CONTEXT_SNAPSHOT_LEGACY_FRAME_CHAIN_RULE_V2",
    "MIXED_CONTEXT_SNAPSHOT_MAX_BYTES_V2",
    "MIXED_CONTEXT_SNAPSHOT_MAX_INTEGER_COUNT_V2",
    "MIXED_CONTEXT_SNAPSHOT_RECORD_V2",
    "MixedContextSnapshotError",
    "decode_mixed_conversation_context_snapshot_v2_bytes",
    "encode_mixed_conversation_context_snapshot_v2_bytes",
    "mixed_context_snapshot_codec_identity_v2",
    "mixed_context_snapshot_codec_revision_v2",
    "restore_mixed_conversation_context_v2",
    "snapshot_mixed_conversation_context_v2",
]
