"""DLG-RAW-12：V4 mixed session 与 V3 focus ledger 的独立 outer snapshot。

本模块不升级或修改旧 cursor V5。它把已经冻结的 V4 session snapshot 与 V3
append-only ledger snapshot 并列嵌入新的 outer record；恢复顺序固定为 V4、V3、
outer projection、runtime course replay。V3 admission 中的外来 catalog/form/
candidate record 不以末尾 SHA 作为充分证据，必须在当前公开 runtime 中逐 event
重新生成并逐 record 比较。

Python 代码只提供当前 reference implementation 的结构体和 bytes adapter。逻辑
snapshot、course binding、整数 framing、readback 与 identity 均可由任意支持整数
和有限 byte sequence 的语言重现。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MixedConversationContextStateV2,
    ProviderOriginContextTurnV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_chain import (
    ProviderOriginDiscourseFocusV1,
    ProviderOriginFocusChainError,
    run_provider_origin_focus_followup_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    FrameQuestionAnswerTurnV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFollowupFocusTurnV1,
    ProviderOriginFocusContextError,
    provider_origin_focus_admission_from_followup_result_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context_snapshot import (
    MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3,
    MixedFocusContextSnapshotError,
    mixed_focus_context_snapshot_codec_identity_v3,
    mixed_focus_context_snapshot_codec_revision_v3,
    restore_mixed_conversation_focus_context_v3,
    snapshot_mixed_conversation_focus_context_v3,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupCatalogV1,
    ProviderOriginFollowupError,
    provider_origin_followup_schema_identity_v1,
    provider_origin_followup_schema_record_v1,
    run_provider_origin_followup_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_RECORD_V1,
    UTF8_STRICT_V1,
    ConversationRawIntake,
    ConversationRawIntakeError,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_snapshot import (
    RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4,
    ConversationRawMixedDialogueSnapshotError,
    mixed_dialogue_runtime_binding_v4,
    mixed_dialogue_runtime_identity_v4,
    restore_public_mixed_frame_dialogue_state_v4,
    snapshot_public_mixed_frame_dialogue_state_v4,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1,
    RAW_MIXED_FOCUS_DIALOGUE_TURN_RECORD_V1,
    ConversationRawMixedFocusDialogueSessionError,
    ConversationRawMixedFocusDialogueStateV1,
)


RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1 = 8
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 = (
    RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4
    + MIXED_FOCUS_CONTEXT_SNAPSHOT_MAX_BYTES_V3
    + 1024 * 1024)
RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1 = (
    RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1 // 9)

RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-12/MIXED-FOCUS-DIALOGUE-RUNTIME/V1")

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-12
class ConversationRawMixedFocusDialogueSnapshotError(ValueError):
    """outer record、embedded snapshot、course replay 或 bytes transport 不闭合。"""


def _u64(value: int, *, label: str) -> int:
    """核验 count、length 和 transport version 使用显式无符号 64-bit 整数。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 必须是非负 u64")
    return value


def _record(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证外层 record 是预算内的有序非负严格整数。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or len(value) > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 必须是预算内的{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _pack(
        result: list[int],
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool = True,
        ) -> None:
    """以 ``u64 count || payload`` 写入一段 logical integer record。"""
    record = _record(value, label=label, allow_empty=allow_empty)
    result.extend((_u64(len(record), label=f"{label} count"), *record))


def _read_scalar(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """读取一个已整体验证 record 的标量，拒绝截断。"""
    if cursor >= len(record):
        raise ConversationRawMixedFocusDialogueSnapshotError(f"{label} 截断")
    return record[cursor], cursor + 1


def _read_segment(
        record: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[tuple[int, ...], int]:
    """读取 length-framed segment，不借用宿主序列化或隐式截断。"""
    count, cursor = _read_scalar(record, cursor, label=f"{label} count")
    _u64(count, label=f"{label} count")
    if count > len(record) - cursor:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 长度越界")
    value = _record(
        record[cursor:cursor + count], label=label, allow_empty=allow_empty)
    return value, cursor + count


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """以 frozen portable SHA raw-u8 framing 形成跨语言 runtime identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 无法形成") from error


def _restore_intake_record(
        record: tuple[int, ...],
        *,
        label: str,
        ) -> ConversationRawIntake:
    """从完整 RAW-00 record 恢复 intake，供 course replay 回读外来输入证据。"""
    record = _record(record, label=label, allow_empty=False)
    cursor = 0
    version, cursor = _read_scalar(record, cursor, label=f"{label} version")
    if version != DLG_RAW_RECORD_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} version 未注册")
    result_code, cursor = _read_scalar(record, cursor, label=f"{label} result code")
    utf8_rule, cursor = _read_scalar(record, cursor, label=f"{label} UTF-8 rule")
    if utf8_rule != UTF8_STRICT_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} UTF-8 rule 未注册")
    values: list[tuple[int, ...]] = []
    for field in (
            "raw input", "canonical body", "unicode scalars", "typed record",
            "output bytes", "state delta"):
        value, cursor = _read_segment(
            record,
            cursor,
            label=f"{label} {field}",
            allow_empty=True,
        )
        values.append(value)
    if cursor != len(record):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 含尾随整数")
    try:
        intake = ConversationRawIntake(
            result_code,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
        )
    except (ConversationRawIntakeError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 无法恢复") from error
    if intake.canonical_record() != record:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} canonical readback 漂移")
    return intake


def mixed_focus_dialogue_snapshot_transport_record_v1() -> tuple[int, ...]:
    """冻结 outer bytes transport 的版本、预算、宽度、字节序和最短整数规则。"""
    return (
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1,
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V1,
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1,
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1,
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1,
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1,
    )


def mixed_focus_dialogue_runtime_binding_v1(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 outer snapshot 必须精确匹配的 V4/V3/course/transport binding。"""
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed focus dialogue snapshot runtime 类型错误")
    catalog = runtime.provider_origin_followup_catalog
    if type(catalog) is not ProviderOriginFollowupCatalogV1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue snapshot 缺公开 follow-up catalog")
    result = [RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V1]
    for label, value in (
            ("mixed dialogue V4 runtime binding", mixed_dialogue_runtime_binding_v4(runtime)),
            ("mixed dialogue V4 runtime identity", mixed_dialogue_runtime_identity_v4(runtime)),
            ("follow-up catalog canonical record", catalog.canonical_record()),
            ("follow-up catalog identity", catalog.catalog_identity_u8),
            ("follow-up schema", provider_origin_followup_schema_record_v1()),
            ("follow-up schema identity", provider_origin_followup_schema_identity_v1()),
            ("focus ledger snapshot codec revision", mixed_focus_context_snapshot_codec_revision_v3()),
            ("focus ledger snapshot codec identity", mixed_focus_context_snapshot_codec_identity_v3()),
            ("outer state record", (RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1,)),
            ("outer turn record", (RAW_MIXED_FOCUS_DIALOGUE_TURN_RECORD_V1,)),
            ("outer snapshot transport", mixed_focus_dialogue_snapshot_transport_record_v1()),
    ):
        _pack(result, value, label=label, allow_empty=False)
    return tuple(result)


def mixed_focus_dialogue_runtime_identity_v1(
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """返回完整 course/runtime binding 的 raw u8 identity。"""
    return _identity(
        RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V1,
        mixed_focus_dialogue_runtime_binding_v1(runtime),
        label="mixed focus dialogue runtime identity",
    )


def _immediate_parent(
        state: ConversationRawMixedFocusDialogueStateV1,
        event: ProviderOriginFollowupFocusTurnV1,
        ) -> ProviderOriginContextTurnV3 | ProviderOriginFollowupFocusTurnV1:
    """从 append ordinal 找到唯一立即父轮，拒绝向早期历史扫描替代它。"""
    ordinal = event.append_ordinal
    turns = state.focus_context.turns
    if ordinal < 1 or ordinal > len(turns) - 1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event parent ordinal 非法")
    parent = turns[ordinal - 1]
    if (type(parent) is not ProviderOriginContextTurnV3
            and type(parent) is not ProviderOriginFollowupFocusTurnV1):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event parent kind 非法")
    if parent.turn_identity_u8 != event.parent_turn_identity_u8:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event parent identity 漂移")
    anchor = (parent.anchor_projection
              if type(parent) is ProviderOriginContextTurnV3
              else parent.parent_anchor_projection)
    if (anchor.canonical_record()
            != event.parent_anchor_projection.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event parent anchor 漂移")
    return parent


def _provider_ancestor(
        state: ConversationRawMixedFocusDialogueStateV1,
        parent: ProviderOriginContextTurnV3 | ProviderOriginFollowupFocusTurnV1,
        ) -> ProviderOriginContextTurnV3:
    """沿 immediate-parent edge 回到本条 focus 链唯一的 provider anchor。"""
    current = parent
    while type(current) is ProviderOriginFollowupFocusTurnV1:
        current = _immediate_parent(state, current)
    if type(current) is not ProviderOriginContextTurnV3:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue provider ancestor 缺失")
    return current


def _v2_prefix_for_provider(
        state: ConversationRawMixedFocusDialogueStateV1,
        provider: ProviderOriginContextTurnV3,
        ) -> MixedConversationContextStateV2:
    """取得 provider 入场时的 V2 prefix，供历史 focus event 重演 reducer。"""
    inner_turns = state.mixed_state.context.turns
    inner_index = 0
    for outer in state.focus_context.turns:
        if (type(outer) is not FrameQuestionAnswerTurnV3
                and type(outer) is not ProviderOriginContextTurnV3):
            continue
        if inner_index >= len(inner_turns):
            raise ConversationRawMixedFocusDialogueSnapshotError(
                "mixed focus dialogue V4/V3 projection 长度漂移")
        inner = inner_turns[inner_index]
        if (type(outer) is ProviderOriginContextTurnV3
                and outer.turn_identity_u8 == provider.turn_identity_u8):
            if (type(inner) is not ProviderOriginContextTurnV1
                    or (inner.anchor_projection.canonical_record()
                        != provider.anchor_projection.canonical_record())):
                raise ConversationRawMixedFocusDialogueSnapshotError(
                    "mixed focus dialogue provider V4/V3 projection 漂移")
            prefix = inner_turns[:inner_index + 1]
            previous = (() if not prefix
                        else prefix[-1].previous_snapshot_digest_u8)
            try:
                return MixedConversationContextStateV2(
                    state.conversation_key,
                    len(prefix),
                    previous,
                    prefix,
                )
            except (TypeError, ValueError) as error:
                raise ConversationRawMixedFocusDialogueSnapshotError(
                    "mixed focus dialogue V2 provider prefix 无法恢复") from error
        inner_index += 1
    raise ConversationRawMixedFocusDialogueSnapshotError(
        "mixed focus dialogue provider 未进入 V4/V3 projection")


def _legacy_focus_from_event(
        context: MixedConversationContextStateV2,
        event: ProviderOriginFollowupFocusTurnV1,
        ) -> ProviderOriginDiscourseFocusV1:
    """临时恢复旧 reducer 所需 focus；它不进入 outer state 或 snapshot。"""
    read = context.read(1)
    if len(read.turns) != 1 or type(read.turns[0]) is not ProviderOriginContextTurnV1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue focus event 缺 provider V2 tail")
    provider_turn = read.turns[0]
    anchor = provider_turn.anchor_projection
    if (anchor.canonical_record()
            != event.parent_anchor_projection.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue focus event/provider prefix anchor 漂移")
    try:
        return ProviderOriginDiscourseFocusV1(
            context.revision,
            context.digest(),
            provider_turn.turn_identity_u8,
            anchor.provider_kind,
            anchor.provider_identity_u8,
            anchor.runtime_identity_u8,
            anchor.catalog_record_identity_u8,
            anchor.provider_result_identity_u8,
            anchor.anchor_identity_u8,
            anchor.source_record_key,
            anchor.source_ref_stable_key,
            anchor.source_commitment_u8,
            anchor.w03_observation_key,
            anchor.w04_observation_key,
            anchor.w05_observation_key,
            anchor.generation_construction_key,
            anchor.proposition_key,
            anchor.predicate_key,
            anchor.relation_kind_code,
            event.current_role_binding_key,
            event.current_role_key,
            event.current_filler_key,
            event.current_occurrence_key,
            event.current_start,
            event.current_end,
            event.admission.candidate_identity_u8,
            event.admission.form_identity_u8,
        )
    except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue 临时 focus 无法恢复") from error


def _form_for_admission(
        catalog: ProviderOriginFollowupCatalogV1,
        event: ProviderOriginFollowupFocusTurnV1,
        ):
    """按完整 identity 和 canonical record 锁定唯一公开 form。"""
    admission = event.admission
    if (admission.catalog_identity_u8 != catalog.catalog_identity_u8
            or admission.catalog_record != catalog.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event catalog external record 漂移")
    forms = tuple(
        form for form in catalog.forms
        if form.form_identity_u8 == admission.form_identity_u8)
    if len(forms) != 1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event form identity 未唯一绑定 runtime catalog")
    form = forms[0]
    if admission.form_record != form.canonical_record():
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event form external record 漂移")
    return form


def _input_for_admission(
        event: ProviderOriginFollowupFocusTurnV1,
        form,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> ConversationRawIntake:
    """回读 event input，证明它就是当前 form 的 exact ingress。"""
    admission = event.admission
    intake = _restore_intake_record(
        admission.input_intake_record,
        label="mixed focus dialogue event input intake",
    )
    try:
        replayed = intake_raw_conversation_vector(intake.raw_input_bytes)
        expected_body = encode_utf8_v1(form.input_scalars)
    except (ConversationRawIntakeError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event input readback 无法形成") from error
    matches = catalog.matching_forms(intake.unicode_scalars) if intake.accepted else ()
    if (not intake.accepted
            or intake.canonical_record() != admission.input_intake_record
            or replayed.canonical_record() != admission.input_intake_record
            or intake.unicode_scalars != form.input_scalars
            or intake.canonical_body_bytes != expected_body
            or len(matches) != 1
            or matches[0].canonical_record() != form.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event input 不是 form exact ingress")
    return intake


def _validate_event_course_replay(
        state: ConversationRawMixedFocusDialogueStateV1,
        catalog: ProviderOriginFollowupCatalogV1,
        event: ProviderOriginFollowupFocusTurnV1,
        ) -> None:
    """逐 event 用 runtime catalog/profile/anchor 重新构造 candidate 与 readback。"""
    admission = event.admission
    form = _form_for_admission(catalog, event)
    intake = _input_for_admission(event, form, catalog)
    parent = _immediate_parent(state, event)
    provider = _provider_ancestor(state, parent)
    prefix = _v2_prefix_for_provider(state, provider)
    anchor = event.parent_anchor_projection
    if (anchor.canonical_record()
            != provider.anchor_projection.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event provider ancestor anchor 漂移")
    try:
        if type(parent) is ProviderOriginContextTurnV3:
            result = run_provider_origin_followup_v1(
                intake,
                prefix.read(1),
                catalog,
            )
        else:
            result = run_provider_origin_focus_followup_v1(
                intake,
                prefix,
                catalog,
                _legacy_focus_from_event(prefix, parent),
            )
    except (ProviderOriginFocusChainError, ProviderOriginFollowupError,
            TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event runtime course replay 失败") from error
    if (result is None or not result.accepted or result.form is None
            or result.candidate is None
            or result.intake.canonical_record() != admission.input_intake_record
            or result.form.canonical_record() != admission.form_record):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event runtime course replay 未得唯一 answer")
    try:
        rebuilt_admission = provider_origin_focus_admission_from_followup_result_v1(
            result,
            catalog,
            anchor,
        )
    except (ProviderOriginFocusContextError, ProviderOriginFollowupError,
            TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event admission 无法由 runtime course 重建") from error
    if (rebuilt_admission.canonical_record() != admission.canonical_record()):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue event runtime admission record 漂移")


def validate_public_mixed_focus_dialogue_runtime_v1(
        state: ConversationRawMixedFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """验证每条 V3 focus event 均可由当前公开 course 精确重放。"""
    if type(state) is not ConversationRawMixedFocusDialogueStateV1:
        raise TypeError("mixed focus dialogue course state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed focus dialogue course runtime 类型错误")
    catalog = runtime.provider_origin_followup_catalog
    if type(catalog) is not ProviderOriginFollowupCatalogV1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue course 缺 runtime catalog")
    for turn in state.focus_context.turns:
        if type(turn) is ProviderOriginFollowupFocusTurnV1:
            _validate_event_course_replay(state, catalog, turn)


def snapshot_public_mixed_focus_dialogue_state_v1(
        state: ConversationRawMixedFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> tuple[int, ...]:
    """导出 outer logical snapshot，嵌入独立 V4 和 V3 snapshot 而非对象引用。"""
    if type(state) is not ConversationRawMixedFocusDialogueStateV1:
        raise TypeError("mixed focus dialogue snapshot state 类型错误")
    binding = mixed_focus_dialogue_runtime_binding_v1(runtime)
    validate_public_mixed_focus_dialogue_runtime_v1(state, runtime)
    inner = snapshot_public_mixed_frame_dialogue_state_v4(
        state.mixed_state,
        runtime,
    )
    ledger = snapshot_mixed_conversation_focus_context_v3(state.focus_context)
    result = [RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V1]
    for label, value in (
            ("mixed focus dialogue runtime binding", binding),
            ("mixed focus dialogue V4 inner snapshot", inner),
            ("mixed focus dialogue V3 ledger snapshot", ledger),
    ):
        _pack(result, value, label=label, allow_empty=False)
    result.append(_u64(
        state.next_operation_ordinal,
        label="mixed focus dialogue outer next operation ordinal",
    ))
    restored = restore_public_mixed_focus_dialogue_state_v1(tuple(result), runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue snapshot encoder readback 漂移")
    return tuple(result)


def restore_public_mixed_focus_dialogue_state_v1(
        record: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedFocusDialogueStateV1:
    """恢复 outer state：固定顺序为 V4、V3、projection、course replay。"""
    record = _record(
        record,
        label="mixed focus dialogue outer snapshot",
        allow_empty=False,
    )
    expected_binding = mixed_focus_dialogue_runtime_binding_v1(runtime)
    cursor = 0
    version, cursor = _read_scalar(
        record,
        cursor,
        label="mixed focus dialogue outer snapshot version",
    )
    if version != RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer snapshot version 未注册")
    binding, cursor = _read_segment(
        record,
        cursor,
        label="mixed focus dialogue runtime binding",
        allow_empty=False,
    )
    if binding != expected_binding:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue runtime binding 漂移")
    inner_record, cursor = _read_segment(
        record,
        cursor,
        label="mixed focus dialogue V4 inner snapshot",
        allow_empty=False,
    )
    ledger_record, cursor = _read_segment(
        record,
        cursor,
        label="mixed focus dialogue V3 ledger snapshot",
        allow_empty=False,
    )
    ordinal, cursor = _read_scalar(
        record,
        cursor,
        label="mixed focus dialogue outer next operation ordinal",
    )
    _u64(ordinal, label="mixed focus dialogue outer next operation ordinal")
    if ordinal < 1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer next operation ordinal 必须大于零")
    if cursor != len(record):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer snapshot 含尾随整数")
    try:
        mixed_state = restore_public_mixed_frame_dialogue_state_v4(
            inner_record,
            runtime,
        )
    except (ConversationRawMixedDialogueSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue V4 inner snapshot 无法恢复") from error
    try:
        focus_context = restore_mixed_conversation_focus_context_v3(ledger_record)
    except (MixedFocusContextSnapshotError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue V3 ledger snapshot 无法恢复") from error
    try:
        state = ConversationRawMixedFocusDialogueStateV1(
            mixed_state,
            focus_context,
            ordinal,
        )
    except (ConversationRawMixedFocusDialogueSessionError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer projection 无法恢复") from error
    validate_public_mixed_focus_dialogue_runtime_v1(state, runtime)
    return state


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """将非负数学整数编码为最短 unsigned big-endian byte vector。"""
    if type(value) is not int or value < 0:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 必须是非负严格整数")
    size = max(1, (value.bit_length() + 7) // 8)
    _u64(size, label=f"{label} byte length")
    if size > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            f"{label} 超出 outer snapshot bytes 预算")
    return value.to_bytes(size, "big")


def _read_u64_bytes(
        payload: bytes,
        cursor: int,
        *,
        label: str,
        ) -> tuple[int, int]:
    """逐 byte 读取 big-endian u64，不依赖宿主 struct 序列化。"""
    if cursor > len(payload) - 8:
        raise ConversationRawMixedFocusDialogueSnapshotError(f"{label} 截断")
    value = 0
    for item in payload[cursor:cursor + 8]:
        value = (value << 8) | item
    return value, cursor + 8


def encode_public_mixed_focus_dialogue_snapshot_v1_bytes(
        state: ConversationRawMixedFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> bytes:
    """以固定 count/length/minimal-unsigned transport 编码 outer logical snapshot。"""
    record = snapshot_public_mixed_focus_dialogue_state_v1(state, runtime)
    result = bytearray()
    result.extend(_u64(
        RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V1,
        label="mixed focus dialogue bytes version",
    ).to_bytes(8, "big"))
    result.extend(_u64(
        len(record),
        label="mixed focus dialogue bytes integer count",
    ).to_bytes(8, "big"))
    for index, value in enumerate(record):
        encoded = _unsigned_integer_bytes(
            value,
            label=f"mixed focus dialogue integer[{index}]",
        )
        result.extend(_u64(
            len(encoded),
            label=f"mixed focus dialogue integer[{index}] length",
        ).to_bytes(8, "big"))
        result.extend(encoded)
        if len(result) > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
            raise ConversationRawMixedFocusDialogueSnapshotError(
                "mixed focus dialogue outer snapshot bytes 超出固定预算")
    payload = bytes(result)
    restored = decode_public_mixed_focus_dialogue_snapshot_v1_bytes(payload, runtime)
    if restored.canonical_record() != state.canonical_record():
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer bytes encoder readback 漂移")
    return payload


def decode_public_mixed_focus_dialogue_snapshot_v1_bytes(
        payload: bytes,
        runtime: PublicDialogueRuntimeV1,
        ) -> ConversationRawMixedFocusDialogueStateV1:
    """严格解码 outer bytes，拒绝 V4/V3、leading zero、截断与尾随 bytes。"""
    if type(payload) is not bytes or not payload:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer snapshot bytes 必须是非空 raw bytes")
    if len(payload) > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer snapshot bytes 超出固定预算")
    cursor = 0
    version, cursor = _read_u64_bytes(
        payload,
        cursor,
        label="mixed focus dialogue bytes version",
    )
    if version != RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V1:
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer bytes version 未注册")
    count, cursor = _read_u64_bytes(
        payload,
        cursor,
        label="mixed focus dialogue bytes integer count",
    )
    if (count > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_INTEGER_COUNT_V1
            or count > (len(payload) - cursor) // 9):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue bytes integer count 越界")
    values: list[int] = []
    for index in range(count):
        size, cursor = _read_u64_bytes(
            payload,
            cursor,
            label=f"mixed focus dialogue integer[{index}] length",
        )
        if (size < 1 or size > len(payload) - cursor
                or size > RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1):
            raise ConversationRawMixedFocusDialogueSnapshotError(
                f"mixed focus dialogue integer[{index}] length 越界")
        encoded = payload[cursor:cursor + size]
        cursor += size
        if len(encoded) > 1 and encoded[0] == 0:
            raise ConversationRawMixedFocusDialogueSnapshotError(
                f"mixed focus dialogue integer[{index}] 非规范 leading zero")
        value = 0
        for item in encoded:
            value = (value << 8) | item
        values.append(value)
    if cursor != len(payload):
        raise ConversationRawMixedFocusDialogueSnapshotError(
            "mixed focus dialogue outer snapshot bytes 含尾随 bytes")
    return restore_public_mixed_focus_dialogue_state_v1(tuple(values), runtime)


__all__ = [
    "RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_BINDING_RECORD_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_RECORD_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_BIG_ENDIAN_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_RECORD_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_U64_WIDTH_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_SNAPSHOT_TRANSPORT_UNSIGNED_MINIMAL_V1",
    "ConversationRawMixedFocusDialogueSnapshotError",
    "decode_public_mixed_focus_dialogue_snapshot_v1_bytes",
    "encode_public_mixed_focus_dialogue_snapshot_v1_bytes",
    "mixed_focus_dialogue_runtime_binding_v1",
    "mixed_focus_dialogue_runtime_identity_v1",
    "mixed_focus_dialogue_snapshot_transport_record_v1",
    "restore_public_mixed_focus_dialogue_state_v1",
    "snapshot_public_mixed_focus_dialogue_state_v1",
    "validate_public_mixed_focus_dialogue_runtime_v1",
]
