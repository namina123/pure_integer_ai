"""DLG-RAW-12 V3 append-only focus context snapshot codec 专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextState,
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
    PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
    PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
    ProviderOriginAnchorProjectionV1,
    ProviderOriginOccurrenceV1,
    ProviderOriginRoleBindingV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1,
    MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
    MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    FrameQuestionAnswerTurnV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFocusAdmissionV1,
    start_mixed_conversation_focus_context_v3,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context_snapshot import (
    MixedFocusContextSnapshotError,
    decode_mixed_conversation_focus_context_snapshot_v3_bytes,
    encode_mixed_conversation_focus_context_snapshot_v3_bytes,
    mixed_focus_context_snapshot_codec_identity_v3,
    mixed_focus_context_snapshot_codec_revision_v3,
    restore_mixed_conversation_focus_context_v3,
    snapshot_mixed_conversation_focus_context_v3,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    intake_raw_conversation_vector,
)


_CONVERSATION_KEY = (91301, 1, 1)
_LEGACY_KEY = (91301, 2, 1)


def _key(value: int) -> tuple[int, ...]:
    """构造本 codec 专项的稳定整数 key。"""
    return (91301, value, 1)


def _digest(value: int) -> tuple[int, ...]:
    """构造格式正确的 raw u8[32] 测试 identity。"""
    return (value,) * 32


def _identity(domain: bytes, record: tuple[int, ...]) -> tuple[int, ...]:
    """按 portable SHA framing 形成测试 admission 的输入/回读 identity。"""
    return tuple(portable_sha256_v1(domain, (record,)))


def _foreign_record(identity: tuple[int, ...]) -> tuple[int, ...]:
    """构造末尾 self identity 已冻结的外来 canonical evidence record。"""
    return (1, 32, *identity)


def _anchor() -> ProviderOriginAnchorProjectionV1:
    """构造具有两个同 anchor occurrence 的最小 source projection。"""
    first_binding = ProviderOriginRoleBindingV1(
        _key(101), _key(111), _key(121), 0)
    second_binding = ProviderOriginRoleBindingV1(
        _key(102), _key(112), _key(122), 1)
    first_occurrence = ProviderOriginOccurrenceV1(
        _key(131), first_binding.filler_key, 0, 0, 1)
    second_occurrence = ProviderOriginOccurrenceV1(
        _key(132), second_binding.filler_key, 1, 1, 2)
    return ProviderOriginAnchorProjectionV1(
        PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        _digest(1), _digest(2), _digest(3), _digest(4), _digest(5), _digest(6),
        _key(141), _key(142), _digest(7), _key(143), _key(144), _key(145),
        _key(146), _key(147), PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
        _key(148), first_binding.binding_key, first_binding.role_key,
        first_binding.filler_key, first_occurrence.occurrence_key,
        first_occurrence.start, first_occurrence.end,
        (first_binding, second_binding), (first_occurrence, second_occurrence),
        (65, 66), (65, 66),
    )


def _admission(
        anchor: ProviderOriginAnchorProjectionV1,
        *,
        reference_index: int,
        target_index: int,
        marker: int,
        ) -> ProviderOriginFocusAdmissionV1:
    """构造一个 target/output 都能被 anchor 原始 slice 回读的 admission。"""
    reference_binding = anchor.ordered_role_bindings[reference_index]
    reference_occurrence = anchor.ordered_occurrences[reference_index]
    target_binding = anchor.ordered_role_bindings[target_index]
    target_occurrence = anchor.ordered_occurrences[target_index]
    catalog_identity = _digest(30 + marker)
    form_identity = _digest(40 + marker)
    candidate_identity = _digest(50 + marker)
    input_record = intake_raw_conversation_vector((89, marker)).canonical_record()
    output = anchor.output_u8[target_occurrence.start:target_occurrence.end]
    scalars = anchor.output_scalars[target_occurrence.start:target_occurrence.end]
    readback = intake_raw_conversation_vector(output).canonical_record()
    return ProviderOriginFocusAdmissionV1(
        _foreign_record(catalog_identity), catalog_identity,
        _foreign_record(form_identity), form_identity,
        _foreign_record(candidate_identity), candidate_identity,
        input_record,
        _identity(MIXED_FOCUS_CONTEXT_INPUT_INTAKE_IDENTITY_DOMAIN_V1, input_record),
        readback,
        _identity(MIXED_FOCUS_CONTEXT_OUTPUT_READBACK_IDENTITY_DOMAIN_V1, readback),
        anchor.provider_kind, anchor.provider_identity_u8, anchor.runtime_identity_u8,
        anchor.catalog_record_identity_u8, anchor.provider_result_identity_u8,
        anchor.anchor_identity_u8, anchor.source_record_key,
        anchor.source_ref_stable_key, anchor.source_commitment_u8,
        anchor.w03_observation_key, anchor.w04_observation_key,
        anchor.w05_observation_key, anchor.generation_construction_key,
        anchor.proposition_key, anchor.predicate_key, anchor.relation_kind_code,
        reference_binding.binding_key, reference_binding.role_key,
        reference_binding.filler_key, reference_occurrence.occurrence_key,
        reference_occurrence.start, reference_occurrence.end,
        target_binding.binding_key, target_binding.role_key,
        target_binding.filler_key, target_occurrence.occurrence_key,
        target_occurrence.start, target_occurrence.end, scalars, output,
    )


def _source() -> SourceRef:
    """构造 legacy Frame typed record 所需的 citation。"""
    return SourceRef(91301, 3, 4, GLOBAL_OWNER_SCOPE, VersionBundle())


def _frame_turn(
        ordinal: int,
        legacy_context: ConversationContextState,
        ) -> ConversationTurnState:
    """构造一条带真实 legacy read 的 Frame turn。"""
    return ConversationTurnState(
        ordinal,
        _key(200 + ordinal), _key(210 + ordinal), _key(220 + ordinal),
        _key(230 + ordinal), ObjectIdentity(OBJECT_CONCEPT, _key(240 + ordinal)),
        (_key(250 + ordinal),), (_source(),), (_key(260 + ordinal),),
        _key(270 + ordinal), _key(280 + ordinal), legacy_context.read(1),
    )


def _state_after_two_focuses(*, with_frames: bool):
    """形成 provider -> focus -> focus，可选插入两条 Frame 以验证旧链隔离。"""
    anchor = _anchor()
    state = start_mixed_conversation_focus_context_v3(_CONVERSATION_KEY)
    first_frame: ConversationTurnState | None = None
    second_frame: ConversationTurnState | None = None
    if with_frames:
        legacy_initial = start_conversation_context(_LEGACY_KEY)
        first_frame = _frame_turn(0, legacy_initial)
        frame_append = state.admit_frame_qa_run(first_frame, state.read(0))
        assert frame_append.accepted
        state = frame_append.after
        legacy_after_first = ConversationContextState(
            _LEGACY_KEY,
            1,
            legacy_initial.digest(),
            (first_frame,),
        )
        second_frame = _frame_turn(1, legacy_after_first)
    provider = state.admit_provider_origin_projection(anchor, state.read(1 if with_frames else 0))
    assert provider.accepted
    first = provider.after.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=0, target_index=1, marker=1),
        provider.after.read(1),
    )
    assert first.accepted
    second = first.after.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=1, target_index=0, marker=2),
        first.after.read(1),
    )
    assert second.accepted
    state = second.after
    if with_frames:
        assert second_frame is not None
        final = state.admit_frame_qa_run(second_frame, state.read(1))
        assert final.accepted
        state = final.after
    return state, anchor, first.after, first_frame, second_frame


def _turn_start(snapshot: tuple[int, ...], index: int) -> int:
    """定位 top-level V3 snapshot 中第 index 条 count-framed turn 的内容起点。"""
    cursor = 2
    assert snapshot[cursor] == 3
    cursor += 2  # state record version + schema
    key_count = snapshot[cursor]
    cursor += 1 + key_count
    cursor += 1  # revision
    digest_count = snapshot[cursor]
    cursor += 1 + digest_count
    turn_count = snapshot[cursor]
    cursor += 1
    assert index < turn_count
    for _ in range(index):
        size = snapshot[cursor]
        cursor += 1 + size
    return cursor + 1


def test_snapshot_round_trip_replays_focus_ledger_and_frame_legacy_chain() -> None:
    """Frame/provider/focus interleave 恢复后，focus ledger 与第二 Frame legacy read 均不漂移。"""
    state, _anchor_value, _after_first, first_frame, second_frame = (
        _state_after_two_focuses(with_frames=True))
    assert first_frame is not None and second_frame is not None

    record = snapshot_mixed_conversation_focus_context_v3(state)
    restored = restore_mixed_conversation_focus_context_v3(record)
    payload = encode_mixed_conversation_focus_context_snapshot_v3_bytes(state)
    restored_bytes = decode_mixed_conversation_focus_context_snapshot_v3_bytes(payload)

    assert restored.canonical_record() == state.canonical_record()
    assert restored.digest() == state.digest()
    assert restored_bytes.canonical_record() == state.canonical_record()
    assert isinstance(restored.turns[0], FrameQuestionAnswerTurnV3)
    assert restored.turns[0].frame_turn.stable_key() == first_frame.stable_key()
    assert isinstance(restored.turns[1], ProviderOriginContextTurnV3)
    assert restored.turns[-1].frame_turn.stable_key() == second_frame.stable_key()
    assert encode_mixed_conversation_focus_context_snapshot_v3_bytes(restored) == payload


def test_snapshot_restore_then_next_focus_event_is_deterministic() -> None:
    """A/B restore 后同一第三焦点输入必须产生相同 event、state 与 bytes。"""
    state, anchor, after_first, _first_frame, _second_frame = (
        _state_after_two_focuses(with_frames=False))
    # ``after_first`` 的 tail 是第一条 focus，适合验证同一下一输入的 A/B replay。
    restored = restore_mixed_conversation_focus_context_v3(
        snapshot_mixed_conversation_focus_context_v3(after_first))
    third_a = after_first.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=1, target_index=0, marker=9),
        after_first.read(1),
    )
    third_b = restored.admit_provider_origin_followup_focus(
        _admission(anchor, reference_index=1, target_index=0, marker=9),
        restored.read(1),
    )

    assert third_a.accepted and third_b.accepted
    assert third_a.appended_turn.canonical_record() == third_b.appended_turn.canonical_record()
    assert third_a.after.canonical_record() == third_b.after.canonical_record()
    assert (encode_mixed_conversation_focus_context_snapshot_v3_bytes(third_a.after)
            == encode_mixed_conversation_focus_context_snapshot_v3_bytes(third_b.after))
    assert state.revision == 3


def test_snapshot_rejects_unknown_tag_versions_and_bytes_noncanonicality() -> None:
    """未知 tag、V2/V4 版本、截断、尾随和 leading-zero transport 必须 fail closed。"""
    state, _anchor_value, _after_first, _first_frame, _second_frame = (
        _state_after_two_focuses(with_frames=False))
    record = snapshot_mixed_conversation_focus_context_v3(state)
    provider_start = _turn_start(record, 0)
    assert record[provider_start + 1] == MIXED_FOCUS_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION
    unknown = list(record)
    unknown[provider_start + 1] = 99
    with pytest.raises(MixedFocusContextSnapshotError):
        restore_mixed_conversation_focus_context_v3(tuple(unknown))
    with pytest.raises(MixedFocusContextSnapshotError):
        restore_mixed_conversation_focus_context_v3((2, *record[1:]))
    with pytest.raises(MixedFocusContextSnapshotError):
        restore_mixed_conversation_focus_context_v3((4, *record[1:]))
    with pytest.raises(MixedFocusContextSnapshotError):
        restore_mixed_conversation_focus_context_v3(record[:-1])
    with pytest.raises(MixedFocusContextSnapshotError):
        restore_mixed_conversation_focus_context_v3((*record, 0))

    payload = encode_mixed_conversation_focus_context_snapshot_v3_bytes(state)
    with pytest.raises(MixedFocusContextSnapshotError):
        decode_mixed_conversation_focus_context_snapshot_v3_bytes(payload[:-1])
    with pytest.raises(MixedFocusContextSnapshotError):
        decode_mixed_conversation_focus_context_snapshot_v3_bytes(payload + b"\x00")
    noncanonical = (
        payload[:16]
        + (2).to_bytes(8, "big")
        + b"\x00"
        + payload[24:]
    )
    with pytest.raises(MixedFocusContextSnapshotError):
        decode_mixed_conversation_focus_context_snapshot_v3_bytes(noncanonical)


def test_codec_revision_identity_is_stable_raw_u8() -> None:
    """codec binding 必须暴露确定 record 与 raw u8[32] identity。"""
    revision = mixed_focus_context_snapshot_codec_revision_v3()
    identity = mixed_focus_context_snapshot_codec_identity_v3()

    assert revision == mixed_focus_context_snapshot_codec_revision_v3()
    assert all(type(item) is int and item >= 0 for item in revision)
    assert len(identity) == 32
    assert all(type(item) is int and 0 <= item <= 255 for item in identity)
