"""DLG-RAW-11B mixed context V2 snapshot codec 的定向严格回归。"""
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
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    ProviderOriginContextTurnV1,
    start_mixed_conversation_context_v2,
)
from pure_integer_ai.experiments.conversation_provider_origin_context_snapshot import (
    MixedContextSnapshotError,
    decode_mixed_conversation_context_snapshot_v2_bytes,
    encode_mixed_conversation_context_snapshot_v2_bytes,
    mixed_context_snapshot_codec_identity_v2,
    mixed_context_snapshot_codec_revision_v2,
    restore_mixed_conversation_context_v2,
    snapshot_mixed_conversation_context_v2,
)


_MIXED_KEY = (71101, 1, 1)
_LEGACY_KEY = (71101, 2, 1)


def _key(item: int) -> tuple[int, ...]:
    """构造只服务于 codec 结构回归的稳定整数 key。"""
    return (71101, item, 1)


def _digest(item: int) -> tuple[int, ...]:
    """构造格式正确且可区分的 raw u8[32] 测试 identity。"""
    return (item,) * 32


def _source(item: int) -> SourceRef:
    """构造 Frame typed payload 所需的无表层 citation。"""
    return SourceRef(71101, item, item + 1, GLOBAL_OWNER_SCOPE, VersionBundle())


def _frame_turn(
        ordinal: int,
        legacy_context: ConversationContextState,
        ) -> ConversationTurnState:
    """构造一条带真实 legacy read 的 Frame typed turn。"""
    return ConversationTurnState(
        ordinal,
        _key(10 + ordinal),
        _key(20 + ordinal),
        _key(30 + ordinal),
        _key(40 + ordinal),
        ObjectIdentity(OBJECT_CONCEPT, _key(50 + ordinal)),
        (_key(60 + ordinal),),
        (_source(70 + ordinal),),
        (_key(80 + ordinal),),
        _key(90 + ordinal),
        _key(100 + ordinal),
        legacy_context.read(1),
    )


def _anchor() -> ProviderOriginAnchorProjectionV1:
    """构造只用于 codec record 可逆性的已接纳 source anchor。"""
    binding = ProviderOriginRoleBindingV1(
        _key(201),
        _key(202),
        _key(203),
        4,
    )
    occurrence = ProviderOriginOccurrenceV1(
        _key(204),
        _key(203),
        5,
        0,
        1,
    )
    return ProviderOriginAnchorProjectionV1(
        PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        _digest(1),
        _digest(2),
        _digest(3),
        _digest(4),
        _digest(5),
        _digest(6),
        _key(205),
        _key(206),
        _digest(7),
        _key(207),
        _key(208),
        _key(209),
        _key(210),
        _key(211),
        PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
        _key(212),
        binding.binding_key,
        binding.role_key,
        binding.filler_key,
        occurrence.occurrence_key,
        0,
        1,
        (binding,),
        (occurrence,),
        (65,),
        (65,),
    )


def _mixed_state():
    """形成 Frame -> provider -> Frame，第二 Frame 消费非空 legacy read。"""
    legacy_initial = start_conversation_context(_LEGACY_KEY)
    first = _frame_turn(0, legacy_initial)
    legacy_after_first = ConversationContextState(
        _LEGACY_KEY,
        1,
        legacy_initial.digest(),
        (first,),
    )
    second = _frame_turn(1, legacy_after_first)

    state = start_mixed_conversation_context_v2(_MIXED_KEY)
    first_append = state.admit_frame_qa_run(first, state.read(0))
    assert first_append.accepted
    provider_append = first_append.after.admit_provider_origin_projection(
        _anchor(),
        first_append.after.read(1),
    )
    assert provider_append.accepted
    second_append = provider_append.after.admit_frame_qa_run(
        second,
        provider_append.after.read(2),
    )
    assert second_append.accepted
    return second_append.after, first, second


def _turn_start(record: tuple[int, ...], index: int) -> int:
    """定位 top-level snapshot 中某条 count-framed V2 turn 的起点。"""
    cursor = 2
    cursor += 2  # context record version + schema
    key_count = record[cursor]
    cursor += 1 + key_count
    cursor += 1  # revision
    digest_count = record[cursor]
    cursor += 1 + digest_count
    turn_count = record[cursor]
    cursor += 1
    assert index < turn_count
    for _ in range(index):
        size = record[cursor]
        cursor += 1 + size
    return cursor + 1


def _witness_requested_limit_position(record: tuple[int, ...], turn_start: int) -> int:
    """定位一个 turn 内 nested read witness 的 requested-limit 标量。"""
    cursor = turn_start + 3
    digest_count = record[cursor]
    cursor += 1 + digest_count
    witness_count = record[cursor]
    witness_start = cursor + 1
    assert witness_count > 0
    cursor = witness_start + 1
    key_count = record[cursor]
    cursor += 1 + key_count
    cursor += 1  # witness revision
    snapshot_digest_count = record[cursor]
    cursor += 1 + snapshot_digest_count
    return cursor


def test_mixed_snapshot_round_trip_replays_nonempty_legacy_frame_read() -> None:
    """V2 prefix 的 Frame 链可恢复 legacy read，provider 不会污染该旧 compatibility chain。"""
    state, first, second = _mixed_state()

    record = snapshot_mixed_conversation_context_v2(state)
    restored = restore_mixed_conversation_context_v2(record)
    payload = encode_mixed_conversation_context_snapshot_v2_bytes(state)
    restored_bytes = decode_mixed_conversation_context_snapshot_v2_bytes(payload)

    assert restored.canonical_record() == state.canonical_record()
    assert restored.digest() == state.digest()
    assert restored_bytes.canonical_record() == state.canonical_record()
    assert restored.turns[0].frame_turn.stable_key() == first.stable_key()
    assert isinstance(restored.turns[1], ProviderOriginContextTurnV1)
    assert restored.turns[2].frame_turn.stable_key() == second.stable_key()
    assert (encode_mixed_conversation_context_snapshot_v2_bytes(restored)
            == payload)


def test_snapshot_rejects_turn_predecessor_witness_kind_and_record_drift() -> None:
    """未知 tag、前驱摘要和 nested read witness 的任何漂移均不能恢复。"""
    state, _first, _second = _mixed_state()
    record = snapshot_mixed_conversation_context_v2(state)

    provider_start = _turn_start(record, 1)
    assert record[provider_start + 1] == MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION
    kind_drift = list(record)
    kind_drift[provider_start + 1] = 99
    with pytest.raises(MixedContextSnapshotError):
        restore_mixed_conversation_context_v2(tuple(kind_drift))

    predecessor_drift = list(record)
    predecessor_drift[provider_start + 4] ^= 1
    with pytest.raises(MixedContextSnapshotError):
        restore_mixed_conversation_context_v2(tuple(predecessor_drift))

    first_start = _turn_start(record, 0)
    witness_drift = list(record)
    witness_drift[_witness_requested_limit_position(record, first_start)] = 1
    with pytest.raises(MixedContextSnapshotError):
        restore_mixed_conversation_context_v2(tuple(witness_drift))

    with pytest.raises(MixedContextSnapshotError):
        restore_mixed_conversation_context_v2(record[:-1])
    with pytest.raises(MixedContextSnapshotError):
        restore_mixed_conversation_context_v2((*record, 0))


def test_snapshot_bytes_reject_truncation_tail_and_noncanonical_integer() -> None:
    """bytes transport 必须拒绝截断、尾随和 leading-zero unsigned integer。"""
    state, _first, _second = _mixed_state()
    payload = encode_mixed_conversation_context_snapshot_v2_bytes(state)

    with pytest.raises(MixedContextSnapshotError):
        decode_mixed_conversation_context_snapshot_v2_bytes(payload[:-1])
    with pytest.raises(MixedContextSnapshotError):
        decode_mixed_conversation_context_snapshot_v2_bytes(payload + b"\x00")

    # bytes header 为两个 u64；第一整数的原始 one-byte ``2`` 改成 ``00 02``。
    noncanonical = (
        payload[:16]
        + (2).to_bytes(8, "big")
        + b"\x00"
        + payload[24:]
    )
    with pytest.raises(MixedContextSnapshotError):
        decode_mixed_conversation_context_snapshot_v2_bytes(noncanonical)


def test_codec_revision_identity_is_stable_raw_u8() -> None:
    """runtime binding 可消费一个确定的 codec revision record 和 raw identity。"""
    revision = mixed_context_snapshot_codec_revision_v2()
    identity = mixed_context_snapshot_codec_identity_v2()

    assert revision == mixed_context_snapshot_codec_revision_v2()
    assert all(type(item) is int and item >= 0 for item in revision)
    assert len(identity) == 32
    assert all(type(item) is int and 0 <= item <= 255 for item in identity)
