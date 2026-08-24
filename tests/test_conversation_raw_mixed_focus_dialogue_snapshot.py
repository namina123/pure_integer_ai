"""DLG-RAW-12 V4/V3 outer snapshot 与 runtime evidence 回查专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    MixedConversationFocusContextStateV3,
    ProviderOriginFollowupFocusTurnV1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    ConversationRawMixedFocusDialogueSessionError,
    ConversationRawMixedFocusDialogueStateV1,
    run_public_mixed_focus_dialogue_turn_v1,
    start_public_mixed_focus_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_snapshot import (
    ConversationRawMixedFocusDialogueSnapshotError,
    decode_public_mixed_focus_dialogue_snapshot_v1_bytes,
    encode_public_mixed_focus_dialogue_snapshot_v1_bytes,
    restore_public_mixed_focus_dialogue_state_v1,
    snapshot_public_mixed_focus_dialogue_state_v1,
    validate_public_mixed_focus_dialogue_runtime_v1,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以显式 UTF-8 u8 tuple 形成与 terminal 相同的输入。"""
    return tuple(text.encode("utf-8"))


def _output(turn) -> str:
    """从唯一 accepted carrier 读取原始输出，不重组语言文本。"""
    result = (turn.answer or turn.provider_answer
              or turn.provider_followup_answer)
    assert result is not None and result.accepted
    payload = (result.output_u8 if turn.provider_followup_answer is result
               else result.output_bytes)
    return bytes(payload).decode("utf-8")


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载完整公开 runtime；不读取 private、训练或长期 memory。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


def _after_first_focus(runtime: PublicDialogueRuntimeV1):
    """构造 provider -> focus 的真实公开状态，供 snapshot 测试共享。"""
    state = start_public_mixed_focus_dialogue((65214, 1, 1))
    first = run_public_mixed_focus_dialogue_turn_v1(
        state, _raw("什么导致路面结冰？"), runtime)
    second = run_public_mixed_focus_dialogue_turn_v1(
        first.after, _raw("它的结果是什么？"), runtime)
    assert _output(first) == "寒潮使得路面结冰。"
    assert _output(second) == "路面结冰"
    return second.after


def test_outer_snapshot_replays_v4_v3_then_next_focus_turn_bit_exact(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """A/B restore 后第三问的 carrier、state、logical/bytes snapshot 必须完全相同。"""
    state = _after_first_focus(runtime)
    logical = snapshot_public_mixed_focus_dialogue_state_v1(state, runtime)
    payload = encode_public_mixed_focus_dialogue_snapshot_v1_bytes(state, runtime)
    restored_logical = restore_public_mixed_focus_dialogue_state_v1(logical, runtime)
    restored_bytes = decode_public_mixed_focus_dialogue_snapshot_v1_bytes(
        payload, runtime)

    assert restored_logical.canonical_record() == state.canonical_record()
    assert restored_bytes.canonical_record() == state.canonical_record()
    assert (encode_public_mixed_focus_dialogue_snapshot_v1_bytes(
        restored_logical, runtime) == payload)

    left = run_public_mixed_focus_dialogue_turn_v1(
        state, _raw("它的原因是什么？"), runtime)
    right = run_public_mixed_focus_dialogue_turn_v1(
        restored_bytes, _raw("它的原因是什么？"), runtime)
    assert _output(left) == "寒潮"
    assert _output(right) == "寒潮"
    assert left.canonical_record() == right.canonical_record()
    assert left.after.canonical_record() == right.after.canonical_record()
    assert (encode_public_mixed_focus_dialogue_snapshot_v1_bytes(
        left.after, runtime)
            == encode_public_mixed_focus_dialogue_snapshot_v1_bytes(
                right.after, runtime))


def test_runtime_validator_rejects_self_identified_foreign_catalog_evidence(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """合法尾部 identity 不能替代 event evidence 对当前 catalog 的全记录绑定。"""
    state = _after_first_focus(runtime)
    event = state.focus_context.turns[-1]
    assert type(event) is ProviderOriginFollowupFocusTurnV1
    foreign_identity = (251,) * 32
    foreign_record = (1, 32, *foreign_identity)
    admission = replace(
        event.admission,
        catalog_record=foreign_record,
        catalog_identity_u8=foreign_identity,
        admission_identity_u8=(),
    )
    forged_event = replace(event, admission=admission, turn_identity_u8=())
    forged_context = MixedConversationFocusContextStateV3(
        state.focus_context.conversation_key,
        state.focus_context.revision,
        state.focus_context.previous_snapshot_digest_u8,
        (*state.focus_context.turns[:-1], forged_event),
    )
    forged_state = ConversationRawMixedFocusDialogueStateV1(
        state.mixed_state,
        forged_context,
        state.next_operation_ordinal,
    )

    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        validate_public_mixed_focus_dialogue_runtime_v1(forged_state, runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        snapshot_public_mixed_focus_dialogue_state_v1(forged_state, runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSessionError):
        run_public_mixed_focus_dialogue_turn_v1(
            forged_state,
            _raw("它的原因是什么？"),
            runtime,
        )


def test_outer_snapshot_rejects_wrong_versions_and_noncanonical_bytes(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """V2/V4/V5 header、截断、尾随和 leading-zero transport 均不得被 outer 接受。"""
    state = _after_first_focus(runtime)
    logical = snapshot_public_mixed_focus_dialogue_state_v1(state, runtime)
    for version in (2, 4, 5):
        with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
            restore_public_mixed_focus_dialogue_state_v1(
                (version, *logical[1:]), runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        restore_public_mixed_focus_dialogue_state_v1(logical[:-1], runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        restore_public_mixed_focus_dialogue_state_v1((*logical, 0), runtime)

    payload = encode_public_mixed_focus_dialogue_snapshot_v1_bytes(state, runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        decode_public_mixed_focus_dialogue_snapshot_v1_bytes(payload[:-1], runtime)
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        decode_public_mixed_focus_dialogue_snapshot_v1_bytes(payload + b"\x00", runtime)
    noncanonical = (
        payload[:16]
        + (2).to_bytes(8, "big")
        + b"\x00"
        + payload[24:]
    )
    with pytest.raises(ConversationRawMixedFocusDialogueSnapshotError):
        decode_public_mixed_focus_dialogue_snapshot_v1_bytes(noncanonical, runtime)
