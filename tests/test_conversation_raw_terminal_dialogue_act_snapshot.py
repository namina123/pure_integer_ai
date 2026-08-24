"""DLG-RAW-13 outer snapshot 的唯一 A/B bytes continuation 回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    build_terminal_dialogue_act_runtime_v1,
    run_public_terminal_dialogue_act_turn_v1,
    start_public_terminal_dialogue_act,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act_snapshot import (
    ConversationRawTerminalDialogueActSnapshotError,
    decode_public_terminal_dialogue_act_snapshot_v1_bytes,
    encode_public_terminal_dialogue_act_snapshot_v1_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]


def test_bytes_snapshot_restores_terminal_act_outer_state_for_a_b_continuation() -> None:
    """7 的 state-none act 经 bytes A/B 后仍得到逐 record 相同的下一 7 response。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    inner = build_public_dialogue_runtime_v1(
        closure,
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )
    runtime = build_terminal_dialogue_act_runtime_v1(inner)
    initial = start_public_terminal_dialogue_act((65001, 34, 13))
    first = run_public_terminal_dialogue_act_turn_v1(
        initial,
        tuple(b"uncovered-route?\n"),
        runtime,
    )

    payload = encode_public_terminal_dialogue_act_snapshot_v1_bytes(
        first.after,
        runtime,
    )
    restored = decode_public_terminal_dialogue_act_snapshot_v1_bytes(
        payload,
        runtime,
    )
    direct = run_public_terminal_dialogue_act_turn_v1(
        first.after,
        tuple(b"uncovered-route-again?\r\n"),
        runtime,
    )
    replay = run_public_terminal_dialogue_act_turn_v1(
        restored,
        tuple(b"uncovered-route-again?\r\n"),
        runtime,
    )

    assert replay.canonical_record() == direct.canonical_record()
    assert b"".join((payload, b"\x00")) != payload
    with pytest.raises(ConversationRawTerminalDialogueActSnapshotError):
        decode_public_terminal_dialogue_act_snapshot_v1_bytes(
            b"".join((payload, b"\x00")), runtime)
