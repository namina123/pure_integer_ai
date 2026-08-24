"""DLG-RAW-14 snapshot 的最小 A/B continuation 与 fail-closed 专项。"""
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
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue import (
    ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1,
    ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1,
    build_public_route_clarification_dialogue_runtime_v1,
    run_public_route_clarification_dialogue_turn_v1,
    start_public_route_clarification_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue_snapshot import (
    ConversationRawRouteClarificationDialogueSnapshotError,
    decode_public_route_clarification_dialogue_snapshot_v1_bytes,
    encode_public_route_clarification_dialogue_snapshot_v1_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]


def _runtime():
    """从安装内公开 closure 构造唯一 DLG-RAW-14 runtime。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    inner = build_public_dialogue_runtime_v1(
        closure,
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(_ROOT),
    )
    return build_public_route_clarification_dialogue_runtime_v1(inner)


def _offered_state():
    """运行真实 V3 ambiguity，得到有 source-bound candidates 的 active pending。"""
    runtime = _runtime()
    initial = start_public_route_clarification_dialogue((65001, 34, 14))
    offer = run_public_route_clarification_dialogue_turn_v1(
        initial,
        tuple("东岸入口何时启用？\n".encode("utf-8")),
        runtime,
    )
    assert offer.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1)
    assert offer.after.pending is not None
    assert len(offer.after.pending.candidates) == 2
    return runtime, offer


def test_bytes_snapshot_preserves_active_offer_and_both_full_question_selections() -> None:
    """同一 offer 经 bytes A/B 后，两个完整候选重输均得到逐 record 相同结果。"""
    runtime, offer = _offered_state()
    payload = encode_public_route_clarification_dialogue_snapshot_v1_bytes(
        offer.after,
        runtime,
    )
    restored = decode_public_route_clarification_dialogue_snapshot_v1_bytes(
        payload,
        runtime,
    )

    assert restored.canonical_record() == offer.after.canonical_record()
    assert restored.pending is not None
    for candidate in restored.pending.candidates:
        raw = (*candidate.option_surface_u8, 0x0A)
        direct = run_public_route_clarification_dialogue_turn_v1(
            offer.after,
            raw,
            runtime,
        )
        replay = run_public_route_clarification_dialogue_turn_v1(
            restored,
            raw,
            runtime,
        )
        assert direct.response.response_kind == (
            ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1)
        assert direct.selection_event is not None
        assert replay.canonical_record() == direct.canonical_record()


def test_bytes_snapshot_rejects_mutated_or_trailing_physical_bytes() -> None:
    """version 篡改和 trailing byte 都不得被解释为可恢复状态。"""
    runtime, offer = _offered_state()
    payload = encode_public_route_clarification_dialogue_snapshot_v1_bytes(
        offer.after,
        runtime,
    )
    mutated = bytearray(payload)
    mutated[7] ^= 1

    with pytest.raises(ConversationRawRouteClarificationDialogueSnapshotError):
        decode_public_route_clarification_dialogue_snapshot_v1_bytes(
            bytes(mutated), runtime)
    with pytest.raises(ConversationRawRouteClarificationDialogueSnapshotError):
        decode_public_route_clarification_dialogue_snapshot_v1_bytes(
            b"".join((payload, b"\x00")), runtime)
