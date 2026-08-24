"""DLG-RAW-11C V4 outer snapshot 的 binding 与 A/B state replay。"""
from __future__ import annotations

from pathlib import Path

import pytest

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
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    run_public_mixed_frame_dialogue_turn_v4,
    start_public_mixed_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_snapshot import (
    ConversationRawMixedDialogueSnapshotError,
    RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4,
    RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4,
    decode_public_mixed_frame_dialogue_snapshot_v4_bytes,
    encode_public_mixed_frame_dialogue_snapshot_v4_bytes,
    mixed_dialogue_snapshot_transport_record_v4,
    restore_public_mixed_frame_dialogue_state,
    restore_public_mixed_frame_dialogue_state_v4,
    snapshot_public_mixed_frame_dialogue_state,
    snapshot_public_mixed_frame_dialogue_state_v4,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以显式 UTF-8 u8 tuple 调用公开对话入口。"""
    return tuple(text.encode("utf-8"))


def test_v4_transport_record_freezes_bytes_layout_and_budget() -> None:
    """V4 binding 必须显式携带 bytes version、上限、u64 width、endian 与最短整数规则。"""
    assert mixed_dialogue_snapshot_transport_record_v4() == (
        1,
        RAW_MIXED_DIALOGUE_SNAPSHOT_BYTES_V4,
        RAW_MIXED_DIALOGUE_SNAPSHOT_MAX_BYTES_V4,
        8,
        1,
        1,
    )


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载完整 V4 runtime，而非读取 private 或训练资源。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


def test_v4_snapshot_replays_after_no_write_followup(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """follow-up 不写 inner context，但 operation ordinal 和 V4 grammar 必须被 snapshot 锁定。"""
    first = run_public_mixed_frame_dialogue_turn_v4(
        start_public_mixed_frame_dialogue((65111, 5, 1)),
        _raw("寒潮导致什么？"),
        runtime,
    )
    second = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw("它的原因是什么？"),
        runtime,
    )
    assert second.after.context == first.after.context

    logical = snapshot_public_mixed_frame_dialogue_state_v4(second.after, runtime)
    encoded = encode_public_mixed_frame_dialogue_snapshot_v4_bytes(
        second.after,
        runtime,
    )
    assert (restore_public_mixed_frame_dialogue_state_v4(logical, runtime)
            .canonical_record() == second.after.canonical_record())
    restored = decode_public_mixed_frame_dialogue_snapshot_v4_bytes(encoded, runtime)
    third = run_public_mixed_frame_dialogue_turn_v4(
        restored,
        _raw("它的原因是什么？"),
        runtime,
    )
    assert third.provider_followup_answer is not None
    assert bytes(third.provider_followup_answer.output_u8).decode("utf-8") == "寒潮"


def test_v3_and_v4_outer_snapshot_versions_do_not_upgrade_each_other(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """旧 V3 只恢复旧 binding；V4 不猜测如何降级为 V3。"""
    state = start_public_mixed_frame_dialogue((65111, 5, 2))
    v3 = snapshot_public_mixed_frame_dialogue_state(state, runtime)
    v4 = snapshot_public_mixed_frame_dialogue_state_v4(state, runtime)

    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        restore_public_mixed_frame_dialogue_state_v4(v3, runtime)
    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        restore_public_mixed_frame_dialogue_state(v4, runtime)
