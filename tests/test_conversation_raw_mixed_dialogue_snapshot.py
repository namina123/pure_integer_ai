"""DLG-RAW-11B V3 mixed-session binding 与 snapshot 的定向回归。"""
from __future__ import annotations

from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH,
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    ConversationRawMixedDialogueStateV2,
    run_public_mixed_frame_dialogue_turn,
    start_public_mixed_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONTEXT,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_snapshot import (
    ConversationRawMixedDialogueSnapshotError,
    decode_public_mixed_frame_dialogue_snapshot_bytes,
    encode_public_mixed_frame_dialogue_snapshot_bytes,
    mixed_dialogue_runtime_binding_v3,
    mixed_dialogue_runtime_identity_v3,
    restore_public_mixed_frame_dialogue_state,
    snapshot_public_mixed_frame_dialogue_state,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
)


_ROOT = Path(__file__).resolve().parents[1]


def _runtime_from_root(root: Path) -> PublicDialogueRuntimeV1:
    """只由 host 读取同一份公开资源闭包和已冻结 provider snapshot。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(root),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(root),
    )


def _copy_public_dialogue_root(tmp_path: Path) -> Path:
    """复制所有逻辑公开资源和 provider snapshot，形成物理 root B。"""
    root = tmp_path / "public-dialogue-root-b"
    relative_paths = (
        *PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
        *(item.encode("ascii") for item in PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS),
        PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH.encode("ascii"),
    )
    for relative in relative_paths:
        destination = root / Path(*relative.decode("ascii").split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / Path(*relative.decode("ascii").split("/")), destination)
    return root


@pytest.fixture(scope="module")
def runtime_a() -> PublicDialogueRuntimeV1:
    """避免每个断言重复重建公开课程与 proof provider。"""
    return _runtime_from_root(_ROOT)


@pytest.fixture(scope="module")
def mixed_state(runtime_a: PublicDialogueRuntimeV1) -> ConversationRawMixedDialogueStateV2:
    """形成真实 Frame 后接真实 provider 的 V2 tagged session state。"""
    initial = start_public_mixed_frame_dialogue((65212, 1, 1))
    frame = run_public_mixed_frame_dialogue_turn(
        initial,
        tuple("北川站东门何时启用？".encode("utf-8")),
        runtime_a,
    )
    assert frame.answer is not None and frame.answer.accepted
    provider = run_public_mixed_frame_dialogue_turn(
        frame.after,
        tuple("寒潮导致什么？".encode("utf-8")),
        runtime_a,
    )
    assert provider.provider_answer is not None and provider.provider_answer.accepted
    return provider.after


def test_v3_snapshot_round_trips_across_physical_public_roots(
        tmp_path: Path,
        runtime_a: PublicDialogueRuntimeV1,
        mixed_state: ConversationRawMixedDialogueStateV2,
        ) -> None:
    """A/B 的同一公开资源必须得到相同 binding、snapshot bytes 和恢复 state。"""
    root_b = _copy_public_dialogue_root(tmp_path)
    runtime_b = _runtime_from_root(root_b)
    assert mixed_dialogue_runtime_binding_v3(runtime_a) == (
        mixed_dialogue_runtime_binding_v3(runtime_b))
    assert mixed_dialogue_runtime_identity_v3(runtime_a) == (
        mixed_dialogue_runtime_identity_v3(runtime_b))

    record_a = snapshot_public_mixed_frame_dialogue_state(mixed_state, runtime_a)
    bytes_a = encode_public_mixed_frame_dialogue_snapshot_bytes(mixed_state, runtime_a)
    restored_record = restore_public_mixed_frame_dialogue_state(record_a, runtime_b)
    restored_bytes = decode_public_mixed_frame_dialogue_snapshot_bytes(bytes_a, runtime_b)

    assert restored_record.canonical_record() == mixed_state.canonical_record()
    assert restored_bytes.canonical_record() == mixed_state.canonical_record()
    assert snapshot_public_mixed_frame_dialogue_state(restored_record, runtime_b) == record_a
    assert encode_public_mixed_frame_dialogue_snapshot_bytes(restored_bytes, runtime_b) == bytes_a

    blocked = run_public_mixed_frame_dialogue_turn(
        restored_bytes,
        tuple("它是在什么时候启用的？".encode("utf-8")),
        runtime_b,
    )
    assert blocked.answer is not None
    assert blocked.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert blocked.after.context == restored_bytes.context


def test_v3_snapshot_rejects_runtime_binding_and_context_tail_drift(
        runtime_a: PublicDialogueRuntimeV1,
        mixed_state: ConversationRawMixedDialogueStateV2,
        ) -> None:
    """binding、inner context 和 outer tail 的任何漂移都不得形成部分恢复 state。"""
    record = snapshot_public_mixed_frame_dialogue_state(mixed_state, runtime_a)
    binding_count = record[1]
    binding_drift = list(record)
    binding_drift[2] ^= 1
    with pytest.raises(ConversationRawMixedDialogueSnapshotError, match="binding"):
        restore_public_mixed_frame_dialogue_state(tuple(binding_drift), runtime_a)

    context_start = 2 + binding_count
    conversation_key_count = record[context_start]
    context_start += 1 + conversation_key_count
    context_size_index = context_start
    context_drift = list(record)
    context_drift[context_size_index + 2] ^= 1
    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        restore_public_mixed_frame_dialogue_state(tuple(context_drift), runtime_a)

    with pytest.raises(ConversationRawMixedDialogueSnapshotError, match="尾随"):
        restore_public_mixed_frame_dialogue_state((*record, 0), runtime_a)
    with pytest.raises(ConversationRawMixedDialogueSnapshotError, match="截断|长度越界"):
        restore_public_mixed_frame_dialogue_state(record[:-1], runtime_a)


def test_v3_snapshot_bytes_reject_noncanonical_and_tail_drift(
        runtime_a: PublicDialogueRuntimeV1,
        mixed_state: ConversationRawMixedDialogueStateV2,
        ) -> None:
    """outer bytes transport 也必须拒绝截断、尾随及 unsigned leading zero。"""
    payload = encode_public_mixed_frame_dialogue_snapshot_bytes(
        mixed_state,
        runtime_a,
    )
    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        decode_public_mixed_frame_dialogue_snapshot_bytes(payload[:-1], runtime_a)
    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        decode_public_mixed_frame_dialogue_snapshot_bytes(payload + b"\x00", runtime_a)

    noncanonical = (
        payload[:16]
        + (2).to_bytes(8, "big")
        + b"\x00"
        + payload[24:]
    )
    with pytest.raises(ConversationRawMixedDialogueSnapshotError):
        decode_public_mixed_frame_dialogue_snapshot_bytes(noncanonical, runtime_a)
