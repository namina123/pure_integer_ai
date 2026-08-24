"""DLG-RAW-12 旧 V5 snapshot 与公开终端连续行为的定向验证。"""
from __future__ import annotations

from io import BytesIO
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
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_snapshot import (
    snapshot_public_mixed_frame_dialogue_state_v4,
)
from pure_integer_ai.experiments.conversation_raw_provider_focus_dialogue_session import (
    run_public_provider_focus_dialogue_turn_v1,
    start_public_provider_focus_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_provider_focus_dialogue_snapshot import (
    ConversationRawProviderFocusDialogueSnapshotError,
    RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5,
    RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5,
    decode_public_provider_focus_dialogue_snapshot_v5_bytes,
    encode_public_provider_focus_dialogue_snapshot_v5_bytes,
    provider_focus_dialogue_snapshot_transport_record_v5,
    restore_public_provider_focus_dialogue_state_v5,
    snapshot_public_provider_focus_dialogue_state_v5,
)
from pure_integer_ai.experiments.run_public_frame_dialogue import (
    run_public_mixed_frame_dialogue_terminal,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """构造明确 UTF-8 raw input vector。"""
    return tuple(text.encode("utf-8"))


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载完整公开 runtime，避免测试伪造 focus 所依赖的来源绑定。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


def _focused_state(runtime: PublicDialogueRuntimeV1):
    """形成已由真实 11C result 投影出的 V5 focus state。"""
    first = run_public_provider_focus_dialogue_turn_v1(
        start_public_provider_focus_dialogue((65212, 6, 1)),
        _raw("寒潮导致什么？"),
        runtime,
    )
    second = run_public_provider_focus_dialogue_turn_v1(
        first.after,
        _raw("它的原因是什么？"),
        runtime,
    )
    assert second.after.focus is not None
    return second.after


def test_v5_transport_record_freezes_budget_and_integer_layout() -> None:
    """V5 transport 必须显式携带 bytes version、预算、u64、字节序和最短整数规则。"""
    assert provider_focus_dialogue_snapshot_transport_record_v5() == (
        1,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_BYTES_V5,
        RAW_PROVIDER_FOCUS_DIALOGUE_SNAPSHOT_MAX_BYTES_V5,
        8,
        1,
        1,
    )


def test_v5_snapshot_replays_focus_chain_after_logical_and_bytes_restore(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """snapshot 必须保存 focus，而非只恢复 V4 inner context 后重答旧锚点。"""
    state = _focused_state(runtime)
    logical = snapshot_public_provider_focus_dialogue_state_v5(state, runtime)
    payload = encode_public_provider_focus_dialogue_snapshot_v5_bytes(state, runtime)
    logical_restored = restore_public_provider_focus_dialogue_state_v5(logical, runtime)
    bytes_restored = decode_public_provider_focus_dialogue_snapshot_v5_bytes(payload, runtime)
    assert logical_restored.canonical_record() == state.canonical_record()
    assert bytes_restored.canonical_record() == state.canonical_record()

    replay = run_public_provider_focus_dialogue_turn_v1(
        bytes_restored,
        _raw("它的结果是什么？"),
        runtime,
    )
    result = replay.provider_followup_answer
    assert result is not None and result.accepted
    assert bytes(result.output_u8).decode("utf-8") == "路面结冰"


def test_v5_snapshot_does_not_upgrade_v4_outer_record(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """V5 restore 不得猜测旧 V4 record 应如何获得 focus optional 分支。"""
    state = _focused_state(runtime)
    v4 = snapshot_public_mixed_frame_dialogue_state_v4(state.mixed_state, runtime)
    with pytest.raises(ConversationRawProviderFocusDialogueSnapshotError):
        restore_public_provider_focus_dialogue_state_v5(v4, runtime)


def test_default_terminal_runs_the_real_three_turn_chain(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """公开 shell 必须保留三轮连续焦点行为，状态 owner 由当前主链决定。"""
    source = BytesIO(
        "寒潮导致什么？\n它的原因是什么？\n它的结果是什么？\n:quit\n".encode("utf-8"))
    target = BytesIO()
    run_public_mixed_frame_dialogue_terminal(
        runtime,
        source,
        target,
        prompts=False,
    )
    assert target.getvalue() == (
        "系统> 寒潮使得路面结冰。\n"
        "系统> 寒潮\n"
        "系统> 路面结冰\n"
    ).encode("utf-8")
