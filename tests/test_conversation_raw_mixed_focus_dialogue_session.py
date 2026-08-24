"""DLG-RAW-12 V4/V3 连续来源焦点 outer dialogue 定向验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    conversation_raw_mixed_focus_dialogue_session,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    FrameQuestionAnswerTurnV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFollowupFocusTurnV1,
    ProviderOriginFocusContextError,
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
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_RUNTIME,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    ConversationRawMixedFocusDialogueSessionError,
    ConversationRawMixedFocusDialogueStateV1,
    run_public_mixed_focus_dialogue_turn_v1,
    start_public_mixed_focus_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以明确 UTF-8 u8 tuple 形成公开 runtime 输入。"""
    return tuple(text.encode("utf-8"))


def _output(turn) -> str:
    """从唯一 accepted carrier 读取已验证输出，不重组或归一化文本。"""
    result = (turn.answer or turn.provider_answer
              or turn.provider_followup_answer)
    assert result is not None and result.accepted
    payload = (result.output_u8 if turn.provider_followup_answer is result
               else result.output_bytes)
    return bytes(payload).decode("utf-8")


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载公开 source closure 与 proof provider，不触碰 private 或长期 memory。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


@pytest.mark.parametrize(
    ("origin", "sentence", "first_form", "first_answer", "second_form", "second_answer"),
    (
        (
            "什么导致路面结冰？",
            "寒潮使得路面结冰。",
            "它的结果是什么？",
            "路面结冰",
            "它的原因是什么？",
            "寒潮",
        ),
        (
            "什么导致河水上涨？",
            "暴雨使得河水上涨。",
            "它的结果是什么？",
            "河水上涨",
            "它的原因是什么？",
            "暴雨",
        ),
    ),
)
@pytest.mark.parametrize("line_suffix", ("", "\n", "\r\n"))
def test_v3_focus_ledger_runs_two_domains_and_terminal_framings(
        runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        origin: str,
        sentence: str,
        first_form: str,
        first_answer: str,
        second_form: str,
        second_answer: str,
        line_suffix: str,
        ) -> None:
    """provider -> focus -> focus 必须仅以 V3 visible tail 连续推进。"""
    state = start_public_mixed_focus_dialogue(
        (65213, len(origin), len(line_suffix) + 1))
    first = run_public_mixed_focus_dialogue_turn_v1(
        state,
        _raw(origin + line_suffix),
        runtime,
    )
    assert _output(first) == sentence
    assert first.v4_turn is not None
    assert first.focus_context_append is not None
    assert first.focus_context_append.accepted
    assert isinstance(first.after.focus_context.turns[-1], ProviderOriginContextTurnV3)
    assert first.after.focus_context.revision == 1
    assert first.after.mixed_state.context.revision == 1

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("已匹配 continuous focus form 不得进入 V4 fallback")

    monkeypatch.setattr(
        conversation_raw_mixed_focus_dialogue_session,
        "run_public_mixed_frame_dialogue_turn_v4",
        forbidden,
    )
    second = run_public_mixed_focus_dialogue_turn_v1(
        first.after,
        _raw(first_form + line_suffix),
        runtime,
    )
    assert _output(second) == first_answer
    assert second.v4_turn is None
    assert second.provider_focus_followup_answer is not None
    assert second.focus_context_append is not None
    assert second.focus_context_append.accepted
    assert second.after.mixed_state.context == first.after.mixed_state.context
    assert isinstance(
        second.after.focus_context.turns[-1], ProviderOriginFollowupFocusTurnV1)

    third = run_public_mixed_focus_dialogue_turn_v1(
        second.after,
        _raw(second_form + line_suffix),
        runtime,
    )
    assert _output(third) == second_answer
    assert third.v4_turn is None
    assert third.provider_focus_followup_answer is not None
    assert third.focus_context_append is not None
    assert third.focus_context_append.accepted
    assert third.after.mixed_state.context == first.after.mixed_state.context
    assert third.after.focus_context.revision == 3
    assert tuple(type(item) for item in third.after.focus_context.turns) == (
        ProviderOriginContextTurnV3,
        ProviderOriginFollowupFocusTurnV1,
        ProviderOriginFollowupFocusTurnV1,
    )


def test_replay_is_a_b_deterministic_at_outer_state_boundary(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """相同输入序列的两个独立 state 必须逐 turn 得到相同整数 record。"""
    inputs = (
        "什么导致路面结冰？",
        "它的结果是什么？",
        "它的原因是什么？",
    )
    left = start_public_mixed_focus_dialogue((65213, 71, 1))
    right = start_public_mixed_focus_dialogue((65213, 71, 1))
    for text in inputs:
        left_turn = run_public_mixed_focus_dialogue_turn_v1(
            left, _raw(text), runtime)
        right_turn = run_public_mixed_focus_dialogue_turn_v1(
            right, _raw(text), runtime)
        assert left_turn.canonical_record() == right_turn.canonical_record()
        assert (left_turn.after.canonical_record()
                == right_turn.after.canonical_record())
        left = left_turn.after
        right = right_turn.after


def test_frame_parent_blocks_focus_and_mismatched_projection_is_rejected(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """Frame 尾轮不能穿透到旧 provider，且 V4/V3 mapping 不可手工错配。"""
    first = run_public_mixed_focus_dialogue_turn_v1(
        start_public_mixed_focus_dialogue((65213, 72, 1)),
        _raw("什么导致路面结冰？"),
        runtime,
    )
    frame = run_public_mixed_focus_dialogue_turn_v1(
        first.after,
        _raw("北川站东门何时启用？"),
        runtime,
    )
    assert frame.answer is not None and frame.answer.accepted
    assert frame.focus_context_append is not None
    assert frame.focus_context_append.accepted
    assert isinstance(frame.after.focus_context.turns[-1], FrameQuestionAnswerTurnV3)

    rejected = run_public_mixed_focus_dialogue_turn_v1(
        frame.after,
        _raw("它的结果是什么？"),
        runtime,
    )
    result = rejected.provider_followup_answer
    assert rejected.v4_turn is not None
    assert result is not None and not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_CONTEXT
    assert rejected.focus_context_append is None
    assert (rejected.after.focus_context.canonical_record()
            == frame.after.focus_context.canonical_record())

    with pytest.raises(ConversationRawMixedFocusDialogueSessionError):
        ConversationRawMixedFocusDialogueStateV1(
            first.after.mixed_state,
            frame.after.focus_context,
            first.after.next_operation_ordinal,
        )


def test_bad_focus_admission_is_runtime_reject_without_any_state_write(
        runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """已得 candidate 但 V3 admission 失效时，不能泄露答案或半写入 ledger。"""
    first = run_public_mixed_focus_dialogue_turn_v1(
        start_public_mixed_focus_dialogue((65213, 73, 1)),
        _raw("什么导致路面结冰？"),
        runtime,
    )

    def broken_admission(*_args: object, **_kwargs: object) -> object:
        raise ProviderOriginFocusContextError("专项模拟 admission failure")

    monkeypatch.setattr(
        conversation_raw_mixed_focus_dialogue_session,
        "provider_origin_focus_admission_from_followup_result_v1",
        broken_admission,
    )
    rejected = run_public_mixed_focus_dialogue_turn_v1(
        first.after,
        _raw("它的结果是什么？"),
        runtime,
    )
    result = rejected.provider_focus_followup_answer
    assert result is not None and not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_RUNTIME
    assert result.output_u8 == ()
    assert rejected.focus_context_append is None
    assert rejected.after.mixed_state.context == first.after.mixed_state.context
    assert (rejected.after.focus_context.canonical_record()
            == first.after.focus_context.canonical_record())
