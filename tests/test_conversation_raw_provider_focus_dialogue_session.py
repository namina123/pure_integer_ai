"""DLG-RAW-12 V5 同锚点话语焦点链的定向验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    conversation_raw_provider_focus_dialogue_session,
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
from pure_integer_ai.experiments.conversation_raw_provider_focus_dialogue_session import (
    ConversationRawProviderFocusDialogueStateV1,
    run_public_provider_focus_dialogue_turn_v1,
    start_public_provider_focus_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以明确 UTF-8 u8 tuple 形成 terminal 等价输入。"""
    return tuple(text.encode("utf-8"))


def _output(turn) -> str:
    """读取三种公开 carrier 的唯一已验证输出，不重组回答文本。"""
    result = (turn.answer or turn.provider_answer
              or turn.provider_followup_answer)
    assert result is not None and result.accepted
    payload = (result.output_u8 if turn.provider_followup_answer is result
               else result.output_bytes)
    return bytes(payload).decode("utf-8")


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载完整公开 runtime，不读取 private、训练或长期 memory。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


@pytest.mark.parametrize(
    ("origin", "sentence", "cause", "effect"),
    (
        ("寒潮导致什么？", "寒潮使得路面结冰。", "寒潮", "路面结冰"),
        ("暴雨导致什么？", "暴雨使得河水上涨。", "暴雨", "河水上涨"),
    ),
)
@pytest.mark.parametrize("line_suffix", ("", "\n", "\r\n"))
def test_same_anchor_focus_chain_moves_between_verified_occurrences(
        runtime: PublicDialogueRuntimeV1,
        origin: str,
        sentence: str,
        cause: str,
        effect: str,
        line_suffix: str,
        ) -> None:
    """首句、原因追问、结果追问均只消费同一 provider tail。"""
    state = start_public_provider_focus_dialogue((65212, 1, len(origin) + len(line_suffix)))
    first = run_public_provider_focus_dialogue_turn_v1(
        state, _raw(origin + line_suffix), runtime)
    assert _output(first) == sentence
    assert first.after.focus is None

    second = run_public_provider_focus_dialogue_turn_v1(
        first.after, _raw("它的原因是什么？" + line_suffix), runtime)
    assert _output(second) == cause
    assert second.v4_turn is not None
    assert second.after.focus is not None
    assert second.after.context == first.after.context

    third = run_public_provider_focus_dialogue_turn_v1(
        second.after, _raw("它的结果是什么？" + line_suffix), runtime)
    assert _output(third) == effect
    assert third.v4_turn is None
    assert third.provider_focus_followup_answer is not None
    assert third.after.focus is not None
    assert third.after.context == second.after.context


def test_repeat_form_is_handled_without_v4_dispatch(
        runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """active focus 已匹配的同 form 必须拒绝，不能重新走 base-anchor reducer。"""
    first = run_public_provider_focus_dialogue_turn_v1(
        start_public_provider_focus_dialogue((65212, 2, 1)),
        _raw("寒潮导致什么？"),
        runtime,
    )
    second = run_public_provider_focus_dialogue_turn_v1(
        first.after, _raw("它的原因是什么？"), runtime)
    third = run_public_provider_focus_dialogue_turn_v1(
        second.after, _raw("它的结果是什么？"), runtime)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("focus form 不得进入 V4 Frame/provider dispatch")

    monkeypatch.setattr(
        conversation_raw_provider_focus_dialogue_session,
        "run_public_mixed_frame_dialogue_turn_v4",
        forbidden,
    )
    duplicate = run_public_provider_focus_dialogue_turn_v1(
        third.after, _raw("它的结果是什么？"), runtime)

    result = duplicate.provider_focus_followup_answer
    assert result is not None and not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_CONTEXT
    assert duplicate.v4_turn is None
    assert duplicate.after.focus == third.after.focus
    assert duplicate.after.context == third.after.context


def test_alternating_form_moves_focus_then_same_form_rejects(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """结果到原因的交替可恢复，连续同 form 不得在新焦点上重复回答。"""
    state = start_public_provider_focus_dialogue((65212, 3, 1))
    for text in ("寒潮导致什么？", "它的原因是什么？", "它的结果是什么？"):
        state = run_public_provider_focus_dialogue_turn_v1(
            state, _raw(text), runtime).after
    alternate = run_public_provider_focus_dialogue_turn_v1(
        state, _raw("它的原因是什么？"), runtime)
    assert _output(alternate) == "寒潮"
    duplicate = run_public_provider_focus_dialogue_turn_v1(
        alternate.after, _raw("它的原因是什么？"), runtime)
    result = duplicate.provider_focus_followup_answer
    assert result is not None and not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_CONTEXT


def test_successful_new_frame_or_provider_answer_clears_focus(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """焦点不能越过新的真实回答，即使老 V4 不会为其 append context。"""
    first = run_public_provider_focus_dialogue_turn_v1(
        start_public_provider_focus_dialogue((65212, 4, 1)),
        _raw("寒潮导致什么？"), runtime)
    focused = run_public_provider_focus_dialogue_turn_v1(
        first.after, _raw("它的原因是什么？"), runtime)
    assert focused.after.focus is not None

    frame = run_public_provider_focus_dialogue_turn_v1(
        focused.after, _raw("北川站东门何时启用？"), runtime)
    assert frame.answer is not None and frame.answer.accepted
    assert frame.after.focus is None

    focused_again = run_public_provider_focus_dialogue_turn_v1(
        first.after, _raw("它的原因是什么？"), runtime)
    provider = run_public_provider_focus_dialogue_turn_v1(
        focused_again.after, _raw("暴雨导致什么？"), runtime)
    assert provider.provider_answer is not None and provider.provider_answer.accepted
    assert provider.after.focus is None


def test_stale_focus_fails_closed_without_v4_dispatch(
        runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """tail replacement 使 focus 不可用，已学习 form 必须以 runtime reject 结束。"""
    first = run_public_provider_focus_dialogue_turn_v1(
        start_public_provider_focus_dialogue((65212, 5, 1)),
        _raw("寒潮导致什么？"), runtime)
    focused = run_public_provider_focus_dialogue_turn_v1(
        first.after, _raw("它的原因是什么？"), runtime)
    replacement = run_public_provider_focus_dialogue_turn_v1(
        focused.after, _raw("暴雨导致什么？"), runtime)
    stale = ConversationRawProviderFocusDialogueStateV1(
        replacement.after.mixed_state,
        focused.after.focus,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stale active form 不得 fallback 到 V4")

    monkeypatch.setattr(
        conversation_raw_provider_focus_dialogue_session,
        "run_public_mixed_frame_dialogue_turn_v4",
        forbidden,
    )
    rejected = run_public_provider_focus_dialogue_turn_v1(
        stale, _raw("它的结果是什么？"), runtime)
    result = rejected.provider_focus_followup_answer
    assert result is not None and not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_RUNTIME
    assert rejected.v4_turn is None
    assert rejected.after.focus == stale.focus
