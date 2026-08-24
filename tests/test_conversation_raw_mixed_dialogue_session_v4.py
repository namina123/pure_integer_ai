"""DLG-RAW-11C V4 mixed session 的真实 follow-up 与零写边界。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import conversation_raw_mixed_dialogue_session
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
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    run_public_mixed_frame_dialogue_turn_v4,
    start_public_mixed_frame_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以明确 UTF-8 bytes 形成 terminal 等价输入。"""
    return tuple(text.encode("utf-8"))


@pytest.fixture(scope="module")
def runtime() -> PublicDialogueRuntimeV1:
    """加载完整公开 runtime，其中 V3 base binding 仍不含 follow-up grammar。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


def test_provider_followup_is_a_real_no_write_transition(
        runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """匹配后不调用 V3 provider/Frame path，只复用最后一个 provider anchor。"""
    state = start_public_mixed_frame_dialogue((65111, 4, 1))
    first = run_public_mixed_frame_dialogue_turn_v4(
        state,
        _raw("寒潮导致什么？"),
        runtime,
    )
    assert first.provider_answer is not None and first.provider_answer.accepted
    assert first.after.context.revision == 1

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("已匹配 follow-up 不得重入 V3 Frame/provider dispatch")

    monkeypatch.setattr(
        conversation_raw_mixed_dialogue_session,
        "run_public_mixed_frame_dialogue_turn",
        forbidden,
    )
    followup = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw("它的原因是什么？"),
        runtime,
    )

    assert followup.answer is None
    assert followup.provider_answer is None
    assert followup.provider_followup_answer is not None
    assert followup.provider_followup_answer.accepted
    assert bytes(followup.provider_followup_answer.output_u8).decode("utf-8") == "寒潮"
    assert followup.context_append is None
    assert followup.context_write_origin == 0
    assert followup.after.context == first.after.context
    assert followup.after.next_operation_ordinal == first.after.next_operation_ordinal + 1


@pytest.mark.parametrize("line_suffix", ("\n", "\r\n"))
def test_provider_followup_preserves_registered_terminal_line_framing(
        runtime: PublicDialogueRuntimeV1,
        line_suffix: str,
        ) -> None:
    """LF/CRLF 都有独立来源 profile，不能通过宿主文本归一化偷过 identity。"""
    state = start_public_mixed_frame_dialogue((65111, 4, 10))
    first = run_public_mixed_frame_dialogue_turn_v4(
        state,
        _raw("寒潮导致什么？" + line_suffix),
        runtime,
    )
    followup = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw("它的原因是什么？" + line_suffix),
        runtime,
    )

    assert followup.provider_followup_answer is not None
    assert followup.provider_followup_answer.accepted
    assert bytes(followup.provider_followup_answer.output_u8).decode("utf-8") == "寒潮"


@pytest.mark.parametrize(
    ("origin_surface", "expected"),
    (
        ("什么导致路面结冰？", "路面结冰"),
        ("什么导致河水上涨？", "河水上涨"),
    ),
)
@pytest.mark.parametrize("line_suffix", ("", "\n", "\r\n"))
def test_provider_result_followup_reuses_reverse_focus_occurrence(
        runtime: PublicDialogueRuntimeV1,
        origin_surface: str,
        expected: str,
        line_suffix: str,
        ) -> None:
    """11D 只能从原因焦点 anchor 的已验证结果 occurrence 切出回答。"""
    state = start_public_mixed_frame_dialogue((65111, 4, 11))
    first = run_public_mixed_frame_dialogue_turn_v4(
        state,
        _raw(origin_surface + line_suffix),
        runtime,
    )
    followup = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw("它的结果是什么？" + line_suffix),
        runtime,
    )

    assert first.provider_answer is not None and first.provider_answer.accepted
    assert followup.provider_followup_answer is not None
    assert followup.provider_followup_answer.accepted
    assert bytes(followup.provider_followup_answer.output_u8).decode("utf-8") == expected
    assert followup.context_append is None
    assert followup.after.context == first.after.context


@pytest.mark.parametrize(
    ("origin_surface", "followup_surface"),
    (
        ("寒潮导致什么？", "它的结果是什么？"),
        ("什么导致路面结冰？", "它的原因是什么？"),
    ),
)
def test_followup_form_does_not_reverse_or_self_bind_without_profile(
        runtime: PublicDialogueRuntimeV1,
        origin_surface: str,
        followup_surface: str,
        ) -> None:
    """同句、同 source 或角色文字不能替代课程明确的 origin-focus binding。"""
    first = run_public_mixed_frame_dialogue_turn_v4(
        start_public_mixed_frame_dialogue((65111, 4, 12)),
        _raw(origin_surface),
        runtime,
    )
    followup = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw(followup_surface),
        runtime,
    )

    assert followup.provider_followup_answer is not None
    assert (followup.provider_followup_answer.mapped_dlg_result_code
            == DLG_RAW_REJECT_CONTEXT)
    assert followup.after.context == first.after.context


def test_provider_tail_with_wrong_focus_is_context_miss(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """同一完整命题的不同焦点不能作为 follow-up 的结构候选。"""
    first = run_public_mixed_frame_dialogue_turn_v4(
        start_public_mixed_frame_dialogue((65111, 4, 2)),
        _raw("什么导致路面结冰？"),
        runtime,
    )
    assert first.provider_answer is not None and first.provider_answer.accepted
    followup = run_public_mixed_frame_dialogue_turn_v4(
        first.after,
        _raw("它的原因是什么？"),
        runtime,
    )

    assert followup.provider_followup_answer is not None
    assert not followup.provider_followup_answer.accepted
    assert (followup.provider_followup_answer.mapped_dlg_result_code
            == DLG_RAW_REJECT_CONTEXT)
    assert followup.after.context == first.after.context


def test_frame_tail_blocks_provider_followup_without_cross_turn_lookup(
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """provider 后若写入一个 Frame turn，follow-up 不得越过最后一轮继续引用 provider。"""
    state = start_public_mixed_frame_dialogue((65111, 4, 3))
    provider = run_public_mixed_frame_dialogue_turn_v4(
        state,
        _raw("寒潮导致什么？"),
        runtime,
    )
    frame = run_public_mixed_frame_dialogue_turn_v4(
        provider.after,
        _raw("北川站东门何时启用？"),
        runtime,
    )
    assert frame.answer is not None and frame.answer.accepted
    assert frame.after.context.revision == 2

    followup = run_public_mixed_frame_dialogue_turn_v4(
        frame.after,
        _raw("它的原因是什么？"),
        runtime,
    )

    assert followup.provider_followup_answer is not None
    assert (followup.provider_followup_answer.mapped_dlg_result_code
            == DLG_RAW_REJECT_CONTEXT)
    assert followup.after.context == frame.after.context
