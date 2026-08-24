"""DLG-RAW-11B mixed session 接入公开运行时的定向验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import conversation_raw_mixed_dialogue_session
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
    MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
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
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    run_public_mixed_frame_dialogue_turn,
    start_public_mixed_frame_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]


def _raw(text: str) -> tuple[int, ...]:
    """以明确 UTF-8 bytes 形成 terminal 等价 raw 输入。"""
    return tuple(text.encode("utf-8"))


@pytest.fixture(scope="module")
def provider_runtime() -> PublicDialogueRuntimeV1:
    """加载公开 closure 和已冻结 proof provider，不读取 private 或 K 盘。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )


def test_pure_frame_follow_up_remains_a_v2_frame_chain(
        provider_runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """没有 provider 插入时，原有 target-anchor 继续实际生成同一回答。"""
    state = start_public_mixed_frame_dialogue((65211, 1, 1))
    first = run_public_mixed_frame_dialogue_turn(
        state,
        _raw("北川站东侧入口何时启用？"),
        provider_runtime,
    )
    assert first.answer is not None and first.answer.accepted
    assert first.provider_answer is None
    assert first.context_write_origin == MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
    assert first.after.context.revision == 1
    assert first.after.context.turns[-1].turn_kind == (
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN)

    follow_up = run_public_mixed_frame_dialogue_turn(
        first.after,
        _raw("它是在什么时候启用的？"),
        provider_runtime,
    )
    assert follow_up.answer is not None and follow_up.answer.accepted
    assert follow_up.provider_answer is None
    assert follow_up.answer.output_bytes == first.answer.output_bytes
    assert (follow_up.context_write_origin
            == MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN)
    assert follow_up.after.context.revision == 2
    assert follow_up.after.context.turns[-1].turn_kind == (
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN)


def test_provider_tail_blocks_old_frame_target_anchor_without_provider_fallback(
        provider_runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """provider 后的旧省略输入必须为 10，不能越过 tagged tail 或重入 provider。"""
    state = start_public_mixed_frame_dialogue((65211, 1, 2))
    frame = run_public_mixed_frame_dialogue_turn(
        state,
        _raw("北川站东门何时启用？"),
        provider_runtime,
    )
    provider = run_public_mixed_frame_dialogue_turn(
        frame.after,
        _raw("寒潮导致什么？"),
        provider_runtime,
    )
    assert provider.answer is None
    assert provider.provider_answer is not None
    assert provider.provider_answer.accepted
    assert (provider.context_write_origin
            == MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)
    assert provider.after.context.revision == 2
    assert provider.after.context.turns[-1].turn_kind == (
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider tail 的 target-anchor 不得 fallback proof provider")

    monkeypatch.setattr(
        conversation_raw_mixed_dialogue_session,
        "run_public_proof_sentence_provider_vector_with_typed_proof",
        forbidden,
    )
    blocked = run_public_mixed_frame_dialogue_turn(
        provider.after,
        _raw("它是在什么时候启用的？"),
        provider_runtime,
    )
    assert blocked.answer is not None
    assert blocked.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert blocked.provider_answer is None
    assert blocked.context_write_origin == 0
    assert blocked.after.context == provider.after.context


def test_provider_answer_is_projected_without_frame_answer_runtime(
        provider_runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """真实 provider ANSWER 只走同次 proof 与 anchor admission，不伪造 Frame run。"""
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider ANSWER 不得调用 Frame answer runtime")

    monkeypatch.setattr(
        conversation_raw_mixed_dialogue_session,
        "run_public_frame_answer",
        forbidden,
    )
    turn = run_public_mixed_frame_dialogue_turn(
        start_public_mixed_frame_dialogue((65211, 1, 3)),
        _raw("寒潮导致什么？"),
        provider_runtime,
    )

    assert turn.answer is None
    assert turn.provider_answer is not None and turn.provider_answer.accepted
    assert turn.provider_anchor is not None and turn.provider_anchor.accepted
    assert (turn.context_write_origin
            == MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)
    assert turn.after.context.revision == 1
    assert turn.after.context.turns[-1].turn_kind == (
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION)
