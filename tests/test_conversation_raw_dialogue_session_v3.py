"""DLG-RAW-11B V3 runtime binding 与 mixed production session 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments import conversation_raw_dialogue_session_v3
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime_v3 import (
    PublicDialogueRuntimeV3Error,
    build_public_dialogue_runtime_v3,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONTEXT,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session_v3 import (
    run_public_frame_dialogue_turn_v3,
    start_public_frame_dialogue_v3,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_v3():
    """以完整 public closure/provider 构造唯一可运行的 V3 binding。"""
    legacy = build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(_ROOT),
    )
    return build_public_dialogue_runtime_v3(legacy)


def _frame_surface(runtime_v3, requirement: int) -> tuple[int, ...]:
    """从 runtime 的冻结 catalog 取得唯一 Frame 输入，不写样例捷径。"""
    frames = tuple(
        frame for frame in runtime_v3.legacy_runtime.base_catalog.frames
        if frame.context_requirement == requirement)
    assert len(frames) == 1
    return frames[0].surface_bytes


def _provider_surface(runtime_v3) -> tuple[int, ...]:
    """从 provider 实际 route 取一条 exact 输入。"""
    routes = tuple(
        route for route in runtime_v3.provider.legacy_catalog.routes
        if route.route_kind == PUBLIC_SENTENCE_DEMO_ROUTE_EXACT)
    assert routes
    return tuple(routes[0].request.question_surface.encode("utf-8"))


def test_v3_runtime_binding_locks_full_provider_and_codec(runtime_v3) -> None:
    """V3 binding 必须含 full provider、schema、admission 与 codec record，而非 identity 简写。"""
    binding = runtime_v3.binding_record()
    provider_record = runtime_v3.provider.canonical_record()
    codec_record = runtime_v3.snapshot_codec_revision

    assert provider_record in tuple(
        binding[index + 1:index + 1 + binding[index]]
        for index in range(len(binding) - 1)
        if binding[index] == len(provider_record))
    assert codec_record in tuple(
        binding[index + 1:index + 1 + binding[index]]
        for index in range(len(binding) - 1)
        if binding[index] == len(codec_record))
    assert len(runtime_v3.runtime_identity()) == 32

    with pytest.raises(PublicDialogueRuntimeV3Error, match="codec identity"):
        replace(runtime_v3, snapshot_codec_identity_u8=(0,) * 32)


def test_provider_answer_writes_only_tagged_origin_projection(runtime_v3) -> None:
    """真实 provider ANSWER 写 V2 provider turn，旧 Frame compatibility context 保持空。"""
    state = start_public_frame_dialogue_v3((65001, 111, 1))
    turn = run_public_frame_dialogue_turn_v3(
        state,
        _provider_surface(runtime_v3),
        runtime_v3,
    )

    assert turn.provider_answer is not None
    assert turn.provider_answer.accepted
    assert (turn.provider_answer.context_policy
            == PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1)
    assert turn.provider_anchor is not None and turn.provider_anchor.accepted
    assert turn.mixed_admission is not None and turn.mixed_admission.accepted
    assert turn.after.mixed_context.revision == 1
    assert turn.after.mixed_context.turns[-1].turn_kind == (
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION)
    assert turn.after.legacy_frame_context.revision == 0


def test_pure_frame_follow_up_remains_actual_and_replayable(runtime_v3) -> None:
    """无 provider 插入时，既有真实 Frame 二轮仍可经 append 后写入 V2 wrapper。"""
    initial = start_public_frame_dialogue_v3((65001, 111, 2))
    first = run_public_frame_dialogue_turn_v3(
        initial,
        _frame_surface(runtime_v3, PUBLIC_FRAME_CONTEXT_NONE),
        runtime_v3,
    )
    follow_up = run_public_frame_dialogue_turn_v3(
        first.after,
        _frame_surface(runtime_v3, PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR),
        runtime_v3,
    )

    assert first.frame_answer is not None and first.frame_answer.result_code == DLG_RAW_ACCEPT
    assert first.mixed_admission is not None and first.mixed_admission.accepted
    assert follow_up.frame_answer is not None
    assert follow_up.frame_answer.result_code == DLG_RAW_ACCEPT
    assert follow_up.mixed_admission is not None and follow_up.mixed_admission.accepted
    assert follow_up.after.legacy_frame_context.revision == 2
    assert follow_up.after.mixed_context.revision == 2
    assert tuple(turn.turn_kind for turn in follow_up.after.mixed_context.turns) == (
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
    )


def test_provider_tail_rejects_target_anchor_before_frame_runtime(
        runtime_v3,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """provider 后不得越过 tagged tail 回溯旧 Frame，也不得触发回答执行。"""
    initial = start_public_frame_dialogue_v3((65001, 111, 3))
    first = run_public_frame_dialogue_turn_v3(
        initial,
        _frame_surface(runtime_v3, PUBLIC_FRAME_CONTEXT_NONE),
        runtime_v3,
    )
    provider = run_public_frame_dialogue_turn_v3(
        first.after,
        _provider_surface(runtime_v3),
        runtime_v3,
    )
    assert provider.provider_anchor is not None and provider.provider_anchor.accepted

    def forbidden(*_args: object, **_kwargs: object) -> object:
        """TARGET_ANCHOR gate 后不允许触碰 RAW-02/G-01/G-03/G-04。"""
        raise AssertionError("provider tail 后不得调用 frame runtime")

    monkeypatch.setattr(
        conversation_raw_dialogue_session_v3,
        "run_public_frame_answer",
        forbidden,
    )
    rejected = run_public_frame_dialogue_turn_v3(
        provider.after,
        _frame_surface(runtime_v3, PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR),
        runtime_v3,
    )

    assert rejected.frame_answer is not None
    assert rejected.frame_answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert rejected.frame_answer.ingress.request is None
    assert rejected.mixed_admission is None
    assert rejected.after.mixed_context.canonical_record() == (
        provider.after.mixed_context.canonical_record())
    assert rejected.after.legacy_frame_context.stable_key() == (
        provider.after.legacy_frame_context.stable_key())
