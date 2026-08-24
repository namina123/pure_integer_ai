"""DLG-RAW-10 接入 RAW-04 主对话入口的定向验证。"""
from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from pure_integer_ai.experiments import conversation_raw_dialogue_session
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
    load_public_proof_sentence_provider_from_root,
    run_public_proof_sentence_provider_vector,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    run_public_frame_dialogue_turn,
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_RUNTIME,
)
from pure_integer_ai.experiments.run_public_frame_dialogue import (
    main,
    run_public_frame_dialogue_terminal,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provider_runtime():
    """主对话 runtime 只接收已冻结 provider binding，不读取 provider 路径。"""
    provider = load_public_proof_sentence_provider_from_root(_ROOT)
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=provider,
    )


def _route_bytes(runtime, route_kind: int) -> tuple[int, ...]:
    """从 provider 的已学习整数 route 获得一条真实 DLG-RAW-10 输入。"""
    provider = runtime.proof_sentence_provider
    assert provider is not None
    route = next(item for item in provider.legacy_catalog.routes
                 if item.route_kind == route_kind)
    return tuple(route.request.question_surface.encode("utf-8"))


def test_provider_answers_in_raw04_without_faking_a_frame_run(provider_runtime) -> None:
    """第二 provider 成功只携带 proof carrier，不生成 QuestionAnswerRun 或 context 写入。"""
    state = start_public_frame_dialogue((65001, 90, 1))
    provider = provider_runtime.proof_sentence_provider
    assert provider is not None

    for route_kind in (1, 2, 3):
        raw = _route_bytes(provider_runtime, route_kind)
        direct = run_public_proof_sentence_provider_vector(provider, raw)
        turn = run_public_frame_dialogue_turn(state, raw, provider_runtime)

        assert turn.answer is None
        assert turn.provider_answer is not None
        assert turn.provider_answer.accepted
        assert turn.provider_answer.route_kind == route_kind
        assert turn.provider_answer.output_scalars == direct.output_scalars
        assert turn.provider_answer.output_bytes == direct.output_bytes
        assert turn.provider_answer.source_record_key == direct.source_record_key
        assert turn.context_written == 0
        assert turn.after.context == state.context
        assert turn.after.next_operation_ordinal == state.next_operation_ordinal + 1
        assert turn.provider_answer.demo_record
        assert turn.provider_answer.output_bytes
        state = turn.after


def test_provider_miss_falls_back_to_existing_dlg_lexical_miss(provider_runtime) -> None:
    """provider 自身 miss 不应覆盖 RAW-06/09 的最终 lexical miss 语义。"""
    state = start_public_frame_dialogue((65001, 90, 2))
    turn = run_public_frame_dialogue_turn(
        state,
        tuple("未学习的公开问题？".encode("utf-8")),
        provider_runtime,
    )

    assert turn.provider_answer is None
    assert turn.answer is not None
    assert turn.answer.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert turn.context_written == 0
    assert turn.after.context == state.context


def test_existing_multi_target_ambiguity_precedes_provider(provider_runtime, monkeypatch) -> None:
    """RAW-09 的 `8` 必须在 provider dispatch 前结束，禁止二次猜测。"""
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("RAW-09 ambiguity 不得调用 DLG-RAW-10 provider")

    monkeypatch.setattr(
        conversation_raw_dialogue_session,
        "run_public_proof_sentence_provider_vector",
        forbidden,
    )
    state = start_public_frame_dialogue((65001, 90, 3))
    turn = run_public_frame_dialogue_turn(
        state,
        tuple("东岸入口何时启用？".encode("utf-8")),
        provider_runtime,
    )

    assert turn.provider_answer is None
    assert turn.answer is not None
    assert turn.answer.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert turn.context_written == 0


def test_main_terminal_routes_one_public_proof_sentence() -> None:
    """用户可执行入口必须装载 provider，而非只通过测试直接调用 adapter。"""
    provider = load_public_proof_sentence_provider_from_root(_ROOT)
    route = next(item for item in provider.legacy_catalog.routes
                 if item.route_kind == 2)
    direct = run_public_proof_sentence_provider_vector(
        provider,
        tuple(route.request.question_surface.encode("utf-8")),
    )
    output = BytesIO()

    assert main(
        [],
        stdin=BytesIO(route.request.question_surface.encode("utf-8") + b"\n:quit\n"),
        stdout=output,
    ) == 0

    assert direct.accepted
    assert output.getvalue() == (
        b"\xe4\xbd\xa0> \xe7\xb3\xbb\xe7\xbb\x9f> "
        + bytes(direct.output_bytes) + b"\n\xe4\xbd\xa0> "
    )


def test_terminal_rejects_forged_provider_carrier_without_crashing(
        provider_runtime,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """provider replay 漂移必须归一为 11，terminal 不得访问不存在的 frame result_code。"""
    provider = provider_runtime.proof_sentence_provider
    assert provider is not None
    raw = _route_bytes(provider_runtime, 2)
    actual = run_public_proof_sentence_provider_vector(provider, raw)
    forged = replace(
        actual,
        output_scalars=(65,),
        output_bytes=(65,),
    )
    monkeypatch.setattr(
        conversation_raw_dialogue_session,
        "run_public_proof_sentence_provider_vector",
        lambda *_args, **_kwargs: forged,
    )
    target = BytesIO()

    run_public_frame_dialogue_terminal(
        provider_runtime,
        BytesIO(bytes(raw) + b"\n:quit\n"),
        target,
        prompts=False,
    )

    assert target.getvalue() == b"\xe7\xb3\xbb\xe7\xbb\x9f> [REJECT:11]\n"
    state = start_public_frame_dialogue((65001, 90, 9))
    turn = run_public_frame_dialogue_turn(state, raw, provider_runtime)
    assert turn.provider_answer is not None
    assert turn.provider_answer.provider_status == (
        PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT)
    assert turn.provider_answer.mapped_dlg_result_code == DLG_RAW_REJECT_RUNTIME
    assert turn.context_written == 0
    assert turn.after.context == state.context
