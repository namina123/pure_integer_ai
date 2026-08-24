"""DLG-RAW-14 来源绑定候选选择 outer dialogue 的有界实际回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
)
from pure_integer_ai.experiments.conversation_raw_route_clarification_dialogue import (
    ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1,
    ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1,
    ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1,
    ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1,
    ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1,
    ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1,
    RouteClarificationDialogueRuntimeV1,
    build_public_route_clarification_dialogue_runtime_v1,
    run_public_route_clarification_dialogue_turn_v1,
    start_public_route_clarification_dialogue,
)


_ROOT = Path(__file__).resolve().parents[1]
_KEY = (65001, 34, 14)
_AMBIGUOUS = "东岸入口何时启用？"
_OPTIONS_OUTPUT = (
    "此输入对应多个已学习路径，请重输其中一个完整问题：\n"
    "澄川码头何时启用？\n"
    "北川站东门何时启用？")


@pytest.fixture(scope="module")
def runtime() -> RouteClarificationDialogueRuntimeV1:
    """从同一公开 closure 建立一次完整 DLG-RAW-13/14 runtime。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    inner = build_public_dialogue_runtime_v1(
        closure,
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )
    return build_public_route_clarification_dialogue_runtime_v1(inner)


def _turn(state, text: str | bytes, runtime: RouteClarificationDialogueRuntimeV1):
    """以 raw UTF-8 bytes 驱动真实 outer transition，缓存不定义协议语义。"""
    raw = text if type(text) is bytes else text.encode("utf-8")
    return run_public_route_clarification_dialogue_turn_v1(
        state,
        tuple(raw),
        runtime,
        preparation_cache=PublicCoursePreparationCache(),
        preflight_cache=AliasRelationPreflightCache(),
    )


def test_real_ambiguity_opens_one_source_bound_complete_question_offer(
        runtime: RouteClarificationDialogueRuntimeV1) -> None:
    """真实 code-8 只能由 V3 candidate 重演后打开一轮、两项完整问句的 offer。"""
    turn = _turn(
        start_public_route_clarification_dialogue(_KEY),
        _AMBIGUOUS,
        runtime,
    )

    assert turn.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1)
    assert turn.response.state_effect == ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_OPEN_V1
    assert bytes(turn.response.output_u8).decode("utf-8") == _OPTIONS_OUTPUT
    assert turn.after.pending is not None
    assert turn.response.pending_identity_u8 == turn.after.pending.pending_identity_u8
    assert turn.response.output_readback.line_bodies_u8 == tuple(
        tuple(line.encode("utf-8")) for line in _OPTIONS_OUTPUT.split("\n"))
    assert not turn.after.selection_events


@pytest.mark.parametrize(("question", "answer"), (
    ("澄川码头何时启用？", "澄川码头于2023年启用。"),
    ("北川站东门何时启用？", "北川站东门于2024年启用。"),
))
def test_complete_reentry_selects_only_actual_matching_frame_answer(
        question: str,
        answer: str,
        runtime: RouteClarificationDialogueRuntimeV1) -> None:
    """两项完整重输各自重走 DLG-RAW-13 answer，并只追加一条 matching event。"""
    first = _turn(
        start_public_route_clarification_dialogue(_KEY),
        _AMBIGUOUS,
        runtime,
    )
    selected = _turn(first.after, question, runtime)

    assert selected.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1)
    assert selected.response.state_effect == (
        ROUTE_CLARIFICATION_STATE_EFFECT_SELECTION_APPEND_V1)
    assert selected.response.base_result_code == DLG_RAW_ACCEPT
    assert bytes(selected.response.output_u8).decode("utf-8") == answer
    assert selected.selection_event is not None
    assert selected.after.pending is None
    assert len(selected.after.selection_events) == 1
    assert selected.after.selection_events[0] == selected.selection_event


@pytest.mark.parametrize("suffix", ("", "\n", "\r\n"))
def test_complete_reentry_accepts_all_raw00_line_framings(
        suffix: str,
        runtime: RouteClarificationDialogueRuntimeV1) -> None:
    """body/LF/CRLF 的同一完整选项均可在唯一下一轮形成同一选择结果。"""
    first = _turn(
        start_public_route_clarification_dialogue(_KEY),
        _AMBIGUOUS + suffix,
        runtime,
    )
    selected = _turn(first.after, "澄川码头何时启用？" + suffix, runtime)

    assert first.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_ROUTE_OPTIONS_V1)
    assert selected.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_SELECTION_ANSWER_V1)
    assert selected.selection_event is not None


def test_bare_or_late_input_expires_pending_without_selection(
        runtime: RouteClarificationDialogueRuntimeV1) -> None:
    """裸实体与隔轮完整问句都不能借用旧 offer，也不得写 selection ledger。"""
    first = _turn(
        start_public_route_clarification_dialogue(_KEY),
        _AMBIGUOUS,
        runtime,
    )
    bare = _turn(first.after, "澄川码头", runtime)
    late = _turn(bare.after, "澄川码头何时启用？", runtime)

    assert bare.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1)
    assert bare.response.state_effect == ROUTE_CLARIFICATION_STATE_EFFECT_PENDING_EXPIRE_V1
    assert bare.after.pending is None
    assert not bare.after.selection_events
    assert late.response.response_kind == (
        ROUTE_CLARIFICATION_RESPONSE_KIND_PASSTHROUGH_V1)
    assert late.response.base_result_code == DLG_RAW_ACCEPT
    assert late.selection_event is None
    assert not late.after.selection_events
