"""DLG-RAW-13 outer response/state 的有界实际 runtime 回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
)
from pure_integer_ai.experiments.conversation_raw_terminal_dialogue_act import (
    TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1,
    TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1,
    TerminalDialogueActRuntimeV1,
    build_terminal_dialogue_act_runtime_v1,
    run_public_terminal_dialogue_act_turn_v1,
    start_public_terminal_dialogue_act,
    terminal_dialogue_response_schema_record_v1,
)


_ROOT = Path(__file__).resolve().parents[1]
_KEY = (65001, 34, 13)


@pytest.fixture(scope="module")
def runtime() -> TerminalDialogueActRuntimeV1:
    """从同一公开 closure 建立一次完整 provider + terminal-act runtime。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    inner = build_public_dialogue_runtime_v1(
        closure,
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(
            _ROOT),
    )
    return build_terminal_dialogue_act_runtime_v1(inner)


def _turn(state, text: str | bytes, runtime: TerminalDialogueActRuntimeV1):
    """以 raw UTF-8 bytes 调用真实 outer transition，不走宿主文本 ingress。"""
    raw = text if type(text) is bytes else text.encode("utf-8")
    return run_public_terminal_dialogue_act_turn_v1(state, tuple(raw), runtime)


def test_coverage_act_handles_body_lf_crlf_without_touching_inner_ledger(
        runtime: TerminalDialogueActRuntimeV1) -> None:
    """7 的三种 RAW framing 必须只产生 state-none meta-act，不写 V2/V3。"""
    state = start_public_terminal_dialogue_act(_KEY)
    focus_before = state.inner_state.focus_context.canonical_record()
    context_before = state.inner_state.mixed_state.context.canonical_record()
    for raw in ("这是什么？", "这是什么？\n", "这是什么？\r\n"):
        turn = _turn(state, raw, runtime)
        assert turn.response.response_kind == TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1
        assert turn.response.base_result_code == DLG_RAW_REJECT_LEXICAL_MISS
        assert bytes(turn.response.output_u8) == (
            "当前公开对话资料尚未覆盖此输入。".encode("utf-8"))
        assert turn.response.dialogue_state_effect == 0
        assert turn.after.inner_state.focus_context.canonical_record() == focus_before
        assert turn.after.inner_state.mixed_state.context.canonical_record() == context_before
        state = turn.after


def test_route_ambiguity_becomes_only_route_clarification_act(
        runtime: TerminalDialogueActRuntimeV1) -> None:
    """真实 8 不得偷选答案，也不得被称为事实层 CLARIFY。"""
    turn = _turn(start_public_terminal_dialogue_act(_KEY), "东岸入口何时启用？", runtime)

    assert turn.response.response_kind == TERMINAL_DIALOGUE_RESPONSE_KIND_META_ACT_V1
    assert turn.response.base_result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert bytes(turn.response.output_u8) == (
        "此输入对应多个已学习路径，请补充限定。".encode("utf-8"))
    assert turn.response.act_code == 2
    assert turn.response.dialogue_state_effect == 0


def test_existing_source_bound_three_turn_chain_is_passthrough(
        runtime: TerminalDialogueActRuntimeV1) -> None:
    """DLG-RAW-13 不得改写既有 provider/focus output 或连续 ledger。"""
    state = start_public_terminal_dialogue_act(_KEY)
    expected = (
        "寒潮使得路面结冰。",
        "路面结冰",
        "寒潮",
    )
    for raw, output in zip((
            "什么导致路面结冰？",
            "它的结果是什么？",
            "它的原因是什么？",
    ), expected):
        turn = _turn(state, raw, runtime)
        assert turn.response.response_kind == TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1
        assert bytes(turn.response.output_u8).decode("utf-8") == output
        state = turn.after
    assert len(state.inner_state.focus_context.turns) == 3


def test_catalog_memory_drift_falls_back_to_original_protocol_reject(
        runtime: TerminalDialogueActRuntimeV1) -> None:
    """内存中被改写的 act form 不能输出新文本，必须保持 base 7 reject。"""
    isolated = build_terminal_dialogue_act_runtime_v1(runtime.inner_runtime)
    form = isolated.catalog.forms[0]
    object.__setattr__(form, "output_u8", tuple(b"forged"))

    turn = _turn(start_public_terminal_dialogue_act(_KEY), "这是什么？", isolated)

    assert turn.response.response_kind == TERMINAL_DIALOGUE_RESPONSE_KIND_PASSTHROUGH_V1
    assert turn.response.base_result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert bytes(turn.response.output_u8) == b"[REJECT:7]"


def test_response_schema_has_closed_portable_tags() -> None:
    """renderer/port 不得通过 carrier class 或空 segment 推断 response 分型。"""
    assert terminal_dialogue_response_schema_record_v1() == (
        1, 1, 1, 2, 1, 2, 3, 0, 1, 2, 0)
