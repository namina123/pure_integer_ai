"""DLG-RAW-12 public focus-chain conformance fixture replay."""
from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    load_public_proof_sentence_provider_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_session import (
    run_public_mixed_focus_dialogue_turn_v1,
    start_public_mixed_focus_dialogue,
)
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_line


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests/fixtures/dlg_raw_public_focus_chain_v1_conformance.json"


def _load_fixture() -> dict:
    payload = _FIXTURE.read_bytes()
    assert canonical_json_line(json.loads(payload.decode("utf-8"))) == payload
    value = json.loads(payload.decode("utf-8"))
    assert value["schema"] == 1
    assert value["trace_kind"] == "dlg_raw_public_focus_chain_v1"
    assert value["expected_exit_code"] == 0
    return value


def _u8(hex_value: str) -> tuple[int, ...]:
    raw = bytes.fromhex(hex_value)
    return tuple(raw)


def _output_u8(turn) -> tuple[int, ...]:
    carrier = turn.answer or turn.provider_answer or turn.provider_focus_followup_answer
    assert carrier is not None and carrier.accepted
    if turn.provider_focus_followup_answer is not None:
        return tuple(carrier.output_u8)
    return tuple(carrier.output_bytes)


def test_public_focus_chain_fixture_replays_real_outer_transitions() -> None:
    fixture = _load_fixture()
    runtime = build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_ROOT),
        proof_sentence_provider=load_public_proof_sentence_provider_from_root(_ROOT),
    )
    state = start_public_mixed_focus_dialogue((65213, 71, 1))
    records = []
    for expected in fixture["steps"]:
        turn = run_public_mixed_focus_dialogue_turn_v1(
            state, _u8(expected["input_u8_hex"]), runtime)
        response_kind = (
            "provider_answer" if turn.provider_answer is not None
            else "provider_focus_followup"
            if turn.provider_focus_followup_answer is not None
            else "frame")
        assert response_kind == expected["response_kind"]
        assert _output_u8(turn) == _u8(expected["output_u8_hex"])
        assert turn.context_write_origin == expected["context_write_origin"]
        assert turn.after.focus_context.revision == expected["focus_revision"]
        assert turn.after.mixed_state.context.revision == expected["mixed_revision"]
        records.append(turn.canonical_record())
        state = turn.after
    assert len(records) == len(fixture["steps"])
    assert state.next_operation_ordinal == 4
