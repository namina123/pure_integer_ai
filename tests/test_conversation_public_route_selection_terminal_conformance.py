"""DLG-RAW-14 默认终端 route-selection 的最小公开 conformance 轨迹。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from pure_integer_ai.experiments import run_public_frame_dialogue as runner
from pure_integer_ai.experiments.ph2_dataset_core import (
    parse_canonical_json_bytes,
)


_VECTOR = Path(__file__).parent / "fixtures" / (
    "dlg_raw_public_route_selection_terminal_v1_conformance.json")


def _load_vector() -> dict:
    """读取唯一公开 raw-u8 vector，并拒绝非规范 JSON。"""
    payload = _VECTOR.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    assert isinstance(value, dict)
    return value


def test_default_outer_runner_matches_route_selection_terminal_trace(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """一次真实 ``main`` 必须保留两种 response kind，并按 quit 正常退出。"""
    value = _load_vector()
    steps = value["steps"]
    assert value["schema"] == 1
    assert value["trace_kind"] == "route_selection_terminal_v1"
    assert value["terminal_input_u8"] == [
        item
        for step in steps
        for item in step["input_u8"]
    ] + value["quit_input_u8"]

    observed: list[tuple[int, int, tuple[int, ...]]] = []
    actual_turn = runner.run_public_route_clarification_dialogue_turn_v1

    def capture_turn(*args, **kwargs):
        turn = actual_turn(*args, **kwargs)
        observed.append((
            turn.response.response_kind,
            turn.response.state_effect,
            tuple(turn.response.output_u8),
        ))
        return turn

    # 只在边界收集真实 turn；transition 本身仍由默认 runner 调用。
    monkeypatch.setattr(
        runner,
        "run_public_route_clarification_dialogue_turn_v1",
        capture_turn,
    )
    source = BytesIO(bytes(value["terminal_input_u8"]))
    target = BytesIO()

    exit_code = runner.main([], stdin=source, stdout=target)

    assert exit_code == value["expected_exit_code"] == 0
    assert source.tell() == len(value["terminal_input_u8"])
    assert observed == [
        (
            step["response_kind"],
            step["state_effect"],
            tuple(step["output_u8"]),
        )
        for step in steps
    ]
    assert target.getvalue() == bytes(value["terminal_output_u8"])
