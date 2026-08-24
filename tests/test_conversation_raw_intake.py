"""DLG-RAW-00 bytes/int reference corpus 与无副作用边界专项。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_MAX_INPUT_BYTES,
    DLG_RAW_RECORD_V1,
    DLG_RAW_REJECT_BOM,
    DLG_RAW_REJECT_INPUT_BUDGET,
    DLG_RAW_REJECT_LINE_FRAMING,
    UTF8_STRICT_V1,
    encode_utf8_v1,
    intake_raw_conversation_bytes,
    intake_raw_conversation_vector,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "dlg_raw_intake_v1.json"


def _vectors() -> tuple[dict[str, object], ...]:
    """读取仅含整数的公开 reference vectors；JSON 不参与生产语义。"""
    value = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise AssertionError("DLG-RAW fixture root 非法")
    return tuple(value)


@pytest.mark.parametrize("vector", _vectors(), ids=lambda item: item["name"])
def test_dlg_raw_reference_vectors(vector: dict[str, object]) -> None:
    """所有公开 bytes/int vector 必须得到固定 result/body/scalar record。"""
    raw = tuple(vector["raw"])
    result = intake_raw_conversation_vector(raw)

    assert result.result_code == vector["code"]
    assert result.canonical_body_bytes == tuple(vector["body"])
    assert result.unicode_scalars == tuple(vector["scalars"])
    assert result.typed_record == result.output_bytes == result.state_delta == ()
    assert result.canonical_record() == (
        DLG_RAW_RECORD_V1,
        vector["code"],
        UTF8_STRICT_V1,
        len(raw),
        *raw,
        len(tuple(vector["body"])),
        *tuple(vector["body"]),
        len(tuple(vector["scalars"])),
        *tuple(vector["scalars"]),
        0,
        0,
        0,
    )


def test_dlg_raw_error_priority_is_frozen() -> None:
    """预算、BOM、framing 的优先级不能随宿主 decoder 或异常路径漂移。"""
    too_long_bom = (0xEF, 0xBB, 0xBF) + (0x0A,) * DLG_RAW_MAX_INPUT_BYTES
    assert intake_raw_conversation_vector(too_long_bom).result_code == (
        DLG_RAW_REJECT_INPUT_BUDGET)
    assert intake_raw_conversation_vector((0xEF, 0xBB, 0xBF, 0x0D)).result_code == (
        DLG_RAW_REJECT_BOM)
    assert intake_raw_conversation_vector((0xC0, 0x0A, 0x80)).result_code == (
        DLG_RAW_REJECT_LINE_FRAMING)


def test_dlg_raw_bytes_adapter_is_only_a_copying_host_edge() -> None:
    """bytes adapter 与核心 vector 的结果必须逐整数一致。"""
    raw = "北川站的东门有启用记录吗？".encode("utf-8")
    from_bytes = intake_raw_conversation_bytes(raw)
    from_vector = intake_raw_conversation_vector(tuple(raw))
    assert from_bytes == from_vector
    assert from_bytes.accepted is True
    assert from_bytes.result_code == DLG_RAW_ACCEPT
    with pytest.raises(TypeError):
        intake_raw_conversation_bytes(bytearray(raw))


def test_dlg_raw_utf8_encoder_round_trips_a_non_bmp_scalar_without_host_codec() -> None:
    """输出侧必须可由同一整数 UTF-8 规则编码并经输入规则还原。"""
    scalars = (0x4F60, 0x1F642, 0x597D)
    encoded = encode_utf8_v1(scalars)
    result = intake_raw_conversation_vector(encoded)
    assert result.accepted is True
    assert result.unicode_scalars == scalars


def test_dlg_raw_budget_rejection_does_not_decode_or_materialize_body() -> None:
    """超限输入只保留 physical trace，不能扫描、解码或形成部分语义。"""
    raw = (0xE4,) * (DLG_RAW_MAX_INPUT_BYTES + 1)
    result = intake_raw_conversation_vector(raw)
    assert result.result_code == DLG_RAW_REJECT_INPUT_BUDGET
    assert result.canonical_body_bytes == result.unicode_scalars == ()
    assert result.accepted is False
