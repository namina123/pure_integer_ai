"""T1-G0 raw observation 到显式 scalar/span unit 的窄验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservationError,
    RawTextSpanUnit,
    compile_raw_text_observation_json,
    compile_raw_text_observation,
    load_raw_text_observation_jsonl,
    parse_raw_text_observation_record,
    raw_text_observation_to_json_object,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    parse_canonical_json_bytes,
)


_SAMPLE = Path(__file__).resolve().parents[1] / (
    "data/ph2/dlg_raw_text_observation_v1.jsonl.sample")


def _causal_units() -> tuple[RawTextSpanUnit, ...]:
    return (
        RawTextSpanUnit("u1", "cause", 0, 2, 0, 6),
        RawTextSpanUnit("u2", "relation", 2, 4, 6, 12),
        RawTextSpanUnit("u3", "effect", 4, 8, 12, 24),
    )


def test_raw_observation_compiles_explicit_scalar_and_byte_spans() -> None:
    observation = compile_raw_text_observation(
        tuple("暴雨导致河水上涨。".encode("utf-8")),
        observation_id="obs-causal-01", source_id="src-causal-01",
        context_id="ctx-causal-01", family_id="fam-causal-01",
        source_namespace="t1-g0-cc0-v1", split="train", units=_causal_units(),
    )
    assert observation.unit_scalars(observation.units[0]) == tuple(map(ord, "暴雨"))
    assert observation.unit_bytes(observation.units[1]) == tuple("导致".encode("utf-8"))
    assert observation.unit_scalars(observation.units[2]) == tuple(map(ord, "河水上涨"))
    assert all(type(item) is int and item >= 0 for item in observation.canonical_record())


def test_unseen_text_can_reuse_explicit_roles_without_training_surface_lookup() -> None:
    observation = compile_raw_text_observation(
        tuple("台风导致港口封闭。".encode("utf-8")),
        observation_id="obs-causal-heldout-01", source_id="src-causal-heldout-01",
        context_id="ctx-causal-heldout-01", family_id="fam-causal-heldout-01",
        source_namespace="t1-g0-cc0-v1", split="heldout",
        units=_causal_units(),
    )
    assert observation.unit_scalars(observation.units[0]) == tuple(map(ord, "台风"))
    assert observation.unit_scalars(observation.units[2]) == tuple(map(ord, "港口封闭"))
    assert "暴雨" not in "".join(chr(item) for item in observation.scalars)


def test_bad_span_or_utf8_fails_closed_before_any_semantic_claim() -> None:
    with pytest.raises(RawTextObservationError, match="scalar/byte span"):
        compile_raw_text_observation(
            tuple("暴雨导致河水上涨。".encode("utf-8")),
            observation_id="obs-bad-span", source_id="src-bad-span",
            context_id="ctx-bad-span", family_id="fam-bad-span",
            source_namespace="t1-g0-cc0-v1", split="negative",
            units=(RawTextSpanUnit("u1", "cause", 0, 2, 0, 5),),
        )
    with pytest.raises(RawTextObservationError, match="严格 UTF-8"):
        compile_raw_text_observation(
            (0xE0, 0x80, 0x80),
            observation_id="obs-bad-utf8", source_id="src-bad-utf8",
            context_id="ctx-bad-utf8", family_id="fam-bad-utf8",
            source_namespace="t1-g0-cc0-v1", split="negative", units=(),
        )


def test_public_jsonl_round_trip_preserves_scalar_u8_and_integer_record() -> None:
    payload = _SAMPLE.read_bytes()
    observations = load_raw_text_observation_jsonl(payload)
    assert [item.split for item in observations] == ["train", "heldout", "negative"]
    for observation in observations:
        wire = raw_text_observation_to_json_object(observation)
        assert parse_raw_text_observation_record(wire).canonical_record() == (
            observation.canonical_record())
        assert compile_raw_text_observation_json(observation).rstrip(b"\n") == (
            next(line[:-1] for line in payload.splitlines(keepends=True)
                 if observation.observation_id.encode("utf-8") in line))
        parsed = parse_canonical_json_bytes(
            compile_raw_text_observation_json(observation)[:-1], require_object=True)
        assert tuple(parsed["raw_u8"]) == observation.raw_bytes
        assert tuple(parsed["scalars"]) == observation.scalars
        assert all(type(value) is int for value in observation.canonical_record())


def test_jsonl_split_boundary_rejects_mixed_pack() -> None:
    payload = _SAMPLE.read_bytes()
    with pytest.raises(RawTextObservationError, match="split"):
        load_raw_text_observation_jsonl(payload, expected_split="train")
    train_line = payload.splitlines(keepends=True)[0]
    assert len(load_raw_text_observation_jsonl(train_line, expected_split="train")) == 1
