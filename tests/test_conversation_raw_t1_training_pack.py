"""T1-G14：四个 raw constituent 装配为可消费训练 pack 的窄验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_training_pack import (
    RawT1TrainingPackError,
    assemble_raw_t1_training_pack,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservationError,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data/ph2"
_OBS = _DATA / "dlg_raw_text_observation_v1.jsonl.sample"
_LEX = _DATA / "dlg_raw_lexical_evidence_v1.jsonl.sample"
_PROP = _DATA / "dlg_raw_proposition_relation_evidence_v1.jsonl.sample"
_QUAL = _DATA / "dlg_raw_proposition_qualification_v1.jsonl.sample"


def _payloads() -> tuple[bytes, bytes, bytes, bytes]:
    return (_OBS.read_bytes(), _LEX.read_bytes(), _PROP.read_bytes(), _QUAL.read_bytes())


def _without_id(payload: bytes, observation_id: str) -> bytes:
    rows = [line for line in payload.splitlines(keepends=True)
            if observation_id.encode("utf-8") not in line]
    return b"".join(rows)


def test_train_heldout_and_negative_close_into_a_deterministic_pack() -> None:
    pack = assemble_raw_t1_training_pack(*_payloads())
    repeat = assemble_raw_t1_training_pack(*_payloads())

    assert pack.split_counts == (("train", 1), ("heldout", 1))
    assert tuple(item.observation_id for item in pack.cases) == (
        "obs-g0-heldout-01", "obs-g0-train-01")
    assert tuple(item.observation.observation_id for item in pack.negatives) == (
        "obs-g0-negative-01",)
    assert pack.canonical_record() == repeat.canonical_record()
    assert all(type(value) is int and value >= 0 for value in pack.canonical_record())


def test_negative_is_preserved_as_a_binding_witness() -> None:
    pack = assemble_raw_t1_training_pack(*_payloads())
    witness = pack.negatives[0]

    assert witness.failure_stage == 1
    assert witness.observation.observation_id == "obs-g0-negative-01"
    assert witness.proposition.proposition_id == "p-g0-negative-01"


def test_orphan_lexical_evidence_fails_closed_before_consumer() -> None:
    observation, lexical, proposition, qualification = _payloads()
    orphan = lexical.replace(
        b"obs-g0-train-01", b"obs-g0-orphan-01", 1)

    with pytest.raises(RawT1TrainingPackError, match="未知 observation"):
        assemble_raw_t1_training_pack(observation, orphan, proposition, qualification)


def test_missing_proposition_or_qualification_fails_closed() -> None:
    observation, lexical, proposition, qualification = _payloads()
    with pytest.raises(RawT1TrainingPackError, match="缺 proposition"):
        assemble_raw_t1_training_pack(
            observation,
            lexical,
            _without_id(proposition, "obs-g0-heldout-01"),
            qualification,
        )
    with pytest.raises(RawT1TrainingPackError, match="缺 proposition"):
        assemble_raw_t1_training_pack(
            observation,
            lexical,
            proposition,
            _without_id(qualification, "obs-g0-heldout-01"),
        )


def test_pack_requires_heldout_and_negative_witness() -> None:
    observation, lexical, proposition, qualification = _payloads()
    train_only = tuple(
        _without_id(payload, "obs-g0-heldout-01")
        for payload in (observation, lexical, proposition, qualification)
    )
    with pytest.raises(RawT1TrainingPackError, match="train 与 heldout"):
        assemble_raw_t1_training_pack(*train_only)

    no_negative = tuple(
        _without_id(payload, "obs-g0-negative-01")
        for payload in (observation, lexical, proposition, qualification)
    )
    with pytest.raises(RawT1TrainingPackError, match="negative witness"):
        assemble_raw_t1_training_pack(*no_negative)


def test_duplicate_observation_id_cannot_be_silently_overwritten() -> None:
    observation, lexical, proposition, qualification = _payloads()
    duplicated_observation = observation + observation.splitlines(keepends=True)[0]

    with pytest.raises((RawT1TrainingPackError, RawTextObservationError),
                       match="observation_id 不得重复"):
        assemble_raw_t1_training_pack(
            duplicated_observation, lexical, proposition, qualification)
