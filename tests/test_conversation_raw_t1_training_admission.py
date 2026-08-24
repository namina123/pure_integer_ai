"""T1-G20：G19 物理闸门到 G14 pack 的 admission 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_training_admission import (
    ADMISSION_ACCEPTED,
    RawT1TrainingAdmissionError,
    admit_raw_t1_training_pack,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data/ph2"


def _payloads() -> tuple[bytes, bytes, bytes, bytes]:
    return tuple((
        _DATA / name).read_bytes()
        for name in (
            "dlg_raw_text_observation_v1.jsonl.sample",
            "dlg_raw_lexical_evidence_v1.jsonl.sample",
            "dlg_raw_proposition_relation_evidence_v1.jsonl.sample",
            "dlg_raw_proposition_qualification_v1.jsonl.sample",
        ))


def test_admission_runs_g19_before_g14_and_preserves_witnesses() -> None:
    admission = admit_raw_t1_training_pack(*_payloads())

    assert admission.status == ADMISSION_ACCEPTED
    assert admission.pack.split_counts == (("train", 1), ("heldout", 1))
    assert tuple(item.status for item in admission.audits) == (1, 1, 2)
    assert admission.canonical_record() == admit_raw_t1_training_pack(
        *_payloads()).canonical_record()
    assert all(type(value) is int and value >= 0
               for value in admission.canonical_record())


def test_missing_lexical_evidence_cannot_bypass_admission() -> None:
    observation, lexical, proposition, qualification = _payloads()
    lexical_without_heldout = b"".join(
        line for line in lexical.splitlines(keepends=True)
        if b"obs-g0-heldout-01" not in line)

    with pytest.raises(RawT1TrainingAdmissionError, match="candidate admission"):
        admit_raw_t1_training_pack(
            observation, lexical_without_heldout, proposition, qualification)
