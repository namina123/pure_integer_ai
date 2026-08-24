"""T1-G15：训练 pack census 的确定性与整数投影边界。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_t1_training_census import (
    RawT1TrainingCensusError,
    build_raw_t1_training_census,
)
from pure_integer_ai.experiments.conversation_raw_t1_training_pack import (
    RawT1TrainingPackError,
    assemble_raw_t1_training_pack,
)


_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data/ph2"


def _pack():
    return assemble_raw_t1_training_pack(*tuple(
        (_DATA / name).read_bytes()
        for name in (
            "dlg_raw_text_observation_v1.jsonl.sample",
            "dlg_raw_lexical_evidence_v1.jsonl.sample",
            "dlg_raw_proposition_relation_evidence_v1.jsonl.sample",
            "dlg_raw_proposition_qualification_v1.jsonl.sample",
        )))


def test_census_freezes_counts_and_replay_identity() -> None:
    first = build_raw_t1_training_census(_pack())
    second = build_raw_t1_training_census(_pack())

    assert first.case_count == 2
    assert first.negative_count == 1
    assert first.lexical_evidence_count == 9
    assert first.split_counts == (("train", 1), ("heldout", 1))
    assert first.negative_stage_counts == ((1, 1), (2, 0))
    assert first.canonical_sha256_u8 == second.canonical_sha256_u8
    assert all(type(value) is int and value >= 0
               for value in first.canonical_integer_record)


def test_census_is_value_only_and_rejects_wrong_input() -> None:
    with pytest.raises(TypeError, match="RawT1TrainingPack"):
        build_raw_t1_training_census(object())
    census = build_raw_t1_training_census(_pack())
    with pytest.raises(RawT1TrainingCensusError, match="32 个 u8"):
        type(census)(census.source_namespace, census.license_id,
                     census.case_count, census.negative_count,
                     census.lexical_evidence_count, census.split_counts,
                     census.negative_stage_counts, (1,))
