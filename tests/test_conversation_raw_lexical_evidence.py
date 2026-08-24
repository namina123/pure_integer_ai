"""T1-G1 显式 lexical/structural evidence 的窄验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
    RawLexicalEvidenceError,
    compile_raw_lexical_evidence_json,
    compile_raw_lexical_evidence_pack,
    load_raw_lexical_evidence_jsonl,
    parse_raw_lexical_evidence_record,
    raw_lexical_evidence_to_json_object,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_OBSERVATIONS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"


def _rows() -> tuple[tuple, tuple]:
    observations = load_raw_text_observation_jsonl(_OBSERVATIONS.read_bytes())
    evidence = load_raw_lexical_evidence_jsonl(_EVIDENCE.read_bytes())
    return observations, evidence


def test_public_evidence_binds_every_unit_without_surface_lookup() -> None:
    observations, evidence = _rows()
    bindings = compile_raw_lexical_evidence_pack(observations, evidence)
    assert len(bindings) == 9
    assert {item.observation_id for item in bindings} == {
        "obs-g0-train-01", "obs-g0-heldout-01", "obs-g0-negative-01",
    }
    assert all(item.unit_kind == "explicit_span" for item in bindings)
    assert all(type(value) is int for item in bindings
               for value in item.evidence_record)
    # heldout 的 scalar 只来自该 observation，不查询 train surface。
    heldout = tuple(item.unit_scalars for item in bindings
                    if item.observation_id == "obs-g0-heldout-01")
    assert heldout
    train = tuple(item.unit_scalars for item in bindings
                  if item.observation_id == "obs-g0-train-01")
    assert heldout[0] != train[0]


def test_evidence_json_round_trip_is_canonical_and_integer_bound() -> None:
    _, evidence = _rows()
    for item in evidence:
        wire = raw_lexical_evidence_to_json_object(item)
        assert parse_raw_lexical_evidence_record(wire) == item
        assert all(type(value) is int for value in item.canonical_record())
        parsed = parse_canonical_json_bytes(
            compile_raw_lexical_evidence_json(item)[:-1], require_object=True)
        assert parsed == wire


def test_span_or_identity_drift_fails_closed() -> None:
    observations, evidence = _rows()
    changed = RawLexicalEvidence(
        evidence[0].evidence_id, evidence[0].observation_id,
        evidence[0].source_id, evidence[0].context_id, evidence[0].family_id,
        evidence[0].source_namespace, evidence[0].split, evidence[0].unit_id,
        evidence[0].unit_kind, evidence[0].authority,
        evidence[0].start_scalar, evidence[0].end_scalar,
        evidence[0].start_byte + 1, evidence[0].end_byte,
    )
    with pytest.raises(RawLexicalEvidenceError, match="span"):
        compile_raw_lexical_evidence_pack(observations, (changed, *evidence[1:]))
    changed_identity = RawLexicalEvidence(
        evidence[0].evidence_id, evidence[0].observation_id,
        evidence[0].source_id, evidence[0].context_id, evidence[0].family_id,
        evidence[0].source_namespace, "heldout", evidence[0].unit_id,
        evidence[0].unit_kind, evidence[0].authority,
        evidence[0].start_scalar, evidence[0].end_scalar,
        evidence[0].start_byte, evidence[0].end_byte,
    )
    with pytest.raises(RawLexicalEvidenceError, match="identity"):
        compile_raw_lexical_evidence_pack(observations, (changed_identity, *evidence[1:]))


def test_pack_requires_exact_unit_coverage_and_split_isolation() -> None:
    observations, evidence = _rows()
    with pytest.raises(RawLexicalEvidenceError, match="覆盖"):
        compile_raw_lexical_evidence_pack(observations[:1], evidence[:2])
    with pytest.raises(RawLexicalEvidenceError, match="split"):
        compile_raw_lexical_evidence_pack(observations, evidence, expected_split="train")
    train_observation = observations[:1]
    train_evidence = evidence[:3]
    assert len(compile_raw_lexical_evidence_pack(
        train_observation, train_evidence, expected_split="train")) == 3
