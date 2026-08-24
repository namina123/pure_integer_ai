"""T1-G2 显式 proposition/relation evidence 的窄验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionEvidenceError,
    RawPropositionRelationEvidence,
    RawRelationArgument,
    bind_raw_proposition_relation,
    compile_raw_proposition_json,
    load_raw_proposition_jsonl,
    parse_raw_proposition_record,
    raw_proposition_to_json_object,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"
_LEX = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"
_PROP = _ROOT / "data/ph2/dlg_raw_proposition_relation_evidence_v1.jsonl.sample"


def _rows():
    return (
        load_raw_text_observation_jsonl(_OBS.read_bytes()),
        load_raw_lexical_evidence_jsonl(_LEX.read_bytes()),
        load_raw_proposition_jsonl(_PROP.read_bytes()),
    )


def test_train_and_heldout_relations_close_over_explicit_evidence() -> None:
    observations, lexical, propositions = _rows()
    results = []
    for observation, proposition in zip(observations[:2], propositions[:2]):
        evidence = tuple(item for item in lexical
                         if item.observation_id == observation.observation_id)
        result = bind_raw_proposition_relation(observation, evidence, proposition)
        results.append(result)
        assert len(result.arguments) == 3
        assert result.relation_kind == "annotated_relation"
        assert all(type(value) is int for value in result.canonical_record)
    assert results[0].arguments[0].unit_scalars != results[1].arguments[0].unit_scalars


def test_proposition_json_round_trip_is_canonical() -> None:
    _, _, propositions = _rows()
    payload = _PROP.read_bytes().splitlines(keepends=True)
    for item, line in zip(propositions, payload):
        assert parse_raw_proposition_record(raw_proposition_to_json_object(item)) == item
        assert compile_raw_proposition_json(item) == line
        assert parse_canonical_json_bytes(line[:-1], require_object=True) == (
            raw_proposition_to_json_object(item))


def test_unknown_evidence_or_identity_drift_fails_closed() -> None:
    observations, lexical, propositions = _rows()
    negative = propositions[2]
    with pytest.raises(RawPropositionEvidenceError, match="未知 evidence"):
        bind_raw_proposition_relation(
            observations[2],
            tuple(item for item in lexical if item.observation_id == observations[2].observation_id),
            negative,
        )
    changed = RawPropositionRelationEvidence(
        negative.proposition_id, negative.observation_id, negative.source_id,
        negative.context_id, negative.family_id, negative.source_namespace,
        "heldout", negative.relation_kind, negative.authority, negative.arguments,
    )
    with pytest.raises(RawPropositionEvidenceError, match="identity"):
        bind_raw_proposition_relation(observations[2], (), changed)


def test_argument_order_and_unit_identity_are_frozen() -> None:
    with pytest.raises(RawPropositionEvidenceError, match="连续"):
        RawPropositionRelationEvidence(
            "p", "o", "s", "c", "f", "ns", "train", "r", "a",
            (RawRelationArgument("e1", "u1", "left", 1),
             RawRelationArgument("e2", "u2", "right", 3)),
        )
    with pytest.raises(RawPropositionEvidenceError, match="不得重复"):
        RawPropositionRelationEvidence(
            "p", "o", "s", "c", "f", "ns", "train", "r", "a",
            (RawRelationArgument("e1", "u1", "left", 1),
             RawRelationArgument("e1", "u2", "right", 2)),
        )


def test_proposition_split_boundary_isolated() -> None:
    _, _, propositions = _rows()
    with pytest.raises(RawPropositionEvidenceError, match="split"):
        load_raw_proposition_jsonl(_PROP.read_bytes(), expected_split="train")
    assert len(load_raw_proposition_jsonl(
        _PROP.read_bytes().splitlines(keepends=True)[0], expected_split="train")) == 1
