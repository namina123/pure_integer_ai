"""T1-G3 source/evidence qualification 与 response-act 消费边界验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    load_raw_lexical_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerError,
    RawPropositionQualification,
    compile_raw_qualification_json,
    consume_raw_proposition_relation,
    load_raw_qualification_jsonl,
    parse_raw_qualification_record,
    raw_qualification_to_json_object,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionEvidenceError,
    bind_raw_proposition_relation,
    load_raw_proposition_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    load_raw_text_observation_jsonl,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_OBS = _ROOT / "data/ph2/dlg_raw_text_observation_v1.jsonl.sample"
_LEX = _ROOT / "data/ph2/dlg_raw_lexical_evidence_v1.jsonl.sample"
_PROP = _ROOT / "data/ph2/dlg_raw_proposition_relation_evidence_v1.jsonl.sample"
_QUAL = _ROOT / "data/ph2/dlg_raw_proposition_qualification_v1.jsonl.sample"


def _rows():
    return (
        load_raw_text_observation_jsonl(_OBS.read_bytes()),
        load_raw_lexical_evidence_jsonl(_LEX.read_bytes()),
        load_raw_proposition_jsonl(_PROP.read_bytes()),
        load_raw_qualification_jsonl(_QUAL.read_bytes()),
    )


def _bind(index: int):
    observations, lexical, propositions, qualifications = _rows()
    observation = observations[index]
    evidence = tuple(item for item in lexical
                     if item.observation_id == observation.observation_id)
    binding = bind_raw_proposition_relation(observation, evidence, propositions[index])
    return binding, qualifications[index]


def test_supported_unknown_and_conflict_map_to_distinct_response_obligations() -> None:
    supported = consume_raw_proposition_relation(*_bind(0))
    unknown = consume_raw_proposition_relation(*_bind(1))
    assert (supported.state, supported.response_act) == ("SUPPORTED", "ANSWER")
    assert (unknown.state, unknown.response_act) == ("UNKNOWN", "UNKNOWN")
    binding, qualification = _bind(0)
    conflict = RawPropositionQualification(
        "q-conflict", qualification.proposition_id, "obs-g0-train-01",
        "src-g0-train-01", "ctx-g0-train-01", "fam-g0-train-01",
        "t1-g0-public-v1", "train", "CONFLICT", "conflict-v1",
        tuple(item.evidence_id for item in binding.arguments),
        "public-authored-qualification-v1",
    )
    result = consume_raw_proposition_relation(binding, conflict)
    assert (result.state, result.response_act) == ("CONFLICT", "CLARIFY")
    assert not hasattr(result, "surface")


def test_qualification_json_round_trip_is_canonical() -> None:
    _, _, _, qualifications = _rows()
    payload = _QUAL.read_bytes().splitlines(keepends=True)
    for item, line in zip(qualifications, payload):
        assert parse_raw_qualification_record(raw_qualification_to_json_object(item)) == item
        assert compile_raw_qualification_json(item) == line
        assert parse_canonical_json_bytes(line[:-1], require_object=True) == (
            raw_qualification_to_json_object(item))


def test_qualification_evidence_chain_mismatch_fails_closed() -> None:
    binding, qualification = _bind(0)
    changed = RawPropositionQualification(
        qualification.qualification_id, qualification.proposition_id,
        qualification.observation_id, qualification.source_id,
        qualification.context_id, qualification.family_id,
        qualification.source_namespace, qualification.split, qualification.state,
        qualification.reason_id, ("e-other", *qualification.evidence_ids[1:]),
        qualification.authority,
    )
    with pytest.raises(RawPropositionConsumerError, match="evidence 链"):
        consume_raw_proposition_relation(binding, changed)


def test_conflict_requires_multiple_evidence_and_split_isolation() -> None:
    with pytest.raises(RawPropositionConsumerError, match="两个"):
        RawPropositionQualification(
            "q", "p", "o", "s", "c", "f", "ns", "train", "CONFLICT",
            "reason", ("e1",), "authority")
    with pytest.raises(RawPropositionConsumerError, match="split"):
        load_raw_qualification_jsonl(_QUAL.read_bytes(), expected_split="train")
    assert len(load_raw_qualification_jsonl(
        _QUAL.read_bytes().splitlines(keepends=True)[0], expected_split="train")) == 1


def test_negative_qualification_cannot_override_missing_relation_evidence() -> None:
    observations, lexical, propositions, qualifications = _rows()
    evidence = tuple(item for item in lexical
                     if item.observation_id == observations[2].observation_id)
    with pytest.raises(RawPropositionEvidenceError, match="未知 evidence"):
        binding = bind_raw_proposition_relation(observations[2], evidence, propositions[2])
        consume_raw_proposition_relation(binding, qualifications[2])
