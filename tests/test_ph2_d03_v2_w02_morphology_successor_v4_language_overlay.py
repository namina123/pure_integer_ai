"""Bounded tests for the append-only language-conditioned morphology overlay."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CandidatePrediction,
    W02CarrierGeneration,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologySuccessorPrediction,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Prediction,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySuccessorV3Prediction,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_language_overlay import (
    W02_MORPH_SUCCESSOR_V4_VERSION,
    W02MorphologySuccessorV4Error,
    build_w02_morphology_successor_v4_from_counts,
    predict_w02_morphology_successor_v4,
    rank_w02_morphology_successor_v4,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_artifact import (
    W02MorphologySuccessorV4ArtifactError,
    publish_w02_morphology_successor_v4_artifact,
    read_w02_morphology_successor_v4_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_public_probe import (
    W02MorphologySuccessorV4PublicTraining,
)


def _index():
    return build_w02_morphology_successor_v4_from_counts((
        ("lzh", "學", "學", "VERB", "{}", 9),
        ("lzh", "學", "學", "NOUN", "{}", 3),
        ("lzh", "道", "道", "NOUN", "{}", 8),
        ("lzh", "行", "行", "VERB", '{"VerbForm":"Conv"}', 5),
    ))


def _observation(surface: str):
    source = _source_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1,
        snapshot_id="v4-public-fixture",
        revision_id="v4-public-fixture-revision",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-PublicFixture"),
        source_identity="v4-public-fixture:1",
        upstream_checksum="sha256:" + "1" * 64,
        local_sha256="2" * 64,
        license_id="CC-BY-SA-4.0",
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value="1",
        span_end=len(surface),
    )
    original = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, source,
        carrier_kind="plain_text", surface=surface, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
    return replace(original, language="lzh")


def _v3(observation, *, route_authorized: int = 1):
    surface = observation.typed_payload.to_value()["language_payload"]["surface"]
    base = W02CandidatePrediction(
        observation.stable_key.components,
        "PREDICTED",
        W02CarrierGeneration(
            "GENERATED", "plain_text", surface, surface, 0, len(surface)),
        tuple(range(len(surface) + 1)),
        (),
        (),
        (),
    )
    v1 = W02MorphologySuccessorPrediction(base, 0, 0, "a" * 64)
    v2 = W02MorphologySuccessorV2Prediction(base, 0, 0, "b" * 64)
    return W02MorphologySuccessorV3Prediction(
        v1, v2, route_authorized, 1, "c" * 64)


def test_v4_exact_ranking_is_language_scoped_and_bounded() -> None:
    index = _index()
    ranking = rank_w02_morphology_successor_v4(index, "lzh", "學")

    assert W02_MORPH_SUCCESSOR_V4_VERSION.endswith("V1")
    assert ranking.evidence_mode == "EXACT_LEXEME"
    assert tuple(row[:3] for row in ranking.candidates) == (
        ("學", "VERB", "{}"),
        ("學", "NOUN", "{}"),
    )
    assert not rank_w02_morphology_successor_v4(
        index, "zh", "學").candidates


def test_v4_unseen_form_uses_only_bounded_language_backoff() -> None:
    ranking = rank_w02_morphology_successor_v4(_index(), "lzh", "新")

    assert ranking.evidence_mode == "LANGUAGE_BACKOFF"
    assert 0 < len(ranking.candidates) <= 8
    assert all(row[0] == "新" for row in ranking.candidates)


def test_v4_prediction_extends_v3_without_mutating_parent() -> None:
    index = _index()
    observation = _observation("學道")
    parent = _v3(observation)
    before = parent.v2.prediction.to_dict()

    result = predict_w02_morphology_successor_v4(
        index, observation, parent, requested_spans=((0, 1), (1, 2)))

    predicted = {
        (row.form, row.lemma, row.upos, row.feats_json)
        for row in result.prediction.morphology_candidates
    }
    assert ("學", "學", "VERB", "{}") in predicted
    assert ("道", "道", "NOUN", "{}") in predicted
    assert result.exact_candidate_count == 3
    assert result.backoff_candidate_count == 0
    assert parent.v2.prediction.to_dict() == before


def test_v4_registered_language_requires_authorized_v3_route() -> None:
    observation = _observation("學")
    with pytest.raises(W02MorphologySuccessorV4Error,
                       match="authorized source route"):
        predict_w02_morphology_successor_v4(
            _index(), observation, _v3(observation, route_authorized=0),
            requested_spans=((0, 1),))


def test_v4_rejects_duplicate_aggregate_lexeme() -> None:
    row = ("lzh", "學", "學", "VERB", "{}", 1)
    with pytest.raises(W02MorphologySuccessorV4Error, match="duplicated"):
        build_w02_morphology_successor_v4_from_counts((row, row))


def test_v4_artifact_round_trip_and_transport_tamper_fail_closed(
        tmp_path: Path) -> None:
    index = _index()
    training = W02MorphologySuccessorV4PublicTraining(
        index, 3, index.training_token_count, 3, 4,
        ({"relative_path": "fixture.conllu", "sha256": "d" * 64,
          "size_bytes": 10},),
    )
    result = publish_w02_morphology_successor_v4_artifact(
        training, tmp_path / "artifact")
    clone = read_w02_morphology_successor_v4_artifact(result.artifact_path)

    assert clone.semantic_sha256 == index.semantic_sha256
    assert clone.index.semantic_rows() == index.semantic_rows()
    rows_path = result.artifact_path / (
        "morphology-v4-language-overlay.rows.jsonl.gz")
    rows_path.write_bytes(rows_path.read_bytes() + b"x")
    with pytest.raises(W02MorphologySuccessorV4ArtifactError,
                       match="transport identity"):
        read_w02_morphology_successor_v4_artifact(result.artifact_path)


def test_public_probe_module_does_not_reference_private_payload_paths() -> None:
    source = Path(
        "src/pure_integer_ai/experiments/"
        "ph2_d03_v2_w02_morphology_successor_v4_public_probe.py"
    ).read_text(encoding="utf-8")
    assert "lzh_kyoto-ud-test.conllu" not in source
    assert "private-owner-r5" not in source
    assert "private-formal-r5" not in source
