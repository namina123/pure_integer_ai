"""Short structural tests for the V3 source-route dev probe."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_dev_probe import (
    W02_MORPH_V3_DEV_ARTIFACT_KEY,
    W02_MORPH_V3_DEV_DATASET_KEY,
    W02_MORPH_V3_DEV_SOURCE_KEY,
    _remap_pair,
    _remap_source,
    read_w02_morphology_successor_v3_dev_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair():
    form = "新词"
    source = _source_record(
        "UD_ZH_GSDSIMP_R2_18", "dev", 1,
        snapshot_id="ud-zh-gsdsimp-r2.18", revision_id="revision",
        official_url=(
            "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
        source_identity="public-dev:1",
        upstream_checksum="sha1:" + "a" * 40,
        local_sha256=_sha("public-dev:1"),
        license_id="CC-BY-SA-4.0", attribution="public fixture",
        locator_kind="record", locator_value="1", span_end=len(form))
    observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "dev", 1, source,
        carrier_kind="plain_text", surface=form, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
    evaluation = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "dev", 1, source, observation, {
            "boundary_spans": [{"end": 2, "form": form, "start": 0}],
            "carrier_kind": "plain_text",
            "definitive_truth_authoritative": 0,
            "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
            "morphology": [{
                "feats": [], "form": form, "lemma": form,
                "node_id": [1], "upos": "NOUN",
            }],
            "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
        }, dimension_name=W02_DEV_DIMENSIONS[2])
    assert isinstance(evaluation, EvaluatorLabelRecord)
    return source, observation, evaluation


def test_dev_probe_remaps_only_identity_and_preserves_payload() -> None:
    source, observation, evaluation = _pair()
    mapped_source = _remap_source(source, 1)
    mapped_observation, mapped_evaluation = _remap_pair(
        observation, evaluation, mapped_source, 1)
    assert mapped_source.source_key == W02_MORPH_V3_DEV_SOURCE_KEY
    assert mapped_source.dataset_key == W02_MORPH_V3_DEV_DATASET_KEY
    assert mapped_observation.dataset_key == W02_MORPH_V3_DEV_DATASET_KEY
    assert mapped_observation.artifact_key == W02_MORPH_V3_DEV_ARTIFACT_KEY
    assert mapped_observation.source_ref_key == mapped_source.stable_key
    assert mapped_evaluation.observation_key == mapped_observation.stable_key
    assert mapped_observation.typed_payload == observation.typed_payload
    assert mapped_evaluation.expected_payload == evaluation.expected_payload


def test_dev_probe_remap_is_deterministic() -> None:
    source, observation, evaluation = _pair()
    first_source = _remap_source(source, 7)
    second_source = _remap_source(replace(source), 7)
    assert first_source == second_source
    assert (_remap_pair(observation, evaluation, first_source, 7)
            == _remap_pair(observation, evaluation, second_source, 7))


def test_v3_route_dev_freeze_matches_live_code() -> None:
    repository = Path(__file__).resolve().parents[1]
    freeze = read_w02_morphology_successor_v3_dev_freeze(repository)
    assert freeze["status"] == "W02_SUCCESSOR_V3_ROUTE_DEV_FREEZE_COMPLETE"
    assert freeze["expected_counts"]["observation_count"] == 500
    assert freeze["parent_failed_aggregate_sha256"].startswith("d61e836e")
