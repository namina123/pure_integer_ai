"""Short structural tests for the V3 source-route shadow probe."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_shadow_probe import (
    W02_MORPH_V3_DEV_DATASET_KEY,
    W02_MORPH_V3_PARENT_SOURCE_KEY,
    W02_MORPH_V3_TRAIN_DATASET_KEY,
    _remap_observation,
    _remap_source,
    _shadow_route_specs,
    read_w02_morphology_successor_v3_shadow_freeze,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_shadow_route_specs_separate_train_and_dev_blobs() -> None:
    repository = Path(__file__).resolve().parents[1]
    train, dev = _shadow_route_specs(repository)
    assert train.split == "train"
    assert dev.split == "dev"
    assert train.dataset_key == W02_MORPH_V3_TRAIN_DATASET_KEY
    assert dev.dataset_key == W02_MORPH_V3_DEV_DATASET_KEY
    assert train.source_key != dev.source_key
    assert train.upstream_checksum != dev.upstream_checksum


def test_shadow_remap_changes_only_identity() -> None:
    repository = Path(__file__).resolve().parents[1]
    spec = _shadow_route_specs(repository)[0]
    source = _source_record(
        W02_MORPH_V3_PARENT_SOURCE_KEY, "train", 1,
        snapshot_id="ud-zh-gsdsimp-r2.18",
        revision_id=spec.capability.revision_id,
        official_url=spec.capability.official_url,
        source_identity="train:train-s1",
        upstream_checksum=spec.upstream_checksum,
        local_sha256=_sha("train:train-s1"),
        license_id="CC-BY-SA-4.0", attribution="public fixture",
        locator_kind="record", locator_value="1", span_end=2)
    observation = _observation_record(
        W02_MORPH_V3_PARENT_SOURCE_KEY, "train", 1, source,
        carrier_kind="plain_text", surface="新词", family_ordinal=1,
        sample_role="read_only_probe",
        perturbation_kind="HELD_OUT_DOCUMENT")
    mapped_source = _remap_source(source, 1, spec)
    mapped_observation = _remap_observation(
        observation, mapped_source, 1, spec)
    assert mapped_source.dataset_key == spec.dataset_key
    assert mapped_observation.dataset_key == spec.dataset_key
    assert mapped_observation.source_ref_key == mapped_source.stable_key
    assert mapped_observation.typed_payload == observation.typed_payload
    assert mapped_observation.language == observation.language
    assert mapped_observation.representation == observation.representation


def test_v3_shadow_freeze_matches_live_code() -> None:
    repository = Path(__file__).resolve().parents[1]
    freeze = read_w02_morphology_successor_v3_shadow_freeze(repository)
    assert freeze["status"] == (
        "W02_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE_COMPLETE")
    assert freeze["expected_counts"]["routed_observation_count"] == 4_497
    assert freeze["formal_private_evaluation_runs"] == 0
