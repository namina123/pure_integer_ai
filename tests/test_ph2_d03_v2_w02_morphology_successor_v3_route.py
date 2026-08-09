"""Public synthetic tests for source-capability morphology routing."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    blind_private_source_specs,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    run_w02_morphology_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    run_w02_morphology_successor_v2_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySuccessorV3RouteError,
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    StableRecordKey,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(split: str, ordinal: int, surface: str):
    return _source_record(
        "UD_ZH_GSDSIMP_R2_18", split, ordinal,
        snapshot_id="ud-zh-gsdsimp-route-train",
        revision_id="train-revision",
        official_url="https://github.com/UniversalDependencies/UD_Chinese-GSDSimp",
        source_identity=f"route-train:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{split}:{ordinal}"),
        license_id="CC-BY-SA-4.0",
        attribution="public synthetic route fixture",
        locator_kind="record", locator_value=str(ordinal),
        span_end=len(surface))


def _expected(form: str, lemma: str, upos: str) -> dict[str, object]:
    return {
        "boundary_spans": [{"end": len(form), "form": form, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [], "form": form, "lemma": lemma,
            "node_id": [1], "upos": upos,
        }],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _training_pairs():
    rows = (
        ("有", "有", "VERB"),
        ("没有", "有", "VERB"),
        ("未有", "有", "VERB"),
        ("猫化", "猫", "VERB"),
        ("纸化", "纸", "VERB"),
        ("新词", "新词", "NOUN"),
    )
    pairs = []
    for ordinal, (form, lemma, upos) in enumerate(rows, start=1):
        source = _source("train", ordinal, form)
        observation = _observation_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source,
            carrier_kind="plain_text", surface=form, family_ordinal=ordinal,
            sample_role="support", perturbation_kind="NONE")
        evidence = _owner_record(
            "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source, observation,
            _expected(form, lemma, upos),
            dimension_name=W02_DEV_DIMENSIONS[2])
        assert isinstance(evidence, TeacherEvidenceRecord)
        pairs.append((observation, evidence))
    return tuple(pairs)


def _artifacts(tmp_path: Path):
    candidate = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate", pairs=_training_pairs(),
        run_id=1, requested_workers=2, mode="fresh")
    v1 = run_w02_morphology_overlay_fixture(
        fixture_root=tmp_path / "v1",
        candidate_artifact_root=candidate.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    v2 = run_w02_morphology_successor_v2_overlay_fixture(
        fixture_root=tmp_path / "v2",
        candidate_artifact_root=candidate.artifact_path,
        v1_overlay_artifact_root=v1.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    return candidate.artifact_path, v1.artifact_path, v2.artifact_path


@pytest.fixture(scope="module")
def artifact_roots(tmp_path_factory: pytest.TempPathFactory):
    """Build the immutable parent chain once for all route assertions."""
    return _artifacts(tmp_path_factory.mktemp("w02-v3-route"))


def _new_route_observation():
    form = "新化"
    parent = _source("held_out", 1, form)
    dataset_key = StableRecordKey((9, 9, 7, 1))
    source_key = StableRecordKey((9, 9, 7, 10, 1))
    source = replace(
        parent,
        dataset_key=dataset_key,
        artifact_key=StableRecordKey((9, 9, 7, 2)),
        stable_key=source_key,
        source_cluster_key=StableRecordKey((9, 9, 7, 50, 1)),
        source_key="UD_ZH_PUBLIC_ROUTE_FIXTURE",
        snapshot_id="ud-zh-public-route-r1",
        revision_id="route-revision",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Chinese-PublicRouteFixture"),
        source_identity="public-route-fixture:sentence:1",
    )
    observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, parent,
        carrier_kind="plain_text", surface=form, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
    observation = replace(
        observation,
        dataset_key=dataset_key,
        artifact_key=source.artifact_key,
        stable_key=StableRecordKey((9, 9, 7, 20, 1)),
        source_ref_key=source_key,
    )
    capability = w02_ud_morphology_source_capability({
        "annotation_provenance": "public synthetic manual annotation",
        "commit_sha1": source.revision_id,
        "language": "zh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": source.source_key,
        "upstream_checksum": source.upstream_checksum,
    })
    return source, observation, capability


def test_v3_routes_new_ud_identity_without_mutating_frozen_parents(
        artifact_roots) -> None:
    candidate_root, v1_root, v2_root = artifact_roots
    v1_index = load_w02_morphology_overlay_index(v1_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
    source, observation, capability = _new_route_observation()
    routes = authorize_w02_morphology_source_routes((source,), (capability,))
    indexes = build_w02_morphology_routed_indexes(v1_index, v2_index, routes)
    parent_v1_identity = (v1_index.dataset_keys, v1_index.semantic_sha256)
    parent_v2_identity = (v2_index.dataset_keys, v2_index.semantic_sha256)

    with open_w02_candidate_predictor(candidate_root) as predictor:
        base = predictor.predict(observation)
    old_v1 = predict_w02_morphology_successor(
        v1_index, observation, base, requested_spans=((0, 2),))
    old_v2 = predict_w02_morphology_successor_v2(
        v2_index, observation, old_v1, requested_spans=((0, 2),))
    assert old_v1.generalized_candidate_count == 0
    assert old_v2.edge_candidate_count == 0

    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        routed = predict_w02_morphology_successor_v3(
            indexes, observation, base, requested_spans=((0, 2),),
            v1_cache=v1_cache, v2_cache=v2_cache)
    finally:
        v1_cache.close()
        v2_cache.close()
    assert routed.route_authorized == 1
    assert routed.v1.generalized_candidate_count > 0
    assert routed.v2.edge_candidate_count > 0
    assert any(
        row.form == "新化" and row.lemma == "新" and row.upos == "VERB"
        for row in routed.v2.prediction.morphology_candidates)
    assert (v1_index.dataset_keys, v1_index.semantic_sha256) == parent_v1_identity
    assert (v2_index.dataset_keys, v2_index.semantic_sha256) == parent_v2_identity


def test_v3_unlisted_source_fails_closed(artifact_roots) -> None:
    candidate_root, v1_root, v2_root = artifact_roots
    source, observation, capability = _new_route_observation()
    routes = authorize_w02_morphology_source_routes((source,), (capability,))
    indexes = build_w02_morphology_routed_indexes(
        load_w02_morphology_overlay_index(v1_root),
        load_w02_morphology_successor_v2_overlay_index(v2_root),
        routes)
    unlisted = replace(
        observation, source_ref_key=StableRecordKey((9, 9, 7, 10, 2)))
    with open_w02_candidate_predictor(candidate_root) as predictor:
        base = predictor.predict(unlisted)
    result = predict_w02_morphology_successor_v3(
        indexes, unlisted, base, requested_spans=((0, 2),))
    assert result.route_authorized == 0
    assert result.v1.generalized_candidate_count == 0
    assert result.v2.edge_candidate_count == 0


def test_v3_rejects_provenance_drift_and_duplicate_routes() -> None:
    source, _observation, capability = _new_route_observation()
    with pytest.raises(
            W02MorphologySuccessorV3RouteError, match="provenance"):
        authorize_w02_morphology_source_routes(
            (replace(source, revision_id="drifted"),), (capability,))
    with pytest.raises(
            W02MorphologySuccessorV3RouteError, match="duplicated"):
        authorize_w02_morphology_source_routes(
            (source, source), (capability,))


def test_v3_builds_capabilities_from_payload_free_public_ud_specs() -> None:
    capabilities = tuple(
        w02_ud_morphology_source_capability(spec)
        for spec in blind_private_source_specs())
    assert len(capabilities) == 2
    assert all(row.upstream_checksum.startswith("sha1:")
               for row in capabilities)
    assert len({row.sha256() for row in capabilities}) == 2
