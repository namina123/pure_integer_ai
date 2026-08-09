"""W-02 morphology successor V2 edge-edit 的公开合成专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
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
    W02MorphologySuccessorV2Error,
    build_w02_morphology_successor_v2_from_counts,
    derive_w02_morphology_successor_v2_from_candidate,
    predict_w02_morphology_successor_v2,
    w02_morphology_edge_rule,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    W02MorphologySuccessorV2OverlayBudget,
    W02MorphologySuccessorV2OverlayError,
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
    run_w02_morphology_successor_v2_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(split: str, ordinal: int, surface: str):
    return _source_record(
        "UD_ZH_GSDSIMP_R2_18", split, ordinal,
        snapshot_id=f"v2-public-{split}", revision_id="1",
        official_url="https://example.invalid/v2-public",
        source_identity=f"v2-public:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{split}:{ordinal}"),
        license_id="CC-BY-SA-4.0", attribution="public synthetic fixture",
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
        ("有", "有", "VERB"), ("是", "是", "AUX"),
        ("他", "他", "PRON"), ("没有", "有", "VERB"),
        ("不是", "是", "AUX"), ("他们", "他", "PRON"),
        ("见义勇为", "为", "VERB"),
        ("猫化", "猫化", "VERB"), ("纸化", "纸化", "VERB"),
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


def _fixture(tmp_path: Path):
    candidate = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate", pairs=_training_pairs(),
        run_id=1, requested_workers=2, mode="fresh")
    overlay = run_w02_morphology_overlay_fixture(
        fixture_root=tmp_path / "overlay",
        candidate_artifact_root=candidate.artifact_path,
        run_id=1, requested_workers=2, mode="fresh")
    return candidate, overlay


def _held_out():
    form = "未有"
    source = _source("held_out", 1, form)
    observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, source,
        carrier_kind="plain_text", surface=form, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT")
    evaluation = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, source, observation,
        _expected(form, "有", "VERB"),
        dimension_name=W02_DEV_DIMENSIONS[2])
    assert isinstance(evaluation, EvaluatorLabelRecord)
    return observation, evaluation


def test_edge_rule_is_bounded_and_train_structural() -> None:
    assert w02_morphology_edge_rule("没有", "有") == "DROP_PREFIX_1"
    assert w02_morphology_edge_rule("他们", "他") == "DROP_SUFFIX_1"
    assert w02_morphology_edge_rule("见义勇为", "为") == "DROP_PREFIX_3"
    assert w02_morphology_edge_rule("一分为二", "为") is None
    assert w02_morphology_edge_rule("猫", "猫") is None


def test_v2_preserves_v1_and_adds_unseen_edge_lemma(tmp_path: Path) -> None:
    candidate, overlay = _fixture(tmp_path)
    v1_index = load_w02_morphology_overlay_index(overlay.artifact_path)
    v2_index = derive_w02_morphology_successor_v2_from_candidate(
        candidate.artifact_path, v1_index)
    observation, _ = _held_out()
    v1_cache = W02MorphologyRankingCache.empty()
    v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        with open_w02_candidate_predictor(candidate.artifact_path) as predictor:
            base = predictor.predict(observation)
        v1 = predict_w02_morphology_successor(
            v1_index, observation, base, requested_spans=((0, 2),),
            ranking_cache=v1_cache)
        v2 = predict_w02_morphology_successor_v2(
            v2_index, observation, v1, requested_spans=((0, 2),),
            cache=v2_cache)
    finally:
        v1_cache.close()
        v2_cache.close()
    v1_rows = [item.to_dict() for item in v1.prediction.morphology_candidates]
    v2_rows = [item.to_dict() for item in v2.prediction.morphology_candidates]
    assert not any(row["start"] == 0 and row["end"] == 2
                   and row["form"] == "未有" and row["lemma"] == "有"
                   for row in v1_rows)
    assert any(row["form"] == "未有" and row["lemma"] == "有"
               and row["upos"] == "VERB" for row in v2_rows)
    assert {canonical_json_bytes(item.to_dict())
            for item in v1.prediction.morphology_candidates}.issubset(
        {canonical_json_bytes(item.to_dict())
         for item in v2.prediction.morphology_candidates})
    assert 0 < v2.edge_candidate_count <= 8


def test_v2_count_build_is_order_independent_and_reports_unsupported(
        tmp_path: Path) -> None:
    candidate, overlay = _fixture(tmp_path)
    v1_index = load_w02_morphology_overlay_index(overlay.artifact_path)
    rows = (
        ("没有", "有", "VERB", "[]", 10),
        ("他们", "他", "PRON", "[]", 5),
        ("一分为二", "为", "VERB", "[]", 1),
        ("猫", "猫", "NOUN", "[]", 20),
    )
    forward = build_w02_morphology_successor_v2_from_counts(
        v1_index=v1_index, lexeme_counts=rows)
    reverse = build_w02_morphology_successor_v2_from_counts(
        v1_index=v1_index, lexeme_counts=tuple(reversed(rows)))
    assert forward.semantic_sha256 == reverse.semantic_sha256
    assert forward.semantic_rows() == reverse.semantic_rows()
    assert forward.accepted_lexeme_rows == 2
    assert forward.accepted_support_count == 15
    assert forward.unsupported_lexeme_rows == 1
    assert forward.unsupported_support_count == 1


def test_v2_span_and_cache_resources_fail_closed(tmp_path: Path) -> None:
    candidate, overlay = _fixture(tmp_path)
    v1_index = load_w02_morphology_overlay_index(overlay.artifact_path)
    v2_index = derive_w02_morphology_successor_v2_from_candidate(
        candidate.artifact_path, v1_index)
    observation, _ = _held_out()
    with open_w02_candidate_predictor(candidate.artifact_path) as predictor:
        base = predictor.predict(observation)
    v1 = predict_w02_morphology_successor(
        v1_index, observation, base, requested_spans=((0, 2),))
    with pytest.raises(W02MorphologySuccessorV2Error, match="span"):
        predict_w02_morphology_successor_v2(
            v2_index, observation, v1,
            requested_spans=tuple((0, 2) for _ in range(513)))
    cache = W02MorphologySuccessorV2Cache.empty()
    predict_w02_morphology_successor_v2(
        v2_index, observation, v1, requested_spans=((0, 2),), cache=cache)
    assert cache.values
    cache.close()
    assert not cache.values


def test_v2_overlay_workers_and_readback_are_canonical(tmp_path: Path) -> None:
    candidate, overlay = _fixture(tmp_path / "parent")
    results = []
    for workers in (1, 2, 4):
        result = run_w02_morphology_successor_v2_overlay_fixture(
            fixture_root=tmp_path / f"v2-{workers}",
            candidate_artifact_root=candidate.artifact_path,
            v1_overlay_artifact_root=overlay.artifact_path,
            run_id=1, requested_workers=workers, mode="fresh")
        results.append(result)
        index = load_w02_morphology_successor_v2_overlay_index(
            result.artifact_path)
        assert index.semantic_sha256 == result.semantic_sha256
        assert index.accepted_lexeme_rows == 4
        before = tuple(
            (path.relative_to(result.artifact_path).as_posix(),
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(result.artifact_path.rglob("*")) if path.is_file())
        resumed = read_w02_morphology_successor_v2_overlay_artifact(
            result.artifact_path, requested_workers=workers)
        after = tuple(
            (path.relative_to(result.artifact_path).as_posix(),
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(result.artifact_path.rglob("*")) if path.is_file())
        assert resumed.semantic_sha256 == result.semantic_sha256
        assert after == before
    assert len({result.semantic_sha256 for result in results}) == 1
    assert len({result.artifact_manifest_sha256 for result in results}) == 1


def test_v2_overlay_restart_and_resource_stop(tmp_path: Path) -> None:
    candidate, overlay = _fixture(tmp_path / "parent")
    failed_root = tmp_path / "failed"
    with pytest.raises(W02MorphologySuccessorV2OverlayError, match="injected"):
        run_w02_morphology_successor_v2_overlay_fixture(
            fixture_root=failed_root,
            candidate_artifact_root=candidate.artifact_path,
            v1_overlay_artifact_root=overlay.artifact_path,
            run_id=1, requested_workers=4, mode="fresh", fault_after_shard=3)
    failed_delta = (
        failed_root / "morphology-v2-overlay-store" / ".run-000001.staging"
        / "shards" / "000.delta.jsonl.gz")
    original_delta = failed_delta.read_bytes()
    failed_delta.write_bytes(original_delta + b"corrupt")
    with pytest.raises(
            W02MorphologySuccessorV2OverlayError, match="shard transport"):
        run_w02_morphology_successor_v2_overlay_fixture(
            fixture_root=failed_root, candidate_artifact_root=None,
            v1_overlay_artifact_root=None, run_id=1, requested_workers=2,
            mode="restart")
    failed_delta.write_bytes(original_delta)
    restarted = run_w02_morphology_successor_v2_overlay_fixture(
        fixture_root=failed_root, candidate_artifact_root=None,
        v1_overlay_artifact_root=None, run_id=1, requested_workers=2,
        mode="restart")
    clean = run_w02_morphology_successor_v2_overlay_fixture(
        fixture_root=tmp_path / "clean",
        candidate_artifact_root=candidate.artifact_path,
        v1_overlay_artifact_root=overlay.artifact_path,
        run_id=1, requested_workers=1, mode="fresh")
    assert restarted.semantic_sha256 == clean.semantic_sha256
    assert restarted.artifact_manifest_sha256 == clean.artifact_manifest_sha256

    stopped = tmp_path / "stopped"
    with pytest.raises(W02MorphologySuccessorV2OverlayError, match="resource stop"):
        run_w02_morphology_successor_v2_overlay_fixture(
            fixture_root=stopped,
            candidate_artifact_root=candidate.artifact_path,
            v1_overlay_artifact_root=overlay.artifact_path,
            run_id=1, requested_workers=1, mode="fresh",
            budget=W02MorphologySuccessorV2OverlayBudget(max_rule_rows=1))
    assert not tuple(stopped.rglob("morphology-v2-overlay.artifact.json"))
