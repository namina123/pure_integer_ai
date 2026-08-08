"""W-02 morphology successor 的公开合成归纳与资源专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    run_w02_candidate_fixture,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    W02MorphologyOverlayBudget,
    W02MorphologyOverlayError,
    derive_w02_morphology_successor_from_candidate,
    load_w02_morphology_overlay_index,
    run_w02_morphology_overlay_formal,
    run_w02_morphology_overlay_fixture,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_contract import (
    W02_MORPH_SUCCESSOR_GUARD_AVAILABLE,
    publish_w02_morphology_successor_guard,
    read_w02_morphology_successor_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02MorphologySuccessorPublicationError,
    build_w02_morphology_successor_receipt,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _owner_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION,
    W02MorphologyRankingCache,
    W02MorphologySuccessorError,
    learn_w02_morphology_successor,
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    TeacherEvidenceRecord,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(source_key: str, split: str, ordinal: int, surface: str):
    license_id = "CC0-1.0" if source_key == "AUTHORED_CC0" else "CC-BY-SA-4.0"
    return _source_record(
        source_key, split, ordinal,
        snapshot_id=f"fixture-{source_key.casefold()}-{split}",
        revision_id="1",
        official_url="https://example.invalid/public-fixture",
        source_identity=f"fixture:{source_key}:{split}:{ordinal}",
        upstream_checksum="sha256:" + _sha(f"upstream:{source_key}:{split}:{ordinal}"),
        local_sha256=_sha(f"local:{source_key}:{split}:{ordinal}"),
        license_id=license_id,
        attribution="public synthetic fixture",
        locator_kind="record",
        locator_value=str(ordinal),
        span_end=len(surface),
    )


def _ud_expected(form: str, upos: str) -> dict[str, object]:
    return {
        "boundary_spans": [{"end": len(form), "form": form, "start": 0}],
        "carrier_kind": "plain_text",
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [],
            "form": form,
            "lemma": form,
            "node_id": [1],
            "upos": upos,
        }],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _ud_pair(ordinal: int, form: str, upos: str):
    source = _source("UD_ZH_GSDSIMP_R2_18", "train", ordinal, form)
    observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source,
        carrier_kind="plain_text",
        surface=form,
        family_ordinal=ordinal,
        sample_role="support",
        perturbation_kind="NONE",
    )
    evidence = _owner_record(
        "UD_ZH_GSDSIMP_R2_18", "train", ordinal, source, observation,
        _ud_expected(form, upos),
        dimension_name="W-02-V2-NEW-CONTENT-MORPHOLOGY",
    )
    assert isinstance(evidence, TeacherEvidenceRecord)
    return observation, evidence


def _training_pairs():
    rows = (
        ("a0", "NOUN"),
        ("b1", "NOUN"),
        ("c2", "NOUN"),
        ("d3", "NOUN"),
        ("猫化", "VERB"),
        ("纸化", "VERB"),
        ("木化", "VERB"),
    )
    return tuple(
        _ud_pair(ordinal, form, upos)
        for ordinal, (form, upos) in enumerate(rows, start=1)
    )


def _dev_observation(source_key: str = "UD_ZH_GSDSIMP_R2_18"):
    form = "新化"
    source = _source(source_key, "dev", 1, form)
    observation = _observation_record(
        source_key, "dev", 1, source,
        carrier_kind="plain_text",
        surface=form,
        family_ordinal=1,
        sample_role="read_only_probe",
        perturbation_kind="HELD_OUT_DOCUMENT",
    )
    evaluation = _owner_record(
        source_key, "dev", 1, source, observation,
        _ud_expected(form, "VERB"),
        dimension_name="W-02-V2-NEW-CONTENT-MORPHOLOGY",
    )
    assert isinstance(evaluation, EvaluatorLabelRecord)
    return observation, evaluation


def _fixture(tmp_path: Path):
    pairs = _training_pairs()
    result = run_w02_candidate_fixture(
        fixture_root=tmp_path / "candidate",
        pairs=pairs,
        run_id=1,
        requested_workers=2,
        mode="fresh",
    )
    return pairs, result


def test_successor_learns_order_independent_train_only_index(tmp_path: Path) -> None:
    pairs, _ = _fixture(tmp_path)
    forward = learn_w02_morphology_successor(pairs)
    reverse = learn_w02_morphology_successor(tuple(reversed(pairs)))
    assert forward.semantic_sha256 == reverse.semantic_sha256
    assert forward.semantic_rows() == reverse.semantic_rows()
    assert forward.training_pair_count == len(pairs)
    assert forward.morphology_observation_count == len(pairs)
    assert forward.morphology_token_count == len(pairs)
    assert forward.logic_operations > 0


def test_successor_queries_span_lazily_and_preserves_base(tmp_path: Path) -> None:
    pairs, result = _fixture(tmp_path)
    index = learn_w02_morphology_successor(pairs)
    observation, _ = _dev_observation()
    cache = W02MorphologyRankingCache.empty()
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        base = predictor.predict(observation)
        lazy = predict_w02_morphology_successor(index, observation, base)
        queried = predict_w02_morphology_successor(
            index, observation, base,
            requested_spans=((0, 2),),
            candidate_limit=1,
            ranking_cache=cache,
        )
        repeated = predict_w02_morphology_successor(
            index, observation, base,
            requested_spans=((0, 2),),
            candidate_limit=1,
            ranking_cache=cache,
        )
    assert lazy.prediction.to_dict() == base.to_dict()
    assert lazy.generalized_candidate_count == 0
    assert any(
        item.start == 0 and item.end == 2 and item.form == "新化"
        and item.lemma == "新化" and item.upos == "VERB"
        for item in queried.prediction.morphology_candidates
    )
    assert queried.prediction.to_dict() == repeated.prediction.to_dict()
    assert cache.hit_count >= 1 and cache.miss_count == 1
    assert queried.generalized_candidate_count <= 1


def test_script_feature_ablation_breaks_held_out_top_one(tmp_path: Path) -> None:
    pairs, result = _fixture(tmp_path)
    index = learn_w02_morphology_successor(pairs)
    observation, _ = _dev_observation()
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        base = predictor.predict(observation)
        full = predict_w02_morphology_successor(
            index, observation, base,
            requested_spans=((0, 2),), candidate_limit=1)
        ablated = predict_w02_morphology_successor(
            index, observation, base,
            requested_spans=((0, 2),),
            enabled_features=("LENGTH_BUCKET",),
            candidate_limit=1,
        )
    assert any(item.upos == "VERB"
               for item in full.prediction.morphology_candidates)
    assert not any(item.upos == "VERB"
                   for item in ablated.prediction.morphology_candidates)


def test_successor_route_and_span_budget_fail_closed(tmp_path: Path) -> None:
    pairs, result = _fixture(tmp_path)
    index = learn_w02_morphology_successor(pairs)
    authored, _ = _dev_observation("AUTHORED_CC0")
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        base = predictor.predict(authored)
        routed = predict_w02_morphology_successor(
            index, authored, base, requested_spans=((0, 2),))
    assert routed.generalized_candidate_count == 0
    assert routed.prediction.to_dict() == base.to_dict()

    observation, _ = _dev_observation()
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        base = predictor.predict(observation)
    too_many = tuple((0, 2) for _ in range(
        W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION + 1))
    with pytest.raises(W02MorphologySuccessorError, match="requested spans"):
        predict_w02_morphology_successor(
            index, observation, base, requested_spans=too_many)
    with pytest.raises(W02MorphologySuccessorError, match="span 越界"):
        predict_w02_morphology_successor(
            index, observation, base, requested_spans=((0, 1),))


def test_ranking_cache_cleanup_is_explicit(tmp_path: Path) -> None:
    pairs, result = _fixture(tmp_path)
    index = learn_w02_morphology_successor(pairs)
    observation, _ = _dev_observation()
    cache = W02MorphologyRankingCache.empty()
    with open_w02_candidate_predictor(result.artifact_path) as predictor:
        base = predictor.predict(observation)
        predict_w02_morphology_successor(
            index, observation, base,
            requested_spans=((0, 2),), ranking_cache=cache)
    assert cache.values
    cache.close()
    assert not cache.values


def _tree_identity(root: Path):
    return tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*")) if path.is_file()
    )


def test_sealed_candidate_rebuilds_exact_train_pair_index(tmp_path: Path) -> None:
    pairs, candidate = _fixture(tmp_path)
    expected = learn_w02_morphology_successor(pairs)
    before = _tree_identity(candidate.artifact_path)
    rebuilt = derive_w02_morphology_successor_from_candidate(
        candidate.artifact_path)
    assert rebuilt.semantic_rows() == expected.semantic_rows()
    assert rebuilt.semantic_sha256 == expected.semantic_sha256
    assert rebuilt.training_pair_count == expected.training_pair_count
    assert rebuilt.logic_operations == expected.logic_operations
    assert _tree_identity(candidate.artifact_path) == before


def test_overlay_is_canonical_for_workers_and_readback(tmp_path: Path) -> None:
    pairs, candidate = _fixture(tmp_path / "parent")
    expected = learn_w02_morphology_successor(pairs)
    parent_before = _tree_identity(candidate.artifact_path)
    results = []
    for workers in (1, 2, 4):
        root = tmp_path / f"overlay-{workers}"
        result = run_w02_morphology_overlay_fixture(
            fixture_root=root,
            candidate_artifact_root=candidate.artifact_path,
            run_id=1,
            requested_workers=workers,
            mode="fresh",
        )
        results.append(result)
        index = load_w02_morphology_overlay_index(result.artifact_path)
        assert index.semantic_rows() == expected.semantic_rows()
        manifest = read_canonical_object(
            result.artifact_path / "morphology-overlay.artifact.json")
        assert manifest["formal_successor_transform_runs"] == 0
        assert manifest["formal_training_runs"] == 0
        assert manifest["candidate_writes"] == 0
        assert manifest["private_payload_reads"] == 0
        before = _tree_identity(result.artifact_path)
        resumed = run_w02_morphology_overlay_fixture(
            fixture_root=root,
            candidate_artifact_root=None,
            run_id=1,
            requested_workers=workers,
            mode="resume",
        )
        assert resumed.overlay_semantic_sha256 == result.overlay_semantic_sha256
        assert _tree_identity(result.artifact_path) == before
    assert len({item.overlay_semantic_sha256 for item in results}) == 1
    assert len({item.artifact_manifest_sha256 for item in results}) == 1
    assert _tree_identity(candidate.artifact_path) == parent_before
    with pytest.raises(W02MorphologySuccessorPublicationError, match="不是正式"):
        build_w02_morphology_successor_receipt(
            Path(__file__).resolve().parents[1], results[0].artifact_path)


def test_overlay_restart_matches_clean_semantics(tmp_path: Path) -> None:
    _, candidate = _fixture(tmp_path / "parent")
    failed_root = tmp_path / "failed"
    with pytest.raises(W02MorphologyOverlayError, match="injected overlay"):
        run_w02_morphology_overlay_fixture(
            fixture_root=failed_root,
            candidate_artifact_root=candidate.artifact_path,
            run_id=1,
            requested_workers=4,
            mode="fresh",
            fault_after_shard=3,
        )
    assert not (
        failed_root / "morphology-overlay-store" / "run-000001"
        / "morphology-overlay.artifact.json").exists()
    restarted = run_w02_morphology_overlay_fixture(
        fixture_root=failed_root,
        candidate_artifact_root=None,
        run_id=1,
        requested_workers=2,
        mode="restart",
    )
    clean = run_w02_morphology_overlay_fixture(
        fixture_root=tmp_path / "clean",
        candidate_artifact_root=candidate.artifact_path,
        run_id=1,
        requested_workers=1,
        mode="fresh",
    )
    assert restarted.overlay_semantic_sha256 == clean.overlay_semantic_sha256
    assert restarted.logic_operations == clean.logic_operations


def test_overlay_resource_stop_never_publishes_manifest(tmp_path: Path) -> None:
    _, candidate = _fixture(tmp_path / "parent")
    root = tmp_path / "stopped"
    with pytest.raises(W02MorphologyOverlayError, match="input row resource stop"):
        run_w02_morphology_overlay_fixture(
            fixture_root=root,
            candidate_artifact_root=candidate.artifact_path,
            run_id=1,
            requested_workers=1,
            mode="fresh",
            budget=W02MorphologyOverlayBudget(max_input_rows=1),
        )
    assert not tuple(root.rglob("morphology-overlay.artifact.json"))


def test_formal_overlay_rejects_fixture_parent_before_guard_consumption(
        tmp_path: Path) -> None:
    _, candidate = _fixture(tmp_path / "parent")
    repository = Path(__file__).resolve().parents[1]
    freeze = read_w02_morphology_successor_runtime_freeze(repository)
    formal_root = tmp_path / "formal-successor"
    publish_w02_morphology_successor_guard(formal_root, freeze)
    available = formal_root / Path(*W02_MORPH_SUCCESSOR_GUARD_AVAILABLE.split("/"))
    before = available.read_bytes()
    with pytest.raises(W02MorphologyOverlayError, match="不是正式零泄漏"):
        run_w02_morphology_overlay_formal(
            successor_root=formal_root,
            candidate_artifact_root=candidate.artifact_path,
            runtime_freeze_sha256=freeze.sha256(),
            expected_parent_manifest_sha256=candidate.artifact_manifest_sha256,
            expected_parent_semantic_sha256=candidate.candidate_semantic_sha256,
            expected_guard_sha256=freeze.first_run_guard_sha256,
            run_id=1,
            requested_workers=1,
            mode="fresh",
            budget=freeze.resource_budget,
        )
    assert available.read_bytes() == before
