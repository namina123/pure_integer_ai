"""W-02 morphology successor V2 dev calibration 的公开合同测试。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_dev_calibration import (
    W02_MORPH_SUCCESSOR_V2_DEV_CODE_PATHS,
    W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS,
    W02MorphologySuccessorV2DevCalibrationError,
    _assert_v1_retained,
    _requested_spans,
    build_w02_morphology_successor_v2_dev_calibration_freeze,
)


def _evaluation(value: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        expected_payload=SimpleNamespace(to_value=lambda: value))


def _candidate(identity: int) -> SimpleNamespace:
    return SimpleNamespace(
        start=0, end=1, form="f", lemma=f"l{identity}", upos="NOUN",
        feats_json="[]", support_count=identity)


def test_successor_v2_dev_requested_spans_use_boundaries_only() -> None:
    evaluation = _evaluation({
        "boundary_spans": [
            {"end": 4, "form": "ignored", "start": 2},
            {"end": 2, "form": "ignored", "start": 0},
        ],
        "morphology": [{
            "feats": [], "lemma": "not-forwarded", "upos": "NOUN",
        }],
    })
    assert _requested_spans(evaluation) == ((0, 2), (2, 4))


def test_successor_v2_dev_rejects_dropped_v1_candidate() -> None:
    v1 = SimpleNamespace(
        prediction=SimpleNamespace(morphology_candidates=(_candidate(1),)))
    v2 = SimpleNamespace(
        prediction=SimpleNamespace(morphology_candidates=(_candidate(2),)),
        edge_candidate_count=0)
    with pytest.raises(
            W02MorphologySuccessorV2DevCalibrationError,
            match="V1 候选"):
        _assert_v1_retained(v1, v2, ((0, 1),))


def test_successor_v2_dev_rejects_per_span_candidate_overflow() -> None:
    v1 = SimpleNamespace(
        prediction=SimpleNamespace(morphology_candidates=()))
    v2 = SimpleNamespace(
        prediction=SimpleNamespace(
            morphology_candidates=tuple(_candidate(index)
                                        for index in range(1, 10))),
        edge_candidate_count=9)
    with pytest.raises(
            W02MorphologySuccessorV2DevCalibrationError,
            match="逐 span 上界"):
        _assert_v1_retained(v1, v2, ((0, 1),))


def test_successor_v2_dev_freeze_binds_three_artifacts() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_v2_dev_calibration_freeze(repository)
    assert value["status"] == (
        "W02_MORPHOLOGY_SUCCESSOR_V2_DEV_CALIBRATION_FREEZE_COMPLETE")
    assert value["formal_training_runs"] == 1
    assert value["formal_successor_transform_runs"] == 1
    assert value["formal_successor_v2_transform_runs"] == 1
    assert value["formal_dev_calibration_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["expected_counts"] == (
        W02_MORPH_SUCCESSOR_V2_DEV_EXPECTED_COUNTS)
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_SUCCESSOR_V2_DEV_CODE_PATHS)
    assert len(value["candidate_artifact_manifest_sha256"]) == 64
    assert len(value["v1_overlay_artifact_manifest_sha256"]) == 64
    assert len(value["v2_overlay_artifact_manifest_sha256"]) == 64
