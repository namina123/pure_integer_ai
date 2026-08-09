"""W-02 morphology successor dev calibration 的公开合同测试。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    W02_MORPH_SUCCESSOR_DEV_CODE_PATHS,
    W02_MORPH_SUCCESSOR_DEV_EXPECTED_COUNTS,
    W02MorphologySuccessorDevCalibrationError,
    _requested_spans,
    build_w02_morphology_successor_dev_calibration_freeze,
)


def _evaluation(value: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        expected_payload=SimpleNamespace(to_value=lambda: value))


def test_successor_dev_requested_spans_use_boundaries_only() -> None:
    evaluation = _evaluation({
        "boundary_spans": [
            {"end": 4, "form": "ignored", "start": 2},
            {"end": 2, "form": "ignored", "start": 0},
            {"end": 4, "form": "ignored", "start": 2},
        ],
        "morphology": [{
            "feats": [], "form": "ignored", "lemma": "not-forwarded",
            "upos": "NOUN",
        }],
    })
    assert _requested_spans(evaluation) == ((0, 2), (2, 4))


def test_successor_dev_requested_spans_reject_invalid_boundary() -> None:
    evaluation = _evaluation({
        "boundary_spans": [{"end": 0, "start": 0}],
        "morphology": [],
    })
    with pytest.raises(
            W02MorphologySuccessorDevCalibrationError,
            match="boundary span"):
        _requested_spans(evaluation)


def test_successor_dev_freeze_binds_public_parent_artifacts() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_dev_calibration_freeze(repository)
    assert value["status"] == (
        "W02_MORPHOLOGY_SUCCESSOR_DEV_CALIBRATION_FREEZE_COMPLETE")
    assert value["formal_training_runs"] == 1
    assert value["formal_successor_transform_runs"] == 1
    assert value["formal_dev_calibration_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["expected_counts"] == W02_MORPH_SUCCESSOR_DEV_EXPECTED_COUNTS
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_SUCCESSOR_DEV_CODE_PATHS)
