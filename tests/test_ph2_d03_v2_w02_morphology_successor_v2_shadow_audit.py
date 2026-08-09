"""W-02 morphology successor V2 shadow audit 的公开合同测试。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_audit import (
    W02_MORPH_SUCCESSOR_V2_SHADOW_CODE_PATHS,
    W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS,
    W02MorphologySuccessorV2ShadowAuditError,
    _audit_v2_extension,
    build_w02_morphology_successor_v2_shadow_audit_freeze,
)


def _candidate(identity: int, *, start: int = 0, end: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        start=start, end=end, form="f", lemma=f"l{identity}", upos="NOUN",
        feats_json="[]", support_count=identity)


def test_v2_shadow_extension_retains_v1_and_bounds_each_span() -> None:
    v1 = SimpleNamespace(prediction=SimpleNamespace(
        morphology_candidates=(_candidate(1),)))
    v2 = SimpleNamespace(
        prediction=SimpleNamespace(morphology_candidates=(
            _candidate(1), _candidate(2), _candidate(3, start=0, end=2))),
        edge_candidate_count=2)
    assert _audit_v2_extension(v1, v2, ((0, 1), (0, 2))) == (True, 1)


def test_v2_shadow_extension_rejects_nonrequested_span() -> None:
    v1 = SimpleNamespace(prediction=SimpleNamespace(morphology_candidates=()))
    v2 = SimpleNamespace(
        prediction=SimpleNamespace(
            morphology_candidates=(_candidate(1, start=1, end=2),)),
        edge_candidate_count=1)
    assert _audit_v2_extension(v1, v2, ((0, 1),)) == (False, 0)


def test_v2_shadow_extension_rejects_per_span_overflow() -> None:
    v1 = SimpleNamespace(prediction=SimpleNamespace(morphology_candidates=()))
    v2 = SimpleNamespace(
        prediction=SimpleNamespace(
            morphology_candidates=tuple(_candidate(index)
                                        for index in range(1, 10))),
        edge_candidate_count=9)
    assert _audit_v2_extension(v1, v2, ((0, 1),)) == (False, 9)


def test_v2_shadow_freeze_binds_v2_dev_pass() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_v2_shadow_audit_freeze(repository)
    assert value["status"] == (
        "W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_AUDIT_FREEZE_COMPLETE")
    assert value["formal_dev_calibration_runs"] == 1
    assert value["formal_shadow_audit_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["label_reads"] == 0
    assert value["expected_counts"] == (
        W02_MORPH_SUCCESSOR_V2_SHADOW_EXPECTED_COUNTS)
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_SUCCESSOR_V2_SHADOW_CODE_PATHS)
