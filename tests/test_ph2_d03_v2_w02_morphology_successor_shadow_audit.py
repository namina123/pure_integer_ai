"""W-02 morphology successor shadow audit 的公开合同测试。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_shadow_audit import (
    W02_MORPH_SUCCESSOR_SHADOW_CODE_PATHS,
    W02_SHADOW_EXPECTED_COUNTS,
    W02MorphologySuccessorShadowAuditError,
    W02ShadowInputRoot,
    _select_shadow_spans,
    build_w02_morphology_successor_shadow_audit_freeze,
)


def test_shadow_span_policy_is_label_free_and_dual() -> None:
    base = SimpleNamespace(
        boundary_lattice=(0, 1, 2, 3),
        generation=SimpleNamespace(surface="abc"),
        morphology_candidates=(SimpleNamespace(start=0, end=1),),
    )
    assert _select_shadow_spans(base, 17) == ((0, 1), (0, 2))


def test_shadow_root_rejects_wrong_owner(tmp_path: Path) -> None:
    root = tmp_path / "not-shadow"
    root.mkdir()
    with pytest.raises(W02MorphologySuccessorShadowAuditError,
                       match="shadow root"):
        W02ShadowInputRoot(root)


def test_shadow_freeze_binds_successor_dev_pass() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_shadow_audit_freeze(repository)
    assert value["status"] == (
        "W02_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT_FREEZE_COMPLETE")
    assert value["formal_dev_calibration_runs"] == 1
    assert value["formal_shadow_audit_runs"] == 0
    assert value["formal_private_evaluation_runs"] == 0
    assert value["label_reads"] == 0
    assert value["expected_counts"] == W02_SHADOW_EXPECTED_COUNTS
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_SUCCESSOR_SHADOW_CODE_PATHS)
