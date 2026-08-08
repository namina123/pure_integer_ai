"""FT00-07 bounded public release-gate tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_ft00_release import (
    FT00_CHECK_ORDER,
    FT00_RELEASE_GATE_PATH,
    FT00ReleaseGateError,
    read_ft00_release_gate,
    run_ft00_release_gate,
    write_ft00_release_gate,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / Path(*FT00_RELEASE_GATE_PATH.split("/"))


def test_repository_gate_is_exact_nine_way_pass_with_zero_formal_runs() -> None:
    report = read_ft00_release_gate(REPORT)
    replay = run_ft00_release_gate(ROOT)
    assert report == replay
    assert report.status == "FT00_COMPLETE"
    assert report.ft00_complete == 1
    assert report.check_order == FT00_CHECK_ORDER
    assert len(report.checks) == 9
    assert all(item.status == "PASS" for item in report.checks)
    assert report.formal_training_runs == 0
    assert report.formal_private_evaluation_runs == 0
    assert report.private_payload_reads == 0
    assert report.candidate_writes == report.teacher_calls == 0
    validate_v2_safe_report(report.to_dict())


def test_gate_public_projection_has_no_paths_or_capability_claim() -> None:
    value = read_canonical_object(REPORT)
    serialized = str(value).lower()
    assert "absolute_path" not in serialized
    assert "relative_path" not in serialized
    assert "language_capability_mastered" not in serialized
    assert "language_readiness" not in serialized
    assert value["next_stage"] == "W-02"


def test_gate_publish_is_immutable_and_invalid_status_is_rejected(tmp_path: Path) -> None:
    report = run_ft00_release_gate(ROOT)
    target = tmp_path / "gate.json"
    write_ft00_release_gate(report, target)
    assert read_ft00_release_gate(target) == report
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Exception):
        write_ft00_release_gate(report, target)

    value = report.to_dict()
    value["status"] = "PASS"
    invalid = tmp_path / "invalid.json"
    write_immutable_json(value, invalid)
    with pytest.raises(FT00ReleaseGateError):
        read_ft00_release_gate(invalid)
