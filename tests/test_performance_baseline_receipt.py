"""PERF-P0 receipt 的 canonical、外部证据和 append-only 边界。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from scripts.performance_baseline_receipt import (
    RECEIPT_PATH,
    STATUS,
    build_performance_baseline_receipt,
    publish_performance_baseline_receipt,
    read_performance_baseline_receipt,
)
from scripts.performance_baseline_contract import PerformanceBaselineError


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / ".perf-p0-20260807-r1"
PUBLISHED_SHA256 = (
    "5c9f3d866ba5bb922e5c2f77a3da3c8728618b521fde7f6a308450eec050eaa1"
)


def test_published_p0_receipt_is_canonical_without_private_checkpoint() -> None:
    target = ROOT / RECEIPT_PATH
    payload = target.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PUBLISHED_SHA256
    assert payload.endswith(b"\n")
    value = read_performance_baseline_receipt(ROOT, verify_external=False)
    assert canonical_json_bytes(value) + b"\n" == payload
    assert value["status"] == STATUS
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    assert tuple(
        item["name"] for item in value["scenario_results"]
    ) == (
        "long_input_hierarchy",
        "long_session_checkpoint",
        "long_memory_projection",
        "storage_dict",
        "storage_sqlite",
    )


@pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists(),
    reason="本地 Git 外 P0 checkpoint 不存在",
)
def test_p0_receipt_replays_external_artifacts() -> None:
    value = read_performance_baseline_receipt(ROOT, CHECKPOINT)
    assert len(value["external_logs"]) == 5
    assert len(value["external_databases"]) == 1


def test_p0_receipt_publish_is_append_only_when_checkpoint_exists(
        tmp_path: Path,
        ) -> None:
    if not (CHECKPOINT / "state.json").exists():
        pytest.skip("本地 Git 外 P0 checkpoint 不存在")
    target = tmp_path / "performance_baseline_receipt.json"
    value = publish_performance_baseline_receipt(
        ROOT, CHECKPOINT, target=target)
    assert read_performance_baseline_receipt(ROOT, CHECKPOINT, target) == value
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_performance_baseline_receipt(ROOT, CHECKPOINT, target=target)


def test_p0_receipt_rejects_malformed_result_and_external_path(
        tmp_path: Path,
        ) -> None:
    value = read_performance_baseline_receipt(ROOT, verify_external=False)
    malformed_result = deepcopy(value)
    malformed_result["scenario_results"][0] = 1
    result_path = tmp_path / "malformed-result.json"
    result_path.write_bytes(canonical_json_bytes(malformed_result) + b"\n")
    with pytest.raises(ValueError, match="scenario_result"):
        read_performance_baseline_receipt(
            ROOT, path=result_path, verify_external=False)

    malformed_path = deepcopy(value)
    malformed_path["external_logs"][0]["relative_path"] = "C:/outside.log"
    external_path = tmp_path / "malformed-path.json"
    external_path.write_bytes(canonical_json_bytes(malformed_path) + b"\n")
    with pytest.raises(ValueError, match="external artifact path"):
        read_performance_baseline_receipt(
            ROOT, path=external_path, verify_external=False)


def test_p0_receipt_build_rejects_in_repo_checkpoint() -> None:
    with pytest.raises(PerformanceBaselineError, match="公开 Git 根之外"):
        build_performance_baseline_receipt(ROOT, ROOT / "state")
