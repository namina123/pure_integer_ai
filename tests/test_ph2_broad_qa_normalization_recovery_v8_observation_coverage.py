"""覆盖 recovery-v8 Observation coverage/collision audit。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import BroadQaExternalDataError
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_coverage import (
    publish_normalization_recovery_v8_observation_coverage,
    read_normalization_recovery_v8_observation_coverage,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _observation(family: str, output: str) -> dict[str, object]:
    """构造同source/input、不同output的eligible single-Han Observation。"""
    return {
        "eligibility": {"pair_features": {
            "single_han_difference": 1, "v8_training_eligible": 1}},
        "official_source_text": "Open",
        "source_family": family,
        "zh_hans": {"translation": output},
        "zh_hans_structure_tokens": ["%1"],
        "zh_hant": {"translation": "開"},
        "zh_hant_structure_tokens": ["%1"],
    }


def _state() -> dict[str, tuple[dict[str, object], ...]]:
    """两家支持開->开，第三家形成同source/input collision。"""
    return {
        "qbittorrent-observations.jsonl": (
            _observation("QBITTORRENT_PROJECT", "开"),),
        "stellarium-observations.jsonl": (
            _observation("STELLARIUM_PROJECT", "启"),),
        "keepassxc-observations.jsonl": (
            _observation("KEEPASSXC_PROJECT", "开"),),
    }


def _dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的九个输入目录。"""
    values = tuple(tmp_path / str(index) for index in range(9))
    for value in values:
        value.mkdir()
    return values


def _publish(tmp_path: Path, inputs: tuple[Path, ...], target: Path):
    """集中publisher参数。"""
    return publish_normalization_recovery_v8_observation_coverage(
        run_root=tmp_path, observation_dir=inputs[0], v2_roster_dir=inputs[1],
        v1_roster_dir=inputs[2], v1_content_audit_dir=inputs[3],
        v2_content_audit_dir=inputs[4], source_overlap_dir=inputs[5],
        qbittorrent_source_pack_dir=inputs[6],
        stellarium_source_pack_dir=inputs[7], keepassxc_source_pack_dir=inputs[8],
        target_dir=target)


def test_v8_coverage_round_trip_conflict_and_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """候选保留multi-family collision，且不可覆盖并拒绝同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_coverage as module

    inputs = _dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda *_args, **_kwargs: _state())
    target = tmp_path / "coverage"
    published = _publish(tmp_path, inputs, target)
    manifest, outputs = read_normalization_recovery_v8_observation_coverage(
        target, observation_dir=inputs[0], v2_roster_dir=inputs[1],
        v1_roster_dir=inputs[2], v1_content_audit_dir=inputs[3],
        v2_content_audit_dir=inputs[4], source_overlap_dir=inputs[5],
        qbittorrent_source_pack_dir=inputs[6], stellarium_source_pack_dir=inputs[7],
        keepassxc_source_pack_dir=inputs[8],
        expected_manifest_sha256=str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs["exact-input-mappings.jsonl"][0]["candidate_status"] == (
        "MULTI_FAMILY_CONFLICT")
    assert outputs["source-conditioned-mappings.jsonl"][0][
        "candidate_status"] == "MULTI_FAMILY_CONFLICT"
    assert outputs["orthographic-atoms.jsonl"][0]["candidate_status"] == (
        "MULTI_FAMILY_CONFLICT")
    assert outputs["structure-obligations.jsonl"][0][
        "candidate_status"] == "MULTI_FAMILY_OBSERVED"
    assert manifest["summary"]["authorization_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        _publish(tmp_path, inputs, target)

    path = target / "coverage-census.jsonl"
    changed = canonical_json_line({"format_version": 1, "record_kind": "CHANGED"})
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    item = next(v for v in stored["files"] if v["relative_path"] == path.name)
    item["bytes"] = len(changed)
    item["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_observation_coverage(
            target, observation_dir=inputs[0], v2_roster_dir=inputs[1],
            v1_roster_dir=inputs[2], v1_content_audit_dir=inputs[3],
            v2_content_audit_dir=inputs[4], source_overlap_dir=inputs[5],
            qbittorrent_source_pack_dir=inputs[6],
            stellarium_source_pack_dir=inputs[7], keepassxc_source_pack_dir=inputs[8],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest())
