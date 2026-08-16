"""覆盖 recovery-v10 五 family local feasibility artifact I/O。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_feasibility import (
    publish_normalization_recovery_v10_source_expansion_local_feasibility,
    read_normalization_recovery_v10_source_expansion_local_feasibility,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _derived():
    """构造不含surface的最小NE feasibility派生结果。"""
    summary = {
        "authorization_count": 0,
        "projection_summary_sha256": "1" * 64,
    }
    audit = {
        "authorization_rule_count": 0,
        "loso_audit_sha256": "2" * 64,
        "novel_survivor_count": 0,
        "outcome": "NE_PREDECESSOR_ONLY_SURVIVORS",
        "outcomes": {"EXACT": 5, "UNKNOWN": 0, "WRONG": 0},
    }
    survivors = ({"predecessor_covered": 1},)
    payloads = {
        "projection-summary.json": canonical_json_line(summary),
        "loso-audit.json": canonical_json_line(audit),
        "survivors.jsonl": canonical_json_line(survivors[0]),
    }
    files = [{
        "bytes": len(payloads[name]),
        "record_count": 1 if name == "survivors.jsonl" else None,
        "relative_path": name,
        "role": role,
        "sha256": hashlib.sha256(payloads[name]).hexdigest(),
    } for name, role in (
        ("projection-summary.json", "FIVE_FAMILY_LOCAL_PROJECTION_SUMMARY"),
        ("loso-audit.json", "FIVE_FAMILY_LOCAL_LOSO_AGGREGATE_AUDIT"),
        ("survivors.jsonl", "FIVE_FAMILY_LOCAL_LOSO_SURVIVORS"),
    )]
    for item in files:
        if item["record_count"] is None:
            del item["record_count"]
    manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_LOCAL_FEASIBILITY_V1"),
        "files": files,
        "status": (
            "TRAIN_ONLY_NE_NO_NOVEL_FIVE_FAMILY_LOCAL_AUTHORIZATION_NOT_FORMAL"),
    }
    return manifest, summary, audit, survivors, payloads


def test_expanded_local_feasibility_round_trip_and_duplicate(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """紧凑artifact逐字节回读，并在大输入前拒绝重复发布。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_feasibility as module

    inputs = tuple(tmp_path / name for name in (
        "observation", "audit", "opencc", "predecessor"))
    for value in inputs:
        value.mkdir()
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_derive", lambda **_kwargs: _derived())
    target = tmp_path / "feasibility"
    published = publish_normalization_recovery_v10_source_expansion_local_feasibility(
        run_root=tmp_path,
        observation_dir=inputs[0],
        five_family_audit_dir=inputs[1],
        opencc_source_pack_dir=inputs[2],
        predecessor_feasibility_dir=inputs[3],
        target_dir=target,
    )
    manifest, summary, audit, survivors = (
        read_normalization_recovery_v10_source_expansion_local_feasibility(
            target,
            observation_dir=inputs[0],
            five_family_audit_dir=inputs[1],
            opencc_source_pack_dir=inputs[2],
            predecessor_feasibility_dir=inputs[3],
            expected_manifest_sha256=str(published["manifest_sha256"]),
        ))
    assert manifest == published
    assert summary["authorization_count"] == 0
    assert audit["outcome"] == "NE_PREDECESSOR_ONLY_SURVIVORS"
    assert survivors[0]["predecessor_covered"] == 1
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        publish_normalization_recovery_v10_source_expansion_local_feasibility(
            run_root=tmp_path,
            observation_dir=inputs[0],
            five_family_audit_dir=inputs[1],
            opencc_source_pack_dir=inputs[2],
            predecessor_feasibility_dir=inputs[3],
            target_dir=target,
        )
