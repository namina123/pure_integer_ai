"""覆盖 recovery-v10 local hypothesis NE feasibility artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def test_local_hypothesis_feasibility_round_trip_and_duplicate_rejection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NE artifact必须不可覆盖，strict reader逐字节重派生。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_feasibility as module

    inputs = []
    for name in ("observations", "opencc", "v2-feasibility"):
        path = tmp_path / name
        path.mkdir()
        inputs.append(path)
    summary = {"authorization_count": 0, "observation_count": 2}
    audit = {
        "authorization_rule_count": 0,
        "novel_survivor_count": 0,
        "outcome": "NE_PREDECESSOR_ONLY_SURVIVORS",
    }
    survivors = ({
        "predecessor_covered": 1,
        "rule_semantic_id": "a" * 64,
    },)
    payloads = {
        "projection-summary.json": canonical_json_line(summary),
        "loso-audit.json": canonical_json_line(audit),
        "survivors.jsonl": canonical_json_line(survivors[0]),
    }
    manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_FEASIBILITY_V1"),
        "files": [{
            "bytes": len(payload),
            "relative_path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        } for name, payload in payloads.items()],
        "loso_outcome": "NE_PREDECESSOR_ONLY_SURVIVORS",
    }
    monkeypatch.setattr(
        module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_derive",
        lambda **kwargs: (manifest, summary, audit, survivors, payloads))
    target = tmp_path / "published"
    published = (
        module.publish_normalization_recovery_v10_local_hypothesis_feasibility(
            run_root=tmp_path,
            observation_dir=inputs[0],
            opencc_source_pack_dir=inputs[1],
            feasibility_dir=inputs[2],
            target_dir=target,
        ))
    reread = (
        module.read_normalization_recovery_v10_local_hypothesis_feasibility(
            target,
            observation_dir=inputs[0],
            opencc_source_pack_dir=inputs[1],
            feasibility_dir=inputs[2],
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    assert reread == (published, summary, audit, survivors)
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        module.publish_normalization_recovery_v10_local_hypothesis_feasibility(
            run_root=tmp_path,
            observation_dir=inputs[0],
            opencc_source_pack_dir=inputs[1],
            feasibility_dir=inputs[2],
            target_dir=target,
        )
