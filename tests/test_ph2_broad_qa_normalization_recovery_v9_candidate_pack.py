"""覆盖 recovery-v9 标签盲candidate rebind与pack。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_candidate_pack import (
    V9_EVALUATION_COMMITMENT_MANIFEST_SHA256,
    publish_normalization_recovery_v9_candidate_pack,
    read_normalization_recovery_v9_candidate_pack,
    rebind_normalization_recovery_v9_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate import (
    V8_CANDIDATE_PROGRAM_SHA256,
)


def test_rebind_changes_only_commitment_and_program_identity() -> None:
    """v9 rebind不得改写规则账、runtime顺序或训练lineage。"""
    base = {
        "candidate_program_sha256": V8_CANDIDATE_PROGRAM_SHA256,
        "evaluation_commitment_manifest_sha256": "a" * 64,
        "inventories": {"rules": [1, 2]},
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_order": ["RULE", "UNKNOWN"],
        "training_audit_manifest_sha256": "b" * 64,
    }
    rebound = rebind_normalization_recovery_v9_candidate(
        base,
        evaluation_commitment_manifest_sha256=(
            V9_EVALUATION_COMMITMENT_MANIFEST_SHA256))
    assert rebound["evaluation_commitment_manifest_sha256"] == (
        V9_EVALUATION_COMMITMENT_MANIFEST_SHA256)
    assert rebound["candidate_program_sha256"] != (
        V8_CANDIDATE_PROGRAM_SHA256)
    assert rebound["inventories"] == base["inventories"]
    assert rebound["runtime_order"] == base["runtime_order"]
    assert rebound["training_audit_manifest_sha256"] == (
        base["training_audit_manifest_sha256"])


def test_candidate_pack_round_trip_and_duplicate_rejection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pack必须不可覆盖且strict reader逐字节重派生。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_candidate_pack as module

    inputs = []
    for name in ("base", "commitment", "source", "gate"):
        path = tmp_path / name
        path.mkdir()
        inputs.append(path)
    target = tmp_path / "candidate"
    candidate = {"candidate_program_sha256": "c" * 64}
    preflight = {"failure_count": 0}
    payloads = {
        "candidate-program.json": b"{\"candidate_program_sha256\":\"c\"}\n",
        "preflight.json": b"{\"failure_count\":0}\n",
    }
    manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_V1"),
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "files": [{
            "bytes": len(payload),
            "relative_path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        } for name, payload in payloads.items()],
    }
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_derive",
        lambda **kwargs: (manifest, candidate, preflight, payloads))
    published = publish_normalization_recovery_v9_candidate_pack(
        run_root=tmp_path,
        base_candidate_dir=inputs[0],
        evaluation_commitment_dir=inputs[1],
        source_pack_dir=inputs[2],
        runtime_gate_dir=inputs[3],
        target_dir=target,
    )
    reread, reread_candidate, reread_preflight = (
        read_normalization_recovery_v9_candidate_pack(
            target,
            base_candidate_dir=inputs[0],
            evaluation_commitment_dir=inputs[1],
            source_pack_dir=inputs[2],
            runtime_gate_dir=inputs[3],
            expected_manifest_sha256=published["manifest_sha256"]))
    assert reread == published
    assert reread_candidate == candidate
    assert reread_preflight == preflight
    with pytest.raises(BroadQaExternalDataError):
        publish_normalization_recovery_v9_candidate_pack(
            run_root=tmp_path,
            base_candidate_dir=inputs[0],
            evaluation_commitment_dir=inputs[1],
            source_pack_dir=inputs[2],
            runtime_gate_dir=inputs[3],
            target_dir=target,
        )
