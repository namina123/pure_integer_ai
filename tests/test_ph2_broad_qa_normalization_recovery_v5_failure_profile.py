"""Recovery-v5 TRAIN-only LOSO failure profile 测试。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_failure_profile as profile_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    publish_normalization_recovery_v5_failure_profile,
    read_normalization_recovery_v5_failure_profile,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile_records import (
    derive_normalization_recovery_v5_failure_profile,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    derive_normalization_recovery_v5_training_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from test_ph2_broad_qa_normalization_recovery_v5_learner import _material


def _sha(value: str) -> str:
    """返回 synthetic artifact identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _profile_material():
    """构造 synthetic protocol、sealed audit manifest 与 profile。"""
    protocol_sha = _sha("v5-failure-profile-protocol")
    material = _material(protocol_sha)
    outputs, _learning_summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            work=material[4],
        ))
    pack_manifest = {
        "manifest_sha256": _sha("v5-failure-profile-pack"),
        "mastery_claimed": 0,
        "production_enabled": 0,
    }
    _runtime, _loso, audit_summary = (
        derive_normalization_recovery_v5_training_audit(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            pack_manifest=pack_manifest,
            outputs=outputs,
        ))
    audit_sha = _sha("v5-failure-profile-audit")
    audit_manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_V1"),
        "evaluation_payload_read_count": 0,
        "formal_run_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "status": "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED",
        "summary": audit_summary,
    }
    profile = derive_normalization_recovery_v5_failure_profile(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        audit_manifest_sha256=audit_sha,
        audit_manifest=audit_manifest,
    )
    return material, audit_sha, audit_manifest, profile


def test_v5_failure_profile_attributes_wrong_without_changing_denominator() -> None:
    """profile 必须重现 sealed WRONG 并按 rule/authority 形成诊断。"""
    _material_values, _audit_sha, _audit_manifest, profile = (
        _profile_material())
    cases, impacts, family_summaries, summary = profile
    assert len(cases) == summary["wrong_case_count"] == 3
    assert len(impacts) == summary["rule_impact_count"] == 3
    assert len(family_summaries) == 4
    assert summary["identity_false_change_count"] == 0
    assert summary["multi_rule_wrong_case_count"] == 0
    assert summary["rule_class_wrong_case_counts"] == {"CONTEXT_HUNK": 3}
    assert summary["support_family_count_wrong_case_counts"] == {"2": 3}
    assert all(item["outcome"] == "WRONG" for item in cases)
    assert all(item["selected_rule_ids"] for item in cases)
    assert all(step["mode"] == "LONGEST_LOCAL_MATCH"
               for item in cases for step in item["selected_steps"])
    assert all(item["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
               for item in impacts)
    assert sum(item["wrong_case_count"] for item in family_summaries) == 3


def test_v5_failure_profile_rejects_audit_denominator_drift() -> None:
    """sealed audit 的 family/bucket 任一计数漂移都不得发布 profile。"""
    material, audit_sha, audit_manifest, _profile = _profile_material()
    tampered = copy.deepcopy(audit_manifest)
    key = next(iter(tampered["summary"]["loso_counts"]))
    tampered["summary"]["loso_counts"][key] += 1
    with pytest.raises(BroadQaExternalDataError, match="分母漂移"):
        derive_normalization_recovery_v5_failure_profile(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            audit_manifest_sha256=audit_sha,
            audit_manifest=tampered,
        )


def _install_readers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, Path, tuple[dict[str, object], ...], str,
                   dict[str, object]]:
    """把 publisher readers 固定到 synthetic protocol/audit manifest。"""
    material, audit_sha, audit_manifest, _profile = _profile_material()
    protocol_dir = tmp_path / "protocol"
    audit_dir = tmp_path / "audit"
    protocol_dir.mkdir()
    audit_dir.mkdir()
    monkeypatch.setattr(
        profile_module,
        "read_normalization_recovery_v5_learner_input",
        lambda *args, **kwargs: material,
    )
    monkeypatch.setattr(
        profile_module,
        "_read_audit_manifest_only",
        lambda *args, **kwargs: audit_manifest,
    )
    monkeypatch.setattr(
        profile_module, "_require_k_root", lambda value: Path(value).resolve())
    return protocol_dir, audit_dir, material, audit_sha, audit_manifest


def test_v5_failure_profile_publishes_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """profile 必须不可覆盖、manifest-last 且严格回读拒绝篡改。"""
    protocol_dir, audit_dir, material, audit_sha, _audit_manifest = (
        _install_readers(tmp_path, monkeypatch))
    target = tmp_path / "profile"
    report = publish_normalization_recovery_v5_failure_profile(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        audit_dir=audit_dir,
        expected_audit_manifest_sha256=audit_sha,
        target_dir=target,
    )
    assert report["status"] == (
        "TRAIN_ONLY_DIAGNOSTIC_COMPLETE_NOT_SELECTION_NOT_EVALUATION")
    assert report["training_audit_manifest_only_read_count"] == 1
    assert report["training_audit_non_manifest_read_count"] == 0
    assert report["evaluation_payload_read_count"] == 0
    restored, outputs = read_normalization_recovery_v5_failure_profile(
        target,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        audit_dir=audit_dir,
        expected_audit_manifest_sha256=audit_sha,
        expected_profile_manifest_sha256=str(report["manifest_sha256"]),
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(outputs["wrong-cases.jsonl"]) == 3
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v5_failure_profile(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            audit_dir=audit_dir,
            expected_audit_manifest_sha256=audit_sha,
            target_dir=target,
        )
    path = target / "rule-impacts.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v5_failure_profile(
            target,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            audit_dir=audit_dir,
            expected_audit_manifest_sha256=audit_sha,
            expected_profile_manifest_sha256=str(report["manifest_sha256"]),
        )


def test_v5_profile_manifest_only_reader_rejects_encoding_tamper(
        tmp_path: Path,
        ) -> None:
    """profile 对 training-audit 只读 manifest 且拒绝非规范附加字节。"""
    _material_values, _audit_sha, audit_manifest, _profile = (
        _profile_material())
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    path = audit_dir / "manifest.json"
    path.write_bytes(canonical_json_line(audit_manifest))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = profile_module._read_audit_manifest_only(
        audit_dir, expected_manifest_sha256=sha)
    assert restored == audit_manifest
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="identity/encoding"):
        profile_module._read_audit_manifest_only(
            audit_dir, expected_manifest_sha256=hashlib.sha256(
                path.read_bytes()).hexdigest())


def test_v5_failure_profile_rejects_non_k_root_before_write(
        tmp_path: Path,
        ) -> None:
    """正式 publisher 不得把诊断 artifact 回退到 D 盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v5_failure_profile(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            audit_dir=tmp_path / "audit",
            expected_audit_manifest_sha256="b" * 64,
            target_dir=tmp_path / "profile",
        )
    assert not (tmp_path / "profile").exists()
