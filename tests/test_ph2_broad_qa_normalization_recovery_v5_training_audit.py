"""Recovery-v5 TRAIN-only runtime、四方向 LOSO 与严格发布测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_training_audit as audit_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_training_audit_records as audit_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit import (
    publish_normalization_recovery_v5_training_audit,
    read_normalization_recovery_v5_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    AUDIT_BUCKETS,
    LOSO_OUTCOMES,
    RULE_BARE_OUTCOMES,
    derive_normalization_recovery_v5_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
    derive_normalization_recovery_v5_fragments,
    derive_normalization_recovery_v5_groups,
)
from test_ph2_broad_qa_normalization_recovery_v5_learner import (
    _material,
    _work,
)


def _sha(value: str) -> str:
    """返回 synthetic artifact identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _audit_material():
    """派生四来源 synthetic protocol、disabled pack 与 audit。"""
    protocol_sha = _sha("v5-training-audit-protocol")
    material = _material(protocol_sha)
    outputs, _summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            work=material[4],
        ))
    pack_manifest = {
        "manifest_sha256": _sha("v5-training-audit-pack"),
        "mastery_claimed": 0,
        "production_enabled": 0,
    }
    runtime, loso, summary = derive_normalization_recovery_v5_training_audit(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        groups=material[3],
        pack_manifest=pack_manifest,
        outputs=outputs,
    )
    return material, pack_manifest, outputs, runtime, loso, summary


def test_v5_audit_distinguishes_facility_from_loso_capability() -> None:
    """四方向、五桶和优先级遮蔽必须审计，能力失败不得污染设施。"""
    material, _pack, outputs, runtime, loso, summary = _audit_material()
    assert len(loso) == len(V5_SOURCE_FAMILIES) == 4
    assert summary["facility_failure_count"] == 0
    assert summary["capability_gate_pass"] == 0
    assert summary["audit_outcome"] == "FACILITY_PASS_CAPABILITY_FAIL"
    assert summary["rule_bare_outcome_counts"]["PRIORITY_SHADOWED"] >= 1
    assert summary["rule_bare_outcome_counts"]["UNEXPECTED"] == 0
    assert set(summary["rule_bare_outcome_counts"]) == set(
        RULE_BARE_OUTCOMES)
    assert summary["runtime_case_count"] == len(runtime)
    assert summary["rule_case_count"] == (
        len(outputs["target-phrase-rules.jsonl"])
        + len(outputs["source-phrase-rules.jsonl"]))
    assert set(summary["bucket_outcome_counts"]) == {
        f"{bucket}:{outcome}"
        for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES}
    for record in loso:
        assert set(record["bucket_outcome_counts"]) == set(AUDIT_BUCKETS)
        assert set(record["outcome_counts"]) == set(LOSO_OUTCOMES)
        assert len(record["training_source_families"]) == 3
        assert record["held_out_source_family"] not in record[
            "training_source_families"]
        assert record["held_out_observation_read_for_learning_count"] == 0
        assert record["selection_leakage_count"] == 0
        assert record["source_leak_count"] == 0
    assert sum(item["held_out_observation_count"] for item in loso) == len(
        material[1])


def test_v5_edit_core_routes_to_character_local_bucket() -> None:
    """五桶合同把 learned EDIT_CORE 与 character backoff 归入同一局部桶。"""
    bucket = audit_records._bucket_from_result(
        {"identity_preservation": 0},
        {"steps": [{
            "mode": "LONGEST_LOCAL_MATCH",
            "rule_class": "EDIT_CORE",
        }]},
    )
    assert bucket == "CHARACTER_LOCAL"


def test_v5_loso_accepts_exactly_three_known_training_sources() -> None:
    """LOSO 可用三来源重学，但两来源子集仍须被 learner 拒绝。"""
    protocol_sha = _sha("v5-training-audit-two-source")
    manifest, observations, _fragments, _groups, _ordered = _material(
        protocol_sha)
    retained = set(V5_SOURCE_FAMILIES[:2])
    subset = tuple(item for item in observations
                   if item["source_family"] in retained)
    fragments = derive_normalization_recovery_v5_fragments(subset)
    groups = derive_normalization_recovery_v5_groups(fragments)
    with pytest.raises(BroadQaExternalDataError, match="source roster"):
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=manifest,
            observations=subset,
            fragments=fragments,
            groups=groups,
            work=_work(subset, fragments, groups),
        )


def _install_readers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, Path, tuple[dict[str, object], ...],
                   dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """把 publisher 的外部 readers 固定到 synthetic material。"""
    material, pack_manifest, outputs, _runtime, _loso, _summary = (
        _audit_material())
    protocol_dir = tmp_path / "protocol"
    pack_dir = tmp_path / "pack"
    protocol_dir.mkdir()
    pack_dir.mkdir()
    monkeypatch.setattr(
        audit_module,
        "read_normalization_recovery_v5_learner_input",
        lambda *args, **kwargs: material,
    )
    monkeypatch.setattr(
        audit_module,
        "read_normalization_recovery_v5_rule_pack",
        lambda *args, **kwargs: (pack_manifest, outputs),
    )
    monkeypatch.setattr(
        audit_module, "_require_k_root", lambda value: Path(value).resolve())
    return protocol_dir, pack_dir, material, pack_manifest, outputs


def test_v5_audit_publishes_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit 必须 manifest-last、不可覆盖并以三 SHA 严格重派生。"""
    protocol_dir, pack_dir, material, pack_manifest, _outputs = (
        _install_readers(tmp_path, monkeypatch))
    target = tmp_path / "audit"
    report = publish_normalization_recovery_v5_training_audit(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_manifest["manifest_sha256"]),
        target_dir=target,
    )
    assert report["status"] == "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED"
    assert report["formal_run_count"] == 0
    assert report["evaluation_payload_read_count"] == 0
    restored, outputs = read_normalization_recovery_v5_training_audit(
        target,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_manifest["manifest_sha256"]),
        expected_audit_manifest_sha256=str(report["manifest_sha256"]),
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(outputs["loso-audit.jsonl"]) == 4
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v5_training_audit(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            pack_dir=pack_dir,
            expected_pack_manifest_sha256=str(
                pack_manifest["manifest_sha256"]),
            target_dir=target,
        )
    runtime_path = target / "runtime-audit.jsonl"
    runtime_path.write_bytes(runtime_path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v5_training_audit(
            target,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            pack_dir=pack_dir,
            expected_pack_manifest_sha256=str(
                pack_manifest["manifest_sha256"]),
            expected_audit_manifest_sha256=str(report["manifest_sha256"]),
        )


def test_v5_audit_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 publisher 不得把 audit artifact 回退到 D 盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v5_training_audit(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            pack_dir=tmp_path / "pack",
            expected_pack_manifest_sha256="b" * 64,
            target_dir=tmp_path / "audit",
        )
    assert not (tmp_path / "audit").exists()
