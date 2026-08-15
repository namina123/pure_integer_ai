"""Recovery-v6 TRAIN-only audit records 测试。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_training_audit as audit_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_training_audit_records as audit_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation import (
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_KIND,
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation_records import (
    derive_normalization_recovery_v5_successor_simulation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_records import (
    derive_normalization_recovery_v6_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_rule_pack import (
    NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND,
    NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_training_audit_records import (
    derive_normalization_recovery_v6_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_training_audit import (
    publish_normalization_recovery_v6_training_audit,
    read_normalization_recovery_v6_training_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from test_ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    _profile_material,
)


def _sha(value: str) -> str:
    """返回 synthetic manifest identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _audit_material():
    """构造同一 protocol 下的 sealed denominator、simulation 与 v6 pack。"""
    material, audit_sha, audit_manifest, profile = _profile_material()
    protocol_sha = str(material[0]["manifest_sha256"])
    predecessor_sha = _sha("v6-audit-predecessor-pack")
    audit_manifest = {
        **audit_manifest,
        "pack_manifest_sha256": predecessor_sha,
    }
    profile_sha = _sha("v6-audit-profile")
    profile_manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
        "formal_run_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "selection_or_threshold_changed": 0,
        "status": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
        "summary": profile[3],
        "training_audit_manifest_sha256": audit_sha,
    }
    cases, families, strategies, simulation_summary = (
        derive_normalization_recovery_v5_successor_simulation(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            audit_manifest_sha256=audit_sha,
            audit_manifest=audit_manifest,
            profile_manifest_sha256=profile_sha,
            profile_manifest=profile_manifest,
        ))
    del cases
    simulation_sha = _sha("v6-audit-simulation")
    simulation_manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_KIND,
        "manifest_sha256": simulation_sha,
        "protocol_manifest_sha256": protocol_sha,
        "status": NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_STATUS,
        "summary": simulation_summary,
        "training_audit_manifest_sha256": audit_sha,
    }
    predecessor, _summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            work=material[4],
        ))
    pack_outputs, pack_summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=protocol_sha,
        predecessor_pack_manifest_sha256=predecessor_sha,
        predecessor_outputs=predecessor,
    )
    pack_sha = _sha("v6-audit-pack")
    pack_manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND,
        "manifest_sha256": pack_sha,
        "mastery_claimed": 0,
        "predecessor_rule_pack_manifest_sha256": predecessor_sha,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "status": NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS,
        "summary": pack_summary,
    }
    return (material, audit_sha, audit_manifest, simulation_sha,
            simulation_manifest, families, strategies, pack_manifest,
            pack_outputs)


def test_v6_audit_matches_whole_strong_with_only_four_loso_relearns(
        monkeypatch,
        ) -> None:
    """v6 audit 必须四方向各重学一次并逐字段等于 WHOLE_STRONG。"""
    values = _audit_material()
    calls = []
    original = audit_records.derive_normalization_recovery_v5_loso_execution

    def counted(**kwargs):
        """记录 audit 自身 LOSO 调用并转发真实 pure derivation。"""
        calls.append(kwargs["held_out_source_family"])
        return original(**kwargs)

    monkeypatch.setattr(
        audit_records,
        "derive_normalization_recovery_v5_loso_execution",
        counted,
    )
    material, audit_sha, audit_manifest, simulation_sha, simulation_manifest, families, strategies, pack_manifest, pack_outputs = values
    runtime, loso, summary = derive_normalization_recovery_v6_training_audit(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        pack_manifest=pack_manifest,
        pack_outputs=pack_outputs,
        audit_manifest_sha256=audit_sha,
        audit_manifest=audit_manifest,
        simulation_manifest_sha256=simulation_sha,
        simulation_manifest=simulation_manifest,
        simulation_family_records=families,
        simulation_strategy_records=strategies,
    )
    assert len(calls) == len(set(calls)) == 4
    assert len(runtime) == 5
    assert len(loso) == 4
    assert summary["audit_outcome"] == "FACILITY_PASS_CAPABILITY_FAIL"
    assert summary["facility_failure_count"] == 0
    assert summary["capability_gate_pass"] == 0
    assert summary["outcome_counts"]["WRONG"] == 0
    assert summary["identity_false_change_count"] == 0
    assert summary["simulation_strategy_reference_equal"] == 1
    expected = next(item for item in strategies
                    if item["strategy"] == "WHOLE_STRONG")
    assert summary["non_identity_exact_count"] == expected[
        "non_identity_exact_count"] == 0


def test_v6_audit_reports_reference_drift_without_lowering_gate() -> None:
    """successor family 参照漂移必须成为 facility FAIL，不得改分母。"""
    values = _audit_material()
    material, audit_sha, audit_manifest, simulation_sha, simulation_manifest, families, strategies, pack_manifest, pack_outputs = values
    tampered = list(copy.deepcopy(families))
    target = next(index for index, item in enumerate(tampered)
                  if item["strategy"] == "WHOLE_STRONG")
    tampered[target]["outcome_counts"]["UNKNOWN"] += 1
    _runtime, loso, summary = derive_normalization_recovery_v6_training_audit(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        pack_manifest=pack_manifest,
        pack_outputs=pack_outputs,
        audit_manifest_sha256=audit_sha,
        audit_manifest=audit_manifest,
        simulation_manifest_sha256=simulation_sha,
        simulation_manifest=simulation_manifest,
        simulation_family_records=tuple(tampered),
        simulation_strategy_records=strategies,
    )
    assert any(item["simulation_family_reference_equal"] == 0 for item in loso)
    assert summary["audit_outcome"] == "FACILITY_FAIL"
    assert summary["capability_gate_pass"] == 0


def test_v6_audit_rejects_denominator_and_predecessor_lineage_drift() -> None:
    """v6 pack、v5 denominator 与 simulation 必须绑定同一冻结血缘。"""
    values = _audit_material()
    (material, audit_sha, audit_manifest, simulation_sha,
     simulation_manifest, families, strategies, pack_manifest,
     pack_outputs) = values
    arguments = {
        "protocol_manifest": material[0],
        "observations": material[1],
        "fragments": material[2],
        "pack_manifest": pack_manifest,
        "pack_outputs": pack_outputs,
        "audit_manifest_sha256": audit_sha,
        "audit_manifest": audit_manifest,
        "simulation_manifest_sha256": simulation_sha,
        "simulation_manifest": simulation_manifest,
        "simulation_family_records": families,
        "simulation_strategy_records": strategies,
    }
    bad_simulation = copy.deepcopy(simulation_manifest)
    bad_simulation["training_audit_manifest_sha256"] = _sha(
        "different denominator")
    with pytest.raises(BroadQaExternalDataError, match="predecessor contract"):
        derive_normalization_recovery_v6_training_audit(
            **{**arguments, "simulation_manifest": bad_simulation})

    bad_pack = copy.deepcopy(pack_manifest)
    bad_pack["predecessor_rule_pack_manifest_sha256"] = _sha(
        "different predecessor")
    with pytest.raises(BroadQaExternalDataError, match="predecessor contract"):
        derive_normalization_recovery_v6_training_audit(
            **{**arguments, "pack_manifest": bad_pack})


def test_v6_simulation_reader_projects_verified_external_manifest_sha(
        tmp_path: Path,
        ) -> None:
    """compact reader 必须把已核验的外部 SHA 投影到合同视图。"""
    values = _audit_material()
    simulation_manifest = copy.deepcopy(values[4])
    families = values[5]
    strategies = values[6]
    simulation_manifest.pop("manifest_sha256")
    root = tmp_path / "simulation"
    root.mkdir()
    commitments = []
    for name, records in (
            ("strategy-family-results.jsonl", families),
            ("strategy-results.jsonl", strategies)):
        payload = b"".join(canonical_json_line(item) for item in records)
        (root / name).write_bytes(payload)
        commitments.append({
            "bytes": len(payload),
            "record_count": len(records),
            "relative_path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    simulation_manifest["files"] = commitments
    encoded = canonical_json_line(simulation_manifest)
    (root / "manifest.json").write_bytes(encoded)
    manifest_sha = hashlib.sha256(encoded).hexdigest()

    restored, restored_families, restored_strategies = (
        audit_module._read_simulation_reference(
            root,
            expected_manifest_sha256=manifest_sha,
        ))
    assert restored["manifest_sha256"] == manifest_sha
    assert restored_families == families
    assert restored_strategies == strategies


def _install_publisher_readers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """把 publisher 的五个 sealed 输入固定到 synthetic TRAIN material。"""
    values = _audit_material()
    (material, audit_sha, audit_manifest, simulation_sha,
     simulation_manifest, families, strategies, pack_manifest,
     pack_outputs) = values
    roots = tuple(tmp_path / name for name in (
        "protocol", "predecessor-pack", "v6-pack", "denominator-audit",
        "successor-simulation"))
    for root in roots:
        root.mkdir()
    manifest_reads = []
    simulation_reads = []

    def read_manifest(root, *, expected_manifest_sha256, label):
        """只替代 denominator manifest reader 并记录读取边界。"""
        manifest_reads.append((Path(root), expected_manifest_sha256, label))
        return audit_manifest

    def read_simulation(root, *, expected_manifest_sha256):
        """只替代 compact simulation reader 并记录读取边界。"""
        simulation_reads.append((Path(root), expected_manifest_sha256))
        return simulation_manifest, families, strategies

    monkeypatch.setattr(
        audit_module,
        "read_normalization_recovery_v5_learner_input",
        lambda *args, **kwargs: material,
    )
    monkeypatch.setattr(
        audit_module,
        "read_normalization_recovery_v6_rule_pack",
        lambda *args, **kwargs: (pack_manifest, pack_outputs),
    )
    monkeypatch.setattr(audit_module, "_read_manifest_only", read_manifest)
    monkeypatch.setattr(
        audit_module, "_read_simulation_reference", read_simulation)
    monkeypatch.setattr(
        audit_module, "_require_k_root", lambda value: Path(value).resolve())
    return (
        roots,
        material,
        audit_sha,
        simulation_sha,
        pack_manifest,
        manifest_reads,
        simulation_reads,
    )


def test_v6_audit_publishes_manifest_last_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher 必须不可覆盖、manifest-last 且严格回读拒绝篡改。"""
    (roots, material, audit_sha, simulation_sha, pack_manifest,
     manifest_reads, simulation_reads) = _install_publisher_readers(
         tmp_path, monkeypatch)
    protocol_dir, predecessor_dir, pack_dir, denominator_dir, simulation_dir = roots
    target = tmp_path / "v6-audit"
    write_order = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        """记录 target 内 exclusive writes，同时保持真实文件语义。"""
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "xb" and path.parent == target:
            write_order.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    arguments = {
        "run_root": tmp_path,
        "protocol_dir": protocol_dir,
        "expected_protocol_manifest_sha256": str(
            material[0]["manifest_sha256"]),
        "predecessor_pack_dir": predecessor_dir,
        "expected_predecessor_pack_manifest_sha256": _sha(
            "v6-audit-predecessor-pack"),
        "pack_dir": pack_dir,
        "expected_pack_manifest_sha256": str(
            pack_manifest["manifest_sha256"]),
        "denominator_audit_dir": denominator_dir,
        "expected_denominator_audit_manifest_sha256": audit_sha,
        "simulation_dir": simulation_dir,
        "expected_simulation_manifest_sha256": simulation_sha,
        "target_dir": target,
    }
    report = publish_normalization_recovery_v6_training_audit(**arguments)
    assert write_order == [
        "runtime-audit.jsonl", "loso-audit.jsonl", "manifest.json"]
    assert report["status"] == "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED"
    assert report["formal_run_count"] == 0
    assert report["evaluation_payload_read_count"] == 0
    assert report["simulation_case_file_read_count"] == 0
    assert report["simulation_compact_file_read_count"] == 2
    assert len(manifest_reads) == len(simulation_reads) == 1

    read_arguments = dict(arguments)
    del read_arguments["run_root"]
    del read_arguments["target_dir"]
    restored, outputs = read_normalization_recovery_v6_training_audit(
        target,
        expected_audit_manifest_sha256=str(report["manifest_sha256"]),
        **read_arguments,
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(outputs["runtime-audit.jsonl"]) == 5
    assert len(outputs["loso-audit.jsonl"]) == 4
    assert len(manifest_reads) == len(simulation_reads) == 2

    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v6_training_audit(**arguments)
    runtime_path = target / "runtime-audit.jsonl"
    runtime_path.write_bytes(runtime_path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v6_training_audit(
            target,
            expected_audit_manifest_sha256=str(report["manifest_sha256"]),
            **read_arguments,
        )


def test_v6_audit_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 publisher 不得把 v6 audit artifact 回退到 D 盘。"""
    target = tmp_path / "v6-audit"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v6_training_audit(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            predecessor_pack_dir=tmp_path / "predecessor-pack",
            expected_predecessor_pack_manifest_sha256="b" * 64,
            pack_dir=tmp_path / "v6-pack",
            expected_pack_manifest_sha256="c" * 64,
            denominator_audit_dir=tmp_path / "denominator-audit",
            expected_denominator_audit_manifest_sha256="d" * 64,
            simulation_dir=tmp_path / "successor-simulation",
            expected_simulation_manifest_sha256="e" * 64,
            target_dir=target,
        )
    assert not target.exists()
