"""Recovery-v3 TRAIN-only runtime、defeater 与 LOSO audit 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v3_training_audit as audit_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_rule_pack import (
    publish_normalization_recovery_v3_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_audit import (
    publish_normalization_recovery_v3_training_audit,
    read_normalization_recovery_v3_training_audit,
)
from test_ph2_broad_qa_normalization_recovery_v3_learner import (
    _publish_protocol,
    _run_pair,
)


def _prepare(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    """发布 synthetic protocol、learner lineage 和 disabled pack。"""
    protocol_dir, protocol_report = _publish_protocol(tmp_path, monkeypatch)
    protocol_sha = str(protocol_report["manifest_sha256"])
    fresh, resumed, _fresh_report, _resumed_report = _run_pair(
        tmp_path, protocol_dir, protocol_sha)
    pack_dir = tmp_path / "pack"
    pack_report = publish_normalization_recovery_v3_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=pack_dir,
    )
    monkeypatch.setattr(
        audit_module, "_require_k_root", lambda value: Path(value).resolve())
    return protocol_dir, pack_dir, protocol_report, pack_report


def test_audit_publishes_strict_runtime_defeater_and_loso_records(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """审计必须真实重学 held-out family 且不把诊断冒充 capability PASS。"""
    protocol_dir, pack_dir, protocol_report, pack_report = _prepare(
        tmp_path, monkeypatch)
    target = tmp_path / "audit"
    report = publish_normalization_recovery_v3_training_audit(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(protocol_report["manifest_sha256"]),
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        target_dir=target,
    )
    assert report["status"] == (
        "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED")
    assert report["summary"]["facility_failure_count"] == 0
    assert report["summary"]["rule_case_count"] >= 2
    assert report["summary"]["defeater_mismatch_count"] == 0
    assert report["summary"]["context_interpreter"][
        "indexed_reference_mismatch_count"] == 0
    assert report["summary"]["loso_family_count"] == 2
    assert all(item["selection_leakage_count"] == 0
               for item in _read_loso(target))
    restored, outputs = read_normalization_recovery_v3_training_audit(
        target,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(protocol_report["manifest_sha256"]),
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        expected_audit_manifest_sha256=str(report["manifest_sha256"]),
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(outputs["runtime-audit.jsonl"]) == report["summary"][
        "runtime_case_count"]


def _read_loso(target: Path) -> tuple[dict[str, object], ...]:
    """读取测试 audit 的 LOSO 记录。"""
    import json
    return tuple(json.loads(line) for line in (
        target / "loso-audit.jsonl").read_bytes().splitlines())


def test_audit_reader_rejects_tamper_and_target_republish(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit JSONL 或同名 target 变化都必须 fail closed。"""
    protocol_dir, pack_dir, protocol_report, pack_report = _prepare(
        tmp_path, monkeypatch)
    target = tmp_path / "audit"
    report = publish_normalization_recovery_v3_training_audit(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(protocol_report["manifest_sha256"]),
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        target_dir=target,
    )
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v3_training_audit(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                protocol_report["manifest_sha256"]),
            pack_dir=pack_dir,
            expected_pack_manifest_sha256=str(
                pack_report["manifest_sha256"]),
            target_dir=target,
        )
    path = target / "runtime-audit.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v3_training_audit(
            target,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(protocol_report["manifest_sha256"]),
            pack_dir=pack_dir,
            expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
            expected_audit_manifest_sha256=str(report["manifest_sha256"]),
        )


def test_audit_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 audit publisher 不得回退到 D 盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v3_training_audit(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            pack_dir=tmp_path / "pack",
            expected_pack_manifest_sha256="b" * 64,
            target_dir=tmp_path / "audit",
        )
    assert not (tmp_path / "audit").exists()
