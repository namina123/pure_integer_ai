"""Recovery-v6 strong-whole rule pack 发布与严格回读测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_rule_pack as pack_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_rule_pack import (
    publish_normalization_recovery_v6_rule_pack,
    read_normalization_recovery_v6_rule_pack,
)
from test_ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    _program_material,
    _sha,
)


def _install_reader(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """把 predecessor pack reader 固定到 synthetic 双运行等价输出。"""
    material, predecessor, _v5, _outputs, _summary, _program = (
        _program_material())
    protocol_sha = str(material[0]["manifest_sha256"])
    predecessor_sha = _sha("v6-runtime-predecessor-pack")
    predecessor_manifest = {
        "fresh_resume_output_bytes_equal": 1,
        "learner_lineages": [
            {"role": "FRESH", "run_id": _sha("fresh")},
            {"role": "RESUMED", "run_id": _sha("resumed")},
        ],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
    }
    protocol_dir = tmp_path / "protocol"
    predecessor_dir = tmp_path / "predecessor"
    protocol_dir.mkdir()
    predecessor_dir.mkdir()
    monkeypatch.setattr(
        pack_module,
        "read_normalization_recovery_v5_rule_pack",
        lambda *args, **kwargs: (predecessor_manifest, predecessor),
    )
    monkeypatch.setattr(
        pack_module, "_require_k_root", lambda value: Path(value).resolve())
    return protocol_dir, predecessor_dir, protocol_sha, predecessor_sha


def test_v6_rule_pack_publishes_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """v6 pack 必须不可覆盖、manifest-last 且严格回读拒绝篡改。"""
    protocol_dir, predecessor_dir, protocol_sha, predecessor_sha = (
        _install_reader(tmp_path, monkeypatch))
    target = tmp_path / "pack"
    report = publish_normalization_recovery_v6_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        predecessor_pack_dir=predecessor_dir,
        expected_predecessor_pack_manifest_sha256=predecessor_sha,
        target_dir=target,
    )
    assert report["status"] == (
        "FROZEN_POLICY_PROJECTED_NOT_EVALUATED_NOT_DEPLOYED")
    assert report["predecessor_rule_pack_read_count"] == 1
    assert report["summary"]["executable_local_rule_count"] == 0
    restored, outputs = read_normalization_recovery_v6_rule_pack(
        target,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        predecessor_pack_dir=predecessor_dir,
        expected_predecessor_pack_manifest_sha256=predecessor_sha,
        expected_pack_manifest_sha256=str(report["manifest_sha256"]),
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert outputs["target-whole-rules.jsonl"]
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v6_rule_pack(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
            predecessor_pack_dir=predecessor_dir,
            expected_predecessor_pack_manifest_sha256=predecessor_sha,
            target_dir=target,
        )
    path = target / "target-index.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v6_rule_pack(
            target,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
            predecessor_pack_dir=predecessor_dir,
            expected_predecessor_pack_manifest_sha256=predecessor_sha,
            expected_pack_manifest_sha256=str(report["manifest_sha256"]),
        )


def test_v6_rule_pack_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 publisher 不得把 v6 pack 回退到 D 盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v6_rule_pack(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            predecessor_pack_dir=tmp_path / "predecessor",
            expected_predecessor_pack_manifest_sha256="b" * 64,
            target_dir=tmp_path / "pack",
        )
    assert not (tmp_path / "pack").exists()
