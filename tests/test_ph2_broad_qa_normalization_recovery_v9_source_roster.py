"""覆盖 recovery-v9 GIMP 标签盲 source roster。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster import (
    publish_normalization_recovery_v9_source_roster,
    read_normalization_recovery_v9_source_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster_records import (
    derive_normalization_recovery_v9_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def test_v9_roster_freezes_all_gimp_domains_without_locale_read() -> None:
    """八个完整 domain pair 全选且不包含翻译正文。"""
    records, census = derive_normalization_recovery_v9_source_roster()
    assert len(records) == 1
    record = records[0]
    assert record["source_family"] == "GIMP_PROJECT"
    assert record["repository"] == "https://gitlab.gnome.org/GNOME/gimp.git"
    assert record["root_tree"] == "efc8a0d0df6606bc8b61b86b936e151f496013c8"
    assert record["license"]["expression"] == "GPL-3.0-or-later"
    assert record["locale_pair_count"] == 8
    assert record["locale_file_count"] == 16
    assert record["locale_blob_content_read_count"] == 0
    assert record["label_or_translation_read_count"] == 0
    assert record["prior_source_repository_overlap_count"] == 0
    assert census["locale_total_bytes"] == 3_228_085
    assert census["all_discovered_complete_domain_pairs_selected"] == 1
    encoded = canonical_json_line(record).decode("utf-8")
    assert '"translation"' not in encoded
    assert '"msgstr"' not in encoded
    assert '"surface"' not in encoded


def test_v9_roster_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """publisher 不覆盖，reader 拒绝同步篡改与额外文件。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_roster as module

    license_root = tmp_path / "license"
    license_root.mkdir()
    target = tmp_path / "artifact"
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_verify_license_root", lambda value: None)
    published = publish_normalization_recovery_v9_source_roster(
        run_root=tmp_path,
        license_root=license_root,
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v9_source_roster(
        target,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert manifest == published
    assert len(outputs["source-roster.jsonl"]) == 1
    with pytest.raises(BroadQaExternalDataError, match="target"):
        publish_normalization_recovery_v9_source_roster(
            run_root=tmp_path,
            license_root=license_root,
            target_dir=target,
        )

    roster_path = target / "source-roster.jsonl"
    roster = json.loads(roster_path.read_bytes())
    roster["root_tree"] = "f" * 40
    roster_path.write_bytes(canonical_json_line(roster))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    payload = roster_path.read_bytes()
    stored["files"][0]["bytes"] = len(payload)
    stored["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_line(stored))
    forged_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="records"):
        read_normalization_recovery_v9_source_roster(
            target, expected_manifest_sha256=forged_sha)

    roster_path.write_bytes(canonical_json_line(outputs["source-roster.jsonl"][0]))
    # 恢复正式 artifact 后单独验证额外物理文件 gate。
    manifest_without_runtime_sha = dict(published)
    manifest_without_runtime_sha.pop("manifest_sha256")
    manifest_path.write_bytes(canonical_json_line(manifest_without_runtime_sha))
    (target / "extra.txt").write_text("x", encoding="utf-8")
    restored_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="physical inventory"):
        read_normalization_recovery_v9_source_roster(
            target, expected_manifest_sha256=restored_sha)


def test_v9_roster_rejects_non_k_run_root(tmp_path: Path) -> None:
    """正式 publisher 不得把 source roster 写回 D 或临时盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K盘"):
        publish_normalization_recovery_v9_source_roster(
            run_root=tmp_path,
            license_root=tmp_path,
            target_dir=tmp_path / "artifact",
        )
