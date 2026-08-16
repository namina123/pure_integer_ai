"""覆盖 recovery-v10 独立来源扩充名册。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    publish_normalization_recovery_v10_source_expansion_roster,
    read_normalization_recovery_v10_source_expansion_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster_records import (
    derive_normalization_recovery_v10_source_expansion_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def test_v10_source_expansion_roster_freezes_decisions_without_surface() -> None:
    """两家进 TRAIN 内容审计，OBS 保留 formal，Wireshark 因缺繁中拒绝。"""
    candidates, exclusions, census = (
        derive_normalization_recovery_v10_source_expansion_roster())
    by_family = {str(item["source_family"]): item for item in candidates}
    assert set(by_family) == {
        "MIXXX_PROJECT", "MUMBLE_PROJECT", "OBS_STUDIO_PROJECT",
        "WIRESHARK_PROJECT",
    }
    assert by_family["MIXXX_PROJECT"]["selection_status"] == (
        "SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING")
    assert by_family["MUMBLE_PROJECT"]["selection_status"] == (
        "SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING")
    assert by_family["OBS_STUDIO_PROJECT"]["selection_status"] == (
        "RESERVED_UNREAD_FRESH_FORMAL_CANDIDATE")
    assert by_family["OBS_STUDIO_PROJECT"]["complete_locale_pair_count"] == 38
    assert by_family["WIRESHARK_PROJECT"]["selection_status"] == (
        "REJECTED_NO_ZH_TW_COUNTERPART")
    assert census == {
        "candidate_count": 4,
        "candidate_locale_blob_content_read_count": 0,
        "candidate_surface_published": 0,
        "historical_exclusion_count": 14,
        "rejected_candidate_count": 1,
        "reserved_formal_candidate_count": 1,
        "selected_train_candidate_count": 2,
        "selected_train_complete_locale_pair_count": 2,
        "selected_train_locale_file_count": 4,
        "selected_train_locale_total_bytes": 2_614_980,
    }
    encoded = b"".join(canonical_json_line(item) for item in candidates)
    assert b'"translation"' not in encoded
    assert b'"source"' not in encoded
    excluded = {str(item["source_family"]): item for item in exclusions}
    assert excluded["BITCOIN_CORE_PROJECT"]["exclusion_reason"] == (
        "PREDECESSOR_REJECTED_ZERO_ACTIVE_COMMON_PAIR")
    assert excluded["CC_CEDICT"]["exclusion_reason"] == (
        "LICENSE_RECONCILIATION_BLOCKED")
    for family in (
            "AUDACITY_PROJECT", "FIREFOX_PROJECT", "GIMP_PROJECT",
            "QT_PROJECT", "VLC_PROJECT"):
        assert excluded[family]["exclusion_reason"] == "FORMAL_FAMILY_CONSUMED"


def test_v10_source_expansion_roster_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """publisher 不覆盖，reader 拒绝同步篡改与额外文件。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster as module

    target = tmp_path / "artifact"
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    published = publish_normalization_recovery_v10_source_expansion_roster(
        run_root=tmp_path,
        target_dir=target,
    )
    manifest, outputs = (
        read_normalization_recovery_v10_source_expansion_roster(
            target,
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    assert manifest == published
    assert len(outputs["source-candidates.jsonl"]) == 4
    assert len(outputs["source-exclusions.jsonl"]) == 14
    with pytest.raises(BroadQaExternalDataError, match="target"):
        publish_normalization_recovery_v10_source_expansion_roster(
            run_root=tmp_path,
            target_dir=target,
        )

    roster_path = target / "source-candidates.jsonl"
    lines = roster_path.read_bytes().splitlines(keepends=True)
    changed = json.loads(lines[0])
    changed["root_tree"] = "f" * 40
    lines[0] = canonical_json_line(changed)
    roster_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    payload = roster_path.read_bytes()
    stored["files"][0]["bytes"] = len(payload)
    stored["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_line(stored))
    forged_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="records"):
        read_normalization_recovery_v10_source_expansion_roster(
            target, expected_manifest_sha256=forged_sha)

    roster_path.write_bytes(b"".join(
        canonical_json_line(item)
        for item in outputs["source-candidates.jsonl"]))
    manifest_without_runtime_sha = dict(published)
    manifest_without_runtime_sha.pop("manifest_sha256")
    manifest_path.write_bytes(canonical_json_line(manifest_without_runtime_sha))
    (target / "extra.txt").write_text("x", encoding="utf-8")
    restored_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="physical inventory"):
        read_normalization_recovery_v10_source_expansion_roster(
            target, expected_manifest_sha256=restored_sha)


def test_v10_source_expansion_roster_rejects_non_k_run_root(
        tmp_path: Path) -> None:
    """正式 publisher 不得把来源名册写回 D 或临时盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v10_source_expansion_roster(
            run_root=tmp_path,
            target_dir=tmp_path / "artifact",
        )
