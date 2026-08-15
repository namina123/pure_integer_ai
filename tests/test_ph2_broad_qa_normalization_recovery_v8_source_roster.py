"""覆盖 recovery-v8 新 TRAIN source roster commitment。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster import (
    publish_normalization_recovery_v8_source_roster,
    read_normalization_recovery_v8_source_roster,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_records import (
    derive_normalization_recovery_v8_source_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def test_v8_source_roster_freezes_three_independent_surface_free_families(
        ) -> None:
    """三家新来源均有简繁、英文 source binding 与独立上游。"""
    records, census = derive_normalization_recovery_v8_source_roster()
    assert [item["source_family"] for item in records] == [
        "BITCOIN_CORE_PROJECT",
        "QBITTORRENT_PROJECT",
        "STELLARIUM_PROJECT",
    ]
    assert census == {
        "family_independence_group_count": 3,
        "locale_blob_content_read_count": 0,
        "locale_file_count": 26,
        "locale_pair_count": 13,
        "locale_total_bytes": 20_765_660,
        "license_file_count": 5,
        "official_source_bound_family_count": 3,
        "selected_source_family_count": 3,
        "surface_published": 0,
    }
    assert all(item["locale_blob_content_read_count"] == 0
               for item in records)
    assert all(item["surface_published"] == 0 for item in records)
    forbidden = {"input_text", "output_text", "translation", "surface_text"}
    assert all(not forbidden.intersection(item) for item in records)
    assert {item["official_source_binding"] for item in records} == {
        "GETTEXT_MSGID", "QT_TS_SOURCE_ELEMENT"}


def test_v8_source_roster_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader逐字段一致，并拒绝覆盖和同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster as module

    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    target = tmp_path / "roster"
    published = publish_normalization_recovery_v8_source_roster(
        run_root=tmp_path, target_dir=target)
    manifest, outputs = read_normalization_recovery_v8_source_roster(
        target,
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert len(outputs["source-roster.jsonl"]) == 3
    assert manifest["locale_blob_content_read_count"] == 0
    assert manifest["consumed_source_exclusion"][
        "qt_individual_or_derivative_read_count"] == 0
    assert manifest["consumed_source_exclusion"][
        "audacity_individual_or_translation_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v8_source_roster(
            run_root=tmp_path, target_dir=target)

    path = target / "source-roster.jsonl"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    records[0]["root_tree"] = "0" * 40
    payload = b"".join(canonical_json_line(item) for item in records)
    path.write_bytes(payload)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(payload)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_source_roster(
            target,
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v8_source_roster_rejects_non_k_run_root(tmp_path: Path) -> None:
    """真实 publisher 不得把 source roster artifact 回退到 D/临时盘。"""
    with pytest.raises(BroadQaExternalDataError, match="必须在 K 盘"):
        publish_normalization_recovery_v8_source_roster(
            run_root=tmp_path,
            target_dir=tmp_path / "roster",
        )
