"""覆盖 recovery-v8 source roster v2 replacement commitment。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_v2 import (
    publish_normalization_recovery_v8_source_roster_v2,
    read_normalization_recovery_v8_source_roster_v2,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_v2_records import (
    derive_normalization_recovery_v8_source_roster_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _v1_record(family: str) -> dict[str, object]:
    """构造一个最小v1 predecessor roster record。"""
    return {
        "commit": family.lower().ljust(40, "0")[:40],
        "commit_date": "2026-08-15T00:00:00Z",
        "family_independence_group": family + "_UPSTREAM",
        "license": {
            "expression": "MIT",
            "files": [{
                "bytes": 1,
                "git_blob_sha1": "1" * 40,
                "relative_path": "COPYING",
            }],
            "primary_bytes": 1,
            "primary_sha256": "2" * 64,
        },
        "locale_blob_content_read_count": 0,
        "locale_file_count": 2,
        "locale_files": [{
            "bytes": 2,
            "git_blob_sha1": character * 40,
            "relative_path": f"{family}/{locale}",
        } for character, locale in (("3", "zh_CN"), ("4", "zh_TW"))],
        "locale_pair_count": 1,
        "official_source_binding": "QT_TS_SOURCE_ELEMENT",
        "parser_identity": "QT_TS_XML_V1",
        "record_id": (family.lower().encode("utf-8").hex().ljust(64, "0")[:64]),
        "record_kind": "V1",
        "repository": f"https://github.com/example/{family}.git",
        "root_tree": "5" * 40,
        "selection_status": "SELECTED",
        "source_family": family,
        "source_policy_scope": family + "_SCOPE",
        "surface_published": 0,
        "target_scope": "TARGET",
    }


def _inputs():
    """构造v1三家roster与一拒两过content records。"""
    families = (
        "BITCOIN_CORE_PROJECT",
        "QBITTORRENT_PROJECT",
        "STELLARIUM_PROJECT",
    )
    roster = tuple(_v1_record(family) for family in families)
    content = tuple({
        "content_outcome": (
            "REJECTED_ZERO_ACTIVE_COMMON_PAIR"
            if family == "BITCOIN_CORE_PROJECT"
            else "PASS_NONZERO_ACTIVE_COMMON_PAIR"),
        "source_family": family,
    } for family in families)
    return roster, content


def test_v8_roster_v2_replaces_only_rejected_bitcoin() -> None:
    """v2继承两家PASS并加入未读KeePassXC。"""
    roster, content = _inputs()
    records, summary = derive_normalization_recovery_v8_source_roster_v2(
        v1_roster=roster, content_records=content)
    assert [item["source_family"] for item in records] == [
        "KEEPASSXC_PROJECT", "QBITTORRENT_PROJECT", "STELLARIUM_PROJECT"]
    by_family = {item["source_family"]: item for item in records}
    assert by_family["KEEPASSXC_PROJECT"][
        "locale_blob_content_read_count"] == 0
    assert by_family["KEEPASSXC_PROJECT"]["commit"] == (
        "0e1510d71ab63ce1edddb71257bce34a7cee2f0d")
    assert by_family["QBITTORRENT_PROJECT"][
        "selection_status"] == "INHERITED_V1_CONTENT_PASS"
    assert "BITCOIN_CORE_PROJECT" not in by_family
    assert summary["content_pass_inherited_family_count"] == 2
    assert summary["replacement_locale_blob_content_read_count"] == 0


def test_v8_roster_v2_rejects_wrong_predecessor_outcome() -> None:
    """Bitcoin非reject或继承family非PASS时拒绝派生v2。"""
    roster, content = _inputs()
    changed = tuple(
        {**item, "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR"}
        if item["source_family"] == "BITCOIN_CORE_PROJECT" else item
        for item in content)
    with pytest.raises(BroadQaExternalDataError, match="outcome 漂移"):
        derive_normalization_recovery_v8_source_roster_v2(
            v1_roster=roster, content_records=changed)


def test_v8_roster_v2_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader逐字段一致，并拒绝覆盖和同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_roster_v2 as module

    roster, content_records = _inputs()
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_inputs", lambda **_kwargs: (roster, content_records))
    v1 = tmp_path / "v1"
    content = tmp_path / "content"
    v1.mkdir()
    content.mkdir()
    target = tmp_path / "v2"
    published = publish_normalization_recovery_v8_source_roster_v2(
        run_root=tmp_path,
        v1_roster_dir=v1,
        content_audit_dir=content,
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v8_source_roster_v2(
        target,
        v1_roster_dir=v1,
        content_audit_dir=content,
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert len(outputs["source-roster-v2.jsonl"]) == 3
    assert manifest["keepassxc_locale_blob_content_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        publish_normalization_recovery_v8_source_roster_v2(
            run_root=tmp_path,
            v1_roster_dir=v1,
            content_audit_dir=content,
            target_dir=target,
        )

    path = target / "source-roster-v2.jsonl"
    values = [json.loads(line) for line in path.read_bytes().splitlines()]
    values[0]["root_tree"] = "0" * 40
    changed = b"".join(canonical_json_line(item) for item in values)
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_source_roster_v2(
            target,
            v1_roster_dir=v1,
            content_audit_dir=content,
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )
