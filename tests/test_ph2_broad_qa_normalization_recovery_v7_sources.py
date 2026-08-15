"""覆盖 recovery-v7 VLC held-out 来源与标签盲 commitment。"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import polib
import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_evaluation_commitment
    as commitment,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_vlc_source_pack as vlc_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vlc_source_records import (
    VLC_LICENSE_EXPRESSION,
    parse_normalization_recovery_v7_vlc_archive,
    validate_vlc_license,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _copying() -> bytes:
    """构造含固定 v2-or-later 声明的最小测试许可文本。"""
    return (
        b"                    GNU GENERAL PUBLIC LICENSE\n"
        b"                       Version 2, June 1991\n"
        b"\nSynthetic test body.\n"
        b"    it under the terms of the GNU General Public License as published by\n"
        b"    the Free Software Foundation; either version 2 of the License, or\n"
        b"    (at your option) any later version.\n")


def _entry(
        msgid: str,
        msgstr: str,
        *,
        msgctxt: str = "",
        fuzzy: bool = False,
        obsolete: bool = False,
        plural: bool = False,
        ) -> polib.POEntry:
    """构造一个 synthetic VLC PO entry。"""
    return polib.POEntry(
        msgctxt=msgctxt or None,
        msgid=msgid,
        msgid_plural=f"{msgid} plural" if plural else "",
        msgstr="" if plural else msgstr,
        msgstr_plural={0: msgstr} if plural else {},
        flags=["fuzzy"] if fuzzy else [],
        obsolete=obsolete,
    )


def _po(
        *,
        locale: str,
        traditional: bool,
        duplicate: bool = False,
        ) -> bytes:
    """构造含 plain 与各类排除项的固定 synthetic PO。"""
    po = polib.POFile(wrapwidth=0)
    po.header = (
        "Synthetic Chinese translation\n"
        "This file is distributed under the same license as the VLC package.")
    po.metadata = {
        "Content-Type": "text/plain; charset=UTF-8",
        "Language": locale,
        "Plural-Forms": "nplurals=1; plural=0;",
        "Project-Id-Version": "vlc synthetic",
    }
    po.extend([
        _entry("identity", "相同", msgctxt="menu"),
        _entry("variable", "繁體較長" if traditional else "简体"),
        _entry("structured", "開啟 %s" if traditional else "打开 %s"),
        _entry("fuzzy", "模糊", fuzzy=True),
        _entry("obsolete", "舊" if traditional else "旧", obsolete=True),
        _entry("plural", "複數" if traditional else "复数", plural=True),
        _entry("empty", "" if traditional else "非空"),
    ])
    if duplicate:
        po.append(_entry("identity", "重複", msgctxt="menu"))
    return str(po).encode("utf-8")


def _zip(
        *,
        duplicate_hant: bool = False,
        extra: bool = False,
        ) -> bytes:
    """构造严格 inventory 的 synthetic VLC source ZIP。"""
    values = {
        "COPYING": _copying(),
        "po/zh_CN.po": _po(locale="zh_CN", traditional=False),
        "po/zh_TW.po": _po(
            locale="zh_TW",
            traditional=True,
            duplicate=duplicate_hant,
        ),
    }
    if extra:
        values["po/extra.po"] = b"extra\n"
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in values.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload)
    return target.getvalue()


def _source_files(
        payload: bytes,
        ) -> dict[str, dict[str, object]]:
    """从 synthetic archive 派生 pack 常量形状的物理 identity。"""
    file_records, _pairs, _summary = (
        parse_normalization_recovery_v7_vlc_archive(payload))
    return {
        str(item["relative_path"]): {
            "bytes": item["bytes"],
            "git_blob_sha1": item["git_blob_sha1"],
            "sha256": item["sha256"],
        }
        for item in file_records
    }


def _patch_official(
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes,
        ) -> dict[str, object]:
    """把 source pack/commitment 固定常量切到 synthetic archive。"""
    _files, _pairs, summary = parse_normalization_recovery_v7_vlc_archive(
        payload)
    source_files = _source_files(payload)
    archive_sha = hashlib.sha256(payload).hexdigest()
    for module in (vlc_pack, commitment):
        monkeypatch.setattr(module, "VLC_ARCHIVE_SHA256", archive_sha)
        monkeypatch.setattr(module, "VLC_OFFICIAL_SUMMARY", summary)
        monkeypatch.setattr(module, "VLC_SOURCE_FILES", source_files)
    monkeypatch.setattr(vlc_pack, "VLC_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(vlc_pack, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        commitment, "_require_k_root", lambda value: Path(value))
    return summary


def test_vlc_adapter_aligns_plain_pairs_and_preserves_structure() -> None:
    """VLC adapter 排除 plural/fuzzy/empty/obsolete 并保留结构。"""
    files, pairs, summary = parse_normalization_recovery_v7_vlc_archive(
        _zip())
    assert len(files) == 3
    assert len(pairs) == 3
    assert summary["common_identity_count"] == 7
    assert summary["excluded_common_pair_counts"] == {
        "any": 4,
        "empty": 2,
        "fuzzy": 1,
        "obsolete": 1,
        "plural": 1,
    }
    assert summary["plain_pair_count"] == 3
    structured = next(
        item for item in pairs
        if item["source_identity"]["msgid"] == "structured")
    assert structured["zh_hans_structure_tokens"] == ["%s"]
    assert structured["zh_hant_structure_tokens"] == ["%s"]
    assert structured["structure_equal"] == 1


def test_vlc_adapter_rejects_duplicate_identity_and_extra_member() -> None:
    """重复 source identity 与额外 archive member 都 fail closed。"""
    with pytest.raises(BroadQaExternalDataError, match="identity 重复"):
        parse_normalization_recovery_v7_vlc_archive(
            _zip(duplicate_hant=True))
    with pytest.raises(BroadQaExternalDataError, match="inventory"):
        parse_normalization_recovery_v7_vlc_archive(_zip(extra=True))


def test_vlc_license_expression_is_structurally_verified() -> None:
    """COPYING 必须同时具有 GPL v2 title 与 or-later 声明。"""
    validate_vlc_license(
        _copying(), expected_expression=VLC_LICENSE_EXPRESSION)
    with pytest.raises(BroadQaExternalDataError, match="license 表达式"):
        validate_vlc_license(
            _copying().replace(b"any later version", b"version 2 only"),
            expected_expression=VLC_LICENSE_EXPRESSION,
        )


def test_vlc_source_pack_round_trip_publishes_no_labels(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source pack 从 raw 重派生，公开 roster 不含简繁 translation。"""
    payload = _zip()
    summary = _patch_official(monkeypatch, payload)
    source = tmp_path / "vlc.zip"
    source.write_bytes(payload)
    target = tmp_path / "vlc-pack"
    manifest = vlc_pack.publish_normalization_recovery_v7_vlc_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=target,
    )
    read_manifest, stored_files, inventory = (
        vlc_pack.read_normalization_recovery_v7_vlc_source_pack(target))
    inventory_payload = (
        target / "evaluation-inventory.identity.jsonl").read_bytes()
    assert read_manifest == manifest
    assert len(stored_files) == 3
    assert len(inventory) == summary["plain_pair_count"]
    assert manifest["selection_boundary"][
        "source_selected_before_translation_label_parse"] == 1
    assert manifest["training_exclusion"][
        "learner_profiler_selector_case_browser_read_count"] == 0
    assert "msgstr" not in inventory[0]
    assert "zh_hans" not in inventory[0]
    assert "zh_hant" not in inventory[0]
    assert "繁體較長".encode("utf-8") not in inventory_payload
    assert "简体".encode("utf-8") not in inventory_payload
    assert not (target / "translation-pairs.jsonl").exists()
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        vlc_pack.publish_normalization_recovery_v7_vlc_source_pack(
            run_root=tmp_path,
            archive_path=source,
            target_dir=target,
        )


def test_vlc_source_pack_rejects_roster_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改 roster 物理摘要仍不能绕过 raw 重派生。"""
    payload = _zip()
    _patch_official(monkeypatch, payload)
    source = tmp_path / "vlc.zip"
    source.write_bytes(payload)
    target = tmp_path / "vlc-pack"
    vlc_pack.publish_normalization_recovery_v7_vlc_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=target,
    )
    identity_path = target / "evaluation-inventory.identity.jsonl"
    lines = identity_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["pair_id"] = "0" * 64
    lines[0] = canonical_json_line(value)
    identity_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = next(
        item for item in manifest["files"]
        if item["relative_path"] == identity_path.name)
    artifact["bytes"] = identity_path.stat().st_size
    artifact["sha256"] = sha256_hex(identity_path.read_bytes())
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        vlc_pack.read_normalization_recovery_v7_vlc_source_pack(target)


def test_v7_commitment_reads_only_vlc_manifest(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """commitment 在 source 非 manifest 文件不可用时仍冻结完整分母。"""
    payload = _zip()
    summary = _patch_official(monkeypatch, payload)
    source = tmp_path / "vlc.zip"
    source.write_bytes(payload)
    pack = tmp_path / "vlc-pack"
    source_manifest = (
        vlc_pack.publish_normalization_recovery_v7_vlc_source_pack(
            run_root=tmp_path,
            archive_path=source,
            target_dir=pack,
        ))
    for path in pack.iterdir():
        if path.name != "manifest.json":
            path.unlink()
    target = tmp_path / "commitment"
    manifest = (
        commitment.publish_normalization_recovery_v7_evaluation_commitment(
            run_root=tmp_path,
            vlc_source_pack_dir=pack,
            expected_vlc_source_manifest_sha256=source_manifest[
                "manifest_sha256"],
            target_dir=target,
        ))
    assert manifest["source_non_manifest_file_read_count"] == 0
    assert manifest["training_source_read_count"] == 0
    assert manifest["denominator"]["record_count"] == summary[
        "plain_pair_count"]
    read = commitment.read_normalization_recovery_v7_evaluation_commitment(
        target,
        vlc_source_pack_dir=pack,
        expected_vlc_source_manifest_sha256=source_manifest[
            "manifest_sha256"],
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    assert read == manifest
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        commitment.publish_normalization_recovery_v7_evaluation_commitment(
            run_root=tmp_path,
            vlc_source_pack_dir=pack,
            expected_vlc_source_manifest_sha256=source_manifest[
                "manifest_sha256"],
            target_dir=target,
        )


def test_official_vlc_archive_identity_when_available() -> None:
    """提供 Git 外 fixture 时核对固定 VLC archive 与完整 census。"""
    value = os.environ.get("PURE_INTEGER_AI_VLC_PO_ARCHIVE")
    if not value:
        pytest.skip("official VLC PO archive fixture is unavailable")
    payload = Path(value).read_bytes()
    files, pairs, summary = parse_normalization_recovery_v7_vlc_archive(
        payload)
    assert len(payload) == vlc_pack.VLC_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == vlc_pack.VLC_ARCHIVE_SHA256
    assert len(files) == 3
    assert len(pairs) == 3_656
    assert all(
        summary[key] == expected
        for key, expected in vlc_pack.VLC_OFFICIAL_SUMMARY.items())
    assert {
        item["relative_path"]: item["git_blob_sha1"] for item in files
    } == {
        path: values["git_blob_sha1"]
        for path, values in vlc_pack.VLC_SOURCE_FILES.items()
    }


def test_source_file_hash_helpers_match_synthetic_archive() -> None:
    """测试 fixture 的 COPYING identity 可由公开 hash 规则复算。"""
    payload = _copying()
    assert git_blob_sha1(payload) == hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()
