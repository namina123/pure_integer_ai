"""normalization recovery-v3 Thunderbird source pack 测试。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack as module,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _archive(*, malformed: bool = False) -> bytes:
    """构造含 Fluent 与必须保留的非 Fluent 文件的双 locale archive。"""
    cn = "pair = 导入信息\nidentity = 保持内容\nwith-var = 项目 { $name }\n"
    tw = "pair = 匯入資訊\nidentity = 保持内容\nwith-var = 項目 { $name }\n"
    if malformed:
        cn += "broken = {\n"
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE", b"Mozilla Public License Version 2.0\n")
        archive.writestr("zh-CN/app/test.ftl", cn.encode())
        archive.writestr("zh-TW/app/test.ftl", tw.encode())
        archive.writestr("zh-CN/app/legacy.dtd", b"<!ENTITY item 'value'>\n")
        archive.writestr("zh-TW/app/legacy.dtd", b"<!ENTITY item 'value'>\n")
    return target.getvalue()


def _freeze_synthetic_identity(
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes,
        ) -> None:
    """注入 synthetic archive 身份，不放宽 parser/reader。"""
    files, pairs, summary = (
        module.parse_normalization_recovery_v3_thunderbird_archive(payload))
    monkeypatch.setattr(module, "THUNDERBIRD_L10N_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_ARCHIVE_FILE_COUNT", len(files))
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_FILE_EXTENSION_COUNTS",
        summary["file_extension_counts"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_LICENSE_BYTES", summary["license_bytes"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_LICENSE_SHA256", summary["license_sha256"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_LICENSE_BLOB_SHA1",
        summary["license_git_blob_sha1"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_LOCALE_TREES", summary["locale_trees"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_COMMON_PATTERN_COUNT", len(pairs))
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_STRUCTURE_EQUAL_COUNT",
        summary["structure_equal_count"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_PLAIN_PAIR_COUNT",
        summary["plain_pair_count"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_NONIDENTITY_PLAIN_PAIR_COUNT",
        summary["nonidentity_plain_pair_count"])
    monkeypatch.setattr(
        module, "THUNDERBIRD_L10N_IDENTITY_PLAIN_PAIR_COUNT",
        summary["identity_plain_pair_count"])
    summaries = summary["locale_summaries"]
    monkeypatch.setattr(module, "THUNDERBIRD_L10N_FTL_FILE_COUNTS", {
        key: value["ftl_file_count"] for key, value in summaries.items()
    })
    monkeypatch.setattr(module, "THUNDERBIRD_L10N_PATTERN_COUNTS", {
        key: value["pattern_count"] for key, value in summaries.items()
    })


def _publish(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> Path:
    """发布 synthetic source pack。"""
    payload = _archive()
    _freeze_synthetic_identity(monkeypatch, payload)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    target = tmp_path / "pack"
    module.publish_normalization_recovery_v3_thunderbird_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=target,
    )
    return target


def test_parser_preserves_fluent_structure_and_non_fluent_boundary() -> None:
    """Fluent 走 AST，对其他格式只保留原文件和格式边界。"""
    files, pairs, summary = (
        module.parse_normalization_recovery_v3_thunderbird_archive(_archive()))
    assert len(files) == 5
    assert summary["file_extension_counts"] == {
        ".dtd": 2, ".ftl": 2, "<none>": 1}
    assert summary["source_format_policy"] == {
        "fluent_ast_aligned": 1,
        "non_fluent_files_preserved_not_parsed": 1,
        "non_fluent_plain_text_training_allowed": 0,
    }
    assert all(item["record_kind"]
               == module.THUNDERBIRD_PATTERN_PAIR_RECORD_KIND
               for item in pairs)
    pair = next(item for item in pairs if item["message_id"] == "pair")
    assert pair["plain_pair_eligible"] == 1
    placeable = next(item for item in pairs
                     if item["message_id"] == "with-var")
    assert placeable["structure_equal"] == 1
    assert placeable["plain_pair_eligible"] == 0
    with pytest.raises(BroadQaExternalDataError, match="parser Junk"):
        module.parse_normalization_recovery_v3_thunderbird_archive(
            _archive(malformed=True))


def test_source_pack_round_trip_is_immutable_and_zero_read(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source pack 从 raw archive 重派生且不读取评测/训练消费者。"""
    source = _publish(tmp_path, monkeypatch)
    manifest, files, pairs = (
        module.read_normalization_recovery_v3_thunderbird_source_pack(source))
    assert manifest["evaluation_or_reserve_read_count"] == 0
    assert manifest["training_read_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(files) == 5
    assert pairs
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        module.publish_normalization_recovery_v3_thunderbird_source_pack(
            run_root=tmp_path,
            archive_path=source / module.THUNDERBIRD_L10N_ARCHIVE_NAME,
            target_dir=source,
        )


def test_reader_rejects_synchronized_pair_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改 pair 与 manifest 仍不能绕过 raw archive 重派生。"""
    source = _publish(tmp_path, monkeypatch)
    pair_path = source / "pattern-pairs.jsonl"
    lines = pair_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["zh_cn"]["surface_text"] = "篡改"
    lines[0] = canonical_json_line(value)
    pair_path.write_bytes(b"".join(lines))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = next(item for item in manifest["files"]
                    if item["relative_path"] == "pattern-pairs.jsonl")
    artifact["bytes"] = pair_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        module.read_normalization_recovery_v3_thunderbird_source_pack(source)
