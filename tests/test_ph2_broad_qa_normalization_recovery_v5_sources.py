"""recovery-v5 本地化来源、held-out commitment 与隔离测试。"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_evaluation_commitment as commitment,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_libreoffice_source_pack as lo_pack,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_qt_source_pack as qt_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_records import (
    parse_normalization_recovery_v5_libreoffice_archive,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_qt_source_records import (
    QT_MODULES,
    parse_normalization_recovery_v5_qt_archive,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _zip(values: dict[str, bytes]) -> bytes:
    """构造 synthetic exact-path ZIP。"""
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path, payload in values.items():
            archive.writestr(path, payload)
    return target.getvalue()


def _po(*, traditional: bool) -> bytes:
    """构造含 identity、变长与 LibreOffice placeholder 的 PO。"""
    language = "zh-TW" if traditional else "zh-CN"
    phrase = "匯入資訊" if traditional else "导入信息"
    content = "內容" if traditional else "内容"
    return (
        'msgid ""\n'
        'msgstr ""\n'
        f'"Language: {language}\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        '\n'
        'msgctxt "identity"\n'
        'msgid "Keep"\n'
        'msgstr "保持"\n'
        '\n'
        'msgctxt "phrase"\n'
        'msgid "Import information"\n'
        f'msgstr "{phrase}"\n'
        '\n'
        'msgctxt "tagged"\n'
        'msgid "Product content"\n'
        f'msgstr "%PRODUCTNAME {content}"\n'
    ).encode("utf-8")


def _lo_archive() -> bytes:
    """构造 LibreOffice 固定三文件 synthetic archive。"""
    return _zip({
        "README": b"synthetic translations\n",
        "source/zh-CN/cui/messages.po": _po(traditional=False),
        "source/zh-TW/cui/messages.po": _po(traditional=True),
    })


def _ts(*, locale: str, populated: bool) -> bytes:
    """构造 active、inactive、numerus 与结构 token TS。"""
    if not populated:
        body = ""
    else:
        phrase = "匯入資訊" if locale == "zh_TW" else "导入信息"
        content = "內容" if locale == "zh_TW" else "内容"
        body = f"""
<context>
 <name>Main</name>
 <message><source>Keep</source><translation>保持</translation></message>
 <message><source>Import information</source><translation>{phrase}</translation></message>
 <message><source>Tagged</source><comment>button</comment><translation>%1 {content}</translation></message>
 <message><source>Ignored</source><translation type="unfinished"></translation></message>
 <message numerus="yes"><source>Items</source><translation><numerusform>%n</numerusform></translation></message>
</context>"""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE TS>\n'
        f'<TS version="2.1" language="{locale}">{body}\n</TS>\n'
    ).encode("utf-8")


def _license_rule() -> bytes:
    """构造 ordinary module 默认许可规则。"""
    value = [{
        "comment": "All other files",
        "location": {"": {
            "comment": "Default",
            "file type": "module and plugin",
            "spdx": [qt_pack.QT_LICENSE_EXPRESSION],
        }},
    }]
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _qt_archive() -> bytes:
    """构造八模块、两 locale 与 licenseRule synthetic archive。"""
    values = {"licenseRule.json": _license_rule()}
    for module in QT_MODULES:
        for locale in ("zh_CN", "zh_TW"):
            values[f"translations/{module}_{locale}.ts"] = _ts(
                locale=locale,
                populated=module == "assistant",
            )
    return _zip(values)


def _summary_subset(
        summary: dict[str, object],
        template: dict[str, object],
        ) -> dict[str, object]:
    """按 official gate 的 key 形成 synthetic census。"""
    return {key: summary[key] for key in template}


def test_shared_structure_preserves_markup_and_product_placeholder() -> None:
    """共享结构层保留 HTML、BBCode、Qt 与 LibreOffice placeholder 序。"""
    assert localization_structure_tokens(
        "<b>[code]%PRODUCTNAME %1 $[officename]</b>") == (
            "HTML_OPEN:b",
            "BBCODE_OPEN:code",
            "%PRODUCTNAME",
            "%1",
            "$[officename]",
            "HTML_CLOSE:b",
        )


def test_libreoffice_parser_aligns_po_and_preserves_source() -> None:
    """PO adapter 按完整 source identity 对齐且保留原 entry。"""
    files, pairs, summary = parse_normalization_recovery_v5_libreoffice_archive(
        _lo_archive())
    assert len(files) == 3
    assert len(pairs) == 3
    assert summary["training_eligible_pair_count"] == 3
    phrase = next(item for item in pairs
                  if item["source_identity"]["msgctxt"] == "phrase")
    assert phrase["zh_hant"]["msgstr"] == "匯入資訊"
    assert phrase["zh_hans"]["msgstr"] == "导入信息"
    tagged = next(item for item in pairs
                  if item["source_identity"]["msgctxt"] == "tagged")
    assert tagged["zh_hant_structure_tokens"] == ["%PRODUCTNAME"]


def test_qt_parser_freezes_active_plain_inventory_without_stripping() -> None:
    """TS adapter 排除 inactive/numerus，并保留 source/comment/结构。"""
    files, pairs, summary = parse_normalization_recovery_v5_qt_archive(
        _qt_archive())
    assert len(files) == 17
    assert len(pairs) == 3
    assert summary["module_pair_counts"] == {"assistant": 3}
    assert summary["training_eligible_pair_count"] == 3
    tagged = next(item for item in pairs
                  if item["source_identity"]["source"] == "Tagged")
    assert tagged["source_identity"]["comment"] == "button"
    assert tagged["zh_hans_structure_tokens"] == ["%1"]


def test_exact_archive_inventory_rejects_extra_member() -> None:
    """固定来源 archive 不接受未冻结的额外文件。"""
    values = {
        "README": b"source\n",
        "source/zh-CN/cui/messages.po": _po(traditional=False),
        "source/zh-TW/cui/messages.po": _po(traditional=True),
        "source/zh-CN/extra.po": b"extra\n",
    }
    with pytest.raises(BroadQaExternalDataError, match="inventory"):
        parse_normalization_recovery_v5_libreoffice_archive(_zip(values))


def test_source_packs_round_trip_and_qt_publishes_no_labels(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """两个 source pack 从 raw 重派生，Qt 只发布无 label identity。"""
    lo_payload = _lo_archive()
    lo_files, _lo_pairs, lo_summary = (
        parse_normalization_recovery_v5_libreoffice_archive(lo_payload))
    monkeypatch.setattr(lo_pack, "LIBREOFFICE_ARCHIVE_BYTES", len(lo_payload))
    monkeypatch.setattr(
        lo_pack, "LIBREOFFICE_ARCHIVE_SHA256",
        hashlib.sha256(lo_payload).hexdigest())
    monkeypatch.setattr(
        lo_pack, "LIBREOFFICE_OFFICIAL_SUMMARY",
        _summary_subset(lo_summary, lo_pack.LIBREOFFICE_OFFICIAL_SUMMARY))
    monkeypatch.setattr(lo_pack, "_require_k_root", lambda value: Path(value))
    lo_source = tmp_path / "lo.zip"
    lo_source.write_bytes(lo_payload)
    lo_target = tmp_path / "lo-pack"
    lo_pack.publish_normalization_recovery_v5_libreoffice_source_pack(
        run_root=tmp_path,
        archive_path=lo_source,
        target_dir=lo_target,
    )
    lo_manifest, stored_lo_files, stored_lo_pairs = (
        lo_pack.read_normalization_recovery_v5_libreoffice_source_pack(
            lo_target))
    assert lo_manifest["training_read_count"] == 0
    assert stored_lo_files == lo_files
    assert len(stored_lo_pairs) == 3

    qt_payload = _qt_archive()
    _qt_files, _qt_pairs, qt_summary = (
        parse_normalization_recovery_v5_qt_archive(qt_payload))
    license_payload = _license_rule()
    monkeypatch.setattr(qt_pack, "QT_ARCHIVE_BYTES", len(qt_payload))
    monkeypatch.setattr(
        qt_pack, "QT_ARCHIVE_SHA256",
        hashlib.sha256(qt_payload).hexdigest())
    monkeypatch.setattr(
        qt_pack, "QT_LICENSE_RULE_BYTES", len(license_payload))
    monkeypatch.setattr(
        qt_pack, "QT_LICENSE_RULE_GIT_BLOB_SHA1",
        git_blob_sha1(license_payload))
    monkeypatch.setattr(
        qt_pack, "QT_LICENSE_RULE_SHA256",
        hashlib.sha256(license_payload).hexdigest())
    monkeypatch.setattr(
        qt_pack, "QT_OFFICIAL_SUMMARY",
        _summary_subset(qt_summary, qt_pack.QT_OFFICIAL_SUMMARY))
    monkeypatch.setattr(qt_pack, "_require_k_root", lambda value: Path(value))
    qt_source = tmp_path / "qt.zip"
    qt_source.write_bytes(qt_payload)
    qt_target = tmp_path / "qt-pack"
    qt_pack.publish_normalization_recovery_v5_qt_source_pack(
        run_root=tmp_path,
        archive_path=qt_source,
        target_dir=qt_target,
    )
    qt_manifest, _stored_qt_files, inventory = (
        qt_pack.read_normalization_recovery_v5_qt_source_pack(qt_target))
    inventory_payload = (
        qt_target / "evaluation-inventory.identity.jsonl").read_bytes()
    assert qt_manifest["training_exclusion"]["learner_read_count"] == 0
    assert len(inventory) == 3
    assert "translation" not in inventory[0]
    assert "匯入資訊".encode("utf-8") not in inventory_payload
    assert not (qt_target / "translation-pairs.jsonl").exists()


def test_commitment_reads_only_qt_manifest(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """commitment 在 source 非 manifest 文件不可用时仍能冻结分母。"""
    qt_payload = _qt_archive()
    _files, _pairs, qt_summary = parse_normalization_recovery_v5_qt_archive(
        qt_payload)
    license_payload = _license_rule()
    archive_sha = hashlib.sha256(qt_payload).hexdigest()
    summary_subset = _summary_subset(
        qt_summary, qt_pack.QT_OFFICIAL_SUMMARY)
    for module in (qt_pack,):
        monkeypatch.setattr(module, "QT_ARCHIVE_BYTES", len(qt_payload))
        monkeypatch.setattr(module, "QT_ARCHIVE_SHA256", archive_sha)
        monkeypatch.setattr(module, "QT_LICENSE_RULE_BYTES", len(license_payload))
        monkeypatch.setattr(
            module, "QT_LICENSE_RULE_GIT_BLOB_SHA1",
            git_blob_sha1(license_payload))
        monkeypatch.setattr(
            module, "QT_LICENSE_RULE_SHA256",
            hashlib.sha256(license_payload).hexdigest())
        monkeypatch.setattr(module, "QT_OFFICIAL_SUMMARY", summary_subset)
    monkeypatch.setattr(commitment, "QT_ARCHIVE_SHA256", archive_sha)
    monkeypatch.setattr(commitment, "QT_OFFICIAL_SUMMARY", summary_subset)
    monkeypatch.setattr(qt_pack, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        commitment, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "qt.zip"
    source.write_bytes(qt_payload)
    pack = tmp_path / "qt-pack"
    source_manifest = qt_pack.publish_normalization_recovery_v5_qt_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=pack,
    )
    for path in pack.iterdir():
        if path.name != "manifest.json":
            path.unlink()
    target = tmp_path / "commitment"
    manifest = commitment.publish_normalization_recovery_v5_evaluation_commitment(
        run_root=tmp_path,
        qt_source_pack_dir=pack,
        expected_qt_source_manifest_sha256=source_manifest["manifest_sha256"],
        target_dir=target,
    )
    assert manifest["source_non_manifest_file_read_count"] == 0
    assert manifest["training_source_read_count"] == 0
    read = commitment.read_normalization_recovery_v5_evaluation_commitment(
        target,
        qt_source_pack_dir=pack,
        expected_qt_source_manifest_sha256=source_manifest["manifest_sha256"],
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    assert read == manifest
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        commitment.publish_normalization_recovery_v5_evaluation_commitment(
            run_root=tmp_path,
            qt_source_pack_dir=pack,
            expected_qt_source_manifest_sha256=(
                source_manifest["manifest_sha256"]),
            target_dir=target,
        )


def test_manifest_tamper_cannot_rewrite_raw_derived_records(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改 pair 与 manifest 仍不能绕过 LibreOffice raw 重派生。"""
    payload = _lo_archive()
    _files, _pairs, summary = parse_normalization_recovery_v5_libreoffice_archive(
        payload)
    monkeypatch.setattr(lo_pack, "LIBREOFFICE_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        lo_pack, "LIBREOFFICE_ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        lo_pack, "LIBREOFFICE_OFFICIAL_SUMMARY",
        _summary_subset(summary, lo_pack.LIBREOFFICE_OFFICIAL_SUMMARY))
    monkeypatch.setattr(lo_pack, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "lo.zip"
    source.write_bytes(payload)
    target = tmp_path / "pack"
    lo_pack.publish_normalization_recovery_v5_libreoffice_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=target,
    )
    pair_path = target / "translation-pairs.jsonl"
    lines = pair_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["zh_hans"]["msgstr"] = "篡改"
    lines[0] = canonical_json_line(value)
    pair_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = next(item for item in manifest["files"]
                    if item["relative_path"] == "translation-pairs.jsonl")
    artifact["bytes"] = pair_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        lo_pack.read_normalization_recovery_v5_libreoffice_source_pack(target)


def test_official_libreoffice_archive_identity_when_available() -> None:
    """提供 Git 外 fixture 时核对固定 PO archive 与完整 census。"""
    value = os.environ.get("PURE_INTEGER_AI_LIBREOFFICE_CUI_ARCHIVE")
    if not value:
        pytest.skip("official LibreOffice CUI archive fixture is unavailable")
    payload = Path(value).read_bytes()
    files, pairs, summary = parse_normalization_recovery_v5_libreoffice_archive(
        payload)
    assert len(payload) == lo_pack.LIBREOFFICE_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == (
        lo_pack.LIBREOFFICE_ARCHIVE_SHA256)
    assert len(files) == 3
    assert len(pairs) == 3_887
    assert _summary_subset(
        summary, lo_pack.LIBREOFFICE_OFFICIAL_SUMMARY
    ) == lo_pack.LIBREOFFICE_OFFICIAL_SUMMARY


def test_official_qt_archive_identity_when_available() -> None:
    """提供 Git 外 fixture 时核对固定 TS archive 与 held-out census。"""
    value = os.environ.get("PURE_INTEGER_AI_QT_TRANSLATIONS_ARCHIVE")
    if not value:
        pytest.skip("official Qt translations archive fixture is unavailable")
    payload = Path(value).read_bytes()
    files, pairs, summary = parse_normalization_recovery_v5_qt_archive(payload)
    assert len(payload) == qt_pack.QT_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == qt_pack.QT_ARCHIVE_SHA256
    assert len(files) == 17
    assert len(pairs) == 3_531
    assert _summary_subset(
        summary, qt_pack.QT_OFFICIAL_SUMMARY
    ) == qt_pack.QT_OFFICIAL_SUMMARY
