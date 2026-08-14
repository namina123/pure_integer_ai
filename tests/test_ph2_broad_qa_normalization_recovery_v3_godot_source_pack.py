"""normalization recovery-v3 Godot PO source pack 测试。"""
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
    ph2_broad_qa_normalization_recovery_v3_godot_source_pack as module,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _po(locale: str, *, traditional: bool) -> bytes:
    """构造含 identity、变长、标记和 plural 的 synthetic PO。"""
    conversion = "匯入資訊" if traditional else "导入信息"
    tagged = "[hint=內容]%s[/hint]" if traditional else "[hint=内容]%s[/hint]"
    value = f'''msgid ""
msgstr ""
"Language: {locale}\\n"
"Content-Type: text/plain; charset=UTF-8\\n"

#: editor/a.cpp:1
msgid "Import information"
msgstr "{conversion}"

msgid "Keep"
msgstr "保持"

msgid "Tagged"
msgstr "{tagged}"

msgid "File"
msgid_plural "Files"
msgstr[0] "文件"
'''
    return value.encode()


def _archive() -> bytes:
    """构造许可与两份 PO 的受限 archive。"""
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE.txt", b"MIT License\n")
        archive.writestr(
            "editor/translations/editor/zh_Hans.po",
            _po("zh_Hans", traditional=False),
        )
        archive.writestr(
            "editor/translations/editor/zh_Hant.po",
            _po("zh_Hant", traditional=True),
        )
    return target.getvalue()


def _freeze_synthetic_identity(
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes,
        ) -> None:
    """注入 synthetic archive、blob 与库存身份。"""
    files, pairs, summary = module.parse_normalization_recovery_v3_godot_archive(
        payload)
    monkeypatch.setattr(module, "GODOT_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        module, "GODOT_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "GODOT_SOURCE_FILES", {
        item["relative_path"]: {
            "bytes": item["bytes"],
            "git_blob_sha1": item["git_blob_sha1"],
            "sha256": item["sha256"],
        } for item in files
    })
    monkeypatch.setattr(module, "GODOT_ENTRY_COUNTS", {
        key: value["entry_count"]
        for key, value in summary["locale_summaries"].items()
    })
    for name, key in (
        ("GODOT_COMMON_ENTRY_COUNT", "common_entry_count"),
        ("GODOT_PLURAL_ENTRY_COUNT", "plural_pair_count"),
        ("GODOT_SIMPLE_PAIR_COUNT", "simple_pair_count"),
        ("GODOT_STRUCTURE_EQUAL_COUNT", "structure_equal_count"),
        ("GODOT_TRAINING_ELIGIBLE_COUNT", "training_eligible_count"),
        ("GODOT_NONIDENTITY_PAIR_COUNT", "nonidentity_pair_count"),
        ("GODOT_IDENTITY_PAIR_COUNT", "identity_pair_count"),
        ("GODOT_EQUAL_LENGTH_PAIR_COUNT", "equal_length_pair_count"),
        ("GODOT_VARIABLE_LENGTH_PAIR_COUNT", "variable_length_pair_count"),
    ):
        monkeypatch.setattr(module, name, summary[key])


def _publish(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> Path:
    """发布 synthetic Godot source pack。"""
    payload = _archive()
    _freeze_synthetic_identity(monkeypatch, payload)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    target = tmp_path / "pack"
    module.publish_normalization_recovery_v3_godot_source_pack(
        run_root=tmp_path, archive_path=source, target_dir=target)
    return target


def test_parser_aligns_po_and_preserves_structure() -> None:
    """同英文 source 对齐，标记参数不被当作语言表面丢弃。"""
    files, pairs, summary = module.parse_normalization_recovery_v3_godot_archive(
        _archive())
    assert len(files) == 3
    assert len(pairs) == 4
    assert summary["plural_pair_count"] == 1
    assert summary["simple_pair_count"] == 3
    tagged = next(item for item in pairs
                  if item["source_identity"]["msgid"] == "Tagged")
    assert tagged["structure_equal"] == 1
    assert tagged["zh_hans"]["structure_tokens"] == [
        "BBCODE_OPEN:hint", "%s", "BBCODE_CLOSE:hint"]
    plural = next(item for item in pairs if item["plural"] == 1)
    assert plural["training_eligible"] == 0


def test_source_pack_round_trip_is_immutable_and_zero_read(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """raw blob 重派生，且 source pack 不读取评测或 learner。"""
    source = _publish(tmp_path, monkeypatch)
    manifest, files, pairs = (
        module.read_normalization_recovery_v3_godot_source_pack(source))
    assert manifest["evaluation_or_reserve_read_count"] == 0
    assert manifest["training_read_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(files) == 3
    assert len(pairs) == 4
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        module.publish_normalization_recovery_v3_godot_source_pack(
            run_root=tmp_path,
            archive_path=source / module.GODOT_ARCHIVE_NAME,
            target_dir=source,
        )


def test_reader_rejects_synchronized_pair_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改派生 pair 与 manifest 仍不能绕过 raw PO。"""
    source = _publish(tmp_path, monkeypatch)
    pair_path = source / "translation-pairs.jsonl"
    lines = pair_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["zh_hans"]["msgstr"] = "篡改"
    lines[0] = canonical_json_line(value)
    pair_path.write_bytes(b"".join(lines))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = next(item for item in manifest["files"]
                    if item["relative_path"] == "translation-pairs.jsonl")
    artifact["bytes"] = pair_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        module.read_normalization_recovery_v3_godot_source_pack(source)
