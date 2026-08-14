"""normalization recovery-v4 VS Code JSON source pack 测试。"""
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
    ph2_broad_qa_normalization_recovery_v4_vscode_source_pack as module,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _translation(*, traditional: bool) -> bytes:
    """构造含 identity、变长与结构 token 的 synthetic translation JSON。"""
    contents = {
        "module": {
            "identity": "保持",
            "phrase": "匯入資訊" if traditional else "导入信息",
            "tagged": (
                "<b>{0}</b> $(check) 內容"
                if traditional else "<b>{0}</b> $(check) 内容"),
        },
    }
    value = {
        "": ["generated"],
        "version": "1.0.0",
        "contents": contents,
    }
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _archive() -> bytes:
    """构造许可与一对同路径 JSON 的受限 archive。"""
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE.md", b"MIT License\n")
        archive.writestr("README.md", b"Synthetic localization source\n")
        for locale, traditional in (("zh-hans", False), ("zh-hant", True)):
            root = f"i18n/vscode-language-pack-{locale}"
            archive.writestr(f"{root}/package.json", b"{}\n")
            archive.writestr(
                f"{root}/translations/main.i18n.json",
                _translation(traditional=traditional),
            )
    return target.getvalue()


def _git_blob_sha1(payload: bytes) -> str:
    """按 Git blob 编码形成 synthetic 许可 identity。"""
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def _freeze_synthetic_identity(
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes,
        ) -> None:
    """注入 synthetic archive、许可与 census 身份。"""
    _files, _pairs, summary = (
        module.parse_normalization_recovery_v4_vscode_archive(payload))
    license_payload = b"MIT License\n"
    monkeypatch.setattr(module, "VSCODE_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        module, "VSCODE_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "VSCODE_LICENSE_BYTES", len(license_payload))
    monkeypatch.setattr(
        module, "VSCODE_LICENSE_GIT_BLOB_SHA1",
        _git_blob_sha1(license_payload))
    monkeypatch.setattr(
        module, "VSCODE_LICENSE_SHA256",
        hashlib.sha256(license_payload).hexdigest())
    monkeypatch.setattr(module, "VSCODE_OFFICIAL_SUMMARY", summary)


def _publish(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> Path:
    """发布 synthetic VS Code source pack。"""
    payload = _archive()
    _freeze_synthetic_identity(monkeypatch, payload)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    target = tmp_path / "pack"
    module.publish_normalization_recovery_v4_vscode_source_pack(
        run_root=tmp_path,
        archive_path=source,
        target_dir=target,
    )
    return target


def test_parser_aligns_complete_json_path_and_preserves_structure() -> None:
    """同文件完整 key path 对齐，HTML、placeholder 与 codicon 不被剥离。"""
    files, pairs, summary = (
        module.parse_normalization_recovery_v4_vscode_archive(_archive()))
    assert len(files) == 6
    assert len(pairs) == 3
    assert summary["training_eligible_pair_count"] == 3
    phrase = next(item for item in pairs
                  if item["json_path"] == ["module", "phrase"])
    assert phrase["zh_hant_text"] == "匯入資訊"
    assert phrase["zh_hans_text"] == "导入信息"
    tagged = next(item for item in pairs
                  if item["json_path"] == ["module", "tagged"])
    assert tagged["structure_equal"] == 1
    assert tagged["zh_hans_structure_tokens"] == [
        "HTML_OPEN:b", "{0}", "HTML_CLOSE:b", "$(check)"]


def test_parser_rejects_duplicate_key_and_locale_inventory_drift() -> None:
    """重复 JSON key 或两侧文件 inventory 不同必须失败关闭。"""
    duplicate = (
        b'{"":[],"version":"1.0.0","contents":'
        b'{"module":{"key":"a","key":"b"}}}')
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE.md", b"MIT\n")
        archive.writestr("README.md", b"source\n")
        archive.writestr(
            "i18n/vscode-language-pack-zh-hans/translations/main.i18n.json",
            duplicate,
        )
        archive.writestr(
            "i18n/vscode-language-pack-zh-hant/translations/main.i18n.json",
            duplicate,
        )
    with pytest.raises(BroadQaExternalDataError, match="重复 key"):
        module.parse_normalization_recovery_v4_vscode_archive(
            target.getvalue())

    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE.md", b"MIT\n")
        archive.writestr("README.md", b"source\n")
        archive.writestr(
            "i18n/vscode-language-pack-zh-hans/translations/main.i18n.json",
            _translation(traditional=False),
        )
        archive.writestr(
            "i18n/vscode-language-pack-zh-hant/translations/other.i18n.json",
            _translation(traditional=True),
        )
    with pytest.raises(BroadQaExternalDataError, match="inventory 未对齐"):
        module.parse_normalization_recovery_v4_vscode_archive(
            target.getvalue())


def test_source_pack_round_trip_is_immutable_and_zero_read(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """raw archive 重派生，且同名 target 不能重发。"""
    source = _publish(tmp_path, monkeypatch)
    manifest, files, pairs = (
        module.read_normalization_recovery_v4_vscode_source_pack(source))
    assert manifest["evaluation_or_reserve_read_count"] == 0
    assert manifest["training_read_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(files) == 6
    assert len(pairs) == 3
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        module.publish_normalization_recovery_v4_vscode_source_pack(
            run_root=tmp_path,
            archive_path=source / module.VSCODE_ARCHIVE_NAME,
            target_dir=source,
        )


def test_reader_rejects_synchronized_pair_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改派生 pair 与 manifest 仍不能绕过 raw JSON。"""
    source = _publish(tmp_path, monkeypatch)
    pair_path = source / "translation-pairs.jsonl"
    lines = pair_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["zh_hans_text"] = "篡改"
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
        module.read_normalization_recovery_v4_vscode_source_pack(source)


def test_publisher_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 publisher 不得把 source artifact 回退到 D 盘或临时目录。"""
    target = tmp_path / "pack"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        module.publish_normalization_recovery_v4_vscode_source_pack(
            run_root=tmp_path,
            archive_path=tmp_path / "missing.zip",
            target_dir=target,
        )
    assert not target.exists()


def test_publisher_rejects_raw_archive_drift_before_unpack(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """raw identity 漂移必须在解压和 target 创建之前失败。"""
    payload = _archive()
    _freeze_synthetic_identity(monkeypatch, payload)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    source = tmp_path / "source.zip"
    source.write_bytes(payload + b"drift")
    target = tmp_path / "pack"
    with pytest.raises(BroadQaExternalDataError, match="archive identity"):
        module.publish_normalization_recovery_v4_vscode_source_pack(
            run_root=tmp_path,
            archive_path=source,
            target_dir=target,
        )
    assert not target.exists()


def test_official_vscode_archive_identity_when_available() -> None:
    """提供 Git 外 fixture 时核对固定提交的全量 census。"""
    value = os.environ.get("PURE_INTEGER_AI_VSCODE_LOC_ARCHIVE")
    if not value:
        pytest.skip("official VS Code localization archive fixture is unavailable")
    payload = Path(value).read_bytes()
    files, pairs, summary = (
        module.parse_normalization_recovery_v4_vscode_archive(payload))
    assert len(payload) == module.VSCODE_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == module.VSCODE_ARCHIVE_SHA256
    assert len(files) == 202
    assert len(pairs) == 25_851
    assert summary == module.VSCODE_OFFICIAL_SUMMARY
