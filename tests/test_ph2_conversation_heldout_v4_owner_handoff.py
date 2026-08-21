"""DLG-05 v4 owner handoff metadata-only 合同专项。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_heldout_v4_family import (
    build_v4_family,
    write_v4_family_artifacts,
)
from pure_integer_ai.experiments.conversation_heldout_v4_owner_handoff import (
    ConversationHeldOutV4OwnerHandoffError,
    read_v4_owner_metadata,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _receipt(family, source_root: Path, label_path: str, label_size: int) -> bytes:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    document = {
        "bundle_index": family.bundle.index,
        "bundle_payload_sha256": bytes(family.bundle.payload_sha256).hex(),
        "bundle_payload_size": family.bundle.payload_size,
        "case_count": 6,
        "candidate_count": sum(len(item.candidates) for item in family.bundle.turns),
        "formal_run": 0,
        "label_payload_path": label_path,
        "label_payload_sha256": hashlib.sha256(b"owner-label-placeholder").hexdigest(),
        "label_payload_size": label_size,
        "labels_read": 0,
        "owner_namespace": [20260821, 405, 900],
        "projection": {
            "html_sha256": digest(source_root / "projection" / "dlg05_v4_reading.html"),
            "markdown_sha256": digest(source_root / "projection" / "dlg05_v4_reading.md"),
        },
        "schema": "dlg05-v4-owner-receipt-v1",
        "selection_run_count": 0,
        "source_count": len(family.bundle.sources),
        "status": "SEALED_UNREAD",
        "turn_count": len(family.bundle.turns),
    }
    # document_sha256 is computed over the document without the self field.
    document["document_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return _canonical(document)


def _setup(tmp_path):
    family = build_v4_family()
    source = tmp_path / "source"
    write_v4_family_artifacts(family, source)
    owner = tmp_path / "owner"
    (owner / "labels").mkdir(parents=True)
    label = owner / "labels" / "owner.labels.bin"
    label.write_bytes(b"owner-label-placeholder")
    (owner / "owner_receipt.json").write_bytes(
        _receipt(family, source, "labels/owner.labels.bin", label.stat().st_size))
    return family, source, owner, label


def test_owner_metadata_reader_does_not_read_label_payload(tmp_path, monkeypatch):
    """无效/不可解析 label 正文也不影响 metadata-only 回读。"""
    family, source, owner, label = _setup(tmp_path)
    # 用同长度非 JSON 内容替换，并让任何 open 立即失败，机械证明只读取 stat。
    label.write_bytes(b"X" * label.stat().st_size)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == label.resolve():
            raise AssertionError("metadata-only reader 不得打开 label payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    metadata = read_v4_owner_metadata(
        owner, source, require_k_drive=False, family=family)
    assert metadata.status == "SEALED_UNREAD"
    assert metadata.labels_read == 0
    assert metadata.formal_run == 0


def test_owner_metadata_reader_rejects_drift_and_path_escape(tmp_path):
    """receipt 漂移、路径穿越和额外文件必须 fail closed。"""
    family, source, owner, _label = _setup(tmp_path)
    receipt = owner / "owner_receipt.json"
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["status"] = "READ"
    receipt.write_bytes(_canonical(document))
    with pytest.raises(ConversationHeldOutV4OwnerHandoffError, match="document SHA"):
        read_v4_owner_metadata(owner, source, require_k_drive=False, family=family)
    family, source, owner, _label = _setup(tmp_path / "escape")
    document = json.loads((owner / "owner_receipt.json").read_text(encoding="utf-8"))
    document["label_payload_path"] = "../outside.bin"
    document.pop("document_sha256")
    document["document_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    (owner / "owner_receipt.json").write_bytes(_canonical(document))
    with pytest.raises(ConversationHeldOutV4OwnerHandoffError, match="相对路径"):
        read_v4_owner_metadata(owner, source, require_k_drive=False, family=family)
    family, source, owner, _label = _setup(tmp_path / "extra")
    (owner / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ConversationHeldOutV4OwnerHandoffError, match="文件闭包"):
        read_v4_owner_metadata(owner, source, require_k_drive=False, family=family)


def test_owner_metadata_reader_requires_k_drive_in_production(tmp_path):
    """生产路径不能把临时目录或 D 盘当 owner/source 根。"""
    family, source, owner, _label = _setup(tmp_path)
    with pytest.raises(ConversationHeldOutV4OwnerHandoffError, match="K 盘"):
        read_v4_owner_metadata(owner, source, family=family)


def test_owner_metadata_reader_rejects_symlink_root(tmp_path):
    """根目录链接必须在 resolve 前拒绝，不能借目标目录通过隔离门。"""
    family, source, owner, _label = _setup(tmp_path)
    linked = tmp_path / "owner-link"
    try:
        linked.symlink_to(owner, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前 Windows 环境不允许创建目录 symlink")
    with pytest.raises(ConversationHeldOutV4OwnerHandoffError, match="链接|reparse"):
        read_v4_owner_metadata(linked, source, require_k_drive=False, family=family)
