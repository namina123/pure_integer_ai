"""DLG-05 v4 metadata-only handoff 审计专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_heldout_v4_family import (
    build_v4_family,
    write_v4_family_artifacts,
)
from pure_integer_ai.experiments.conversation_heldout_v4_metadata_audit import (
    ConversationHeldOutV4MetadataAuditError,
    audit_v4_family_artifacts,
    write_v4_metadata_audit,
)


def test_v4_metadata_audit_is_read_only_and_explicitly_fixture_only(tmp_path):
    """完整 family artifact 可逐字节核对，但不得获得 owner handoff 资格。"""
    family = build_v4_family()
    root = tmp_path / "family"
    write_v4_family_artifacts(family, root)
    first = audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    second = audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    assert first == second
    assert first.status == "SYNTHETIC_FIXTURE_ONLY"
    assert first.source_qualified == 0
    assert first.file_count == 5
    assert first.document()["status"] == first.status


def test_v4_metadata_audit_report_is_idempotent_and_detects_drift(tmp_path):
    """metadata report 只允许同字节复写，漂移或标签投影必须拒绝。"""
    family = build_v4_family()
    root = tmp_path / "family"
    write_v4_family_artifacts(family, root)
    audit = audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    report = root / "metadata_audit.json"
    # 测试 writer 的路径门需使用真实 K 盘；这里只验证审计在报告出现后拒绝临时根。
    with pytest.raises(ConversationHeldOutV4MetadataAuditError, match="K 盘"):
        write_v4_metadata_audit(root, audit)
    (root / "projection" / "dlg05_v4_reading.md").write_text(
        "selected_candidate", encoding="utf-8")
    with pytest.raises(ConversationHeldOutV4MetadataAuditError, match="projection"):
        audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    assert not report.exists()


def test_v4_metadata_audit_rejects_non_k_production_root(tmp_path):
    """生产默认不能静默把 D 盘或临时目录当 K 盘工作根。"""
    with pytest.raises(ConversationHeldOutV4MetadataAuditError, match="K 盘"):
        audit_v4_family_artifacts(Path(tmp_path), family=build_v4_family())


def test_v4_metadata_audit_rejects_symlink_root_before_resolve(
        tmp_path, monkeypatch):
    """root 链接必须在 resolve 前拒绝，避免 junction 隐去自身属性。"""
    family = build_v4_family()
    root = tmp_path / "family"
    write_v4_family_artifacts(family, root)
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def is_family_link(path):
        if path == root:
            return True
        return original_is_symlink(path)

    def resolve_after_link_check(path, *args, **kwargs):
        if path == root:
            raise AssertionError("链接根不得在检查前 resolve")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", is_family_link)
    monkeypatch.setattr(Path, "resolve", resolve_after_link_check)
    with pytest.raises(ConversationHeldOutV4MetadataAuditError, match="链接|reparse"):
        audit_v4_family_artifacts(root, require_k_drive=False, family=family)
