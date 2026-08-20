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


def test_v4_metadata_audit_is_read_only_and_deterministic(tmp_path):
    """完整 family artifact 可由同一 typed bundle 逐字节核对。"""
    family = build_v4_family()
    root = tmp_path / "family"
    write_v4_family_artifacts(family, root)
    first = audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    second = audit_v4_family_artifacts(root, require_k_drive=False, family=family)
    assert first == second
    assert first.status == "READY_FOR_OWNER_HANDOFF"
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
        audit_v4_family_artifacts(Path(tmp_path))
