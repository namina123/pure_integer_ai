"""PERF-P3 中文说明后继 receipt 的严格边界检查。"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.performance_p3_sqlite_trial_successor_receipt import (
    EXECUTABLE_AST_SHA256,
    PARENT_RECEIPT_SHA256,
    RECEIPT_PATH,
    build_performance_p3_sqlite_trial_successor_receipt,
    publish_performance_p3_sqlite_trial_successor_receipt,
    read_performance_p3_sqlite_trial_successor_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def test_p3_successor_only_changes_explanations() -> None:
    """后继合同须绑定当前 worker 且可执行 AST 与 v1 完全一致。"""
    value = build_performance_p3_sqlite_trial_successor_receipt(ROOT)
    assert value["parent_receipt"]["sha256"] == PARENT_RECEIPT_SHA256
    assert value["transformation"] == {
        "documentation_language": "ZH_CN",
        "executable_ast_changed": 0,
        "executable_ast_sha256": EXECUTABLE_AST_SHA256,
        "external_evidence_rerun": 0,
        "runtime_contract_changed": 0,
    }
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }


def test_p3_successor_publish_is_append_only(tmp_path: Path) -> None:
    """临时发布须规范回读，重复发布必须在覆盖前拒绝。"""
    target = tmp_path / "p3-v2.json"
    value = publish_performance_p3_sqlite_trial_successor_receipt(
        ROOT, target=target)
    before = target.read_bytes()
    assert read_performance_p3_sqlite_trial_successor_receipt(
        ROOT, target) == value
    with pytest.raises(ValueError, match="禁止覆盖"):
        publish_performance_p3_sqlite_trial_successor_receipt(
            ROOT, target=target)
    assert target.read_bytes() == before


def test_p3_successor_rejects_executable_change(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """worker 可执行 AST 变化时不得按说明文字修正授权。"""
    import scripts.performance_p3_sqlite_trial_successor_receipt as receipt

    monkeypatch.setattr(receipt, "EXECUTABLE_AST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="可执行 AST"):
        build_performance_p3_sqlite_trial_successor_receipt(ROOT)


def test_p3_worker_module_docstring_is_chinese() -> None:
    """worker 模块说明必须保留中文解释性内容。"""
    worker = ROOT / "scripts/performance_p3_sqlite_trial_worker.py"
    tree = ast.parse(worker.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree, clean=False)
    assert docstring is not None
    assert any("\u4e00" <= char <= "\u9fff" for char in docstring)
    assert RECEIPT_PATH.endswith("performance_p3_sqlite_trial_receipt_v2.json")
