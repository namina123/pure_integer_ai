"""PW-01 正式 successor 的 base 继承、因果证据和重启专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.publish_pw00a_formal_start_receipt import execute_formal_start
from scripts.publish_pw01_formal_successor_receipt import (
    STATUS,
    execute_formal_successor,
)


def test_pw01_formal_successor_inherits_base_without_overwrite(tmp_path: Path):
    """由临时 PW-00A 原件产生 successor，base 不变且目标不可覆盖。"""
    root = Path(__file__).resolve().parents[1]
    base_database = tmp_path / "pw00a-base.sqlite3"
    successor_database = tmp_path / "pw01-successor.sqlite3"
    base_receipt = execute_formal_start(root, base_database)
    base_bytes = base_database.read_bytes()

    def base_reader(*args, **kwargs):
        """返回本测试刚产生且与显式 database 绑定的 base receipt。"""
        del args, kwargs
        return base_receipt

    value = execute_formal_successor(
        root,
        base_database,
        successor_database,
        base_receipt_reader=base_reader,
    )
    assert value["status"] == STATUS
    assert value["readiness_transition"] == {
        "PW00A_STARTED": 1,
        "PW01_COMPLETE": 0,
        "PW01_CONTROLLED_READING_EVIDENCED": 1,
    }
    assert value["formal_evidence"]["before_complete"] == 0
    assert value["formal_evidence"]["exact_ablation_complete"] == 0
    assert value["formal_evidence"]["restart_complete"] == 1
    assert value["formal_evidence"]["after_answer_sha256"] == (
        value["formal_evidence"]["restart_answer_sha256"])
    assert value["formal_evidence"]["fresh_projection_record_count"] == 3
    assert value["formal_evidence"]["projection_record_count"] == 3
    assert base_database.read_bytes() == base_bytes
    assert successor_database.read_bytes() != base_bytes

    with pytest.raises(RuntimeError, match="已存在"):
        execute_formal_successor(
            root,
            base_database,
            successor_database,
            base_receipt_reader=base_reader,
        )
