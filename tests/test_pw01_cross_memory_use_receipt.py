"""PW-01 跨 Memory Use 正式 receipt 的公开回读专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.publish_pw01_cross_memory_use_receipt import (
    RECEIPT_PATH,
    STATUS,
    read_cross_memory_use_receipt,
    run_and_publish,
)


RECEIPT_SHA256 = "8919bcc6f2dc6c338845c22d23fef2baa5ec0f8b762af43e408c0dc8fbc19589"


def test_public_cross_memory_use_receipt_is_strict_and_append_only(tmp_path):
    """公开固定字节可严格回读，重复发布在读取数据库前即拒绝。"""
    root = Path(__file__).resolve().parents[1]
    target = root / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == RECEIPT_SHA256
    value = read_cross_memory_use_receipt(root)
    assert value["status"] == STATUS
    assert value["evidence"]["bridge_record_count"] == 1
    assert value["evidence"]["read_last_used_seq"] == 0
    assert value["evidence"]["same_user_other_session_fact_visible"] == 1
    assert value["evidence"]["other_session_payload_visible"] == 0
    with pytest.raises(ValueError, match="receipt 已存在"):
        run_and_publish(
            root,
            tmp_path / "missing-base.sqlite3",
            tmp_path / "missing-successor.sqlite3",
        )
