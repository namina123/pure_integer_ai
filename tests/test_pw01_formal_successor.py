"""PW-01 正式 successor 的 base 继承、因果证据和重启专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.facility_readiness_scenarios import (
    prepare_facility_context,
)
from pure_integer_ai.experiments.pw00a_formal_transaction import (
    PW00AFormalEventStore,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend
from scripts.publish_pw01_formal_successor_receipt import (
    RECEIPT_PATH,
    STATUS,
    execute_formal_successor,
    read_formal_successor_receipt,
    run_and_publish,
)


RECEIPT_SHA256 = "236f02c5750589a1bafea9d204be3d7895643e498cd0bcb65c0a94e4dbf67ed7"


def _identity(path: Path) -> dict[str, int | str]:
    """返回测试 base 的文件身份。"""
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _base_fixture(root: Path, database: Path):
    """不调用历史 authority，构造含唯一 PW-00A 双事件的最小继承 base。"""
    del root
    backend = SQLiteBackend(str(database))
    try:
        ctx = make_train_context(backend, companion=True)
        prepare_facility_context(ctx)
        store = PW00AFormalEventStore(backend)
        manifest_sha256 = "1" * 64
        prepared = store.prepared(
            run_id=2026080701,
            publish_epoch=1,
            manifest_sha256=manifest_sha256,
            payload={"fixture": 1},
        )
        published = store.published(
            run_id=2026080701,
            publish_epoch=1,
            manifest_sha256=manifest_sha256,
            payload={"fixture": 2},
        )
    finally:
        backend.close()
    events = [
        {
            "event_kind": item.event_kind,
            "event_seq": item.event_seq,
            "manifest_sha256": item.manifest_sha256,
            "payload_sha256": item.payload_sha256,
            "publish_epoch": item.publish_epoch,
            "run_id": item.run_id,
        }
        for item in (prepared, published)
    ]
    return {"formal_events": events, "runtime_database": _identity(database)}


def test_pw01_formal_successor_inherits_base_without_overwrite(tmp_path: Path):
    """由临时 PW-00A 原件产生 successor，base 不变且目标不可覆盖。"""
    root = Path(__file__).resolve().parents[1]
    base_database = tmp_path / "pw00a-base.sqlite3"
    successor_database = tmp_path / "pw01-successor.sqlite3"
    base_receipt = _base_fixture(root, base_database)
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


def test_public_pw01_successor_receipt_is_strict_and_append_only(tmp_path: Path):
    """公开 receipt 固定字节可严格回读，重复发布在读取数据库前即拒绝。"""
    root = Path(__file__).resolve().parents[1]
    target = root / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == RECEIPT_SHA256
    value = read_formal_successor_receipt(root)
    assert value["status"] == STATUS
    assert value["formal_evidence"]["after_answer_sha256"] == (
        value["formal_evidence"]["restart_answer_sha256"])
    with pytest.raises(ValueError, match="receipt 已存在"):
        run_and_publish(
            root,
            tmp_path / "missing-base.sqlite3",
            tmp_path / "missing-successor.sqlite3",
        )
