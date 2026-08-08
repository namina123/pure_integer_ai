"""PW-00A 唯一正式启动 receipt 的固定字节和严格回读测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.artifact_verification_mode import (
    CURRENT_HEAD_COMPATIBILITY_VERIFY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from scripts.publish_pw00a_formal_start_receipt import (
    RECEIPT_PATH,
    read_formal_start_receipt,
    run_and_publish,
)


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT.parent / ".pw00a-formal-start-20260807-a.sqlite3"
PUBLISHED_SHA256 = (
    "c4cec590cce1a5c1956933dca726cd9148b3da06b953909d8ba2c6a84454595a"
)
DATABASE_SHA256 = (
    "df8e0376ad3c31848d60ccbfab11de4c7618842fac8ad5148680fe9b6581af84"
)


def test_pw00a_formal_start_receipt_fixed_and_database_bound() -> None:
    """公开 receipt 和 Git 外 SQLite 必须保持正式运行时的固定字节。"""
    target = ROOT / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PUBLISHED_SHA256
    assert hashlib.sha256(DATABASE.read_bytes()).hexdigest() == DATABASE_SHA256
    value = read_formal_start_receipt(ROOT, database_path=DATABASE)
    assert value["status"] == "PW00A_FORMAL_RUNTIME_STARTED"
    assert value["readiness_transition"] == {
        "LANGUAGE_CAPABILITY_MASTERED": 1,
        "LANGUAGE_READINESS": 1,
        "PW00A_STARTED": 1,
    }
    assert value["runtime_boundaries"]["inference_rule_count"] == 299
    assert value["runtime_boundaries"]["formal_event_count"] == 2
    assert tuple(item["event_kind"] for item in value["formal_events"]) == (1, 2)


def test_pw00a_formal_start_receipt_rejects_rerun_and_tamper(
        tmp_path: Path,
        ) -> None:
    """正式运行和 receipt 不可覆盖，任一启动位改写必须 fail closed。"""
    before = hashlib.sha256((ROOT / RECEIPT_PATH).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="禁止重跑"):
        run_and_publish(ROOT, DATABASE)
    assert hashlib.sha256((ROOT / RECEIPT_PATH).read_bytes()).hexdigest() == before

    value = read_formal_start_receipt(ROOT)
    changed = dict(value)
    changed["readiness_transition"] = dict(value["readiness_transition"])
    changed["readiness_transition"]["PW00A_STARTED"] = 0
    target = tmp_path / "tampered-start.json"
    target.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="readiness"):
        read_formal_start_receipt(ROOT, target)


def test_pw00a_formal_start_current_mode_rejects_historical_authority() -> None:
    """历史 start receipt 可审计，但不得给当前源码授予启动 authority。"""
    with pytest.raises(RuntimeError, match="source leaf 漂移"):
        read_formal_start_receipt(
            ROOT,
            verification_mode=CURRENT_HEAD_COMPATIBILITY_VERIFY,
        )
