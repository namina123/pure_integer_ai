"""W-09-00 authority 的现场、规范和破坏测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_w09_authority as authority
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_AUTHORITY_RELATIVE_PATH,
    W09_DIMENSION_KEYS,
    W09AuthorityError,
    build_w09_authority,
    canonical_w09_authority_bytes,
    publish_w09_authority,
    validate_w09_authority,
)


ROOT = Path(__file__).resolve().parents[1]


def test_w09_authority_recomputes_stage_inventory_and_parents() -> None:
    value = build_w09_authority(ROOT)
    validate_w09_authority(value)
    assert tuple(value["dimension_keys"]) == W09_DIMENSION_KEYS
    assert tuple(value["ablation_keys"]) == W09_ABLATION_KEYS
    assert value["stage_inventory"]["train_pack_count"] == 34
    assert value["stage_inventory"]["future_pack_count"] == 0
    assert value["execution_state"]["W09_STARTED"] == 0
    assert value["historical_exposure"]["public_sample_heldout_label_exposure"] == 1


def test_w09_authority_is_canonical_and_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / W09_AUTHORITY_RELATIVE_PATH
    destination.parent.mkdir(parents=True)
    value = build_w09_authority(ROOT)
    # 用真实父身份构造一个只读副本，验证规范字节和排他写的行为。
    destination.write_bytes(canonical_w09_authority_bytes(value))
    assert json.loads(destination.read_bytes()) == value
    monkeypatch.setattr(authority, "W09_AUTHORITY_RELATIVE_PATH", W09_AUTHORITY_RELATIVE_PATH)
    monkeypatch.setattr(authority, "build_w09_authority", lambda _root: value)
    # 真实 publisher 在临时仓库上回读规范字节，并拒绝第二次发布。
    destination.unlink()
    digest = authority.publish_w09_authority(tmp_path)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert authority.read_w09_authority(tmp_path) == value
    with pytest.raises(W09AuthorityError, match="append-only"):
        authority.publish_w09_authority(tmp_path)
    with pytest.raises(FileExistsError):
        with destination.open("xb"):
            pass


def test_w09_authority_rejects_dimension_or_future_drift() -> None:
    value = build_w09_authority(ROOT)
    changed = dict(value)
    changed["dimension_keys"] = list(value["dimension_keys"][:-1])
    with pytest.raises(W09AuthorityError):
        validate_w09_authority(changed)
    changed = dict(value)
    changed["stage_inventory"] = dict(value["stage_inventory"], future_pack_count=1)
    with pytest.raises(W09AuthorityError):
        validate_w09_authority(changed)


def test_w09_authority_publisher_uses_exclusive_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = build_w09_authority(ROOT)
    monkeypatch.setattr(authority, "build_w09_authority", lambda _root: value)
    monkeypatch.setattr(authority, "W09_AUTHORITY_RELATIVE_PATH", W09_AUTHORITY_RELATIVE_PATH)
    digest = publish_w09_authority(tmp_path)
    assert len(digest) == 64
    with pytest.raises(W09AuthorityError, match="append-only"):
        publish_w09_authority(tmp_path)
