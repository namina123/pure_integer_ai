"""PW-00A 正式装载 authority 的固定发布物与无 Git 回读测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.pw00a_authority import (
    PW00AAuthorityError,
    RECEIPT_PATH,
    publish_pw00a_formal_load_authority,
    read_pw00a_formal_load_authority,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SHA256 = (
    "7c65b4d88e2bbfa8b1b2bc43dcd2c5a7f30456e6954739a192614d868608daeb"
)


def test_pw00a_authority_fixed_bytes_and_chain() -> None:
    """发布物必须绑定完整 source/dependency 链且不提前设置启动位。"""
    target = ROOT / RECEIPT_PATH
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PUBLISHED_SHA256
    value = read_pw00a_formal_load_authority(ROOT)
    assert len(value["source_bindings"]) == 20
    assert len(value["dependency_bindings"]) == 14
    assert value["readiness_transition"] == {
        "LANGUAGE_CAPABILITY_MASTERED": 1,
        "LANGUAGE_READINESS_REPUBLISHED": 1,
        "PW00A_STARTED": 0,
    }
    correction = value["v5_metadata_correction"]
    assert len(correction["recorded_parent_sha256"]) == 63
    assert len(correction["correct_parent_sha256"]) == 64
    assert correction["current_leaf_sha256"] == (
        "88ce77e8274c8dc5bede0f8ab786667de21872185f8c9e61ea5a0b9b6600219a"
    )


def test_pw00a_authority_runtime_reader_does_not_call_git(monkeypatch) -> None:
    """正式 runtime reader 只能核当前字节，不得依赖 Git 可执行文件。"""
    def reject_git(*args, **kwargs):
        """任何 subprocess 调用都说明 runtime 边界倒退。"""
        del args, kwargs
        raise AssertionError("runtime reader called Git")

    monkeypatch.setattr(
        "pure_integer_ai.experiments.pw00a_authority.subprocess.run",
        reject_git,
    )
    assert read_pw00a_formal_load_authority(ROOT)["status"] == (
        "PW00A_FORMAL_LOAD_AUTHORITY_EVIDENCED")


def test_pw00a_authority_tamper_and_overwrite_fail_closed(tmp_path: Path) -> None:
    """readiness 或 source leaf 漂移必须拒绝，正式发布物禁止覆盖。"""
    value = read_pw00a_formal_load_authority(ROOT)
    changed = dict(value)
    changed["readiness_transition"] = dict(value["readiness_transition"])
    changed["readiness_transition"]["PW00A_STARTED"] = 1
    target = tmp_path / "tampered-authority.json"
    target.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(PW00AAuthorityError, match="readiness"):
        read_pw00a_formal_load_authority(ROOT, target)
    before = hashlib.sha256((ROOT / RECEIPT_PATH).read_bytes()).hexdigest()
    with pytest.raises(PW00AAuthorityError, match="禁止覆盖"):
        publish_pw00a_formal_load_authority(ROOT)
    assert hashlib.sha256((ROOT / RECEIPT_PATH).read_bytes()).hexdigest() == before
