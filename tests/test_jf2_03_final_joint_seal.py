"""JF2-03 final joint seal 的合取、canonical 与发布边界测试。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pure_integer_ai.experiments.j_f2_final_joint_seal as final_seal_module
from pure_integer_ai.experiments.j_f2_final_joint_seal import (
    SEAL_PATH,
    FinalJointSeal,
    FinalJointSealError,
    build_final_joint_seal,
    publish_final_joint_seal,
    read_final_joint_seal,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from tests.jf2_historical_context import build_historical_final_joint_seal


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_seal() -> FinalJointSeal:
    """共享一次历史闭包 preflight，控制 J-F1 回验成本。"""
    return build_historical_final_joint_seal(ROOT)


def test_historical_seal_has_complete_joint_conjuncts(live_seal: FinalJointSeal) -> None:
    """历史公开依赖必须全 PASS，墙维保持 NE，状态转换保持受控。"""
    assert all(item.status == "PASS" for item in live_seal.dependency_bindings)
    assert all(item.status == "PASS" for item in live_seal.hard_conjuncts)
    assert live_seal.w09_evidence["j_lc"]["wall_dimension_states"] == [
        ["W-09-W1_PHYSICAL_GROUNDING", "NE"],
        ["W-09-W2_DEFINITIVE_TRUTH", "NE"],
    ]
    assert live_seal.readiness_transition == {
        "LANGUAGE_CAPABILITY_MASTERED": 1,
        "LANGUAGE_READINESS_AFTER_EXCLUSIVE_PUBLICATION": 1,
        "LANGUAGE_READINESS_BEFORE_PUBLICATION": 0,
        "PW00A_STARTED": 0,
        "can_ween_language_modified": 0,
    }
    assert read_final_joint_seal(
        ROOT, verify_dependencies=False) == live_seal


def test_canonical_readback_rejects_state_drift(
        tmp_path: Path, live_seal: FinalJointSeal) -> None:
    """canonical 回读必须拒绝 readiness 预置、字段漂移和多余换行。"""
    target = tmp_path / "seal.json"
    target.write_bytes(live_seal.canonical_bytes())
    assert read_final_joint_seal(
        ROOT, target, verify_dependencies=False) == live_seal

    value = json.loads(target.read_text(encoding="utf-8"))
    value["readiness_transition"]["LANGUAGE_READINESS_BEFORE_PUBLICATION"] = 1
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(FinalJointSealError):
        read_final_joint_seal(ROOT, target, verify_dependencies=False)

    target.write_bytes(live_seal.canonical_bytes() + b"\n")
    with pytest.raises(FinalJointSealError):
        read_final_joint_seal(ROOT, target, verify_dependencies=False)


def test_reader_rejects_path_and_dependency_drift(
        tmp_path: Path, live_seal: FinalJointSeal, monkeypatch) -> None:
    """seal 内路径越界或公开依赖身份变化时必须 fail closed。"""
    target = tmp_path / "seal.json"
    value = live_seal.to_dict()
    value["seal_relative_path"] = "../outside.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(FinalJointSealError):
        read_final_joint_seal(ROOT, target, verify_dependencies=False)

    value = live_seal.to_dict()
    value["dependency_bindings"][0]["sha256"] = "1" * 64
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    monkeypatch.setattr(
        final_seal_module, "build_final_joint_seal", lambda _root: live_seal)
    with pytest.raises(FinalJointSealError, match="依赖身份漂移"):
        read_final_joint_seal(ROOT, target, verify_dependencies=True)


def test_build_fails_closed_when_preflight_is_blocked(
        live_seal: FinalJointSeal, monkeypatch) -> None:
    """preflight 出现阻断时不得形成带 readiness 转换的 seal。"""
    blocked = SimpleNamespace(
        status="BLOCKED", blockers=("FORCED_BLOCKER",),
        language_capability_mastered=1, language_readiness=0,
        dependencies=live_seal.dependency_bindings)
    monkeypatch.setattr(
        final_seal_module, "build_jf2_preflight", lambda _root: blocked)
    with pytest.raises(FinalJointSealError, match="未满足正式封存条件"):
        build_final_joint_seal(ROOT)


def test_publish_is_append_only_and_prechecks_existing_target(
        tmp_path: Path, live_seal: FinalJointSeal, monkeypatch) -> None:
    """临时首次发布可回读，重复发布须在昂贵 preflight 前拒绝。"""
    target = tmp_path / Path(SEAL_PATH).name
    monkeypatch.setattr(
        final_seal_module, "build_final_joint_seal", lambda _root: live_seal)
    published = publish_final_joint_seal(ROOT, target=target)
    assert published == live_seal

    def must_not_rebuild(_root):
        """重复发布若进入 builder，测试必须失败。"""
        raise AssertionError("duplicate publication reran preflight")

    monkeypatch.setattr(
        final_seal_module, "build_final_joint_seal", must_not_rebuild)
    with pytest.raises(FinalJointSealError, match="禁止覆盖"):
        publish_final_joint_seal(ROOT, target=target)
