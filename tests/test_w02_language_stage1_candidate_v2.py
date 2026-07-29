"""W-02 v2 base fence 的公开身份、负结果绑定和漂移反例。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_w02_candidate_v2 import (
    W02_V2_HOST_CODE_PATHS,
    W02_V2_HOST_TEST_PATHS,
    build_w02_v2_base_fence,
    w02_v2_base_fence_key,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_ENGINEERING_ROOT = _REPOSITORY.parent
_V1_ROOT = _ENGINEERING_ROOT / "w02_artifacts" / "formal_candidate_v1"
_REMOTE_COMMIT = "5d00b703ada7d41bfa96e466a06af61026da3a64"


def _fence(remote_commit: str = _REMOTE_COMMIT):
    """从公开仓库和 Git 外 v1 摘要构造 v2 base fence。"""
    return build_w02_v2_base_fence(
        _REPOSITORY,
        remote_commit_sha1=remote_commit,
        v1_freeze_path=_V1_ROOT / "candidate_host_freeze.json",
        v1_report_path=_V1_ROOT / "w02_private_evaluation_first_run.json",
    )


def test_v2_base_fence_is_deterministic_and_binds_public_runtime_inventory():
    """同输入 fence bit-identical，且必须覆盖 v2 adapter/runtime 与对抗测试。"""
    first = _fence()
    second = _fence()
    assert first == second
    assert w02_v2_base_fence_key(first) == w02_v2_base_fence_key(second)
    assert len(w02_v2_base_fence_key(first)) == 32
    assert tuple(item["path"] for item in first["code_inventory"]) == (
        W02_V2_HOST_CODE_PATHS)
    assert tuple(item["path"] for item in first["test_inventory"]) == (
        W02_V2_HOST_TEST_PATHS)
    assert any(item["path"].endswith("ph2_w02_learning_v2.py")
               for item in first["code_inventory"])
    assert any(item["path"].endswith("ph2_w02_runtime_v2.py")
               for item in first["code_inventory"])
    assert first["prior_v1_failure"] == {
        "candidate_freeze_sha256": (
            "78d3e96463912ac358bded536dc5cb2cd15ba6a94302e0a8188180d84a8cb255"),
        "first_report_sha256": (
            "1a2a79561acb6f9aa50841dac7b7cd8104945e9be8628281df33a83a30c94412"),
    }


def test_remote_commit_change_invalidates_v2_base_fence_key():
    """只改 remote commit 也必须改变完整 base fence，不能沿用旧 candidate。"""
    changed = "0" * 40
    assert w02_v2_base_fence_key(_fence()) != w02_v2_base_fence_key(
        _fence(changed))
