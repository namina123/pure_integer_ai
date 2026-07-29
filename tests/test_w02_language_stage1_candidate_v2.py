"""W-02 v2 base fence 的公开身份、负结果绑定和漂移反例。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w02_candidate_v2 import (
    W02_V2_HOST_CODE_PATHS,
    W02_V2_HOST_TEST_PATHS,
    build_w02_v2_base_fence,
    w02_v2_base_fence_key,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_REMOTE_COMMIT = "5d00b703ada7d41bfa96e466a06af61026da3a64"
_PRIOR_FREEZE = {
    "ablation_order": [
        "WITHOUT_BOUNDARY_WITHDRAWAL",
        "WITHOUT_MULTI_CANDIDATE",
        "WITHOUT_MORPHOLOGY",
        "WITHOUT_OOV",
    ],
    "artifact_kind": "SYNTHETIC_W02_V1_CANDIDATE_FREEZE",
    "base_fence_key": [1, 2, 3],
    "d03_thresholds": {
        "all_bearing_dimensions_must_pass": 1,
        "fail_allowed": 0,
        "ne_blocks": 1,
        "required_pass_denominator": 1,
        "required_pass_numerator": 1,
    },
    "evaluation_order": [
        "W-02-BOUNDARY_WITHDRAWAL",
        "W-02-MULTI_CANDIDATE",
        "W-02-NEW_CONTENT_MORPHOLOGY",
        "W-02-OOV",
        "W-02-GENERATION-HARD-CONJUNCT",
    ],
    "format_version": 1,
}
_PRIOR_REPORT = {
    "artifact_kind": "SYNTHETIC_W02_V1_AGGREGATE",
    "format_version": 1,
}


def _json_bytes(value: object) -> bytes:
    """形成稳定 UTF-8 JSON，只服务公开合成 prior artifact。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_prior_artifacts(
        root: Path,
        *,
        freeze: object = _PRIOR_FREEZE,
        report: object = _PRIOR_REPORT,
        ) -> tuple[Path, Path]:
    """在临时目录写入无 private case/label 的 prior 合同占位件。"""
    root.mkdir(parents=True, exist_ok=True)
    freeze_path = root / "candidate_host_freeze.json"
    report_path = root / "w02_private_evaluation_first_run.json"
    freeze_path.write_bytes(_json_bytes(freeze))
    report_path.write_bytes(_json_bytes(report))
    return freeze_path, report_path


def _fence(
        freeze_path: Path,
        report_path: Path,
        remote_commit: str = _REMOTE_COMMIT,
        ):
    """从公开仓库和临时 prior 合同构造 v2 base fence。"""
    return build_w02_v2_base_fence(
        _REPOSITORY,
        remote_commit_sha1=remote_commit,
        v1_freeze_path=freeze_path,
        v1_report_path=report_path,
    )


def test_v2_base_fence_is_deterministic_and_binds_public_runtime_inventory(
        tmp_path: Path,
        ):
    """同输入 fence bit-identical，且必须覆盖 v2 adapter/runtime 与对抗测试。"""
    freeze_path, report_path = _write_prior_artifacts(tmp_path / "prior")
    first = _fence(freeze_path, report_path)
    second = _fence(freeze_path, report_path)
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
        "candidate_freeze_sha256": hashlib.sha256(
            freeze_path.read_bytes()).hexdigest(),
        "first_report_sha256": hashlib.sha256(
            report_path.read_bytes()).hexdigest(),
    }


def test_remote_commit_change_invalidates_v2_base_fence_key(tmp_path: Path):
    """只改 remote commit 也必须改变完整 base fence，不能沿用旧 candidate。"""
    freeze_path, report_path = _write_prior_artifacts(tmp_path / "prior")
    changed = "0" * 40
    assert w02_v2_base_fence_key(
        _fence(freeze_path, report_path)
    ) != w02_v2_base_fence_key(
        _fence(freeze_path, report_path, changed)
    )


@pytest.mark.parametrize("missing_name", [
    "candidate_host_freeze.json",
    "w02_private_evaluation_first_run.json",
])
def test_missing_prior_artifact_fails_closed(
        tmp_path: Path,
        missing_name: str,
        ):
    """任一 prior artifact 缺失都必须阻断 base fence 构造。"""
    freeze_path, report_path = _write_prior_artifacts(tmp_path / "prior")
    (freeze_path.parent / missing_name).unlink()
    with pytest.raises(RuntimeError, match="不存在|无法回读"):
        _fence(freeze_path, report_path)


def test_malformed_prior_freeze_json_fails_closed(tmp_path: Path):
    """无法解析的 prior freeze 不能被当成合法 canonical JSON 合同。"""
    freeze_path, report_path = _write_prior_artifacts(tmp_path / "prior")
    freeze_path.write_bytes(b"{not-json")
    with pytest.raises(RuntimeError, match="无法回读"):
        _fence(freeze_path, report_path)


@pytest.mark.parametrize("missing_field", sorted(_PRIOR_FREEZE))
def test_incomplete_prior_freeze_fails_closed(
        tmp_path: Path,
        missing_field: str,
        ):
    """prior freeze 六个合同字段任一缺失都必须阻断。"""
    incomplete = dict(_PRIOR_FREEZE)
    incomplete.pop(missing_field)
    freeze_path, report_path = _write_prior_artifacts(
        tmp_path / "prior",
        freeze=incomplete,
    )
    with pytest.raises(RuntimeError, match="字段不完整"):
        _fence(freeze_path, report_path)


@pytest.mark.parametrize("changed_artifact", ["freeze", "report"])
def test_prior_artifact_change_invalidates_v2_base_fence_key(
        tmp_path: Path,
        changed_artifact: str,
        ):
    """任一 prior artifact 字节变化都必须改变 v2 base fence identity。"""
    freeze_path, report_path = _write_prior_artifacts(tmp_path / "prior")
    original = w02_v2_base_fence_key(_fence(freeze_path, report_path))
    changed = dict(_PRIOR_FREEZE if changed_artifact == "freeze"
                   else _PRIOR_REPORT)
    changed["synthetic_revision"] = 2
    target = freeze_path if changed_artifact == "freeze" else report_path
    target.write_bytes(_json_bytes(changed))
    assert original != w02_v2_base_fence_key(
        _fence(freeze_path, report_path)
    )
