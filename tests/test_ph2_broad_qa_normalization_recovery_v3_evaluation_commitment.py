"""normalization recovery v3 标签盲分母 commitment 测试。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v3_evaluation_commitment as module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _prior_manifest() -> dict[str, object]:
    """构造只含冻结 identity、没有 label 的旧 manifest 视图。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
        "inventory_summary": {
            **module.PRIOR_RESERVE_SUMMARY,
            "evaluation_count": 6_343,
        },
        "manifest_sha256": module.PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256,
        "reserve_identity": deepcopy(module.PRIOR_RESERVE_IDENTITY),
        "source_pack_manifest_sha256": (
            module.EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256),
    }


def _install_reader(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """安装只返回 manifest 的 reader，并记录调用路径。"""
    calls: list[Path] = []

    def reader(path, *, expected_manifest_sha256):
        calls.append(Path(path))
        assert expected_manifest_sha256 == (
            module.PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256)
        return _prior_manifest()

    monkeypatch.setattr(
        module, "read_normalization_recovery_evaluation_manifest_only", reader)
    return calls


def test_build_freezes_entire_reserve_and_whole_source_exclusion() -> None:
    """完整 reserve 与 Firefox 整包派生禁训边界不可缩减。"""
    value = module.build_normalization_recovery_v3_evaluation_commitment(
        _prior_manifest())
    assert value["denominator"] == {
        "context_count": 31,
        "coverage_count": 1_557,
        "identity_count": 52,
        "label_blind": 1,
        "local_mapping_count": 80,
        "phrase_count": 1_505,
        "record_count": 1_558,
        "selection": "ENTIRE_PRIOR_UNREAD_RESERVE_WITHOUT_RESELECTION",
    }
    assert value["reserve_identity_read_count"] == 0
    assert value["reserve_payload_read_count"] == 0
    assert value["training_source_read_count"] == 0
    assert value["source_exclusion"][
        "derivative_message_or_pair_allowed_in_v3_train"] == 0
    assert value["formal_contract"][
        "candidate_applicability_cannot_shrink_denominator"] == 1
    assert value["formal_contract"][
        "label_materialization_allowed_before_candidate_code_family_freeze"] == 0
    assert len(value["dimensions"]) == 6


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("reserve_identity", "record_count"), 1_557),
        (("reserve_identity", "sha256"), "0" * 64),
        (("inventory_summary", "context_reserve_count"), 30),
        (("source_pack_manifest_sha256",), "0" * 64),
    ),
)
def test_build_rejects_prior_identity_drift(
        path: tuple[str, ...], replacement: object) -> None:
    """旧 reserve/source 任一身份漂移均失败关闭。"""
    value = _prior_manifest()
    target: dict[str, object] = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = replacement
    with pytest.raises(BroadQaExternalDataError, match="commitment 漂移"):
        module.build_normalization_recovery_v3_evaluation_commitment(value)


def test_publish_and_read_are_manifest_only_and_immutable(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """发布/回读均不打开 reserve，且 target 不可覆盖。"""
    prior = tmp_path / "prior"
    prior.mkdir()
    target = tmp_path / "commitment"
    calls = _install_reader(monkeypatch)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    published = module.publish_normalization_recovery_v3_evaluation_commitment(
        run_root=tmp_path,
        prior_evaluation_protocol_dir=prior,
        target_dir=target,
    )
    assert calls == [prior]
    stored = module.read_normalization_recovery_v3_evaluation_commitment(
        target,
        prior_evaluation_protocol_dir=prior,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert calls == [prior, prior]
    assert stored == published
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        module.publish_normalization_recovery_v3_evaluation_commitment(
            run_root=tmp_path,
            prior_evaluation_protocol_dir=prior,
            target_dir=target,
        )


def test_reader_rejects_synchronized_manifest_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步重算外层 SHA 也不能修改分母或判分合同。"""
    prior = tmp_path / "prior"
    prior.mkdir()
    target = tmp_path / "commitment"
    _install_reader(monkeypatch)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    published = module.publish_normalization_recovery_v3_evaluation_commitment(
        run_root=tmp_path,
        prior_evaluation_protocol_dir=prior,
        target_dir=target,
    )
    manifest_path = target / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    value["denominator"]["record_count"] = 1_557
    encoded = canonical_json_line(value)
    manifest_path.write_bytes(encoded)
    tampered_sha = hashlib.sha256(encoded).hexdigest()
    assert tampered_sha != published["manifest_sha256"]
    with pytest.raises(BroadQaExternalDataError, match="字段漂移"):
        module.read_normalization_recovery_v3_evaluation_commitment(
            target,
            prior_evaluation_protocol_dir=prior,
            expected_manifest_sha256=tampered_sha,
        )
