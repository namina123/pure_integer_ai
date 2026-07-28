"""R-03 修正、精确来源撤回和恢复生产吸收 artifact 测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_correction_recovery_catalog import (
    MANIFEST_PATH,
    build_correction_recovery_manifest,
)
from pure_integer_ai.experiments.ph2_correction_recovery_contract import (
    CorrectionRecoveryContractError,
    CorrectionRecoveryEvidenceFile,
    read_correction_recovery_manifest,
    verify_correction_recovery_files,
    write_correction_recovery_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_manifest():
    """从最终实现和直接依赖字节构建正式 R-03 证据。"""
    return build_correction_recovery_manifest(_REPOSITORY_ROOT)


def test_manifest_round_trip_and_file_identity(
        tmp_path: Path,
        formal_manifest):
    """R-03 artifact 可规范回读并逐字节闭合全部证据。"""
    target = tmp_path / "r03_correction_recovery_absorption_v1.json"
    assert write_correction_recovery_manifest(formal_manifest, target) == target
    assert read_correction_recovery_manifest(target) == formal_manifest
    verify_correction_recovery_files(
        formal_manifest,
        repository_root=_REPOSITORY_ROOT,
    )


def test_manifest_rejects_fake_fault_coverage(formal_manifest):
    """三故障点证据不能由较小计数冒充。"""
    coverage = formal_manifest.coverage_contract.to_value()
    coverage["three_fault_points"] = 2
    with pytest.raises(
            CorrectionRecoveryContractError,
            match="coverage contract 漂移"):
        replace(
            formal_manifest,
            coverage_contract=CanonicalJsonObject.from_value(coverage),
        )


def test_manifest_rejects_file_identity_drift(formal_manifest):
    """任一生产或测试文件漂移都必须失败关闭。"""
    first = formal_manifest.evidence_files[0]
    drifted = CorrectionRecoveryEvidenceFile(
        first.relative_path,
        first.role,
        first.byte_count,
        "0" * 64,
    )
    manifest = replace(
        formal_manifest,
        evidence_files=(drifted, *formal_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            CorrectionRecoveryContractError,
            match="evidence 文件身份漂移"):
        verify_correction_recovery_files(
            manifest,
            repository_root=_REPOSITORY_ROOT,
        )


def test_manifest_is_idempotent_but_non_overwritable(
        tmp_path: Path,
        formal_manifest):
    """同字节可重放，同版本不同字节不得覆盖。"""
    target = tmp_path / "r03_correction_recovery_absorption_v1.json"
    write_correction_recovery_manifest(formal_manifest, target)
    assert write_correction_recovery_manifest(formal_manifest, target) == target
    first = formal_manifest.evidence_files[0]
    different = replace(
        formal_manifest,
        evidence_files=(
            replace(first, sha256="f" * 64),
            *formal_manifest.evidence_files[1:],
        ),
    )
    with pytest.raises(
            CorrectionRecoveryContractError,
            match="已存在且内容不同"):
        write_correction_recovery_manifest(different, target)


def test_stored_manifest_is_current_and_readable():
    """仓内正式 R-03 artifact 必须等于当前确定构建。"""
    stored = read_correction_recovery_manifest(
        _REPOSITORY_ROOT / MANIFEST_PATH)
    expected = build_correction_recovery_manifest(_REPOSITORY_ROOT)
    assert stored == expected
    verify_correction_recovery_files(
        stored,
        repository_root=_REPOSITORY_ROOT,
    )
