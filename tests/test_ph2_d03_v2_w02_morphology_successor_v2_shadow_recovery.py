"""W-02 successor V2 shadow recovery 的公开合同测试。"""
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery import (
    W02_MORPH_V2_SHADOW_ABORTED_FAILURE_SEAL_SHA256,
    W02_MORPH_V2_SHADOW_RECOVERY_CODE_PATHS,
    W02_MORPH_V2_SHADOW_RECOVERY_FAMILY_NAME,
    W02MorphologySuccessorV2ShadowRecoveryError,
    build_w02_morphology_successor_v2_shadow_recovery_freeze,
    consume_w02_morphology_successor_v2_shadow_recovery_guard,
    publish_w02_morphology_successor_v2_shadow_recovery_guard,
    read_w02_morphology_successor_v2_shadow_recovery_state,
)


def test_shadow_recovery_freeze_binds_aborted_family_and_parent() -> None:
    repository = Path(__file__).resolve().parents[1]
    value = build_w02_morphology_successor_v2_shadow_recovery_freeze(repository)
    assert value["status"] == (
        "W02_MORPHOLOGY_SUCCESSOR_V2_SHADOW_RECOVERY_FREEZE_COMPLETE")
    assert value["aborted_failure_seal_sha256"] == (
        W02_MORPH_V2_SHADOW_ABORTED_FAILURE_SEAL_SHA256)
    assert value["aborted_formal_shadow_attempts"] == 1
    assert value["aborted_formal_shadow_passes"] == 0
    assert value["formal_shadow_recovery_runs"] == 0
    assert value["private_payload_reads"] == 0
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_V2_SHADOW_RECOVERY_CODE_PATHS)


def test_shadow_recovery_guard_is_single_use(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / W02_MORPH_V2_SHADOW_RECOVERY_FAMILY_NAME
    publish_w02_morphology_successor_v2_shadow_recovery_guard(
        repository, root)
    state = consume_w02_morphology_successor_v2_shadow_recovery_guard(
        repository, root)
    assert state == read_w02_morphology_successor_v2_shadow_recovery_state(
        repository, root)
    with pytest.raises(
            W02MorphologySuccessorV2ShadowRecoveryError,
            match="已消费"):
        consume_w02_morphology_successor_v2_shadow_recovery_guard(
            repository, root)
