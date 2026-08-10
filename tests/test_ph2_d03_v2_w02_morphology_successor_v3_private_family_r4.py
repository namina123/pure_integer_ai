"""Public freeze and guard tests for the successor V3 R4 family."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_family_r4 import (
    W02_MORPH_V3_PRIVATE_R4_FORMAL_FAMILY_NAME,
    W02MorphologySuccessorV3PrivateR4FamilyError,
    build_w02_morphology_successor_v3_private_r4_family_freeze,
    consume_w02_morphology_successor_v3_private_r4_guard,
    publish_w02_morphology_successor_v3_private_r4_family_root,
    read_w02_morphology_successor_v3_private_r4_family_freeze,
    verify_w02_morphology_successor_v3_private_r4_consumed_guard,
    w02_morphology_successor_v3_private_r4_run_identity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_contract import (
    W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_family_r4 import (
    W02_MORPH_V3_PRIVATE_R4_FAMILY_FREEZE_PATH,
)


def test_r4_family_freeze_binds_v5_owner_and_consumed_r3() -> None:
    repository = Path(__file__).resolve().parents[1]
    built = build_w02_morphology_successor_v3_private_r4_family_freeze(
        repository)
    frozen = read_w02_morphology_successor_v3_private_r4_family_freeze(
        repository)

    assert frozen == built
    assert frozen["status"] == "W02_SUCCESSOR_V3_R4_BLIND_PRIVATE_FAMILY_FROZEN"
    assert frozen["owner_metadata_sha256"] == W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256
    assert frozen["owner_source_count"] == 500
    assert frozen["owner_pair_count"] == 500
    assert frozen["previous_consumed_r2_reuse_authorized"] == 0
    assert frozen["previous_consumed_r3_reuse_authorized"] == 0
    assert frozen["private_payload_reads"] == 0
    assert frozen["formal_private_evaluation_runs"] == 0
    assert all(
        row["license_ids"] == ["CC-BY-SA-4.0"]
        for row in frozen["owner_input_files"])
    assert "source_extension_v5_file_sha256" in frozen["artifact_chain"]
    assert "source_extension_v5_code_file_sha256" in frozen["artifact_chain"]


def test_r4_guard_is_single_use(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / W02_MORPH_V3_PRIVATE_R4_FORMAL_FAMILY_NAME
    guard_sha = publish_w02_morphology_successor_v3_private_r4_family_root(
        repository, root)
    freeze = read_w02_morphology_successor_v3_private_r4_family_freeze(
        repository)
    freeze_path = repository / Path(
        *W02_MORPH_V3_PRIVATE_R4_FAMILY_FREEZE_PATH.split("/"))
    run_identity = w02_morphology_successor_v3_private_r4_run_identity(
        freeze, _sha256_file(freeze_path)[1])

    consume_w02_morphology_successor_v3_private_r4_guard(
        root, expected_guard_sha256=guard_sha,
        run_identity_sha256=run_identity)
    verify_w02_morphology_successor_v3_private_r4_consumed_guard(
        root, expected_guard_sha256=guard_sha,
        run_identity_sha256=run_identity)
    with pytest.raises(W02MorphologySuccessorV3PrivateR4FamilyError,
                       match="already consumed"):
        consume_w02_morphology_successor_v3_private_r4_guard(
            root, expected_guard_sha256=guard_sha,
            run_identity_sha256=run_identity)
