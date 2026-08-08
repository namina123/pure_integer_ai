"""W-02 morphology successor freeze、guard 与公开状态专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_contract import (
    W02_MORPH_SUCCESSOR_CODE_PATHS,
    W02_MORPH_SUCCESSOR_FREEZE_PATH,
    W02_MORPH_SUCCESSOR_GUARD_AVAILABLE,
    W02_MORPH_SUCCESSOR_GUARD_CONSUMED,
    W02_MORPH_SUCCESSOR_RUN_INTENT,
    W02MorphologySuccessorContractError,
    build_w02_morphology_successor_runtime_freeze,
    consume_w02_morphology_successor_guard,
    publish_w02_morphology_successor_guard,
    read_w02_morphology_successor_runtime_freeze,
    verify_w02_morphology_successor_consumed_guard,
    w02_morphology_successor_guard_value,
)


ROOT = Path(__file__).resolve().parents[1]


def test_successor_runtime_freeze_binds_parent_fail_and_live_code() -> None:
    freeze = read_w02_morphology_successor_runtime_freeze(ROOT)
    rebuilt = build_w02_morphology_successor_runtime_freeze(ROOT)
    assert freeze == rebuilt
    assert tuple(item.repository_path for item in freeze.code_files) == (
        W02_MORPH_SUCCESSOR_CODE_PATHS)
    assert freeze.expected_overlay_semantic_sha256 == (
        "a72c0ccb0d054537c04ac3e683a5730a7e81b767339957e2957e1cc172de0676")
    value = freeze.to_dict()
    assert value["parent_dev_status"] == "FAIL"
    assert value["parent_formal_training_runs"] == 1
    assert value["formal_successor_transform_runs"] == 0
    assert value["formal_training_runs"] == 0
    assert value["private_family_registered"] == 0
    target = ROOT / Path(*W02_MORPH_SUCCESSOR_FREEZE_PATH.split("/"))
    assert target.read_bytes() == freeze.canonical_bytes()


def test_successor_guard_is_bound_and_consumed_once(tmp_path: Path) -> None:
    freeze = read_w02_morphology_successor_runtime_freeze(ROOT)
    root = tmp_path / "formal-successor"
    guard_sha = publish_w02_morphology_successor_guard(root, freeze)
    available = root / Path(*W02_MORPH_SUCCESSOR_GUARD_AVAILABLE.split("/"))
    expected = w02_morphology_successor_guard_value(
        runtime_code_freeze_sha256=freeze.runtime_code_freeze_sha256,
        parent_candidate_manifest_sha256=freeze.parent_candidate_manifest_sha256,
        expected_overlay_semantic_sha256=freeze.expected_overlay_semantic_sha256,
    )
    assert available.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert hashlib.sha256(available.read_bytes()).hexdigest() == guard_sha
    run_identity = hashlib.sha256(b"public-guard-test-run").hexdigest()
    consume_w02_morphology_successor_guard(
        root,
        expected_guard_sha256=guard_sha,
        run_id=1,
        run_identity_sha256=run_identity,
    )
    consumed = root / Path(*W02_MORPH_SUCCESSOR_GUARD_CONSUMED.split("/"))
    intent = root / Path(*W02_MORPH_SUCCESSOR_RUN_INTENT.split("/"))
    assert not available.exists()
    assert consumed.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert read_canonical_object(intent)["formal_successor_transform_runs"] == 1
    verify_w02_morphology_successor_consumed_guard(
        root,
        expected_guard_sha256=guard_sha,
        run_id=1,
        run_identity_sha256=run_identity,
    )
    with pytest.raises(W02MorphologySuccessorContractError, match="intent"):
        verify_w02_morphology_successor_consumed_guard(
            root,
            expected_guard_sha256=guard_sha,
            run_id=1,
            run_identity_sha256="1" * 64,
        )
    with pytest.raises(W02MorphologySuccessorContractError, match="已消费"):
        consume_w02_morphology_successor_guard(
            root,
            expected_guard_sha256=guard_sha,
            run_id=1,
            run_identity_sha256=run_identity,
        )


def test_successor_guard_rejects_wrong_sha_before_consumption(tmp_path: Path) -> None:
    freeze = read_w02_morphology_successor_runtime_freeze(ROOT)
    root = tmp_path / "formal-successor"
    publish_w02_morphology_successor_guard(root, freeze)
    available = root / Path(*W02_MORPH_SUCCESSOR_GUARD_AVAILABLE.split("/"))
    before = available.read_bytes()
    with pytest.raises(W02MorphologySuccessorContractError, match="字节漂移"):
        consume_w02_morphology_successor_guard(
            root,
            expected_guard_sha256="0" * 64,
            run_id=1,
            run_identity_sha256=hashlib.sha256(b"wrong-sha-test").hexdigest(),
        )
    assert available.read_bytes() == before
    assert not (root / Path(*W02_MORPH_SUCCESSOR_GUARD_CONSUMED.split("/"))).exists()
