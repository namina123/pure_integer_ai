"""W-02 morphology successor V2 freeze 与 guard 专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_contract import (
    W02_MORPH_V2_CODE_PATHS,
    W02_MORPH_V2_FREEZE_PATH,
    W02_MORPH_V2_GUARD_AVAILABLE,
    W02_MORPH_V2_GUARD_CONSUMED,
    W02_MORPH_V2_RUN_INTENT,
    W02MorphologySuccessorV2ContractError,
    build_w02_morphology_successor_v2_runtime_freeze,
    consume_w02_morphology_successor_v2_guard,
    publish_w02_morphology_successor_v2_guard,
    read_w02_morphology_successor_v2_runtime_freeze,
    verify_w02_morphology_successor_v2_consumed_guard,
    w02_morphology_successor_v2_guard_value,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_runtime_freeze_binds_both_parents_preflight_and_code() -> None:
    freeze = read_w02_morphology_successor_v2_runtime_freeze(ROOT)
    assert freeze == build_w02_morphology_successor_v2_runtime_freeze(ROOT)
    assert tuple(item.repository_path for item in freeze.code_files) == (
        W02_MORPH_V2_CODE_PATHS)
    value = freeze.to_dict()
    assert value["expected_counts"] == {
        "accepted_lexeme_rows": 112,
        "accepted_support_count": 651,
        "logic_operations": 4_611,
        "rule_row_count": 383,
        "unsupported_lexeme_rows": 4,
        "unsupported_support_count": 4,
    }
    assert value["formal_successor_v2_transform_runs"] == 0
    assert value["parent_formal_training_runs"] == 1
    assert value["parent_formal_successor_transform_runs"] == 1
    target = ROOT / Path(*W02_MORPH_V2_FREEZE_PATH.split("/"))
    assert target.read_bytes() == freeze.canonical_bytes()


def test_v2_guard_is_bound_and_consumed_once(tmp_path: Path) -> None:
    freeze = read_w02_morphology_successor_v2_runtime_freeze(ROOT)
    root = tmp_path / "formal-v2"
    guard_sha = publish_w02_morphology_successor_v2_guard(root, freeze)
    available = root / Path(*W02_MORPH_V2_GUARD_AVAILABLE.split("/"))
    expected = w02_morphology_successor_v2_guard_value(
        runtime_code_freeze_sha256=freeze.runtime_code_freeze_sha256,
        parent_candidate_manifest_sha256=(
            freeze.parent_candidate_manifest_sha256),
        parent_v1_manifest_sha256=freeze.parent_v1_manifest_sha256,
        expected_semantic_sha256=freeze.expected_semantic_sha256)
    assert available.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert hashlib.sha256(available.read_bytes()).hexdigest() == guard_sha
    identity = hashlib.sha256(b"public-v2-guard-test").hexdigest()
    consume_w02_morphology_successor_v2_guard(
        root, expected_guard_sha256=guard_sha,
        run_id=1, run_identity_sha256=identity)
    consumed = root / Path(*W02_MORPH_V2_GUARD_CONSUMED.split("/"))
    intent = root / Path(*W02_MORPH_V2_RUN_INTENT.split("/"))
    assert not available.exists()
    assert consumed.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert read_canonical_object(intent)[
        "formal_successor_v2_transform_runs"] == 1
    verify_w02_morphology_successor_v2_consumed_guard(
        root, expected_guard_sha256=guard_sha,
        run_id=1, run_identity_sha256=identity)
    with pytest.raises(W02MorphologySuccessorV2ContractError, match="intent"):
        verify_w02_morphology_successor_v2_consumed_guard(
            root, expected_guard_sha256=guard_sha,
            run_id=1, run_identity_sha256="1" * 64)
    with pytest.raises(W02MorphologySuccessorV2ContractError, match="已消费"):
        consume_w02_morphology_successor_v2_guard(
            root, expected_guard_sha256=guard_sha,
            run_id=1, run_identity_sha256=identity)


def test_v2_guard_rejects_wrong_sha_before_consumption(tmp_path: Path) -> None:
    freeze = read_w02_morphology_successor_v2_runtime_freeze(ROOT)
    root = tmp_path / "formal-v2"
    publish_w02_morphology_successor_v2_guard(root, freeze)
    available = root / Path(*W02_MORPH_V2_GUARD_AVAILABLE.split("/"))
    before = available.read_bytes()
    with pytest.raises(W02MorphologySuccessorV2ContractError, match="字节漂移"):
        consume_w02_morphology_successor_v2_guard(
            root, expected_guard_sha256="0" * 64,
            run_id=1,
            run_identity_sha256=hashlib.sha256(b"wrong-v2").hexdigest())
    assert available.read_bytes() == before
    assert not (root / Path(*W02_MORPH_V2_GUARD_CONSUMED.split("/"))).exists()
