"""LC-COVERAGE-V2 全任务、载体、方向缺口基线测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_language_coverage_v2_catalog import (
    MANIFEST_PATH,
    build_language_capability_coverage_v2,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    CARRIER_KEYS,
    DIRECTIONS,
    EXECUTION_STATE,
    INVARIANTS,
    TASK_KEYS,
    W02_RECEIPT_SHA256,
    LanguageCoverageV2Error,
    read_language_capability_coverage_v2,
    verify_language_capability_coverage_v2_files,
    write_language_capability_coverage_v2,
)


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_manifest():
    return build_language_capability_coverage_v2(REPOSITORY)


def test_v2_build_is_deterministic_and_matches_canonical_manifest(
        built_manifest):
    rebuilt = build_language_capability_coverage_v2(REPOSITORY)
    assert rebuilt.canonical_bytes() == built_manifest.canonical_bytes()
    path = REPOSITORY / MANIFEST_PATH
    assert path.read_bytes() == built_manifest.canonical_bytes()
    assert read_language_capability_coverage_v2(path) == built_manifest


def test_all_tasks_carriers_and_directions_are_explicit(built_manifest):
    assert tuple(item.task_key for item in built_manifest.task_records) == TASK_KEYS
    assert tuple(
        item.carrier_key for item in built_manifest.carrier_records
    ) == CARRIER_KEYS
    assert len(built_manifest.cells) == 16 * 10 * 3 == 480
    assert tuple(item.key for item in built_manifest.cells) == tuple(
        (task, carrier, direction)
        for task in TASK_KEYS
        for carrier in CARRIER_KEYS
        for direction in DIRECTIONS
    )


def test_v2_splits_legacy_mixed_scope_without_rewriting_v1(built_manifest):
    split = built_manifest.legacy_split
    assert split.legacy_capability_key == "NON_TEXT_MEDIA"
    assert split.in_scope_capability_key == "TYPED_ARTIFACT_CARRIERS"
    assert split.wall_capability_key == "SENSORY_GROUNDING"
    assert split.migration_state == "DEPRECATED_SPLIT_ONLY"
    v1_path = REPOSITORY / "data/ph2/manifests/language_capability_baseline_v39.json"
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == (
        "7c96579c900e9ca25390abd097d7f330d949fcf9e4288b7a311367d03e7f18f4")


def test_typed_carrier_cells_are_required_and_honestly_absent(built_manifest):
    typed = tuple(
        item for item in built_manifest.cells
        if item.carrier_key != "SENSORY_GROUNDING")
    assert len(typed) == 16 * 9 * 3
    for item in typed:
        assert item.applicability == "REQUIRED"
        assert {
            item.observation_state,
            item.representation_state,
            item.adapter_state,
            item.projection_state,
            item.consumer_state,
            item.verifier_state,
            item.retention_state,
            item.coverage_state,
        } == {"ABSENT"}
        assert "HISTORICAL_RECEIPTS_DO_NOT_EXTEND_TO_THIS_CELL" in item.ne_reasons


def test_sensory_grounding_stays_wall_blocked_in_every_cell(built_manifest):
    wall = tuple(
        item for item in built_manifest.cells
        if item.carrier_key == "SENSORY_GROUNDING")
    assert len(wall) == 16 * 3
    for item in wall:
        assert item.applicability == "WALL"
        assert {
            item.observation_state,
            item.representation_state,
            item.adapter_state,
            item.projection_state,
            item.consumer_state,
            item.verifier_state,
            item.retention_state,
            item.coverage_state,
        } == {"WALL_BLOCKED"}


def test_lc16_is_audited_absent_and_old_tasks_only_keep_historical_scope(
        built_manifest):
    by_key = {item.task_key: item for item in built_manifest.task_records}
    assert by_key["LC-16"].baseline_state == "AUDITED_ABSENT"
    assert by_key["LC-16"].historical_scope_authority == 0
    assert all(
        by_key[key].baseline_state == "HISTORICAL_SCOPE_ONLY"
        and by_key[key].historical_scope_authority == 1
        for key in TASK_KEYS if key != "LC-16")
    assert all(
        item.carrier_qualified_runtime_authority == 0
        for item in built_manifest.task_records)


def test_v39_v41_d03_and_receipt_evidence_are_exact(built_manifest):
    verify_language_capability_coverage_v2_files(
        built_manifest, repository_root=REPOSITORY)
    by_role = {
        item.role: item for item in built_manifest.evidence_files
        if item.role != "IMPLEMENTATION"
    }
    assert by_role["COVERAGE_BASE"].sha256 == (
        "7c96579c900e9ca25390abd097d7f330d949fcf9e4288b7a311367d03e7f18f4")
    assert by_role["LINEAGE_HEAD"].sha256 == (
        "386b08975bc4368ad52e95a997a3d362fa37ff6f19eecb3043ba03cf09cc5c5c")
    assert by_role["D03_GLOBAL"].sha256 == (
        "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6")
    assert by_role["D03_RECEIPT"].sha256 == (
        "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef")
    assert by_role["W03_RECEIPT"].sha256 == (
        "ef64636ab287eacbacae4040f59da74bb4105374cba31d756e1ddefaf86043f6")
    assert built_manifest.w02_receipt_sha256 == W02_RECEIPT_SHA256


def test_v2_keeps_training_mastery_readiness_and_w04_closed(built_manifest):
    assert built_manifest.invariants.to_value() == INVARIANTS
    assert built_manifest.execution_state.to_value() == EXECUTION_STATE
    assert built_manifest.invariants.to_value()["carrier_qualified_passes"] == 0
    assert built_manifest.execution_state.to_value()["W04_STARTED"] == 0


def test_contract_rejects_cell_omission_and_pass_inflation(built_manifest):
    with pytest.raises(LanguageCoverageV2Error, match="单元必须精确列全"):
        replace(built_manifest, cells=built_manifest.cells[:-1])
    cell = next(
        item for item in built_manifest.cells
        if item.carrier_key == "SOURCE_CODE" and item.task_key == "LC-16")
    with pytest.raises(LanguageCoverageV2Error, match="cell state 非法"):
        replace(cell, coverage_state="RUNTIME_EVIDENCED")


def test_contract_rejects_mastery_and_w04_state_inflation(built_manifest):
    state = dict(EXECUTION_STATE)
    state["LANGUAGE_CAPABILITY_MASTERED"] = 1
    with pytest.raises(LanguageCoverageV2Error, match="execution_state"):
        replace(
            built_manifest,
            execution_state=CanonicalJsonObject.from_value(state),
        )
    state = dict(EXECUTION_STATE)
    state["W04_STARTED"] = 1
    with pytest.raises(LanguageCoverageV2Error, match="execution_state"):
        replace(
            built_manifest,
            execution_state=CanonicalJsonObject.from_value(state),
        )


def test_nonoverwrite_writer_is_idempotent_and_rejects_corruption(
        tmp_path, built_manifest):
    path = tmp_path / MANIFEST_PATH.name
    write_language_capability_coverage_v2(built_manifest, path)
    write_language_capability_coverage_v2(built_manifest, path)
    assert read_language_capability_coverage_v2(path) == built_manifest
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(LanguageCoverageV2Error, match="内容不同"):
        write_language_capability_coverage_v2(built_manifest, path)
