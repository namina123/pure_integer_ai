"""D-03 LC-16 后继 overlay 的课程、覆盖、边界与分账对抗测试。"""
from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_d03_lc16_overlay_catalog as catalog
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_catalog import (
    DIRECTIONAL_PARENT_PATH,
    DIRECTIONAL_PARENT_SHA256,
    OVERLAY_MANIFEST_PATH,
    TYPED_CARRIER_PACK_SHA256,
    build_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    D03Lc16SuccessorOverlay,
    read_d03_lc16_successor_overlay,
    verify_d03_lc16_overlay_files,
    write_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_records import (
    D03Lc16OverlayError,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import (
    RENDERER_KEY,
    RENDERER_MODE,
    RENDERER_VERSION,
    SCOPE_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    DIRECTIONS,
    IN_SCOPE_CARRIER_KEYS,
    W02_RECEIPT_SHA256,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    SAMPLE_KINDS,
    SAMPLE_SPLITS,
    read_typed_carrier_pack_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAL_PATH = REPOSITORY_ROOT / Path(*OVERLAY_MANIFEST_PATH.split("/"))
D03_GLOBAL_SHA256 = (
    "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6")
D03_RECEIPT_SHA256 = (
    "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef")


def _sha(relative_path: str) -> str:
    """返回仓内文件的 SHA-256。"""
    path = REPOSITORY_ROOT / Path(*relative_path.split("/"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built_overlay() -> D03Lc16SuccessorOverlay:
    """读取已经 append-only 发布的唯一历史 overlay。"""
    return read_d03_lc16_successor_overlay(FORMAL_PATH)


def test_formal_overlay_stays_canonical_and_current_rebuild_fails_closed(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """正式字节保持冻结；证据演进后不得伪装成当前可重建对象。"""
    formal = read_d03_lc16_successor_overlay(FORMAL_PATH)
    assert formal == built_overlay
    assert formal.canonical_bytes() == FORMAL_PATH.read_bytes()
    assert formal.sha256() == _sha(OVERLAY_MANIFEST_PATH)
    with pytest.raises(D03Lc16OverlayError, match="身份漂移"):
        verify_d03_lc16_overlay_files(formal, repository_root=REPOSITORY_ROOT)
    with pytest.raises(catalog.D03Lc16OverlayCatalogError, match="无法严格回验"):
        build_d03_lc16_successor_overlay(REPOSITORY_ROOT)


def test_overlay_preserves_d03_v1_and_binds_directional_parent(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """后继只能追加绑定，不得覆盖 D-03 v1、W-02/W-03 或方向 parent。"""
    assert built_overlay.parent_directional_manifest.relative_path == (
        DIRECTIONAL_PARENT_PATH)
    assert built_overlay.parent_directional_manifest.sha256 == (
        DIRECTIONAL_PARENT_SHA256)
    assert built_overlay.d03_v1_global_manifest.sha256 == D03_GLOBAL_SHA256
    assert built_overlay.d03_v1_publication_receipt.sha256 == D03_RECEIPT_SHA256
    assert built_overlay.typed_carrier_pack.sha256 == TYPED_CARRIER_PACK_SHA256
    assert built_overlay.w02_parent_receipt_sha256 == W02_RECEIPT_SHA256
    assert _sha(FORMAL_GLOBAL_MANIFEST_PATH) == D03_GLOBAL_SHA256
    assert _sha(FORMAL_RECEIPT_PATH) == D03_RECEIPT_SHA256
    assert built_overlay.execution_state.to_value()["d03_v1_preserved"] == 1


def test_nine_courses_bind_materialized_cases_owner_split_parser_and_renderer(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """九类课程必须把 63 个 payload 与 parent owner/split 及 parser/renderer 绑死。"""
    parent = read_typed_carrier_pack_manifest(
        REPOSITORY_ROOT / "data/ph2/manifests/lc16_typed_carrier_pack_v1.json")
    parent_cases = {item.case_key: item for item in parent.cases}
    assert tuple(item.carrier_key for item in built_overlay.carrier_courses) == (
        IN_SCOPE_CARRIER_KEYS)
    seen_cases = set()
    seen_owners = set()
    for course in built_overlay.carrier_courses:
        assert course.course_state == "PAYLOAD_FROZEN_NOT_QUALIFIED"
        assert tuple(item.sample_kind for item in course.cases) == SAMPLE_KINDS
        assert (course.renderer_key, course.renderer_version,
                course.renderer_mode) == (
                    RENDERER_KEY, RENDERER_VERSION, RENDERER_MODE)
        assert course.parser_package
        assert course.parser_version
        assert course.budget.to_value()["carrier_key"] == course.carrier_key
        for case in course.cases:
            parent_case = parent_cases[case.case_key]
            assert case.owner_key == parent_case.owner_key
            assert case.split == parent_case.split == SAMPLE_SPLITS[case.sample_kind]
            assert case.directions == DIRECTIONS
            assert case.materialization_size_bytes > 0
            assert len(case.materialization_sha256) == 64
            seen_cases.add(case.case_key)
            seen_owners.add(case.owner_key)
    assert len(seen_cases) == len(IN_SCOPE_CARRIER_KEYS) * len(SAMPLE_KINDS)
    assert len(seen_owners) == len(seen_cases)


def test_carrier_direction_scope_matrix_is_complete_and_w04_stays_locked(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """9×3×8 单元必须全部显式存在，资格完成前 W-04 以后保持锁定。"""
    cells = built_overlay.coverage_cells
    assert len(cells) == (
        len(IN_SCOPE_CARRIER_KEYS) * len(DIRECTIONS) * len(SCOPE_KEYS))
    assert len({item.key for item in cells}) == len(cells)
    assert {item.current_state for item in cells if item.scope_key in {
        "BOUNDARY_OOV", "SENSE_CONCEPT",
    }} == {"SUPPLEMENTAL_QUALIFICATION_REQUIRED"}
    assert {item.current_state for item in cells if item.scope_key not in {
        "BOUNDARY_OOV", "SENSE_CONCEPT",
    }} == {"LOCKED_NOT_STARTED"}
    state = built_overlay.execution_state.to_value()
    assert state["W04_STARTED"] == 0
    assert state["w02_lc16_supplemental_qualified"] == 0
    assert state["w03_lc16_supplemental_qualified"] == 0


def test_evaluator_boundaries_separate_decidable_ne_and_future_private_runs(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """每个独立 evaluator 都必须显式分离可判维度、NE 与未运行事实。"""
    assert len(built_overlay.evaluator_boundaries) == 5
    for boundary in built_overlay.evaluator_boundaries:
        assert boundary.decidable_dimensions
        assert boundary.ne_conditions
        assert not (set(boundary.decidable_dimensions)
                    & set(boundary.ne_conditions))
    private = [
        item for item in built_overlay.evaluator_boundaries
        if item.owner_key == "PH2_PRIVATE_EVALUATOR"
    ]
    assert len(private) == 2
    assert {item.implementation_state for item in private} == {"FROZEN_NOT_RUN"}
    assert {item.ne_policy for item in private} == {
        "BLOCK_QUALIFICATION_ON_NE"}


def test_generation_replay_and_open_generation_are_never_aggregated(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """来源化 surface 回放不得替开放生成获得 PASS。"""
    by_key = {item.account_key: item
              for item in built_overlay.generation_accounts}
    replay = by_key["SOURCE_GROUNDED_SURFACE_REPLAY"]
    open_generation = by_key["OPEN_GENERATION"]
    assert replay.current_status == "BOUNDED_RUNTIME_EVIDENCED"
    assert replay.runtime_evidenced == 1
    assert replay.included_in_current_directional_evidence == 1
    assert open_generation.current_status == "NE_NOT_YET_EVALUABLE"
    assert open_generation.runtime_evidenced == 0
    assert open_generation.included_in_current_directional_evidence == 0
    assert all(item.aggregate_with_other_account == 0 for item in by_key.values())
    assert built_overlay.execution_state.to_value()["open_generation_pass"] == 0


def test_resource_and_failure_suffixes_are_bounded_and_complete(
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """W-02/W-03 补充资格必须只读有界，失败必须传播到完整下游。"""
    assert {item.family_key for item in built_overlay.resource_budgets} == {
        "W-02-LC16-SUPPLEMENTAL", "W-03-LC16-SUPPLEMENTAL"}
    for budget in built_overlay.resource_budgets:
        assert budget.max_records == 63
        assert budget.max_direction_evaluations == 189
        assert budget.max_workers == 4
        assert budget.max_host_writes == 0
    by_key = {item.failure_key: item
              for item in built_overlay.failure_dependencies}
    assert by_key["W02_SUPPLEMENTAL_FAIL_OR_NE"].invalidation_suffix[-2:] == (
        "J-LC-W09", "J-F2")
    assert "W-04" in (
        by_key["W03_SUPPLEMENTAL_FAIL_OR_NE"].invalidation_suffix)
    assert by_key["OPEN_GENERATION_FAIL_OR_NE"].invalidation_suffix == (
        "W-08", "W-09", "J-LC-W09", "J-F2")


@pytest.mark.parametrize(
    "mutate, message",
    (
        (lambda value: value["coverage_cells"].pop(), "scope"),
        (lambda value: value["carrier_courses"][0].update(
            {"parser_version": "drift"}), "parser"),
        (lambda value: value["carrier_courses"][0]["cases"][0].update(
            {"split": "held_out"}), "split"),
        (lambda value: value["generation_accounts"][1].update(
            {"aggregate_with_other_account": 1}), "聚合"),
        (lambda value: value["execution_state"].update(
            {"W04_STARTED": 1}), "execution_state"),
        (lambda value: value["resource_budgets"][0].update(
            {"max_host_writes": 1}), "只读"),
    ),
)
def test_contract_fails_closed_on_bearing_drift(
        built_overlay: D03Lc16SuccessorOverlay,
        mutate,
        message: str,
        ) -> None:
    """删除承重 cell 或放宽边界都必须在回读阶段失败。"""
    payload = copy.deepcopy(built_overlay.to_dict())
    mutate(payload)
    with pytest.raises(D03Lc16OverlayError, match=message):
        D03Lc16SuccessorOverlay.from_dict(payload)


def test_writer_is_append_only_and_rejects_different_payload(
        tmp_path: Path,
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """overlay 写入必须可幂等重放但不得覆盖异内容。"""
    target = tmp_path / "overlay.json"
    assert write_d03_lc16_successor_overlay(built_overlay, target) == target
    assert write_d03_lc16_successor_overlay(built_overlay, target) == target
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(D03Lc16OverlayError, match="内容不同"):
        write_d03_lc16_successor_overlay(built_overlay, target)


def test_file_verifier_detects_bound_dependency_drift(
        tmp_path: Path,
        built_overlay: D03Lc16SuccessorOverlay,
        ) -> None:
    """任一直接绑定文件改变都必须令 overlay 文件回验失败。"""
    identities = [
        built_overlay.parent_directional_manifest,
        built_overlay.d03_v1_global_manifest,
        built_overlay.d03_v1_publication_receipt,
        built_overlay.typed_carrier_pack,
        built_overlay.w03_parent_receipt,
        *(item.manifest_identity for item in built_overlay.carrier_courses),
        *(item.sample_identity for item in built_overlay.carrier_courses),
        *(item.file_identity for item in built_overlay.evidence_files),
    ]
    for identity in identities:
        source = REPOSITORY_ROOT / Path(*identity.relative_path.split("/"))
        target = tmp_path / Path(*identity.relative_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    drifted = tmp_path / Path(*identities[-1].relative_path.split("/"))
    drifted.write_bytes(drifted.read_bytes() + b" ")
    with pytest.raises(D03Lc16OverlayError, match="身份漂移"):
        verify_d03_lc16_overlay_files(built_overlay, repository_root=tmp_path)


def test_catalog_rejects_frozen_directional_parent_sha_drift(monkeypatch) -> None:
    """目录构造器不得用运行时实际 hash 悄悄替换冻结 parent。"""
    monkeypatch.setattr(catalog, "DIRECTIONAL_PARENT_SHA256", "0" * 64)
    with pytest.raises(catalog.D03Lc16OverlayCatalogError, match="directional"):
        catalog.build_d03_lc16_successor_overlay(REPOSITORY_ROOT)
