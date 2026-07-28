"""D-03 正式课程 catalog、全局 manifest 和 reader 的集成验收。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    COURSE_COMPILER_SPECS,
    FORMAL_GLOBAL_MANIFEST_PATH,
    build_d03_candidate,
)
from pure_integer_ai.experiments.ph2_d03_release_contract import (
    D03ContractError,
    STAGE_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_reader() -> D03ReleaseReader:
    """从正式全局 manifest 打开唯一 D-03 candidate reader。"""
    return D03ReleaseReader.open(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        require_publication=False,
    )


def _tree_identity(root: Path) -> tuple[tuple[str, int, str], ...]:
    """返回一个候选构建目录的稳定路径、大小和摘要。"""
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )


def test_formal_global_manifest_has_nine_stages_and_all_course_packs(formal_reader):
    """全局计划唯一回读九阶段，并冻结全部课程及公开 source pack。"""
    assert tuple(item.stage_identity.stage_key for item in formal_reader.stages) == (
        STAGE_KEYS
    )
    assert len(COURSE_COMPILER_SPECS) == 31
    assert len(formal_reader.global_manifest.pack_bindings) == 37
    assert formal_reader.global_manifest.publication_state.state == (
        "CANDIDATE_VERIFIED"
    )
    assert formal_reader.global_manifest.execution_state["d03_published"] == 0


def test_every_pack_has_physical_owner_separation_and_exact_identity(formal_reader):
    """所有 pack 的 train/held-out/evaluator/teacher 路径物理互斥且逐字节有效。"""
    for pack in formal_reader.global_manifest.pack_bindings:
        groups = (
            set(pack.train_observation_paths),
            set(pack.dev_observation_paths),
            set(pack.held_out_observation_paths),
            set(pack.teacher_evidence_paths),
            set(pack.evaluator_label_paths),
        )
        for index, group in enumerate(groups):
            for other in groups[index + 1:]:
                assert group.isdisjoint(other)
        assert all("/owners/teacher/" in path for path in pack.teacher_evidence_paths)
        assert all("/owners/evaluator/" in path for path in pack.evaluator_label_paths)
        formal_reader.verify_pack_files(pack.pack_key)


def test_candidate_visibility_never_returns_future_or_private_paths(formal_reader):
    """candidate 只见累计 train/source 路径，拒绝 future、held-out 和私有 label。"""
    for stage_key in STAGE_KEYS:
        view = formal_reader.visibility(stage_key, "candidate")
        assert view.payload_reads == 0
        assert view.payload_bytes == 0
        assert set(view.allowed_paths).isdisjoint(view.rejected_paths)
        assert all("/owners/" not in path for path in view.allowed_paths)
        assert all("held_out" not in path for path in view.allowed_paths)
        for path in view.rejected_paths:
            with pytest.raises(D03ContractError, match="不可见"):
                formal_reader.require_visible_path(stage_key, "candidate", path)


def test_fresh_resume_plan_and_invalidation_suffixes_are_exact(formal_reader):
    """同 identity 的 fresh/resume 起点相同，pack/evaluator/version 变化返回完整后缀。"""
    fresh = formal_reader.execution_suffix("W-01", mode="fresh")
    resume = formal_reader.execution_suffix("W-01", mode="resume")
    assert fresh == resume == STAGE_KEYS

    pack = next(
        item for item in formal_reader.global_manifest.pack_bindings
        if item.earliest_stage == "W-06"
    )
    pack_result = formal_reader.invalidation("PACK_CONTENT", pack.pack_key)
    assert pack_result.earliest_stage == "W-06"
    assert pack_result.invalidated_stage_keys == STAGE_KEYS[5:]
    evaluator_result = formal_reader.invalidation("EVALUATOR_VERSION", "W-08")
    assert evaluator_result.invalidated_stage_keys == STAGE_KEYS[7:]
    schema_result = formal_reader.invalidation("SCHEMA_VERSION", "GLOBAL")
    assert schema_result.invalidated_stage_keys == STAGE_KEYS
    with pytest.raises(D03ContractError, match="未知"):
        formal_reader.invalidation("PACK_CONTENT", "UNKNOWN-PACK")


def test_two_candidate_builds_are_byte_identical(tmp_path: Path):
    """两个独立输出根在双构建中形成逐文件相同的 D-03 candidate。"""
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_d03_candidate(REPOSITORY, first)
    build_d03_candidate(REPOSITORY, second)
    assert _tree_identity(first) == _tree_identity(second)
    first_reader = D03ReleaseReader.open(
        first,
        FORMAL_GLOBAL_MANIFEST_PATH,
        dependency_root=REPOSITORY,
        require_publication=False,
    )
    second_reader = D03ReleaseReader.open(
        second,
        FORMAL_GLOBAL_MANIFEST_PATH,
        dependency_root=REPOSITORY,
        require_publication=False,
    )
    assert first_reader.global_manifest.to_dict() == second_reader.global_manifest.to_dict()
