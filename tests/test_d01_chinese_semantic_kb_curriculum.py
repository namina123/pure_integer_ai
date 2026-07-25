"""D-01 ChineseSemanticKB 课程切片、来源簇和只读产物测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.chinese_semantic_kb_adapter import (
    PARSER_DECIMAL_TAB,
    PARSER_DOCUMENT,
    PARSER_RELATION_MARKER,
    PARSER_SURFACE_LINE,
    PARSER_SYMMETRIC_AT,
    PROFILES,
    ChineseKBRecord,
    ChineseSemanticKBAdapter,
    build_manifest,
)
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    KIND_ANOMALY,
    KIND_PATTERN_CANDIDATE,
    KIND_RELATION_CANDIDATE,
    KIND_WORD_FORM,
    SPLIT_HELD_OUT,
    ChineseSemanticKBCurriculum,
    CourseSplitPolicy,
    build_curriculum_artifacts,
    read_curriculum_manifest,
)
from pure_integer_ai.experiments.data_manifest import ManifestIntegrityError


def _valid_line(profile) -> str:
    """按 adapter profile 生成一个合法来源记录。"""
    if profile.parser_kind == PARSER_DOCUMENT:
        return "# 来源说明"
    if profile.parser_kind == PARSER_RELATION_MARKER:
        return f"甲,{profile.relation_marker},乙"
    if profile.parser_kind == PARSER_SYMMETRIC_AT:
        return "甲@乙"
    if profile.parser_kind == PARSER_DECIMAL_TAB:
        return "很\t2.50"
    if profile.parser_kind == PARSER_SURFACE_LINE:
        return "词形"
    raise AssertionError(profile.parser_kind)


def _write_snapshot(root: Path) -> None:
    """创建覆盖 12 类词典和 README 的最小课程快照。"""
    for profile in PROFILES:
        path = root / profile.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_valid_line(profile) + "\n", encoding="utf-8")
    (root / "dict/程度副词.txt").write_text(
        "过低\t-1.25\n",
        encoding="utf-8",
    )
    (root / "dict/否定词.txt").write_text(
        "要不要\n不\n",
        encoding="utf-8",
    )
    (root / "dict/停用词.txt").write_text(
        "的\n",
        encoding="utf-8",
    )
    (root / "dict/反义关系库.txt").write_text(
        "上@下\n下@上\n",
        encoding="utf-8",
    )


def _source_manifest(root: Path):
    """构造测试快照的 D-00C manifest。"""
    manifest, _ = build_manifest(
        root,
        dataset_version="fixture-course-v1",
        unicode_sequence_family=(7001,),
    )
    return manifest


def _policy() -> CourseSplitPolicy:
    """返回测试显式声明的切分策略。"""
    return CourseSplitPolicy(1, 800, 100, 100)


def test_curriculum_preserves_all_surfaces_as_forms_and_only_candidates(
        tmp_path):
    raw = tmp_path / "raw"
    _write_snapshot(raw)
    source_manifest = _source_manifest(raw)
    course_root = tmp_path / "course"
    built = build_curriculum_artifacts(
        source_manifest,
        raw,
        course_root,
        split_policy=_policy(),
    )
    loaded = read_curriculum_manifest(course_root / "manifest.json")
    curriculum = ChineseSemanticKBCurriculum(loaded, course_root)

    expected_surfaces: set[tuple[str, str]] = set()
    relation_count = 0
    pattern_count = 0
    for event in ChineseSemanticKBAdapter(
            source_manifest, raw).iter_events():
        if not isinstance(event, ChineseKBRecord):
            continue
        if event.parser_kind == PARSER_DOCUMENT:
            continue
        surfaces = (event.fields[:1]
                    if event.parser_kind == PARSER_DECIMAL_TAB
                    else event.fields)
        expected_surfaces.update(
            (event.category, surface) for surface in surfaces)
        if event.parser_kind in {
                PARSER_RELATION_MARKER, PARSER_SYMMETRIC_AT}:
            relation_count += 1
        else:
            pattern_count += 1

    forms = tuple(curriculum.iter_word_forms())
    relations = tuple(curriculum.iter_relation_candidates())
    patterns = tuple(curriculum.iter_pattern_candidates())

    assert built.canonical_bytes() == loaded.canonical_bytes()
    assert {(item.category, item.surface) for item in forms} == expected_surfaces
    assert len(relations) == relation_count
    assert len(patterns) == pattern_count
    assert next(item for item in patterns
                if item.category == "degree_adverb").rational == (-5, 4)
    assert {item.fields for item in patterns
            if item.category == "negation"} == {("要不要",), ("不",)}
    assert {item.fields for item in patterns
            if item.category == "stoplist"} == {("的",)}
    assert loaded.artifact(KIND_WORD_FORM).record_count == len(forms)
    assert loaded.artifact(KIND_RELATION_CANDIDATE).record_count == len(relations)
    assert loaded.artifact(KIND_PATTERN_CANDIDATE).record_count == len(patterns)
    curriculum.verify_record_counts()


def test_reverse_duplicate_is_anomaly_not_independent_relation_source(tmp_path):
    raw = tmp_path / "raw"
    _write_snapshot(raw)
    source_manifest = _source_manifest(raw)
    course_root = tmp_path / "course"
    manifest = build_curriculum_artifacts(
        source_manifest,
        raw,
        course_root,
        split_policy=_policy(),
    )
    curriculum = ChineseSemanticKBCurriculum(manifest, course_root)

    relations = tuple(curriculum.iter_relation_candidates(
        category="antonym"))
    anomalies = tuple(curriculum.iter_anomalies(category="antonym"))
    summary = next(item for item in manifest.categories
                   if item.category == "antonym")

    assert len(relations) == 1
    assert relations[0].fields == ("上", "下")
    assert [item.kind for item in anomalies] == ["duplicate_key"]
    assert relations[0].provenance_cluster_id == (
        anomalies[0].provenance_cluster_id)
    assert relations[0].source_ref.source_id == (
        anomalies[0].source_ref.source_id)
    assert relations[0].source_ref.document_id == 1
    assert anomalies[0].source_ref.document_id == 2
    assert summary.provenance_cluster_count == 1
    anomaly_counts = next(item for item in summary.counts
                          if item[0] == KIND_ANOMALY)
    assert anomaly_counts == (KIND_ANOMALY, 1, 0, 0, 0)


def test_split_policy_is_injected_and_filters_without_reordering(tmp_path):
    raw = tmp_path / "raw"
    _write_snapshot(raw)
    source_manifest = _source_manifest(raw)
    course_root = tmp_path / "course"
    held_out_only = CourseSplitPolicy(7, 0, 0, 1000)
    manifest = build_curriculum_artifacts(
        source_manifest,
        raw,
        course_root,
        split_policy=held_out_only,
    )
    curriculum = ChineseSemanticKBCurriculum(manifest, course_root)

    all_forms = tuple(curriculum.iter_word_forms(category="negation"))
    held_out_forms = tuple(curriculum.iter_word_forms(
        category="negation", split=SPLIT_HELD_OUT))
    assert all_forms == held_out_forms
    assert all(item.split == SPLIT_HELD_OUT for item in all_forms)
    assert manifest.split_policy == held_out_only

    with pytest.raises(ValueError):
        CourseSplitPolicy(1, 500, 500, 1)
    with pytest.raises(ValueError):
        tuple(curriculum.iter_word_forms(split=99))


def test_course_build_is_bit_identical_and_never_overwrites(tmp_path):
    raw = tmp_path / "raw"
    _write_snapshot(raw)
    source_manifest = _source_manifest(raw)
    first_root = tmp_path / "course-a"
    second_root = tmp_path / "course-b"

    first = build_curriculum_artifacts(
        source_manifest, raw, first_root, split_policy=_policy())
    second = build_curriculum_artifacts(
        source_manifest, raw, second_root, split_policy=_policy())

    assert first.canonical_bytes() == second.canonical_bytes()
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json").read_bytes()
    for artifact in first.artifacts:
        assert (first_root / artifact.relative_path).read_bytes() == (
            second_root / artifact.relative_path).read_bytes()
    with pytest.raises(ManifestIntegrityError):
        build_curriculum_artifacts(
            source_manifest, raw, first_root, split_policy=_policy())
    with pytest.raises(ManifestIntegrityError):
        build_curriculum_artifacts(
            source_manifest,
            raw,
            raw / "generated-course",
            split_policy=_policy(),
        )


def test_course_reader_rejects_artifact_mutation(tmp_path):
    raw = tmp_path / "raw"
    _write_snapshot(raw)
    source_manifest = _source_manifest(raw)
    course_root = tmp_path / "course"
    manifest = build_curriculum_artifacts(
        source_manifest, raw, course_root, split_policy=_policy())
    curriculum = ChineseSemanticKBCurriculum(manifest, course_root)
    word_forms = manifest.artifact(KIND_WORD_FORM)
    with (course_root / word_forms.relative_path).open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ManifestIntegrityError):
        ChineseSemanticKBCurriculum(manifest, course_root)
    with pytest.raises(ManifestIntegrityError):
        tuple(curriculum.iter_word_forms())
