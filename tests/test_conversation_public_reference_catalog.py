"""DLG-RAW-05B V3 公开双命题/reference catalog 验证。"""
from __future__ import annotations

from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameReferenceRuntimeRecipe,
    load_public_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    PublicReferenceCatalogError,
    load_public_reference_frame_catalog_from_closure,
    materialize_public_reference_planning_from_closure,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    load_public_response_act_frame_catalog_from_closure,
    merge_public_frame_catalogs,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    PublicSourcePayloadClosureV1,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_public_reference_planning,
    public_response_act_planning_input_from_episode,
)
from pure_integer_ai.cognition.shared.identity import language_branch_identity


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PATH = "data/ph2/dlg_raw_public_reference_frame_v3.jsonl.sample"
_V1_PATH = "data/ph2/dlg_raw_public_frame_v1.jsonl.sample"
_V2_PATH = "data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample"
_COURSE_PATH = "data/ph2/grounded_answer_train_v1.jsonl.sample"
_REFERENCE_SOURCE_PATHS = (
    _REFERENCE_PATH,
    _COURSE_PATH,
    "data/ph2/dlg_raw_public_reference_input_v3_a.txt.sample",
    "data/ph2/dlg_raw_public_reference_input_v3_b.txt.sample",
    "data/ph2/dlg_raw_public_reference_antecedent_v3_a.txt.sample",
    "data/ph2/dlg_raw_public_reference_antecedent_v3_b.txt.sample",
    "data/ph2/dlg_raw_public_reference_explicit_v3_a.txt.sample",
    "data/ph2/dlg_raw_public_reference_explicit_v3_b.txt.sample",
)


def _copy_reference_sources(tmp_path: Path) -> tuple[Path, Path]:
    """建立包含完整 logical closure 的独立物理根。"""
    root = tmp_path / "public-reference-repository"
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        source = _REPOSITORY_ROOT / relative_path
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
    return root, root / _REFERENCE_PATH


def _closure_from_root(root: Path) -> PublicSourcePayloadClosureV1:
    """host 将物理根冻结为 closure 后，catalog 只消费纯 payload record。"""
    return load_public_source_payload_closure_from_root(root)


def _replace_once(path: Path, before: bytes, after: bytes) -> None:
    """只替换一处冻结 transport 字段，避免测试改变无关输入。"""
    payload = path.read_bytes()
    assert payload.count(before) == 1
    path.write_bytes(payload.replace(before, after, 1))


def _contains_bytes(record: tuple[int, ...], value: bytes) -> bool:
    """检查规范整数 record 未夹带一个完整的禁入 UTF-8 表面。"""
    needle = tuple(value)
    return any(record[index:index + len(needle)] == needle
               for index in range(len(record) - len(needle) + 1))


def test_reference_catalog_loads_twice_and_materializes_label_free_inputs() -> None:
    """V3 只重派生 Evidence/planning/reference lexeme，不保存答案标签。"""
    closure = _closure_from_root(_REPOSITORY_ROOT)
    first = load_public_reference_frame_catalog_from_closure(closure)
    second = load_public_reference_frame_catalog_from_closure(closure)

    assert first.canonical_record() == second.canonical_record()
    assert len(first.frames) == 1
    frame = first.frames[0]
    assert isinstance(frame.recipe, PublicFrameReferenceRuntimeRecipe)
    request = frame.question.request_for((65001, 53, 90))
    build = materialize_public_reference_planning_from_closure(
        frame, request, closure)

    ordered_candidates = tuple(
        build.planning_build.candidate_for(proposition_id)
        for proposition_id in frame.recipe.ordered_proposition_ids)
    assert frame.recipe.ordered_proposition_ids == ("p-year", "p-registration")
    assert len(set(ordered_candidates)) == 2
    assert set(ordered_candidates) == set(build.planning_build.planning.candidates)
    assert len(build.planning_build.planning.candidates) == 2
    assert build.antecedent_reference_scalars == tuple(map(ord, "前述"))
    assert build.explicit_repetition_scalars == tuple(map(ord, "北川站东门的"))
    assert not hasattr(build.planning_build.planning_input, "answer_plan")
    assert not hasattr(build.planning_build.planning_input, "reference_course")

    record = first.canonical_record()
    for forbidden in (
            b'"answer_plan"',
            b'"surfaces"',
            b'"reference_course"',
            b'"surface_labels"',
            "北川站东门于2024年启用。前述启用事项已登记入档。".encode("utf-8"),
            "北川站东门于2024年启用。北川站东门的启用事项已登记入档。".encode("utf-8")):
        assert not _contains_bytes(record, forbidden)


def test_public_planning_is_independent_from_answer_plan_and_reference_labels() -> None:
    """公共 planning 只取 Evidence；旧课程标签即使失效也不得进入其语义。"""
    episode = next(
        item for item in read_grounded_answer_episodes(
            _REPOSITORY_ROOT / _COURSE_PATH)
        if item.episode_id == "train-grounded-reference-event-v2")
    baseline = public_response_act_planning_input_from_episode(episode)
    original_plan = episode.question.answer_plan
    original_reference = episode.reference_course
    object.__setattr__(episode.question, "answer_plan", object())
    object.__setattr__(episode, "reference_course", object())
    try:
        changed = public_response_act_planning_input_from_episode(episode)
    finally:
        object.__setattr__(episode.question, "answer_plan", original_plan)
        object.__setattr__(episode, "reference_course", original_reference)

    assert changed.canonical_record() == baseline.canonical_record()
    branch = language_branch_identity((65001, 53, 2))
    assert compile_public_reference_planning(
        changed, branch, ("p-year", "p-registration")).stable_key() == (
            compile_public_reference_planning(
                baseline, branch, ("p-year", "p-registration")).stable_key())


@pytest.mark.parametrize(
    ("relative_path", "before", "after", "message"),
    (
        (
            _REFERENCE_PATH,
            b'"relation_kind_code":1',
            b'"relation_kind_code":2',
            "relation kind 未注册",
        ),
        (
            _REFERENCE_PATH,
            (b'"relative_path":"data/ph2/dlg_raw_public_reference_'
             b'antecedent_v3_a.txt.sample","span_utf8_hex":"e5898de8bfb0"'),
            (b'"relative_path":"data/ph2/dlg_raw_public_reference_'
             b'antecedent_v3_a.txt.sample","span_utf8_hex":"e58c97e5b79e"'),
            "span",
        ),
        (
            "data/ph2/dlg_raw_public_reference_antecedent_v3_a.txt.sample",
            b"\xe5\x89\x8d\xe8\xbf\xb0",
            b"\xe5\x89\x8d\xe8\xbf\xaf",
            "raw SHA-256 漂移",
        ),
    ),
    ids=("relation-kind", "reference-span", "lexical-raw-sha"),
)
def test_reference_catalog_rejects_manifest_or_source_drift(
        tmp_path: Path,
        relative_path: str,
        before: bytes,
        after: bytes,
        message: str,
        ) -> None:
    """任何 V3 结构字段、span 或公开 source 漂移都必须 fail closed。"""
    root, catalog_path = _copy_reference_sources(tmp_path)
    _replace_once(root / relative_path, before, after)

    with pytest.raises(PublicReferenceCatalogError, match=message):
        load_public_reference_frame_catalog_from_closure(_closure_from_root(root))


def test_three_catalog_merge_preserves_existing_catalog_identities() -> None:
    """V1/V2/V3 合并只增加无歧义 route，不修改旧 catalog 的规范记录。"""
    closure = _closure_from_root(_REPOSITORY_ROOT)
    v1 = load_public_frame_catalog_from_closure(closure)
    v2 = load_public_response_act_frame_catalog_from_closure(
        closure, _V2_PATH.encode("ascii"))
    v3 = load_public_reference_frame_catalog_from_closure(closure)
    v1_before = v1.canonical_record()
    v2_before = v2.canonical_record()

    merged = merge_public_frame_catalogs(v1, v2, v3)

    assert v1.canonical_record() == v1_before
    assert v2.canonical_record() == v2_before
    assert len(merged.frames) == len(v1.frames) + len(v2.frames) + len(v3.frames)
    assert merged.frames == tuple(sorted(
        merged.frames, key=lambda item: item.canonical_record()))
    assert {item.surface_scalars for item in merged.frames} == {
        item.surface_scalars
        for catalog in (v1, v2, v3) for item in catalog.frames
    }


def test_reference_catalog_is_root_independent_and_drift_fails_closed(
        tmp_path: Path) -> None:
    """两物理根的同一 closure 必须重放相同 catalog；漂移必须拒绝。"""
    clone_root, _catalog_path = _copy_reference_sources(tmp_path)
    source_closure = _closure_from_root(_REPOSITORY_ROOT)
    clone_closure = _closure_from_root(clone_root)

    assert source_closure.closure_identity == clone_closure.closure_identity
    source_catalog = load_public_reference_frame_catalog_from_closure(
        source_closure)
    clone_catalog = load_public_reference_frame_catalog_from_closure(
        clone_closure)
    assert source_catalog.canonical_record() == clone_catalog.canonical_record()
    source_frame = source_catalog.frames[0]
    clone_frame = clone_catalog.frames[0]
    source_build = materialize_public_reference_planning_from_closure(
        source_frame, source_frame.question.request_for((65001, 53, 91)),
        source_closure)
    clone_build = materialize_public_reference_planning_from_closure(
        clone_frame, clone_frame.question.request_for((65001, 53, 91)),
        clone_closure)
    assert source_build.canonical_record() == clone_build.canonical_record()

    lexical = clone_root / (
        "data/ph2/dlg_raw_public_reference_antecedent_v3_a.txt.sample")
    lexical.write_bytes(b"X" + lexical.read_bytes())
    with pytest.raises(PublicReferenceCatalogError, match="raw SHA-256 漂移"):
        load_public_reference_frame_catalog_from_closure(
            _closure_from_root(clone_root))
