"""DLG-RAW-05A 公开课程派生 response-act catalog 的有界回归。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    load_public_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1,
    PublicResponseActCatalogError,
    load_public_response_act_frame_catalog_from_closure,
    materialize_public_response_act_planning_from_closure,
    merge_public_frame_catalogs,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    PublicSourcePayloadClosureV1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_public_response_act_planning,
    public_response_act_planning_input_from_episode,
)


_ROOT = Path(__file__).resolve().parents[1]
_V2_MANIFEST = "data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample"
_DERIVED_MANIFEST = "data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample"
_CONTEXTUAL_DERIVED_MANIFEST = (
    "data/ph2/dlg_raw_public_contextual_ellipsis_frame_v4.jsonl.sample")
_V1_MANIFEST = "data/ph2/dlg_raw_public_frame_v1.jsonl.sample"
_COURSE = "data/ph2/grounded_answer_train_v1.jsonl.sample"
_LEXICAL_A = "data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample"
_LEXICAL_B = "data/ph2/dlg_raw_public_response_act_lexical_v2_b.txt.sample"
_V2_KEY = _V2_MANIFEST.encode("ascii")
_DERIVED_KEY = _DERIVED_MANIFEST.encode("ascii")
_CONTEXTUAL_DERIVED_KEY = _CONTEXTUAL_DERIVED_MANIFEST.encode("ascii")


def _copy_public_sources(tmp_path: Path) -> tuple[Path, Path]:
    """复制完整 27-resource closure，供独立物理根 A/B 验证使用。"""
    root = tmp_path / "response-act-public-repository"
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        source = _ROOT / relative_path
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
    return root, root / _V2_MANIFEST


def _copy_contextual_sources(tmp_path: Path) -> tuple[Path, Path]:
    """复制完整 closure，之后只变更 V4 相关 bytes。"""
    root = tmp_path / "contextual-public-repository"
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        source = _ROOT / relative_path
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
    return root, root / _CONTEXTUAL_DERIVED_MANIFEST


def _closure_from_root(root: Path) -> PublicSourcePayloadClosureV1:
    """测试 host 只负责物理读取，catalog core 只获得已冻结 closure。"""
    return load_public_source_payload_closure_from_root(root)


def _read_manifest_rows(path: Path) -> list[dict[str, object]]:
    """只读取 canonical JSONL，避免测试用宽松 JSON 掩盖 transport 问题。"""
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    return [parse_canonical_json_bytes(line, require_object=True)
            for line in payload[:-1].split(b"\n")]


def _write_manifest_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """以同一 canonical JSONL 编码回写测试副本。"""
    path.write_bytes(b"".join(canonical_json_line(item) for item in rows))


def _contains_bytes(record: tuple[int, ...], value: bytes) -> bool:
    """检查整数 canonical record 是否错误嵌入一段原始字节。"""
    needle = tuple(value)
    return any(record[index:index + len(needle)] == needle
               for index in range(len(record) - len(needle) + 1))


def test_v2_catalog_loads_three_course_derived_frames_stably() -> None:
    """三条公开课程样例必须两次得到同一完整整数 catalog 和 SourceRef。"""
    closure = _closure_from_root(_ROOT)
    first = load_public_response_act_frame_catalog_from_closure(closure, _V2_KEY)
    second = load_public_response_act_frame_catalog_from_closure(closure, _V2_KEY)

    assert len(first.frames) == 3
    assert first.canonical_record() == second.canonical_record()
    assert tuple(frame.recipe.episode_id for frame in first.frames) == (
        "train-grounded-clarify-site-v1",
        "train-grounded-conflict-date-v1",
        "train-grounded-unknown-budget-v1",
    )
    for before, after in zip(first.frames, second.frames, strict=True):
        assert before.recipe.planning_input_record
        assert before.recipe.planning_input_record == after.recipe.planning_input_record
        assert tuple(item.source.stable_key() for item in before.source_records) == (
            tuple(item.source.stable_key() for item in after.source_records))
        assert all(type(value) is int for value in before.canonical_record())


def test_v3_catalog_derives_omission_frame_without_answer_labels() -> None:
    """V3 omission 只能从公开 question bytes 派生，planning 仍不读取答案标签。"""
    closure = _closure_from_root(_ROOT)
    catalog = load_public_response_act_frame_catalog_from_closure(
        closure, _DERIVED_KEY)

    assert len(catalog.frames) == 1
    frame = catalog.frames[0]
    assert bytes(frame.surface_bytes).decode("utf-8") == "北川站东门预算是多少？"
    assert frame.recipe.episode_id == "train-grounded-unknown-budget-v1"
    assert not _contains_bytes(frame.canonical_record(), b'"answer_plan"')
    build = materialize_public_response_act_planning_from_closure(
        frame, frame.question.request_for((65001, 80, 99)), closure)
    assert build.planning_input.canonical_record() == frame.recipe.planning_input_record


def test_v4_catalog_derives_contextual_ellipsis_without_answer_labels() -> None:
    """V4 只能由 self QuestionRequest target 形成 context anchor。"""
    closure = _closure_from_root(_ROOT)
    catalog = load_public_response_act_frame_catalog_from_closure(
        closure, _CONTEXTUAL_DERIVED_KEY)

    assert len(catalog.frames) == 1
    frame = catalog.frames[0]
    assert bytes(frame.surface_bytes).decode("utf-8") == "预算是多少？"
    assert frame.recipe.episode_id == "train-grounded-unknown-budget-v1"
    assert frame.context_requirement == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR
    assert frame.context_target_key == frame.question.target.stable_key()
    assert not _contains_bytes(frame.canonical_record(), b'"answer_plan"')
    build = materialize_public_response_act_planning_from_closure(
        frame,
        frame.question.request_for((65001, 80, 100)),
        closure,
    )
    assert build.planning_input.canonical_record() == frame.recipe.planning_input_record


def test_v4_catalog_rejects_context_selector_omission_and_source_drift(
        tmp_path: Path) -> None:
    """V4 不允许替换 target selector、宽松 omission 或伪造 lexical source。"""
    cases = (
        ({}, "字段集合漂移"),
        ({"kind": "UNKNOWN_CONTEXT_SELECTOR"}, "kind 未注册"),
        ({"kind": "SELF_QUESTION_TARGET_V1", "extra": 1}, "字段集合漂移"),
    )
    for ordinal, (derivation, message) in enumerate(cases, start=1):
        root, manifest = _copy_contextual_sources(tmp_path / f"selector-{ordinal}")
        rows = _read_manifest_rows(manifest)
        rows[0]["context_derivation"] = derivation
        _write_manifest_rows(manifest, rows)
        with pytest.raises(PublicResponseActCatalogError, match=message):
            load_public_response_act_frame_catalog_from_closure(
                _closure_from_root(root), _CONTEXTUAL_DERIVED_KEY)

    root, manifest = _copy_contextual_sources(tmp_path / "omission")
    rows = _read_manifest_rows(manifest)
    rows[0]["question_derivation"]["omitted_utf8_hex"] = "00"
    _write_manifest_rows(manifest, rows)
    with pytest.raises(PublicResponseActCatalogError,
                       match="omission 在课程 question 中缺失或不唯一"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _CONTEXTUAL_DERIVED_KEY)

    root, manifest = _copy_contextual_sources(tmp_path / "lexical")
    lexical = root / (
        "data/ph2/dlg_raw_public_contextual_ellipsis_lexical_v4_a.txt.sample")
    lexical.write_bytes(b"X" + lexical.read_bytes())
    with pytest.raises(PublicResponseActCatalogError, match="raw SHA-256 漂移"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _CONTEXTUAL_DERIVED_KEY)


def test_merge_rejects_unreachable_contextual_frame() -> None:
    """独立 V4 可供审计，但终端最终合并不得接受无 NONE anchor 的组件。"""
    contextual = load_public_response_act_frame_catalog_from_closure(
        _closure_from_root(_ROOT), _CONTEXTUAL_DERIVED_KEY)

    with pytest.raises(PublicResponseActCatalogError,
                       match="TARGET_ANCHOR 没有可达 NONE target"):
        merge_public_frame_catalogs(contextual)


def test_v2_catalog_record_does_not_embed_course_labels_or_surfaces() -> None:
    """V2 frame 只留非标签 observation span，不能携带课程 answer plan 或表层。"""
    catalog = load_public_response_act_frame_catalog_from_closure(
        _closure_from_root(_ROOT), _V2_KEY)
    record = catalog.canonical_record()
    forbidden = (
        b'"answer_plan"',
        b'"response_act"',
        "现有来源没有提供所问信息。".encode("utf-8"),
        "存在多个候选，请补充更明确的限定。".encode("utf-8"),
        "来源说法彼此矛盾，无法给出确定答案。".encode("utf-8"),
    )

    assert all(not _contains_bytes(record, item) for item in forbidden)


def test_v2_materializer_rebuilds_only_label_free_planning() -> None:
    """runtime materializer 必须用课程重投影的 record 建立 planning，不反解 recipe。"""
    closure = _closure_from_root(_ROOT)
    catalog = load_public_response_act_frame_catalog_from_closure(closure, _V2_KEY)

    for ordinal, frame in enumerate(catalog.frames, start=1):
        request = frame.question.request_for((65001, 80, ordinal))
        build = materialize_public_response_act_planning_from_closure(
            frame, request, closure)

        assert build.planning_input.canonical_record() == (
            frame.recipe.planning_input_record)
        assert build.planning.goal.proposition == request.target
        assert build.response_scope == request.response_scope
        assert build.planning.goal.target_branch == request.target_branch
        assert all(type(value) is int for value in build.planning_input.stable_key())


def test_v2_materializer_rechecks_course_content_lock_each_call(
        tmp_path: Path) -> None:
    """重新构造发生内容漂移的 closure 时，runtime materializer 必须拒绝。"""
    root, manifest = _copy_public_sources(tmp_path)
    baseline_closure = _closure_from_root(root)
    catalog = load_public_response_act_frame_catalog_from_closure(
        baseline_closure, _V2_KEY)
    frame = catalog.frames[0]
    course = root / _COURSE
    payload = course.read_bytes()
    course.write_bytes(b"X" + payload[1:])

    with pytest.raises(PublicResponseActCatalogError, match="raw SHA-256 漂移"):
        materialize_public_response_act_planning_from_closure(
            frame, frame.question.request_for((65001, 81, 1)),
            _closure_from_root(root))


def test_v2_materializer_rejects_drifted_complete_question_request() -> None:
    """动态 occurrence 以外的任一 F-00 字段均不能被 runtime 替换。"""
    closure = _closure_from_root(_ROOT)
    catalog = load_public_response_act_frame_catalog_from_closure(closure, _V2_KEY)
    frame = catalog.frames[0]
    request = frame.question.request_for((65001, 81, 2))
    drifted = replace(
        request,
        intent=minimal_instruction_identity((65001, 81, 3)),
    )

    with pytest.raises(PublicResponseActCatalogError,
                       match="漂移完整 QuestionRequest"):
        materialize_public_response_act_planning_from_closure(
            frame, drifted, closure)


def test_v2_catalog_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    """manifest 字段增加也必须 fail closed，不能被默认值或宽松 parser 吞掉。"""
    root, manifest = _copy_public_sources(tmp_path)
    rows = _read_manifest_rows(manifest)
    rows[0]["unexpected"] = 1
    _write_manifest_rows(manifest, rows)

    with pytest.raises(PublicResponseActCatalogError, match="字段集合漂移"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _V2_KEY)


def test_v2_catalog_rejects_course_and_lexical_source_drift(
        tmp_path: Path) -> None:
    """课程或词汇 raw bytes 的任一漂移都不得进入已派生目录。"""
    root, manifest = _copy_public_sources(tmp_path)
    course = root / _COURSE
    payload = course.read_bytes()
    course.write_bytes(b"X" + payload[1:])

    with pytest.raises(PublicResponseActCatalogError, match="公开课程 raw SHA-256 漂移"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _V2_KEY)

    root, manifest = _copy_public_sources(tmp_path / "lexical")
    lexical = root / _LEXICAL_A
    payload = lexical.read_bytes()
    lexical.write_bytes(b"X" + payload[1:])

    with pytest.raises(PublicResponseActCatalogError, match="raw SHA-256 漂移"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _V2_KEY)


def test_v2_catalog_rejects_lexical_span_and_source_identity_drift(
        tmp_path: Path) -> None:
    """即使攻击者同步 manifest SHA，缺少 span 或两个 source 同一仍须拒绝。"""
    root, manifest = _copy_public_sources(tmp_path / "span")
    lexical = root / _LEXICAL_A
    payload = lexical.read_bytes()
    before = "北川站东门的建设预算是多少？".encode("utf-8")
    after = "南川站东门的建设预算是多少？".encode("utf-8")
    assert payload.count(before) == 1
    lexical.write_bytes(payload.replace(before, after, 1))
    rows = _read_manifest_rows(manifest)
    rows[0]["lexical_source_a"]["raw_sha256"] = hashlib.sha256(
        lexical.read_bytes()).hexdigest()
    _write_manifest_rows(manifest, rows)

    with pytest.raises(PublicResponseActCatalogError, match="surface.*缺失或不唯一"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _V2_KEY)

    root, manifest = _copy_public_sources(tmp_path / "identity")
    rows = _read_manifest_rows(manifest)
    rows[0]["lexical_source_a"] = dict(rows[0]["lexical_source_b"])
    _write_manifest_rows(manifest, rows)

    with pytest.raises(PublicResponseActCatalogError, match="两个不同 SourceRef"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(root), _V2_KEY)


def test_public_planning_is_invariant_to_course_response_act_label() -> None:
    """替换合法字符串标签不得改变公开 input、candidate 或 planning 状态。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_ROOT / _COURSE)
        if item.episode_id == "train-grounded-clarify-site-v1")
    relabeled = replace(
        episode,
        question=replace(
            episode.question,
            answer_plan=replace(
                episode.question.answer_plan,
                response_act="UNKNOWN",
            ),
        ),
    )
    branch = language_branch_identity((65001, 82, 1))
    original_input = public_response_act_planning_input_from_episode(episode)
    relabeled_input = public_response_act_planning_input_from_episode(relabeled)
    original = compile_public_response_act_planning(original_input, branch)
    relabeled_build = compile_public_response_act_planning(relabeled_input, branch)

    assert original_input.canonical_record() == relabeled_input.canonical_record()
    assert tuple(item.candidate.stable_key() for item in original.candidate_bindings) == (
        tuple(item.candidate.stable_key()
              for item in relabeled_build.candidate_bindings))
    assert original.planning.stable_key() == relabeled_build.planning.stable_key()


def test_v2_catalog_merges_with_v1_without_changing_v1_identity() -> None:
    """V2 必须直接贡献同构 frame，且合并不改变原 V1 catalog 本身。"""
    closure = _closure_from_root(_ROOT)
    v1 = load_public_frame_catalog_from_closure(closure)
    v1_record = v1.canonical_record()
    v2 = load_public_response_act_frame_catalog_from_closure(closure, _V2_KEY)
    merged = merge_public_frame_catalogs(v1, v2)

    assert v1.canonical_record() == v1_record
    assert len(merged.frames) == len(v1.frames) + len(v2.frames)
    assert all(len(merged.matching_frames(frame.surface_scalars)) == 1
               for frame in merged.frames)
    with pytest.raises(PublicResponseActCatalogError, match="frame_key 重复"):
        merge_public_frame_catalogs(v2, v2)


def test_response_act_catalog_is_root_independent_and_drift_fails_closed(
        tmp_path: Path) -> None:
    """相同 27 项 raw bytes 跨根同一；任一受锁内容漂移必须拒绝。"""
    clone_root, _manifest = _copy_public_sources(tmp_path)
    source_closure = _closure_from_root(_ROOT)
    clone_closure = _closure_from_root(clone_root)

    assert source_closure.closure_identity == clone_closure.closure_identity
    assert set(PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1) == {
        _V2_KEY, _DERIVED_KEY, _CONTEXTUAL_DERIVED_KEY}
    source_catalog = load_public_response_act_frame_catalog_from_closure(
        source_closure, _V2_KEY)
    clone_catalog = load_public_response_act_frame_catalog_from_closure(
        clone_closure, _V2_KEY)
    assert source_catalog.canonical_record() == clone_catalog.canonical_record()

    course = clone_root / _COURSE
    payload = course.read_bytes()
    course.write_bytes(b"X" + payload[1:])
    with pytest.raises(PublicResponseActCatalogError, match="公开课程 raw SHA-256 漂移"):
        load_public_response_act_frame_catalog_from_closure(
            _closure_from_root(clone_root), _V2_KEY)
