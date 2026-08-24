"""DLG-RAW-07 V1 public frame catalog 的 logical closure 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1,
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    PublicFrameCatalog,
    PublicFrameCatalogError,
    load_public_frame_catalog,
    load_public_frame_catalog_from_closure,
    load_public_frame_catalog_from_host_root,
    materialize_public_frame_candidate,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
)


_ROOT = Path(__file__).resolve().parents[1]
_FRAME_CATALOG_SOURCE = (
    _ROOT / "src/pure_integer_ai/experiments/"
    "conversation_public_frame_catalog.py"
)
_LEXICAL_SOURCE_LOGICAL_KEY = (
    b"data/ph2/dlg_raw_public_lexical_evidence_v1.txt.sample")


def _copy_public_dialogue_resources(target: Path) -> Path:
    """复制冻结 27 项 payload，供独立物理根与内容漂移专项使用。"""
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative_path, destination)
    return target


def _closure_at(resource_root: Path):
    """只在测试 host 边界读物理文件，core 测试始终接收 closure。"""
    return load_public_source_payload_closure_from_root(resource_root)


def _replace_catalog_bytes(
        resource_root: Path,
        before: bytes,
        after: bytes,
        ) -> None:
    """对独立 closure transport 作一次精确替换，防止静默修改其他字段。"""
    catalog_path = resource_root / PUBLIC_FRAME_CATALOG_LOGICAL_KEY_V1.decode("ascii")
    payload = catalog_path.read_bytes()
    assert payload.count(before) == 1
    catalog_path.write_bytes(payload.replace(before, after, 1))


def test_public_frame_catalog_loads_full_request_and_candidate_from_closure() -> None:
    """完整 closure 必须可物化 F-00 request 和无标签 G-00 candidate。"""
    closure = _closure_at(_ROOT)
    catalog = load_public_frame_catalog_from_closure(closure)

    assert load_public_frame_catalog(closure).canonical_record() == (
        catalog.canonical_record())
    assert load_public_frame_catalog_from_host_root(_ROOT).canonical_record() == (
        catalog.canonical_record())
    assert len(catalog.frames) == 2
    frame = next(item for item in catalog.frames
                 if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    assert frame.context_requirement == PUBLIC_FRAME_CONTEXT_NONE
    assert frame.context_target_key == ()
    assert frame.question.authorized_candidate_targets == (frame.question.target,)

    request = frame.question.request_for((65001, 31, 1))
    candidate, planning = materialize_public_frame_candidate(frame, request)

    assert isinstance(request, QuestionRequest)
    assert request.target == frame.question.target
    assert request.authorized_candidate_targets == (frame.question.target,)
    assert request.trace == (*frame.question.trace_prefix, 65001, 31, 1)
    assert isinstance(candidate, GenerationCandidate)
    assert candidate.proposition == request.target
    assert candidate.evidence == frame.recipe.candidate_evidence
    assert isinstance(planning, GenerationPlanningRequest)
    assert planning.goal.proposition == request.target
    assert planning.candidates == (candidate,)
    assert all(type(item) is int for item in catalog.canonical_record())


def test_equal_payloads_in_two_physical_roots_form_identical_catalogs(
        tmp_path: Path,
        ) -> None:
    """物理根、复制顺序和 mtime 均不得进入 catalog 的规范整数结果。"""
    first = load_public_frame_catalog_from_closure(_closure_at(_ROOT))
    copied_root = _copy_public_dialogue_resources(tmp_path / "copied-public")
    second = load_public_frame_catalog_from_closure(_closure_at(copied_root))

    assert first.source_sha256 == second.source_sha256
    assert first.canonical_record() == second.canonical_record()


def test_public_frame_catalog_rejects_closure_source_raw_sha_drift(
        tmp_path: Path,
        ) -> None:
    """manifest 不得只信声明 hash，closure 中任一 source 漂移必须拒绝。"""
    copied_root = _copy_public_dialogue_resources(tmp_path / "drift-public")
    lexical_source = copied_root / _LEXICAL_SOURCE_LOGICAL_KEY.decode("ascii")
    payload = lexical_source.read_bytes()
    lexical_source.write_bytes(b"X" + payload[1:])

    with pytest.raises(PublicFrameCatalogError, match="raw SHA-256 漂移"):
        load_public_frame_catalog_from_closure(_closure_at(copied_root))


@pytest.mark.parametrize(
    "replacement",
    (
        (
            b'"evidence_source_record_ids":["lexical-a-north","lexical-b-north"]',
            b'"evidence_source_record_ids":["lexical-a-north","lexical-a-north"]',
        ),
        (
            b'"evidence_source_record_ids":["lexical-a-north","lexical-b-north"]',
            b'"evidence_source_record_ids":["lexical-a-north"]',
        ),
    ),
    ids=("duplicate-source-ref", "insufficient-source-ref"),
)
def test_public_frame_catalog_rejects_non_independent_lexical_evidence(
        tmp_path: Path,
        replacement: tuple[bytes, bytes],
        ) -> None:
    """每个词汇路由必须保留至少两个不同的公开 SourceRef。"""
    copied_root = _copy_public_dialogue_resources(tmp_path / "manifest-public")
    before, after = replacement
    _replace_catalog_bytes(copied_root, before, after)

    with pytest.raises(PublicFrameCatalogError, match="缺独立 lexical evidence"):
        load_public_frame_catalog_from_closure(_closure_at(copied_root))


def test_none_context_frame_rejects_any_context_anchor(tmp_path: Path) -> None:
    """首轮 NONE frame 不得携带可被误当作前文的 context anchor。"""
    copied_root = _copy_public_dialogue_resources(tmp_path / "anchor-public")
    _replace_catalog_bytes(
        copied_root,
        b'"context_requirement":"NONE","context_target_key":[]',
        b'"context_requirement":"NONE","context_target_key":[1]',
    )

    with pytest.raises(PublicFrameCatalogError, match="NONE frame 不得携带 context anchor"):
        load_public_frame_catalog_from_closure(_closure_at(copied_root))


def test_catalog_rejects_target_anchor_without_a_catalog_target() -> None:
    """follow-up anchor 必须指向本 catalog 内完整 target。"""
    catalog = load_public_frame_catalog_from_closure(_closure_at(_ROOT))
    follow_up = next(item for item in catalog.frames
                     if item.context_requirement
                     == PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR)
    malformed = replace(follow_up, context_target_key=(1,))
    frames = tuple(sorted(
        (item if item != follow_up else malformed for item in catalog.frames),
        key=lambda item: item.canonical_record(),
    ))

    with pytest.raises(PublicFrameCatalogError, match="未绑定 catalog 内完整 target"):
        PublicFrameCatalog(catalog.source_sha256, frames)


def test_closure_loader_rejects_nonclosure_input_before_parsing() -> None:
    """core loader 不能重新解释路径、bytes 或任意 Python provider 对象。"""
    with pytest.raises(PublicFrameCatalogError, match="完整 payload closure"):
        load_public_frame_catalog_from_closure(object())  # type: ignore[arg-type]
    with pytest.raises(PublicFrameCatalogError, match="路径式 public frame catalog 加载已废止"):
        load_public_frame_catalog(object())  # type: ignore[arg-type]


def test_pure_frame_catalog_ast_has_no_physical_filesystem_boundary() -> None:
    """frame core 不得重新引入路径、物理读取或旧 root 语义。"""
    source = _FRAME_CATALOG_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ) + tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "pathlib" not in imported_modules
    assert "repository_root" not in source
    assert "read_bytes" not in source
    assert "PurePosixPath" not in source
    assert "Path(" not in source
